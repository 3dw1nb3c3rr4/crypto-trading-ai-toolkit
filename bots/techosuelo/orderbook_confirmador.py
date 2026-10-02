#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orderbook_confirmador.py
=========================
"Cerebro 3.5" -- filtro de CONFIRMACION DE ENTRADA en tiempo real, basado en
el libro de ordenes (order book) y en las operaciones que se van ejecutando
de verdad en el mercado (tape / trade flow).

Que problema resuelve
----------------------
El Cerebro 1 (binance_topsuelo_bot / okx_topsuelo_bot) detecta con velas
DIARIAS que una moneda esta en zona de techo o de suelo. El Cerebro 2
(binance_meta_filtro / okx_meta_filtro) le agrega una probabilidad historica
de que esa señal se cumpla en <=3 dias. Ninguno de los dos mira que esta
pasando AHORA MISMO en el libro de ordenes -- pueden marcar "vender" justo
en el momento en que el libro muestra que en realidad hay una pared de
compradores absorbiendo toda la oferta, lo que suele preceder a un rebote,
no a una caida.

Este modulo agrega esa ULTIMA capa, de mucho mas corto plazo (5 a 15
minutos) y mas alta frecuencia: observa el libro de ordenes y el flujo de
trades ejecutados de cada candidata durante una ventana de tiempo, calcula
un pequeño conjunto de metricas de microestructura de mercado, y con eso da
un veredicto de "confirma" o "no confirma" para el LADO especifico (LONG o
SHORT) que la señal propone. No decide QUE moneda operar (eso ya lo
decidieron los Cerebros 1 y 2) -- decide si el TIMING de entrada es bueno.

Metricas usadas (todas calculadas solo con datos publicos del libro y del
tape, sin indicadores tecnicos de velas):

  1) Order Book Imbalance (OBI): desequilibrio entre el volumen ofertado en
     los primeros N niveles de compra vs. venta. OBI > 0 = mas compradores
     esperando en el libro que vendedores.
  2) Tendencia del OBI: si el desequilibrio se esta ACENTUANDO durante la
     ventana (no solo un valor puntual, que puede ser ruido).
  3) Microprice drift: el "microprice" (precio medio ponderado por el
     tamaño del lado contrario en el mejor nivel) es la mejor estimacion de
     hacia donde se va a mover el proximo trade. Si el microprice se separa
     consistentemente del precio medio simple hacia un lado, ese lado tiene
     presion real.
  4) Momentum + consistencia: el retorno neto del precio medio durante la
     ventana y que tan parejo (no erratico) fue ese movimiento.
  5) Flujo de trades ejecutados (tape reading): de las operaciones que
     realmente se ejecutaron (no solo ordenes puestas, que se pueden
     cancelar), que proporcion fue iniciada por compradores agresivos vs.
     vendedores agresivos.
  6) Filtros de calidad: spread relativo (si esta muy ancho, el libro esta
     poco liquido y cualquier metrica es poco confiable) y notional minimo
     operado durante la ventana (evita confirmar sobre una moneda casi sin
     actividad).

Todo se combina en un "score de confirmacion" 0-1 para el lado pedido. Por
diseño el filtro es CONSERVADOR: mientras no se junten suficientes datos
(snapshots) o no pase suficiente tiempo de observacion, NO confirma (mejor
perder una entrada que entrar a ciegas).

Este modulo es agnostico de exchange: solo necesita un objeto `exchange` de
ccxt ya conectado (ex.fetch_order_book, ex.fetch_trades), asi que sirve
igual para Binance Futures y para OKX -- se pasa el mismo objeto `ex` que ya
usan binance_topsuelo_bot.py / okx_topsuelo_bot.py.

⚠️ Esto sigue sin ser asesoria financiera. Es un filtro estadistico mas,
no una garantia de que la entrada vaya a salir bien.
"""

import math
import time
import threading
from collections import deque

from red_reintentos import AvisadorThrottled


# ---------------------------------------------------------------------------
# CONFIG POR DEFECTO
# ---------------------------------------------------------------------------
VENTANA_SEG_DEFAULT        = 600     # 10 minutos de observacion antes de decidir
POLL_SECS_DEFAULT          = 5       # cada cuanto se refresca el libro/trades por simbolo
NIVELES_LIBRO_DEFAULT      = 10      # profundidad del libro que se analiza
MIN_SCORE_DEFAULT          = 0.60
MAX_SPREAD_PCT_DEFAULT     = 0.0015  # 0.15% -- por encima de esto se considera libro poco liquido
MIN_NOTIONAL_OPERADO_DEF   = 15000.0 # USDT operados durante la ventana, minimo para confiar en el flujo
MIN_SNAPSHOTS_DEFAULT      = 8       # minimo de fotos del libro antes de calcular nada


# ---------------------------------------------------------------------------
# BUFFER POR SIMBOLO
# ---------------------------------------------------------------------------
class _Ventana:
    """Buffer acotado por TIEMPO (no por cantidad) con los snapshots del
    libro de ordenes y los trades ejecutados de un simbolo."""

    def __init__(self, ventana_seg):
        self.ventana_seg = ventana_seg
        self.snapshots = deque()   # (ts, mid, microprice, obi, spread_pct)
        self.trades = deque()      # (ts, side, qty)
        self.first_ts = None

    def _purgar(self, ahora):
        limite = ahora - self.ventana_seg
        while self.snapshots and self.snapshots[0][0] < limite:
            self.snapshots.popleft()
        while self.trades and self.trades[0][0] < limite:
            self.trades.popleft()
        # una vez que se purga el primer snapshot original, la "antiguedad"
        # de la ventana debe recalcularse desde lo que queda
        if self.snapshots:
            self.first_ts = self.snapshots[0][0]

    def agregar_snapshot(self, ts, mid, microprice, obi, spread_pct):
        if self.first_ts is None:
            self.first_ts = ts
        self.snapshots.append((ts, mid, microprice, obi, spread_pct))
        self._purgar(ts)

    def agregar_trades(self, ts, trades_side_qty):
        for side, qty in trades_side_qty:
            self.trades.append((ts, side, qty))
        self._purgar(ts)

    def antiguedad(self, ahora):
        if self.first_ts is None:
            return 0.0
        return ahora - self.first_ts


# ---------------------------------------------------------------------------
# PROCESAMIENTO DE UN SNAPSHOT DEL LIBRO
# ---------------------------------------------------------------------------
def _procesar_orderbook(ob, niveles=10):
    """A partir de un order book crudo de ccxt (dict con 'bids'/'asks', cada
    nivel [precio, cantidad, ...]) calcula: precio medio, microprice, OBI y
    spread relativo. Devuelve None si el libro viene vacio/invalido."""
    bids = (ob.get("bids") or [])[:niveles]
    asks = (ob.get("asks") or [])[:niveles]
    if not bids or not asks:
        return None

    best_bid, bid_qty0 = bids[0][0], bids[0][1]
    best_ask, ask_qty0 = asks[0][0], asks[0][1]
    if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
        return None

    mid = (best_bid + best_ask) / 2.0
    spread_pct = (best_ask - best_bid) / mid if mid else 0.0

    vol_bid = sum(lvl[1] for lvl in bids if len(lvl) >= 2)
    vol_ask = sum(lvl[1] for lvl in asks if len(lvl) >= 2)
    total = vol_bid + vol_ask
    obi = (vol_bid - vol_ask) / total if total > 0 else 0.0

    denom = bid_qty0 + ask_qty0
    # microprice: pondera cada precio por la cantidad del lado CONTRARIO en
    # el mejor nivel -- si hay mucha cantidad esperando del lado ask, el
    # "verdadero" precio de ejecucion esta mas cerca del bid, y viceversa.
    microprice = (best_bid * ask_qty0 + best_ask * bid_qty0) / denom if denom > 0 else mid

    return mid, microprice, obi, spread_pct


# ---------------------------------------------------------------------------
# METRICAS SOBRE TODA LA VENTANA
# ---------------------------------------------------------------------------
def _calcular_metricas(ventana: "_Ventana"):
    snaps = list(ventana.snapshots)
    if len(snaps) < 2:
        return None

    mids = [s[1] for s in snaps]
    microprices = [s[2] for s in snaps]
    obis = [s[3] for s in snaps]
    spreads = [s[4] for s in snaps]

    mid_prom = sum(mids) / len(mids)
    obi_prom = sum(obis) / len(obis)

    mitad = len(obis) // 2
    obi_tendencia = 0.0
    if mitad:
        obi_tendencia = (sum(obis[mitad:]) / len(obis[mitad:])) - (sum(obis[:mitad]) / mitad)

    micro_drift = sum((mp - m) / m for mp, m in zip(microprices, mids) if m) / len(mids)

    momentum = (mids[-1] - mids[0]) / mids[0] if mids[0] else 0.0

    # consistencia: de los movimientos sub-tramo a sub-tramo, que fraccion
    # fue en la MISMA direccion que el movimiento neto de la ventana
    signo_neto = 1 if momentum > 0 else (-1 if momentum < 0 else 0)
    pasos = []
    for i in range(1, len(mids)):
        d = mids[i] - mids[i - 1]
        if d != 0:
            pasos.append(1 if d > 0 else -1)
    consistencia = (sum(1 for p in pasos if p == signo_neto) / len(pasos)) if pasos and signo_neto else 0.0

    spread_prom = sum(spreads) / len(spreads)

    trades = list(ventana.trades)
    vol_buy = sum(q for (_, side, q) in trades if side == "buy")
    vol_sell = sum(q for (_, side, q) in trades if side == "sell")
    vol_total = vol_buy + vol_sell
    flow_imbalance = (vol_buy - vol_sell) / vol_total if vol_total > 0 else 0.0
    notional_operado = vol_total * mid_prom

    return {
        "mid_prom": mid_prom,
        "obi_prom": obi_prom,
        "obi_tendencia": obi_tendencia,
        "micro_drift": micro_drift,
        "momentum": momentum,
        "consistencia": consistencia,
        "spread_prom": spread_prom,
        "flow_imbalance": flow_imbalance,
        "vol_total": vol_total,
        "notional_operado": notional_operado,
        "n_snapshots": len(snaps),
        "n_trades": len(trades),
    }


# ---------------------------------------------------------------------------
# SCORE DE CONFIRMACION PARA UN LADO (LONG/SHORT)
# ---------------------------------------------------------------------------
def _puntaje_signo(valor, lado, escala):
    """Convierte una metrica con signo (positivo favorece LONG, negativo
    favorece SHORT) en un puntaje 0-1 para el lado pedido, con una sigmoide
    suave (evita que un solo valor extremo domine el score entero)."""
    x = (valor / escala) if escala else valor
    if lado == "SHORT":
        x = -x
    try:
        return 1.0 / (1.0 + math.exp(-x * 4.0))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


PESOS_DEFAULT = {
    "flow": 0.30,          # flujo de trades ejecutados (lo mas "real": dinero que ya se movio)
    "obi": 0.20,           # desequilibrio del libro
    "obi_tendencia": 0.15, # si el desequilibrio se esta acentuando
    "micro_drift": 0.15,   # hacia donde "empuja" el microprice
    "momentum": 0.10,      # direccion del precio en la ventana
    "consistencia": 0.10,  # que tan parejo fue ese movimiento
}


def _score_confirmacion(m, lado, pesos=None):
    pesos = pesos or PESOS_DEFAULT

    s_flow = _puntaje_signo(m["flow_imbalance"], lado, escala=0.35)
    s_obi = _puntaje_signo(m["obi_prom"], lado, escala=0.35)
    s_obi_t = _puntaje_signo(m["obi_tendencia"], lado, escala=0.20)
    s_micro = _puntaje_signo(m["micro_drift"], lado, escala=0.0015)
    s_mom = _puntaje_signo(m["momentum"], lado, escala=0.01)

    signo_esperado = 1 if lado == "LONG" else -1
    signo_mom = 1 if m["momentum"] > 0 else (-1 if m["momentum"] < 0 else 0)
    if signo_mom == signo_esperado:
        s_cons = 0.5 + 0.5 * m["consistencia"]
    elif signo_mom == 0:
        s_cons = 0.5
    else:
        s_cons = 0.5 - 0.5 * m["consistencia"]

    detalle = {
        "s_flow": round(s_flow, 3), "s_obi": round(s_obi, 3),
        "s_obi_tendencia": round(s_obi_t, 3), "s_micro_drift": round(s_micro, 3),
        "s_momentum": round(s_mom, 3), "s_consistencia": round(s_cons, 3),
    }
    score = (pesos["flow"] * s_flow + pesos["obi"] * s_obi +
             pesos["obi_tendencia"] * s_obi_t + pesos["micro_drift"] * s_micro +
             pesos["momentum"] * s_mom + pesos["consistencia"] * s_cons)
    return round(score, 4), detalle


# ---------------------------------------------------------------------------
# CONFIRMADOR -- clase principal, con hilo de fondo propio
# ---------------------------------------------------------------------------
class ConfirmadorLibroOrdenes:
    """
    Uso tipico:

        confirmador = ConfirmadorLibroOrdenes(ex, ventana_seg=600, min_score=0.6)
        confirmador.iniciar(["BTC/USDT:USDT", "ETH/USDT:USDT"])
        ...
        estado = confirmador.evaluar("BTC/USDT:USDT", "LONG")
        if estado["confirma"]:
            # abrir la posicion

    O de forma mas directa, filtrando una lista completa de candidatos
    (dicts con al menos 'symbol' y 'lado'):

        confirmadas = confirmador.filtrar_confirmados(candidatos)

    El objeto mantiene UN hilo de fondo que va rotando entre todos los
    simbolos "en observacion", pidiendo su libro de ordenes + trades
    recientes cada `poll_secs` segundos por simbolo. Las metricas se
    recalculan sobre una ventana movil de `ventana_seg` segundos.
    """

    def __init__(self, exchange, ventana_seg=VENTANA_SEG_DEFAULT,
                 poll_secs=POLL_SECS_DEFAULT, niveles_libro=NIVELES_LIBRO_DEFAULT,
                 min_score=MIN_SCORE_DEFAULT, max_spread_pct=MAX_SPREAD_PCT_DEFAULT,
                 min_notional_operado=MIN_NOTIONAL_OPERADO_DEF,
                 min_snapshots=MIN_SNAPSHOTS_DEFAULT, pesos=None):
        self.exchange = exchange
        self.ventana_seg = ventana_seg
        self.poll_secs = poll_secs
        self.niveles_libro = niveles_libro
        self.min_score = min_score
        self.max_spread_pct = max_spread_pct
        self.min_notional_operado = min_notional_operado
        self.min_snapshots = min_snapshots
        self.pesos = pesos or PESOS_DEFAULT

        self._ventanas = {}
        self._last_trade_ts = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._avisador = AvisadorThrottled(intervalo_seg=30)

    # -- ciclo de vida -------------------------------------------------------

    def iniciar(self, symbols=None):
        """Agrega simbolos nuevos a observacion (si ya estaban, no reinicia
        su ventana) y arranca el hilo de fondo si aun no esta corriendo."""
        with self._lock:
            for sym in (symbols or []):
                if sym not in self._ventanas:
                    self._ventanas[sym] = _Ventana(self.ventana_seg)
                    self._last_trade_ts[sym] = 0
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def detener_simbolos(self, symbols):
        """Deja de observar y libera memoria de simbolos que ya no
        interesan (ej. dejaron de ser candidatos tras un re-escaneo)."""
        with self._lock:
            for sym in symbols:
                self._ventanas.pop(sym, None)
                self._last_trade_ts.pop(sym, None)

    def stop(self):
        self._stop_event.set()

    # -- hilo de fondo ---------------------------------------------------------

    def _loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                symbols = list(self._ventanas.keys())
            if not symbols:
                self._stop_event.wait(self.poll_secs)
                continue
            extra_espera = 0.0
            for sym in symbols:
                if self._stop_event.is_set():
                    break
                extra_espera = max(extra_espera, self._poll_symbol(sym))
                # pequeña pausa entre simbolos: cuida el rate limit del
                # exchange cuando hay muchas candidatas en observacion a la vez
                time.sleep(0.15)
            # espera lo que falte para completar el ciclo de poll_secs (mas
            # una espera extra si el exchange respondio con rate limit, para
            # no seguir insistiendo y empeorar el bloqueo);
            # si hay muchos simbolos y ya se gasto mas de poll_secs recorriendolos,
            # sigue directo a la proxima vuelta sin dormir de mas
            self._stop_event.wait(max(0.2, self.poll_secs) + extra_espera)

    def _poll_symbol(self, sym) -> float:
        """Refresca libro de ordenes + trades de UN simbolo. Devuelve la
        espera extra (segundos) recomendada antes del proximo ciclo, segun
        el peor error visto (0 si no hubo problemas o fueron menores)."""
        extra_espera = 0.0
        ahora = time.time()
        try:
            ob = self.exchange.fetch_order_book(sym, limit=self.niveles_libro)
            procesado = _procesar_orderbook(ob, niveles=self.niveles_libro)
            if procesado:
                mid, microprice, obi, spread_pct = procesado
                with self._lock:
                    v = self._ventanas.get(sym)
                    if v:
                        v.agregar_snapshot(ahora, mid, microprice, obi, spread_pct)
        except Exception as e:
            # error de red / rate limit puntual: se avisa (throttled) y se
            # reintenta en el siguiente poll -- no se detiene la observacion
            # del simbolo por un fallo pasajero.
            extra_espera = max(extra_espera,
                                self._avisador.avisar(sym, e, contexto="[Libro de ordenes] libro de"))

        try:
            trades = self.exchange.fetch_trades(sym, limit=100)
            last_ts = self._last_trade_ts.get(sym, 0)
            nuevos = [t for t in trades if (t.get("timestamp") or 0) > last_ts]
            if nuevos:
                self._last_trade_ts[sym] = max(t.get("timestamp") or 0 for t in nuevos)
                side_qty = [(t.get("side") or "buy", float(t.get("amount") or 0)) for t in nuevos]
                with self._lock:
                    v = self._ventanas.get(sym)
                    if v:
                        v.agregar_trades(ahora, side_qty)
        except Exception as e:
            extra_espera = max(extra_espera,
                                self._avisador.avisar(sym, e, contexto="[Libro de ordenes] trades de"))
        return extra_espera

    # -- evaluacion --------------------------------------------------------

    def evaluar(self, symbol, lado):
        """Devuelve el veredicto ACTUAL de confirmacion para `symbol` en el
        `lado` pedido ("LONG"/"SHORT"). Si el simbolo todavia no estaba en
        observacion, lo agrega ahora y devuelve 'listo': False (la ventana
        recien empieza, hay que volver a preguntar mas adelante)."""
        with self._lock:
            v = self._ventanas.get(symbol)
        if v is None:
            self.iniciar([symbol])
            return {"listo": False, "confirma": False, "score": 0.0,
                    "antiguedad_seg": 0.0, "motivo": "iniciando observacion del libro de ordenes"}

        ahora = time.time()
        with self._lock:
            antiguedad = v.antiguedad(ahora)
            metricas = _calcular_metricas(v)

        if metricas is None or metricas["n_snapshots"] < self.min_snapshots:
            n = metricas["n_snapshots"] if metricas else 0
            return {"listo": False, "confirma": False, "score": 0.0,
                    "antiguedad_seg": antiguedad,
                    "motivo": f"acumulando datos del libro ({n}/{self.min_snapshots} snapshots)"}

        # se exige al menos el 80% de la ventana pedida para dar un veredicto
        # definitivo (si se exigiera el 100% siempre, un pequeño hueco de red
        # reiniciaria la espera indefinidamente)
        listo = antiguedad >= self.ventana_seg * 0.8

        score, detalle = _score_confirmacion(metricas, lado, self.pesos)

        spread_ok = metricas["spread_prom"] <= self.max_spread_pct
        notional_ok = metricas["notional_operado"] >= self.min_notional_operado
        liquidez_ok = spread_ok and notional_ok

        confirma = bool(listo and liquidez_ok and score >= self.min_score)

        if not listo:
            motivo = f"ventana incompleta ({antiguedad:.0f}/{self.ventana_seg}s)"
        elif not spread_ok:
            motivo = f"spread demasiado ancho ({metricas['spread_prom']*100:.3f}%)"
        elif not notional_ok:
            motivo = f"muy poco volumen operado (${metricas['notional_operado']:.0f})"
        elif not confirma:
            motivo = f"score insuficiente ({score:.2f} < {self.min_score:.2f})"
        else:
            motivo = "confirmado"

        return {
            "listo": listo, "confirma": confirma, "score": score,
            "antiguedad_seg": antiguedad, "spread_prom_pct": metricas["spread_prom"] * 100,
            "notional_operado": metricas["notional_operado"], "liquidez_ok": liquidez_ok,
            "metricas": metricas, "detalle": detalle, "motivo": motivo,
        }

    def filtrar_confirmados(self, candidatos, campo_symbol="symbol", campo_lado="lado"):
        """Dada una lista de candidatos (dicts con al menos 'symbol' y
        'lado'), se asegura de que todos queden en observacion y devuelve
        SOLO los que YA se confirmaron en este momento. Los que aun no
        completan la ventana o no confirman simplemente no aparecen esta
        vuelta -- pero siguen siendo observados y pueden confirmar en una
        llamada posterior (ideal para pegarlo dentro de un loop que ya se
        repite cada poco tiempo, como el tick del engine)."""
        symbols = [c[campo_symbol] for c in candidatos]
        self.iniciar(symbols)
        confirmados = []
        for c in candidatos:
            estado = self.evaluar(c[campo_symbol], c.get(campo_lado, "LONG"))
            c["_confirmacion"] = estado
            if estado["confirma"]:
                confirmados.append(c)
        return confirmados

    def estado_resumen(self, candidatos, campo_symbol="symbol", campo_lado="lado"):
        """Igual que filtrar_confirmados pero devuelve el detalle de TODOS
        (confirmados y no), util para mostrar progreso en el dashboard."""
        symbols = [c[campo_symbol] for c in candidatos]
        self.iniciar(symbols)
        out = []
        for c in candidatos:
            estado = self.evaluar(c[campo_symbol], c.get(campo_lado, "LONG"))
            out.append((c, estado))
        return out
