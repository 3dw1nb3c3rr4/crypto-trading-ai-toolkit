#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trading_engine.py
==================
Motor de trading (PAPER y REAL) para el Sistema Binance Futures Techo/Suelo.

Dos modos de operacion:

  MODO 1 - CanastaEngine ("canasta"):
    El usuario dice cuantas LONG y cuantas SHORT quiere. El motor toma esa
    cantidad de las mejores candidatas (ya rankeadas por Cerebro 1 + Cerebro 2,
    es decir por prob_exito_3d) y abre TODAS las posiciones de una vez.
    TP/SL puede ser:
      - "individual": cada posicion se cierra sola cuando toca su propio TP
        o SL (igual que cualquier bot normal).
      - "grupal": toda la canasta se cierra JUNTA cuando la ganancia/perdida
        NETA acumulada de todas las posiciones abiertas llega al TP o SL
        grupal (en USDT) -- igual logica que el backtest de canasta del
        notebook (TP_GLOBAL_USD / SL_GLOBAL_USD).

  MODO 2 - PiramideEngine ("piramide"):
    Se abre 1 sola posicion: la candidata con mayor probabilidad de
    cumplirse. Se definen TP% y SL% fijos desde la entrada. Cuando el precio
    se mueve a favor un "trigger_pct" (por defecto = SL%), el stop de ESA
    posicion se mueve a un colchon de ganancia asegurada ("lock_pct", por
    defecto 1%) y ESE evento habilita abrir UNA posicion nueva mas (la
    siguiente mejor candidata disponible, con las MISMAS condiciones de
    TP/SL). Se repite: cada gatillo tocado = +1 posicion nueva habilitada.
    Cerrar una posicion (por TP o por el stop ya asegurado) NO habilita otra
    por si sola -- SOLO tocar el gatillo lo hace. Esto limita el riesgo: solo
    se agrega exposicion nueva cuando ya hay "colchon" ganado en otra parte.
    Si no hay ninguna posicion abierta, se abre una (arranca con 1 slot).

Ambos modos comparten:
  - Ejecucion REAL en Binance Futures USDT-M via ccxt, en Hedge Mode
    (positionSide LONG/SHORT), igual patron que BinanceRealTrader.
  - CSV de trades cerrados y de posiciones activas.

⚠️ El modo REAL coloca ordenes con dinero real. Revisa bien apalancamiento,
nocional y TP/SL antes de confirmar.
"""

import os
import csv
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import ccxt

from red_reintentos import AvisadorThrottled, buscar_posicion_abierta

_avisador_precios = AvisadorThrottled(intervalo_seg=30)

DEFAULT_FEE = 0.0005  # 0.05% por lado (comision estandar Binance Futures Taker)

PAPER_SLIPPAGE = 0.0002   # [parche paper] deslizamiento adverso por pata en paper (0.02%)


def precio_vivo_paper(exchange, symbol, side):
    """[parche paper] Precio de entrada simulado EN VIVO: LONG compra al ask, SHORT vende al bid, mas slippage adverso.
    Devuelve None si no hay precio en vivo (en ese caso NO se debe abrir con un precio viejo)."""
    try:
        t = exchange.fetch_ticker(symbol)
        last = t.get("last") or t.get("close")
        px = (t.get("ask") if side == "LONG" else t.get("bid")) or last
        if not px:
            return None
        return float(px) * (1 + PAPER_SLIPPAGE) if side == "LONG" else float(px) * (1 - PAPER_SLIPPAGE)
    except Exception:
        return None


def precio_salida_paper(price, side):
    """[parche paper] Salida simulada con slippage adverso: LONG vende un poco mas bajo, SHORT recompra un poco mas alto."""
    return price * (1 - PAPER_SLIPPAGE) if side == "LONG" else price * (1 + PAPER_SLIPPAGE)

TRADES_LOG_CANASTA   = "trades_canasta.csv"
TRADES_LOG_PIRAMIDE  = "trades_piramide.csv"
ACTIVE_POSITIONS_CSV = "active_positions.csv"


# ---------------------------------------------------------------------------
# Utilidades compartidas: CSV de trades cerrados / posiciones activas
# ---------------------------------------------------------------------------

def _init_csv(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "ts_entry", "ts_close", "symbol", "side", "modo",
                "notional", "entry_price", "exit_price",
                "tp_price", "sl_price", "motivo",
                "pnl_pct", "pnl_usd", "fees", "prob_exito", "result",
            ])


def _log_trade(path: str, trade: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            trade["ts_entry"], trade["ts_close"].isoformat(), trade["symbol"],
            trade["side"], trade["modo"],
            f'{trade["notional"]:.2f}', f'{trade["entry_price"]:.8g}',
            f'{trade["exit_price"]:.8g}', f'{trade["tp_price"]:.8g}',
            f'{trade["sl_price"]:.8g}', trade["motivo"],
            f'{trade["pnl_pct"]:.3f}', f'{trade["pnl_usd"]:.6f}',
            f'{trade["fees"]:.6f}', f'{trade.get("prob_exito", 0):.3f}',
            trade["result"],
        ])


def _save_active_positions_csv(all_positions: Dict[str, dict]) -> None:
    rows = []
    now_utc = datetime.now(timezone.utc)
    for pos in all_positions.values():
        elapsed_min = (now_utc - pos["entry_ts"]).total_seconds() / 60
        rows.append({
            "symbol":       pos["symbol"],
            "side":         pos["side"],
            "modo":         pos["modo"],
            "notional":     f'{pos["notional"]:.2f}',
            "entry_price":  f'{pos["entry_price"]:.8g}',
            "tp_price":     f'{pos["tp_price"]:.8g}',
            "sl_price":     f'{pos["sl_price"]:.8g}',
            "trigger_hit":  pos.get("trigger_hit", False),
            "prob_exito":   f'{pos.get("prob_exito", 0):.3f}',
            "min_abierto":  f'{elapsed_min:.1f}',
        })
    try:
        with open(ACTIVE_POSITIONS_CSV, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write("symbol,side,modo,notional,entry_price,tp_price,sl_price,"
                         "trigger_hit,prob_exito,min_abierto\n")
    except Exception:
        pass  # no interrumpir el bot por un error de CSV


# ---------------------------------------------------------------------------
# TRAILING STOP -- logica UNICA y thread-safe, compartida por los 3 lugares
# que la usan
# ---------------------------------------------------------------------------
# Esta funcion la llaman: CanastaEngine._check_individual (tick principal,
# cada 15s) y, en main_binance_techosuelo.py, los hilos _actualizar_precios
# (cada 3s) y _monitorear_trailing_activo (cada 1s) -- los 3 pueden evaluar
# la MISMA posicion casi al mismo tiempo, en hilos distintos.
#
# Antes esta logica estaba copiada por separado en esos 3 lugares. Eso traia
# dos problemas:
#   1) Ninguna copia tomaba ningun lock: dos hilos podian leer y escribir
#      pos["trailing_peak"] / pos["trailing_activo"] al mismo tiempo, y la
#      escritura de uno (con un precio mas viejo, si a ese hilo le toco
#      consultar el exchange un instante antes) podia pisar la del otro. El
#      resultado practico es un trailing MAS FLOJO que el % configurado --
#      el bot le da mas margen del debido antes de cerrar y devuelve mas
#      ganancia de la que deberia.
#   2) Al mantenerse 3 copias, una correccion aplicada en una (el reset de
#      SHORT cuando el precio revierte por completo hasta el entry) se quedo
#      sin replicar en las otras 2, y tampoco existia para LONG en ninguna
#      de las 3 -- quedaba la posibilidad (con trailing_pct configurado
#      grande en relacion al TP) de cerrar por debajo del entry etiquetado
#      como "TP_TRAILING".
#
# Ahora hay una sola implementacion, simetrica entre LONG y SHORT, y con un
# lock propio de cada posicion (`pos["_trailing_lock"]`) que serializa toda
# lectura+escritura de esos dos campos: un solo hilo a la vez puede tocarlos
# para una posicion dada (posiciones distintas no se bloquean entre si).

def evaluar_trailing(pos: dict, price: float, trailing_pct: Optional[float],
                      log=print) -> Optional[str]:
    """Evalua (y aplica) el trailing stop de UNA posicion para el precio
    `price` ya conocido (no vuelve a golpear la red). Debe llamarse DESPUES
    de comprobar el SL -- el SL nunca tiene trailing, se evalua aparte y
    tiene prioridad siempre.

    LONG: al tocar tp_price se activa el trailing y se guarda el maximo
    alcanzado (peak). Cierra cuando el precio retrocede trailing_pct% desde
    ese maximo. Si el precio vuelve a caer hasta el entry (la subida se
    revirtio del todo), se desactiva el trailing y queda el SL original
    protegiendo.
    SHORT: simetrico, con minimo (valley) en vez de maximo.

    side/entry/tp se leen con fallback de nombres de campo para ser
    compatible con los distintos formatos de `pos` usados en el proyecto
    (trading_engine.py siempre usa "side"/"entry_price"/"tp_price"; algunos
    lugares de main_binance_techosuelo.py contemplan alias legados).

    Devuelve "TP_TRAILING" si corresponde cerrar la posicion ahora mismo, o
    None si sigue abierta (todavia no activo el trailing, o esta activo pero
    no llego al umbral de cierre)."""
    if not trailing_pct:
        return None
    side = (pos.get("side") or pos.get("lado") or pos.get("direction")
            or pos.get("type") or "").upper()
    entry = pos.get("entry_price") or pos.get("open_price")
    tp    = pos.get("tp_price") or pos.get("take_profit")
    if side not in ("LONG", "SHORT") or not entry or not tp:
        return None

    lock = pos.get("_trailing_lock")
    if lock is None:
        lock = pos["_trailing_lock"] = threading.Lock()

    with lock:
        if not pos.get("trailing_activo"):
            activa = (price >= tp) if side == "LONG" else (price <= tp)
            if not activa:
                return None
            pos["trailing_activo"] = True
            pos["trailing_peak"] = price
            log(f"    [TRAILING] {side} {pos.get('symbol', '?')} activado @ "
                f"{price:.6g} ({'peak' if side == 'LONG' else 'valley'} inicial)")
            return None  # se acaba de activar -- el cierre se evalua desde el proximo precio

        peak = pos["trailing_peak"]
        if side == "LONG":
            if price <= entry:
                # Reversion completa hasta el entry: se apaga el trailing y
                # el SL original (evaluado aparte, antes de llamar aca) es
                # quien protege de aca en adelante.
                pos["trailing_activo"] = False
                pos["trailing_peak"] = None
                return None
            if price > peak:
                pos["trailing_peak"] = peak = price
            umbral = peak * (1.0 - trailing_pct)
            if price <= umbral:
                return "TP_TRAILING"
        else:  # SHORT
            if price >= entry:
                pos["trailing_activo"] = False
                pos["trailing_peak"] = None
                return None
            if price < peak:
                pos["trailing_peak"] = peak = price
            umbral = peak * (1.0 + trailing_pct)
            if price >= umbral:
                return "TP_TRAILING"
    return None


# ---------------------------------------------------------------------------
# EJECUCION REAL (Binance Futures, Hedge Mode) -- compartida por ambos modos
# ---------------------------------------------------------------------------

class BinanceFuturesExecutor:
    """
    Envoltorio delgado sobre ccxt para abrir/cerrar posiciones REALES en
    Binance Futures USDT-M, en Hedge Mode (positionSide LONG/SHORT) -- mismo
    patron que BinanceRealTrader en tu bot de XRP, pero generico para
    cualquier simbolo.

    Requiere que tu cuenta de Binance Futures este en HEDGE MODE (no
    One-way). Si no lo esta, actívalo una vez en la app de Binance o con
    exchange.set_position_mode(True) antes de operar.
    """

    def __init__(self, api_key: str, api_secret: str, leverage: int = 20):
        self.leverage = leverage
        self.exchange = ccxt.binance({
            "apiKey":        api_key,
            "secret":        api_secret,
            "enableRateLimit": True,
            "options":       {"defaultType": "future"},
        })
        self.exchange.load_markets()
        # Cache del ultimo precio conocido por simbolo (se actualiza en cada
        # get_price() exitoso) -- se usa como ultimo recurso en cerrar() si
        # el precio de fill no viene en la respuesta de la orden Y la
        # consulta de fallback tambien falla por red, para no perder el
        # registro del trade solo porque no se pudo determinar el precio
        # exacto (la orden ya se ejecuto en el exchange de todas formas).
        self._ultimo_precio: Dict[str, float] = {}

    def get_price(self, symbol: str) -> float:
        t = self.exchange.fetch_ticker(symbol)
        price = float(t.get("last") or t.get("close"))
        self._ultimo_precio[symbol] = price
        return price

    def get_available_balance(self) -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0))

    def set_leverage(self, symbol: str) -> None:
        try:
            self.exchange.set_leverage(self.leverage, symbol)
        except Exception as e:
            print(f"  [!] Aviso set_leverage {symbol}: {e}")

    def _amount(self, symbol: str, notional: float, price: float) -> float:
        cantidad = notional / price
        try:
            return float(self.exchange.amount_to_precision(symbol, cantidad))
        except Exception:
            return round(cantidad, 4)

    def _buscar_posicion_abierta(self, symbol: str, side: str, notional: float) -> Optional[dict]:
        """Consulta fetch_positions() para ver si el exchange ya tiene una
        posicion abierta del lado `side` para `symbol` (aunque la orden de
        apertura haya fallado por red antes de que llegara la respuesta).
        Delega en `buscar_posicion_abierta()` de red_reintentos (logica
        centralizada, reutilizable por cualquier executor del proyecto).

        SOLO se llama cuando `abrir()` captura una excepcion -- es seguro
        porque fetch_positions() es lectura pura (idempotente, no duplica
        ordenes). Si `abrir()` termina sin excepcion, no hay ambiguedad y
        no se necesita consultar nada."""
        return buscar_posicion_abierta(self.exchange, symbol, side, notional)

    def sincronizar_posiciones(self, posiciones: Dict[str, str]):
        """Lee el estado REAL de una o varias posiciones en el exchange, en
        una sola llamada a fetch_positions(). `posiciones`: {symbol: side}
        con el side ORIGINAL que el bot cree que tiene cada una.

        Se usa para dos cosas:
          1) Detectar cuando el usuario agrega/quita tamano manualmente en
             el exchange sobre una posicion que el bot gestiona -- Binance
             ya calcula solo el entryPrice PROMEDIO cada vez que cambia el
             tamano, asi que alcanza con leerlo y adoptarlo.
          2) Detectar cuando el usuario CIERRA manualmente una posicion en
             el exchange (fuera del bot) -- si ya no aparece con cantidad >0
             para ese symbol/side, se interpreta como cerrada.

        Devuelve una tupla (actualizadas, cerradas):
          - actualizadas: {symbol: {"entry_price":.., "cantidad":..}} para
            las que siguen abiertas en el exchange.
          - cerradas: set() de symbols que se pidieron consultar pero YA NO
            tienen posicion real abierta con ese side -- se cerraron por
            fuera del bot.

        Si la consulta al exchange falla por completo (red/rate-limit),
        devuelve (None, None) -- asi quien llama sabe que NO se pudo
        determinar nada y no debe borrar ni tocar ninguna posicion (una
        falla de red no es lo mismo que "la posicion ya no existe")."""
        if not posiciones:
            return {}, set()
        try:
            todas = self.exchange.fetch_positions(list(posiciones.keys()))
        except Exception as e:
            print(f"  [!] No se pudo resincronizar posiciones con el exchange: {e}")
            return None, None

        actualizadas: Dict[str, dict] = {}
        for p in todas:
            symbol = p.get("symbol")
            side_esperado = posiciones.get(symbol)
            if side_esperado is None:
                continue
            p_side = str(p.get("side") or p.get("positionSide") or "").lower()
            if p_side != side_esperado.lower():
                continue
            p_contracts = float(p.get("contracts") or p.get("contractSize") or 0)
            p_entry = float(p.get("entryPrice") or p.get("entry_price") or 0)
            if p_contracts <= 0 or p_entry <= 0:
                continue  # aparece pero en 0 -> se trata como cerrada abajo
            actualizadas[symbol] = {"entry_price": p_entry, "cantidad": p_contracts}

        cerradas = {s for s in posiciones if s not in actualizadas}
        return actualizadas, cerradas

    def tiene_posicion_externa(self, symbol: str) -> Optional[dict]:
        """Consulta si YA existe una posicion real abierta en el exchange
        para `symbol` (en cualquier side). Se usa para que el bot NUNCA
        abra una operacion sobre una moneda que el usuario ya tiene operada
        manualmente por fuera del bot -- debe esperar a que la cierres para
        poder operarla.

        Devuelve {"side":.., "cantidad":..} si encuentra una posicion con
        contratos > 0, o None si el simbolo esta libre.

        Si la consulta al exchange falla (red/rate-limit), devuelve un
        marcador de bloqueo generico en vez de None -- mejor saltear este
        intento de apertura (se reintenta en el proximo escaneo) que abrir
        a ciegas sin saber si hay algo ya operado ahi.

        NOTA: este metodo no distingue "posicion que abriste vos a mano" de
        "posicion que el bot mismo abrio en una sesion anterior y perdio de
        vista al reiniciarse" -- el bot no recarga su estado de posiciones
        abiertas contra el exchange al arrancar. Si reiniciás el proceso
        con posiciones reales del propio bot todavia abiertas, este chequeo
        tambien las va a tratar como "externas" y no las va a volver a
        tocar. Es el comportamiento SEGURO (nunca duplica una orden ni
        pisa nada), pero no equivale a "recuperar" esa posicion perdida --
        para eso haria falta una funcion aparte que reconstruya
        open_positions desde el exchange al arrancar."""
        try:
            positions = self.exchange.fetch_positions([symbol])
        except Exception as e:
            print(f"  [!] No se pudo verificar si {symbol} ya tiene posicion externa "
                  f"abierta: {e}. Se omite la apertura por seguridad (se reintenta en "
                  f"el proximo escaneo).")
            return {"side": "desconocido", "cantidad": 0.0}
        for p in positions:
            p_contracts = float(p.get("contracts") or p.get("contractSize") or 0)
            if p_contracts > 0:
                p_side = str(p.get("side") or p.get("positionSide") or "").upper()
                return {"side": p_side, "cantidad": p_contracts}
        return None

    def apalancamiento_disponible(self, symbol: str, notional: float) -> bool:
        """Verifica en el momento de abrir que el exchange realmente
        soporte `self.leverage` para ESTE nocional especifico -- puede
        diferir del nocional base si el bot uso sizing dinamico (0.6x-1.6x
        segun el score), asi que el chequeo inicial de candidatos (hecho
        con el nocional base en main_binance_techosuelo.py) es solo una
        primera pasada; este es el chequeo final y autoritativo, con el
        dato mas fresco posible (no usa cache, se llama una sola vez por
        apertura).

        Binance limita el apalancamiento maximo permitido segun el TRAMO
        de nocional en el que cae la posicion -- una moneda puede anunciar
        un maximo alto que solo aplica a nocionales chicos. Si el tramo
        que corresponde a este nocional no alcanza `self.leverage`,
        devuelve False y NO se abre la posicion."""
        try:
            tiers = self.exchange.fetch_market_leverage_tiers(symbol)
        except Exception as e:
            print(f"  [!] No se pudo verificar apalancamiento de {symbol} antes de "
                  f"abrir: {e}. Se omite la apertura por seguridad.")
            return False
        if not tiers:
            return False
        tier = None
        for t in tiers:
            min_n = float(t.get("minNotional") or 0)
            max_n_raw = t.get("maxNotional")
            max_n = float(max_n_raw) if max_n_raw is not None else float("inf")
            if min_n <= notional < max_n:
                tier = t
                break
        if tier is None:
            tier = tiers[0]
        max_lev_tramo = float(tier.get("maxLeverage") or 0)
        if max_lev_tramo < self.leverage:
            print(f"  [!] Salteando apertura de {symbol}: el tramo de apalancamiento "
                  f"para un nocional de ${notional:.2f} solo permite {max_lev_tramo:.0f}x, "
                  f"pero esta configurado {self.leverage}x.")
            return False
        return True

    def abrir(self, symbol: str, side: str, notional: float, es_reentrada: bool = False) -> Optional[dict]:
        """side: 'LONG' o 'SHORT'. Devuelve entry_price/cantidad/order_id o
        None si fallo (nocional invalido, error de red, moneda ya operada
        manualmente, apalancamiento no disponible, etc.).

        Antes de mandar la orden, chequea:
          1) que no exista ya una posicion abierta (de cualquier side) para
             ese simbolo que el bot no este gestionando -- SALVO que
             `es_reentrada=True`: en ese caso el llamador YA SABE que hay una
             posicion abierta en ese symbol/side (es la posicion propia del
             bot que se esta promediando/reforzando), asi que este chequeo
             se omite a proposito -- si no, la reentrada nunca podria
             ejecutarse porque siempre "ya existe una posicion".
          2) que el exchange realmente soporte el apalancamiento configurado
             para ESTE nocional (los tramos de apalancamiento varian segun
             el tamano de la posicion).
        Si cualquiera de las dos falla, no abre nada y devuelve None; el
        candidato simplemente se salta.

        Si la llamada al exchange falla por red / rate-limit al mandar la
        orden, antes de rendirse consulta si el exchange ya tiene la
        posicion abierta (la orden puede haberse ejecutado aunque la
        respuesta se perdio). Si la encuentra, la adopta y el bot la
        registra normalmente."""
        if not es_reentrada:
            externa = self.tiene_posicion_externa(symbol)
            if externa is not None:
                if externa["side"] == "desconocido":
                    pass  # ya se logueo el motivo dentro de tiene_posicion_externa()
                else:
                    print(f"  [!] Salteando apertura de {side} {symbol}: ya existe una "
                          f"posicion {externa['side']} abierta en el exchange para esa "
                          f"moneda ({externa['cantidad']:.6g} contratos) que el bot no "
                          f"gestiona. Esperando a que la cierres para poder operarla.")
                return None
        if not self.apalancamiento_disponible(symbol, notional):
            return None  # el mensaje ya se logueo dentro de apalancamiento_disponible()
        try:
            self.set_leverage(symbol)
            price = self.get_price(symbol)
            cantidad = self._amount(symbol, notional, price)
            if cantidad <= 0:
                print(f"  [!] Cantidad invalida para {symbol}, se omite.")
                return None
            order_side = "buy" if side == "LONG" else "sell"
            order = self.exchange.create_order(
                symbol=symbol, type="market", side=order_side,
                amount=cantidad, params={"positionSide": side},
            )
            fill_price = float(order.get("average") or price)
            print(f"    ✅ {side} REAL {symbol} @ {fill_price:.6g} "
                  f"(cantidad {cantidad}, ID {order.get('id')})")
            return {"entry_price": fill_price, "cantidad": cantidad, "order_id": order.get("id")}
        except Exception as e:
            print(f"  [!] Error abriendo {side} {symbol}: {e}")
            # ----------------------------------------------------------------
            # RECUPERACION POR FALLO DE RED EN APERTURA
            # La orden puede haberse ejecutado en el exchange aunque la
            # respuesta se perdio por el camino. Consultamos si ya hay una
            # posicion abierta compatible antes de descartar la operacion.
            # ----------------------------------------------------------------
            adoptado = self._buscar_posicion_abierta(symbol, side, notional)
            if adoptado is not None:
                return adoptado
            print(f"  [!] No se encontro posicion preexistente para {symbol} {side} "
                  f"-- la apertura se descarta.")
            return None

    def cerrar(self, symbol: str, side: str, cantidad: float) -> Optional[float]:
        """Cierra una posicion real. `side` es el lado ORIGINAL de la
        posicion (LONG/SHORT), no la direccion de la orden de cierre.

        IMPORTANTE: en Binance Futures Hedge Mode (positionSide=LONG/SHORT)
        el parametro `reduceOnly` NO se acepta -- el exchange ya interpreta
        una orden con side opuesto sobre el mismo positionSide como un
        cierre. Mandar reduceOnly=True junto con positionSide dispara el
        error -1106 ("Parameter 'reduceonly' sent when not required.") y la
        orden de cierre nunca se ejecuta, dejando la posicion abierta para
        siempre en modo real (aunque en paper trading "funcione" porque ahi
        no se valida contra la API real).

        IMPORTANTE 2: la `cantidad` se redondea con `amount_to_precision()`
        antes de mandarla -- si trae mas decimales que el stepSize que
        Binance define para ese simbolo, la orden se rechaza con el error
        -1111 "Precision is over the maximum defined for this asset" y el
        cierre nunca llega a ejecutarse.

        IMPORTANTE 3: una vez que `create_order()` responde sin excepcion,
        la orden de cierre YA SE EJECUTO en Binance -- lo unico que puede
        faltar es el precio exacto de fill para el registro interno. Por
        eso la busqueda del precio de salida (fallback si `average` viene
        vacio, tipico en ordenes de mercado inmediatas) esta en su PROPIO
        try/except: si `get_price()` falla por red justo en ese momento, NO
        se debe interpretar como que el cierre fallo (ya se ejecuto) -- se
        cae a un ultimo precio conocido en vez de relanzar la excepcion y
        que el bot crea erroneamente que la posicion sigue abierta."""
        try:
            try:
                cantidad_precisa = float(self.exchange.amount_to_precision(symbol, cantidad))
            except Exception:
                cantidad_precisa = cantidad  # fallback si amount_to_precision no esta disponible
            order_side = "sell" if side == "LONG" else "buy"
            order = self.exchange.create_order(
                symbol=symbol, type="market", side=order_side,
                amount=cantidad_precisa, params={"positionSide": side},
            )
        except Exception as e:
            print(f"  [!] Error cerrando {side} {symbol}: {e}")
            return None

        # La orden ya se ejecuto en el exchange llegado a este punto -- lo
        # de aca abajo es solo para obtener el precio de fill mas preciso
        # posible, nunca debe hacer que se reporte el cierre como fallido.
        exit_price = order.get("average")
        if not exit_price:
            try:
                exit_price = self.get_price(symbol)
            except Exception as e:
                print(f"  [!] Cierre de {symbol} SI se ejecuto en el exchange, pero no se "
                      f"pudo confirmar el precio exacto de salida ({e}). Se usa el ultimo "
                      f"precio conocido como aproximacion.")
                exit_price = self._ultimo_precio.get(symbol, 0.0)
        exit_price = float(exit_price or 0.0)
        self._ultimo_precio[symbol] = exit_price
        print(f"    ✅ Cerrado REAL {side} {symbol} @ {exit_price:.6g}")
        return exit_price


# ---------------------------------------------------------------------------
# MODO 1 -- CANASTA
# ---------------------------------------------------------------------------

class CanastaEngine:
    """MODO 1: abre N_LONG + N_SHORT posiciones a la vez. Ver docstring del
    modulo para la explicacion completa de "individual" vs "grupal"."""

    def __init__(self, exchange, n_long: int, n_short: int,
                 notional_per_trade: float, leverage: int,
                 tp_pct: float, sl_pct: float, tp_sl_mode: str = "individual",
                 tp_global_usd: float = None, sl_global_usd: float = None,
                 paper: bool = True, real_executor: BinanceFuturesExecutor = None,
                 initial_balance: float = 1000.0,
                 trailing_pct: float = None):
        self.exchange = exchange  # instancia ccxt (solo se usa para leer precios)
        self.n_long = n_long
        self.n_short = n_short
        self.notional = notional_per_trade
        self.leverage = leverage
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.tp_sl_mode = tp_sl_mode  # "individual" | "grupal"
        self.tp_global_usd = tp_global_usd
        self.sl_global_usd = sl_global_usd
        self.paper = paper
        self.real_executor = real_executor
        # trailing_pct: None = sin trailing (cierre clasico al tocar TP);
        # float = % de retroceso desde el precio extremo para disparar el cierre.
        # Solo aplica en modo "individual" -- en modo "grupal" el TP es en USDT
        # totales de toda la canasta y el trailing no tiene sentido por moneda.
        self.trailing_pct = trailing_pct

        self.wallet_balance = initial_balance
        self.pnl_realizado = 0.0
        self.open_positions: Dict[str, dict] = {}
        self.closed_trades: List[dict] = []
        # Protege _cerrar(): hay 2 hilos concurrentes que pueden detectar
        # TP/SL sobre la MISMA posicion casi al mismo tiempo -- el hilo
        # rapido de precios (cada ~3s, cierra de inmediato en modo TP/SL
        # individual) y el tick principal del engine (cada 15s, check_tp_sl
        # / modo grupal). Sin este lock, ambos podrian pasar el chequeo
        # "pos = open_positions.get(symbol)" antes de que ninguno la haya
        # sacado del dict todavia, y mandar 2 ordenes de cierre reales a
        # Binance para la misma posicion.
        self._close_lock = threading.Lock()

        _init_csv(TRADES_LOG_CANASTA)

    # -- apertura ------------------------------------------------------------

    def abrir_canasta(self, candidatos_long: List[dict], candidatos_short: List[dict]) -> None:
        """candidatos_*: listas de dicts (symbol, last_close, prob_exito_3d,
        ...) ya ordenadas por probabilidad descendente. Abre las primeras
        n_long / n_short."""
        elegidos_long = candidatos_long[: self.n_long]
        elegidos_short = candidatos_short[: self.n_short]
        print(f"\n  Abriendo canasta: {len(elegidos_long)} LONG + {len(elegidos_short)} SHORT "
              f"(TP/SL {self.tp_sl_mode})")
        for c in elegidos_long:
            self._abrir(c["symbol"], "LONG", c)
        for c in elegidos_short:
            self._abrir(c["symbol"], "SHORT", c)
        _save_active_positions_csv(self.open_positions)

    def _abrir(self, symbol: str, side: str, cand: dict) -> None:
        # Si el candidato trae "notional_override" (ej. sizing dinamico del
        # Cerebro 3 en main.py), se usa ese monto; si no, el nocional fijo
        # configurado para el motor. Mantiene compatibilidad total con el
        # comportamiento anterior cuando no se envia el campo.
        notional_usado = cand.get("notional_override", self.notional)
        entry_price = float(cand["last_close"])
        cantidad = order_id = None
        if self.paper:  # [parche paper]
            px_vivo = precio_vivo_paper(self.exchange, symbol, side)
            if px_vivo is None:
                print(f"    [paper] sin precio en vivo para {symbol}: no se abre (evita entrada con precio viejo)")
                return
            entry_price = px_vivo
        if not self.paper:
            r = self.real_executor.abrir(symbol, side, notional_usado)
            if r is None:
                return
            entry_price, cantidad, order_id = r["entry_price"], r["cantidad"], r["order_id"]

        if side == "LONG":
            tp_price = entry_price * (1 + self.tp_pct)
            sl_price = entry_price * (1 - self.sl_pct)
        else:
            tp_price = entry_price * (1 - self.tp_pct)
            sl_price = entry_price * (1 + self.sl_pct)

        entry_fee = notional_usado * DEFAULT_FEE
        if self.paper:
            self.wallet_balance -= entry_fee

        pos = {
            "symbol": symbol, "side": side, "modo": "canasta",
            "notional": notional_usado, "entry_price": entry_price,
            "tp_price": tp_price, "sl_price": sl_price,
            "entry_ts": datetime.now(timezone.utc),
            "prob_exito": cand.get("prob_exito_3d", cand.get("confidence", 0)),
            "entry_fee": entry_fee, "cantidad": cantidad, "order_id": order_id,
            # Trailing stop: trailing_activo se pone True cuando el precio toca
            # el tp_price por primera vez (en modo individual con trailing_pct
            # configurado). trailing_peak guarda el precio extremo mas favorable
            # alcanzado desde que se activo el trailing (maximo para LONG,
            # minimo para SHORT). El cierre se dispara cuando el precio
            # retrocede trailing_pct% desde ese extremo.
            "trailing_activo": False,
            "trailing_peak":   None,
        }
        self.open_positions[symbol] = pos
        print(f"    {side} {symbol} @ {entry_price:.6g}  TP {tp_price:.6g}  SL {sl_price:.6g}"
              f"{'  (nocional ' + str(round(notional_usado,2)) + ')' if 'notional_override' in cand else ''}")

    # -- pnl / cierre ---------------------------------------------------------

    def _pnl_pct(self, pos: dict, price: float) -> float:
        if pos["side"] == "LONG":
            return price / pos["entry_price"] - 1.0
        return 1.0 - price / pos["entry_price"]

    def _cerrar(self, symbol: str, exit_price: float, motivo: str) -> None:
        with self._close_lock:
            pos = self.open_positions.get(symbol)
            if pos is None:
                return  # ya la cerro el otro hilo mientras esperabamos el lock

            if not self.paper:
                real_exit = self.real_executor.cerrar(symbol, pos["side"], pos["cantidad"])
                if real_exit is None:
                    # El cierre en el exchange fallo -- la posicion SIGUE ABIERTA
                    # en Binance. NO la sacamos del dict para que el bot continue
                    # vigilandola en el proximo tick en lugar de dejarla huerfana.
                    print(f"  [!] Cierre REAL de {symbol} fallo -- posicion conservada "
                          f"como activa para reintentar en el proximo tick.")
                    return
                exit_price = real_exit
            else:  # [parche paper]
                exit_price = precio_salida_paper(exit_price, pos["side"])

            # Cierre confirmado (paper o real exitoso): ahora si sacamos la posicion.
            self.open_positions.pop(symbol)

            pnl_pct = self._pnl_pct(pos, exit_price)
            gross = pos["notional"] * pnl_pct
            exit_fee = pos["notional"] * DEFAULT_FEE
            net_pnl = gross - pos["entry_fee"] - exit_fee

            if self.paper:
                self.wallet_balance += gross - exit_fee
            self.pnl_realizado += net_pnl

            trade = {
                "ts_entry": pos["entry_ts"].isoformat(), "ts_close": datetime.now(timezone.utc),
                "symbol": symbol, "side": pos["side"], "modo": "canasta",
                "notional": pos["notional"], "entry_price": pos["entry_price"],
                "exit_price": exit_price, "tp_price": pos["tp_price"], "sl_price": pos["sl_price"],
                "motivo": motivo, "pnl_pct": pnl_pct * 100, "pnl_usd": net_pnl,
                "fees": pos["entry_fee"] + exit_fee, "prob_exito": pos.get("prob_exito", 0),
                "result": "WIN" if net_pnl > 0 else "LOSS",
            }
        self.closed_trades.append(trade)
        _log_trade(TRADES_LOG_CANASTA, trade)
        _save_active_positions_csv(self.open_positions)
        print(f"    Cerrado {pos['side']} {symbol} @ {exit_price:.6g} ({motivo}) "
              f"PnL {net_pnl:+.4f} USDT")

    # -- monitoreo --------------------------------------------------------------

    def _cerrar_detectada_externamente(self, symbol: str) -> None:
        """El usuario cerro esta posicion manualmente en el exchange (fuera
        del bot) -- ya no aparece en fetch_positions(). La sacamos del panel
        de posiciones activas para que no quede fantasma ahi para siempre
        (el bot seguiria pidiendo su precio y evaluando TP/SL de una
        posicion que ya no existe).

        El PnL que se registra es APROXIMADO: el bot no vio la orden de
        cierre real ni su fill exacto, asi que usa el ultimo precio de
        mercado disponible como proxy del precio de salida. Se deja
        marcado con motivo "CIERRE_MANUAL" en el CSV para que quede claro
        que no es un cierre por TP/SL del propio bot."""
        with self._close_lock:
            pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return
        try:
            exit_price = self.exchange.fetch_ticker(symbol)["last"]
        except Exception:
            exit_price = pos["entry_price"]  # ultimo recurso si ni el precio se puede leer

        pnl_pct = self._pnl_pct(pos, exit_price)
        gross = pos["notional"] * pnl_pct
        exit_fee = pos["notional"] * DEFAULT_FEE
        net_pnl = gross - pos["entry_fee"] - exit_fee
        self.pnl_realizado += net_pnl

        trade = {
            "ts_entry": pos["entry_ts"].isoformat(), "ts_close": datetime.now(timezone.utc),
            "symbol": symbol, "side": pos["side"], "modo": pos.get("modo", "canasta"),
            "notional": pos["notional"], "entry_price": pos["entry_price"],
            "exit_price": exit_price, "tp_price": pos["tp_price"], "sl_price": pos["sl_price"],
            "motivo": "CIERRE_MANUAL", "pnl_pct": pnl_pct * 100, "pnl_usd": net_pnl,
            "fees": pos["entry_fee"] + exit_fee, "prob_exito": pos.get("prob_exito", 0),
            "result": "WIN" if net_pnl > 0 else "LOSS",
        }
        self.closed_trades.append(trade)
        _log_trade(TRADES_LOG_CANASTA, trade)
        _save_active_positions_csv(self.open_positions)
        print(f"\n  🗑️  {pos['side']} {symbol} ya no existe en el exchange -- se cerro "
              f"manualmente por fuera del bot. Removida del panel de activas "
              f"(PnL aprox {net_pnl:+.4f} USDT, exit ~{exit_price:.6g} aproximado).")

    def _resincronizar_con_exchange(self) -> None:
        """Antes de chequear TP/SL: compara cada posicion que el bot recuerda
        contra el estado REAL de esa posicion en el exchange. Dos casos:

          1) La posicion sigue abierta pero cambio de tamano (el usuario
             agrego o quito manualmente en Binance) -> adopta el
             entry_price PROMEDIO real y la cantidad real, y recalcula
             TP/SL sobre ese nuevo entry_price usando los mismos %
             configurados desde el arranque, para que el bot cierre la
             posicion COMPLETA al tocar TP/SL.
          2) La posicion ya no existe en el exchange (el usuario la cerro
             manualmente) -> se remueve del panel de posiciones activas
             (ver `_cerrar_detectada_externamente`).

        Solo aplica en modo real. Si la consulta al exchange falla (red),
        no se toca nada -- una falla de red no significa que la posicion
        se haya cerrado."""
        if self.paper or self.real_executor is None or not self.open_positions:
            return
        posiciones = {s: p["side"] for s, p in self.open_positions.items()}
        reales, cerradas = self.real_executor.sincronizar_posiciones(posiciones)
        if reales is None:
            return  # fallo de red al consultar -- no se pudo determinar nada

        for symbol in cerradas:
            self._cerrar_detectada_externamente(symbol)

        for symbol, real in reales.items():
            pos = self.open_positions[symbol]
            cambio_rel = abs(real["cantidad"] - pos["cantidad"]) / max(pos["cantidad"], 1e-9)
            if cambio_rel < 0.01:
                continue

            entry_ant, cant_ant, notional_ant = pos["entry_price"], pos["cantidad"], pos["notional"]
            pos["entry_price"] = real["entry_price"]
            pos["cantidad"] = real["cantidad"]
            pos["notional"] = real["cantidad"] * real["entry_price"]
            pos["entry_fee"] = pos["notional"] * DEFAULT_FEE

            if pos["side"] == "LONG":
                pos["tp_price"] = pos["entry_price"] * (1 + self.tp_pct)
                pos["sl_price"] = pos["entry_price"] * (1 - self.sl_pct)
            else:
                pos["tp_price"] = pos["entry_price"] * (1 - self.tp_pct)
                pos["sl_price"] = pos["entry_price"] * (1 + self.sl_pct)

            signo = "+" if real["cantidad"] > cant_ant else "-"
            print(f"\n  🔄 Resincronizado {pos['side']} {symbol}: tamano cambiado "
                  f"manualmente en el exchange ({signo}{abs(real['cantidad']-cant_ant):.6g}).")
            print(f"     Entry {entry_ant:.6g} -> {pos['entry_price']:.6g} (promedio real de "
                  f"Binance)  |  Nocional {notional_ant:.2f} -> {pos['notional']:.2f} USDT")
            print(f"     TP/SL recalculados sobre el nuevo entry: "
                  f"TP {pos['tp_price']:.6g}  SL {pos['sl_price']:.6g}")

    def check_tp_sl(self) -> None:
        """Revisa TP/SL (individual o grupal) de todas las posiciones abiertas
        contra el precio actual de cada simbolo."""
        if not self.open_positions:
            return
        self._resincronizar_con_exchange()
        precios = {}
        for symbol in list(self.open_positions.keys()):
            try:
                precios[symbol] = self.exchange.fetch_ticker(symbol)["last"]
            except Exception as e:
                _avisador_precios.avisar(symbol, e, contexto="[Canasta] refrescando precio de")
                continue

        if self.tp_sl_mode == "grupal":
            self._check_grupal(precios)
        else:
            self._check_individual(precios)

    def _check_individual(self, precios: dict) -> None:
        """Revisa TP/SL de cada posicion. Si trailing_pct esta configurado,
        el TP activa el modo trailing en vez de cerrar de inmediato:
          - LONG: cuando precio >= tp_price, activa trailing y guarda el
            maximo alcanzado. Cierra solo cuando retrocede trailing_pct%
            desde ese maximo.
          - SHORT: cuando precio <= tp_price, activa trailing y guarda el
            minimo alcanzado. Cierra solo cuando rebota trailing_pct% desde
            ese minimo.
        El SL siempre funciona como antes, sin trailing."""
        to_close = []
        for symbol, pos in self.open_positions.items():
            price = precios.get(symbol)
            if price is None:
                continue

            # ── SL: evaluar SIEMPRE primero, sin trailing ─────────────────
            if pos["side"] == "LONG" and price <= pos["sl_price"]:
                to_close.append((symbol, price, "SL"))
                continue
            if pos["side"] == "SHORT" and price >= pos["sl_price"]:
                to_close.append((symbol, price, "SL"))
                continue

            # ── TP con Trailing Stop (logica compartida y thread-safe, ─────
            #    ver evaluar_trailing() mas arriba en este mismo archivo) ──
            if self.trailing_pct and self.tp_sl_mode == "individual":
                hit = evaluar_trailing(pos, price, self.trailing_pct)
                if hit:
                    to_close.append((symbol, price, hit))
            else:
                # TP clásico: cierre inmediato al tocar tp_price
                if pos["side"] == "LONG" and price >= pos["tp_price"]:
                    to_close.append((symbol, price, "TP"))
                elif pos["side"] == "SHORT" and price <= pos["tp_price"]:
                    to_close.append((symbol, price, "TP"))

        for symbol, price, motivo in to_close:
            self._cerrar(symbol, price, motivo)

    def _check_grupal(self, precios: dict) -> None:
        """PnL NETO no-realizado de TODA la canasta abierta; si llega al TP o
        al SL grupal (en USDT), se cierra TODA la canasta de una vez.

        IMPORTANTE: se descuenta el round-trip completo de fees -- la
        entry_fee que YA se pago de verdad al abrir (pos["entry_fee"]) MAS
        la exit_fee estimada de cerrar ahora. Restar solo la exit_fee
        subestima el costo real y hace que el TP grupal se dispare antes de
        lo que realmente corresponde (o el SL grupal se retrase), igual que
        ya se hace correctamente en el cierre individual (`_cerrar`)."""
        pnl_no_realizado = 0.0
        for symbol, pos in self.open_positions.items():
            price = precios.get(symbol)
            if price is None:
                continue
            pnl_pct = self._pnl_pct(pos, price)
            exit_fee_est = pos["notional"] * DEFAULT_FEE
            pnl_no_realizado += pos["notional"] * pnl_pct - pos["entry_fee"] - exit_fee_est

        if self.tp_global_usd and pnl_no_realizado >= self.tp_global_usd:
            print(f"\n  🎯 TP GRUPAL alcanzado (+{pnl_no_realizado:.2f} USDT) -> cerrando canasta")
            self._cerrar_todo(precios, "TP_GRUPAL")
        elif self.sl_global_usd and pnl_no_realizado <= -abs(self.sl_global_usd):
            print(f"\n  🛑 SL GRUPAL alcanzado ({pnl_no_realizado:.2f} USDT) -> cerrando canasta")
            self._cerrar_todo(precios, "SL_GRUPAL")

    def _cerrar_todo(self, precios: dict, motivo: str) -> None:
        for symbol in list(self.open_positions.keys()):
            price = precios.get(symbol, self.open_positions[symbol]["entry_price"])
            self._cerrar(symbol, price, motivo)


# ---------------------------------------------------------------------------
# MODO 2 -- PIRAMIDE
# ---------------------------------------------------------------------------

class PiramideEngine:
    """MODO 2: abre 1 posicion, y cada gatillo tocado habilita abrir otra.
    Ver docstring del modulo para la explicacion completa del mecanismo."""

    def __init__(self, exchange, notional_per_trade: float, leverage: int,
                 tp_pct: float, sl_pct: float, trigger_pct: float = None,
                 lock_pct: float = 0.01, max_positions: int = 0,
                 paper: bool = True, real_executor: BinanceFuturesExecutor = None,
                 initial_balance: float = 1000.0):
        self.exchange = exchange
        self.notional = notional_per_trade
        self.leverage = leverage
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trigger_pct = trigger_pct if trigger_pct is not None else sl_pct
        self.lock_pct = lock_pct
        self.max_positions = max_positions  # 0 = sin limite
        self.paper = paper
        self.real_executor = real_executor

        self.wallet_balance = initial_balance
        self.pnl_realizado = 0.0
        self.open_positions: Dict[str, dict] = {}
        self.closed_trades: List[dict] = []
        self.pending_slots = 1  # arranca con 1 slot -> "si no hay operaciones activas, se abre una"
        # Protege _cerrar() -- mismo motivo que en CanastaEngine (ver ahi el
        # docstring completo): el hilo rapido de precios y el tick principal
        # pueden detectar TP/SL sobre la misma posicion casi al mismo tiempo.
        self._close_lock = threading.Lock()

        _init_csv(TRADES_LOG_PIRAMIDE)

    # -- ciclo principal --------------------------------------------------------

    def tick(self, candidatos: List[dict]) -> None:
        """Un ciclo: revisa gatillos/TP/SL de lo abierto y, si hay slots
        pendientes, abre nuevas posiciones con las mejores candidatas aun no
        abiertas. `candidatos`: lista con symbol/last_close/lado/prob_exito_3d,
        ya ordenada por probabilidad descendente."""
        self._check_positions()
        self._abrir_pendientes(candidatos)

    def _abrir_pendientes(self, candidatos: List[dict]) -> None:
        # Re-arme de seguridad: la piramide NUNCA deberia quedar "muerta". Si
        # no hay ninguna posicion abierta y tampoco quedan slots pendientes
        # (ej. las ultimas posiciones cerraron en perdida sin llegar a
        # disparar su gatillo), se repone 1 slot para que siga buscando
        # entradas -- exactamente la misma regla que al arrancar el motor
        # ("si no hay operaciones activas, se abre una").
        if not self.open_positions and self.pending_slots <= 0:
            self.pending_slots = 1
            print("    [Piramide] Sin posiciones abiertas y sin slots pendientes -> "
                  "re-armando 1 slot para seguir buscando entradas.")

        while self.pending_slots > 0:
            if self.max_positions and len(self.open_positions) >= self.max_positions:
                break
            cand = next((c for c in candidatos if c["symbol"] not in self.open_positions), None)
            if cand is None:
                break  # no hay mas candidatas disponibles por ahora
            self._abrir(cand)
            self.pending_slots -= 1

    def _abrir(self, cand: dict) -> None:
        symbol = cand["symbol"]
        side = cand["lado"]  # "LONG" o "SHORT"
        notional_usado = cand.get("notional_override", self.notional)
        entry_price = float(cand["last_close"])
        cantidad = order_id = None
        if self.paper:  # [parche paper]
            px_vivo = precio_vivo_paper(self.exchange, symbol, side)
            if px_vivo is None:
                print(f"    [paper] sin precio en vivo para {symbol}: no se abre (evita entrada con precio viejo)")
                return
            entry_price = px_vivo
        if not self.paper:
            r = self.real_executor.abrir(symbol, side, notional_usado)
            if r is None:
                return
            entry_price, cantidad, order_id = r["entry_price"], r["cantidad"], r["order_id"]

        if side == "LONG":
            tp_price = entry_price * (1 + self.tp_pct)
            sl_price = entry_price * (1 - self.sl_pct)
        else:
            tp_price = entry_price * (1 - self.tp_pct)
            sl_price = entry_price * (1 + self.sl_pct)

        entry_fee = notional_usado * DEFAULT_FEE
        if self.paper:
            self.wallet_balance -= entry_fee

        pos = {
            "symbol": symbol, "side": side, "modo": "piramide",
            "notional": notional_usado, "entry_price": entry_price,
            "tp_price": tp_price, "sl_price": sl_price,
            "entry_ts": datetime.now(timezone.utc),
            "prob_exito": cand.get("prob_exito_3d", cand.get("confidence", 0)),
            "entry_fee": entry_fee, "trigger_hit": False,
            "cantidad": cantidad, "order_id": order_id,
        }
        self.open_positions[symbol] = pos
        print(f"    [Piramide] Abierta {side} {symbol} @ {entry_price:.6g}  "
              f"TP {tp_price:.6g}  SL {sl_price:.6g}  (prob {pos['prob_exito']:.2f})")
        _save_active_positions_csv(self.open_positions)

    # -- monitoreo / gatillo ------------------------------------------------------

    def _cerrar_detectada_externamente(self, symbol: str) -> None:
        """El usuario cerro esta posicion manualmente en el exchange (fuera
        del bot). Se remueve del panel de activas -- misma logica que en
        CanastaEngine (ver ahi el docstring completo sobre el PnL
        aproximado). IMPORTANTE: no toca `pending_slots` -- el cierre
        normal de la piramide (`_cerrar`) tampoco lo hace, el slot solo se
        libera cuando se dispara el gatillo de aseguramiento (ver
        `_revisar_gatillo`), asi que un cierre externo se trata igual que
        cualquier otro cierre en ese sentido. Si esta era la ultima
        posicion abierta y no quedan slots pendientes, el re-armado de
        seguridad en `_abrir_pendientes` se encarga de reponer 1 slot en el
        siguiente paso del tick."""
        with self._close_lock:
            pos = self.open_positions.pop(symbol, None)
        if pos is None:
            return
        try:
            exit_price = self.exchange.fetch_ticker(symbol)["last"]
        except Exception:
            exit_price = pos["entry_price"]

        pnl_pct = (exit_price / pos["entry_price"] - 1.0 if pos["side"] == "LONG"
                   else 1.0 - exit_price / pos["entry_price"])
        gross = pos["notional"] * pnl_pct
        exit_fee = pos["notional"] * DEFAULT_FEE
        net_pnl = gross - pos["entry_fee"] - exit_fee
        self.pnl_realizado += net_pnl

        trade = {
            "ts_entry": pos["entry_ts"].isoformat(), "ts_close": datetime.now(timezone.utc),
            "symbol": symbol, "side": pos["side"], "modo": "piramide",
            "notional": pos["notional"], "entry_price": pos["entry_price"],
            "exit_price": exit_price, "tp_price": pos["tp_price"], "sl_price": pos["sl_price"],
            "motivo": "CIERRE_MANUAL", "pnl_pct": pnl_pct * 100, "pnl_usd": net_pnl,
            "fees": pos["entry_fee"] + exit_fee, "prob_exito": pos.get("prob_exito", 0),
            "result": "WIN" if net_pnl > 0 else "LOSS",
        }
        self.closed_trades.append(trade)
        _log_trade(TRADES_LOG_PIRAMIDE, trade)
        _save_active_positions_csv(self.open_positions)
        print(f"\n  🗑️  [Piramide] {pos['side']} {symbol} ya no existe en el exchange -- se "
              f"cerro manualmente por fuera del bot. Removida del panel de activas "
              f"(PnL aprox {net_pnl:+.4f} USDT, exit ~{exit_price:.6g} aproximado).")

    def _resincronizar_con_exchange(self) -> None:
        """Misma logica que en CanastaEngine (ver docstring alla): adopta
        entry_price/cantidad reales del exchange cuando el usuario agrega o
        quita tamano manualmente, recalcula TP/SL sobre el nuevo entry, y
        remueve del panel las posiciones que ya no existen en el exchange
        (cerradas manualmente).

        Diferencia con Canasta: si el gatillo de aseguramiento YA se disparo
        para esta posicion (`trigger_hit`), el SL no vuelve al SL original --
        se reubica el stop asegurado (`lock_pct`) sobre el nuevo entry_price,
        para no perder la proteccion de "stop en ganancia" que ya se habia
        ganado antes del resincronizado."""
        if self.paper or self.real_executor is None or not self.open_positions:
            return
        posiciones = {s: p["side"] for s, p in self.open_positions.items()}
        reales, cerradas = self.real_executor.sincronizar_posiciones(posiciones)
        if reales is None:
            return  # fallo de red al consultar -- no se pudo determinar nada

        for symbol in cerradas:
            self._cerrar_detectada_externamente(symbol)

        for symbol, real in reales.items():
            pos = self.open_positions[symbol]
            cambio_rel = abs(real["cantidad"] - pos["cantidad"]) / max(pos["cantidad"], 1e-9)
            if cambio_rel < 0.01:
                continue

            entry_ant, cant_ant, notional_ant = pos["entry_price"], pos["cantidad"], pos["notional"]
            pos["entry_price"] = real["entry_price"]
            pos["cantidad"] = real["cantidad"]
            pos["notional"] = real["cantidad"] * real["entry_price"]
            pos["entry_fee"] = pos["notional"] * DEFAULT_FEE

            if pos["side"] == "LONG":
                pos["tp_price"] = pos["entry_price"] * (1 + self.tp_pct)
                pos["sl_price"] = (pos["entry_price"] * (1 + self.lock_pct) if pos["trigger_hit"]
                                    else pos["entry_price"] * (1 - self.sl_pct))
            else:
                pos["tp_price"] = pos["entry_price"] * (1 - self.tp_pct)
                pos["sl_price"] = (pos["entry_price"] * (1 - self.lock_pct) if pos["trigger_hit"]
                                    else pos["entry_price"] * (1 + self.sl_pct))

            signo = "+" if real["cantidad"] > cant_ant else "-"
            print(f"\n  🔄 [Piramide] Resincronizado {pos['side']} {symbol}: tamano cambiado "
                  f"manualmente en el exchange ({signo}{abs(real['cantidad']-cant_ant):.6g}).")
            print(f"     Entry {entry_ant:.6g} -> {pos['entry_price']:.6g} (promedio real de "
                  f"Binance)  |  Nocional {notional_ant:.2f} -> {pos['notional']:.2f} USDT")
            print(f"     TP/SL recalculados sobre el nuevo entry: "
                  f"TP {pos['tp_price']:.6g}  SL {pos['sl_price']:.6g}"
                  f"{'  (stop asegurado reubicado)' if pos['trigger_hit'] else ''}")

    def _revisar_gatillo(self, symbol: str, price: float) -> bool:
        """Revisa (y aplica) SOLO el gatillo de aseguramiento de una posicion,
        dado un precio que YA se conoce (sin volver a golpear la red). Se
        expone por separado de `_check_positions` para que el hilo rapido de
        refresco de precios (cada 3s) pueda invocarlo ANTES de cerrar una
        posicion por TP/SL -- si no se hace asi, una posicion que sube rapido
        y cruza el gatillo y el TP dentro de la misma ventana de 15s del tick
        del engine se cerraria por TP sin que el gatillo llegara a dispararse
        nunca, dejando `pending_slots` congelado en 0 para siempre (la
        piramide deja de reponer posiciones). Devuelve True si el gatillo se
        disparo en esta llamada."""
        pos = self.open_positions.get(symbol)
        if pos is None or pos.get("trigger_hit"):
            return False
        side = pos["side"]
        pnl_pct = (price / pos["entry_price"] - 1.0 if side == "LONG"
                   else 1.0 - price / pos["entry_price"])
        if pnl_pct < self.trigger_pct:
            return False
        lock_price = (pos["entry_price"] * (1 + self.lock_pct) if side == "LONG"
                      else pos["entry_price"] * (1 - self.lock_pct))
        pos["sl_price"] = lock_price
        pos["trigger_hit"] = True
        self.pending_slots += 1
        print(f"    [Piramide] {symbol} toco el gatillo (+{pnl_pct*100:.2f}%) -> "
              f"stop asegurado en {lock_price:.6g} (+{self.lock_pct*100:.1f}%). "
              f"Nuevo slot habilitado.")
        return True

    def _check_positions(self) -> None:
        if not self.open_positions:
            return
        self._resincronizar_con_exchange()
        precios = {}
        for symbol in list(self.open_positions.keys()):
            try:
                precios[symbol] = self.exchange.fetch_ticker(symbol)["last"]
            except Exception as e:
                _avisador_precios.avisar(symbol, e, contexto="[Piramide] refrescando precio de")
                continue

        to_close = []
        for symbol, pos in self.open_positions.items():
            price = precios.get(symbol)
            if price is None:
                continue
            side = pos["side"]

            # 1) Gatillo de aseguramiento (se dispara UNA sola vez por posicion)
            self._revisar_gatillo(symbol, price)

            # 2) TP / SL (el SL puede ya estar en zona de ganancia si el gatillo se disparo)
            if side == "LONG":
                if price >= pos["tp_price"]:
                    to_close.append((symbol, price, "TP"))
                elif price <= pos["sl_price"]:
                    to_close.append((symbol, price, "SL_ASEGURADO" if pos["trigger_hit"] else "SL"))
            else:
                if price <= pos["tp_price"]:
                    to_close.append((symbol, price, "TP"))
                elif price >= pos["sl_price"]:
                    to_close.append((symbol, price, "SL_ASEGURADO" if pos["trigger_hit"] else "SL"))

        for symbol, price, motivo in to_close:
            self._cerrar(symbol, price, motivo)

    def _cerrar(self, symbol: str, exit_price: float, motivo: str) -> None:
        with self._close_lock:
            pos = self.open_positions.get(symbol)
            if pos is None:
                return  # ya la cerro el otro hilo mientras esperabamos el lock

            if not self.paper:
                real_exit = self.real_executor.cerrar(symbol, pos["side"], pos["cantidad"])
                if real_exit is None:
                    # El cierre en el exchange fallo -- la posicion SIGUE ABIERTA
                    # en Binance. NO la sacamos del dict para que el bot continue
                    # vigilandola en el proximo tick en lugar de dejarla huerfana.
                    # IMPORTANTE: tampoco sumamos pending_slots aqui -- el slot se
                    # sumara cuando el cierre sea exitoso en un proximo intento.
                    print(f"  [!] [Piramide] Cierre REAL de {symbol} fallo -- posicion "
                          f"conservada como activa para reintentar en el proximo tick.")
                    return
                exit_price = real_exit
            else:  # [parche paper]
                exit_price = precio_salida_paper(exit_price, pos["side"])

            # Cierre confirmado (paper o real exitoso): ahora si sacamos la posicion.
            self.open_positions.pop(symbol)

            pnl_pct = (exit_price / pos["entry_price"] - 1.0 if pos["side"] == "LONG"
                       else 1.0 - exit_price / pos["entry_price"])
            gross = pos["notional"] * pnl_pct
            exit_fee = pos["notional"] * DEFAULT_FEE
            net_pnl = gross - pos["entry_fee"] - exit_fee

            if self.paper:
                self.wallet_balance += gross - exit_fee
            self.pnl_realizado += net_pnl

            trade = {
                "ts_entry": pos["entry_ts"].isoformat(), "ts_close": datetime.now(timezone.utc),
                "symbol": symbol, "side": pos["side"], "modo": "piramide",
                "notional": pos["notional"], "entry_price": pos["entry_price"],
                "exit_price": exit_price, "tp_price": pos["tp_price"], "sl_price": pos["sl_price"],
                "motivo": motivo, "pnl_pct": pnl_pct * 100, "pnl_usd": net_pnl,
                "fees": pos["entry_fee"] + exit_fee, "prob_exito": pos.get("prob_exito", 0),
                "result": "WIN" if net_pnl > 0 else "LOSS",
            }
        self.closed_trades.append(trade)
        _log_trade(TRADES_LOG_PIRAMIDE, trade)
        _save_active_positions_csv(self.open_positions)
        print(f"    [Piramide] Cerrado {pos['side']} {symbol} @ {exit_price:.6g} "
              f"({motivo}) PnL {net_pnl:+.4f} USDT | slots pendientes: {self.pending_slots}")
