#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
red_reintentos.py
==================
Utilidades compartidas para manejar fallos de red / rate-limit del exchange
de forma consistente en todo el proyecto (escaneo, refresco de precios,
libro de ordenes). Dos ideas centrales:

  1) Distinguir el TIPO de error (rate limit / bloqueo temporal vs. caida de
     red generica vs. otro error) para esperar el tiempo adecuado antes de
     reintentar -- un rate limit necesita esperar MAS que un timeout de red
     comun, o el reintento inmediato solo empeora el bloqueo.

  2) Dos modos de reintento:
       - `reintentar_acotado`: para llamadas puntuales/interactivas (ej. el
         usuario esta esperando en un menu) -- reintenta un numero limitado
         de veces y despues relanza la excepcion para que quien llama decida.
       - `reintentar_persistente`: para procesos de fondo que NO deben
         morir nunca por un problema de red pasajero (el auto-escaneo, por
         ejemplo) -- reintenta PARA SIEMPRE con backoff exponencial acotado,
         mostrando el error cada vez, hasta que la llamada funcione.

No se usa para llamadas que ESCRIBEN (colocar/cancelar ordenes reales): ahi
un reintento ciego puede terminar mandando la misma orden dos veces si la
primera si se ejecuto pero la respuesta se perdio por el camino. Para esas,
usar `reintentar_acotado` con pocos intentos y dejar que el que llama decida
que hacer con el error (nunca `reintentar_persistente`).

Para el caso especifico de APERTURA de posiciones que falla por red, el
patron correcto es:
  1) Intentar la orden UNA sola vez (sin reintento automatico).
  2) Si falla, llamar a `buscar_posicion_abierta()` de este modulo para
     preguntar al exchange si la posicion existe ya (la orden pudo haberse
     ejecutado aunque la respuesta se perdio). Si existe y coincide con lo
     que pedimos, adoptarla como entrada real sin colocar otra orden.
  3) Solo si el exchange confirma que NO existe la posicion, descartar la
     apertura. El modulo `trading_engine.py` implementa este patron en
     BinanceFuturesExecutor._buscar_posicion_abierta().
"""

import time
from typing import Optional

try:
    import ccxt
except ImportError:  # el modulo debe poder importarse igual sin ccxt instalado
    ccxt = None


# ---------------------------------------------------------------------------
# CLASIFICACION DEL ERROR
# ---------------------------------------------------------------------------
def clasificar_error(e: Exception) -> str:
    """Devuelve "rate_limit", "red" u "otro" segun el tipo de excepcion."""
    if ccxt is not None:
        if isinstance(e, (ccxt.RateLimitExceeded, ccxt.DDoSProtection)):
            return "rate_limit"
        if isinstance(e, ccxt.NetworkError):
            return "red"
    nombre = type(e).__name__.lower()
    mensaje = str(e).lower()
    if "rate limit" in mensaje or "too many requests" in mensaje or "429" in mensaje:
        return "rate_limit"
    if any(p in nombre for p in ("timeout", "connection", "network", "resolve", "socket")):
        return "red"
    return "otro"


def _espera_para(tipo: str, intento: int, espera_min: float, espera_max: float) -> float:
    """Backoff exponencial acotado, mas agresivo para rate limits."""
    factor = 2.0 if tipo == "rate_limit" else 1.6
    piso = espera_min * (3.0 if tipo == "rate_limit" else 1.0)
    espera = min(espera_max, piso * (factor ** (intento - 1)))
    return espera


# ---------------------------------------------------------------------------
# REINTENTO ACOTADO (llamadas interactivas / puntuales)
# ---------------------------------------------------------------------------
def reintentar_acotado(fn, *args, intentos=4, espera_min=3, espera_max=60,
                        descripcion="operacion", log=print, **kwargs):
    """Ejecuta fn(*args, **kwargs); si falla, reintenta hasta `intentos`
    veces con backoff (mas largo si el error es de rate limit). Si se
    agotan los intentos, relanza la ultima excepcion."""
    ultimo_error = None
    for intento in range(1, intentos + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            ultimo_error = e
            tipo = clasificar_error(e)
            if intento < intentos:
                espera = _espera_para(tipo, intento, espera_min, espera_max)
                etiqueta = {"rate_limit": "RATE LIMIT / bloqueo temporal del exchange",
                            "red": "problema de red", "otro": "error"}[tipo]
                log(f"  [!] {etiqueta} en {descripcion} (intento {intento}/{intentos}): "
                    f"{type(e).__name__}: {e}")
                log(f"      Reintentando en {espera:.0f}s...")
                time.sleep(espera)
    log(f"  [!] {descripcion.capitalize()} fallo tras {intentos} intentos: "
        f"{type(ultimo_error).__name__}: {ultimo_error}")
    raise ultimo_error


# ---------------------------------------------------------------------------
# REINTENTO PERSISTENTE (procesos de fondo -- nunca se rinden)
# ---------------------------------------------------------------------------
def reintentar_persistente(fn, *args, espera_min=5, espera_max=90,
                            descripcion="operacion", detener=None, log=print, **kwargs):
    """Ejecuta fn(*args, **kwargs) reintentando INDEFINIDAMENTE hasta que
    funcione, con backoff exponencial acotado entre intentos (mas largo si
    el exchange esta aplicando rate limit). Pensado para loops de fondo
    (auto-escaneo) donde "rendirse" no es una opcion valida -- el bot debe
    seguir insistiendo mientras la red o el exchange esten con problemas.

    `detener`: threading.Event opcional -- si se activa, se corta el loop de
    reintentos y se relanza la ultima excepcion (para poder apagar el hilo
    ordenadamente en vez de reintentar para siempre incluso al cerrar)."""
    intento = 0
    ultimo_error = None
    while True:
        intento += 1
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            ultimo_error = e
            tipo = clasificar_error(e)
            espera = _espera_para(tipo, intento, espera_min, espera_max)
            etiqueta = {"rate_limit": "RATE LIMIT / bloqueo temporal del exchange",
                        "red": "problema de red", "otro": "error"}[tipo]
            log(f"  [!] {etiqueta} en {descripcion} (intento {intento}): "
                f"{type(e).__name__}: {e}")
            log(f"      Reintentando en {espera:.0f}s (sin limite de intentos)...")
            if detener is not None:
                if detener.wait(timeout=espera):
                    raise ultimo_error
            else:
                time.sleep(espera)


# ---------------------------------------------------------------------------
# VERIFICACION DE POSICION TRAS FALLO DE RED EN APERTURA
# ---------------------------------------------------------------------------
def buscar_posicion_abierta(exchange, symbol: str, side: str, notional: float,
                             tolerancia: float = 0.10, log=print) -> Optional[dict]:
    """Despues de que una orden de apertura fallo por red, consulta si el
    exchange ya tiene registrada una posicion abierta compatible para ese
    symbol y side. Si la encuentra (nocional dentro de `tolerancia` del
    pedido original), devuelve un dict con entry_price / cantidad / order_id
    listo para ser adoptado por el bot como si la apertura hubiera funcionado.
    Devuelve None si no hay posicion o si la consulta al exchange tambien falla.

    Por que es seguro: fetch_positions() es lectura pura (idempotente). Solo
    se llama DESPUES de haber capturado una excepcion en la orden original, asi
    que nunca se duplica una orden que ya tuvo exito.

    Parametros:
      exchange   -- instancia ccxt ya configurada (con la cuenta correcta).
      symbol     -- ej. "BTC/USDT:USDT".
      side       -- "LONG" o "SHORT" (como lo maneja el bot, en mayusculas).
      notional   -- USDT que se intento abrir (para verificar que la posicion
                    encontrada corresponde a la nuestra y no a otra previa).
      tolerancia -- diferencia maxima relativa entre notional del exchange y el
                    pedido original para considerarlo un match (default 10%).
      log        -- funcion de logging (por defecto print).
    """
    try:
        positions = exchange.fetch_positions([symbol])
    except Exception as e:
        log(f"  [!] buscar_posicion_abierta: no se pudo consultar posiciones "
            f"para {symbol}: {type(e).__name__}: {e}")
        return None

    for p in positions:
        p_side      = str(p.get("side") or p.get("positionSide") or "").lower()
        p_contracts = float(p.get("contracts")  or p.get("contractSize") or 0)
        p_notional  = float(p.get("notional")   or 0)
        p_entry     = float(p.get("entryPrice") or p.get("entry_price") or 0)

        if p_side != side.lower():
            continue
        if p_entry <= 0:
            continue
        if p_notional <= 0 and p_contracts <= 0:
            continue

        notional_exchange = p_notional if p_notional > 0 else p_contracts * p_entry
        diff_rel = abs(notional_exchange - notional) / max(notional, 1e-9)
        if diff_rel > tolerancia:
            continue

        log(f"  [~] Posicion preexistente en exchange: {symbol} {side} "
            f"@ {p_entry:.6g} (nocional {notional_exchange:.2f} USDT, "
            f"diff {diff_rel*100:.1f}%) -- adoptada como entrada real.")
        return {
            "entry_price": p_entry,
            "cantidad":    p_contracts if p_contracts > 0 else notional / p_entry,
            "order_id":    p.get("id") or "adoptado_de_exchange",
        }

    log(f"  [~] buscar_posicion_abierta: no se encontro posicion {side} "
        f"compatible para {symbol} en el exchange.")
    return None


# ---------------------------------------------------------------------------
# AVISOS THROTTLEADOS (para loops muy frecuentes: no floodear la consola)
# ---------------------------------------------------------------------------
class AvisadorThrottled:
    """Para loops que corren cada 3-5s (refresco de precios, poll del libro
    de ordenes): si cada fallo imprimiera un mensaje, un problema de red de
    unos minutos generaria decenas de lineas repetidas. Esta clase solo deja
    pasar un aviso cada `intervalo_seg` por `clave` (ej. por simbolo), y
    ademas indica cuanto conviene esperar antes del proximo intento segun el
    tipo de error (para no seguir hammereando un exchange que ya esta
    limitando por rate limit)."""

    def __init__(self, intervalo_seg=30):
        self.intervalo_seg = intervalo_seg
        self._ultimo_aviso = {}

    def avisar(self, clave, e: Exception, contexto="", log=print) -> float:
        """Registra el error y, si corresponde (throttle), lo imprime.
        Devuelve la espera EXTRA (segundos) recomendada antes del proximo
        intento para esta `clave` (0 si es un error comun y no hace falta
        frenar, mayor si es rate limit)."""
        tipo = clasificar_error(e)
        ahora = time.monotonic()
        ultimo = self._ultimo_aviso.get(clave)  # None = nunca se aviso -> siempre corresponde avisar
        if ultimo is None or ahora - ultimo >= self.intervalo_seg:
            self._ultimo_aviso[clave] = ahora
            etiqueta = {"rate_limit": "RATE LIMIT / bloqueo temporal", "red": "problema de red",
                        "otro": "error"}[tipo]
            pref = f"{contexto} " if contexto else ""
            log(f"  [!] {pref}{etiqueta} en {clave}: {type(e).__name__}: {e}")
        if tipo == "rate_limit":
            return 20.0
        if tipo == "red":
            return 3.0
        return 0.0
