"""Datos privados de TU cuenta del exchange (con tus claves) para que los cálculos usen valores reales.

Con claves (solo consultas de lectura, nunca envía órdenes):
  - comisiones reales maker/taker por símbolo (tu nivel VIP / descuento BNB)        -> ccxt fetch_trading_fee
  - escalones de apalancamiento: apalancamiento máximo por tamaño de posición,
    margen de mantenimiento y "cum" de cada escalón                                    -> ccxt fetch_leverage_tiers
  - modo de posición (one-way o hedge)                                                 -> ccxt fetch_position_mode
Sin claves (o si fallan) todo vuelve a los valores de la configuración, y el panel dice qué fuente usa.
Se guarda en ~/.cerebro/datos_cuenta.json y se refresca cada 12 h (o con el botón del panel).
Para comisiones se prefieren las claves REAL (las de demo devuelven la tarifa por defecto).
"""
from __future__ import annotations

import json
import os
import threading
import time

import config_local as cl

CACHE = os.path.join(cl.DIR, "datos_cuenta.json")
TTL = 12 * 3600


def liq_price(side, entry, qty, margin, mmr, cum=0.0):
    """Precio de liquidación en margen aislado, modo one-way (fórmula de Binance):
    LP = (WB + cum - s·Q·EP) / (Q·MMR - s·Q), con WB = margen de la posición, s = +1 largo / -1 corto."""
    s = 1 if side == "LONG" else -1
    den = qty * mmr - s * qty
    if qty <= 0 or den == 0:
        return None
    lp = (margin + cum - s * qty * entry) / den
    return lp if lp > 0 else 0.0


class AccountData:
    def __init__(self, cfg, make_exchange=None):
        self.cfg = cfg
        self._make = make_exchange
        self.lock = threading.Lock()
        self.d = {}
        if os.path.exists(CACHE):
            try:
                self.d = json.load(open(CACHE, encoding="utf-8"))
            except Exception:
                self.d = {}
        if self.d.get("exchange") != cfg["exchange"]:
            self.d = {}

    # ------------------------------------------------------------------ fuente
    def _keys(self):
        """Claves a usar: REAL si existen (comisiones verdaderas), si no DEMO."""
        for mode in ("real", "demo"):
            k = cl.load_keys(self.cfg["exchange"], mode)
            if k:
                return mode, k
        return None, None

    def _exchange(self, mode, keys):
        if self._make:
            return self._make(mode, keys)
        import cuenta_real as cr
        return cr.make_exchange({**self.cfg, "mode": mode}, keys)

    def stale(self):
        return not self.d or time.time() - self.d.get("updated", 0) > TTL

    def refresh(self, symbols=("BTC/USDT:USDT",)):
        mode, keys = self._keys()
        if not keys:
            raise ValueError("no hay claves guardadas: los cálculos usan los valores de la configuración")
        ex = self._exchange(mode, keys)
        ex.load_markets()
        out = dict(exchange=self.cfg["exchange"], source=mode, updated=time.time(), fees={}, tiers={}, errors=[])
        try:
            raw = ex.fetch_leverage_tiers()
            for sym, tl in raw.items():
                out["tiers"][sym] = [[t.get("minNotional") or 0, t.get("maxNotional") or 1e18, t.get("maxLeverage") or 1,
                                      t.get("maintenanceMarginRate") or self.cfg["mmr"], float((t.get("info") or {}).get("cum") or 0)] for t in tl]
        except Exception as e:                                    # noqa: BLE001
            out["errors"].append(f"escalones: {type(e).__name__}: {str(e)[:120]}")
        for sym in symbols:
            try:
                f = ex.fetch_trading_fee(sym)
                out["fees"][sym] = [float(f["maker"]), float(f["taker"])]
            except Exception as e:                                # noqa: BLE001
                out["errors"].append(f"comisión {sym}: {type(e).__name__}: {str(e)[:120]}")
        try:
            out["hedged"] = bool(ex.fetch_position_mode("BTC/USDT:USDT").get("hedged"))
        except Exception as e:                                    # noqa: BLE001
            out["errors"].append(f"modo de posición: {type(e).__name__}: {str(e)[:80]}")
        if not out["fees"] and not out["tiers"]:
            raise ValueError("no se pudo leer nada de tu cuenta: " + "; ".join(out["errors"]))
        with self.lock:
            self.d = out
            os.makedirs(cl.DIR, exist_ok=True)
            json.dump(out, open(CACHE, "w", encoding="utf-8"))
        return self.status()

    def fee_for(self, symbol, fetch=None):
        """(maker, taker, fuente). Si no está en caché y hay un exchange autenticado en `fetch`, lo consulta."""
        f = self.d.get("fees", {}).get(symbol)
        if f:
            return f[0], f[1], "tu cuenta"
        base = self.d.get("fees", {}).get("BTC/USDT:USDT")
        if base:                                                  # misma tarifa VIP para casi todos los símbolos
            return base[0], base[1], "tu cuenta (tarifa general)"
        return self.cfg["maker"], self.cfg["taker"], "configuración"

    def tiers_for(self, symbol):
        return self.d.get("tiers", {}).get(symbol)

    def bracket(self, symbol, notional):
        """(apalancamiento máximo, mmr, cum, fuente) para ese tamaño de posición."""
        tl = self.tiers_for(symbol)
        if tl:
            for lo, hi, lev, mmr, cum in tl:
                if lo <= notional < hi:
                    return lev, mmr, cum, "tu cuenta"
            lo, hi, lev, mmr, cum = tl[-1]
            return lev, mmr, cum, "tu cuenta"
        return None, self.cfg["mmr"], 0.0, "configuración"

    def status(self):
        d = self.d
        f = d.get("fees", {}).get("BTC/USDT:USDT")
        return dict(available=bool(d), source=d.get("source"), updated=d.get("updated"), n_tiers=len(d.get("tiers", {})),
                    maker=f[0] if f else None, taker=f[1] if f else None, hedged=d.get("hedged"), errors=d.get("errors", []))

    def fetch_fee(self, symbol):
        """Consulta (y guarda) la comisión de un símbolo concreto. Silencioso si no hay claves."""
        if symbol in self.d.get("fees", {}) or not self.d:
            return
        mode, keys = self._keys()
        if not keys:
            return
        try:
            ex = self._exchange(mode, keys)
            ex.load_markets()
            f = ex.fetch_trading_fee(symbol)
            with self.lock:
                self.d.setdefault("fees", {})[symbol] = [float(f["maker"]), float(f["taker"])]
                json.dump(self.d, open(CACHE, "w", encoding="utf-8"))
        except Exception:
            pass
