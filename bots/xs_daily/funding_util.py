"""Funding de perpetuos: cuánto pagó o cobró una posición entre dos instantes.

Convención: devuelve el COSTO (positivo = pagaste). Con tasa positiva los largos pagan a los cortos.
Si el exchange no da el historial (o no hay red), devuelve (0.0, False) para que el llamador lo indique.
"""
from __future__ import annotations


def funding_events(ex, symbol, since_ms, until_ms):
    """Lista [(ts_ms, tasa)] con ts en (since_ms, until_ms]."""
    out, since = [], int(since_ms) + 1
    for _ in range(20):                                      # paginación hacia adelante (máx. ~20 páginas)
        rows = ex.fetch_funding_rate_history(symbol, since=since, limit=100)
        if not rows:
            break
        new = [(int(r["timestamp"]), float(r["fundingRate"])) for r in rows
               if r.get("timestamp") and r.get("fundingRate") is not None and since_ms < r["timestamp"] <= until_ms]
        out.extend(new)
        last = max(int(r["timestamp"]) for r in rows)
        if last >= until_ms or last < since or len(rows) < 100:
            break
        since = last + 1
    return sorted(set(out))


def funding_cost(ex, symbol, side, notional, since_ms, until_ms):
    """Costo de funding (USDT) de una posición de `notional` USDT. (costo, ok)."""
    try:
        ev = funding_events(ex, symbol, since_ms, until_ms)
    except Exception:
        return 0.0, False
    sgn = 1 if side == "LONG" else -1
    return float(sum(sgn * notional * r for _, r in ev)), True
