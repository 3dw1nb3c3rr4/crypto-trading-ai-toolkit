"""Descarga funding y open interest (OI) históricos de perpetuos USDT con ccxt (API pública, sin claves).

Uso en tu PC:  python download_derivs.py --exchange bybit --universe ../data/ohlcv_daily_long.pkl --days 2200 --out derivs_bybit.pkl
Fuentes (el que funcione en tu país): bybit (funding + OI diario con historial largo), binanceusdm (funding completo; el OI
histórico de la API solo cubre ~30 días), okx (funding ~3 meses; OI limitado). Cada una guarda su propio archivo.

Salida: pickle {"funding": {simbolo: DataFrame[ts, rate]}, "oi": {simbolo: DataFrame[ts, oi_amt, oi_usd]}, "meta": {...}}
  - funding: rate = tasa por liquidación (fracción; positiva => los largos pagan a los cortos).
  - oi: ts en ms, oi_amt = open interest en unidades base/contratos, oi_usd = en USDT si el exchange lo informa (si no, NaN).
Nota sobre fechas: el OI diario es una *foto* a la hora ts; el modelo la usa con retraso de un día por seguridad.
Probado con exchanges simulados; no se ha probado contra las APIs reales.
"""
import argparse
import pickle
import time

import ccxt
import pandas as pd

DEFAULT_SYMBOLS = [f"{b}/USDT:USDT" for b in (
    "BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC DOT TRX BCH NEAR ATOM UNI APT ARB OP SUI").split()]
DAY = 86_400_000


def _retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.RequestTimeout):
            time.sleep(2 ** k)
    raise RuntimeError("demasiados fallos de red/rate limit")


def _backward(fetch, start, now, ts_of, max_pages=400):
    """Pagina hacia atrás con 'until' (los exchanges devuelven lo más reciente anterior a until).
    fetch(until) -> lista de registros; ts_of(registro) -> ms. Corta al llegar a `start` o cuando no hay progreso."""
    rows, until, pages = {}, now, 0
    while pages < max_pages:
        batch = _retry(lambda: fetch(until))
        pages += 1
        if not batch:
            break
        for b in batch:
            rows[ts_of(b)] = b
        oldest = min(ts_of(b) for b in batch)
        if oldest >= until or oldest <= start:
            break
        until = oldest
    return rows


def fetch_funding(ex, symbol, days):
    now = ex.milliseconds()
    start = now - days * DAY
    if ex.id in ("okx", "bybit"):
        rows = _backward(lambda u: ex.fetch_funding_rate_history(symbol, limit=200 if ex.id == "bybit" else 100, params={"until": u}),
                         start, now, lambda b: b["timestamp"])
        pairs = [(t, float(b["fundingRate"])) for t, b in rows.items()]
    else:                                                                    # binance: hacia adelante con since
        rows, since, last = {}, start, None
        while since < now:
            batch = _retry(lambda: ex.fetch_funding_rate_history(symbol, since=since, limit=1000))
            if not batch:
                break
            for b in batch:
                rows[b["timestamp"]] = float(b["fundingRate"])
            nl = max(b["timestamp"] for b in batch)
            if last is not None and nl <= last:
                break
            last, since = nl, nl + 1
        pairs = list(rows.items())
    df = pd.DataFrame(sorted(pairs), columns=["ts", "rate"])
    return df[df["ts"] >= start].reset_index(drop=True)


def fetch_oi(ex, symbol, days):
    """OI diario. Bybit/OKX: hacia atrás con 'until'. Binance: la API solo guarda ~30 días (se pide y se informa lo que haya)."""
    now = ex.milliseconds()
    start = now - days * DAY
    if ex.id == "binanceusdm":
        batch = _retry(lambda: ex.fetch_open_interest_history(symbol, "1d", limit=500))
        rows = {b["timestamp"]: b for b in batch}
    else:
        rows = _backward(lambda u: ex.fetch_open_interest_history(symbol, "1d", limit=200, params={"until": u}),
                         start, now, lambda b: b["timestamp"])
    data = [(t, b.get("openInterestAmount"), b.get("openInterestValue")) for t, b in rows.items() if t >= start]
    df = pd.DataFrame(sorted(data), columns=["ts", "oi_amt", "oi_usd"]).astype("float64")
    return df.reset_index(drop=True)


def quality(df, col):
    if df.empty:
        return "vacío"
    d = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return f"{len(df)} filas, {d.iloc[0]:%Y-%m-%d} -> {d.iloc[-1]:%Y-%m-%d}, {df[col].notna().mean():.0%} con dato"


def download(exchange_id, symbols, days, log=print):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    out = {"funding": {}, "oi": {}, "meta": {"exchange": exchange_id, "days": days, "errors": {}}}
    for s in symbols:
        if s not in ex.markets:
            log(f"{s}: no existe en {exchange_id}")
            continue
        for kind, fn, col in (("funding", fetch_funding, "rate"), ("oi", fetch_oi, "oi_amt")):
            try:
                df = fn(ex, s, days)
                out[kind][s] = df
                log(f"{s} {kind}: {quality(df, col)}")
            except Exception as e:                                       # un símbolo o dato roto no detiene el resto
                out["meta"]["errors"][f"{s}:{kind}"] = f"{type(e).__name__}: {str(e)[:120]}"
                log(f"{s} {kind}: ERROR {type(e).__name__}: {str(e)[:100]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="bybit", help="bybit | binanceusdm | okx")
    ap.add_argument("--universe", default=None, help="pickle diario (p.ej. ohlcv_daily_long.pkl) para tomar la lista de símbolos")
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--symbols-file", default=None, help="archivo de texto con un símbolo por línea (p.ej. universe_symbols.txt)")
    ap.add_argument("--days", type=int, default=2200)
    ap.add_argument("--out", default="derivs.pkl")
    a = ap.parse_args()
    syms = a.symbols or ([x.strip() for x in open(a.symbols_file) if x.strip()] if a.symbols_file else
                         (list(pickle.load(open(a.universe, "rb"))) if a.universe else DEFAULT_SYMBOLS))
    data = download(a.exchange, syms, a.days)
    pickle.dump(data, open(a.out, "wb"))
    print(f"guardado {a.out}: funding {len(data['funding'])} símbolos, OI {len(data['oi'])} símbolos, errores {len(data['meta']['errors'])}")
