"""Descarga velas de OKX (perpetuos USDT) con ccxt, con reanudación por símbolo.

Local:  python download_okx_5m.py --days 365 --out okx_5m.pkl
Colab:  ver colab/descargar_velas_okx.ipynb (usa estas mismas funciones).
Salida: pickle {simbolo: DataFrame[ts(ms), open, high, low, close, volume]} compatible con backtest_cerebro_rl_5m.py.
Probado con un exchange simulado; no se ha probado contra OKX real.
"""
import argparse
import os
import pickle
import time

import ccxt
import pandas as pd

DEFAULT_SYMBOLS = [f"{b}/USDT:USDT" for b in (
    "BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC DOT TRX BCH NEAR ATOM UNI APT ARB OP SUI").split()]


def _fetch_with_retry(ex, symbol, tf, since, limit, tries=5):
    for k in range(tries):
        try:
            return ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.RequestTimeout):
            time.sleep(2 ** k)
    raise RuntimeError(f"{symbol}: demasiados fallos de red/rate limit")


def fetch_symbol(ex, symbol, days, tf="5m", limit=100, log=print):
    """Pagina hacia adelante desde now-days; descarta la vela en curso; ordena y quita duplicados."""
    tf_ms = ex.parse_timeframe(tf) * 1000
    now = ex.milliseconds()
    since = now - days * 86_400_000
    rows, last_ts, calls = [], None, 0
    while since < now:
        batch = _fetch_with_retry(ex, symbol, tf, since, limit)
        calls += 1
        if not batch:
            break
        rows += batch
        new_last = batch[-1][0]
        if last_ts is not None and new_last <= last_ts:
            break                               # sin progreso: fin de datos
        last_ts, since = new_last, new_last + tf_ms
        if calls % 100 == 0:
            log(f"  {symbol}: {len(rows)} velas, hasta {pd.to_datetime(new_last, unit='ms', utc=True):%Y-%m-%d}")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[df["ts"] + tf_ms <= now].reset_index(drop=True)   # solo velas cerradas


def quality(df, tf_ms=300_000):
    """Resumen de calidad: filas, rango y huecos (velas faltantes)."""
    if df.empty:
        return dict(rows=0, first=None, last=None, missing=0)
    expected = (df["ts"].iloc[-1] - df["ts"].iloc[0]) // tf_ms + 1
    return dict(rows=len(df), first=pd.to_datetime(df["ts"].iloc[0], unit="ms", utc=True),
                last=pd.to_datetime(df["ts"].iloc[-1], unit="ms", utc=True), missing=int(expected - len(df)))


def download_all(symbols, days, out_dir, tf="5m", ex=None, log=print):
    """Descarga todos los símbolos; guarda cada uno en out_dir/parts (si ya existe, lo reutiliza)."""
    ex = ex or ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    parts = os.path.join(out_dir, "parts")
    os.makedirs(parts, exist_ok=True)
    result = {}
    for s in symbols:
        path = os.path.join(parts, s.replace("/", "_").replace(":", "_") + f"_{days}d.pkl")
        if os.path.exists(path):
            result[s] = pickle.load(open(path, "rb"))
            log(f"{s}: reutilizado ({len(result[s])} velas)")
            continue
        if s not in ex.markets:
            log(f"{s}: no existe en OKX, se omite")
            continue
        df = fetch_symbol(ex, s, days, tf, log=log)
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("float32")
        pickle.dump(df, open(path, "wb"))
        result[s] = df
        q = quality(df)
        log(f"{s}: {q['rows']} velas, {q['first']:%Y-%m-%d} -> {q['last']:%Y-%m-%d}, faltan {q['missing']}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default="okx_5m.pkl")
    a = ap.parse_args()
    data = download_all(a.symbols, a.days, os.path.dirname(os.path.abspath(a.out)))
    pickle.dump(data, open(a.out, "wb"))
    print("guardado", a.out, f"({len(data)} símbolos)")
