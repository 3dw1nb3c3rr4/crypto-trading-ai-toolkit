"""Descarga velas de 5m de Binance Futures (API pública, sin claves) para backtest_cerebro_rl_5m.py.

Uso:  python download_5m.py --symbols BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT --days 60 --out ohlcv_5m.pkl
Ejecútalo en tu PC (este entorno de desarrollo no tiene acceso a Binance). Sin probar contra el exchange real.
"""
import argparse
import pickle
import time

import ccxt
import pandas as pd


def fetch(ex, symbol, days):
    ms_5m = 5 * 60 * 1000
    since = ex.milliseconds() - days * 24 * 3600 * 1000
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + ms_5m
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("ts").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default="ohlcv_5m.pkl")
    a = ap.parse_args()
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    out = {}
    for s in a.symbols:
        out[s] = fetch(ex, s, a.days)
        print(s, len(out[s]), "velas", flush=True)
    pickle.dump(out, open(a.out, "wb"))
    print("guardado", a.out)
