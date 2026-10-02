"""Descarga historial de funding de perpetuos (OKX o Binance) y velas spot de 1h para calcular la base.

Colab:  ver colab/celda_funding.py (usa estas mismas funciones).
Local:  python download_funding.py --exchange binance --days 365 --out funding_binance.pkl
Notas:  OKX solo expone ~3 meses de historial de funding; Binance expone todo el historial pero bloquea IPs de EE.UU.
        (Colab suele estar en EE.UU.: usa OKX allí y Binance desde tu PC). Probado con exchanges simulados, no con APIs reales.
Salida: pickle {simbolo: DataFrame[ts(ms), rate]}  (rate = tasa por liquidación, fracción; positiva => los cortos cobran).
"""
import argparse
import pickle
import time

import ccxt
import pandas as pd

DEFAULT_SYMBOLS = [f"{b}/USDT:USDT" for b in (
    "BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC DOT TRX BCH NEAR ATOM UNI APT ARB OP SUI").split()]


def _retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.RequestTimeout):
            time.sleep(2 ** k)
    raise RuntimeError("demasiados fallos de red/rate limit")


def fetch_funding(ex, symbol, days):
    """OKX: pagina hacia atrás con 'until' (devuelve las 100 más recientes anteriores a until, máx ~3 meses).
    Otros (Binance): pagina hacia adelante con 'since' (hasta 1000 por llamada)."""
    now = ex.milliseconds()
    start = now - days * 86_400_000
    rows = {}
    if ex.id == "okx":
        until = now
        while True:
            batch = _retry(lambda: ex.fetch_funding_rate_history(symbol, limit=100, params={"until": until}))
            if not batch:
                break
            for b in batch:
                rows[b["timestamp"]] = float(b["fundingRate"])
            oldest = min(b["timestamp"] for b in batch)
            if oldest >= until or oldest <= start:
                break
            until = oldest
    else:
        since, last = start, None
        while since < now:
            batch = _retry(lambda: ex.fetch_funding_rate_history(symbol, since=since, limit=1000))
            if not batch:
                break
            for b in batch:
                rows[b["timestamp"]] = float(b["fundingRate"])
            new_last = max(b["timestamp"] for b in batch)
            if last is not None and new_last <= last:
                break
            last, since = new_last, new_last + 1
    df = pd.DataFrame(sorted(rows.items()), columns=["ts", "rate"])
    return df[df["ts"] >= start].reset_index(drop=True)


def funding_quality(df):
    if df.empty:
        return dict(n=0, first=None, last=None, per_day=0.0, mean_8h=float("nan"))
    span_days = max((df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86_400_000, 1e-9)
    return dict(n=len(df), first=pd.to_datetime(df["ts"].iloc[0], unit="ms", utc=True),
                last=pd.to_datetime(df["ts"].iloc[-1], unit="ms", utc=True),
                per_day=round((len(df) - 1) / span_days, 2), mean_8h=float(df["rate"].mean()))


def download_funding(exchange_id, symbols, days, log=print):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    out = {}
    for s in symbols:
        if s not in ex.markets:
            log(f"{exchange_id} {s}: no existe, se omite")
            continue
        try:
            df = fetch_funding(ex, s, days)
        except Exception as e:  # un símbolo con error no detiene el resto
            log(f"{exchange_id} {s}: ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        out[s] = df
        q = funding_quality(df)
        log(f"{exchange_id} {s}: {q['n']} liquidaciones, {q['first']:%Y-%m-%d} -> {q['last']:%Y-%m-%d}, "
            f"{q['per_day']}/día, media {q['mean_8h']:.5%}")
    return out


def download_spot_1h(symbols, days, fetch_symbol, log=print):
    """Velas spot de 1h de OKX (para el P&L de la base). fetch_symbol viene de download_okx_5m.py."""
    ex = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    ex.load_markets()
    out = {}
    for perp in symbols:
        spot = perp.split(":")[0]
        if spot not in ex.markets:
            log(f"spot {spot}: no existe, se omite")
            continue
        out[perp] = fetch_symbol(ex, spot, days, tf="1h", limit=100, log=log)
        log(f"spot {spot}: {len(out[perp])} velas 1h")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binanceusdm", help="okx | binanceusdm")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", default="funding.pkl")
    a = ap.parse_args()
    ex_id = "binanceusdm" if a.exchange == "binance" else a.exchange
    data = download_funding(ex_id, a.symbols, a.days)
    pickle.dump(data, open(a.out, "wb"))
    print("guardado", a.out, f"({len(data)} símbolos)")
