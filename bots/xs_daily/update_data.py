"""Pone al día los datos de entrenamiento sin pasos manuales.

1. Velas diarias: añade a data/ohlcv_daily_long.pkl las velas CERRADAS que falten (desde el mismo exchange del que salió
   el archivo, OKX por defecto, para no mezclar volúmenes de distintos exchanges). Guarda antes una copia .bak.
2. Funding y open interest (para la variante FO): actualiza bots/xs_daily/derivs_live.pkl desde la API pública de Binance.
Uso: python bots/xs_daily/update_data.py [--exchange okx] [--derivs-exchange binanceusdm] [--no-derivs]
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xs_core  # noqa: E402

DAY = 86_400_000
DATA = os.path.join(xs_core.ROOT, "data", "ohlcv_daily_long.pkl")


def last_day(path=DATA):
    """Fecha de la última vela del archivo (la más reciente entre todos los símbolos)."""
    if not os.path.exists(path):
        return None
    raw = pickle.load(open(path, "rb"))
    return max(pd.to_datetime(df["ts"], utc=True).max() for df in raw.values() if len(df))


def update_candles(ex, path=DATA, log=print):
    raw = pickle.load(open(path, "rb"))
    ex.load_markets()
    now = ex.milliseconds()
    added, missing, errors = 0, 0, 0
    for i, (sym, df) in enumerate(raw.items(), 1):
        if sym not in ex.markets:
            missing += 1
            continue
        ts = pd.to_datetime(df["ts"], utc=True)
        since = int(ts.max().value // 10**6) + DAY
        rows = []
        try:
            while since + DAY <= now:                           # solo velas cerradas
                batch = ex.fetch_ohlcv(sym, "1d", since=since, limit=100)
                batch = [r for r in batch if r[0] >= since and r[0] + DAY <= now]
                if not batch:
                    break
                rows += batch
                since = batch[-1][0] + DAY
        except Exception as e:                                  # noqa: BLE001
            errors += 1
            log(f"  {sym}: {type(e).__name__}: {str(e)[:80]}")
        if rows:
            new = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            new["ts"] = pd.to_datetime(new["ts"], unit="ms", utc=True)
            base = df.copy()
            base["ts"] = pd.to_datetime(base["ts"], utc=True)
            raw[sym] = pd.concat([base, new]).drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
            added += len(rows)
        if i % 20 == 0:
            log(f"  velas: {i}/{len(raw)} símbolos revisados, {added} velas nuevas")
    shutil.copy2(path, path + ".bak")
    pickle.dump(raw, open(path, "wb"))
    log(f"velas: {added} velas diarias nuevas añadidas | {missing} símbolos ya no existen en {ex.id} | {errors} con error | copia en {os.path.basename(path)}.bak")
    return added


def main():
    import ccxt
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="okx", help="exchange de las velas (el mismo del archivo original: okx)")
    ap.add_argument("--derivs-exchange", default="binanceusdm", help="exchange del funding/OI (el historial es de Binance)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--no-derivs", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    before = last_day(a.data)
    print(f"última vela antes de actualizar: {before.date() if before is not None else '—'}", flush=True)
    ex = getattr(ccxt, a.exchange)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    update_candles(ex, a.data)
    print(f"última vela ahora: {last_day(a.data).date()}", flush=True)
    if not a.no_derivs:
        import derivs_live
        dex = getattr(ccxt, a.derivs_exchange)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        dex.load_markets()
        syms = [s for s in pickle.load(open(a.data, "rb")) if s in dex.markets]
        derivs_live.update(dex, syms)
    print(f"datos actualizados en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
