"""Funding y open interest diarios para el bot variante FO, mantenidos al día en un archivo local.

- Semilla: tu historial largo (derivs_metrics.pkl de descargar_todo_pc.py) para no empezar de cero.
- Cada ejecución añade lo reciente desde la API pública del exchange: funding (historial completo) y OI diario
  (la API de Binance solo da ~30 días, por eso se acumula en el archivo: basta con ejecutar el bot al menos una vez al mes).
- Mismo formato que los archivos de backtesting: funding {sym: DataFrame[ts, rate]}, oi {sym: DataFrame[ts, oi_amt, oi_usd]}.
"""
from __future__ import annotations

import os
import pickle
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STORE = os.path.join(HERE, "derivs_live.pkl")
SEEDS = [os.path.join(ROOT, "backtesting", "derivados", "derivs_metrics.pkl"), os.path.join(ROOT, "data", "derivs_metrics.pkl"),
         os.path.join(ROOT, "backtesting", "derivados", "derivs_funding_bin.pkl")]
DAY = 86_400_000


def _merge(old, new, cols):
    if old is None or not len(old):
        df = new
    elif new is None or not len(new):
        df = old
    else:
        df = pd.concat([old, new])
    if df is None or not len(df):
        return pd.DataFrame(columns=cols)
    df = df[cols].copy()
    df["ts"] = df["ts"].astype("int64")
    return df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)   # se guarda todo: sirve para reentrenar


def load_seed(paths=None):
    import ml_daily_xs_v2 as v2
    paths = [p for p in (paths or SEEDS) if os.path.exists(p)]
    return v2.load_derivs(paths) if paths else ({}, {})


def load(store=STORE, seeds=None):
    """Historial (semilla) + lo acumulado en vivo, unidos por símbolo (lo más reciente gana si se repite una fecha)."""
    fund, oi = load_seed(seeds)
    if os.path.exists(store):
        d = pickle.load(open(store, "rb"))
        for s, df in d.get("funding", {}).items():
            fund[s] = _merge(fund.get(s), df, ["ts", "rate"])
        for s, df in d.get("oi", {}).items():
            oi[s] = _merge(oi.get(s), df, ["ts", "oi_amt", "oi_usd"])
    return fund, oi


def fetch_recent(ex, sym, days=70):
    """(funding DataFrame, oi DataFrame) recientes desde la API. Cualquier fallo devuelve vacío para ese dato."""
    now = int(time.time() * 1000)
    f = o = None
    try:
        import funding_util
        ev = funding_util.funding_events(ex, sym, now - days * DAY, now)
        f = pd.DataFrame(ev, columns=["ts", "rate"])
    except Exception:
        pass
    try:
        rows = ex.fetch_open_interest_history(sym, "1d", limit=30)
        o = pd.DataFrame([dict(ts=int(r["timestamp"]), oi_amt=float(r.get("openInterestAmount") or float("nan")),
                               oi_usd=float(r.get("openInterestValue") or float("nan"))) for r in rows if r.get("timestamp")])
    except Exception:
        pass
    return f, o


def update(ex, symbols, store=STORE, seeds=None, log=print):
    """Actualiza el archivo local con lo reciente de la API y devuelve (funding, oi)."""
    fund, oi = load(store, seeds)
    ok_f = ok_o = 0
    for s in symbols:
        f, o = fetch_recent(ex, s)
        if f is not None and len(f):
            fund[s] = _merge(fund.get(s), f, ["ts", "rate"]); ok_f += 1
        if o is not None and len(o):
            oi[s] = _merge(oi.get(s), o, ["ts", "oi_amt", "oi_usd"]); ok_o += 1
    pickle.dump({"funding": fund, "oi": oi, "updated": time.strftime("%Y-%m-%d %H:%M")}, open(store, "wb"))
    days_oi = sorted(((d.ts.max() - d.ts.min()) // DAY) for d in oi.values() if len(d) > 1)
    log(f"derivados: funding actualizado en {ok_f}/{len(symbols)} símbolos, OI en {ok_o}/{len(symbols)} | "
        f"historial OI mediano {days_oi[len(days_oi) // 2] if days_oi else 0} días (se necesitan ~60 para todos los factores)")
    return fund, oi
