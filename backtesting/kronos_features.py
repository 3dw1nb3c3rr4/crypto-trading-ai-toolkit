"""Pronósticos de un modelo fundacional de velas (Kronos, shiyu-coder/Kronos) convertidos en características diarias sin fuga.

Flujo:
  1) En Colab con GPU (colab/celda_kronos.py) se genera `kronos_forecasts.pkl` con `make_forecasts(...)`.
  2) Aquí `build()` lo alinea al índice diario y crea características que entran como variante "K" de ml_daily_xs_v2.py.

Anti-fuga: el pronóstico de la fecha t usa SOLO velas con índice <= t (la vela t cierra en t+1d, igual que el resto de
características; la entrada es el open de t+1). `test_kronos_features.py` lo comprueba alterando el futuro.
Limitación conocida: Kronos se preentrenó con 12B+ velas de 45+ exchanges sin fecha de corte publicada; los periodos
antiguos pueden haber estado en su preentrenamiento (contaminación). Solo las fechas posteriores a su corte son una prueba limpia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 7
CTX = 400                      # velas de contexto (Kronos-small/base admiten 512)
FEATS = ("kr_ret", "kr_vol", "kr_up", "kr_hi", "kr_lo")


class MockPredictor:
    """Sustituto determinista para pruebas sin pesos: extrapola el momentum de 5 días. Misma interfaz que KronosPredictor."""

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **kw):
        out = []
        for df, yts in zip(df_list, y_timestamp_list):
            c = df["close"].to_numpy(dtype="float64")
            drift = (c[-1] / c[-6]) ** (1 / 5) - 1 if len(c) > 6 else 0.0
            path = c[-1] * (1 + drift) ** np.arange(1, pred_len + 1)
            out.append(pd.DataFrame({"open": path, "high": path * 1.01, "low": path * 0.99, "close": path,
                                     "volume": df["volume"].iloc[-1], "amount": df["amount"].iloc[-1]}, index=pd.DatetimeIndex(yts)))
        return out


def _window(df: pd.DataFrame, i: int, ctx: int):
    lo = max(0, i - ctx + 1)
    w = df.iloc[lo:i + 1]
    x = w[["open", "high", "low", "close", "volume"]].astype("float64").copy()
    x["amount"] = (w["close"] * w["volume"]).astype("float64").to_numpy()
    return x.reset_index(drop=True), pd.Series(pd.to_datetime(w["ts"], utc=True).dt.tz_localize(None).to_numpy())


def make_forecasts(uni: dict, predictor, step: int = 1, start=None, horizon: int = HORIZON, ctx: int = CTX,
                   batch: int = 64, min_ctx: int = 120, log=print, **kw) -> pd.DataFrame:
    """uni: símbolo -> DataFrame diario (ts, open, high, low, close, volume). Devuelve una tabla larga con una fila por (símbolo, fecha t)."""
    jobs = []
    start = pd.Timestamp(start, tz="UTC") if start is not None else None
    for s, df in uni.items():
        ts = pd.to_datetime(df["ts"], utc=True)
        for i in range(min_ctx, len(df), step):
            if start is not None and ts.iloc[i] < start:
                continue
            jobs.append((s, i))
    rows = []
    for b in range(0, len(jobs), batch):
        chunk = jobs[b:b + batch]
        # predict_batch exige historiales de igual longitud: se agrupa por longitud de ventana
        groups = {}
        for s, i in chunk:
            groups.setdefault(min(i + 1, ctx), []).append((s, i))
        for L, items in groups.items():
            xs, xts, yts = [], [], []
            for s, i in items:
                x, xt = _window(uni[s], i, ctx)
                last = xt.iloc[-1]
                xs.append(x); xts.append(xt); yts.append(pd.Series(pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="1D")))
            preds = predictor.predict_batch(df_list=xs, x_timestamp_list=xts, y_timestamp_list=yts, pred_len=horizon, **kw)
            for (s, i), x, p in zip(items, xs, preds):
                c0 = float(x["close"].iloc[-1]); pc = p["close"].to_numpy(dtype="float64")
                lr = np.diff(np.log(np.r_[c0, pc]))
                rows.append((s, uni[s]["ts"].iloc[i], pc[-1] / c0 - 1, float(lr.std()), float((lr > 0).mean()),
                             float(p["high"].max() / c0 - 1), float(p["low"].min() / c0 - 1)))
        if (b // batch) % 20 == 0:
            log(f"  pronósticos {min(b + batch, len(jobs))}/{len(jobs)}")
    out = pd.DataFrame(rows, columns=["sym", "ts", *FEATS])
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def build(forecasts: pd.DataFrame, idx: pd.DatetimeIndex, cols: list[str]) -> dict:
    """Tabla larga -> {nombre: DataFrame (T x S)} alineada al índice diario. Sin relleno hacia adelante: sin pronóstico = NaN."""
    out = {}
    f = forecasts.copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    for k in FEATS:
        piv = f.pivot_table(index="ts", columns="sym", values=k, aggfunc="last")
        out[k] = piv.reindex(index=idx, columns=cols)
    return out
