"""Núcleo compartido del bot cross-sectional diario (variante D: features Alpha158 normalizadas por ranking, etiqueta rank, H=7)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "backtesting"))
import ml_daily_xs as base  # noqa: E402
import ml_daily_xs_v2 as v2  # noqa: E402

H, K = v2.H, v2.K
LIQ = 0.30
PARAMS = dict(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=400, l2_regularization=50.0, max_features=0.8, random_state=0)


def features(hist: dict):
    """hist: {simbolo: DataFrame[ts, open, high, low, close, volume]} diario. Devuelve (idx, P, F rank-normalizadas, liq_ok)."""
    idx, P = base.panels(hist)
    F = v2.to_rank(v2.alpha_features(P))
    quote30 = (P["close"] * P["volume"]).rolling(30).mean()
    liq_ok = (quote30.rank(axis=1, pct=True) >= LIQ).to_numpy()
    return idx, P, F, liq_ok


def train(hist: dict):
    idx, P, F, liq_ok = features(hist)
    names = list(F)
    _, ylab = v2.labels(P, "rank")
    X3 = np.stack([F[k].to_numpy() for k in names], axis=-1)
    y3 = ylab.to_numpy()
    T, S, nf = X3.shape
    rows = np.arange(70, T - H - 2)
    X = X3[rows].reshape(-1, nf); y = y3[rows].reshape(-1); ok = liq_ok[rows].reshape(-1)
    m = np.isfinite(y) & ok & np.isfinite(X).sum(axis=1).astype(bool)
    X, y = X[m], y[m]
    lo, hi = np.percentile(y, [0.5, 99.5])
    mdl = HistGradientBoostingRegressor(**PARAMS).fit(X, np.clip(y, lo, hi))
    return dict(model=mdl, features=names, H=H, K=K, liq=LIQ, variant="D", trained_until=str(idx[-1].date()),
                symbols=list(hist), n_rows=int(len(y)))


def pick(bundle, hist: dict):
    """Señal con la última vela diaria cerrada: (fecha, longs, shorts, predicciones)."""
    idx, P, F, liq_ok = features(hist)
    X = np.stack([F[k].to_numpy()[-1] for k in bundle["features"]], axis=-1)
    cols = list(P["close"].columns)
    valid = liq_ok[-1] & (np.isfinite(X).sum(axis=1) > 0) & np.isfinite(P["close"].to_numpy()[-1])
    if valid.sum() < 2 * bundle["K"] + 10:
        return idx[-1], [], [], {}
    iv = np.where(valid)[0]
    p = bundle["model"].predict(X[iv])
    order = iv[np.argsort(p)]
    k = bundle["K"]
    preds = {cols[i]: float(pv) for i, pv in zip(iv, p)}
    return idx[-1], [cols[i] for i in order[-k:]], [cols[i] for i in order[:k]], preds
