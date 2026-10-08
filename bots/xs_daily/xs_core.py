"""Núcleo compartido del bot cross-sectional diario (etiqueta rank, H=7).

Variantes:
  D  : 111 factores Alpha158 normalizados por ranking (la original del bot).
  FO : factores base + Alpha158 + funding + open interest (la mejor del backtest de 6 años, sección 16 de REPORT.md:
       +0.45 % por cohorte frente a +0.29 % de E, mejora NO significativa). Necesita derivados (derivs_live.py).
"""
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


VARIANTS = ("D", "FO")


def features(hist: dict, variant: str = "D", derivs=None):
    """hist: {simbolo: DataFrame[ts, open, high, low, close, volume]} diario; derivs = (funding, oi) para FO.
    Devuelve (idx, P, F rank-normalizadas, liq_ok). Sin fuga: los derivados se alinean as-of (OI con 1 día de retraso)."""
    idx, P = base.panels(hist)
    if variant == "FO":
        import derivs_features as dfe
        fund, oi = derivs or ({}, {})
        Ff, Fo = dfe.build(P, idx, fund, oi, 1)
        F = v2.to_rank({**base.build_features(P), **v2.alpha_features(P), **Ff, **Fo})
    else:
        F = v2.to_rank(v2.alpha_features(P))
    quote30 = (P["close"] * P["volume"]).rolling(30).mean()
    liq_ok = (quote30.rank(axis=1, pct=True) >= LIQ).to_numpy()
    return idx, P, F, liq_ok


def train(hist: dict, variant: str = "D", derivs=None):
    idx, P, F, liq_ok = features(hist, variant, derivs)
    names = list(F)
    _, ylab = v2.labels(P, "rank")
    X3 = np.stack([F[k].to_numpy() for k in names], axis=-1)
    y3 = ylab.to_numpy()
    T, S, nf = X3.shape
    rows = np.arange(70, T - H - 2)
    X = X3[rows].reshape(-1, nf); y = y3[rows].reshape(-1); ok = liq_ok[rows].reshape(-1)
    m = np.isfinite(y) & ok & np.isfinite(X).sum(axis=1).astype(bool)
    X, y = X[m], y[m]
    keep = [j for j in range(nf) if len(np.unique(X[np.isfinite(X[:, j]), j])) > 1]   # fuera factores constantes o vacíos
    X, names = X[:, keep], [names[j] for j in keep]
    lo, hi = np.percentile(y, [0.5, 99.5])
    mdl = HistGradientBoostingRegressor(**PARAMS).fit(X, np.clip(y, lo, hi))
    return dict(model=mdl, features=names, H=H, K=K, liq=LIQ, variant=variant, kfrac=0.12 if variant == "FO" else None, trained_until=str(idx[-1].date()),
                symbols=list(hist), n_rows=int(len(y)))


def k_for(K, n_valid, kfrac=None):
    """Monedas por lado: K fijo (o kfrac de las válidas, como en el backtest de FO), a lo sumo 1/4 de las válidas."""
    k = round(kfrac * n_valid) if kfrac else K
    return int(max(2, min(k, n_valid // 4)))


def pick(bundle, hist: dict, derivs=None):
    """Señal con la última vela diaria cerrada: (fecha, longs, shorts, predicciones)."""
    idx, P, F, liq_ok = features(hist, bundle.get("variant", "D"), derivs)
    nan = np.full(len(P["close"].columns), np.nan)         # si un factor no se pudo calcular hoy (p. ej. sin funding/OI) queda vacío
    X = np.stack([F[k].to_numpy()[-1] if k in F else nan for k in bundle["features"]], axis=-1)
    cols = list(P["close"].columns)
    valid = liq_ok[-1] & (np.isfinite(X).sum(axis=1) > 0) & np.isfinite(P["close"].to_numpy()[-1])
    if valid.sum() < 8:
        return idx[-1], [], [], {}
    iv = np.where(valid)[0]
    p = bundle["model"].predict(X[iv])
    order = iv[np.argsort(p)]
    k = k_for(bundle["K"], len(iv), bundle.get("kfrac"))
    preds = {cols[i]: float(pv) for i, pv in zip(iv, p)}
    return idx[-1], [cols[i] for i in order[-k:]], [cols[i] for i in order[:k]], preds
