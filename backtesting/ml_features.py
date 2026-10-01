"""Características causales de 5m (solo datos hasta el cierre de la vela t) y objetivos de retorno futuro."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

LAGS = (1, 3, 6, 12, 48, 144, 288)
WARMUP = 900   # velas descartadas al inicio (ventanas largas)


def load_panel(path: str):
    """Carga el pickle y alinea todos los símbolos al rango común de timestamps (se descargaron en momentos distintos)."""
    d = pickle.load(open(path, "rb"))
    syms = list(d)
    common = d[syms[0]]["ts"].to_numpy()
    for s in syms[1:]:
        common = np.intersect1d(common, d[s]["ts"].to_numpy())
    out = {s: d[s][d[s]["ts"].isin(common)].sort_values("ts").reset_index(drop=True) for s in syms}
    ts = out[syms[0]]["ts"].to_numpy()
    for s in syms:
        assert len(out[s]) == len(ts) and (out[s]["ts"].to_numpy() == ts).all(), f"{s}: timestamps distintos"
    return out, syms, ts


def _rsi(c: pd.Series, n: int) -> pd.Series:
    delta = c.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype("float64") for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    r1 = lc.diff()
    f = {f"r{k}": lc.diff(k) for k in LAGS}
    for n in (12, 48, 288):
        f[f"vol{n}"] = r1.rolling(n).std()
    f["volratio12"] = f["vol12"] / (f["vol288"] + 1e-12)
    f["volratio48"] = f["vol48"] / (f["vol288"] + 1e-12)
    rng = (h - l) / c
    f["rng12"], f["rng48"] = rng.rolling(12).mean(), rng.rolling(48).mean()
    f["rng_now"] = rng / (f["rng48"] + 1e-12)
    span = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / o
    f["upw"] = (h - np.maximum(o, c)) / span
    f["loww"] = (np.minimum(o, c) - l) / span
    f["cpos"] = (c - l) / span
    f["volu288"] = np.log1p(v) - np.log1p(v.rolling(288).mean())
    f["volu_z48"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-12)
    f["rsi14"], f["rsi168"] = _rsi(c, 14), _rsi(c, 168)
    for n in (20, 288):
        f[f"bbz{n}"] = (c - c.rolling(n).mean()) / (c.rolling(n).std() + 1e-12)
    for n in (288, 864):
        f[f"dhi{n}"] = c / h.rolling(n).max() - 1
        f[f"dlo{n}"] = c / l.rolling(n).min() - 1
    return pd.DataFrame(f)


def build_dataset(data: dict, syms: list[str], ts: np.ndarray, horizons=(12, 48)):
    """Devuelve (X, y, meta): X[s] DataFrame de features; y[(s,H)] retorno simple futuro neto de entrada en open[t+1]."""
    feats = {s: symbol_features(data[s]) for s in syms}
    # contexto de mercado y fuerza relativa
    wide = {k: pd.DataFrame({s: feats[s][k] for s in syms}) for k in ("r12", "r48", "r288", "vol48")}
    mkt = {k: wide[k].mean(axis=1) for k in ("r12", "r48", "r288")}
    rank = {k: wide[k].rank(axis=1, pct=True) for k in ("r12", "r48", "r288", "vol48")}
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    time_f = pd.DataFrame({"hsin": np.sin(2 * np.pi * dt.hour / 24), "hcos": np.cos(2 * np.pi * dt.hour / 24),
                           "dsin": np.sin(2 * np.pi * dt.dayofweek / 7), "dcos": np.cos(2 * np.pi * dt.dayofweek / 7)})
    btc = feats[syms[0]]
    X = {}
    for s in syms:
        x = feats[s].copy()
        for k in ("r12", "r48", "r288"):
            x[f"mkt_{k}"] = mkt[k].to_numpy()
            x[f"rel_{k}"] = (feats[s][k] - mkt[k]).to_numpy()
            x[f"rank_{k}"] = rank[k][s].to_numpy()
            x[f"btc_{k}"] = btc[k].to_numpy()
        x["rank_vol48"] = rank["vol48"][s].to_numpy()
        x["btc_vol48"] = btc["vol48"].to_numpy()
        x = pd.concat([x, time_f], axis=1)
        X[s] = x.astype("float32")
    y = {}
    for s in syms:
        o = data[s]["open"].to_numpy(dtype="float64")
        c = data[s]["close"].to_numpy(dtype="float64")
        for H in horizons:
            out = np.full(len(c), np.nan)
            out[: len(c) - H - 1] = c[H: len(c) - 1] / o[1: len(c) - H] - 1.0   # salida en close[t+H], entrada open[t+1]
            y[(s, H)] = out
    return X, y
