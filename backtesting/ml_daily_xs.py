"""Modelo cross-sectional DIARIO sobre el universo completo (117 perpetuos, ~2 años), con walk-forward y costos.

Predice el retorno futuro relativo (H días, entrada en open[t+1], salida en open[t+1+H]) de cada símbolo y opera
long los K mejores / short los K peores (neutral al mercado). Cada día se abre una cartera que dura H días
(carteras superpuestas, 1/H del capital cada una). Costo: se asume que TODAS las posiciones de cada cartera se abren
y se cierran (taker + slippage por lado = 0.14% sobre el nocional), lo que es conservador.

Diseño fijado antes de ver resultados: H in {3, 7} x modelo in {gbm, ridge}. Controles: --shuffle (nulo) y --plant (sensibilidad).
Uso: python ml_daily_xs.py [--shuffle] [--plant 6]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Costs, load_universe  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "trading_history", "ohlcv_cache_2y.pkl")
K = 10
RT = Costs().round_trip()
TEST_START_DAY, TEST_LEN = 300, 90


def panels(uni):
    idx = pd.DatetimeIndex(sorted(set().union(*[set(df.ts) for df in uni.values()])))
    g = lambda col: pd.DataFrame({s: df.set_index("ts")[col] for s, df in uni.items()}).reindex(idx)
    return idx, {c: g(c) for c in ("open", "high", "low", "close", "volume")}


def build_features(P):
    o, h, l, c, v = (P[k] for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    r1 = lc.diff()
    F = {}
    for n in (1, 3, 7, 14, 30, 60):
        F[f"ret{n}"] = lc.diff(n)
    F["vol7"], F["vol30"] = r1.rolling(7).std(), r1.rolling(30).std()
    F["volratio"] = F["vol7"] / (F["vol30"] + 1e-9)
    F["range7"] = ((h - l) / c).rolling(7).mean()
    for n in (30, 90):
        F[f"dhi{n}"] = c / h.rolling(n).max() - 1
        F[f"dlo{n}"] = c / l.rolling(n).min() - 1
    quote = c * v
    F["turn"] = np.log1p(quote.rolling(7).mean()) - np.log1p(quote.rolling(30).mean())
    F["lturn30"] = np.log1p(quote.rolling(30).mean())
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    F["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-12))
    span = (h - l).replace(0, np.nan)
    F["body"], F["upw"], F["loww"] = (c - o) / o, (h - np.maximum(o, c)) / span, (np.minimum(o, c) - l) / span
    mkt = r1.mean(axis=1)
    F["age"] = np.log1p(c.notna().cumsum())
    for n in (7, 30):
        mk = lc.diff(n).mean(axis=1)
        F[f"rel{n}"] = F[f"ret{n}"].sub(mk, axis=0)
    cov = r1.rolling(60).cov(mkt)
    F["beta60"] = cov.div(mkt.rolling(60).var(), axis=0)
    F["idio60"] = r1.sub(F["beta60"].mul(mkt, axis=0)).rolling(60).std()
    F["mkt7"] = pd.DataFrame({s: lc.diff(7).mean(axis=1) for s in c.columns})
    F["btc7"] = pd.DataFrame({s: F["ret7"].iloc[:, 0] for s in c.columns}) if c.columns[0].startswith("BTC") else F["mkt7"]
    for k in ("ret7", "ret30", "vol30", "lturn30", "dhi30", "rsi14"):
        F[f"rk_{k}"] = F[k].rank(axis=1, pct=True)
    return F


def target(P, H):
    o = P["open"]
    fwd = o.shift(-(1 + H)) / o.shift(-1) - 1.0
    return fwd


def stack(F, y, idx, rows_mask):
    cols = list(F)
    arrs = [F[k].to_numpy()[rows_mask] for k in cols]
    X = np.stack([a for a in arrs], axis=-1)           # (dias, simbolos, features)
    return X, y.to_numpy()[rows_mask], cols


def run(H, model_name, shuffle, plant, rng, uni_idx, P, F, liq_ok):
    global AGE, MKT
    AGE = F_AGE.to_numpy()
    MKT = target(P, H).to_numpy()
    y = target(P, H)
    ydm = y.sub(y.mean(axis=1), axis=0)
    n = len(uni_idx)
    fwd = y.to_numpy()
    Fnp = np.stack([F[k].to_numpy() for k in F], axis=-1)           # (T, S, f)
    if plant:
        pl = ydm.to_numpy() + plant * np.nanstd(ydm.to_numpy()) * rng.standard_normal(ydm.shape)
        Fnp = np.concatenate([Fnp, pl[..., None]], axis=-1)
    T, S, nf = Fnp.shape
    ydmn = ydm.to_numpy()
    rows, folds = [], []
    starts = list(range(TEST_START_DAY, n - H - 2, TEST_LEN))
    for fi, t0 in enumerate(starts, 1):
        t1 = min(t0 + TEST_LEN, n - H - 2)
        tr_end = t0 - (H + 2)
        tr_idx = np.arange(70, tr_end)
        X = Fnp[tr_idx].reshape(-1, nf); yy = ydmn[tr_idx].reshape(-1); ok_ = liq_ok[tr_idx].reshape(-1)
        m = np.isfinite(yy) & ok_ & np.isfinite(X).sum(axis=1).astype(bool)
        X, yy = X[m], yy[m]
        lo_c, hi_c = np.percentile(yy, [0.5, 99.5]); yy = np.clip(yy, lo_c, hi_c)
        if shuffle:
            yy = rng.permutation(yy)
        if model_name == "gbm":
            mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.04, max_depth=4, min_samples_leaf=300, l2_regularization=5.0, random_state=0).fit(X, yy)
            pred = lambda Z: mdl.predict(Z)
        else:
            mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0) + 1e-9
            Xn = np.clip(np.nan_to_num((X - mu) / sd), -5, 5)
            mdl = Ridge(alpha=100.0).fit(Xn, yy)
            pred = lambda Z: mdl.predict(np.clip(np.nan_to_num((Z - mu) / sd), -5, 5))
        for t in range(t0, t1):
            valid = liq_ok[t] & np.isfinite(fwd[t]) & (np.isfinite(Fnp[t]).sum(axis=1) > 0)
            if valid.sum() < 2 * K + 10:
                continue
            idxv = np.where(valid)[0]
            p = pred(Fnp[t][idxv])
            order = np.argsort(p)
            lo_i, hi_i = idxv[order[:K]], idxv[order[-K:]]
            lr, sr = np.nanmean(fwd[t][hi_i]), np.nanmean(fwd[t][lo_i])
            gross = 0.5 * lr - 0.5 * sr
            rows.append((uni_idx[t + 1], fi, gross, gross - RT, lr, sr, float(np.nanmean(AGE[t][hi_i])), float(np.nanmean(AGE[t][lo_i])),
                         float(np.nanmean(MKT[t]))))
        folds.append(fi)
    return pd.DataFrame(rows, columns=["entry_ts", "fold", "gross", "net", "long_ret", "short_ret", "age_long", "age_short", "mkt"])


def summarize_run(df, H):
    d = pd.DatetimeIndex(df.entry_ts).date
    arrs = [v.to_numpy() for _, v in df.net.groupby(d)]
    r = np.random.default_rng(0)
    # bootstrap por bloques de H días para respetar la superposición de carteras
    blk = [np.concatenate(arrs[i:i + H]) for i in range(0, len(arrs) - H + 1, H)]
    m = [np.concatenate([blk[i] for i in r.integers(0, len(blk), len(blk))]).mean() for _ in range(3000)]
    lo, hi = np.percentile(m, [2.5, 97.5])
    ann = 365 / H
    sh = df.net.mean() / df.net.std() * np.sqrt(ann)
    byf = df.groupby("fold").agg(n=("net", "size"), gross=("gross", "mean"), net=("net", "mean"))
    return lo, hi, sh, byf


def evaluate_variant(H=7, model="gbm", stable=False, liq=0.30, drop=(), shuffle=False, plant=0.0, data=DATA, seed=1, rt=None):
    """Corre una variante completa y devuelve el DataFrame de carteras (entry_ts, fold, gross, net, ...). Usada por selector.py."""
    global K, RT, F_AGE
    if rt is not None:
        RT = rt
    uni = load_universe(data, min_bars=700 if stable else 150)
    idx, P = panels(uni)
    F = build_features(P)
    F_AGE = F["age"].copy()
    groups = dict(momentum=["ret1", "ret3", "ret7", "ret14", "ret30", "ret60", "rel7", "rel30", "rk_ret7", "rk_ret30"],
                  liquidez=["turn", "lturn30", "rk_lturn30", "age"])
    for item in drop:
        for k in groups.get(item, [item]):
            F.pop(k, None)
    quote30 = (P["close"] * P["volume"]).rolling(30).mean()
    liq_ok = (quote30.rank(axis=1, pct=True) >= liq).to_numpy()
    return run(H, model, shuffle, plant, np.random.default_rng(seed), idx, P, F, liq_ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--plant", type=float, default=0.0)
    ap.add_argument("--drop", default="", help="grupos o features a quitar: momentum,volatilidad,liquidez,posicion,vela,mercado | nombres")
    ap.add_argument("--stable", action="store_true", help="solo símbolos con historia completa desde el inicio")
    ap.add_argument("--liq", type=float, default=0.30, help="descarta el menor X de liquidez (0.30 = 30%)")
    ap.add_argument("--data", default=DATA, help="pickle diario (por defecto el cache de 2 años; usa el de colab/celda_historia_diaria.py para más años)")
    ap.add_argument("--only", default="", help="p.ej. 7:gbm para correr una sola variante")
    a = ap.parse_args()
    t0 = time.time()
    uni = load_universe(a.data, min_bars=700 if a.stable else 150)
    idx, P = panels(uni)
    F = build_features(P)
    global F_AGE
    F_AGE = F["age"].copy()
    GROUPS = dict(momentum=["ret1", "ret3", "ret7", "ret14", "ret30", "ret60", "rel7", "rel30", "rk_ret7", "rk_ret30"],
                  volatilidad=["vol7", "vol30", "volratio", "range7", "idio60", "beta60", "rk_vol30"],
                  liquidez=["turn", "lturn30", "rk_lturn30", "age"], posicion=["dhi30", "dlo30", "dhi90", "dlo90", "rk_dhi30"],
                  vela=["body", "upw", "loww", "rsi14", "rk_rsi14"], mercado=["mkt7", "btc7"])
    for item in [x for x in a.drop.split(",") if x]:
        for k in GROUPS.get(item, [item]):
            F.pop(k, None)
    quote30 = (P["close"] * P["volume"]).rolling(30).mean()
    liq_ok = (quote30.rank(axis=1, pct=True) >= a.liq).to_numpy()          # fuera el 30% menos líquido
    print(f"símbolos={len(uni)} | días={len(idx)} | features={len(F)} | K={K}/lado | costo por cartera {RT:.2%} | {time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(1)
    tag = "SHUFFLE" if a.shuffle else ("PLANT" if a.plant else "REAL")
    variants = [(H, mn) for H in (3, 7) for mn in ("gbm", "ridge")]
    if a.only:
        h_, m_ = a.only.split(":"); variants = [(int(h_), m_)]
    for H, mn in variants:
        if True:
            df = run(H, mn, a.shuffle, a.plant, rng, idx, P, F, liq_ok)
            lo, hi, sh, byf = summarize_run(df, H)
            print(f"\n[{tag}] H={H}d modelo={mn}: carteras={len(df)} bruto={df.gross.mean():+.3%} neto={df.net.mean():+.3%} IC95%[{lo:+.3%},{hi:+.3%}] sharpe_anual={sh:+.2f}")
            print("  por periodo:", "  ".join(f"P{f}: bruto {r.gross:+.2%} neto {r.net:+.2%} (n={int(r.n)})" for f, r in byf.iterrows()))
            print(f"  patas: long {df.long_ret.mean():+.2%} | short {df.short_ret.mean():+.2%} | mercado {df.mkt.mean():+.2%} | edad(rank) longs {df.age_long.mean():.2f} vs shorts {df.age_short.mean():.2f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
