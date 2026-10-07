"""Ideas de repos de referencia aplicadas al candidato cross-sectional diario (ml_daily_xs.py).

  Qlib (microsoft/qlib): features tipo Alpha158 (velas, MA/STD/MAX/MIN/QTL/RANK/RSV/IMAX/IMIN/CORR/CNTP/SUMP/VMA/VSTD...),
      normalización cross-sectional (rank por fecha), etiqueta de ranking, LightGBM muy regularizado, evaluación por IC/RankIC
      y estrategia TopkDropout (mantener K posiciones y reemplazar solo n_drop por día para recortar costos).
  FreqAI (freqtrade): features en varios periodos; filtrar valores atípicos.   ML4T (Jansen): walk-forward + Sharpe deflactado.

Variantes fijadas ANTES de ver resultados (H=7 días, K=10 por lado):
  A  features base (33) crudas,   etiqueta demeaned,  carteras de 7 días       (= referencia de ml_daily_xs.py)
  B  features alpha crudas,       etiqueta demeaned,  carteras de 7 días
  C  features base en rank,       etiqueta rank,      carteras de 7 días
  D  features alpha en rank,      etiqueta rank,      carteras de 7 días
  E  base+alpha en rank,          etiqueta rank,      carteras de 7 días
  Ad A con portafolio TopkDropout (n_drop=2)
  Ed E con portafolio TopkDropout (n_drop=2)
Controles: --shuffle (nulo) y --plant (sensibilidad). Robustez: --stable, --liq.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ml_daily_xs as base  # noqa: E402
from engine import Costs, load_universe  # noqa: E402

H, K, N_DROP = 7, 10, 2
BETA_NEUTRAL = False    # dimensiona las patas para que beta_largos * w_largos = beta_cortos * w_cortos
BETA = None             # matriz (T, S) de beta60 de cada símbolo (se asigna en main / run_named)
KFRAC = None            # si se define, K = 12% de los símbolos válidos cada día (universos que cambian de tamaño)
NEW_DATA_CUTOFF = "2024-09-05"   # todo lo anterior nunca se usó en análisis previos (el cache de 2 años empieza aquí)


def kk(nvalid):
    return max(3, int(round(KFRAC * nvalid))) if KFRAC else K


def min_valid():
    return 20 if KFRAC else 2 * K + 10
RT = Costs().round_trip()
ONE_WAY = RT / 2
MARKET_LEVEL = ("mkt7", "btc7")


def alpha_features(P):
    o, h, l, c, v = (P[k] for k in ("open", "high", "low", "close", "volume"))
    lv = np.log1p(v)
    ret = c.pct_change()
    absr, up = ret.abs(), ret.clip(lower=0)
    sp = (h - l).replace(0, np.nan)
    F = {"kmid": (c - o) / o, "klen": (h - l) / o, "kmid2": (c - o) / sp, "kup": (h - np.maximum(o, c)) / o,
         "kup2": (h - np.maximum(o, c)) / sp, "klow": (np.minimum(o, c) - l) / o, "klow2": (np.minimum(o, c) - l) / sp,
         "ksft": (2 * c - h - l) / o, "ksft2": (2 * c - h - l) / sp}
    dvol = np.log1p(v / v.shift(1))

    def fit(n):
        x = np.arange(n) - (n - 1) / 2
        sxx = (x ** 2).sum()

        def stats(a):
            ym = a.mean(); dy = a - ym
            sxy = (x * dy).sum(); syy = (dy ** 2).sum()
            slope = sxy / sxx
            return slope, (sxy ** 2 / (sxx * syy) if syy > 0 else np.nan), a[-1] - (ym + slope * x[-1])
        return stats

    for n in (5, 10, 20, 30, 60):
        F[f"roc{n}"] = c.shift(n) / c - 1
        F[f"ma{n}"] = c.rolling(n).mean() / c
        F[f"std{n}"] = c.rolling(n).std() / c
        F[f"max{n}"] = h.rolling(n).max() / c
        F[f"min{n}"] = l.rolling(n).min() / c
        F[f"qtlu{n}"] = c.rolling(n).quantile(0.8) / c
        F[f"qtld{n}"] = c.rolling(n).quantile(0.2) / c
        F[f"rank{n}"] = c.rolling(n).rank(pct=True)
        F[f"rsv{n}"] = (c - l.rolling(n).min()) / (h.rolling(n).max() - l.rolling(n).min() + 1e-12)
        F[f"imax{n}"] = h.rolling(n).apply(lambda a: (len(a) - 1 - np.argmax(a)) / len(a), raw=True)
        F[f"imin{n}"] = l.rolling(n).apply(lambda a: (len(a) - 1 - np.argmin(a)) / len(a), raw=True)
        F[f"corr{n}"] = c.rolling(n).corr(lv)
        F[f"cord{n}"] = ret.rolling(n).corr(dvol)
        F[f"cntp{n}"] = (ret > 0).astype(float).where(ret.notna()).rolling(n).mean()
        F[f"sump{n}"] = up.rolling(n).sum() / (absr.rolling(n).sum() + 1e-12)
        F[f"vma{n}"] = v.rolling(n).mean() / (v + 1e-12)
        F[f"vstd{n}"] = v.rolling(n).std() / (v + 1e-12)
        F[f"wvma{n}"] = (absr * v).rolling(n).std() / ((absr * v).rolling(n).mean() + 1e-12)
        if n >= 10:
            st = fit(n)
            cn = c / c.rolling(n).mean()
            F[f"beta{n}"] = cn.rolling(n).apply(lambda a: st(a)[0], raw=True)
            F[f"rsqr{n}"] = cn.rolling(n).apply(lambda a: st(a)[1], raw=True)
            F[f"resi{n}"] = cn.rolling(n).apply(lambda a: st(a)[2], raw=True)
    return F


def to_rank(F):
    return {k: (v if k in MARKET_LEVEL else v.rank(axis=1, pct=True)) for k, v in F.items() if k not in MARKET_LEVEL}


def load_derivs(paths):
    """Une varios pickles de download_derivs.py / download_binance_metrics.py. Por tipo (funding / oi) y símbolo se usa UNA sola fuente:
    la de mayor historial (no se mezclan fuentes porque las unidades y niveles de OI difieren entre exchanges)."""
    import pickle
    fund, oi = {}, {}

    def span(df):
        return (float(df["ts"].max()) - float(df["ts"].min())) if len(df) > 1 else 0.0
    for p in paths or []:
        d = pickle.load(open(p, "rb"))
        for store, key in ((fund, "funding"), (oi, "oi")):
            for sym, df in d.get(key, {}).items():
                if len(df) and (sym not in store or span(df) > span(store[sym])):
                    store[sym] = df
    return fund, oi


def make_variants(Fbase, Falpha, Ff=None, Fo=None):
    extra = {}
    if Ff:
        extra["F"] = (to_rank({**Fbase, **Falpha, **Ff}), "rank", "cohort")
    if Fo:
        extra["O"] = (to_rank({**Fbase, **Falpha, **Fo}), "rank", "cohort")
    if Ff and Fo:
        extra["FO"] = (to_rank({**Fbase, **Falpha, **Ff, **Fo}), "rank", "cohort")
    return {**extra,
        "A": (Fbase, "demean", "cohort"), "B": (Falpha, "demean", "cohort"),
        "C": (to_rank(Fbase), "rank", "cohort"), "D": (to_rank(Falpha), "rank", "cohort"),
        "E": (to_rank({**Fbase, **Falpha}), "rank", "cohort"),
        "Ad": (Fbase, "demean", "dropout"), "Ed": (to_rank({**Fbase, **Falpha}), "rank", "dropout"),
    }


def labels(P, mode):
    o = P["open"]
    fwd = o.shift(-(1 + H)) / o.shift(-1) - 1.0
    if mode == "demean":
        return fwd, fwd.sub(fwd.mean(axis=1), axis=0)
    r = fwd.rank(axis=1, pct=True)
    return fwd, (r - 0.5) * 3.4641


def predict_walk_forward(F, fwd, ylab, liq_ok, shuffle, plant, rng):
    names = list(F)
    X3 = np.stack([F[k].to_numpy() for k in names], axis=-1)
    if plant:
        pl = ylab.to_numpy() + plant * np.nanstd(ylab.to_numpy()) * rng.standard_normal(ylab.shape)
        X3 = np.concatenate([X3, pl[..., None]], axis=-1)
    T, S, nf = X3.shape
    y3 = ylab.to_numpy()
    PRED = np.full((T, S), np.nan)
    folds = np.zeros(T, dtype=int)
    for fi, t0 in enumerate(range(base.TEST_START_DAY, T - H - 2, base.TEST_LEN), 1):
        t1 = min(t0 + base.TEST_LEN, T - H - 2)
        tr = np.arange(70, t0 - (H + 2))
        X = X3[tr].reshape(-1, nf); y = y3[tr].reshape(-1); ok = liq_ok[tr].reshape(-1)
        m = np.isfinite(y) & ok & np.isfinite(X).sum(axis=1).astype(bool)
        X, y = X[m], y[m]
        lo, hi = np.percentile(y, [0.5, 99.5]); y = np.clip(y, lo, hi)
        if shuffle:
            y = rng.permutation(y)
        mdl = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=400,
                                            l2_regularization=50.0, max_features=0.8, random_state=0).fit(X, y)
        for t in range(t0, t1):
            valid = liq_ok[t] & np.isfinite(fwd[t]) & (np.isfinite(X3[t]).sum(axis=1) > 0)
            if valid.sum() >= min_valid():
                idv = np.where(valid)[0]
                PRED[t, idv] = mdl.predict(X3[t][idv])
        folds[t0:t1] = fi
    return PRED, folds


def cohort_returns(PRED, fwd, folds, idx, neutral=None):
    neutral = BETA_NEUTRAL if neutral is None else neutral
    rows = []
    for t in np.where(folds > 0)[0]:
        p = PRED[t]; v = np.where(np.isfinite(p) & np.isfinite(fwd[t]))[0]
        if len(v) < min_valid():
            continue
        k = kk(len(v))
        o_ = v[np.argsort(p[v])]
        lr, sr = np.mean(fwd[t][o_[-k:]]), np.mean(fwd[t][o_[:k]])
        wl = 0.5
        if neutral and BETA is not None:
            bl = np.clip(np.nanmean(BETA[t][o_[-k:]]), 0.2, 3.0); bs = np.clip(np.nanmean(BETA[t][o_[:k]]), 0.2, 3.0)
            if np.isfinite(bl) and np.isfinite(bs):
                wl = bs / (bl + bs)          # beta_largos*wl = beta_cortos*(1-wl)
        g = wl * lr - (1 - wl) * sr
        rows.append((idx[t + 1], folds[t], g, g - RT, float(np.mean(fwd[t][v])), k, len(v), wl))
    return pd.DataFrame(rows, columns=["entry_ts", "fold", "gross", "net", "mkt", "k", "nvalid", "w_long"])


def dropout_returns(PRED, P, folds, idx):
    """TopkDropout long/short: mantiene K por lado y reemplaza n_drop por día. Rendimiento open->open diario, costo por posición cambiada."""
    o = P["open"].to_numpy()
    oo = np.full_like(o, np.nan); oo[:-1] = o[1:] / o[:-1] - 1          # retorno de la apertura t a la t+1
    longs, shorts, rows = set(), set(), []
    for t in np.where(folds > 0)[0]:
        p = PRED[t]; v = np.where(np.isfinite(p))[0]
        if len(v) < min_valid() or t + 2 >= len(o):
            continue
        order = v[np.argsort(p[v])]
        k = kk(len(v))
        changed = 0
        if not longs:
            longs, shorts = set(order[-k:]), set(order[:k]); changed = 2 * k
        else:
            vs = set(v)
            longs &= vs; shorts &= vs
            for side, hold, cand_order in (("L", longs, order[::-1]), ("S", shorts, order)):
                worst = sorted(hold, key=lambda i: p[i], reverse=(side == "S"))[:N_DROP]          # peores de lo que ya tengo
                cands = [i for i in cand_order if i not in hold and i not in (shorts if side == "L" else longs)][:N_DROP]
                for w_, c_ in zip(worst, cands):
                    better = p[c_] > p[w_] if side == "L" else p[c_] < p[w_]
                    if better:
                        hold.discard(w_); hold.add(c_); changed += 2
                while len(hold) < k:                                     # reponer si se perdió alguna o creció el universo
                    c_ = next(i for i in cand_order if i not in hold and i not in (shorts if side == "L" else longs)); hold.add(c_); changed += 1
                while len(hold) > k:                                     # recortar si el universo se achicó
                    w_ = min(hold, key=lambda i: p[i]) if side == "L" else max(hold, key=lambda i: p[i]); hold.discard(w_); changed += 1
        r_long = np.nanmean(oo[t + 1][list(longs)]); r_short = np.nanmean(oo[t + 1][list(shorts)])
        gross = 0.5 * r_long - 0.5 * r_short
        cost = changed * (0.5 / k) * ONE_WAY                              # cada posición que entra o sale paga un lado
        rows.append((idx[t + 1], folds[t], gross, gross - cost, changed))
    return pd.DataFrame(rows, columns=["entry_ts", "fold", "gross", "net", "changed"])


def ic_stats(PRED, fwd, folds):
    ics = []
    for t in np.where(folds > 0)[0]:
        p = PRED[t]; v = np.isfinite(p) & np.isfinite(fwd[t])
        if v.sum() >= 30:
            ics.append(spearmanr(p[v], fwd[t][v])[0])
    ics = np.array(ics)
    r = np.random.default_rng(0)
    blk = [ics[i:i + H] for i in range(0, len(ics) - H + 1, H)]
    bs = [np.concatenate([blk[i] for i in r.integers(0, len(blk), len(blk))]).mean() for _ in range(2000)]
    return ics.mean(), ics.mean() / ics.std(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def dsr(returns, n_trials):
    r = np.asarray(returns); T = len(r); sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    sd = np.sqrt((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1))
    emc = 0.5772156649
    sr0 = sd * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    return float(norm.cdf((sr - sr0) / sd))


def block_ci(vals, step, n=3000, seed=0):
    r = np.random.default_rng(seed)
    blk = [vals[i:i + step] for i in range(0, len(vals) - step + 1, step)]
    m = [np.concatenate([blk[i] for i in r.integers(0, len(blk), len(blk))]).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def regress_mkt(df):
    x, y_ = df.mkt.to_numpy(), df.net.to_numpy()
    b_, a_ = np.polyfit(x, y_, 1)
    rr = np.random.default_rng(0)
    nb = len(y_) // 7
    bs_ = []
    for _ in range(1500):
        ii = np.concatenate([np.arange(j * 7, j * 7 + 7) for j in rr.integers(0, nb, nb)])
        bb, aa = np.polyfit(x[ii], y_[ii], 1); bs_.append((aa, bb))
    bs_ = np.array(bs_)
    return dict(alpha=a_, beta=b_, alpha_ci=np.percentile(bs_[:, 0], [2.5, 97.5]), beta_ci=np.percentile(bs_[:, 1], [2.5, 97.5]))


def evaluate(variant_name, spec, P, idx, liq_ok, shuffle=False, plant=0.0, seed=1):
    F, lab, port = spec
    fwd_df, ylab = labels(P, lab)
    rng = np.random.default_rng(seed)
    PRED, folds = predict_walk_forward(F, fwd_df.to_numpy(), ylab, liq_ok, shuffle, plant, rng)
    fwd = fwd_df.to_numpy()
    ic, icir, ic_lo, ic_hi = ic_stats(PRED, fwd, folds)
    if port == "cohort":
        df = cohort_returns(PRED, fwd, folds, idx)
        step, per_year = H, 365 / H
        daily_equiv = df.net.to_numpy() / H
    else:
        df = dropout_returns(PRED, P, folds, idx)
        step, per_year = 7, 365
        daily_equiv = df.net.to_numpy()
    lo, hi = block_ci(df.net.to_numpy(), step)
    sharpe = df.net.mean() / df.net.std() * np.sqrt(per_year)
    byf = df.groupby("fold").net.mean()
    eq = np.cumprod(1 + daily_equiv); dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    reg = regress_mkt(df) if ("mkt" in df and port == "cohort") else None
    plain = None
    if BETA_NEUTRAL and port == "cohort":                       # misma predicción, cartera NO neutral, para comparar
        dfp = cohort_returns(PRED, fwd, folds, idx, neutral=False)
        plain = dict(reg=regress_mkt(dfp), net=dfp.net.mean(), ci=block_ci(dfp.net.to_numpy(), 7), df=dfp,
                     yearly=dfp.groupby(pd.DatetimeIndex(dfp.entry_ts).year).agg(n=("net", "size"), net=("net", "mean"), mkt=("mkt", "mean")))
    yr = df.groupby(pd.DatetimeIndex(df.entry_ts).year).agg(n=("net", "size"), net=("net", "mean"),
                                                           mkt=("mkt", "mean") if "mkt" in df else ("net", "size"))
    new = df[pd.DatetimeIndex(df.entry_ts) < pd.Timestamp(NEW_DATA_CUTOFF, tz="UTC")]
    new_ci = block_ci(new.net.to_numpy(), step) if len(new) > 4 * step else (np.nan, np.nan)
    return dict(plain=plain, reg=reg, yearly=yr, new_n=len(new), new_net=float(new.net.mean()) if len(new) else np.nan, new_ci=new_ci,
                name=variant_name, port=port, n=len(df), ic=ic, icir=icir, ic_lo=ic_lo, ic_hi=ic_hi,
                gross=df.gross.mean(), net=df.net.mean(), ci_lo=lo, ci_hi=hi, sharpe=sharpe,
                folds=[float(x) for x in byf.values], maxdd=dd, dsr20=dsr(df.net.to_numpy(), 20),
                dsr700=dsr(df.net.to_numpy(), 700), turnover=float(df.changed.mean()) if "changed" in df else None, df=df)


def run_named(vname, stable=False, liq=0.30, rt=None, data=None, shuffle=False):
    """Prepara datos y corre una variante con nombre (A..Ed). Usada por selector.py y por el bot."""
    global RT, ONE_WAY
    old = RT
    if rt is not None:
        RT, ONE_WAY = rt, rt / 2
    try:
        uni = load_universe(data or base.DATA, min_bars=700 if stable else 150)
        idx, P = base.panels(uni)
        Fbase, Falpha = base.build_features(P), alpha_features(P)
        global BETA
        BETA = Fbase["beta60"].to_numpy()
        quote30 = (P["close"] * P["volume"]).rolling(30).mean()
        liq_ok = (quote30.rank(axis=1, pct=True) >= liq).to_numpy()
        return evaluate(vname, make_variants(Fbase, Falpha)[vname], P, idx, liq_ok, shuffle=shuffle)
    finally:
        RT, ONE_WAY = old, old / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=base.DATA)
    ap.add_argument("--variants", default="A,B,C,D,E,Ad,Ed")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--plant", type=float, default=0.0)
    ap.add_argument("--stable", action="store_true")
    ap.add_argument("--liq", type=float, default=0.30)
    ap.add_argument("--derivs", nargs="+", default=None, help="pickles de download_derivs.py (funding / open interest)")
    ap.add_argument("--oi-lag", type=int, default=1, help="días de retraso del OI (1 = seguro; 0 solo si la foto es instantánea)")
    ap.add_argument("--betaneutral", action="store_true", help="patas dimensionadas para beta neto cero")
    ap.add_argument("--save", default=None, help="guardar las cohortes en este CSV")
    ap.add_argument("--kfrac", type=float, default=None, help="K = fracción de símbolos válidos por lado (p.ej. 0.12)")
    ap.add_argument("--test-len", type=int, default=90, help="días por periodo de prueba (reentrena en cada uno)")
    ap.add_argument("--stable-bars", type=int, default=700, help="velas mínimas para --stable")
    a = ap.parse_args()
    t0 = time.time()
    global KFRAC, BETA_NEUTRAL, BETA
    KFRAC = a.kfrac
    BETA_NEUTRAL = a.betaneutral
    base.TEST_LEN = a.test_len
    uni = load_universe(a.data, min_bars=a.stable_bars if a.stable else 150)
    idx, P = base.panels(uni)
    Fbase, Falpha = base.build_features(P), alpha_features(P)
    BETA = Fbase["beta60"].to_numpy()
    quote30 = (P["close"] * P["volume"]).rolling(30).mean()
    liq_ok = (quote30.rank(axis=1, pct=True) >= a.liq).to_numpy()
    Ff, Fo = {}, {}
    if a.derivs:
        import derivs_features as dfe
        fund, oi = load_derivs(a.derivs)
        Ff, Fo = dfe.build(P, idx, fund, oi, a.oi_lag)
        cov = {k: int(v.notna().any().sum()) for k, v in {**Ff, **Fo}.items() if k in ("fund_7d", "oi_chg7")}
        first = {k: str(v.notna().sum(axis=1).gt(20).idxmax().date()) for k, v in {**Ff, **Fo}.items() if k in ("fund_7d", "oi_chg7")}
        print(f"derivados: funding {len(fund)} símbolos, OI {len(oi)} | símbolos con dato: {cov} | primer día con >20 símbolos: {first}", flush=True)
    specs = make_variants(Fbase, Falpha, Ff or None, Fo or None)
    print(f"símbolos={len(uni)} días={len(idx)} | features base={len(Fbase)} alpha={len(Falpha)} | K={K} n_drop={N_DROP} | "
          f"costo ida/vuelta {RT:.2%} | listo en {time.time()-t0:.0f}s", flush=True)
    tag = "SHUFFLE" if a.shuffle else ("PLANT" if a.plant else "REAL")
    results = {}
    for v in a.variants.split(","):
        r = evaluate(v, specs[v], P, idx, liq_ok, a.shuffle, a.plant)
        results[v] = r
        unit = "por cohorte 7d" if r["port"] == "cohort" else "por día"
        print(f"\n[{tag}] {v:2s} ({r['port']}): n={r['n']} IC={r['ic']:+.3f} [{r['ic_lo']:+.3f},{r['ic_hi']:+.3f}] ICIR={r['icir']:+.2f} | "
              f"bruto={r['gross']:+.3%} neto={r['net']:+.3%} {unit} IC95%[{r['ci_lo']:+.3%},{r['ci_hi']:+.3%}] Sharpe={r['sharpe']:+.2f} "
              f"maxDD={r['maxdd']:.1%} DSR(N=20)={r['dsr20']:.2f} DSR(N=700)={r['dsr700']:.2f}"
              + (f" rotación={r['turnover']:.1f} cambios/día" if r["turnover"] else ""))
        print("      por periodo:", " ".join(f"{x:+.2%}" for x in r["folds"]), flush=True)
        if "mkt" in r["df"]:
            print("      por año:", " | ".join(f"{y}: neto {row.net:+.2%} mercado {row.mkt:+.2%} (n={int(row.n)})" for y, row in r["yearly"].iterrows()))
        else:
            print("      por año:", " | ".join(f"{y}: neto {row.net:+.3%}/día (n={int(row.n)})" for y, row in r["yearly"].iterrows()))
        if r["reg"]:
            g = r["reg"]
            print(f"      REGRESIÓN contra el mercado: beta={g['beta']:+.3f} [{g['beta_ci'][0]:+.3f},{g['beta_ci'][1]:+.3f}] | "
                  f"alfa={g['alpha']:+.3%} por cohorte IC95%[{g['alpha_ci'][0]:+.3%},{g['alpha_ci'][1]:+.3%}]"
                  + (f" | peso medio de largos {r['df'].w_long.mean():.2f}" if BETA_NEUTRAL else ""))
        if r.get("plain"):
            g, pl = r["plain"]["reg"], r["plain"]
            print(f"      (misma predicción SIN neutralizar beta: neto {pl['net']:+.3%} IC95%[{pl['ci'][0]:+.3%},{pl['ci'][1]:+.3%}] | "
                  f"beta={g['beta']:+.3f} [{g['beta_ci'][0]:+.3f},{g['beta_ci'][1]:+.3f}] alfa={g['alpha']:+.3%} [{g['alpha_ci'][0]:+.3%},{g['alpha_ci'][1]:+.3%}])")
            print("      por año SIN neutralizar:", " | ".join(f"{y}: {row.net:+.2%} (mercado {row.mkt:+.2%})" for y, row in pl["yearly"].iterrows()))
        if a.save:
            r["df"].to_csv(a.save.replace(".csv", f"_{v}.csv"), index=False)
        if r["new_n"]:
            print(f"      DATOS NUEVOS (antes de {NEW_DATA_CUTOFF}, nunca usados): n={r['new_n']} neto={r['new_net']:+.3%} IC95%[{r['new_ci'][0]:+.3%},{r['new_ci'][1]:+.3%}]")
    if "E" in results:                         # comparación pareada contra E sobre las mismas fechas (con datos de derivados)
        first_day = None
        if Ff or Fo:
            cnt = sum(((F_[k].notna().sum(axis=1) > 20) for F_ in (Ff, Fo) for k in F_ if k in ("fund_7d", "oi_chg7")))
            first_day = (cnt > 0).idxmax() + pd.Timedelta(days=45) if (cnt > 0).any() else None
        for v in [x for x in results if x in ("F", "O", "FO")]:
            A = results["E"]["df"].set_index("entry_ts").net; B = results[v]["df"].set_index("entry_ts").net
            common = A.index.intersection(B.index)
            if first_day is not None:
                common = common[common >= first_day]
            if len(common) > 30:
                d = (B.loc[common] - A.loc[common]).to_numpy()
                lo, hi = block_ci(d, 7)
                print(f"\nPAREADO {v} - E sobre {len(common)} cohortes desde {common.min().date()}: diferencia media {d.mean():+.3%} "
                      f"IC95%[{lo:+.3%},{hi:+.3%}] | neto E={A.loc[common].mean():+.3%} {v}={B.loc[common].mean():+.3%}"
                      f" | {'MEJORA significativa' if lo > 0 else ('EMPEORA significativamente' if hi < 0 else 'diferencia no significativa')}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
