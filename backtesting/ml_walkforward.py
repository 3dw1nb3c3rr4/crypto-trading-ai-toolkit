"""Walk-forward del modelo nuevo (gradient boosting) sobre velas 5m de OKX, con comisiones.

Diseño fijado ANTES de ver resultados (4 variantes, se reportan todas):
  horizonte H in {12, 48} velas (1h, 4h) x modo {dir, xs}
  - dir: modelo predice el retorno futuro de cada símbolo; opera si |pred| supera un umbral elegido en validación.
  - xs : modelo predice el retorno relativo al mercado; long los 3 mejores / short los 3 peores cada H velas.
Entrenamiento expansivo con 4 periodos de prueba nunca vistos; purga de 2 días; costo taker 0.05%/lado + slippage 0.02%/lado.
`--shuffle` baraja el objetivo en entrenamiento: control nulo (debe dar ~ -costo).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Costs, portfolio_curve, simulate_symbol, summarize  # noqa: E402
from ml_features import WARMUP, add_funding_features, build_dataset, load_panel  # noqa: E402

BAR_DAY = 288
TEST_STARTS_DAYS = (150, 210, 270, 330)
TEST_LEN_DAYS = 60
PURGE = 2 * BAR_DAY
TRAIN_STEP = 6
COSTS = Costs()
XS_K = 3


DEFAULT_PARAMS = dict(max_iter=150, learning_rate=0.04, max_depth=5, min_samples_leaf=400, l2_regularization=5.0)
SPACE = dict(max_depth=[3, 4, 6, 8], min_samples_leaf=[200, 400, 1000, 3000], learning_rate=[0.02, 0.04, 0.08],
             max_iter=[60, 150, 300], l2_regularization=[1.0, 5.0, 20.0])


def new_model(prm=None):
    return HistGradientBoostingRegressor(random_state=0, **(prm or DEFAULT_PARAMS))


def candidates(n_tune, seed=7):
    """Configuración por defecto + n_tune aleatorias. La elección se hace SOLO con la ventana de validación."""
    r = np.random.default_rng(seed)
    out = [DEFAULT_PARAMS]
    for _ in range(n_tune):
        out.append({k: v[r.integers(len(v))] for k, v in SPACE.items()})
    return out


def stack(X, y, syms, H, lo, hi, step, demean=False):
    xs, ys, idx = [], [], []
    yy = {s: y[(s, H)] for s in syms}
    if demean:
        m = np.nanmean(np.stack([yy[s] for s in syms]), axis=0)
        yy = {s: yy[s] - m for s in syms}
    for si, s in enumerate(syms):
        rows = np.arange(lo, hi, step)
        t = yy[s][rows]
        ok = np.isfinite(t) & X[s].iloc[rows].notna().sum(axis=1).to_numpy().astype(bool)
        rows = rows[ok]
        xs.append(X[s].iloc[rows].to_numpy()); ys.append(yy[s][rows]); idx.append(np.c_[np.full(len(rows), si), rows])
    return np.vstack(xs), np.concatenate(ys), np.vstack(idx)


def predict_all(model, X, syms, lo, hi):
    return {s: model.predict(X[s].iloc[lo:hi].to_numpy()) for s in syms}


def dir_trades(data, syms, preds, lo, hi, thr):
    """Trades direccionales por umbral con salida a H velas (tp/sl inalcanzables => salida por tiempo)."""
    out = []
    for s in syms:
        p = preds[s]
        sig = np.zeros(len(data[s]))
        sig[lo:hi] = np.where(p > thr, 1.0, np.where(p < -thr, -1.0, 0.0))
        out.append((s, sig))
    return out


def run_dir(data, syms, ts, X, y, H, shuffle, rng, n_tune=0):
    all_tr, rows = [], []
    n = len(ts)
    for fi, d0 in enumerate(TEST_STARTS_DAYS, 1):
        t0, t1 = d0 * BAR_DAY, min((d0 + TEST_LEN_DAYS) * BAR_DAY, n - H - 2)
        tr_end = t0 - PURGE
        val_lo = int(WARMUP + 0.8 * (tr_end - WARMUP))
        Xt, yt, _ = stack(X, y, syms, H, WARMUP, val_lo - PURGE, TRAIN_STEP)
        lo_c, hi_c = np.nanpercentile(yt, [0.5, 99.5]); yt = np.clip(yt, lo_c, hi_c)
        if shuffle:
            yt = rng.permutation(yt)
        best = None   # (val_exp, model, thr, params)
        for prm in candidates(n_tune):
            model = new_model(prm).fit(Xt, yt)
            pv = predict_all(model, X, syms, val_lo, tr_end)
            absall = np.concatenate([np.abs(pv[s]) for s in syms])
            cb = (None, -9.0)
            for q in (0.80, 0.90, 0.95):
                thr = float(np.quantile(absall, q))
                trs = []
                for s, sig in dir_trades(data, syms, pv, val_lo, tr_end, thr):
                    trs += simulate_symbol(data[s], sig, 10.0, 10.0, H, COSTS, start_idx=val_lo, end_idx=tr_end)
                if len(trs) >= 200:
                    e = float(np.mean([t[3] for t in trs]))
                    if e > cb[1]:
                        cb = (thr, e)
            thr = cb[0] if cb[0] is not None else float(np.quantile(absall, 0.95))
            if best is None or cb[1] > best[0]:
                best = (cb[1], model, thr, prm)
        val_exp, model, thr, prm = best
        pt = predict_all(model, X, syms, t0, t1)
        ftr = []
        for s, sig in dir_trades(data, syms, pt, t0, t1, thr):
            for (e, x, side, net, gross, reason) in simulate_symbol(data[s], sig, 10.0, 10.0, H, COSTS, start_idx=t0, end_idx=t1):
                ftr.append((s, ts[e], ts[x], side, net, gross, reason))
        f = pd.DataFrame(ftr, columns=["symbol", "entry_ms", "exit_ms", "side", "net", "gross", "reason"])
        f["fold"] = fi
        rows.append(dict(fold=fi, thr=thr, val_exp=val_exp, params=str(prm) if n_tune else "default", **summarize(f)))
        all_tr.append(f)
    return pd.concat(all_tr, ignore_index=True), pd.DataFrame(rows)


def xs_periods(pred, lo, hi, H, c, o, ts):
    vals = []
    for off in (0, H // 3, 2 * H // 3):
        for k in range(off, hi - lo, H):
            t = lo + k
            if t + H >= len(c):
                break
            order = np.argsort(pred[k]); lo_i, hi_i = order[:XS_K], order[-XS_K:]
            fwd = c[t + H] / o[t + 1] - 1.0
            gross = 0.5 * fwd[hi_i].mean() - 0.5 * fwd[lo_i].mean()
            vals.append((ts[t + 1], gross, gross - COSTS.round_trip()))
    return vals


def run_xs(data, syms, ts, X, y, H, shuffle, rng, n_tune=0):
    n = len(ts)
    c = np.stack([data[s]["close"].to_numpy(dtype="float64") for s in syms], axis=1)
    o = np.stack([data[s]["open"].to_numpy(dtype="float64") for s in syms], axis=1)
    per, rows = [], []
    for fi, d0 in enumerate(TEST_STARTS_DAYS, 1):
        t0, t1 = d0 * BAR_DAY, min((d0 + TEST_LEN_DAYS) * BAR_DAY, n - H - 2)
        tr_end = t0 - PURGE
        val_lo = int(WARMUP + 0.8 * (tr_end - WARMUP))
        Xt, yt, _ = stack(X, y, syms, H, WARMUP, val_lo - PURGE, TRAIN_STEP, demean=True)
        lo_c, hi_c = np.nanpercentile(yt, [0.5, 99.5]); yt = np.clip(yt, lo_c, hi_c)
        if shuffle:
            yt = rng.permutation(yt)
        best = None
        for prm in candidates(n_tune):
            model = new_model(prm).fit(Xt, yt)
            pv = np.stack([model.predict(X[s].iloc[val_lo:tr_end].to_numpy()) for s in syms], axis=1)
            v = xs_periods(pv, val_lo, tr_end, H, c, o, ts)
            score = float(np.mean([t[2] for t in v])) if v else -9.0
            if best is None or score > best[0]:
                best = (score, model, prm)
        val_exp, model, prm = best
        pt = np.stack([model.predict(X[s].iloc[t0:t1].to_numpy()) for s in syms], axis=1)
        f = pd.DataFrame(xs_periods(pt, t0, t1, H, c, o, ts), columns=["entry_ms", "gross", "net"])
        f["fold"] = fi
        rows.append(dict(fold=fi, n=len(f), val_net=val_exp, gross=f.gross.mean(), net=f.net.mean(), win=(f.net > 0).mean(),
                         params=str(prm) if n_tune else "default"))
        per.append(f)
    return pd.concat(per, ignore_index=True), pd.DataFrame(rows)


def day_bootstrap(entry_ms, net, n=4000, seed=0):
    day = pd.DatetimeIndex(pd.to_datetime(np.asarray(entry_ms), unit="ms", utc=True)).date
    arrs = [v.to_numpy() for _, v in pd.Series(np.asarray(net)).groupby(day)]
    r = np.random.default_rng(seed)
    m = [np.concatenate([arrs[i] for i in r.integers(0, len(arrs), len(arrs))]).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(HERE), "data", "okx_5m.pkl"))
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--plant", type=float, default=0.0, help="prueba de sensibilidad: agrega una feature = retorno futuro + ruido de N sigmas")
    ap.add_argument("--variants", default="dir12,dir48,xs12,xs48")
    ap.add_argument("--funding", default=None, help="pickle con funding (ver colab/celda_funding.py)")
    ap.add_argument("--funding-key", default=None, help="clave dentro del pickle: okx_funding | binance_funding")
    ap.add_argument("--tune", type=int, default=0, help="n configuraciones aleatorias de hiperparámetros a probar por fold (elección solo con validación)")
    a = ap.parse_args()
    t0 = time.time()
    data, syms, ts = load_panel(a.data)
    X, y = build_dataset(data, syms, ts, horizons=(12, 48))
    print(f"datos: {len(syms)} símbolos x {len(ts)} velas | features={X[syms[0]].shape[1]} | listo en {time.time()-t0:.0f}s", flush=True)
    if a.funding:
        raw = pickle.load(open(a.funding, "rb"))
        add_funding_features(X, syms, ts, raw[a.funding_key] if a.funding_key else raw)
        print(f"funding agregado: features={X[syms[0]].shape[1]}", flush=True)
    rng = np.random.default_rng(1)
    if a.plant:
        for v in a.variants.split(","):
            Hp = int(v[-2:])
            for s_ in syms:
                yy = y[(s_, Hp)]
                X[s_]["plant"] = (yy + a.plant * np.nanstd(yy) * rng.standard_normal(len(yy))).astype("float32")
            break
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tag = ("FUNDING_" if a.funding else "") + ("SHUFFLE" if a.shuffle else ("PLANT" if a.plant else "REAL")) + (f"_TUNE{a.tune}" if a.tune else "")
    for v in a.variants.split(","):
        H = int(v[-2:])
        mode = "dir" if v.startswith("dir") else "xs"
        print(f"\n===== {tag} | {v} (modo={mode}, H={H} velas = {H*5/60:.0f}h) =====", flush=True)
        if mode == "dir":
            tr, folds = run_dir(data, syms, ts, X, y, H, a.shuffle, rng, a.tune)
            print(folds[["fold", "thr", "val_exp", "n", "win", "exp_gross", "exp_net", "pf"]].round(4).to_string(index=False))
            if a.tune: print(folds[["fold", "params"]].to_string(index=False))
            s = summarize(tr); lo, hi = day_bootstrap(tr.entry_ms, tr.net)
            print(f"POOLED OOS: n={s['n']} win={s['win']:.1%} bruta={s['exp_gross']:+.3%} neta={s['exp_net']:+.3%} "
                  f"IC95%[{lo:+.3%},{hi:+.3%}] pf={s['pf']:.2f}")
            tr2 = tr.assign(entry_ts=pd.to_datetime(tr.entry_ms, unit="ms", utc=True), exit_ts=pd.to_datetime(tr.exit_ms, unit="ms", utc=True))
            _, st = portfolio_curve(tr2, alloc=0.05, max_pos=10)
            print(f"cartera 5%/trade máx 10: retorno={st['ret']:.1%} maxDD={st['maxdd']:.1%} trades={st['taken']}")
            for lab, rt in (("sin costos", 0), ("maker+maker", 2*0.0002 + 2*0.0002), ("taker (base)", COSTS.round_trip())):
                print(f"   sensibilidad {lab:13s} exp={np.mean(tr.gross - rt):+.3%}")
            tr.to_csv(os.path.join(HERE, "results", f"ml_{tag.lower()}_{v}_trades.csv"), index=False)
        else:
            per, folds = run_xs(data, syms, ts, X, y, H, a.shuffle, rng, a.tune)
            print(folds.round(5).to_string(index=False))
            lo, hi = day_bootstrap(per.entry_ms, per.net)
            ann = 365 * 24 / (H * 5 / 60)
            sh = per.net.mean() / per.net.std() * np.sqrt(ann)
            print(f"POOLED OOS: periodos={len(per)} bruta={per.gross.mean():+.3%} neta={per.net.mean():+.3%} "
                  f"IC95%[{lo:+.3%},{hi:+.3%}] sharpe_anual={sh:.2f} (costo por periodo {COSTS.round_trip():.3%})")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
