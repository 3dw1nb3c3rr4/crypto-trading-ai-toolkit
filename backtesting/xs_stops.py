"""¿Mejora el bot cross-sectional con TP / SL calculados según la volatilidad de cada moneda?

Mismas predicciones walk-forward que ml_daily_xs_v2 (variante E por defecto, FO con --derivs), mismas cohortes de 7 días.
Para cada posición: ATR de 14 días / precio, conocido al cierre del día de la señal.
  SL = entrada × (1 ∓ k_sl·ATR%)   TP = entrada × (1 ± k_tp·ATR%)
Se recorre día a día (velas t+1 … t+H con máximo y mínimo). Si un mismo día toca SL y TP se asume SL (conservador).
Si el día abre más allá del SL (hueco) se sale al open. SL paga deslizamiento extra (0.10 %); sin toques se sale al open
de t+1+H, igual que la etiqueta del modelo. Comparación PAREADA contra el bot sin TP/SL sobre las mismas cohortes.
Uso: python xs_stops.py --data ../data/ohlcv_daily_long.pkl [--derivs derivados/derivs_metrics.pkl]
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

import ml_daily_xs as base
import ml_daily_xs_v2 as v2
from engine import Costs, load_universe

STOP_SLIP = Costs().stop_slippage


def atr_pct(P, n=14):
    h, l, c = P["high"], P["low"], P["close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    return (tr.rolling(n, min_periods=n).mean() / c).to_numpy()


def leg_returns(O, Hh, L, t, syms, side, atr, k_sl, k_tp, Hd):
    """Retornos de un lado (side=+1 largos, -1 cortos) con TP/SL; devuelve (retornos, sl_hits, tp_hits)."""
    entry = O[t + 1, syms]
    a = atr[t, syms]
    a = np.where(np.isfinite(a), a, np.nanmedian(atr[t]))
    sl = entry * (1 - side * k_sl * a) if k_sl else None
    tp = entry * (1 + side * k_tp * a) if k_tp else None
    out = np.full(len(syms), np.nan)
    done = np.zeros(len(syms), bool)
    nsl = ntp = 0
    for d in range(1, Hd + 1):
        j = t + d
        o, hi, lo = O[j, syms], Hh[j, syms], L[j, syms]
        if sl is not None:
            if side == 1:
                gap, hit = o <= sl, lo <= sl
            else:
                gap, hit = o >= sl, hi >= sl
            hit = hit & ~done
            px = np.where(gap, o, sl) * (1 - side * STOP_SLIP)
            out = np.where(hit, side * (px / entry - 1), out)
            nsl += int(hit.sum()); done |= hit
        if tp is not None:
            hit = ((hi >= tp) if side == 1 else (lo <= tp)) & ~done
            px = np.where((o >= tp) if side == 1 else (o <= tp), o, tp)
            out = np.where(hit, side * (px / entry - 1), out)
            ntp += int(hit.sum()); done |= hit
    fin = O[t + 1 + Hd, syms]
    out = np.where(done, out, side * (fin / entry - 1))
    return out, nsl, ntp


def run(P, PRED, fwd, folds, idx, atr, k_sl, k_tp):
    O, Hh, L = (P[k].to_numpy() for k in ("open", "high", "low"))
    T = O.shape[0]
    rows = []
    for t in np.where(folds > 0)[0]:
        if t + 1 + v2.H >= T:
            continue
        p = PRED[t]
        v = np.where(np.isfinite(p) & np.isfinite(fwd[t]))[0]
        if len(v) < v2.min_valid():
            continue
        k = v2.kk(len(v))
        o_ = v[np.argsort(p[v])]
        lr, sl1, tp1 = leg_returns(O, Hh, L, t, o_[-k:], +1, atr, k_sl, k_tp, v2.H)
        sr, sl2, tp2 = leg_returns(O, Hh, L, t, o_[:k], -1, atr, k_sl, k_tp, v2.H)
        g = 0.5 * np.nanmean(lr) + 0.5 * np.nanmean(sr)
        rows.append((idx[t + 1], g, g - v2.RT, (sl1 + sl2) / (2 * k), (tp1 + tp2) / (2 * k)))
    return pd.DataFrame(rows, columns=["entry_ts", "gross", "net", "sl_rate", "tp_rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/ohlcv_daily_long.pkl")
    ap.add_argument("--derivs", nargs="+", default=None)
    ap.add_argument("--kfrac", type=float, default=0.12)
    ap.add_argument("--test-len", type=int, default=180)
    a = ap.parse_args()
    t0 = time.time()
    v2.KFRAC = a.kfrac
    base.TEST_LEN = a.test_len
    uni = load_universe(a.data, min_bars=150)
    idx, P = base.panels(uni)
    Fbase, Falpha = base.build_features(P), v2.alpha_features(P)
    Ff, Fo = {}, {}
    name = "E"
    if a.derivs:
        import derivs_features as dfe
        Ff, Fo = dfe.build(P, idx, *v2.load_derivs(a.derivs), 1)
        name = "FO"
    F = v2.to_rank({**Fbase, **Falpha, **Ff, **Fo})
    quote30 = (P["close"] * P["volume"]).rolling(30).mean()
    liq_ok = (quote30.rank(axis=1, pct=True) >= 0.30).to_numpy()
    fwd_df, ylab = v2.labels(P, "rank")
    PRED, folds = v2.predict_walk_forward(F, fwd_df.to_numpy(), ylab, liq_ok, False, 0.0, np.random.default_rng(1))
    fwd = fwd_df.to_numpy()
    atr = atr_pct(P)
    print(f"variante {name} | {len(uni)} símbolos | predicciones listas en {time.time() - t0:.0f}s | ATR% mediano {np.nanmedian(atr):.2%}", flush=True)
    base_df = run(P, PRED, fwd, folds, idx, atr, None, None).set_index("entry_ts")
    lo, hi = v2.block_ci(base_df.net.to_numpy(), 7)
    print(f"\nSIN TP/SL: n={len(base_df)} neto {base_df.net.mean():+.3%} IC95%[{lo:+.3%},{hi:+.3%}] peor cohorte {base_df.net.min():+.2%}")
    res = []
    grid = list(itertools.product([None, 1.5, 2.0, 3.0, 4.0], [None, 2.0, 3.0, 4.0, 6.0]))[1:]
    for k_sl, k_tp in grid:
        df = run(P, PRED, fwd, folds, idx, atr, k_sl, k_tp).set_index("entry_ts")
        c = df.index.intersection(base_df.index)
        d = (df.net.loc[c] - base_df.net.loc[c]).to_numpy()
        dlo, dhi = v2.block_ci(d, 7)
        half = len(c) // 2
        res.append(dict(sl=k_sl, tp=k_tp, net=df.net.mean(), worst=df.net.min(), sl_rate=df.sl_rate.mean(), tp_rate=df.tp_rate.mean(),
                        diff=d.mean(), dlo=dlo, dhi=dhi, d1=d[:half].mean(), d2=d[half:].mean()))
        r = res[-1]
        print(f"SL {str(k_sl):>4}×ATR TP {str(k_tp):>4}×ATR | neto {r['net']:+.3%} | peor {r['worst']:+.2%} | SL toca {r['sl_rate']:.0%} TP toca {r['tp_rate']:.0%} | "
              f"vs sin TP/SL {r['diff']:+.3%} IC95%[{dlo:+.3%},{dhi:+.3%}] | 1ª mitad {r['d1']:+.3%} 2ª mitad {r['d2']:+.3%}"
              + ("  <- MEJORA significativa" if dlo > 0 else ("  <- EMPEORA" if dhi < 0 else "")), flush=True)
    best = max(res, key=lambda r: r["diff"])
    print(f"\nMejor combinación: SL {best['sl']}×ATR, TP {best['tp']}×ATR: {best['diff']:+.3%} por cohorte frente a sin TP/SL "
          f"(IC95% [{best['dlo']:+.3%}, {best['dhi']:+.3%}]). Se probaron {len(res)} combinaciones: elegir la mejor infla el resultado.")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
