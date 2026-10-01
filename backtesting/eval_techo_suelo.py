"""Evalúa las señales del modelo Techo/Suelo (Cerebro 1) con comisiones, SOLO en el tramo de validación
(último 15% de cada símbolo, el que el propio entrenamiento reservó) para no medir sobre datos de entrenamiento.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bots", "techosuelo"))
import binance_topsuelo_bot as tm  # noqa: E402
import binance_meta_filtro as mf  # noqa: E402
from engine import Costs, load_universe, portfolio_curve, summarize, trades_frame  # noqa: E402

VAL_FRAC = 0.15


def main():
    uni = load_universe(os.path.join(ROOT, "data", "trading_history", "ohlcv_cache_2y.pkl"))
    model, bundle = tm.load_bundle(os.path.join(ROOT, "models", "techo_suelo_model.pt"), device="cpu")
    costs = Costs()
    sigs, starts, probs = {}, {}, {}
    n_sig = 0
    for sym, df in uni.items():
        sc = mf.score_symbol_full_history(model, bundle, df, device="cpu")
        if sc is None:
            continue
        sc = sc.set_index("ts")
        p_t, p_s, p_n = sc["p_techo"], sc["p_suelo"], sc["p_neutral"]
        side = pd.Series(0.0, index=sc.index)
        side[(p_t > p_n) & (p_t >= p_s)] = -1.0     # TECHO -> SHORT
        side[(p_s > p_n) & (p_s > p_t)] = 1.0       # SUELO -> LONG
        strength = np.where(side < 0, p_t, np.where(side > 0, p_s, 0.0))
        s = pd.Series(0.0, index=pd.DatetimeIndex(df["ts"]))
        s.loc[side.index] = side.values
        p = pd.Series(0.0, index=pd.DatetimeIndex(df["ts"]))
        p.loc[side.index] = strength
        sigs[sym], probs[sym] = s.to_numpy(), p.to_numpy()
        n = len(df)
        starts[sym] = int(tm_cut(n))
        n_sig += int((side != 0).sum())
    print(f"símbolos evaluados={len(sigs)} | señales totales (todo el historial)={n_sig}")

    t_cut = {s: uni[s]["ts"].iloc[starts[s]] for s in sigs}
    print("rango de fecha de corte de validación:", min(t_cut.values()).date(), "->", max(t_cut.values()).date())
    sub = {s: uni[s] for s in sigs}

    def run(tp, sl, hold, min_p=0.0, side_filter=0):
        out = []
        for s, df in sub.items():
            sg = np.where(probs[s] >= min_p, sigs[s], 0.0)
            if side_filter:
                sg = np.where(sg == side_filter, sg, 0.0)
            tr = trades_frame({s: df}, {s: sg}, tp, sl, hold, costs, start_ts=t_cut[s])
            out.append(tr)
        return pd.concat(out, ignore_index=True)

    print("\n=== Config nativa del modelo (TP 4% / SL 4% / 7 días), umbral p>=0 ===")
    for label, sf in (("ambos", 0), ("solo LONG (suelo)", 1), ("solo SHORT (techo)", -1)):
        s = summarize(run(0.04, 0.04, 7, 0.0, sf))
        print(f"{label:20s} n={s['n']:4d} win={s['win']:.1%} bruta={s['exp_gross']:+.3%} neta={s['exp_net']:+.3%} pf={s['pf']:.2f}")

    print("\n=== Grilla (solo tramo de validación; ojo: más combinaciones = más riesgo de azar) ===")
    rows = []
    for tp in (0.03, 0.04, 0.06, 0.10):
        for sl in (0.03, 0.04, 0.06):
            for hold in (3, 7):
                for min_p in (0.0, 0.5, 0.7):
                    tr = run(tp, sl, hold, min_p)
                    s = summarize(tr)
                    if s["n"] >= 30:
                        rows.append(dict(tp=tp, sl=sl, hold=hold, min_p=min_p, **s))
    g = pd.DataFrame(rows).sort_values("exp_net", ascending=False)
    g.to_csv(os.path.join(HERE, "results", "techo_suelo_val_grid.csv"), index=False)
    print(g.head(8).round(4).to_string(index=False))
    print(f"\nconfigs con expectativa neta > 0: {(g.exp_net > 0).sum()} de {len(g)}")

    base = run(0.04, 0.04, 7, 0.0)
    curve, st = portfolio_curve(base, alloc=0.10, max_pos=10)
    print(f"\ncartera (config nativa, 10%/trade, máx 10): retorno={st['ret']:.1%} maxDD={st['maxdd']:.1%} trades={st['taken']}")


def tm_cut(n: int) -> int:
    max_w = max(tm.SCALES.values())
    return int(max_w + (1 - VAL_FRAC) * (n - tm.FUTURE_HORIZON_DAYS - max_w))


if __name__ == "__main__":
    main()
