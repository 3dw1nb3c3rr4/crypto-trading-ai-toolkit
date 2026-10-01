"""Optimización walk-forward anclada con comisiones.

1. Simula cada configuración (estrategia x parámetros x TP/SL/hold) sobre TODO el histórico.
2. En cada fold se elige la mejor configuración usando SOLO datos de entrenamiento
   (por expectativa neta de comisiones, con mínimo de trades) y se mide en el periodo
   siguiente, que esa elección nunca vio.
3. Los trades fuera de muestra de los 3 folds se unen y se reporta cartera, drawdown y sensibilidad a costos.
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Costs, FEE_MAKER, load_universe, portfolio_curve, summarize, trades_frame  # noqa: E402
from strategies import EXIT_GRID, REGISTRY  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "trading_history", "ohlcv_cache_2y.pkl")
MIN_TRAIN_TRADES = 150
COSTS = Costs()

UNIVERSE = load_universe(DATA)
SIGNALS: dict[tuple, dict[str, np.ndarray]] = {}


def build_signals():
    for name, (fn, grid) in REGISTRY.items():
        for i, p in enumerate(grid):
            SIGNALS[(name, i)] = {s: fn(df, **p) for s, df in UNIVERSE.items()}


def run_cfg(cfg):
    (name, i), tp, sl, hold = cfg
    tr = trades_frame(UNIVERSE, SIGNALS[(name, i)], tp, sl, hold, COSTS)
    tr["cfg"] = f"{name}#{i}|tp{tp}|sl{sl}|h{hold}"
    tr["strategy"] = name
    return cfg, tr


def main():
    t0 = time.time()
    build_signals()
    cfgs = [(k, tp, sl, h) for k in SIGNALS for tp, sl, h in
            itertools.product(EXIT_GRID["tp"], EXIT_GRID["sl"], EXIT_GRID["hold"])]
    print(f"universo={len(UNIVERSE)} símbolos | configuraciones={len(cfgs)} | CPU={mp.cpu_count()}", flush=True)
    with mp.Pool(min(mp.cpu_count(), 8)) as pool:
        results = pool.map(run_cfg, cfgs, chunksize=8)
    all_tr = {r[0]: r[1] for r in results}
    print(f"simulación completa en {time.time()-t0:.0f}s", flush=True)

    t_min = min(df["ts"].iloc[0] for df in UNIVERSE.values())
    t_max = max(df["ts"].iloc[-1] for df in UNIVERSE.values())
    span = (t_max - t_min)
    # folds anclados: entrena [t_min, a) y prueba [a, b)
    cuts = [t_min + span * f for f in (0.50, 0.67, 0.84, 1.0000001)]
    folds = [(cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], cuts[3])]
    print("periodo:", t_min.date(), "->", t_max.date())

    rows, oos_parts = [], []
    for fi, (a, b) in enumerate(folds, 1):
        table = []
        for cfg, tr in all_tr.items():
            trn = tr[tr.entry_ts < a]
            if len(trn) < MIN_TRAIN_TRADES:
                continue
            s = summarize(trn)
            table.append((cfg, s["exp_net"], s["n"]))
        table.sort(key=lambda x: -x[1])
        top = table[:10]
        pos_oos = 0
        for rank, (cfg, exp_tr, n_tr) in enumerate(top):
            tst = all_tr[cfg]
            tst = tst[(tst.entry_ts >= a) & (tst.entry_ts < b)]
            so = summarize(tst)
            pos_oos += so["exp_net"] > 0 if so["n"] else 0
            rows.append(dict(fold=fi, rank=rank + 1, cfg=tst.cfg.iloc[0] if len(tst) else str(cfg),
                             train_exp=exp_tr, train_n=n_tr, test_exp=so["exp_net"], test_n=so["n"],
                             test_win=so["win"], test_pf=so["pf"]))
            if rank == 0:
                oos_parts.append(tst)
        print(f"fold {fi}: entrena<{a.date()} prueba {a.date()}..{b.date()} | top10 con expectativa OOS>0: {pos_oos}/10", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(HERE, "results", "walkforward_top10.csv"), index=False)

    print("\n=== Mejor config de cada fold (elegida SOLO con datos de entrenamiento) ===")
    print(res[res["rank"] == 1][["fold", "cfg", "train_exp", "train_n", "test_exp", "test_n", "test_win", "test_pf"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    oos = pd.concat(oos_parts, ignore_index=True)
    s = summarize(oos)
    print(f"\n=== OOS combinado (3 folds, 1 config por fold) ===")
    print(f"trades={s['n']} win={s['win']:.1%} exp_bruta={s['exp_gross']:.3%} exp_neta={s['exp_net']:.3%} pf={s['pf']:.2f}")
    print(f"costo ida y vuelta (taker+slippage) = {COSTS.round_trip():.3%}")
    curve, st = portfolio_curve(oos, alloc=0.10, max_pos=10)
    print(f"cartera 10%/trade, máx 10 abiertas: retorno={st['ret']:.1%} maxDD={st['maxdd']:.1%} trades tomados={st['taken']}")

    # Mejor estrategia por promedio OOS entre folds (referencia, NO es selección limpia)
    allrows = []
    for cfg, tr in all_tr.items():
        for fi, (a, b) in enumerate(folds, 1):
            t = tr[(tr.entry_ts >= a) & (tr.entry_ts < b)]
            if len(t) >= 20:
                allrows.append((tr.cfg.iloc[0], fi, t.net.mean(), len(t)))
    ar = pd.DataFrame(allrows, columns=["cfg", "fold", "exp", "n"])
    piv = ar.pivot_table(index="cfg", columns="fold", values="exp")
    robust = piv.dropna()
    robust = robust[(robust > 0).all(axis=1)]
    print(f"\nConfigs con expectativa NETA positiva en los 3 periodos de prueba: {len(robust)} de {len(piv.dropna())}")
    robust.assign(mean=robust.mean(axis=1)).sort_values("mean", ascending=False).head(15).to_csv(
        os.path.join(HERE, "results", "robust_configs.csv"))
    print(robust.assign(mean=robust.mean(axis=1)).sort_values("mean", ascending=False).head(10).round(4).to_string())

    # sensibilidad a costos de lo OOS elegido
    print("\n=== Sensibilidad a comisiones (misma OOS, recalculando neto) ===")
    for label, rt in [("sin costos", 0.0), ("maker+maker + slip", 2 * FEE_MAKER + 2 * 0.0002),
                      ("taker+taker + slip (base)", COSTS.round_trip()), ("taker + slippage x3", 0.001 + 6 * 0.0002)]:
        net = oos.gross - rt
        print(f"{label:28s} exp_neta/trade={net.mean():+.3%}  win={(net>0).mean():.1%}")
    oos.to_csv(os.path.join(HERE, "results", "oos_trades.csv"), index=False)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
