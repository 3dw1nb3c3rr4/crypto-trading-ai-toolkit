"""Backtest del Cerebro RL sobre velas de 5m, con comisiones.

Uso:  python backtest_cerebro_rl_5m.py --data ohlcv_5m.pkl [--max-symbols 40] [--oos-from 2026-08-01]
`--data` es un pickle {simbolo: DataFrame[ts(ms), open, high, low, close, volume]} (ver download_5m.py).

Inferencia por lotes (misma red y features que CerebroRL.decidir_entrada, sin posición abierta).
Salida: TP=SL=1.5% y máx. 288 velas (config con que se entrenó el modelo), taker 0.05%/lado + slippage 0.02%/lado.
Solo se reportan los trades con entrada >= --oos-from si se indica (recomendado: fecha posterior al entrenamiento).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bots", "cerebro_rl"))
import cerebro_rl_core as crc  # noqa: E402
from engine import Costs, simulate_symbol, summarize  # noqa: E402


def batch_signals(model, df_ms: pd.DataFrame, min_prob: float | None = None, bs: int = 4096) -> np.ndarray:
    """+1 long / -1 short / 0 esperar para cada vela, usando solo datos hasta su cierre."""
    X = crc.normalizar(crc.build_features(df_ms), model.mu, model.sd)
    P = crc.pos_features(*[np.zeros(len(X), dtype=np.float32)] * 6)
    obs = np.concatenate([X, P], axis=1)
    min_prob = model.min_prob if min_prob is None else min_prob
    sig = np.zeros(len(X))
    with torch.no_grad():
        for s in range(0, len(obs), bs):
            out = model.net(torch.tensor(obs[s:s + bs], device=model.device))
            prob = torch.softmax(out["entry"], -1).cpu().numpy()
            a = prob.argmax(1)
            a = np.where((a != 0) & (prob[np.arange(len(a)), a] < min_prob), 0, a)
            sig[s:s + bs] = np.where(a == 0, 0, np.where(a <= 3, 1, -1))
    sig[:crc.WARMUP] = 0
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "cerebro_rl.pt"))
    ap.add_argument("--max-symbols", type=int, default=40)
    ap.add_argument("--oos-from", default=None, help="solo contar entradas desde esta fecha (UTC)")
    ap.add_argument("--min-prob", type=float, default=None)
    args = ap.parse_args()

    model = crc.CerebroRL.cargar(args.model, device="cpu")
    data = pickle.load(open(args.data, "rb"))
    costs = Costs()
    rows = []
    for sym in list(data)[: args.max_symbols]:
        df = data[sym]
        if len(df) < crc.WARMUP + crc.MAX_HOLD + 50:
            continue
        df_ms = crc.a_dataframe(df)
        sig = batch_signals(model, df_ms, args.min_prob)
        ts = pd.to_datetime(df_ms["ts"], unit="ms", utc=True)
        start = 0 if not args.oos_from else int(ts.searchsorted(pd.Timestamp(args.oos_from, tz="UTC")))
        for (e, x, side, net, gross, reason) in simulate_symbol(
                df_ms, sig, crc.TP_PCT, crc.SL_PCT, crc.MAX_HOLD, costs, start_idx=start):
            rows.append((sym, ts.iloc[e], ts.iloc[x], side, net, gross, reason))
    tr = pd.DataFrame(rows, columns=["symbol", "entry_ts", "exit_ts", "side", "net", "gross", "reason"])
    s = summarize(tr)
    print(f"símbolos={tr.symbol.nunique()} trades={s['n']} win={s['win']:.1%} "
          f"bruta={s['exp_gross']:+.3%} neta={s['exp_net']:+.3%} pf={s['pf']:.2f} (costo ida/vuelta {costs.round_trip():.3%})")
    for side, name in ((1, "LONG"), (-1, "SHORT")):
        t = summarize(tr[tr.side == side])
        print(f"  {name}: n={t['n']} win={t['win']:.1%} neta={t['exp_net']:+.3%} pf={t['pf']:.2f}")
    print(tr.reason.value_counts().to_dict())
    tr.to_csv(os.path.join(HERE, "results", "cerebro_rl_5m_trades.csv"), index=False)


if __name__ == "__main__":
    main()
