"""Long/short cross-sectional neutral al mercado (diario), con costos.

Cada H días: ranking por retorno de los últimos L días; long a un extremo y short al otro
(momentum o reversión), 50% del nocional en cada pata, entrada en open[t+1], salida en open[t+1+H].
Costo: se asume que TODAS las posiciones se cierran y se reabren cada periodo (conservador):
round-trip = 2*fee + 2*slippage sobre el nocional total.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import Costs, load_universe  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data", "trading_history", "ohlcv_cache_2y.pkl")


def matrices(uni):
    idx = sorted(set().union(*[set(df.ts) for df in uni.values()]))
    idx = pd.DatetimeIndex(idx)
    o = pd.DataFrame({s: df.set_index("ts")["open"] for s, df in uni.items()}).reindex(idx)
    c = pd.DataFrame({s: df.set_index("ts")["close"] for s, df in uni.items()}).reindex(idx)
    return o, c


def run(o, c, L, H, q, mode, rt_cost, offset):
    """Devuelve Serie de retornos netos por periodo (cartera neutral, nocional 1)."""
    ret_L = c.pct_change(L)
    out = {}
    n = len(o)
    for t in range(L + offset, n - H - 1, H):
        sig = ret_L.iloc[t].dropna()
        if len(sig) < 20:
            continue
        entry, exit_ = o.iloc[t + 1], o.iloc[t + 1 + H]
        r = (exit_ / entry - 1).reindex(sig.index).dropna()
        sig = sig.reindex(r.index)
        k = max(int(len(sig) * q), 3)
        ranked = sig.sort_values()
        low, high = ranked.index[:k], ranked.index[-k:]
        longs, shorts = (high, low) if mode == "momentum" else (low, high)
        gross = 0.5 * r[longs].mean() - 0.5 * r[shorts].mean()
        out[o.index[t + 1]] = gross - rt_cost
    return pd.Series(out)


def main():
    uni = load_universe(DATA)
    o, c = matrices(uni)
    costs = Costs()
    rt = costs.round_trip()
    t0, t1 = o.index[0], o.index[-1]
    span = t1 - t0
    cuts = [t0 + span * f for f in (0.50, 0.67, 0.84, 1.0000001)]
    periods = [("P1", cuts[0], cuts[1]), ("P2", cuts[1], cuts[2]), ("P3", cuts[2], cuts[3])]
    rows = []
    for L, H, q, mode in itertools.product((3, 7, 14, 30), (3, 7, 14), (0.2,), ("momentum", "reversal")):
        # promedio de offsets para no depender del día de inicio
        series = [run(o, c, L, H, q, mode, rt, off) for off in range(H)]
        net = pd.concat(series).sort_index()
        gross = net + rt
        row = dict(L=L, H=H, mode=mode, n=len(net), gross_per_period=gross.mean(), net_per_period=net.mean())
        ann = 365 / H
        row["net_ann_sharpe"] = net.mean() / net.std() * np.sqrt(ann) if net.std() > 0 else np.nan
        for name, a, b in periods:
            s = net[(net.index >= a) & (net.index < b)]
            row[f"{name}_net"] = s.mean()
        rows.append(row)
    df = pd.DataFrame(rows)
    df["all_pos"] = (df[["P1_net", "P2_net", "P3_net"]] > 0).all(axis=1)
    df.sort_values("net_per_period", ascending=False).to_csv(os.path.join(HERE, "results", "cross_sectional.csv"), index=False)
    pd.set_option("display.width", 200)
    cols = ["L", "H", "mode", "gross_per_period", "net_per_period", "net_ann_sharpe", "P1_net", "P2_net", "P3_net", "all_pos"]
    print(f"costo por periodo (round trip): {rt:.3%}")
    print(df.sort_values("net_per_period", ascending=False)[cols].round(4).to_string(index=False))
    print(f"\nconfigs con retorno neto > 0 en los 3 periodos: {int(df.all_pos.sum())} de {len(df)}")


if __name__ == "__main__":
    main()
