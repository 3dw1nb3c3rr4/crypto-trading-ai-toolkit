"""Seguimiento de tendencia (Donchian) con salida trailing y stop por ATR. Sin take profit: deja correr las ganancias.

Señal al cierre de t: long si cierra por encima del máximo de los N_entrada días previos, short si por debajo del mínimo.
Entrada en open[t+1]. Salida: (a) stop inicial/ATR intravela a entrada -/+ k*ATR, (b) cierre por debajo del mínimo (long) o por
encima del máximo (short) de los N_salida días previos, ejecutado en la apertura siguiente. Costos: taker + slippage por lado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import Costs


def _atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().to_numpy()


def trend_trades(df: pd.DataFrame, n_entry: int, n_exit: int, atr_mult: float, costs: Costs,
                 allow_long=True, allow_short=True) -> list[tuple]:
    """Devuelve (idx_entrada, idx_salida, lado, neto, bruto, motivo)."""
    o, h, l, c = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    n = len(df)
    hh = df["high"].shift(1).rolling(n_entry).max().to_numpy()
    ll = df["low"].shift(1).rolling(n_entry).min().to_numpy()
    xh = df["high"].shift(1).rolling(n_exit).max().to_numpy()
    xl = df["low"].shift(1).rolling(n_exit).min().to_numpy()
    atr = _atr(df)
    rt = costs.round_trip()
    out, t = [], max(n_entry, n_exit, 15)
    while t < n - 2:
        side = 0
        if allow_long and c[t] > hh[t]:
            side = 1
        elif allow_short and c[t] < ll[t]:
            side = -1
        if side == 0 or not np.isfinite(atr[t]):
            t += 1
            continue
        e = t + 1
        entry = o[e]
        stop = entry - side * atr_mult * atr[t]
        exit_px, motivo, x = c[n - 1], "FIN", n - 1
        for j in range(e, n - 1):
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                gap = o[j] if ((side > 0 and o[j] < stop) or (side < 0 and o[j] > stop)) else stop
                exit_px, motivo, x = gap * (1 - side * costs.stop_slippage), "STOP", j
                break
            if (side > 0 and c[j] < xl[j]) or (side < 0 and c[j] > xh[j]):
                exit_px, motivo, x = o[j + 1], "TRAILING", j + 1
                break
        gross = side * (exit_px / entry - 1)
        out.append((e, x, side, gross - rt, gross, motivo))
        t = x + 1
    return out
