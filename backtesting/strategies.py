"""Estrategias candidatas (señales causales: usan solo datos hasta el cierre de t).

Cada función devuelve un array +1 (long) / -1 (short) / 0 para cada vela del DataFrame.
Usa los indicadores del propio toolkit (indicators.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import rsi, sma, bollinger_bands  # noqa: E402


def _arr(x) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)


def rsi_reversion(df, lo=25, hi=75, period=14):
    r = rsi(df["close"], period)
    return _arr(np.where(r < lo, 1, np.where(r > hi, -1, 0)))


def bollinger_reversion(df, k=2.0, period=20):
    mid, up, low = bollinger_bands(df["close"], period, k)
    c = df["close"]
    return _arr(np.where(c < low, 1, np.where(c > up, -1, 0)))


def donchian_breakout(df, n=20):
    hh = df["high"].shift(1).rolling(n).max()
    ll = df["low"].shift(1).rolling(n).min()
    c = df["close"]
    return _arr(np.where(c > hh, 1, np.where(c < ll, -1, 0)))


def trend_pullback(df, fast=20, slow=50, rsi_lo=40):
    c = df["close"]
    r = rsi(c, 14)
    up = sma(c, fast) > sma(c, slow)
    dn = sma(c, fast) < sma(c, slow)
    return _arr(np.where(up & (r < rsi_lo), 1, np.where(dn & (r > 100 - rsi_lo), -1, 0)))


def exhaustion_reversal(df, lookback=30, move=0.35, rsi_ext=72):
    """Estilo Techo/Suelo: tras una subida/caída sostenida con RSI extremo, apuesta a la reversión."""
    c = df["close"]
    ret = c.pct_change(lookback)
    r = rsi(c, 14)
    return _arr(np.where((ret > move) & (r > rsi_ext), -1, np.where((ret < -move) & (r < 100 - rsi_ext), 1, 0)))


def momentum(df, n=30, thr=0.10):
    ret = df["close"].pct_change(n)
    return _arr(np.where(ret > thr, 1, np.where(ret < -thr, -1, 0)))


REGISTRY = {
    "rsi_reversion":       (rsi_reversion,       [dict(lo=l, hi=h) for l, h in [(20, 80), (25, 75), (30, 70)]]),
    "bollinger_reversion": (bollinger_reversion, [dict(k=k) for k in (1.5, 2.0, 2.5)]),
    "donchian_breakout":   (donchian_breakout,   [dict(n=n) for n in (10, 20, 55)]),
    "trend_pullback":      (trend_pullback,      [dict(fast=f, slow=s, rsi_lo=r) for f, s, r in [(20, 50, 40), (10, 30, 35), (20, 50, 30)]]),
    "exhaustion_reversal": (exhaustion_reversal, [dict(lookback=lb, move=m, rsi_ext=e) for lb, m, e in [(30, 0.30, 70), (30, 0.50, 72), (14, 0.25, 70)]]),
    "momentum":            (momentum,            [dict(n=n, thr=t) for n, t in [(30, 0.10), (30, 0.20), (14, 0.10)]]),
}

EXIT_GRID = dict(tp=(0.03, 0.05, 0.08, 0.12), sl=(0.02, 0.04, 0.06), hold=(3, 7, 14))
