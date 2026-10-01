"""Motor de backtesting con comisiones realistas (Binance Futures USDT-M).

Reglas anti-sesgo:
- La señal se calcula con el cierre de la vela t y se ejecuta en la apertura de t+1.
- Si TP y SL se tocan en la misma vela se asume SL primero (conservador).
- Un solo trade abierto por símbolo a la vez.
- Costos: comisión taker por lado + slippage por lado + funding opcional.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEE_TAKER = 0.0005     # 0.05% por lado (Binance Futures USDT-M taker)
FEE_MAKER = 0.0002     # 0.02% por lado
SLIPPAGE = 0.0002      # 0.02% por lado


@dataclass
class Costs:
    fee_entry: float = FEE_TAKER
    fee_exit: float = FEE_TAKER
    slippage: float = SLIPPAGE
    funding_per_day: float = 0.0   # >0 : longs pagan / shorts cobran

    def round_trip(self) -> float:
        return self.fee_entry + self.fee_exit + 2 * self.slippage


def load_universe(path: str, min_bars: int = 300) -> dict[str, pd.DataFrame]:
    raw = pickle.load(open(path, "rb"))
    out = {}
    for sym, df in raw.items():
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        if len(df) >= min_bars:
            out[sym] = df
    return out


def simulate_symbol(df: pd.DataFrame, signal: np.ndarray, tp: float, sl: float,
                    max_hold: int, costs: Costs, allow_long=True, allow_short=True,
                    start_idx: int = 0, end_idx: int | None = None) -> list[tuple]:
    """Devuelve trades (entry_ts_idx, exit_idx, side, net_ret, gross_ret, reason).

    signal[t] in {+1,-1,0} calculada al cierre de t; entra en open[t+1].
    start_idx/end_idx acotan el rango de FECHAS DE ENTRADA (para train/test).
    """
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    n = len(df)
    end_idx = n - 1 if end_idx is None else min(end_idx, n - 1)
    trades = []
    t = max(start_idx, 0)
    rt = costs.round_trip()
    while t < end_idx:
        s = signal[t]
        if s == 0 or np.isnan(s) or (s > 0 and not allow_long) or (s < 0 and not allow_short):
            t += 1
            continue
        e = t + 1
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            t += 1
            continue
        side = int(s)
        tp_px = entry * (1 + side * tp)
        sl_px = entry * (1 - side * sl)
        exit_px, reason, x = None, "TIME", min(e + max_hold - 1, n - 1)
        for j in range(e, min(e + max_hold, n)):
            hi, lo = h[j], l[j]
            hit_sl = (lo <= sl_px) if side > 0 else (hi >= sl_px)
            hit_tp = (hi >= tp_px) if side > 0 else (lo <= tp_px)
            if hit_sl:                       # SL primero (conservador)
                exit_px, reason, x = sl_px, "SL", j
                break
            if hit_tp:
                exit_px, reason, x = tp_px, "TP", j
                break
        if exit_px is None:
            x = min(e + max_hold - 1, n - 1)
            exit_px = c[x]
        gross = side * (exit_px / entry - 1)
        days = max(x - e + 1, 1)
        net = gross - rt - side * costs.funding_per_day * days
        trades.append((e, x, side, net, gross, reason))
        t = x + 1                            # sin solapamiento por símbolo
    return trades


def trades_frame(universe, signals, tp, sl, max_hold, costs, start_ts=None, end_ts=None,
                 allow_long=True, allow_short=True) -> pd.DataFrame:
    rows = []
    for sym, df in universe.items():
        sig = signals[sym]
        ts = df["ts"]
        s_i = 0 if start_ts is None else int(ts.searchsorted(start_ts))
        e_i = None if end_ts is None else int(ts.searchsorted(end_ts)) - 1
        for (e, x, side, net, gross, reason) in simulate_symbol(
                df, sig, tp, sl, max_hold, costs, allow_long, allow_short, s_i, e_i):
            rows.append((sym, ts.iloc[e], ts.iloc[x], side, net, gross, reason))
    return pd.DataFrame(rows, columns=["symbol", "entry_ts", "exit_ts", "side", "net", "gross", "reason"])


def summarize(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return dict(n=0, win=np.nan, exp_net=np.nan, exp_gross=np.nan, pf=np.nan, total_net=0.0)
    w = tr.net[tr.net > 0].sum(); lo = -tr.net[tr.net < 0].sum()
    return dict(n=len(tr), win=(tr.net > 0).mean(), exp_net=tr.net.mean(),
                exp_gross=tr.gross.mean(), pf=(w / lo if lo > 0 else np.inf),
                total_net=tr.net.sum())


def portfolio_curve(tr: pd.DataFrame, alloc: float = 0.10, max_pos: int = 10,
                    start_equity: float = 1000.0):
    """Cartera simple sin apalancamiento: cada trade usa `alloc` del equity, máx `max_pos` abiertos.

    Los trades se toman por orden de entrada; si ya hay `max_pos` abiertos se omiten.
    El PnL se acredita en la fecha de salida. Devuelve (equity_series, stats).
    """
    if tr.empty:
        return pd.Series([start_equity]), dict(ret=0.0, maxdd=0.0, taken=0)
    pending = []            # (exit_ts, net, alloc_usd)
    curve = []
    eq, taken = start_equity, 0
    for r in tr.sort_values("entry_ts").itertuples():
        for p in sorted(p for p in pending if p[0] <= r.entry_ts):
            eq += p[2] * p[1]
            curve.append((p[0], eq))
        pending = [p for p in pending if p[0] > r.entry_ts]
        if len(pending) >= max_pos:
            continue
        pending.append((r.exit_ts, r.net, eq * alloc))
        taken += 1
    for p in sorted(pending):
        eq += p[2] * p[1]
        curve.append((p[0], eq))
    s = pd.Series({t: v for t, v in curve}).sort_index()
    peak = s.cummax()
    return s, dict(ret=eq / start_equity - 1, maxdd=float(((s - peak) / peak).min()), taken=taken)
