"""Arbitraje de funding (idea de Hummingbot v2_funding_rate_arb.py) evaluado con datos diarios y costos.

1) cross: dos perpetuos del mismo activo en exchanges distintos. Largo en el de menor funding, corto en el de mayor:
   gana (funding_alto - funding_bajo) por día. Entra si |media móvil de la diferencia diaria| >= umbral (en fracción por día,
   como la normalización diaria de Hummingbot); sale si cae a la mitad del umbral o cambia de signo. 4 patas: costo ida y vuelta
   2 * 2 * (comisión taker + slippage).
2) carry: corto en el perpetuo + largo en spot (el spot no está en los datos, así que la base spot-perp se ASUME CERO: es una
   COTA SUPERIOR del resultado real). Entra si la media móvil del funding diario >= umbral.
Retornos sobre el nocional de cada posición; el equity combina las posiciones activas a partes iguales.

Uso:  python funding_arb.py --a derivs_binance.pkl --b derivs_bybit.pkl       (cross, y carry con --a)
      python funding_arb.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import derivs_features as dfe  # noqa: E402

FEE, SLIP = 0.0005, 0.0002
CROSS_RT = 4 * (FEE + SLIP)                  # 4 patas: abrir y cerrar dos perpetuos
CARRY_RT = 2 * (0.0010 + FEE + 2 * SLIP)     # spot taker 0.10% + perp taker + slippage en ambas patas, entrada y salida
ENTRY_DAY_FRAC = 2 / 3                       # el día de entrada solo se cobran 2 de las 3 liquidaciones


def daily_panel(funding, cols, start, end):
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    return dfe.funding_daily(funding, idx, cols), idx


def simulate(signal_series: pd.Series, earn: pd.Series, theta: float, L: int, rt_cost: float, sign_free: bool):
    """signal_series: media móvil (fracción/día) visible al cierre de t; earn: lo ganado en el día t+1 con la posición abierta
    (con el signo de la posición). Devuelve serie diaria de retorno sobre nocional y número de operaciones."""
    n = len(signal_series)
    sig, earn_v = signal_series.to_numpy(), earn.to_numpy()
    pos = 0                                                   # +1 / -1 / 0
    ret = np.zeros(n)
    trades = 0
    for t in range(n - 1):
        s = sig[t]
        if not np.isfinite(s):
            continue
        if pos == 0:
            if abs(s) >= theta and (sign_free or s > 0):
                pos = 1 if s > 0 else -1
                ret[t + 1] -= rt_cost / 2
                trades += 1
                e = earn_v[t + 1]
                ret[t + 1] += pos * e * ENTRY_DAY_FRAC if np.isfinite(e) else 0.0
                continue
        else:
            if abs(s) < theta / 2 or np.sign(s) != pos:
                ret[t + 1] -= rt_cost / 2
                pos = 0
                continue
            e = earn_v[t + 1]
            ret[t + 1] += pos * e if np.isfinite(e) else 0.0
    return pd.Series(ret, index=signal_series.index), trades


def stats(daily: np.ndarray, label=""):
    d = np.asarray(daily, dtype=float)
    eq = np.cumprod(1 + d)
    r = np.random.default_rng(0)
    blk = [d[i:i + 7] for i in range(0, len(d) - 7, 7)]
    m = [np.concatenate([blk[i] for i in r.integers(0, len(blk), len(blk))]).mean() for _ in range(2000)] if len(blk) > 5 else [np.nan]
    return dict(mean_day=d.mean(), apr=d.mean() * 365, sharpe=d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan,
                maxdd=float((eq / np.maximum.accumulate(eq) - 1).min()), ci=(np.nanpercentile(m, 2.5), np.nanpercentile(m, 97.5)))


def portfolio(per_symbol: dict, max_pos=None):
    """Combina las series por símbolo: promedio entre las posiciones activas (retorno sobre nocional)."""
    df = pd.DataFrame(per_symbol)
    active = (df != 0).sum(axis=1).replace(0, np.nan)
    return (df.sum(axis=1) / active).fillna(0.0)


def run_cross(fa, fb, cols, grid=None, log=print):
    start = min(min(f["ts"].min() for f in fa.values()), min(f["ts"].min() for f in fb.values()))
    end = max(max(f["ts"].max() for f in fa.values()), max(f["ts"].max() for f in fb.values()))
    start, end = pd.to_datetime(start, unit="ms", utc=True).floor("D"), pd.to_datetime(end, unit="ms", utc=True).floor("D")
    A, idx = daily_panel(fa, cols, start, end)
    B, _ = daily_panel(fb, cols, start, end)
    common = [s for s in cols if A[s].notna().sum() > 120 and B[s].notna().sum() > 120]
    spread = (A - B)[common]
    log(f"cross: {len(common)} símbolos con ambos exchanges | {idx[0].date()} -> {idx[-1].date()} | diferencia media {spread.stack().mean():+.4%}/día "
        f"| |diferencia| media {spread.abs().stack().mean():.4%}/día | costo ida y vuelta {CROSS_RT:.2%}")
    return sweep(spread, spread, CROSS_RT, True, grid, log, "cross")


def run_carry(fa, cols, grid=None, log=print):
    start = pd.to_datetime(min(f["ts"].min() for f in fa.values()), unit="ms", utc=True).floor("D")
    end = pd.to_datetime(max(f["ts"].max() for f in fa.values()), unit="ms", utc=True).floor("D")
    A, idx = daily_panel(fa, cols, start, end)
    common = [s for s in cols if A[s].notna().sum() > 120]
    log(f"carry (SIN base spot, cota superior): {len(common)} símbolos | funding medio {A[common].stack().mean():+.4%}/día | costo ida y vuelta {CARRY_RT:.2%}")
    return sweep(A[common], A[common], CARRY_RT, False, grid, log, "carry")


def sweep(basis: pd.DataFrame, earn: pd.DataFrame, rt, sign_free, grid, log, name):
    grid = grid or list(itertools.product((3, 7), (0.0002, 0.0004, 0.0008)))
    res = {}
    for L, theta in grid:
        ps, ntr = {}, 0
        for s in basis.columns:
            m = basis[s].rolling(L, min_periods=max(2, L - 1)).mean()
            ret, k = simulate(m, earn[s].shift(-1), theta, L, rt, sign_free)
            ps[s] = ret; ntr += k
        port = portfolio(ps)
        res[(L, theta)] = (port, ntr)
    half = len(next(iter(res.values()))[0]) // 2
    log(f"\n{name}: grilla (L días de media móvil, umbral por día) -> retorno sobre nocional")
    log(f"{'L':>3s} {'umbral/día':>11s} {'operaciones':>11s} {'APR':>8s} {'Sharpe':>7s} {'maxDD':>7s} {'IC95% diario':>22s}  1a mitad -> 2a mitad (APR)")
    rows = []
    for (L, th), (port, ntr) in res.items():
        s_all, s1, s2 = stats(port.to_numpy()), stats(port.to_numpy()[:half]), stats(port.to_numpy()[half:])
        rows.append((L, th, s1["sharpe"], s2["apr"]))
        log(f"{L:>3d} {th:>11.4%} {ntr:>11d} {s_all['apr']:>8.1%} {s_all['sharpe']:>7.2f} {s_all['maxdd']:>7.1%} "
            f"[{s_all['ci'][0]:+.4%},{s_all['ci'][1]:+.4%}]  {s1['apr']:+.1%} -> {s2['apr']:+.1%}")
    best = max(rows, key=lambda r: r[2] if np.isfinite(r[2]) else -9)
    log(f"WALK-FORWARD: se elige en la 1a mitad (L={best[0]}, umbral {best[1]:.4%}/día) y rinde en la 2a mitad (no vista): APR {best[3]:+.1%}")
    return res


def selftest():
    """Casos de respuesta conocida con funding SINTÉTICO (no son resultados de mercado)."""
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    ts = np.arange(int(idx[0].value // 10**6), int(idx[0].value // 10**6) + n * 86_400_000, 8 * 3600_000)
    def mk(rate): return {"X/USDT:USDT": pd.DataFrame({"ts": ts, "rate": np.full(len(ts), rate)})}
    cols = ["X/USDT:USDT"]
    # A) diferencia constante 0.04%/día repartida en 3 liquidaciones: entra una vez y se queda
    fa, fb = mk(0.0004 / 3 + 0.0001), mk(0.0001)
    A, _ = daily_panel(fa, cols, idx[0], idx[-1]); B, _ = daily_panel(fb, cols, idx[0], idx[-1])
    spread = (A - B)
    port, ntr = simulate(spread["X/USDT:USDT"].rolling(3, min_periods=2).mean(), spread["X/USDT:USDT"].shift(-1), 0.0002, 3, CROSS_RT, True)
    exp = 0.0004 * (n - 5) - CROSS_RT / 2 - 0.0004 / 3 * 0    # ~ todos los días menos el calentamiento, menos el costo de entrada
    print(f"A) diferencia constante 0.04%/día: operaciones={ntr} (esperado 1) | retorno acumulado={port.sum():.4%} (esperado ~{exp:.4%}) | "
          f"costo de entrada cobrado={CROSS_RT/2:.4%}")
    # B) sin diferencia: no debe operar ni perder
    port2, ntr2 = simulate((A - A)["X/USDT:USDT"].rolling(3, min_periods=2).mean(), (A - A)["X/USDT:USDT"].shift(-1), 0.0002, 3, CROSS_RT, True)
    print(f"B) sin diferencia: operaciones={ntr2} (esperado 0) | retorno={port2.sum():.4%} (esperado 0)")
    # C) diferencia que se invierte cada 5 días: el costo debe superar la ganancia (no es rentable)
    sgn = np.where((np.arange(n) // 5) % 2 == 0, 1, -1)
    sp = pd.Series(0.0004 * sgn, index=idx)
    port3, ntr3 = simulate(sp.rolling(3, min_periods=2).mean(), sp.shift(-1), 0.0002, 3, CROSS_RT, True)
    print(f"C) diferencia que cambia de signo cada 5 días: operaciones={ntr3} | retorno acumulado={port3.sum():+.4%} (con costo {CROSS_RT:.2%} por ciclo)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    fa = pickle.load(open(a.a, "rb"))["funding"]
    cols = sorted(fa)
    if a.b:
        fb = pickle.load(open(a.b, "rb"))["funding"]
        run_cross(fa, fb, sorted(set(fa) & set(fb)))
    run_carry(fa, cols)
