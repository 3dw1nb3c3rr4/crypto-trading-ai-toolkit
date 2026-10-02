"""Carry de funding delta-neutral: long spot + short perpetuo, con costos reales.

P&L por liquidación = funding cobrado por el corto + P&L de la base (spot vs perp) - costos de entrar/salir.
Estrategia: cada R liquidaciones se eligen los N perpetuos con mayor funding medio reciente (ventana L, umbral mínimo),
se mantienen hasta el siguiente rebalanceo. Capital: nocional = capital * lev/(lev+1) (margen del perp con apalancamiento lev).

Uso:  python funding_carry.py --funding funding_data.pkl --source okx_funding --prices ../data/okx_5m.pkl [--spot-key okx_spot_1h]
      python funding_carry.py --selftest      (casos de respuesta conocida con funding sintético; NO es un resultado real)
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
STEP_MS = 8 * 3600_000
SPOT_FEE, PERP_FEE, SLIP = 0.0010, 0.0005, 0.0002      # taker spot OKX, taker perp, slippage por pata
ONE_WAY = SPOT_FEE + PERP_FEE + 2 * SLIP                 # costo por unidad de nocional que entra o sale (ambas patas)
BURN = 63                                                # liquidaciones de calentamiento (ventana máxima)


def load_inputs(funding_path, source, prices_path, spot_key):
    raw = pickle.load(open(funding_path, "rb"))
    if source:
        funding = raw[source]
        spot = raw.get(spot_key) if spot_key else None
    else:
        funding, spot = raw, None
    prices = pickle.load(open(prices_path, "rb"))
    return funding, prices, spot


def build_matrices(funding, prices, spot=None):
    syms = [s for s in funding if s in prices and len(funding[s]) > BURN + 10]
    grid = np.unique(np.concatenate([funding[s]["ts"].to_numpy() for s in syms]))
    first_px = max(prices[s]["ts"].iloc[0] for s in syms) + 300_000
    last_px = min(prices[s]["ts"].iloc[-1] for s in syms) + 300_000
    grid = grid[(grid >= first_px) & (grid <= last_px)]
    grid = grid[(grid % STEP_MS) == 0] if (grid % STEP_MS == 0).mean() > 0.9 else grid
    K, S = len(grid), len(syms)
    R, P, Sp = np.full((K, S), np.nan), np.full((K, S), np.nan), np.full((K, S), np.nan)
    for j, s in enumerate(syms):
        f = funding[s].drop_duplicates("ts").set_index("ts")["rate"]
        R[:, j] = f.reindex(grid).to_numpy()
        pts = prices[s]["ts"].to_numpy()
        idx = np.searchsorted(pts, grid - 300_000)
        ok = (idx < len(pts)) & (pts[np.minimum(idx, len(pts) - 1)] == grid - 300_000)
        P[ok, j] = prices[s]["close"].to_numpy(dtype="float64")[idx[ok]]
        if spot is not None and s in spot:
            sdf = spot[s]; sts = sdf["ts"].to_numpy()
            idx = np.searchsorted(sts, grid - 3600_000)
            ok = (idx < len(sts)) & (sts[np.minimum(idx, len(sts) - 1)] == grid - 3600_000)
            Sp[ok, j] = sdf["close"].to_numpy(dtype="float64")[idx[ok]]
    return syms, grid, R, P, Sp


def simulate(R, P, Sp, highs, N, Lb, Rb, thr, lev, start_k, end_k, one_way=ONE_WAY):
    """Devuelve dict con retornos por liquidación (sobre capital) y ventanas mantenidas para el chequeo de liquidación."""
    K, S = R.shape
    eff = lev / (lev + 1.0)
    w = np.zeros(S)
    entry_k = np.full(S, -1)
    steps = end_k - start_k
    net, fund, basis, cost = (np.zeros(steps) for _ in range(4))
    windows = []
    for i, k in enumerate(range(start_k, end_k)):
        c_k = 0.0
        if i % Rb == 0:
            score = np.nanmean(R[k - Lb + 1:k + 1], axis=0)
            order = np.argsort(-np.where(np.isfinite(score), score, -9))
            sel = [j for j in order[:N] if np.isfinite(score[j]) and score[j] > thr and np.isfinite(P[k, j])]
            w_new = np.zeros(S)
            if sel:
                w_new[sel] = 1.0 / len(sel)
            c_k = np.abs(w_new - w).sum() * one_way * eff
            for j in range(S):
                if w[j] > 0 and w_new[j] == 0:
                    windows.append((j, entry_k[j], k))
                if w_new[j] > 0 and w[j] == 0:
                    entry_k[j] = k
            w = w_new
        if k + 1 < K:
            fr = np.nan_to_num(R[k + 1])
            perp_ret = P[k + 1] / P[k] - 1.0
            spot_ret = np.where(np.isfinite(Sp[k + 1] / Sp[k]), Sp[k + 1] / Sp[k] - 1.0, perp_ret)
            b = np.nan_to_num(spot_ret - perp_ret)
            fund[i] = eff * float((w * fr).sum())
            basis[i] = eff * float((w * b).sum())
        cost[i] = c_k
        net[i] = fund[i] + basis[i] - cost[i]
    for j in range(S):
        if w[j] > 0:
            windows.append((j, entry_k[j], end_k - 1))
    return dict(net=net, fund=fund, basis=basis, cost=cost, windows=windows)


def liquidation_events(windows, grid, P, highs, lev, mmr=0.01):
    """Ventanas en que el precio máximo (5m) superó el umbral de liquidación del corto: sube > 1/lev - mmr."""
    n = 0
    for j, ka, kb in windows:
        if ka < 0 or kb <= ka or not np.isfinite(P[ka, j]):
            continue
        hts, hv = highs[j]
        lo, hi = np.searchsorted(hts, grid[ka]), np.searchsorted(hts, grid[kb], side="right")
        if hi > lo and hv[lo:hi].max() / P[ka, j] - 1.0 > 1.0 / lev - mmr:
            n += 1
    return n


def stats(net, steps_per_day=3):
    d = len(net) // steps_per_day
    daily = net[: d * steps_per_day].reshape(d, steps_per_day).sum(axis=1)
    eq = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(eq)
    total = eq[-1] - 1
    return dict(days=d, total=total, apr_simple=daily.mean() * 365,
                sharpe=(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else float("nan"),
                maxdd=float(((eq - peak) / peak).min()), worst_day=float(daily.min()), pos_days=float((daily > 0).mean()))


def evaluate(R, P, Sp, highs, grid, cfg, lev, start_k, end_k):
    N, Lb, Rb, thr = cfg
    r = simulate(R, P, Sp, highs, N, Lb, Rb, thr, lev, start_k, end_k)
    s = stats(r["net"])
    s.update(fund=r["fund"].sum(), basis=r["basis"].sum(), cost=r["cost"].sum(),
             liq=liquidation_events(r["windows"], grid, P, highs, lev), net_steps=r["net"])
    return s


def report(funding_path, source, prices_path, spot_key, lev):
    funding, prices, spot = load_inputs(funding_path, source, prices_path, spot_key)
    syms, grid, R, P, Sp = build_matrices(funding, prices, spot)
    highs = [(prices[s]["ts"].to_numpy() + 300_000, prices[s]["high"].to_numpy(dtype="float64")) for s in syms]
    K = len(grid)
    days = (grid[-1] - grid[0]) / 86_400_000
    print(f"símbolos={len(syms)} | liquidaciones={K} | rango {pd.to_datetime(grid[0], unit='ms', utc=True):%Y-%m-%d} -> "
          f"{pd.to_datetime(grid[-1], unit='ms', utc=True):%Y-%m-%d} ({days:.0f} días) | base spot: "
          f"{'modelada' if np.isfinite(Sp).any() else 'NO disponible (se asume 0: optimista)'} | apalancamiento perp {lev}x")
    print(f"funding medio por liquidación (todos): {np.nanmean(R):.5%} = {np.nanmean(R)*3*365:.1%} anual simple | "
          f"costo entrar+salir: {2*ONE_WAY:.2%} del nocional")
    if K < BURN + 60:
        print("Datos insuficientes para evaluar."); return
    grid_cfg = list(itertools.product((3, 5, 10), (9, 21, 63), (21, 90), (0.0, 0.00005)))
    rows = []
    for cfg in grid_cfg:
        s = evaluate(R, P, Sp, highs, grid, cfg, lev, BURN, K - 1)
        rows.append(dict(N=cfg[0], L=cfg[1], R=cfg[2], thr=cfg[3], apr=s["apr_simple"], sharpe=s["sharpe"], maxdd=s["maxdd"],
                         worst_day=s["worst_day"], fund=s["fund"], basis=s["basis"], cost=s["cost"], liq=s["liq"]))
    base = evaluate(R, P, Sp, highs, grid, (len(syms), 9, 90, -1.0), lev, BURN, K - 1)
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(os.path.join(HERE, "results", "funding_carry_grid.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\nBASE (mantener los {len(syms)} símbolos sin elegir): APR simple {base['apr_simple']:+.1%} | Sharpe {base['sharpe']:.2f} | "
          f"maxDD {base['maxdd']:.2%} | peor día {base['worst_day']:.2%} | liq.risk {base['liq']}")
    print("\nTop 8 de la grilla (por Sharpe, sobre todo el periodo: puede estar sobreajustado):")
    print(df.head(8).round(4).to_string(index=False))
    # selección en la primera mitad, evaluación en la segunda
    mid = BURN + (K - 1 - BURN) // 2
    if (K - 1 - BURN) >= 2 * 60:
        tr = {c: evaluate(R, P, Sp, highs, grid, c, lev, BURN, mid) for c in grid_cfg}
        best = max(grid_cfg, key=lambda c: tr[c]["sharpe"] if np.isfinite(tr[c]["sharpe"]) else -9)
        te = evaluate(R, P, Sp, highs, grid, best, lev, mid, K - 1)
        te_base = evaluate(R, P, Sp, highs, grid, (len(syms), 9, 90, -1.0), lev, mid, K - 1)
        pos = sum(1 for c in grid_cfg if evaluate(R, P, Sp, highs, grid, c, lev, mid, K - 1)["apr_simple"] > 0)
        print(f"\nWALK-FORWARD: config elegida en la 1a mitad = N{best[0]} L{best[1]} R{best[2]} thr{best[3]}  "
              f"(train APR {tr[best]['apr_simple']:+.1%}, Sharpe {tr[best]['sharpe']:.2f})")
        print(f"  2a mitad (no vista): APR {te['apr_simple']:+.1%} | Sharpe {te['sharpe']:.2f} | maxDD {te['maxdd']:.2%} | "
              f"peor día {te['worst_day']:.2%} | liq.risk {te['liq']}   [BASE misma mitad: APR {te_base['apr_simple']:+.1%}]")
        print(f"  configs con APR>0 en la 2a mitad: {pos} de {len(grid_cfg)}")
    else:
        print("\n(Historia corta: sin walk-forward; los resultados son solo descriptivos.)")


def selftest():
    """Casos de respuesta conocida con funding SINTÉTICO y precios perp reales. No son resultados de mercado."""
    prices = pickle.load(open(os.path.join(os.path.dirname(HERE), "data", "okx_5m.pkl"), "rb"))
    syms = list(prices)[:8]
    t0 = (max(prices[s]["ts"].iloc[0] for s in syms) // STEP_MS + 2) * STEP_MS
    t1 = min(prices[s]["ts"].iloc[-1] for s in syms) - 3600_000
    ts = np.arange(t0, t1, STEP_MS)
    out = {}
    for name, rate in (("A) funding constante 0.01%/8h, base 0", 0.0001), ("B) funding 0, base 0", 0.0)):
        fund = {s: pd.DataFrame({"ts": ts, "rate": np.full(len(ts), rate)}) for s in syms}
        _, grid, R, P, Sp = build_matrices(fund, {s: prices[s] for s in syms}, None)
        highs = [(prices[s]["ts"].to_numpy() + 300_000, prices[s]["high"].to_numpy(dtype="float64")) for s in syms]
        lev = 3.0
        r = evaluate(R, P, Sp, highs, grid, (8, 9, 10_000, -1.0), lev, BURN, len(grid) - 1)
        d = r["days"]
        exp_f = rate * 3 * lev / (lev + 1) * d
        exp_c = ONE_WAY * lev / (lev + 1)
        print(f"{name}: días={d} | funding acumulado={r['fund']:.4%} (esperado {exp_f:.4%}) | costo={r['cost']:.4%} "
              f"(esperado {exp_c:.4%} una sola entrada) | neto total={r['total']:+.4%} (esperado {exp_f - exp_c:+.4%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--funding"); ap.add_argument("--source", default=None); ap.add_argument("--spot-key", default=None)
    ap.add_argument("--prices", default=os.path.join(os.path.dirname(HERE), "data", "okx_5m.pkl"))
    ap.add_argument("--lev", type=float, default=3.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    selftest() if a.selftest else report(a.funding, a.source, a.prices, a.spot_key, a.lev)
