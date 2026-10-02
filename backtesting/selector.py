"""Selector de estrategias: solo se habilita una estrategia si pasa una prueba estricta, fuera de muestra y con costos.

Cada candidata se evalúa con walk-forward (la configuración se elige solo con datos previos al periodo de prueba),
costos reales (taker + slippage + deslizamiento extra en stops) y un recorte de seguridad por trade
(las pruebas con martingalas sintéticas mostraron ~+0.15% de optimismo residual en salidas por stop).

Puerta (todas obligatorias):
  - al menos `min_trades` operaciones fuera de muestra
  - IC95% (bootstrap por día) del neto medio, después del recorte, con límite inferior > 0
  - neto medio > 0 en TODOS los periodos de prueba con suficientes operaciones (mínimo 2 periodos)
  - factor de beneficio >= 1.10
  - caída máxima de la cartera simulada no peor que -30% (si aplica)
Resultado: strategy_selection.json (lo lee el bot). Si ninguna pasa, el bot no opera.

Uso: python selector.py [--data ../data/okx_5m.pkl] [--funding funding_data.pkl --funding-key okx_funding --spot-key okx_spot_1h]
                        [--skip familias,tendencia,techosuelo,cerebro,ml,xsdiario,carry]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from engine import Costs, portfolio_curve  # noqa: E402

HAIRCUT = 0.0015
GATE = dict(min_trades=300, min_pf=1.10, min_fold_trades=30, min_folds=2, max_dd=-0.30)
FOLD_FRACS = (0.50, 0.67, 0.84, 1.0000001)       # mismos periodos de prueba que run_optimization.py
MIN_TRAIN_TRADES = 150


# ------------------------------------------------------------------ evidencia y puerta
def _day_ci(entry_ts: pd.Series, vals: np.ndarray, n=3000, seed=0):
    days = pd.DatetimeIndex(pd.to_datetime(entry_ts, utc=True)).date
    arrs = [v.to_numpy() for _, v in pd.Series(vals).groupby(days)]
    r = np.random.default_rng(seed)
    m = [np.concatenate([arrs[i] for i in r.integers(0, len(arrs), len(arrs))]).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def make_evidence(name, tr: pd.DataFrame, kind="trades", config=None, haircut=HAIRCUT, notes=""):
    """tr: columnas entry_ts, net, [exit_ts], [fold]. Aplica el recorte de seguridad al neto de cada trade."""
    if tr is None or tr.empty:
        return dict(name=name, kind=kind, n=0, exp_net=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), pf=float("nan"),
                    win=float("nan"), folds=[], port_ret=None, port_dd=None, config=config, notes=notes or "sin operaciones")
    adj = tr["net"].to_numpy() - haircut
    lo, hi = _day_ci(tr["entry_ts"], adj)
    w, l = adj[adj > 0].sum(), -adj[adj < 0].sum()
    folds = []
    if "fold" in tr:
        for f, g in tr.assign(adj=adj).groupby("fold"):
            folds.append(dict(fold=int(f), n=int(len(g)), exp_net=float(g["adj"].mean())))
    port_ret = port_dd = None
    if "exit_ts" in tr:
        t2 = tr.assign(net=adj)[["entry_ts", "exit_ts", "net"]]
        _, st = portfolio_curve(t2, alloc=0.05, max_pos=10)
        port_ret, port_dd = float(st["ret"]), float(st["maxdd"])
    return dict(name=name, kind=kind, n=int(len(tr)), exp_net=float(adj.mean()), exp_net_raw=float(tr["net"].mean()),
                ci_lo=lo, ci_hi=hi, pf=float(w / l) if l > 0 else float("inf"), win=float((adj > 0).mean()),
                folds=folds, port_ret=port_ret, port_dd=port_dd, config=config, notes=notes)


def gate(ev: dict, g=GATE):
    r = []
    if ev["n"] < g["min_trades"]:
        r.append(f"pocas operaciones ({ev['n']} < {g['min_trades']})")
    if not (ev["ci_lo"] > 0):
        r.append(f"IC95% del neto incluye cero o es negativo [{ev['ci_lo']:+.3%}, {ev['ci_hi']:+.3%}]")
    ok = [f for f in ev["folds"] if f["n"] >= g["min_fold_trades"]]
    if len(ok) < g["min_folds"]:
        r.append(f"menos de {g['min_folds']} periodos con datos suficientes")
    bad = [f["fold"] for f in ok if f["exp_net"] <= 0]
    if bad:
        r.append(f"neto <= 0 en el/los periodo(s) {bad}")
    if not (ev["pf"] >= g["min_pf"]):
        r.append(f"factor de beneficio {ev['pf']:.2f} < {g['min_pf']}")
    if ev["port_dd"] is not None and ev["port_dd"] < g["max_dd"]:
        r.append(f"caída máxima {ev['port_dd']:.0%} peor que {g['max_dd']:.0%}")
    return (len(r) == 0), r


# ------------------------------------------------------------------ utilidades de walk-forward por familia
def fold_cuts(universe):
    t_min = min(df["ts"].iloc[0] for df in universe.values())
    t_max = max(df["ts"].iloc[-1] for df in universe.values())
    span = t_max - t_min
    cuts = [t_min + span * f for f in FOLD_FRACS]
    return [(cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], cuts[3])]


def walk_forward_pick(all_tr: dict, folds, label: str):
    """all_tr: {cfg: DataFrame(entry_ts, net, ...)}. Elige por fold la mejor cfg con SOLO datos previos y devuelve trades OOS."""
    parts, last_cfg = [], None
    for fi, (a, b) in enumerate(folds, 1):
        best, best_e = None, -9
        for cfg, tr in all_tr.items():
            trn = tr[tr.entry_ts < a]
            if len(trn) >= MIN_TRAIN_TRADES and trn.net.mean() > best_e:
                best, best_e = cfg, trn.net.mean()
        if best is None:
            continue
        t = all_tr[best]
        t = t[(t.entry_ts >= a) & (t.entry_ts < b)].assign(fold=fi)
        parts.append(t)
        last_cfg = best
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), last_cfg


# ------------------------------------------------------------------ evaluadores
_RO = None


def _ro():
    global _RO
    if _RO is None:
        import run_optimization as ro
        _RO = ro
    return _RO


def eval_familias():
    ro = _ro()
    ro.build_signals()
    cfgs = [(k, tp, sl, h) for k in ro.SIGNALS for tp, sl, h in itertools.product(*[ro.EXIT_GRID[x] for x in ("tp", "sl", "hold")])]
    with mp.Pool(min(mp.cpu_count(), 8), initializer=ro.build_signals) as pool:
        res = pool.map(ro.run_cfg, cfgs, chunksize=8)
    by_family: dict[str, dict] = {}
    for cfg, tr in res:
        by_family.setdefault(cfg[0][0], {})[tr.cfg.iloc[0] if len(tr) else str(cfg)] = tr
    folds = fold_cuts(ro.UNIVERSE)
    out = []
    for fam, trs in by_family.items():
        oos, cfg = walk_forward_pick(trs, folds, fam)
        out.append(make_evidence(f"indicadores:{fam}", oos, config=dict(familia=fam, ultima_config=cfg),
                                 notes="walk-forward: config elegida por periodo con datos previos"))
    return out


def _trend_cfg(cfg):
    from trend_following import trend_trades
    ro = _ro()
    ne, nx, k = cfg
    rows = []
    for sym, df in ro.UNIVERSE.items():
        ts = df["ts"]
        for (e, x, side, net, gross, mot) in trend_trades(df, ne, nx, k, Costs()):
            rows.append((sym, ts.iloc[e], ts.iloc[x], side, net, gross, mot))
    return cfg, pd.DataFrame(rows, columns=["symbol", "entry_ts", "exit_ts", "side", "net", "gross", "reason"])


def eval_tendencia():
    ro = _ro()
    cfgs = [(ne, nx, k) for ne in (20, 55, 100) for nx in (10, 20, 50) for k in (2.0, 3.0, 4.0) if nx < ne]
    with mp.Pool(min(mp.cpu_count(), 8)) as pool:
        res = pool.map(_trend_cfg, cfgs)
    all_tr = {f"N{c[0]}|X{c[1]}|ATR{c[2]}": tr for c, tr in res}
    oos, cfg = walk_forward_pick(all_tr, fold_cuts(ro.UNIVERSE), "tendencia")
    return [make_evidence("tendencia:donchian_trailing", oos, config=dict(ultima_config=cfg),
                          notes="Donchian + stop ATR + salida trailing, sin take profit; walk-forward")]


def eval_techosuelo():
    from eval_techo_suelo import native_trades
    tr = native_trades()
    tr = tr.sort_values("entry_ts").reset_index(drop=True)
    tr["fold"] = pd.qcut(tr["entry_ts"].rank(method="first"), 3, labels=False) + 1
    return [make_evidence("modelo:techo_suelo", tr, notes="config nativa TP4/SL4/7d en el tramo de validación")]


def _panel(path):
    from ml_features import load_panel
    return load_panel(path)


def eval_cerebro(path):
    import torch  # noqa: F401
    import ml_walkforward as mw
    sys.path.insert(0, os.path.join(ROOT, "bots", "cerebro_rl"))
    import cerebro_rl_core as crc
    from backtest_cerebro_rl_5m import batch_signals
    from engine import simulate_symbol
    data, syms, ts = _panel(path)
    model = crc.CerebroRL.cargar(os.path.join(ROOT, "models", "cerebro_rl.pt"), device="cpu")
    n = len(ts)
    rows = []
    for s in syms:
        df_ms = crc.a_dataframe(data[s])
        sig = batch_signals(model, df_ms)
        for fi, d0 in enumerate(mw.TEST_STARTS_DAYS, 1):
            t0, t1 = d0 * mw.BAR_DAY, min((d0 + mw.TEST_LEN_DAYS) * mw.BAR_DAY, n - 2)
            for (e, x, side, net, gross, mot) in simulate_symbol(df_ms, sig, crc.TP_PCT, crc.SL_PCT, crc.MAX_HOLD, Costs(),
                                                                 start_idx=t0, end_idx=t1):
                rows.append((s, ts[e], ts[x], side, net, gross, fi))
    tr = pd.DataFrame(rows, columns=["symbol", "entry_ms", "exit_ms", "side", "net", "gross", "fold"])
    tr["entry_ts"] = pd.to_datetime(tr.entry_ms, unit="ms", utc=True)
    tr["exit_ts"] = pd.to_datetime(tr.exit_ms, unit="ms", utc=True)
    return [make_evidence("modelo:cerebro_rl", tr, notes="velas 5m OKX, TP=SL=1.5%, 4 periodos de prueba")]


def eval_ml(path):
    import ml_walkforward as mw
    from ml_features import build_dataset
    data, syms, ts = _panel(path)
    X, y = build_dataset(data, syms, ts, horizons=(12, 48))
    rng = np.random.default_rng(1)
    out = []
    for v in ("dir12", "dir48", "xs12", "xs48"):
        H = int(v[-2:])
        if v.startswith("dir"):
            tr, _ = mw.run_dir(data, syms, ts, X, y, H, False, rng, 0)
            tr["entry_ts"] = pd.to_datetime(tr.entry_ms, unit="ms", utc=True)
            tr["exit_ts"] = pd.to_datetime(tr.exit_ms, unit="ms", utc=True)
        else:
            tr, _ = mw.run_xs(data, syms, ts, X, y, H, False, rng, 0)
            tr["entry_ts"] = pd.to_datetime(tr.entry_ms, unit="ms", utc=True)
        out.append(make_evidence(f"modelo:gradient_boosting_{v}", tr, notes="walk-forward con purga, 5m OKX"))
    return out


def eval_daily_xs(data=None):
    """Modelo cross-sectional diario (117 símbolos, H=7d, gradient boosting) + batería de robustez FIJA:
    universo estable (sin listados nuevos), 50% más líquido y costo doble. Para aprobarse debe mantener IC95% > 0 en todas."""
    import ml_daily_xs as dx
    kw = dict(H=7, model="gbm", data=data or dx.DATA)
    base_rt = Costs().round_trip()

    def ev_from(df, name, notes):
        tr = df.rename(columns={"entry_ts": "entry_ts"})[["entry_ts", "fold", "net"]].copy()
        tr["entry_ts"] = pd.to_datetime(tr["entry_ts"], utc=True)
        return make_evidence(name, tr, notes=notes)

    base = ev_from(dx.evaluate_variant(**kw), "modelo:xs_diario_gbm_7d", "117 perpetuos, diario, long/short K=10, carteras superpuestas de 7 días")
    variants = {
        "universo_estable": dx.evaluate_variant(stable=True, **kw),
        "50%_mas_liquido": dx.evaluate_variant(liq=0.5, **kw),
        "costo_doble": dx.evaluate_variant(rt=2 * base_rt, **kw),
    }
    dx.RT = base_rt
    base["robustness"] = []
    for k, df in variants.items():
        e = ev_from(df, k, "")
        base["robustness"].append(dict(name=k, exp_net=e["exp_net"], ci_lo=e["ci_lo"], ci_hi=e["ci_hi"], n=e["n"]))
    return [base]


def eval_carry(funding_path, key, spot_key):
    import funding_carry as fc
    funding, prices, spot = fc.load_inputs(funding_path, key, os.path.join(ROOT, "data", "okx_5m.pkl"), spot_key)
    syms, grid, R, P, Sp = fc.build_matrices(funding, prices, spot)
    K = len(grid)
    if K < fc.BURN + 120:
        return [dict(name="carry:funding_delta_neutral", kind="carry", n=0, exp_net=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                     pf=float("nan"), win=float("nan"), folds=[], port_ret=None, port_dd=None, config=None,
                     notes=f"historial de funding insuficiente ({K} liquidaciones)")]
    highs = [(prices[s]["ts"].to_numpy() + 300_000, prices[s]["high"].to_numpy(dtype="float64")) for s in syms]
    lev = 1.0
    grid_cfg = list(itertools.product((3, 5, 10), (9, 21, 63), (21, 90), (0.0, 0.00005)))
    mid = fc.BURN + (K - 1 - fc.BURN) // 2
    tr = {c: fc.evaluate(R, P, Sp, highs, grid, c, lev, fc.BURN, mid) for c in grid_cfg}
    best = max(grid_cfg, key=lambda c: tr[c]["sharpe"] if np.isfinite(tr[c]["sharpe"]) else -9)
    r = fc.simulate(R, P, Sp, highs, best[0], best[1], best[2], best[3], lev, mid, K - 1)
    steps = r["net"]
    d = len(steps) // 3
    daily = steps[: d * 3].reshape(d, 3).sum(axis=1)
    rng = np.random.default_rng(0)
    boots = [daily[rng.integers(0, d, d)].mean() for _ in range(3000)]
    half = d // 2
    folds = [dict(fold=1, n=half, exp_net=float(daily[:half].mean())), dict(fold=2, n=d - half, exp_net=float(daily[half:].mean()))]
    eq = np.cumprod(1 + daily)
    wl = daily[daily > 0].sum(), -daily[daily < 0].sum()
    liq = fc.liquidation_events(r["windows"], grid, P, highs, lev)
    ev = dict(name="carry:funding_delta_neutral", kind="carry", n=int(d), exp_net=float(daily.mean()), ci_lo=float(np.percentile(boots, 2.5)),
              ci_hi=float(np.percentile(boots, 97.5)), pf=float(wl[0] / wl[1]) if wl[1] > 0 else float("inf"), win=float((daily > 0).mean()),
              folds=folds, port_ret=float(eq[-1] - 1), port_dd=float((eq / np.maximum.accumulate(eq) - 1).min()),
              config=dict(N=best[0], L=best[1], R=best[2], thr=best[3], lev=lev), notes=f"config elegida en la 1a mitad, evaluada en la 2a; riesgo de liquidación {liq} ventanas a {lev}x")
    return [ev]


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "okx_5m.pkl"))
    ap.add_argument("--daily-data", default=None, help="pickle diario largo para el modelo cross-sectional (colab/celda_historia_diaria.py)")
    ap.add_argument("--funding", default=None)
    ap.add_argument("--funding-key", default=None)
    ap.add_argument("--spot-key", default=None)
    ap.add_argument("--skip", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "strategy_selection.json"))
    a = ap.parse_args()
    skip = set(x for x in a.skip.split(",") if x)
    jobs = [("familias", eval_familias), ("tendencia", eval_tendencia), ("techosuelo", eval_techosuelo),
            ("cerebro", lambda: eval_cerebro(a.data)), ("ml", lambda: eval_ml(a.data)), ("xsdiario", lambda: eval_daily_xs(a.daily_data))]
    if a.funding:
        jobs.append(("carry", lambda: eval_carry(a.funding, a.funding_key, a.spot_key)))
    results, t0 = [], time.time()
    for name, fn in jobs:
        if name in skip:
            continue
        print(f"[{time.time()-t0:4.0f}s] evaluando {name} ...", flush=True)
        try:
            results += fn()
        except Exception as e:
            results.append(dict(name=f"error:{name}", kind="error", n=0, exp_net=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                                pf=float("nan"), win=float("nan"), folds=[], port_ret=None, port_dd=None, config=None, notes=f"{type(e).__name__}: {e}"))
    rows = []
    for ev in results:
        ok, why = (False, [ev["notes"]]) if ev["kind"] == "error" else gate(ev, dict(GATE, min_trades=90) if ev["kind"] == "carry" else GATE)
        ev["status"] = "aprobada" if ok else "rechazada"
        if ok and ev.get("robustness"):
            fails = [r["name"] for r in ev["robustness"] if not (r["ci_lo"] > 0)]
            if fails:       # pasa la prueba principal pero no la batería de robustez: solo observación / paper
                ev["status"], ok = "en_observacion", False
                why = [f"no mantiene IC95% > 0 en: {', '.join(fails)}"]
        ev["enabled"], ev["reasons"] = bool(ok), why
        rows.append(ev)
    print(f"\n{'estrategia':40s} {'n':>6s} {'neto/trade*':>11s} {'IC95% inferior':>14s} {'pf':>5s}  veredicto")
    for ev in rows:
        print(f"{ev['name']:40s} {ev['n']:6d} {ev['exp_net']:+11.3%} {ev['ci_lo']:+14.3%} {ev['pf']:5.2f}  "
              f"{ev['status'].upper() if ev['status'] != 'rechazada' else 'rechazada'}")
        for r in ev["reasons"][:3]:
            print(f"{'':42s}- {r}")
        for r in ev.get("robustness", []):
            print(f"{'':42s}  robustez {r['name']:18s} neto {r['exp_net']:+.3%}  IC95% [{r['ci_lo']:+.3%}, {r['ci_hi']:+.3%}]")
    print(f"\n* neto por trade DESPUÉS del recorte de seguridad de {HAIRCUT:.2%}; costo base ida/vuelta {Costs().round_trip():.2%} + stop {Costs().stop_slippage:.2%}")
    enabled = [e["name"] for e in rows if e["enabled"]]
    watch = [e["name"] for e in rows if e["status"] == "en_observacion"]
    print("ESTRATEGIAS APROBADAS:", enabled or "NINGUNA -> el bot no debe operar")
    if watch:
        print("EN OBSERVACION (solo paper, no dinero real):", watch)
    payload = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gate=GATE, haircut=HAIRCUT, enabled=enabled, watchlist=watch, strategies=rows)
    json.dump(payload, open(a.out, "w"), indent=1, default=lambda o: None if o != o else str(o))
    print("guardado", a.out)


if __name__ == "__main__":
    main()
