"""Simula el bot paper día a día con un exchange falso construido con velas reales (ejecución a la apertura de cada día).
Verifica la contabilidad y da una prueba fuera de muestra: el modelo se entrena solo con datos anteriores a --start.
Uso: python bots/xs_daily/test_paper_sim.py [--start 2026-06-01]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paper as rp  # noqa: E402
import xs_core  # noqa: E402
from engine import load_universe  # noqa: E402

DAY = 86_400_000


class FakeExchange:
    def __init__(self, uni):
        self.uni = {s: df.assign(ms=df.ts.astype("int64") // 10**6 if df.ts.dtype.kind == "M" else df.ts) for s, df in uni.items()}
        for s, df in self.uni.items():
            df["ms"] = (pd.to_datetime(df.ts, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
        self.now = None

    def milliseconds(self):
        return self.now + 3600_000

    def fetch_ohlcv(self, s, tf, limit=150):
        df = self.uni[s]; d = df[df.ms < self.now].tail(limit)
        return d[["ms", "open", "high", "low", "close", "volume"]].values.tolist()

    def fetch_ticker(self, s):
        row = self.uni[s][self.uni[s].ms == self.now]
        if row.empty:
            raise RuntimeError("sin precio")
        o = float(row.open.iloc[0])
        return {"last": o, "bid": o * 0.9999, "ask": o * 1.0001}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--data", default=os.path.join(xs_core.ROOT, "data", "trading_history", "ohlcv_cache_2y.pkl"))
    ap.add_argument("--save", default=None, help="CSV con las cohortes cerradas")
    ap.add_argument("--variant", default="D", choices=("D", "FO"))
    ap.add_argument("--derivs", nargs="+", default=None, help="historial de funding/OI para FO")
    a = ap.parse_args()
    uni = load_universe(a.data, min_bars=150)
    train_uni = {s: df[df.ts < a.start].reset_index(drop=True) for s, df in uni.items()}
    derivs = None
    if a.variant == "FO":
        import derivs_live
        derivs = derivs_live.load_seed(a.derivs)
        print(f"derivados: funding {len(derivs[0])} símbolos, OI {len(derivs[1])}")
    bundle = xs_core.train({s: d for s, d in train_uni.items() if len(d) >= 150}, a.variant, derivs)
    print(f"modelo entrenado hasta {bundle['trained_until']} con {bundle['n_rows']} filas (sin ver nada desde {a.start})")
    ex = FakeExchange(uni)
    days = pd.date_range(pd.Timestamp(a.start, tz="UTC"), max(df.ts.iloc[-1] for df in uni.values()), freq="D")
    state = {"equity": 1000.0, "cohorts": [], "closed": []}
    errors = 0
    for d in days:
        ex.now = int(d.value // 10**6)
        before = {c["opened"]: c for c in state["cohorts"]}
        rp.step(state, ex, bundle, str(d.date()), log=lambda *x: None, derivs=derivs)   # derivs se alinean as-of a cada día
        # verificación independiente de cada cohorte cerrada hoy
        for c in state["closed"]:
            if c["closed"] != str(d.date()) or c["opened"] not in before:
                continue
            coh = before[c["opened"]]
            age = (d - pd.Timestamp(c["opened"], tz="UTC")).days
            exp = 0.0
            for p in coh["positions"]:
                t = ex.fetch_ticker(p["symbol"]) if not ex.uni[p["symbol"]][ex.uni[p["symbol"]].ms == ex.now].empty else None
                px = (t["bid"] * (1 - rp.SLIP) if p["side"] == "LONG" else t["ask"] * (1 + rp.SLIP)) if t else p["entry"]
                sgn = 1 if p["side"] == "LONG" else -1
                exp += p["notional"] * sgn * (px / p["entry"] - 1) - p["notional"] * rp.FEE
            if age != bundle["H"] or abs(exp - c["pnl"]) > 1e-9:
                errors += 1
    cl = pd.DataFrame(state["closed"])
    print(f"días simulados={len(days)} | cohortes cerradas={len(cl)} | errores de contabilidad/edad={errors}")
    if a.save:
        cl.to_csv(a.save, index=False)
    r = cl["ret"].to_numpy()
    print(f"FUERA DE MUESTRA ({a.start} -> fin): retorno medio por cohorte {r.mean():+.3%} (sin comisión de entrada) | "
          f"cohortes positivas {np.mean(r > 0):.0%} | equity final realizada {state['equity']:.2f} desde 1000 "
          f"(quedan {len(state['cohorts'])} cohortes abiertas)")
