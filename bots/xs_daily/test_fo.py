"""Prueba sin red de la variante FO: entrena con datos sintéticos y verifica que alterar el funding/OI FUTURO
no cambia la señal de hoy (sin fuga) y que el funding/OI de hoy sí la puede cambiar.
Uso: python bots/xs_daily/test_fo.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xs_core  # noqa: E402

DAY = 86_400_000


def synth(n_sym=40, n=420, seed=0):
    r = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    hist, fund, oi = {}, {}, {}
    for i in range(n_sym):
        c = 10 * np.exp(np.cumsum(r.normal(0, 0.03, n)))
        v = r.uniform(1e5, 1e6, n) * (1 + i)
        hist[f"S{i}/USDT:USDT"] = pd.DataFrame({"ts": ts, "open": c, "high": c * 1.02, "low": c * 0.98, "close": c, "volume": v})
        ms = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy()   # ms, sin depender de la resolución
        f_ts = np.concatenate([ms + h * 3600_000 for h in (0, 8, 16)])
        fund[f"S{i}/USDT:USDT"] = pd.DataFrame({"ts": np.sort(f_ts), "rate": r.normal(0.0001, 0.0002, len(f_ts))})
        oi[f"S{i}/USDT:USDT"] = pd.DataFrame({"ts": ms + 300_000, "oi_amt": r.uniform(1e5, 2e5, n), "oi_usd": r.uniform(1e6, 2e6, n)})
    return hist, fund, oi


def main():
    hist, fund, oi = synth()
    b = xs_core.train(hist, "FO", (fund, oi))
    assert b["variant"] == "FO" and any(f.startswith("fund_") for f in b["features"]) and any(f.startswith("oi_") for f in b["features"])
    t = hist["S0/USDT:USDT"].ts.iloc[-1]
    cut = int(t.value // 10**6) + DAY                     # todo lo posterior al cierre de la última vela es "futuro"
    fut_f = {s: d.assign(rate=np.where(d.ts >= cut, 0.05, d.rate)) for s, d in fund.items()}
    fut_o = {s: d.assign(oi_usd=np.where(d.ts > cut - DAY + 3600_000, 9e9, d.oi_usd)) for s, d in oi.items()}
    _, L1, S1, P1 = xs_core.pick(b, hist, (fund, oi))
    _, L2, S2, P2 = xs_core.pick(b, hist, (fut_f, fut_o))
    assert L1 == L2 and S1 == S2 and np.allclose(list(P1.values()), list(P2.values())), "FUGA: el futuro cambió la señal"
    big = {s: d.assign(rate=np.where((d.ts >= cut - DAY) & (d.ts < cut), (i % 7) * 0.01, d.rate)) for i, (s, d) in enumerate(fund.items())}
    _, _, _, P3 = xs_core.pick(b, hist, (big, oi))
    assert not np.allclose(list(P1.values()), list(P3.values())), "el funding de hoy no influye en nada"
    _, L4, S4, _ = xs_core.pick(b, hist, ({}, {}))              # sin derivados: el modelo sigue funcionando (NaN)
    assert len(L4) == len(L1)
    print(f"test_fo: OK (K={len(L1)} por lado, {len(b['features'])} factores)")


if __name__ == "__main__":
    main()
