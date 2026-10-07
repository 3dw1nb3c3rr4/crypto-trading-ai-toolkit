"""Pruebas de alineación y anti-fuga de derivs_features.py con datos sintéticos de respuesta conocida.
Uso: python backtesting/test_derivs_features.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import derivs_features as df_  # noqa: E402

DAY = 86_400_000
idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
cols = ["AAA/USDT:USDT", "BBB/USDT:USDT", "CCC/USDT:USDT"]
rng = np.random.default_rng(0)
close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .02, (120, 3)), axis=0)), index=idx, columns=cols)
P = {"close": close, "volume": pd.DataFrame(1000.0, index=idx, columns=cols), "open": close, "high": close, "low": close}
t0 = int(idx[0].value // 10**6)


def make_funding(seed=1, extra_after=None):
    r = np.random.default_rng(seed); fd = {}
    for s in cols:
        ts = np.arange(t0 - 10 * DAY, t0 + 130 * DAY, 8 * 3600_000)
        rate = r.normal(1e-4, 5e-5, len(ts))
        if extra_after is not None:                                   # corrompe todo lo posterior a extra_after
            rate = np.where(ts > extra_after, 9.99, rate)
        fd[s] = pd.DataFrame({"ts": ts, "rate": rate})
    return fd


def make_oi(seed=2, extra_after=None):
    r = np.random.default_rng(seed); oi = {}
    for s in cols:
        ts = np.arange(t0 - 40 * DAY, t0 + 130 * DAY, DAY)
        usd = 1e8 * np.exp(np.cumsum(r.normal(0, .03, len(ts))))
        if extra_after is not None:
            usd = np.where(ts > extra_after, 1e15, usd)
        oi[s] = pd.DataFrame({"ts": ts, "oi_amt": usd / 100, "oi_usd": usd})
    return oi


ok = True
def check(name, cond):
    global ok
    print(("OK   " if cond else "FALLA"), name); ok &= bool(cond)

# 1) funding: la suma del día D incluye solo las liquidaciones 00:00, 08:00 y 16:00 de D
fund = {s: pd.DataFrame({"ts": np.arange(t0, t0 + 120 * DAY, 8 * 3600_000), "rate": np.tile([1.0, 10.0, 100.0], 120)}) for s in cols}
fdaily = df_.funding_daily(fund, idx, cols)
check("funding: suma diaria = 111 (00h + 08h + 16h del mismo día)", np.allclose(fdaily.iloc[5:60].to_numpy(), 111.0))

# 2) OI: con lag=1 la vela t ve la foto de su inicio (ts_t); con lag=0 la foto de ts_t + 1 día
oi = {s: pd.DataFrame({"ts": np.arange(t0 - 5 * DAY, t0 + 125 * DAY, DAY), "oi_amt": np.arange(130.0) + 1, "oi_usd": (np.arange(130.0) + 1) * 1e6}) for s in cols}
o1 = df_.oi_daily(oi, idx, cols, close, lag_days=1)
o0 = df_.oi_daily(oi, idx, cols, close, lag_days=0)
t = 40
snap_start = (np.arange(130.0) + 1)[(t + 5)] * 1e6                  # foto con ts == inicio del día t
check("OI lag=1: la vela t ve la foto de su inicio (no la del cierre)", np.isclose(o1.iloc[t, 0], snap_start))
check("OI lag=0: la vela t ve la foto del cierre (ts_t + 1 día)", np.isclose(o0.iloc[t, 0], snap_start + 1e6))

# 2b) fotos a las 00:05 (como los archivos reales de Binance): con lag=1 la vela t ve la foto 00:05 de su propio día, no la del día anterior
oi5 = {s: pd.DataFrame({"ts": np.arange(t0 - 5 * DAY, t0 + 125 * DAY, DAY) + 300_000, "oi_amt": np.arange(130.0) + 1, "oi_usd": (np.arange(130.0) + 1) * 1e6}) for s in cols}
o5 = df_.oi_daily(oi5, idx, cols, close, lag_days=1)
check("OI con fotos 00:05 y lag=1: la vela t ve la foto 00:05 de su día (23 h antes del cierre)", np.isclose(o5.iloc[t, 0], snap_start))
o5b = df_.oi_daily(oi5, idx, cols, close, lag_days=0)
check("OI con fotos 00:05 y lag=0: no ve la foto de las 00:05 posterior al cierre", np.isclose(o5b.iloc[t, 0], snap_start))

# 3) perturbación: alterar datos posteriores a la fecha T no cambia ninguna característica en fechas <= T
T = 70
cut_ts = t0 + T * DAY                                               # inicio del día T
F_f, F_o = df_.build(P, idx, make_funding(), make_oi(), oi_lag_days=1)
G_f, G_o = df_.build(P, idx, make_funding(extra_after=cut_ts + DAY), make_oi(extra_after=cut_ts), oi_lag_days=1)
for nombre, A, B in (("funding", F_f, G_f), ("OI", F_o, G_o)):
    diffs = [k for k in A if not np.allclose(A[k].iloc[:T + 1].to_numpy(), B[k].iloc[:T + 1].to_numpy(), equal_nan=True)]
    check(f"anti-fuga {nombre}: modificar el futuro no cambia las características hasta el día {T}", not diffs)
    if diffs: print("   cambian:", diffs)
changed_after = any(not np.allclose(F_o[k].iloc[T + 5:].to_numpy(), G_o[k].iloc[T + 5:].to_numpy(), equal_nan=True) for k in F_o)
check("sensibilidad: la corrupción SÍ afecta a fechas posteriores (la prueba detecta cambios)", changed_after)

# 4) símbolos sin datos -> NaN, sin errores; datos vacíos -> diccionarios vacíos
f2, o2 = df_.build(P, idx, {cols[0]: make_funding()[cols[0]]}, {cols[1]: make_oi()[cols[1]]})
check("símbolo sin datos de funding queda en NaN", f2["fund_7d"][cols[1]].isna().all() and f2["fund_7d"][cols[0]].notna().any())
check("símbolo sin datos de OI queda en NaN", o2["oi_chg7"][cols[0]].isna().all() and o2["oi_chg7"][cols[1]].notna().any())
check("sin datos: diccionarios vacíos", df_.build(P, idx, None, None) == ({}, {}))
print("\nTODOS LOS TESTS OK" if ok else "\nHAY FALLAS"); sys.exit(0 if ok else 1)
