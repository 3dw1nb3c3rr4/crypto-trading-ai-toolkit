"""Características diarias de funding y open interest (OI) para el modelo cross-sectional.

Alineación (anti-sesgo), todas "as-of" respecto al cierre de la vela diaria t (cierra en ts_t + 1 día):
  - funding: se suman las liquidaciones con ts dentro del día t (00:00, 08:00, 16:00 UTC); la de las 00:00 del día siguiente
    pertenece al día t+1. Nada posterior al cierre entra.
  - OI: es una foto en ts. Por seguridad, la vela t solo ve fotos con ts <= ts_t (inicio del día t), es decir un día de retraso
    (oi_lag_days=1). Con oi_lag_days=0 usaría la foto de las 00:00 del día siguiente, que solo es válida si la foto es instantánea.
Ideas tomadas de: Hummingbot (funding normalizado a base diaria, diferencias entre exchanges), cryptofeed (canales FUNDING/OPEN_INTEREST).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400_000


def _to_index(ts_ms) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(ts_ms, dtype="int64"), unit="ms", utc=True))


def funding_daily(funding: dict, idx: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """Funding acumulado por día calendario (fracción por día) para cada símbolo. NaN si el símbolo no tiene datos ese día."""
    out = {}
    for s in cols:
        f = funding.get(s)
        if f is None or len(f) < 10:
            continue
        ser = pd.Series(f["rate"].to_numpy(dtype="float64"), index=_to_index(f["ts"])).sort_index()
        ser = ser[~ser.index.duplicated()]
        daily = ser.resample("1D", label="left", closed="left").sum(min_count=1)
        out[s] = daily.reindex(idx)
    return pd.DataFrame(out, index=idx).reindex(columns=cols)


def oi_daily(oi: dict, idx: pd.DatetimeIndex, cols: list[str], close: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """OI en USDT visible al cierre de cada vela diaria: última foto con ts <= idx + (1 - lag_days) días."""
    out = {}
    for s in cols:
        o = oi.get(s)
        if o is None or len(o) < 10:
            continue
        ts = _to_index(o["ts"])
        usd = o["oi_usd"].to_numpy(dtype="float64") if "oi_usd" in o else np.full(len(o), np.nan)
        amt = o["oi_amt"].to_numpy(dtype="float64")
        ser = pd.Series(usd, index=ts).sort_index()
        ser = ser[~ser.index.duplicated()]
        if ser.isna().mean() > 0.5:                                       # el exchange no da valor en USDT: unidades x precio de cierre
            a = pd.Series(amt, index=ts).sort_index()
            a = a[~a.index.duplicated()]
            px = close[s].reindex(a.index.floor("1D"), method="ffill").to_numpy()
            ser = pd.Series(a.to_numpy() * px, index=a.index)
        # tolerancia de 1 h con lag>=1: las fotos de los archivos reales salen a las 00:05 (siguen siendo 23 h antes del cierre)
        visible_until = idx + pd.Timedelta(days=1 - lag_days) + (pd.Timedelta(hours=1) if lag_days >= 1 else pd.Timedelta(0))
        out[s] = ser.reindex(visible_until, method="ffill").set_axis(idx)
    return pd.DataFrame(out, index=idx).reindex(columns=cols)


def build(P: dict, idx: pd.DatetimeIndex, funding: dict | None, oi: dict | None, oi_lag_days: int = 1):
    """Devuelve (F_funding, F_oi): diccionarios nombre -> DataFrame (T x S). Vacío si no hay datos."""
    cols = list(P["close"].columns)
    c, v = P["close"], P["volume"]
    Ff, Fo = {}, {}
    if funding:
        fd = funding_daily(funding, idx, cols)
        f7 = fd.rolling(7, min_periods=5).mean()
        f30 = fd.rolling(30, min_periods=20)
        Ff = {"fund_1d": fd, "fund_3d": fd.rolling(3, min_periods=2).mean(), "fund_7d": f7,
              "fund_z30": (f7 - f30.mean()) / (f30.std() + 1e-9), "fund_chg3": fd - fd.shift(3),
              "fund_pos7": (fd > 0).astype(float).where(fd.notna()).rolling(7, min_periods=5).mean()}
    if oi:
        o = oi_daily(oi, idx, cols, c, oi_lag_days)
        lo = np.log(o.where(o > 0))
        dvol = (c * v).rolling(30, min_periods=20).mean()
        ret7 = np.log(c).diff(7)
        oc7 = lo.diff(7)
        Fo = {"oi_chg1": lo.diff(1), "oi_chg7": oc7, "oi_chg30": lo.diff(30),
              "oi_z30": (lo - lo.rolling(30, min_periods=20).mean()) / (lo.rolling(30, min_periods=20).std() + 1e-9),
              "oi_vol": np.log((o / (dvol + 1e-9)).where(o > 0)),
              "oi_px_div": oc7.rank(axis=1, pct=True) - ret7.rank(axis=1, pct=True)}      # OI sube sin que suba el precio = posicionamiento
    return Ff, Fo
