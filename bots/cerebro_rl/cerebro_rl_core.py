# -*- coding: utf-8 -*-
"""
cerebro_rl_core.py — Cerebro RL, reemplazo de JEV.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyarrow, pyarrow.parquet

TP_PCT      = 0.015
SL_PCT      = 0.015
FEE_TAKER   = 0.0005
SLIPPAGE    = 0.0002
MAX_HOLD    = 288
WARMUP      = 150

SIZING_NOMBRES = ["pequeño", "medio", "completo"]
SIZING_FACTOR  = np.array([0.33, 0.66, 1.00], dtype=np.float32)

ACCIONES   = ["esperar", "long_pequeño", "long_medio", "long_completo",
              "short_pequeño", "short_medio", "short_completo"]
N_ACC      = len(ACCIONES)
REGIMENES  = ["tendencia_alcista", "tendencia_bajista", "rango",
              "trampa_o_manipulacion", "caos_alta_volatilidad"]
QUIENES    = ["compradores_agresivos", "vendedores_agresivos",
              "pasividad_equilibrio", "absorcion_institucional"]

FUERZA_MINIMA_BASE = 2.0
RIESGO_MAX_BASE    = 0.50
TIMING_MINIMO      = 0.45
N_POS_FEATS = 6

def _rmean(a, n):  return pd.Series(a).rolling(n, min_periods=n).mean().to_numpy()
def _rmax(a, n):   return pd.Series(a).rolling(n, min_periods=n).max().to_numpy()
def _rmin(a, n):   return pd.Series(a).rolling(n, min_periods=n).min().to_numpy()
def _rsum(a, n):   return pd.Series(a).rolling(n, min_periods=n).sum().to_numpy()
def _shift(a, k):
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[:-k] if k > 0 else a
    return out

def a_dataframe(ohlcv) -> pd.DataFrame:
    if isinstance(ohlcv, pd.DataFrame):
        df = ohlcv.copy()
        if "ts" not in df.columns:
            t = pd.to_datetime(df["timestamp"], utc=True)
            df["ts"] = (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    else:
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
    df["ts"] = df["ts"].astype("int64")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    return df.reset_index(drop=True)

FEATURE_NAMES = (
    ["body", "rango", "mecha_sup", "mecha_inf", "cierre_en_rango", "racha"]
    + [f"cuerpo_lag{k}" for k in range(1, 12)]
    + ["hh10", "ll10", "dist_max20", "dist_min20", "dist_max50", "dist_min50",
       "vol_media20", "vol_ult_vs_media", "volumen_rel",
       "ret3", "ret12", "ret48", "ret144", "flujo6", "flujo20",
       "dist_sma20", "dist_sma50", "pend_sma20", "hora_sin", "hora_cos"]
)
N_FEATS = len(FEATURE_NAMES)

def build_features(df_in) -> np.ndarray:
    df = a_dataframe(df_in)
    o, h, l, c, v = (df[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
    T = len(df); eps = 1e-12
    body = (c - o) / np.maximum(o, eps) * 100
    rng  = (h - l) / np.maximum(o, eps) * 100
    span = np.where(h - l > 0, h - l, np.nan)
    loc  = np.where(np.isnan(span), 0.5, (c - l) / np.where(np.isnan(span), 1, span))
    upw  = np.where(np.isnan(span), 0.0, (h - np.maximum(o, c)) / np.where(np.isnan(span), 1, span))
    loww = np.where(np.isnan(span), 0.0, (np.minimum(o, c) - l) / np.where(np.isnan(span), 1, span))
    color  = (c > o).astype(np.int8)
    change = np.r_[True, color[1:] != color[:-1]]
    idx    = np.arange(T)
    start  = np.maximum.accumulate(np.where(change, idx, 0))
    racha  = (idx - start + 1) * np.where(color == 1, 1, -1)
    cols = [body, rng, upw, loww, loc, racha.astype(float)]
    cols += [_shift(body, k) for k in range(1, 12)]
    hh = _rsum(np.r_[0, (h[1:] > h[:-1]).astype(float)], 9) / 9
    ll = _rsum(np.r_[0, (l[1:] < l[:-1]).astype(float)], 9) / 9
    max20, min20 = _rmax(h, 20), _rmin(l, 20)
    max50, min50 = _rmax(h, 50), _rmin(l, 50)
    cols += [hh, ll, (max20 - c) / c * 100, (c - min20) / c * 100,
             (max50 - c) / c * 100, (c - min50) / c * 100]
    vol_med = _rmean(rng, 20); vmean20 = _rmean(v, 20)
    volrel  = np.log((v + 1e-9) / (vmean20 + 1e-9) + 1e-3)
    cols += [vol_med, rng / np.maximum(vol_med, 1e-9), volrel]
    for n in (3, 12, 48, 144):
        cols.append((c / _shift(c, n) - 1) * 100)
    signed = v * (2 * loc - 1)
    cols += [_rsum(signed, 6) / np.maximum(_rsum(v, 6), 1e-12),
             _rsum(signed, 20) / np.maximum(_rsum(v, 20), 1e-12)]
    sma20, sma50 = _rmean(c, 20), _rmean(c, 50)
    cols += [(c / sma20 - 1) * 100, (c / sma50 - 1) * 100, (sma20 / _shift(sma20, 5) - 1) * 100]
    hora = ((df["ts"].to_numpy() // 60000) % 1440) / 1440.0
    cols += [np.sin(2 * np.pi * hora), np.cos(2 * np.pi * hora)]
    X = np.stack(cols, axis=1).astype(np.float64)
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e4, 1e4)
    X[:WARMUP] = 0.0
    return X.astype(np.float32)

def normalizar(X, mu, sd): return np.clip((X - mu) / sd, -5.0, 5.0).astype(np.float32)

def pos_features(side, size, ret, hold, mfe, mae) -> np.ndarray:
    side, size, ret, hold, mfe, mae = (np.asarray(a, dtype=np.float32) for a in (side, size, ret, hold, mfe, mae))
    out = np.stack([side, size * (side != 0), np.clip(ret / SL_PCT, -3, 3),
                    np.clip(hold / MAX_HOLD, 0, 1), np.clip(mfe / TP_PCT, 0, 4),
                    np.clip(mae / SL_PCT, -3, 0)], axis=-1)
    return out.astype(np.float32)

class CerebroNet(nn.Module):
    def __init__(self, n_mkt=N_FEATS, hid=128):
        super().__init__()
        self.n_mkt = n_mkt
        self.trunk = nn.Sequential(
            nn.Linear(n_mkt + N_POS_FEATS, hid), nn.LayerNorm(hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU())
        self.h_entry = nn.Linear(hid, N_ACC); self.h_exit = nn.Linear(hid, 2)
        self.h_value = nn.Linear(hid, 1)
        self.h_aux_e = nn.Linear(hid, 3 + len(REGIMENES) + len(QUIENES))
        self.h_aux_x = nn.Linear(hid, 3)
    def forward(self, x):
        z = self.trunk(x)
        return {"entry": self.h_entry(z), "exit": self.h_exit(z),
                "value": self.h_value(z).squeeze(-1),
                "aux_e": self.h_aux_e(z), "aux_x": self.h_aux_x(z)}

def logits_politica(out, en_pos):
    ent = out["entry"]; ex = torch.full_like(ent, -1e9)
    ex[:, :2] = out["exit"]
    return torch.where(en_pos.unsqueeze(-1), ex, ent)

def split_aux_entrada(aux_e):
    calidad = 4.0 * torch.sigmoid(aux_e[:, 0]); riesgo = torch.sigmoid(aux_e[:, 1])
    timing  = torch.sigmoid(aux_e[:, 2]); reg = aux_e[:, 3:3 + len(REGIMENES)]
    quien   = aux_e[:, 3 + len(REGIMENES):]
    return calidad, riesgo, timing, reg, quien

class CerebroRL:
    def __init__(self, net, mu, sd, device="cpu", min_prob=0.0):
        self.net, self.mu, self.sd, self.device = net.eval(), mu, sd, device
        self.min_prob = float(min_prob)   # umbral de confianza calibrado en validación
    @classmethod
    def cargar(cls, path, device="cpu"):
        b = torch.load(path, map_location=device, weights_only=False)
        net = CerebroNet(b["n_mkt"], b["hid"]).to(device)
        net.load_state_dict(b["state_dict"])
        return cls(net, np.asarray(b["mu"], np.float32), np.asarray(b["sd"], np.float32), device, b.get("min_prob", 0.0))
    def _obs(self, ohlcv, side=0, size=0.0, ret=0.0, hold=0, mfe=0.0, mae=0.0):
        df = a_dataframe(ohlcv)
        if len(df) < WARMUP + 1:
            raise ValueError(f"Se necesitan >= {WARMUP + 1} velas de 5m.")
        X = normalizar(build_features(df)[-1:], self.mu, self.sd)
        P = pos_features([side], [size], [ret], [hold], [mfe], [mae])
        return df, torch.tensor(np.concatenate([X, P], axis=1), device=self.device)
    @torch.no_grad()
    def decidir_entrada(self, ohlcv, min_prob=None):
        """Mismo formato que _jev_decidir_entrada(). Solo entra si la confianza >= umbral calibrado."""
        min_prob = self.min_prob if min_prob is None else min_prob
        df, obs = self._obs(ohlcv); out = self.net(obs)
        prob = torch.softmax(out["entry"], -1)[0].cpu().numpy(); a = int(prob.argmax())
        if a != 0 and prob[a] < min_prob: a = 0
        cal, rie, tim, reg, quien = split_aux_entrada(out["aux_e"])
        cal, rie, tim = float(cal[0]), float(rie[0]), float(tim[0])
        direccion = "esperar" if a == 0 else ("long" if a <= 3 else "short")
        sizing    = SIZING_NOMBRES[(a - 1) % 3] if a > 0 else "pequeño"
        if a > 0: cal, rie, tim = max(cal, FUERZA_MINIMA_BASE), min(rie, RIESGO_MAX_BASE), max(tim, 0.5)
        return {"direccion": direccion, "calidad": cal, "riesgo": rie,
                "momentum": tim, "timing": tim, "precio": float(df["close"].iloc[-1]),
                "sizing": sizing, "regimen": REGIMENES[int(reg[0].argmax())],
                "conf_regimen": float(torch.softmax(reg[0], -1).max()),
                "quien_controla": QUIENES[int(quien[0].argmax())],
                "funding_peligroso": False, "oi_confirma": True,
                "prob_accion": float(prob[a]), "probs": {n: float(p) for n, p in zip(ACCIONES, prob)}}
    @torch.no_grad()
    def decidir_cierre(self, ohlcv, lado, entry_price, size_factor=1.0, entry_ts_ms=None,
                        hold_velas=0, umbral_cierre=0.5):
        """umbral_cierre: prob_cerrar minima para que 'cerrar'/'cerrar_ya' den True.
        Default 0.5 (el original). Un backtest sobre 40 monedas reales del universo
        del bot (ver sim_barrido_umbral.py) mostro que 0.05 recorta perdidas frente
        a 0.5 en la ventana probada -- ver umbral_cierre en quien llama a esta funcion
        para saber si esta usando el default o el ajustado."""
        side = 1 if lado.upper() == "LONG" else -1
        df = a_dataframe(ohlcv); precio = float(df["close"].iloc[-1])
        ret = side * (precio / entry_price - 1.0); mfe = mae = 0.0
        if entry_ts_ms is not None:
            tramo = df[df["ts"] > entry_ts_ms]; hold_velas = len(tramo)
            if len(tramo):
                r = side * (tramo["close"].to_numpy() / entry_price - 1.0)
                mfe, mae = float(max(r.max(), 0.0)), float(min(r.min(), 0.0))
        _, obs = self._obs(df, side, size_factor, ret, hold_velas, max(mfe, ret, 0.0), min(mae, ret, 0.0))
        out = self.net(obs); p_cl = float(torch.softmax(out["exit"], -1)[0, 1])
        ax = torch.sigmoid(out["aux_x"])[0].cpu().numpy()
        cerrar = p_cl >= umbral_cierre
        return {"cerrar_ya": cerrar, "prob_cerrar": p_cl, "pnl_pct": ret,
                "cerrar": cerrar, "revertida": bool(ax[1] >= 0.55),
                "contexto_ok": bool(ax[2] >= 0.50)}
