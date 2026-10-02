"""
okx_topsuelo_bot.py
====================
Bot "cerebro" multi-moneda para OKX Futures (perpetuos USDT-margined) que:

  1) UNIVERSO: toma las 300 monedas con mayor volumen de OKX Futures hoy.
  2) VOLATILIDAD: de esas 300, se queda con las 150 mas volatiles (las que
     tuvieron subidas explosivas y bajadas explosivas en las ultimas semanas).
  3) CALIDAD DE HISTORIA: de esas 150 escoge las 50 "mejores" -- las de mayor
     score de volatilidad QUE ADEMAS tengan al menos ~2 anios de velas diarias
     de historia. Si una de las 150 no tiene suficiente historia, se descarta
     y se reemplaza por la siguiente candidata (primero dentro de las 150, y
     si hace falta, dentro de las 300) hasta completar 50.
  4) ENTRENAMIENTO: con los 2 anios de velas diarias de esas 50 monedas se
     entrena una red neuronal multi-escala (90d / 30d / 14d / 7d) que aprende
     a reconocer dos patrones:
       - "TECHO"  -> la moneda tuvo una subida sostenida (pump) de mas de un
                     mes, llego a una zona de techo y esta empezando a caer.
       - "SUELO"  -> la moneda tuvo una caida sostenida (dump) de mas de un
                     mes y esta empezando a girar hacia arriba.
     Todo lo demas se etiqueta "NEUTRAL".
  5) ESCANER / BACKTEST: una vez entrenado, el modelo puede evaluar CUALQUIER
     moneda de la lista de OKX Futures (no solo las 50 de entrenamiento) y
     mostrar las 20 mejores candidatas en cada categoria.

Nada de esto coloca ordenes reales ni es asesoria financiera -- es un
clasificador de patrones tecnicos para investigacion / paper trading.
"""

import os
import time
import math
import json
import pickle
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ccxt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0) CONFIG POR DEFECTO (se puede sobreescribir desde la celda de driver)
# ---------------------------------------------------------------------------

TIMEFRAME          = "1d"          # velas diarias para el analisis "macro"
DAYS_HISTORY_NEEDED = 700           # ~2 anios de velas diarias minimo exigido
TARGET_DAYS_HISTORY = 730           # 2 anios objetivo (se pide un poco de mas margen)

TOP_BY_VOLUME       = 300
TOP_BY_VOLATILITY   = 150
FINAL_UNIVERSE_SIZE = 50

VOL_LOOKBACK_DAYS    = 90    # ventana usada para medir "explosividad"
PUMP_DUMP_WINDOW     = 30    # "mas de 1 mes" de subida/caida sostenida
FUTURE_HORIZON_DAYS  = 7     # a cuantos dias mira el label para saber si giro

REQUEST_SLEEP = 0.15         # pausa extra entre requests (ademas del rateLimit de ccxt)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1) CONEXION A OKX (FUTUROS / SWAP PERPETUOS USDT-M)
# ---------------------------------------------------------------------------

def make_okx_exchange():
    """Crea el cliente ccxt de OKX configurado para futuros perpetuos (swap)."""
    ex = ccxt.okx({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    ex.load_markets()
    return ex


def list_usdt_perpetuals(ex):
    """Devuelve la lista de simbolos de perpetuos USDT-margined activos en OKX."""
    symbols = []
    for sym, m in ex.markets.items():
        if not m.get("active", True):
            continue
        if m.get("swap") and m.get("linear") and m.get("quote") == "USDT":
            symbols.append(sym)
    return sorted(set(symbols))


# ---------------------------------------------------------------------------
# 2) PASO 1 -- TOP 300 POR VOLUMEN DE HOY
# ---------------------------------------------------------------------------

def get_top_by_volume(ex, top_n=TOP_BY_VOLUME):
    """Consulta los tickers de OKX Futures y devuelve los `top_n` simbolos
    con mayor volumen en USDT de las ultimas 24h (volumen de "hoy")."""
    usdt_perp = set(list_usdt_perpetuals(ex))
    tickers = ex.fetch_tickers(list(usdt_perp))

    rows = []
    for sym, t in tickers.items():
        if sym not in usdt_perp:
            continue
        qvol = t.get("quoteVolume")
        if qvol is None:
            # fallback: baseVolume * ultimo precio
            base_vol = t.get("baseVolume") or 0
            last = t.get("last") or 0
            qvol = base_vol * last
        if qvol and qvol > 0:
            rows.append((sym, float(qvol)))

    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[:top_n]
    print(f"[1/4] {len(rows)} perpetuos USDT activos -> top {len(top)} por volumen 24h.")
    return top  # lista de (symbol, quote_volume)


# ---------------------------------------------------------------------------
# 3) DESCARGA DE VELAS CON PAGINACION (para historia larga)
# ---------------------------------------------------------------------------

def fetch_ohlcv_paginated(ex, symbol, timeframe="1d", since_ms=None, max_candles=800,
                           per_call_limit=300, sleep=REQUEST_SLEEP):
    """Descarga velas OHLCV hacia atras en el tiempo hasta reunir `max_candles`
    (o hasta que OKX ya no tenga mas historia). Devuelve lista ordenada por
    tiempo ascendente, sin duplicados."""
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    if since_ms is None:
        since_ms = ex.milliseconds() - max_candles * tf_ms

    all_rows = {}
    cursor = since_ms
    attempts = 0
    max_attempts = math.ceil(max_candles / per_call_limit) + 5

    while len(all_rows) < max_candles and attempts < max_attempts:
        attempts += 1
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=per_call_limit)
        except Exception:
            break
        if not batch:
            break
        for row in batch:
            all_rows[row[0]] = row
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            # no avanzo, evitar loop infinito
            cursor = last_ts + tf_ms
        else:
            cursor = last_ts + tf_ms
        if cursor >= ex.milliseconds():
            break
        time.sleep(sleep)

    rows = sorted(all_rows.values(), key=lambda r: r[0])
    return rows[-max_candles:] if len(rows) > max_candles else rows


def ohlcv_to_df(rows):
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 4) PASO 2 -- TOP 150 POR VOLATILIDAD ("explosividad")
# ---------------------------------------------------------------------------

def volatility_score(df, lookback_days=VOL_LOOKBACK_DAYS):
    """Mide que tan 'explosiva' es una moneda: la mayor subida en ventana
    corta + la mayor bajada en ventana corta, dentro de los ultimos
    `lookback_days`. Cuanto mas grandes los pumps Y los dumps, mas alto
    el score."""
    d = df.tail(lookback_days).copy()
    if len(d) < 20:
        return -1.0
    close = d["close"].values
    # retorno rodante de 3 dias (pump/dump corto y explosivo)
    window = 3
    rets = close[window:] / close[:-window] - 1.0
    if len(rets) == 0:
        return -1.0
    max_pump = float(np.nanmax(rets))
    max_dump = float(-np.nanmin(rets))
    # tambien premiamos el rango total recorrido (high/low) como proxy de rango
    rng = float((d["high"].max() - d["low"].min()) / d["close"].mean())
    score = max(max_pump, 0) + max(max_dump, 0) + 0.5 * rng
    return score


def rank_by_volatility(ex, candidates, top_n=TOP_BY_VOLATILITY, lookback_days=VOL_LOOKBACK_DAYS):
    """Descarga velas cortas (solo lo necesario para medir volatilidad) para
    cada candidato y devuelve la lista ordenada por score de volatilidad."""
    scored = []
    for sym, qvol in tqdm(candidates, desc="Midiendo volatilidad"):
        try:
            rows = fetch_ohlcv_paginated(ex, sym, timeframe=TIMEFRAME,
                                          max_candles=lookback_days + 10)
            if len(rows) < 20:
                continue
            df = ohlcv_to_df(rows)
            score = volatility_score(df, lookback_days=lookback_days)
            scored.append((sym, qvol, score))
        except Exception as e:
            continue
        time.sleep(REQUEST_SLEEP)

    scored.sort(key=lambda r: r[2], reverse=True)
    top = scored[:top_n]
    print(f"[2/4] {len(scored)} monedas evaluadas -> top {len(top)} por volatilidad.")
    return top  # lista de (symbol, quote_volume, vol_score)


# ---------------------------------------------------------------------------
# 5) PASO 3 -- FINAL 50 CON >= 2 ANIOS DE HISTORIA DIARIA
# ---------------------------------------------------------------------------

def build_final_universe(ex, volatile_ranked, all_ranked_by_volume,
                          final_n=FINAL_UNIVERSE_SIZE,
                          days_needed=DAYS_HISTORY_NEEDED,
                          target_days=TARGET_DAYS_HISTORY):
    """Recorre las candidatas (primero las 150 mas volatiles, y si faltan,
    el resto de las 300 por volumen) y se queda con las primeras `final_n`
    que tengan al menos `days_needed` velas diarias de historia real."""
    # orden de intentos: primero las volatiles (ya vienen ordenadas por score),
    # luego el resto del pool original por volumen (por si hace falta rellenar)
    volatile_syms = [s for s, *_ in volatile_ranked]
    fallback_syms = [s for s, *_ in all_ranked_by_volume if s not in set(volatile_syms)]
    order = volatile_syms + fallback_syms

    final = {}
    data_cache = {}
    for sym in tqdm(order, desc="Verificando 2 anios de historia"):
        if len(final) >= final_n:
            break
        try:
            rows = fetch_ohlcv_paginated(ex, sym, timeframe=TIMEFRAME, max_candles=target_days)
        except Exception:
            continue
        if len(rows) < days_needed:
            continue  # no tiene suficiente historia, se descarta y se sigue con la siguiente
        df = ohlcv_to_df(rows)
        final[sym] = df
        data_cache[sym] = df
        time.sleep(REQUEST_SLEEP)

    print(f"[3/4] Universo final: {len(final)}/{final_n} monedas con >= {days_needed} velas diarias.")
    if len(final) < final_n:
        print("     (No alcanzaron todas las candidatas posibles; se sigue con las que si califican.)")
    return data_cache  # dict symbol -> DataFrame


# ---------------------------------------------------------------------------
# 6) INDICADORES / FEATURES TECNICOS
# ---------------------------------------------------------------------------

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df):
    """Construye el set de indicadores tecnicos usados como input del modelo
    y como reglas para etiquetar los patrones de techo/suelo."""
    d = df.copy()
    close = d["close"]
    vol = d["volume"]

    d["ret_1"] = close.pct_change(1)
    d["ret_7"] = close.pct_change(7)
    d["ret_14"] = close.pct_change(14)
    d["ret_30"] = close.pct_change(PUMP_DUMP_WINDOW)

    d["ma7"] = close.rolling(7).mean()
    d["ma30"] = close.rolling(30).mean()
    d["ma_slope_30"] = d["ma30"].pct_change(20)

    roll_max_30 = close.rolling(30).max()
    roll_min_30 = close.rolling(30).min()
    d["dist_high_30"] = (close - roll_max_30) / roll_max_30
    d["dist_low_30"] = (close - roll_min_30) / roll_min_30

    d["rsi_14"] = rsi(close, 14)

    vol_ma20 = vol.rolling(20).mean()
    d["vol_ratio"] = vol / vol_ma20.replace(0, np.nan)

    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    d["bb_width"] = (4 * bb_std) / bb_ma.replace(0, np.nan)  # ancho de banda (squeeze si es chico)
    d["bb_width_chg"] = d["bb_width"].pct_change(10)  # crecio el ancho => salida de lateral

    # posicion del cierre dentro del rango de la vela (proxy de presion compradora/vendedora)
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["candle_pos"] = (close - d["low"]) / rng

    # rompimiento: cierre por fuera del rango de los ultimos 20 dias (excluyendo hoy)
    prev_high_20 = d["high"].shift(1).rolling(20).max()
    prev_low_20 = d["low"].shift(1).rolling(20).min()
    d["breakout_up"] = (close > prev_high_20).astype(float)
    d["breakout_down"] = (close < prev_low_20).astype(float)

    return d


# ---------------------------------------------------------------------------
# 7) ETIQUETAS (TECHO / SUELO / NEUTRAL) PARA ENTRENAMIENTO
# ---------------------------------------------------------------------------
# Label = 2 -> TECHO   (subio sostenido >1 mes y AHORA esta empezando a caer)
# Label = 0 -> SUELO   (bajo sostenido >1 mes y AHORA esta empezando a subir)
# Label = 1 -> NEUTRAL (todo lo demas)
#
# El label usa el retorno FUTURO (siguientes `FUTURE_HORIZON_DAYS`) solo como
# variable objetivo de entrenamiento -- nunca como feature de entrada del
# modelo. Esto es supervision estandar: "dado lo que paso hasta hoy, que fue
# lo que ocurrio despues".

PUMP_THRESHOLD    = 0.25   # +25% en el mes se considera "subida sostenida"
DUMP_THRESHOLD    = -0.25  # -25% en el mes se considera "caida sostenida"
REVERSAL_THRESHOLD = 0.04  # 4% en la direccion contraria en el horizonte futuro


def build_labels(d):
    future_ret = d["close"].shift(-FUTURE_HORIZON_DAYS) / d["close"] - 1.0

    was_pumping = d["ret_30"] >= PUMP_THRESHOLD
    was_dumping = d["ret_30"] <= DUMP_THRESHOLD

    turning_down = future_ret <= -REVERSAL_THRESHOLD
    turning_up = future_ret >= REVERSAL_THRESHOLD

    label = pd.Series(1, index=d.index)  # NEUTRAL por defecto
    label[was_pumping & turning_down] = 2   # TECHO
    label[was_dumping & turning_up] = 0     # SUELO
    return label


LABEL_NAMES = {0: "SUELO", 1: "NEUTRAL", 2: "TECHO"}


# ---------------------------------------------------------------------------
# 8) VENTANAS MULTI-ESCALA (90d / 30d / 14d / 7d)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "ret_1", "vol_ratio", "candle_pos", "dist_high_30", "dist_low_30",
    "bb_width", "breakout_up", "breakout_down",
]
SCALAR_COLS = [
    "ret_7", "ret_14", "ret_30", "ma_slope_30", "rsi_14", "bb_width_chg",
]

SCALES = {
    "w90": 90,
    "w30": 30,
    "w14": 14,
    "w7": 7,
}


def make_samples(d, symbol):
    """A partir del DataFrame con features+labels ya calculados, arma una
    lista de ejemplos (uno por dia) con las 4 ventanas multi-escala, el
    vector de features escalares y el label."""
    feat = d[FEATURE_COLS].values.astype(np.float32)
    scal = d[SCALAR_COLS].values.astype(np.float32)
    labels = d["label"].values
    n = len(d)
    max_w = max(SCALES.values())

    samples = []
    for i in range(max_w, n - FUTURE_HORIZON_DAYS):
        if not np.isfinite(scal[i]).all():
            continue
        windows = {}
        ok = True
        for name, w in SCALES.items():
            block = feat[i - w:i]
            if not np.isfinite(block).all():
                ok = False
                break
            windows[name] = block
        if not ok:
            continue
        samples.append({
            "symbol": symbol,
            "idx": i,
            "windows": windows,
            "scalar": scal[i],
            "label": int(labels[i]),
        })
    return samples


# ---------------------------------------------------------------------------
# 9) RED NEURONAL MULTI-ESCALA
# ---------------------------------------------------------------------------

class ScaleEncoder(nn.Module):
    """Codifica una ventana de tiempo (n_features x n_pasos) con Conv1d + GRU."""
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out_dim = hidden

    def forward(self, x):
        # x: (batch, n_pasos, n_features)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (batch, n_pasos, hidden)
        _, hn = self.gru(h)
        return hn.squeeze(0)  # (batch, hidden)


class MultiScaleTechoSueloNet(nn.Module):
    def __init__(self, n_features=len(FEATURE_COLS), n_scalar=len(SCALAR_COLS),
                 hidden=32, n_classes=3):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: ScaleEncoder(n_features, hidden) for name in SCALES
        })
        merged_dim = hidden * len(SCALES) + n_scalar
        self.head = nn.Sequential(
            nn.Linear(merged_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, n_classes)
        self.confidence = nn.Linear(32, 1)

    def forward(self, windows_dict, scalar):
        parts = [self.encoders[name](windows_dict[name]) for name in SCALES]
        parts.append(scalar)
        merged = torch.cat(parts, dim=1)
        h = self.head(merged)
        logits = self.classifier(h)
        conf_logit = self.confidence(h)
        return logits, conf_logit


# ---------------------------------------------------------------------------
# 10) DATASET / NORMALIZACION
# ---------------------------------------------------------------------------

class TechoSueloDataset(torch.utils.data.Dataset):
    def __init__(self, samples, scalar_mean, scalar_std):
        self.samples = samples
        self.scalar_mean = scalar_mean
        self.scalar_std = scalar_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        windows = {name: torch.tensor(w, dtype=torch.float32) for name, w in s["windows"].items()}
        scalar = (s["scalar"] - self.scalar_mean) / self.scalar_std
        scalar = torch.tensor(scalar, dtype=torch.float32)
        label = torch.tensor(s["label"], dtype=torch.long)
        return windows, scalar, label


def collate_fn(batch):
    windows_batch = {name: torch.stack([b[0][name] for b in batch]) for name in SCALES}
    scalar_batch = torch.stack([b[1] for b in batch])
    label_batch = torch.stack([b[2] for b in batch])
    return windows_batch, scalar_batch, label_batch


# ---------------------------------------------------------------------------
# 11) PIPELINE DE DATOS: 50 monedas -> features -> labels -> samples
# ---------------------------------------------------------------------------

def build_dataset_from_universe(universe_data):
    """universe_data: dict symbol -> DataFrame OHLCV crudo (>=2 anios diarios).
    Devuelve la lista completa de samples de todas las monedas."""
    all_samples = []
    per_symbol_df = {}
    for sym, df in tqdm(universe_data.items(), desc="Construyendo features/labels"):
        d = build_features(df)
        d["label"] = build_labels(d)
        d = d.dropna(subset=SCALAR_COLS).reset_index(drop=True)
        per_symbol_df[sym] = d
        all_samples.extend(make_samples(d, sym))
    print(f"[4/4] {len(all_samples)} ejemplos de entrenamiento construidos de {len(universe_data)} monedas.")
    return all_samples, per_symbol_df


def time_based_split(samples, val_frac=0.15):
    """Separa train/val por tiempo DENTRO de cada simbolo (los ultimos
    `val_frac` de cada moneda van a validacion) para evitar fuga de
    informacion futura hacia el pasado."""
    by_symbol = {}
    for s in samples:
        by_symbol.setdefault(s["symbol"], []).append(s)

    train, val = [], []
    for sym, items in by_symbol.items():
        items = sorted(items, key=lambda x: x["idx"])
        cut = int(len(items) * (1 - val_frac))
        train.extend(items[:cut])
        val.extend(items[cut:])
    return train, val


# ---------------------------------------------------------------------------
# 12) ENTRENAMIENTO
# ---------------------------------------------------------------------------

def train_model(train_samples, val_samples, epochs=25, batch_size=128, lr=1e-3,
                 device=DEVICE):
    scalar_stack = np.stack([s["scalar"] for s in train_samples])
    scalar_mean = scalar_stack.mean(axis=0)
    scalar_std = scalar_stack.std(axis=0)
    scalar_std[scalar_std == 0] = 1.0

    train_ds = TechoSueloDataset(train_samples, scalar_mean, scalar_std)
    val_ds = TechoSueloDataset(val_samples, scalar_mean, scalar_std)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    labels_arr = np.array([s["label"] for s in train_samples])
    class_counts = np.bincount(labels_arr, minlength=3).astype(np.float32)
    class_weights = (class_counts.sum() / (3 * np.clip(class_counts, 1, None)))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = MultiScaleTechoSueloNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for windows, scalar, label in train_dl:
            windows = {k: v.to(device) for k, v in windows.items()}
            scalar = scalar.to(device)
            label = label.to(device)

            logits, conf_logit = model(windows, scalar)
            loss_cls = ce_loss(logits, label)

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct_mask = (pred == label).float()
            loss_conf = bce_loss(conf_logit.squeeze(1), correct_mask)

            loss = loss_cls + 0.3 * loss_conf
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * label.size(0)

        model.eval()
        correct, n_total = 0, 0
        with torch.no_grad():
            for windows, scalar, label in val_dl:
                windows = {k: v.to(device) for k, v in windows.items()}
                scalar = scalar.to(device)
                label = label.to(device)
                logits, _ = model(windows, scalar)
                pred = logits.argmax(dim=1)
                correct += (pred == label).sum().item()
                n_total += label.size(0)
        val_acc = correct / max(n_total, 1)
        train_loss = total_loss / max(len(train_ds), 1)
        print(f"Epoch {epoch:02d}/{epochs} | loss={train_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    bundle = {
        "scalar_mean": scalar_mean,
        "scalar_std": scalar_std,
        "feature_cols": FEATURE_COLS,
        "scalar_cols": SCALAR_COLS,
        "scales": SCALES,
        "best_val_acc": best_val_acc,
    }
    return model, bundle


def save_bundle(model, bundle, path="techo_suelo_model.pt"):
    torch.save({"state_dict": model.state_dict(), "bundle": bundle}, path)
    print(f"Modelo guardado en {path}")


def load_bundle(path="techo_suelo_model.pt", device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    model = MultiScaleTechoSueloNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["bundle"]


# ---------------------------------------------------------------------------
# 13) ESCANER: EVALUAR CUALQUIER MONEDA DE OKX FUTURES
# ---------------------------------------------------------------------------

def prepare_last_window(df, bundle):
    """Toma el OHLCV crudo de una moneda cualquiera, calcula features y arma
    la ventana multi-escala del ULTIMO dia disponible (sin usar el futuro)."""
    d = build_features(df).dropna(subset=SCALAR_COLS).reset_index(drop=True)
    max_w = max(SCALES.values())
    if len(d) < max_w + 1:
        return None

    feat = d[FEATURE_COLS].values.astype(np.float32)
    scal = d[SCALAR_COLS].values.astype(np.float32)
    i = len(d) - 1  # ultimo dia cerrado

    windows = {}
    for name, w in SCALES.items():
        block = feat[i - w:i]
        if len(block) < w or not np.isfinite(block).all():
            return None
        windows[name] = block

    scalar = scal[i]
    if not np.isfinite(scalar).all():
        return None

    scalar_norm = (scalar - bundle["scalar_mean"]) / bundle["scalar_std"]
    return {
        "windows": windows,
        "scalar": scalar_norm,
        "last_close": float(d["close"].iloc[-1]),
        "ret_30": float(d["ret_30"].iloc[-1]) if not math.isnan(d["ret_30"].iloc[-1]) else None,
        "ret_7": float(d["ret_7"].iloc[-1]) if not math.isnan(d["ret_7"].iloc[-1]) else None,
        "date": d["ts"].iloc[-1],
    }


def score_symbol(model, bundle, df, device=DEVICE):
    prep = prepare_last_window(df, bundle)
    if prep is None:
        return None
    windows = {name: torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
               for name, w in prep["windows"].items()}
    scalar = torch.tensor(prep["scalar"], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, conf_logit = model(windows, scalar)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        conf = float(torch.sigmoid(conf_logit).cpu().numpy()[0, 0])
    return {
        "p_suelo": float(probs[0]),
        "p_neutral": float(probs[1]),
        "p_techo": float(probs[2]),
        "confidence": conf,
        "last_close": prep["last_close"],
        "ret_30": prep["ret_30"],
        "ret_7": prep["ret_7"],
        "date": prep["date"],
    }


def scan_okx_futures(ex, model, bundle, symbols=None, top_k=20,
                      history_days=140, device=DEVICE, verbose=True):
    """Evalua una lista de simbolos de OKX Futures (por defecto TODOS los
    perpetuos USDT activos) y devuelve dos tablas ordenadas:
      - top_techo: las que subieron sostenido y ya estan empezando a caer.
      - top_suelo: las que bajaron sostenido y ya estan empezando a subir.
    """
    if symbols is None:
        symbols = list_usdt_perpetuals(ex)

    rows = []
    it = tqdm(symbols, desc="Escaneando OKX Futures") if verbose else symbols
    for sym in it:
        try:
            raw = fetch_ohlcv_paginated(ex, sym, timeframe=TIMEFRAME, max_candles=history_days)
            if len(raw) < history_days * 0.8:
                continue
            df = ohlcv_to_df(raw)
            res = score_symbol(model, bundle, df, device=device)
            if res is None:
                continue
            res["symbol"] = sym
            rows.append(res)
        except Exception:
            continue
        time.sleep(REQUEST_SLEEP)

    table = pd.DataFrame(rows)
    if table.empty:
        print("No se pudo evaluar ninguna moneda (revisa conexion / simbolos).")
        return table, table

    top_techo = table.sort_values(["p_techo", "confidence"], ascending=False).head(top_k).reset_index(drop=True)
    top_suelo = table.sort_values(["p_suelo", "confidence"], ascending=False).head(top_k).reset_index(drop=True)

    cols = ["symbol", "last_close", "ret_30", "ret_7", "p_techo", "p_suelo", "p_neutral", "confidence", "date"]
    return top_techo[cols], top_suelo[cols]
