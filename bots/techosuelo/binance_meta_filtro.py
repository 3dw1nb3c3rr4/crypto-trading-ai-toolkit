"""
binance_meta_filtro.py
===================
"Cerebro 2": un segundo modelo, mas chico y especializado, que aprende sobre
la HISTORIA de lo que el Cerebro 1 (binance_topsuelo_bot) hubiera detectado dia
por dia, cuales de esas señales de techo/suelo realmente se cumplieron en
los siguientes hasta 3 dias y cuales fueron falsas alarmas.

Flujo completo:
  1) Se descarga (y se cachea en disco, para no repetir descargas) ~2 anios
     de velas diarias de un universo grande de monedas de Binance Futures.
  2) Para cada moneda y cada dia de su historia, se corre el Cerebro 1 usando
     SOLO los datos disponibles hasta ese dia (walk-forward, sin ver el
     futuro) y se guarda su prediccion (p_techo, p_suelo, confidence, etc.).
     Esto se hace en una sola pasada vectorizada por moneda (mucho mas rapido
     que llamar al modelo dia por dia en un loop de Python).
  3) Cada vez que el Cerebro 1 marco una señal (techo o suelo, no neutral),
     se mira que paso REALMENTE en los siguientes `META_HORIZON` dias
     (maximo 3): si el precio se movio lo suficiente a favor de la señal, se
     etiqueta como "exito" (1); si no, "fallo" (0).
  4) Con ese dataset de señales historicas + resultado real se entrena el
     Cerebro 2: un clasificador (HistGradientBoostingClassifier de
     scikit-learn, robusto para datos tabulares con relativamente pocas
     columnas) que aprende a distinguir señales del Cerebro 1 que SI se
     cumplen de las que NO.
  5) En produccion: el Cerebro 1 detecta candidatas (igual que antes); el
     Cerebro 2 les agrega un "prob_exito_3d" -- la probabilidad REAL,
     calibrada con historia, de que la señal se cumpla en <= 3 dias -- y
     solo se muestran/operan las que superan un umbral de confianza.

Nada de esto coloca ordenes reales ni es asesoria financiera -- es un
segundo filtro estadistico para reducir falsas señales del Cerebro 1.
"""

import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

import binance_topsuelo_bot as tm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
META_HORIZON       = 3      # el filtro mira MAXIMO 3 dias hacia adelante
TARGET_MOVE_LONG   = 0.02   # +2% de recorrido a favor para contar como "exito" LONG
TARGET_MOVE_SHORT  = 0.02   # -2% de recorrido a favor para contar como "exito" SHORT

# Features que ve el Cerebro 2: la salida del Cerebro 1 + contexto tecnico
# del dia de la señal (no incluye nada del futuro).
META_FEATURE_COLS = [
    "p_techo", "p_suelo", "p_neutral", "confidence",
    "ret_7", "ret_14", "ret_30", "ma_slope_30", "rsi_14",
    "vol_ratio", "bb_width", "bb_width_chg",
    "dist_high_30", "dist_low_30", "candle_pos",
]

DEVICE = tm.DEVICE


# ---------------------------------------------------------------------------
# 1) DESCARGA MASIVA CON CACHE EN DISCO
# ---------------------------------------------------------------------------

def download_universe_history(ex, symbols, days=tm.TARGET_DAYS_HISTORY,
                                cache_path="ohlcv_cache_2y.pkl", force_refresh=False):
    """Descarga (o reusa un cache en disco) velas diarias largas de una lista
    de simbolos. Guarda todo en un .pkl para que las siguientes corridas NO
    tengan que volver a descargar nada de Binance (esto es lo que mas tarda)."""
    data_dict = {}
    if os.path.exists(cache_path) and not force_refresh:
        with open(cache_path, "rb") as f:
            data_dict = pickle.load(f)
        print(f"Cache cargado: {len(data_dict)} monedas desde '{cache_path}'.")

    faltantes = [s for s in symbols if s not in data_dict]
    if not faltantes:
        print("Todas las monedas pedidas ya estan en cache.")
        return {s: data_dict[s] for s in symbols if s in data_dict}

    for n_done, sym in enumerate(tqdm(faltantes, desc=f"Descargando {len(faltantes)} monedas (~{days}d)"), 1):
        try:
            raw = tm.fetch_ohlcv_paginated(ex, sym, timeframe=tm.TIMEFRAME, max_candles=days)
            if len(raw) < days * 0.5:
                continue
            data_dict[sym] = tm.ohlcv_to_df(raw)
        except Exception:
            continue
        time.sleep(tm.REQUEST_SLEEP)
        if n_done % 20 == 0:  # guardado incremental por si se corta la sesion
            with open(cache_path, "wb") as f:
                pickle.dump(data_dict, f)

    with open(cache_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Cache final: {len(data_dict)} monedas -> guardado en '{cache_path}'.")
    return {s: data_dict[s] for s in symbols if s in data_dict}


# ---------------------------------------------------------------------------
# 2) CEREBRO 1 EN MODO "WALK-FORWARD" (todas las predicciones historicas de
#    una moneda en UNA sola pasada vectorizada, sin recalcular indicadores
#    dia por dia). Los indicadores de build_features solo miran hacia atras
#    (rolling / pct_change / shift positivo), por eso es seguro calcularlos
#    una sola vez sobre todo el DataFrame y despues cortar por dia -- da
#    exactamente el mismo resultado que recalcular sobre cada sub-DataFrame
#    truncado, pero muchisimo mas rapido.
# ---------------------------------------------------------------------------

def score_symbol_full_history(model, bundle, df, device=DEVICE, batch_size=512):
    """Corre el Cerebro 1 sobre TODOS los dias validos de la historia de una
    moneda. Devuelve un DataFrame con una fila por dia evaluado: OHLCV +
    features + prediccion del Cerebro 1 para ESE dia (sin fuga de futuro)."""
    d = tm.build_features(df)
    max_w = max(tm.SCALES.values())
    n = len(d)
    if n < max_w + META_HORIZON + 1:
        return None

    feat = d[tm.FEATURE_COLS].values.astype(np.float32)
    scal = d[tm.SCALAR_COLS].values.astype(np.float32)

    valid_idx = []
    for i in range(max_w, n):
        if not np.isfinite(scal[i]).all():
            continue
        ok = True
        for w in tm.SCALES.values():
            blk = feat[i - w:i]
            if not np.isfinite(blk).all():
                ok = False
                break
        if ok:
            valid_idx.append(i)
    if not valid_idx:
        return None

    windows_all = {name: np.stack([feat[i - w:i] for i in valid_idx])
                   for name, w in tm.SCALES.items()}
    scalar_all = np.stack([scal[i] for i in valid_idx])
    scalar_norm = (scalar_all - bundle["scalar_mean"]) / bundle["scalar_std"]

    probs_chunks, conf_chunks = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(valid_idx), batch_size):
            e = s + batch_size
            wb = {name: torch.tensor(windows_all[name][s:e], dtype=torch.float32).to(device)
                  for name in tm.SCALES}
            sb = torch.tensor(scalar_norm[s:e], dtype=torch.float32).to(device)
            logits, conf_logit = model(wb, sb)
            probs_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
            conf_chunks.append(torch.sigmoid(conf_logit).cpu().numpy()[:, 0])
    probs = np.concatenate(probs_chunks, axis=0)
    conf = np.concatenate(conf_chunks, axis=0)

    out = d.iloc[valid_idx].reset_index(drop=False).rename(columns={"index": "orig_idx"})
    out["p_suelo"] = probs[:, 0]
    out["p_neutral"] = probs[:, 1]
    out["p_techo"] = probs[:, 2]
    out["confidence"] = conf
    return out


# ---------------------------------------------------------------------------
# 3) ETIQUETAS REALES: que paso en los siguientes <=3 dias de cada señal
# ---------------------------------------------------------------------------

def build_meta_samples_for_symbol(scored_df, symbol, horizon=META_HORIZON,
                                   target_long=TARGET_MOVE_LONG,
                                   target_short=TARGET_MOVE_SHORT):
    """A partir de las predicciones historicas del Cerebro 1 para UNA moneda,
    arma ejemplos (features del dia + lado de la señal + si se cumplio de
    verdad en los siguientes `horizon` dias) para entrenar el Cerebro 2."""
    samples = []
    highs = scored_df["high"].values
    lows = scored_df["low"].values
    closes = scored_df["close"].values
    n = len(scored_df)

    for i in range(n - horizon):
        row = scored_df.iloc[i]
        p_techo, p_suelo, p_neutral = row["p_techo"], row["p_suelo"], row["p_neutral"]

        if p_techo > p_neutral and p_techo >= p_suelo:
            side = "SHORT"
        elif p_suelo > p_neutral and p_suelo > p_techo:
            side = "LONG"
        else:
            continue  # el Cerebro 1 no marco señal ese dia, no hay nada que filtrar

        entry = closes[i]
        fut_highs = highs[i + 1:i + 1 + horizon]
        fut_lows = lows[i + 1:i + 1 + horizon]
        if len(fut_highs) < horizon:
            continue

        if side == "SHORT":
            max_favor = (entry - fut_lows.min()) / entry     # cuanto llego a caer
            label = int(max_favor >= target_short)
        else:
            max_favor = (fut_highs.max() - entry) / entry    # cuanto llego a subir
            label = int(max_favor >= target_long)

        samples.append({
            "symbol": symbol,
            "side": side,
            "features": [float(row[c]) for c in META_FEATURE_COLS],
            "label": label,
        })
    return samples


def build_meta_dataset(model, bundle, data_dict, horizon=META_HORIZON,
                        target_long=TARGET_MOVE_LONG, target_short=TARGET_MOVE_SHORT,
                        device=DEVICE):
    """Genera el dataset completo del Cerebro 2 recorriendo TODAS las monedas
    de `data_dict` en modo walk-forward (sin fuga de informacion futura)."""
    all_samples = []
    for sym, df in tqdm(data_dict.items(), desc="Generando dataset del Cerebro 2"):
        scored = score_symbol_full_history(model, bundle, df, device=device)
        if scored is None or len(scored) < horizon + 10:
            continue
        all_samples.extend(build_meta_samples_for_symbol(
            scored, sym, horizon=horizon, target_long=target_long, target_short=target_short))

    n = len(all_samples)
    n_pos = sum(s["label"] for s in all_samples)
    n_long = sum(1 for s in all_samples if s["side"] == "LONG")
    n_short = n - n_long
    print(f"Dataset del Cerebro 2: {n} señales historicas de {len(data_dict)} monedas "
          f"({n_long} LONG / {n_short} SHORT).")
    print(f"  Se cumplieron en <= {horizon} dias: {n_pos} ({100*n_pos/max(n,1):.1f}%)")
    return all_samples


# ---------------------------------------------------------------------------
# 4) ENTRENAR EL CEREBRO 2 (clasificador de "esta señal se va a cumplir?")
# ---------------------------------------------------------------------------

def train_meta_filter(meta_samples, val_frac=0.15, random_state=42):
    X = np.array([s["features"] for s in meta_samples], dtype=np.float32)
    y = np.array([s["label"] for s in meta_samples], dtype=np.int64)
    side_long = np.array([1.0 if s["side"] == "LONG" else 0.0 for s in meta_samples],
                          dtype=np.float32)
    X_full = np.column_stack([X, side_long])  # ultima columna = lado (1=LONG, 0=SHORT)

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y, test_size=val_frac, random_state=random_state, stratify=y)

    clf = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=random_state,
    )
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    acc = (val_preds == y_val).mean()
    base_rate = y_val.mean()
    print(f"Cerebro 2 (filtro) -> val_acc={acc:.3f} | tasa base de exito={base_rate:.3f} "
          f"| n_train={len(y_train)} | n_val={len(y_val)}")
    if acc <= base_rate + 0.02:
        print("  ⚠️ El filtro apenas mejora sobre la tasa base: prueba con mas monedas/mas "
              "historia, o ajusta TARGET_MOVE_LONG/TARGET_MOVE_SHORT en este modulo.")
    return clf


def save_meta_filter(clf, path="meta_filtro_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Cerebro 2 guardado en '{path}'.")


def load_meta_filter(path="meta_filtro_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 5) ESCANER COMBINADO: CEREBRO 1 (detecta) + CEREBRO 2 (confirma / filtra)
# ---------------------------------------------------------------------------

def meta_filter_score(clf, feature_row, side):
    feats = [float(feature_row[c]) for c in META_FEATURE_COLS]
    feats.append(1.0 if side == "LONG" else 0.0)
    X = np.array(feats, dtype=np.float32).reshape(1, -1)
    return float(clf.predict_proba(X)[0, 1])


def scan_with_meta_filter(ex, model, bundle, clf, symbols=None, top_k=20,
                           history_days=140, min_meta_prob=0.55, device=DEVICE):
    """Escanea monedas de BINANCE FUTURES con el Cerebro 1 y, para cada señal (techo o suelo),
    le pide al Cerebro 2 una probabilidad real de que se cumpla en <=3 dias.
    Solo deja pasar (y ordena) las que superan `min_meta_prob`. Devuelve
    (top_short, top_long)."""
    if symbols is None:
        symbols = tm.list_usdt_perpetuals(ex)

    rows = []
    for sym in tqdm(symbols, desc="Cerebro 1 + Cerebro 2"):
        try:
            raw = tm.fetch_ohlcv_paginated(ex, sym, timeframe=tm.TIMEFRAME, max_candles=history_days)
            if len(raw) < history_days * 0.8:
                continue
            df = tm.ohlcv_to_df(raw)
            res = tm.score_symbol(model, bundle, df, device=device)
            if res is None:
                continue

            if res["p_techo"] > res["p_neutral"] and res["p_techo"] >= res["p_suelo"]:
                side = "SHORT"
            elif res["p_suelo"] > res["p_neutral"] and res["p_suelo"] > res["p_techo"]:
                side = "LONG"
            else:
                continue  # Cerebro 1 no marca señal clara -> nada que confirmar

            last = tm.build_features(df).iloc[-1]
            feature_row = {**res, **{c: last[c] for c in META_FEATURE_COLS if c not in res}}
            prob_exito = meta_filter_score(clf, feature_row, side)

            res["symbol"] = sym
            res["lado"] = side
            res["prob_exito_3d"] = prob_exito
            if prob_exito >= min_meta_prob:
                rows.append(res)
        except Exception:
            continue
        time.sleep(tm.REQUEST_SLEEP)

    table = pd.DataFrame(rows)
    if table.empty:
        print("Ninguna moneda supero el umbral del Cerebro 2 con la configuracion actual "
              f"(min_meta_prob={min_meta_prob}). Prueba bajarlo un poco.")
        return table, table

    cols = ["symbol", "lado", "last_close", "ret_30", "ret_7",
            "p_techo", "p_suelo", "confidence", "prob_exito_3d", "date"]
    top_short = (table[table["lado"] == "SHORT"]
                 .sort_values("prob_exito_3d", ascending=False).head(top_k).reset_index(drop=True))
    top_long = (table[table["lado"] == "LONG"]
                .sort_values("prob_exito_3d", ascending=False).head(top_k).reset_index(drop=True))
    return top_short[cols], top_long[cols]