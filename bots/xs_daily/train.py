"""Entrena el modelo final del bot cross-sectional diario con todo el historial disponible.
Uso: python bots/xs_daily/train.py [--data data/trading_history/ohlcv_cache_2y.pkl] [--until 2026-07-01] [--out bots/xs_daily/model_xs_D.pkl]
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xs_core  # noqa: E402
from engine import load_universe  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(xs_core.ROOT, "data", "trading_history", "ohlcv_cache_2y.pkl"))
    ap.add_argument("--until", default=None, help="entrenar solo con velas anteriores a esta fecha (para pruebas fuera de muestra)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_xs_D.pkl"))
    a = ap.parse_args()
    uni = load_universe(a.data, min_bars=150)
    if a.until:
        uni = {s: df[df.ts < a.until].reset_index(drop=True) for s, df in uni.items()}
        uni = {s: df for s, df in uni.items() if len(df) >= 150}
    b = xs_core.train(uni)
    pickle.dump(b, open(a.out, "wb"))
    print(f"modelo guardado en {a.out} | símbolos={len(b['symbols'])} filas={b['n_rows']} entrenado hasta {b['trained_until']}")
