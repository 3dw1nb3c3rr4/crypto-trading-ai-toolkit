"""Entrena el modelo final del bot cross-sectional diario con todo el historial disponible.

Uso: python bots/xs_daily/train.py                       (variante D, la original)
     python bots/xs_daily/train.py --variant FO          (base + Alpha158 + funding + open interest)
Para FO usa el historial de derivados de backtesting/derivados/derivs_metrics.pkl (descargar_todo_pc.py) o --derivs.
Opciones: [--data ...] [--until 2026-07-01] [--out ...]
"""
import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xs_core  # noqa: E402
from engine import load_universe  # noqa: E402


def model_path(variant):
    return os.path.join(HERE, f"model_xs_{variant}.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=next((p for p in (os.path.join(xs_core.ROOT, "data", "ohlcv_daily_long.pkl"),) if os.path.exists(p)),
                                           os.path.join(xs_core.ROOT, "data", "trading_history", "ohlcv_cache_2y.pkl")),
                    help="por defecto el historial largo de 6 años si existe (data/ohlcv_daily_long.pkl)")
    ap.add_argument("--variant", default="D", choices=xs_core.VARIANTS)
    ap.add_argument("--derivs", nargs="+", default=None, help="pickles de funding/OI (por defecto los de backtesting/derivados)")
    ap.add_argument("--until", default=None, help="entrenar solo con velas anteriores a esta fecha (para pruebas fuera de muestra)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    uni = load_universe(a.data, min_bars=150)
    if a.until:
        uni = {s: df[df.ts < a.until].reset_index(drop=True) for s, df in uni.items()}
        uni = {s: df for s, df in uni.items() if len(df) >= 150}
    derivs = None
    if a.variant == "FO":
        import derivs_live
        derivs = derivs_live.load_seed(a.derivs) if a.derivs else derivs_live.load()   # historial + lo acumulado en vivo
        if not derivs[0] and not derivs[1]:
            sys.exit("FO necesita el historial de funding/OI: ejecuta backtesting/descargar_todo_pc.py o pasa --derivs ruta.pkl")
        print(f"derivados: funding {len(derivs[0])} símbolos, OI {len(derivs[1])} símbolos")
    b = xs_core.train(uni, a.variant, derivs)
    out = a.out or model_path(a.variant)
    pickle.dump(b, open(out, "wb"))
    print(f"modelo {a.variant} guardado en {out} | símbolos={len(b['symbols'])} factores={len(b['features'])} filas={b['n_rows']} entrenado hasta {b['trained_until']}")
