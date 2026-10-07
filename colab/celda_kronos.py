# ===== CELDA ÚNICA (Colab con GPU: Entorno de ejecución > Cambiar tipo > T4) =====
# Genera pronósticos de Kronos (https://github.com/shiyu-coder/Kronos) a 7 días para cada símbolo y día, y los guarda en
# kronos_forecasts.pkl. Necesita tu ohlcv_daily_long.pkl (el de colab/celda_historia_diaria.py).
# Pasos: 1) sube ohlcv_daily_long.pkl a /content (o ponlo en Drive y cambia DATA)  2) pega y ejecuta esta celda
#        3) descarga kronos_forecasts.pkl (se descarga solo al final) y pásamelo.
import os, sys, subprocess, pickle, time
if not os.path.exists("/content/Kronos"):
    subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos", "/content/Kronos"], check=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "huggingface_hub", "einops", "safetensors"], check=False)
sys.path.insert(0, "/content/Kronos")
import numpy as np, pandas as pd, torch
from model import Kronos, KronosTokenizer, KronosPredictor

DATA = "/content/ohlcv_daily_long.pkl"
OUT = "/content/kronos_forecasts.pkl"
MODEL = "NeoQuasar/Kronos-small"          # small (24.7M, contexto 512) = equilibrio; "Kronos-base" (102M) es más lento
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
START = "2024-01-01"                      # primera fecha a pronosticar. Cuanto más atrás, más tarda
STEP = 1                                  # 1 = todos los días (lo ideal); 3 = cada 3 días (3x más rápido, menos datos de entrenamiento)
TOP = 60                                  # solo los TOP símbolos por liquidez (60 va bien en una T4; 135 = todo)
CTX, H, BATCH = 400, 7, 64
PROBE = True                              # primero mide la velocidad con 128 pronósticos y muestra el tiempo total estimado

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("dispositivo:", dev, "(en CPU es inviable)" if dev == "cpu" else "")
raw = pickle.load(open(DATA, "rb"))
uni = {}
for s, df in raw.items():
    df = df.copy(); df["ts"] = pd.to_datetime(df["ts"], utc=True); uni[s] = df.sort_values("ts").reset_index(drop=True)
liq = {s: float((d["close"] * d["volume"]).tail(60).mean()) for s, d in uni.items() if len(d) > 200}
keep = sorted(liq, key=liq.get, reverse=True)[:TOP]
uni = {s: uni[s] for s in keep}
print("símbolos:", len(uni))

# --- las funciones de abajo son las mismas de backtesting/kronos_features.py (copiadas para que la celda sea autónoma) ---
FEATS = ("kr_ret", "kr_vol", "kr_up", "kr_hi", "kr_lo")
def _window(df, i, ctx):
    w = df.iloc[max(0, i - ctx + 1):i + 1]
    x = w[["open", "high", "low", "close", "volume"]].astype("float64").copy()
    x["amount"] = (w["close"] * w["volume"]).astype("float64").to_numpy()
    return x.reset_index(drop=True), pd.Series(pd.to_datetime(w["ts"], utc=True).dt.tz_localize(None).to_numpy())

def make_forecasts(uni, predictor, step, start, horizon, ctx, batch, min_ctx=120, limit=None, **kw):
    jobs, st = [], pd.Timestamp(start, tz="UTC")
    for s, df in uni.items():
        ts = df["ts"]
        jobs += [(s, i) for i in range(min_ctx, len(df), step) if ts.iloc[i] >= st]
    if limit: jobs = jobs[:limit]
    rows, t0 = [], time.time()
    for b in range(0, len(jobs), batch):
        groups = {}
        for s, i in jobs[b:b + batch]:
            groups.setdefault(min(i + 1, ctx), []).append((s, i))
        for L, items in groups.items():
            xs, xts, yts = [], [], []
            for s, i in items:
                x, xt = _window(uni[s], i, ctx)
                xs.append(x); xts.append(xt); yts.append(pd.Series(pd.date_range(xt.iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="1D")))
            preds = predictor.predict_batch(df_list=xs, x_timestamp_list=xts, y_timestamp_list=yts, pred_len=horizon, **kw)
            for (s, i), x, p in zip(items, xs, preds):
                c0 = float(x["close"].iloc[-1]); pc = p["close"].to_numpy(dtype="float64")
                lr = np.diff(np.log(np.r_[c0, pc]))
                rows.append((s, uni[s]["ts"].iloc[i], pc[-1] / c0 - 1, float(lr.std()), float((lr > 0).mean()),
                             float(p["high"].max() / c0 - 1), float(p["low"].min() / c0 - 1)))
        if (b // batch) % 25 == 0:
            el = time.time() - t0
            print(f"  {min(b + batch, len(jobs))}/{len(jobs)} | {el/60:.1f} min | falta ~{el/(b+batch)*(len(jobs)-b-batch)/60:.0f} min", flush=True)
    out = pd.DataFrame(rows, columns=["sym", "ts", *FEATS]); out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out, len(jobs), time.time() - t0

tok = KronosTokenizer.from_pretrained(TOKENIZER)
mdl = Kronos.from_pretrained(MODEL)
pred = KronosPredictor(mdl, tok, device=dev, max_context=512)
kw = dict(T=1.0, top_p=0.9, sample_count=1, verbose=False)

if PROBE:
    _, n_probe, secs = make_forecasts(uni, pred, STEP, START, H, CTX, BATCH, limit=128, **kw)
    n_all = sum(len(range(120, len(d), STEP)) for d in uni.values())
    print(f"128 pronósticos en {secs:.0f}s -> total ~{n_all} pronósticos ≈ {n_all/128*secs/60:.0f} min. Si es demasiado: sube STEP o baja TOP/usa START más reciente.")
fc, n, secs = make_forecasts(uni, pred, STEP, START, H, CTX, BATCH, **kw)
fc.to_pickle(OUT)
print("listo:", len(fc), "pronósticos ->", OUT)
try:
    from google.colab import files; files.download(OUT)
except Exception:
    pass
