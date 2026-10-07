# ===== CELDA ÚNICA: funding + open interest para el modelo =====
#   Etapa 1: OKX por API (rápida; el historial que OKX entregue se informa al final).
#   Etapa 2: archivos públicos de Binance (data.binance.vision): funding mensual desde 2020 y OI diario desde 2021-12.
#            Primero comprueba que existen y se leen; puede tardar 1-2 h; si se corta, vuelve a ejecutar y reanuda.
# (antes de ejecutar este archivo en Colab: !pip -q install ccxt)
import os, sys, time, pickle, urllib.request
import pandas as pd

LIMIT = None                 # número de símbolos para una prueba rápida (p.ej. 10); None = todos
START_OI = "2021-12-01"      # inicio del OI diario de Binance
START_FUNDING = "2020-01-01"
DO_METRICS = True            # etapa 2
RAW = "https://raw.githubusercontent.com/3dw1nb3c3rr4/crypto-trading-ai-toolkit/claude/hola-d7y87v/backtesting/"

OUT_DIR = "/content/derivados"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/derivados"
except Exception:
    pass
CODE_DIR = "/content/derivados_code"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CODE_DIR, exist_ok=True)
for f in ("download_derivs.py", "download_binance_metrics.py", "universe_symbols.txt"):
    urllib.request.urlretrieve(RAW + f, os.path.join(CODE_DIR, f))
sys.path.insert(0, CODE_DIR)
import download_derivs as dd
import download_binance_metrics as dm

symbols = [x.strip() for x in open(os.path.join(CODE_DIR, "universe_symbols.txt")) if x.strip()]
MAYORES = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC", "DOT", "TRX", "BCH", "NEAR", "ATOM"]
symbols.sort(key=lambda s: (MAYORES.index(s.split("/")[0]) if s.split("/")[0] in MAYORES else 99, s))
if LIMIT:
    symbols = symbols[:LIMIT]                       # la prueba rápida usa primero los símbolos más líquidos


def cobertura(nombre, d, col):
    filas = []
    for s, df in d.items():
        if len(df):
            t = pd.to_datetime(df["ts"], unit="ms", utc=True)
            filas.append((s, len(df), t.iloc[0].date(), t.iloc[-1].date(), (t.iloc[-1] - t.iloc[0]).days))
    if not filas:
        print(f"  {nombre}: SIN DATOS")
        return
    r = pd.DataFrame(filas, columns=["sym", "filas", "desde", "hasta", "dias"])
    print(f"  {nombre}: {len(r)} símbolos | historial mediano {int(r.dias.median())} días (mín {r.dias.min()}, máx {r.dias.max()}) | "
          f"más antiguo {r.desde.min()} | más reciente {r.hasta.max()}")


t0 = time.time()
print(f"== Etapa 1: OKX por API ({len(symbols)} símbolos) ==")
okx = dd.download("okx", symbols, 2200, log=lambda *a: None)
pickle.dump(okx, open(os.path.join(OUT_DIR, "derivs_okx.pkl"), "wb"))
print(f"[{time.time()-t0:.0f}s] errores: {len(okx['meta']['errors'])}")
cobertura("OKX funding", okx["funding"], "rate"); cobertura("OKX open interest", okx["oi"], "oi_amt")

files = ["derivs_okx.pkl"]
if DO_METRICS:
    print("\n== Etapa 2: archivos públicos de Binance ==")
    ok_oi, ok_f = dm.probe(), dm.probe_funding()
    data = {"funding": {}, "oi": {}, "meta": {"source": "data.binance.vision"}}
    cache = os.path.join(OUT_DIR, "metrics_cache")
    end = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if ok_f:
        data["funding"] = dm.download_funding(symbols, START_FUNDING, end, cache)
        print(f"[{time.time()-t0:.0f}s] funding Binance listo")
    if ok_oi:
        data["oi"] = dm.download(symbols, START_OI, end, cache)["oi"]
        print(f"[{time.time()-t0:.0f}s] OI Binance listo")
    pickle.dump(data, open(os.path.join(OUT_DIR, "derivs_metrics.pkl"), "wb"))
    cobertura("Binance funding", data["funding"], "rate"); cobertura("Binance open interest", data["oi"], "oi_amt")
    files.append("derivs_metrics.pkl")

print(f"\nListo en {time.time()-t0:.0f}s. Archivos en {OUT_DIR}: {files}")
try:
    from google.colab import files as cfiles
    for f in files:
        cfiles.download(os.path.join(OUT_DIR, f))
except Exception:
    print("Descárgalos manualmente desde el panel de archivos de Colab.")
