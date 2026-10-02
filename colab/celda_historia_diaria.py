# ===== CELDA ÚNICA: velas DIARIAS largas (varios años) de OKX para los perpetuos USDT más líquidos =====
# (antes de ejecutar este archivo en Colab: !pip -q install ccxt)
import os, time, pickle
import ccxt, pandas as pd

YEARS = 6            # años de historia (cada contrato empieza en su fecha de listado)
TOP_N = 150          # cuántos símbolos (los de mayor volumen actual). Ojo: el universo "de hoy" tiene sesgo de supervivencia
# TOP_N = 10; YEARS = 1    # <- descomenta para una prueba rápida

OUT_DIR = "/content/historia_diaria"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/historia_diaria"
except Exception:
    pass
os.makedirs(OUT_DIR, exist_ok=True)

DAY_MS = 86_400_000


def _retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.RequestTimeout):
            time.sleep(2 ** k)
    raise RuntimeError("demasiados fallos de red/rate limit")


def top_symbols(ex, n):
    ex.load_markets()
    swaps = [s for s, m in ex.markets.items() if m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("active")]
    try:
        tk = _retry(lambda: ex.fetch_tickers(swaps))
        swaps.sort(key=lambda s: -float((tk.get(s) or {}).get("quoteVolume") or 0))
    except Exception as e:
        print("no se pudo ordenar por volumen, se usa el orden del exchange:", type(e).__name__)
    return swaps[:n]


def fetch_daily(ex, symbol, years):
    """Pagina hacia adelante. Si un tramo viene vacío (el contrato aún no existía) avanza al siguiente."""
    now = ex.milliseconds()
    since = now - int(years * 365.25) * DAY_MS
    rows, last, empties = [], None, 0
    while since < now:
        batch = _retry(lambda: ex.fetch_ohlcv(symbol, "1d", since=since, limit=100))
        if not batch:
            since += 100 * DAY_MS
            empties += 1
            if empties > 80:
                break
            continue
        rows += batch
        nl = batch[-1][0]
        if last is not None and nl <= last:
            break
        last, since = nl, nl + DAY_MS
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df[df["ts"] + DAY_MS <= now].reset_index(drop=True)          # solo velas cerradas
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


ex = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
symbols = top_symbols(ex, TOP_N)
print(f"{len(symbols)} símbolos; descargando hasta {YEARS} años de velas diarias ...")
data, parts = {}, os.path.join(OUT_DIR, "parts")
os.makedirs(parts, exist_ok=True)
for i, s in enumerate(symbols, 1):
    path = os.path.join(parts, s.replace("/", "_").replace(":", "_") + f"_{YEARS}y.pkl")
    if os.path.exists(path):
        data[s] = pickle.load(open(path, "rb"))
        continue
    try:
        df = fetch_daily(ex, s, YEARS)
    except Exception as e:
        print(f"[{i}/{len(symbols)}] {s}: ERROR {type(e).__name__}: {str(e)[:100]}")
        continue
    if len(df) < 60:
        print(f"[{i}/{len(symbols)}] {s}: solo {len(df)} velas, se omite")
        continue
    pickle.dump(df, open(path, "wb"))
    data[s] = df
    if i % 10 == 0 or i == len(symbols):
        print(f"[{i}/{len(symbols)}] {s}: {len(df)} velas desde {df['ts'].iloc[0]:%Y-%m-%d}")

resumen = pd.DataFrame({s: dict(velas=len(d), desde=d["ts"].iloc[0].date(), hasta=d["ts"].iloc[-1].date()) for s, d in data.items()}).T
print(f"\n{len(data)} símbolos. Historia: mediana {int(resumen.velas.median())} velas; con >= 1500 velas: {(resumen.velas >= 1500).sum()}; "
      f"más antiguo: {min(resumen.desde)}")
out = os.path.join(OUT_DIR, "ohlcv_daily_long.pkl")
pickle.dump(data, open(out, "wb"))
print(f"Archivo listo: {out} ({os.path.getsize(out)/1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(out)
except Exception:
    print("Descárgalo manualmente desde el panel de archivos de Colab:", out)
