# ===== CELDA ÚNICA: funding (OKX y, si se puede, Binance) + velas spot 1h de OKX =====
# (antes de ejecutar este archivo en Colab: !pip -q install ccxt)
import os, time, pickle
import ccxt, pandas as pd

DAYS = 365          # OKX solo entrega ~3 meses de funding; Binance entrega todo (pero suele bloquear IPs de EE.UU.)
SYMBOLS = [f"{b}/USDT:USDT" for b in "BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC DOT TRX BCH NEAR ATOM UNI APT ARB OP SUI".split()]
# SYMBOLS = SYMBOLS[:3]; DAYS = 30    # <- descomenta para una prueba rápida

OUT_DIR = "/content/funding_okx"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = "/content/drive/MyDrive/funding_okx"
except Exception:
    pass
os.makedirs(OUT_DIR, exist_ok=True)

def _fetch_with_retry(ex, symbol, tf, since, limit, tries=5):
    for k in range(tries):
        try:
            return ex.fetch_ohlcv(symbol, tf, since=since, limit=limit)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.RequestTimeout):
            time.sleep(2 ** k)
    raise RuntimeError(f"{symbol}: demasiados fallos de red/rate limit")


def fetch_symbol(ex, symbol, days, tf="5m", limit=100, log=print):
    """Pagina hacia adelante desde now-days; descarta la vela en curso; ordena y quita duplicados."""
    tf_ms = ex.parse_timeframe(tf) * 1000
    now = ex.milliseconds()
    since = now - days * 86_400_000
    rows, last_ts, calls = [], None, 0
    while since < now:
        batch = _fetch_with_retry(ex, symbol, tf, since, limit)
        calls += 1
        if not batch:
            break
        rows += batch
        new_last = batch[-1][0]
        if last_ts is not None and new_last <= last_ts:
            break                               # sin progreso: fin de datos
        last_ts, since = new_last, new_last + tf_ms
        if calls % 100 == 0:
            log(f"  {symbol}: {len(rows)} velas, hasta {pd.to_datetime(new_last, unit='ms', utc=True):%Y-%m-%d}")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[df["ts"] + tf_ms <= now].reset_index(drop=True)   # solo velas cerradas


def quality(df, tf_ms=300_000):
    """Resumen de calidad: filas, rango y huecos (velas faltantes)."""
    if df.empty:
        return dict(rows=0, first=None, last=None, missing=0)
    expected = (df["ts"].iloc[-1] - df["ts"].iloc[0]) // tf_ms + 1
    return dict(rows=len(df), first=pd.to_datetime(df["ts"].iloc[0], unit="ms", utc=True),
                last=pd.to_datetime(df["ts"].iloc[-1], unit="ms", utc=True), missing=int(expected - len(df)))

def _retry(fn, tries=5):
    for k in range(tries):
        try:
            return fn()
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.RequestTimeout):
            time.sleep(2 ** k)
    raise RuntimeError("demasiados fallos de red/rate limit")


def fetch_funding(ex, symbol, days):
    """OKX: pagina hacia atrás con 'until' (devuelve las 100 más recientes anteriores a until, máx ~3 meses).
    Otros (Binance): pagina hacia adelante con 'since' (hasta 1000 por llamada)."""
    now = ex.milliseconds()
    start = now - days * 86_400_000
    rows = {}
    if ex.id == "okx":
        until = now
        while True:
            batch = _retry(lambda: ex.fetch_funding_rate_history(symbol, limit=100, params={"until": until}))
            if not batch:
                break
            for b in batch:
                rows[b["timestamp"]] = float(b["fundingRate"])
            oldest = min(b["timestamp"] for b in batch)
            if oldest >= until or oldest <= start:
                break
            until = oldest
    else:
        since, last = start, None
        while since < now:
            batch = _retry(lambda: ex.fetch_funding_rate_history(symbol, since=since, limit=1000))
            if not batch:
                break
            for b in batch:
                rows[b["timestamp"]] = float(b["fundingRate"])
            new_last = max(b["timestamp"] for b in batch)
            if last is not None and new_last <= last:
                break
            last, since = new_last, new_last + 1
    df = pd.DataFrame(sorted(rows.items()), columns=["ts", "rate"])
    return df[df["ts"] >= start].reset_index(drop=True)


def funding_quality(df):
    if df.empty:
        return dict(n=0, first=None, last=None, per_day=0.0, mean_8h=float("nan"))
    span_days = max((df["ts"].iloc[-1] - df["ts"].iloc[0]) / 86_400_000, 1e-9)
    return dict(n=len(df), first=pd.to_datetime(df["ts"].iloc[0], unit="ms", utc=True),
                last=pd.to_datetime(df["ts"].iloc[-1], unit="ms", utc=True),
                per_day=round((len(df) - 1) / span_days, 2), mean_8h=float(df["rate"].mean()))


def download_funding(exchange_id, symbols, days, log=print):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ex.load_markets()
    out = {}
    for s in symbols:
        if s not in ex.markets:
            log(f"{exchange_id} {s}: no existe, se omite")
            continue
        try:
            df = fetch_funding(ex, s, days)
        except Exception as e:  # un símbolo con error no detiene el resto
            log(f"{exchange_id} {s}: ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        out[s] = df
        q = funding_quality(df)
        log(f"{exchange_id} {s}: {q['n']} liquidaciones, {q['first']:%Y-%m-%d} -> {q['last']:%Y-%m-%d}, "
            f"{q['per_day']}/día, media {q['mean_8h']:.5%}")
    return out


def download_spot_1h(symbols, days, fetch_symbol, log=print):
    """Velas spot de 1h de OKX (para el P&L de la base). fetch_symbol viene de download_okx_5m.py."""
    ex = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    ex.load_markets()
    out = {}
    for perp in symbols:
        spot = perp.split(":")[0]
        if spot not in ex.markets:
            log(f"spot {spot}: no existe, se omite")
            continue
        out[perp] = fetch_symbol(ex, spot, days, tf="1h", limit=100, log=log)
        log(f"spot {spot}: {len(out[perp])} velas 1h")
    return out


result = {}
print("== Funding OKX ==")
result["okx_funding"] = download_funding("okx", SYMBOLS, DAYS)
pickle.dump(result, open(os.path.join(OUT_DIR, "funding_data.pkl"), "wb"))      # guardado parcial

print("\n== Funding Binance (puede fallar desde Colab por bloqueo regional; entonces hazlo desde tu PC) ==")
try:
    result["binance_funding"] = download_funding("binanceusdm", SYMBOLS, DAYS)
except Exception as e:
    print("Binance no disponible aquí:", type(e).__name__, str(e)[:150])
    result["binance_funding"] = {}
pickle.dump(result, open(os.path.join(OUT_DIR, "funding_data.pkl"), "wb"))

print("\n== Velas spot 1h de OKX (para la base spot-perp; tarda varios minutos) ==")
result["okx_spot_1h"] = download_spot_1h(SYMBOLS, DAYS, fetch_symbol, log=lambda *a: None)
for s, df in result["okx_spot_1h"].items():
    print(f"spot {s.split(':')[0]}: {len(df)} velas 1h")

out = os.path.join(OUT_DIR, "funding_data.pkl")
pickle.dump(result, open(out, "wb"))
print(f"\nArchivo listo: {out} ({os.path.getsize(out)/1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(out)
except Exception:
    print("Descárgalo manualmente desde el panel de archivos de Colab:", out)
