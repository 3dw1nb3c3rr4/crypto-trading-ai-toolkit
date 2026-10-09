"""Descarga el historial diario de open interest (OI) desde los archivos públicos de Binance (data.binance.vision, carpeta
futures/um/daily/metrics). Cada archivo trae una foto cada 5 minutos del día: se guarda la primera (00:00 UTC) como foto del OI.

También baja el funding mensual (carpeta futures/um/monthly/fundingRate), con historial desde 2019-2020.
Primero comprueba que los archivos existen:   python download_binance_metrics.py --probe
Luego descarga:  python download_binance_metrics.py --symbols-file universe_symbols.txt --start 2021-12-01 --what both --out derivs_metrics.pkl
Reanuda solo (cache por símbolo en --cache). Salida compatible con ml_daily_xs_v2.py --derivs.
Probado con respuestas simuladas; la existencia y el formato reales los verifica --probe (sin eso, no se asume nada).
"""
import argparse
import time
import concurrent.futures as cf
import datetime as dt
import io
import os
import pickle
import urllib.error
import urllib.request
import zipfile

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
FUNDING_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"


def url_for(binance_symbol, day):
    return f"{BASE}/{binance_symbol}/{binance_symbol}-metrics-{day:%Y-%m-%d}.zip"


def to_ms(t):
    """Serie de tiempos -> milisegundos desde epoch (int64), robusta a: números en ms o s, texto, y resoluciones ns/us/ms de pandas."""
    if pd.api.types.is_numeric_dtype(t):
        v = t.astype("int64")
        return v * 1000 if len(v) and v.max() < 10**11 else v
    d = pd.to_datetime(t, utc=True)
    return ((d - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).astype("int64")


def funding_url(binance_symbol, year, month):
    return f"{FUNDING_BASE}/{binance_symbol}/{binance_symbol}-fundingRate-{year:04d}-{month:02d}.zip"


def parse_funding_zip(blob):
    """Archivo mensual de funding -> DataFrame[ts, rate]. None si el formato no es el esperado."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    tcol = next((c for c in df.columns if "time" in c.lower()), None)
    rcol = next((c for c in df.columns if "rate" in c.lower()), None)
    if tcol is None or rcol is None:
        return None
    return pd.DataFrame({"ts": to_ms(df[tcol]), "rate": df[rcol].astype(float)})


def http_get(url, timeout=20, tries=3):
    """Devuelve (codigo, bytes|None). 404 = el archivo no existe (normal antes de la fecha de inicio del símbolo).
    Errores de red / 429 / 5xx se reintentan; si siguen fallando devuelve (0, None) y el llamador NO guarda caché."""
    code = 0
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, None
            code = e.code
        except Exception:
            code = 0
        time.sleep(1.5 * (k + 1))
    return code or 0, None


def parse_zip(blob):
    """Primera fila del día -> (ts_ms, oi_amt, oi_usd). Devuelve None si el formato no es el esperado."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    tcol = next((c for c in df.columns if "time" in c.lower()), None)
    if tcol is None or "sum_open_interest" not in df.columns:
        return None
    df["_ms"] = to_ms(df[tcol])
    r = df.sort_values("_ms").iloc[0]
    return int(r["_ms"]), float(r["sum_open_interest"]), float(r.get("sum_open_interest_value", float("nan")))


def to_binance(sym):
    return sym.split("/")[0] + sym.split("/")[1].split(":")[0]            # BTC/USDT:USDT -> BTCUSDT


def probe_funding(get=None):
    get = get or http_get
    ok = False
    for y, m in ((2022, 6), (2024, 1)):
        code, blob = get(funding_url("BTCUSDT", y, m))
        info = ""
        if code == 200 and blob:
            try:
                p = parse_funding_zip(blob)
                info = f" -> formato OK, {len(p)} liquidaciones" if p is not None and len(p) else " -> formato NO reconocido"
                ok = ok or (p is not None and len(p) > 0)
            except Exception as e:
                info = f" -> error al leer: {e}"
        print(f"  funding {y}-{m:02d}: HTTP {code}{info}")
    print("RESULTADO funding: existe y se lee bien" if ok else "RESULTADO funding: no disponible aquí")
    return ok


def probe(get=None):
    get = get or http_get
    ok = False
    for day in (dt.date(2021, 12, 15), dt.date(2022, 6, 15), dt.date(2024, 1, 15)):
        code, blob = get(url_for("BTCUSDT", day))
        info = ""
        if code == 200 and blob:
            try:
                p = parse_zip(blob)
                info = f" -> formato OK, OI={p[1]:.0f}" if p else " -> formato NO reconocido (revisar columnas)"
                ok = ok or bool(p)
            except Exception as e:
                info = f" -> error al leer: {e}"
        print(f"  {day}: HTTP {code}{info}")
    print("RESULTADO: los archivos de OI existen y se leen bien" if ok else
          "RESULTADO: no se pudo leer OI diario aquí (red bloqueada, archivos ausentes o formato distinto); no sigas con la descarga masiva")
    return ok


def download_funding(symbols, start, end, out_dir, workers=16, get=None, log=print):
    """Funding mensual desde data.binance.vision para cada símbolo (cache por símbolo para poder reanudar)."""
    get = get or http_get
    os.makedirs(out_dir, exist_ok=True)
    months = [(d.year, d.month) for d in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")]
    result = {}
    for sym in symbols:
        path = os.path.join(out_dir, "fund_" + to_binance(sym) + ".pkl")
        if os.path.exists(path):
            result[sym] = pickle.load(open(path, "rb"))
            log(f"{sym} funding: en caché")
            continue
        bs = to_binance(sym)
        bad = []

        def one(ym):
            code, blob = get(funding_url(bs, *ym))
            if code not in (200, 404):
                bad.append(code)
            if code == 200 and blob:
                try:
                    return parse_funding_zip(blob)
                except Exception:
                    return None
            return None
        with cf.ThreadPoolExecutor(workers) as ex:
            parts = [p for p in ex.map(one, months) if p is not None and len(p)]
        df = (pd.concat(parts).drop_duplicates("ts").sort_values("ts").reset_index(drop=True) if parts
              else pd.DataFrame(columns=["ts", "rate"]))
        if not bad:                                       # con fallos de red NO se guarda caché (se reintenta al volver a ejecutar)
            pickle.dump(df, open(path, "wb"))
        result[sym] = df
        log(f"{sym} funding:{' [' + str(len(bad)) + ' ARCHIVOS CON ERROR DE RED, sin caché]' if bad else ''} {len(df)} liquidaciones" + (f" ({pd.to_datetime(df.ts.iloc[0], unit='ms', utc=True):%Y-%m-%d} -> "
                                                       f"{pd.to_datetime(df.ts.iloc[-1], unit='ms', utc=True):%Y-%m-%d})" if len(df) else " (sin archivos)"))
    return {s: d for s, d in result.items() if len(d) > 10}


def download(symbols, start, end, out_dir, workers=16, get=None, log=print):
    get = get or http_get
    os.makedirs(out_dir, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    result = {}
    for sym in symbols:
        path = os.path.join(out_dir, to_binance(sym) + ".pkl")
        if os.path.exists(path):
            result[sym] = pickle.load(open(path, "rb"))
            log(f"{sym} OI: en caché")
            continue
        bs = to_binance(sym)
        bad = []

        def one(day):
            code, blob = get(url_for(bs, day.date()))
            if code not in (200, 404):
                bad.append(code)
            if code == 200 and blob:
                try:
                    return parse_zip(blob)
                except Exception:
                    return None
            return None
        with cf.ThreadPoolExecutor(workers) as ex:
            rows = [r for r in ex.map(one, days) if r]
        df = pd.DataFrame(rows, columns=["ts", "oi_amt", "oi_usd"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        if not bad:
            pickle.dump(df, open(path, "wb"))
        result[sym] = df
        log(f"{sym}:{' [' + str(len(bad)) + ' ARCHIVOS CON ERROR DE RED, sin caché]' if bad else ''} {len(df)} días con OI" + (f" ({pd.to_datetime(df.ts.iloc[0], unit='ms', utc=True):%Y-%m-%d} -> "
                                              f"{pd.to_datetime(df.ts.iloc[-1], unit='ms', utc=True):%Y-%m-%d})" if len(df) else " (sin archivos)"))
    return {"funding": {}, "oi": {s: d for s, d in result.items() if len(d) > 10}, "meta": {"source": "data.binance.vision metrics"}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--what", default="both", help="oi | funding | both")
    ap.add_argument("--funding-start", default="2020-01-01")
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--start", default="2021-12-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cache", default="metrics_cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="derivs_metrics.pkl")
    a = ap.parse_args()
    if a.probe:
        r1, r2 = probe(), probe_funding()
        raise SystemExit(0 if (r1 or r2) else 1)
    syms = a.symbols or [x.strip() for x in open(a.symbols_file) if x.strip()]
    end = a.end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    data = {"funding": {}, "oi": {}, "meta": {"source": "data.binance.vision"}}
    if a.what in ("oi", "both"):
        data["oi"] = download(syms, a.start, end, a.cache, a.workers)["oi"]
    if a.what in ("funding", "both"):
        data["funding"] = download_funding(syms, a.funding_start, end, a.cache, a.workers)
    pickle.dump(data, open(a.out, "wb"))
    print(f"guardado {a.out}: OI de {len(data['oi'])} símbolos, funding de {len(data['funding'])} símbolos")
