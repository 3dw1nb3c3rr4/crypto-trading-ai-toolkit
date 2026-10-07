"""Descarga el historial diario de open interest (OI) desde los archivos públicos de Binance (data.binance.vision, carpeta
futures/um/daily/metrics). Cada archivo trae una foto cada 5 minutos del día: se guarda la primera (00:00 UTC) como foto del OI.

Primero comprueba que los archivos existen:   python download_binance_metrics.py --probe
Luego descarga:  python download_binance_metrics.py --symbols-file universe_symbols.txt --start 2021-12-01 --out derivs_metrics.pkl
Reanuda solo (cache por símbolo en --cache). Salida compatible con ml_daily_xs_v2.py --derivs.
Probado con respuestas simuladas; la existencia y el formato reales los verifica --probe (sin eso, no se asume nada).
"""
import argparse
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


def url_for(binance_symbol, day):
    return f"{BASE}/{binance_symbol}/{binance_symbol}-metrics-{day:%Y-%m-%d}.zip"


def http_get(url, timeout=30):
    """Devuelve (codigo, bytes|None). 404 = el archivo no existe (normal antes de la fecha de inicio del símbolo)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return 0, None


def parse_zip(blob):
    """Primera fila del día -> (ts_ms, oi_amt, oi_usd). Devuelve None si el formato no es el esperado."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]))
    tcol = next((c for c in df.columns if "time" in c.lower()), None)
    if tcol is None or "sum_open_interest" not in df.columns:
        return None
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    r = df.sort_values(tcol).iloc[0]
    return int(r[tcol].value // 10**6), float(r["sum_open_interest"]), float(r.get("sum_open_interest_value", float("nan")))


def to_binance(sym):
    return sym.split("/")[0] + sym.split("/")[1].split(":")[0]            # BTC/USDT:USDT -> BTCUSDT


def probe(get=http_get):
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


def download(symbols, start, end, out_dir, workers=16, get=http_get, log=print):
    os.makedirs(out_dir, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    result = {}
    for sym in symbols:
        path = os.path.join(out_dir, to_binance(sym) + ".pkl")
        if os.path.exists(path):
            result[sym] = pickle.load(open(path, "rb"))
            continue
        bs = to_binance(sym)

        def one(day):
            code, blob = get(url_for(bs, day.date()))
            if code == 200 and blob:
                try:
                    return parse_zip(blob)
                except Exception:
                    return None
            return None
        with cf.ThreadPoolExecutor(workers) as ex:
            rows = [r for r in ex.map(one, days) if r]
        df = pd.DataFrame(rows, columns=["ts", "oi_amt", "oi_usd"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        pickle.dump(df, open(path, "wb"))
        result[sym] = df
        log(f"{sym}: {len(df)} días con OI" + (f" ({pd.to_datetime(df.ts.iloc[0], unit='ms', utc=True):%Y-%m-%d} -> "
                                              f"{pd.to_datetime(df.ts.iloc[-1], unit='ms', utc=True):%Y-%m-%d})" if len(df) else " (sin archivos)"))
    return {"funding": {}, "oi": {s: d for s, d in result.items() if len(d) > 10}, "meta": {"source": "data.binance.vision metrics"}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--symbols-file", default=None)
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--start", default="2021-12-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cache", default="metrics_cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="derivs_metrics.pkl")
    a = ap.parse_args()
    if a.probe:
        raise SystemExit(0 if probe() else 1)
    syms = a.symbols or [x.strip() for x in open(a.symbols_file) if x.strip()]
    end = a.end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    data = download(syms, a.start, end, a.cache, a.workers)
    pickle.dump(data, open(a.out, "wb"))
    print(f"guardado {a.out}: OI de {len(data['oi'])} símbolos")
