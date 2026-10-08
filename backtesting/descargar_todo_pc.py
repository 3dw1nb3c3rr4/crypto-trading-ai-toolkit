"""Descarga funding + open interest en tu PC, todas las fuentes EN PARALELO (OKX, Bybit y Binance por API; Binance por archivos públicos).
Cada fuente es independiente: si una está bloqueada en tu país o falla, las demás siguen. Reanuda los archivos públicos de Binance.

  pip install ccxt pandas numpy
  python descargar_todo_pc.py                 # todo (puede tardar; las 4 fuentes corren a la vez)
  python descargar_todo_pc.py --limit 10      # prueba rápida con 10 símbolos
  python descargar_todo_pc.py --skip bybit    # omitir fuentes (okx,bybit,binance_api,binance_files)

Salida en ./derivados/: derivs_okx.pkl, derivs_bybit.pkl, derivs_binanceusdm.pkl, derivs_metrics.pkl.
Luego:  python ml_daily_xs_v2.py --data ../data/ohlcv_daily_long.pkl --kfrac 0.12 --test-len 180 --derivs derivados/derivs_*.pkl --variants E,F,O,FO
(el cargador elige por símbolo la fuente con más historial).
"""
import argparse, concurrent.futures as cf, os, pickle, time, traceback
import pandas as pd
import download_derivs as dd
import download_binance_metrics as dm

HERE = os.path.dirname(os.path.abspath(__file__))


def cobertura(nombre, d):
    filas = [(len(df), (pd.to_datetime(df["ts"].iloc[-1], unit="ms") - pd.to_datetime(df["ts"].iloc[0], unit="ms")).days) for df in d.values() if len(df) > 1]
    print(f"   {nombre}: {len(filas)} símbolos" + (f", historial mediano {int(pd.Series([f[1] for f in filas]).median())} días (máx {max(f[1] for f in filas)})" if filas else " SIN DATOS"), flush=True)


def job_api(exchange, symbols, out_dir, days):
    d = dd.download(exchange, symbols, days, log=lambda *a: None)
    pickle.dump(d, open(os.path.join(out_dir, f"derivs_{exchange}.pkl"), "wb"))
    print(f"[{exchange}] errores: {len(d['meta']['errors'])}", flush=True)
    cobertura(f"{exchange} funding", d["funding"]); cobertura(f"{exchange} OI", d["oi"])


def job_files(symbols, out_dir, workers, what="both", oi_limit=None):
    t0, n = time.time(), [0]

    def log(msg):                                     # un renglón por símbolo terminado, con avance y tiempo
        n[0] += 1
        print(f"  [{n[0]}/{len(symbols) * 2}] {time.time()-t0:5.0f}s  {msg}", flush=True)
    ok_oi, ok_f = dm.probe(), dm.probe_funding()
    end = (pd.Timestamp.now("UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache = os.path.join(out_dir, "metrics_cache")
    data = {"funding": {}, "oi": {}, "meta": {"source": "data.binance.vision"}}
    oi_syms = symbols[:oi_limit] if oi_limit else symbols
    if ok_f and what in ("both", "funding"):
        data["funding"] = dm.download_funding(symbols, "2020-01-01", end, cache, workers, log=log)
    if ok_oi and what in ("both", "oi"):
        data["oi"] = dm.download(oi_syms, "2021-12-01", end, cache, workers, log=log)["oi"]
    name = {"both": "derivs_metrics.pkl", "funding": "derivs_funding_bin.pkl", "oi": "derivs_oi_bin.pkl"}[what]
    pickle.dump(data, open(os.path.join(out_dir, name), "wb"))
    print(f"guardado {name}")
    print(f"[binance_files] probe OI={ok_oi} funding={ok_f}", flush=True)
    cobertura("binance_files funding", data["funding"]); cobertura("binance_files OI", data["oi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", default="")
    ap.add_argument("--days", type=int, default=2200)
    ap.add_argument("--workers", type=int, default=32, help="hilos para los archivos públicos de Binance")
    ap.add_argument("--out", default="derivados")
    ap.add_argument("--what", default="both", choices=["both", "funding", "oi"], help="solo los archivos públicos de Binance: funding (rápido) / oi (lento) / both")
    ap.add_argument("--rank-by", default=None, help="pickle diario (p.ej. ../data/ohlcv_daily_long.pkl): ordena los símbolos por volumen en USDT reciente")
    ap.add_argument("--oi-limit", type=int, default=None, help="OI solo para los N primeros símbolos (con --rank-by = los N más líquidos)")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out); os.makedirs(out_dir, exist_ok=True)
    symbols = [x.strip() for x in open(os.path.join(HERE, "universe_symbols.txt")) if x.strip()]
    may = "BTC ETH SOL BNB XRP DOGE ADA AVAX LINK LTC DOT TRX BCH NEAR ATOM".split()
    symbols.sort(key=lambda s: (may.index(s.split("/")[0]) if s.split("/")[0] in may else 99, s))
    if a.rank_by:
        raw = pickle.load(open(a.rank_by, "rb"))
        vol = {k: float((v["close"] * v["volume"]).tail(60).mean()) for k, v in raw.items() if len(v)}
        symbols.sort(key=lambda x: -vol.get(x, 0.0))
    if a.limit:
        symbols = symbols[:a.limit]
    skip = set(x for x in a.skip.split(",") if x)
    jobs = {}
    for ex, name in (("okx", "okx"), ("bybit", "bybit"), ("binanceusdm", "binance_api")):
        if name not in skip:
            jobs[name] = lambda ex=ex: job_api(ex, symbols, out_dir, a.days)
    if "binance_files" not in skip:
        jobs["binance_files"] = lambda: job_files(symbols, out_dir, a.workers, a.what, a.oi_limit)
    t0 = time.time()
    print(f"{len(symbols)} símbolos | fuentes en paralelo: {list(jobs)} | salida: {out_dir}", flush=True)
    with cf.ThreadPoolExecutor(len(jobs)) as pool:
        futs = {pool.submit(fn): n for n, fn in jobs.items()}
        for f in cf.as_completed(futs):
            try:
                f.result(); print(f"== {futs[f]} terminó ({time.time()-t0:.0f}s)", flush=True)
            except Exception:
                print(f"== {futs[f]} FALLÓ (¿bloqueado en tu país?):\n{traceback.format_exc(limit=2)}", flush=True)
    print(f"\nListo en {time.time()-t0:.0f}s. Archivos: {sorted(f for f in os.listdir(out_dir) if f.endswith('.pkl'))}")


if __name__ == "__main__":
    main()
