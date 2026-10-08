"""Dashboard web del bot cross-sectional diario (PAPER: no envía órdenes reales).

Uso:   python bots/xs_daily/dashboard.py                 -> abre http://127.0.0.1:8765 en el navegador
       python bots/xs_daily/dashboard.py --exchange okx
       python bots/xs_daily/dashboard.py --demo          -> sin internet: usa data/ohlcv_daily_long.pkl y un estado de ejemplo
Requiere: pandas, numpy, scikit-learn, ccxt (lo mismo que el bot). El gráfico es TradingView Lightweight Charts (Apache-2.0),
incluido en bots/xs_daily/web/ para que funcione sin CDN.

Qué muestra:
  - Velas tipo TradingView (1h / 4h / 1d) de cualquier moneda, con las entradas del bot marcadas y precio en vivo.
  - Posiciones abiertas con PnL en vivo (se refresca solo), equity, cohortes cerradas y estadística honesta.
  - Análisis del modelo: ranking de todas las monedas, por qué eligió cada una (contribución de cada factor medida
    quitando ese factor y viendo cuánto cambia la predicción), importancia global y contexto de mercado.
  - Botones para ejecutar el paper de hoy y reentrenar (corren run_paper.py / train.py en un proceso aparte).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xs_core  # noqa: E402

ROOT = xs_core.ROOT
WEB = os.path.join(HERE, "web")
STATE = os.path.join(HERE, "paper_state.json")
LOG = os.path.join(HERE, "paper_log.csv")
MODEL = os.path.join(HERE, "model_xs_D.pkl")
ANALYSIS = os.path.join(HERE, "analysis_cache.json")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BACKTEST_REF = dict(mean=0.0029, lo=-0.0023, hi=0.0081, note="backtest 6 años, 1725 cohortes")

# ------------------------------------------------------------------ descripción legible de los factores (Alpha158)
GROUPS = {
    "kmid": ("Vela", "cuerpo de la vela de ayer (cierre vs apertura)"), "klen": ("Vela", "rango de la vela de ayer"),
    "kmid2": ("Vela", "cuerpo relativo al rango"), "kup": ("Vela", "mecha superior"), "kup2": ("Vela", "mecha superior relativa"),
    "klow": ("Vela", "mecha inferior"), "klow2": ("Vela", "mecha inferior relativa"), "ksft": ("Vela", "cierre cerca del máximo o del mínimo"),
    "ksft2": ("Vela", "cierre dentro del rango"),
}
PREFIX = [  # (prefijo, grupo, plantilla); n = ventana en días
    ("roc", "Momentum", "precio de hace {n} d / precio actual (alto = cayó en {n} d)"),
    ("ma", "Tendencia", "media de {n} d / precio (alto = precio por debajo de su media)"),
    ("std", "Volatilidad", "volatilidad de {n} d"),
    ("max", "Rango", "máximo de {n} d / precio (alto = lejos del máximo)"),
    ("min", "Rango", "mínimo de {n} d / precio (alto = cerca del mínimo)"),
    ("qtlu", "Rango", "percentil 80 de {n} d / precio"), ("qtld", "Rango", "percentil 20 de {n} d / precio"),
    ("rank", "Rango", "posición del precio en su historial de {n} d"),
    ("rsv", "Rango", "estocástico de {n} d (0 = en el mínimo, 1 = en el máximo)"),
    ("imax", "Tiempo", "días desde el máximo de {n} d"), ("imin", "Tiempo", "días desde el mínimo de {n} d"),
    ("corr", "Volumen", "correlación precio-volumen {n} d"), ("cord", "Volumen", "correlación de cambios precio-volumen {n} d"),
    ("cntp", "Momentum", "% de días alcistas en {n} d"), ("sump", "Momentum", "fuerza alcista tipo RSI en {n} d"),
    ("vma", "Volumen", "volumen medio {n} d / volumen de ayer"), ("vstd", "Volumen", "variabilidad del volumen {n} d"),
    ("wvma", "Volumen", "volatilidad ponderada por volumen {n} d"),
    ("beta", "Tendencia", "pendiente de la tendencia de {n} d"), ("rsqr", "Tendencia", "qué tan limpia es la tendencia de {n} d (R²)"),
    ("resi", "Tendencia", "desvío del precio respecto a su tendencia de {n} d"),
]


def describe(f):
    if f in GROUPS:
        return GROUPS[f]
    for p, g, t in PREFIX:
        if f.startswith(p) and f[len(p):].isdigit():
            return g, t.format(n=f[len(p):])
    return "Otro", f


# ------------------------------------------------------------------ fuente de mercado (exchange real o datos locales en --demo)
class LiveSource:
    def __init__(self, exchange):
        import ccxt
        self.ex = getattr(ccxt, exchange)({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.name = exchange
        self.markets_ok = False
        self.lock = threading.Lock()

    def _markets(self):
        with self.lock:
            if not self.markets_ok:
                self.ex.load_markets()
                self.markets_ok = True

    def tickers(self, syms):
        self._markets()
        syms = [s for s in syms if s in self.ex.markets]
        out = {}
        if not syms:
            return out
        try:
            t = self.ex.fetch_tickers(syms)
        except Exception:                                   # algunos exchanges no aceptan lista: uno por uno
            t = {}
            for s in syms:
                try:
                    t[s] = self.ex.fetch_ticker(s)
                except Exception:
                    pass
        for s, v in t.items():
            px = v.get("last") or v.get("close")
            if px:
                out[s] = dict(price=float(px), change=float(v.get("percentage") or 0.0) / 100.0)
        return out

    def candles(self, sym, tf, limit=500):
        self._markets()
        rows = self.ex.fetch_ohlcv(sym, tf, limit=limit)
        return [dict(time=int(r[0] // 1000), open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5]) for r in rows]

    def history(self, syms, days=150):
        import run_paper as rp
        self._markets()
        return rp.fetch_history(self.ex, [s for s in syms if s in self.ex.markets], days)


class DemoSource:
    """Sin red: velas diarias del archivo local; el 'precio en vivo' es el último cierre con un pequeño ruido."""

    def __init__(self, path):
        from engine import load_universe
        self.uni = load_universe(path, min_bars=150)
        self.name = "demo (datos locales)"
        self.rng = np.random.default_rng(0)

    def tickers(self, syms):
        out = {}
        for s in syms:
            if s in self.uni:
                c = self.uni[s]["close"].to_numpy()
                out[s] = dict(price=float(c[-1] * (1 + self.rng.normal(0, 0.001))), change=float(c[-1] / c[-2] - 1))
        return out

    def candles(self, sym, tf, limit=500):
        df = self.uni[sym].tail(limit)
        return [dict(time=int(pd.Timestamp(t).timestamp()), open=o, high=h, low=l, close=c, volume=v)
                for t, o, h, l, c, v in df[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False)]

    def history(self, syms, days=150):
        return {s: self.uni[s].tail(days + 1).reset_index(drop=True) for s in syms if s in self.uni}


# ------------------------------------------------------------------ análisis del modelo
def compute_analysis(bundle, hist, progress=lambda *_: None):
    idx, P, F, liq_ok = xs_core.features(hist)
    names = bundle["features"]
    cols = list(P["close"].columns)
    X = np.stack([F[k].to_numpy()[-1] for k in names], axis=-1)
    valid = liq_ok[-1] & (np.isfinite(X).sum(axis=1) > 0) & np.isfinite(P["close"].to_numpy()[-1])
    iv = np.where(valid)[0]
    mdl = bundle["model"]
    Xv = X[iv]
    pred = mdl.predict(Xv)
    progress("contribuciones de cada factor")
    contrib = np.zeros_like(Xv)
    for j in range(len(names)):                              # ablación: el factor j se lleva a la mediana (0.5 en ranking)
        Xa = Xv.copy()
        Xa[:, j] = 0.5
        contrib[:, j] = pred - mdl.predict(Xa)
    order = np.argsort(-pred)
    K = bundle["K"]
    longs, shorts = set(order[:K]), set(order[-K:])
    c, v = P["close"], P["volume"]
    r1, r7, r30 = c.pct_change(1).iloc[-1], c.pct_change(7).iloc[-1], c.pct_change(30).iloc[-1]
    vol30 = c.pct_change().rolling(30).std().iloc[-1] * np.sqrt(365)
    qv30 = (c * v).rolling(30).mean().iloc[-1]
    rows = []
    for rank, k in enumerate(order):
        i = iv[k]
        s = cols[i]
        top = np.argsort(-np.abs(contrib[k]))[:8]
        drivers = []
        for j in top:
            g, d = describe(names[j])
            drivers.append(dict(feature=names[j], group=g, desc=d, value=float(Xv[k, j]) if np.isfinite(Xv[k, j]) else None,
                                contrib=float(contrib[k, j])))
        grp = {}
        for j, n in enumerate(names):
            g = describe(n)[0]
            grp[g] = grp.get(g, 0.0) + float(contrib[k, j])
        rows.append(dict(symbol=s, score=float(pred[k]), rank=rank + 1, pct=float(1 - rank / max(len(order) - 1, 1)),
                         decision="LONG" if k in longs else ("SHORT" if k in shorts else "—"),
                         ret1=float(r1[s]), ret7=float(r7[s]), ret30=float(r30[s]), vol30=float(vol30[s]), qv30=float(qv30[s]),
                         drivers=drivers, groups=grp))
    imp = np.abs(contrib).mean(axis=0)
    gimp = {}
    for j, n in enumerate(names):
        g = describe(n)[0]
        gimp[g] = gimp.get(g, 0.0) + float(imp[j])
    top_feats = [dict(feature=names[j], group=describe(names[j])[0], desc=describe(names[j])[1], importance=float(imp[j]))
                 for j in np.argsort(-imp)[:15]]
    ma20 = c.rolling(20).mean().iloc[-1]
    btc = next((s for s in cols if s.startswith("BTC/")), None)
    market = dict(
        date=str(idx[-1].date()), n_symbols=len(cols), n_valid=int(len(iv)),
        breadth=float((c.iloc[-1] > ma20).mean()), avg_ret7=float(r7.mean()), dispersion7=float(r7.std()),
        median_vol=float(vol30.median()), btc_ret7=float(r7[btc]) if btc else None, btc_ret30=float(r30[btc]) if btc else None,
        long_avg_ret7=float(np.mean([r["ret7"] for r in rows if r["decision"] == "LONG"])),
        short_avg_ret7=float(np.mean([r["ret7"] for r in rows if r["decision"] == "SHORT"])),
    )
    return dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), model_trained_until=bundle.get("trained_until"), K=K, H=bundle["H"],
                market=market, ranking=rows, groups_importance=gimp, top_features=top_feats)


# ------------------------------------------------------------------ estado de la app
class Hub:
    def __init__(self, source, capital, demo=False):
        self.src, self.capital, self.demo = source, capital, demo
        self.cache = {}
        self.analysis = json.load(open(ANALYSIS)) if os.path.exists(ANALYSIS) else None
        self.an_status = dict(state="idle", msg="")
        self.job = dict(running=False, name="", lines=[], code=None)
        self.lock = threading.Lock()

    def cached(self, key, ttl, fn):
        now = time.time()
        hit = self.cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        val = fn()
        self.cache[key] = (now, val)
        return val

    def state(self):
        st = json.load(open(STATE)) if os.path.exists(STATE) else {"equity": self.capital, "cohorts": [], "closed": []}
        log = pd.read_csv(LOG).to_dict("records") if os.path.exists(LOG) else []
        cl = st["closed"]
        r = np.array([c["ret"] for c in cl]) if cl else np.array([])
        stats = None
        if len(r):
            n_eff = max(len(r) / 7, 1)
            se = r.std(ddof=1) / np.sqrt(n_eff) if len(r) > 1 else float("nan")
            stats = dict(mean=float(r.mean()), lo=float(r.mean() - 1.96 * se), hi=float(r.mean() + 1.96 * se), n=len(r),
                         n_eff=n_eff, win=float((r > 0).mean()), best=float(r.max()), worst=float(r.min()))
        model = None
        if os.path.exists(MODEL):
            model = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(MODEL)))
        return dict(state=st, log=log, stats=stats, capital=self.capital, model=model, backtest=BACKTEST_REF,
                    source=self.src.name, demo=self.demo, today_done=any(c["opened"] == time.strftime("%Y-%m-%d", time.gmtime())
                                                                          for c in st["cohorts"]))

    def symbols_of_interest(self):
        st = json.load(open(STATE)) if os.path.exists(STATE) else {"cohorts": []}
        s = {p["symbol"] for c in st["cohorts"] for p in c["positions"]}
        if self.analysis:
            s |= {r["symbol"] for r in self.analysis["ranking"][:15]} | {r["symbol"] for r in self.analysis["ranking"][-15:]}
        s.add("BTC/USDT:USDT")
        return sorted(s)

    def prices(self, extra=None):
        syms = self.symbols_of_interest() + ([extra] if extra else [])
        return self.cached(("px", tuple(sorted(set(syms)))), 5, lambda: self.src.tickers(sorted(set(syms))))

    def candles(self, sym, tf):
        ttl = 20 if tf in ("1m", "5m", "15m", "1h") else 60
        return self.cached(("c", sym, tf), ttl, lambda: self.src.candles(sym, tf if not self.demo else "1d"))

    def start_analysis(self, force=False):
        with self.lock:
            if self.an_status["state"] == "running":
                return
            if not force and self.analysis and self.analysis.get("generated", "")[:10] == time.strftime("%Y-%m-%d"):
                return
            if not os.path.exists(MODEL):
                self.an_status = dict(state="error", msg="No hay modelo entrenado: pulsa «Reentrenar modelo».")
                return
            self.an_status = dict(state="running", msg="descargando velas diarias de todas las monedas…")

        def work():
            try:
                bundle = pickle.load(open(MODEL, "rb"))
                hist = self.src.history(bundle["symbols"])
                self.an_status["msg"] = f"calculando modelo sobre {len(hist)} monedas…"
                a = compute_analysis(bundle, hist, progress=lambda m: self.an_status.update(msg=m))
                json.dump(a, open(ANALYSIS, "w"))
                self.analysis = a
                self.an_status = dict(state="done", msg="")
            except Exception as e:                           # noqa: BLE001
                self.an_status = dict(state="error", msg=f"{type(e).__name__}: {str(e)[:200]}")
        threading.Thread(target=work, daemon=True).start()

    def run_job(self, name, args):
        if self.job["running"]:
            return False
        self.job = dict(running=True, name=name, lines=[], code=None)

        def work():
            try:
                p = subprocess.Popen([sys.executable, "-u", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     cwd=ROOT, creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
                for line in p.stdout:
                    self.job["lines"].append(line.rstrip("\n"))
                p.wait()
                self.job["code"] = p.returncode
            except Exception as e:                           # noqa: BLE001
                self.job["lines"].append(f"ERROR: {e}")
                self.job["code"] = 1
            self.job["running"] = False
            if name == "train" and self.job["code"] == 0:
                self.start_analysis(force=True)
        threading.Thread(target=work, daemon=True).start()
        return True


def make_handler(hub: Hub, exchange):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, open(os.path.join(WEB, "index.html"), "rb").read(), "text/html; charset=utf-8")
                if u.path.startswith("/web/"):
                    f = os.path.join(WEB, os.path.basename(u.path))
                    return self._send(200, open(f, "rb").read(), "application/javascript" if f.endswith(".js") else "text/plain")
                if u.path == "/api/state":
                    return self._send(200, hub.state())
                if u.path == "/api/prices":
                    return self._send(200, hub.prices(q.get("extra")))
                if u.path == "/api/candles":
                    return self._send(200, hub.candles(q["symbol"], q.get("tf", "1h")))
                if u.path == "/api/analysis":
                    if q.get("refresh"):
                        hub.start_analysis(force=True)
                    elif hub.analysis is None or hub.analysis.get("generated", "")[:10] != time.strftime("%Y-%m-%d"):
                        hub.start_analysis()
                    return self._send(200, dict(status=hub.an_status, analysis=hub.analysis))
                if u.path == "/api/job":
                    return self._send(200, hub.job)
                return self._send(404, {"error": "no encontrado"})
            except Exception as e:                           # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {str(e)[:300]}"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/run":
                ok = hub.run_job("run", [os.path.join(HERE, "run_paper.py"), "--force", "--capital", str(hub.capital), "--exchange", exchange])
            elif u.path == "/api/train":
                ok = hub.run_job("train", [os.path.join(HERE, "train.py")])
            else:
                return self._send(404, {"error": "no encontrado"})
            return self._send(200 if ok else 409, {"started": ok})
    return H


def build_demo_state(uni, capital, days=40, seed=7):
    """Estado de ejemplo con precios reales del archivo local (selección aleatoria, solo para ver la interfaz)."""
    rng = np.random.default_rng(seed)
    syms = [s for s, d in uni.items() if len(d) > 200]
    end = min(d.ts.iloc[-1] for d in (uni[s] for s in syms[:20]))
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    st = {"equity": capital, "cohorts": [], "closed": []}
    rows = []

    def px(s, d):
        df = uni[s]
        r = df[df.ts == d]
        return float(r.open.iloc[0]) if len(r) else None
    for d in dates:
        keep = []
        for c in st["cohorts"]:
            if (d - pd.Timestamp(c["opened"], tz="UTC")).days >= 7:
                pnl = sum(p["notional"] * (1 if p["side"] == "LONG" else -1) * ((px(p["symbol"], d) or p["entry"]) / p["entry"] - 1) - p["notional"] * 0.0005
                          for p in c["positions"])
                st["equity"] += pnl
                st["closed"].append(dict(opened=c["opened"], closed=str(d.date()), pnl=pnl, capital=c["capital"], ret=pnl / c["capital"]))
            else:
                keep.append(c)
        st["cohorts"] = keep
        pick = rng.choice(syms, 20, replace=False)
        cap = st["equity"] / 7
        pos = [dict(symbol=s, side="LONG" if i < 10 else "SHORT", entry=px(s, d), notional=cap * 0.5 / 10) for i, s in enumerate(pick) if px(s, d)]
        st["cohorts"].append(dict(opened=str(d.date()), signal_date=str((d - pd.Timedelta(days=1)).date()), capital=cap, positions=pos))
        rows.append(dict(fecha=str(d.date()), equity_realizada=st["equity"], no_realizado=0.0, equity_total=st["equity"], cohortes_abiertas=len(st["cohorts"])))
    return st, pd.DataFrame(rows)


def main():
    global STATE, LOG, ANALYSIS
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--demo", action="store_true", help="sin internet: datos locales")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "ohlcv_daily_long.pkl"))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    src = DemoSource(a.data) if a.demo else LiveSource(a.exchange)
    if a.demo:                                               # el demo nunca toca tu estado real
        STATE, LOG, ANALYSIS = (os.path.join(HERE, f) for f in ("demo_state.json", "demo_log.csv", "demo_analysis.json"))
        if not os.path.exists(STATE):
            st, lg = build_demo_state(src.uni, a.capital)
            json.dump(st, open(STATE, "w"))
            lg.to_csv(LOG, index=False)
    hub = Hub(src, a.capital, a.demo)
    hub.start_analysis()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(hub, a.exchange))
    url = f"http://127.0.0.1:{a.port}"
    print(f"Dashboard en {url}  (Ctrl+C para cerrar) | fuente: {src.name} | MODO PAPER: no se envían órdenes reales")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
