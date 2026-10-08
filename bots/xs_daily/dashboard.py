"""Panel de trading local (estilo TradingView) del proyecto Cerebro.

Uso:   python bots/xs_daily/dashboard.py            -> abre http://127.0.0.1:8765 en el navegador
       python bots/xs_daily/dashboard.py --demo     -> sin internet: datos locales y cuentas de ejemplo (no toca tus archivos)

Funciones:
  - Velas TradingView (Lightweight Charts, Apache-2.0, incluida en web/) con precio en vivo, búsqueda de cualquier perpetuo,
    funding actual y cuenta regresiva, entradas marcadas.
  - Trading MANUAL como un exchange: mercado / límite, apalancamiento, valor de la posición, TP/SL, cierre total o parcial,
    cancelar órdenes. Modo PAPER (simulado con comisiones, deslizamiento, funding real y liquidación), DEMO (cuenta demo del
    exchange) o REAL (tu cuenta, con topes de seguridad y confirmación). Claves API guardadas solo en tu PC (config_local.py).
  - Bot de la estrategia (siempre en paper): ejecutar hoy, reentrenar, cerrar posiciones del bot antes de tiempo, equity,
    cohortes con funding, ranking del modelo y por qué eligió cada moneda.
  - Universo: solo cripto, solo acciones/TradFi tokenizadas o ambos. Reinicio de cuentas con copia de seguridad.
El servidor solo escucha en 127.0.0.1 (tu PC).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
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
import config_local as cl  # noqa: E402
import cuenta_paper as cp  # noqa: E402
import cuenta_real as cr  # noqa: E402
import datos_cuenta as dc  # noqa: E402
import derivs_live  # noqa: E402
import funding_util  # noqa: E402
import universo  # noqa: E402
import xs_core  # noqa: E402

ROOT = xs_core.ROOT
WEB = os.path.join(HERE, "web")
STATE = os.path.join(HERE, "paper_state.json")
LOG = os.path.join(HERE, "paper_log.csv")
MODEL = os.path.join(HERE, "model_xs_D.pkl")
ANALYSIS = os.path.join(HERE, "analysis_cache.json")
DEMO = False


def use_variant(v, demo=None):
    """Apunta modelo / estado / log / análisis del bot a la variante elegida (D original, FO con funding y OI)."""
    global STATE, LOG, MODEL, ANALYSIS, DEMO
    import run_paper
    DEMO = DEMO if demo is None else demo
    MODEL = os.path.join(HERE, f"model_xs_{v}.pkl")
    sfx = "" if v == "D" else f"_{v}"
    if DEMO:
        STATE, LOG, ANALYSIS = (os.path.join(HERE, f"demo_{n}{sfx}.{e}") for n, e in (("state", "json"), ("log", "csv"), ("analysis", "json")))
    else:
        _, STATE, LOG = run_paper.paths(v)
        ANALYSIS = os.path.join(HERE, f"analysis_cache{sfx}.json")
MANUAL = os.path.join(HERE, "cuenta_manual.json")
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


BASE_DESC = {
    "ret1": "retorno de 1 d", "ret3": "retorno de 3 d", "ret7": "retorno de 7 d", "ret14": "retorno de 14 d", "ret30": "retorno de 30 d",
    "ret60": "retorno de 60 d", "vol7": "volatilidad de 7 d", "vol30": "volatilidad de 30 d", "volratio": "volatilidad 7 d / 30 d",
    "range7": "rango de 7 d", "dhi30": "distancia al máximo de 30 d", "dlo30": "distancia al mínimo de 30 d",
    "dhi90": "distancia al máximo de 90 d", "dlo90": "distancia al mínimo de 90 d", "turn": "volumen de ayer vs su media",
    "lturn30": "volumen medio de 30 d (liquidez)", "rsi14": "RSI de 14 d", "body": "cuerpo de la vela", "upw": "mecha superior",
    "loww": "mecha inferior", "age": "antigüedad del listado", "rel7": "retorno de 7 d frente al mercado", "rel30": "retorno de 30 d frente al mercado",
    "beta60": "beta frente al mercado (60 d)", "idio60": "volatilidad propia (no explicada por el mercado)",
    "fund_1d": "funding de ayer (alto = largos pagando mucho)", "fund_3d": "funding medio de 3 d", "fund_7d": "funding medio de 7 d",
    "fund_z30": "funding de 7 d frente a su media de 30 d", "fund_chg3": "cambio del funding en 3 d", "fund_pos7": "% de días con funding positivo (7 d)",
    "oi_chg1": "cambio del open interest en 1 d", "oi_chg7": "cambio del open interest en 7 d", "oi_chg30": "cambio del open interest en 30 d",
    "oi_z30": "open interest frente a su media de 30 d", "oi_vol": "open interest / volumen negociado",
    "oi_px_div": "sube el OI sin que suba el precio (posicionamiento)",
}


def describe(f):
    if f.startswith("rk_"):
        return "Ranking", "ranking de " + BASE_DESC.get(f[3:], f[3:])
    if f.startswith("fund_"):
        return "Funding", BASE_DESC.get(f, f)
    if f.startswith("oi_"):
        return "Open interest", BASE_DESC.get(f, f)
    if f in BASE_DESC:
        return ("Momentum" if f.startswith(("ret", "rel", "rsi")) else "Volatilidad" if f.startswith(("vol", "idio", "range", "beta")) else
                "Volumen" if f in ("turn", "lturn30") else "Rango" if f.startswith(("dhi", "dlo")) else "Vela" if f in ("body", "upw", "loww") else "Otro"), BASE_DESC[f]
    if f in GROUPS:
        return GROUPS[f]
    for p, g, t in PREFIX:
        if f.startswith(p) and f[len(p):].isdigit():
            return g, t.format(n=f[len(p):])
    return "Otro", f


# ------------------------------------------------------------------ fuente de mercado (datos públicos; exchange real o datos locales en --demo)
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
        return self.ex.markets

    def markets(self):
        m = self._markets()
        return {s: v for s, v in m.items() if v.get("swap") and v.get("linear") and v.get("quote") == "USDT" and v.get("active", True)}

    def tickers(self, syms):
        mk = self._markets()
        syms = [s for s in syms if s in mk]
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
                out[s] = dict(price=float(px), bid=float(v.get("bid") or px), ask=float(v.get("ask") or px),
                              change=float(v.get("percentage") or 0.0) / 100.0, high=v.get("high"), low=v.get("low"),
                              volume=v.get("quoteVolume"))
        return out

    def candles(self, sym, tf, limit=500):
        self._markets()
        rows = self.ex.fetch_ohlcv(sym, tf, limit=limit)
        return [dict(time=int(r[0] // 1000), open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5]) for r in rows]

    def history(self, syms, days=150):
        import run_paper as rp
        mk = self._markets()
        return rp.fetch_history(self.ex, [s for s in syms if s in mk], days)

    def funding_rate(self, sym):
        try:
            f = self.ex.fetch_funding_rate(sym)
            return dict(rate=f.get("fundingRate"), next=f.get("fundingTimestamp") or f.get("nextFundingTimestamp"),
                        mark=f.get("markPrice"), interval=f.get("interval"))
        except Exception:
            return {}

    def funding_events(self, sym, since_ms, until_ms):
        return funding_util.funding_events(self.ex, sym, since_ms, until_ms)


class DemoSource:
    """Sin red: velas diarias del archivo local; el 'precio en vivo' es el último cierre con un pequeño ruido; funding 0.01 %/8 h."""

    def __init__(self, path):
        from engine import load_universe
        self.uni = load_universe(path, min_bars=150)
        self.name = "demo (datos locales)"
        self.rng = np.random.default_rng(0)

    def markets(self):
        return {s: {"symbol": s, "info": {}, "limits": {"leverage": {"max": 50}}} for s in self.uni}

    def tickers(self, syms):
        out = {}
        for s in syms:
            if s in self.uni:
                df = self.uni[s]
                c = df["close"].to_numpy()
                p = float(c[-1] * (1 + self.rng.normal(0, 0.001)))
                out[s] = dict(price=p, bid=p * 0.9998, ask=p * 1.0002, change=float(c[-1] / c[-2] - 1), high=float(df.high.iloc[-1]),
                              low=float(df.low.iloc[-1]), volume=float(df.volume.iloc[-1] * c[-1]))
        return out

    def candles(self, sym, tf, limit=500):
        df = self.uni[sym].tail(limit)
        return [dict(time=int(pd.Timestamp(t).timestamp()), open=o, high=h, low=l, close=c, volume=v)
                for t, o, h, l, c, v in df[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False)]

    def history(self, syms, days=150):
        return {s: self.uni[s].tail(days + 1).reset_index(drop=True) for s in syms if s in self.uni}

    def funding_rate(self, sym):
        h8 = 8 * 3600_000
        return dict(rate=0.0001, next=(int(time.time() * 1000) // h8 + 1) * h8, mark=None, interval="8h")

    def funding_events(self, sym, since_ms, until_ms):
        h8 = 8 * 3600_000
        return [(t, 0.0001) for t in range((since_ms // h8 + 1) * h8, until_ms + 1, h8)]


# ------------------------------------------------------------------ análisis del modelo
def compute_analysis(bundle, hist, progress=lambda *_: None, derivs=None):
    idx, P, F, liq_ok = xs_core.features(hist, bundle.get("variant", "D"), derivs)
    names = bundle["features"]
    cols = list(P["close"].columns)
    nan = np.full(len(cols), np.nan)
    X = np.stack([F[k].to_numpy()[-1] if k in F else nan for k in names], axis=-1)
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
    K = xs_core.k_for(bundle["K"], len(iv), bundle.get("kfrac"))
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
    def __init__(self, cfg, demo=False, data=None):
        self.cfg, self.demo, self.data = cfg, demo, data
        self.src = DemoSource(data) if demo else LiveSource(cfg["exchange"])
        self.cache = {}
        self.analysis = json.load(open(ANALYSIS)) if os.path.exists(ANALYSIS) else None
        self.an_status = dict(state="idle", msg="")
        self.job = dict(running=False, name="", lines=[], code=None)
        self.lock = threading.Lock()
        self.events = []                                     # avisos para el navegador (llenados, TP/SL, liquidaciones, funding)
        self.acct = dc.AccountData(cfg)                       # comisiones / escalones / modo de posición de TU cuenta
        self.paper = cp.PaperAccount(MANUAL, cfg, self.acct)
        if not demo and self.acct.stale():
            threading.Thread(target=self._refresh_acct_bg, daemon=True).start()
        self.exacc = None
        self.last_funding_check = 0.0
        threading.Thread(target=self._engine, daemon=True).start()

    # ---------------------------------------------------------------- utilidades
    def cached(self, key, ttl, fn):
        now = time.time()
        hit = self.cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        val = fn()
        self.cache[key] = (now, val)
        return val

    def note(self, msg, level="info"):
        self.events.append(dict(t=time.time(), msg=msg, level=level))
        self.events = self.events[-200:]

    def account(self):
        """Cuenta manual activa: la simulada (paper) o la del exchange (demo/real) con tus claves."""
        if self.cfg["mode"] == "paper":
            return self.paper
        if self.exacc is None:
            self.exacc = cr.ExchangeAccount(self.cfg, hedged=self.acct.d.get("hedged"))
        return self.exacc

    def markets(self):
        return self.cached("markets", 3600, self.src.markets)

    def set_config(self, new):
        old = self.cfg
        cfg = cl.save_config({**old, **new})
        self.cfg = cfg
        self.paper.cfg = cfg
        self.acct.cfg = cfg
        if cfg["exchange"] != old["exchange"] and not self.demo:
            self.src = LiveSource(cfg["exchange"])
            self.cache.clear()
            self.acct = dc.AccountData(cfg)
            self.paper.data = self.acct
            threading.Thread(target=self._refresh_acct_bg, daemon=True).start()
        if cfg["exchange"] != old["exchange"] or cfg["mode"] != old["mode"]:
            self.exacc = None
        if cfg["bot_variant"] != old["bot_variant"]:
            use_variant(cfg["bot_variant"])
            self.analysis = json.load(open(ANALYSIS)) if os.path.exists(ANALYSIS) else None
            if DEMO and not os.path.exists(STATE):
                st, lg = build_demo_state(self.src.uni, cfg["bot_capital"])
                json.dump(st, open(STATE, "w"))
                lg.to_csv(LOG, index=False)
        if cfg["universe"] != old["universe"] or cfg["bot_variant"] != old["bot_variant"]:
            self.start_analysis(force=True)
        return cfg

    # ---------------------------------------------------------------- motor de la cuenta paper manual
    def _engine(self):
        while True:
            try:
                if self.cfg["mode"] == "paper":
                    syms = set(self.paper.s["positions"]) | {o["symbol"] for o in self.paper.s["orders"]}
                    if syms:
                        quotes = self.src.tickers(sorted(syms))
                        for e in self.paper.on_tick(quotes):
                            self.note(e, "warn" if "LIQUID" in e or "stop" in e else "ok")
                        if time.time() - self.last_funding_check > 600:   # funding real cada 10 min
                            self.last_funding_check = time.time()
                            now = int(time.time() * 1000)
                            for sym, p in list(self.paper.s["positions"].items()):
                                ev = self.src.funding_events(sym, p["last_funding"], now)
                                if ev and sym in quotes:
                                    c = self.paper.apply_funding(sym, ev, quotes[sym]["price"])
                                    self.note(f"funding {sym.split('/')[0]}: {'pagaste' if c > 0 else 'cobraste'} {abs(c):.4f} USDT", "info")
            except Exception as e:                           # noqa: BLE001
                self.note(f"motor paper: {type(e).__name__}: {str(e)[:120]}", "err")
            time.sleep(3)

    # ---------------------------------------------------------------- bot (estrategia)
    def bot_state(self):
        st = json.load(open(STATE)) if os.path.exists(STATE) else {"equity": self.cfg["bot_capital"], "cohorts": [], "closed": []}
        log = pd.read_csv(LOG).to_dict("records") if os.path.exists(LOG) else []
        cl_ = st["closed"]
        r = np.array([c["ret"] for c in cl_]) if cl_ else np.array([])
        stats = None
        if len(r):
            n_eff = max(len(r) / 7, 1)
            se = r.std(ddof=1) / np.sqrt(n_eff) if len(r) > 1 else float("nan")
            stats = dict(mean=float(r.mean()), lo=float(r.mean() - 1.96 * se), hi=float(r.mean() + 1.96 * se), n=len(r),
                         n_eff=n_eff, win=float((r > 0).mean()), best=float(r.max()), worst=float(r.min()),
                         funding=float(sum(c.get("funding", 0.0) for c in cl_)))
        model = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(MODEL))) if os.path.exists(MODEL) else None
        return dict(state=st, log=log, stats=stats, capital=self.cfg["bot_capital"], model=model, backtest=BACKTEST_REF, variant=self.cfg["bot_variant"],
                    today_done=any(c["opened"] == time.strftime("%Y-%m-%d", time.gmtime()) for c in st["cohorts"]))

    def bot_close(self, symbol, opened):
        """Cierre anticipado de una posición del bot (paper): precio vivo + deslizamiento + comisión + funding real."""
        st = json.load(open(STATE))
        for c in st["cohorts"]:
            if c["opened"] != opened:
                continue
            for p in list(c["positions"]):
                if p["symbol"] != symbol:
                    continue
                q = self.src.tickers([symbol]).get(symbol)
                if not q:
                    raise ValueError("sin precio para cerrar")
                long = p["side"] == "LONG"
                px = q["bid"] * (1 - self.cfg["slippage"]) if long else q["ask"] * (1 + self.cfg["slippage"])
                pnl = p["notional"] * (1 if long else -1) * (px / p["entry"] - 1) - p["notional"] * self.cfg["taker"]
                t0 = int(pd.Timestamp(opened, tz="UTC").value // 10**6)
                fund = sum((1 if long else -1) * p["notional"] * r for _, r in self.src.funding_events(symbol, t0, int(time.time() * 1000)))
                pnl -= fund
                c["positions"].remove(p)
                st["equity"] += pnl
                st.setdefault("early_closed", []).append(dict(symbol=symbol, side=p["side"], opened=opened, closed=time.strftime("%Y-%m-%d %H:%M"),
                                                              entry=p["entry"], exit=px, pnl=pnl, funding=fund))
                json.dump(st, open(STATE, "w"), indent=1)
                return f"posición del bot {symbol.split('/')[0]} cerrada antes de tiempo: PnL {pnl:+.2f} USDT (funding {-fund:+.4f})"
        raise ValueError("posición no encontrada")

    # ---------------------------------------------------------------- precios / velas / info
    def symbols_of_interest(self):
        st = json.load(open(STATE)) if os.path.exists(STATE) else {"cohorts": []}
        s = {p["symbol"] for c in st["cohorts"] for p in c["positions"]}
        s |= set(self.paper.s["positions"]) | {o["symbol"] for o in self.paper.s["orders"]}
        if self.analysis:
            s |= {r["symbol"] for r in self.analysis["ranking"][:15]} | {r["symbol"] for r in self.analysis["ranking"][-15:]}
        s.add("BTC/USDT:USDT")
        return sorted(s)

    def prices(self, extra=None):
        syms = sorted(set(self.symbols_of_interest() + ([extra] if extra else [])))
        return self.cached(("px", tuple(syms)), 4, lambda: self.src.tickers(syms))

    def candles(self, sym, tf):
        ttl = 15 if tf in ("1m", "5m", "15m", "1h") else 60
        return self.cached(("c", sym, tf), ttl, lambda: self.src.candles(sym, tf if not self.demo else "1d"))

    def _refresh_acct(self):
        try:
            st = self.acct.refresh(self.symbols_of_interest()[:30])
            self.note(f"datos de tu cuenta ({st['source']}): comisión taker {st['taker']:.4%} · escalones de {st['n_tiers']} símbolos"
                      if st["taker"] is not None else f"datos de tu cuenta ({st['source']}): escalones de {st['n_tiers']} símbolos", "ok")
            return st
        except Exception as e:                               # noqa: BLE001
            if cl.load_keys(self.cfg["exchange"], "real") or cl.load_keys(self.cfg["exchange"], "demo"):
                self.note(f"no se pudieron leer los datos de tu cuenta: {str(e)[:160]}", "warn")
            raise

    def _refresh_acct_bg(self):
        try:
            self._refresh_acct()
        except Exception:
            pass                                             # ya quedó el aviso; se usan los valores de la configuración

    def symbol_info(self, sym):
        m = self.markets().get(sym, {})
        if self.acct.d and sym not in self.acct.d.get("fees", {}):
            threading.Thread(target=self.acct.fetch_fee, args=(sym,), daemon=True).start()
        maker, taker, fsrc = self.acct.fee_for(sym)
        t = self.src.tickers([sym]).get(sym, {})
        f = self.cached(("f", sym), 60, lambda: self.src.funding_rate(sym))
        lev = ((m.get("limits") or {}).get("leverage") or {}).get("max")
        lim = m.get("limits") or {}
        tiers = self.acct.tiers_for(sym)
        return dict(symbol=sym, cls=universo.classify(sym, m), ticker=t, funding=f, max_leverage=(tiers[0][2] if tiers else lev),
                    fee=dict(maker=maker, taker=taker, source=fsrc), tiers=tiers, tiers_source="tu cuenta" if tiers else None,
                    min_cost=(lim.get("cost") or {}).get("min"), contract_size=m.get("contractSize"))

    def market_list(self):
        mk = self.markets()
        return sorted([dict(symbol=s, cls=universo.classify(s, m)) for s, m in mk.items()
                       if universo.allowed(s, self.cfg["universe"], m)], key=lambda x: x["symbol"])

    # ---------------------------------------------------------------- análisis del modelo
    def start_analysis(self, force=False):
        with self.lock:
            if self.an_status["state"] == "running":
                return
            fresh = self.analysis and self.analysis.get("generated", "")[:10] == time.strftime("%Y-%m-%d") \
                and self.analysis.get("universe", "both") == self.cfg["universe"] \
                and self.analysis.get("variant", "D") == self.cfg["bot_variant"]
            if not force and fresh:
                return
            if not os.path.exists(MODEL):
                self.an_status = dict(state="error", msg=f"No hay modelo {self.cfg['bot_variant']} entrenado: pulsa «Reentrenar».")
                return
            self.an_status = dict(state="running", msg="descargando velas diarias de todas las monedas…")

        def work():
            try:
                bundle = pickle.load(open(MODEL, "rb"))
                syms = universo.filter_symbols(bundle["symbols"], self.cfg["universe"], self.markets())
                hist = self.src.history(syms)
                derivs = None
                if bundle.get("variant") == "FO":
                    self.an_status["msg"] = "actualizando funding y open interest…"
                    derivs = derivs_live.load_seed() if self.demo else derivs_live.update(self.src.ex, syms, log=lambda *_: None)
                self.an_status["msg"] = f"calculando modelo sobre {len(hist)} monedas…"
                a = compute_analysis(bundle, hist, progress=lambda m: self.an_status.update(msg=m), derivs=derivs)
                a["universe"] = self.cfg["universe"]
                a["variant"] = bundle.get("variant", "D")
                json.dump(a, open(ANALYSIS, "w"))
                self.analysis = a
                self.an_status = dict(state="done", msg="")
            except Exception as e:                           # noqa: BLE001
                self.an_status = dict(state="error", msg=f"{type(e).__name__}: {str(e)[:200]}")
        threading.Thread(target=work, daemon=True).start()

    # ---------------------------------------------------------------- tareas (bot / reentrenar)
    def run_job(self, name, args, steps=None):
        """Ejecuta uno o varios scripts seguidos (steps = lista de listas de argumentos); se detiene en el primer error."""
        if self.job["running"]:
            return False
        self.job = dict(running=True, name=name, lines=[], code=None)

        def work():
            try:
                for st in steps or [args]:
                    self.job["lines"].append(f"$ {os.path.basename(st[0])} {' '.join(st[1:])}")
                    p = subprocess.Popen([sys.executable, "-u", *st], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         cwd=ROOT, creationflags=NO_WINDOW, encoding="utf-8", errors="replace")
                    for line in p.stdout:
                        self.job["lines"].append(line.rstrip("\n"))
                    p.wait()
                    self.job["code"] = p.returncode
                    if p.returncode != 0:
                        break
            except Exception as e:                           # noqa: BLE001
                self.job["lines"].append(f"ERROR: {e}")
                self.job["code"] = 1
            self.job["running"] = False
            if name == "train" and self.job["code"] == 0:
                self.start_analysis(force=True)
        threading.Thread(target=work, daemon=True).start()
        return True

    def bot_args(self):
        c = self.cfg
        return [os.path.join(HERE, "run_paper.py"), "--force", "--capital", str(c["bot_capital"]), "--exchange", c["exchange"],
                "--universe", c["universe"], "--fee", str(self.acct.fee_for("BTC/USDT:USDT")[1]), "--slip", str(c["slippage"]), "--state", STATE, "--log", LOG,
                "--variant", c["bot_variant"], "--model", MODEL]

    # ---------------------------------------------------------------- reinicio con copia de seguridad
    def reset(self, what, capital):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bk = os.path.join(HERE, "backups", stamp)
        os.makedirs(bk, exist_ok=True)
        for f in (STATE, LOG, MANUAL):
            if os.path.exists(f):
                shutil.copy2(f, bk)
        done = []
        if what in ("bot", "all"):
            for f in (STATE, LOG):
                if os.path.exists(f):
                    os.remove(f)
            self.cfg = cl.save_config({**self.cfg, "bot_capital": capital})
            done.append(f"bot reiniciado con {capital:.2f} USDT")
        if what in ("manual", "all"):
            self.cfg = cl.save_config({**self.cfg, "manual_capital": capital})
            self.paper.cfg = self.cfg
            self.paper.reset(capital)
            done.append(f"cuenta manual reiniciada con {capital:.2f} USDT")
        return " | ".join(done) + f" (copia de seguridad en backups/{stamp})"


def make_handler(hub: Hub):
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

        def _local_only(self):                               # el servidor solo escucha en 127.0.0.1; doble control del Origin
            o = self.headers.get("Origin")
            return not o or o.startswith(("http://127.0.0.1", "http://localhost"))

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
                    return self._send(200, dict(bot=hub.bot_state(), cfg=hub.cfg, demo=hub.demo, source=hub.src.name,
                                                events=[e for e in hub.events if e["t"] > float(q.get("since", 0))]))
                if u.path == "/api/account":
                    acc = hub.account()
                    syms = list(hub.paper.s["positions"]) if hub.cfg["mode"] == "paper" else []
                    quotes = hub.src.tickers(syms) if syms else {}
                    return self._send(200, hub.cached(("acc", hub.cfg["mode"]), 2 if hub.cfg["mode"] == "paper" else 4, lambda: acc.snapshot(quotes)))
                if u.path == "/api/prices":
                    return self._send(200, hub.prices(q.get("extra")))
                if u.path == "/api/candles":
                    return self._send(200, hub.candles(q["symbol"], q.get("tf", "1h")))
                if u.path == "/api/symbol":
                    return self._send(200, hub.symbol_info(q["symbol"]))
                if u.path == "/api/account_data":
                    return self._send(200, hub.acct.status())
                if u.path == "/api/markets":
                    return self._send(200, hub.market_list())
                if u.path == "/api/config":
                    return self._send(200, dict(cfg=hub.cfg, keys=cl.keys_status(hub.cfg["exchange"], hub.cfg["mode"]),
                                                keys_demo=cl.keys_status(hub.cfg["exchange"], "demo"),
                                                keys_real=cl.keys_status(hub.cfg["exchange"], "real"), config_dir=cl.DIR))
                if u.path == "/api/analysis":
                    if q.get("refresh"):
                        hub.start_analysis(force=True)
                    else:
                        hub.start_analysis()
                    return self._send(200, dict(status=hub.an_status, analysis=hub.analysis))
                if u.path == "/api/job":
                    return self._send(200, hub.job)
                if u.path == "/api/export":
                    rows = hub.paper.s["history"] if q.get("what") == "manual" else hub.bot_state()["state"]["closed"]
                    csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/csv")
                    self.send_header("Content-Disposition", f"attachment; filename=historial_{q.get('what', 'bot')}.csv")
                    self.end_headers()
                    self.wfile.write(csv)
                    return None
                return self._send(404, {"error": "no encontrado"})
            except Exception as e:                           # noqa: BLE001
                return self._send(500, {"error": f"{type(e).__name__}: {str(e)[:300]}"})

        def do_POST(self):
            if not self._local_only():
                return self._send(403, {"error": "origen no permitido"})
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            b = json.loads(self.rfile.read(n) or b"{}") if n else {}
            try:
                if u.path == "/api/run":
                    ok = hub.run_job("run", hub.bot_args())
                    return self._send(200 if ok else 409, {"started": ok})
                if u.path == "/api/train":
                    tr = [os.path.join(HERE, "train.py"), "--variant", hub.cfg["bot_variant"]]
                    steps = ([[os.path.join(HERE, "update_data.py")] + (["--no-derivs"] if hub.cfg["bot_variant"] == "D" else [])] if b.get("update") else []) + [tr]
                    ok = hub.run_job("train", None, steps)
                    return self._send(200 if ok else 409, {"started": ok})
                if u.path == "/api/config":
                    new = b.get("cfg", {})
                    if new.get("mode") == "real" and hub.cfg["mode"] != "real" and b.get("confirm") != "CONFIRMO":
                        raise ValueError("para activar el modo REAL escribe CONFIRMO")
                    cfg = hub.set_config(new)
                    return self._send(200, {"ok": True, "cfg": cfg, "msg": "configuración guardada"})
                if u.path == "/api/keys":
                    where = cl.save_keys(hub.cfg["exchange"], b.get("mode", hub.cfg["mode"]), b.get("apiKey", ""), b.get("secret", ""), b.get("password", ""))
                    hub.exacc = None
                    if not hub.demo:
                        threading.Thread(target=hub._refresh_acct_bg, daemon=True).start()
                    return self._send(200, {"ok": True, "msg": f"claves guardadas en tu PC ({where}); leyendo datos de tu cuenta…"})
                if u.path == "/api/keys/delete":
                    cl.delete_keys(hub.cfg["exchange"], b.get("mode", hub.cfg["mode"]))
                    hub.exacc = None
                    return self._send(200, {"ok": True, "msg": "claves borradas"})
                if u.path == "/api/keys/test":
                    mode = b.get("mode", hub.cfg["mode"])
                    keys = cl.load_keys(hub.cfg["exchange"], mode)
                    if not keys:
                        raise ValueError(f"no hay claves guardadas para la cuenta {mode.upper()}")
                    res = cr.diagnose(hub.cfg, keys)
                    good = [r for r in res if r["ok"]]
                    want = "real" if mode == "real" else "demo"
                    ok = any(r["env"] == want for r in good)
                    if ok:
                        msg = f"Conexión OK en {mode.upper()}: " + next(r["msg"] for r in good if r["env"] == want)
                    elif good:
                        msg = (f"Estas claves NO funcionan en {mode.upper()}, pero sí en {', '.join(r['label'] for r in good)}. "
                               f"Guárdalas en la cuenta correcta (o crea claves de {mode.upper()}).")
                    elif all(r["msg"].startswith(("NetworkError", "RequestTimeout", "ExchangeNotAvailable")) for r in res):
                        msg = "No hay conexión con el exchange (internet, VPN o DNS): no se pudo comprobar nada."
                    else:
                        msg = ("Las claves no funcionan en ningún entorno. Revisa en el exchange: 1) que copiaste bien la clave y el secreto, "
                               "2) que la clave tenga activado el permiso de Futuros (Enable Futures), "
                               "3) si la limitaste por IP, que incluya la IP pública de tu PC, "
                               "4) para DEMO, que las creaste en demo.binance.com (Futuros demo), no en la cuenta normal.")
                    return self._send(200, {"ok": ok, "msg": msg, "results": res})
                if u.path == "/api/account_data/refresh":
                    st = hub._refresh_acct()
                    return self._send(200, {"ok": True, "msg": "datos de tu cuenta actualizados", "status": st})
                if u.path == "/api/train_info":
                    import update_data
                    info = dict(variant=hub.cfg["bot_variant"], model_until=None)
                    if os.path.exists(MODEL):
                        info["model_until"] = pickle.load(open(MODEL, "rb")).get("trained_until")
                    ld = update_data.last_day()
                    info["data_until"] = str(ld.date()) if ld is not None else None
                    info["new_days"] = (pd.Timestamp(info["data_until"]) - pd.Timestamp(info["model_until"])).days if info["model_until"] and info["data_until"] else None
                    info["stale_days"] = (pd.Timestamp.now(tz="UTC").normalize().tz_localize(None) - pd.Timestamp(info["data_until"])).days - 1 if info["data_until"] else None
                    return self._send(200, info)
                if u.path == "/api/order":
                    mode = hub.cfg["mode"]
                    if mode == "real" and not b.get("confirm"):
                        raise ValueError("las órdenes reales deben confirmarse")
                    sym = b["symbol"]
                    quote = hub.src.tickers([sym]).get(sym)
                    if not quote:
                        raise ValueError("sin precio en vivo para ese símbolo")
                    tp = float(b["tp"]) if b.get("tp") else None
                    sl = float(b["sl"]) if b.get("sl") else None
                    msg = hub.account().place(sym, b["side"], b.get("type", "market"), float(b["usdt"]), int(b["leverage"]), quote,
                                              float(b["price"]) if b.get("price") else None, tp, sl)
                    hub.note(f"[{mode.upper()}] {msg}", "ok")
                    hub.cache.pop(("acc", mode), None)
                    return self._send(200, {"ok": True, "msg": msg})
                if u.path == "/api/close":
                    sym = b["symbol"]
                    if b.get("source") == "bot":
                        msg = hub.bot_close(sym, b["opened"])
                    else:
                        quote = hub.src.tickers([sym]).get(sym)
                        msg = hub.account().close(sym, float(b.get("fraction", 1.0)), quote)
                    hub.note(msg, "ok")
                    hub.cache.pop(("acc", hub.cfg["mode"]), None)
                    return self._send(200, {"ok": True, "msg": msg})
                if u.path == "/api/cancel":
                    acc = hub.account()
                    msg = acc.cancel(b["id"]) if hub.cfg["mode"] == "paper" else acc.cancel(b["id"], b["symbol"])
                    hub.cache.pop(("acc", hub.cfg["mode"]), None)
                    return self._send(200, {"ok": True, "msg": msg})
                if u.path == "/api/tpsl":
                    if hub.cfg["mode"] != "paper":
                        raise ValueError("en demo/real crea el TP/SL desde el formulario de orden o en el exchange")
                    msg = hub.paper.set_tpsl(b["symbol"], float(b["tp"]) if b.get("tp") else None, float(b["sl"]) if b.get("sl") else None)
                    return self._send(200, {"ok": True, "msg": msg})
                if u.path == "/api/reset":
                    msg = hub.reset(b.get("what", "manual"), float(b.get("capital", 1000)))
                    hub.note(msg, "warn")
                    return self._send(200, {"ok": True, "msg": msg})
                return self._send(404, {"error": "no encontrado"})
            except Exception as e:                           # noqa: BLE001
                return self._send(400, {"ok": False, "error": f"{str(e)[:300]}" if isinstance(e, (ValueError, KeyError)) else f"{type(e).__name__}: {str(e)[:300]}"})
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
    global MANUAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default=None, help="sobrescribe el exchange de la configuración")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--demo", action="store_true", help="sin internet: datos locales")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "ohlcv_daily_long.pkl"))
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    cfg = cl.load_config()
    if a.exchange:
        cfg["exchange"] = a.exchange
    if a.demo:                                               # el demo nunca toca tus archivos ni tus claves
        MANUAL = os.path.join(HERE, "demo_manual.json")
        cfg["mode"] = "paper"
    use_variant(cfg["bot_variant"], a.demo)
    hub = Hub(cfg, a.demo, a.data)
    if a.demo and not os.path.exists(STATE):
        st, lg = build_demo_state(hub.src.uni, cfg["bot_capital"])
        json.dump(st, open(STATE, "w"))
        lg.to_csv(LOG, index=False)
    hub.start_analysis()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(hub))
    url = f"http://127.0.0.1:{a.port}"
    print(f"Panel en {url}  (Ctrl+C para cerrar) | fuente: {hub.src.name} | modo de la cuenta manual: {cfg['mode'].upper()}")
    print(f"Configuración y claves en {cl.DIR}")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
