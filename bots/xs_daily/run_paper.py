"""Bot cross-sectional diario en PAPER TRADING (no envía órdenes reales).

Cada ejecución (una vez al día, después de las 00:00 UTC):
  1. cierra las cohortes con 7 días o más al precio en vivo (long vende al bid, short recompra al ask, + slippage y comisión);
  2. abre una cohorte nueva: long los 10 mejores y short los 10 peores según el modelo, con 1/7 del capital;
  3. valora lo abierto y registra todo en el log CSV.
Solo opera si la estrategia figura en strategy_selection.json como aprobada o en observación (o con --force).
Uso: python bots/xs_daily/run_paper.py [--exchange binanceusdm|okx] [--capital 1000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import funding_util  # noqa: E402
import universo  # noqa: E402
import xs_core  # noqa: E402

FEE, SLIP = 0.0005, 0.0002
STOP_SLIP = 0.0010                    # deslizamiento extra al ejecutarse un stop (igual que en el backtest)
SL_ATR = 2.0                          # stop loss = entrada ∓ SL_ATR × ATR(14)% (0 = sin stop). Ver backtesting/xs_stops.py
HOUR = 3_600_000
STRATEGY_NAME = "modelo:xs_v2_alpha158_rank_7d"
DAY = 86_400_000


def fetch_history(ex, symbols, days=150):
    now = ex.milliseconds()
    out = {}
    for s in symbols:
        try:
            rows = ex.fetch_ohlcv(s, "1d", limit=days + 1)
        except Exception as e:
            print(f"  {s}: sin datos ({type(e).__name__})")
            continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df[df.ts + DAY <= now]                                   # solo velas cerradas
        if len(df) >= 100:
            df["ts"] = pd.to_datetime(df.ts, unit="ms", utc=True)
            out[s] = df.reset_index(drop=True)
    return out


def exec_price(ex, symbol, side, opening):
    """Precio simulado en vivo: compras al ask, ventas al bid, más slippage adverso. None si no hay precio."""
    try:
        t = ex.fetch_ticker(symbol)
    except Exception:
        return None
    buy = (side == "LONG") == opening
    px = (t.get("ask") if buy else t.get("bid")) or t.get("last")
    if not px:
        return None
    return float(px) * (1 + SLIP) if buy else float(px) * (1 - SLIP)


def load_state(path, capital):
    if os.path.exists(path):
        return json.load(open(path))
    return {"equity": capital, "cohorts": [], "closed": []}


def atr14(df):
    """ATR de 14 días en % del precio, con las velas diarias cerradas hasta la señal."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    v = tr.rolling(14, min_periods=14).mean().iloc[-1] / c.iloc[-1]
    return float(v) if pd.notna(v) else None


def check_stops(state, ex, log=print):
    """Revisa con velas de 1 h si alguna posición tocó su stop desde la última revisión. Si lo tocó, queda cerrada
    a ese precio (o al open si abrió más allá: hueco) con deslizamiento de stop; se realiza al cerrar la cohorte."""
    hits = 0
    for c in state["cohorts"]:
        t0 = c.get("opened_ts") or int(pd.Timestamp(c["opened"], tz="UTC").value // 10**6)
        for p in c["positions"]:
            if not p.get("sl") or p.get("exit"):
                continue
            since = p.get("checked_ts", t0)
            try:
                rows = ex.fetch_ohlcv(p["symbol"], "1h", since=since - since % HOUR, limit=500)
            except Exception:
                continue
            long = p["side"] == "LONG"
            for ts, o, hi, lo, cl, v in rows:
                if ts + HOUR <= since:                      # vela anterior a la entrada / a la última revisión
                    continue
                if (long and lo <= p["sl"]) or (not long and hi >= p["sl"]):
                    base_px = min(o, p["sl"]) if long else max(o, p["sl"])
                    p["exit"] = base_px * (1 - STOP_SLIP) if long else base_px * (1 + STOP_SLIP)
                    p["exit_ts"], p["exit_reason"] = int(ts), "stop loss"
                    hits += 1
                    log(f"  STOP {p['side']} {p['symbol'].split('/')[0]} (cohorte {c['opened']}): salida {p['exit']:.6g} "
                        f"({(1 if long else -1) * (p['exit'] / p['entry'] - 1):+.2%})")
                    break
            if rows and not p.get("exit"):
                p["checked_ts"] = int(rows[-1][0])
    return hits


def step(state, ex, bundle, today, log=print, derivs=None):
    check_stops(state, ex, log)
    realized = 0.0
    keep = []
    for c in state["cohorts"]:
        age = (pd.Timestamp(today) - pd.Timestamp(c["opened"])).days
        if age < bundle["H"]:
            keep.append(c)
            continue
        pnl, fund, fund_ok = 0.0, 0.0, True
        t_open = int(pd.Timestamp(c["opened"], tz="UTC").value // 10**6)
        for p in c["positions"]:
            px = p.get("exit") or exec_price(ex, p["symbol"], p["side"], opening=False) or p["entry"]   # si tocó el stop, sale a ese precio
            sgn = 1 if p["side"] == "LONG" else -1
            pnl += p["notional"] * sgn * (px / p["entry"] - 1) - p["notional"] * FEE
            if hasattr(ex, "fetch_funding_rate_history"):       # funding pagado/cobrado mientras la posición estuvo abierta
                until = p.get("exit_ts") or int(pd.Timestamp(today, tz="UTC").value // 10**6)
                f, ok = funding_util.funding_cost(ex, p["symbol"], p["side"], p["notional"], t_open, until)
                fund += f
                fund_ok &= ok
        pnl -= fund
        realized += pnl
        state["closed"].append({"opened": c["opened"], "closed": today, "pnl": pnl, "capital": c["capital"],
                                "ret": pnl / c["capital"] if c["capital"] else 0.0, "funding": fund, "funding_ok": fund_ok,
                                "stops": sum(1 for p in c["positions"] if p.get("exit"))})
        log(f"  cohorte del {c['opened']} cerrada: PnL {pnl:+.2f} ({pnl / c['capital']:+.2%}) | funding {-fund:+.2f}"
            + ("" if fund_ok else " (sin datos de funding para algún símbolo)"))
    state["cohorts"] = keep
    state["equity"] += realized
    if any(c["opened"] == today for c in state["cohorts"]):
        log("  ya se abrió la cohorte de hoy; no se abre otra")
        return realized
    hist = fetch_history(ex, bundle["symbols"])
    date, longs, shorts, _ = xs_core.pick(bundle, hist, derivs)
    if not longs:
        log("  datos insuficientes para elegir posiciones; no se abre cohorte")
        return realized
    cap = state["equity"] / bundle["H"]
    notional = cap * 0.5 / len(longs)                       # K puede ser menor con universos pequeños
    positions = []
    for side, syms in (("LONG", longs), ("SHORT", shorts)):
        for s in syms:
            px = exec_price(ex, s, side, opening=True)
            if px is None:
                continue
            pos = {"symbol": s, "side": side, "entry": px, "notional": notional, "fee": notional * FEE}
            a = atr14(hist[s]) if (SL_ATR and s in hist) else None
            if a:
                pos["atr"] = a
                pos["sl"] = px * (1 - SL_ATR * a) if side == "LONG" else px * (1 + SL_ATR * a)
            positions.append(pos)
            state["equity"] -= notional * FEE
    state["cohorts"].append({"opened": today, "opened_ts": int(ex.milliseconds()), "signal_date": str(date.date()), "capital": cap,
                             "positions": positions})   # opened_ts = momento real de la entrada (para marcarla en el gráfico)
    log(f"  cohorte abierta ({len(positions)} posiciones, {notional:.2f} c/u, stop {SL_ATR:g}×ATR" + (")" if SL_ATR else " = sin stop)") + f": LONG {', '.join(s.split('/')[0] for s in longs)} | "
        f"SHORT {', '.join(s.split('/')[0] for s in shorts)}")
    return realized


def unrealized(state, ex):
    u = 0.0
    for c in state["cohorts"]:
        for p in c["positions"]:
            px = p.get("exit") or exec_price(ex, p["symbol"], p["side"], opening=False)
            if px:
                u += p["notional"] * (1 if p["side"] == "LONG" else -1) * (px / p["entry"] - 1)
    return u


def paths(variant):
    """(modelo, estado, log) por variante. D conserva los nombres originales para no perder tu historial."""
    if variant == "D":
        return os.path.join(HERE, "model_xs_D.pkl"), os.path.join(HERE, "paper_state.json"), os.path.join(HERE, "paper_log.csv")
    return (os.path.join(HERE, f"model_xs_{variant}.pkl"), os.path.join(HERE, f"paper_state_{variant}.json"),
            os.path.join(HERE, f"paper_log_{variant}.csv"))


def main():
    global FEE, SLIP, SL_ATR
    import ccxt
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--variant", default="D", choices=xs_core.VARIANTS, help="D (original) o FO (con funding y open interest)")
    ap.add_argument("--model", default=None, help="por defecto model_xs_<variante>.pkl")
    ap.add_argument("--state", default=None, help="por defecto paper_state.json (D) / paper_state_FO.json (FO)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--universe", default="both", choices=["crypto", "stocks", "both"], help="qué opera el bot: cripto, acciones tokenizadas o ambos")
    ap.add_argument("--fee", type=float, default=FEE, help="comisión taker por lado")
    ap.add_argument("--sl-atr", type=float, default=SL_ATR, help="stop loss en múltiplos de ATR(14) (0 = sin stop)")
    ap.add_argument("--slip", type=float, default=SLIP, help="deslizamiento por lado")
    a = ap.parse_args()
    sel_path = os.path.join(xs_core.ROOT, "strategy_selection.json")
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {}
    allowed = STRATEGY_NAME in sel.get("enabled", []) + sel.get("watchlist", [])
    if not allowed and not a.force:
        sys.exit(f"{STRATEGY_NAME} no está aprobada ni en observación en strategy_selection.json; no se opera (usa --force para paper de todos modos).")
    print("MODO PAPER: no se envían órdenes reales.", "Estado en el selector:", "aprobada" if STRATEGY_NAME in sel.get("enabled", []) else "en observación")
    a.model, a.state, a.log = (a.model or paths(a.variant)[0], a.state or paths(a.variant)[1], a.log or paths(a.variant)[2])
    bundle = pickle.load(open(a.model, "rb"))
    if bundle.get("variant", "D") != a.variant:
        sys.exit(f"el modelo {a.model} es de la variante {bundle.get('variant', 'D')}, no {a.variant}: entrénalo con train.py --variant {a.variant}")
    ex = getattr(ccxt, a.exchange)({"enableRateLimit": True})
    for k in range(6):                                           # reintentos: fallos de DNS/red pasajeros no deben tumbar la ejecución diaria
        try:
            ex.load_markets()
            break
        except Exception as e:
            if k == 5:
                sys.exit(f"Sin conexión con {a.exchange} tras 6 intentos ({type(e).__name__}). Revisa internet/VPN/DNS y vuelve a ejecutar.")
            print(f"  sin conexión ({type(e).__name__}), reintento {k + 1}/5 en {5 * (k + 1)} s ...", flush=True)
            time.sleep(5 * (k + 1))
    FEE, SLIP, SL_ATR = a.fee, a.slip, a.sl_atr
    bundle["symbols"] = universo.filter_symbols([s for s in bundle["symbols"] if s in ex.markets], a.universe, ex.markets)
    print(f"universo: {a.universe} -> {len(bundle['symbols'])} símbolos | comisión {FEE:.4%} | deslizamiento {SLIP:.4%}")
    state = load_state(a.state, a.capital)
    today = time.strftime("%Y-%m-%d", time.gmtime())
    derivs = None
    if a.variant == "FO":                                    # funding y OI recientes (API) sobre el historial local
        import derivs_live
        derivs = derivs_live.update(ex, bundle["symbols"])
    print(f"variante {a.variant} | modelo entrenado hasta {bundle.get('trained_until')} | estado {os.path.basename(a.state)}")
    step(state, ex, bundle, today, derivs=derivs)
    u = unrealized(state, ex)
    json.dump(state, open(a.state, "w"), indent=1)
    new = not os.path.exists(a.log)
    if not new and any(line.startswith(today + ",") for line in open(a.log)):
        print(f"equity realizada {state['equity']:.2f} | no realizado {u:+.2f} (el log ya tiene una fila de hoy)")
        return
    with open(a.log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["fecha", "equity_realizada", "no_realizado", "equity_total", "cohortes_abiertas"])
        w.writerow([today, f"{state['equity']:.2f}", f"{u:.2f}", f"{state['equity'] + u:.2f}", len(state["cohorts"])])
    print(f"equity realizada {state['equity']:.2f} | no realizado {u:+.2f} | total {state['equity'] + u:.2f} | cohortes abiertas {len(state['cohorts'])}")


if __name__ == "__main__":
    main()
