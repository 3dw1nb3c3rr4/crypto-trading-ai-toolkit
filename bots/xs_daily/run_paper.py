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


def step(state, ex, bundle, today, log=print, derivs=None):
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
            px = exec_price(ex, p["symbol"], p["side"], opening=False) or p["entry"]
            sgn = 1 if p["side"] == "LONG" else -1
            pnl += p["notional"] * sgn * (px / p["entry"] - 1) - p["notional"] * FEE
            if hasattr(ex, "fetch_funding_rate_history"):       # funding pagado/cobrado durante los 7 días
                f, ok = funding_util.funding_cost(ex, p["symbol"], p["side"], p["notional"], t_open, int(pd.Timestamp(today, tz="UTC").value // 10**6))
                fund += f
                fund_ok &= ok
        pnl -= fund
        realized += pnl
        state["closed"].append({"opened": c["opened"], "closed": today, "pnl": pnl, "capital": c["capital"],
                                "ret": pnl / c["capital"] if c["capital"] else 0.0, "funding": fund, "funding_ok": fund_ok})
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
            positions.append({"symbol": s, "side": side, "entry": px, "notional": notional, "fee": notional * FEE})
            state["equity"] -= notional * FEE
    state["cohorts"].append({"opened": today, "opened_ts": int(ex.milliseconds()), "signal_date": str(date.date()), "capital": cap,
                             "positions": positions})   # opened_ts = momento real de la entrada (para marcarla en el gráfico)
    log(f"  cohorte abierta ({len(positions)} posiciones, {notional:.2f} c/u): LONG {', '.join(s.split('/')[0] for s in longs)} | "
        f"SHORT {', '.join(s.split('/')[0] for s in shorts)}")
    return realized


def unrealized(state, ex):
    u = 0.0
    for c in state["cohorts"]:
        for p in c["positions"]:
            px = exec_price(ex, p["symbol"], p["side"], opening=False)
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
    global FEE, SLIP
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
    FEE, SLIP = a.fee, a.slip
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
