"""Operación en el exchange (cuenta DEMO/testnet o REAL) con ccxt y tus claves API guardadas en el PC.

Seguridad incorporada (no se puede saltar desde el navegador):
  - tope de valor por orden (max_position_usdt) y de apalancamiento (max_leverage) de la configuración;
  - cada orden real exige confirmar en el navegador y el modo real exige escribir CONFIRMO al activarlo;
  - los cierres usan reduceOnly (nunca abren posición contraria por error);
  - modo demo: Binance demo trading / OKX y Bybit sandbox, con claves de demo separadas.
Probado contra un exchange simulado (test_cuentas.py). NO se ha podido probar contra el exchange real desde el entorno
de desarrollo: empieza en modo demo y con montos mínimos.
"""
from __future__ import annotations

import ccxt

import config_local as cl


def make_exchange(cfg, keys=None, authenticated=True):
    exid = cfg["exchange"]
    opts = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    if authenticated:
        k = keys or cl.load_keys(exid, cfg["mode"])
        if not k:
            raise ValueError(f"no hay claves API guardadas para {exid} ({'demo' if cfg['mode'] == 'demo' else 'real'}). Ve a Configuración.")
        opts.update(apiKey=k["apiKey"], secret=k["secret"])
        if k.get("password"):
            opts["password"] = k["password"]
    ex = getattr(ccxt, exid)(opts)
    if cfg["mode"] == "demo":
        if exid == "binanceusdm" and hasattr(ex, "enable_demo_trading"):
            ex.enable_demo_trading(True)
        else:
            ex.set_sandbox_mode(True)
    return ex


class ExchangeAccount:
    def __init__(self, cfg, ex=None):
        self.cfg = cfg
        self.ex = ex or make_exchange(cfg)
        self._markets = False

    def markets(self):
        if not self._markets:
            self.ex.load_markets()
            self._markets = True
        return self.ex.markets

    def test(self):
        b = self.ex.fetch_balance()
        usdt = b.get("USDT") or {}
        return dict(ok=True, total=usdt.get("total"), free=usdt.get("free"))

    def snapshot(self, quotes=None):
        self.markets()
        b = self.ex.fetch_balance()
        usdt = b.get("USDT") or {}
        pos = []
        for p in self.ex.fetch_positions():
            c = float(p.get("contracts") or 0)
            if not c:
                continue
            side = "LONG" if p.get("side") == "long" else "SHORT"
            pos.append(dict(symbol=p["symbol"], side=side, qty=c * float(p.get("contractSize") or 1), entry=float(p.get("entryPrice") or 0),
                            mark=float(p.get("markPrice") or 0), upnl=float(p.get("unrealizedPnl") or 0),
                            margin=float(p.get("initialMargin") or p.get("collateral") or 0), lev=float(p.get("leverage") or 0),
                            liq=float(p.get("liquidationPrice") or 0), notional=abs(float(p.get("notional") or 0)),
                            roe=float(p.get("percentage") or 0) / 100, funding=None, fees=None, opened=p.get("timestamp") or 0))
        orders = []
        try:
            for o in self.ex.fetch_open_orders():
                orders.append(dict(id=o["id"], symbol=o["symbol"], side="LONG" if o["side"] == "buy" else "SHORT", price=o.get("price") or o.get("triggerPrice"),
                                   usdt=(o.get("amount") or 0) * (o.get("price") or 0), lev=None, type=o.get("type"), created=o.get("timestamp")))
        except Exception:
            pass
        return dict(mode=self.cfg["mode"], balance=usdt.get("free"), equity=usdt.get("total"), upnl=sum(p["upnl"] for p in pos),
                    used_margin=usdt.get("used"), available=usdt.get("free"), initial=None, fees=None, funding=None,
                    positions=pos, orders=orders, history=self.history())

    def history(self, limit=50):
        try:
            tr = self.ex.fetch_my_trades(limit=limit) if self.ex.has.get("fetchMyTrades") else []
        except Exception:
            return []
        return [dict(symbol=t["symbol"], side="LONG" if t["side"] == "buy" else "SHORT", qty=t.get("amount"), entry=None, exit=t.get("price"),
                     fees=(t.get("fee") or {}).get("cost"), funding=None, net=None, gross=None, closed=t.get("timestamp"), reason="trade") for t in tr]

    # ------------------------------------------------------------------ órdenes
    def amount_for(self, symbol, usdt, price):
        m = self.markets()[symbol]
        cs = float(m.get("contractSize") or 1)
        amt = float(self.ex.amount_to_precision(symbol, usdt / price / cs))
        lim = (m.get("limits") or {})
        mn_amt = ((lim.get("amount") or {}).get("min") or 0)
        mn_cost = ((lim.get("cost") or {}).get("min") or 0)
        if amt <= 0 or amt < mn_amt or amt * cs * price < mn_cost:
            raise ValueError(f"monto demasiado pequeño para {symbol}: mínimo ~{max(mn_cost, mn_amt * cs * price):.2f} USDT")
        return amt

    def place(self, symbol, side, otype, usdt, lev, quote, price=None, tp=None, sl=None):
        if usdt > self.cfg["max_position_usdt"]:
            raise ValueError(f"supera tu tope de {self.cfg['max_position_usdt']:.0f} USDT por orden (Configuración)")
        if not 1 <= int(lev) <= self.cfg["max_leverage"]:
            raise ValueError(f"apalancamiento entre 1 y {self.cfg['max_leverage']}")
        self.markets()
        try:
            self.ex.set_margin_mode(self.cfg["margin_mode"], symbol)
        except Exception:
            pass                                             # ya estaba así o el exchange no lo permite con posición abierta
        self.ex.set_leverage(int(lev), symbol)
        ref = price if otype == "limit" else quote["price"]
        amt = self.amount_for(symbol, usdt, ref)
        bs = "buy" if side == "LONG" else "sell"
        o = self.ex.create_order(symbol, "limit" if otype == "limit" else "market", bs, amt, price if otype == "limit" else None)
        extra = []
        opp = "sell" if bs == "buy" else "buy"
        for kind, val in (("stopLossPrice", sl), ("takeProfitPrice", tp)):
            if val:
                try:
                    self.ex.create_order(symbol, "market", opp, amt, None, {kind: float(val), "reduceOnly": True})
                    extra.append("SL" if kind.startswith("stop") else "TP")
                except Exception as e:                       # la entrada ya está hecha: avisar sin ocultarlo
                    extra.append(f"{'SL' if kind.startswith('stop') else 'TP'} FALLÓ ({type(e).__name__}: {str(e)[:80]})")
        return f"orden {o.get('id')} enviada: {side} {amt} {symbol.split('/')[0]}" + (f" | {', '.join(extra)}" if extra else "")

    def close(self, symbol, fraction, quote=None, reason=""):
        for p in self.ex.fetch_positions([symbol]):
            c = float(p.get("contracts") or 0)
            if c:
                amt = float(self.ex.amount_to_precision(symbol, c * min(max(float(fraction), 0.0), 1.0)))
                bs = "sell" if p.get("side") == "long" else "buy"
                o = self.ex.create_order(symbol, "market", bs, amt, None, {"reduceOnly": True})
                return f"cierre enviado ({o.get('id')}): {amt} {symbol.split('/')[0]}"
        raise ValueError("no hay posición abierta en ese símbolo")

    def cancel(self, oid, symbol):
        self.ex.cancel_order(oid, symbol)
        return "orden cancelada"
