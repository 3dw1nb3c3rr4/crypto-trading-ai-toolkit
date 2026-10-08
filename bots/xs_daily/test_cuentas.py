"""Pruebas sin red de la cuenta paper manual, la cuenta de exchange (con un exchange simulado), funding, universo y config.
Uso: python bots/xs_daily/test_cuentas.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["CEREBRO_HOME"] = tempfile.mkdtemp()             # nunca toca tu ~/.cerebro real

import config_local as cl  # noqa: E402
import cuenta_paper as cp  # noqa: E402
import cuenta_real as cr  # noqa: E402
import datos_cuenta as dc  # noqa: E402
import funding_util as fu  # noqa: E402
import universo as un  # noqa: E402

CFG = cl.validate(dict(manual_capital=1000, taker=0.0005, maker=0.0002, slippage=0.0, mmr=0.005, max_leverage=20, default_leverage=3))
OK = []


def q(p):
    return dict(price=p, bid=p, ask=p)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1, abs(b))


def check(name, cond):
    assert cond, name
    OK.append(name)


def t_paper():
    d = tempfile.mkdtemp()
    a = cp.PaperAccount(os.path.join(d, "c.json"), CFG)
    a.place("BTC/USDT:USDT", "LONG", "market", 1000, 10, q(100.0))
    p = a.s["positions"]["BTC/USDT:USDT"]
    check("margen = nocional/apalancamiento", close(p["margin"], 100.0))
    check("comisión taker al abrir", close(a.s["balance"], 1000 - 100 - 0.5))
    check("liquidación long 10x", close(p["liq"], 100 * (1 - 0.1 + 0.005)))
    s = a.snapshot({"BTC/USDT:USDT": q(110.0)})
    check("PnL no realizado", close(s["upnl"], 100.0))
    check("equity = saldo + margen + upnl", close(s["equity"], 1000 - 0.5 + 100))
    a.apply_funding("BTC/USDT:USDT", [(p["last_funding"] + 1, 0.0001)], 110.0)
    check("funding: long paga con tasa positiva", close(a.s["funding"], 10 * 110 * 0.0001))
    a.close("BTC/USDT:USDT", 0.5, q(110.0))
    check("cierre parcial deja mitad", close(a.s["positions"]["BTC/USDT:USDT"]["qty"], 5.0))
    a.close("BTC/USDT:USDT", 1.0, q(110.0))
    h = a.s["history"]
    net = sum(x["net"] for x in h)
    exp = 100.0 - 0.5 - 1100 * 0.0005 - 10 * 110 * 0.0001       # bruto - comisión entrada - comisión salida - funding
    check("neto del historial = bruto - comisiones - funding", close(net, exp))
    check("saldo final cuadra con el neto", close(a.s["balance"], 1000 + exp))
    # short + TP
    a.place("ETH/USDT:USDT", "SHORT", "market", 500, 5, q(50.0), tp=45.0, sl=55.0)
    ev = a.on_tick({"ETH/USDT:USDT": q(44.0)})
    check("take profit ejecutado", "ETH/USDT:USDT" not in a.s["positions"] and ev and "take profit" in ev[0])
    # límite + liquidación
    bal0 = a.s["balance"]
    a.place("SOL/USDT:USDT", "LONG", "limit", 200, 20, q(10.0), price=9.0)
    check("límite reserva margen", close(a.s["balance"], bal0 - 10.0))
    a.on_tick({"SOL/USDT:USDT": q(9.5)})
    check("límite no se llena antes de tiempo", len(a.s["orders"]) == 1)
    a.on_tick({"SOL/USDT:USDT": q(8.99)})
    check("límite llenada con comisión maker", "SOL/USDT:USDT" in a.s["positions"] and not a.s["orders"])
    liq = a.s["positions"]["SOL/USDT:USDT"]["liq"]
    a.on_tick({"SOL/USDT:USDT": q(liq * 0.999)})
    check("liquidación pierde el margen", "SOL/USDT:USDT" not in a.s["positions"] and a.s["history"][-1]["reason"] == "LIQUIDADA")
    # cancelación devuelve margen
    b0 = a.s["balance"]
    a.place("XRP/USDT:USDT", "LONG", "limit", 100, 2, q(1.0), price=0.5)
    a.cancel(a.s["orders"][0]["id"])
    check("cancelar devuelve el margen", close(a.s["balance"], b0))
    # one-way: orden contraria reduce
    a.place("BTC/USDT:USDT", "LONG", "market", 100, 2, q(100.0))
    a.place("BTC/USDT:USDT", "SHORT", "market", 50, 2, q(100.0))
    check("orden contraria reduce (one-way)", close(a.s["positions"]["BTC/USDT:USDT"]["qty"], 0.5))
    try:
        a.place("BTC/USDT:USDT", "LONG", "market", 10 ** 7, 2, q(100.0))
        check("rechaza saldo insuficiente", False)
    except ValueError:
        check("rechaza saldo insuficiente", True)
    try:
        a.place("BTC/USDT:USDT", "LONG", "market", 10, 50, q(100.0))
        check("rechaza apalancamiento > tope", False)
    except ValueError:
        check("rechaza apalancamiento > tope", True)
    a.reset(500)
    check("reinicio", a.s["balance"] == 500 and not a.s["positions"])


class FakeEx:
    """Exchange simulado con la interfaz de ccxt que usa ExchangeAccount."""
    has = {"fetchMyTrades": False}

    def __init__(self):
        self.markets = {"BTC/USDT:USDT": dict(contractSize=1, limits=dict(amount=dict(min=0.001), cost=dict(min=5)))}
        self.calls = []
        self.pos = []

    def load_markets(self):
        return self.markets

    def amount_to_precision(self, s, a):
        return f"{int(a * 1000) / 1000:.3f}"

    def set_margin_mode(self, m, s):
        self.calls.append(("margin", m))

    def set_leverage(self, l, s):
        self.calls.append(("lev", l))

    def create_order(self, s, t, side, amt, price=None, params=None):
        self.calls.append(("order", t, side, amt, price, params or {}))
        return {"id": str(len(self.calls))}

    def fetch_positions(self, syms=None):
        return self.pos

    def fetch_funding_rate_history(self, s, since=None, limit=100):
        return [dict(timestamp=t, fundingRate=0.0001) for t in range(since or 0, (since or 0) + 3 * 8 * 3600_000, 8 * 3600_000)]


def t_real():
    cfg = dict(CFG, mode="demo", max_position_usdt=100, max_leverage=10)
    ex = FakeEx()
    acc = cr.ExchangeAccount(cfg, ex)
    acc.place("BTC/USDT:USDT", "LONG", "market", 50, 3, q(20000.0), sl=19000, tp=22000)
    kinds = [c for c in ex.calls if c[0] == "order"]
    check("fija apalancamiento antes de ordenar", ("lev", 3) in ex.calls)
    check("cantidad = USDT / precio con precisión", kinds[0][3] == 0.002)
    check("SL y TP son reduceOnly", all(c[5].get("reduceOnly") for c in kinds[1:]) and len(kinds) == 3)
    try:
        acc.place("BTC/USDT:USDT", "LONG", "market", 500, 3, q(20000.0))
        check("tope por orden", False)
    except ValueError:
        check("tope por orden", True)
    try:
        acc.place("BTC/USDT:USDT", "LONG", "market", 2, 3, q(20000.0))
        check("rechaza bajo el mínimo del exchange", False)
    except ValueError:
        check("rechaza bajo el mínimo del exchange", True)
    ex.pos = [dict(contracts=0.004, side="long", symbol="BTC/USDT:USDT")]
    ex.calls.clear()
    acc.close("BTC/USDT:USDT", 0.5)
    c = ex.calls[-1]
    check("cierre real reduceOnly y lado opuesto", c[2] == "sell" and c[3] == 0.002 and c[5].get("reduceOnly"))
    cost, ok = fu.funding_cost(ex, "BTC/USDT:USDT", "SHORT", 1000, 0, 3 * 8 * 3600_000)
    check("funding: corto cobra con tasa positiva", ok and close(cost, -1000 * 0.0001 * 3))


def t_misc():
    check("universo: acción", un.classify("AAPL/USDT:USDT") == "stock")
    check("universo: cripto", un.classify("BTC/USDT:USDT") == "crypto")
    check("universo: info del exchange", un.classify("XYZ/USDT:USDT", {"info": {"underlyingType": "TRADFI"}}) == "stock")
    check("filtro solo cripto", un.filter_symbols(["AAPL/USDT:USDT", "BTC/USDT:USDT"], "crypto") == ["BTC/USDT:USDT"])
    cl.save_keys("okx", "demo", "abcdefghijkl", "s3cr3t", "pp")
    st = cl.keys_status("okx", "demo")
    check("claves guardadas y pista sin secreto", st["saved"] and "s3cr3t" not in str(st) and st["hint"] == "abcd…ijkl")
    check("claves demo y real separadas", not cl.keys_status("okx", "real")["saved"])
    cl.delete_keys("okx", "demo")
    check("borrar claves", not cl.keys_status("okx", "demo")["saved"])
    try:
        cl.save_config(dict(taker=0.5))
        check("valida comisiones absurdas", False)
    except ValueError:
        check("valida comisiones absurdas", True)


class FakeAuthEx(FakeEx):
    def fetch_leverage_tiers(self, symbols=None):
        return {"BTC/USDT:USDT": [dict(minNotional=0, maxNotional=50000, maxLeverage=125, maintenanceMarginRate=0.004, info={"cum": "0"}),
                                  dict(minNotional=50000, maxNotional=250000, maxLeverage=100, maintenanceMarginRate=0.005, info={"cum": "50"})]}

    def fetch_trading_fee(self, s):
        return dict(maker=0.00018, taker=0.00045)

    def fetch_position_mode(self, s):
        return dict(hedged=True)


def t_datos():
    check("liquidación fórmula Binance (ejemplo 10x)", close(dc.liq_price("LONG", 10000, 1, 1000, 0.004, 0), 9000 / 0.996))
    cl.save_keys("binanceusdm", "real", "abcdefghijkl", "s3cr3t")
    cfg = dict(CFG, exchange="binanceusdm")
    d = dc.AccountData(cfg, make_exchange=lambda m, k: FakeAuthEx())
    st = d.refresh()
    check("lee comisiones y escalones de la cuenta", st["taker"] == 0.00045 and st["n_tiers"] == 1 and st["hedged"] is True and st["source"] == "real")
    check("escalón según tamaño", d.bracket("BTC/USDT:USDT", 60000)[:3] == (100, 0.005, 50.0))
    a = cp.PaperAccount(os.path.join(tempfile.mkdtemp(), "c.json"), dict(cfg, max_leverage=125), d)
    a.place("BTC/USDT:USDT", "LONG", "market", 1000, 10, q(100.0))
    p = a.s["positions"]["BTC/USDT:USDT"]
    check("paper usa la comisión real", close(a.s["fees"], 1000 * 0.00045))
    check("paper usa la liquidación del escalón", close(p["liq"], dc.liq_price("LONG", 100.0, 10, 100, 0.004, 0)))
    try:
        a.place("BTC/USDT:USDT", "LONG", "market", 60000, 110, q(100.0))
        check("respeta el apalancamiento máximo del escalón", False)
    except ValueError:
        check("respeta el apalancamiento máximo del escalón", True)
    ex = FakeAuthEx()
    acc = cr.ExchangeAccount(dict(cfg, mode="demo", max_position_usdt=100, max_leverage=10), ex)
    acc.place("BTC/USDT:USDT", "SHORT", "market", 50, 3, q(20000.0), sl=21000)
    o = [c for c in ex.calls if c[0] == "order"]
    check("modo hedge: positionSide en entrada y SL, sin reduceOnly", o[0][5] == {"positionSide": "SHORT"} and o[1][5].get("positionSide") == "SHORT" and "reduceOnly" not in o[1][5])
    cl.delete_keys("binanceusdm", "real")


if __name__ == "__main__":
    t_paper(); t_real(); t_misc(); t_datos()
    print(f"test_cuentas: {len(OK)} comprobaciones OK")
