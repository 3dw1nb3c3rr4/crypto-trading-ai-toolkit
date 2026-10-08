"""Cuenta de trading MANUAL simulada (paper) que se comporta como un exchange de perpetuos en modo one-way.

- Órdenes a mercado (taker: ask/bid + deslizamiento) y límite (maker: se llenan cuando el precio cruza).
- Apalancamiento por posición, margen aislado, precio de liquidación aproximado (margen de mantenimiento `mmr`).
- TP / SL que cierran a mercado al tocarse (taker + deslizamiento).
- Funding: se aplica con el historial real de tasas del exchange (cada 8 h normalmente); largos pagan si la tasa es positiva.
- Comisiones en cada apertura/cierre. Todo queda en el historial con comisiones y funding por separado.
Estado en JSON (cuenta_manual.json). No envía órdenes a ningún sitio.
"""
from __future__ import annotations

import json
import os
import time
import uuid


def now_ms():
    return int(time.time() * 1000)


class PaperAccount:
    def __init__(self, path, cfg):
        self.path, self.cfg = path, cfg
        self.s = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else self._fresh(cfg["manual_capital"])

    @staticmethod
    def _fresh(capital):
        return dict(balance=float(capital), initial=float(capital), positions={}, orders=[], history=[],
                    fees=0.0, funding=0.0, created=now_ms())

    def save(self):
        tmp = self.path + ".tmp"
        json.dump(self.s, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, self.path)

    def reset(self, capital):
        self.s = self._fresh(capital)
        self.save()

    # ------------------------------------------------------------------ cálculos
    def liq_price(self, side, entry, lev):
        m = self.cfg["mmr"]
        return entry * (1 - 1 / lev + m) if side == "LONG" else entry * (1 + 1 / lev - m)

    def used_margin(self):
        return sum(p["margin"] for p in self.s["positions"].values()) + sum(o["margin"] for o in self.s["orders"])

    def snapshot(self, quotes):
        pos, upnl = [], 0.0
        for sym, p in self.s["positions"].items():
            q = quotes.get(sym)
            mark = q["price"] if q else None
            u = (1 if p["side"] == "LONG" else -1) * (mark - p["entry"]) * p["qty"] if mark else 0.0
            upnl += u
            pos.append({**p, "symbol": sym, "mark": mark, "upnl": u, "roe": u / p["margin"] if p["margin"] else 0.0,
                        "notional": p["qty"] * (mark or p["entry"])})
        eq = self.s["balance"] + self.used_margin() + upnl          # el margen de órdenes límite también es tuyo
        return dict(mode="paper", balance=self.s["balance"], equity=eq, upnl=upnl, used_margin=self.used_margin(),
                    available=self.s["balance"], initial=self.s["initial"], fees=self.s["fees"], funding=self.s["funding"],
                    positions=sorted(pos, key=lambda x: x["opened"]), orders=self.s["orders"], history=self.s["history"][-300:])

    # ------------------------------------------------------------------ órdenes
    def place(self, symbol, side, otype, usdt, lev, quote, price=None, tp=None, sl=None):
        """side: LONG/SHORT. usdt = valor de la posición (nocional). Devuelve un mensaje."""
        lev = int(lev)
        if usdt <= 0:
            raise ValueError("el valor de la posición debe ser > 0")
        if not 1 <= lev <= self.cfg["max_leverage"]:
            raise ValueError(f"apalancamiento entre 1 y {self.cfg['max_leverage']} (cámbialo en Configuración)")
        if otype == "limit":
            if not price or price <= 0:
                raise ValueError("falta el precio límite")
            margin = usdt / lev
            fee = usdt * self.cfg["maker"]
            if margin + fee > self.s["balance"]:
                raise ValueError(f"saldo insuficiente: necesitas {margin + fee:.2f} USDT de margen+comisión")
            self.s["balance"] -= margin                     # el margen queda reservado mientras la orden está abierta
            o = dict(id=uuid.uuid4().hex[:10], symbol=symbol, side=side, price=float(price), usdt=float(usdt), lev=lev,
                     margin=margin, tp=tp, sl=sl, created=now_ms())
            self.s["orders"].append(o)
            self.save()
            return f"orden límite {side} {symbol.split('/')[0]} a {price} creada"
        px = quote["ask"] * (1 + self.cfg["slippage"]) if side == "LONG" else quote["bid"] * (1 - self.cfg["slippage"])
        msg = self._fill(symbol, side, usdt / px, px, lev, self.cfg["taker"], tp, sl)
        self.save()
        return msg

    def _fill(self, symbol, side, qty, px, lev, fee_rate, tp=None, sl=None, reserved=0.0):
        """Ejecuta qty al precio px (one-way: reduce/cierra si la posición abierta es contraria)."""
        notional = qty * px
        fee = notional * fee_rate
        p = self.s["positions"].get(symbol)
        if p and p["side"] != side:                          # reduce o invierte
            close_qty = min(qty, p["qty"])
            self._realize(symbol, close_qty, px, fee_rate, "reducida por orden contraria")
            self.s["balance"] += reserved
            rest = qty - close_qty
            if rest > 1e-12:
                return self._fill(symbol, side, rest, px, lev, fee_rate, tp, sl)
            return f"posición {symbol.split('/')[0]} reducida en {close_qty:.6g}"
        margin = notional / lev
        if margin + fee > self.s["balance"] + reserved + 1e-9:
            raise ValueError(f"saldo insuficiente: necesitas {margin + fee:.2f} USDT de margen+comisión (disponible {self.s['balance'] + reserved:.2f})")
        self.s["balance"] += reserved - margin - fee
        self.s["fees"] += fee
        if p:                                                # aumenta: precio medio ponderado
            tot = p["qty"] + qty
            p["entry"] = (p["entry"] * p["qty"] + px * qty) / tot
            p["qty"], p["margin"] = tot, p["margin"] + margin
            p["lev"] = round(p["qty"] * p["entry"] / p["margin"], 2)
            p["fees"] += fee
            if tp:
                p["tp"] = tp
            if sl:
                p["sl"] = sl
        else:
            p = dict(side=side, qty=qty, entry=px, margin=margin, lev=lev, fees=fee, funding=0.0, opened=now_ms(),
                     last_funding=now_ms(), tp=tp, sl=sl)
            self.s["positions"][symbol] = p
        p["liq"] = self.liq_price(p["side"], p["entry"], max(p["lev"], 1))
        return f"{side} {symbol.split('/')[0]}: {qty:.6g} a {px:.6g} (comisión {fee:.4f} USDT)"

    def _realize(self, symbol, qty, px, fee_rate, reason, liquidation=False):
        p = self.s["positions"][symbol]
        frac = qty / p["qty"]
        sgn = 1 if p["side"] == "LONG" else -1
        gross = sgn * (px - p["entry"]) * qty
        fee = 0.0 if liquidation else qty * px * fee_rate
        margin = p["margin"] * frac
        if liquidation:
            gross = -margin                                  # en aislado se pierde el margen de la posición
        self.s["balance"] += margin + gross - fee
        self.s["fees"] += fee
        fund = p["funding"] * frac
        self.s["history"].append(dict(symbol=symbol, side=p["side"], qty=qty, entry=p["entry"], exit=px, lev=p["lev"],
                                      gross=gross, fees=p["fees"] * frac + fee, funding=fund,
                                      net=gross - fee - p["fees"] * frac - fund, opened=p["opened"], closed=now_ms(), reason=reason))
        p["fees"] *= (1 - frac)
        p["funding"] *= (1 - frac)
        p["qty"] -= qty
        p["margin"] -= margin
        if p["qty"] <= 1e-12 or frac >= 0.999999:
            del self.s["positions"][symbol]

    def close(self, symbol, fraction, quote, reason="cierre manual"):
        p = self.s["positions"].get(symbol)
        if not p:
            raise ValueError("no hay posición abierta en ese símbolo")
        fraction = min(max(float(fraction), 0.0), 1.0)
        px = quote["bid"] * (1 - self.cfg["slippage"]) if p["side"] == "LONG" else quote["ask"] * (1 + self.cfg["slippage"])
        self._realize(symbol, p["qty"] * fraction, px, self.cfg["taker"], reason)
        self.save()
        return f"{symbol.split('/')[0]} cerrada {fraction:.0%} a {px:.6g}"

    def cancel(self, oid):
        for o in list(self.s["orders"]):
            if o["id"] == oid:
                self.s["orders"].remove(o)
                self.s["balance"] += o["margin"]
                self.save()
                return "orden cancelada"
        raise ValueError("orden no encontrada")

    def set_tpsl(self, symbol, tp=None, sl=None):
        p = self.s["positions"].get(symbol)
        if not p:
            raise ValueError("no hay posición abierta en ese símbolo")
        p["tp"], p["sl"] = tp or None, sl or None
        self.save()
        return "TP/SL actualizados"

    # ------------------------------------------------------------------ motor (llamar cada pocos segundos)
    def on_tick(self, quotes):
        """Llena órdenes límite, ejecuta TP/SL y liquida. Devuelve eventos (texto)."""
        ev, changed = [], False
        for o in list(self.s["orders"]):
            q = quotes.get(o["symbol"])
            if not q:
                continue
            hit = (o["side"] == "LONG" and q["ask"] <= o["price"]) or (o["side"] == "SHORT" and q["bid"] >= o["price"])
            if hit:
                self.s["orders"].remove(o)
                try:
                    ev.append("límite llenada: " + self._fill(o["symbol"], o["side"], o["usdt"] / o["price"], o["price"], o["lev"],
                                                              self.cfg["maker"], o.get("tp"), o.get("sl"), reserved=o["margin"]))
                except ValueError as e:
                    self.s["balance"] += o["margin"]
                    ev.append(f"límite cancelada: {e}")
                changed = True
        for sym, p in list(self.s["positions"].items()):
            q = quotes.get(sym)
            if not q:
                continue
            m = q["price"]
            long = p["side"] == "LONG"
            if (long and m <= p["liq"]) or (not long and m >= p["liq"]):
                self._realize(sym, p["qty"], p["liq"], 0.0, "LIQUIDADA", liquidation=True)
                ev.append(f"{sym.split('/')[0]} LIQUIDADA a {p['liq']:.6g}")
                changed = True
                continue
            if p.get("sl") and ((long and m <= p["sl"]) or (not long and m >= p["sl"])):
                ev.append("stop loss: " + self.close(sym, 1.0, q, "stop loss"))
                changed = True
                continue
            if p.get("tp") and ((long and m >= p["tp"]) or (not long and m <= p["tp"])):
                ev.append("take profit: " + self.close(sym, 1.0, q, "take profit"))
                changed = True
        if changed:
            self.save()
        return ev

    def apply_funding(self, symbol, events, mark):
        """events: [(ts_ms, tasa)] posteriores a last_funding. Cobra/paga sobre qty × precio actual."""
        p = self.s["positions"].get(symbol)
        if not p or not events:
            return 0.0
        sgn = 1 if p["side"] == "LONG" else -1
        cost = 0.0
        for ts, r in events:
            if ts > p["last_funding"]:
                cost += sgn * p["qty"] * mark * r
                p["last_funding"] = ts
        p["funding"] += cost
        self.s["funding"] += cost
        self.s["balance"] -= cost
        self.save()
        return cost
