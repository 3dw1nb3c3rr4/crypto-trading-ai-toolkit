"""Clasifica los perpetuos en cripto o 'acciones/TradFi tokenizadas' (acciones, ETFs, materias primas).

Usa la información del exchange cuando la trae (Binance: underlyingType / underlyingSubType) y, si no, una lista
conocida. Puedes añadir tickers en ~/.cerebro/acciones_extra.txt (uno por línea) sin tocar el código.
"""
from __future__ import annotations

import os

from config_local import DIR

STOCKS = set("""
AAOI AAPL ADBE AEHR ALAB AMAT AMD AMZN ANTHROPIC APLD APP ARM ASML ASTS AVGO AXTI BABA BE BILL BMNR BRKB BX BZ CBRS CGNX
CIEN CL COHR COIN COST CRCL CRDO CRM CRWD CRWV CSCO DELL DKNG DRAM EWJ EWY EWZ GOOGL GOOG GLD HOOD IBM INTC JPM LLY META
MSFT MSTR MU NFLX NVDA OPENAI ORCL PLTR PYPL QQQ RDDT RKLB SHOP SLV SMCI SNOW SPY SPACEX TSLA TSM UBER UNH V WMT XAU XAG
XOM NG HG PL PA
""".split())


def _extra():
    p = os.path.join(DIR, "acciones_extra.txt")
    if os.path.exists(p):
        return {x.strip().upper() for x in open(p, encoding="utf-8") if x.strip()}
    return set()


def classify(symbol: str, market: dict | None = None) -> str:
    base = symbol.split("/")[0].upper()
    info = (market or {}).get("info") or {}
    ut = str(info.get("underlyingType", "")).upper()
    sub = " ".join(map(str, info.get("underlyingSubType") or [])).upper()
    if ut in ("TRADFI", "EQUITY", "STOCK", "COMMODITY") or any(w in sub for w in ("TRADFI", "STOCK", "EQUITY", "COMMODIT")):
        return "stock"
    if base in STOCKS or base in _extra():
        return "stock"
    return "crypto"


def allowed(symbol: str, universe: str, market: dict | None = None) -> bool:
    if universe == "both":
        return True
    c = classify(symbol, market)
    return (c == "stock") if universe == "stocks" else (c == "crypto")


def filter_symbols(symbols, universe, markets=None):
    markets = markets or {}
    return [s for s in symbols if allowed(s, universe, markets.get(s))]
