"""Tests del bot maestro con selecciones FABRICADAS (solo para verificar la lógica; no son resultados de mercado).
Uso: python bots/test_master_bot.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from master_trading_bot import MasterTradingBot  # noqa: E402


def daily(n=120, breakout_up=False, breakout_down=False, seed=0):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(0, 0.3, n))
    if breakout_up:
        c[-1] = c[:-1].max() + 5
    if breakout_down:
        c[-1] = c[:-1].min() - 5
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(dict(open=o, high=np.maximum(o, c) + 0.2, low=np.minimum(o, c) - 0.2, close=c, volume=1.0))


def selection(strategies):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"generated": "test", "enabled": [s["name"] for s in strategies if s["enabled"]], "strategies": strategies}, f)
    f.close()
    return f.name


def check(nombre, cond):
    print(("OK   " if cond else "FALLA"), nombre)
    return cond


ok = True
# 1) sin archivo de selección -> nada aprobado -> HOLD
bot = MasterTradingBot(use_cerebro_rl=False, selection_path="/no/existe.json")
sig = bot.generate_signal(bot.calculate_indicators(daily()))
ok &= check("sin selección: HOLD", sig["signal"] == "HOLD" and sig["approved"] == [])

# 2) selección con todo rechazado -> HOLD
p = selection([{"name": "indicadores:rsi_reversion", "enabled": False, "config": {}}])
bot = MasterTradingBot(use_cerebro_rl=False, selection_path=p)
ok &= check("todo rechazado: HOLD", bot.generate_signal(bot.calculate_indicators(daily()))["signal"] == "HOLD")

# 3) tendencia aprobada: ruptura alcista -> LONG con distancia de stop; ruptura bajista -> SHORT; sin ruptura -> sin señal
p = selection([{"name": "tendencia:donchian_trailing", "enabled": True, "config": {"ultima_config": "N55|X20|ATR3.0"}}])
bot = MasterTradingBot(use_cerebro_rl=False, selection_path=p)
up = bot.approved_signals(daily(breakout_up=True))
dn = bot.approved_signals(daily(breakout_down=True))
no = bot.approved_signals(daily())
ok &= check("tendencia: ruptura alcista -> LONG con stop > 0", len(up) == 1 and up[0]["side"] == "LONG" and up[0]["stop_distance"] > 0)
ok &= check("tendencia: ruptura bajista -> SHORT", len(dn) == 1 and dn[0]["side"] == "SHORT")
ok &= check("tendencia: sin ruptura -> sin señal", no == [])
ok &= check("generate_signal con aprobada devuelve APPROVED", bot.generate_signal(bot.calculate_indicators(daily(breakout_up=True)))["signal"] == "APPROVED")

# 4) familia de indicadores aprobada: donchian_breakout#1 (n=20) con TP/SL/hold de la configuración
p = selection([{"name": "indicadores:donchian_breakout", "enabled": True, "config": {"ultima_config": "donchian_breakout#1|tp0.08|sl0.04|h7"}}])
bot = MasterTradingBot(use_cerebro_rl=False, selection_path=p)
s = bot.approved_signals(daily(breakout_up=True))
ok &= check("familia: ruptura -> LONG con tp=0.08, sl=0.04, hold=7", len(s) == 1 and s[0]["side"] == "LONG" and s[0]["tp"] == 0.08 and s[0]["sl"] == 0.04 and s[0]["max_hold_days"] == 7)

# 5) aprobada sin ejecutor -> nota explícita, sin señal inventada
p = selection([{"name": "modelo:otra", "enabled": True, "config": None}])
bot = MasterTradingBot(use_cerebro_rl=False, selection_path=p)
s = bot.approved_signals(daily())
ok &= check("sin ejecutor: devuelve nota y side=None", len(s) == 1 and s[0]["side"] is None and "ejecutor" in s[0]["note"])

# 6) config corrupta -> error controlado, no excepción
p = selection([{"name": "tendencia:donchian_trailing", "enabled": True, "config": {"ultima_config": "basura"}}])
bot = MasterTradingBot(use_cerebro_rl=False, selection_path=p)
s = bot.approved_signals(daily())
ok &= check("config corrupta: error controlado", len(s) == 1 and s[0]["side"] is None and "error" in s[0]["note"])

print("\nTODOS LOS TESTS OK" if ok else "\nHAY FALLAS")
sys.exit(0 if ok else 1)
