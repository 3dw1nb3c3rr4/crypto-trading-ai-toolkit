"""Bot maestro: opera SOLO las estrategias aprobadas por backtesting/selector.py (strategy_selection.json).

Si ninguna estrategia está aprobada, el bot responde HOLD. El modo investigación (require_approval=False) conserva la
combinación de indicadores original, que NO está validada: los backtests de `backtesting/REPORT.md` no encontraron una
configuración rentable neta de comisiones. No es un sistema listo para operar con dinero real.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "bots", "cerebro_rl"))
sys.path.insert(0, os.path.join(BASE_DIR, "backtesting"))

from indicators import atr, bollinger_bands, ema, macd, rsi, sma  # noqa: E402
from risk_management import position_size  # noqa: E402

CEREBRO_MODEL = os.path.join(BASE_DIR, "models", "cerebro_rl.pt")
SELECTION_PATH = os.path.join(BASE_DIR, "strategy_selection.json")
RL_MIN_CANDLES_5M = 151   # WARMUP + 1 del Cerebro RL


def _load_cerebro():
    try:
        from cerebro_rl_core import CerebroRL
        if os.path.exists(CEREBRO_MODEL):
            return CerebroRL.cargar(CEREBRO_MODEL, device="cpu")
    except Exception as exc:  # torch no instalado, checkpoint incompatible, etc.
        print(f"Cerebro RL no disponible: {exc}")
    return None


def load_selection(path: str = SELECTION_PATH) -> dict:
    """Lee strategy_selection.json; si no existe, devuelve una selección vacía (nada aprobado)."""
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": [], "strategies": [], "generated": None}


class MasterTradingBot:
    def __init__(self, use_cerebro_rl: bool = True, selection_path: str = SELECTION_PATH, require_approval: bool = True):
        self.require_approval = require_approval
        self.selection = load_selection(selection_path)
        self.cerebro_rl = _load_cerebro() if use_cerebro_rl else None
        self.trades_history: list[dict] = []

    def approved_strategies(self) -> list[str]:
        return [s["name"] for s in self.selection.get("strategies", []) if s.get("enabled")]

    def approved_signals(self, df: pd.DataFrame) -> list[dict]:
        """Señales de la última vela diaria SOLO de las estrategias aprobadas. df: OHLC diario con columnas open/high/low/close."""
        out = []
        for ev in self.selection.get("strategies", []):
            if not ev.get("enabled"):
                continue
            name, cfg = ev["name"], (ev.get("config") or {})
            try:
                if name.startswith("indicadores:"):
                    from strategies import REGISTRY
                    fam_idx, tp, sl, hold = cfg["ultima_config"].split("|")
                    fam, idx = fam_idx.split("#")
                    fn, grid = REGISTRY[fam]
                    sig = fn(df, **grid[int(idx)])[-1]
                    if sig != 0:
                        out.append(dict(strategy=name, side="LONG" if sig > 0 else "SHORT", tp=float(tp[2:]), sl=float(sl[2:]),
                                        max_hold_days=int(hold[1:])))
                elif name.startswith("tendencia:"):
                    ne, nx, k = (x[1:] if x[0] in "NX" else x[3:] for x in cfg["ultima_config"].split("|"))
                    ne, k = int(ne), float(k)
                    c, hi, lo = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
                    atr = float(pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                                           (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1).rolling(14).mean().iloc[-1])
                    if c[-1] > hi[-ne - 1:-1].max():
                        out.append(dict(strategy=name, side="LONG", stop_atr=k, stop_distance=k * atr, exit="trailing Donchian"))
                    elif c[-1] < lo[-ne - 1:-1].min():
                        out.append(dict(strategy=name, side="SHORT", stop_atr=k, stop_distance=k * atr, exit="trailing Donchian"))
                else:
                    out.append(dict(strategy=name, side=None, note="aprobada, pero el bot aún no tiene un ejecutor para esta estrategia"))
            except Exception as exc:
                out.append(dict(strategy=name, side=None, note=f"error al calcular la señal: {type(exc).__name__}: {exc}"))
        return out

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 50:
            raise ValueError(f"Se necesitan al menos 50 velas (hay {len(df)})")
        out = df.copy()
        c = out["close"]
        out["sma20"], out["sma50"] = sma(c, 20), sma(c, 50)
        out["ema12"], out["ema26"] = ema(c, 12), ema(c, 26)
        out["rsi14"] = rsi(c, 14)
        out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
        out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger_bands(c, 20, 2.0)
        out["atr"] = atr(out["high"], out["low"], c, 14)
        return out

    @staticmethod
    def _technical_score(last: pd.Series, prev: pd.Series) -> float:
        score = 0.0
        if last["sma20"] > last["sma50"]:
            score += 0.25
        elif last["sma20"] < last["sma50"]:
            score -= 0.25
        if last["close"] < last["bb_lower"]:
            score += 0.25
        elif last["close"] > last["bb_upper"]:
            score -= 0.25
        if last["rsi14"] < 30:
            score += 0.25
        elif last["rsi14"] > 70:
            score -= 0.25
        if last["macd_hist"] > 0 >= prev["macd_hist"]:
            score += 0.25
        elif last["macd_hist"] < 0 <= prev["macd_hist"]:
            score -= 0.25
        return score

    def rl_decision(self, ohlcv_5m) -> dict | None:
        """Decisión real del Cerebro RL. Requiere >=151 velas de 5m (no sirve con velas diarias)."""
        if self.cerebro_rl is None or ohlcv_5m is None or len(ohlcv_5m) < RL_MIN_CANDLES_5M:
            return None
        return self.cerebro_rl.decidir_entrada(ohlcv_5m)

    def generate_signal(self, df: pd.DataFrame, ohlcv_5m=None) -> dict:
        """df: velas con indicadores (calculate_indicators). ohlcv_5m: opcional, para el Cerebro RL.
        Con require_approval=True (por defecto) solo opera estrategias aprobadas por el selector; si no hay, HOLD."""
        if self.require_approval:
            appr = self.approved_signals(df)
            if not self.approved_strategies():
                return {"signal": "HOLD", "approved": [], "reason": "ninguna estrategia aprobada por el selector (ejecuta backtesting/selector.py)",
                        "selection_generated": self.selection.get("generated"), "timestamp": datetime.now(timezone.utc).isoformat()}
            return {"signal": "APPROVED", "approved": appr, "selection_generated": self.selection.get("generated"),
                    "timestamp": datetime.now(timezone.utc).isoformat()}
        last, prev = df.iloc[-1], df.iloc[-2]
        score = self._technical_score(last, prev)
        action = "BUY" if score >= 0.5 else "SELL" if score <= -0.5 else "HOLD"
        rl = self.rl_decision(ohlcv_5m)
        if rl is not None and rl["direccion"] != "esperar":
            rl_action = "BUY" if rl["direccion"] == "long" else "SELL"
            if action != "HOLD" and rl_action != action:
                action = "HOLD"          # desacuerdo -> no operar
        return {"signal": action, "technical_score": score, "rl": rl,
                "timestamp": datetime.now(timezone.utc).isoformat()}

    def calculate_position_size(self, balance: float, risk_pct: float, entry: float, stop: float) -> dict:
        qty = position_size(balance, risk_pct / 100.0, entry, stop)
        return {"quantity": qty, "notional": qty * entry, "risk_amount": balance * risk_pct / 100.0}

    def get_performance_summary(self) -> dict:
        pnl = np.array([t.get("pnl", 0.0) for t in self.trades_history])
        if len(pnl) == 0:
            return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0}
        return {"trades": len(pnl), "win_rate": float((pnl > 0).mean()), "total_pnl": float(pnl.sum())}


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(BASE_DIR, "sample_data.csv"))
    bot = MasterTradingBot()
    enriched = bot.calculate_indicators(data)
    sig = bot.generate_signal(enriched)
    print("estrategias aprobadas:", bot.approved_strategies() or "NINGUNA")
    print(sig["signal"], "-", sig.get("reason", sig.get("approved")))
    research = MasterTradingBot(use_cerebro_rl=False, require_approval=False).generate_signal(enriched)
    print("modo investigación (NO validado):", research["signal"], f"score={research['technical_score']:+.2f}")
    last = enriched["close"].iloc[-1]
    print(bot.calculate_position_size(10_000, 1.0, last, last * 0.98))
