#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Trading Bot - Integración unificada de todos los sistemas
=================================================================
Combina:
- Indicadores básicos (RSI, MACD, Bollinger Bands, ATR)
- Modelos RL avanzados (Cerebro RL)
- Sistemas Techo/Suelo
- Gestión de riesgo
"""

import os
import sys
import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import json

# Añadir el directorio del proyecto al path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "bots"))

# Importar indicadores del toolkit
from indicators import sma, ema, rsi, macd, bollinger_bands, atr
from risk_management import position_size, calculate_stop_loss

# Intentar cargar modelos RL avanzados
try:
    from cerebro_rl.cerebro_rl_core import CerebroRL
    CEREBRO_RL_AVAILABLE = True
except ImportError:
    CEREBRO_RL_AVAILABLE = False
    print("⚠️  Cerebro RL no disponible - usando indicadores básicos")


class MasterTradingBot:
    """Bot maestro que combina múltiples estrategias de trading"""

    def __init__(self, exchange: str = "binance", sandbox: bool = True):
        """
        Inicializar el bot maestro

        Args:
            exchange: 'binance' o 'okx'
            sandbox: Usar testnet (True) o mainnet (False)
        """
        self.exchange_name = exchange
        self.sandbox = sandbox
        self.exchange = self._init_exchange(exchange, sandbox)

        # Cargar modelos RL si están disponibles
        self.cerebro_rl = None
        if CEREBRO_RL_AVAILABLE:
            self._load_cerebro_rl()

        # Estado del bot
        self.positions = {}
        self.trades_history = []
        self.performance_metrics = {}

        print(f"✅ Master Trading Bot inicializado ({exchange} - {'sandbox' if sandbox else 'mainnet'})")

    def _init_exchange(self, exchange: str, sandbox: bool):
        """Inicializar conexión con el exchange"""
        if exchange.lower() == "binance":
            exc = ccxt.binance({
                'enableRateLimit': True,
                'sandbox': sandbox
            })
        elif exchange.lower() == "okx":
            exc = ccxt.okx({
                'enableRateLimit': True,
                'sandbox': sandbox
            })
        else:
            raise ValueError(f"Exchange no soportado: {exchange}")

        return exc

    def _load_cerebro_rl(self):
        """Cargar modelo Cerebro RL pre-entrenado"""
        try:
            model_path = os.path.join(BASE_DIR, "models", "cerebro_rl.pt")
            if os.path.exists(model_path):
                self.cerebro_rl = CerebroRL.cargar(model_path, device="cpu")
                print(f"🧠 Cerebro RL cargado (umbral: {self.cerebro_rl.min_prob:.2f})")
            else:
                print(f"⚠️  Modelo Cerebro RL no encontrado en {model_path}")
        except Exception as e:
            print(f"⚠️  Error cargando Cerebro RL: {e}")

    def calculate_indicators(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Calcular todos los indicadores técnicos

        Args:
            df: DataFrame con OHLCV
            symbol: Símbolo del par (para logging)

        Returns:
            DataFrame enriquecido con indicadores
        """
        if len(df) < 50:
            raise ValueError(f"Se necesitan al menos 50 velas (tenemos {len(df)})")

        # Copiar para no modificar el original
        result = df.copy()

        # Indicadores básicos
        result['sma20'] = sma(result['close'], 20)
        result['sma50'] = sma(result['close'], 50)
        result['ema12'] = ema(result['close'], 12)
        result['ema26'] = ema(result['close'], 26)

        result['rsi14'] = rsi(result['close'], 14)

        macd_vals, signal_vals = macd(result['close'])
        result['macd'] = macd_vals
        result['signal'] = signal_vals
        result['macd_histogram'] = result['macd'] - result['signal']

        bb_upper, bb_middle, bb_lower = bollinger_bands(result['close'], 20, 2)
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower

        result['atr'] = atr(result['high'], result['low'], result['close'], 14)

        return result

    def generate_signal(self, df: pd.DataFrame,
                       use_rl: bool = True) -> Dict[str, any]:
        """
        Generar señal de trading combinando todos los indicadores

        Args:
            df: DataFrame con OHLCV e indicadores calculados
            use_rl: Usar modelo RL si está disponible

        Returns:
            Dict con señal, confianza, detalles
        """
        if len(df) == 0:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "Sin datos"}

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        signals = {
            "technical": self._analyze_technical(last, prev),
            "momentum": self._analyze_momentum(last, prev),
            "volatility": self._analyze_volatility(last),
        }

        # Si Cerebro RL está disponible, usarlo para confirmar
        if use_rl and self.cerebro_rl:
            rl_signal = self._cerebro_rl_decision(df)
            signals["rl"] = rl_signal

        # Consenso
        consensus = self._calculate_consensus(signals)

        return {
            "signal": consensus["action"],
            "confidence": consensus["confidence"],
            "signals_breakdown": signals,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _analyze_technical(self, last, prev) -> Dict:
        """Análisis técnico con MA y BB"""
        signal = "HOLD"
        score = 0.0

        # Cruces de promedios móviles
        if not pd.isna(last['sma20']) and not pd.isna(last['sma50']):
            if last['sma20'] > last['sma50']:
                score += 0.25
                signal = "BUY"
            elif last['sma20'] < last['sma50']:
                score -= 0.25
                signal = "SELL"

        # Bollinger Bands
        if not pd.isna(last['bb_lower']) and not pd.isna(last['bb_upper']):
            if last['close'] < last['bb_lower']:
                score += 0.25
                signal = "BUY"
            elif last['close'] > last['bb_upper']:
                score -= 0.25
                signal = "SELL"

        return {
            "signal": signal,
            "score": score,
            "reasoning": "Cruces de MA + Bollinger Bands"
        }

    def _analyze_momentum(self, last, prev) -> Dict:
        """Análisis de momentum con RSI y MACD"""
        signal = "HOLD"
        score = 0.0

        # RSI
        if not pd.isna(last['rsi14']):
            if last['rsi14'] < 30:
                score += 0.25
                signal = "BUY"
            elif last['rsi14'] > 70:
                score -= 0.25
                signal = "SELL"

        # MACD
        if not pd.isna(last['macd_histogram']):
            if (last['macd_histogram'] > 0 and prev['macd_histogram'] <= 0):
                score += 0.25
                signal = "BUY"
            elif (last['macd_histogram'] < 0 and prev['macd_histogram'] >= 0):
                score -= 0.25
                signal = "SELL"

        return {
            "signal": signal,
            "score": score,
            "reasoning": "RSI + MACD"
        }

    def _analyze_volatility(self, last) -> Dict:
        """Análisis de volatilidad con ATR"""
        signal = "HOLD"
        score = 0.0

        # ATR para gestión de riesgo
        if not pd.isna(last['atr']):
            atr_pct = (last['atr'] / last['close']) * 100
            if atr_pct > 2:
                score -= 0.1  # Alta volatilidad = más cautela
            else:
                score += 0.1  # Baja volatilidad = más confianza

        return {
            "signal": signal,
            "score": score,
            "reasoning": "ATR para evaluación de volatilidad"
        }

    def _cerebro_rl_decision(self, df: pd.DataFrame) -> Dict:
        """Consultar decisión del modelo Cerebro RL"""
        try:
            if len(df) < 150:  # Cerebro RL necesita al menos 150 velas para warmup
                return {"signal": "HOLD", "confidence": 0.0, "reason": "Insuficientes datos históricos"}

            # Aquí iría la integración con el Cerebro RL real
            # Por ahora es un placeholder
            return {
                "signal": "EVALUATE",
                "confidence": 0.5,
                "reason": "Cerebro RL requiere datos de mercado en vivo"
            }
        except Exception as e:
            print(f"⚠️  Error en Cerebro RL: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "error": str(e)}

    def _calculate_consensus(self, signals: Dict) -> Dict:
        """Calcular consenso de todas las señales"""
        scores = []
        for system, data in signals.items():
            if isinstance(data, dict) and "score" in data:
                scores.append(data["score"])

        if not scores:
            return {"action": "HOLD", "confidence": 0.0}

        avg_score = np.mean(scores)
        confidence = abs(avg_score) / len(scores)

        if avg_score > 0.1:
            action = "BUY"
        elif avg_score < -0.1:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "confidence": min(confidence, 0.95),
            "avg_score": avg_score
        }

    def calculate_position_size(self, account_balance: float,
                               risk_pct: float = 1.0,
                               entry_price: float = None,
                               stop_loss_price: float = None) -> Dict:
        """
        Calcular tamaño de posición basado en riesgo

        Args:
            account_balance: Balance total de la cuenta
            risk_pct: Porcentaje del balance en riesgo (default 1%)
            entry_price: Precio de entrada
            stop_loss_price: Precio del stop loss

        Returns:
            Dict con cálculos de posición
        """
        risk_amount = account_balance * (risk_pct / 100)

        if entry_price and stop_loss_price:
            risk_per_unit = abs(entry_price - stop_loss_price)
            position_qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        else:
            position_qty = risk_amount / entry_price if entry_price else 0

        return {
            "risk_amount": risk_amount,
            "position_quantity": position_qty,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "risk_per_unit": abs(entry_price - stop_loss_price) if entry_price and stop_loss_price else None
        }

    def get_performance_summary(self) -> Dict:
        """Obtener resumen de desempeño"""
        if not self.trades_history:
            return {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0}

        winning_trades = [t for t in self.trades_history if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trades_history if t.get("pnl", 0) < 0]

        total_pnl = sum(t.get("pnl", 0) for t in self.trades_history)
        win_rate = len(winning_trades) / len(self.trades_history) if self.trades_history else 0

        return {
            "total_trades": len(self.trades_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": np.mean([t.get("pnl", 0) for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t.get("pnl", 0) for t in losing_trades]) if losing_trades else 0,
        }


def main():
    """Ejemplo de uso del Master Trading Bot"""
    import pandas as pd

    # Crear bot
    bot = MasterTradingBot(exchange="binance", sandbox=True)

    # Cargar datos de ejemplo
    sample_data = pd.read_csv(os.path.join(BASE_DIR, "sample_data.csv"))

    # Calcular indicadores
    print("\n📊 Calculando indicadores...")
    enriched_df = bot.calculate_indicators(sample_data, symbol="BTC/USDT")

    # Generar señal
    print("\n🤖 Generando señal de trading...")
    signal = bot.generate_signal(enriched_df, use_rl=True)

    print(f"\n✅ Resultado:")
    print(f"  Señal: {signal['signal']}")
    print(f"  Confianza: {signal['confidence']:.2%}")
    print(f"  Detalles: {json.dumps(signal['signals_breakdown'], indent=2, default=str)}")

    # Calcular posición
    print("\n💰 Cálculo de posición:")
    position = bot.calculate_position_size(
        account_balance=10000,
        risk_pct=2.0,
        entry_price=float(enriched_df['close'].iloc[-1]),
        stop_loss_price=float(enriched_df['close'].iloc[-1] * 0.98)
    )
    for key, value in position.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
