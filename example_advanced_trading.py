#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo Avanzado: Trading Integrado con Master Bot
====================================================
Demuestra:
1. Análisis multi-timeframe
2. Gestión de riesgo automática
3. Histórico y métricas
4. Exportación de resultados
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Agregar el proyecto al path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from bots import MasterTradingBot


class AdvancedTradingStrategy:
    """Estrategia avanzada con múltiples timeframes"""

    def __init__(self, exchange: str = "binance"):
        self.bot = MasterTradingBot(exchange=exchange, sandbox=True)
        self.trades = []
        self.signals_history = []

    def analyze_multi_timeframe(self,
                               data_5m: pd.DataFrame,
                               data_1h: pd.DataFrame,
                               data_4h: pd.DataFrame) -> dict:
        """
        Análisis en 3 timeframes

        Regla: Operar solo cuando:
        - 4h está en tendencia (RSI < 30 para LONG o > 70 para SHORT)
        - 1h confirma (MACD positivo para LONG)
        - 5m da entrada (Bollinger Band menor para LONG)
        """

        # Enriquecer datos
        df_5m = self.bot.calculate_indicators(data_5m.copy())
        df_1h = self.bot.calculate_indicators(data_1h.copy())
        df_4h = self.bot.calculate_indicators(data_4h.copy())

        # Obtener señales
        signal_5m = self.bot.generate_signal(df_5m, use_rl=True)
        signal_1h = self.bot.generate_signal(df_1h, use_rl=True)
        signal_4h = self.bot.generate_signal(df_4h, use_rl=True)

        # Analizar los últimos candles
        last_5m = df_5m.iloc[-1]
        last_1h = df_1h.iloc[-1]
        last_4h = df_4h.iloc[-1]

        # Lógica de consenso
        consensus = self._calculate_consensus_multi_tf(
            signal_5m, signal_1h, signal_4h,
            last_5m, last_1h, last_4h
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "signals": {
                "5m": signal_5m,
                "1h": signal_1h,
                "4h": signal_4h
            },
            "consensus": consensus,
            "last_prices": {
                "5m": float(last_5m['close']),
                "1h": float(last_1h['close']),
                "4h": float(last_4h['close'])
            },
            "last_indicators": {
                "5m_rsi": float(last_5m['rsi14']) if not pd.isna(last_5m['rsi14']) else None,
                "1h_rsi": float(last_1h['rsi14']) if not pd.isna(last_1h['rsi14']) else None,
                "4h_rsi": float(last_4h['rsi14']) if not pd.isna(last_4h['rsi14']) else None,
            }
        }

    def _calculate_consensus_multi_tf(self, signal_5m, signal_1h, signal_4h,
                                      last_5m, last_1h, last_4h) -> dict:
        """Consenso entre timeframes"""

        # Contar votos
        buy_votes = 0
        sell_votes = 0

        # 4h es la más importante (peso 3)
        if signal_4h['signal'] == 'BUY':
            buy_votes += 3
        elif signal_4h['signal'] == 'SELL':
            sell_votes += 3

        # 1h es importante (peso 2)
        if signal_1h['signal'] == 'BUY':
            buy_votes += 2
        elif signal_1h['signal'] == 'SELL':
            sell_votes += 2

        # 5m para entrada (peso 1)
        if signal_5m['signal'] == 'BUY':
            buy_votes += 1
        elif signal_5m['signal'] == 'SELL':
            sell_votes += 1

        # Determinar decisión
        if buy_votes > sell_votes:
            action = "BUY"
            confidence = buy_votes / (buy_votes + sell_votes) if (buy_votes + sell_votes) > 0 else 0
        elif sell_votes > buy_votes:
            action = "SELL"
            confidence = sell_votes / (buy_votes + sell_votes) if (buy_votes + sell_votes) > 0 else 0
        else:
            action = "HOLD"
            confidence = 0.5

        return {
            "action": action,
            "confidence": confidence,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "rationale": f"{action} con {int(confidence*100)}% confianza (votos: BUY {buy_votes}, SELL {sell_votes})"
        }

    def execute_trade(self,
                     symbol: str,
                     action: str,
                     entry_price: float,
                     account_balance: float = 10000,
                     risk_pct: float = 1.0) -> dict:
        """Ejecutar trade con gestión de riesgo automática"""

        if action == "HOLD":
            return {"status": "NO_TRADE", "reason": "Señal HOLD"}

        # Calcular SL/TP
        if action == "BUY":
            stop_loss = entry_price * 0.98  # -2%
            take_profit = entry_price * 1.03  # +3%
        else:  # SELL
            stop_loss = entry_price * 1.02  # +2%
            take_profit = entry_price * 0.97  # -3%

        # Calcular posición
        position = self.bot.calculate_position_size(
            account_balance=account_balance,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_loss_price=stop_loss
        )

        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_quantity": position['position_quantity'],
            "risk_amount": position['risk_amount'],
            "status": "PENDING"
        }

        self.trades.append(trade)
        return trade

    def backtest(self,
                data_5m: pd.DataFrame,
                data_1h: pd.DataFrame,
                data_4h: pd.DataFrame,
                initial_balance: float = 10000):
        """
        Backtest de la estrategia multi-timeframe

        Limitaciones:
        - Simula análisis en cada vela 5m
        - No incluye slippage ni comisiones
        - Propósito: verificación lógica
        """

        balance = initial_balance
        position = None
        trades_executed = 0
        wins = 0
        losses = 0

        print(f"\n📊 BACKTEST INICIADO")
        print(f"Balance inicial: ${balance:,.2f}")
        print(f"Período: {data_5m.iloc[0]['timestamp']} a {data_5m.iloc[-1]['timestamp']}")
        print("=" * 80)

        # Iterar sobre datos
        for i in range(200, min(len(data_5m), 300)):  # Analizar últimas 100 velas
            # Crear subset de datos hasta el momento actual
            subset_5m = data_5m.iloc[:i].copy()
            subset_1h = data_1h.iloc[:int(i/12)].copy() if int(i/12) < len(data_1h) else data_1h.copy()
            subset_4h = data_4h.iloc[:int(i/48)].copy() if int(i/48) < len(data_4h) else data_4h.copy()

            # Análisis
            analysis = self.analyze_multi_timeframe(subset_5m, subset_1h, subset_4h)

            entry_price = analysis['last_prices']['5m']
            action = analysis['consensus']['action']

            # Lógica de trades
            if position is None and action in ["BUY", "SELL"]:
                # Abrir posición
                trade = self.execute_trade(
                    symbol="BTC/USDT",
                    action=action,
                    entry_price=entry_price,
                    account_balance=balance,
                    risk_pct=1.0
                )
                position = trade
                position['entry_idx'] = i
                trades_executed += 1

                print(f"\n[Vela {i}] 📈 ENTRADA {action}")
                print(f"  Precio: ${entry_price:.2f}")
                print(f"  Cantidad: {position['position_quantity']:.4f}")
                print(f"  SL: ${position['stop_loss']:.2f} | TP: ${position['take_profit']:.2f}")

            elif position is not None:
                # Verificar salida
                if (position['action'] == "BUY" and entry_price >= position['take_profit']):
                    pnl = (entry_price - position['entry_price']) * position['position_quantity']
                    balance += pnl
                    wins += 1

                    print(f"[Vela {i}] ✅ SALIDA TP - PnL: +${pnl:.2f} | Balance: ${balance:,.2f}")
                    position = None

                elif (position['action'] == "BUY" and entry_price <= position['stop_loss']):
                    pnl = (entry_price - position['entry_price']) * position['position_quantity']
                    balance += pnl
                    losses += 1

                    print(f"[Vela {i}] ❌ SALIDA SL - PnL: ${pnl:.2f} | Balance: ${balance:,.2f}")
                    position = None

        # Resumen
        print("\n" + "=" * 80)
        print(f"📋 RESUMEN DEL BACKTEST")
        print(f"Trades ejecutados: {trades_executed}")
        print(f"Trades ganadores: {wins}")
        print(f"Trades perdedores: {losses}")
        print(f"Win rate: {(wins/trades_executed*100) if trades_executed > 0 else 0:.1f}%")
        print(f"Balance final: ${balance:,.2f}")
        print(f"P&L total: ${(balance - initial_balance):,.2f}")
        print(f"Retorno: {((balance - initial_balance) / initial_balance * 100):.1f}%")

        return {
            "balance_inicial": initial_balance,
            "balance_final": balance,
            "pnl": balance - initial_balance,
            "trades_totales": trades_executed,
            "trades_ganadores": wins,
            "trades_perdedores": losses,
            "win_rate": (wins/trades_executed) if trades_executed > 0 else 0
        }


def main():
    """Función principal con ejemplos"""

    print("\n" + "="*80)
    print("🚀 CRYPTO TRADING AI TOOLKIT - EJEMPLO AVANZADO")
    print("="*80)

    # Cargar datos de ejemplo
    print("\n📂 Cargando datos de ejemplo...")
    try:
        sample_data = pd.read_csv(os.path.join(BASE_DIR, "sample_data.csv"))
        print(f"✅ Datos cargados: {len(sample_data)} velas")
    except FileNotFoundError:
        print("❌ Error: sample_data.csv no encontrado")
        return

    # Crear estrategia
    print("\n🤖 Inicializando estrategia...")
    strategy = AdvancedTradingStrategy(exchange="binance")

    # Simular múltiples timeframes (en realidad es el mismo dato)
    data_5m = sample_data.copy()
    data_1h = sample_data.iloc[::12].copy()  # Cada 12 velas
    data_4h = sample_data.iloc[::48].copy()  # Cada 48 velas

    # Análisis multi-timeframe
    print("\n📊 Análisis multi-timeframe...")
    analysis = strategy.analyze_multi_timeframe(data_5m, data_1h, data_4h)

    print(f"\n✅ Análisis completado:")
    print(f"  Consenso: {analysis['consensus']['action']} ({analysis['consensus']['confidence']:.2%})")
    print(f"  Razón: {analysis['consensus']['rationale']}")

    print(f"\nÚltimos RSI:")
    print(f"  5m: {analysis['last_indicators']['5m_rsi']:.2f}")
    print(f"  1h: {analysis['last_indicators']['1h_rsi']:.2f}")
    print(f"  4h: {analysis['last_indicators']['4h_rsi']:.2f}")

    # Ejecutar trade
    print("\n💰 Ejecutando trade...")
    entry_price = analysis['last_prices']['5m']
    trade = strategy.execute_trade(
        symbol="BTC/USDT",
        action=analysis['consensus']['action'],
        entry_price=entry_price,
        account_balance=10000,
        risk_pct=2.0
    )

    if trade['status'] != 'NO_TRADE':
        print(f"✅ Trade ejecutado:")
        print(f"  Acción: {trade['action']}")
        print(f"  Entrada: ${trade['entry_price']:.2f}")
        print(f"  SL: ${trade['stop_loss']:.2f}")
        print(f"  TP: ${trade['take_profit']:.2f}")
        print(f"  Cantidad: {trade['position_quantity']:.4f}")
        print(f"  Riesgo: ${trade['risk_amount']:.2f}")

    # Backtest
    print("\n" + "="*80)
    print("📈 Ejecutando backtest...")
    print("="*80)

    backtest_results = strategy.backtest(data_5m, data_1h, data_4h)

    # Exportar resultados
    print("\n💾 Exportando resultados...")
    with open(os.path.join(BASE_DIR, "backtest_results.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "consensus": analysis['consensus'],
                "signals_5m": analysis['signals']['5m'],
                "signals_1h": analysis['signals']['1h'],
                "signals_4h": analysis['signals']['4h']
            },
            "backtest_results": backtest_results
        }, f, indent=2, default=str)

    print("✅ Resultados exportados a backtest_results.json")


if __name__ == "__main__":
    main()
