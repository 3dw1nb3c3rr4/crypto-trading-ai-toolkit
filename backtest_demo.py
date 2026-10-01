#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo de Backtesting - Master Trading Bot
=========================================
Backtesting simplificado para demostración
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Importar indicadores básicos directamente
from indicators import sma, ema, rsi, macd, bollinger_bands, atr
from risk_management import position_size, risk_reward_ratio, max_drawdown


class SimpleBacktest:
    """Backtesting simplificado sin dependencias complejas"""

    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = [initial_balance]

    def calculate_indicators(self, df):
        """Calcular indicadores técnicos"""
        df = df.copy()

        # Promedios móviles
        df['sma20'] = sma(df['close'], 20)
        df['sma50'] = sma(df['close'], 50)
        df['ema12'] = ema(df['close'], 12)
        df['ema26'] = ema(df['close'], 26)

        # RSI
        df['rsi'] = rsi(df['close'], 14)

        # MACD (retorna macd, signal, histogram)
        macd_vals, signal_vals, macd_hist = macd(df['close'])
        df['macd'] = macd_vals
        df['signal'] = signal_vals
        df['macd_hist'] = macd_hist

        # Bollinger Bands (retorna middle, upper, lower)
        bb_middle, bb_upper, bb_lower = bollinger_bands(df['close'], 20, 2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower

        # ATR
        df['atr'] = atr(df['high'], df['low'], df['close'], 14)

        return df

    def generate_signal(self, row, prev_row=None):
        """Generar señal de trading simple"""
        signal = "HOLD"
        score = 0

        # Cruce de SMA
        if not pd.isna(row['sma20']) and not pd.isna(row['sma50']):
            if row['sma20'] > row['sma50']:
                score += 0.3
                signal = "BUY"
            elif row['sma20'] < row['sma50']:
                score -= 0.3
                signal = "SELL"

        # RSI
        if not pd.isna(row['rsi']):
            if row['rsi'] < 30:
                score += 0.3
                signal = "BUY"
            elif row['rsi'] > 70:
                score -= 0.3
                signal = "SELL"

        # Bollinger Bands
        if not pd.isna(row['bb_lower']) and not pd.isna(row['bb_upper']):
            if row['close'] < row['bb_lower']:
                score += 0.2
                signal = "BUY"
            elif row['close'] > row['bb_upper']:
                score -= 0.2
                signal = "SELL"

        confidence = abs(score)

        if abs(score) < 0.3:
            signal = "HOLD"
        elif score > 0.3:
            signal = "BUY"
        elif score < -0.3:
            signal = "SELL"

        return {"signal": signal, "score": score, "confidence": min(confidence, 1.0)}

    def run_backtest(self, df, risk_pct=1.0):
        """Ejecutar backtesting completo"""

        print(f"\n{'='*80}")
        print(f"🚀 BACKTESTING INICIADO")
        print(f"{'='*80}")
        print(f"Balance inicial: ${self.initial_balance:,.2f}")
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        print(f"Período: {df.iloc[0][date_col]} a {df.iloc[-1][date_col]}")
        print(f"Total de velas: {len(df)}")
        print(f"Riesgo por trade: {risk_pct}%\n")

        # Calcular indicadores
        df = self.calculate_indicators(df)

        win_count = 0
        loss_count = 0

        # Iterar sobre datos
        for i in range(50, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            price = current['close']

            # Generar señal
            signal_data = self.generate_signal(current, prev)
            signal = signal_data['signal']

            # Lógica de entrada
            if self.position is None and signal != "HOLD":
                # Abrir posición
                if signal == "BUY":
                    stop_loss = price * 0.98
                    take_profit = price * 1.03
                else:  # SELL
                    stop_loss = price * 1.02
                    take_profit = price * 0.97

                # Calcular posición
                risk_amount = self.balance * (risk_pct / 100)
                risk_per_unit = abs(price - stop_loss)
                qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                self.position = {
                    'type': signal,
                    'entry_price': price,
                    'entry_idx': i,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'quantity': qty,
                    'risk_amount': risk_amount
                }

                print(f"[Vela {i:3d}] 📈 ENTRADA {signal:4s} @ ${price:8.2f} | "
                      f"Qty: {qty:.4f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")

            # Lógica de salida
            elif self.position is not None:
                if self.position['type'] == "BUY":
                    if price >= self.position['take_profit']:
                        pnl = (price - self.position['entry_price']) * self.position['quantity']
                        self.balance += pnl
                        win_count += 1
                        print(f"[Vela {i:3d}] ✅ SALIDA TP     @ ${price:8.2f} | "
                              f"PnL: +${pnl:8.2f} | Balance: ${self.balance:10,.2f}")
                        self.position = None

                    elif price <= self.position['stop_loss']:
                        pnl = (price - self.position['entry_price']) * self.position['quantity']
                        self.balance += pnl
                        loss_count += 1
                        print(f"[Vela {i:3d}] ❌ SALIDA SL     @ ${price:8.2f} | "
                              f"PnL: ${pnl:8.2f} | Balance: ${self.balance:10,.2f}")
                        self.position = None

                else:  # SHORT
                    if price <= self.position['take_profit']:
                        pnl = (self.position['entry_price'] - price) * self.position['quantity']
                        self.balance += pnl
                        win_count += 1
                        print(f"[Vela {i:3d}] ✅ SALIDA TP     @ ${price:8.2f} | "
                              f"PnL: +${pnl:8.2f} | Balance: ${self.balance:10,.2f}")
                        self.position = None

                    elif price >= self.position['stop_loss']:
                        pnl = (self.position['entry_price'] - price) * self.position['quantity']
                        self.balance += pnl
                        loss_count += 1
                        print(f"[Vela {i:3d}] ❌ SALIDA SL     @ ${price:8.2f} | "
                              f"PnL: ${pnl:8.2f} | Balance: ${self.balance:10,.2f}")
                        self.position = None

            self.equity_curve.append(self.balance)

        # Resumen
        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        pnl = self.balance - self.initial_balance
        return_pct = (pnl / self.initial_balance * 100)

        print(f"\n{'='*80}")
        print(f"📋 RESUMEN DEL BACKTESTING")
        print(f"{'='*80}")
        print(f"Total de trades:      {total_trades}")
        print(f"Trades ganadores:     {win_count} ✅")
        print(f"Trades perdedores:    {loss_count} ❌")
        print(f"Win rate:             {win_rate:.1f}%")
        print(f"Balance inicial:      ${self.initial_balance:,.2f}")
        print(f"Balance final:        ${self.balance:,.2f}")
        print(f"P&L total:            ${pnl:,.2f}")
        print(f"Retorno:              {return_pct:.2f}%")
        print(f"{'='*80}\n")

        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'balance_inicial': self.initial_balance,
            'balance_final': self.balance,
            'pnl': pnl,
            'return_pct': return_pct
        }


def main():
    """Ejecutar backtesting"""

    print("\n🎯 CRYPTO TRADING AI TOOLKIT - BACKTESTING DEMO\n")

    # Cargar datos
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "sample_data.csv"))
        print(f"✅ Datos cargados: {len(df)} velas")
        print(f"   Rango: {df.iloc[0]['date']} a {df.iloc[-1]['date']}")
    except FileNotFoundError:
        print("❌ Error: sample_data.csv no encontrado")
        return

    # Ejecutar backtesting
    backtest = SimpleBacktest(initial_balance=10000)
    results = backtest.run_backtest(df, risk_pct=2.0)

    # Exportar resultados
    print("💾 Exportando resultados a CSV...")
    results_df = pd.DataFrame({
        'candle': range(len(backtest.equity_curve)),
        'equity': backtest.equity_curve,
        'balance_change': [0] + [backtest.equity_curve[i] - backtest.equity_curve[i-1]
                                 for i in range(1, len(backtest.equity_curve))]
    })

    results_df.to_csv(os.path.join(BASE_DIR, "backtest_results.csv"), index=False)
    print(f"✅ Resultados guardados en backtest_results.csv\n")


if __name__ == "__main__":
    main()
