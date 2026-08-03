"""
backtest_example.py

A minimal, educational backtest demonstrating how to combine the
indicators, risk management, and sentiment modules from this toolkit.

Strategy: simple SMA crossover with fixed fractional risk sizing.
This is NOT investment advice — for research/educational use only.
"""

import pandas as pd

from indicators import sma
from risk_management import position_size, max_drawdown


def run_backtest(csv_path: str, initial_balance: float = 10_000,
                  risk_per_trade: float = 0.01,
                  fast_period: int = 10, slow_period: int = 30):
    df = pd.read_csv(csv_path)
    df["sma_fast"] = sma(df["close"], fast_period)
    df["sma_slow"] = sma(df["close"], slow_period)

    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1   # long
    df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1  # flat/short

    balance = initial_balance
    equity_curve = [balance]
    position = 0
    entry_price = 0.0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_signal = df.iloc[i - 1]["signal"]

        # Enter long
        if position == 0 and row["signal"] == 1:
            stop_loss = row["close"] * 0.98  # 2% stop
            size = position_size(balance, risk_per_trade, row["close"], stop_loss)
            position = size
            entry_price = row["close"]

        # Exit long
        elif position > 0 and row["signal"] != 1:
            pnl = (row["close"] - entry_price) * position
            balance += pnl
            position = 0

        equity_curve.append(balance)

    total_return = (balance - initial_balance) / initial_balance
    dd = max_drawdown(equity_curve)

    print(f"Initial balance: ${initial_balance:,.2f}")
    print(f"Final balance:   ${balance:,.2f}")
    print(f"Total return:    {total_return:.2%}")
    print(f"Max drawdown:    {dd:.2%}")

    return df, equity_curve


if __name__ == "__main__":
    run_backtest("examples/sample_data.csv")
