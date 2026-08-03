# Crypto Trading AI Toolkit

An open-source Python toolkit for quantitative cryptocurrency trading research.

## Features

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- SMA / EMA
- Risk management utilities
- Position sizing
- Simple market sentiment scoring
- Backtesting example

## Installation

```bash
git clone https://github.com/3dw1nb3c3rr4/crypto-trading-ai-toolkit.git
cd crypto-trading-ai-toolkit
pip install -r requirements.txt
```

## Usage

```python
import pandas as pd
from indicators import sma, ema, rsi, macd, bollinger_bands, atr

df = pd.read_csv("examples/sample_data.csv")
df["rsi"] = rsi(df["close"])
df["macd"], df["signal"] = macd(df["close"])
```

See `backtest_example.py` for a full end-to-end example using the sample dataset.

## Project Structure

```
crypto-trading-ai-toolkit/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── indicators.py
├── sentiment.py
├── risk_management.py
├── backtest_example.py
└── examples/
    └── sample_data.csv
```

## Technologies

- Python
- Pandas
- NumPy
- CCXT
- TA

## Disclaimer

This project is for **educational and research purposes only**. Nothing here constitutes financial advice. Trading cryptocurrencies involves substantial risk of loss.

## Goal

This project aims to provide educational tools for developers interested in algorithmic trading and machine learning applied to financial markets.

## License

MIT
