# Crypto Trading AI Toolkit

Una plataforma integrada de trading cuantitativo con IA, que combina indicadores técnicos clásicos con modelos de aprendizaje por refuerzo avanzados.

## 🎯 Características Principales

### Indicadores Técnicos Clásicos
- RSI (Índice de Fuerza Relativa)
- MACD (Convergencia/Divergencia de Promedios Móviles)
- Bollinger Bands
- ATR (Rango Verdadero Promedio)
- SMA / EMA (Promedios Móviles)

### Modelos Avanzados de IA
- **Cerebro RL**: Red neuronal de Refuerzo Aprendizaje para decisiones de entrada
- **Techo/Suelo**: Detección de niveles de resistencia y soporte
- **Meta Filtro**: Análisis meta de market-making y filtrado de señales

### Características del Bot
- Integración multi-exchange (Binance, OKX)
- Gestión automática de riesgo y posiciones
- Análisis de volatilidad con ATR
- Consenso de múltiples estrategias
- Historial de trades y métricas de desempeño
- Soporte para testnet/mainnet

## 🚀 Instalación Rápida

```bash
git clone https://github.com/3dw1nb3c3rr4/crypto-trading-ai-toolkit.git
cd crypto-trading-ai-toolkit
pip install -r requirements.txt
```

## 💡 Ejemplos de Uso

### Uso Básico - Indicadores Técnicos

```python
import pandas as pd
from indicators import sma, ema, rsi, macd, bollinger_bands, atr

df = pd.read_csv("sample_data.csv")
df["rsi"] = rsi(df["close"])
df["macd"], df["signal"] = macd(df["close"])
df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger_bands(df["close"], 20, 2)
```

### Bot Maestro Integrado

```python
from bots import MasterTradingBot
import pandas as pd

# Crear bot
bot = MasterTradingBot(exchange="binance", sandbox=True)

# Cargar datos
df = pd.read_csv("sample_data.csv")

# Calcular indicadores
enriched_df = bot.calculate_indicators(df, symbol="BTC/USDT")

# Generar señal combinada (técnica + RL)
signal = bot.generate_signal(enriched_df, use_rl=True)
print(f"Señal: {signal['signal']}")
print(f"Confianza: {signal['confidence']:.2%}")

# Calcular tamaño de posición
position = bot.calculate_position_size(
    account_balance=10000,
    risk_pct=2.0,
    entry_price=enriched_df['close'].iloc[-1],
    stop_loss_price=enriched_df['close'].iloc[-1] * 0.98
)
print(f"Cantidad: {position['position_quantity']:.4f}")
```

Ver `backtest_example.py` para ejemplo completo con datos históricos.

## 📁 Estructura del Proyecto

```
crypto-trading-ai-toolkit/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias
├── LICENSE
│
├── 📊 INDICADORES Y BASE
├── indicators.py                      # Indicadores técnicos
├── sentiment.py                       # Análisis de sentimiento
├── risk_management.py                 # Gestión de riesgo
├── backtest_example.py               # Ejemplo de backtest
│
├── 🤖 BOTS
├── bots/
│   ├── __init__.py
│   ├── master_trading_bot.py         # Bot maestro integrado
│   ├── sesion_cerebro_db.py          # DB de sesiones
│   │
│   ├── cerebro_rl/                   # Sistema RL avanzado
│   │   └── cerebro_rl_core.py        # Core del Cerebro RL
│   │
│   └── techosuelo/                   # Sistema Techo/Suelo
│       ├── binance_topsuelo_bot.py   # Bot Binance
│       ├── binance_meta_filtro.py    # Meta filtro Binance
│       ├── okx_topsuelo_bot.py       # Bot OKX
│       ├── okx_meta_filtro.py        # Meta filtro OKX
│       ├── orderbook_confirmador.py  # Confirmador de orderbook
│       └── red_reintentos.py         # Lógica de reintentos
│
├── 📈 MODELOS PRE-ENTRENADOS
├── models/
│   ├── cerebro_rl.pt                 # Modelo RL entrenado
│   ├── techo_suelo_model.pt          # Modelo Techo/Suelo
│   └── meta_filtro_model.pkl         # Modelo Meta Filtro
│
├── 💾 DATOS
├── data/
│   ├── trading_history/
│   │   ├── ohlcv_cache_2y.pkl        # Cache OHLCV 2 años
│   │   ├── trades_*.csv              # Histórico de trades
│   │   └── active_positions.csv      # Posiciones activas
│   ├── sesion_cerebro.sqlite3        # Base de datos sesiones
│   └── sample_data.csv               # Datos de ejemplo
│
└── 📋 LOGS
    └── logs/                         # Registros de ejecución
```

## 🛠 Tecnologías

**Lenguaje & Procesamiento:**
- Python 3.8+
- Pandas (análisis de datos)
- NumPy (computación numérica)
- PyArrow (manejo eficiente de datos)

**Trading & Exchanges:**
- CCXT (conexión multi-exchange)
- Binance API
- OKX API

**Machine Learning:**
- PyTorch (modelos neuronales)
- Scikit-learn (procesamiento ML)

**Herramientas:**
- TA-Lib (análisis técnico)
- Matplotlib (visualización)
- Python-dotenv (gestión de configs)

## Disclaimer

This project is for **educational and research purposes only**. Nothing here constitutes financial advice. Trading cryptocurrencies involves substantial risk of loss.

## Goal

This project aims to provide educational tools for developers interested in algorithmic trading and machine learning applied to financial markets.

## License

MIT
