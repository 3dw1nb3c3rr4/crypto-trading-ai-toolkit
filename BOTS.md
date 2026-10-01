# 🤖 Documentación de Bots

## Master Trading Bot

El **Master Trading Bot** es la puerta de entrada principal para usar todas las capacidades del toolkit. Combina:

1. **Indicadores Técnicos Clásicos**: Promedios móviles, RSI, MACD, Bollinger Bands, ATR
2. **Modelos RL Avanzados**: Cerebro RL entrenado con histórico de 2 años
3. **Gestión Automática de Riesgo**: Cálculo de posiciones y stop-loss

### Instalación

```bash
pip install -r requirements.txt
```

### Ejemplo Básico

```python
from bots import MasterTradingBot
import pandas as pd

# Inicializar el bot
bot = MasterTradingBot(exchange="binance", sandbox=True)

# Cargar datos
df = pd.read_csv("sample_data.csv")

# Calcular indicadores
df_enriched = bot.calculate_indicators(df)

# Generar señal
signal = bot.generate_signal(df_enriched, use_rl=True)

print(f"Acción: {signal['signal']}")
print(f"Confianza: {signal['confidence']:.2%}")
```

### Estructura de Señal

La salida `generate_signal()` retorna un diccionario con:

```python
{
    "signal": "BUY" | "SELL" | "HOLD",
    "confidence": 0.0 - 1.0,
    "signals_breakdown": {
        "technical": {
            "signal": "BUY",
            "score": 0.25,
            "reasoning": "Cruces de MA + Bollinger Bands"
        },
        "momentum": {
            "signal": "HOLD", 
            "score": 0.0,
            "reasoning": "RSI + MACD"
        },
        "volatility": {
            "signal": "HOLD",
            "score": 0.1,
            "reasoning": "ATR para evaluación de volatilidad"
        },
        "rl": {  # Opcional, si está disponible Cerebro RL
            "signal": "EVALUATE",
            "confidence": 0.5
        }
    },
    "timestamp": "2024-10-01T12:34:56+00:00"
}
```

## Cerebro RL (Reinforcement Learning)

Sistema avanzado de toma de decisiones entrenado con aprendizaje por refuerzo.

### Características

- **Modelos Pre-entrenados**: Incluye modelos entrenados con 2 años de datos
- **87 Características Técnicas**: Análisis profundo del mercado
- **Análisis de Regímenes**: Identifica tendencias, rangos, manipulación, caos
- **Tamaños de Posición**: Pequeño, medio, completo (33%, 66%, 100%)

### Archivo de Configuración Clave

`bots/cerebro_rl/cerebro_rl_core.py`:

```python
# Parámetros principales
TP_PCT = 0.015      # Take profit 1.5%
SL_PCT = 0.015      # Stop loss 1.5%
MAX_HOLD = 288      # Máximo 288 velas
WARMUP = 150        # Calentamiento 150 velas

# Estados posibles
ACCIONES = [
    "esperar",
    "long_pequeño", "long_medio", "long_completo",
    "short_pequeño", "short_medio", "short_completo"
]

REGIMENES = [
    "tendencia_alcista",
    "tendencia_bajista", 
    "rango",
    "trampa_o_manipulacion",
    "caos_alta_volatilidad"
]
```

### Uso en Master Bot

```python
bot = MasterTradingBot()
df_enriched = bot.calculate_indicators(df)

# Automáticamente usa Cerebro RL si está disponible
signal = bot.generate_signal(df_enriched, use_rl=True)

if signal['signals_breakdown'].get('rl'):
    print(f"Decisión RL: {signal['signals_breakdown']['rl']['signal']}")
```

## Sistema Techo/Suelo

Detección automática de niveles de resistencia (techo) y soporte (suelo).

### Componentes

1. **binance_topsuelo_bot.py** / **okx_topsuelo_bot.py**
   - Bot principal de Techo/Suelo
   - Soporte multi-exchange (Binance, OKX)
   - Confirmación con orderbook

2. **binance_meta_filtro.py** / **okx_meta_filtro.py**
   - Filtro meta para señales
   - Análisis de market-making
   - Confirmación de tendencias

3. **orderbook_confirmador.py**
   - Lectura de orderbook en tiempo real
   - Confirmación de niveles de soporte/resistencia
   - Detección de acumulación institucional

### Red de Reintentos

`red_reintentos.py` - Sistema robusto de reconexión:

```python
from bots.techosuelo.red_reintentos import reintentar_acotado

# Reintentos con backoff exponencial
resultado = reintentar_acotado(
    func=lambda: exchange.fetch_ticker(symbol),
    max_intentos=5,
    delay_inicial=1.0
)
```

## Gestión de Riesgo y Posiciones

### Cálculo Automático de Posición

```python
position = bot.calculate_position_size(
    account_balance=10000,      # Saldo total
    risk_pct=2.0,               # Riesgo por trade (2%)
    entry_price=50000,          # Precio de entrada
    stop_loss_price=49000       # Stop loss
)

print(f"Cantidad: {position['position_quantity']:.4f}")
print(f"Riesgo en USD: ${position['risk_amount']:.2f}")
```

### Métricas de Desempeño

```python
metrics = bot.get_performance_summary()
print(f"Total de trades: {metrics['total_trades']}")
print(f"Win rate: {metrics['win_rate']:.2%}")
print(f"P&L total: ${metrics['total_pnl']:.2f}")
print(f"Promedio ganancia: ${metrics['avg_win']:.2f}")
print(f"Promedio pérdida: ${metrics['avg_loss']:.2f}")
```

## Base de Datos de Sesiones

`bots/sesion_cerebro_db.py` - Almacenamiento persistente:

```python
from bots.sesion_cerebro_db import SesionDB

db = SesionDB("data/sesion_cerebro.sqlite3")

# Registrar operación
db.registrar_operacion(
    symbol="BTC/USDT",
    tipo="LONG",
    entrada=50000,
    cantidad=0.1,
    stop_loss=49000
)

# Consultar historial
historial = db.obtener_operaciones(limit=100)
```

## Integraciones Multi-Exchange

### Binance

```python
from bots import MasterTradingBot

bot = MasterTradingBot(exchange="binance", sandbox=True)
# Para mainnet: sandbox=False (requiere credenciales)
```

**API Keys necesarias:**
```bash
export BINANCE_API_KEY="tu_clave"
export BINANCE_API_SECRET="tu_secreto"
```

### OKX

```python
bot = MasterTradingBot(exchange="okx", sandbox=True)
```

**API Keys necesarias:**
```bash
export OKX_API_KEY="tu_clave"
export OKX_SECRET_KEY="tu_secreto"
export OKX_PASSPHRASE="tu_passphrase"
```

## Datos Históricos

### Cache de OHLCV (2 años)

`data/trading_history/ohlcv_cache_2y.pkl`

- Precargado con datos históricos de 2 años
- Usado por Cerebro RL para contexto histórico
- Formato: pickle de pandas DataFrame

### Histórico de Trades

`data/trading_history/trades_*.csv`

Ejemplos incluidos:
- `trades_canasta.py` - Sistema de cestas de trading
- `trades_cerebro_jev.py` - Trades del Cerebro anterior
- `trades_piramide.py` - Estrategia de pirámides

### Posiciones Activas

`data/trading_history/active_positions.csv`

Formato:
```
symbol,side,entry_price,quantity,entry_time,stop_loss,take_profit
BTC/USDT,LONG,50000,0.1,2024-10-01T12:00:00Z,49000,51500
```

## Ejemplos Avanzados

### 1. Análisis Multi-Timeframe

```python
from bots import MasterTradingBot
import pandas as pd

bot = MasterTradingBot()

# Cargar datos de múltiples timeframes
df_1h = pd.read_csv("data_1h.csv")
df_4h = pd.read_csv("data_4h.csv")
df_1d = pd.read_csv("data_1d.csv")

# Enriquecer con indicadores
df_1h = bot.calculate_indicators(df_1h)
df_4h = bot.calculate_indicators(df_4h)
df_1d = bot.calculate_indicators(df_1d)

# Generar señales en cada timeframe
signal_1h = bot.generate_signal(df_1h)
signal_4h = bot.generate_signal(df_4h)
signal_1d = bot.generate_signal(df_1d)

# Consenso: operar si todos están en la misma dirección
if signal_1d['signal'] == 'BUY' and signal_4h['signal'] == 'BUY':
    print(f"Señal fuerte BUY con 1h confirmando")
```

### 2. Backtest con Master Bot

```python
import pandas as pd
from bots import MasterTradingBot

bot = MasterTradingBot()
df = pd.read_csv("historical_data.csv")

# Simular trading
results = []
for i in range(150, len(df)):
    current_data = df[:i].copy()
    enriched = bot.calculate_indicators(current_data)
    signal = bot.generate_signal(enriched)
    
    results.append({
        'timestamp': df.iloc[i]['timestamp'],
        'price': df.iloc[i]['close'],
        'signal': signal['signal'],
        'confidence': signal['confidence']
    })

results_df = pd.DataFrame(results)
results_df.to_csv('backtest_results.csv', index=False)
print(f"Backtest completado: {len(results)} velas analizadas")
```

### 3. Monitoreo en Tiempo Real

```python
import time
from bots import MasterTradingBot
from datetime import datetime

bot = MasterTradingBot(exchange="binance")

# Monitorear continuamente
while True:
    try:
        # Obtener datos del exchange
        ohlcv = bot.exchange.fetch_ohlcv('BTC/USDT', '5m', limit=200)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Procesar
        enriched = bot.calculate_indicators(df)
        signal = bot.generate_signal(enriched)
        
        # Log
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] {signal['signal']} (confianza: {signal['confidence']:.2%})")
        
        time.sleep(300)  # Esperar 5 minutos
        
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
```

## Troubleshooting

### Error: "Cerebro RL no disponible"

```
⚠️  Cerebro RL no disponible - usando indicadores básicos
```

**Solución:**
```bash
# Verificar que el modelo existe
ls -la models/cerebro_rl.pt

# Verificar dependencias
pip install torch scikit-learn
```

### Error: "Insuficientes datos históricos"

El Cerebro RL necesita al menos 150 velas para el warmup.

```python
if len(df) >= 150:
    signal = bot.generate_signal(df, use_rl=True)
else:
    # Usar solo indicadores básicos
    signal = bot.generate_signal(df, use_rl=False)
```

### Conexión Lenta al Exchange

Usar el sistema de reintentos:

```python
from bots.techosuelo.red_reintentos import reintentar_persistente

@reintentar_persistente(max_intentos=10, delay=2.0)
def obtener_ticker(exchange, symbol):
    return exchange.fetch_ticker(symbol)

ticker = obtener_ticker(bot.exchange, 'BTC/USDT')
```

## Contribuciones

Para mejorar los modelos o agregar nuevas estrategias:

1. Crear rama: `git checkout -b feature/nueva-estrategia`
2. Implementar cambios
3. Entrenar modelos (si aplica)
4. Hacer commit: `git commit -m "Add nueva estrategia X"`
5. Push y PR

## Licencia

MIT - Ver LICENSE para detalles
