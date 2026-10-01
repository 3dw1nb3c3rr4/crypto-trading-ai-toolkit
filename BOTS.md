# Componentes del proyecto Cerebro (referencia)

Resumen verificado contra el código y los checkpoints. Para el rendimiento medido ver `backtesting/REPORT.md`.

## Cerebro RL (`bots/cerebro_rl/cerebro_rl_core.py`, `models/cerebro_rl.pt`)
- Red de decisión de entrada/salida con **37 features** de velas de **5m** (`FEATURE_NAMES` en el código).
- Acciones: esperar, long/short en tamaño pequeño (33%), medio (66%) o completo (100%).
- Config de entrenamiento: TP=SL=1.5%, máx. 288 velas, fee 0.05%, slippage 0.02%.
- El checkpoint declara `modelo: "Imitación pura"` (imita al sistema JEV) y `val_usd: -107.84`.
- Requiere al menos 151 velas de 5m. `CerebroRL.cargar(ruta).decidir_entrada(ohlcv)`.

## Techo/Suelo (`bots/techosuelo/`, `models/techo_suelo_model.pt`, `models/meta_filtro_model.pkl`)
- Cerebro 1 (`binance_topsuelo_bot.py`, `okx_topsuelo_bot.py`): red multi-escala sobre velas **diarias**; detecta posibles techos (SHORT) y suelos (LONG) con horizonte de 7 días.
- Cerebro 2 (`*_meta_filtro.py`): clasificador que estima `prob_exito_3d` de cada señal.
- `orderbook_confirmador.py`: confirmación con libro de órdenes. `red_reintentos.py`: reintentos ante errores de red (`reintentar_acotado`, `reintentar_persistente`, `buscar_posicion_abierta`).
- `sesion_cerebro_db.py`: guarda/carga la sesión en SQLite (`guardar_sesion`, `cargar_sesion`, `borrar_sesion_y_archivos`).

## Qué falta para ejecutarlos desde este repo
Las copias no incluyen `trading_engine.py` (motor paper/real, modos canasta y pirámide) ni los `main_*`; la versión ejecutable es la carpeta original. Los modelos se buscan en la misma carpeta que los scripts.

## Master Trading Bot (`bots/master_trading_bot.py`)
Combina puntuaciones de SMA, Bollinger, RSI y MACD y, si se le pasan velas 5m, la decisión real del Cerebro RL (si discrepan, no opera). Es una herramienta de investigación: la combinación no está validada.
