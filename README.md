# Crypto Trading AI Toolkit

Toolkit en Python para investigación cuantitativa de trading de criptomonedas: indicadores, gestión de riesgo, un motor de backtesting **con comisiones** y los modelos/bots del proyecto "Cerebro" como referencia.

> **Estado honesto:** los backtests con comisiones (`backtesting/REPORT.md`) **no encontraron una estrategia rentable de forma robusta**. Nada de aquí es asesoría financiera ni está listo para operar con dinero real.

## Instalación

```bash
git clone https://github.com/3dw1nb3c3rr4/crypto-trading-ai-toolkit.git
cd crypto-trading-ai-toolkit
pip install -r requirements.txt
```
Windows sin Git: descarga el ZIP de la rama desde GitHub y descomprímelo. Si `ta` falla al instalarse, puedes omitirlo (el toolkit usa sus propios indicadores).

## Qué hay

| Carpeta / archivo | Contenido |
|---|---|
| `indicators.py`, `risk_management.py`, `sentiment.py` | SMA/EMA, RSI, MACD, Bollinger, ATR; tamaño de posición, R/R, max drawdown |
| `backtesting/` | Motor con comisiones (taker 0.05% + slippage por lado), estrategias, optimización walk-forward, evaluación del modelo Techo/Suelo, backtest del Cerebro RL en 5m y descargador de velas |
| `backtesting/REPORT.md` | Resultados y conclusiones de los backtests |
| `colab/descargar_velas_okx.ipynb`, `backtesting/download_okx_5m.py` | Descarga de velas 5m de OKX con ccxt (Colab o local), con reanudación |
| `backtesting/ml_features.py`, `backtesting/ml_walkforward.py` | Modelo nuevo (gradient boosting) con walk-forward, control de etiquetas barajadas y prueba de señal plantada |
| `backtesting/download_funding.py`, `backtesting/funding_carry.py`, `colab/celda_funding.py` | Descarga de funding y evaluación del carry delta-neutral con costos reales (resultado real pendiente de datos) |
| `backtesting/selector.py`, `strategy_selection.json` | Selector: evalúa cada estrategia con una puerta estricta fuera de muestra y con costos; el bot solo opera las aprobadas (hoy: ninguna) |
| `backtesting/ml_daily_xs.py`, `colab/celda_historia_diaria.py` | Modelo cross-sectional diario (117 símbolos) y descarga de ~6 años de velas diarias para confirmarlo; hoy "en observación" |
| `backtesting/ml_daily_xs_v2.py` | Ideas de Qlib/FreqAI/ML4T (Alpha158 en ranking, TopkDropout, IC, Sharpe deflactado); mejor candidato "en observación" |
| `bots/xs_daily/` | Bot nuevo cross-sectional diario, SOLO paper (entrenar, ejecutar a diario, simulación de prueba) |
| `backtesting/download_derivs.py`, `download_binance_metrics.py`, `derivs_features.py`, `funding_arb.py`, `universe_symbols.txt` | Funding y open interest: descarga, características sin fuga de información, arbitraje de funding (Hummingbot) y carry; resultado real pendiente de datos |
| `backtesting/trend_following.py` | Seguimiento de tendencia (Donchian + trailing), candidata evaluada por el selector |
| `bots/master_trading_bot.py` | Bot de investigación: indicadores + (opcional) decisión real del Cerebro RL con velas 5m |
| `bots/cerebro_rl/`, `bots/techosuelo/`, `bots/sesion_cerebro_db.py` | Copias de referencia del código original (no incluyen `trading_engine.py` ni los `main_*`, así que no se ejecutan solas) |
| `models/` | `cerebro_rl.pt`, `techo_suelo_model.pt`, `meta_filtro_model.pkl` |
| `data/trading_history/` | Cache OHLCV diario de 117 símbolos (2 años) y logs de trades del bot |

## Uso

```bash
python backtesting/run_optimization.py     # 648 configs, walk-forward, con comisiones
python backtesting/cross_sectional.py      # long/short neutral al mercado
python backtesting/eval_techo_suelo.py     # modelo Techo/Suelo en validación
python bots/master_trading_bot.py          # demo del bot sobre sample_data.csv
```

Cerebro RL (necesita velas de 5m; ejecútalo en tu PC porque requiere acceso a Binance):
```bash
python backtesting/download_5m.py --days 60 --out ohlcv_5m.pkl
python backtesting/backtest_cerebro_rl_5m.py --data ohlcv_5m.pkl --oos-from 2026-08-01
```

Modelo nuevo sobre velas 5m de OKX (genera `data/okx_5m.pkl` con `colab/celda_unica.py`; no se versiona):
```bash
python backtesting/ml_walkforward.py --variants dir12,dir48,xs12,xs48   # real
python backtesting/ml_walkforward.py --shuffle                          # control nulo
python backtesting/ml_walkforward.py --plant 6 --variants dir48         # prueba de sensibilidad
```

Selector de estrategias (re-ejecútalo cuando tengas más datos o estrategias nuevas):
```bash
python backtesting/selector.py                                    # evalúa todo y actualiza strategy_selection.json
python backtesting/selector.py --funding funding_data.pkl --funding-key okx_funding --spot-key okx_spot_1h   # incluye el carry de funding
python backtesting/ml_daily_xs.py                                 # modelo diario (--shuffle / --plant 6 / --stable / --liq 0.5 / --data <pkl largo>)
python backtesting/ml_daily_xs_v2.py                              # variantes A..Ed (--shuffle / --plant 6 / --stable / --liq 0.5)
python bots/xs_daily/train.py                                     # entrena el modelo del bot nuevo
python bots/xs_daily/run_paper.py                                 # una ejecución diaria en paper (Binance por defecto; --exchange okx)
python bots/xs_daily/test_paper_sim.py                            # simulación día a día con velas reales
python backtesting/test_derivs_features.py                        # pruebas anti-fuga de funding/OI
python backtesting/funding_arb.py --selftest                      # autotest del arbitraje de funding
python backtesting/ml_daily_xs_v2.py --derivs derivs_*.pkl --variants E,F,O,FO   # con datos reales de funding/OI
python bots/test_master_bot.py                                    # tests del bot con selecciones fabricadas
```

`sample_data.csv` es un dataset de ejemplo de 200 velas diarias; no representa un mercado real para evaluar rentabilidad.

## Aviso

Proyecto educativo y de investigación. Operar criptomonedas implica riesgo sustancial de pérdida.

## Licencia

MIT
