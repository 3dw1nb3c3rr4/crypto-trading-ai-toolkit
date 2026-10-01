# Resultados de backtesting (con comisiones)

**Conclusión corta: con los datos disponibles no encontré ninguna estrategia rentable de forma robusta, y no voy a presentar una como si lo fuera.** Abajo está la evidencia, qué se pudo y qué no se pudo probar, y los siguientes pasos.

## Metodología

- Datos: `data/trading_history/ohlcv_cache_2y.pkl`, 117 perpetuos USDT, velas **diarias**, 2024-09-05 a 2026-09-04.
- Costos: taker 0.05% por lado + slippage 0.02% por lado = **0.14% ida y vuelta** (igual que `DEFAULT_FEE` del bot). Sin funding.
- Señal al cierre de la vela t, entrada en la apertura de t+1; si TP y SL se tocan en la misma vela se asume SL primero; un trade por símbolo a la vez.
- Optimización **walk-forward anclada**: 3 folds; la configuración se elige solo con datos previos al fold y se mide en el periodo siguiente (nunca visto al elegir).
- Reproducir: `python backtesting/run_optimization.py`, `cross_sectional.py`, `eval_techo_suelo.py`.

## 1. Estrategias con indicadores del toolkit (RSI, Bollinger, Donchian, tendencia+pullback, agotamiento, momentum)

648 configuraciones (estrategia x parámetros x TP/SL/horizonte).

| Fold | Mejor config en entrenamiento | Expectativa train | Expectativa OOS | Trades OOS |
|---|---|---|---|---|
| 1 | trend_pullback TP12/SL6/14d | +0.85% | +0.92% | 272 |
| 2 | misma | +0.87% | +0.51% | 401 |
| 3 | misma | +0.78% | **-1.49%** | 381 |

- Configuraciones con expectativa neta positiva en los 3 periodos de prueba: **0 de 648**.
- OOS combinado: 1054 trades, 35.6% aciertos, expectativa bruta +0.03%, **neta -0.11%**, factor de beneficio 0.97.
- Cartera (10% del capital por trade, máx. 10 abiertas): **-51%**, drawdown -62%. Las pérdidas llegan agrupadas (muchos SL el mismo día en altcoins correlacionadas), por eso es peor que la suma simple (-11%).

Sensibilidad a comisiones (misma muestra OOS): sin costos +0.03%/trade; maker+maker -0.05%; taker (base) -0.11%; taker con slippage x3 -0.19%. El edge bruto es casi nulo: las comisiones empeoran el resultado, pero no son la causa principal.

## 2. Long/short cross-sectional neutral al mercado

24 variantes (momentum/reversión, lookback 3-30d, rebalanceo 3-14d, 20% extremos). Ninguna rentable en los 3 periodos de forma consistente (la única "positiva" en los 3 tiene retorno neto negativo en los 2 años completos: ruido de comparar 24 variantes).

## 3. Modelo Techo/Suelo (Cerebro 1), solo tramo de validación (último 15% por símbolo)

- Config nativa del modelo (TP 4%, SL 4%, 7 días): 1125 trades, 47.7% aciertos, expectativa bruta -0.20%, **neta -0.34%**, pf 0.84. LONG y SHORT pierden.
- Grilla de 72 combinaciones: 16 con expectativa neta > 0; la mejor +0.30% (pf 1.09). Con tantas combinaciones eso es compatible con azar.
- Cartera con la config nativa: -38.7%.
- El Meta Filtro no se puede medir limpio con este cache: se entrenó con split aleatorio sobre predicciones del Cerebro 1 que incluyen su propio periodo de entrenamiento.

## 4. Cerebro RL

- El checkpoint indica: modelo "Imitación pura" (imita al sistema JEV anterior), 37 features, velas de **5m**, TP=SL=1.5%, y `val_usd = -107.84` guardado en su validación.
- No se pudo backtestear aquí: no hay velas de 5m en el repo y Binance no es accesible desde este entorno.
- Listo para correr en tu PC: `download_5m.py` y luego `backtest_cerebro_rl_5m.py --data ohlcv_5m.pkl --oos-from <fecha posterior al entrenamiento>`. El script se probó solo con datos sintéticos (mecánica), no con datos reales.

## 5. Tus logs de operación (`data/trading_history/trades_*.csv`)

| | Canasta | Pirámide |
|---|---|---|
| Trades / días | 1035 / 22 | 275 / 8 |
| Aciertos | 68% | 84% |
| PnL neto / comisiones | +$69 / $26.6 | +$194 / $9.7 |
| Neto por trade (IC95%, bootstrap por día) | +0.21% [-0.20%, +0.53%] | +1.90% [+0.16%, +2.93%] |

Hallazgos que conviene verificar antes de confiar en estos números:

1. **Canasta: todo el beneficio viene de trades que cierran en menos de 5 min** (438 trades, +$107.65). Los que duran más de 5 min suman -$38. El intervalo de confianza incluye cero.
2. **Pirámide: 55% de los trades cierran en menos de 10 s**, incluso TP de +0.7% en 0.08 s después de abrir. Eso no es alcanzable en ejecución real y es típico de un artefacto de paper trading (relleno al precio de pantalla sin spread, latencia ni cola de órdenes). Si esos logs son de paper, el 84% de aciertos no es evidencia de rentabilidad real.
3. `prob_exito` del Meta Filtro: los trades con prob > 0.55 ganan 50-55% y pierden dinero; los de 0.40-0.55 son los rentables. La confianza del modelo no parece estar bien calibrada en vivo.
4. El PnL de canasta es casi todo SHORT (+$75 vs LONG -$6): puede reflejar un mercado bajista en esas semanas, no una ventaja del modelo.

## Limitaciones de este análisis

Velas diarias (no 5m); solo 2 años y un ciclo de mercado; universo de símbolos listados hoy (sesgo de supervivencia); sin funding ni profundidad del libro; los modelos pudieron entrenarse con parte de estos mismos datos.

## Siguientes pasos recomendados

1. Backtest del Cerebro RL con 5m reales (scripts listos) y solo en fechas posteriores a su entrenamiento.
2. Verificar si los logs de canasta/pirámide son paper o real. Si son paper, repetir con rellenos al bid/ask, taker en ambos lados y latencia, o con nocional mínimo en real durante 2-4 semanas.
3. No operar con dinero real una estrategia hasta que tenga expectativa neta positiva fuera de muestra en al menos 3 periodos y más de ~300 trades.
