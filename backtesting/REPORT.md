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
- Backtest con velas 5m reales de OKX: ver sección 6 (sin ventaja: retorno bruto ≈ 0).
- Scripts: `download_okx_5m.py` / `colab/` para los datos y `backtest_cerebro_rl_5m.py --data <pkl> --oos-from <fecha>`.

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

**Confirmado por el usuario: ambos logs son de paper trading.** Separando los cierres instantáneos (menos de 10 s):

| | Cierres < 10 s | PnL de esos | Resto de trades | PnL del resto | Neto medio del resto |
|---|---|---|---|---|---|
| Canasta | 264 de 1035 (26%) | +$95.16 | 771 | **-$25.90** | -0.072% por trade |
| Pirámide | 152 de 275 (55%) | +$187.84 | 123 | +$6.60 | +0.106% por trade (muestra pequeña) |

- Casi todos los cierres instantáneos son TP, en LONG y en SHORT (canasta: 252 de 264; pirámide: 145 de 152). Eso apunta a un sesgo del paper a favor del TP, no a un mercado bajista.
- Causa probable (no demostrada): en paper, `trading_engine.py` registra la entrada al `last_close` del escáner (líneas ~624 y ~975) y revisa el TP/SL contra el precio en vivo de `fetch_ticker`; si el precio ya se movió desde ese cierre, la posición cierra al instante. En modo real la entrada viene del fill verdadero, así que ese efecto no existe.
- Reproducido con un exchange simulado: el motor original cierra al instante con TP falso (+$0.42 sobre $30) cuando el precio en vivo ya se movió desde el `last_close`; el parche `bots/techosuelo/aplicar_parche_paper.py` lo corrige (ver BOTS.md). Los logs paper anteriores quedan contaminados y conviene rehacerlos con el motor parchado.
- Sin los cierres instantáneos, el rendimiento de canasta es negativo y el de pirámide es pequeño y no concluyente. Los $69 y $194 de ganancia "en paper" no son evidencia de rentabilidad real.

## 6. Velas 5m de OKX (20 perpetuos, 365 días) y modelo nuevo

Datos: 20 símbolos, 105,099 velas comunes (2025-10-01 a 2026-10-01), sin huecos ni anomalías (descargados con `colab/celda_unica.py`).

**Cerebro RL actual:** año completo 31,914 trades, 49.6% de aciertos, retorno **bruto -0.003%**, neto -0.143%, pf 0.82 (puede incluir datos de su entrenamiento, es decir optimista). Solo después del 28-sep: 492 trades, neto -0.27%, pf 0.69. Sin ventaja: TP y SL se alcanzan prácticamente por igual (15,492 vs 15,512).

**Modelo nuevo** (`ml_features.py`, `ml_walkforward.py`): gradient boosting con 47 características causales (retornos de 5m a 24h, volatilidad, velas, volumen, RSI, Bollinger, distancia a extremos, hora/día, contexto de BTC y del mercado, fuerza relativa y ranking entre símbolos). Entrenamiento expansivo, 4 periodos de prueba nunca vistos (días 150-365), purga de 2 días, costo 0.14% por operación. Las 4 variantes se fijaron antes de ver resultados:

| Variante | Operaciones | Bruto | Neto | IC95% neto (por día) | Cartera |
|---|---|---|---|---|---|
| direccional 1h | 29,035 | -0.011% | -0.151% | [-0.181%, -0.123%] | -75.6% |
| direccional 4h | 10,818 | -0.037% | -0.177% | [-0.263%, -0.094%] | -40.9% |
| long/short cross-sectional 1h | 15,472 periodos | -0.001% | -0.141% | [-0.148%, -0.135%] | n/a |
| long/short cross-sectional 4h | 3,866 periodos | -0.009% | -0.149% | [-0.172%, -0.126%] | n/a |

Controles de validez:
- **Etiquetas barajadas** (no puede haber ventaja): bruto -0.002% (1h), -0.034% (4h), -0.017% (xs 4h). Igual que el modelo real: no aprendió nada predictivo.
- **Señal plantada** (feature con correlación ≈ 0.16 con el retorno futuro a 4h): el mismo código gana +0.10% neto por trade (IC95% [+0.03%, +0.17%]) y +58% en cartera con caída máxima -9.8%. Es decir, la canalización sí detecta una ventaja de ese tamaño; su ausencia en datos reales es creíble.

**Conclusión:** con velas OHLCV de 5m y características de mercado, en 20 perpetuos líquidos, no hay ventaja predictiva a 1h/4h antes de comisiones. Cambiar a un modelo más complejo (RL, redes) no lo arregla: el límite es la información, no el modelo. Lo que sí podría tener ventaja y no se ha probado: datos distintos (funding, open interest, libro de órdenes, liquidaciones) o estrategias estructurales que no dependen de predecir dirección (p. ej. carry de funding delta-neutral).

### 6b. Búsqueda automática de hiperparámetros (12 configuraciones por fold, elegidas solo con la ventana de validación)

| Variante | Neto sin búsqueda | Neto con búsqueda | Control barajado + búsqueda |
|---|---|---|---|
| direccional 1h | -0.151% | -0.149% | n/d |
| direccional 4h | -0.177% | -0.186% | -0.176% |
| cross-sectional 1h | -0.141% | -0.141% | n/d |
| cross-sectional 4h | -0.149% | -0.136% | -0.132% |

Sin mejora real: la búsqueda no crea ventaja y el pequeño cambio en xs 4h es igual al del control barajado (ruido). Como la elección usa solo validación, el resultado en prueba sigue siendo limpio.

## 7. Carry de funding delta-neutral (herramientas listas; resultado real pendiente de datos)

- `download_funding.py` y `colab/celda_funding.py`: bajan funding (OKX solo entrega ~3 meses; Binance todo el año pero bloquea EE.UU.) y velas spot 1h de OKX. Probados con exchanges simulados, no con APIs reales.
- `funding_carry.py`: P&L = funding cobrado + base spot-perp - costos (0.19% por entrada y por salida, ambas patas), con grilla de 36 configuraciones, walk-forward y chequeo de liquidación. Autotest con casos de respuesta conocida: costo exacto, funding acumulado coincide con lo esperado.
- `ml_features.add_funding_features`: funding como característica del modelo, alineada sin mirar el futuro (0 filas mal alineadas en la prueba).
- **Riesgo de liquidación del corto en el perp** (precios reales, ventanas de 30 días): a 1x 1.2% de las ventanas, 2x 4.8%, 3x 12.3%, 5x 26.9% suben lo suficiente para liquidar. Los altcoins pueden subir más de 100% en un mes; el carry tiene riesgo de cola real y conviene apalancamiento bajo y símbolos líquidos.

## Limitaciones de este análisis

Velas diarias (no 5m); solo 2 años y un ciclo de mercado; universo de símbolos listados hoy (sesgo de supervivencia); sin funding ni profundidad del libro; los modelos pudieron entrenarse con parte de estos mismos datos.

## Siguientes pasos recomendados

1. Probar fuentes de información nuevas (funding, open interest, libro de órdenes) o carry de funding; con solo OHLCV ya no hay más que explorar (sección 6).
2. Verificar si los logs de canasta/pirámide son paper o real. Si son paper, repetir con rellenos al bid/ask, taker en ambos lados y latencia, o con nocional mínimo en real durante 2-4 semanas.
3. No operar con dinero real una estrategia hasta que tenga expectativa neta positiva fuera de muestra en al menos 3 periodos y más de ~300 trades.
