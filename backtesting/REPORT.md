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

## 8. Selector de estrategias: qué se evaluó y qué quedó aprobado

`backtesting/selector.py` somete cada candidata a la misma puerta estricta y escribe `strategy_selection.json`, que lee el bot. Puerta (todas obligatorias): al menos 300 operaciones fuera de muestra; IC95% del neto (bootstrap por día) con límite inferior > 0; neto > 0 en todos los periodos de prueba; factor de beneficio ≥ 1.10; caída máxima de cartera no peor que -30%. Antes de la puerta se resta un **recorte de seguridad de 0.15% por trade**: con martingalas sintéticas (donde ninguna estrategia puede ganar) el motor mostró un optimismo residual de ~+0.15% por trade en salidas por stop, causado por asumir que el stop se ejecuta exactamente en su precio. Se añadió además 0.10% de deslizamiento extra en los stops.

| Estrategia | Operaciones | Neto/trade* | IC95% inferior | pf | Veredicto |
|---|---|---|---|---|---|
| indicadores: RSI reversión | 2,156 | -0.33% | -1.25% | 0.92 | rechazada |
| indicadores: Bollinger reversión | 5,829 | -0.36% | -0.73% | 0.82 | rechazada |
| indicadores: Donchian breakout | 2,708 | -0.20% | -1.17% | 0.94 | rechazada |
| indicadores: tendencia + pullback | 1,054 | -0.32% | -1.45% | 0.92 | rechazada |
| indicadores: agotamiento | 1,762 | -0.84% | -1.71% | 0.78 | rechazada |
| indicadores: momentum | 6,719 | -0.19% | -0.78% | 0.94 | rechazada |
| **tendencia Donchian + trailing (nueva)** | 508 | **+1.33%** | -3.63% | 1.14 | rechazada |
| modelo Techo/Suelo | 1,125 | -0.54% | -1.29% | 0.76 | rechazada |
| modelo Cerebro RL (5m OKX) | 16,845 | -0.38% | -0.43% | 0.60 | rechazada |
| gradient boosting (4 variantes, 5m OKX) | 3,866 a 29,035 | -0.29% a -0.33% | -0.30% a -0.42% | 0.07 a 0.58 | rechazadas |

\* después del recorte de seguridad. **Estrategias aprobadas: ninguna.** El bot responde HOLD.

Tendencia con trailing, el único caso con neto medio positivo: es el perfil típico de esta clase (mediana por trade -2.9%, 45% de aciertos) y la ganancia depende de pocas operaciones extremas. Los 10 mejores trades aportan el 106% de la ganancia total y sin ellos la media es -0.08%. Por periodo: +5.6% en el primero (caída de altcoins de fin de 2025, casi todo por cortos) y -0.2% / -0.9% en los otros dos. Con solo 2 años de datos no es una ventaja demostrable; conviene reevaluarla cuando haya más historia.

Nota: las secciones 1 a 6 se calcularon con stops ejecutados exactamente en su precio; con el deslizamiento extra los resultados son ligeramente peores y las conclusiones no cambian.

## 9. Búsqueda de ventaja: modelo cross-sectional diario sobre 117 perpetuos

Primera señal que supera la prueba principal. `ml_daily_xs.py`: gradient boosting con 33 características diarias (retornos, volatilidad, volumen/liquidez, distancia a extremos, velas, mercado, rankings), walk-forward con 5 periodos de prueba y purga, long los 10 mejores / short los 10 peores por cohorte de 7 días, costo 0.14% por cohorte sobre todo el nocional.

| Variante (4 fijadas de antemano) | Neto por cohorte | IC95% | Sharpe anual |
|---|---|---|---|
| 3 días, gradient boosting | +0.10% | [-0.17%, +0.39%] | 0.46 |
| 3 días, ridge | +0.01% | [-0.23%, +0.27%] | 0.06 |
| **7 días, gradient boosting** | **+0.81%** | **[+0.29%, +1.41%]** | **1.74** |
| 7 días, ridge | +0.42% | [-0.08%, +0.94%] | 1.02 |

La variante de 7 días con gradient boosting es positiva en los 5 periodos (+0.45%, +1.36%, +0.67%, +0.19%, +1.63%).

Controles: etiquetas barajadas -0.22% (sin ventaja); señal plantada +2.47% (se detecta). Quitar un grupo de información a la vez deja +0.61% a +0.96%: la señal no depende de un solo tipo de dato.

**Pero es frágil en el universo**, y por eso queda "en observación" y no aprobada:

| Variante de robustez | Neto | IC95% |
|---|---|---|
| universo estable (solo símbolos con historia completa) | +0.06% | [-0.25%, +0.34%] |
| 50% más líquido | +0.25% | [-0.05%, +0.55%] |
| costo doble | +0.52% | [+0.21%, +0.84%] |

Hallazgos que explican la cautela:
- La ganancia viene sobre todo de la pata corta: los shorts elegidos cayeron -2.48% por semana contra -0.93% de la moneda promedio. Todo el periodo de prueba fue una caída fuerte de altcoins.
- El universo (117 símbolos) se eligió con el ranking de volumen y volatilidad de hoy: sesgo de supervivencia. Una parte de la ventaja desaparece al quitar monedas nuevas o menos líquidas, que son justo las más caras de operar de verdad (el costo de 0.14% probablemente está subestimado para ellas).
- Solo hay un ciclo de mercado.

Estado en el selector: **EN OBSERVACIÓN** (solo paper). Para confirmarla o descartarla hace falta más historia con otros regímenes (`colab/celda_historia_diaria.py`, ~6 años) y luego `python backtesting/ml_daily_xs.py --data <archivo>` y el selector con `--daily-data`.

## 10. Ideas de repositorios con más de 1,000 estrellas y bot nuevo (paper)

Repos revisados (estrellas verificadas en cada página de GitHub): freqtrade 55.1k, microsoft/qlib 49.2k, stefan-jansen/machine-learning-for-trading 21.2k, hummingbot 20.3k, FinRL 16.6k, jesse 8.6k, hftbacktest 4.9k, passivbot 2.1k.

Ideas aplicadas (`ml_daily_xs_v2.py`):
- **Qlib:** features tipo Alpha158 (velas, MA/STD/MAX/MIN/cuantiles/RANK/RSV/IMAX/IMIN/correlación precio-volumen/CNTP/SUMP/volumen, regresión BETA/RSQR/RESI en 5 ventanas), normalización por ranking entre símbolos, etiqueta de ranking, gradient boosting muy regularizado, métricas IC/ICIR y estrategia TopkDropout (rotación limitada).
- **FreqAI:** expansión de features en varias ventanas.
- **ML4T:** walk-forward y Sharpe deflactado.
- **Descartado por falta de datos:** el market making con órdenes límite (Hummingbot, Passivbot, hftbacktest) necesita libro de órdenes por tick; no se puede evaluar con velas.

Resultado (7 variantes fijadas de antemano, H=7 días, K=10 por lado): normalizar por ranking duplica el IC (de 0.063 a 0.13).

| Variante | IC | Neto por cohorte | Periodos positivos |
|---|---|---|---|
| A referencia (features base) | 0.063 | +0.73% | 4/5 |
| D Alpha158 en ranking | 0.112 | +1.01% | 5/5 |
| E base+Alpha158 en ranking | 0.130 | +1.06% | 4/5 |
| Ed E con TopkDropout | 0.130 | +0.165% por día | 5/5 |

Controles: etiquetas barajadas IC ≈ 0 y neto negativo; señal plantada detectada. El IC es positivo y significativo también en el universo estable (0.09 a 0.12) y en el 50% más líquido (0.15 a 0.16).

Selector (con bootstrap por bloques de 7 días, corregido porque las cohortes se superponen):
- **D: EN OBSERVACIÓN.** Pasa la prueba principal, el 50% más líquido (+0.85%, IC95% [+0.40%, +1.33%]) y el costo doble, pero no el universo estable (+0.11%, IC95% [-0.49%, +0.62%]).
- **E: rechazada** por un periodo levemente negativo, aunque mantiene el IC95% > 0 en los tres universos.

Advertencias: el Sharpe deflactado de la serie diaria (Ed) es 0.58 con 20 pruebas y 0.15 con 700, así que no es concluyente tras corregir por pruebas múltiples. El modelo tiende a ir largo en monedas grandes y corto en altcoins pequeñas y recientes, una apuesta que funcionó en este periodo bajista y podría fallar en una temporada alcista de altcoins. Solo hay 2 años y un ciclo.

**Bot nuevo** (`bots/xs_daily/`, solo paper): entrena la variante D, abre cada día una cohorte de 10 long y 10 short con 1/7 del capital, la cierra a los 7 días y ejecuta al precio en vivo (ask/bid + slippage + comisión). Solo corre si la estrategia está aprobada o en observación. Simulación día a día con velas reales y modelo entrenado solo hasta el 31-may-2026: 89 cohortes, 0 errores de contabilidad, +1.53% por cohorte fuera de muestra, 71% positivas, $1,000 → $1,198 realizados (jun-sep 2026, mismo periodo bajista que el último tramo del walk-forward).

## 11. Confirmación con 6 años de historia (2020-10 a 2026-10, 105 a 135 símbolos)

Archivo: `colab/celda_historia_diaria.py` (OKX, velas diarias, sin huecos). Mismo walk-forward, ahora con reentrenamiento cada 180 días (11 periodos), K = 12% de los símbolos válidos por lado y costo 0.14% por cohorte. Todo lo anterior a 2024-09-05 es **dato nunca usado** en análisis previos.

| Variante | IC | Neto por cohorte | IC95% | Sharpe | Caída máx. | Datos nuevos (< sep-2024) |
|---|---|---|---|---|---|---|
| A (features base, referencia) | 0.057 | +0.25% | [-0.29%, +0.76%] | 0.30 | -54% | -0.30% [-1.09%, +0.39%] |
| D (Alpha158 en ranking) | 0.132 | +0.26% | [-0.27%, +0.78%] | 0.30 | -51% | +0.07% [-0.72%, +0.84%] |
| E (base+Alpha158 en ranking) | 0.141 | +0.26% | [-0.30%, +0.78%] | 0.29 | -42% | +0.02% [-0.78%, +0.83%] |
| Ed (E con TopkDropout) | 0.141 | +0.025% por día | [-0.08%, +0.12%] | 0.24 | -69% | -0.04% por día [-0.17%, +0.08%] |

Control con etiquetas barajadas (6 años): IC ≈ 0, neto -0.25% / -0.23%, datos nuevos -0.19% / -0.06%. No hay fuga de información.

Por año (variante D, neto por cohorte): 2022 +0.47% (mercado -1.91%), 2023 **-0.68%** (mercado +1.86%), 2024 +0.37%, 2025 +0.88% (mercado -1.64%), 2026 +0.26%.

**Conclusión:** el resultado de +0.8% a +1.0% de los 2 años recientes no se sostiene en 6 años. El IC sigue siendo positivo (0.13 a 0.14: el modelo ordena mejor que el azar), pero la cartera long/short de los extremos solo rinde ~+0.26% por cohorte, sin significancia, con caídas de -40% a -69%; el segmento anterior a sep-2024 es estadísticamente cero. Las variantes quedan **rechazadas** por el selector con estos datos. El patrón por año (gana en mercados bajistas, pierde en el alcista de 2023) sugiere dependencia del régimen.

### 11b. Neutralizar el beta, carteras más amplias y costos de ejecución (variante D, 6 años)

| Cartera | Neto por cohorte | IC95% | Alfa vs mercado | Caída máx. | Datos nuevos (< sep-2024) |
|---|---|---|---|---|---|
| K=12% por lado, sin neutralizar | +0.26% | [-0.27%, +0.78%] | +0.24% [-0.30%, +0.75%], beta -0.11 | -51% | +0.07% |
| K=12%, neutral en beta | +0.26% | [-0.22%, +0.76%] | +0.27% [-0.24%, +0.79%], beta +0.02 | -37% | +0.18% [-0.54%, +1.00%] |
| K=25%, neutral en beta | +0.19% | [-0.12%, +0.50%] | +0.19% [-0.12%, +0.51%] | -19.5% | +0.23% [-0.21%, +0.70%] |
| K=40%, neutral en beta | +0.14% | [-0.09%, +0.36%] | +0.14% [-0.09%, +0.36%] | -22% | +0.21% [-0.11%, +0.54%] |

- Parte de la ganancia sin neutralizar venía de un beta significativamente negativo (-0.11: estar levemente corto del mercado). Al neutralizarlo, la caída máxima baja y los años quedan más parejos, pero el alfa sigue sin ser significativo.
- Sensibilidad al costo (K=25%, neutral en beta): 0.14% → +0.19%; 0.08% → +0.25% [-0.06%, +0.56%]; 0.04% (solo maker, hipotético) → +0.29% [-0.02%, +0.60%]. Ni con ejecución maker el intervalo se separa de cero.
- El modelo ordena mejor que el azar (IC 0.13, significativo), pero ese poder de ordenamiento es demasiado pequeño en términos económicos: ~+0.3% bruto por semana contra 0.14% de costo.

**Selector con el historial largo** (`python backtesting/selector.py --daily-data data/ohlcv_daily_long.pkl`): ninguna estrategia aprobada ni en observación. Las variantes D y E quedan rechazadas (neto tras recorte +0.11%, IC95% inferior -0.42%; periodos 2, 3, 5 y 10 negativos). Las tres pruebas de robustez tampoco superan la barrera.

## 12. Funding y open interest: herramientas listas, resultado real pendiente de datos

**Código de GitHub revisado** (estrellas verificadas en cada página): [hummingbot](https://github.com/hummingbot/hummingbot) 20.3k, [cryptofeed](https://github.com/bmoscon/cryptofeed) 2.9k (canales FUNDING, OPEN_INTEREST y LIQUIDATIONS), [binance-public-data](https://github.com/binance/binance-public-data) 2.5k (su README solo documenta trades y velas; no confirma OI ni funding).
- De Hummingbot (`scripts/v2_funding_rate_arb.py`): funding normalizado a base diaria, entrada cuando la diferencia entre exchanges supera un umbral, salida por take profit (PnL + funding cobrado), por inversión de la diferencia o por triple barrera, y comisiones de ambas patas descontadas de la rentabilidad. Implementado en `funding_arb.py` (modo `cross`) con costo de 4 patas.
- De cryptofeed: qué canales existen (funding, OI, liquidaciones) para decidir qué descargar.

**Qué se construyó** (probado con datos sintéticos y exchanges simulados; no se ha corrido contra las APIs reales ni con datos reales de OI):
- `download_derivs.py`: funding + OI diario por ccxt (Bybit, Binance, OKX). `download_binance_metrics.py`: OI diario desde los archivos públicos de Binance, con `--probe` que comprueba si existen y se leen antes de la descarga masiva.
- `derivs_features.py`: características de funding (suma diaria, medias, z-score, cambio, fracción de días positivos) y de OI (cambios 1/7/30 días, z-score, OI/volumen, divergencia OI-precio). Alineación anti-sesgo con 11 pruebas: modificar datos posteriores a t no cambia ninguna característica hasta t; el OI se retrasa un día por seguridad.
- `ml_daily_xs_v2.py --derivs ...`: variantes F (funding), O (OI) y FO, comparadas de forma **pareada** contra E (mismo modelo sin derivados) sobre las mismas fechas.
- `selector.py --derivs ...`: una variante con derivados solo se aprueba si pasa la puerta Y mejora a E de forma significativa.
- `funding_arb.py`: arbitraje de funding entre exchanges (estilo Hummingbot) y carry sin base spot (cota superior). Autotest: diferencia constante entra una vez y acumula lo esperado; sin diferencia no opera; una diferencia que se invierte cada 5 días pierde por costos.

Controles del pipeline: con derivados sin información, F/O/FO no cambian el resultado de E (diferencia pareada -0.05% a 0.00%, no significativa) y el selector rechaza; con información plantada en el OI, la mejora pareada es +12.5% [+11.6%, +13.5%] y el selector aprueba. Es decir, si hay información real en el OI o el funding, el sistema la detectará; si no la hay, no inventará una ventaja.

### 12b. Celda única de Colab para funding y OI (OKX + archivos públicos de Binance)

`colab/celda_derivados.py` descarga en dos etapas y termina con un **reporte de cobertura** (símbolos, historial mediano, fecha más antigua) para saber exactamente qué entregó cada fuente: (1) OKX por API (funding y OI; el historial que entregue se informa, no se asume); (2) Binance `data.binance.vision`: funding mensual (desde 2020) y OI diario (desde 2021-12), comprobando antes con `--probe` que los archivos existen y se leen. Prueba de extremo a extremo con exchange y servidor simulados (incluidos tres formatos de fecha distintos) y carga posterior en el modelo.

Defectos encontrados y corregidos al probarla: (a) pandas 3 trata el texto como `str` (no `object`) y las fechas pueden venir en microsegundos, lo que rompía la conversión a milisegundos del funding; ahora `to_ms` es robusta al tipo y a la resolución; (b) `load_derivs` se quedaba con el primer archivo que tuviera cada símbolo, de modo que 90 días de OKX tapaban 450 días de Binance; ahora elige por símbolo la fuente de mayor historial y no mezcla fuentes (los niveles de OI difieren entre exchanges).

## Limitaciones de este análisis

Velas diarias (no 5m); solo 2 años y un ciclo de mercado; universo de símbolos listados hoy (sesgo de supervivencia); sin funding ni profundidad del libro; los modelos pudieron entrenarse con parte de estos mismos datos.

## Siguientes pasos recomendados

1. Probar fuentes de información nuevas (funding, open interest, libro de órdenes) o carry de funding; con solo OHLCV ya no hay más que explorar (sección 6).
2. Verificar si los logs de canasta/pirámide son paper o real. Si son paper, repetir con rellenos al bid/ask, taker en ambos lados y latencia, o con nocional mínimo en real durante 2-4 semanas.
3. No operar con dinero real una estrategia hasta que tenga expectativa neta positiva fuera de muestra en al menos 3 periodos y más de ~300 trades.


## 13. Kronos (modelo fundacional de velas): integración lista, resultado real pendiente

Se integró un pronosticador tipo Kronos (shiyu-coder/Kronos, MIT; tokenizador + transformer preentrenado con 12B+ velas de 45+ exchanges) como **características** del modelo diario cross-sectional: retorno pronosticado a 7 días, volatilidad pronosticada, fracción de días al alza, máximo y mínimo pronosticados (`kronos_features.py`). Variante **K = E + Kronos**, comparada de forma pareada con E (mismas fechas, desde 180 d después del primer pronóstico) en `ml_daily_xs_v2.py --kronos` y en `selector.py --kronos`.

Estado verificado:
- Pruebas anti-fuga (`test_kronos_features.py`): alterar todo lo posterior a la fecha t no cambia el pronóstico de t; alineación sin relleno hacia adelante; resultado independiente del tamaño de lote.
- Pipeline completo ejecutado con un predictor simulado (extrapola momentum): K - E = -0.07% por cohorte, IC95% [-0.23%, +0.10%] (sin mejora, como debe ser: el simulado no aporta información nueva). El selector lo rechaza.
- **Los pesos reales no se pudieron descargar** (el entorno bloquea huggingface.co), así que no hay ningún resultado con Kronos real. Se ejecuta en Colab con GPU (`colab/celda_kronos.py`).

Advertencias para interpretar el resultado real:
- **Contaminación**: Kronos no publica su fecha de corte; el periodo antiguo pudo estar en su preentrenamiento y daría una mejora falsa. Solo las fechas posteriores al corte son una prueba limpia; se debe mirar el resultado por año y compararlo con el control de etiquetas barajadas.
- Se evalúa con el mismo juicio estricto: IC95% inferior > 0 tras costos y mejora pareada significativa frente a E; si no, no se opera.

## 14. Funding real de Binance (127 símbolos, 2020-2026) en el modelo diario

Datos: archivos públicos de Binance, funding desde 2020 (105 símbolos en el panel, 1725 cohortes de 7 d desde 2022-01, K = 12 %, re-entrenamiento cada 180 d, costo 0.14 % ida/vuelta). Resultado de la corrida del usuario:

| | neto por cohorte | IC95 % | Sharpe | maxDD |
|---|---|---|---|---|
| E (sin derivados) | +0.292 % | [-0.234 %, +0.809 %] | 0.34 | -38 % |
| F (E + funding) | +0.337 % | [-0.226 %, +0.867 %] | 0.38 | -43 % |
| F - E (pareado) | +0.045 % | [-0.134 %, +0.213 %] | | |

Conclusión: el funding **no mejora de forma significativa** al modelo, y E mismo no es distinguible de cero con 6 años de datos (por año: 2022 +0.53 %, 2023 -0.39 %, 2024 +0.17 %, 2025 +1.00 %, 2026 +0.11 %). Antes de 2024-09 (datos nuevos): E +0.14 % [-0.61 %, +0.90 %]. El selector no lo aprueba. El open interest (variantes O/FO) sigue pendiente de la descarga.

## 15. Simulación del bot paper día a día con 6 años de datos (fuera de muestra)

`bots/xs_daily/test_paper_sim.py --data data/ohlcv_daily_long.pkl --start 2025-01-01`: el modelo (variante D del bot) se entrena solo con datos hasta 2024-12-31 y NO se re-entrena; luego opera 644 días con un exchange simulado (entrada al open del día con ask/bid ± slippage, comisión de salida 0.05 %). Contabilidad verificada de forma independiente: 0 errores en 637 cohortes cerradas.

- Retorno medio por cohorte de 7 d: **+0.344 %** (sin la comisión de entrada, ≈ -0.05 % del capital de la cohorte → ≈ +0.29 % neto). Bootstrap por bloques de 7 d: IC95 % [+0.01 %, +0.65 %]; 59 % de cohortes positivas.
- Por trimestre: 25Q1 +0.75 %, 25Q2 -0.04 %, 25Q3 +0.43 %, 25Q4 +0.28 %, 26Q1 +0.42 %, 26Q2 -0.51 %, 26Q3 +1.08 %.
- Peor cohorte -11.6 %, mejor +22.8 %.
- Es consistente con el hallazgo de la sección 14 (≈ +0.3 % por cohorte, ventaja pequeña y marginal). El extremo inferior del IC apenas supera cero y el periodo coincide con el mejor año del modelo (2025), así que NO se considera confirmado: falta validación en vivo.

Paper trading en vivo (`run_paper.py --force`, `resumen_paper.py`): su único propósito es acumular evidencia real sin riesgo; con cohortes diarias solapadas hacen falta ~7 meses (30 cohortes independientes) para concluir algo.

## 16. Open interest real de Binance (127 símbolos, OI desde 2021-12) en el modelo diario

Misma prueba que la sección 14 (105 símbolos en el panel, 1725 cohortes de 7 d desde 2022-01, K = 12 %, costo 0.14 %), con funding + OI de los archivos públicos de Binance (OI con 1 día de retraso).

| variante | neto/cohorte | IC95 % | Sharpe | maxDD | pareado vs E | IC95 % pareado |
|---|---|---|---|---|---|---|
| E (sin derivados) | +0.292 % | [-0.234 %, +0.809 %] | 0.34 | -38 % | | |
| F (+ funding) | +0.337 % | [-0.226 %, +0.867 %] | 0.38 | -43 % | +0.045 % | [-0.134 %, +0.213 %] |
| O (+ OI) | +0.405 % | [-0.155 %, +0.936 %] | 0.46 | -45 % | +0.114 % | [-0.056 %, +0.281 %] |
| FO (+ ambos) | +0.446 % | [-0.126 %, +0.992 %] | 0.50 | -44 % | +0.155 % | [-0.039 %, +0.339 %] |

Lectura: el OI (y más aún funding + OI) empuja en la dirección correcta y de forma bastante consistente (FO mejora 2023 de -0.39 % a -0.07 % y 2026 de +0.11 % a +0.60 %), pero **ninguna mejora es estadísticamente significativa** y ninguna variante tiene IC95 % por encima de cero. Además se probaron 3 variantes, lo que debilita aún más la evidencia (Sharpe deflactado con N=700: 0.43). Conclusión: candidato a seguir midiendo, no aprobado.

## 17. TP / SL según la volatilidad (ATR de 14 días) en el bot cross-sectional

`backtesting/xs_stops.py` (variante E, 1725 cohortes de 7 d desde 2022, mismas predicciones walk-forward; recorrido día a día con máximo/mínimo, SL primero si un día toca ambos, salida al open si hay hueco, deslizamiento extra de 0.10 % en el stop). Se probaron 24 combinaciones de SL ∈ {1.5, 2, 3, 4}×ATR y TP ∈ {2, 3, 4, 6}×ATR.

| | media/cohorte | peor cohorte | máx. caída | Sharpe |
|---|---|---|---|---|
| sin TP/SL | +0.26 % | -52.4 % | -41.7 % | 0.29 |
| SL 2×ATR, sin TP | +0.37 % | -15.3 % | -21.2 % | 0.50 |
| SL 3×ATR, sin TP | +0.31 % | -16.0 % | -24.0 % | 0.42 |

- **El TP empeora siempre** (corta a las ganadoras, que son las que pagan a las perdedoras): no se usa.
- **El SL reduce mucho el riesgo de cola** (squeezes de los cortos): la peor cohorte pasa de -52 % a -15 % y la caída máxima se reduce a la mitad. La mejora de la media (+0.11 % pareado, IC95 % [-0.23 %, +0.50 %]) no es significativa y fue negativa en la segunda mitad del periodo: se adopta como protección, no como fuente de ganancia.
- El bot usa SL = 2×ATR(14) por defecto (configurable), revisado con velas de 1 h.
