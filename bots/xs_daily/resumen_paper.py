"""Resume el paper trading en vivo: lee paper_state.json y paper_log.csv y compara con lo esperado del backtest.
Uso: python bots/xs_daily/resumen_paper.py
Honestidad estadística: con pocas cohortes (< ~30) el resultado NO permite concluir nada; el IC95% lo deja explícito.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
st = json.load(open(os.path.join(HERE, "paper_state.json")))
cl = pd.DataFrame(st["closed"])
print(f"equity realizada {st['equity']:.2f} | cohortes abiertas {len(st['cohorts'])} | cerradas {len(cl)}")
if len(cl):
    r = cl["ret"].to_numpy()
    n_eff = max(len(r) / 7, 1)                                   # las cohortes diarias se solapan 7 días: n efectivo = n/7
    se = r.std(ddof=1) / np.sqrt(n_eff) if len(r) > 1 else float("nan")
    print(f"retorno medio por cohorte (7 d): {r.mean():+.3%} | IC95% aprox. [{r.mean() - 1.96 * se:+.3%}, {r.mean() + 1.96 * se:+.3%}] | positivas {np.mean(r > 0):.0%}")
    print("Referencia del backtest 6 años (sobre el capital de la cohorte): +0.29% por cohorte, IC95% [-0.23%, +0.81%] -> no distinguible de cero.")
    if n_eff < 30:
        print(f"Solo {len(r)} cohortes cerradas (~{n_eff:.0f} independientes): demasiado pronto para concluir (hacen falta 30+ ~ 7 meses de ejecución diaria).")
    print(cl.tail(10).to_string(index=False))
log = os.path.join(HERE, "paper_log.csv")
if os.path.exists(log):
    print("\nÚltimas filas del log diario:\n", pd.read_csv(log).tail(7).to_string(index=False))
