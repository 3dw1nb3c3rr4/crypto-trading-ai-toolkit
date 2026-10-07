"""Pruebas sin pesos de Kronos: forma de salida, alineación y anti-fuga (alterar el futuro NO cambia el pronóstico de t)."""
import numpy as np, pandas as pd
import kronos_features as kf


def synth(n=300, seed=0):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.02, n)))
    ts = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"ts": ts, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": r.uniform(1e3, 2e3, n)})


def main():
    uni = {"A": synth(300, 1), "B": synth(300, 2)}
    f1 = kf.make_forecasts(uni, kf.MockPredictor(), step=5, batch=8, log=lambda *_: None)
    assert set(kf.FEATS) <= set(f1.columns) and len(f1) > 0
    # anti-fuga: alterar TODO lo posterior a la fecha t0 no cambia los pronósticos con ts <= t0
    t0 = uni["A"]["ts"].iloc[200]
    u2 = {k: v.copy() for k, v in uni.items()}
    for v in u2.values():
        m = v["ts"] > t0
        v.loc[m, ["open", "high", "low", "close"]] *= 3.0
        v.loc[m, "volume"] *= 10
    f2 = kf.make_forecasts(u2, kf.MockPredictor(), step=5, batch=8, log=lambda *_: None)
    a = f1[f1.ts <= t0].sort_values(["sym", "ts"]).reset_index(drop=True)
    b = f2[f2.ts <= t0].sort_values(["sym", "ts"]).reset_index(drop=True)
    assert len(a) > 10 and np.allclose(a[list(kf.FEATS)].to_numpy(), b[list(kf.FEATS)].to_numpy()), "FUGA: el pronóstico depende del futuro"
    # alineación: la característica de la fecha t está en la fila t, sin relleno
    idx = pd.DatetimeIndex(uni["A"]["ts"])
    F = kf.build(f1, idx, ["A", "B"])
    row = f1.iloc[3]
    assert np.isclose(F["kr_ret"].loc[row.ts, row.sym], row.kr_ret)
    assert F["kr_ret"].notna().sum().sum() == len(f1)
    # sensibilidad: el mock extrapola momentum, así que kr_ret debe correlacionar con el retorno de 5 días pasado
    c = pd.DataFrame({k: v.set_index("ts").close for k, v in uni.items()})
    past = np.log(c).diff(5).reindex(index=idx)
    x = pd.concat([F["kr_ret"].stack(), past.stack()], axis=1).dropna()
    assert x.corr().iloc[0, 1] > 0.9, "el mock debería correlacionar con el momentum"
    # tamaño de lote distinto -> mismo resultado
    f3 = kf.make_forecasts(uni, kf.MockPredictor(), step=5, batch=3, log=lambda *_: None).sort_values(["sym", "ts"]).reset_index(drop=True)
    f1s = f1.sort_values(["sym", "ts"]).reset_index(drop=True)
    assert np.allclose(f3[list(kf.FEATS)].to_numpy(), f1s[list(kf.FEATS)].to_numpy())
    print("test_kronos_features: OK")


if __name__ == "__main__":
    main()
