"""Compara el trading_engine.py ORIGINAL contra el PARCHADO en modo paper con un exchange simulado.

Uso:  python test_parche_paper.py ruta/al/trading_engine_ORIGINAL.py
(no escribe en tu carpeta: trabaja en un directorio temporal)
"""
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aplicar_parche_paper import aplicar  # noqa: E402


class FakeExchange:
    """Precio en vivo configurable por símbolo: {symbol: (last, bid, ask)}."""
    def __init__(self, quotes):
        self.quotes = dict(quotes)

    def fetch_ticker(self, symbol):
        if symbol not in self.quotes:
            raise RuntimeError("sin precio")
        last, bid, ask = self.quotes[symbol]
        return {"last": last, "close": last, "bid": bid, "ask": ask}


def cargar(src, name):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(HERE, "trading_engine.py")
    exec(compile(src, f"{name}.py", "exec"), mod.__dict__)
    return mod


def escenario(mod, motor):
    """Devuelve lista de (caso, entrada, cerrada_al_instante, motivo, precio_salida)."""
    out = []
    casos = [
        ("SHORT, precio ya cayó 1.5% desde el cierre del escáner", "SHORT", (98.5, 98.49, 98.51)),
        ("LONG, precio ya subió 1.5% desde el cierre del escáner", "LONG", (101.5, 101.49, 101.51)),
        ("LONG, precio sin cambios (control)", "LONG", (100.0, 99.99, 100.01)),
    ]
    for nombre, side, quote in casos:
        ex = FakeExchange({"XXX/USDT:USDT": quote})
        if motor == "canasta":
            eng = mod.CanastaEngine(ex, 1, 1, 30.0, 5, tp_pct=0.01, sl_pct=0.015, paper=True)
            eng._abrir("XXX/USDT:USDT", side, {"last_close": 100.0})
            if "XXX/USDT:USDT" in eng.open_positions:
                eng.check_tp_sl()
        else:
            eng = mod.PiramideEngine(ex, 30.0, 5, tp_pct=0.01, sl_pct=0.015, paper=True)
            eng._abrir({"symbol": "XXX/USDT:USDT", "lado": side, "last_close": 100.0})
            if "XXX/USDT:USDT" in eng.open_positions:
                eng._check_positions()
        pos_abierta = eng.open_positions.get("XXX/USDT:USDT")
        cerrados = eng.closed_trades
        if pos_abierta is not None:
            out.append((nombre, pos_abierta["entry_price"], False, "-", None))
        elif cerrados:
            t = cerrados[-1]
            out.append((nombre, t["entry_price"], True, t["motivo"], t["exit_price"]))
        else:
            out.append((nombre, None, False, "no abrió", None))
    return out


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "trading_engine.py")
    original = open(ruta, encoding="utf-8").read()
    parchado, msg = aplicar(original)
    if parchado is None:
        parchado = original
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            mods = {"ORIGINAL": cargar(original, "te_orig"), "PARCHADO": cargar(parchado, "te_patch")}
            for motor in ("canasta", "piramide"):
                print(f"\n===== motor {motor} =====")
                for etiqueta, mod in mods.items():
                    print(f"[{etiqueta}]")
                    for nombre, entry, inst, motivo, exit_px in escenario(mod, motor):
                        e = f"{entry:.4f}" if entry else "n/a"
                        x = f" salida={exit_px:.4f}" if exit_px else ""
                        print(f"  {nombre:58s} entrada={e:>9s} cierre_instantáneo={'SI' if inst else 'no':2s} {motivo}{x}")
            # salida: el parchado aplica slippage adverso al precio de salida; el original sale exactamente al precio visto
            for etiqueta, mod in mods.items():
                for motor in ("canasta", "piramide"):
                    ex = FakeExchange({"XXX/USDT:USDT": (100.0, 99.99, 100.01)})
                    if motor == "canasta":
                        eng = mod.CanastaEngine(ex, 1, 1, 30.0, 5, tp_pct=0.01, sl_pct=0.015, paper=True)
                        eng._abrir("XXX/USDT:USDT", "LONG", {"last_close": 100.0})
                        ex.quotes["XXX/USDT:USDT"] = (102.0, 101.99, 102.01)
                        eng.check_tp_sl()
                    else:
                        eng = mod.PiramideEngine(ex, 30.0, 5, tp_pct=0.01, sl_pct=0.015, paper=True)
                        eng._abrir({"symbol": "XXX/USDT:USDT", "lado": "LONG", "last_close": 100.0})
                        ex.quotes["XXX/USDT:USDT"] = (102.0, 101.99, 102.01)
                        eng._check_positions()
                    t = eng.closed_trades[-1] if eng.closed_trades else None
                    print(f"[{etiqueta}] {motor}: LONG toca TP con el precio en 102.00 -> "
                          f"{'salida=%.4f (%s)' % (t['exit_price'], t['motivo']) if t else 'sin cierre'}")
            # sin precio en vivo: el parchado no debe abrir con precio viejo
            ex = FakeExchange({})
            for etiqueta, mod in mods.items():
                eng = mod.CanastaEngine(ex, 1, 1, 30.0, 5, tp_pct=0.01, sl_pct=0.015, paper=True)
                eng._abrir("XXX/USDT:USDT", "LONG", {"last_close": 100.0})
                print(f"\n[{etiqueta}] sin precio en vivo -> posiciones abiertas: {len(eng.open_positions)}")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    main()
