"""Parche del modo PAPER de trading_engine.py (no toca el modo real).

Problema: en paper la entrada se registraba al `last_close` del escáner (precio viejo) mientras el TP/SL se evaluaba contra
el precio en vivo; si el precio ya se había movido, la posición cerraba al instante con ganancia/pérdida falsa.

Cambios (solo con self.paper=True):
  1. Entrada al precio EN VIVO del libro: ask (+slippage) para LONG, bid (-slippage) para SHORT. Si no hay precio, no abre.
  2. Salida con slippage adverso (0.02%).
Las comisiones del bot (0.05% por lado) ya se aplicaban y no cambian.

Uso:  python aplicar_parche_paper.py [ruta/a/trading_engine.py]     (crea trading_engine.py.bak antes de modificar)
Es idempotente: si el archivo ya tiene el parche, no hace nada. Si algún punto de anclaje no coincide, no modifica nada.
"""
import shutil
import sys

MARK = "[parche paper]"

HELPERS = '''
PAPER_SLIPPAGE = 0.0002   # [parche paper] deslizamiento adverso por pata en paper (0.02%)


def precio_vivo_paper(exchange, symbol, side):
    """[parche paper] Precio de entrada simulado EN VIVO: LONG compra al ask, SHORT vende al bid, mas slippage adverso.
    Devuelve None si no hay precio en vivo (en ese caso NO se debe abrir con un precio viejo)."""
    try:
        t = exchange.fetch_ticker(symbol)
        last = t.get("last") or t.get("close")
        px = (t.get("ask") if side == "LONG" else t.get("bid")) or last
        if not px:
            return None
        return float(px) * (1 + PAPER_SLIPPAGE) if side == "LONG" else float(px) * (1 - PAPER_SLIPPAGE)
    except Exception:
        return None


def precio_salida_paper(price, side):
    """[parche paper] Salida simulada con slippage adverso: LONG vende un poco mas bajo, SHORT recompra un poco mas alto."""
    return price * (1 - PAPER_SLIPPAGE) if side == "LONG" else price * (1 + PAPER_SLIPPAGE)
'''

ENTRY_OLD = '''        cantidad = order_id = None
        if not self.paper:
            r = self.real_executor.abrir('''
ENTRY_NEW = '''        cantidad = order_id = None
        if self.paper:  # [parche paper]
            px_vivo = precio_vivo_paper(self.exchange, symbol, side)
            if px_vivo is None:
                print(f"    [paper] sin precio en vivo para {symbol}: no se abre (evita entrada con precio viejo)")
                return
            entry_price = px_vivo
        if not self.paper:
            r = self.real_executor.abrir('''

EXIT_OLD = '''                exit_price = real_exit
'''
EXIT_NEW = '''                exit_price = real_exit
            else:  # [parche paper]
                exit_price = precio_salida_paper(exit_price, pos["side"])
'''


def aplicar(src):
    if MARK in src:
        return None, "ya tiene el parche"
    anchors = src.count("DEFAULT_FEE = ")
    if src.count(ENTRY_OLD) != 2 or src.count(EXIT_OLD) != 2 or anchors != 1:
        raise SystemExit(
            f"No se aplicó: puntos de anclaje inesperados (entrada={src.count(ENTRY_OLD)} esperado 2, "
            f"salida={src.count(EXIT_OLD)} esperado 2, DEFAULT_FEE={anchors} esperado 1). "
            "Tu trading_engine.py difiere de la versión analizada; no se modificó nada.")
    out = src.replace(ENTRY_OLD, ENTRY_NEW).replace(EXIT_OLD, EXIT_NEW)
    lines = out.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("DEFAULT_FEE = "))
    lines[i + 1:i + 1] = HELPERS.split("\n")[:-1]
    return "\n".join(lines), "parche aplicado"


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "trading_engine.py"
    src = open(path, encoding="utf-8").read()
    new, msg = aplicar(src)
    if new is None:
        print(f"{path}: {msg}")
    else:
        shutil.copyfile(path, path + ".bak")
        open(path, "w", encoding="utf-8").write(new)
        compile(new, path, "exec")
        print(f"{path}: {msg} (copia de seguridad en {path}.bak)")
