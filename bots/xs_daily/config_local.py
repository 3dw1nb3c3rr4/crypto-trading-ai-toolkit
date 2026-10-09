"""Configuración y claves API guardadas SOLO en tu PC (carpeta ~/.cerebro, fuera del repositorio).

Claves: si está instalado `keyring` (pip install keyring) se guardan en el Administrador de credenciales de Windows,
cifradas por el sistema. Si no, en ~/.cerebro/keys.json (texto plano protegido solo por los permisos de tu usuario).
El dashboard nunca devuelve el secreto al navegador: solo indica si hay claves guardadas y una pista de la clave pública.
Recomendación: crea claves SIN permiso de retiro y, si el exchange lo permite, restringidas a la IP de tu PC.
"""
from __future__ import annotations

import json
import os

DIR = os.environ.get("CEREBRO_HOME", os.path.join(os.path.expanduser("~"), ".cerebro"))
CONFIG = os.path.join(DIR, "config.json")
KEYS = os.path.join(DIR, "keys.json")
SERVICE = "cerebro-trading"

DEFAULTS = dict(
    exchange="binanceusdm",          # binanceusdm | okx | bybit
    mode="paper",                    # paper (simulado local) | demo (cuenta demo/testnet del exchange) | real
    universe="both",                 # crypto | stocks | both
    bot_capital=1000.0, manual_capital=1000.0,
    taker=0.0005, maker=0.0002, slippage=0.0002,   # comisión taker/maker y deslizamiento por lado
    mmr=0.005,                       # margen de mantenimiento aproximado para el precio de liquidación
    default_leverage=3, max_leverage=10,           # tope de apalancamiento (seguridad)
    max_position_usdt=100.0,         # tope de valor por orden en demo/real (seguridad)
    margin_mode="isolated",
    bot_variant="D",                 # D (original) | FO (con funding y open interest)
    bot_sl_atr=2.0,                  # stop loss del bot en múltiplos de ATR(14) (0 = sin stop); ver backtesting/xs_stops.py
)
EXCHANGES = ("binanceusdm", "okx", "bybit")


def _ensure():
    os.makedirs(DIR, exist_ok=True)


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG):
        try:
            cfg.update(json.load(open(CONFIG, encoding="utf-8")))
        except Exception:
            pass
    return cfg


def validate(cfg: dict) -> dict:
    """Devuelve la configuración saneada (lanza ValueError con un mensaje claro si algo no es válido)."""
    out = dict(DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k in DEFAULTS})
    if out["exchange"] not in EXCHANGES:
        raise ValueError(f"exchange no soportado: {out['exchange']}")
    if out["mode"] not in ("paper", "demo", "real"):
        raise ValueError("modo inválido")
    if out["universe"] not in ("crypto", "stocks", "both"):
        raise ValueError("universo inválido")
    for k in ("bot_capital", "manual_capital", "taker", "maker", "slippage", "mmr", "max_position_usdt"):
        out[k] = float(out[k])
        if out[k] < 0:
            raise ValueError(f"{k} no puede ser negativo")
    for k in ("taker", "maker", "slippage", "mmr"):
        if out[k] > 0.05:
            raise ValueError(f"{k} parece demasiado alto (se escribe en fracción: 0.0005 = 0.05 %)")
    out["default_leverage"] = int(out["default_leverage"])
    out["max_leverage"] = int(out["max_leverage"])
    if not 1 <= out["max_leverage"] <= 125 or not 1 <= out["default_leverage"] <= out["max_leverage"]:
        raise ValueError("apalancamiento fuera de rango (1 <= por defecto <= máximo <= 125)")
    out["bot_sl_atr"] = float(out["bot_sl_atr"])
    if not 0 <= out["bot_sl_atr"] <= 10:
        raise ValueError("stop del bot entre 0 (sin stop) y 10 ATR")
    if out["bot_variant"] not in ("D", "FO"):
        raise ValueError("variante del bot inválida")
    if out["margin_mode"] not in ("isolated", "cross"):
        raise ValueError("modo de margen inválido")
    return out


def save_config(cfg: dict) -> dict:
    cfg = validate(cfg)
    _ensure()
    json.dump(cfg, open(CONFIG, "w", encoding="utf-8"), indent=1)
    return cfg


# ------------------------------------------------------------------ claves
def _keyring():
    try:
        import keyring
        keyring.get_keyring()
        return keyring
    except Exception:
        return None


def _file_keys() -> dict:
    if os.path.exists(KEYS):
        try:
            return json.load(open(KEYS, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _slot(exchange, mode):
    return f"{exchange}:{'demo' if mode == 'demo' else 'real'}"   # claves separadas para demo y real


def save_keys(exchange, mode, key, secret, password=""):
    if not key or not secret:
        raise ValueError("faltan la clave API o el secreto")
    data = json.dumps(dict(apiKey=key.strip(), secret=secret.strip(), password=(password or "").strip()))
    kr = _keyring()
    if kr:
        kr.set_password(SERVICE, _slot(exchange, mode), data)
        return "keyring"
    _ensure()
    d = _file_keys()
    d[_slot(exchange, mode)] = json.loads(data)
    with open(KEYS, "w", encoding="utf-8") as f:
        json.dump(d, f)
    try:
        os.chmod(KEYS, 0o600)
    except Exception:
        pass
    return "archivo"


def load_keys(exchange, mode):
    kr = _keyring()
    if kr:
        raw = kr.get_password(SERVICE, _slot(exchange, mode))
        if raw:
            return json.loads(raw)
    return _file_keys().get(_slot(exchange, mode))


def delete_keys(exchange, mode):
    kr = _keyring()
    if kr:
        try:
            kr.delete_password(SERVICE, _slot(exchange, mode))
        except Exception:
            pass
    d = _file_keys()
    if d.pop(_slot(exchange, mode), None) is not None:
        json.dump(d, open(KEYS, "w", encoding="utf-8"))


def keys_status(exchange, mode):
    k = load_keys(exchange, mode)
    hint = None
    if k and k.get("apiKey"):
        a = k["apiKey"]
        hint = a[:4] + "…" + a[-4:] if len(a) > 10 else "****"
    return dict(saved=bool(k), hint=hint, backend="keyring (cifrado por Windows)" if _keyring() else f"archivo {KEYS}")
