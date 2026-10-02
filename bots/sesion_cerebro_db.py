# -*- coding: utf-8 -*-
"""
sesion_cerebro_db.py
=====================
Persistencia en SQLite de la sesion del modo CEREBRO (main_solo_cerebro_rl.py):
posiciones abiertas, contadores de la sesion (win rate, fees, pnl neto),
reparto LONG/SHORT favorita y cooldowns. Permite reanudar el bot despues de
cerrarlo (o de que se caiga) sin perder ese estado.

Modulo aislado: solo lo usa main_solo_cerebro_rl.py. No lo importa ningun
otro archivo del proyecto.

`entry_ts` de cada posicion es un datetime real (UTC): se guarda/restaura tal
cual. El unico valor que usa time.monotonic() (que se reinicia en cada
proceso nuevo y no se puede guardar tal cual) es el cooldown local
`cooldown_perd` de CEREBRO -- se guarda como epoch (time.time()) y se
reconvierte a monotonic al cargar.
"""
import os
import sqlite3
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "sesion_cerebro.sqlite3")
BLOQUEOS_LOG_PATH = os.path.join(BASE_DIR, "bot_bloqueos.log")
TRADES_CANASTA_CSV = os.path.join(BASE_DIR, "trades_canasta.csv")
ACTIVE_POSITIONS_CSV = os.path.join(BASE_DIR, "active_positions.csv")


def _conectar():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _crear_tablas(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sesion (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            modo TEXT, iniciada_en TEXT, actualizada_en TEXT,
            n_total INTEGER, notional REAL, leverage INTEGER, balance_inicial REAL,
            sl_pct REAL, min_neto REAL, max_hold_h REAL, cd_perd_h REAL,
            wallet_balance REAL, pnl_realizado REAL,
            reparto_favorita TEXT, reparto_motivo TEXT, reparto_eval_en INTEGER,
            stats_total INTEGER, stats_wins INTEGER, stats_losses INTEGER,
            stats_fees REAL, stats_pnl_neto REAL, stats_estimados INTEGER,
            stats_wins_long INTEGER, stats_losses_long INTEGER,
            stats_wins_short INTEGER, stats_losses_short INTEGER,
            recup_activo INTEGER, recup_max INTEGER,
            recup_deuda REAL, recup_consecutivas INTEGER
        );
        CREATE TABLE IF NOT EXISTS open_positions (
            symbol TEXT PRIMARY KEY,
            side TEXT, notional REAL, entry_price REAL, tp_price REAL, sl_price REAL,
            entry_ts TEXT, prob_exito REAL, entry_fee REAL, cantidad REAL, order_id TEXT,
            trailing_activo INTEGER, trailing_peak REAL
        );
        CREATE TABLE IF NOT EXISTS cooldown_perd (
            symbol TEXT PRIMARY KEY, deadline_epoch REAL
        );
        CREATE TABLE IF NOT EXISTS cooldown_perdidas (
            symbol TEXT PRIMARY KEY, lado TEXT, exit_price REAL, loss_pct REAL
        );
        CREATE TABLE IF NOT EXISTS cooldown_ganadoras (
            symbol TEXT PRIMARY KEY, lado TEXT, entry_price REAL, timestamp TEXT
        );
    """)


def hay_sesion_guardada():
    if not os.path.exists(DB_PATH):
        return False
    conn = None
    try:
        conn = _conectar()
        _crear_tablas(conn)
        return conn.execute("SELECT 1 FROM sesion WHERE id = 1").fetchone() is not None
    except Exception:
        return False
    finally:
        if conn is not None:
            conn.close()


def modo_guardado():
    """'PAPER'/'REAL' de la sesion guardada, o None si no hay nada guardado."""
    conn = None
    try:
        conn = _conectar()
        _crear_tablas(conn)
        row = conn.execute("SELECT modo FROM sesion WHERE id = 1").fetchone()
        return row["modo"] if row else None
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def guardar_sesion(modo, cfg, engine_raw, engine, reparto, cooldown_perd, recuperacion=None):
    """Guarda TODO el estado de la sesion en una sola transaccion. Nunca lanza
    excepcion hacia afuera -- si falla, el bot sigue corriendo igual (la
    persistencia es "best effort", no debe poder tirar abajo una operacion
    en curso)."""
    conn = None
    try:
        ahora_iso = datetime.now(timezone.utc).isoformat()
        conn = _conectar()
        with conn:
            _crear_tablas(conn)
            existe = conn.execute("SELECT iniciada_en FROM sesion WHERE id = 1").fetchone()
            iniciada_en = existe["iniciada_en"] if existe else ahora_iso
            st = engine._stats
            _r_activo = bool(getattr(recuperacion, "_activo", False)) if recuperacion is not None else False
            _r_max    = getattr(recuperacion, "max_recuperaciones", 0) if recuperacion is not None else 0
            _r_deuda  = getattr(recuperacion, "_deuda", 0.0) if recuperacion is not None else 0.0
            _r_cons   = getattr(recuperacion, "_consecutivas", 0) if recuperacion is not None else 0
            conn.execute("""
                INSERT INTO sesion (id, modo, iniciada_en, actualizada_en,
                    n_total, notional, leverage, balance_inicial, sl_pct, min_neto,
                    max_hold_h, cd_perd_h, wallet_balance, pnl_realizado,
                    reparto_favorita, reparto_motivo, reparto_eval_en,
                    stats_total, stats_wins, stats_losses, stats_fees, stats_pnl_neto, stats_estimados,
                    stats_wins_long, stats_losses_long, stats_wins_short, stats_losses_short,
                    recup_activo, recup_max, recup_deuda, recup_consecutivas)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    modo=excluded.modo, actualizada_en=excluded.actualizada_en,
                    n_total=excluded.n_total, notional=excluded.notional, leverage=excluded.leverage,
                    balance_inicial=excluded.balance_inicial, sl_pct=excluded.sl_pct,
                    min_neto=excluded.min_neto, max_hold_h=excluded.max_hold_h, cd_perd_h=excluded.cd_perd_h,
                    wallet_balance=excluded.wallet_balance, pnl_realizado=excluded.pnl_realizado,
                    reparto_favorita=excluded.reparto_favorita, reparto_motivo=excluded.reparto_motivo,
                    reparto_eval_en=excluded.reparto_eval_en,
                    stats_total=excluded.stats_total, stats_wins=excluded.stats_wins,
                    stats_losses=excluded.stats_losses, stats_fees=excluded.stats_fees,
                    stats_pnl_neto=excluded.stats_pnl_neto, stats_estimados=excluded.stats_estimados,
                    stats_wins_long=excluded.stats_wins_long, stats_losses_long=excluded.stats_losses_long,
                    stats_wins_short=excluded.stats_wins_short, stats_losses_short=excluded.stats_losses_short,
                    recup_activo=excluded.recup_activo, recup_max=excluded.recup_max,
                    recup_deuda=excluded.recup_deuda, recup_consecutivas=excluded.recup_consecutivas
            """, (modo, iniciada_en, ahora_iso,
                  cfg["n_total"], cfg["notional"], cfg["leverage"], cfg["balance"],
                  cfg["sl_pct"], cfg["min_neto"], cfg["max_hold_h"], cfg["cd_perd_h"],
                  engine_raw.wallet_balance, engine_raw.pnl_realizado,
                  reparto.get("favorita"), reparto.get("motivo"), reparto.get("eval_en", 0),
                  st["total"], st["wins"], st["losses"], st["fees"], st["pnl_neto"], st["estimados"],
                  st.get("wins_long", 0), st.get("losses_long", 0),
                  st.get("wins_short", 0), st.get("losses_short", 0),
                  int(_r_activo), _r_max, _r_deuda, _r_cons))

            conn.execute("DELETE FROM open_positions")
            for sym, pos in engine_raw.open_positions.items():
                entry_ts = pos.get("entry_ts")
                entry_ts_iso = entry_ts.isoformat() if hasattr(entry_ts, "isoformat") else entry_ts
                conn.execute("""INSERT INTO open_positions
                    (symbol, side, notional, entry_price, tp_price, sl_price, entry_ts,
                     prob_exito, entry_fee, cantidad, order_id, trailing_activo, trailing_peak)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (sym, pos.get("side"), pos.get("notional"), pos.get("entry_price"),
                     pos.get("tp_price"), pos.get("sl_price"), entry_ts_iso,
                     pos.get("prob_exito"), pos.get("entry_fee"), pos.get("cantidad"),
                     pos.get("order_id"), int(bool(pos.get("trailing_activo"))), pos.get("trailing_peak")))

            conn.execute("DELETE FROM cooldown_perd")
            ahora_epoch, ahora_mono = time.time(), time.monotonic()
            for sym, deadline_mono in cooldown_perd.items():
                deadline_epoch = ahora_epoch + (deadline_mono - ahora_mono)
                conn.execute("INSERT INTO cooldown_perd (symbol, deadline_epoch) VALUES (?,?)",
                             (sym, deadline_epoch))

            conn.execute("DELETE FROM cooldown_perdidas")
            for sym, info in engine._cooldown_perdidas.items():
                conn.execute("INSERT INTO cooldown_perdidas (symbol, lado, exit_price, loss_pct) VALUES (?,?,?,?)",
                             (sym, info.get("lado"), info.get("exit_price"), info.get("loss_pct")))

            conn.execute("DELETE FROM cooldown_ganadoras")
            for sym, info in engine._cooldown_ganadoras.items():
                ts = info.get("timestamp")
                ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else ts
                conn.execute("INSERT INTO cooldown_ganadoras (symbol, lado, entry_price, timestamp) VALUES (?,?,?,?)",
                             (sym, info.get("lado"), info.get("entry_price"), ts_iso))
    except Exception:
        pass
    finally:
        if conn is not None:
            conn.close()


def cargar_sesion():
    """Devuelve un dict con todo el estado guardado, o None si no hay sesion."""
    if not hay_sesion_guardada():
        return None
    conn = None
    try:
        conn = _conectar()
        _crear_tablas(conn)
        s = conn.execute("SELECT * FROM sesion WHERE id = 1").fetchone()
        if s is None:
            return None

        posiciones = {}
        for r in conn.execute("SELECT * FROM open_positions"):
            entry_ts = datetime.fromisoformat(r["entry_ts"]) if r["entry_ts"] else None
            posiciones[r["symbol"]] = dict(
                symbol=r["symbol"], side=r["side"], modo="canasta",
                notional=r["notional"], entry_price=r["entry_price"],
                tp_price=r["tp_price"], sl_price=r["sl_price"], entry_ts=entry_ts,
                prob_exito=r["prob_exito"], entry_fee=r["entry_fee"],
                cantidad=r["cantidad"], order_id=r["order_id"],
                trailing_activo=bool(r["trailing_activo"]), trailing_peak=r["trailing_peak"],
            )

        cooldown_perd = {}
        ahora_epoch, ahora_mono = time.time(), time.monotonic()
        for r in conn.execute("SELECT * FROM cooldown_perd"):
            cooldown_perd[r["symbol"]] = ahora_mono + (r["deadline_epoch"] - ahora_epoch)

        cooldown_perdidas = {}
        for r in conn.execute("SELECT * FROM cooldown_perdidas"):
            cooldown_perdidas[r["symbol"]] = dict(lado=r["lado"], exit_price=r["exit_price"],
                                                   loss_pct=r["loss_pct"])

        cooldown_ganadoras = {}
        for r in conn.execute("SELECT * FROM cooldown_ganadoras"):
            ts = datetime.fromisoformat(r["timestamp"]) if r["timestamp"] else None
            cooldown_ganadoras[r["symbol"]] = dict(lado=r["lado"], entry_price=r["entry_price"],
                                                    timestamp=ts)

        return {
            "modo": s["modo"],
            "cfg": dict(n_total=s["n_total"], notional=s["notional"], leverage=s["leverage"],
                        balance=s["balance_inicial"], sl_pct=s["sl_pct"], min_neto=s["min_neto"],
                        max_hold_h=s["max_hold_h"], cd_perd_h=s["cd_perd_h"]),
            "wallet_balance": s["wallet_balance"], "pnl_realizado": s["pnl_realizado"],
            "reparto": dict(favorita=s["reparto_favorita"], motivo=s["reparto_motivo"] or "",
                             eval_en=s["reparto_eval_en"] or 0),
            "stats": dict(total=s["stats_total"] or 0, wins=s["stats_wins"] or 0,
                          losses=s["stats_losses"] or 0, fees=s["stats_fees"] or 0.0,
                          pnl_neto=s["stats_pnl_neto"] or 0.0, estimados=s["stats_estimados"] or 0,
                          wins_long=s["stats_wins_long"] or 0, losses_long=s["stats_losses_long"] or 0,
                          wins_short=s["stats_wins_short"] or 0, losses_short=s["stats_losses_short"] or 0),
            "recuperacion": dict(activo=bool(s["recup_activo"]), max_recuperaciones=s["recup_max"] or 0,
                                  deuda=s["recup_deuda"] or 0.0, consecutivas=s["recup_consecutivas"] or 0),
            "open_positions": posiciones,
            "cooldown_perd": cooldown_perd,
            "cooldown_perdidas": cooldown_perdidas,
            "cooldown_ganadoras": cooldown_ganadoras,
            "iniciada_en": s["iniciada_en"],
        }
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def borrar_sesion_y_archivos(borrar_csvs=True):
    """Borra la sesion guardada y, si `borrar_csvs`, tambien bot_bloqueos.log,
    trades_canasta.csv y active_positions.csv -- SOLO si existen (nunca un
    glob/wildcard, nunca otra cosa). Devuelve la lista de paths borrados."""
    borrados = []
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
            borrados.append(DB_PATH)
    except Exception:
        pass
    if borrar_csvs:
        for p in (BLOQUEOS_LOG_PATH, TRADES_CANASTA_CSV, ACTIVE_POSITIONS_CSV):
            try:
                if os.path.exists(p):
                    os.remove(p)
                    borrados.append(p)
            except Exception:
                pass
    return borrados
