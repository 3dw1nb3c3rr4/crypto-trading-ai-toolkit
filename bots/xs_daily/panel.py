"""Panel gráfico del bot cross-sectional diario (PAPER: no envía órdenes reales).

Uso:  python bots/xs_daily/panel.py            (requiere:  pip install matplotlib pandas ccxt)
Pestañas: Resumen (curva de equity + estadística honesta), Posiciones abiertas (con precios en vivo),
          Cohortes cerradas, Señales de hoy y Consola. Botones: ejecutar hoy, reentrenar, precios en vivo.
Las acciones corren el mismo run_paper.py / train.py de siempre en un proceso aparte, así que el panel no cambia la lógica del bot.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "paper_state.json")
LOG = os.path.join(HERE, "paper_log.csv")
MODEL = os.path.join(HERE, "model_xs_D.pkl")

BG, PANEL, FG, MUTED = "#0f1419", "#182028", "#e6edf3", "#8b98a5"
GREEN, RED, BLUE, AMBER = "#3fb950", "#f85149", "#58a6ff", "#d29922"
NO_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def money(x):
    return f"{x:+,.2f}"


class App(tk.Tk):
    def __init__(self, exchange="binanceusdm", capital=1000.0):
        super().__init__()
        self.title("Cerebro · Paper trading cross-sectional")
        self.geometry("1180x760")
        self.configure(bg=BG)
        self.exchange, self.capital = exchange, capital
        self.prices: dict[str, float] = {}
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self._style()
        self._build()
        self.refresh()
        self.after(200, self._pump)

    # ------------------------------------------------------------------ estilo / construcción
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(16, 8))
        s.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", FG)])
        s.configure("Treeview", background=PANEL, foreground=FG, fieldbackground=PANEL, rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background=BG, foreground=MUTED, relief="flat")
        s.map("Treeview", background=[("selected", "#243447")])
        s.configure("Accent.TButton", background=BLUE, foreground="#06101a", padding=(14, 8), borderwidth=0)
        s.map("Accent.TButton", background=[("active", "#79b8ff"), ("disabled", "#2a3a4a")])
        s.configure("TButton", background=PANEL, foreground=FG, padding=(12, 8), borderwidth=0)
        s.map("TButton", background=[("active", "#243447"), ("disabled", "#141b22")], foreground=[("disabled", MUTED)])

    def _card(self, parent, title):
        f = tk.Frame(parent, bg=PANEL, padx=14, pady=10)
        tk.Label(f, text=title, bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        v = tk.Label(f, text="–", bg=PANEL, fg=FG, font=("Segoe UI", 18, "bold"))
        v.pack(anchor="w")
        return f, v

    def _build(self):
        top = tk.Frame(self, bg=BG, padx=16, pady=12)
        top.pack(fill="x")
        tk.Label(top, text="Cerebro · cross-sectional diario", bg=BG, fg=FG, font=("Segoe UI", 16, "bold")).pack(side="left")
        tk.Label(top, text="  PAPER · sin órdenes reales  ", bg=AMBER, fg="#1b1400", font=("Segoe UI", 9, "bold")).pack(side="left", padx=12)
        self.status = tk.Label(top, text="", bg=BG, fg=MUTED, font=("Segoe UI", 9))
        self.status.pack(side="right")

        cards = tk.Frame(self, bg=BG, padx=16)
        cards.pack(fill="x")
        self.kpi = {}
        for i, (k, t) in enumerate([("total", "Equity total"), ("real", "Realizado"), ("unreal", "No realizado"),
                                    ("open", "Cohortes abiertas"), ("closed", "Cohortes cerradas"), ("mean", "Retorno medio / cohorte")]):
            f, v = self._card(cards, t)
            f.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            cards.columnconfigure(i, weight=1)
            self.kpi[k] = v

        bar = tk.Frame(self, bg=BG, padx=16, pady=10)
        bar.pack(fill="x")
        self.btn_run = ttk.Button(bar, text="▶  Ejecutar hoy (paper)", style="Accent.TButton", command=self.run_today)
        self.btn_run.pack(side="left")
        self.btn_px = ttk.Button(bar, text="⟳  Precios en vivo", command=self.fetch_prices)
        self.btn_px.pack(side="left", padx=8)
        self.btn_train = ttk.Button(bar, text="⚙  Reentrenar modelo", command=self.retrain)
        self.btn_train.pack(side="left")
        ttk.Button(bar, text="Actualizar vista", command=self.refresh).pack(side="left", padx=8)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.t_sum, self.t_open, self.t_closed, self.t_sig, self.t_con = (tk.Frame(self.nb, bg=BG) for _ in range(5))
        for t, n in ((self.t_sum, "Resumen"), (self.t_open, "Posiciones abiertas"), (self.t_closed, "Cohortes cerradas"),
                     (self.t_sig, "Señales de hoy"), (self.t_con, "Consola")):
            self.nb.add(t, text=n)

        # resumen
        self.fig = Figure(figsize=(8, 4), facecolor=BG)
        self.ax1, self.ax2 = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        self.verdict = tk.Label(self.t_sum, text="", bg=BG, fg=MUTED, justify="left", wraplength=1100, font=("Segoe UI", 10))
        self.verdict.pack(side="bottom", anchor="w", pady=8)                # primero, para que conserve su espacio
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.t_sum)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

        # tablas
        self.tr_open = self._table(self.t_open, ("cohorte", "lado", "símbolo", "entrada", "precio", "PnL %", "PnL USDT"), (90, 70, 130, 110, 110, 90, 100))
        self.tr_closed = self._table(self.t_closed, ("abierta", "cerrada", "capital", "PnL USDT", "retorno"), (110, 110, 110, 110, 110))
        self.sig = tk.Text(self.t_sig, bg=PANEL, fg=FG, relief="flat", font=("Consolas", 11), padx=14, pady=12)
        self.sig.pack(fill="both", expand=True, pady=8)
        self.con = tk.Text(self.t_con, bg="#0b0f13", fg="#b6c2cf", relief="flat", font=("Consolas", 10), padx=10, pady=8)
        self.con.pack(fill="both", expand=True, pady=8)
        for w in (self.sig, self.con):
            w.tag_configure("g", foreground=GREEN)
            w.tag_configure("r", foreground=RED)
            w.tag_configure("m", foreground=MUTED)

    def _table(self, parent, cols, widths):
        fr = tk.Frame(parent, bg=BG)
        fr.pack(fill="both", expand=True, pady=8)
        tr = ttk.Treeview(fr, columns=cols, show="headings")
        for c, w in zip(cols, widths):
            tr.heading(c, text=c)
            tr.column(c, width=w, anchor="e" if c not in ("lado", "símbolo", "cohorte") else "w")
        sb = ttk.Scrollbar(fr, orient="vertical", command=tr.yview)
        tr.configure(yscrollcommand=sb.set)
        tr.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        tr.tag_configure("pos", foreground=GREEN)
        tr.tag_configure("neg", foreground=RED)
        return tr

    # ------------------------------------------------------------------ datos
    def load(self):
        st = json.load(open(STATE)) if os.path.exists(STATE) else {"equity": self.capital, "cohorts": [], "closed": []}
        lg = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
        return st, lg

    def unrealized(self, st):
        u = 0.0
        for c in st["cohorts"]:
            for p in c["positions"]:
                px = self.prices.get(p["symbol"])
                if px:
                    u += p["notional"] * (1 if p["side"] == "LONG" else -1) * (px / p["entry"] - 1)
        return u

    def refresh(self):
        st, lg = self.load()
        cl = pd.DataFrame(st["closed"])
        u = self.unrealized(st) if self.prices else (float(lg["no_realizado"].iloc[-1]) if len(lg) else 0.0)
        tot = st["equity"] + u
        self._set(self.kpi["total"], f"{tot:,.2f}", GREEN if tot >= self.capital else RED)
        self._set(self.kpi["real"], money(st["equity"] - self.capital), GREEN if st["equity"] >= self.capital else RED)
        self._set(self.kpi["unreal"], money(u), GREEN if u >= 0 else RED)
        self._set(self.kpi["open"], str(len(st["cohorts"])))
        self._set(self.kpi["closed"], str(len(cl)))
        mean = cl["ret"].mean() if len(cl) else None
        self._set(self.kpi["mean"], f"{mean:+.2%}" if mean is not None else "–", FG if mean is None else (GREEN if mean >= 0 else RED))
        self._charts(lg, cl)
        self._verdict(cl)
        self._open_table(st)
        self._closed_table(cl)
        self._signals(st)
        self.status.config(text=f"actualizado {time.strftime('%H:%M:%S')} · modelo: {self._model_info()}")

    def _model_info(self):
        return time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(MODEL))) if os.path.exists(MODEL) else "SIN ENTRENAR"

    @staticmethod
    def _set(lbl, text, color=FG):
        lbl.config(text=text, fg=color)

    def _style_ax(self, ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=FG, fontsize=10, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#243447")
        ax.grid(color="#243447", alpha=0.5, linewidth=0.6)

    def _charts(self, lg, cl):
        for ax in (self.ax1, self.ax2):
            ax.clear()
        self._style_ax(self.ax1, "Equity total (USDT)")
        if len(lg) > 1:
            x = pd.to_datetime(lg["fecha"])
            self.ax1.plot(x, lg["equity_total"], color=BLUE, linewidth=2)
            self.ax1.axhline(self.capital, color=MUTED, linewidth=0.8, linestyle="--")
            self.ax1.fill_between(x, self.capital, lg["equity_total"], color=BLUE, alpha=0.12)
            self.fig.autofmt_xdate(rotation=0)
        else:
            self.ax1.text(0.5, 0.5, "La curva aparece con 2+ días de ejecución", transform=self.ax1.transAxes, ha="center", color=MUTED)
        self._style_ax(self.ax2, "Retorno por cohorte cerrada")
        if len(cl):
            r = cl["ret"].to_numpy() * 100
            self.ax2.bar(range(len(r)), r, color=[GREEN if v >= 0 else RED for v in r])
            self.ax2.axhline(0, color=MUTED, linewidth=0.8)
            self.ax2.set_ylabel("%", color=MUTED)
        else:
            self.ax2.text(0.5, 0.5, "Aún no cierra ninguna cohorte\n(tarda 7 días)", transform=self.ax2.transAxes, ha="center", color=MUTED)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _verdict(self, cl):
        if not len(cl):
            self.verdict.config(text="Aún no hay cohortes cerradas: todavía no se puede evaluar nada. Referencia del backtest de 6 años: "
                                     "+0.29 % por cohorte, IC95 % [−0.23 %, +0.81 %] (no distinguible de cero).", fg=MUTED)
            return
        r = cl["ret"].to_numpy()
        n_eff = max(len(r) / 7, 1)
        se = r.std(ddof=1) / np.sqrt(n_eff) if len(r) > 1 else float("nan")
        m = r.mean()
        txt = (f"Media {m:+.3%} por cohorte · IC95 % aprox. [{m - 1.96 * se:+.3%}, {m + 1.96 * se:+.3%}] · positivas {np.mean(r > 0):.0%} · "
               f"{len(r)} cohortes (~{n_eff:.0f} independientes, se solapan 7 días).\n")
        txt += ("Demasiado pronto para concluir: hacen falta ~30 cohortes independientes (≈ 7 meses)." if n_eff < 30
                else "Muestra suficiente para una primera lectura; compárala con el backtest (+0.29 %).")
        self.verdict.config(text=txt, fg=AMBER if n_eff < 30 else FG)

    def _open_table(self, st):
        self.tr_open.delete(*self.tr_open.get_children())
        for c in st["cohorts"]:
            for p in c["positions"]:
                px = self.prices.get(p["symbol"])
                sg = 1 if p["side"] == "LONG" else -1
                pct = sg * (px / p["entry"] - 1) if px else None
                tag = () if pct is None else (("pos",) if pct >= 0 else ("neg",))
                self.tr_open.insert("", "end", tags=tag, values=(
                    c["opened"], p["side"], p["symbol"].split("/")[0], f"{p['entry']:.6g}", f"{px:.6g}" if px else "–",
                    f"{pct:+.2%}" if pct is not None else "–", money(p["notional"] * pct) if pct is not None else "–"))

    def _closed_table(self, cl):
        self.tr_closed.delete(*self.tr_closed.get_children())
        for _, r in cl.iloc[::-1].iterrows():
            self.tr_closed.insert("", "end", tags=("pos",) if r["pnl"] >= 0 else ("neg",),
                                  values=(r["opened"], r["closed"], f"{r['capital']:.2f}", money(r["pnl"]), f"{r['ret']:+.2%}"))

    def _signals(self, st):
        self.sig.delete("1.0", "end")
        if not st["cohorts"]:
            self.sig.insert("end", "Aún no hay señales. Pulsa «Ejecutar hoy (paper)».\n", "m")
            return
        c = st["cohorts"][-1]
        self.sig.insert("end", f"Cohorte abierta el {c['opened']} (señal con la vela cerrada del {c['signal_date']})\n\n", "m")
        for side, tag, label in (("LONG", "g", "COMPRAR (largos)"), ("SHORT", "r", "VENDER EN CORTO (cortos)")):
            self.sig.insert("end", f"{label}\n", tag)
            for p in c["positions"]:
                if p["side"] == side:
                    self.sig.insert("end", f"   {p['symbol'].split('/')[0]:<10} entrada simulada {p['entry']:.6g}   ({p['notional']:.2f} USDT)\n")
            self.sig.insert("end", "\n")
        self.sig.insert("end", "Son operaciones SIMULADAS: el bot no envía órdenes. La ventaja medida es pequeña y no está confirmada.\n", "m")

    # ------------------------------------------------------------------ acciones (procesos aparte, para no bloquear la ventana)
    def _spawn(self, args, done=None):
        if self.busy:
            return
        self.busy = True
        for b in (self.btn_run, self.btn_px, self.btn_train):
            b.state(["disabled"])
        self.nb.select(self.t_con)
        self._log("$ " + " ".join(os.path.basename(a) if os.path.exists(a) else a for a in args) + "\n", "m")

        def work():
            try:
                p = subprocess.Popen([sys.executable, "-u", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     cwd=os.path.dirname(os.path.dirname(HERE)), creationflags=NO_FLAGS)
                for line in p.stdout:
                    self.q.put(("log", line))
                p.wait()
                self.q.put(("done", (p.returncode, done)))
            except Exception as e:                               # noqa: BLE001
                self.q.put(("log", f"ERROR: {e}\n"))
                self.q.put(("done", (1, done)))
        threading.Thread(target=work, daemon=True).start()

    def run_today(self):
        if not os.path.exists(MODEL):
            messagebox.showinfo("Falta el modelo", "Primero pulsa «Reentrenar modelo».")
            return
        self._spawn([os.path.join(HERE, "run_paper.py"), "--force", "--capital", str(self.capital), "--exchange", self.exchange])

    def retrain(self):
        self._spawn([os.path.join(HERE, "train.py")])

    def fetch_prices(self):
        st, _ = self.load()
        syms = sorted({p["symbol"] for c in st["cohorts"] for p in c["positions"]})
        if not syms:
            messagebox.showinfo("Sin posiciones", "No hay posiciones abiertas.")
            return
        self.busy = True
        for b in (self.btn_run, self.btn_px, self.btn_train):
            b.state(["disabled"])

        def work():
            try:
                import ccxt
                ex = getattr(ccxt, self.exchange)({"enableRateLimit": True})
                tk_ = ex.fetch_tickers(syms)
                self.q.put(("prices", {s: float(t["last"]) for s, t in tk_.items() if t.get("last")}))
            except Exception as e:                               # noqa: BLE001
                self.q.put(("log", f"No se pudieron obtener precios: {type(e).__name__}: {str(e)[:160]}\n"))
                self.q.put(("done", (1, None)))
        threading.Thread(target=work, daemon=True).start()

    def _log(self, text, tag=None):
        self.con.insert("end", text, tag or ())
        self.con.see("end")

    def _pump(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "log":
                    self._log(data, "r" if "Error" in data or "ERROR" in data or "Traceback" in data else None)
                elif kind == "prices":
                    self.prices = data
                    self.busy = False
                    self._enable()
                    self.refresh()
                    self.nb.select(self.t_open)
                elif kind == "done":
                    code, _ = data
                    self._log("\n✔ terminado\n" if code == 0 else f"\n✖ terminó con error (código {code})\n", "g" if code == 0 else "r")
                    self.busy = False
                    self._enable()
                    self.refresh()
        except queue.Empty:
            pass
        self.after(200, self._pump)

    def _enable(self):
        for b in (self.btn_run, self.btn_px, self.btn_train):
            b.state(["!disabled"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binanceusdm")
    ap.add_argument("--capital", type=float, default=1000.0)
    a = ap.parse_args()
    App(a.exchange, a.capital).mainloop()
