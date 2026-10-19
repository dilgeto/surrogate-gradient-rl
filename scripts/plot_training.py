"""Curvas de entrenamiento a partir de `runs/<nombre>/log.csv` (lo escribe sgrl.train).

Una figura con 6 paneles: retorno (entrenamiento y evaluación), tasa de disparo, desviación estándar de
la política, KL, norma del gradiente y varianza explicada del crítico. Con varias corridas se
superponen (un color por corrida, en orden fijo) y la tasa de disparo se resume en su media sobre capas.
Imprime además un resumen de texto (pasos, último retorno, mejor evaluación, velocidad, tiempo restante).

  python scripts/plot_training.py runs/Pendulum-lif-s0                       # -> runs/Pendulum-lif-s0/training.png
  python scripts/plot_training.py runs/pend-lif-nominal runs/pend-ann-nominal --out cmp.png
  python scripts/plot_training.py runs/Pendulum-lif-s0 --watch 30            # reescribe la figura cada 30 s
  python scripts/plot_training.py runs/Pendulum-lif-s0 --smooth 10 --summary-only

--smooth N promedia móvilmente `ep_return` sobre N iteraciones (la evaluación no se suaviza).
Acepta directorios de corrida o rutas a un log.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Orden fijo de la paleta categórica de referencia (azul, naranja, aqua, amarillo, magenta, violeta).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
LAYER_SHADES = ["#2a78d6", "#6aa3e8", "#a9c9f2"]  # capas de una misma corrida: un solo tono, claro->oscuro
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"


class Run:
    def __init__(self, path: Path):
        self.dir = path if path.is_dir() else path.parent
        self.name = self.dir.name
        self.rows = self._read(path if path.is_file() else path / "log.csv")
        cfg = self.dir / "config.json"
        self.cfg = json.loads(cfg.read_text()) if cfg.exists() else {}
        self.total_steps = self.cfg.get("args", {}).get("steps")
        self.target_kl = self.cfg.get("ppo", {}).get("target_kl")

    @staticmethod
    def _read(path: Path):
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))

    def col(self, key: str, rows=None) -> np.ndarray:
        rows = self.rows if rows is None else rows
        return np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows])

    def evals(self):
        rows = [r for r in self.rows if r.get("eval_return") not in (None, "")]
        return self.col("steps", rows), self.col("eval_return", rows), self.col("eval_std", rows)

    def rates(self) -> np.ndarray:  # (iteraciones, capas); vacío si la corrida no es SNN
        parsed = [[float(x) for x in r["rates"].split()] for r in self.rows if r.get("rates")]
        n = min((len(p) for p in parsed), default=0)
        return np.array([p[:n] for p in parsed]).reshape(len(parsed), n)


def smooth(y: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return y
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - n + 1): i + 1]
        w = w[~np.isnan(w)]
        if w.size:
            out[i] = w.mean()
    return out


def style(ax, title, ylabel, xlabel=False):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    if xlabel:
        ax.set_xlabel("pasos de entorno", fontsize=8, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"))


def draw(runs, args):
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    (a_ret, a_rate, a_std), (a_kl, a_gn, a_ev) = axs
    multi = len(runs) > 1
    for i, run in enumerate(runs):
        c = SERIES[i % len(SERIES)]
        x = run.col("steps")
        if not x.size:
            continue
        # retorno: entrenamiento (fino, claro) y evaluación nominal (grueso) con banda ±1 desv. entre episodios
        a_ret.plot(x, smooth(run.col("ep_return"), args.smooth), color=c, alpha=0.45, linewidth=1.2,
                   label=None if multi else "entrenamiento")
        xe, ye, se = run.evals()
        if xe.size:
            if not multi:  # con varias corridas las bandas se tapan entre sí
                a_ret.fill_between(xe, ye - se, ye + se, color=c, alpha=0.12, linewidth=0)
            a_ret.plot(xe, ye, color=c, linewidth=2, marker="o", markersize=3.5, markeredgecolor="white",
                       markeredgewidth=0.8, label=run.name if multi else "evaluación (±1 desv.)")
            k = int(np.nanargmax(ye))
            a_ret.plot(xe[k], ye[k], marker="*", markersize=11, color=c, markeredgecolor="white", markeredgewidth=0.8)
        rt = run.rates()
        if rt.size:
            if multi:
                a_rate.plot(x[: len(rt)], rt.mean(1), color=c, linewidth=1.6)
            else:
                for j in range(rt.shape[1]):
                    a_rate.plot(x[: len(rt)], rt[:, j], color=LAYER_SHADES[j % 3] if rt.shape[1] <= 3 else None,
                                linewidth=1.6, label=f"capa {j}")
        a_std.plot(x, run.col("std"), color=c, linewidth=1.6)
        a_kl.plot(x, run.col("kl"), color=c, linewidth=1.2, alpha=0.9)
        a_gn.plot(x, run.col("grad_norm"), color=c, linewidth=1.2, alpha=0.9)
        a_ev.plot(x, run.col("explained_var"), color=c, linewidth=1.6)
        if run.target_kl:
            a_kl.axhline(run.target_kl, color=MUTED, linestyle=":", linewidth=1)
            a_kl.annotate("target_kl", (x[-1], run.target_kl), fontsize=7, color=MUTED, ha="right", va="bottom")

    style(a_ret, "Retorno", "retorno")
    style(a_rate, "Tasa de disparo" + (" (media de capas)" if multi else " por capa"), "spikes / paso")
    style(a_std, "Desviación estándar de la política", "std")
    style(a_kl, "KL aproximado por iteración", "KL")
    style(a_gn, "Norma del gradiente", "‖g‖")
    style(a_ev, "Varianza explicada del crítico", "R²", xlabel=True)
    a_gn.set_yscale("log")
    a_kl.set_yscale("log")
    a_ev.set_ylim(-1.0, 1.02)  # los valores atípicos < -1 se recortan
    a_ev.axhline(0, color=MUTED, linewidth=0.8)
    for ax in (a_rate, a_std, a_kl, a_gn):
        ax.set_xlabel("pasos de entorno", fontsize=8, color=MUTED)
    a_ret.legend(fontsize=8, frameon=False, loc="best")  # multi: identidad de cada corrida
    if multi:
        fig.text(0.01, 0.005, "Retorno: línea gruesa = evaluación nominal, fina = entrenamiento, ★ = mejor evaluación (best.pt)",
                 fontsize=8, color=MUTED)
    elif a_rate.get_legend_handles_labels()[0]:
        a_rate.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(args.title or ", ".join(r.name for r in runs), x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def summary(run: Run, stale_after=None) -> str:
    if not run.rows:
        return f"{run.name}: log.csv vacío"
    last = run.rows[-1]
    steps = int(float(last["steps"]))
    xe, ye, se = run.evals()
    ep = run.col("ep_return")
    ep = ep[~np.isnan(ep)]
    lines = [f"{run.name}: it {last['iteration']}, {steps:,} pasos"
             + (f" de {run.total_steps:,} ({100 * steps / run.total_steps:.0f} %)" if run.total_steps else "")]
    if ep.size:
        lines.append(f"  retorno entrenamiento: último {ep[-1]:.1f}, media últimas 10 it {ep[-10:].mean():.1f}")
    if xe.size:
        k = int(np.nanargmax(ye))
        lines.append(f"  evaluación: última {ye[-1]:.1f} ± {se[-1]:.1f} (paso {int(xe[-1]):,}); "
                     f"mejor {ye[k]:.1f} (paso {int(xe[k]):,}) -> best.pt")
    sps = run.col("sps")[-10:].mean()
    if run.total_steps and sps > 0:
        lines.append(f"  {sps:.0f} pasos/s; restante ≈ {(run.total_steps - steps) / sps / 60:.1f} min")
    kl, std = float(last["kl"]), float(last["std"])
    lines.append(f"  KL {kl:.4f}, std {std:.3f}, R² crítico {float(last['explained_var']):.2f}, "
                 f"grad {float(last['grad_norm']):.2f}")
    rt = run.rates()
    if rt.size:
        r = rt[-1]
        flag = "  ⚠ capa casi muda o saturada" if (r < 0.01).any() or (r > 0.9).any() else ""
        lines.append(f"  tasa de disparo última it: [{' '.join(f'{v:.3f}' for v in r)}]{flag}")
    return "\n".join(lines)


def once(args):
    runs = [Run(Path(p)) for p in args.runs]
    print("\n\n".join(summary(r) for r in runs))
    if args.summary_only:
        return
    out = Path(args.out) if args.out else runs[0].dir / "training.png"
    fig = draw(runs, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="directorios de corrida (runs/<nombre>) o rutas a log.csv")
    p.add_argument("--out", help="PNG de salida (por defecto <primera corrida>/training.png)")
    p.add_argument("--title")
    p.add_argument("--smooth", type=int, default=1, help="ventana (iteraciones) para suavizar ep_return")
    p.add_argument("--watch", type=float, default=0, metavar="SEG", help="repetir cada SEG segundos (Ctrl+C para salir)")
    p.add_argument("--summary-only", action="store_true", help="sólo el resumen de texto, sin figura")
    args = p.parse_args(argv)
    if not args.watch:
        return once(args)
    try:
        while True:
            once(args)
            print(f"-- {time.strftime('%H:%M:%S')}, próxima actualización en {args.watch:g} s --\n", flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
