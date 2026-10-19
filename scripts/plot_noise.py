"""Figuras de robustez a ruido de memristores (puerto de wann-cpp `plot_car_noise*.py`).

Parten de los CSV de scripts/noise_eval.py y usan `summarize_noise.summarize` para agregar (mismo
cálculo que el resumen de texto). La línea es la media; la banda oscura, el IC 95 % bootstrap de la
media (remuestreando chips); la clara, los percentiles 10-90 de las medias por chip. Con filas
`noise_model == none` en el CSV (o `--baseline`) se dibuja el retorno sin ruido como línea punteada.

  --kind intensity  retorno vs escala de ruido (--x-col: prog/read/drift), una línea por modelo
  --kind time       retorno vs tiempo desde la programación (symlog), sin el modelo `simple`
  --kind heatmap    retorno en la grilla escala×tiempo de un modelo (--noise-model pcm|reram|reramcmo)
  --kind rmse       retorno vs error RMS efectivo de los pesos: eje común entre modelos
  --kind ablation   retorno vs escala con UN mecanismo activo cada vez (CSV con columna `tag`,
                    que escribe run_noise_sweeps.py --ablation; --noise-model elige el modelo)

    python scripts/plot_noise.py --kind intensity --csv noise_sweep.csv --out fig.png --title "Pendulum LIF"
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_noise import load_rows, summarize

COLORS = {"simple": "steelblue", "pcm": "darkorange", "reram": "green", "reramcmo": "purple"}
LABELS = {"simple": "Simple (placeholder)", "pcm": "PCM (Nandakumar et al. 2019)",
          "reram": "ReRAM (Wan et al. 2022)", "reramcmo": "ReRAM CMO/HfOx (Falcone et al.)"}
X_LABELS = {
    "prog_noise_scale": "Escala de ruido de programación (--prog-noise-scale; 1 = ajuste por defecto)",
    "read_noise_scale": "Escala de ruido de lectura (--read-noise-scale)",
    "drift_scale": "Escala de deriva (--drift-scale)",
}
TIME_TICKS = {0: "0", 1: "1 s", 60: "1 min", 300: "5 min", 3600: "1 h", 21600: "6 h", 86400: "1 día",
              172800: "2 días", 604800: "1 semana", 2592000: "30 días", 31536000: "1 año"}


def baseline_of(df, override):
    if override is not None:
        return override
    base = df[df["noise_model"] == "none"]
    return float(base["return_mean"].mean()) if not base.empty else None


def band(ax, x, g, color, label, marker="o"):
    ax.plot(x, g["ret_mean"], "-" + marker, color=color, linewidth=1.5, markersize=4, label=label)
    ax.fill_between(x, g["ret_p10"], g["ret_p90"], color=color, alpha=0.10, linewidth=0)
    ax.fill_between(x, g["ret_ci_lo"], g["ret_ci_hi"], color=color, alpha=0.25, linewidth=0)


def finish(ax, fig, args, xlabel, base):
    if base is not None:
        ax.axhline(base, color="gray", linestyle=":", linewidth=1.2, label="sin ruido")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Retorno medio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(args.title)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Guardado: {out}")
    if args.show:
        plt.show()


def plot_intensity(df, summary, args, base):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, g in summary[summary["noise_model"] != "none"].groupby("noise_model"):
        g = g.sort_values(args.x_col)
        band(ax, g[args.x_col], g, COLORS.get(model), LABELS.get(model, model))
    finish(ax, fig, args, X_LABELS[args.x_col], base)


def plot_time(df, summary, args, base):
    summary = summary[~summary["noise_model"].isin(["none", "simple"])]  # simple no tiene deriva
    if summary.empty:
        sys.exit("Error: no quedaron filas tras excluir none/simple.")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    used = set()
    for model, g in summary.groupby("noise_model"):
        g = g.sort_values("t_inference")
        band(ax, g["t_inference"], g, COLORS.get(model), LABELS.get(model, model))
        used.update(g["t_inference"].tolist())
    linthresh = 60
    ax.set_xscale("symlog", linthresh=linthresh)
    # funde ticks a menos de linthresh (p. ej. t=0 de PCM y t=1 s de ReRAM): se solaparían
    ticks, cur = [], []
    for t in sorted(used):
        if cur and t - cur[-1] >= linthresh:
            ticks.append(next((x for x in cur if x in TIME_TICKS), cur[0]))
            cur = []
        cur.append(t)
    ticks.append(next((x for x in cur if x in TIME_TICKS), cur[0]))
    ax.set_xticks(ticks)
    ax.set_xticklabels([TIME_TICKS.get(t, f"{t:g}") for t in ticks])
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    finish(ax, fig, args, "Tiempo desde la programación (--t-inference)", base)


def plot_heatmap(df, summary, args, base):
    if not args.noise_model:
        sys.exit("--kind heatmap necesita --noise-model")
    s = summary[summary["noise_model"] == args.noise_model]
    if s.empty:
        sys.exit(f"Error: no hay filas con noise_model={args.noise_model}.")
    scales, times = sorted(s["prog_noise_scale"].unique()), sorted(s["t_inference"].unique())
    grid = np.full((len(times), len(scales)), np.nan)
    for _, r in s.iterrows():
        grid[times.index(r["t_inference"]), scales.index(r["prog_noise_scale"])] = r["ret_mean"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.pcolormesh(np.arange(len(scales) + 1), np.arange(len(times) + 1), grid, cmap="viridis", shading="flat")
    fig.colorbar(im, ax=ax).set_label("Retorno medio")
    ax.set_xticks(np.arange(len(scales)) + 0.5)
    ax.set_xticklabels([f"{x:g}×" for x in scales])
    ax.set_yticks(np.arange(len(times)) + 0.5)
    ax.set_yticklabels([TIME_TICKS.get(t, f"{t:g}s") for t in times])
    ax.set_xlabel("Escala de ruido (--noise-scale)")
    ax.set_ylabel("Tiempo desde la programación (--t-inference)")
    ax.set_title(args.title if args.title != DEFAULT_TITLE else f"{LABELS[args.noise_model]}: retorno vs escala×tiempo")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Guardado: {out}")
    if args.show:
        plt.show()


def plot_rmse(df, summary, args, base):
    import pandas as pd
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, g in df[df["noise_model"] != "none"].groupby("noise_model"):
        color = COLORS.get(model)
        ax.scatter(g["weight_rmse"], g["return_mean"], color=color, alpha=0.08, s=14, linewidths=0)
        try:  # bins por cuantil de x: weight_rmse no se distribuye uniforme
            bins = pd.qcut(g["weight_rmse"], q=args.bins, duplicates="drop")
        except ValueError:
            continue
        t = (pd.DataFrame({"x": g["weight_rmse"], "y": g["return_mean"], "bin": bins})
             .groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"), s=("y", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan))
             .sort_values("x"))
        ax.plot(t["x"], t["y"], "-o", color=color, linewidth=1.8, markersize=5, label=LABELS.get(model, model))
        ax.fill_between(t["x"], t["y"] - t["s"], t["y"] + t["s"], color=color, alpha=0.15, linewidth=0)
    finish(ax, fig, args, "RMSE efectivo de los pesos realizados (|w_real − w_nominal| / w_max)", base)


def plot_ablation(df, summary, args, base):
    if "tag" not in summary.columns:
        sys.exit("Error: --kind ablation necesita un CSV con columna `tag` (run_noise_sweeps.py --ablation).")
    s = summary[summary["tag"] != ""]
    if args.noise_model:  # el CSV mezcla modelos (tag = "<modelo>:<mecanismo>")
        s = s[s["noise_model"] == args.noise_model]
    if s.empty:
        sys.exit("Error: no hay filas de ablación para ese modelo.")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("tab10")
    for i, (tag, g) in enumerate(s.groupby("tag")):
        # el mecanismo aislado es el único con escala != 0; la escala común es su máximo
        g = g.assign(scale=g[["prog_noise_scale", "read_noise_scale", "drift_scale"]].max(axis=1)).sort_values("scale")
        band(ax, g["scale"], g, cmap(i), tag)
    finish(ax, fig, args, "Escala del mecanismo activo (los otros dos en 0)", base)


DEFAULT_TITLE = "Robustez a ruido de memristores"
KINDS = {"intensity": plot_intensity, "time": plot_time, "heatmap": plot_heatmap, "rmse": plot_rmse,
         "ablation": plot_ablation}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=list(KINDS), required=True)
    ap.add_argument("--csv", nargs="+", required=True, help="uno o más CSV de noise_eval.py (se concatenan)")
    ap.add_argument("--out", required=True, help="ruta del PNG")
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--x-col", default="prog_noise_scale", choices=list(X_LABELS), help="eje X de --kind intensity")
    ap.add_argument("--noise-model", default=None, help="modelo de --kind heatmap y --kind ablation")
    ap.add_argument("--bins", type=int, default=12, help="cuantiles de weight_rmse en --kind rmse")
    ap.add_argument("--baseline", type=float, default=None, help="retorno sin ruido (def.: filas noise_model=none)")
    ap.add_argument("--resamples", type=int, default=2000, help="remuestreos del bootstrap de las bandas")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args(argv)

    df = load_rows(a.csv)
    summary = summarize(df, n_resamples=a.resamples)
    KINDS[a.kind](df, summary, a, baseline_of(df, a.baseline))


if __name__ == "__main__":
    main()
