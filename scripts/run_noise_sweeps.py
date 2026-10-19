"""Orquesta los barridos de ruido de un checkpoint y genera las figuras (puerto de wann-cpp `run_car_noise_sweeps.py`).

  1. INTENSIDAD (--noise-scales, t fijo)        -> <prefix>_noise_sweep.csv  -> *_noise_robustness.png
  2. TIEMPO (--time-points-*, escala 1.0)       -> <prefix>_time_sweep.csv   -> *_time_robustness.png
  3. --grid: grilla escala×tiempo               -> <prefix>_grid_sweep.csv   -> *_heatmap_<modelo>.png
  4. --ablation: un mecanismo activo cada vez    -> <prefix>_ablation.csv     -> *_ablation_<modelo>.png
     (sólo programación / sólo lectura / sólo deriva, sobre pcm y reramcmo, que los separan)
  5. siempre que haya intensidad: *_rmse.png (retorno vs error efectivo de pesos)

Cada invocación de noise_eval.py carga el checkpoint una vez y barre escalas y tiempos por dentro, así que
hay un trabajo por modelo (por mecanismo en la ablación), en paralelo con --jobs. Los CSV intermedios van a
<out-dir>/parts/. Incluye siempre una referencia sin ruido (`--noise-model none`) en el CSV de intensidad.

    python scripts/run_noise_sweeps.py runs/Pendulum-lif-s0/final.pt --n-chips 20 --jobs 3 --grid --ablation
    python scripts/run_noise_sweeps.py runs/Pendulum-lif-s0/final.pt --dry-run     # sólo imprime los comandos
"""
import argparse
import concurrent.futures
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NOISE_EVAL = str(HERE / "noise_eval.py")
PLOT = str(HERE / "plot_noise.py")
MODELS = ["simple", "pcm", "reram", "reramcmo"]
# Modelos que separan programación / lectura / deriva (reram usa un solo valor para los tres)
ABLATION_MODELS = ["pcm", "reramcmo"]
MECHANISMS = {"programacion": ("--prog-noise-scale", "--read-noise-scale", "--drift-scale"),
              "lectura": ("--read-noise-scale", "--prog-noise-scale", "--drift-scale"),
              "deriva": ("--drift-scale", "--prog-noise-scale", "--read-noise-scale")}


def base_cmd(a):
    return [sys.executable, NOISE_EVAL, str(a.checkpoint), "--seeds", *map(str, a.seeds), "--n-chips", str(a.n_chips),
            "--reads-per-chip", str(a.reads_per_chip), "--episodes", str(a.episodes), "--device", a.device,
            "--reramcmo-g-max", str(a.reramcmo_g_max), "--reramcmo-g-min", str(a.reramcmo_g_min),
            "--n-levels", str(a.n_levels)]


def times_for(a, model):
    return {"pcm": a.time_points_pcm, "reram": a.time_points_reram, "reramcmo": a.time_points_reramcmo}[model]


def intensity_jobs(a, parts):
    jobs = [("intensity none", "", base_cmd(a) + ["--noise-model", "none", "--out", str(parts / "intensity_none.csv")])]
    for m in MODELS:
        # reram sólo admite sus claves exactas; reramcmo con t=0 es un bypass total: 1 s ejercita los tres mecanismos
        t = ["--t-inference", "1"] if m in ("reram", "reramcmo") else []
        jobs.append((f"intensity {m}", "", base_cmd(a) + ["--noise-model", m, "--noise-scales", a.noise_scales, *t,
                                                           "--out", str(parts / f"intensity_{m}.csv")]))
    return jobs


def time_jobs(a, parts):  # `simple` no tiene deriva: se excluye, como en plot_noise.py
    return [(f"time {m}", "", base_cmd(a) + ["--noise-model", m, "--noise-scales", "1.0", "--t-inference", times_for(a, m),
                                              "--out", str(parts / f"time_{m}.csv")]) for m in ("pcm", "reram", "reramcmo")]


def grid_jobs(a, parts):
    return [(f"grid {m}", "", base_cmd(a) + ["--noise-model", m, "--noise-scales", a.noise_scales, "--t-inference",
                                              times_for(a, m), "--out", str(parts / f"grid_{m}.csv")])
            for m in ("pcm", "reram", "reramcmo")]


def ablation_jobs(a, parts):
    jobs = []
    for m in ABLATION_MODELS:
        t = ["--t-inference", "1"] if m == "reramcmo" else ["--t-inference", a.ablation_t]
        for tag, (on, off1, off2) in MECHANISMS.items():
            # el mecanismo activo sigue --noise-scales; los otros dos quedan en 0
            jobs.append((f"ablation {m} {tag}", f"{m}:{tag}",
                         base_cmd(a) + ["--noise-model", m, "--noise-scales", a.noise_scales, off1, "0", off2, "0", *t,
                                        "--out", str(parts / f"ablation_{m}_{tag}.csv")]))
    return jobs


def run_one(label, cmd, timeout):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        sys.exit(f"[TIMEOUT] {label}: {' '.join(cmd)}")
    if proc.returncode != 0:
        sys.exit(f"[FAIL] {label}\n$ {' '.join(cmd)}\n{proc.stderr}")
    print(f"  ok: {label}", flush=True)


def run_many(jobs, a):
    """Corre los trabajos (en paralelo) y devuelve [(tag, csv_parcial)]; en --dry-run sólo los imprime."""
    if a.dry_run:
        for label, _, cmd in jobs:
            print(f"[dry-run] {label}: {' '.join(cmd)}")
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(lambda j: run_one(j[0], j[2], a.timeout), jobs))
    return [(tag, Path(cmd[cmd.index("--out") + 1])) for _, tag, cmd in jobs]


def combine(parts, out_csv, tagged=False):
    frames = []
    for tag, path in parts:
        df = pd.read_csv(path)
        if tagged:
            df["tag"] = tag
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Guardado: {out_csv} ({len(out)} filas)")


def plot(kind, csv, png, title, *extra):
    subprocess.run([sys.executable, PLOT, "--kind", kind, "--csv", str(csv), "--out", str(png), "--title", title, *extra],
                   check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-chips", type=int, default=20)
    ap.add_argument("--reads-per-chip", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=10, help="episodios por lectura")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-levels", type=int, default=0, help="cuantización de conductancia (<=1: continua)")
    ap.add_argument("--noise-scales", default="0,0.5,1,2,3,4", help="escalas de los barridos de intensidad, grilla y ablación")
    ap.add_argument("--time-points-pcm", default="0,60,300,3600,21600,86400,604800,2592000,31536000",
                    help="s desde la programación; grilla log de 60 s a 1 año más 0 = recién programado")
    ap.add_argument("--time-points-reram", default="1,86400,172800", help="deben ser EXACTAMENTE estas 3 claves")
    ap.add_argument("--time-points-reramcmo", default="0,60,300,3600,21600,86400,604800,2592000,31536000",
                    help="t=0 es un bypass total en este modelo")
    ap.add_argument("--ablation-t", default="86400",
                    help="t (s) de la ablación en PCM; a t=0 la deriva de PCM no actúa, así que no uses 0")
    ap.add_argument("--reramcmo-g-max", type=float, default=25.0)
    ap.add_argument("--reramcmo-g-min", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=None, help="def.: <carpeta del checkpoint>/noise")
    ap.add_argument("--prefix", default=None, help="def.: nombre de la carpeta del checkpoint")
    ap.add_argument("--title", default=None)
    ap.add_argument("--jobs", type=int, default=3, help="invocaciones de noise_eval.py en paralelo")
    ap.add_argument("--timeout", type=int, default=7200, help="segundos por invocación")
    ap.add_argument("--grid", action="store_true", help="además, la grilla escala×tiempo y sus heatmaps")
    ap.add_argument("--ablation", action="store_true", help="además, la ablación por mecanismo (pcm y reramcmo)")
    ap.add_argument("--skip-intensity", action="store_true")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="sólo imprimir los comandos")
    a = ap.parse_args(argv)

    out_dir = a.out_dir or a.checkpoint.parent / "noise"
    prefix = a.prefix or a.checkpoint.parent.name
    title = a.title or prefix
    parts = out_dir / "parts"
    if not a.dry_run:
        parts.mkdir(parents=True, exist_ok=True)
    csvs = {k: out_dir / f"{prefix}_{k}.csv" for k in ("noise_sweep", "time_sweep", "grid_sweep", "ablation")}

    for enabled, name, make, key, tagged in [
        (not a.skip_intensity, "Barrido de intensidad (--noise-scale)", intensity_jobs, "noise_sweep", False),
        (not a.skip_time, "Barrido de tiempo (--t-inference)", time_jobs, "time_sweep", False),
        (a.grid, "Grilla escala×tiempo", grid_jobs, "grid_sweep", False),
        (a.ablation, "Ablación por mecanismo", ablation_jobs, "ablation", True),
    ]:
        if not enabled:
            continue
        print(f"== {name} ==")
        done = run_many(make(a, parts), a)
        if done:
            combine(done, csvs[key], tagged)

    if a.dry_run or a.skip_plots:
        return
    print("== Figuras ==")
    if csvs["noise_sweep"].is_file():
        plot("intensity", csvs["noise_sweep"], out_dir / f"{prefix}_noise_robustness.png", title)
        plot("rmse", csvs["noise_sweep"], out_dir / f"{prefix}_rmse.png", title)
    if csvs["time_sweep"].is_file():
        plot("time", csvs["time_sweep"], out_dir / f"{prefix}_time_robustness.png", title)
    if a.grid and csvs["grid_sweep"].is_file():
        for m in ("pcm", "reram", "reramcmo"):
            plot("heatmap", csvs["grid_sweep"], out_dir / f"{prefix}_heatmap_{m}.png", title, "--noise-model", m)
    if a.ablation and csvs["ablation"].is_file():
        for m in ABLATION_MODELS:
            plot("ablation", csvs["ablation"], out_dir / f"{prefix}_ablation_{m}.png", title, "--noise-model", m)


if __name__ == "__main__":
    main()
