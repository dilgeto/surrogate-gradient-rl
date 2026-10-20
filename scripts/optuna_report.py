"""Resume un estudio de scripts/optuna_search.py y, opcionalmente, valida los mejores con más semillas.

    python scripts/optuna_report.py pend-f1                       # top 10 + importancias + comandos
    python scripts/optuna_report.py pend-f1 --top 5 --plot        # + optuna/<study>.png (matplotlib)
    python scripts/optuna_report.py pend-f1 --validate 3                 # reentrena el top 3 con 11 semillas (100-110)
    python scripts/optuna_report.py pend-f1 --validate 3 --jobs 12       # ... 12 corridas a la vez

La validación son K x 11 corridas independientes: se lanzan en paralelo, cada una en su propio proceso con 1 hilo
de torch. `--jobs 0` (por defecto) usa el máximo que cabe: min(corridas, núcleos disponibles, RAM libre / ~1.2 GB
por corrida en CPU o ~2 GB en CUDA).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna

COMPLETE = optuna.trial.TrialState.COMPLETE


def auto_jobs(n_runs: int, device: str) -> int:
    """Corridas simultáneas que caben: núcleos disponibles y RAM libre (~1.2 GB por corrida en CPU, ~2 GB en CUDA)."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        kb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
        by_ram = int(kb / 1024 / 1024 / (2.0 if device.startswith("cuda") else 1.2))
    except (OSError, StopIteration, ValueError):
        by_ram = cpus
    return max(1, min(n_runs, cpus, by_ram))


def base_device(argv: list[str]) -> str:
    return next((argv[i + 1] for i in range(len(argv) - 1, -1, -1) if argv[i] == "--device"), "auto")


def run_one(argv: list[str], name: str, out: str) -> dict:
    """Una corrida de sgrl.train en su propio proceso (1 hilo de torch); devuelve su final_eval.json."""
    log = Path(out, "_logs", f"{name}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, "-m", "sgrl.train", *argv, "--threads", "1", "--name", name, "--out", out],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{name} terminó con código {r.returncode}; revisa {log}")
    return json.loads(Path(out, name, "final_eval.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--plot", action="store_true", help="historial, importancias y coordenadas paralelas")
    ap.add_argument("--validate", type=int, default=0, metavar="K", help="reentrenar los K mejores trials")
    ap.add_argument("--seeds", default="100,101,102,103,104,105,106,107,108,109,110",
                    help="semillas de validación (por defecto 11, lejos de las 0, 1, ... de la búsqueda)")
    ap.add_argument("--out", default="runs", help="carpeta de runs de la validación")
    ap.add_argument("--jobs", type=int, default=0, help="corridas de validación simultáneas (0 = el máximo que cabe)")
    ap.add_argument("--device", default=None, help="sustituye al --device de la config base en la validación (p. ej. cpu)")
    a = ap.parse_args()

    storage = a.storage or f"sqlite:///optuna/{a.study}.db"
    study = optuna.load_study(study_name=a.study, storage=storage)
    base = study.user_attrs.get("base_argv", "")
    trials = [t for t in study.trials if t.state == COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    print(f"{a.study}: {len(trials)} completos, {pruned} podados, {len(study.trials) - len(trials) - pruned} otros")
    print(f"base (fija): {base}\nbuscado: {study.user_attrs.get('params')}  ·  semillas/trial: "
          f"{study.user_attrs.get('seeds_per_trial')}  ·  puntuación = "
          f"{'media de toda la curva' if study.user_attrs.get('score') == 'auc' else 'media de las últimas ' + str(study.user_attrs.get('last_k')) + ' evals'}\n")
    if not trials:
        return
    top = sorted(trials, key=lambda t: t.value, reverse=True)[: max(a.top, a.validate)]

    names = list(top[0].params)
    print(f"{'#':>4} {'valor':>9}  " + "  ".join(f"{n:>11}" for n in names))
    for t in top[: a.top]:
        print(f"{t.number:>4} {t.value:9.1f}  " + "  ".join(f"{t.params.get(n, float('nan')):>11.4g}" for n in names))

    if len(trials) >= 10:
        try:
            imp = optuna.importance.get_param_importances(study, target=lambda t: t.value if t.state == COMPLETE else None)
            print("\nimportancia de cada parámetro (fANOVA):")
            for k, v in imp.items():
                print(f"  {k:<16} {v:.2f}")
        except Exception as e:  # pocos datos o parámetros constantes
            print(f"\n(sin importancias: {e})")

    best = top[0]
    print(f"\nmejor: python -m sgrl.train {base} {best.user_attrs['flags']} --seed 1 --name {a.study}-best")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization import matplotlib as ov
        Path("optuna").mkdir(exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        for axis, fn in zip(ax, (ov.plot_optimization_history, ov.plot_param_importances, ov.plot_parallel_coordinate)):
            plt.sca(axis)
            try:
                fn(study)
            except Exception as e:
                axis.set_title(f"{fn.__name__}: {e}", fontsize=8)
        fig.savefig(f"optuna/{a.study}.png", dpi=120, bbox_inches="tight")
        print(f"figura -> optuna/{a.study}.png")

    if a.validate:
        seeds = [int(x) for x in a.seeds.split(",")]
        device_flag = ["--device", a.device] if a.device else []
        runs = []  # (trial, seed, nombre, argv): todas independientes
        for t in top[: a.validate]:
            argv = [*shlex.split(base), *shlex.split(t.user_attrs["flags"]), *device_flag]
            runs += [(t, sd, f"{a.study}-t{t.number}-s{sd}", [*argv, "--seed", str(sd)]) for sd in seeds]
        jobs = a.jobs if a.jobs > 0 else auto_jobs(len(runs), a.device or base_device(shlex.split(base)))
        jobs = max(1, min(jobs, len(runs)))
        print(f"\nvalidación: {len(runs)} corridas ({a.validate} trials x {len(seeds)} semillas), {jobs} a la vez", flush=True)
        finals = {t.number: {} for t in top[: a.validate]}
        failed = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(run_one, argv, name, a.out): (t, sd, name) for t, sd, name, argv in runs}
            for i, fut in enumerate(as_completed(futs), 1):
                t, sd, name = futs[fut]
                try:
                    finals[t.number][sd] = fut.result()["return_mean"]
                    print(f"  [{i}/{len(runs)}] {name}: {finals[t.number][sd]:.1f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    failed.append(name)
                    print(f"  [{i}/{len(runs)}] {name}: FALLÓ ({e})", file=sys.stderr, flush=True)
        summary = {}
        for t in top[: a.validate]:
            vals = [finals[t.number][sd] for sd in seeds if sd in finals[t.number]]  # en el orden de --seeds
            if not vals:
                continue
            summary[t.number] = dict(search_value=t.value, finals=vals, mean=float(np.mean(vals)), std=float(np.std(vals)))
            print(f"trial {t.number}: búsqueda {t.value:.1f} -> validación {np.mean(vals):.1f} ± {np.std(vals):.1f} "
                  f"({len(vals)}/{len(seeds)} semillas)")
        Path(a.out, f"{a.study}-validation.json").write_text(json.dumps(summary, indent=2))
        if failed:
            raise SystemExit(f"fallaron {len(failed)} corridas: {', '.join(failed)}")

if __name__ == "__main__":
    main()
