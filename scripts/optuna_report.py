"""Resume un estudio de scripts/optuna_search.py y, opcionalmente, valida los mejores con más semillas.

    python scripts/optuna_report.py pend-f1                       # top 10 + importancias + comandos
    python scripts/optuna_report.py pend-f1 --top 5 --plot        # + optuna/<study>.png (matplotlib)
    python scripts/optuna_report.py pend-f1 --validate 3                 # reentrena el top 3 con 11 semillas (100-110)
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np
import optuna

from sgrl.train import main as train_main

COMPLETE = optuna.trial.TrialState.COMPLETE


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
        seeds = [int(s) for s in a.seeds.split(",")]
        summary = {}
        for t in top[: a.validate]:
            finals = []
            for s in seeds:
                name = f"{a.study}-t{t.number}-s{s}"
                res = train_main([*shlex.split(base), *shlex.split(t.user_attrs["flags"]), "--seed", str(s),
                                  "--name", name, "--out", a.out])
                finals.append(res["return_mean"])
            summary[t.number] = dict(search_value=t.value, finals=finals, mean=float(np.mean(finals)), std=float(np.std(finals)))
            print(f"trial {t.number}: búsqueda {t.value:.1f} -> validación {np.mean(finals):.1f} ± {np.std(finals):.1f} "
                  f"({len(seeds)} semillas)")
        Path(a.out, f"{a.study}-validation.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
