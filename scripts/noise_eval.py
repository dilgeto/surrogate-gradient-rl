"""Robustez de un checkpoint a ruido de memristores (protocolo de ../wann-cpp `wann_car_noise`).

Para cada (semilla, chip) se programa un chip nuevo (ruido de escritura, fijo), y por cada lectura
(`--reads-per-chip`) y cada `--t-inference` se muestrea ruido de lectura/deriva nuevo, se fijan los
pesos efectivos y se evalúan `--episodes` episodios con la política determinista. Una fila de CSV
por (semilla, chip, lectura, t, escala). Sólo evalúa, nunca entrena.

    python scripts/noise_eval.py runs/Pendulum-lif-s0/final.pt --noise-model pcm \
        --noise-scales 0,1,2,4 --t-inference 0,3600,86400 --n-chips 20 --out noise_pcm.csv
    python scripts/noise_eval.py runs/Pendulum-lif-s0/final.pt --noise-model none   # regresión: = eval.py
    python scripts/summarize_noise.py --csv noise_pcm.csv
    python scripts/run_noise_sweeps.py runs/Pendulum-lif-s0/final.pt --grid --ablation   # barridos + figuras

`weight_rmse`: error RMS de los pesos realizados frente a los nominales, normalizado por el w_max de
cada capa; misma unidad en todos los modelos, sirve de eje común (como en wann-cpp).
Los modelos y sus parámetros están en sgrl/noise.py.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from sgrl.checkpoint import load
from sgrl.noise import Chip, add_noise_args, config_from_args
from sgrl.ppo import evaluate

COLUMNS = ["seed", "chip", "read", "t_inference", "noise_model", "noise_scale", "prog_noise_scale",
           "read_noise_scale", "drift_scale", "n_levels", "weight_rmse", "return_mean", "return_std", "length_mean"]


def floats(csv_s: str):
    return [float(x) for x in csv_s.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-chips", type=int, default=20)
    ap.add_argument("--reads-per-chip", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=10, help="episodios por lectura")
    ap.add_argument("--t-inference", default=None, help="s desde la programación, separados por comas (def. según modelo)")
    ap.add_argument("--noise-scales", default=None, help="barrido de --noise-scale, separado por comas")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="CSV de salida (def. stdout)")
    add_noise_args(ap)
    a = ap.parse_args(argv)

    agent, norm, env_name = load(a.checkpoint, a.device)
    device = torch.device(a.device)
    scales = floats(a.noise_scales) if a.noise_scales else [a.noise_scale]
    ts = floats(a.t_inference) if a.t_inference else [None]
    out = open(a.out, "w", newline="") if a.out else sys.stdout
    w = csv.DictWriter(out, COLUMNS)
    w.writeheader()
    for scale in scales:
        cfg = config_from_args(a, noise_scale=scale)
        pscale, rscale, dscale = cfg.scales()
        for seed in a.seeds:
            for c in range(a.n_chips):
                # un generador por (semilla, chip), como wann_car_noise
                with Chip(agent.actor, cfg, seed=seed * 1_000_003 + c) as chip:
                    chip.program()
                    for read in range(a.reads_per_chip):
                        for t in ts:
                            tt = cfg.t if t is None else t
                            rmse = chip.read(tt)
                            base_seed = seed * 10_000_000 + c * 10_000 + read * 100
                            ev = evaluate(agent, env_name, norm, a.episodes, base_seed, device)
                            w.writerow(dict(seed=seed, chip=c, read=read, t_inference=tt, noise_model=cfg.model,
                                            noise_scale=scale, prog_noise_scale=pscale, read_noise_scale=rscale,
                                            drift_scale=dscale, n_levels=cfg.n_levels, weight_rmse=rmse,
                                            return_mean=ev["return_mean"], return_std=ev["return_std"],
                                            length_mean=ev["length_mean"]))
            out.flush()
    if a.out:
        out.close()
        print(f"-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
