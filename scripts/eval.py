"""Evalúa un checkpoint (`best.pt` / `final.pt`) sin reentrenar.

    python scripts/eval.py runs/Pendulum-lif-s0/best.pt --episodes 100 --seeds 0 1 2
    python scripts/eval.py runs/Pendulum-lif-s0/final.pt --json out.json
    python scripts/eval.py runs/CarTrack-lif-s0/best.pt --episodes 100 --csv episodios.csv   # una fila por episodio
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sgrl.checkpoint import load
from sgrl.ppo import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345], help="semillas del entorno (episodios independientes por semilla)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None, help="retorno y longitud de cada episodio (seed, episode, return, length)")
    a = ap.parse_args()

    agent, norm, env_name = load(a.checkpoint, a.device)
    res = [evaluate(agent, env_name, norm, a.episodes, s, torch.device(a.device)) for s in a.seeds]
    means = np.array([r["return_mean"] for r in res])
    out = {"checkpoint": str(a.checkpoint), "env": env_name, "episodes_per_seed": a.episodes, "seeds": a.seeds,
           "return_mean": float(means.mean()), "return_std_across_seeds": float(means.std()),
           "return_std_within": float(np.mean([r["return_std"] for r in res])),
           "length_mean": float(np.mean([r["length_mean"] for r in res])), "per_seed": res}
    print(f"{env_name} [{a.checkpoint}]: {out['return_mean']:.1f} ± {out['return_std_within']:.1f} "
          f"(episodios) | entre semillas ± {out['return_std_across_seeds']:.1f} | long. {out['length_mean']:.0f}")
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("seed,episode,return,length\n")
            for sd, r in zip(a.seeds, res):
                f.writelines(f"{sd},{i},{x:.6f},{int(n)}\n" for i, (x, n) in enumerate(zip(r["returns"], r["lengths"])))
        print(f"-> {a.csv}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
