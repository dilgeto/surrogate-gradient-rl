"""PPO de stable-baselines3 con ANN sobre los mismos entornos RLTools, para validar el PPO propio.

    python scripts/sb3_baseline.py --env Pendulum --steps 300000 --gamma 0.9 --seed 0
    python -m sgrl.train --env Pendulum --neuron ann --sb3 --steps 300000 --gamma 0.9 --seed 0

Ambos usan los mismos hiperparámetros (los de SB3 por defecto) y la misma evaluación (política
determinista, `--eval-episodes` episodios con semilla distinta de la de entrenamiento); comparar
`log.csv` (steps, ep_return, eval_return) y `final_eval.json` de las dos corridas. SB3 normaliza con
`VecNormalize` (obs y recompensa, clip 10), equivalente al normalizador propio.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from sgrl.sb3_compat import SB3VecEnv


def evaluate(model, vecnorm: VecNormalize, env_name: str, episodes: int, seed: int) -> dict:
    """Mismo protocolo que `sgrl.ppo.evaluate`: política determinista, estadísticos de obs congelados."""
    env = SB3VecEnv(env_name, episodes, seed=seed)
    env.seed(seed)
    obs = env.reset()
    ret, cur, done = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes, bool)
    length, cur_l = np.zeros(episodes), np.zeros(episodes)
    while not done.all():
        act, _ = model.predict(vecnorm.normalize_obs(obs), deterministic=True)
        obs, rew, dn, _ = env.step(act)
        cur, cur_l = cur + rew, cur_l + 1
        fin = dn & ~done
        ret[fin], length[fin] = cur[fin], cur_l[fin]
        done |= dn
    return {"return_mean": float(ret.mean()), "return_std": float(ret.std()), "length_mean": float(length.mean())}


class Logger(BaseCallback):
    def __init__(self, a, vecnorm, writer, fh):
        super().__init__()
        self.a, self.vn, self.w, self.fh = a, vecnorm, writer, fh
        self.recent, self.it, self.t0 = [], 0, time.time()

    def _on_step(self):
        self.recent += [i["episode"]["r"] for i in self.locals["infos"] if "episode" in i]
        return True

    def _on_rollout_end(self):
        self.it += 1
        row = {"iteration": self.it, "steps": self.num_timesteps, "ep_return": float(np.mean(self.recent[-20:])) if self.recent else float("nan"),
               "sps": self.num_timesteps / (time.time() - self.t0)}
        if self.a.eval_every and self.it % self.a.eval_every == 0:
            ev = evaluate(self.model, self.vn, self.a.env, self.a.eval_episodes, self.a.seed + 10_000)
            row["eval_return"], row["eval_std"] = ev["return_mean"], ev["return_std"]
            print(f"it {self.it:4d} steps {row['steps']:8d} ret {row['ep_return']:9.1f} eval {ev['return_mean']:9.1f}", flush=True)
        self.w.writerow(row)
        self.fh.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="Pendulum")
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--rollout-steps", type=int, default=128, help="pasos por entorno (16 × 128 = 2048, como SB3)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    venv = SB3VecEnv(a.env, a.num_envs, seed=a.seed)
    venv.seed(a.seed)
    venv = VecNormalize(venv, gamma=a.gamma)
    model = SB3PPO("MlpPolicy", venv, n_steps=a.rollout_steps, gamma=a.gamma, ent_coef=a.ent_coef, seed=a.seed,
                   policy_kwargs=dict(net_arch=dict(pi=a.hidden, vf=a.hidden)), device=a.device, verbose=0)
    run = Path(a.out) / (a.name or f"{a.env}-sb3-s{a.seed}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"args": vars(a), "sb3_version": __import__("stable_baselines3").__version__}, indent=2))
    fh = open(run / "log.csv", "w", newline="")
    w = csv.DictWriter(fh, ["iteration", "steps", "ep_return", "eval_return", "eval_std", "sps"], extrasaction="ignore")
    w.writeheader()
    t0 = time.time()
    model.learn(a.steps, callback=Logger(a, venv, w, fh))
    final = evaluate(model, venv, a.env, max(a.eval_episodes, 20), a.seed + 20_000)
    (run / "final_eval.json").write_text(json.dumps({**final, "train_seconds": time.time() - t0}, indent=2))
    print(f"final eval: {final['return_mean']:.1f} ± {final['return_std']:.1f}  [{time.time() - t0:.0f} s]  -> {run}")
    fh.close()


if __name__ == "__main__":
    main()
