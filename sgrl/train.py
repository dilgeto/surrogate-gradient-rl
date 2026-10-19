"""Entrena un actor SNN (LIF de snnTorch, gradientes surrogados) o una ANN con PPO.

    python -m sgrl.train --env Acrobot --neuron lif --device cuda
    python -m sgrl.train --env Bottleneck --neuron lif --train-noise-model pcm   # entrenar con ruido
    python -m sgrl.train --env BipedalWalker --neuron ann     # línea base sin espigas
Cada tarea de TASKS trae hiperparámetros por defecto (sgrl/tasks.py); los flags los sobrescriben.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .envs import ENV_NAMES, TASKS, VecEnv
from .policy import ActorCritic
from .ppo import PPO, PPOConfig, evaluate
from .noise import Chip, add_noise_args, config_from_args, noise_disabled
from .snn import NEURONS, SURROGATES
from .tasks import PRESETS


SB3_DEFAULTS = dict(minibatch=64, epochs=10, actor_lr=3e-4, critic_lr=3e-4, target_kl=0.0,
                    no_anneal_lr=True, init_std=1.0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=ENV_NAMES, default="Pendulum")
    ap.add_argument("--neuron", choices=(*NEURONS, "ann"), default="lif")
    ap.add_argument("--steps", type=int, default=500_000, help="pasos de entorno totales")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cpu | cuda | cuda:N | auto (cuda si hay)")
    ap.add_argument("--threads", type=int, default=4, help="hilos de torch")
    ap.add_argument("--env-threads", type=int, default=1, help="hilos OpenMP del entorno")
    ap.add_argument("--out", default="runs", help="directorio de resultados")
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-every", type=int, default=20, help="evaluar cada N iteraciones (0 = sólo al final)")
    ap.add_argument("--eval-episodes", type=int, default=10)
    g = ap.add_argument_group("SNN")
    g.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    g.add_argument("--sim-steps", type=int, default=16, help="pasos de simulación neuronal por paso del entorno")
    g.add_argument("--surrogate", choices=SURROGATES, default="fast_sigmoid")
    g.add_argument("--alpha", type=float, default=None, help="nitidez del surrogado (def. según surrogado)")
    g.add_argument("--beta", type=float, default=0.9, help="decaimiento de membrana de las neuronas LIF por paso (exp(-dt/tau))")
    g.add_argument("--in-gain", type=float, default=None)
    g.add_argument("--hid-gain", type=float, default=None)
    g = ap.add_argument_group("PPO")
    d = PPOConfig()
    g.add_argument("--num-envs", type=int, default=d.num_envs)
    g.add_argument("--rollout-steps", type=int, default=d.rollout_steps)
    g.add_argument("--minibatch", type=int, default=d.minibatch_size)
    g.add_argument("--epochs", type=int, default=d.epochs)
    g.add_argument("--gamma", type=float, default=d.gamma)
    g.add_argument("--gae-lambda", type=float, default=d.gae_lambda)
    g.add_argument("--actor-lr", type=float, default=d.actor_lr)
    g.add_argument("--critic-lr", type=float, default=d.critic_lr)
    g.add_argument("--ent-coef", type=float, default=d.ent_coef)
    g.add_argument("--target-kl", type=float, default=d.target_kl)
    g.add_argument("--no-anneal-lr", action="store_true")
    g.add_argument("--init-std", type=float, default=0.5)
    g.add_argument("--sb3", action="store_true",
                   help="hiperparámetros por defecto de SB3 PPO (minilote 64, 10 épocas, LR único, sin target_kl/"
                        "annealing, recorte global, init ortogonal, std 1); los flags explícitos siguen ganando")
    add_noise_args(ap, "train-")
    ap.add_argument("--train-noise-t", type=float, default=None, help="t_inference (s) de la lectura durante el entrenamiento")
    ap.add_argument("--train-noise-every", type=int, default=1, help="chip nuevo cada N iteraciones de PPO")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = build_parser()
    pre = ap.parse_known_args(argv)[0]
    ap.set_defaults(**PRESETS.get(pre.env, {}))  # los flags explícitos siguen ganando
    if pre.sb3:
        ap.set_defaults(**SB3_DEFAULTS)
    a = ap.parse_args(argv)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda pedido pero torch no ve ninguna GPU (¿build CPU de torch?)")
    return a


def build(a: argparse.Namespace):
    """Fija semillas y construye (env, agente, PPO, config PPO, kwargs de la SNN) a partir de los args."""
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)

    env = VecEnv(a.env, a.num_envs, seed=a.seed, num_threads=a.env_threads)
    snn_kwargs = {}
    if a.neuron != "ann":
        snn_kwargs = dict(steps=a.sim_steps, surrogate=a.surrogate, alpha=a.alpha, beta=a.beta,
                          in_gain=a.in_gain, hid_gain=a.hid_gain)
    agent = ActorCritic(env.obs_dim, env.act_dim, a.neuron, tuple(a.hidden), init_std=a.init_std,
                        n_actions=env.n_actions, n_agents=env.n_agents, ortho_init=a.sb3, **snn_kwargs)
    cfg = PPOConfig(num_envs=a.num_envs, rollout_steps=a.rollout_steps, minibatch_size=a.minibatch,
                    epochs=a.epochs, gamma=a.gamma, gae_lambda=a.gae_lambda, actor_lr=a.actor_lr,
                    critic_lr=a.critic_lr, ent_coef=a.ent_coef, target_kl=a.target_kl or None,
                    anneal_lr=not a.no_anneal_lr, global_grad_clip=a.sb3)
    ppo = PPO(agent, env, cfg, a.device)
    return env, agent, ppo, cfg, snn_kwargs


def main(argv=None):
    a = parse_args(argv)
    env, agent, ppo, cfg, snn_kwargs = build(a)
    noise_cfg = config_from_args(a, "train-", t_inference=a.train_noise_t)
    chip = Chip(agent.actor, noise_cfg, seed=a.seed + 777) if noise_cfg.active() else None

    def new_chip(it: int):  # mismo chip en el rollout y en las épocas de la iteración
        if it % max(a.train_noise_every, 1) == 0:
            chip.resample()

    name = a.name or f"{a.env}-{a.neuron}{'-' + a.train_noise_model if a.train_noise_model != 'none' else ''}-s{a.seed}"
    run = Path(a.out) / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"args": vars(a), "ppo": vars(cfg), "snn": snn_kwargs, "preset": PRESETS.get(a.env, {}), "train_noise": vars(noise_cfg),
                                                  "obs_dim": env.obs_dim, "act_dim": env.act_dim}, indent=2))
    n_act = sum(p.numel() for p in agent.actor.parameters())
    dev = a.device + (f" ({torch.cuda.get_device_name(torch.device(a.device))})" if a.device.startswith("cuda") else "")
    print(f"{name}: obs={env.obs_dim} act={env.act_dim} agentes={env.n_agents} acciones={env.n_actions or 'continuas'} "
          f"actor={n_act} params  {a.steps} pasos  [{dev}]")

    fields = ["iteration", "steps", "ep_return", "ep_len", "eval_return", "eval_std", "pg", "vf", "kl",
              "clipfrac", "ent", "std", "grad_norm", "explained_var", "sps", "rate_mean", "rates"]
    fh = open(run / "log.csv", "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    best = {"v": -np.inf}

    def save(tag: str):
        torch.save({"agent": agent.state_dict(), "norm": ppo.norm.rms.state(), "env": a.env,
                    "neuron": a.neuron, "hidden": a.hidden, "snn": snn_kwargs,
                    "obs_dim": env.obs_dim, "act_dim": env.act_dim, "n_actions": env.n_actions,
                    "n_agents": env.n_agents}, run / f"{tag}.pt")

    def log(info: dict):
        it = info["iteration"]
        if a.eval_every and it % a.eval_every == 0:
            with noise_disabled(agent.actor):  # evaluación con pesos nominales (con ruido: scripts/noise_eval.py)
                ev = evaluate(agent, a.env, ppo.norm, a.eval_episodes, a.seed + 10_000, ppo.device)
            info["eval_return"], info["eval_std"] = ev["return_mean"], ev["return_std"]
            if ev["return_mean"] > best["v"]:
                best["v"] = ev["return_mean"]
                save("best")
        info["rate_mean"] = float(np.mean(info["rates"])) if info["rates"] else ""
        info["rates"] = " ".join(f"{r:.3f}" for r in info["rates"])
        writer.writerow(info)
        fh.flush()
        if it % 5 == 0 or "eval_return" in info:
            ev_s = f" eval {info['eval_return']:9.1f}" if "eval_return" in info else ""
            print(f"it {it:4d} steps {info['steps']:8d} ret {info['ep_return']:9.1f}{ev_s} | "
                  f"kl {info['kl']:.4f} std {info['std']:.3f} gn {info['grad_norm']:.2f} "
                  f"rate [{info['rates']}] | {info['sps']:.0f} sps", flush=True)

    t0 = time.time()
    ppo.learn(a.steps, log, iter_hook=new_chip if chip else None)
    if chip:
        chip.detach()
    final = evaluate(agent, a.env, ppo.norm, max(a.eval_episodes, 20), a.seed + 20_000, ppo.device)
    save("final")
    (run / "final_eval.json").write_text(json.dumps({**final, "train_seconds": time.time() - t0}, indent=2))
    print(f"final eval ({max(a.eval_episodes, 20)} episodios): {final['return_mean']:.1f} ± {final['return_std']:.1f}  "
          f"[{time.time() - t0:.0f} s]  -> {run}")
    fh.close()
    return final


if __name__ == "__main__":
    main()
