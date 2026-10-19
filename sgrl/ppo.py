"""PPO (clip) con GAE sobre entornos vectorizados de RLTools; vale para actores SNN y ANN.

Detalles que importan con SNN:
  * El actor se reevalúa en cada minilote (log-prob de las acciones guardadas), así que el
    gradiente surrogado atraviesa los `steps` pasos de simulación en cada actualización.
  * Normalización de observaciones y escalado de recompensas con estadísticos móviles (como el
    PPO de RLTools con NORMALIZE_OBSERVATIONS): las corrientes de entrada de las neuronas tienen
    que estar en un rango estable para que la tasa de disparo no se sature ni se apague.
  * Los episodios cortados por límite de tiempo hacen bootstrap con V(última observación).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .envs import POTENTIALS, VecEnv
from .policy import ActorCritic


@dataclass
class PPOConfig:
    num_envs: int = 16
    rollout_steps: int = 128          # pasos por entorno y por iteración
    minibatch_size: int = 512
    epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    anneal_lr: bool = True
    target_kl: Optional[float] = 0.03  # corta las épocas si el KL aproximado lo supera 1.5x
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    global_grad_clip: bool = False    # True: una sola norma sobre actor+crítico (como SB3)
    # Escala del reward shaping basado en potencial, como en wann-cpp: a la recompensa de ENTRENAMIENTO se
    # suma escala·(φ(s') - φ(s)) (sin descuento). Los retornos de evaluación y de los logs siguen siendo los
    # originales del entorno. 0 = sin shaping. Sólo Acrobot y MountainCarDiscrete tienen potencial.
    reward_shaping: float = 0.0

    @classmethod
    def sb3(cls, **overrides) -> "PPOConfig":
        """Hiperparámetros por defecto de stable-baselines3 PPO (aquí con 16 entornos × 128 pasos = 2048).

        Un solo LR, minilote 64, 10 épocas, sin `target_kl`, sin annealing y recorte de gradiente
        global. Para igualarlo del todo usa también `ActorCritic(..., ortho_init=True, init_std=1.0)`.
        Con un actor SNN son ~20× más actualizaciones por iteración que la configuración propia.
        """
        base = dict(num_envs=16, rollout_steps=128, minibatch_size=64, epochs=10, actor_lr=3e-4,
                    critic_lr=3e-4, anneal_lr=False, target_kl=None, global_grad_clip=True)
        return cls(**{**base, **overrides})


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        x = x.reshape(-1, *self.mean.shape) if self.mean.shape else x.reshape(-1)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d**2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    def state(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = s["mean"].copy(), s["var"].copy(), s["count"]


class Normalizer:
    """Normalización de observaciones (congelable para evaluación).

    Con `group` > 1 (multiagente) las observaciones son `group` bloques iguales y se comparten
    los estadísticos entre bloques, de modo que un mismo valor normaliza igual para cualquier agente.
    """

    def __init__(self, obs_dim: int, clip: float = 10.0, enabled: bool = True, group: int = 1):
        self.group, self.d = group, obs_dim // group
        self.rms, self.clip, self.enabled = RunningMeanStd((self.d,)), clip, enabled

    def __call__(self, obs: np.ndarray, update: bool = False) -> np.ndarray:
        if not self.enabled:
            return obs.astype(np.float32)
        blocks = obs.reshape(-1, self.d)
        if update:
            self.rms.update(blocks)
        out = np.clip((blocks - self.rms.mean) / np.sqrt(self.rms.var + 1e-8), -self.clip, self.clip)
        return out.reshape(obs.shape).astype(np.float32)


@torch.no_grad()
def evaluate(agent: ActorCritic, env_name: str, norm: Normalizer, episodes: int, seed: int, device) -> dict:
    """Retorno de `episodes` episodios con la política determinista (media de la gaussiana)."""
    env = VecEnv(env_name, episodes, seed=seed)
    obs = env.reset(seed)
    ret, length = np.zeros(episodes), np.zeros(episodes, dtype=np.int64)
    done = np.zeros(episodes, dtype=bool)
    cur_r, cur_l = np.zeros(episodes), np.zeros(episodes, dtype=np.int64)
    agent.eval()
    while not done.all():
        a, *_ = agent.act(torch.as_tensor(norm(obs), device=device), deterministic=True)
        obs, rew, term, trunc, _, _ = env.step(a.cpu().numpy().astype(np.float64))
        cur_r += rew
        cur_l += 1
        fin = (term | trunc) & ~done
        ret[fin], length[fin] = cur_r[fin], cur_l[fin]
        done |= fin
    agent.train()
    return {"return_mean": float(ret.mean()), "return_std": float(ret.std()), "length_mean": float(length.mean()),
            "returns": ret.tolist(), "lengths": length.tolist()}  # por episodio


class PPO:
    def __init__(self, agent: ActorCritic, env: VecEnv, cfg: PPOConfig, device="cpu"):
        self.agent, self.env, self.cfg, self.device = agent.to(device), env, cfg, torch.device(device)
        self.actor_params = list(agent.actor.parameters()) + [agent.log_std]
        self.all_params = self.actor_params + list(agent.critic.parameters())
        self.opt = torch.optim.Adam(
            [{"params": self.actor_params, "lr": cfg.actor_lr}, {"params": agent.critic.parameters(), "lr": cfg.critic_lr}],
            eps=1e-5,
        )
        if cfg.reward_shaping and env.potential(np.zeros((1, env.obs_dim))) is None:
            raise ValueError(f"{env.name} no define potencial: no admite reward_shaping (sólo {sorted(POTENTIALS)})")
        self.norm = Normalizer(env.obs_dim, cfg.clip_obs, cfg.normalize_obs, env.n_agents)
        self.ret_rms = RunningMeanStd()
        self._disc_ret = np.zeros(env.num_envs)
        self.obs = env.reset()
        self.total_steps = 0
        self.recent = deque(maxlen=20)  # (retorno, longitud) de los últimos episodios

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(x, device=self.device)

    def _scale_reward(self, rew: np.ndarray, done: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize_reward:
            return rew
        self._disc_ret = self._disc_ret * self.cfg.gamma + rew
        self.ret_rms.update(self._disc_ret)
        self._disc_ret[done] = 0
        return np.clip(rew / np.sqrt(self.ret_rms.var + 1e-8), -10, 10)

    @torch.no_grad()
    def collect(self):
        c, n, T = self.cfg, self.env.num_envs, self.cfg.rollout_steps
        D, A = self.env.obs_dim, self.env.act_dim
        b_obs = np.zeros((T, n, D), np.float32)
        b_act = np.zeros((T, n, A), np.float32)
        b_logp, b_val = np.zeros((T, n), np.float32), np.zeros((T, n), np.float32)
        b_rew, b_done = np.zeros((T, n), np.float32), np.zeros((T, n), np.float32)
        rates = []
        for t in range(T):
            o = self.norm(self.obs, update=True)
            act, logp, val, r = self.agent.act(self._t(o))
            act_np = act.cpu().numpy()
            obs, rew, term, trunc, final_obs, finished = self.env.step(act_np.astype(np.float64))
            done = term | trunc
            if c.reward_shaping:  # sólo lo ve el aprendizaje; `finished` (retornos reportados) sigue siendo el original
                rew = rew + c.reward_shaping * (self.env.potential(final_obs) - self.env.potential(self.obs))
            rew_s = self._scale_reward(rew, done).astype(np.float32)
            if trunc.any():  # bootstrap del valor en cortes por tiempo
                idx = np.flatnonzero(trunc)
                v_fin = self.agent.value(self._t(self.norm(final_obs[idx]))).cpu().numpy()
                rew_s[idx] += c.gamma * v_fin
            b_obs[t], b_act[t], b_logp[t], b_val[t] = o, act_np, logp.cpu().numpy(), val.cpu().numpy()
            b_rew[t], b_done[t] = rew_s, done
            if r.numel():
                rates.append(r.cpu().numpy())
            self.recent.extend(finished)
            self.obs = obs
        self.total_steps += T * n
        last_val = self.agent.value(self._t(self.norm(self.obs))).cpu().numpy()
        adv = np.zeros((T, n), np.float32)
        gae = np.zeros(n, np.float32)
        for t in reversed(range(T)):
            next_v = last_val if t == T - 1 else b_val[t + 1]
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + c.gamma * next_v * nonterm - b_val[t]
            gae = delta + c.gamma * c.gae_lambda * nonterm * gae
            adv[t] = gae
        batch = dict(obs=b_obs, act=b_act, logp=b_logp, adv=adv, ret=adv + b_val, val=b_val)
        flat = {k: self._t(v.reshape(T * n, *v.shape[2:])) for k, v in batch.items()}
        return flat, (np.mean(rates, axis=0) if rates else np.zeros(0))

    def update(self, batch) -> dict:
        c = self.cfg
        B = batch["obs"].shape[0]
        stats = dict(pg=[], vf=[], ent=[], kl=[], clipfrac=[])
        stop = False
        grad_norm = 0.0
        for _ in range(c.epochs):
            perm = torch.randperm(B, device=self.device)
            for s in range(0, B, c.minibatch_size):
                mb = perm[s : s + c.minibatch_size]
                logp, ent, val, _ = self.agent.evaluate(batch["obs"][mb], batch["act"][mb])
                adv = batch["adv"][mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = logp - batch["logp"][mb]
                ratio = log_ratio.exp()
                pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - c.clip, 1 + c.clip)).mean()
                vf = 0.5 * (val - batch["ret"][mb]).pow(2).mean()
                loss = pg + c.vf_coef * vf - c.ent_coef * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                # Recorte por separado: la norma del gradiente del actor SNN puede ser órdenes de
                # magnitud mayor que la del crítico y un recorte global aplastaría el del crítico.
                if c.global_grad_clip:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.all_params, c.max_grad_norm))
                else:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor_params, c.max_grad_norm))
                    torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - log_ratio).mean().item()
                    stats["pg"].append(pg.item())
                    stats["vf"].append(vf.item())
                    stats["ent"].append(ent.mean().item())
                    stats["kl"].append(kl)
                    stats["clipfrac"].append(((ratio - 1).abs() > c.clip).float().mean().item())
                if c.target_kl is not None and kl > 1.5 * c.target_kl:
                    stop = True
                    break
            if stop:
                break
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["grad_norm"] = grad_norm
        y, yp = batch["ret"], batch["val"]
        out["explained_var"] = float(1 - (y - yp).var() / (y.var() + 1e-8))
        return out

    def learn(self, total_steps: int, log_fn: Optional[Callable[[dict], None]] = None,
              iter_hook: Optional[Callable[[int], None]] = None):
        c = self.cfg
        per_iter = c.rollout_steps * self.env.num_envs
        n_iter = max(1, int(np.ceil(total_steps / per_iter)))
        base_lrs = [g["lr"] for g in self.opt.param_groups]
        t0 = time.time()
        for it in range(n_iter):
            if c.anneal_lr:
                frac = 1.0 - it / n_iter
                for g, lr in zip(self.opt.param_groups, base_lrs):
                    g["lr"] = lr * frac
            if iter_hook:  # p. ej. chip nuevo con ruido de memristor: igual en el rollout y en las épocas
                iter_hook(it)
            t_roll = time.time()
            batch, rates = self.collect()
            t_upd = time.time()
            info = self.update(batch)
            now = time.time()
            rets = [r for r, _ in self.recent]
            info.update(
                iteration=it + 1,
                steps=self.total_steps,
                ep_return=float(np.mean(rets)) if rets else float("nan"),
                ep_len=float(np.mean([l for _, l in self.recent])) if rets else float("nan"),
                std=float(self.agent.log_std.exp().mean()),
                sps=self.total_steps / (now - t0),
                t_rollout=t_upd - t_roll,
                t_update=now - t_upd,
                rates=rates.tolist(),
            )
            if log_fn:
                log_fn(info)
        return self
