"""Adaptador de `VecEnv` (RLTools) a stable-baselines3, para validar el PPO propio contra SB3.

    from sgrl.sb3_compat import SB3VecEnv
    env = VecNormalize(SB3VecEnv("Pendulum", 16, seed=0))
    PPO("MlpPolicy", env).learn(300_000)

Sólo tareas de un agente: en las multiagente el actor propio comparte pesos entre agentes y SB3
trataría el vector conjunto como un único agente, que ya no es la misma política.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3BaseVecEnv

from .envs import VecEnv


class SB3VecEnv(SB3BaseVecEnv):
    def __init__(self, name: str, num_envs: int, seed: int = 0, num_threads: int = 1, max_steps: int = 0):
        self.env = VecEnv(name, num_envs, seed=seed, num_threads=num_threads, max_steps=max_steps)
        if self.env.n_agents > 1:
            raise ValueError(f"{name} es multiagente; la comparación con SB3 sólo vale para tareas de un agente")
        obs_space = spaces.Box(-np.inf, np.inf, (self.env.obs_dim,), np.float32)
        act_space = (spaces.Discrete(self.env.n_actions) if self.env.n_actions
                     else spaces.Box(-1.0, 1.0, (self.env.act_dim,), np.float32))
        super().__init__(num_envs, obs_space, act_space)
        self._seed, self._actions = seed, None
        self.reset_infos = [{} for _ in range(num_envs)]

    def reset(self):
        seed = self._seeds[0]  # SB3 llama a seed() antes de reset(); RLTools siembra todo el lote con una semilla
        self._reset_seeds()
        return self.env.reset(self._seed if seed is None else seed).astype(np.float32)

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        a = self._actions.astype(np.float64)
        obs, rew, term, trunc, final_obs, finished = self.env.step(a.reshape(self.num_envs, -1) if not self.env.n_actions else a)
        infos = [{} for _ in range(self.num_envs)]
        ends = {i: k for k, i in enumerate(np.flatnonzero(term | trunc))}
        for i in ends:
            infos[i]["terminal_observation"] = final_obs[i].astype(np.float32)
            infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
        for (ret, length), i in zip(finished, ends):  # `finished` sigue el orden de índices
            infos[i]["episode"] = {"r": ret, "l": length}
        return obs.astype(np.float32), rew.astype(np.float32), term | trunc, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
