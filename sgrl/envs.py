"""Fábrica de entornos vectorizados (RLTools vía _rlt_envs) con seguimiento de episodios.

Tareas del proyecto (`TASKS`): Acrobot, MountainCarDiscrete, CarTrack (coche de carreras),
Bottleneck (5 agentes) y BipedalWalker. Además quedan Pendulum, AcrobotSwingup,
MountainCarContinuous y BipedalWalkerHardcore.
"""
from __future__ import annotations

import numpy as np

try:
    from . import _rlt_envs
except ImportError as e:  # pragma: no cover
    raise ImportError("Falta compilar el módulo de entornos: ejecuta scripts/build_envs.sh") from e

TASKS = ("Acrobot", "MountainCarDiscrete", "CarTrack", "Bottleneck", "BipedalWalker")
ENV_NAMES = (*TASKS, "Pendulum", "AcrobotSwingup", "MountainCarContinuous", "BipedalWalkerHardcore")

# Mountain Car en RLTools es sólo continuo (fuerza en [-1, 1]). La versión discreta es un
# adaptador: acción k in {0, 1, 2} -> fuerza k-1 (izquierda, nada, derecha) sobre la misma
# dinámica y la misma recompensa (-0.1·fuerza² por paso, +100 en la meta).
_DISCRETE = {"MountainCarDiscrete": ("MountainCarContinuous", 3)}


class VecEnv:
    """Envoltorio de `_rlt_envs.VecEnv`: acciones, retorno y longitud de episodio.

    Atributos útiles:
      obs_dim, act_dim  dimensiones del vector (conjunto) que ve la política
      n_actions         nº de acciones si la tarea es discreta, si no 0 (act_dim entonces es 1)
      n_agents          >1 en tareas multiagente: obs/acciones son n_agents bloques iguales y
                        la política comparte parámetros entre agentes

    `step` recibe (N, act_dim) (continuo: se recorta a [-1, 1]; discreto: índices de acción) y
    devuelve (obs, reward, terminated, truncated, final_obs, finished), con `finished` una lista
    de (retorno, longitud) de los episodios que acaban de cerrarse.
    """

    def __init__(self, name: str, num_envs: int, seed: int = 0, num_threads: int = 1, max_steps: int = 0):
        if name not in ENV_NAMES:
            raise ValueError(f"entorno {name!r}; opciones: {ENV_NAMES}")
        self.name, self.num_envs = name, num_envs
        base, self.n_actions = _DISCRETE.get(name, (name, 0))
        self._env = _rlt_envs.make(base, num_envs, seed, num_threads, max_steps)
        self.obs_dim, self.act_dim = self._env.obs_dim, self._env.act_dim
        self.n_agents = self._env.n_agents
        self.episode_step_limit = self._env.episode_step_limit
        self._ret = np.zeros(num_envs)
        self._len = np.zeros(num_envs, dtype=np.int64)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._ret[:] = 0
        self._len[:] = 0
        return self._env.reset(seed)

    def step(self, actions: np.ndarray):
        if self.n_actions:
            actions = np.asarray(actions).reshape(self.num_envs, 1) - 1.0  # {0,1,2} -> {-1,0,1}
        obs, rew, term, trunc, final_obs = self._env.step(np.clip(actions, -1.0, 1.0))
        self._ret += rew
        self._len += 1
        done = term | trunc
        finished = [(float(self._ret[i]), int(self._len[i])) for i in np.flatnonzero(done)]
        self._ret[done] = 0
        self._len[done] = 0
        return obs, rew, term, trunc, final_obs, finished
