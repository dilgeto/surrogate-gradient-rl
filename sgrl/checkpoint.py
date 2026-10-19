"""Carga de checkpoints (`best.pt` / `final.pt`) escritos por sgrl.train."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .policy import ActorCritic
from .ppo import Normalizer


def load(path: Path, device: str = "cpu"):
    """-> (agent, normalizador congelado, nombre del entorno)."""
    path = Path(path)
    ck = torch.load(path, map_location=device, weights_only=False)
    agent = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["neuron"], tuple(ck["hidden"]),
                        n_actions=ck["n_actions"], n_agents=ck["n_agents"], **ck["snn"])
    agent.load_state_dict(ck["agent"])
    agent.to(device).eval()
    # normalize_obs / clip_obs no van en el checkpoint: se leen del config.json de la corrida
    cfg_path = path.parent / "config.json"
    ppo_cfg = json.loads(cfg_path.read_text())["ppo"] if cfg_path.exists() else {}
    norm = Normalizer(ck["obs_dim"], ppo_cfg.get("clip_obs", 10.0), ppo_cfg.get("normalize_obs", True), ck["n_agents"])
    norm.rms.load(ck["norm"])
    return agent, norm, ck["env"]
