"""Actor-crítico para PPO: actor SNN (o MLP de referencia) y crítico MLP analógico."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .noise import NoisyLinear
from .snn import SNNActor


def mlp(sizes: Sequence[int], act=nn.Tanh, linear=nn.Linear) -> nn.Sequential:
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(linear(a, b))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def _orthogonal(net: nn.Sequential, out_gain: float):
    lins = [m for m in net if isinstance(m, nn.Linear)]
    for m in lins:
        nn.init.orthogonal_(m.weight, 2**0.5)
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(lins[-1].weight, out_gain)


class ANNActor(nn.Module):
    """Línea base no espiguiforme con la misma interfaz que SNNActor (y las mismas capas `NoisyLinear`)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim], linear=NoisyLinear)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, obs: torch.Tensor):
        return self.net(obs), torch.zeros(0)


class ActorCritic(nn.Module):
    """Política gaussiana (acciones continuas) o categórica (`n_actions` > 0) + crítico analógico.

    Multiagente (`n_agents` > 1, p. ej. Bottleneck): el actor se aplica a cada agente por separado
    con los mismos pesos (obs por agente -> acción por agente, el vector conjunto es la
    concatenación) y el crítico es centralizado: ve la observación conjunta y estima el valor del
    equipo, que es lo que recompensa el entorno.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        neuron: str = "lif",
        hidden: Sequence[int] = (64, 64),
        critic_hidden: Sequence[int] = (64, 64),
        init_std: float = 0.5,
        n_actions: int = 0,
        n_agents: int = 1,
        ortho_init: bool = False,
        **snn_kwargs,
    ):
        super().__init__()
        if obs_dim % n_agents or act_dim % n_agents:
            raise ValueError("obs_dim y act_dim deben ser múltiplos de n_agents")
        self.n_actions, self.n_agents, self.act_dim = n_actions, n_agents, act_dim
        self.agent_obs, self.agent_act = obs_dim // n_agents, act_dim // n_agents
        out_dim = n_actions or self.agent_act
        if neuron == "ann":
            self.actor = ANNActor(self.agent_obs, out_dim, hidden)
        else:
            self.actor = SNNActor(self.agent_obs, out_dim, hidden, neuron, **snn_kwargs)
        self.critic = mlp([obs_dim, *critic_hidden, 1])
        if ortho_init:  # como SB3: ortogonal √2 en ocultas, 0.01 en la salida del actor y 1 en la del crítico
            if neuron == "ann":
                _orthogonal(self.actor.net, 0.01)
            _orthogonal(self.critic, 1.0)
        # desviación por dimensión de acción del agente, compartida entre agentes y estados
        self.log_std = nn.Parameter(torch.full((self.agent_act,), float(torch.log(torch.tensor(init_std)))))

    def dist(self, obs: torch.Tensor):
        B = obs.shape[0]
        out, rates = self.actor(obs.reshape(B * self.n_agents, self.agent_obs))
        if self.n_actions:
            return torch.distributions.Categorical(logits=out), rates
        mean = out.reshape(B, self.act_dim)
        return torch.distributions.Normal(mean, self.log_std.exp().repeat(self.n_agents)), rates

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def _log_prob(self, d, actions: torch.Tensor):
        """Acciones siempre como (B, act_dim) float; en discreto act_dim=1 y es el índice."""
        if self.n_actions:
            return d.log_prob(actions.squeeze(-1).long()), d.entropy()
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        d, rates = self.dist(obs)
        logp, ent = self._log_prob(d, actions)
        return logp, ent, self.value(obs), rates

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d, rates = self.dist(obs)
        if self.n_actions:
            a = d.probs.argmax(-1) if deterministic else d.sample()
            a = a.float().unsqueeze(-1)
        else:
            a = d.mean if deterministic else d.sample()
        logp, _ = self._log_prob(d, a)
        return a, logp, self.value(obs), rates
