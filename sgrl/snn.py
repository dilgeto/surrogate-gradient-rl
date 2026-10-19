"""Actor espiguiforme con gradientes surrogados, construido sobre snnTorch.

La política es *feed-forward en el tiempo del entorno*: en cada paso del entorno la observación se
convierte en corriente (codificación directa, una capa lineal analógica), se simulan `steps` pasos
de la dinámica neuronal con estado inicial en reposo y la acción sale de un integrador leaky (LI)
no espiguiforme que lee la última capa. El gradiente atraviesa los `steps` pasos (BPTT) y la
función escalón de cada espiga se sustituye en el backward por un gradiente surrogado de snnTorch.

Neurona: `snn.Leaky` (LIF de un estado, reset a cero). Todas las capas lineales son `NoisyLinear`
(sgrl/noise.py), de modo que se puede inyectar ruido de memristor sin tocar el actor.
"""
from __future__ import annotations

from typing import Optional, Sequence

import snntorch as snn
import torch
from snntorch import surrogate as _sg
from torch import nn

from .noise import NoisyLinear

NEURONS = ("lif",)
# (se omite `snntorch.surrogate.sigmoid`: su backward da NaN cuando exp(-slope·x) desborda con x muy negativo)
SURROGATES = ("fast_sigmoid", "atan", "triangular", "straight_through")

# Nitidez del surrogado: `slope` en fast_sigmoid (el derivado de SuperSpike), `alpha` en atan;
# triangular y straight-through no la usan.
DEFAULT_ALPHA = {"fast_sigmoid": 25.0, "atan": 2.0, "triangular": None, "straight_through": None}
# Constante de membrana de las capas ocultas y de la lectura: β = exp(-dt/τ) con dt = 1 ms.
DEFAULT_BETA, READOUT_BETA = 0.9, 0.8
# Escala de la corriente que reciben las neuronas, para que a la inicialización disparen a una
# tasa razonable (ver scripts/probe_rates.py): (capa de entrada, analógica; capas alimentadas por
# espigas). La corriente se multiplica además por (1-β) para que el potencial de reposo sea la corriente.
DEFAULT_GAIN = (10.0, 40.0)


def surrogate_fn(name: str, alpha: Optional[float] = None):
    if name not in SURROGATES:
        raise ValueError(f"surrogado {name!r}; opciones: {SURROGATES}")
    alpha = DEFAULT_ALPHA[name] if alpha is None else alpha
    if name == "fast_sigmoid":
        return _sg.fast_sigmoid(slope=alpha)
    if name == "atan":
        return _sg.atan(alpha=alpha)
    if name == "triangular":
        return _sg.triangular()
    return _sg.straight_through_estimator()


class SpikingPopulation(nn.Module):
    """Población de `size` neuronas LIF sin sinapsis propias: recibe corriente, devuelve espigas."""

    def __init__(self, size: int, surrogate: str = "fast_sigmoid", alpha: Optional[float] = None,
                 beta: float = DEFAULT_BETA):
        super().__init__()
        self.size, self.beta = size, beta
        self.cell = snn.Leaky(beta=beta, spike_grad=surrogate_fn(surrogate, alpha), reset_mechanism="zero")

    @staticmethod
    def initial_state(current: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(current)

    def forward(self, current: torch.Tensor, mem: torch.Tensor):
        return self.cell(current, mem)


class SNNActor(nn.Module):
    """obs -> (media de la acción, tasas de disparo por capa). La desviación va aparte (PPO)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (64, 64),
        neuron: str = "lif",
        steps: int = 16,
        surrogate: str = "fast_sigmoid",
        alpha: Optional[float] = None,
        beta: float = DEFAULT_BETA,
        in_gain: Optional[float] = None,
        hid_gain: Optional[float] = None,
    ):
        super().__init__()
        if neuron not in NEURONS:
            raise ValueError(f"neurona {neuron!r}; opciones: {NEURONS}")
        self.steps, self.neuron = steps, neuron
        d_in, d_hid = DEFAULT_GAIN
        in_gain = d_in if in_gain is None else in_gain
        hid_gain = d_hid if hid_gain is None else hid_gain
        sizes = [obs_dim, *hidden]
        self.fcs = nn.ModuleList(NoisyLinear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.pops = nn.ModuleList(SpikingPopulation(h, surrogate, alpha, beta) for h in hidden)
        self.out = NoisyLinear(hidden[-1], act_dim)
        # Lectura: integrador leaky sin espigas (potencial de membrana, sin reset), promediado en el tiempo.
        self.readout = snn.Leaky(beta=READOUT_BETA, threshold=1e6, reset_mechanism="none", surrogate_disable=True)
        self.in_gain, self.hid_gain = float(in_gain) * (1 - beta), float(hid_gain) * (1 - beta)
        self.out_gain = 1 - READOUT_BETA
        with torch.no_grad():
            self.out.weight.mul_(0.1)  # acciones iniciales cerca de 0, como en el PPO de RLTools
            self.out.bias.zero_()

    def forward(self, obs: torch.Tensor):
        # Pesos efectivos una sola vez por llamada (con ruido de memristor, una lectura por forward).
        ws = [fc.effective_weight() for fc in self.fcs]
        w_out = self.out.effective_weight()
        # Codificación directa: la misma corriente analógica se inyecta en cada paso temporal.
        drive = torch.nn.functional.linear(obs, ws[0], self.fcs[0].bias) * self.in_gain
        mems = [None] * len(self.pops)
        ro_mem = torch.zeros(obs.shape[0], self.out.out_features, device=obs.device, dtype=obs.dtype)
        ro_sum = torch.zeros_like(ro_mem)
        spike_counts = [torch.zeros((), device=obs.device, dtype=obs.dtype) for _ in self.pops]
        for _ in range(self.steps):
            x = drive
            for k, pop in enumerate(self.pops):
                if mems[k] is None:
                    mems[k] = pop.initial_state(x)
                z, mems[k] = pop(x, mems[k])
                spike_counts[k] = spike_counts[k] + z.detach().mean()
                if k + 1 < len(self.pops):
                    x = torch.nn.functional.linear(z, ws[k + 1], self.fcs[k + 1].bias) * self.hid_gain
                else:
                    x = z
            _, ro_mem = self.readout(torch.nn.functional.linear(x, w_out, self.out.bias) * self.out_gain, ro_mem)
            ro_sum = ro_sum + ro_mem
        rates = torch.stack(spike_counts) / self.steps
        return ro_sum / self.steps, rates
