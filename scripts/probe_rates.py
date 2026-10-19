"""Diagnóstico de un SNNActor sin entrenar: tasas de disparo por capa y norma del gradiente.

    python scripts/probe_rates.py --in-gain 2 --hid-gain 8 --beta 0.9 --steps 16
Sirve para elegir `gain` (escala de corriente), `beta` y `alpha` (nitidez del surrogado) de modo que
la tasa de disparo inicial no sea ~0 ni ~1 y el gradiente llegue a la primera capa.
"""
import argparse

import torch

from sgrl.snn import NEURONS, SNNActor, SURROGATES

ap = argparse.ArgumentParser()
ap.add_argument("--neuron", choices=NEURONS, default="lif")
ap.add_argument("--surrogate", choices=SURROGATES, default="fast_sigmoid")
ap.add_argument("--alpha", type=float, default=None)
ap.add_argument("--in-gain", type=float, default=None)
ap.add_argument("--hid-gain", type=float, default=None)
ap.add_argument("--beta", type=float, default=0.9)
ap.add_argument("--steps", type=int, default=16)
ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
ap.add_argument("--obs-dim", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed)
net = SNNActor(a.obs_dim, 1, a.hidden, a.neuron, a.steps, a.surrogate, a.alpha, a.beta, in_gain=a.in_gain, hid_gain=a.hid_gain)
obs = torch.randn(256, a.obs_dim)
mean, rates = net(obs)
mean.pow(2).sum().backward()
print(f"{a.neuron}/{a.surrogate} steps={a.steps}  tasa de disparo por capa: {[round(r, 3) for r in rates.tolist()]}")
print(f"  |salida| media={mean.abs().mean():.4f}  std entre muestras={mean.std():.4f}")
for i, fc in enumerate(net.fcs):
    print(f"  |grad| fc{i}: {fc.weight.grad.norm():.3e}")
print(f"  |grad| out: {net.out.weight.grad.norm():.3e}")
