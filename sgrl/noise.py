"""Ruido de memristores en las capas sinápticas del actor (puerto a PyTorch de ../wann-cpp/memristor_noise.hpp).

Cada peso w de una `NoisyLinear` se mapea a un par diferencial de conductancias,
    w = (G+ - G-) * (w_max / g_max),    G+ = max(w, 0) · g_max/w_max,  G- = max(-w, 0) · g_max/w_max,
y el ruido tiene dos partes de distinta naturaleza temporal (igual que en wann-cpp):
  1. PROGRAMACIÓN (escritura): se muestrea UNA vez por "chip" (`Chip.program`) y queda fijo.
     En PCM también se sortea ahí el coeficiente de deriva `nu` de cada dispositivo.
  2. LECTURA: ruido nuevo en cada lectura (`Chip.read(t)`), más la deriva a `t` segundos de la
     programación. Una lectura fija los pesos efectivos hasta la siguiente (un chip se lee, luego
     corre episodios), que es el protocolo de `wann_car_noise` (`--reads-per-chip`).

Modelos (fórmulas de aihwkit, MIT, vía el puerto C++; ver las citas en memristor_noise.hpp):
  simple    dos coeficientes, sin calibrar (sólo barridos genéricos de sensibilidad)
  pcm       aihwkit PCMLikeNoiseModel (Nandakumar et al. 2019): programación, deriva y ruido 1/f
  reram     aihwkit ReRamWan2022NoiseModel (Wan et al., Nature 2022): t ∈ {1 s, 1 día, 2 días}
  reramcmo  aihwkit ReRamCMONoiseModel (Falcone et al.): g_max/g_min sin valor por defecto fiable

Los sesgos (`bias`) se quedan digitales, sin ruido (como `digital_bias` de aihwkit). Sólo las capas
`NoisyLinear` del actor llevan ruido; el crítico es un MLP digital.

Entrenamiento con ruido (Joshi et al. 2020): el ruido se suma al peso en el forward y el gradiente
pasa por el peso nominal (`w + delta.detach()`). Se resortea un chip nuevo cada iteración de PPO
(no en cada forward): así la política es la misma en el rollout y en las épocas de actualización y
la razón de PPO vale 1 al empezar.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

NOISE_MODELS = ("none", "simple", "pcm", "reram", "reramcmo")


# --------------------------------------------------------------------------------------------
# Modelos de dispositivo. Trabajan sobre tensores de conductancia (una rama del par diferencial).
# `program(g_target, gen) -> estado` y `read(estado, t, gen) -> conductancia leída`.
# --------------------------------------------------------------------------------------------
def _randn(like: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(like.shape, generator=gen, device=like.device, dtype=like.dtype)


def _quantize(g: torch.Tensor, g_max: float, n_levels: int) -> torch.Tensor:
    """Asienta la conductancia ya escrita en uno de `n_levels` niveles (<=1: continua)."""
    if n_levels <= 1:
        return g
    step = g_max / (n_levels - 1)
    return (torch.floor(g / step + 0.5) * step).clamp(0.0, g_max)


def _poly_noise(g_target, coeff, g_max, g_max_ref, scale, gen):
    """Ruido de programación polinómico de aihwkit (PCM y ReRAM-Wan): g + scale·σ(u)·N(0,1), >= 0."""
    u = g_target / g_max
    mat, sig = torch.ones_like(g_target), torch.full_like(g_target, coeff[0])
    for c in coeff[1:]:
        mat = mat * u
        sig = sig + mat * c
    sig = sig * (g_max / g_max_ref)
    return (g_target + scale * sig * _randn(g_target, gen)).clamp_min(0.0)


@dataclass
class SimpleModel:
    g_max: float = 1.0
    c0: float = 0.03
    c1: float = 0.0
    c2: float = 0.0
    r0: float = 0.01
    read_exp: float = 0.5
    n_levels: int = 0
    prog_noise_scale: float = 1.0
    read_noise_scale: float = 1.0

    def program(self, g_t, gen):
        u = (g_t / self.g_max).clamp(0.0, 1.0)
        sigma = self.g_max * (self.c0 + self.c1 * u + self.c2 * u * u).clamp_min(0.0) * self.prog_noise_scale
        g = (g_t + sigma * _randn(g_t, gen)).clamp(0.0, self.g_max)
        return {"g": _quantize(g, self.g_max, self.n_levels)}

    def read(self, st, t, gen):
        g = st["g"]
        u = (g / self.g_max).clamp(0.0, 1.0)
        shape = torch.ones_like(u) if self.read_exp == 0.0 else u.pow(self.read_exp)
        sigma = self.g_max * self.r0 * self.read_noise_scale * shape
        return (g + sigma * _randn(g, gen)).clamp(0.0, self.g_max)


@dataclass
class PCMModel:
    g_max: float = 25.0  # µS
    prog_coeff: tuple = (0.26348, 1.9650, -1.1731)
    prog_coeff_g_max_reference: float = 25.0
    prog_noise_scale: float = 1.0
    read_noise_scale: float = 1.0
    drift_scale: float = 1.0
    t_read: float = 250.0e-9  # s
    t_0: float = 20.0  # s
    n_levels: int = 0

    def program(self, g_t, gen):
        g = _poly_noise(g_t, self.prog_coeff, self.g_max, self.prog_coeff_g_max_reference, self.prog_noise_scale, gen)
        # nu se sortea a partir del target original (pre-ruido), como generate_drift_coefficients
        lg = torch.log((g_t.abs() / self.g_max).clamp_min(1e-7))
        mu = (-0.0155 * lg + 0.0244).clamp(0.049, 0.1)
        sig = (-0.0125 * lg - 0.0059).clamp(0.008, 0.045)
        nu = self.drift_scale * (mu + sig * _randn(g_t, gen)).abs()
        return {"g": _quantize(g, self.g_max, self.n_levels), "nu": nu}

    def read(self, st, t, gen):
        g, nu = st["g"], st["nu"]
        tt = t + self.t_0
        g_drift = g * (tt / self.t_0) ** (-nu) if tt > self.t_0 else g
        q_s = (0.0088 / (g.abs() / self.g_max).pow(0.65).clamp_min(1e-3)).clamp(max=0.2)
        sig_noise = q_s * math.sqrt(math.log((tt + self.t_read) / (2.0 * self.t_read)))
        out = g_drift + g_drift.abs() * self.read_noise_scale * sig_noise * _randn(g, gen)
        return out.clamp_min(0.0)


@dataclass
class ReRamWanModel:
    g_max: float = 40.0  # µS
    coeff_g_max_reference: float = 40.0
    noise_scale: float = 1.0
    n_levels: int = 0
    coeff_dic: Dict[float, tuple] = field(default_factory=lambda: {
        1.0: (0.348, 16.030, -43.853, 45.393, -16.815),
        86400.0: (0.701, 22.086, -50.773, 47.095, -16.458),
        172800.0: (0.782, 20.274, -43.507, 37.062, -11.934),
    })

    def program(self, g_t, gen):
        coeff0 = self.coeff_dic[min(self.coeff_dic)]
        g = _poly_noise(g_t, coeff0, self.g_max, self.coeff_g_max_reference, self.noise_scale, gen)
        return {"g": _quantize(g, self.g_max, self.n_levels)}

    def read(self, st, t, gen):
        if t not in self.coeff_dic:
            raise ValueError(f"reram: t_inference debe ser una de {sorted(self.coeff_dic)} s (no hay interpolación), no {t}")
        return _poly_noise(st["g"], self.coeff_dic[t], self.g_max, self.coeff_g_max_reference, self.noise_scale, gen)


@dataclass
class ReRamCMOModel:
    g_max: float = 25.0  # µS; sin valor por defecto en aihwkit: ni este ni g_min son de Falcone et al.
    g_min: float = 1.0
    prog_noise_scale: float = 1.0
    read_noise_scale: float = 1.0
    drift_scale: float = 1.0
    acceptance_range: float = 2e-2
    prog_coeff: Dict[float, tuple] = field(default_factory=lambda: {
        0.2e-2: (0.00081107, 0.00106879),
        2.0e-2: (0.0112185391, 0.01129027418),
    })
    decay_mean: float = -0.08900206
    decay_std_slope: float = 0.04201137
    decay_std_intercept: float = 0.41183342
    read_K: float = 0.0277316483
    t_read: float = 1.0e-6
    n_levels: int = 0

    def program(self, g_t, gen):
        coeff = self.prog_coeff.get(self.acceptance_range, self.prog_coeff[min(self.prog_coeff)])
        mat, sig = torch.ones_like(g_t), torch.full_like(g_t, coeff[0])
        for c in coeff[1:]:  # lineal en g_target CRUDO (sin normalizar por g_max), como aihwkit
            mat = mat * g_t
            sig = sig + mat * c
        g = (g_t + self.prog_noise_scale * sig * _randn(g_t, gen)).clamp(0.0, self.g_max)
        return {"g": _quantize(g, self.g_max, self.n_levels)}

    def read(self, st, t, gen):
        g = st["g"]
        if t == 0.0:  # bypass total, igual que aihwkit
            return g
        lt = math.log(t)
        g_mean = g + self.decay_mean * lt * self.drift_scale
        sigma_relax = self.decay_std_slope * lt + self.decay_std_intercept
        g_drift = g_mean + _randn(g, gen) * sigma_relax * self.drift_scale
        sigma_read = self.read_K * torch.log10(g_drift.clamp_min(1e-6)) * math.sqrt(math.log((t + self.t_read) / (2.0 * self.t_read)))
        out = g_drift + _randn(g, gen) * sigma_read * self.read_noise_scale
        return out.clamp(self.g_min, self.g_max)


# --------------------------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------------------------
@dataclass
class NoiseConfig:
    """Equivale a las flags de `wann_car_noise`. `noise_scale` escala todos los mecanismos y
    `prog_/read_noise_scale`, `drift_scale` los sobrescriben uno a uno (ablación)."""

    model: str = "none"
    noise_scale: float = 1.0
    prog_noise_scale: Optional[float] = None
    read_noise_scale: Optional[float] = None
    drift_scale: Optional[float] = None
    n_levels: int = 0  # niveles de conductancia al programar (<=1: continua)
    w_max: Optional[float] = None  # |w| que mapea a g_max; None = max|W| de cada capa
    t_inference: Optional[float] = None  # s; None = defecto del modelo (0 pcm/simple, 1 reram/reramcmo)
    reramcmo_g_max: float = 25.0
    reramcmo_g_min: float = 1.0

    def scales(self):
        pick = lambda v: self.noise_scale if v is None else v
        return pick(self.prog_noise_scale), pick(self.read_noise_scale), pick(self.drift_scale)

    @property
    def t(self) -> float:
        if self.t_inference is not None:
            return float(self.t_inference)
        return 1.0 if self.model in ("reram", "reramcmo") else 0.0

    def active(self) -> bool:
        return self.model != "none"


def build_model(cfg: NoiseConfig):
    if cfg.model not in NOISE_MODELS:
        raise ValueError(f"modelo de ruido {cfg.model!r}; opciones: {NOISE_MODELS}")
    p, r, d = cfg.scales()
    if cfg.model == "none":
        return None
    if cfg.model == "simple":
        m = SimpleModel(prog_noise_scale=p, read_noise_scale=r, n_levels=cfg.n_levels)
    elif cfg.model == "pcm":
        m = PCMModel(prog_noise_scale=p, read_noise_scale=r, drift_scale=d, n_levels=cfg.n_levels)
    elif cfg.model == "reram":  # un solo valor para programación, deriva y lectura
        m = ReRamWanModel(noise_scale=p, n_levels=cfg.n_levels)
    else:
        m = ReRamCMOModel(g_max=cfg.reramcmo_g_max, g_min=cfg.reramcmo_g_min, prog_noise_scale=p,
                          read_noise_scale=r, drift_scale=d, n_levels=cfg.n_levels)
        if not (0.0 <= m.g_min < m.g_max):
            raise ValueError("reramcmo: hace falta 0 <= g_min < g_max")
    return m


def add_noise_args(ap, prefix: str = ""):
    """Flags de ruido; con `prefix="train-"` quedan como `--train-noise-model`, etc."""
    d = NoiseConfig()
    g = ap.add_argument_group("ruido de memristores" + (" (entrenamiento)" if prefix else ""))
    g.add_argument(f"--{prefix}noise-model", choices=NOISE_MODELS, default=d.model)
    g.add_argument(f"--{prefix}noise-scale", type=float, default=d.noise_scale,
                   help="escala de todos los mecanismos (0 = sin ruido, 1 = valores del modelo)")
    g.add_argument(f"--{prefix}prog-noise-scale", type=float, default=None)
    g.add_argument(f"--{prefix}read-noise-scale", type=float, default=None)
    g.add_argument(f"--{prefix}drift-scale", type=float, default=None)
    g.add_argument(f"--{prefix}n-levels", type=int, default=d.n_levels)
    g.add_argument(f"--{prefix}w-max", type=float, default=None, help="def. max|W| de cada capa")
    g.add_argument(f"--{prefix}reramcmo-g-max", type=float, default=d.reramcmo_g_max)
    g.add_argument(f"--{prefix}reramcmo-g-min", type=float, default=d.reramcmo_g_min)
    return g


def config_from_args(a, prefix: str = "", **overrides) -> NoiseConfig:
    p = prefix.replace("-", "_")
    kw = {f.name: getattr(a, p + f.name) for f in fields(NoiseConfig) if hasattr(a, p + f.name)}
    kw["model"] = getattr(a, p + "noise_model")
    kw["noise_scale"] = getattr(a, p + "noise_scale")
    return NoiseConfig(**{**kw, **overrides})


# --------------------------------------------------------------------------------------------
# Crossbar y capa
# --------------------------------------------------------------------------------------------
class Crossbar:
    """Par diferencial G+/G- de una matriz de pesos; `program` una vez, `read` en cada lectura."""

    def __init__(self, weight: torch.Tensor, w_max: float, model):
        self.model, self.w_max = model, float(w_max)
        g = (weight.abs() / self.w_max).clamp(0.0, 1.0) * model.g_max
        zero = torch.zeros_like(g)
        self.g_pos, self.g_neg = torch.where(weight > 0, g, zero), torch.where(weight < 0, g, zero)
        self.state = None

    def program(self, gen):
        self.state = (self.model.program(self.g_pos, gen), self.model.program(self.g_neg, gen))

    def read(self, t: float, gen) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("Crossbar.read antes de program()")
        gp, gn = (self.model.read(s, t, gen) for s in self.state)
        return (gp - gn) * (self.w_max / self.model.g_max)


class NoisyLinear(nn.Linear):
    """`nn.Linear` cuyos pesos pueden leerse con ruido de memristor (el sesgo es digital).

    Sin chip asociado se comporta exactamente como `nn.Linear`. Con un `Chip` activo usa
    `weight + delta` con `delta` constante (straight-through): el gradiente va al peso nominal.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("_delta", None, persistent=False)
        self._xbar: Optional[Crossbar] = None
        self._enabled = True

    def effective_weight(self) -> torch.Tensor:
        if self._delta is None or not self._enabled:
            return self.weight
        return self.weight + self._delta

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def program(self, cfg: NoiseConfig, model, gen):
        w = self.weight.detach()
        w_max = cfg.w_max if cfg.w_max else float(w.abs().max().clamp_min(1e-8))
        self._xbar = Crossbar(w, w_max, model)
        self._xbar.program(gen)

    @torch.no_grad()
    def read(self, t: float, gen):
        """Fija los pesos efectivos con una lectura nueva; devuelve (Σ error² normalizado, nº pesos)."""
        w = self.weight.detach()
        noisy = self._xbar.read(t, gen)
        self._delta = noisy - w
        return float(((self._delta / self._xbar.w_max) ** 2).sum()), w.numel()

    def clear_noise(self):
        self._delta, self._xbar = None, None


class Chip:
    """Un chip simulado: sus `NoisyLinear` comparten un generador (semilla reproducible).

        chip = Chip(agent.actor, NoiseConfig("pcm"), seed=0)
        chip.program()              # ruido de escritura, una vez
        rmse = chip.read(t=3600.0)  # lectura nueva a 1 h; los pesos efectivos quedan fijos
    """

    def __init__(self, module: nn.Module, cfg: NoiseConfig, seed: int = 0):
        self.layers = [m for m in module.modules() if isinstance(m, NoisyLinear)]
        if not self.layers:
            raise ValueError("el módulo no tiene capas NoisyLinear")
        self.cfg, self.model = cfg, build_model(cfg)
        self.gen = torch.Generator(device=self.layers[0].weight.device)
        self.gen.manual_seed(int(seed))

    def program(self):
        if self.model is None:
            return
        for l in self.layers:
            l.program(self.cfg, self.model, self.gen)

    def read(self, t: Optional[float] = None) -> float:
        """RMSE de los pesos realizados frente a los nominales, normalizado por w_max de cada capa."""
        if self.model is None:
            for l in self.layers:
                l.clear_noise()
            return 0.0
        t = self.cfg.t if t is None else t
        sq = n = 0
        for l in self.layers:
            s, k = l.read(t, self.gen)
            sq, n = sq + s, n + k
        return math.sqrt(sq / n)

    def resample(self, t: Optional[float] = None) -> float:
        """Chip nuevo (programación + lectura): lo que hace el entrenamiento en cada iteración."""
        self.program()
        return self.read(t)

    def detach(self):
        for l in self.layers:
            l.clear_noise()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()


@contextmanager
def noise_disabled(module: nn.Module):
    """Pesos nominales dentro del bloque (p. ej. evaluar durante el entrenamiento con ruido)."""
    layers = [m for m in module.modules() if isinstance(m, NoisyLinear)]
    prev = [l._enabled for l in layers]
    for l in layers:
        l._enabled = False
    try:
        yield
    finally:
        for l, p in zip(layers, prev):
            l._enabled = p
