"""Cuantización post-entrenamiento (PTQ) de los pesos del actor, por redondeo al nivel más cercano.

Esquemas (uniformes y por capa; el paso es `Δ = max|W| / q`, con el mismo `w_max` por capa que usa el mapeo a
conductancias de sgrl/noise.py, y cada peso pasa a `Δ · round(w / Δ)`):
  odin       (def.) sinapsis de ODIN: `bits` es la anchura de la MAGNITUD (3 -> 0..7) y el signo va aparte,
             q = 2^bits - 1, o sea 2q+1 = 15 valores con 3 bits. En ODIN el signo es de la neurona
             presináptica (ley de Dale), no de la sinapsis: una neurona con salidas de signo mixto se
             duplica en una gemela excitadora y otra inhibidora (ver wann-cpp/ODIN_DEPLOYMENT.md), de modo
             que cada peso conserva su signo y sólo se cuantiza la magnitud. `quantize_actor` cuenta
             esas gemelas (`dale_twins`).
  symmetric  `bits` bits CON signo: q = 2^(bits-1) - 1 (con 2 bits queda ternario {-Δ, 0, +Δ}).

Sólo se tocan las capas sinápticas `NoisyLinear` del actor. Los sesgos son digitales (como en el modelo
de ruido) y se dejan en coma flotante salvo `include_bias=True`. El crítico no se despliega y no se toca.
La cuantización es "falsa": los pesos siguen guardados como float32 pero sólo toman valores de la rejilla,
así que el mismo `ActorCritic` los evalúa sin cambios; los códigos enteros salen de `w / scale`.
"""
from __future__ import annotations

import torch

from .noise import NoisyLinear


SCHEMES = ("odin", "symmetric")


def n_levels_per_side(bits: int, scheme: str = "odin") -> int:
    """Nivel máximo `q` de la rejilla {-q..q}·Δ."""
    if scheme == "odin":
        if bits < 1:
            raise ValueError("hace falta >= 1 bit de magnitud")
        return 2**bits - 1
    if scheme == "symmetric":
        if bits < 2:
            raise ValueError("hacen falta >= 2 bits (1 bit con signo sólo deja el cero)")
        return 2 ** (bits - 1) - 1
    raise ValueError(f"esquema {scheme!r}; opciones: {SCHEMES}")


def quantize_tensor(w: torch.Tensor, bits: int, scheme: str = "odin") -> tuple[torch.Tensor, float]:
    """-> (pesos redondeados a la rejilla, paso Δ). Un tensor todo ceros se devuelve igual (Δ = 0)."""
    q = n_levels_per_side(bits, scheme)
    w_max = float(w.abs().max())
    if w_max == 0.0:
        return w.clone(), 0.0
    step = w_max / q
    return torch.round(w / step).clamp(-q, q) * step, step


@torch.no_grad()
def quantize_actor(actor: torch.nn.Module, bits: int, include_bias: bool = False, scheme: str = "odin") -> dict:
    """Cuantiza *in place* las `NoisyLinear` de `actor`. -> info por tensor (paso, w_max, error cuadrático).

    Si `actor` es una copia recién cargada del disco el checkpoint original no se ve afectado.
    """
    info = {}
    for name, m in actor.named_modules():
        if not isinstance(m, NoisyLinear):
            continue
        tensors = [("weight", m.weight)] + ([("bias", m.bias)] if include_bias else [])
        for pname, p in tensors:
            wq, step = quantize_tensor(p.data, bits, scheme)
            w_max = step * n_levels_per_side(bits, scheme)
            info[f"{name}.{pname}"] = {"step": step, "w_max": w_max, "numel": p.numel(),
                                       "sq_err": float((wq - p.data).pow(2).sum()),  # sin normalizar
                                       "n_unique": int(torch.unique(wq).numel())}
            if pname == "weight":  # columnas = neuronas presinápticas; signo mixto = hace falta gemela de Dale
                info[f"{name}.weight"]["dale_twins"] = int(((wq > 0).any(0) & (wq < 0).any(0)).sum())
            p.data.copy_(wq)
    if not info:
        raise ValueError("el actor no tiene capas NoisyLinear")
    return info


def weight_rmse(info: dict) -> float:
    """Error RMS de los *pesos* (no sesgos) frente a los nominales, normalizado por el w_max de cada capa:
    la misma definición que `weight_rmse` de scripts/noise_eval.py, para poder compararlos en un mismo eje."""
    sq = n = 0.0
    for k, v in info.items():
        if k.endswith(".weight") and v["w_max"] > 0:
            sq, n = sq + v["sq_err"] / v["w_max"] ** 2, n + v["numel"]
    return (sq / n) ** 0.5
