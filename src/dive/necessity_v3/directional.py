
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

DEFAULT_EPS = 1e-8

def central_difference_times(anchor: float, step: float) -> tuple[float, float]:

    if not step > 0.0:
        raise ValueError(
            f"step must be strictly positive, got {step}; a zero or negative "
            "step leaves the central difference undefined"
        )
    return anchor - step, anchor + step

def _check_scale(scale: torch.Tensor) -> None:
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(
            "velocity scale must be finite and strictly positive; a zero scale "
            "would send every influence to infinity"
        )

def velocity_scale(velocity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    if velocity.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            f"expected velocity [b, n, d] and mask [b, n], got "
            f"{tuple(velocity.shape)} and {tuple(mask.shape)}"
        )
    keep = mask.to(velocity.dtype)
    counts = keep.sum(dim=-1)
    if (counts <= 0).any():
        raise ValueError("an example has no unpadded residues, so it has no scale")
    total = (velocity.square().sum(dim=-1) * keep).sum(dim=-1)
    scale = total / counts
    _check_scale(scale)
    return scale

def influence(
    *, v_less: torch.Tensor, v_more: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:

    if v_less.shape != v_more.shape:
        raise ValueError(
            f"probe velocities must match in shape, got {tuple(v_less.shape)} "
            f"and {tuple(v_more.shape)}"
        )
    _check_scale(scale)
    delta = (v_less - v_more).square().sum(dim=-1)
    return delta / scale[:, None]

def utility(
    *,
    e_more: torch.Tensor,
    e_less: torch.Tensor,
    e_anchor: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:

    for name, tensor in (("e_more", e_more), ("e_less", e_less),
                         ("e_anchor", e_anchor)):
        if (tensor < 0).any():
            raise ValueError(
                f"{name} carries a negative value; E is a sum of squares, so "
                "this indicates a caller bug rather than a real measurement"
            )
    return (e_more - e_less) / (e_anchor + eps)

@dataclass(frozen=True)
class Probe:

    role: str
    direction: str
    step: float
    dt_x: float
    dt_z: float

def probe_plan(*, steps: Sequence[float]) -> list[Probe]:

    steps = tuple(steps)
    if not steps:
        raise ValueError("at least one step size is required to form a contrast")
    for step in steps:
        if not step > 0.0:
            raise ValueError(f"step must be strictly positive, got {step}")

    plan = [Probe("anchor", "", 0.0, 0.0, 0.0)]
    for step in steps:
        more, less = central_difference_times(anchor=0.0, step=step)
        plan.append(Probe("more", "z->x", step, 0.0, more))
        plan.append(Probe("less", "z->x", step, 0.0, less))
        plan.append(Probe("more", "x->z", step, more, 0.0))
        plan.append(Probe("less", "x->z", step, less, 0.0))
    return plan

@dataclass(frozen=True)
class DirectionalEstimate:

    influence: torch.Tensor
    utility: torch.Tensor

def estimate_direction(
    *,
    v_less: torch.Tensor,
    v_more: torch.Tensor,
    v_anchor: torch.Tensor,
    e_less: torch.Tensor,
    e_more: torch.Tensor,
    e_anchor: torch.Tensor,
    mask: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> DirectionalEstimate:

    if v_less.shape[:2] != e_less.shape or v_less.shape[:2] != mask.shape:
        raise ValueError(
            f"velocity [b, n, d] {tuple(v_less.shape)}, error [b, n] "
            f"{tuple(e_less.shape)} and mask [b, n] {tuple(mask.shape)} disagree"
        )
    scale = velocity_scale(v_anchor, mask)
    c = influence(v_less=v_less, v_more=v_more, scale=scale)
    u = utility(e_more=e_more, e_less=e_less, e_anchor=e_anchor, eps=eps)
    blank = ~mask
    c = c.masked_fill(blank, float("nan"))
    u = u.masked_fill(blank, float("nan"))
    return DirectionalEstimate(influence=c, utility=u)

def sign_stable(values: Sequence[float]) -> bool:

    values = list(values)
    if len(values) < 2:
        return False
    if any(math.isnan(v) for v in values):
        return False
    if any(v == 0.0 for v in values):
        return False
    return all(v > 0 for v in values) or all(v < 0 for v in values)
