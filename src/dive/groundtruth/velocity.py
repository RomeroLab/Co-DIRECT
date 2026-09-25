
from __future__ import annotations

import torch
from torch import Tensor

SUPPORTED_CENTERING: tuple[str, ...] = ("none",)
_KNOWN_CENTERING: tuple[str, ...] = ("none", "x_0", "x_1", "x_t")

class VelocityError(RuntimeError):
    pass

def target_velocity(x_0: Tensor, x_1: Tensor, *, centering_mode: str) -> Tensor:

    if centering_mode not in _KNOWN_CENTERING:
        raise VelocityError(
            f"unknown stochastic_centering_mode {centering_mode!r}; "
            f"expected one of {_KNOWN_CENTERING}"
        )
    if centering_mode not in SUPPORTED_CENTERING:
        raise VelocityError(
            f"stochastic_centering_mode {centering_mode!r} perturbs the interpolant, "
            f"so x_1 - x_0 is not the target; this modality is uncalibratable "
            f"until the perturbed target is derived and tested"
        )
    if x_0.shape != x_1.shape:
        raise VelocityError(
            f"shape mismatch: {tuple(x_0.shape)} != {tuple(x_1.shape)}"
        )
    if not (torch.isfinite(x_0).all() and torch.isfinite(x_1).all()):
        raise VelocityError("non-finite value in x_0 or x_1")
    return x_1 - x_0
