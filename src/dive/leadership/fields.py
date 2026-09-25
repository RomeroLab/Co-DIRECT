
from __future__ import annotations

import torch
from torch import Tensor

from dive.leadership.router import DirectionalGates

FIXED_CORNERS: dict[str, tuple[float, float]] = {
    "self_only": (0.0, 0.0),
    "backbone_led": (0.0, 1.0),
    "local_led": (1.0, 0.0),
    "joint": (1.0, 1.0),
}

class FieldError(RuntimeError):
    pass

def fixed_gates(name: str, batch: int, residues: int, device) -> DirectionalGates:

    if name not in FIXED_CORNERS:
        raise FieldError(f"unknown corner {name!r}; expected one of {sorted(FIXED_CORNERS)}")
    z_to_x, x_to_z = FIXED_CORNERS[name]
    shape = (batch, residues)
    return DirectionalGates(
        z_to_x=torch.full(shape, z_to_x, device=device),
        x_to_z=torch.full(shape, x_to_z, device=device),
    )

def route_fields(
    *,
    gates: DirectionalGates,
    v_x_joint: Tensor,
    v_x_self: Tensor,
    v_z_joint: Tensor,
    v_z_self: Tensor,
) -> tuple[Tensor, Tensor]:

    for name, tensor in (
        ("v_x_joint", v_x_joint),
        ("v_x_self", v_x_self),
        ("v_z_joint", v_z_joint),
        ("v_z_self", v_z_self),
    ):
        if not torch.isfinite(tensor).all():
            raise FieldError(
                f"{name} contains non-finite values; routing would carry them into "
                f"the sampler where they surface far from the cause"
            )

    expected = v_x_joint.shape[:2]
    for name, gate in (("z_to_x", gates.z_to_x), ("x_to_z", gates.x_to_z)):
        if tuple(gate.shape) != tuple(expected):
            raise FieldError(
                f"gate {name} shape {tuple(gate.shape)} does not match the "
                f"prediction's [batch, residues] {tuple(expected)}"
            )

    return (
        _combine(gates.z_to_x, joint=v_x_joint, self_only=v_x_self),
        _combine(gates.x_to_z, joint=v_z_joint, self_only=v_z_self),
    )

def _combine(gate: Tensor, *, joint: Tensor, self_only: Tensor) -> Tensor:

    g = gate.unsqueeze(-1)
    interpolated = self_only + g * (joint - self_only)
    at_joint = g == 1.0
    at_self = g == 0.0
    return torch.where(at_joint, joint, torch.where(at_self, self_only, interpolated))
