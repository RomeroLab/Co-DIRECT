
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from dive.leadership.router import DirectionalGates

RANGE_TOLERANCE = 1e-9

def _tolerance(dtype: torch.dtype) -> float:

    try:
        eps = float(torch.finfo(dtype).eps)
    except TypeError:
        eps = 0.0
    return max(RANGE_TOLERANCE, 4.0 * eps)

class AsymmetryError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ShuffleReport:

    groups: int
    permuted_residues: int
    clamped_residues: int
    max_abs_clamp: float

def _validate(gates: DirectionalGates) -> None:
    z, x = gates.z_to_x, gates.x_to_z
    if tuple(z.shape) != tuple(x.shape):
        raise AsymmetryError(
            f"gate shape mismatch: z_to_x {tuple(z.shape)} != x_to_z {tuple(x.shape)}"
        )
    for name, tensor in (("z_to_x", z), ("x_to_z", x)):
        if not torch.isfinite(tensor).all():
            raise AsymmetryError(f"gate {name} contains non-finite values")
        if bool((tensor < 0).any() or (tensor > 1).any()):
            raise AsymmetryError(
                f"gate {name} leaves [0, 1]; the routed field is only an "
                f"interpolation between the self-only and joint predictions "
                f"inside that interval"
            )

def decompose_gates(gates: DirectionalGates) -> tuple[Tensor, Tensor]:

    _validate(gates)
    m = (gates.z_to_x + gates.x_to_z) / 2
    d = (gates.z_to_x - gates.x_to_z) / 2
    return m, d

def recompose_gates(
    m: Tensor, d: Tensor, *, tolerance: float | None = None
) -> DirectionalGates:

    if tuple(m.shape) != tuple(d.shape):
        raise AsymmetryError(
            f"shape mismatch: m {tuple(m.shape)} != d {tuple(d.shape)}"
        )
    if not (torch.isfinite(m).all() and torch.isfinite(d).all()):
        raise AsymmetryError("m or d contains non-finite values")
    if bool((m < 0).any() or (m > 1).any()):
        raise AsymmetryError("mean gate m leaves [0, 1]")

    if tolerance is None:
        tolerance = _tolerance(d.dtype)
    headroom = torch.minimum(m, 1 - m)
    excess = d.abs() - headroom
    if bool((excess > tolerance).any()):
        worst = float(excess.max())
        raise AsymmetryError(
            f"asymmetry d is outside |d| <= min(m, 1-m) by up to {worst:.6g}; "
            f"at m = 1 both gates are already joint and no reallocation exists"
        )
    return DirectionalGates(
        z_to_x=(m + d).clamp(0.0, 1.0),
        x_to_z=(m - d).clamp(0.0, 1.0),
    )

def equal_gate(gates: DirectionalGates) -> DirectionalGates:

    m, _ = decompose_gates(gates)
    return DirectionalGates(z_to_x=m.clone(), x_to_z=m.clone())

def reversed_asymmetry(gates: DirectionalGates) -> DirectionalGates:

    _validate(gates)
    return DirectionalGates(z_to_x=gates.x_to_z.clone(), x_to_z=gates.z_to_x.clone())

def response_magnitudes(
    delta_z_to_x: Tensor,
    delta_x_to_z: Tensor,
    *,
    sigma_x: float,
    sigma_z: float,
) -> tuple[Tensor, Tensor]:

    for name, sigma in (("sigma_x", sigma_x), ("sigma_z", sigma_z)):
        if not (float(sigma) > 0) or not torch.isfinite(torch.tensor(float(sigma))):
            raise AsymmetryError(f"receiver scale {name} must be finite and positive, got {sigma!r}")
    for name, tensor in (("delta_z_to_x", delta_z_to_x), ("delta_x_to_z", delta_x_to_z)):
        if tensor.dim() < 3:
            raise AsymmetryError(
                f"{name} must be [batch, residues, features], got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise AsymmetryError(f"{name} contains non-finite values")
    a = torch.linalg.vector_norm(delta_z_to_x, dim=-1) / float(sigma_x)
    b = torch.linalg.vector_norm(delta_x_to_z, dim=-1) / float(sigma_z)
    return a, b

def magnitude_matched_equal_gate(
    gates: DirectionalGates, *, a: Tensor, b: Tensor
) -> DirectionalGates:

    _validate(gates)
    for name, tensor in (("a", a), ("b", b)):
        if tuple(tensor.shape) != tuple(gates.z_to_x.shape):
            raise AsymmetryError(
                f"{name} shape {tuple(tensor.shape)} does not match the gates "
                f"{tuple(gates.z_to_x.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise AsymmetryError(f"{name} contains non-finite values")
        if bool((tensor < 0).any()):
            raise AsymmetryError(f"{name} is a norm and cannot be negative")

    a2, b2 = a**2, b**2
    denominator = a2 + b2
    numerator = gates.z_to_x**2 * a2 + gates.x_to_z**2 * b2
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    q = torch.sqrt((numerator / safe).clamp_min(0.0))
    q = torch.where(denominator > 0, q, torch.zeros_like(q)).clamp(0.0, 1.0)
    return DirectionalGates(z_to_x=q, x_to_z=q.clone())

def shuffled_asymmetry(
    gates: DirectionalGates,
    *,
    groups: Tensor,
    generator: torch.Generator,
    force_permutation: bool = False,
) -> tuple[DirectionalGates, ShuffleReport]:

    m, d = decompose_gates(gates)
    if tuple(groups.shape) != tuple(m.shape):
        raise AsymmetryError(
            f"groups shape {tuple(groups.shape)} does not match the gates "
            f"{tuple(m.shape)}"
        )

    shuffled = d.clone()
    group_count = 0
    permuted = 0

    device = d.device
    labels_source = groups.cpu()
    for batch_index in range(m.shape[0]):
        row = labels_source[batch_index]
        for label in torch.unique(row):
            positions = torch.nonzero(row == label, as_tuple=False).flatten()
            group_count += 1
            size = int(positions.numel())
            if size < 2:
                continue
            order = _permutation(size, generator=generator, force=force_permutation)
            source = positions[order].to(device)
            destination = positions.to(device)
            shuffled[batch_index, destination] = d[batch_index, source]
            permuted += size

    headroom = torch.minimum(m, 1 - m)
    clamped = shuffled.clamp(-headroom, headroom)
    difference = (shuffled - clamped).abs()
    report = ShuffleReport(
        groups=group_count,
        permuted_residues=permuted,
        clamped_residues=int((difference > _tolerance(d.dtype)).sum()),
        max_abs_clamp=float(difference.max()) if difference.numel() else 0.0,
    )
    return recompose_gates(m, clamped), report

def _permutation(size: int, *, generator: torch.Generator, force: bool) -> Tensor:

    identity = torch.arange(size)
    for _ in range(16):
        order = torch.randperm(size, generator=generator)
        if not force or bool((order != identity).any()):
            return order
    return torch.roll(identity, 1)
