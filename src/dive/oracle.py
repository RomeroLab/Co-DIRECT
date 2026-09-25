
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

class OracleInputError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class DirectionalOracleResult:

    gate: Tensor
    value: Tensor

def directional_oracle(
    a: Tensor,
    b: Tensor,
    u: Tensor,
    valid_mask: Tensor,
    *,
    epsilon: float = 1e-8,
    g_max: float = 2.0,
) -> DirectionalOracleResult:

    _validate(a, b, u, valid_mask, epsilon, g_max)

    accumulation_dtype = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    a_acc = a.to(accumulation_dtype)
    b_acc = b.to(accumulation_dtype)
    u_acc = u.to(accumulation_dtype)

    residual = u_acc - a_acc
    b_squared = (b_acc * b_acc).sum(dim=-1)
    gate = ((b_acc * residual).sum(dim=-1) / (b_squared + epsilon)).clamp(0.0, g_max)

    before = (residual * residual).sum(dim=-1)
    after_residual = residual - gate.unsqueeze(-1) * b_acc
    value = before - (after_residual * after_residual).sum(dim=-1)

    usable = valid_mask & (b_squared > 0)
    zero = torch.zeros((), dtype=accumulation_dtype, device=gate.device)
    return DirectionalOracleResult(
        gate=torch.where(usable, gate, zero),
        value=torch.where(usable, value, zero),
    )

def delta_magnitude(b: Tensor, valid_mask: Tensor) -> Tensor:

    if not b.is_floating_point():
        raise TypeError(f"b must be a floating tensor, observed {b.dtype}")
    if valid_mask.dtype is not torch.bool:
        raise TypeError(f"valid_mask must be bool, observed {valid_mask.dtype}")
    if valid_mask.shape != b.shape[:-1]:
        raise OracleInputError(f"valid_mask shape {tuple(valid_mask.shape)} != {tuple(b.shape[:-1])}")
    if not torch.isfinite(b).all():
        raise OracleInputError("b contains non-finite values")

    accumulation_dtype = torch.float32 if b.dtype in (torch.float16, torch.bfloat16) else b.dtype
    norms = b.to(accumulation_dtype).pow(2).sum(dim=-1).sqrt()
    nan = torch.full((), float("nan"), dtype=accumulation_dtype, device=norms.device)
    return torch.where(valid_mask, norms, nan)

def _validate(
    a: Tensor, b: Tensor, u: Tensor, valid_mask: Tensor, epsilon: float, g_max: float
) -> None:

    for name, tensor in (("a", a), ("b", b), ("u", u)):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, observed {type(tensor)!r}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor, observed {tensor.dtype}")
    if valid_mask.dtype is not torch.bool:
        raise TypeError(f"valid_mask must be bool, observed {valid_mask.dtype}")

    if not (a.shape == b.shape == u.shape):
        raise OracleInputError(
            f"a/b/u shapes differ: {tuple(a.shape)}, {tuple(b.shape)}, {tuple(u.shape)}"
        )
    if not (a.dtype == b.dtype == u.dtype):
        raise OracleInputError(f"a/b/u dtypes differ: {a.dtype}, {b.dtype}, {u.dtype}")
    if not (a.device == b.device == u.device == valid_mask.device):
        raise OracleInputError("a/b/u/valid_mask are not on the same device")
    if valid_mask.shape != a.shape[:-1]:
        raise OracleInputError(
            f"valid_mask shape {tuple(valid_mask.shape)} != {tuple(a.shape[:-1])}"
        )
    for name, tensor in (("a", a), ("b", b), ("u", u)):
        if not torch.isfinite(tensor).all():
            raise OracleInputError(f"{name} contains non-finite values")
    if not epsilon > 0.0:
        raise OracleInputError(f"epsilon must be positive, observed {epsilon}")
    if not g_max > 0.0:
        raise OracleInputError(f"g_max must be positive, observed {g_max}")
