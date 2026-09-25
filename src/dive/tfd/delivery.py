
from __future__ import annotations

import enum
from typing import Mapping

import torch

class DeliveryVerdict(enum.Enum):

    DELIVERED = "delivered"

    NEGLIGIBLE = "negligible"
    NOT_DELIVERED = "not_delivered"

def _random_rotation(generator: torch.Generator | None, device, dtype) -> torch.Tensor:

    a = torch.randn(3, 3, generator=generator, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)

    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if float(torch.det(q)) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)

def perturb_masked_rigid(
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    translation_nm: float = 1.0,
) -> torch.Tensor:

    if coords.shape[-1] != 3:
        raise ValueError(f"expected a trailing axis of 3, got {tuple(coords.shape)}")
    if mask.shape != coords.shape[:2]:
        raise ValueError(f"mask {tuple(mask.shape)} does not match coords {tuple(coords.shape[:2])}")
    out = coords.clone()
    if not bool(mask.any()):
        return out
    rot = _random_rotation(generator, coords.device, coords.dtype)
    shift = torch.randn(3, generator=generator, device=coords.device, dtype=torch.float32)
    shift = (shift / shift.norm().clamp_min(1e-12) * translation_nm).to(coords.dtype)
    moved = coords @ rot.T + shift
    broadcast = mask.reshape(mask.shape + (1,) * (coords.dim() - 2))
    return torch.where(broadcast, moved, out)

def prediction_delta(
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:

    keys = set(a) | set(b)
    missing = [k for k in keys if k not in a or k not in b]
    if missing:
        raise KeyError(f"modality present in only one prediction: {sorted(missing)}")
    out: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        diff = (a[key].float() - b[key].float()).abs()
        if mask is not None:
            broadcast = mask.reshape(mask.shape + (1,) * (diff.dim() - 2))
            diff = diff * broadcast
            denom = float(broadcast.expand_as(diff).sum())
        else:
            denom = float(diff.numel())
        out[key] = {
            "max_abs": float(diff.max()) if diff.numel() else 0.0,
            "mean_abs": float(diff.sum() / denom) if denom else 0.0,
        }
    return out

def verdict_from_delta(
    delta: Mapping[str, Mapping[str, float]], *, tol: float = 0.0
) -> DeliveryVerdict:

    if not delta:
        raise ValueError("empty delta table; a probe that measured nothing has no verdict")
    largest = max(float(v["max_abs"]) for v in delta.values())
    if largest == 0.0:
        return DeliveryVerdict.NOT_DELIVERED
    if largest <= tol:
        return DeliveryVerdict.NEGLIGIBLE
    return DeliveryVerdict.DELIVERED

def perturb_mask_roll(mask: torch.Tensor, *, shift: int = 1) -> torch.Tensor:

    if shift == 0:
        raise ValueError("shift=0 is not a perturbation")
    if mask.dim() < 2:
        raise ValueError(f"expected at least [b, n], got {tuple(mask.shape)}")
    flat = mask
    while flat.dim() > 2:
        flat = flat.any(dim=-1)
    total = int(flat.numel())
    true = int(flat.sum())
    if true == 0 or true == total:
        raise ValueError(
            "mask is uniform, so rolling it changes nothing; a delivery reading "
            "from it would be uninformative"
        )
    return torch.roll(mask, shifts=int(shift), dims=1)
