
from __future__ import annotations

import torch

from dive.stackelberg.joint_update import partner_near_rows, partner_tensors

def near_partner_mask(batch: dict, *, contact_nm: float = 0.5) -> torch.Tensor:

    mask = batch["mask"].bool()
    return mask & ~far_from_partner_mask(batch, contact_nm=contact_nm)

def far_from_partner_mask(batch: dict, *, contact_nm: float = 0.5) -> torch.Tensor:

    mask = batch["mask"].bool()
    if "x_target" not in batch or "bb_ca" not in batch.get("x_t", {}):
        return mask
    partner, keep = partner_tensors(batch)
    near = partner_near_rows(
        batch["x_t"]["bb_ca"], partner, keep, contact_nm=contact_nm, valid=mask)
    return mask & ~near

def bulk_anchor_loss(live: dict, teacher: dict, far: torch.Tensor) -> torch.Tensor:

    if far is None:
        raise ValueError("bulk anchor needs a far-row mask")
    total = None
    n_terms = 0
    for key in ("bb_ca", "local_latents"):
        if key not in live or key not in teacher:
            continue
        err = (live[key] - teacher[key].detach()).pow(2).sum(dim=-1)
        weight = far.to(dtype=err.dtype)
        term = (err * weight).sum() / weight.sum().clamp_min(1.0)
        total = term if total is None else total + term
        n_terms += 1
    if total is None:
        raise ValueError("bulk anchor needs bb_ca or local_latents velocities")
    return total / n_terms
