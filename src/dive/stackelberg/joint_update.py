
from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_CLIP = 0.5

def partner_tensors(batch: dict) -> tuple[Tensor, Tensor]:

    if "x_target" not in batch:
        raise ValueError("no x_target: there is no partner to snap onto")
    x = batch["x_target"]
    m = batch.get("target_mask")
    if x.dim() == 4:
        b = x.shape[0]
        flat = x.reshape(b, -1, 3)
        keep = (m.reshape(b, -1).bool() if m is not None
                else torch.ones(flat.shape[:2], dtype=torch.bool, device=x.device))
        return flat, keep
    if x.dim() == 3:
        keep = (m.bool() if m is not None
                else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device))
        return x, keep
    raise ValueError(f"x_target has shape {tuple(x.shape)}")

def partner_near_rows(ca: Tensor, partner: Tensor, keep: Tensor, *,
                      contact_nm: float = 0.5, valid: Tensor | None = None
                      ) -> Tensor:

    dist = torch.cdist(ca, partner)
    dist = dist.masked_fill(~keep[:, None, :], 1e6)
    near = dist.min(dim=-1).values <= float(contact_nm)
    if valid is not None:
        near = near & valid.bool()
    return near

def intervention_rows(designed, mask):

    if designed is None or mask is None:
        raise ValueError(
            "snap anticipation needs designed and mask; without both, fixed "
            "rows and padding enter the probe, the response loss, and the "
            "velocity replacement")
    active = designed.bool()
    valid = mask.bool()
    while active.dim() > 2:
        active = active.any(dim=-1)
    while valid.dim() > 2:
        valid = valid.any(dim=-1)
    if active.shape != valid.shape:
        raise ValueError(
            f"designed shape {tuple(active.shape)} != mask {tuple(valid.shape)}")
    return active & valid

def snap_anticipation_probe(x_t, velocity, t, weights, partner, keep, *,
                           dt, contact_nm: float = 0.5, valid=None,
                           row_mask=None):

    w_ca = weights["bb_ca"]
    w_lat = weights["local_latents"]
    if w_ca.shape != w_lat.shape:
        raise ValueError(
            f"follower scores differ {tuple(w_ca.shape)} vs {tuple(w_lat.shape)}")
    lat_follows = w_lat >= w_ca
    if row_mask is None:
        active = torch.ones_like(lat_follows)
    else:
        active = row_mask.bool()
        if active.shape != lat_follows.shape:
            raise ValueError(
                f"row_mask shape {tuple(active.shape)} != {tuple(lat_follows.shape)}")
    lat_follows = lat_follows & active
    ca_follows = active & ~lat_follows
    backbone_leads = lat_follows
    latent_leads = ca_follows

    snapped, near = snap_ca_to_partner(
        x_t["bb_ca"], partner, keep, contact_nm=contact_nm, valid=valid)
    lead = backbone_leads
    while lead.dim() < snapped.dim():
        lead = lead[..., None]
    probe_ca = torch.where(lead, snapped, x_t["bb_ca"])

    zt = x_t["local_latents"]
    clean = anticipated_clean_state(
        zt, velocity["local_latents"], t["local_latents"].to(dtype=zt.dtype),
        max_step_nm=PRIOR_SCALE_NM)
    tt = t["local_latents"].to(dtype=zt.dtype)
    while tt.dim() < zt.dim():
        tt = tt.unsqueeze(-1)
    if dt is None or float(dt) <= 0.0:
        frac = torch.zeros((), device=zt.device, dtype=zt.dtype)
    else:
        frac = (float(dt) / (1.0 - tt).clamp_min(1e-6)).clamp(max=1.0)
    stepped = zt + frac * (clean - zt)
    mask = latent_leads
    while mask.dim() < zt.dim():
        mask = mask[..., None]
    probe_z = torch.where(mask, stepped, zt)
    return ({"bb_ca": probe_ca, "local_latents": probe_z},
            {"bb_ca": ca_follows, "local_latents": lat_follows})

def snap_ca_to_partner(ca: Tensor, partner: Tensor, keep: Tensor, *,
                       contact_nm: float = 0.5, valid: Tensor | None = None
                       ) -> tuple[Tensor, Tensor]:

    dist = torch.cdist(ca, partner)
    dist = dist.masked_fill(~keep[:, None, :], 1e6)
    nearest = dist.argmin(dim=-1)
    snapped = torch.gather(partner, 1, nearest.unsqueeze(-1).expand(-1, -1, 3))
    near = dist.min(dim=-1).values <= float(contact_nm)
    if valid is not None:
        near = near & valid.bool()
    probe = torch.where(near.unsqueeze(-1), snapped, ca)
    return probe, near

def rigid_contact_shift(ca: Tensor, partner: Tensor, keep: Tensor, *,
                        contact_nm: float = 0.5, valid: Tensor | None = None
                        ) -> tuple[Tensor, Tensor]:

    dist = torch.cdist(ca, partner)
    dist = dist.masked_fill(~keep[:, None, :], 1e6)
    nearest = dist.argmin(dim=-1)
    snapped = torch.gather(partner, 1, nearest.unsqueeze(-1).expand(-1, -1, 3))
    near = dist.min(dim=-1).values <= float(contact_nm)
    if valid is not None:
        near = near & valid.bool()
    weight = near.to(dtype=ca.dtype)
    count = weight.sum(dim=-1).clamp_min(1.0).unsqueeze(-1)
    centroid = (ca * weight.unsqueeze(-1)).sum(dim=1) / count
    target = (snapped * weight.unsqueeze(-1)).sum(dim=1) / count
    shift = (target - centroid).unsqueeze(1)
    moved = ca + shift
    probe = torch.where(near.unsqueeze(-1), moved, ca)
    return probe, near

def anticipated_partner_state(x_t: Tensor, velocity: Tensor, dt: float) -> Tensor:

    return x_t + dt * velocity

def routed_extra_pass_velocity(released: Tensor, responded: Tensor,
                               follower_weight: Tensor, *,
                               clip: float | None = DEFAULT_CLIP) -> Tensor:

    unit = blend(released, responded, 1.0, clip=clip)
    weight = follower_weight.to(dtype=unit.dtype, device=unit.device)
    while weight.dim() < unit.dim():
        weight = weight[..., None]
    return released + weight * (unit - released)

def blend(released: Tensor, responded: Tensor, alpha: float,
          clip: float | None = DEFAULT_CLIP) -> Tensor:

    if alpha == 0.0:
        return released
    delta = responded - released
    if clip is not None:
        limit = clip * released.norm(dim=-1, keepdim=True)
        norm = delta.norm(dim=-1, keepdim=True)
        delta = torch.where(norm > limit,
                            delta * (limit / norm.clamp_min(1e-12)),
                            delta)
    return released + alpha * delta

PRIOR_SCALE_NM = 3 ** 0.5

def residue_commitment(x_t, velocity, t, weights, dt, *, row_mask=None):

    needed = ("bb_ca", "local_latents")
    for name, bag in (("x_t", x_t), ("velocity", velocity), ("t", t),
                      ("weights", weights)):
        missing = [m for m in needed if m not in bag]
        if missing:
            raise ValueError(
                f"residue commitment {name} missing {missing}; without both "
                "channels the who cannot differ by residue")
    w_ca = weights["bb_ca"]
    w_lat = weights["local_latents"]
    if w_ca.shape != w_lat.shape:
        raise ValueError(
            f"follower scores differ {tuple(w_ca.shape)} vs {tuple(w_lat.shape)}")
    lat_follows = w_lat >= w_ca
    ca_follows = ~lat_follows
    if row_mask is None:
        active = torch.ones_like(lat_follows)
    else:
        active = row_mask.bool()
        if active.shape != lat_follows.shape:
            raise ValueError(
                f"row_mask shape {tuple(active.shape)} != "
                f"{tuple(lat_follows.shape)}")
    lat_follows = lat_follows & active
    ca_follows = ca_follows & active
    ca_commits = active & ~ca_follows
    lat_commits = active & ~lat_follows

    def _commit(modality, commits):
        xt = x_t[modality]
        clean = anticipated_clean_state(
            xt, velocity[modality], t[modality].to(dtype=xt.dtype),
            max_step_nm=PRIOR_SCALE_NM)
        tt = t[modality].to(dtype=xt.dtype)
        while tt.dim() < xt.dim():
            tt = tt.unsqueeze(-1)
        frac = (float(dt) / (1.0 - tt).clamp_min(1e-6)).clamp(max=1.0)
        stepped = xt + frac * (clean - xt)
        mask = commits
        while mask.dim() < xt.dim():
            mask = mask[..., None]
        return torch.where(mask, stepped, xt)

    probe = {
        "bb_ca": _commit("bb_ca", ca_commits),
        "local_latents": _commit("local_latents", lat_commits),
    }
    return probe, {"bb_ca": ca_follows, "local_latents": lat_follows}

def anticipated_clean_state(x_t, velocity, t, *, max_step_nm=None):

    while getattr(t, "dim", lambda: 0)() > 0 and t.dim() < x_t.dim():
        t = t[..., None]
    step = (1.0 - t) * velocity
    if max_step_nm is not None:
        norm = step.norm(dim=-1, keepdim=True)
        scale = (max_step_nm / norm.clamp_min(1e-12)).clamp(max=1.0)
        step = step * scale
    return x_t + step
