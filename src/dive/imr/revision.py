
from __future__ import annotations

import torch

N_SLOT, CA_SLOT, C_SLOT = 0, 1, 2

B_N_CA = 0.1458
B_CA_C = 0.1525
B_C_N = 0.1329

def implicit_vote(decoded, leader_ca, mask):

    coors, atom_mask = decoded["coors_nm"], decoded["atom_mask"]
    have = (atom_mask[..., N_SLOT].bool()
            & atom_mask[..., C_SLOT].bool()
            & mask.bool())
    vote = torch.zeros_like(leader_ca)
    for slot, target in ((N_SLOT, B_N_CA), (C_SLOT, B_CA_C)):
        delta = leader_ca - coors[..., slot, :]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        vote = vote + (target - dist) * (delta / dist)
    vote = 0.5 * vote
    vote = torch.where(have[..., None], vote, torch.zeros_like(vote))
    return vote, vote.norm(dim=-1), have

def cap_per_residue(vote, cap_nm):

    if cap_nm is None:
        return vote
    norm = vote.norm(dim=-1, keepdim=True)
    scale = torch.where(norm > cap_nm, cap_nm / norm.clamp_min(1e-12),
                        torch.ones_like(norm))
    return vote * scale

def in_window(t, window):

    low, high = window
    return bool(((t > low) & (t < high)).any())

def blind_displacement(vote, mask, generator=None):

    noise = torch.randn(vote.shape, generator=generator,
                        dtype=vote.dtype, device="cpu").to(vote.device)
    unit = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = unit * vote.norm(dim=-1, keepdim=True)
    return torch.where(mask.bool()[..., None], out, torch.zeros_like(out))

def scramble_ca(ca, mask, generator=None):

    out = ca.clone()
    sel = mask.bool()
    for b in range(ca.shape[0]):
        idx = sel[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        centroid = ca[b, idx].mean(0)
        offsets = ca[b, idx] - centroid
        perm = torch.randperm(idx.numel(), generator=generator).to(idx.device)
        out[b, idx] = centroid + offsets[perm]
    return out

HEALTH_BAND_NM = (0.10, 0.20)

def follower_is_coherent(decoded, leader_ca, mask, band=HEALTH_BAND_NM):

    coors, atom_mask = decoded["coors_nm"], decoded["atom_mask"]
    have = (atom_mask[..., N_SLOT].bool()
            & atom_mask[..., C_SLOT].bool()
            & mask.bool())
    if not bool(have.any()):
        return False, (None, None)
    n_ca = (coors[..., N_SLOT, :] - leader_ca).norm(dim=-1)[have].median()
    ca_c = (coors[..., C_SLOT, :] - leader_ca).norm(dim=-1)[have].median()
    low, high = band
    ok = bool(low <= float(n_ca) <= high and low <= float(ca_c) <= high)
    return ok, (float(n_ca), float(ca_c))
