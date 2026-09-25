
from __future__ import annotations

import torch

CA = 1

BIN_MIN_A = 2.0
BIN_MAX_A = 42.0
N_IN_RANGE = 40
N_BINS = N_IN_RANGE + 2
BIN_WIDTH_A = (BIN_MAX_A - BIN_MIN_A) / N_IN_RANGE

ANCHOR_FEATURES = 20 + 5

def bin_distances(distance_a: torch.Tensor) -> torch.Tensor:

    index = torch.floor((distance_a - BIN_MIN_A) / BIN_WIDTH_A).to(torch.long) + 1
    index = torch.where(distance_a < BIN_MIN_A, torch.zeros_like(index), index)
    index = torch.where(
        distance_a >= BIN_MAX_A, torch.full_like(index, N_BINS - 1), index
    )
    return index.clamp_(0, N_BINS - 1)

def bin_centres_a(device=None, dtype=torch.float32) -> torch.Tensor:

    centres = torch.empty(N_BINS, device=device, dtype=dtype)
    centres[0] = BIN_MIN_A
    centres[N_BINS - 1] = BIN_MAX_A
    inner = torch.arange(N_IN_RANGE, device=device, dtype=dtype)
    centres[1:N_BINS - 1] = BIN_MIN_A + (inner + 0.5) * BIN_WIDTH_A
    return centres

def anchor_positions(
    x_motif: torch.Tensor, motif_mask: torch.Tensor, *, max_anchors: int
) -> tuple[torch.Tensor, torch.Tensor]:

    required = motif_mask.to(torch.bool)
    b, n_motif, _ = required.shape
    keep = min(n_motif, max_anchors)

    coordinates = x_motif[:, :keep]
    present = required[:, :keep]
    count = present.sum(-1)
    valid = count > 0

    centroid = (coordinates * present[..., None]).sum(-2) / count.clamp(min=1)[..., None]
    has_ca = present[..., CA]
    position = torch.where(has_ca[..., None], coordinates[..., CA, :], centroid)
    position = torch.where(valid[..., None], position, torch.zeros_like(position))

    if keep < max_anchors:
        pad = max_anchors - keep
        position = torch.nn.functional.pad(position, (0, 0, 0, pad))
        valid = torch.nn.functional.pad(valid, (0, pad), value=False)
    return position, valid

def anchor_features(
    x_motif: torch.Tensor,
    motif_mask: torch.Tensor,
    seq_motif: torch.Tensor,
    *,
    max_anchors: int,
) -> torch.Tensor:

    position, valid = anchor_positions(x_motif, motif_mask, max_anchors=max_anchors)
    b, m, _ = position.shape
    required = motif_mask.to(torch.bool)[:, :m]
    if required.shape[1] < m:
        required = torch.nn.functional.pad(required, (0, 0, 0, m - required.shape[1]))
    identity = seq_motif[:, :m].long()
    if identity.shape[1] < m:
        identity = torch.nn.functional.pad(identity, (0, m - identity.shape[1]))
    identity = torch.where(
        (identity >= 0) & (identity < 20) & valid, identity, torch.zeros_like(identity)
    )
    one_hot = torch.nn.functional.one_hot(identity, num_classes=20).to(position.dtype)
    one_hot = one_hot * valid[..., None].to(position.dtype)

    separation = torch.linalg.vector_norm(
        position[:, :, None, :] - position[:, None, :, :], dim=-1
    ) * 10.0
    pair_valid = valid[:, :, None] & valid[:, None, :]
    pair_valid = pair_valid & ~torch.eye(m, dtype=torch.bool, device=position.device)
    others = pair_valid.sum(-1)
    mean_separation = (
        torch.where(pair_valid, separation, torch.zeros_like(separation)).sum(-1)
        / others.clamp(min=1)
    )
    max_separation = torch.where(
        pair_valid, separation, torch.zeros_like(separation)
    ).max(-1).values

    extras = torch.stack([
        required.sum(-1).to(position.dtype) / 8.0,
        required[..., CA].to(position.dtype),
        torch.arange(m, device=position.device, dtype=position.dtype).expand(b, m) / m,
        mean_separation / 40.0,
        max_separation / 40.0,
    ], dim=-1)
    extras = extras * valid[..., None].to(position.dtype)
    return torch.cat([one_hot, extras], dim=-1)

def relation_distances(
    ca_nm: torch.Tensor,
    present: torch.Tensor,
    anchor_nm: torch.Tensor,
    anchor_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    delta = ca_nm[:, :, None, :] * 10.0 - anchor_nm[:, None, :, :] * 10.0
    distance = torch.linalg.vector_norm(delta, dim=-1)
    valid = present[:, :, None] & anchor_valid[:, None, :]
    return torch.where(valid, distance, torch.zeros_like(distance)), valid

def relation_targets(
    coords_nm: torch.Tensor,
    coord_mask: torch.Tensor,
    x_motif: torch.Tensor,
    motif_mask: torch.Tensor,
    seq_motif: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_anchors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    ca = coords_nm[..., CA, :]
    present = coord_mask[..., CA].to(torch.bool) & mask.to(torch.bool)
    anchor_nm, anchor_valid = anchor_positions(x_motif, motif_mask, max_anchors=max_anchors)
    distance, valid = relation_distances(ca, present, anchor_nm, anchor_valid)
    bins = torch.where(valid, bin_distances(distance), torch.zeros_like(distance).long())
    features = anchor_features(x_motif, motif_mask, seq_motif, max_anchors=max_anchors)
    return bins, valid, distance, features
