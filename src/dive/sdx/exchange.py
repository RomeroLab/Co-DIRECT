
from __future__ import annotations

import math

import torch

from dive.sdx.relations import N_BINS, bin_centres_a, bin_distances

FEATURE_NAMES = (
    "nll_identity_mean",
    "nll_identity_max",
    "entropy_identity_mean",
    "shift_identity_mean",
    "shift_identity_absmax",
    "nll_geometry_mean",
    "kl_views_mean",
    "anchor_fraction",
)
N_FEATURES = len(FEATURE_NAMES)

SHIFT_SCALE_A = 10.0

_LOG_BINS = math.log(N_BINS)

def _expected_distance(log_probability: torch.Tensor) -> torch.Tensor:
    centres = bin_centres_a(device=log_probability.device, dtype=log_probability.dtype)
    return (log_probability.exp() * centres).sum(-1)

def message_features(
    identity_logits: torch.Tensor,
    geometry_logits: torch.Tensor,
    distance_a: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:

    log_identity = torch.log_softmax(identity_logits.float(), dim=-1)
    log_geometry = torch.log_softmax(geometry_logits.float(), dim=-1)

    target = bin_distances(distance_a).unsqueeze(-1)
    nll_identity = -torch.gather(log_identity, -1, target).squeeze(-1)
    nll_geometry = -torch.gather(log_geometry, -1, target).squeeze(-1)

    entropy = -(log_identity.exp() * log_identity).sum(-1)
    kl = (log_geometry.exp() * (log_geometry - log_identity)).sum(-1)
    shift = _expected_distance(log_identity) - distance_a

    count = valid.sum(-1)
    any_valid = count > 0
    denominator = count.clamp(min=1).to(distance_a.dtype)
    zero = torch.zeros((), device=distance_a.dtype and distance_a.device,
                       dtype=distance_a.dtype)

    def mean(values: torch.Tensor) -> torch.Tensor:
        return torch.where(valid, values, torch.zeros_like(values)).sum(-1) / denominator

    def maximum(values: torch.Tensor) -> torch.Tensor:
        floored = torch.where(valid, values, torch.full_like(values, -math.inf))
        out = floored.max(-1).values
        return torch.where(any_valid, out, torch.zeros_like(out))

    columns = [
        mean(nll_identity) / _LOG_BINS,
        maximum(nll_identity) / _LOG_BINS,
        mean(entropy) / _LOG_BINS,
        mean(shift) / SHIFT_SCALE_A,
        maximum(shift.abs()) / SHIFT_SCALE_A,
        mean(nll_geometry) / _LOG_BINS,
        mean(kl) / _LOG_BINS,
        count.to(distance_a.dtype) / max(valid.shape[-1], 1),
    ]
    features = torch.stack(columns, dim=-1)
    features = torch.where(any_valid[..., None], features, torch.zeros_like(features))
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

VARIANTS: dict[str, tuple[int, ...]] = {
    "full": tuple(range(N_FEATURES)),

    "mean_only": (3, 4, 7),

    "likelihood_only": (0, 1, 2, 7),
}

def apply_variant(message: torch.Tensor, variant: str) -> torch.Tensor:
    try:
        keep = VARIANTS[variant]
    except KeyError as error:
        raise KeyError(
            f"unknown message variant {variant!r}; expected {sorted(VARIANTS)}"
        ) from error
    out = torch.zeros_like(message)
    for column in keep:
        out[..., column] = message[..., column]
    return out
