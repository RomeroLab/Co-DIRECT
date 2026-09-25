
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

import torch
from torch import Tensor

_HARD_ROUTE_COUNT = 4
_TAP_COUNT = 3
_TEMPERATURE = 0.10
_NONE_MARGIN = 0.02
_PRESERVE_WEIGHT = 0.10
_PRESERVE_EPSILON = 1e-8
_OCCUPANCY_MIN = 0.10
_OCCUPANCY_MAX = 0.40
_ENTROPY_FLOOR = 0.5 * math.log(_HARD_ROUTE_COUNT)

class DirectionalObjectiveError(RuntimeError):
    pass

class DirectionalStage(StrEnum):

    BRIDGE_WARMUP = "bridge_warmup"
    JOINT_ADAPTATION = "joint_adaptation"

@dataclass(frozen=True, slots=True)
class RoutePrediction:

    bb_ca: Tensor
    local_latents: Tensor

@dataclass(frozen=True, slots=True)
class LossWeights:

    rank: float
    balance: float
    entropy: float

@dataclass(frozen=True, slots=True)
class DirectionalLossBreakdown:

    total: Tensor
    flow: Tensor
    rank: Tensor
    preserve: Tensor
    balance: Tensor
    entropy_floor: Tensor
    confident_valid_count: int

def _as_stage(stage: DirectionalStage | str) -> DirectionalStage:
    try:
        return DirectionalStage(stage)
    except ValueError as error:
        raise DirectionalObjectiveError(f"unknown directional stage {stage!r}") from error

def loss_weights_at(stage: DirectionalStage | str, step: int) -> LossWeights:

    stage = _as_stage(stage)
    if isinstance(step, bool) or not isinstance(step, int):
        raise DirectionalObjectiveError(f"step must be an integer, got {step!r}")
    maximum = 9_999 if stage is DirectionalStage.BRIDGE_WARMUP else 19_999
    if not 0 <= step <= maximum:
        raise DirectionalObjectiveError(
            f"step {step} is outside the frozen {stage.value} range [0, {maximum}]"
        )

    if stage is DirectionalStage.BRIDGE_WARMUP:
        early = max(0.0, 1.0 - step / 1999.0) if step <= 1999 else 0.0
        return LossWeights(rank=0.25, balance=0.01 * early, entropy=0.01 * early)

    rank = 0.25 if step < 16_000 else 0.25 - 0.20 * ((step - 16_000) / 3999.0)
    return LossWeights(rank=max(0.05, rank), balance=0.0, entropy=0.0)

def _require_finite(name: str, value: Tensor) -> None:
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise DirectionalObjectiveError(f"{name} must contain only finite floating-point values")

def _require_nonnegative(name: str, value: Tensor) -> None:
    if (value < 0).any():
        raise DirectionalObjectiveError(f"{name} must contain only non-negative values")

def _require_shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise DirectionalObjectiveError(
            f"{name} shape {tuple(value.shape)} does not match expected {expected}"
        )

def counterfactual_targets(hard_losses: Tensor, *, none_index: int = 0) -> tuple[Tensor, Tensor]:

    if hard_losses.ndim != 3 or hard_losses.shape[-1] != _HARD_ROUTE_COUNT:
        raise DirectionalObjectiveError(
            "hard_losses must have shape [B, N, 4] in declared hard-route order"
        )
    if not 0 <= none_index < _HARD_ROUTE_COUNT:
        raise DirectionalObjectiveError(f"none_index must be in [0, 3], got {none_index}")
    _require_finite("hard_losses", hard_losses)
    _require_nonnegative("hard_losses", hard_losses)
    detached = hard_losses.detach()
    q = torch.softmax(-detached / _TEMPERATURE, dim=-1)
    ordered, _ = detached.sort(dim=-1)
    margin = ordered[..., 1] - ordered[..., 0]
    confident = (margin > 0) & (margin >= _NONE_MARGIN * detached[..., none_index])
    return q, confident

def _prediction_residual_energy(
    adaptive: Tensor, none: Tensor, valid_mask: Tensor, *, name: str
) -> tuple[Tensor, Tensor]:
    if adaptive.ndim < 3:
        raise DirectionalObjectiveError(
            f"adaptive_prediction.{name} must have at least [B, N, D] dimensions"
        )
    if none.ndim < 3:
        raise DirectionalObjectiveError(
            f"none_prediction.{name} must have at least [B, N, D] dimensions"
        )
    if adaptive.shape[:2] != tuple(valid_mask.shape):
        raise DirectionalObjectiveError(
            f"adaptive_prediction.{name} must start with the [B, N] shape of valid_mask"
        )
    _require_shape(f"none_prediction.{name}", none, tuple(adaptive.shape))
    _require_finite(f"adaptive_prediction.{name}", adaptive)
    _require_finite(f"none_prediction.{name}", none)
    feature_axes = tuple(range(2, adaptive.ndim))
    numerator = (adaptive - none.detach()).square().sum(dim=feature_axes)
    denominator = none.detach().square().sum(dim=feature_axes)
    return numerator, denominator

def _validate_inputs(
    adaptive_losses: Tensor,
    hard_losses: Tensor,
    probabilities: Tensor,
    adaptive_prediction: RoutePrediction,
    none_prediction: RoutePrediction,
    valid_mask: Tensor,
) -> None:
    if adaptive_losses.ndim != 2:
        raise DirectionalObjectiveError("adaptive_losses must have shape [B, N]")
    _require_finite("adaptive_losses", adaptive_losses)
    _require_nonnegative("adaptive_losses", adaptive_losses)
    batch, residues = adaptive_losses.shape
    _require_shape("hard_losses", hard_losses, (batch, residues, _HARD_ROUTE_COUNT))
    _require_shape("probabilities", probabilities, (_TAP_COUNT, batch, residues, _HARD_ROUTE_COUNT))
    _require_shape("valid_mask", valid_mask, (batch, residues))
    _require_finite("hard_losses", hard_losses)
    _require_nonnegative("hard_losses", hard_losses)
    _require_finite("probabilities", probabilities)
    if valid_mask.dtype != torch.bool:
        raise DirectionalObjectiveError("valid_mask must have boolean dtype")
    if not valid_mask.any():
        raise DirectionalObjectiveError("at least one valid generated residue is required")
    if (probabilities <= 0).any() or not torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]), rtol=1e-5, atol=1e-6
    ):
        raise DirectionalObjectiveError("probabilities must be strictly positive and sum to one")
    _prediction_residual_energy(
        adaptive_prediction.bb_ca, none_prediction.bb_ca, valid_mask, name="bb_ca"
    )
    _prediction_residual_energy(
        adaptive_prediction.local_latents,
        none_prediction.local_latents,
        valid_mask,
        name="local_latents",
    )

def directional_objective(
    *,
    adaptive_losses: Tensor,
    hard_losses: Tensor,
    probabilities: Tensor,
    adaptive_prediction: RoutePrediction,
    none_prediction: RoutePrediction,
    valid_mask: Tensor,
    stage: DirectionalStage | str,
    step: int,
    rank_override: float | None = None,
) -> DirectionalLossBreakdown:

    _validate_inputs(
        adaptive_losses,
        hard_losses,
        probabilities,
        adaptive_prediction,
        none_prediction,
        valid_mask,
    )
    weights = loss_weights_at(stage, step)
    if rank_override is not None:
        if (
            isinstance(rank_override, bool)
            or not isinstance(rank_override, Real)
            or not math.isfinite(rank_override)
            or rank_override != 0.0
        ):
            raise DirectionalObjectiveError("rank_override may only be None or exactly 0.0")
        weights = LossWeights(rank=0.0, balance=weights.balance, entropy=weights.entropy)

    flow = adaptive_losses[valid_mask].mean()
    q, confident = counterfactual_targets(hard_losses, none_index=0)
    confident_valid = confident & valid_mask
    if confident_valid.any():
        kl = (torch.xlogy(q, q) - torch.xlogy(q, probabilities)).sum(dim=-1)
        rank = kl[:, confident_valid].mean()
    else:
        rank = probabilities.sum() * 0.0

    preserve_x_numerator, preserve_x_denominator = _prediction_residual_energy(
        adaptive_prediction.bb_ca, none_prediction.bb_ca, valid_mask, name="bb_ca"
    )
    preserve_z_numerator, preserve_z_denominator = _prediction_residual_energy(
        adaptive_prediction.local_latents, none_prediction.local_latents, valid_mask, name="local_latents"
    )
    preserve = (
        (preserve_x_numerator + preserve_z_numerator)
        / (preserve_x_denominator + preserve_z_denominator + _PRESERVE_EPSILON)
    )[valid_mask].mean()

    valid_probabilities = probabilities[:, valid_mask, :]
    occupancy = valid_probabilities.mean(dim=1)
    balance = (
        torch.relu(_OCCUPANCY_MIN - occupancy).sum(dim=-1)
        + torch.relu(occupancy - _OCCUPANCY_MAX).sum(dim=-1)
    ).mean()
    entropy = -(valid_probabilities * valid_probabilities.log()).sum(dim=-1)
    entropy_floor = torch.relu(torch.as_tensor(_ENTROPY_FLOOR, dtype=entropy.dtype, device=entropy.device) - entropy).mean()

    total = (
        flow
        + weights.rank * rank
        + _PRESERVE_WEIGHT * preserve
        + weights.balance * balance
        + weights.entropy * entropy_floor
    )
    return DirectionalLossBreakdown(
        total=total,
        flow=flow,
        rank=rank,
        preserve=preserve,
        balance=balance,
        entropy_floor=entropy_floor,
        confident_valid_count=int(confident_valid.sum().item()),
    )
