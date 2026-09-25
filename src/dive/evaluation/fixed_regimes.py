
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from dive.leadership.fields import FIXED_CORNERS

MIN_REGIME_SPREAD = 0.02
MIN_ORACLE_GAIN = 0.05
MIN_ORACLE_AFFECTED_FRACTION = 0.20
ALLOWED_GATE2_PARTITIONS = frozenset({"validation", "legacy-dev"})

def assert_gate2_partitions(partitions: Sequence[str]) -> tuple[str, ...]:

    normalized = tuple(str(partition).strip() for partition in partitions)
    forbidden = sorted(set(normalized) - ALLOWED_GATE2_PARTITIONS)
    if forbidden:
        raise ValueError(
            f"Gate 2 may use only validation and legacy-dev; refused {forbidden}"
        )
    if "validation" not in normalized:
        raise ValueError("Gate 2 fixed-flow scan requires validation")
    return normalized

@dataclass(frozen=True, slots=True)
class Gate2Verdict:

    passed: bool
    reasons: tuple[str, ...]
    representationally_blocked: bool

@dataclass(frozen=True, slots=True)
class OracleHeadroom:

    best_global_regime: str
    best_time_only_regime_by_bin: Mapping[int, str]
    control_kind: str
    control_losses: tuple[float, ...]
    oracle_losses: tuple[float, ...]
    gains: tuple[float, ...]

@dataclass(frozen=True, slots=True)
class SharedControl:

    best_global_regime: str
    best_time_only_regime_by_bin: Mapping[int, str]
    control_kind: str
    control_losses: tuple[float, ...]

def shared_family_agnostic_control(
    losses: Mapping[str, Sequence[float]],
    *,
    base_losses: Sequence[float],
    family_ids: Sequence[str],
    time_bins: Sequence[int],
) -> SharedControl:

    if not losses:
        raise ValueError("at least one fixed regime is required")
    size = len(base_losses)
    if size == 0 or len(family_ids) != size or len(time_bins) != size:
        raise ValueError("base losses, family ids, and time bins must align")
    names = tuple(sorted(losses))
    numeric = {name: tuple(float(value) for value in losses[name]) for name in names}
    if any(len(values) != size for values in numeric.values()):
        raise ValueError("every regime must carry one loss per example")
    base = tuple(float(value) for value in base_losses)
    if any(not math.isfinite(value) or value <= 0 for value in base):
        raise ValueError("base losses must be finite and positive")
    if any(
        not math.isfinite(value)
        for values in numeric.values()
        for value in values
    ):
        raise ValueError("fixed-regime losses must be finite")

    families = tuple(sorted(set(str(value) for value in family_ids)))

    def score(values: Sequence[float], indices: Sequence[int]) -> float:
        ratios = []
        for family in families:
            selected = [i for i in indices if str(family_ids[i]) == family]
            if selected:
                ratios.append(
                    sum(values[i] for i in selected) / sum(base[i] for i in selected)
                )
        if not ratios:
            raise ValueError("control selection received an empty slice")
        return sum(ratios) / len(ratios)

    all_indices = tuple(range(size))
    best_global = min(names, key=lambda name: (score(numeric[name], all_indices), name))
    global_losses = numeric[best_global]
    best_by_bin: dict[int, str] = {}
    time_losses = [0.0] * size
    for bin_id in sorted(set(int(value) for value in time_bins)):
        indices = tuple(i for i, value in enumerate(time_bins) if int(value) == bin_id)
        selected = min(names, key=lambda name: (score(numeric[name], indices), name))
        best_by_bin[bin_id] = selected
        for i in indices:
            time_losses[i] = numeric[selected][i]
    if score(time_losses, all_indices) <= score(global_losses, all_indices):
        kind, control = "time_only", tuple(time_losses)
    else:
        kind, control = "global", global_losses
    return SharedControl(
        best_global_regime=best_global,
        best_time_only_regime_by_bin=best_by_bin,
        control_kind=kind,
        control_losses=tuple(control),
    )

def fixed_gate_grid(
    values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, tuple[float, float]]:

    corner_by_pair = {pair: name for name, pair in FIXED_CORNERS.items()}
    schedules = {}
    for z_to_x in values:
        for x_to_z in values:
            pair = (float(z_to_x), float(x_to_z))
            name = corner_by_pair.get(
                pair, f"z_to_x={pair[0]:.2f},x_to_z={pair[1]:.2f}"
            )
            schedules[name] = pair
    if len(schedules) != len(values) ** 2:
        raise ValueError("gate grid values must be unique")
    return schedules

def fixed_regime_predictor(module, *, z_to_x: float, x_to_z: float):

    def predict(batch, mode: str = "full", n_recycle: int = 0):
        if mode != "full":
            raise ValueError(
                "fixed-regime generation requires guidance_w=1; "
                f"refusing sampling mode {mode!r}"
            )
        if n_recycle != 0:
            raise ValueError("fixed-regime generation requires n_recycle=0")
        return module.velocities_at_gates(
            batch, z_to_x=z_to_x, x_to_z=x_to_z
        )

    return predict

def routed_predictor(module):

    from dive.training.stages import contract_for

    corner = contract_for(module.stage).fixed_corner
    if corner is not None:
        raise ValueError(
            f"stage {module.stage} pins the gates to {corner!r}, so this would "
            f"sample the fixed corner under a routed label. Build the module "
            f"with the stage whose checkpoint is being generated from."
        )

    def predict(batch, mode: str = "full", n_recycle: int = 0):
        if mode != "full":
            raise ValueError(
                "routed generation requires guidance_w=1; refusing sampling "
                f"mode {mode!r}"
            )
        if n_recycle != 0:
            raise ValueError("routed generation requires n_recycle=0")
        return module.routed_velocities(batch)

    return predict

def oracle_headroom(
    per_residue_losses: Mapping[str, Tensor],
    *,
    mask: Tensor,
    time_bins: Tensor,
) -> OracleHeadroom:

    if not per_residue_losses:
        raise ValueError("at least one fixed regime is required")
    names = tuple(sorted(per_residue_losses))
    shape = tuple(mask.shape)
    if tuple(time_bins.shape) != (shape[0],):
        raise ValueError("time_bins must carry one bin per example")
    tensors = []
    for name in names:
        value = per_residue_losses[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"regime {name!r} has shape {tuple(value.shape)}, expected {shape}")
        if not torch.isfinite(value).all():
            raise ValueError(f"regime {name!r} contains non-finite loss terms")
        tensors.append(value * mask)
    stacked = torch.stack(tensors, dim=0)
    example_losses = stacked.sum(dim=-1)

    global_index = int(example_losses.mean(dim=1).argmin())
    global_losses = example_losses[global_index]
    best_by_bin: dict[int, str] = {}
    time_only_losses = torch.empty_like(global_losses)
    for raw_bin in torch.unique(time_bins, sorted=True):
        bin_id = int(raw_bin)
        selected = time_bins == raw_bin
        regime_index = int(example_losses[:, selected].mean(dim=1).argmin())
        best_by_bin[bin_id] = names[regime_index]
        time_only_losses[selected] = example_losses[regime_index, selected]

    if float(time_only_losses.mean()) <= float(global_losses.mean()):
        control_kind, control = "time_only", time_only_losses
    else:
        control_kind, control = "global", global_losses
    oracle = stacked.min(dim=0).values.sum(dim=-1)
    gains = torch.where(
        control > 0,
        (control - oracle) / control,
        torch.zeros_like(control),
    )
    return OracleHeadroom(
        best_global_regime=names[global_index],
        best_time_only_regime_by_bin=best_by_bin,
        control_kind=control_kind,
        control_losses=tuple(float(v) for v in control),
        oracle_losses=tuple(float(v) for v in oracle),
        gains=tuple(float(v) for v in gains),
    )

def residual_relative_norms(
    condition_velocities: Mapping[str, Mapping[str, Tensor]], mask: Tensor
) -> dict[str, float]:

    valid = mask[..., None]

    def ratio(modality: str, null: str) -> float:
        joint = condition_velocities["joint"][modality] * valid
        residual = (joint - condition_velocities[null][modality] * valid)
        denominator = torch.linalg.vector_norm(joint)
        if float(denominator) == 0.0:
            return float("inf")
        return float(torch.linalg.vector_norm(residual) / denominator)

    return {
        "z_to_x": ratio("bb_ca", "without_latent"),
        "x_to_z": ratio("local_latents", "without_backbone"),
    }

def target_bootstrap_interval(
    values: Sequence[float],
    *,
    target_ids: Sequence[str],
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:

    if len(values) != len(target_ids) or not values:
        raise ValueError("values and target_ids must have the same non-zero length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grouped: dict[str, list[float]] = {}
    for value, target in zip(values, target_ids, strict=True):
        grouped.setdefault(str(target), []).append(float(value))
    target_means = [sum(group) / len(group) for _, group in sorted(grouped.items())]
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [rng.choice(target_means) for _ in target_means]
        draws.append(sum(sample) / len(sample))
    quantiles = torch.tensor(draws, dtype=torch.float64).quantile(
        torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return float(quantiles[0]), float(quantiles[1])

def target_bootstrap_ratio_interval(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    target_ids: Sequence[str],
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:

    if not (
        len(numerators) == len(denominators) == len(target_ids)
        and len(numerators) > 0
    ):
        raise ValueError("paired losses and target_ids must have equal non-zero length")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for numerator, denominator, target in zip(
        numerators, denominators, target_ids, strict=True
    ):
        grouped.setdefault(str(target), []).append(
            (float(numerator), float(denominator))
        )
    target_pairs = list(dict(sorted(grouped.items())).values())
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled = [rng.choice(target_pairs) for _ in target_pairs]
        numerator = sum(value for group in sampled for value, _ in group)
        denominator = sum(value for group in sampled for _, value in group)
        if denominator <= 0:
            raise ValueError("bootstrap denominator must be positive")
        draws.append(1.0 - numerator / denominator)
    quantiles = torch.tensor(draws, dtype=torch.float64).quantile(
        torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return float(quantiles[0]), float(quantiles[1])

def gate2_verdict(report: Mapping[str, object]) -> Gate2Verdict:

    families = report["families"]
    if not isinstance(families, Mapping):
        raise TypeError("families must be a mapping")

    qualifying_winners = []
    for family_record in families.values():
        if not isinstance(family_record, Mapping):
            raise TypeError("each family record must be a mapping")
        losses = family_record["normalized_flow_loss_by_regime"]
        if not isinstance(losses, Mapping) or not losses:
            raise ValueError("each family needs fixed-regime losses")
        numeric = {str(name): float(value) for name, value in losses.items()}
        best_name, best_loss = min(numeric.items(), key=lambda item: (item[1], item[0]))
        worst_loss = max(numeric.values())
        improvement = (worst_loss - best_loss) / worst_loss
        if improvement >= MIN_REGIME_SPREAD:
            qualifying_winners.append(best_name)

    reasons: list[str] = []
    if len(qualifying_winners) < 2:
        reasons.append("fixed-regime separation below 2%")
    elif len(set(qualifying_winners)) < 2:
        reasons.append("no family crossover")

    gains = _float_sequence(report["oracle_gain_over_best_global_or_time_only"])
    affected = sum(gain >= MIN_ORACLE_GAIN for gain in gains) / len(gains)
    if affected < MIN_ORACLE_AFFECTED_FRACTION:
        reasons.append("insufficient oracle adaptive headroom")

    valid_nulls = report.get("counterfactual_validity_passed") is True
    joint_preserved = report.get("joint_preservation_passed") is True
    if not valid_nulls:
        reasons.append("counterfactual validity did not pass")
    if not joint_preserved:
        reasons.append("joint preservation did not pass")

    weak_residual_families = sum(
        _both_residuals_are_weak(record)
        for record in families.values()
        if isinstance(record, Mapping)
    )
    representationally_blocked = valid_nulls and joint_preserved and (
        weak_residual_families >= 2 or sum(gains) / len(gains) < 0.01
    )
    return Gate2Verdict(
        passed=not reasons,
        reasons=tuple(reasons),
        representationally_blocked=representationally_blocked,
    )

def _both_residuals_are_weak(record: Mapping[str, object]) -> bool:
    norms = record.get("residual_relative_norms")
    if not isinstance(norms, Mapping):
        return False
    return (
        float(norms.get("z_to_x", float("inf"))) < 0.05
        and float(norms.get("x_to_z", float("inf"))) < 0.05
    )

def _float_sequence(values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("oracle gains must be a sequence")
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("oracle gains must not be empty")
    return result

__all__ = [
    "Gate2Verdict",
    "OracleHeadroom",
    "SharedControl",
    "assert_gate2_partitions",
    "fixed_gate_grid",
    "fixed_regime_predictor",
    "gate2_verdict",
    "oracle_headroom",
    "routed_predictor",
    "residual_relative_norms",
    "shared_family_agnostic_control",
    "target_bootstrap_interval",
    "target_bootstrap_ratio_interval",
]

ROUTED_REGIME = "routed"

def predictor_for_regime(module, regime: str, schedules):

    if regime == ROUTED_REGIME:
        return routed_predictor(module)
    z_to_x, x_to_z = schedules[regime]
    return fixed_regime_predictor(module, z_to_x=z_to_x, x_to_z=x_to_z)

CLIP_ONLY_REGIME = "clip_only"

def clip_only_schedule() -> tuple[float, float]:

    from dive.leadership.initialization import GATE_ENDPOINT_CLIP

    value = 1.0 - GATE_ENDPOINT_CLIP
    return (value, value)
