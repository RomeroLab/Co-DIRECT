
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Generator, Tensor

from dive.analysis.heterogeneity import (
    DENSITY_RADIUS_NM,
    Asymmetry,
    NullInterval,
    association_test,
    concentration,
    covariates,
    direction_asymmetry,
    median_across_targets,
    profile_test,
    step_profile,
)
from dive.oracle import delta_magnitude

DIRECTIONS: tuple[str, ...] = ("z_to_x", "x_to_z")
NULLS: tuple[str, ...] = ("zero", "sham")
DECISION_NULL = "sham"
COVARIATE_NAMES: tuple[str, ...] = (
    "dist_to_target",
    "local_density",
    "relative_position",
)
MAGNITUDE_KEYS: tuple[str, ...] = tuple(
    f"{null}.{direction}" for null in NULLS for direction in DIRECTIONS
)

ASYMMETRY_IDENTICAL_ABS_R = 0.95

_DIRECTION_DIM: Mapping[str, int] = {"z_to_x": 3, "x_to_z": 8}

_SHARD_FIELD: Mapping[str, str] = {
    "zero.z_to_x": "b_zero_z_to_x",
    "zero.x_to_z": "b_zero_x_to_z",
    "sham.z_to_x": "b_sham_z_to_x",
    "sham.x_to_z": "b_sham_x_to_z",
}

class CampaignError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class Shard:

    target: str
    seed: int
    steps: int
    n_res: int
    valid: Tensor
    magnitude: Mapping[str, Tensor]
    covariate: Mapping[str, Tensor]

@dataclass(frozen=True, slots=True)
class CampaignGrids:

    targets: tuple[str, ...]
    seeds: tuple[int, ...]
    n_steps: int
    n_res: int
    valid: Tensor
    magnitude: Mapping[str, Tensor]
    covariate: Mapping[str, Tensor]

    def shape_dict(self) -> dict[str, object]:

        return {
            "targets": len(self.targets),
            "seeds": len(self.seeds),
            "steps": self.n_steps,
            "max_residues": self.n_res,
            "valid_cells": int(self.valid.sum()),
        }

@dataclass(frozen=True, slots=True)
class CampaignReport:

    decision_null: str
    shape: Mapping[str, object]
    concentration: Mapping[str, Mapping[str, Mapping[str, float]]]
    association: Mapping[str, Mapping[str, Mapping[str, NullInterval]]]
    sign_agreement: Mapping[str, Mapping[str, float]]
    profile: Mapping[str, Mapping[str, NullInterval]]
    asymmetry: Mapping[str, Mapping[str, object]]
    where_falsified: bool
    when_falsified: bool
    premise_falsified: bool

    @property
    def heterogeneity_falsified(self) -> bool:

        return self.where_falsified and self.when_falsified

    def to_dict(self) -> dict[str, object]:

        def interval(value: NullInterval) -> dict[str, object]:
            return {
                "observed": value.observed,
                "null_low": value.low,
                "null_high": value.high,
                "inside_null": value.inside,
            }

        return {
            "decision_null": self.decision_null,
            "shape": dict(self.shape),
            "concentration": {
                null: {
                    direction: dict(values)
                    for direction, values in directions.items()
                }
                for null, directions in self.concentration.items()
            },
            "covariate_association": {
                null: {
                    direction: {name: interval(value) for name, value in covs.items()}
                    for direction, covs in directions.items()
                }
                for null, directions in self.association.items()
            },
            "sign_agreement_across_seeds": {
                null: dict(values) for null, values in self.sign_agreement.items()
            },
            "timestep_profile": {
                null: {direction: interval(value) for direction, value in directions.items()}
                for null, directions in self.profile.items()
            },
            "direction_asymmetry": {
                null: dict(values) for null, values in self.asymmetry.items()
            },
            "verdict": {
                "where_falsified": self.where_falsified,
                "when_falsified": self.when_falsified,
                "heterogeneity_falsified": self.heterogeneity_falsified,
                "premise_falsified": self.premise_falsified,
                "asymmetry_identical_abs_r": ASYMMETRY_IDENTICAL_ABS_R,
                "reading": (
                    "where_falsified requires every covariate and both directions "
                    "inside their permutation nulls (section 8.3); when_falsified "
                    "requires both directions inside (section 8.4); clause 1 falls "
                    "only if both hold. Decisions are taken on the sham null."
                ),
            },
        }

def load_shard(path: Path | str) -> Shard:

    payload = torch.load(Path(path), weights_only=True)
    for required in ("target", "seed", "valid", "x_t_bb_ca", "x_target", *_SHARD_FIELD.values()):
        if required not in payload:
            raise CampaignError(f"{path} is missing '{required}'")

    valid = _squeeze_sample(payload["valid"], path)
    if valid.dtype is not torch.bool:
        raise CampaignError(f"{path} carries a non-boolean 'valid'")
    steps, n_res = valid.shape

    magnitude = {
        key: delta_magnitude(_squeeze_sample(payload[field], path), valid)
        for key, field in _SHARD_FIELD.items()
    }

    x_t_bb_ca = _squeeze_sample(payload["x_t_bb_ca"], path)
    x_target = payload["x_target"]
    while x_target.dim() > 2:
        if x_target.shape[0] != 1:
            raise CampaignError(f"{path} carries a target with a sample axis > 1")
        x_target = x_target[0]

    try:
        covariate = covariates(
            x_t_bb_ca, x_target, valid, density_radius=DENSITY_RADIUS_NM
        )
    except ValueError as error:
        raise CampaignError(f"{path} cannot yield covariates: {error}") from error

    return Shard(
        target=str(payload["target"]),
        seed=int(payload["seed"]),
        steps=int(steps),
        n_res=int(n_res),
        valid=valid,
        magnitude=magnitude,
        covariate=covariate,
    )

def assemble_campaign(
    paths: Iterable[Path | str], *, expected_steps: int
) -> CampaignGrids:

    shards = [load_shard(path) for path in paths]
    if not shards:
        raise CampaignError("no shard was supplied; a campaign cannot be assembled")

    seen: dict[tuple[str, int], Shard] = {}
    for shard in shards:
        key = (shard.target, shard.seed)
        if key in seen:
            raise CampaignError(f"duplicate shard for {shard.target} seed {shard.seed}")
        if shard.steps != expected_steps:
            raise CampaignError(
                f"{shard.target} seed {shard.seed} recorded {shard.steps} steps, "
                f"expected {expected_steps}"
            )
        seen[key] = shard

    targets = tuple(sorted({shard.target for shard in shards}))
    seeds = tuple(sorted({shard.seed for shard in shards}))
    missing = [
        f"{target} seed {seed}"
        for target in targets
        for seed in seeds
        if (target, seed) not in seen
    ]
    if missing:
        raise CampaignError("incomplete campaign, missing: " + ", ".join(missing))

    n_res = max(shard.n_res for shard in shards)
    shape = (len(targets), len(seeds), expected_steps, n_res)

    valid = torch.zeros(shape, dtype=torch.bool)
    magnitude = {key: torch.full(shape, float("nan"), dtype=torch.float64) for key in MAGNITUDE_KEYS}
    covariate = {
        name: torch.full(shape, float("nan"), dtype=torch.float64) for name in COVARIATE_NAMES
    }

    for t, target in enumerate(targets):
        for s, seed in enumerate(seeds):
            shard = seen[(target, seed)]
            width = shard.n_res
            valid[t, s, :, :width] = shard.valid
            for key in MAGNITUDE_KEYS:
                magnitude[key][t, s, :, :width] = shard.magnitude[key].to(torch.float64)
            for name in COVARIATE_NAMES:
                covariate[name][t, s, :, :width] = shard.covariate[name].to(torch.float64)

    return CampaignGrids(
        targets=targets,
        seeds=seeds,
        n_steps=expected_steps,
        n_res=n_res,
        valid=valid,
        magnitude=magnitude,
        covariate=covariate,
    )

def dimension_matched_concentration_null(
    dim: int, n_valid: int, *, generator: Generator, draws: int = 40, top_fraction: float = 0.2
) -> float:

    if dim < 1:
        raise CampaignError(f"dim must be positive, observed {dim}")
    if n_valid < 2:
        raise CampaignError(f"n_valid must be at least two, observed {n_valid}")
    mask = torch.ones(1, n_valid, dtype=torch.bool)
    values = [
        concentration(
            torch.randn(1, n_valid, dim, generator=generator).pow(2).sum(-1).sqrt(),
            mask,
            top_fraction=top_fraction,
        ).top_mass
        for _ in range(draws)
    ]
    return float(sum(values) / len(values))

def analyze_campaign(
    grids: CampaignGrids, *, generator: Generator, draws: int = 200
) -> CampaignReport:

    concentrations: dict[str, dict[str, dict[str, float]]] = {}
    associations: dict[str, dict[str, dict[str, NullInterval]]] = {}
    agreements: dict[str, dict[str, float]] = {}
    profiles: dict[str, dict[str, NullInterval]] = {}
    asymmetries: dict[str, dict[str, object]] = {}

    for null in NULLS:
        concentrations[null] = {}
        associations[null] = {}
        agreements[null] = {}
        profiles[null] = {}

        for direction in DIRECTIONS:
            norms = grids.magnitude[f"{null}.{direction}"]
            concentrations[null][direction] = _concentration_summary(
                norms, grids.valid, direction, generator=generator
            )
            associations[null][direction] = {
                name: association_test(
                    norms,
                    grids.covariate[name],
                    grids.valid,
                    generator=generator,
                    draws=draws,
                )
                for name in COVARIATE_NAMES
            }
            for name in COVARIATE_NAMES:
                agreements[null][f"{direction}.{name}"] = _sign_agreement(
                    norms, grids.covariate[name], grids.valid
                )
            profiles[null][direction] = profile_test(
                norms, grids.valid, generator=generator, draws=draws
            )

        asymmetries[null] = _asymmetry_summary(grids, null)

    decisive = associations[DECISION_NULL]
    where_falsified = all(
        interval.inside
        for direction in DIRECTIONS
        for interval in decisive[direction].values()
    )
    when_falsified = all(
        profiles[DECISION_NULL][direction].inside for direction in DIRECTIONS
    )
    premise_falsified = bool(asymmetries[DECISION_NULL]["median_abs_correlation"] >= ASYMMETRY_IDENTICAL_ABS_R)

    return CampaignReport(
        decision_null=DECISION_NULL,
        shape=grids.shape_dict(),
        concentration=concentrations,
        association=associations,
        sign_agreement=agreements,
        profile=profiles,
        asymmetry=asymmetries,
        where_falsified=where_falsified,
        when_falsified=when_falsified,
        premise_falsified=premise_falsified,
    )

def _squeeze_sample(tensor: Tensor, path: Path | str) -> Tensor:

    if tensor.dim() < 2:
        raise CampaignError(f"{path} carries a tensor with too few axes: {tuple(tensor.shape)}")
    if tensor.shape[1] != 1:
        raise CampaignError(
            f"{path} has a sample axis of {tensor.shape[1]}; the probe records one sample per run"
        )
    return tensor[:, 0]

def _concentration_summary(
    norms: Tensor, valid: Tensor, direction: str, *, generator: Generator
) -> dict[str, float]:

    per_target: list[float] = []
    counts: list[int] = []
    for t in range(valid.shape[0]):
        result = concentration(norms[t].reshape(-1), valid[t].reshape(-1))
        per_target.append(result.top_mass)
        counts.append(result.n_valid)

    observed = float(torch.tensor(per_target, dtype=torch.float64).median())
    matched = dimension_matched_concentration_null(
        _DIRECTION_DIM[direction],
        n_valid=int(torch.tensor(counts, dtype=torch.float64).median()),
        generator=generator,
    )
    return {
        "median_top_mass": observed,
        "dimension_matched_null": matched,
        "lift_over_matched_null": observed - matched,
        "uniform_baseline": 0.2,
        "vector_dim": float(_DIRECTION_DIM[direction]),
    }

def _sign_agreement(norms: Tensor, covariate: Tensor, valid: Tensor) -> float:

    n_targets, n_seeds = valid.shape[0], valid.shape[1]
    if n_seeds < 2:
        return float("nan")

    agreed = 0
    for t in range(n_targets):
        signs = []
        for s in range(n_seeds):
            block = _flat_association(norms[t, s], covariate[t, s], valid[t, s])
            signs.append(0.0 if block == 0.0 else (1.0 if block > 0 else -1.0))
        if len(set(signs)) == 1 and signs[0] != 0.0:
            agreed += 1
    return agreed / n_targets

def _flat_association(norms: Tensor, covariate: Tensor, valid: Tensor) -> float:

    from dive.analysis.heterogeneity import covariate_association

    return covariate_association(norms.reshape(-1), covariate.reshape(-1), valid.reshape(-1))

def _asymmetry_summary(grids: CampaignGrids, null: str) -> dict[str, object]:

    forward = grids.magnitude[f"{null}.z_to_x"]
    backward = grids.magnitude[f"{null}.x_to_z"]

    correlations: list[float] = []
    agreements: list[bool] = []
    for t in range(grids.valid.shape[0]):
        for s in range(grids.valid.shape[1]):
            result: Asymmetry = direction_asymmetry(
                step_profile(forward[t, s], grids.valid[t, s]),
                step_profile(backward[t, s], grids.valid[t, s]),
            )
            correlations.append(result.correlation)
            agreements.append(result.argmax_agrees)

    values = torch.tensor(correlations, dtype=torch.float64)
    return {
        "median_correlation": float(values.median()),
        "median_abs_correlation": float(values.abs().median()),
        "argmax_agreement_fraction": float(sum(agreements) / len(agreements)),
        "n_target_seed_pairs": len(correlations),
        "note": (
            "a descriptive judgement about what 'effectively the same' means, "
            "not a significance test (design section 8.5)"
        ),
    }
