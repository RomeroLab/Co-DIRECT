
from __future__ import annotations

import hashlib
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from dive.emergent_contract import EmergentContract
from dive.signed_value.roots import EMERGENT_UPSTREAM_COMMIT, EMERGENT_UPSTREAM_ROOT
from dive.training.directional_objective import DirectionalStage

class DirectionalConfigError(RuntimeError):
    pass

_SOURCE_REPO_ROOT = Path(__file__).resolve().parents[3]
_DIRECTIONAL_BASIS_COMMIT = "3608c182f3c98089dae2a4a248fa0fbddbaa57ab"
_DEVICE_TYPE = "NVIDIA L40S"
_CONFIG_PATHS = {
    "resources": "configs/emergent/resources.yaml",
    "model": "configs/emergent/model.yaml",
    "data": "configs/emergent/data.yaml",
}
_DEVELOPMENT_PARTITIONS = ("train", "validation", "legacy-dev")
_MEMORY_VARIANTS = frozenset({(2, 1), (1, 2)})
_STAGE_OPTIMIZERS = {
    DirectionalStage.BRIDGE_WARMUP: (3.0e-4, 0.0),
    DirectionalStage.JOINT_ADAPTATION: (1.0e-4, 0.01),
}
_STAGE_STEPS = {
    DirectionalStage.BRIDGE_WARMUP: 10_000,
    DirectionalStage.JOINT_ADAPTATION: 20_000,
}

_SMOKE_STEPS = 6

_SMOKE_STAGES = frozenset({DirectionalStage.BRIDGE_WARMUP})
_GENERATION_ARMS = ("adaptive", "best_frozen_fixed_control")
_CHECKPOINT_IDENTITIES_V1 = MappingProxyType(
    {
        "common": (
            2_934_289_381,
            "589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9",
        ),
        "autoencoder": (
            4_100_101_779,
            "35f8865efd269995eeaf1670e1c1085acfe2988c40abdeda8e09a0e15eb40816",
        ),
    }
)
_TRAINING_KEYS = frozenset(
    {
        "stage",
        "resources",
        "model",
        "data",
        "partitions",
        "devices",
        "device_type",
        "precision",
        "crop_size",
        "microbatch_size",
        "accumulate_grad_batches",
        "effective_batch_size",
        "seeds",
        "route_equivalent_passes",
        "dropout",
        "ranking_mode",
        "rank_override",
        "optimizer",
        "max_steps",
        "preserve_router_optimizer_state",
    }
)

_OPTIONAL_TRAINING_KEYS = frozenset({"non_scientific_smoke"})
_OPTIMIZER_KEYS = frozenset({"name", "lr", "weight_decay", "betas", "grad_clip_norm"})
_CONFIRMATION_KEYS = frozenset(
    {
        "bootstrap_seed",
        "bootstrap_resamples",
        "parent_groups_per_family",
        "generation",
        "planned_generation_cells",
        "development",
        "blind",
    }
)
_GENERATION_KEYS = frozenset({"seeds", "arms"})
_DEVELOPMENT_KEYS = frozenset(
    {
        "require_counterfactual_validity",
        "require_joint_preservation",
        "max_weak_residual_families",
        "min_directional_residual_norm",
        "max_universal_occupancy",
        "min_non_none_occupancy",
        "min_non_none_states",
        "min_families_per_non_none_state",
        "min_adaptive_gain",
        "min_oracle_gap_recovery",
        "require_rerun_bootstrap_reproduction",
    }
)
_BLIND_KEYS = frozenset(
    {
        "min_families_with_distinct_optima",
        "min_winner_over_worst_gain",
        "min_oracle_example_gain",
        "min_oracle_example_fraction",
        "min_adaptive_gain",
        "min_oracle_gap_recovery",
        "require_counterfactual_validity",
        "require_joint_preservation",
        "require_finite_generation",
        "require_paired_cell_completeness",
        "require_preblind_seed_rerun_agreement",
    }
)

@dataclass(frozen=True, slots=True)
class DirectionalOptimizerConfig:
    name: str
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    grad_clip_norm: float

    def __post_init__(self) -> None:
        _validate_optimizer(self)

@dataclass(frozen=True, slots=True)
class DirectionalRunConfig:

    stage: DirectionalStage
    resources: Path
    model: Path
    data: Path
    partitions: tuple[str, ...]
    devices: int
    device_type: str
    precision: str
    crop_size: int
    microbatch_size: int
    accumulate_grad_batches: int
    declared_effective_batch_size: int
    seeds: tuple[int, int]
    route_equivalent_passes: float
    dropout: float
    ranking_mode: str
    rank_override: float | None
    optimizer: DirectionalOptimizerConfig
    max_steps: int
    preserve_router_optimizer_state: bool

    non_scientific_smoke: bool = False

    def __post_init__(self) -> None:
        _validate_run_config(self)

    @classmethod
    def from_yaml(cls, path: Path) -> "DirectionalRunConfig":
        return cls.from_mapping(_load_mapping(path))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DirectionalRunConfig":
        _exact_keys(
            "training configuration",
            config,
            _TRAINING_KEYS,
            optional=_OPTIONAL_TRAINING_KEYS,
        )
        return cls(
            stage=_stage(config["stage"]),
            resources=_development_path("resources", config["resources"]),
            model=_development_path("model", config["model"]),
            data=_development_path("data", config["data"]),
            partitions=_string_tuple("partitions", config["partitions"]),
            devices=_int("devices", config["devices"]),
            device_type=_string("device_type", config["device_type"]),
            precision=_string("precision", config["precision"]),
            crop_size=_int("crop_size", config["crop_size"]),
            microbatch_size=_int("microbatch_size", config["microbatch_size"]),
            accumulate_grad_batches=_int(
                "accumulate_grad_batches", config["accumulate_grad_batches"]
            ),
            declared_effective_batch_size=_int(
                "effective_batch_size", config["effective_batch_size"]
            ),
            seeds=_int_tuple("seeds", config["seeds"]),
            route_equivalent_passes=_float(
                "route_equivalent_passes", config["route_equivalent_passes"]
            ),
            dropout=_float("dropout", config["dropout"]),
            ranking_mode=_string("ranking_mode", config["ranking_mode"]),
            rank_override=(
                None
                if config["rank_override"] is None
                else _float("rank_override", config["rank_override"])
            ),
            optimizer=_optimizer(config["optimizer"]),
            max_steps=_int("max_steps", config["max_steps"]),
            preserve_router_optimizer_state=_bool(
                "preserve_router_optimizer_state",
                config["preserve_router_optimizer_state"],
            ),
            non_scientific_smoke=(
                _bool("non_scientific_smoke", config["non_scientific_smoke"])
                if "non_scientific_smoke" in config
                else False
            ),
        )

    def effective_batch_size(self, *, world_size: int) -> int:
        _exact_int("world_size", world_size, 4)
        effective = self.microbatch_size * self.accumulate_grad_batches * world_size
        if effective != self.declared_effective_batch_size:
            raise DirectionalConfigError(
                "effective batch size is inconsistent with this frozen config"
            )
        return effective

@dataclass(frozen=True, slots=True)
class DevelopmentGateConfig:
    require_counterfactual_validity: bool
    require_joint_preservation: bool
    max_weak_residual_families: int
    min_directional_residual_norm: float
    max_universal_occupancy: float
    min_non_none_occupancy: float
    min_non_none_states: int
    min_families_per_non_none_state: int
    min_adaptive_gain: float
    min_oracle_gap_recovery: float
    require_rerun_bootstrap_reproduction: bool

    def __post_init__(self) -> None:
        _validate_development_gate_values(self)

@dataclass(frozen=True, slots=True)
class BlindGateConfig:
    min_families_with_distinct_optima: int
    min_winner_over_worst_gain: float
    min_oracle_example_gain: float
    min_oracle_example_fraction: float
    min_adaptive_gain: float
    min_oracle_gap_recovery: float
    require_counterfactual_validity: bool
    require_joint_preservation: bool
    require_finite_generation: bool
    require_paired_cell_completeness: bool
    require_preblind_seed_rerun_agreement: bool

    def __post_init__(self) -> None:
        _validate_blind_gate_values(self)

@dataclass(frozen=True, slots=True)
class DirectionalConfirmationConfig:

    bootstrap_seed: int
    bootstrap_resamples: int
    parent_groups_per_family: int
    generation_seeds: tuple[int, int]
    generation_arms: tuple[str, str]
    planned_generation_cells: int
    development: DevelopmentGateConfig
    blind: BlindGateConfig

    def __post_init__(self) -> None:
        _validate_confirmation_config(self)

    @classmethod
    def from_yaml(cls, path: Path) -> "DirectionalConfirmationConfig":
        return cls.from_mapping(_load_mapping(path))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DirectionalConfirmationConfig":
        _exact_keys("confirmation configuration", config, _CONFIRMATION_KEYS)
        generation = _mapping("generation", config["generation"])
        _exact_keys("generation", generation, _GENERATION_KEYS)
        return cls(
            bootstrap_seed=_int("bootstrap_seed", config["bootstrap_seed"]),
            bootstrap_resamples=_int(
                "bootstrap_resamples", config["bootstrap_resamples"]
            ),
            parent_groups_per_family=_int(
                "parent_groups_per_family", config["parent_groups_per_family"]
            ),
            generation_seeds=_int_tuple("generation.seeds", generation["seeds"]),
            generation_arms=_string_tuple("generation.arms", generation["arms"]),
            planned_generation_cells=_int(
                "planned_generation_cells", config["planned_generation_cells"]
            ),
            development=_development_gates(config["development"]),
            blind=_blind_gates(config["blind"]),
        )

def assert_directional_preflight(
    config: DirectionalRunConfig,
    contract: EmergentContract,
    *,
    repo_root: Path,
    upstream_root: Path,
) -> None:

    _validate_run_config(config)
    if not isinstance(contract, EmergentContract):
        raise DirectionalConfigError("contract must be an EmergentContract")
    source_root = _SOURCE_REPO_ROOT.resolve()
    if Path(repo_root).resolve() != source_root:
        raise DirectionalConfigError(
            "repo_root must be the executing source checkout containing directional_config"
        )
    _assert_clean_committed_repo(source_root, "DIVE")
    _assert_basis_commit(source_root)
    resource_path = source_root / config.resources
    if resource_path.resolve() != source_root / _CONFIG_PATHS["resources"]:
        raise DirectionalConfigError(
            "config.resources must resolve to the frozen resource YAML"
        )
    try:
        frozen_contract = EmergentContract.from_yaml(resource_path)
    except Exception as error:
        raise DirectionalConfigError(
            f"cannot load frozen emergent contract from {resource_path}: {error}"
        ) from error
    _assert_contract_matches_frozen(frozen_contract, contract)
    expected_upstream = EMERGENT_UPSTREAM_ROOT.resolve()
    if frozen_contract.upstream_root != EMERGENT_UPSTREAM_ROOT:
        raise DirectionalConfigError(
            "frozen resources name a different emergent upstream root"
        )
    if frozen_contract.upstream_commit != EMERGENT_UPSTREAM_COMMIT:
        raise DirectionalConfigError(
            "frozen resources name a different emergent upstream pin"
        )
    if Path(upstream_root).resolve() != expected_upstream:
        raise DirectionalConfigError(
            "upstream_root must equal the configured emergent upstream root"
        )
    _assert_clean_committed_repo(expected_upstream, "emergent upstream")
    if _git(expected_upstream, "rev-parse", "HEAD") != EMERGENT_UPSTREAM_COMMIT:
        raise DirectionalConfigError(
            "emergent upstream HEAD differs from the configured pin"
        )
    try:
        frozen_contract.assert_current()
    except Exception as error:
        raise DirectionalConfigError(
            f"frozen emergent contract is not current: {error}"
        ) from error

def _validate_optimizer(config: DirectionalOptimizerConfig) -> None:
    _exact_str("optimizer.name", config.name, "adamw")
    _float("optimizer.lr", config.lr)
    _float("optimizer.weight_decay", config.weight_decay)
    _exact_float_tuple("optimizer.betas", config.betas, (0.9, 0.999))
    _exact_float("optimizer.grad_clip_norm", config.grad_clip_norm, 1.0)
    if (config.lr, config.weight_decay) not in set(_STAGE_OPTIMIZERS.values()):
        raise DirectionalConfigError(
            "optimizer must be one of the frozen stage profiles"
        )

def _validate_run_config(config: DirectionalRunConfig) -> None:
    if not isinstance(config.stage, DirectionalStage):
        raise DirectionalConfigError("stage must be a DirectionalStage instance")
    _direct_path("resources", config.resources, _CONFIG_PATHS["resources"])
    _direct_path("model", config.model, _CONFIG_PATHS["model"])
    _direct_path("data", config.data, _CONFIG_PATHS["data"])
    _exact_str_tuple("partitions", config.partitions, _DEVELOPMENT_PARTITIONS)
    _int("devices", config.devices)
    if config.devices > 4:
        raise DirectionalConfigError(
            "directional campaign permits at most four devices"
        )
    if config.devices != 4:
        raise DirectionalConfigError(
            "directional campaign freezes the four-device world size"
        )
    _exact_str("device_type", config.device_type, _DEVICE_TYPE)
    _exact_str("precision", config.precision, "bf16-mixed")
    _exact_int("crop_size", config.crop_size, 256)
    _int("microbatch_size", config.microbatch_size)
    _int("accumulate_grad_batches", config.accumulate_grad_batches)
    if (config.microbatch_size, config.accumulate_grad_batches) not in _MEMORY_VARIANTS:
        raise DirectionalConfigError(
            "memory contract permits only microbatch/accumulation 2x1 or 1x2"
        )
    _exact_int("effective_batch_size", config.declared_effective_batch_size, 8)
    if config.microbatch_size * config.accumulate_grad_batches * config.devices != 8:
        raise DirectionalConfigError(
            "effective batch size is inconsistent with the four-device memory contract"
        )
    _exact_int_tuple("seeds", config.seeds, (42, 314159))
    _exact_float("route_equivalent_passes", config.route_equivalent_passes, 8.0)
    _exact_float("dropout", config.dropout, 0.0)
    matched_ranking = config.ranking_mode == "matched" and config.rank_override is None
    no_ranking = (
        config.ranking_mode == "no_ranking"
        and config.rank_override == 0.0
        and math.copysign(1.0, config.rank_override) == 1.0
    )
    if not (matched_ranking or no_ranking):
        raise DirectionalConfigError(
            "ranking_mode/rank_override must be exactly matched/None or no_ranking/0.0"
        )
    if not isinstance(config.optimizer, DirectionalOptimizerConfig):
        raise DirectionalConfigError("optimizer must be a DirectionalOptimizerConfig")
    _validate_optimizer(config.optimizer)
    if (config.optimizer.lr, config.optimizer.weight_decay) != _STAGE_OPTIMIZERS[
        config.stage
    ]:
        raise DirectionalConfigError(
            "optimizer does not match the frozen stage profile"
        )
    _bool("non_scientific_smoke", config.non_scientific_smoke)
    if config.non_scientific_smoke and config.stage not in _SMOKE_STAGES:
        raise DirectionalConfigError(
            "non-scientific smoke is permitted only for the bridge warm-up stage"
        )

    _exact_int(
        "max_steps",
        config.max_steps,
        _SMOKE_STEPS if config.non_scientific_smoke else _STAGE_STEPS[config.stage],
    )
    if not isinstance(config.preserve_router_optimizer_state, bool):
        raise DirectionalConfigError(
            "preserve_router_optimizer_state must be a boolean"
        )
    if config.preserve_router_optimizer_state is not (
        config.stage is DirectionalStage.JOINT_ADAPTATION
    ):
        raise DirectionalConfigError(
            "router optimizer preservation must match the declared stage"
        )

def _validate_confirmation_config(config: DirectionalConfirmationConfig) -> None:
    _exact_int("bootstrap_seed", config.bootstrap_seed, 42)
    _exact_int("bootstrap_resamples", config.bootstrap_resamples, 2000)
    _exact_int("parent_groups_per_family", config.parent_groups_per_family, 20)
    _exact_int_tuple("generation.seeds", config.generation_seeds, (42001, 42002))
    _exact_str_tuple("generation.arms", config.generation_arms, _GENERATION_ARMS)
    _exact_int("planned_generation_cells", config.planned_generation_cells, 240)
    if not isinstance(config.development, DevelopmentGateConfig):
        raise DirectionalConfigError("development must be a DevelopmentGateConfig")
    if not isinstance(config.blind, BlindGateConfig):
        raise DirectionalConfigError("blind must be a BlindGateConfig")
    _validate_development_gate_values(config.development)
    _validate_blind_gate_values(config.blind)

def _validate_development_gate_values(config: DevelopmentGateConfig) -> None:
    _required_true(
        "development.require_counterfactual_validity",
        config.require_counterfactual_validity,
    )
    _required_true(
        "development.require_joint_preservation", config.require_joint_preservation
    )
    _exact_int(
        "development.max_weak_residual_families", config.max_weak_residual_families, 1
    )
    _exact_float(
        "development.min_directional_residual_norm",
        config.min_directional_residual_norm,
        0.05,
    )
    _exact_float(
        "development.max_universal_occupancy", config.max_universal_occupancy, 0.90
    )
    _exact_float(
        "development.min_non_none_occupancy", config.min_non_none_occupancy, 0.05
    )
    _exact_int("development.min_non_none_states", config.min_non_none_states, 2)
    _exact_int(
        "development.min_families_per_non_none_state",
        config.min_families_per_non_none_state,
        2,
    )
    _exact_float("development.min_adaptive_gain", config.min_adaptive_gain, 0.02)
    _exact_float(
        "development.min_oracle_gap_recovery", config.min_oracle_gap_recovery, 0.30
    )
    _required_true(
        "development.require_rerun_bootstrap_reproduction",
        config.require_rerun_bootstrap_reproduction,
    )

def _validate_blind_gate_values(config: BlindGateConfig) -> None:
    _exact_int(
        "blind.min_families_with_distinct_optima",
        config.min_families_with_distinct_optima,
        2,
    )
    _exact_float(
        "blind.min_winner_over_worst_gain", config.min_winner_over_worst_gain, 0.02
    )
    _exact_float("blind.min_oracle_example_gain", config.min_oracle_example_gain, 0.05)
    _exact_float(
        "blind.min_oracle_example_fraction", config.min_oracle_example_fraction, 0.20
    )
    _exact_float("blind.min_adaptive_gain", config.min_adaptive_gain, 0.02)
    _exact_float("blind.min_oracle_gap_recovery", config.min_oracle_gap_recovery, 0.30)
    _required_true(
        "blind.require_counterfactual_validity", config.require_counterfactual_validity
    )
    _required_true(
        "blind.require_joint_preservation", config.require_joint_preservation
    )
    _required_true("blind.require_finite_generation", config.require_finite_generation)
    _required_true(
        "blind.require_paired_cell_completeness",
        config.require_paired_cell_completeness,
    )
    _required_true(
        "blind.require_preblind_seed_rerun_agreement",
        config.require_preblind_seed_rerun_agreement,
    )

def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise DirectionalConfigError(
            f"cannot load directional config {path}: {error}"
        ) from error
    return _mapping(str(path), value)

def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DirectionalConfigError(f"{name} must be a mapping with string keys")
    return value

def _exact_keys(
    name: str,
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    unknown = set(value) - allowed - optional
    missing = allowed - set(value)
    if unknown:
        raise DirectionalConfigError(f"{name} has unknown key(s) {sorted(unknown)}")
    if missing:
        raise DirectionalConfigError(f"{name} is missing key(s) {sorted(missing)}")

def _int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DirectionalConfigError(f"{name} must be an integer, got {value!r}")
    return value

def _exact_int(name: str, value: object, expected: int) -> int:
    parsed = _int(name, value)
    if parsed != expected:
        raise DirectionalConfigError(
            f"{name} must equal frozen value {expected}, got {parsed}"
        )
    return parsed

def _float(name: str, value: object) -> float:
    if not isinstance(value, float):
        raise DirectionalConfigError(f"{name} must be a float, got {value!r}")
    return value

def _exact_float(name: str, value: object, expected: float) -> float:
    parsed = _float(name, value)
    if parsed != expected:
        raise DirectionalConfigError(
            f"{name} must equal frozen value {expected}, got {parsed}"
        )
    return parsed

def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise DirectionalConfigError(f"{name} must be a boolean, got {value!r}")
    return value

def _required_true(name: str, value: object) -> bool:
    if not _bool(name, value):
        raise DirectionalConfigError(
            f"{name} must remain true in the frozen gate contract"
        )
    return True

def _string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise DirectionalConfigError(f"{name} must be a string, got {value!r}")
    return value

def _exact_str(name: str, value: object, expected: str) -> str:
    parsed = _string(name, value)
    if parsed != expected:
        raise DirectionalConfigError(
            f"{name} must equal frozen value {expected!r}, got {parsed!r}"
        )
    return parsed

def _sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DirectionalConfigError(f"{name} must be a sequence")
    return value

def _int_tuple(name: str, value: object) -> tuple[int, ...]:
    return tuple(
        _int(f"{name}[{index}]", item)
        for index, item in enumerate(_sequence(name, value))
    )

def _exact_int_tuple(
    name: str, value: object, expected: tuple[int, int]
) -> tuple[int, int]:
    parsed = tuple(
        _int(f"{name}[{index}]", item)
        for index, item in enumerate(_frozen_tuple(name, value))
    )
    if parsed != expected:
        raise DirectionalConfigError(
            f"{name} must equal frozen value {expected}, got {parsed}"
        )
    return parsed

def _string_tuple(name: str, value: object) -> tuple[str, ...]:
    return tuple(
        _string(f"{name}[{index}]", item)
        for index, item in enumerate(_sequence(name, value))
    )

def _exact_str_tuple(
    name: str, value: object, expected: tuple[str, ...]
) -> tuple[str, ...]:
    parsed = tuple(
        _string(f"{name}[{index}]", item)
        for index, item in enumerate(_frozen_tuple(name, value))
    )
    if parsed != expected:
        if any("blind" in item.lower() for item in parsed):
            raise DirectionalConfigError(
                f"blind partition or path hint is forbidden in {name}"
            )
        raise DirectionalConfigError(
            f"{name} must equal frozen value {expected}, got {parsed}"
        )
    return parsed

def _development_path(name: str, value: object) -> Path:
    path_hint = _string(name, value)
    if not path_hint:
        raise DirectionalConfigError(
            f"{name} must be a non-empty development path hint"
        )
    if "blind" in path_hint.lower():
        raise DirectionalConfigError(
            f"blind path hint is forbidden in {name}: {path_hint}"
        )
    return Path(path_hint)

def _direct_path(name: str, value: object, expected: str) -> None:
    if not isinstance(value, Path):
        raise DirectionalConfigError(f"{name} must be a Path")
    path_hint = str(value)
    if "blind" in path_hint.lower():
        raise DirectionalConfigError(
            f"blind path hint is forbidden in {name}: {path_hint}"
        )
    if path_hint != expected:
        raise DirectionalConfigError(
            f"{name} must equal frozen path hint {expected!r}, got {path_hint!r}"
        )

def _stage(value: object) -> DirectionalStage:
    if not isinstance(value, str):
        raise DirectionalConfigError(f"stage must be a string, got {value!r}")
    try:
        return DirectionalStage(value)
    except ValueError as error:
        raise DirectionalConfigError(f"unknown directional stage {value!r}") from error

def _optimizer(value: object) -> DirectionalOptimizerConfig:
    mapping = _mapping("optimizer", value)
    _exact_keys("optimizer", mapping, _OPTIMIZER_KEYS)
    return DirectionalOptimizerConfig(
        name=_string("optimizer.name", mapping["name"]),
        lr=_float("optimizer.lr", mapping["lr"]),
        weight_decay=_float("optimizer.weight_decay", mapping["weight_decay"]),
        betas=_float_tuple("optimizer.betas", mapping["betas"]),
        grad_clip_norm=_float("optimizer.grad_clip_norm", mapping["grad_clip_norm"]),
    )

def _float_tuple(name: str, value: object) -> tuple[float, float]:
    parsed = tuple(
        _float(f"{name}[{index}]", item)
        for index, item in enumerate(_sequence(name, value))
    )
    if len(parsed) != 2:
        raise DirectionalConfigError(f"{name} must be a two-float sequence")
    return parsed

def _exact_float_tuple(
    name: str, value: object, expected: tuple[float, float]
) -> tuple[float, float]:
    parsed = tuple(
        _float(f"{name}[{index}]", item)
        for index, item in enumerate(_frozen_tuple(name, value))
    )
    if parsed != expected:
        raise DirectionalConfigError(
            f"{name} must equal frozen value {expected}, got {parsed}"
        )
    return parsed

def _frozen_tuple(name: str, value: object) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise DirectionalConfigError(f"{name} must be an exact tuple")
    return value

def _development_gates(value: object) -> DevelopmentGateConfig:
    mapping = _mapping("development", value)
    _exact_keys("development", mapping, _DEVELOPMENT_KEYS)
    return DevelopmentGateConfig(
        require_counterfactual_validity=_bool(
            "development.require_counterfactual_validity",
            mapping["require_counterfactual_validity"],
        ),
        require_joint_preservation=_bool(
            "development.require_joint_preservation",
            mapping["require_joint_preservation"],
        ),
        max_weak_residual_families=_int(
            "development.max_weak_residual_families",
            mapping["max_weak_residual_families"],
        ),
        min_directional_residual_norm=_float(
            "development.min_directional_residual_norm",
            mapping["min_directional_residual_norm"],
        ),
        max_universal_occupancy=_float(
            "development.max_universal_occupancy", mapping["max_universal_occupancy"]
        ),
        min_non_none_occupancy=_float(
            "development.min_non_none_occupancy", mapping["min_non_none_occupancy"]
        ),
        min_non_none_states=_int(
            "development.min_non_none_states", mapping["min_non_none_states"]
        ),
        min_families_per_non_none_state=_int(
            "development.min_families_per_non_none_state",
            mapping["min_families_per_non_none_state"],
        ),
        min_adaptive_gain=_float(
            "development.min_adaptive_gain", mapping["min_adaptive_gain"]
        ),
        min_oracle_gap_recovery=_float(
            "development.min_oracle_gap_recovery", mapping["min_oracle_gap_recovery"]
        ),
        require_rerun_bootstrap_reproduction=_bool(
            "development.require_rerun_bootstrap_reproduction",
            mapping["require_rerun_bootstrap_reproduction"],
        ),
    )

def _blind_gates(value: object) -> BlindGateConfig:
    mapping = _mapping("blind", value)
    _exact_keys("blind", mapping, _BLIND_KEYS)
    return BlindGateConfig(
        min_families_with_distinct_optima=_int(
            "blind.min_families_with_distinct_optima",
            mapping["min_families_with_distinct_optima"],
        ),
        min_winner_over_worst_gain=_float(
            "blind.min_winner_over_worst_gain", mapping["min_winner_over_worst_gain"]
        ),
        min_oracle_example_gain=_float(
            "blind.min_oracle_example_gain", mapping["min_oracle_example_gain"]
        ),
        min_oracle_example_fraction=_float(
            "blind.min_oracle_example_fraction", mapping["min_oracle_example_fraction"]
        ),
        min_adaptive_gain=_float(
            "blind.min_adaptive_gain", mapping["min_adaptive_gain"]
        ),
        min_oracle_gap_recovery=_float(
            "blind.min_oracle_gap_recovery", mapping["min_oracle_gap_recovery"]
        ),
        require_counterfactual_validity=_bool(
            "blind.require_counterfactual_validity",
            mapping["require_counterfactual_validity"],
        ),
        require_joint_preservation=_bool(
            "blind.require_joint_preservation", mapping["require_joint_preservation"]
        ),
        require_finite_generation=_bool(
            "blind.require_finite_generation", mapping["require_finite_generation"]
        ),
        require_paired_cell_completeness=_bool(
            "blind.require_paired_cell_completeness",
            mapping["require_paired_cell_completeness"],
        ),
        require_preblind_seed_rerun_agreement=_bool(
            "blind.require_preblind_seed_rerun_agreement",
            mapping["require_preblind_seed_rerun_agreement"],
        ),
    )

def _assert_contract_matches_frozen(
    frozen: EmergentContract, caller: EmergentContract
) -> None:
    for field in fields(EmergentContract):
        if getattr(caller, field.name) != getattr(frozen, field.name):
            raise DirectionalConfigError(
                f"caller contract {field.name} differs from frozen resources"
            )
    for role, path in (
        ("common", caller.common_checkpoint),
        ("autoencoder", caller.autoencoder_checkpoint),
    ):
        if _file_identity(path) != _CHECKPOINT_IDENTITIES_V1[role]:
            raise DirectionalConfigError(
                f"{role} checkpoint does not match reviewed identity v1"
            )

def _file_identity(path: Path) -> tuple[int, str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return path.stat().st_size, digest.hexdigest()
    except OSError as error:
        raise DirectionalConfigError(
            f"checkpoint is not readable for identity check: {path}: {error}"
        ) from error

def _assert_basis_commit(root: Path) -> None:
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", _DIRECTIONAL_BASIS_COMMIT, "HEAD"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise DirectionalConfigError(
            f"DIVE checkout does not contain required basis commit {_DIRECTIONAL_BASIS_COMMIT}"
        )

def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise DirectionalConfigError(
            f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}"
        )
    return result.stdout.strip()

def _assert_clean_committed_repo(root: Path, label: str) -> None:
    if not root.is_dir():
        raise DirectionalConfigError(f"{label} repository is missing: {root}")
    if _git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise DirectionalConfigError(f"{label} root is not a git work tree: {root}")
    _git(root, "rev-parse", "HEAD^{commit}")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise DirectionalConfigError(f"{label} repository is dirty: {root}")
