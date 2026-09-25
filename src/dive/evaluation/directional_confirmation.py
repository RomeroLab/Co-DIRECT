
from __future__ import annotations

from dive.codirect_paths import joined

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import posixpath
import random
from types import MappingProxyType

from dive.evaluation.directional_checks import (
    CHECK_SCHEMA_VERSION,
    DirectionalCheckError,
    counterfactual_validity,
    joint_preservation,
    validate_check_inventory,
)

class DirectionalConfirmationError(ValueError):
    pass

FAMILIES = ("binder", "ame", "antibody")
STRICT_FLOW_ROW_COUNTS = MappingProxyType({"binder": 198, "ame": 292, "antibody": 1106})
STRICT_FLOW_PARENT_COUNTS = MappingProxyType(
    {"binder": 131, "ame": 120, "antibody": 454}
)
HARD_ROUTES = ("none", "z_to_x", "x_to_z", "bidirectional")
ALL_ROUTES = ("adaptive", *HARD_ROUTES)
NON_NONE_ROUTES = ("z_to_x", "x_to_z", "bidirectional")
TAPS = ("tap_0", "tap_1", "tap_2")
DIRECTIONS = ("z_to_x", "x_to_z")
EMERGENT_UPSTREAM_COMMIT = "32b71ae1d9a8767414eae3921cf35969430db82f"
COMMON_CHECKPOINT_PATH = joined('PROTEINA_COMPLEXA_ROOT', 'ckpts', 'complexa.ckpt')
COMMON_CHECKPOINT_HASH = (
    "589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9"
)
COMMON_CHECKPOINT_SIZE = 2_934_289_381
TIME_BIN_SCHEME = "dive-directional-cartesian-time-bin-v1"
TIME_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)

MIN_DIRECTIONAL_RESIDUAL_NORM = 0.05
MAX_UNIVERSAL_OCCUPANCY = 0.90
MIN_NON_NONE_OCCUPANCY = 0.05
MIN_ADAPTIVE_GAIN = 0.02
MIN_ORACLE_GAP_RECOVERY = 0.30
MIN_WINNER_OVER_WORST_GAIN = 0.02
MIN_ORACLE_EXAMPLE_GAIN = 0.05
MIN_ORACLE_EXAMPLE_FRACTION = 0.20

DEVELOPMENT_GATE_NAMES = (
    "condition_1_validity_and_preservation",
    "condition_2_directional_residual_signal",
    "condition_3_route_occupancy_diversity",
    "condition_4_adaptive_gain",
    "condition_5_oracle_gap_recovery",
    "condition_6_rerun_reproduction",
)

BLIND_GATE_NAMES = (
    "condition_1_family_crossover",
    "condition_2_winner_over_worst_spread",
    "condition_3_oracle_example_gain",
    "condition_4_learned_adaptive_gain",
    "condition_5_blind_oracle_gap_recovery",
    "condition_6_validity_generation_and_pairing",
    "condition_7_preblind_agreement_binding",
)

@dataclass(frozen=True, slots=True)
class DevelopmentVerdict:

    passed: bool
    reasons: tuple[str, ...]
    conditions: Mapping[str, bool]
    diagnostics: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class BlindVerdict:

    passed: bool
    reasons: tuple[str, ...]
    conditions: Mapping[str, bool]
    terminal: bool
    claim_label: str
    diagnostics: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class CandidateSelection:

    seed: int
    checkpoint_path: str
    checkpoint_hash: str
    checkpoint_size_bytes: int
    parent_checkpoint_path: str
    parent_checkpoint_hash: str
    parent_checkpoint_size_bytes: int
    common_checkpoint_path: str
    common_checkpoint_hash: str
    common_checkpoint_size_bytes: int
    config_path: str
    provenance_hash: str
    provenance_path: str
    provenance_size_bytes: int
    training_completion_hash: str
    training_completion_path: str
    training_completion_size_bytes: int
    checkpoint_completion_hash: str
    checkpoint_completion_path: str
    checkpoint_completion_size_bytes: int
    validation_record_hash: str
    validation_record_path: str
    validation_record_size_bytes: int
    split_hash: str
    parent_projection_hash: str
    parent_projection_file_hash: str
    parent_projection_spec_hash: str
    resume_hash: str
    ownership_hash: str
    validation_cell_hash: str
    example_statistics_hash: str
    config_hash: str
    repo_commit: str
    upstream_commit: str
    stochastic_budget_hash: str
    confirmation_hash: str
    equal_family_relative_validation_flow_loss: float
    bootstrap_interval: tuple[float, float]
    directional_conclusion: tuple[tuple[str, str], ...]

    def to_mapping(self) -> dict[str, object]:

        return {
            "seed": self.seed,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "parent_checkpoint_path": self.parent_checkpoint_path,
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
            "parent_checkpoint_size_bytes": self.parent_checkpoint_size_bytes,
            "common_checkpoint_path": self.common_checkpoint_path,
            "common_checkpoint_hash": self.common_checkpoint_hash,
            "common_checkpoint_size_bytes": self.common_checkpoint_size_bytes,
            "config_path": self.config_path,
            "provenance_hash": self.provenance_hash,
            "provenance_path": self.provenance_path,
            "provenance_size_bytes": self.provenance_size_bytes,
            "training_completion_hash": self.training_completion_hash,
            "training_completion_path": self.training_completion_path,
            "training_completion_size_bytes": self.training_completion_size_bytes,
            "checkpoint_completion_hash": self.checkpoint_completion_hash,
            "checkpoint_completion_path": self.checkpoint_completion_path,
            "checkpoint_completion_size_bytes": self.checkpoint_completion_size_bytes,
            "validation_record_hash": self.validation_record_hash,
            "validation_record_path": self.validation_record_path,
            "validation_record_size_bytes": self.validation_record_size_bytes,
            "split_hash": self.split_hash,
            "parent_projection_hash": self.parent_projection_hash,
            "parent_projection_file_hash": self.parent_projection_file_hash,
            "parent_projection_spec_hash": self.parent_projection_spec_hash,
            "resume_hash": self.resume_hash,
            "ownership_hash": self.ownership_hash,
            "validation_cell_hash": self.validation_cell_hash,
            "example_statistics_hash": self.example_statistics_hash,
            "config_hash": self.config_hash,
            "repo_commit": self.repo_commit,
            "upstream_commit": self.upstream_commit,
            "stochastic_budget_hash": self.stochastic_budget_hash,
            "confirmation_hash": self.confirmation_hash,
            "equal_family_relative_validation_flow_loss": (
                self.equal_family_relative_validation_flow_loss
            ),
            "bootstrap_interval": list(self.bootstrap_interval),
            "directional_conclusion": [
                [family, route] for family, route in self.directional_conclusion
            ],
        }

@dataclass(frozen=True, slots=True)
class _CandidateMetrics:
    candidate: Mapping[str, object]
    records: tuple[Mapping[str, object], ...]
    first_five: tuple[bool, bool, bool, bool, bool]
    primary_metric: float
    bootstrap_interval: tuple[float, float]
    directional_conclusion: tuple[tuple[str, str], ...]
    checks_passed: bool
    check_diagnostics: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class _CheckMetrics:
    passed: bool
    validity_passed: bool
    preservation_passed: bool
    diagnostics: Mapping[str, object]

_CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "checkpoint_path",
        "checkpoint_hash",
        "checkpoint_size_bytes",
        "parent_checkpoint_path",
        "parent_checkpoint_hash",
        "parent_checkpoint_size_bytes",
        "common_checkpoint_path",
        "common_checkpoint_hash",
        "common_checkpoint_size_bytes",
        "config_path",
        "provenance_hash",
        "provenance_path",
        "provenance_size_bytes",
        "training_completion_hash",
        "training_completion_path",
        "training_completion_size_bytes",
        "checkpoint_completion_hash",
        "checkpoint_completion_path",
        "checkpoint_completion_size_bytes",
        "validation_record_hash",
        "validation_record_path",
        "validation_record_size_bytes",
        "split_hash",
        "parent_projection_hash",
        "parent_projection_file_hash",
        "parent_projection_spec_hash",
        "resume_hash",
        "ownership_hash",
        "validation_cell_hash",
        "example_statistics_hash",
        "config_hash",
        "repo_commit",
        "upstream_commit",
        "stochastic_budget_hash",
        "lineage",
        "confirmation",
        "expected_cells",
        "example_statistics",
        "check_schema_version",
        "check_records",
        "check_records_hash",
    }
)
_LEGACY_CHECK_KEYS = frozenset(
    {"counterfactual_validity_passed", "joint_preservation_passed"}
)
_IDENTITY_KEYS = frozenset({"path", "sha256", "size_bytes"})
_LINEAGE_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint",
        "parent_checkpoint",
        "common_checkpoint",
        "provenance",
        "provenance_hash",
        "training_completion",
        "training_completion_hash",
        "checkpoint_completion",
        "checkpoint_completion_hash",
        "validation_record",
        "validation_record_hash",
        "resume_state",
        "resume_hash",
        "ownership",
        "ownership_hash",
        "stochastic_budget",
        "stochastic_budget_hash",
    }
)
_RERUN_EXTRA_KEYS = frozenset(
    {
        "selected_checkpoint_path",
        "selected_checkpoint_hash",
        "selected_candidate_hash",
        "selected_selection_hash",
        "expected_bootstrap_interval",
    }
)
_EXAMPLE_KEYS = frozenset(
    {
        "family",
        "example_id",
        "parent_id",
        "cell_hash",
        "family_hash",
        "example_id_hash",
        "parent_id_hash",
        "identity_hash",
        "route_losses",
        "per_residue_oracle",
        "corruption_time",
        "time_bin",
        "tap_probabilities",
        "tap_residuals",
        "derivable_controls",
        "non_finite_count",
        "non_finite_events",
    }
)
_EXPECTED_CELL_KEYS = frozenset({"family", "example_id", "parent_id", "cell_hash"})
_SELECTION_KEYS = frozenset(CandidateSelection.__dataclass_fields__)
_ABLATION_KEYS = frozenset(
    {
        "bridge_removed",
        "shuffled_router",
        "fixed_joint",
        "best_fixed_time_only",
        "no_ranking",
    }
)
_BLIND_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "confirmation",
        "seal_payload",
        "seal_hash",
        "preblind",
        "strict_flow_inventory",
        "flow_examples",
        "blind_checks",
        "generation_inventory",
        "availability_manifest",
        "generation_cells",
    }
)
_PREBLIND_KEYS = frozenset(
    {
        "development_report",
        "development_report_hash",
        "development_verdict",
        "development_verdict_hash",
        "selection",
        "selection_hash",
        "rerun",
        "rerun_hash",
        "seal_payload",
        "seal_hash",
    }
)
_STRICT_INVENTORY_KEYS = frozenset({"schema_version", "rows", "rows_hash"})
_GENERATION_INVENTORY_KEYS = frozenset(
    {
        "schema_version",
        "parents_by_family",
        "parent_inventory_hash",
        "seeds",
        "arms",
        "preregistered_cells",
        "preregistered_cells_hash",
        "availability_manifest_hash",
    }
)
_GENERATION_IDENTITY_KEYS = frozenset(
    {"family", "parent_id", "seed", "arm", "cell_hash"}
)
_GENERATION_CELL_KEYS = _GENERATION_IDENTITY_KEYS | frozenset(
    {"loss", "non_finite_count", "non_finite_events"}
)
_SEAL_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "development_report_hash",
        "development_verdict_hash",
        "selection_hash",
        "rerun_hash",
        "ablations_hash",
        "strict_flow_inventory_hash",
        "generation_inventory_hash",
        "availability_manifest_hash",
    }
)
_AVAILABILITY_MANIFEST_KEYS = frozenset({"schema_version", "deterministic_exclusions"})
METRIC_SOURCE_PATHS = (
    "src/dive/evaluation/directional_confirmation.py",
    "src/dive/evaluation/directional_checks.py",
)

def _frozen_confirmation() -> dict[str, object]:

    return {
        "bootstrap_seed": 42,
        "bootstrap_resamples": 2000,
        "parent_groups_per_family": 20,
        "generation": {
            "seeds": [42001, 42002],
            "arms": ["adaptive", "best_frozen_fixed_control"],
        },
        "planned_generation_cells": 240,
        "development": {
            "require_counterfactual_validity": True,
            "require_joint_preservation": True,
            "max_weak_residual_families": 1,
            "min_directional_residual_norm": 0.05,
            "max_universal_occupancy": 0.90,
            "min_non_none_occupancy": 0.05,
            "min_non_none_states": 2,
            "min_families_per_non_none_state": 2,
            "min_adaptive_gain": 0.02,
            "min_oracle_gap_recovery": 0.30,
            "require_rerun_bootstrap_reproduction": True,
        },
        "blind": {
            "min_families_with_distinct_optima": 2,
            "min_winner_over_worst_gain": 0.02,
            "min_oracle_example_gain": 0.05,
            "min_oracle_example_fraction": 0.20,
            "min_adaptive_gain": 0.02,
            "min_oracle_gap_recovery": 0.30,
            "require_counterfactual_validity": True,
            "require_joint_preservation": True,
            "require_finite_generation": True,
            "require_paired_cell_completeness": True,
            "require_preblind_seed_rerun_agreement": True,
        },
    }

def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise DirectionalConfirmationError(
            f"aggregate is not canonical JSON: {error}"
        ) from error

def metric_source_inventory(
    identities: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:

    sequence = _sequence("metric source identities", identities)
    if len(sequence) != len(METRIC_SOURCE_PATHS):
        raise DirectionalConfirmationError(
            "metric source identities must contain exactly both reducer sources"
        )
    sources: list[dict[str, object]] = []
    for index, expected_path in enumerate(METRIC_SOURCE_PATHS):
        name = f"metric source identities[{index}]"
        identity = _mapping(name, sequence[index])
        _exact_keys(name, identity, frozenset({"path", "sha256", "size_bytes"}))
        if _string(f"{name}.path", identity["path"]) != expected_path:
            raise DirectionalConfirmationError(f"{name}.path is not the fixed source")
        source_hash = _hash(f"{name}.sha256", identity["sha256"])
        size = _exact_int(f"{name}.size_bytes", identity["size_bytes"])
        if size <= 0:
            raise DirectionalConfirmationError(f"{name}.size_bytes must be positive")
        sources.append(
            {"path": expected_path, "sha256": source_hash, "size_bytes": size}
        )
    payload = {
        "schema_version": 1,
        "check_schema_version": CHECK_SCHEMA_VERSION,
        "sources": sources,
    }
    return MappingProxyType(
        {
            "schema_version": 1,
            "check_schema_version": CHECK_SCHEMA_VERSION,
            "sources": tuple(MappingProxyType(source) for source in sources),
            "inventory_hash": hashlib.sha256(_canonical(payload)).hexdigest(),
        }
    )

def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DirectionalConfirmationError(f"{name} must be a mapping")
    return value

def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DirectionalConfirmationError(f"{name} must be a sequence")
    return value

def _exact_keys(
    name: str, value: Mapping[str, object], expected: frozenset[str]
) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise DirectionalConfirmationError(f"{name} schema: {'; '.join(details)}")

def _exact_keys_with_optional(
    name: str,
    value: Mapping[str, object],
    required: frozenset[str],
    optional: frozenset[str],
) -> None:
    observed = set(value)
    missing = sorted(required - observed)
    unknown = sorted(observed - required - optional)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise DirectionalConfirmationError(f"{name} schema: {'; '.join(details)}")

def _exact_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise DirectionalConfirmationError(f"{name} must be an integer, not bool")
    return value

def _exact_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise DirectionalConfirmationError(f"{name} must be a boolean")
    return value

def _float(name: str, value: object, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise DirectionalConfirmationError(f"{name} must be a number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise DirectionalConfirmationError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise DirectionalConfirmationError(f"{name} must not be negative")
    return result

def _inclusive_at_least(value: float, threshold: float) -> bool:
    return value >= threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )

def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise DirectionalConfirmationError(f"{name} must be a non-empty string")
    return value

def _validate_ranking_pairing(
    name: str,
    ranking_mode: object,
    rank_override: object,
) -> tuple[str, float | None]:
    parsed_mode = _string(f"{name}.ranking_mode", ranking_mode)
    matched = parsed_mode == "matched" and rank_override is None
    no_ranking = (
        parsed_mode == "no_ranking"
        and type(rank_override) is float
        and rank_override == 0.0
        and math.copysign(1.0, rank_override) == 1.0
    )
    if not (matched or no_ranking):
        raise DirectionalConfirmationError(
            f"{name}.ranking_mode/rank_override is not a frozen pairing"
        )
    return parsed_mode, rank_override

def _validate_ranking_record(name: str, value: object) -> tuple[str, float | None]:
    record = _mapping(name, value)
    missing = sorted({"ranking_mode", "rank_override"} - set(record))
    if missing:
        raise DirectionalConfirmationError(f"{name} schema: missing {missing}")
    return _validate_ranking_pairing(
        name,
        record["ranking_mode"],
        record["rank_override"],
    )

def _hash(name: str, value: object, *, length: int = 64) -> str:
    text = _string(name, value)
    if len(text) != length or any(char not in "0123456789abcdef" for char in text):
        raise DirectionalConfirmationError(
            f"{name} must be a lowercase {length}-hex identity"
        )
    return text

def _assert_deep_exact(name: str, value: object, expected: object) -> None:
    if isinstance(expected, dict):
        mapping = _mapping(name, value)
        _exact_keys(name, mapping, frozenset(expected))
        for key, item in expected.items():
            _assert_deep_exact(f"{name}.{key}", mapping[key], item)
        return
    if isinstance(expected, list):
        sequence = _sequence(name, value)
        if len(sequence) != len(expected):
            raise DirectionalConfirmationError(f"{name} length drifted")
        for index, item in enumerate(expected):
            _assert_deep_exact(f"{name}[{index}]", sequence[index], item)
        return
    if type(value) is not type(expected) or value != expected:
        raise DirectionalConfirmationError(
            f"{name} differs from the frozen confirmation configuration"
        )

def _assert_confirmation(value: object) -> Mapping[str, object]:
    expected = _frozen_confirmation()
    _assert_deep_exact("confirmation", value, expected)
    return _mapping("confirmation", value)

def _sufficient(
    name: str,
    value: object,
    *,
    expected_denominator: int | None = None,
) -> tuple[float, int]:
    mapping = _mapping(name, value)
    _exact_keys(name, mapping, frozenset({"numerator", "denominator"}))
    numerator = _float(f"{name}.numerator", mapping["numerator"], nonnegative=True)
    denominator = _exact_int(f"{name}.denominator", mapping["denominator"])
    if denominator <= 0:
        raise DirectionalConfirmationError(f"{name}.denominator must be positive")
    if expected_denominator is not None and denominator != expected_denominator:
        raise DirectionalConfirmationError(
            f"{name}.denominator makes the validation cell unpaired"
        )
    return numerator, denominator

def _relative_residual_sufficient(
    name: str,
    value: object,
    *,
    expected_denominator: int,
) -> tuple[float, float, int]:
    mapping = _mapping(name, value)
    _exact_keys(
        name,
        mapping,
        frozenset(
            {
                "applied_residual_squared_sum",
                "joint_field_squared_norm_sum",
                "valid_denominator",
            }
        ),
    )
    applied = _float(
        f"{name}.applied_residual_squared_sum",
        mapping["applied_residual_squared_sum"],
        nonnegative=True,
    )
    joint = _float(
        f"{name}.joint_field_squared_norm_sum",
        mapping["joint_field_squared_norm_sum"],
        nonnegative=True,
    )
    if joint <= 0.0:
        raise DirectionalConfirmationError(
            f"{name}.joint_field_squared_norm_sum denominator must be positive"
        )
    denominator = _exact_int(f"{name}.valid_denominator", mapping["valid_denominator"])
    if denominator <= 0:
        raise DirectionalConfirmationError(f"{name}.valid_denominator must be positive")
    if denominator != expected_denominator:
        raise DirectionalConfirmationError(
            f"{name}.valid_denominator makes the validation cell unpaired"
        )
    return applied, joint, denominator

def _validate_time_record(record: Mapping[str, object], *, prefix: str) -> int:
    times = _mapping(f"{prefix}.corruption_time", record["corruption_time"])
    _exact_keys(
        f"{prefix}.corruption_time",
        times,
        frozenset({"bb_ca", "local_latents"}),
    )
    bb_ca = _float(f"{prefix}.corruption_time.bb_ca", times["bb_ca"])
    latent = _float(f"{prefix}.corruption_time.local_latents", times["local_latents"])
    if not TIME_BIN_EDGES[0] <= bb_ca <= TIME_BIN_EDGES[-1] or not (
        TIME_BIN_EDGES[0] <= latent <= TIME_BIN_EDGES[-1]
    ):
        raise DirectionalConfirmationError(f"{prefix}.corruption_time is outside bins")
    bin_record = _mapping(f"{prefix}.time_bin", record["time_bin"])
    _exact_keys(
        f"{prefix}.time_bin",
        bin_record,
        frozenset(
            {
                "scheme",
                "edges",
                "bb_ca_index",
                "local_latents_index",
                "index",
            }
        ),
    )
    if bin_record["scheme"] != TIME_BIN_SCHEME:
        raise DirectionalConfirmationError(f"{prefix}.time_bin scheme drifted")
    edges = _sequence(f"{prefix}.time_bin.edges", bin_record["edges"])
    if len(edges) != len(TIME_BIN_EDGES) or any(
        type(value) is not float or value != expected
        for value, expected in zip(edges, TIME_BIN_EDGES, strict=True)
    ):
        raise DirectionalConfirmationError(f"{prefix}.time_bin edges drifted")
    internal = TIME_BIN_EDGES[1:-1]
    expected_bb = bisect.bisect_left(internal, bb_ca)
    expected_latent = bisect.bisect_left(internal, latent)
    bb_index = _exact_int(f"{prefix}.time_bin.bb_ca_index", bin_record["bb_ca_index"])
    latent_index = _exact_int(
        f"{prefix}.time_bin.local_latents_index",
        bin_record["local_latents_index"],
    )
    index = _exact_int(f"{prefix}.time_bin.index", bin_record["index"])
    if (
        bb_index != expected_bb
        or latent_index != expected_latent
        or index != expected_bb * 5 + expected_latent
    ):
        raise DirectionalConfirmationError(f"{prefix}.time_bin is inconsistent")
    return index

def _validate_example(value: object, *, index: int) -> Mapping[str, object]:
    name = f"example_statistics[{index}]"
    record = _mapping(name, value)
    _exact_keys(name, record, _EXAMPLE_KEYS)
    family = _string(f"{name}.family", record["family"])
    if family not in FAMILIES:
        raise DirectionalConfirmationError(f"{name}.family is unknown")
    example_id = _string(f"{name}.example_id", record["example_id"])
    parent_id = _string(f"{name}.parent_id", record["parent_id"])
    cell_hash = _hash(f"{name}.cell_hash", record["cell_hash"])
    identity = {
        "family": family,
        "example_id": example_id,
        "parent_id": parent_id,
        "cell_hash": cell_hash,
    }
    expected_hashes = {
        "family_hash": hashlib.sha256(family.encode()).hexdigest(),
        "example_id_hash": hashlib.sha256(example_id.encode()).hexdigest(),
        "parent_id_hash": hashlib.sha256(parent_id.encode()).hexdigest(),
        "identity_hash": hashlib.sha256(_canonical(identity) + b"\n").hexdigest(),
    }
    for field, expected in expected_hashes.items():
        if _hash(f"{name}.{field}", record[field]) != expected:
            raise DirectionalConfirmationError(
                f"{name}.{field} identity is inconsistent"
            )

    route_losses = _mapping(f"{name}.route_losses", record["route_losses"])
    _exact_keys(f"{name}.route_losses", route_losses, frozenset(ALL_ROUTES))
    parsed_losses: dict[str, tuple[float, int]] = {}
    for route in ALL_ROUTES:
        parsed_losses[route] = _sufficient(
            f"{name}.route_losses.{route}", route_losses[route]
        )
    denominator = parsed_losses["adaptive"][1]
    if any(item[1] != denominator for item in parsed_losses.values()):
        raise DirectionalConfirmationError(f"{name}.route_losses are unpaired")
    oracle, oracle_denominator = _sufficient(
        f"{name}.per_residue_oracle",
        record["per_residue_oracle"],
        expected_denominator=denominator,
    )
    if oracle > min(parsed_losses[route][0] for route in HARD_ROUTES) + 1e-12:
        raise DirectionalConfirmationError(
            f"{name}.per_residue_oracle exceeds a hard route loss"
        )

    _validate_time_record(record, prefix=name)
    probabilities = _mapping(f"{name}.tap_probabilities", record["tap_probabilities"])
    _exact_keys(f"{name}.tap_probabilities", probabilities, frozenset(TAPS))
    for tap in TAPS:
        routes = _mapping(f"{name}.tap_probabilities.{tap}", probabilities[tap])
        _exact_keys(f"{name}.tap_probabilities.{tap}", routes, frozenset(HARD_ROUTES))
        means = []
        for route in HARD_ROUTES:
            numerator, _ = _sufficient(
                f"{name}.tap_probabilities.{tap}.{route}",
                routes[route],
                expected_denominator=denominator,
            )
            mean = numerator / denominator
            if mean > 1.0:
                raise DirectionalConfirmationError(
                    f"{name}.tap_probabilities.{tap}.{route} exceeds one"
                )
            means.append(mean)
        if not math.isclose(sum(means), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise DirectionalConfirmationError(
                f"{name}.tap_probabilities.{tap} do not sum to one"
            )

    residuals = _mapping(f"{name}.tap_residuals", record["tap_residuals"])
    _exact_keys(f"{name}.tap_residuals", residuals, frozenset(TAPS))
    for tap in TAPS:
        directions = _mapping(f"{name}.tap_residuals.{tap}", residuals[tap])
        _exact_keys(f"{name}.tap_residuals.{tap}", directions, frozenset(DIRECTIONS))
        for direction in DIRECTIONS:
            _relative_residual_sufficient(
                f"{name}.tap_residuals.{tap}.{direction}",
                directions[direction],
                expected_denominator=denominator,
            )

    derivable = _sequence(f"{name}.derivable_controls", record["derivable_controls"])
    if list(derivable) != [
        "per_residue_oracle",
        "best_fixed",
        "global",
        "time_only",
    ]:
        raise DirectionalConfirmationError(f"{name}.derivable_controls drifted")
    non_finite_count = _exact_int(
        f"{name}.non_finite_count", record["non_finite_count"]
    )
    events = _sequence(f"{name}.non_finite_events", record["non_finite_events"])
    if non_finite_count != 0 or events:
        raise DirectionalConfirmationError(f"{name} contains non-finite losses")
    return record

def _expected_cell(value: object, *, index: int) -> dict[str, str]:
    name = f"expected_cells[{index}]"
    record = _mapping(name, value)
    _exact_keys(name, record, _EXPECTED_CELL_KEYS)
    family = _string(f"{name}.family", record["family"])
    if family not in FAMILIES:
        raise DirectionalConfirmationError(f"{name}.family is unknown")
    return {
        "family": family,
        "example_id": _string(f"{name}.example_id", record["example_id"]),
        "parent_id": _string(f"{name}.parent_id", record["parent_id"]),
        "cell_hash": _hash(f"{name}.cell_hash", record["cell_hash"]),
    }

def _record_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()

def _flow_identities(
    records: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            str(record["family"]),
            str(record["example_id"]),
            str(record["parent_id"]),
            str(record["cell_hash"]),
        )
        for record in records
    )

def _validate_check_bundle(
    value: object,
    *,
    records: Sequence[Mapping[str, object]],
    name: str,
    schema_version: object = CHECK_SCHEMA_VERSION,
    expected_hash: object | None = None,
    legacy: Mapping[str, object] | None = None,
) -> _CheckMetrics:
    if (
        _exact_int(f"{name}.check_schema_version", schema_version)
        != CHECK_SCHEMA_VERSION
    ):
        raise DirectionalConfirmationError(
            f"{name}.check_schema_version must be exactly {CHECK_SCHEMA_VERSION}"
        )
    if type(value) is not list:
        raise DirectionalConfirmationError(
            f"{name}.check_records must be a built-in list"
        )
    if expected_hash is not None:
        recorded_hash = _hash(f"{name}.check_records_hash", expected_hash)
        if recorded_hash != _record_hash(value):
            raise DirectionalConfirmationError(
                f"{name}.check_records_hash is inconsistent with check_records"
            )
    try:
        validate_check_inventory(value, expected_identities=_flow_identities(records))
        validity_passed = all(
            counterfactual_validity(check)["failure_count"] == 0 for check in value
        )
        preservation = joint_preservation(value)
    except DirectionalCheckError as error:
        raise DirectionalConfirmationError(f"{name}.check_records: {error}") from error

    if legacy is not None:
        expected_legacy = {
            "counterfactual_validity_passed": validity_passed,
            "joint_preservation_passed": preservation.passed,
        }
        for field, recomputed in expected_legacy.items():
            if field in legacy:
                observed = _exact_bool(f"{name}.{field}", legacy[field])
                if observed is not recomputed:
                    raise DirectionalConfirmationError(
                        f"{name}.{field} derived value disagrees with raw checks"
                    )
    diagnostics = MappingProxyType(
        {
            "validity_passed": validity_passed,
            "preservation_passed": preservation.passed,
            "equal_family_ratio": preservation.equal_family_ratio,
            "family_ratios": tuple(preservation.family_ratios),
        }
    )
    return _CheckMetrics(
        passed=validity_passed and preservation.passed,
        validity_passed=validity_passed,
        preservation_passed=preservation.passed,
        diagnostics=diagnostics,
    )

def _absolute_path(name: str, value: object) -> str:
    path = _string(name, value)
    if (
        not path.startswith("/")
        or path == "/"
        or "\x00" in path
        or "//" in path
        or posixpath.normpath(path) != path
    ):
        raise DirectionalConfirmationError(f"{name} must be a normalized absolute path")
    return path

def _checkpoint_identity(name: str, value: object) -> Mapping[str, object]:
    record = _mapping(name, value)
    _exact_keys(name, record, _IDENTITY_KEYS)
    _absolute_path(f"{name}.path", record["path"])
    _hash(f"{name}.sha256", record["sha256"])
    size = _exact_int(f"{name}.size_bytes", record["size_bytes"])
    if size <= 0:
        raise DirectionalConfirmationError(f"{name}.size_bytes must be positive")
    return record

def _hashed_record(
    lineage: Mapping[str, object],
    *,
    record_name: str,
    prefix: str,
    hash_name: str | None = None,
) -> Mapping[str, object]:
    record = _mapping(f"{prefix}.{record_name}", lineage[record_name])
    hash_name = hash_name or f"{record_name}_hash"
    supplied = _hash(f"{prefix}.{hash_name}", lineage[hash_name])
    if supplied != _record_hash(record):
        raise DirectionalConfirmationError(
            f"{prefix}.{record_name}_hash is inconsistent with its record"
        )
    return record

def _validate_ownership(
    value: object, *, name: str, expected_stage: str
) -> Mapping[str, object]:
    record = _mapping(name, value)
    _exact_keys(
        name,
        record,
        frozenset(
            {
                "stage",
                "trainable_names",
                "trainable_name_count",
                "trainable_parameters",
                "groups",
            }
        ),
    )
    if record["stage"] != expected_stage:
        raise DirectionalConfirmationError(f"{name}.stage identity mismatch")
    names = tuple(
        _string(f"{name}.trainable_names[{index}]", item)
        for index, item in enumerate(
            _sequence(f"{name}.trainable_names", record["trainable_names"])
        )
    )
    if not names or names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise DirectionalConfirmationError(
            f"{name}.trainable_names must be non-empty, unique, and sorted"
        )
    count = _exact_int(f"{name}.trainable_name_count", record["trainable_name_count"])
    if count != len(names):
        raise DirectionalConfirmationError(
            f"{name}.trainable_name_count is inconsistent"
        )
    parameters = _exact_int(
        f"{name}.trainable_parameters", record["trainable_parameters"]
    )
    if parameters <= 0:
        raise DirectionalConfirmationError(
            f"{name}.trainable_parameters must be positive"
        )
    groups = _mapping(f"{name}.groups", record["groups"])
    expected_groups = (
        ("bridge", "router")
        if expected_stage == "bridge_warmup"
        else ("bridge", "lora", "router")
    )
    _exact_keys(f"{name}.groups", groups, frozenset(expected_groups))
    grouped: list[str] = []
    for group in expected_groups:
        values = tuple(
            _string(f"{name}.groups.{group}[{index}]", item)
            for index, item in enumerate(
                _sequence(f"{name}.groups.{group}", groups[group])
            )
        )
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise DirectionalConfirmationError(
                f"{name}.groups.{group} must be unique and sorted"
            )
        if not values or any(group not in value.lower() for value in values):
            raise DirectionalConfirmationError(
                f"{name}.groups.{group} does not contain exact {group} names"
            )
        grouped.extend(values)
    if sorted(grouped) != list(names) or len(grouped) != len(set(grouped)):
        raise DirectionalConfirmationError(f"{name}.groups do not partition ownership")
    if expected_stage == "bridge_warmup" and any(
        "lora" in value.lower() for value in names
    ):
        raise DirectionalConfirmationError(
            f"{name} bridge_warmup ownership must contain no LoRA names"
        )
    return record

def _terminal_artifact_identity(
    name: str, value: object, record: Mapping[str, object]
) -> Mapping[str, object]:
    identity = _checkpoint_identity(name, value)
    raw = _canonical(record) + b"\n"
    if identity["sha256"] != hashlib.sha256(raw).hexdigest():
        raise DirectionalConfirmationError(
            f"{name}.sha256 differs from canonical bytes"
        )
    if identity["size_bytes"] != len(raw):
        raise DirectionalConfirmationError(
            f"{name}.size_bytes differs from canonical bytes"
        )
    return identity

def _validate_directional_config(
    value: object, identity_value: object, *, name: str
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    config = _mapping(name, value)
    expected_keys = frozenset(
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
    _exact_keys(name, config, expected_keys)
    stage = _string(f"{name}.stage", config["stage"])
    ranking_mode, _ = _validate_ranking_record(name, config)
    if stage not in {"bridge_warmup", "joint_adaptation"}:
        raise DirectionalConfirmationError(f"{name}.stage is not directional")
    expected = {
        "resources": "configs/emergent/resources.yaml",
        "model": "configs/emergent/model.yaml",
        "data": "configs/emergent/data.yaml",
        "partitions": ["train", "validation", "legacy-dev"],
        "devices": 4,
        "device_type": "NVIDIA L40S",
        "precision": "bf16-mixed",
        "crop_size": 256,
        "microbatch_size": 2,
        "accumulate_grad_batches": 1,
        "effective_batch_size": 8,
        "seeds": [42, 314159],
        "route_equivalent_passes": 8.0,
        "dropout": 0.0,
        "optimizer": (
            {
                "name": "adamw",
                "lr": 0.0003,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "grad_clip_norm": 1.0,
            }
            if stage == "bridge_warmup"
            else {
                "name": "adamw",
                "lr": 0.0001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "grad_clip_norm": 1.0,
            }
        ),
        "max_steps": 10_000 if stage == "bridge_warmup" else 20_000,
        "preserve_router_optimizer_state": stage == "joint_adaptation",
    }
    for field, expected_value in expected.items():
        _assert_deep_exact(f"{name}.{field}", config[field], expected_value)

    identity = _checkpoint_identity(f"{name}_identity", identity_value)
    basename = str(identity["path"]).rsplit("/", 1)[-1]
    specs = {
        ("bridge_warmup", "matched"): (
            "directional_bridge_warmup.yaml",
            "3b132b616b37ba55c45ce52abd4404d0a43f1efc6df0f8b2f2e43f91feeccbaf",
            817,
        ),
        ("joint_adaptation", "matched"): (
            "directional_bridge_joint.yaml",
            "f7b10ac2e7c390347c9d1b94461a28eacf911af3f0d47229ef139084abd1076e",
            735,
        ),
        ("bridge_warmup", "no_ranking"): (
            "directional_bridge_no_ranking_warmup.yaml",
            "be7401bf0a3acd633ef0486f782b56d22aa7c812d1c6a746082ae354d1846b59",
            730,
        ),
        ("joint_adaptation", "no_ranking"): (
            "directional_bridge_no_ranking_joint.yaml",
            "a5d5c8afac99294d9bb81f95d37e41764f72cf194c070f679b6afe4c0515e724",
            730,
        ),
    }
    expected_basename, expected_hash, expected_size = specs[(stage, ranking_mode)]
    if (
        basename != expected_basename
        or identity["sha256"] != expected_hash
        or identity["size_bytes"] != expected_size
    ):
        raise DirectionalConfirmationError(
            f"{name}_identity does not bind the reviewed canonical config"
        )
    return config, identity

def _validate_candidate_lineage(
    candidate: Mapping[str, object],
    *,
    name: str,
    records: tuple[Mapping[str, object], ...],
    example_statistics_hash: str,
    validation_cell_hash: str,
) -> None:
    prefix = f"{name}.lineage"
    lineage = _mapping(prefix, candidate["lineage"])
    exact_lineage_keys = _LINEAGE_KEYS | frozenset(
        {
            "config",
            "config_identity",
            "provenance_identity",
            "training_completion_identity",
            "checkpoint_completion_identity",
            "validation_record_identity",
        }
    )
    _exact_keys(prefix, lineage, exact_lineage_keys)
    if _exact_int(f"{prefix}.schema_version", lineage["schema_version"]) != 1:
        raise DirectionalConfirmationError(f"{prefix}.schema_version must be exactly 1")

    checkpoint = _checkpoint_identity(f"{prefix}.checkpoint", lineage["checkpoint"])
    parent = _checkpoint_identity(
        f"{prefix}.parent_checkpoint", lineage["parent_checkpoint"]
    )
    common = _checkpoint_identity(
        f"{prefix}.common_checkpoint", lineage["common_checkpoint"]
    )
    if dict(common) != {
        "path": COMMON_CHECKPOINT_PATH,
        "sha256": COMMON_CHECKPOINT_HASH,
        "size_bytes": COMMON_CHECKPOINT_SIZE,
    }:
        raise DirectionalConfirmationError(f"{prefix}.common_checkpoint drifted")
    if checkpoint["path"] == parent["path"] or checkpoint["sha256"] == parent["sha256"]:
        raise DirectionalConfirmationError(f"{prefix}.checkpoint is not create-new")

    config, config_identity = _validate_directional_config(
        lineage["config"], lineage["config_identity"], name=f"{prefix}.config"
    )
    config_ranking = _validate_ranking_record(f"{prefix}.config", config)
    if config["stage"] != "joint_adaptation":
        raise DirectionalConfirmationError(f"{prefix}.config must be joint_adaptation")

    ownership = _hashed_record(lineage, record_name="ownership", prefix=prefix)
    _validate_ownership(
        ownership, name=f"{prefix}.ownership", expected_stage="joint_adaptation"
    )
    resume = _hashed_record(
        lineage, record_name="resume_state", hash_name="resume_hash", prefix=prefix
    )
    resume_keys = frozenset(
        {
            "schema_version",
            "stage",
            "seed",
            "repo_commit",
            "upstream_commit",
            "common_checkpoint_hash",
            "parent_checkpoint_hash",
            "split_hash",
            "config_hash",
            "trainable_name_hash",
            "route_equivalent_passes",
            "dropout",
            "ranking_mode",
            "rank_override",
            "optimizer_state_hash",
            "parent_projection_hash",
            "parent_projection_file_hash",
            "parent_projection_spec_hash",
        }
    )
    _exact_keys(f"{prefix}.resume_state", resume, resume_keys)
    resume_ranking = _validate_ranking_record(f"{prefix}.resume_state", resume)
    if (
        _exact_int(f"{prefix}.resume_state.schema_version", resume["schema_version"])
        != 3
    ):
        raise DirectionalConfirmationError(
            f"{prefix}.resume_state.schema_version must be exactly 3"
        )
    seed = _exact_int(f"{prefix}.resume_state.seed", resume["seed"])
    if seed != candidate["seed"] or resume["stage"] != "joint_adaptation":
        raise DirectionalConfirmationError(f"{prefix}.resume_state stage/seed mismatch")
    if resume_ranking != config_ranking:
        raise DirectionalConfirmationError(f"{prefix}.resume_state ranking mismatch")
    for field, length in (("repo_commit", 40), ("upstream_commit", 40)):
        _hash(f"{prefix}.resume_state.{field}", resume[field], length=length)
    for field in (
        "common_checkpoint_hash",
        "parent_checkpoint_hash",
        "split_hash",
        "config_hash",
        "trainable_name_hash",
        "optimizer_state_hash",
        "parent_projection_hash",
        "parent_projection_file_hash",
        "parent_projection_spec_hash",
    ):
        _hash(f"{prefix}.resume_state.{field}", resume[field])
    if (
        resume["common_checkpoint_hash"] != common["sha256"]
        or resume["parent_checkpoint_hash"] != parent["sha256"]
        or resume["config_hash"] != config_identity["sha256"]
        or resume["route_equivalent_passes"] != 8.0
        or resume["dropout"] != 0.0
    ):
        raise DirectionalConfirmationError(f"{prefix}.resume_state identity drifted")
    expected_name_hash = hashlib.sha256(
        ("\n".join(str(item) for item in ownership["trainable_names"]) + "\n").encode()
    ).hexdigest()
    if resume["trainable_name_hash"] != expected_name_hash:
        raise DirectionalConfirmationError(
            f"{prefix}.resume_state.trainable_name_hash is inconsistent"
        )

    stochastic = _hashed_record(lineage, record_name="stochastic_budget", prefix=prefix)
    _exact_keys(
        f"{prefix}.stochastic_budget",
        stochastic,
        frozenset(
            {
                "schema_version",
                "route_equivalent_passes",
                "dropout",
                "rng_scheme",
                "validation_cell_hash",
                "example_statistics_hash",
            }
        ),
    )
    if stochastic != {
        "schema_version": 1,
        "route_equivalent_passes": 8.0,
        "dropout": 0.0,
        "rng_scheme": "dive-directional-rng-v1",
        "validation_cell_hash": validation_cell_hash,
        "example_statistics_hash": example_statistics_hash,
    }:
        raise DirectionalConfirmationError(f"{prefix}.stochastic_budget mismatch")

    provenance = _mapping(f"{prefix}.provenance", lineage["provenance"])
    provenance_keys = frozenset(
        {
            "schema_version",
            "mode",
            "run_id",
            "seed",
            "stage",
            "repo_root",
            "repo_commit",
            "upstream_root",
            "upstream_commit",
            "config_path",
            "config_hash",
            "split_manifest_root",
            "split_hash",
            "loader_manifests",
            "parent_projection_hash",
            "parent_projection_file_hash",
            "parent_projection_spec_hash",
            "common_checkpoint",
            "common_checkpoint_hash",
            "autoencoder_checkpoint",
            "autoencoder_checkpoint_hash",
            "parent_checkpoint",
            "route_equivalent_passes",
            "dropout",
            "ranking_mode",
            "rank_override",
            "declared_devices",
            "uses_gpu",
            "loads_blind_data",
            "validation_interval",
            "validation_batches_per_family",
            "validation_metric_schema",
            "rng_scheme",
            "rng_contract",
            "launch_command_policy",
        }
    )
    _exact_keys(f"{prefix}.provenance", provenance, provenance_keys)
    _validate_ranking_record(f"{prefix}.provenance", provenance)
    provenance_identity = _terminal_artifact_identity(
        f"{prefix}.provenance_identity", lineage["provenance_identity"], provenance
    )
    if lineage["provenance_hash"] != provenance_identity["sha256"]:
        raise DirectionalConfirmationError(f"{prefix}.provenance_hash mismatch")
    provenance_expected = {
        "schema_version": 1,
        "mode": "real",
        "seed": seed,
        "stage": "joint_adaptation",
        "repo_commit": resume["repo_commit"],
        "upstream_commit": resume["upstream_commit"],
        "config_path": config_identity["path"],
        "config_hash": config_identity["sha256"],
        "split_hash": resume["split_hash"],
        "parent_projection_hash": resume["parent_projection_hash"],
        "parent_projection_file_hash": resume["parent_projection_file_hash"],
        "parent_projection_spec_hash": resume["parent_projection_spec_hash"],
        "common_checkpoint": common["path"],
        "common_checkpoint_hash": common["sha256"],
        "parent_checkpoint": parent["path"],
        "route_equivalent_passes": 8.0,
        "dropout": 0.0,
        "ranking_mode": config["ranking_mode"],
        "rank_override": config["rank_override"],
        "declared_devices": 4,
        "uses_gpu": True,
        "loads_blind_data": False,
        "validation_interval": 250,
        "validation_batches_per_family": 8,
        "rng_scheme": "dive-directional-rng-v1",
    }
    for field, expected in provenance_expected.items():
        if (
            type(provenance[field]) is not type(expected)
            or provenance[field] != expected
        ):
            raise DirectionalConfirmationError(f"{prefix}.provenance.{field} mismatch")
    manifest_names = (
        "binder_train.parquet",
        "binder_validation.parquet",
        "ame_train.parquet",
        "ame_validation.parquet",
        "antibody_train.parquet",
        "antibody_validation.parquet",
    )
    manifests = _sequence(
        f"{prefix}.provenance.loader_manifests", provenance["loader_manifests"]
    )
    if len(manifests) != 6:
        raise DirectionalConfirmationError(
            f"{prefix}.provenance loader inventory incomplete"
        )
    for index, (raw_manifest, expected_name) in enumerate(
        zip(manifests, manifest_names, strict=True)
    ):
        manifest = _mapping(
            f"{prefix}.provenance.loader_manifests[{index}]", raw_manifest
        )
        _exact_keys(
            f"{prefix}.provenance.loader_manifests[{index}]",
            manifest,
            frozenset(
                {
                    "name",
                    "family",
                    "partition",
                    "sha256",
                    "size_bytes",
                    "row_count",
                    "example_id_hash",
                }
            ),
        )
        if manifest["name"] != expected_name:
            raise DirectionalConfirmationError(
                f"{prefix}.provenance manifest order drifted"
            )
        expected_family = expected_name.split("_", 1)[0]
        expected_partition = "validation" if "_validation" in expected_name else "train"
        if (
            manifest["family"] != expected_family
            or manifest["partition"] != expected_partition
        ):
            raise DirectionalConfirmationError(
                f"{prefix}.provenance manifest family/partition drifted"
            )
        _hash(
            f"{prefix}.provenance.loader_manifests[{index}].sha256", manifest["sha256"]
        )
        _hash(
            f"{prefix}.provenance.loader_manifests[{index}].example_id_hash",
            manifest["example_id_hash"],
        )
        if (
            _exact_int(
                f"{prefix}.provenance.loader_manifests[{index}].size_bytes",
                manifest["size_bytes"],
            )
            <= 0
            or _exact_int(
                f"{prefix}.provenance.loader_manifests[{index}].row_count",
                manifest["row_count"],
            )
            <= 0
        ):
            raise DirectionalConfirmationError(f"{prefix}.provenance manifest is empty")
    expected_metric_schema = (
        "step",
        "loss_breakdown",
        "route_occupancy",
        "residual_norms_by_tap",
        "non_finite_count",
        "non_finite_events",
        "elapsed_seconds",
        "allocated_gpu_count",
        "aggregation",
        "family_statistics",
        "cell_statistics",
        "example_statistics",
        "example_statistics_hash",
        "route_loss_sufficient",
        "per_residue_oracle_sufficient",
        "tap_residual_relative_sufficient",
        "time_bin_route_loss_sufficient",
        "time_bin_scheme",
        "rng_scheme",
    )
    metric_schema = _sequence(
        f"{prefix}.provenance.validation_metric_schema",
        provenance["validation_metric_schema"],
    )
    if tuple(metric_schema) != expected_metric_schema:
        raise DirectionalConfirmationError(
            f"{prefix}.provenance validation metric schema drifted"
        )

    validation = _mapping(f"{prefix}.validation_record", lineage["validation_record"])
    _exact_keys(
        f"{prefix}.validation_record",
        validation,
        frozenset(
            {
                "schema_version",
                "stage",
                "seed",
                "step_completed",
                "record",
                "ownership",
                "validation_cell_hash",
                "check_schema_version",
                "check_records_hash",
            }
        ),
    )
    aggregate = _mapping(f"{prefix}.validation_record.record", validation["record"])
    aggregate_keys = frozenset(
        {
            "step",
            "loss_breakdown",
            "route_occupancy",
            "residual_norms_by_tap",
            "non_finite_count",
            "non_finite_events",
            "elapsed_seconds",
            "allocated_gpu_count",
            "aggregation",
            "family_statistics",
            "cell_statistics",
            "example_statistics",
            "example_statistics_hash",
            "route_loss_sufficient",
            "per_residue_oracle_sufficient",
            "tap_residual_relative_sufficient",
            "time_bin_route_loss_sufficient",
            "time_bin_scheme",
            "rng_scheme",
        }
    )
    _exact_keys(f"{prefix}.validation_record.record", aggregate, aggregate_keys)
    if (
        validation["schema_version"] != 1
        or validation["stage"] != "joint_adaptation"
        or validation["seed"] != seed
        or validation["step_completed"] != 20_000
        or validation["ownership"] != ownership
        or aggregate["step"] != 19_999
        or aggregate["example_statistics"] != list(records)
        or aggregate["example_statistics_hash"] != example_statistics_hash
        or aggregate["rng_scheme"] != "dive-directional-validation-cell-v2"
        or validation["validation_cell_hash"] != validation_cell_hash
        or validation["check_schema_version"] != candidate["check_schema_version"]
        or validation["check_records_hash"] != candidate["check_records_hash"]
    ):
        raise DirectionalConfirmationError(f"{prefix}.validation_record mismatch")
    validation_identity = _terminal_artifact_identity(
        f"{prefix}.validation_record_identity",
        lineage["validation_record_identity"],
        validation,
    )
    if lineage["validation_record_hash"] != validation_identity["sha256"]:
        raise DirectionalConfirmationError(f"{prefix}.validation_record_hash mismatch")

    completion = _mapping(
        f"{prefix}.checkpoint_completion", lineage["checkpoint_completion"]
    )
    _exact_keys(
        f"{prefix}.checkpoint_completion",
        completion,
        frozenset(
            {
                "schema_version",
                "checkpoint",
                "step_completed",
                "resume_state",
                "ownership",
                "validation_record",
            }
        ),
    )
    _validate_ranking_record(
        f"{prefix}.checkpoint_completion.resume_state",
        completion["resume_state"],
    )
    if completion != {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "step_completed": 20_000,
        "resume_state": resume,
        "ownership": ownership,
        "validation_record": validation_identity,
    }:
        raise DirectionalConfirmationError(f"{prefix}.checkpoint_completion mismatch")
    completion_identity = _terminal_artifact_identity(
        f"{prefix}.checkpoint_completion_identity",
        lineage["checkpoint_completion_identity"],
        completion,
    )
    if lineage["checkpoint_completion_hash"] != completion_identity["sha256"]:
        raise DirectionalConfirmationError(
            f"{prefix}.checkpoint_completion_hash mismatch"
        )

    training = _mapping(f"{prefix}.training_completion", lineage["training_completion"])
    training_keys = frozenset(
        {
            "schema_version",
            "mode",
            "stage",
            "seed",
            "ranking_mode",
            "rank_override",
            "steps",
            "start_step",
            "checkpoint",
            "checkpoint_hash",
            "checkpoint_completion",
            "resume_state",
            "ownership",
            "splice",
            "validation",
            "elapsed_seconds",
            "allocated_gpu_count",
            "uses_gpu",
            "loads_blind_data",
            "uses_synthetic_data",
            "rng_state",
            "precision_policy",
        }
    )
    _exact_keys(f"{prefix}.training_completion", training, training_keys)
    training_ranking = _validate_ranking_record(
        f"{prefix}.training_completion", training
    )
    _validate_ranking_record(
        f"{prefix}.training_completion.resume_state",
        training["resume_state"],
    )
    training_validation = _sequence(
        f"{prefix}.training_completion.validation", training["validation"]
    )
    if (
        training["schema_version"] != 1
        or training["mode"] != "real"
        or training["stage"] != "joint_adaptation"
        or training["seed"] != seed
        or training_ranking != config_ranking
        or training["start_step"] != 0
        or training["steps"] != 20_000
        or training["checkpoint"] != checkpoint["path"]
        or training["checkpoint_hash"] != checkpoint["sha256"]
        or training["checkpoint_completion"] != completion_identity["path"]
        or training["resume_state"] != resume
        or training["ownership"] != ownership
        or not training_validation
        or training_validation[-1] != aggregate
        or training["allocated_gpu_count"] != 4
        or training["uses_gpu"] is not True
        or training["loads_blind_data"] is not False
        or training["uses_synthetic_data"] is not False
    ):
        raise DirectionalConfirmationError(f"{prefix}.training_completion mismatch")
    training_identity = _terminal_artifact_identity(
        f"{prefix}.training_completion_identity",
        lineage["training_completion_identity"],
        training,
    )
    if lineage["training_completion_hash"] != training_identity["sha256"]:
        raise DirectionalConfirmationError(
            f"{prefix}.training_completion_hash mismatch"
        )

    cross_bindings = {
        "checkpoint_path": checkpoint["path"],
        "checkpoint_hash": checkpoint["sha256"],
        "checkpoint_size_bytes": checkpoint["size_bytes"],
        "parent_checkpoint_path": parent["path"],
        "parent_checkpoint_hash": parent["sha256"],
        "parent_checkpoint_size_bytes": parent["size_bytes"],
        "common_checkpoint_path": common["path"],
        "common_checkpoint_hash": common["sha256"],
        "common_checkpoint_size_bytes": common["size_bytes"],
        "config_path": config_identity["path"],
        "provenance_hash": provenance_identity["sha256"],
        "provenance_path": provenance_identity["path"],
        "provenance_size_bytes": provenance_identity["size_bytes"],
        "training_completion_hash": training_identity["sha256"],
        "training_completion_path": training_identity["path"],
        "training_completion_size_bytes": training_identity["size_bytes"],
        "checkpoint_completion_hash": completion_identity["sha256"],
        "checkpoint_completion_path": completion_identity["path"],
        "checkpoint_completion_size_bytes": completion_identity["size_bytes"],
        "validation_record_hash": validation_identity["sha256"],
        "validation_record_path": validation_identity["path"],
        "validation_record_size_bytes": validation_identity["size_bytes"],
        "split_hash": resume["split_hash"],
        "parent_projection_hash": resume["parent_projection_hash"],
        "parent_projection_file_hash": resume["parent_projection_file_hash"],
        "parent_projection_spec_hash": resume["parent_projection_spec_hash"],
        "resume_hash": lineage["resume_hash"],
        "ownership_hash": lineage["ownership_hash"],
        "example_statistics_hash": example_statistics_hash,
        "validation_cell_hash": validation_cell_hash,
        "config_hash": config_identity["sha256"],
        "repo_commit": resume["repo_commit"],
        "upstream_commit": resume["upstream_commit"],
        "stochastic_budget_hash": lineage["stochastic_budget_hash"],
    }
    for field, expected in cross_bindings.items():
        if candidate[field] != expected:
            raise DirectionalConfirmationError(f"{name}.{field} lineage mismatch")

def _validate_candidate(
    value: object, *, name: str, rerun: bool = False
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...], _CheckMetrics]:
    candidate = _mapping(name, value)
    expected_keys = _CANDIDATE_KEYS | (_RERUN_EXTRA_KEYS if rerun else frozenset())
    _exact_keys_with_optional(name, candidate, expected_keys, _LEGACY_CHECK_KEYS)
    if _exact_int(f"{name}.schema_version", candidate["schema_version"]) != 1:
        raise DirectionalConfirmationError(f"{name}.schema_version must be exactly 1")
    seed = _exact_int(f"{name}.seed", candidate["seed"])
    if seed not in (42, 314159):
        raise DirectionalConfirmationError(f"{name}.seed is not frozen")
    _absolute_path(f"{name}.checkpoint_path", candidate["checkpoint_path"])
    checkpoint_hash = _hash(f"{name}.checkpoint_hash", candidate["checkpoint_hash"])
    _absolute_path(
        f"{name}.parent_checkpoint_path", candidate["parent_checkpoint_path"]
    )
    for field in (
        "common_checkpoint_path",
        "config_path",
        "provenance_path",
        "training_completion_path",
        "checkpoint_completion_path",
        "validation_record_path",
    ):
        _absolute_path(f"{name}.{field}", candidate[field])
    for field in (
        "checkpoint_size_bytes",
        "parent_checkpoint_size_bytes",
        "common_checkpoint_size_bytes",
        "provenance_size_bytes",
        "training_completion_size_bytes",
        "checkpoint_completion_size_bytes",
        "validation_record_size_bytes",
    ):
        if _exact_int(f"{name}.{field}", candidate[field]) <= 0:
            raise DirectionalConfirmationError(f"{name}.{field} must be positive")
    parent_checkpoint_hash = _hash(
        f"{name}.parent_checkpoint_hash", candidate["parent_checkpoint_hash"]
    )
    if checkpoint_hash == parent_checkpoint_hash:
        raise DirectionalConfirmationError(
            f"{name}.checkpoint_hash must identify a create-new checkpoint"
        )
    for field in (
        "provenance_hash",
        "training_completion_hash",
        "checkpoint_completion_hash",
        "validation_record_hash",
        "common_checkpoint_hash",
        "split_hash",
        "parent_projection_hash",
        "parent_projection_file_hash",
        "parent_projection_spec_hash",
        "resume_hash",
        "ownership_hash",
        "example_statistics_hash",
    ):
        _hash(f"{name}.{field}", candidate[field])
    validation_hash = _hash(
        f"{name}.validation_cell_hash", candidate["validation_cell_hash"]
    )
    _hash(f"{name}.config_hash", candidate["config_hash"])
    _hash(f"{name}.repo_commit", candidate["repo_commit"], length=40)
    upstream = _hash(f"{name}.upstream_commit", candidate["upstream_commit"], length=40)
    if upstream != EMERGENT_UPSTREAM_COMMIT:
        raise DirectionalConfirmationError(f"{name}.upstream_commit pin drifted")
    _hash(f"{name}.stochastic_budget_hash", candidate["stochastic_budget_hash"])
    _assert_confirmation(candidate["confirmation"])
    raw_records = _sequence(
        f"{name}.example_statistics", candidate["example_statistics"]
    )
    if not raw_records:
        raise DirectionalConfirmationError(f"{name}.example_statistics is incomplete")
    records = tuple(
        _validate_example(record, index=index)
        for index, record in enumerate(raw_records)
    )
    observed_identities = [
        (
            str(record["family"]),
            str(record["example_id"]),
            str(record["parent_id"]),
            str(record["cell_hash"]),
        )
        for record in records
    ]
    if len(set(observed_identities)) != len(observed_identities):
        raise DirectionalConfirmationError(
            f"{name}.example_statistics has duplicate cells"
        )
    cell_families: dict[str, str] = {}
    for family, _, _, cell_hash in observed_identities:
        previous = cell_families.setdefault(cell_hash, family)
        if previous != family:
            raise DirectionalConfirmationError(
                f"{name}.cell_hash is inconsistent across families"
            )

    raw_expected = _sequence(f"{name}.expected_cells", candidate["expected_cells"])
    expected_cells = [
        _expected_cell(record, index=index) for index, record in enumerate(raw_expected)
    ]
    expected_identities = [
        (
            record["family"],
            record["example_id"],
            record["parent_id"],
            record["cell_hash"],
        )
        for record in expected_cells
    ]
    if len(set(expected_identities)) != len(expected_identities):
        raise DirectionalConfirmationError(f"{name}.expected_cells has duplicate cells")
    if set(observed_identities) != set(expected_identities):
        raise DirectionalConfirmationError(
            f"{name}.example_statistics is incomplete or has inconsistent parent identity"
        )
    if {record["family"] for record in expected_cells} != set(FAMILIES):
        raise DirectionalConfirmationError(
            f"{name}.expected_cells is incomplete by family"
        )
    ordered_expected = sorted(
        expected_cells, key=lambda record: (record["family"], record["example_id"])
    )
    observed_validation_hash = hashlib.sha256(
        _canonical({"cells": ordered_expected})
    ).hexdigest()
    if validation_hash != observed_validation_hash:
        raise DirectionalConfirmationError(
            f"{name}.validation_cell_hash is inconsistent with expected cells"
        )
    example_statistics_hash = hashlib.sha256(
        _canonical({"examples": list(records)}) + b"\n"
    ).hexdigest()
    if candidate["example_statistics_hash"] != example_statistics_hash:
        raise DirectionalConfirmationError(
            f"{name}.example_statistics_hash is inconsistent with example statistics"
        )
    check_metrics = _validate_check_bundle(
        candidate["check_records"],
        records=records,
        name=name,
        schema_version=candidate["check_schema_version"],
        expected_hash=candidate["check_records_hash"],
        legacy=candidate,
    )
    _validate_candidate_lineage(
        candidate,
        name=name,
        records=records,
        example_statistics_hash=example_statistics_hash,
        validation_cell_hash=validation_hash,
    )
    if rerun:
        _absolute_path(
            f"{name}.selected_checkpoint_path",
            candidate["selected_checkpoint_path"],
        )
        _hash(f"{name}.selected_checkpoint_hash", candidate["selected_checkpoint_hash"])
        _hash(f"{name}.selected_candidate_hash", candidate["selected_candidate_hash"])
        _hash(f"{name}.selected_selection_hash", candidate["selected_selection_hash"])
        interval = _sequence(
            f"{name}.expected_bootstrap_interval",
            candidate["expected_bootstrap_interval"],
        )
        if len(interval) != 2:
            raise DirectionalConfirmationError(
                f"{name}.expected_bootstrap_interval must contain two bounds"
            )
        for index, bound in enumerate(interval):
            _float(f"{name}.expected_bootstrap_interval[{index}]", bound)
    return candidate, records, check_metrics

def _numerator(record: Mapping[str, object], route: str) -> float:
    return float(record["route_losses"][route]["numerator"])

def _family_ratio(
    records: Sequence[Mapping[str, object]], numerator_name: str
) -> float:
    numerator = sum(_numerator(record, numerator_name) for record in records)
    denominator = sum(_numerator(record, "none") for record in records)
    if denominator <= 0.0:
        raise DirectionalConfirmationError(
            "relative validation denominator is non-positive"
        )
    return numerator / denominator

def _equal_family_route_score(
    records: Sequence[Mapping[str, object]], route: str
) -> float:
    ratios = []
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        if not family_records:
            raise DirectionalConfirmationError(f"validation is incomplete for {family}")
        ratios.append(_family_ratio(family_records, route))
    return sum(ratios) / len(ratios)

def _best_fixed_score(records: Sequence[Mapping[str, object]]) -> float:
    ratios = []
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        if not family_records:
            raise DirectionalConfirmationError(f"validation is incomplete for {family}")
        ratios.append(
            min(_family_ratio(family_records, route) for route in HARD_ROUTES)
        )
    return sum(ratios) / len(ratios)

def _oracle_score(records: Sequence[Mapping[str, object]]) -> float:
    ratios = []
    for family in FAMILIES:
        family_records = [record for record in records if record["family"] == family]
        numerator = sum(
            float(record["per_residue_oracle"]["numerator"])
            for record in family_records
        )
        denominator = sum(_numerator(record, "none") for record in family_records)
        if denominator <= 0.0:
            raise DirectionalConfirmationError("oracle denominator is non-positive")
        ratios.append(numerator / denominator)
    return sum(ratios) / len(ratios)

def _time_only_score(records: Sequence[Mapping[str, object]]) -> float:
    selected_by_bin: dict[int, str] = {}
    for bin_id in sorted({int(record["time_bin"]["index"]) for record in records}):
        in_bin = [
            record for record in records if int(record["time_bin"]["index"]) == bin_id
        ]
        scores = []
        for route in HARD_ROUTES:
            ratios = []
            for family in FAMILIES:
                subset = [record for record in in_bin if record["family"] == family]
                if subset:
                    ratios.append(_family_ratio(subset, route))
            if not ratios:
                raise DirectionalConfirmationError(
                    "time-only control received an empty bin"
                )
            scores.append((sum(ratios) / len(ratios), route))
        selected_by_bin[bin_id] = min(scores)[1]

    family_ratios = []
    for family in FAMILIES:
        subset = [record for record in records if record["family"] == family]
        numerator = sum(
            _numerator(record, selected_by_bin[int(record["time_bin"]["index"])])
            for record in subset
        )
        denominator = sum(_numerator(record, "none") for record in subset)
        if denominator <= 0.0:
            raise DirectionalConfirmationError("time-only denominator is non-positive")
        family_ratios.append(numerator / denominator)
    return sum(family_ratios) / len(family_ratios)

def _parent_bootstrap_interval(
    records: Sequence[Mapping[str, object]], *, resamples: int, seed: int
) -> tuple[float, float]:
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = {
        family: {} for family in FAMILIES
    }
    for record in records:
        grouped[str(record["family"])].setdefault(str(record["parent_id"]), []).append(
            (_numerator(record, "adaptive"), _numerator(record, "none"))
        )
    if any(not groups for groups in grouped.values()):
        raise DirectionalConfirmationError("parent bootstrap is incomplete by family")
    ordered = {
        family: [values for _, values in sorted(groups.items())]
        for family, groups in grouped.items()
    }
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        family_ratios = []
        for family in FAMILIES:
            parents = ordered[family]
            sampled = [rng.choice(parents) for _ in parents]
            numerator = sum(value for group in sampled for value, _ in group)
            denominator = sum(value for group in sampled for _, value in group)
            if denominator <= 0.0:
                raise DirectionalConfirmationError(
                    "parent bootstrap denominator is non-positive"
                )
            family_ratios.append(numerator / denominator)
        draws.append(sum(family_ratios) / len(family_ratios))
    return _linear_quantile(draws, 0.025), _linear_quantile(draws, 0.975)

def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise DirectionalConfirmationError("bootstrap produced no draws")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

def _candidate_metrics(
    value: object, *, name: str, rerun: bool = False
) -> _CandidateMetrics:
    candidate, records, checks = _validate_candidate(value, name=name, rerun=rerun)
    adaptive_score = _equal_family_route_score(records, "adaptive")
    fixed_scores = {
        route: _equal_family_route_score(records, route) for route in HARD_ROUTES
    }
    global_score = min(fixed_scores.values())
    time_score = _time_only_score(records)
    best_fixed_score = _best_fixed_score(records)
    control_score = min(best_fixed_score, global_score, time_score)
    if control_score <= 0.0:
        raise DirectionalConfirmationError("best-control denominator is non-positive")
    oracle_score = _oracle_score(records)
    oracle_gap = control_score - oracle_score
    if oracle_gap <= 0.0:
        raise DirectionalConfirmationError("oracle-gap denominator is non-positive")
    adaptive_gain = 1.0 - adaptive_score / control_score
    gap_recovery = (control_score - adaptive_score) / oracle_gap

    residual_by_family: dict[str, dict[str, float]] = {}
    occupancy_by_family: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        subset = [record for record in records if record["family"] == family]
        residual_by_family[family] = {}
        for direction in DIRECTIONS:
            applied = sum(
                float(
                    record["tap_residuals"][tap][direction][
                        "applied_residual_squared_sum"
                    ]
                )
                for record in subset
                for tap in TAPS
            )
            joint = sum(
                float(
                    record["tap_residuals"][tap][direction][
                        "joint_field_squared_norm_sum"
                    ]
                )
                for record in subset
                for tap in TAPS
            )
            if joint <= 0.0:
                raise DirectionalConfirmationError(
                    "residual denominator is non-positive"
                )
            residual_by_family[family][direction] = math.sqrt(applied / joint)
        occupancy_by_family[family] = {}
        for route in HARD_ROUTES:
            numerator = sum(
                float(record["tap_probabilities"][tap][route]["numerator"])
                for record in subset
                for tap in TAPS
            )
            denominator = sum(
                int(record["tap_probabilities"][tap][route]["denominator"])
                for record in subset
                for tap in TAPS
            )
            if denominator <= 0:
                raise DirectionalConfirmationError(
                    "occupancy denominator is non-positive"
                )
            occupancy_by_family[family][route] = numerator / denominator

    weak_families = sum(
        residual_by_family[family]["z_to_x"] < MIN_DIRECTIONAL_RESIDUAL_NORM
        and residual_by_family[family]["x_to_z"] < MIN_DIRECTIONAL_RESIDUAL_NORM
        for family in FAMILIES
    )
    universal_dominance = any(
        all(
            occupancy_by_family[family][route] > MAX_UNIVERSAL_OCCUPANCY
            for family in FAMILIES
        )
        for route in HARD_ROUTES
    )
    qualifying_non_none = sum(
        sum(
            occupancy_by_family[family][route] > MIN_NON_NONE_OCCUPANCY
            for family in FAMILIES
        )
        >= 2
        for route in NON_NONE_ROUTES
    )
    first_five = (
        checks.passed,
        weak_families <= 1,
        not universal_dominance and qualifying_non_none >= 2,
        _inclusive_at_least(adaptive_gain, MIN_ADAPTIVE_GAIN),
        _inclusive_at_least(gap_recovery, MIN_ORACLE_GAP_RECOVERY),
    )
    conclusion = tuple(
        (
            family,
            min(
                HARD_ROUTES,
                key=lambda route: (
                    _family_ratio(
                        [record for record in records if record["family"] == family],
                        route,
                    ),
                    route,
                ),
            ),
        )
        for family in sorted(FAMILIES)
    )
    confirmation = _mapping("confirmation", candidate["confirmation"])
    interval = _parent_bootstrap_interval(
        records,
        resamples=int(confirmation["bootstrap_resamples"]),
        seed=int(confirmation["bootstrap_seed"]),
    )
    return _CandidateMetrics(
        candidate=candidate,
        records=records,
        first_five=first_five,
        primary_metric=adaptive_score,
        bootstrap_interval=interval,
        directional_conclusion=conclusion,
        checks_passed=checks.passed,
        check_diagnostics=checks.diagnostics,
    )

def _validate_candidate_pair(
    candidates: object,
) -> tuple[_CandidateMetrics, _CandidateMetrics]:
    sequence = _sequence("candidates", candidates)
    if len(sequence) != 2:
        raise DirectionalConfirmationError("candidates must contain exactly two seeds")
    metrics = tuple(
        _candidate_metrics(value, name=f"candidates[{index}]")
        for index, value in enumerate(sequence)
    )
    if {metric.candidate["seed"] for metric in metrics} != {42, 314159}:
        raise DirectionalConfirmationError("candidates must bind both frozen seeds")
    if len({metric.candidate["checkpoint_hash"] for metric in metrics}) != 2:
        raise DirectionalConfirmationError(
            "candidate checkpoint identities must be distinct"
        )
    if len({metric.candidate["checkpoint_path"] for metric in metrics}) != 2:
        raise DirectionalConfirmationError(
            "candidate checkpoint paths must be distinct create-new identities"
        )
    common_fields = (
        "config_hash",
        "repo_commit",
        "upstream_commit",
        "split_hash",
        "parent_projection_hash",
        "parent_projection_file_hash",
        "parent_projection_spec_hash",
    )
    for field in common_fields:
        if metrics[0].candidate[field] != metrics[1].candidate[field]:
            raise DirectionalConfirmationError(
                f"candidate {field} identities are inconsistent"
            )
    first_examples = sorted(
        (
            record["family"],
            record["example_id"],
            record["parent_id"],
            record["time_bin"]["index"],
        )
        for record in metrics[0].records
    )
    second_examples = sorted(
        (
            record["family"],
            record["example_id"],
            record["parent_id"],
            record["time_bin"]["index"],
        )
        for record in metrics[1].records
    )
    if first_examples != second_examples:
        raise DirectionalConfirmationError(
            "candidate validation cells or parent identities are inconsistent"
        )
    return metrics[0], metrics[1]

def _selection_from_metrics(
    metrics: tuple[_CandidateMetrics, _CandidateMetrics], *, require_eligible: bool
) -> CandidateSelection:
    if require_eligible and not all(all(metric.first_five) for metric in metrics):
        raise DirectionalConfirmationError(
            "both candidates must pass the first five development gates"
        )
    if require_eligible and (
        metrics[0].directional_conclusion != metrics[1].directional_conclusion
    ):
        raise DirectionalConfirmationError(
            "both candidates must agree on the directional conclusion"
        )
    if metrics[0].primary_metric == metrics[1].primary_metric:
        raise DirectionalConfirmationError(
            "candidate primary metric tie; no gate-statistic tiebreak is authorized"
        )
    selected = min(metrics, key=lambda metric: metric.primary_metric)
    return _selection_for_metric(selected)

def _selection_for_metric(selected: _CandidateMetrics) -> CandidateSelection:
    candidate = selected.candidate
    return CandidateSelection(
        seed=int(candidate["seed"]),
        checkpoint_path=str(candidate["checkpoint_path"]),
        checkpoint_hash=str(candidate["checkpoint_hash"]),
        checkpoint_size_bytes=int(candidate["checkpoint_size_bytes"]),
        parent_checkpoint_path=str(candidate["parent_checkpoint_path"]),
        parent_checkpoint_hash=str(candidate["parent_checkpoint_hash"]),
        parent_checkpoint_size_bytes=int(candidate["parent_checkpoint_size_bytes"]),
        common_checkpoint_path=str(candidate["common_checkpoint_path"]),
        common_checkpoint_hash=str(candidate["common_checkpoint_hash"]),
        common_checkpoint_size_bytes=int(candidate["common_checkpoint_size_bytes"]),
        config_path=str(candidate["config_path"]),
        provenance_hash=str(candidate["provenance_hash"]),
        provenance_path=str(candidate["provenance_path"]),
        provenance_size_bytes=int(candidate["provenance_size_bytes"]),
        training_completion_hash=str(candidate["training_completion_hash"]),
        training_completion_path=str(candidate["training_completion_path"]),
        training_completion_size_bytes=int(candidate["training_completion_size_bytes"]),
        checkpoint_completion_hash=str(candidate["checkpoint_completion_hash"]),
        checkpoint_completion_path=str(candidate["checkpoint_completion_path"]),
        checkpoint_completion_size_bytes=int(
            candidate["checkpoint_completion_size_bytes"]
        ),
        validation_record_hash=str(candidate["validation_record_hash"]),
        validation_record_path=str(candidate["validation_record_path"]),
        validation_record_size_bytes=int(candidate["validation_record_size_bytes"]),
        split_hash=str(candidate["split_hash"]),
        parent_projection_hash=str(candidate["parent_projection_hash"]),
        parent_projection_file_hash=str(candidate["parent_projection_file_hash"]),
        parent_projection_spec_hash=str(candidate["parent_projection_spec_hash"]),
        resume_hash=str(candidate["resume_hash"]),
        ownership_hash=str(candidate["ownership_hash"]),
        validation_cell_hash=str(candidate["validation_cell_hash"]),
        example_statistics_hash=str(candidate["example_statistics_hash"]),
        config_hash=str(candidate["config_hash"]),
        repo_commit=str(candidate["repo_commit"]),
        upstream_commit=str(candidate["upstream_commit"]),
        stochastic_budget_hash=str(candidate["stochastic_budget_hash"]),
        confirmation_hash=hashlib.sha256(
            _canonical(candidate["confirmation"])
        ).hexdigest(),
        equal_family_relative_validation_flow_loss=selected.primary_metric,
        bootstrap_interval=selected.bootstrap_interval,
        directional_conclusion=selected.directional_conclusion,
    )

def select_candidate(candidates: Sequence[Mapping[str, object]]) -> CandidateSelection:

    return _selection_from_metrics(
        _validate_candidate_pair(candidates), require_eligible=True
    )

def _selection_mapping(value: object, *, name: str) -> Mapping[str, object]:
    record = _mapping(name, value)
    _exact_keys(name, record, _SELECTION_KEYS)
    seed = _exact_int(f"{name}.seed", record["seed"])
    if seed not in (42, 314159):
        raise DirectionalConfirmationError(f"{name}.seed is not frozen")
    _absolute_path(f"{name}.checkpoint_path", record["checkpoint_path"])
    _absolute_path(f"{name}.parent_checkpoint_path", record["parent_checkpoint_path"])
    for field in (
        "common_checkpoint_path",
        "config_path",
        "provenance_path",
        "training_completion_path",
        "checkpoint_completion_path",
        "validation_record_path",
    ):
        _absolute_path(f"{name}.{field}", record[field])
    for field in (
        "checkpoint_size_bytes",
        "parent_checkpoint_size_bytes",
        "common_checkpoint_size_bytes",
        "provenance_size_bytes",
        "training_completion_size_bytes",
        "checkpoint_completion_size_bytes",
        "validation_record_size_bytes",
    ):
        if _exact_int(f"{name}.{field}", record[field]) <= 0:
            raise DirectionalConfirmationError(f"{name}.{field} must be positive")
    for field in (
        "checkpoint_hash",
        "parent_checkpoint_hash",
        "common_checkpoint_hash",
        "provenance_hash",
        "training_completion_hash",
        "checkpoint_completion_hash",
        "validation_record_hash",
        "split_hash",
        "parent_projection_hash",
        "parent_projection_file_hash",
        "parent_projection_spec_hash",
        "resume_hash",
        "ownership_hash",
        "validation_cell_hash",
        "example_statistics_hash",
        "config_hash",
        "stochastic_budget_hash",
        "confirmation_hash",
    ):
        _hash(f"{name}.{field}", record[field])
    _hash(f"{name}.repo_commit", record["repo_commit"], length=40)
    upstream_commit = _hash(
        f"{name}.upstream_commit", record["upstream_commit"], length=40
    )
    if upstream_commit != EMERGENT_UPSTREAM_COMMIT:
        raise DirectionalConfirmationError(f"{name}.upstream_commit pin drifted")
    _float(
        f"{name}.equal_family_relative_validation_flow_loss",
        record["equal_family_relative_validation_flow_loss"],
        nonnegative=True,
    )
    interval = _sequence(f"{name}.bootstrap_interval", record["bootstrap_interval"])
    if len(interval) != 2:
        raise DirectionalConfirmationError(
            f"{name}.bootstrap_interval needs two bounds"
        )
    low = _float(f"{name}.bootstrap_interval[0]", interval[0])
    high = _float(f"{name}.bootstrap_interval[1]", interval[1])
    if low > high:
        raise DirectionalConfirmationError(
            f"{name}.bootstrap_interval bounds are reversed"
        )
    conclusion = _sequence(
        f"{name}.directional_conclusion", record["directional_conclusion"]
    )
    if len(conclusion) != len(FAMILIES):
        raise DirectionalConfirmationError(
            f"{name}.directional_conclusion is incomplete"
        )
    parsed_conclusion = []
    for index, item in enumerate(conclusion):
        pair = _sequence(f"{name}.directional_conclusion[{index}]", item)
        if len(pair) != 2:
            raise DirectionalConfirmationError(
                f"{name}.directional_conclusion[{index}] must be a pair"
            )
        family = _string(f"{name}.directional_conclusion[{index}][0]", pair[0])
        route = _string(f"{name}.directional_conclusion[{index}][1]", pair[1])
        if family not in FAMILIES or route not in HARD_ROUTES:
            raise DirectionalConfirmationError(
                f"{name}.directional_conclusion[{index}] is unknown"
            )
        parsed_conclusion.append((family, route))
    if tuple(parsed_conclusion) != tuple(
        sorted(parsed_conclusion, key=lambda item: item[0])
    ) or {family for family, _ in parsed_conclusion} != set(FAMILIES):
        raise DirectionalConfirmationError(
            f"{name}.directional_conclusion is incomplete or unordered"
        )
    return record

def _candidate_selection_from_mapping(
    value: object, *, name: str
) -> CandidateSelection:
    record = _selection_mapping(value, name=name)
    interval = _sequence(f"{name}.bootstrap_interval", record["bootstrap_interval"])
    conclusion = _sequence(
        f"{name}.directional_conclusion", record["directional_conclusion"]
    )
    return CandidateSelection(
        seed=int(record["seed"]),
        checkpoint_path=str(record["checkpoint_path"]),
        checkpoint_hash=str(record["checkpoint_hash"]),
        checkpoint_size_bytes=int(record["checkpoint_size_bytes"]),
        parent_checkpoint_path=str(record["parent_checkpoint_path"]),
        parent_checkpoint_hash=str(record["parent_checkpoint_hash"]),
        parent_checkpoint_size_bytes=int(record["parent_checkpoint_size_bytes"]),
        common_checkpoint_path=str(record["common_checkpoint_path"]),
        common_checkpoint_hash=str(record["common_checkpoint_hash"]),
        common_checkpoint_size_bytes=int(record["common_checkpoint_size_bytes"]),
        config_path=str(record["config_path"]),
        provenance_hash=str(record["provenance_hash"]),
        provenance_path=str(record["provenance_path"]),
        provenance_size_bytes=int(record["provenance_size_bytes"]),
        training_completion_hash=str(record["training_completion_hash"]),
        training_completion_path=str(record["training_completion_path"]),
        training_completion_size_bytes=int(record["training_completion_size_bytes"]),
        checkpoint_completion_hash=str(record["checkpoint_completion_hash"]),
        checkpoint_completion_path=str(record["checkpoint_completion_path"]),
        checkpoint_completion_size_bytes=int(
            record["checkpoint_completion_size_bytes"]
        ),
        validation_record_hash=str(record["validation_record_hash"]),
        validation_record_path=str(record["validation_record_path"]),
        validation_record_size_bytes=int(record["validation_record_size_bytes"]),
        split_hash=str(record["split_hash"]),
        parent_projection_hash=str(record["parent_projection_hash"]),
        parent_projection_file_hash=str(record["parent_projection_file_hash"]),
        parent_projection_spec_hash=str(record["parent_projection_spec_hash"]),
        resume_hash=str(record["resume_hash"]),
        ownership_hash=str(record["ownership_hash"]),
        validation_cell_hash=str(record["validation_cell_hash"]),
        example_statistics_hash=str(record["example_statistics_hash"]),
        config_hash=str(record["config_hash"]),
        repo_commit=str(record["repo_commit"]),
        upstream_commit=str(record["upstream_commit"]),
        stochastic_budget_hash=str(record["stochastic_budget_hash"]),
        confirmation_hash=str(record["confirmation_hash"]),
        equal_family_relative_validation_flow_loss=float(
            record["equal_family_relative_validation_flow_loss"]
        ),
        bootstrap_interval=(float(interval[0]), float(interval[1])),
        directional_conclusion=tuple(
            (str(pair[0]), str(pair[1]))
            for pair in (
                _sequence(f"{name}.directional_conclusion item", item)
                for item in conclusion
            )
        ),
    )

def _assert_selection_equals(
    observed: Mapping[str, object], expected: CandidateSelection
) -> None:
    expected_mapping = expected.to_mapping()
    if _canonical(observed) != _canonical(expected_mapping):
        raise DirectionalConfirmationError(
            "selection differs from the primary-metric candidate selection"
        )

def _record_replay_identity(record: Mapping[str, object]) -> tuple[object, ...]:
    denominator = int(record["route_losses"]["adaptive"]["denominator"])
    return (
        record["family"],
        record["example_id"],
        record["parent_id"],
        record["cell_hash"],
        record["identity_hash"],
        _canonical(record["corruption_time"]),
        _canonical(record["time_bin"]),
        denominator,
    )

def _validated_rerun_metrics(
    selection: CandidateSelection,
    rerun: Mapping[str, object],
    *,
    selected: Mapping[str, object],
) -> _CandidateMetrics:
    if not isinstance(selection, CandidateSelection):
        raise DirectionalConfirmationError("selection must be a CandidateSelection")
    selected_metric = _candidate_metrics(selected, name="selected")
    expected_selection = _selection_for_metric(selected_metric)
    for field in CandidateSelection.__dataclass_fields__:
        if getattr(selection, field) != getattr(expected_selection, field):
            raise DirectionalConfirmationError(
                f"selection {field} does not bind the selected candidate"
            )
    rerun_candidate, rerun_records, _ = _validate_candidate(
        rerun, name="rerun", rerun=True
    )
    bindings = {
        "seed": selection.seed,
        "parent_checkpoint_path": selection.parent_checkpoint_path,
        "parent_checkpoint_hash": selection.parent_checkpoint_hash,
        "parent_checkpoint_size_bytes": selection.parent_checkpoint_size_bytes,
        "common_checkpoint_path": selection.common_checkpoint_path,
        "common_checkpoint_hash": selection.common_checkpoint_hash,
        "common_checkpoint_size_bytes": selection.common_checkpoint_size_bytes,
        "config_path": selection.config_path,
        "validation_cell_hash": selection.validation_cell_hash,
        "split_hash": selection.split_hash,
        "parent_projection_hash": selection.parent_projection_hash,
        "parent_projection_file_hash": selection.parent_projection_file_hash,
        "parent_projection_spec_hash": selection.parent_projection_spec_hash,
        "config_hash": selection.config_hash,
        "repo_commit": selection.repo_commit,
        "upstream_commit": selection.upstream_commit,
        "selected_checkpoint_path": selection.checkpoint_path,
        "selected_checkpoint_hash": selection.checkpoint_hash,
        "selected_candidate_hash": _record_hash(selected_metric.candidate),
        "selected_selection_hash": _record_hash(selection.to_mapping()),
        "expected_bootstrap_interval": list(selection.bootstrap_interval),
    }
    for field, expected in bindings.items():
        if rerun_candidate[field] != expected:
            raise DirectionalConfirmationError(f"rerun {field} identity mismatch")
    if rerun_candidate["checkpoint_hash"] in {
        selection.checkpoint_hash,
        selection.parent_checkpoint_hash,
    }:
        raise DirectionalConfirmationError(
            "rerun checkpoint_hash must identify a create-new retrain"
        )
    if rerun_candidate["checkpoint_path"] in {
        selection.checkpoint_path,
        selection.parent_checkpoint_path,
    }:
        raise DirectionalConfirmationError(
            "rerun checkpoint_path must identify a create-new retrain"
        )
    if sorted(_record_replay_identity(record) for record in rerun_records) != sorted(
        _record_replay_identity(record) for record in selected_metric.records
    ):
        raise DirectionalConfirmationError(
            "rerun cell, parent, corruption, or denominator identity mismatch"
        )
    return _candidate_metrics(rerun_candidate, name="rerun", rerun=True)

def _rerun_scientific_outcome(
    selection: CandidateSelection, rerun_metrics: _CandidateMetrics
) -> bool:
    rerun_metric = rerun_metrics.primary_metric
    low, high = selection.bootstrap_interval
    return (
        rerun_metrics.checks_passed
        and low <= rerun_metric <= high
        and rerun_metrics.directional_conclusion == selection.directional_conclusion
    )

def assert_rerun_reproduces(
    selection: CandidateSelection,
    rerun: Mapping[str, object],
    *,
    selected: Mapping[str, object],
) -> None:

    rerun_metrics = _validated_rerun_metrics(selection, rerun, selected=selected)
    if not rerun_metrics.checks_passed:
        raise DirectionalConfirmationError(
            "rerun failed recomputed counterfactual validity or preservation"
        )
    low, high = selection.bootstrap_interval
    if not low <= rerun_metrics.primary_metric <= high:
        raise DirectionalConfirmationError(
            "rerun primary metric is outside the frozen bootstrap interval"
        )
    if rerun_metrics.directional_conclusion != selection.directional_conclusion:
        raise DirectionalConfirmationError(
            "rerun changed the frozen directional conclusion"
        )

def _validate_no_ranking_warmup(
    value: object, *, selection: CandidateSelection
) -> Mapping[str, object]:
    name = "ablations.no_ranking.warmup"
    lineage = _mapping(name, value)
    _exact_keys(
        name,
        lineage,
        _LINEAGE_KEYS
        | frozenset(
            {
                "config",
                "config_identity",
                "provenance_identity",
                "training_completion_identity",
                "checkpoint_completion_identity",
                "validation_record_identity",
            }
        ),
    )
    if _exact_int(f"{name}.schema_version", lineage["schema_version"]) != 1:
        raise DirectionalConfirmationError(f"{name}.schema_version must be exactly 1")
    checkpoint = _checkpoint_identity(f"{name}.checkpoint", lineage["checkpoint"])
    parent = _checkpoint_identity(
        f"{name}.parent_checkpoint", lineage["parent_checkpoint"]
    )
    common = _checkpoint_identity(
        f"{name}.common_checkpoint", lineage["common_checkpoint"]
    )
    frozen_common = {
        "path": COMMON_CHECKPOINT_PATH,
        "sha256": COMMON_CHECKPOINT_HASH,
        "size_bytes": COMMON_CHECKPOINT_SIZE,
    }
    if dict(parent) != frozen_common or dict(common) != frozen_common:
        raise DirectionalConfirmationError(
            f"{name}.parent_checkpoint must be the common base checkpoint"
        )
    if checkpoint["path"] in {
        selection.parent_checkpoint_path,
        selection.checkpoint_path,
    } or checkpoint["sha256"] in {
        selection.parent_checkpoint_hash,
        selection.checkpoint_hash,
    }:
        raise DirectionalConfirmationError(
            f"{name}.checkpoint must be an independent warmup"
        )

    config, config_identity = _validate_directional_config(
        lineage["config"], lineage["config_identity"], name=f"{name}.config"
    )
    config_ranking = _validate_ranking_record(f"{name}.config", config)
    if config["stage"] != "bridge_warmup" or config_ranking[0] != "no_ranking":
        raise DirectionalConfirmationError(
            f"{name}.config must be the canonical no-ranking warmup"
        )
    ownership = _hashed_record(lineage, record_name="ownership", prefix=name)
    _validate_ownership(
        ownership, name=f"{name}.ownership", expected_stage="bridge_warmup"
    )
    resume = _hashed_record(
        lineage, record_name="resume_state", hash_name="resume_hash", prefix=name
    )
    resume_keys = frozenset(
        {
            "schema_version",
            "stage",
            "seed",
            "repo_commit",
            "upstream_commit",
            "common_checkpoint_hash",
            "parent_checkpoint_hash",
            "split_hash",
            "config_hash",
            "trainable_name_hash",
            "route_equivalent_passes",
            "dropout",
            "ranking_mode",
            "rank_override",
            "optimizer_state_hash",
            "parent_projection_hash",
            "parent_projection_file_hash",
            "parent_projection_spec_hash",
        }
    )
    _exact_keys(f"{name}.resume_state", resume, resume_keys)
    resume_ranking = _validate_ranking_record(f"{name}.resume_state", resume)
    if (
        resume["schema_version"] != 3
        or resume["stage"] != "bridge_warmup"
        or resume["seed"] != selection.seed
        or resume["parent_checkpoint_hash"] is not None
        or resume["common_checkpoint_hash"] != common["sha256"]
        or resume["config_hash"] != config_identity["sha256"]
        or resume_ranking[0] != "no_ranking"
        or resume["route_equivalent_passes"] != 8.0
        or resume["dropout"] != 0.0
        or resume["split_hash"] != selection.split_hash
        or resume["parent_projection_hash"] != selection.parent_projection_hash
        or resume["parent_projection_file_hash"]
        != selection.parent_projection_file_hash
        or resume["parent_projection_spec_hash"]
        != selection.parent_projection_spec_hash
        or resume["repo_commit"] != selection.repo_commit
        or resume["upstream_commit"] != selection.upstream_commit
    ):
        raise DirectionalConfirmationError(f"{name}.resume_state identity mismatch")
    expected_name_hash = hashlib.sha256(
        ("\n".join(str(item) for item in ownership["trainable_names"]) + "\n").encode()
    ).hexdigest()
    if resume["trainable_name_hash"] != expected_name_hash:
        raise DirectionalConfirmationError(
            f"{name}.resume_state trainable identity mismatch"
        )

    stochastic = _hashed_record(lineage, record_name="stochastic_budget", prefix=name)
    _exact_keys(
        f"{name}.stochastic_budget",
        stochastic,
        frozenset(
            {
                "schema_version",
                "route_equivalent_passes",
                "dropout",
                "rng_scheme",
                "validation_cell_hash",
                "example_statistics_hash",
            }
        ),
    )
    if (
        stochastic["schema_version"] != 1
        or stochastic["validation_cell_hash"] != selection.validation_cell_hash
        or stochastic["rng_scheme"] != "dive-directional-rng-v1"
    ):
        raise DirectionalConfirmationError(f"{name}.stochastic_budget mismatch")

    provenance = _mapping(f"{name}.provenance", lineage["provenance"])
    provenance_keys = frozenset(
        {
            "schema_version",
            "mode",
            "run_id",
            "seed",
            "stage",
            "repo_root",
            "repo_commit",
            "upstream_root",
            "upstream_commit",
            "config_path",
            "config_hash",
            "split_manifest_root",
            "split_hash",
            "loader_manifests",
            "parent_projection_hash",
            "parent_projection_file_hash",
            "parent_projection_spec_hash",
            "common_checkpoint",
            "common_checkpoint_hash",
            "autoencoder_checkpoint",
            "autoencoder_checkpoint_hash",
            "parent_checkpoint",
            "route_equivalent_passes",
            "dropout",
            "ranking_mode",
            "rank_override",
            "declared_devices",
            "uses_gpu",
            "loads_blind_data",
            "validation_interval",
            "validation_batches_per_family",
            "validation_metric_schema",
            "rng_scheme",
            "rng_contract",
            "launch_command_policy",
        }
    )
    _exact_keys(f"{name}.provenance", provenance, provenance_keys)
    _validate_ranking_record(f"{name}.provenance", provenance)
    provenance_identity = _terminal_artifact_identity(
        f"{name}.provenance_identity", lineage["provenance_identity"], provenance
    )
    expected_provenance = {
        "schema_version": 1,
        "mode": "real",
        "seed": selection.seed,
        "stage": "bridge_warmup",
        "repo_commit": selection.repo_commit,
        "upstream_commit": selection.upstream_commit,
        "config_path": config_identity["path"],
        "config_hash": config_identity["sha256"],
        "split_hash": selection.split_hash,
        "parent_projection_hash": selection.parent_projection_hash,
        "parent_projection_file_hash": selection.parent_projection_file_hash,
        "parent_projection_spec_hash": selection.parent_projection_spec_hash,
        "common_checkpoint": common["path"],
        "common_checkpoint_hash": common["sha256"],
        "parent_checkpoint": None,
        "route_equivalent_passes": 8.0,
        "dropout": 0.0,
        "ranking_mode": "no_ranking",
        "rank_override": 0.0,
        "declared_devices": 4,
        "uses_gpu": True,
        "loads_blind_data": False,
        "validation_interval": 250,
        "validation_batches_per_family": 8,
        "rng_scheme": "dive-directional-rng-v1",
    }
    if lineage["provenance_hash"] != provenance_identity["sha256"]:
        raise DirectionalConfirmationError(f"{name}.provenance_hash mismatch")
    for field, expected in expected_provenance.items():
        if (
            type(provenance[field]) is not type(expected)
            or provenance[field] != expected
        ):
            raise DirectionalConfirmationError(
                f"{name}.provenance.{field} identity mismatch"
            )

    validation = _mapping(f"{name}.validation_record", lineage["validation_record"])
    _exact_keys(
        f"{name}.validation_record",
        validation,
        frozenset(
            {
                "schema_version",
                "stage",
                "seed",
                "step_completed",
                "record",
                "ownership",
            }
        ),
    )
    aggregate = _mapping(f"{name}.validation_record.record", validation["record"])
    metric_schema = _sequence(
        f"{name}.provenance.validation_metric_schema",
        provenance["validation_metric_schema"],
    )
    if (
        set(aggregate) != set(metric_schema)
        or validation["schema_version"] != 1
        or validation["stage"] != "bridge_warmup"
        or validation["seed"] != selection.seed
        or validation["step_completed"] != 10_000
        or validation["ownership"] != ownership
        or aggregate.get("step") != 9_999
        or aggregate.get("example_statistics_hash")
        != stochastic["example_statistics_hash"]
        or aggregate.get("rng_scheme") != "dive-directional-validation-cell-v2"
    ):
        raise DirectionalConfirmationError(f"{name}.validation_record mismatch")
    validation_identity = _terminal_artifact_identity(
        f"{name}.validation_record_identity",
        lineage["validation_record_identity"],
        validation,
    )
    if lineage["validation_record_hash"] != validation_identity["sha256"]:
        raise DirectionalConfirmationError(f"{name}.validation_record_hash mismatch")

    completion = _mapping(
        f"{name}.checkpoint_completion", lineage["checkpoint_completion"]
    )
    _exact_keys(
        f"{name}.checkpoint_completion",
        completion,
        frozenset(
            {
                "schema_version",
                "checkpoint",
                "step_completed",
                "resume_state",
                "ownership",
                "validation_record",
            }
        ),
    )
    _validate_ranking_record(
        f"{name}.checkpoint_completion.resume_state",
        completion["resume_state"],
    )
    if completion != {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "step_completed": 10_000,
        "resume_state": resume,
        "ownership": ownership,
        "validation_record": validation_identity,
    }:
        raise DirectionalConfirmationError(f"{name}.checkpoint_completion mismatch")
    completion_identity = _terminal_artifact_identity(
        f"{name}.checkpoint_completion_identity",
        lineage["checkpoint_completion_identity"],
        completion,
    )
    if lineage["checkpoint_completion_hash"] != completion_identity["sha256"]:
        raise DirectionalConfirmationError(
            f"{name}.checkpoint_completion_hash mismatch"
        )

    training = _mapping(f"{name}.training_completion", lineage["training_completion"])
    _exact_keys(
        f"{name}.training_completion",
        training,
        frozenset(
            {
                "schema_version",
                "mode",
                "stage",
                "seed",
                "ranking_mode",
                "rank_override",
                "steps",
                "start_step",
                "checkpoint",
                "checkpoint_hash",
                "checkpoint_completion",
                "resume_state",
                "ownership",
                "splice",
                "validation",
                "elapsed_seconds",
                "allocated_gpu_count",
                "uses_gpu",
                "loads_blind_data",
                "uses_synthetic_data",
                "rng_state",
                "precision_policy",
            }
        ),
    )
    training_ranking = _validate_ranking_record(f"{name}.training_completion", training)
    _validate_ranking_record(
        f"{name}.training_completion.resume_state",
        training["resume_state"],
    )
    training_validation = _sequence(
        f"{name}.training_completion.validation", training["validation"]
    )
    if (
        training["schema_version"] != 1
        or training["mode"] != "real"
        or training["stage"] != "bridge_warmup"
        or training["seed"] != selection.seed
        or training_ranking[0] != "no_ranking"
        or training["start_step"] != 0
        or training["steps"] != 10_000
        or training["checkpoint"] != checkpoint["path"]
        or training["checkpoint_hash"] != checkpoint["sha256"]
        or training["checkpoint_completion"] != completion_identity["path"]
        or training["resume_state"] != resume
        or training["ownership"] != ownership
        or not training_validation
        or training_validation[-1] != aggregate
        or training["uses_gpu"] is not True
        or training["loads_blind_data"] is not False
        or training["uses_synthetic_data"] is not False
    ):
        raise DirectionalConfirmationError(f"{name}.training_completion mismatch")
    training_identity = _terminal_artifact_identity(
        f"{name}.training_completion_identity",
        lineage["training_completion_identity"],
        training,
    )
    if lineage["training_completion_hash"] != training_identity["sha256"]:
        raise DirectionalConfirmationError(f"{name}.training_completion_hash mismatch")
    return checkpoint

def _validate_evaluation_record(
    ablation: Mapping[str, object],
    *,
    name: str,
    operation_parameters: Mapping[str, object],
) -> tuple[str, str]:
    evaluation_name = f"{name}.evaluation_record"
    evaluation = _mapping(evaluation_name, ablation["evaluation_record"])
    expected = {
        "schema_version": 1,
        "mode": "evaluation",
        "operation": ablation["operation"],
        "operation_parameters": dict(operation_parameters),
        "seed": ablation["seed"],
        "checkpoint": {
            "path": ablation["checkpoint_path"],
            "sha256": ablation["checkpoint_hash"],
            "size_bytes": ablation["checkpoint_size_bytes"],
        },
        "config_hash": ablation["config_hash"],
        "selection_hash": ablation["selection_hash"],
        "validation_cell_hash": ablation["validation_cell_hash"],
        "check_schema_version": ablation["check_schema_version"],
        "check_records_hash": ablation["check_records_hash"],
    }
    _assert_deep_exact(evaluation_name, evaluation, expected)
    identity = _terminal_artifact_identity(
        f"{name}.evaluation_record_identity",
        ablation["evaluation_record_identity"],
        evaluation,
    )
    recorded_hash = _hash(
        f"{name}.evaluation_record_hash", ablation["evaluation_record_hash"]
    )
    if recorded_hash != identity["sha256"]:
        raise DirectionalConfirmationError(
            f"{name}.evaluation_record_hash identity mismatch"
        )
    return str(identity["path"]), str(identity["sha256"])

def _scientific_artifact_path_registry(
    entries: Sequence[tuple[str, str]],
) -> Mapping[str, str]:
    labels = tuple(label for label, _ in entries)
    paths = tuple(path for _, path in entries)
    if len(set(labels)) != len(labels):
        raise DirectionalConfirmationError(
            "scientific artifact registry contains duplicate labels"
        )
    if len(set(paths)) != len(paths):
        raise DirectionalConfirmationError(
            "scientific artifact paths must be globally unique"
        )
    return MappingProxyType(dict(entries))

def _validate_ablations(
    value: object,
    *,
    selection: CandidateSelection,
    expected_records: Sequence[Mapping[str, object]],
) -> tuple[
    _CandidateMetrics,
    bool,
    Mapping[str, object],
    tuple[tuple[str, str], ...],
]:
    inventory = _mapping("ablations", value)
    _exact_keys("ablations", inventory, _ABLATION_KEYS)
    common_keys = frozenset(
        {
            "mode",
            "seed",
            "checkpoint_path",
            "checkpoint_hash",
            "checkpoint_size_bytes",
            "config_hash",
            "validation_cell_hash",
            "stochastic_budget_hash",
            "operation",
            "selection_hash",
            "check_schema_version",
            "check_records",
            "check_records_hash",
            "evaluation_record",
            "evaluation_record_identity",
            "evaluation_record_hash",
        }
    )
    check_results: dict[str, _CheckMetrics] = {}
    evaluation_identities: list[tuple[str, str, str]] = []
    for name in (
        "bridge_removed",
        "fixed_joint",
        "best_fixed_time_only",
    ):
        record = _mapping(f"ablations.{name}", inventory[name])
        _exact_keys(f"ablations.{name}", record, common_keys)
        expected = {
            "mode": "evaluation",
            "seed": selection.seed,
            "checkpoint_path": selection.checkpoint_path,
            "checkpoint_hash": selection.checkpoint_hash,
            "checkpoint_size_bytes": selection.checkpoint_size_bytes,
            "config_hash": selection.config_hash,
            "validation_cell_hash": selection.validation_cell_hash,
            "stochastic_budget_hash": selection.stochastic_budget_hash,
            "operation": name,
            "selection_hash": _record_hash(selection.to_mapping()),
        }
        for field, expected_value in expected.items():
            if record[field] != expected_value:
                raise DirectionalConfirmationError(
                    f"ablations.{name}.{field} identity mismatch"
                )
        evaluation_identities.append(
            (
                f"ablation.{name}.evaluation",
                *_validate_evaluation_record(
                    record,
                    name=f"ablations.{name}",
                    operation_parameters={},
                ),
            )
        )
        check_results[name] = _validate_check_bundle(
            record["check_records"],
            records=expected_records,
            name=f"ablations.{name}",
            schema_version=record["check_schema_version"],
            expected_hash=record["check_records_hash"],
        )
    shuffled = _mapping("ablations.shuffled_router", inventory["shuffled_router"])
    _exact_keys(
        "ablations.shuffled_router",
        shuffled,
        common_keys | frozenset({"shuffle_matching"}),
    )
    expected_shuffled = {
        "mode": "evaluation",
        "seed": selection.seed,
        "checkpoint_path": selection.checkpoint_path,
        "checkpoint_hash": selection.checkpoint_hash,
        "checkpoint_size_bytes": selection.checkpoint_size_bytes,
        "config_hash": selection.config_hash,
        "validation_cell_hash": selection.validation_cell_hash,
        "stochastic_budget_hash": selection.stochastic_budget_hash,
        "operation": "shuffled_router",
        "shuffle_matching": "length_and_mask",
        "selection_hash": _record_hash(selection.to_mapping()),
    }
    for field, expected_value in expected_shuffled.items():
        if shuffled[field] != expected_value:
            raise DirectionalConfirmationError(
                f"ablations.shuffled_router.{field} identity mismatch"
            )
    evaluation_identities.append(
        (
            "ablation.shuffled_router.evaluation",
            *_validate_evaluation_record(
                shuffled,
                name="ablations.shuffled_router",
                operation_parameters={"shuffle_matching": "length_and_mask"},
            ),
        )
    )
    check_results["shuffled_router"] = _validate_check_bundle(
        shuffled["check_records"],
        records=expected_records,
        name="ablations.shuffled_router",
        schema_version=shuffled["check_schema_version"],
        expected_hash=shuffled["check_records_hash"],
    )

    no_ranking = _mapping("ablations.no_ranking", inventory["no_ranking"])
    no_ranking_keys = frozenset(
        {
            "mode",
            "seed",
            "warmup_steps",
            "joint_steps",
            "rank_override",
            "ranked_warmup_checkpoint",
            "warmup",
            "joint_candidate",
        }
    )
    _exact_keys("ablations.no_ranking", no_ranking, no_ranking_keys)
    expected_no_ranking = {
        "mode": "full_retrain",
        "seed": selection.seed,
        "warmup_steps": 10_000,
        "joint_steps": 20_000,
    }
    _validate_ranking_pairing(
        "ablations.no_ranking",
        "no_ranking",
        no_ranking["rank_override"],
    )
    for field, expected_value in expected_no_ranking.items():
        if type(no_ranking[field]) is not type(expected_value) or (
            no_ranking[field] != expected_value
        ):
            raise DirectionalConfirmationError(
                f"ablations.no_ranking.{field} must prove a full_retrain"
            )
    ranked_warmup = _checkpoint_identity(
        "ablations.no_ranking.ranked_warmup_checkpoint",
        no_ranking["ranked_warmup_checkpoint"],
    )
    if (
        ranked_warmup["path"] != selection.parent_checkpoint_path
        or ranked_warmup["sha256"] != selection.parent_checkpoint_hash
        or ranked_warmup["size_bytes"] != selection.parent_checkpoint_size_bytes
    ):
        raise DirectionalConfirmationError(
            "ablations.no_ranking.ranked_warmup_checkpoint identity mismatch"
        )

    checkpoint = _validate_no_ranking_warmup(no_ranking["warmup"], selection=selection)
    joint = _candidate_metrics(
        no_ranking["joint_candidate"], name="ablations.no_ranking.joint_candidate"
    )
    joint_candidate = joint.candidate
    expected_joint_bindings = {
        "seed": selection.seed,
        "parent_checkpoint_path": checkpoint["path"],
        "parent_checkpoint_hash": checkpoint["sha256"],
        "parent_checkpoint_size_bytes": checkpoint["size_bytes"],
        "validation_cell_hash": selection.validation_cell_hash,
        "split_hash": selection.split_hash,
        "parent_projection_hash": selection.parent_projection_hash,
        "parent_projection_file_hash": selection.parent_projection_file_hash,
        "parent_projection_spec_hash": selection.parent_projection_spec_hash,
        "repo_commit": selection.repo_commit,
        "upstream_commit": selection.upstream_commit,
    }
    for field, expected in expected_joint_bindings.items():
        if joint_candidate[field] != expected:
            raise DirectionalConfirmationError(
                f"ablations.no_ranking.joint_candidate.{field} identity mismatch"
            )
    joint_config = _mapping(
        "ablations.no_ranking.joint_candidate.lineage.config",
        _mapping("joint lineage", joint_candidate["lineage"])["config"],
    )
    joint_ranking = _validate_ranking_record(
        "ablations.no_ranking.joint_candidate.lineage.config",
        joint_config,
    )
    if joint_ranking[0] != "no_ranking":
        raise DirectionalConfirmationError(
            "ablations.no_ranking.joint_candidate must bind the canonical "
            "no_ranking config"
        )
    if joint_candidate["checkpoint_hash"] in {
        selection.checkpoint_hash,
        selection.parent_checkpoint_hash,
        checkpoint["sha256"],
    }:
        raise DirectionalConfirmationError(
            "ablations.no_ranking.joint_candidate checkpoint is not create-new"
        )
    diagnostics = MappingProxyType(
        {
            **{name: result.diagnostics for name, result in check_results.items()},
            "no_ranking.joint_candidate": joint.check_diagnostics,
        }
    )
    return (
        joint,
        joint.checks_passed and all(result.passed for result in check_results.values()),
        diagnostics,
        tuple((label, path) for label, path, _ in evaluation_identities),
    )

def development_verdict(report: Mapping[str, object]) -> DevelopmentVerdict:

    payload = _mapping("development report", report)
    _exact_keys(
        "development report",
        payload,
        frozenset({"schema_version", "candidates", "selection", "rerun", "ablations"}),
    )
    if _exact_int("development report.schema_version", payload["schema_version"]) != 1:
        raise DirectionalConfirmationError(
            "development report.schema_version must be exactly 1"
        )
    metrics = _validate_candidate_pair(payload["candidates"])
    observed_selection = _selection_mapping(payload["selection"], name="selection")
    first_five = tuple(
        all(metric.first_five[index] for metric in metrics) for index in range(5)
    )
    seed_agreement = (
        metrics[0].directional_conclusion == metrics[1].directional_conclusion
    )
    if all(first_five) and seed_agreement:
        expected_selection = _selection_from_metrics(metrics, require_eligible=False)
        _assert_selection_equals(observed_selection, expected_selection)
    else:
        selected_metric = next(
            (
                metric
                for metric in metrics
                if metric.candidate["seed"] == observed_selection["seed"]
            ),
            None,
        )
        if selected_metric is None:
            raise DirectionalConfirmationError("selection seed identity mismatch")
        expected_selection = _selection_for_metric(selected_metric)
        identity_only = (
            "seed",
            "checkpoint_path",
            "checkpoint_hash",
            "parent_checkpoint_path",
            "parent_checkpoint_hash",
            "provenance_hash",
            "training_completion_hash",
            "checkpoint_completion_hash",
            "validation_record_hash",
            "split_hash",
            "parent_projection_hash",
            "parent_projection_file_hash",
            "parent_projection_spec_hash",
            "resume_hash",
            "ownership_hash",
            "validation_cell_hash",
            "example_statistics_hash",
            "config_hash",
            "repo_commit",
            "upstream_commit",
            "stochastic_budget_hash",
            "confirmation_hash",
        )
        expected_map = expected_selection.to_mapping()
        for field in identity_only:
            if observed_selection[field] != expected_map[field]:
                raise DirectionalConfirmationError(
                    f"selection {field} identity mismatch"
                )
    selected_index = next(
        index
        for index, metric in enumerate(metrics)
        if metric.candidate["seed"] == expected_selection.seed
    )
    (
        no_ranking_metrics,
        ablation_checks_passed,
        ablation_diagnostics,
        evaluation_artifact_paths,
    ) = _validate_ablations(
        payload["ablations"],
        selection=expected_selection,
        expected_records=metrics[selected_index].records,
    )
    first_five = (
        first_five[0] and ablation_checks_passed,
        *first_five[1:],
    )
    rerun_metrics = _validated_rerun_metrics(
        expected_selection,
        payload["rerun"],
        selected=metrics[selected_index].candidate,
    )
    scientific_artifact_paths = _scientific_artifact_path_registry(
        (
            *(
                (
                    f"candidate.seed-{metric.candidate['seed']}.validation",
                    str(metric.candidate["validation_record_path"]),
                )
                for metric in metrics
            ),
            (
                "selected_rerun.validation",
                str(rerun_metrics.candidate["validation_record_path"]),
            ),
            (
                "no_ranking.joint_candidate.validation",
                str(no_ranking_metrics.candidate["validation_record_path"]),
            ),
            *evaluation_artifact_paths,
        )
    )
    rerun_passed = _rerun_scientific_outcome(expected_selection, rerun_metrics)
    conditions = dict(
        zip(DEVELOPMENT_GATE_NAMES, (*first_five, rerun_passed), strict=True)
    )
    reasons = [name for name in DEVELOPMENT_GATE_NAMES if not conditions[name]]
    if not seed_agreement:
        reasons.append("seed_directional_conclusion_agreement")
    return DevelopmentVerdict(
        passed=not reasons,
        reasons=tuple(reasons),
        conditions=MappingProxyType(conditions),
        diagnostics=MappingProxyType(
            {
                "candidates": MappingProxyType(
                    {
                        str(metric.candidate["seed"]): metric.check_diagnostics
                        for metric in metrics
                    }
                ),
                "rerun": rerun_metrics.check_diagnostics,
                "ablations": ablation_diagnostics,
                "scientific_artifact_paths": scientific_artifact_paths,
            }
        ),
    )

def _verdict_record(verdict: DevelopmentVerdict) -> dict[str, object]:
    return {
        "passed": verdict.passed,
        "reasons": list(verdict.reasons),
        "conditions": dict(verdict.conditions),
        "diagnostics": {
            "candidates": {
                seed: {
                    **dict(values),
                    "family_ratios": [list(pair) for pair in values["family_ratios"]],
                }
                for seed, values in verdict.diagnostics["candidates"].items()
            },
            "rerun": {
                **dict(verdict.diagnostics["rerun"]),
                "family_ratios": [
                    list(pair) for pair in verdict.diagnostics["rerun"]["family_ratios"]
                ],
            },
            "ablations": {
                name: {
                    **dict(values),
                    "family_ratios": [list(pair) for pair in values["family_ratios"]],
                }
                for name, values in verdict.diagnostics["ablations"].items()
            },
            "scientific_artifact_paths": dict(
                verdict.diagnostics["scientific_artifact_paths"]
            ),
        },
    }

def _validate_strict_flow(
    inventory_value: object, examples_value: object
) -> tuple[tuple[Mapping[str, object], ...], str]:
    inventory = _mapping("strict_flow_inventory", inventory_value)
    _exact_keys("strict_flow_inventory", inventory, _STRICT_INVENTORY_KEYS)
    if (
        _exact_int("strict_flow_inventory.schema_version", inventory["schema_version"])
        != 1
    ):
        raise DirectionalConfirmationError(
            "strict_flow_inventory.schema_version must be exactly 1"
        )
    raw_rows = _sequence("strict_flow_inventory.rows", inventory["rows"])
    rows = tuple(
        _expected_cell(value, index=index) for index, value in enumerate(raw_rows)
    )
    if not rows or len(rows) != len(set(tuple(row.values()) for row in rows)):
        raise DirectionalConfirmationError(
            "strict_flow_inventory.rows is empty or contains duplicate identities"
        )
    if tuple(rows) != tuple(
        sorted(rows, key=lambda row: (row["family"], row["example_id"]))
    ):
        raise DirectionalConfirmationError(
            "strict_flow_inventory.rows must be canonically ordered"
        )
    if {row["family"] for row in rows} != set(FAMILIES):
        raise DirectionalConfirmationError(
            "strict_flow_inventory.rows is incomplete by family"
        )
    observed_counts = {
        family: sum(row["family"] == family for row in rows) for family in FAMILIES
    }
    if observed_counts != STRICT_FLOW_ROW_COUNTS:
        raise DirectionalConfirmationError(
            "strict flow row counts must be exactly binder=198, ame=292, antibody=1106"
        )
    observed_parent_counts = {
        family: len({row["parent_id"] for row in rows if row["family"] == family})
        for family in FAMILIES
    }
    if observed_parent_counts != STRICT_FLOW_PARENT_COUNTS:
        raise DirectionalConfirmationError(
            "strict independent parent counts must be exactly binder=131, ame=120, antibody=454"
        )
    rows_hash = _hash("strict_flow_inventory.rows_hash", inventory["rows_hash"])
    if rows_hash != _record_hash({"rows": list(rows)}):
        raise DirectionalConfirmationError(
            "strict_flow_inventory.rows_hash is inconsistent"
        )

    raw_examples = _sequence("flow_examples", examples_value)
    examples = tuple(
        _validate_example(value, index=index)
        for index, value in enumerate(raw_examples)
    )
    observed = {
        (
            record["family"],
            record["example_id"],
            record["parent_id"],
            record["cell_hash"],
        )
        for record in examples
    }
    expected = {
        (row["family"], row["example_id"], row["parent_id"], row["cell_hash"])
        for row in rows
    }
    if len(observed) != len(examples):
        raise DirectionalConfirmationError("flow_examples contains duplicate cells")
    if observed != expected:
        raise DirectionalConfirmationError(
            "flow_examples is incomplete or inconsistent with the sealed strict rows"
        )
    return examples, _record_hash(inventory)

def _generation_identity(value: object, *, name: str) -> dict[str, object]:
    record = _mapping(name, value)
    _exact_keys(name, record, _GENERATION_IDENTITY_KEYS)
    family = _string(f"{name}.family", record["family"])
    if family not in FAMILIES:
        raise DirectionalConfirmationError(f"{name}.family is unknown")
    parent_id = _string(f"{name}.parent_id", record["parent_id"])
    seed = _exact_int(f"{name}.seed", record["seed"])
    if seed not in (42001, 42002):
        raise DirectionalConfirmationError(f"{name}.seed is not frozen")
    arm = _string(f"{name}.arm", record["arm"])
    if arm not in ("adaptive", "best_frozen_fixed_control"):
        raise DirectionalConfirmationError(f"{name}.arm is unknown")
    cell_hash = _hash(f"{name}.cell_hash", record["cell_hash"])
    identity = {"family": family, "parent_id": parent_id, "seed": seed, "arm": arm}
    if cell_hash != _record_hash(identity):
        raise DirectionalConfirmationError(f"{name}.cell_hash identity mismatch")
    return {**identity, "cell_hash": cell_hash}

def _validate_blind_checks(
    value: object, *, records: Sequence[Mapping[str, object]]
) -> _CheckMetrics:
    return _validate_check_bundle(value, records=records, name="blind_checks")

def _validate_generation(
    inventory_value: object,
    availability_value: object,
    cells_value: object,
) -> tuple[bool, str, str]:
    inventory = _mapping("generation_inventory", inventory_value)
    _exact_keys("generation_inventory", inventory, _GENERATION_INVENTORY_KEYS)
    if (
        _exact_int("generation_inventory.schema_version", inventory["schema_version"])
        != 1
    ):
        raise DirectionalConfirmationError(
            "generation_inventory.schema_version must be exactly 1"
        )
    parents_mapping = _mapping(
        "generation_inventory.parents_by_family", inventory["parents_by_family"]
    )
    _exact_keys(
        "generation_inventory.parents_by_family",
        parents_mapping,
        frozenset(FAMILIES),
    )
    parents_by_family: dict[str, tuple[str, ...]] = {}
    for family in FAMILIES:
        values = tuple(
            _string(f"generation_inventory.parents_by_family.{family}[{index}]", value)
            for index, value in enumerate(
                _sequence(
                    f"generation_inventory.parents_by_family.{family}",
                    parents_mapping[family],
                )
            )
        )
        if len(values) != 20 or len(values) != len(set(values)):
            raise DirectionalConfirmationError(
                f"generation_inventory.parents_by_family.{family} must contain exactly 20 unique sealed parents"
            )
        expected_order = tuple(
            sorted(
                values,
                key=lambda parent: hashlib.sha256(
                    f"{family}{parent}".encode()
                ).hexdigest(),
            )
        )
        if values != expected_order:
            raise DirectionalConfirmationError(
                f"generation_inventory.parents_by_family.{family} selection order drifted"
            )
        parents_by_family[family] = values
    parent_hash = _hash(
        "generation_inventory.parent_inventory_hash",
        inventory["parent_inventory_hash"],
    )
    if parent_hash != _record_hash(
        {
            "parents_by_family": {
                family: list(parents_by_family[family]) for family in FAMILIES
            }
        }
    ):
        raise DirectionalConfirmationError(
            "generation_inventory.parent_inventory_hash is inconsistent"
        )
    seeds = _sequence("generation_inventory.seeds", inventory["seeds"])
    if list(seeds) != [42001, 42002] or any(type(value) is not int for value in seeds):
        raise DirectionalConfirmationError(
            "generation_inventory.seeds must be exactly 42001 and 42002"
        )
    arms = _sequence("generation_inventory.arms", inventory["arms"])
    if list(arms) != ["adaptive", "best_frozen_fixed_control"]:
        raise DirectionalConfirmationError(
            "generation_inventory.arms must contain the two frozen arms"
        )
    raw_preregistered = _sequence(
        "generation_inventory.preregistered_cells",
        inventory["preregistered_cells"],
    )
    preregistered = tuple(
        _generation_identity(
            value, name=f"generation_inventory.preregistered_cells[{index}]"
        )
        for index, value in enumerate(raw_preregistered)
    )
    expected_preregistered = tuple(
        {
            **identity,
            "cell_hash": _record_hash(identity),
        }
        for family in FAMILIES
        for parent_id in parents_by_family[family]
        for seed in (42001, 42002)
        for arm in ("adaptive", "best_frozen_fixed_control")
        for identity in (
            {"family": family, "parent_id": parent_id, "seed": seed, "arm": arm},
        )
    )
    if len(preregistered) != 240 or preregistered != expected_preregistered:
        raise DirectionalConfirmationError(
            "generation_inventory must bind exactly 240 preregistered cells"
        )
    preregistered_hash = _hash(
        "generation_inventory.preregistered_cells_hash",
        inventory["preregistered_cells_hash"],
    )
    if preregistered_hash != _record_hash({"cells": list(preregistered)}):
        raise DirectionalConfirmationError(
            "generation_inventory.preregistered_cells_hash is inconsistent"
        )
    availability = _mapping("availability_manifest", availability_value)
    _exact_keys("availability_manifest", availability, _AVAILABILITY_MANIFEST_KEYS)
    if (
        _exact_int(
            "availability_manifest.schema_version", availability["schema_version"]
        )
        != 1
    ):
        raise DirectionalConfirmationError(
            "availability_manifest.schema_version must be exactly 1"
        )
    availability_hash = _record_hash(availability)
    if (
        _hash(
            "generation_inventory.availability_manifest_hash",
            inventory["availability_manifest_hash"],
        )
        != availability_hash
    ):
        raise DirectionalConfirmationError(
            "generation_inventory availability manifest identity mismatch"
        )
    exclusions = _sequence(
        "availability_manifest.deterministic_exclusions",
        availability["deterministic_exclusions"],
    )
    excluded: set[str] = set()
    preregistered_hashes = {str(cell["cell_hash"]) for cell in preregistered}
    for index, value in enumerate(exclusions):
        name = f"availability_manifest.deterministic_exclusions[{index}]"
        record = _mapping(name, value)
        _exact_keys(name, record, frozenset({"cell_hash", "reason"}))
        cell_hash = _hash(f"{name}.cell_hash", record["cell_hash"])
        _string(f"{name}.reason", record["reason"])
        if cell_hash not in preregistered_hashes or cell_hash in excluded:
            raise DirectionalConfirmationError(
                f"{name} is duplicate or not preregistered"
            )
        excluded.add(cell_hash)
    excluded_by_parent: dict[tuple[str, str], set[str]] = {}
    for cell in preregistered:
        cell_hash = str(cell["cell_hash"])
        if cell_hash in excluded:
            key = (str(cell["family"]), str(cell["parent_id"]))
            excluded_by_parent.setdefault(key, set()).add(cell_hash)
    if any(len(cell_hashes) != 4 for cell_hashes in excluded_by_parent.values()):
        raise DirectionalConfirmationError(
            "generation deterministic exclusions leave seed or arm cells unpaired"
        )

    raw_cells = _sequence("generation_cells", cells_value)
    observed: list[dict[str, object]] = []
    finite = True
    for index, value in enumerate(raw_cells):
        name = f"generation_cells[{index}]"
        record = _mapping(name, value)
        _exact_keys(name, record, _GENERATION_CELL_KEYS)
        identity = _generation_identity(
            {field: record[field] for field in _GENERATION_IDENTITY_KEYS}, name=name
        )
        _sufficient(f"{name}.loss", record["loss"])
        count = _exact_int(f"{name}.non_finite_count", record["non_finite_count"])
        if count < 0:
            raise DirectionalConfirmationError(
                f"{name}.non_finite_count must not be negative"
            )
        events = _sequence(f"{name}.non_finite_events", record["non_finite_events"])
        for event_index, event in enumerate(events):
            _string(f"{name}.non_finite_events[{event_index}]", event)
        if count != len(events):
            raise DirectionalConfirmationError(
                f"{name}.non_finite_count is inconsistent with events"
            )
        finite = finite and count == 0
        observed.append(identity)
    observed_hashes = [str(cell["cell_hash"]) for cell in observed]
    if len(observed_hashes) != len(set(observed_hashes)):
        raise DirectionalConfirmationError("generation_cells contains duplicate cells")
    expected_hashes = preregistered_hashes - excluded
    if set(observed_hashes) != expected_hashes:
        raise DirectionalConfirmationError(
            "generation_cells is incomplete or unpaired after deterministic exclusions"
        )
    coverage = {
        (str(cell["family"]), int(cell["seed"]), str(cell["arm"])) for cell in observed
    }
    expected_coverage = {
        (family, seed, arm)
        for family in FAMILIES
        for seed in (42001, 42002)
        for arm in ("adaptive", "best_frozen_fixed_control")
    }
    if coverage != expected_coverage:
        raise DirectionalConfirmationError(
            "generation screen is vacuous or lacks complete family/seed/arm coverage"
        )
    return finite, _record_hash(inventory), availability_hash

def _shared_time_routes(records: Sequence[Mapping[str, object]]) -> dict[int, str]:
    routes: dict[int, str] = {}
    for bin_id in sorted({int(record["time_bin"]["index"]) for record in records}):
        subset = [
            record for record in records if int(record["time_bin"]["index"]) == bin_id
        ]
        routes[bin_id] = min(
            HARD_ROUTES,
            key=lambda route: (
                sum(
                    _family_ratio(
                        [record for record in subset if record["family"] == family],
                        route,
                    )
                    for family in FAMILIES
                    if any(record["family"] == family for record in subset)
                )
                / len({record["family"] for record in subset}),
                route,
            ),
        )
    return routes

def _blind_metrics(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], dict[str, float], dict[str, float], float]:
    winners: dict[str, str] = {}
    spreads: dict[str, float] = {}
    for family in FAMILIES:
        subset = [record for record in records if record["family"] == family]
        totals = {
            route: sum(_numerator(record, route) for record in subset)
            for route in HARD_ROUTES
        }
        winner = min(HARD_ROUTES, key=lambda route: (totals[route], route))
        worst = max(HARD_ROUTES, key=lambda route: (totals[route], route))
        if totals[worst] <= 0.0:
            raise DirectionalConfirmationError(
                "blind winner-over-worst denominator is non-positive"
            )
        winners[family] = winner
        spreads[family] = (totals[worst] - totals[winner]) / totals[worst]

    global_route = min(
        HARD_ROUTES,
        key=lambda route: (_equal_family_route_score(records, route), route),
    )
    global_score = _equal_family_route_score(records, global_route)
    time_routes = _shared_time_routes(records)
    time_score = _time_only_score(records)
    shared_strategy = "global" if global_score <= time_score else "time_only"
    qualifying_oracle = 0
    for record in records:
        control_route = (
            global_route
            if shared_strategy == "global"
            else time_routes[int(record["time_bin"]["index"])]
        )
        control = _numerator(record, control_route)
        oracle = float(record["per_residue_oracle"]["numerator"])
        if control <= 0.0:
            raise DirectionalConfirmationError(
                "blind oracle-example control denominator is non-positive"
            )
        qualifying_oracle += _inclusive_at_least(
            (control - oracle) / control, MIN_ORACLE_EXAMPLE_GAIN
        )
    scores = {
        "adaptive": _equal_family_route_score(records, "adaptive"),
        "best_fixed": _best_fixed_score(records),
        "global": global_score,
        "time_only": time_score,
        "oracle": _oracle_score(records),
    }
    return winners, spreads, scores, qualifying_oracle / len(records)

def _validate_preblind(
    value: object,
) -> tuple[CandidateSelection, bool, _CandidateMetrics, Mapping[str, object]]:
    record = _mapping("preblind", value)
    _exact_keys("preblind", record, _PREBLIND_KEYS)
    development = _mapping("preblind.development_report", record["development_report"])
    if _hash(
        "preblind.development_report_hash", record["development_report_hash"]
    ) != _record_hash(development):
        raise DirectionalConfirmationError(
            "preblind.development_report_hash is inconsistent"
        )
    computed_verdict = development_verdict(development)
    expected_verdict = _verdict_record(computed_verdict)
    observed_verdict = _mapping(
        "preblind.development_verdict", record["development_verdict"]
    )
    if _canonical(observed_verdict) != _canonical(expected_verdict):
        raise DirectionalConfirmationError(
            "preblind.development_verdict differs from real development validation"
        )
    if _hash(
        "preblind.development_verdict_hash", record["development_verdict_hash"]
    ) != _record_hash(observed_verdict):
        raise DirectionalConfirmationError(
            "preblind.development_verdict_hash is inconsistent"
        )
    selection = select_candidate(
        _sequence("development candidates", development["candidates"])
    )
    selection_record = _selection_mapping(
        record["selection"], name="preblind.selection"
    )
    _assert_selection_equals(selection_record, selection)
    if _hash("preblind.selection_hash", record["selection_hash"]) != _record_hash(
        selection_record
    ):
        raise DirectionalConfirmationError("preblind.selection_hash is inconsistent")
    development_selection = _selection_mapping(
        development["selection"], name="preblind.development_report.selection"
    )
    _assert_selection_equals(development_selection, selection)
    rerun = _mapping("preblind.rerun", record["rerun"])
    if _hash("preblind.rerun_hash", record["rerun_hash"]) != _record_hash(rerun):
        raise DirectionalConfirmationError("preblind.rerun_hash is inconsistent")
    if _canonical(rerun) != _canonical(development["rerun"]):
        raise DirectionalConfirmationError(
            "preblind.rerun differs from the canonical development rerun"
        )
    candidates = _sequence("development candidates", development["candidates"])
    selected_candidate = next(
        candidate
        for candidate in candidates
        if _mapping("development candidate", candidate)["seed"] == selection.seed
    )
    rerun_metrics = _validated_rerun_metrics(
        selection, rerun, selected=_mapping("selected candidate", selected_candidate)
    )
    rerun_passed = _rerun_scientific_outcome(selection, rerun_metrics)
    selected_metrics = _candidate_metrics(selected_candidate, name="selected candidate")
    no_ranking_metrics, _, _, _ = _validate_ablations(
        development["ablations"],
        selection=selection,
        expected_records=selected_metrics.records,
    )
    seal_payload = _mapping("preblind.seal_payload", record["seal_payload"])
    _exact_keys("preblind.seal_payload", seal_payload, _SEAL_PAYLOAD_KEYS)
    seal_hash = _hash("preblind.seal_hash", record["seal_hash"])
    if seal_hash != _record_hash(seal_payload):
        raise DirectionalConfirmationError("preblind.seal_hash is inconsistent")
    return (
        selection,
        computed_verdict.passed and rerun_passed,
        no_ranking_metrics,
        seal_payload,
    )

def blind_verdict(
    report: Mapping[str, object],
    *,
    expected_seal_hash: str,
    expected_strict_inventory_hash: str,
    expected_generation_inventory_hash: str,
    expected_availability_manifest_hash: str,
) -> BlindVerdict:

    trusted_seal_hash = _hash("expected_seal_hash", expected_seal_hash)
    trusted_strict_hash = _hash(
        "expected_strict_inventory_hash", expected_strict_inventory_hash
    )
    trusted_generation_hash = _hash(
        "expected_generation_inventory_hash", expected_generation_inventory_hash
    )
    trusted_availability_hash = _hash(
        "expected_availability_manifest_hash", expected_availability_manifest_hash
    )
    payload = _mapping("blind report", report)
    _exact_keys("blind report", payload, _BLIND_REPORT_KEYS)
    if _exact_int("blind report.schema_version", payload["schema_version"]) != 1:
        raise DirectionalConfirmationError(
            "blind report.schema_version must be exactly 1"
        )
    confirmation = _assert_confirmation(payload["confirmation"])
    records, strict_inventory_hash = _validate_strict_flow(
        payload["strict_flow_inventory"], payload["flow_examples"]
    )
    if strict_inventory_hash != trusted_strict_hash:
        raise DirectionalConfirmationError(
            "expected_strict_inventory_hash differs from the sealed report inventory"
        )
    blind_checks = _validate_blind_checks(payload["blind_checks"], records=records)
    (
        finite_generation,
        generation_inventory_hash,
        availability_manifest_hash,
    ) = _validate_generation(
        payload["generation_inventory"],
        payload["availability_manifest"],
        payload["generation_cells"],
    )
    if generation_inventory_hash != trusted_generation_hash:
        raise DirectionalConfirmationError(
            "expected_generation_inventory_hash differs from the sealed report inventory"
        )
    if availability_manifest_hash != trusted_availability_hash:
        raise DirectionalConfirmationError(
            "expected_availability_manifest_hash differs from the predeclared manifest"
        )
    selection, preblind_passed, no_ranking, preblind_seal = _validate_preblind(
        payload["preblind"]
    )
    if selection.confirmation_hash != _record_hash(confirmation):
        raise DirectionalConfirmationError(
            "preblind selection confirmation_hash does not bind blind confirmation"
        )
    seal_payload = _mapping("blind report.seal_payload", payload["seal_payload"])
    _exact_keys("blind report.seal_payload", seal_payload, _SEAL_PAYLOAD_KEYS)
    if _canonical(seal_payload) != _canonical(preblind_seal):
        raise DirectionalConfirmationError(
            "blind report.seal_payload differs from the preblind seal payload"
        )
    expected_seal = {
        "schema_version": 1,
        "development_report_hash": payload["preblind"]["development_report_hash"],
        "development_verdict_hash": payload["preblind"]["development_verdict_hash"],
        "selection_hash": payload["preblind"]["selection_hash"],
        "rerun_hash": payload["preblind"]["rerun_hash"],
        "ablations_hash": _record_hash(
            payload["preblind"]["development_report"]["ablations"]
        ),
        "strict_flow_inventory_hash": strict_inventory_hash,
        "generation_inventory_hash": generation_inventory_hash,
        "availability_manifest_hash": availability_manifest_hash,
    }
    if _canonical(seal_payload) != _canonical(expected_seal):
        raise DirectionalConfirmationError(
            "blind report.seal_payload does not bind every sealed identity"
        )
    seal_hash = _hash("blind report.seal_hash", payload["seal_hash"])
    if (
        seal_hash != _record_hash(seal_payload)
        or seal_hash != payload["preblind"]["seal_hash"]
    ):
        raise DirectionalConfirmationError("blind report.seal_hash is inconsistent")
    if seal_hash != trusted_seal_hash:
        raise DirectionalConfirmationError(
            "expected_seal_hash differs from the externally verified Task 10 seal"
        )

    winners, spreads, scores, oracle_fraction = _blind_metrics(records)
    control_score = min(
        scores[control] for control in ("best_fixed", "global", "time_only")
    )
    if control_score <= 0.0:
        raise DirectionalConfirmationError(
            "blind best-control denominator is non-positive"
        )
    oracle_gap = control_score - scores["oracle"]
    if oracle_gap <= 0.0:
        raise DirectionalConfirmationError(
            "blind oracle-gap denominator is non-positive"
        )
    adaptive_gain = 1.0 - scores["adaptive"] / control_score
    oracle_gap_recovery = (control_score - scores["adaptive"]) / oracle_gap
    distinct_winners = set(winners.values())
    qualifying_winners = {
        winners[family]
        for family in FAMILIES
        if _inclusive_at_least(spreads[family], MIN_WINNER_OVER_WORST_GAIN)
    }
    ranking_necessary = (
        not all(no_ranking.first_five)
        or no_ranking.directional_conclusion != selection.directional_conclusion
    )
    conditions = {
        BLIND_GATE_NAMES[0]: len(distinct_winners) >= 2,
        BLIND_GATE_NAMES[1]: len(qualifying_winners) >= 2,
        BLIND_GATE_NAMES[2]: _inclusive_at_least(
            oracle_fraction, MIN_ORACLE_EXAMPLE_FRACTION
        ),
        BLIND_GATE_NAMES[3]: _inclusive_at_least(adaptive_gain, MIN_ADAPTIVE_GAIN),
        BLIND_GATE_NAMES[4]: _inclusive_at_least(
            oracle_gap_recovery, MIN_ORACLE_GAP_RECOVERY
        ),
        BLIND_GATE_NAMES[5]: blind_checks.passed and finite_generation,
        BLIND_GATE_NAMES[6]: preblind_passed,
    }
    reasons = tuple(name for name in BLIND_GATE_NAMES if not conditions[name])
    passed = not reasons
    claim_label = (
        "terminal negative result"
        if not passed
        else "task-label-free self-supervised directional routing"
        if ranking_necessary
        else "task-label-free directional routing"
    )
    return BlindVerdict(
        passed=passed,
        reasons=reasons,
        conditions=MappingProxyType(conditions),
        terminal=True,
        claim_label=claim_label,
        diagnostics=MappingProxyType({"blind_checks": blind_checks.diagnostics}),
    )

__all__ = [
    "BLIND_GATE_NAMES",
    "BlindVerdict",
    "CandidateSelection",
    "DEVELOPMENT_GATE_NAMES",
    "DevelopmentVerdict",
    "DirectionalConfirmationError",
    "METRIC_SOURCE_PATHS",
    "assert_rerun_reproduces",
    "blind_verdict",
    "development_verdict",
    "metric_source_inventory",
    "select_candidate",
]

MUTABLE_SOURCE_REALM_EXCLUSIONS: Mapping[str, type] = MappingProxyType({})
