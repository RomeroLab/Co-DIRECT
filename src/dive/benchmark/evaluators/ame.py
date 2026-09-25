
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import overload

from dive.benchmark.evaluators.binder import (
    _evaluate_authenticated_family,
    _optional_from_evaluator,
)
from dive.benchmark.evaluators.contracts import (
    EvaluatorAvailability,
    EvaluatorRegistry,
    MetricValue,
    RawFieldContract,
)
from dive.benchmark.evaluators.records import FamilyMetricRecord, RawValue, Runner
from dive.benchmark.evaluators.worker_io import DisposableEvidencePolicy
from dive.benchmark.evaluators.worker_protocol import (
    WorkerFailureReason,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResult,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AME_FIELD_CONTRACTS = {
    "binder_scRMSD_bb3": ("float", "angstrom", True),
    "binder_scRMSD_bb3_source_present": ("boolean", "boolean", True),
    "clash_rate": ("float", "normalized", False),
    "diversity": ("float", "normalized", False),
    "exact_motif_sequence_recovery": ("boolean", "boolean", True),
    "exact_motif_sequence_recovery_source_present": ("boolean", "boolean", True),
    "foldability": ("float", "normalized", False),
    "ligand_clash": ("boolean", "boolean", True),
    "ligand_clash_source_present": ("boolean", "boolean", True),
    "local_all_atom_quality": ("float", "normalized", False),
    "min_ipAE": ("float", "raw_rf3", True),
    "min_ipAE_source_present": ("boolean", "boolean", True),
    "motif_all_atom_rmsd": ("float", "angstrom", True),
    "motif_all_atom_rmsd_source_present": ("boolean", "boolean", True),
    "motif_geometry": ("float", "normalized", False),
    "novelty": ("float", "normalized", False),
    "redesign_id": ("string", "sequence", True),
    "sequence": ("string", "sequence", True),
}
_AME_ROW_FIELDS = frozenset(
    {
        "redesign_id",
        "sequence",
        "binder_scRMSD_bb3",
        "motif_all_atom_rmsd",
        "exact_motif_sequence_recovery",
        "ligand_clash",
        "min_ipAE",
        "binder_scRMSD_bb3_source_present",
        "motif_all_atom_rmsd_source_present",
        "exact_motif_sequence_recovery_source_present",
        "ligand_clash_source_present",
        "min_ipAE_source_present",
    }
)
_AME_RAW_FIELDS = frozenset(set(_AME_FIELD_CONTRACTS) - _AME_ROW_FIELDS)
AME_OPTIONAL_METRIC_NAMES = tuple(sorted(_AME_RAW_FIELDS))
_SOURCE_PRESENCE_FIELDS = {
    "binder_scRMSD_bb3": "binder_scRMSD_bb3_source_present",
    "motif_all_atom_rmsd": "motif_all_atom_rmsd_source_present",
    "exact_motif_sequence_recovery": "exact_motif_sequence_recovery_source_present",
    "ligand_clash": "ligand_clash_source_present",
    "min_ipAE": "min_ipAE_source_present",
}

def _refuse(reason: WorkerFailureReason, detail: str) -> None:
    raise WorkerProtocolError(reason, detail)

def _validated_manifest_hash(value: object) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid evaluator manifest hash"
        )
    return value

@dataclass(frozen=True, slots=True)
class AmeRedesign:

    redesign_id: str
    sequence: str
    binder_scRMSD_bb3: float
    motif_all_atom_rmsd: float
    exact_motif_sequence_recovery: bool
    ligand_clash: bool
    min_ipAE: float
    binder_scRMSD_bb3_source_present: bool
    motif_all_atom_rmsd_source_present: bool
    exact_motif_sequence_recovery_source_present: bool
    ligand_clash_source_present: bool
    min_ipAE_source_present: bool
    raw: Mapping[str, str | bool | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.redesign_id) is not str or not self.redesign_id:
            _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid redesign_id")
        if type(self.sequence) is not str or not self.sequence:
            _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid sequence")
        for name, value in (
            ("binder_scRMSD_bb3", self.binder_scRMSD_bb3),
            ("motif_all_atom_rmsd", self.motif_all_atom_rmsd),
            ("min_ipAE", self.min_ipAE),
        ):
            if type(value) is not float:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID, f"{name} must be a float"
                )
            if not math.isfinite(value):
                _refuse(WorkerFailureReason.NONFINITE_METRIC, f"{name} is non-finite")
        for name, value in (
            ("exact_motif_sequence_recovery", self.exact_motif_sequence_recovery),
            ("ligand_clash", self.ligand_clash),
        ):
            if type(value) is not bool:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID,
                    f"{name} must be a boolean",
                )
        for name, value in (
            ("binder_scRMSD_bb3_source_present", self.binder_scRMSD_bb3_source_present),
            (
                "motif_all_atom_rmsd_source_present",
                self.motif_all_atom_rmsd_source_present,
            ),
            (
                "exact_motif_sequence_recovery_source_present",
                self.exact_motif_sequence_recovery_source_present,
            ),
            ("ligand_clash_source_present", self.ligand_clash_source_present),
            ("min_ipAE_source_present", self.min_ipAE_source_present),
        ):
            if type(value) is not bool:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID,
                    f"{name} must be a boolean",
                )
            if not value:
                _refuse(
                    WorkerFailureReason.UPSTREAM_NUMERIC_SENTINEL,
                    f"source field absent or defaulted: {name.removesuffix('_source_present')}",
                )
        if not isinstance(self.raw, Mapping):
            _refuse(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "row raw values must be a mapping",
            )
        checked: dict[str, str | bool | float] = {}
        for name, value in self.raw.items():
            if type(name) is not str or not name:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid raw field name"
                )
            if name in _AME_ROW_FIELDS:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID,
                    f"primary raw shadow: {name}",
                )
            if name not in _AME_RAW_FIELDS:
                _refuse(
                    WorkerFailureReason.UNKNOWN_OUTPUT_FIELD,
                    f"unknown raw field: {name}",
                )
            expected_type = {"string": str, "boolean": bool, "float": float}[
                _AME_FIELD_CONTRACTS[name][0]
            ]
            if type(value) is not expected_type:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID,
                    f"invalid raw field: {name}",
                )
            if type(value) is float and not math.isfinite(value):
                _refuse(WorkerFailureReason.NONFINITE_METRIC, f"{name} is non-finite")
            checked[name] = value
        object.__setattr__(self, "raw", MappingProxyType(checked))

def _contracts_by_name(field_contracts: object) -> Mapping[str, RawFieldContract]:
    if not isinstance(field_contracts, tuple) or not field_contracts:
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "raw field contract is invalid"
        )
    contracts: dict[str, RawFieldContract] = {}
    for contract in field_contracts:
        if not isinstance(contract, RawFieldContract) or contract.name in contracts:
            _refuse(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "raw field contract is invalid",
            )
        if (
            type(contract.name) is not str
            or type(contract.value_type) is not str
            or type(contract.unit) is not str
            or type(contract.required) is not bool
        ):
            _refuse(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "raw field contract is invalid",
            )
        contracts[contract.name] = contract
    if set(contracts) != set(_AME_FIELD_CONTRACTS):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "raw field contract names drifted",
        )
    for name, (value_type, unit, required) in _AME_FIELD_CONTRACTS.items():
        contract = contracts[name]
        if (
            contract.value_type != value_type
            or contract.unit != unit
            or contract.required is not required
        ):
            _refuse(WorkerFailureReason.UNIT_CONTRACT_ERROR, f"contract drift: {name}")
    return MappingProxyType(contracts)

def _typed_vector(
    name: str, value: object, contract: RawFieldContract, length: int | None
) -> tuple[str, ...] | tuple[bool, ...] | tuple[float, ...]:
    if type(value) is not tuple:
        _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, f"{name} must be a vector")
    if not value or (length is not None and len(value) != length):
        _refuse(
            WorkerFailureReason.VECTOR_LENGTH_MISMATCH, f"invalid vector length: {name}"
        )
    scalar_type = {"string": str, "boolean": bool, "float": float}.get(
        contract.value_type
    )
    if scalar_type is None or any(type(item) is not scalar_type for item in value):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            f"raw field has wrong type: {name}",
        )
    if scalar_type is float and not all(math.isfinite(item) for item in value):
        _refuse(
            WorkerFailureReason.NONFINITE_METRIC, f"raw field is non-finite: {name}"
        )
    return value

def parse_ame_rows(
    raw: Mapping[str, RawValue], field_contracts: tuple[RawFieldContract, ...]
) -> tuple[AmeRedesign, ...]:

    contracts = _contracts_by_name(field_contracts)
    if not isinstance(raw, Mapping):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "raw result must be a mapping"
        )
    unknown = set(raw) - set(contracts)
    if unknown:
        _refuse(
            WorkerFailureReason.UNKNOWN_OUTPUT_FIELD,
            f"unknown raw fields: {sorted(unknown)}",
        )
    missing = [
        name
        for name, contract in contracts.items()
        if contract.required and name not in raw
    ]
    if missing:
        _refuse(
            WorkerFailureReason.MISSING_REQUIRED_METRIC,
            f"missing raw fields: {sorted(missing)}",
        )

    vectors: dict[str, tuple[str, ...] | tuple[bool, ...] | tuple[float, ...]] = {}
    vector_length: int | None = None
    for name, contract in contracts.items():
        if name not in raw:
            continue
        vector = _typed_vector(name, raw[name], contract, vector_length)
        vector_length = len(vector)
        vectors[name] = vector
    if vector_length is None:
        _refuse(WorkerFailureReason.VECTOR_LENGTH_MISMATCH, "raw result has no vectors")

    redesign_ids = vectors["redesign_id"]
    if len(set(redesign_ids)) != len(redesign_ids) or any(
        not item for item in redesign_ids
    ):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "redesign identities must be unique",
        )
    if any(not item for item in vectors["sequence"]):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "sequences must be non-empty"
        )
    for value_name, presence_name in _SOURCE_PRESENCE_FIELDS.items():
        presence = vectors[presence_name]
        if any(not item for item in presence):
            _refuse(
                WorkerFailureReason.UPSTREAM_NUMERIC_SENTINEL,
                f"source field absent or defaulted: {value_name}",
            )

    rows: list[AmeRedesign] = []
    for index in range(vector_length):
        row_raw = {
            name: vector[index]
            for name, vector in vectors.items()
            if name not in _AME_ROW_FIELDS
        }
        rows.append(
            AmeRedesign(
                redesign_ids[index],
                vectors["sequence"][index],
                vectors["binder_scRMSD_bb3"][index],
                vectors["motif_all_atom_rmsd"][index],
                vectors["exact_motif_sequence_recovery"][index],
                vectors["ligand_clash"][index],
                vectors["min_ipAE"][index],
                vectors["binder_scRMSD_bb3_source_present"][index],
                vectors["motif_all_atom_rmsd_source_present"][index],
                vectors["exact_motif_sequence_recovery_source_present"][index],
                vectors["ligand_clash_source_present"][index],
                vectors["min_ipAE_source_present"][index],
                row_raw,
            )
        )
    return tuple(rows)

def _reduce_rows(
    rows: tuple[AmeRedesign, ...], manifest_hash: str
) -> FamilyMetricRecord:
    manifest_hash = _validated_manifest_hash(manifest_hash)
    if not isinstance(rows, tuple) or not rows:
        _refuse(
            WorkerFailureReason.VECTOR_LENGTH_MISMATCH,
            "ame result has no redesign rows",
        )
    if any(not isinstance(row, AmeRedesign) for row in rows):
        _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, "ame rows are invalid")
    if len({row.redesign_id for row in rows}) != len(rows):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "redesign identities must be unique",
        )
    raw_names = set(rows[0].raw)
    if any(set(row.raw) != raw_names for row in rows):
        _refuse(WorkerFailureReason.VECTOR_LENGTH_MISMATCH, "raw fields differ by row")
    raw: dict[str, RawValue] = {
        "redesign_id": tuple(row.redesign_id for row in rows),
        "sequence": tuple(row.sequence for row in rows),
        "binder_scRMSD_bb3": tuple(row.binder_scRMSD_bb3 for row in rows),
        "motif_all_atom_rmsd": tuple(row.motif_all_atom_rmsd for row in rows),
        "exact_motif_sequence_recovery": tuple(
            row.exact_motif_sequence_recovery for row in rows
        ),
        "ligand_clash": tuple(row.ligand_clash for row in rows),
        "min_ipAE": tuple(row.min_ipAE for row in rows),
        "binder_scRMSD_bb3_source_present": tuple(
            row.binder_scRMSD_bb3_source_present for row in rows
        ),
        "motif_all_atom_rmsd_source_present": tuple(
            row.motif_all_atom_rmsd_source_present for row in rows
        ),
        "exact_motif_sequence_recovery_source_present": tuple(
            row.exact_motif_sequence_recovery_source_present for row in rows
        ),
        "ligand_clash_source_present": tuple(
            row.ligand_clash_source_present for row in rows
        ),
        "min_ipAE_source_present": tuple(row.min_ipAE_source_present for row in rows),
    }
    for name in raw_names:
        raw[name] = tuple(row.raw[name] for row in rows)
    success = any(
        row.binder_scRMSD_bb3 <= 2.0
        and row.motif_all_atom_rmsd <= 1.5
        and row.exact_motif_sequence_recovery
        and not row.ligand_clash
        for row in rows
    )
    return FamilyMetricRecord(
        family="ame",
        primary=MetricValue(
            "ame_motif_ligand_success", float(success), "higher", manifest_hash
        ),
        raw=raw,
    )

@overload
def reduce_ame(
    rows: tuple[AmeRedesign, ...], manifest_hash: str
) -> FamilyMetricRecord: ...

@overload
def reduce_ame(
    result: WorkerResult, request: WorkerRequest, evaluator: object
) -> FamilyMetricRecord: ...

def reduce_ame(
    rows: tuple[AmeRedesign, ...] | WorkerResult,
    manifest_hash: str | WorkerRequest,
    evaluator: object | None = None,
) -> FamilyMetricRecord:

    if evaluator is None:
        return _reduce_rows(rows, manifest_hash)
    if not isinstance(rows, WorkerResult) or not isinstance(
        manifest_hash, WorkerRequest
    ):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "invalid verifier reducer arguments",
        )
    field_contracts = getattr(evaluator, "raw_fields", None)
    if not isinstance(field_contracts, tuple):
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "raw fields are unauthenticated",
        )
    reduced = _reduce_rows(
        parse_ame_rows(rows.raw, field_contracts),
        manifest_hash.evaluator_semantic_sha256,
    )
    return FamilyMetricRecord(
        reduced.family,
        reduced.primary,
        reduced.raw,
        _optional_from_evaluator(evaluator, manifest_hash.evaluator_semantic_sha256),
    )

@dataclass(frozen=True, slots=True)
class AmeEvaluationRequest:

    generated_path: str
    native_path: str
    frozen_roles: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.generated_path, str) or not self.generated_path:
            raise ValueError("generated_path must be a non-empty string")
        if not isinstance(self.native_path, str) or not self.native_path:
            raise ValueError("native_path must be a non-empty string")
        if not isinstance(self.frozen_roles, Mapping):
            raise TypeError("frozen_roles must be a mapping")
        object.__setattr__(
            self, "frozen_roles", MappingProxyType(dict(self.frozen_roles))
        )

def evaluate_ame(
    request: WorkerRequest,
    registry: EvaluatorRegistry,
    availability: EvaluatorAvailability,
    *,
    runner: Runner | None = None,
    policy: DisposableEvidencePolicy | None = None,
) -> FamilyMetricRecord:

    return _evaluate_authenticated_family(
        family="ame",
        primary_name="ame_motif_ligand_success",
        reducer=reduce_ame,
        request=request,
        registry=registry,
        availability=availability,
        runner=runner,
        policy=policy,
    )
