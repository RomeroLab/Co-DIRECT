
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import overload

from dive.benchmark.evaluators.contracts import (
    EvaluatorAsset,
    EvaluatorAvailability,
    EvaluatorContract,
    EvaluatorRegistry,
    EvidenceContract,
    FamilyAvailability,
    MetricUnavailable,
    MetricValue,
    RawFieldContract,
    ReducerContract,
    WorkerContract,
)
from dive.benchmark.evaluators.records import FamilyMetricRecord, RawValue, Runner
from dive.benchmark.evaluators.worker_io import DisposableEvidencePolicy
from dive.benchmark.evaluators.worker_protocol import (
    WorkerFailureReason,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResult,
    _validate_request_against_evaluator,
    read_and_verify_result,
    write_request_create_new,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BINDER_FIELD_CONTRACTS = {
    "binder_interface_dSASA": ("float", "normalized", False),
    "binder_interface_fraction": ("float", "normalized", False),
    "binder_interface_hydrophobicity": ("float", "normalized", False),
    "binder_interface_nres": ("float", "normalized", False),
    "binder_interface_sc": ("float", "normalized", False),
    "binder_scRMSD_ca": ("float", "angstrom", True),
    "binder_surface_hydrophobicity": ("float", "normalized", False),
    "diversity_foldseek": ("float", "normalized", False),
    "has_clash": ("boolean", "boolean", False),
    "i_pAE": ("float", "normalized", True),
    "novelty_from_list": ("float", "normalized", False),
    "pLDDT": ("float", "normalized", True),
    "redesign_id": ("string", "sequence", True),
    "self_binder_scRMSD_ca": ("float", "angstrom", False),
    "self_complex_i_pAE": ("float", "normalized", False),
    "self_complex_pLDDT": ("float", "normalized", False),
    "self_esm_pseudo_perplexity": ("float", "normalized", False),
    "sequence": ("string", "sequence", True),
}
_BINDER_PRIMARY_FIELDS = frozenset(
    {"redesign_id", "sequence", "i_pAE", "pLDDT", "binder_scRMSD_ca"}
)
_BINDER_OPTIONAL_FIELDS = frozenset(
    set(_BINDER_FIELD_CONTRACTS) - _BINDER_PRIMARY_FIELDS
)
BINDER_OPTIONAL_METRIC_NAMES = tuple(sorted(_BINDER_OPTIONAL_FIELDS))
_UNCOMPUTED_OPTIONAL_REASON = "independent_optional_uncomputed"
_UNCOMPUTED_OPTIONAL_DETAIL = (
    "authenticated fixture does not compute independent optional scalars"
)

def _refuse(reason: WorkerFailureReason, detail: str) -> None:
    raise WorkerProtocolError(reason, detail)

def uncomputed_optional_records(
    names: object, manifest_hash: str
) -> dict[str, MetricUnavailable]:

    if type(manifest_hash) is not str or not _SHA256.fullmatch(manifest_hash):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid evaluator manifest hash"
        )
    if not isinstance(names, (tuple, list, frozenset)):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "optional metric names are unauthenticated",
        )
    records: dict[str, MetricUnavailable] = {}
    for name in names:
        if type(name) is not str or not name or name in records:
            _refuse(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "invalid optional metric name",
            )
        records[name] = MetricUnavailable(
            name,
            _UNCOMPUTED_OPTIONAL_REASON,
            _UNCOMPUTED_OPTIONAL_DETAIL,
            manifest_hash,
        )
    return records

def _optional_from_evaluator(
    evaluator: object, manifest_hash: str
) -> dict[str, MetricUnavailable]:
    metrics = getattr(evaluator, "optional_metrics", ())
    if not isinstance(metrics, tuple):
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "optional metrics are unauthenticated",
        )
    names: list[str] = []
    for metric in metrics:
        name = getattr(metric, "name", None)
        if type(name) is not str or not name:
            _refuse(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                "optional metrics are unauthenticated",
            )
        names.append(name)
    return uncomputed_optional_records(tuple(names), manifest_hash)

def _validated_manifest_hash(value: object) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid evaluator manifest hash"
        )
    return value

@dataclass(frozen=True, slots=True)
class BinderRedesign:

    redesign_id: str
    sequence: str
    i_pAE: float
    pLDDT: float
    binder_scRMSD_ca: float
    raw: Mapping[str, str | bool | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.redesign_id) is not str or not self.redesign_id:
            _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid redesign_id")
        if type(self.sequence) is not str or not self.sequence:
            _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid sequence")
        for name, value in (
            ("i_pAE", self.i_pAE),
            ("pLDDT", self.pLDDT),
            ("binder_scRMSD_ca", self.binder_scRMSD_ca),
        ):
            if type(value) is not float:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID, f"{name} must be a float"
                )
            if not math.isfinite(value):
                _refuse(WorkerFailureReason.NONFINITE_METRIC, f"{name} is non-finite")
        if not 0.0 <= self.pLDDT <= 1.0:
            _refuse(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "pLDDT must be normalized"
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
            if name in _BINDER_PRIMARY_FIELDS:
                _refuse(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID,
                    f"primary raw shadow: {name}",
                )
            if name not in _BINDER_OPTIONAL_FIELDS:
                _refuse(
                    WorkerFailureReason.UNKNOWN_OUTPUT_FIELD,
                    f"unknown raw field: {name}",
                )
            expected_type = {"string": str, "boolean": bool, "float": float}[
                _BINDER_FIELD_CONTRACTS[name][0]
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

def _contracts_by_name(
    field_contracts: object,
) -> Mapping[str, RawFieldContract]:
    if not isinstance(field_contracts, tuple) or not field_contracts:
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "raw field contract is invalid"
        )
    contracts: dict[str, RawFieldContract] = {}
    for contract in field_contracts:
        if not isinstance(contract, RawFieldContract):
            _refuse(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "raw field contract is invalid",
            )
        if contract.name in contracts:
            _refuse(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "raw field contract repeats a name",
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
    if set(contracts) != set(_BINDER_FIELD_CONTRACTS):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "raw field contract names drifted",
        )
    for name, (value_type, unit, required) in _BINDER_FIELD_CONTRACTS.items():
        contract = contracts.get(name)
        assert contract is not None
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

def parse_binder_rows(
    raw: Mapping[str, RawValue], field_contracts: tuple[RawFieldContract, ...]
) -> tuple[BinderRedesign, ...]:

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
    sequences = vectors["sequence"]
    if any(not item for item in sequences):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "sequences must be non-empty"
        )
    plddt = vectors["pLDDT"]
    if any(not 0.0 <= item <= 1.0 for item in plddt):
        _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, "pLDDT must be normalized")

    rows: list[BinderRedesign] = []
    primary_names = _BINDER_PRIMARY_FIELDS
    for index in range(vector_length):
        optional = {
            name: vector[index]
            for name, vector in vectors.items()
            if name not in primary_names
        }
        rows.append(
            BinderRedesign(
                redesign_ids[index],
                sequences[index],
                vectors["i_pAE"][index],
                plddt[index],
                vectors["binder_scRMSD_ca"][index],
                optional,
            )
        )
    return tuple(rows)

def _reduce_rows(
    rows: tuple[BinderRedesign, ...], manifest_hash: str
) -> FamilyMetricRecord:
    manifest_hash = _validated_manifest_hash(manifest_hash)
    if not isinstance(rows, tuple) or not rows:
        _refuse(
            WorkerFailureReason.VECTOR_LENGTH_MISMATCH,
            "binder result has no redesign rows",
        )
    if any(not isinstance(row, BinderRedesign) for row in rows):
        _refuse(WorkerFailureReason.RESULT_SCHEMA_INVALID, "binder rows are invalid")
    if len({row.redesign_id for row in rows}) != len(rows):
        _refuse(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "redesign identities must be unique",
        )
    optional_names = set(rows[0].raw)
    if any(set(row.raw) != optional_names for row in rows):
        _refuse(
            WorkerFailureReason.VECTOR_LENGTH_MISMATCH,
            "optional raw fields differ by row",
        )
    raw: dict[str, RawValue] = {
        "redesign_id": tuple(row.redesign_id for row in rows),
        "sequence": tuple(row.sequence for row in rows),
        "i_pAE": tuple(row.i_pAE for row in rows),
        "pLDDT": tuple(row.pLDDT for row in rows),
        "binder_scRMSD_ca": tuple(row.binder_scRMSD_ca for row in rows),
    }
    for name in optional_names:
        raw[name] = tuple(row.raw[name] for row in rows)
    success = any(
        row.i_pAE * 31 <= 7 and row.pLDDT >= 0.9 and row.binder_scRMSD_ca < 1.5
        for row in rows
    )
    return FamilyMetricRecord(
        family="binder",
        primary=MetricValue(
            "target_conditioned_success", float(success), "higher", manifest_hash
        ),
        raw=raw,
    )

@overload
def reduce_binder(
    rows: tuple[BinderRedesign, ...], manifest_hash: str
) -> FamilyMetricRecord: ...

@overload
def reduce_binder(
    result: WorkerResult, request: WorkerRequest, evaluator: object
) -> FamilyMetricRecord: ...

def reduce_binder(
    rows: tuple[BinderRedesign, ...] | WorkerResult,
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
        parse_binder_rows(rows.raw, field_contracts),
        manifest_hash.evaluator_semantic_sha256,
    )
    return FamilyMetricRecord(
        reduced.family,
        reduced.primary,
        reduced.raw,
        _optional_from_evaluator(evaluator, manifest_hash.evaluator_semantic_sha256),
    )

@dataclass(frozen=True, slots=True)
class BinderEvaluationRequest:

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

def evaluate_binder(
    request: WorkerRequest,
    registry: EvaluatorRegistry,
    availability: EvaluatorAvailability,
    *,
    runner: Runner | None = None,
    policy: DisposableEvidencePolicy | None = None,
) -> FamilyMetricRecord:

    return _evaluate_authenticated_family(
        family="binder",
        primary_name="target_conditioned_success",
        reducer=reduce_binder,
        request=request,
        registry=registry,
        availability=availability,
        runner=runner,
        policy=policy,
    )

def _unavailable(
    family: str, primary_name: str, reason_code: str, detail: str, manifest_hash: str
) -> FamilyMetricRecord:
    return FamilyMetricRecord(
        family=family,
        primary=MetricUnavailable(primary_name, reason_code, detail, manifest_hash),
    )

def _required_asset_path_missing(evaluator: EvaluatorContract) -> bool:
    for asset in evaluator.assets:
        if type(asset) is not EvaluatorAsset:
            _refuse(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                "evaluator assets are unauthenticated",
            )
        if not asset.required:
            continue
        if asset.identity is None or not Path(asset.identity.path).exists():
            return True
    return False

def _authenticate_adapter_inputs(
    family: str,
    request: object,
    registry: object,
    availability: object,
) -> tuple[WorkerRequest, EvaluatorRegistry, EvaluatorAvailability, EvaluatorContract]:
    if type(request) is not WorkerRequest:
        _refuse(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID,
            "adapter request must be a WorkerRequest",
        )
    if type(registry) is not EvaluatorRegistry:
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "registry must be an EvaluatorRegistry",
        )
    if type(availability) is not EvaluatorAvailability:
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "availability must be an EvaluatorAvailability",
        )
    if any(type(item) is not EvaluatorContract for item in registry.evaluators):
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "registry family records are unauthenticated",
        )
    if any(type(item) is not FamilyAvailability for item in availability.families):
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "availability family records are unauthenticated",
        )
    if tuple(item.family for item in availability.families) != registry.families:
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "availability families do not match the registry",
        )
    if request.family != family:
        _refuse(
            WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
            "request family differs from adapter",
        )
    evaluator = registry.for_family(family)
    if (
        type(evaluator) is not EvaluatorContract
        or evaluator.family != family
        or type(evaluator.worker) is not WorkerContract
        or type(evaluator.reducer) is not ReducerContract
        or type(evaluator.evidence) is not EvidenceContract
        or any(type(field) is not RawFieldContract for field in evaluator.raw_fields)
    ):
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "family registry record is unauthenticated",
        )
    return request, registry, availability, evaluator

def _evaluate_authenticated_family(
    *,
    family: str,
    primary_name: str,
    reducer: object,
    request: object,
    registry: object,
    availability: object,
    runner: Runner | None,
    policy: DisposableEvidencePolicy | None,
) -> FamilyMetricRecord:
    request, registry, availability, evaluator = _authenticate_adapter_inputs(
        family, request, registry, availability
    )
    if evaluator.primary.name != primary_name:
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            f"registry does not bind {primary_name} as the {family} primary",
        )
    family_availability = availability.for_family(family)
    if type(family_availability) is not FamilyAvailability:
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "family availability record is unauthenticated",
        )
    blocked = family_availability.primary
    if blocked is not None:
        if type(blocked) is not MetricUnavailable:
            _refuse(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                "availability blocker is unauthenticated",
            )
        return _unavailable(
            family,
            primary_name,
            WorkerFailureReason.REQUIRED_ASSET_UNAVAILABLE.value,
            blocked.detail,
            registry.semantic_sha256,
        )
    if _required_asset_path_missing(evaluator):
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "availability claimed ready while a required asset is missing",
        )
    if runner is None or type(policy) is not DisposableEvidencePolicy:
        _refuse(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "production worker execution is not authorized",
        )
    _validate_request_against_evaluator(request, registry, request.dive_commit)
    write_request_create_new(
        request,
        Path(request.request_path),
        registry=registry,
        expected_dive_commit=request.dive_commit,
        policy=policy,
    )
    result_path = runner(
        request,
        expected_dive_commit=request.dive_commit,
        policy=policy,
    )
    if not isinstance(result_path, Path):
        _refuse(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            "runner did not return an authenticated result path",
        )
    return read_and_verify_result(
        result_path,
        request,
        registry,
        reducer,
        expected_dive_commit=request.dive_commit,
        policy=policy,
    )
