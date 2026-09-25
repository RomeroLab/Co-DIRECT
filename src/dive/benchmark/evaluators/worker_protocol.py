
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.evaluators.contracts import MetricUnavailable, MetricValue
from dive.benchmark.evaluators.worker_errors import (
    WorkerFailureReason,
    WorkerProtocolError,
    protocol_error as _protocol_error,
)
from dive.benchmark.evaluators.worker_io import (
    DisposableEvidencePolicy,
    close_descriptors,
    hash_identity,
    open_read_fd,
    preflight_output_paths,
    read_descriptor_bytes,
    resolve_evidence_roots,
    write_create_new,
)
from dive.benchmark.evaluators.records import (
    FamilyMetricRecord,
    RawValue,
    validate_raw_value,
)
from dive.training.preflight import PreflightError, canonical_json_bytes

REQUEST_SCHEMA = "dive-evaluator-request-v1"
RESULT_SCHEMA = "dive-evaluator-result-v1"
EXPECTED_FAILURE_REASONS = (
    "registry_authentication_failed",
    "required_asset_unavailable",
    "request_schema_invalid",
    "request_identity_drift",
    "worker_timeout",
    "worker_nonzero_exit",
    "result_missing",
    "result_schema_invalid",
    "unknown_output_field",
    "nonfinite_metric",
    "missing_required_metric",
    "vector_length_mismatch",
    "unit_contract_error",
    "upstream_numeric_sentinel",
    "output_identity_drift",
    "evidence_path_violation",
)

class Reducer(Protocol):

    def __call__(
        self, result: WorkerResult, request: WorkerRequest, evaluator: object
    ) -> FamilyMetricRecord: ...

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "run_id",
        "family",
        "benchmark_contract_sha256",
        "registry_semantic_sha256",
        "evaluator_semantic_sha256",
        "dive_commit",
        "upstream_commit",
        "artifacts",
        "worker_source",
        "worker_interpreter",
        "worker_environment",
        "worker_callable",
        "worker_schema_version",
        "result_schema",
        "reducer_identifier",
        "reducer_semantic_sha256",
        "frozen_ids",
        "frozen_roles",
        "argv",
        "working_directory",
        "environment",
        "environment_allowlist",
        "request_path",
        "result_path",
        "stdout_path",
        "stderr_path",
        "bulk_directory",
        "timeout_seconds",
        "resource_limits",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "run_id",
        "family",
        "evaluator_semantic_sha256",
        "status",
        "return_code",
        "duration_seconds",
        "failure_reason",
        "failure_detail",
        "stdout",
        "stderr",
        "produced_artifacts",
        "raw",
        "primary",
        "optional",
    }
)

def _check_identifier(
    value: object,
    label: str,
    reason: WorkerFailureReason = WorkerFailureReason.REQUEST_SCHEMA_INVALID,
) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise _protocol_error(reason, f"invalid {label}")
    return value

def _check_hash(
    value: object,
    label: str,
    *,
    commit: bool = False,
    reason: WorkerFailureReason = WorkerFailureReason.REQUEST_SCHEMA_INVALID,
) -> str:
    pattern = _COMMIT if commit else _SHA256
    if type(value) is not str or not pattern.fullmatch(value):
        raise _protocol_error(reason, f"invalid {label}")
    return value

def _identity_record(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

def _identity_from_record(
    value: object, label: str, *, result: bool = False, absolute: bool = True
) -> ArtifactIdentity:
    reason = (
        WorkerFailureReason.RESULT_SCHEMA_INVALID
        if result
        else WorkerFailureReason.REQUEST_SCHEMA_INVALID
    )
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise _protocol_error(reason, f"invalid {label} identity")
    path, digest, size = value["path"], value["sha256"], value["size_bytes"]
    if (
        type(path) is not str
        or (absolute and not Path(path).is_absolute())
        or type(size) is not int
        or size < 0
    ):
        raise _protocol_error(reason, f"invalid {label} identity")
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise _protocol_error(reason, f"invalid {label} identity")
    return ArtifactIdentity(path, digest, size)

def _freeze_artifacts(
    value: Mapping[str, ArtifactIdentity], label: str
) -> Mapping[str, ArtifactIdentity]:
    if not isinstance(value, Mapping):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID, f"{label} must be a mapping"
        )
    checked: dict[str, ArtifactIdentity] = {}
    for name, identity in value.items():
        _check_identifier(name, f"{label} key")
        if not isinstance(identity, ArtifactIdentity):
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID, f"invalid {label} identity"
            )
        _identity_from_record(_identity_record(identity), label)
        checked[name] = identity
    return MappingProxyType(checked)

def _freeze_string_vectors(
    value: Mapping[str, tuple[str, ...]], label: str
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID, f"{label} must be a mapping"
        )
    checked: dict[str, tuple[str, ...]] = {}
    for name, records in value.items():
        _check_identifier(name, f"{label} key")
        if (
            not isinstance(records, tuple)
            or not records
            or any(type(item) is not str or not item for item in records)
        ):
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID, f"invalid {label} value"
            )
        checked[name] = records
    return MappingProxyType(checked)

def _freeze_roles(
    value: Mapping[str, str | tuple[str, ...]],
) -> Mapping[str, str | tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID, "frozen_roles must be a mapping"
        )
    checked: dict[str, str | tuple[str, ...]] = {}
    for name, role in value.items():
        _check_identifier(name, "frozen_roles key")
        if type(role) is str and role:
            checked[name] = role
        elif (
            isinstance(role, tuple)
            and role
            and all(type(item) is str and item for item in role)
        ):
            checked[name] = role
        else:
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID, "invalid frozen role"
            )
    return MappingProxyType(checked)

def _freeze_environment(
    value: Mapping[str, str], allowlist: tuple[str, ...]
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(
        sorted(allowlist)
    ):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID,
            "environment keys differ from allowlist",
        )
    if len(set(allowlist)) != len(allowlist) or any(
        type(name) is not str or not name for name in allowlist
    ):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID, "invalid environment allowlist"
        )
    if any(
        type(name) is not str or type(item) is not str for name, item in value.items()
    ):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID,
            "environment entries must be strings",
        )
    return MappingProxyType(dict(value))

def _freeze_limits(value: Mapping[str, int | float]) -> Mapping[str, int | float]:
    if not isinstance(value, Mapping) or not value:
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID,
            "resource_limits must be non-empty",
        )
    checked: dict[str, int | float] = {}
    for name, limit in value.items():
        _check_identifier(name, "resource limit")
        if (
            type(limit) not in {int, float}
            or isinstance(limit, bool)
            or not math.isfinite(float(limit))
            or limit <= 0
        ):
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID, "invalid resource limit"
            )
        checked[name] = limit
    return MappingProxyType(checked)

@dataclass(frozen=True, slots=True)
class WorkerRequest:
    schema_version: str
    request_id: str
    run_id: str
    family: str
    benchmark_contract_sha256: str
    registry_semantic_sha256: str
    evaluator_semantic_sha256: str
    dive_commit: str
    upstream_commit: str
    artifacts: Mapping[str, ArtifactIdentity]
    worker_source: ArtifactIdentity
    worker_interpreter: ArtifactIdentity
    worker_environment: ArtifactIdentity
    worker_callable: str
    worker_schema_version: str
    result_schema: str
    reducer_identifier: str
    reducer_semantic_sha256: str
    frozen_ids: Mapping[str, tuple[str, ...]]
    frozen_roles: Mapping[str, str | tuple[str, ...]]
    argv: tuple[str, ...]
    working_directory: str
    environment: Mapping[str, str]
    environment_allowlist: tuple[str, ...]
    request_path: str
    result_path: str
    stdout_path: str
    stderr_path: str
    bulk_directory: str
    timeout_seconds: float
    resource_limits: Mapping[str, int | float]

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA or self.family not in {
            "binder",
            "ame",
        }:
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID,
                "unsupported request schema or family",
            )
        for label in ("request_id", "run_id"):
            _check_identifier(getattr(self, label), label)
        for label in (
            "benchmark_contract_sha256",
            "registry_semantic_sha256",
            "evaluator_semantic_sha256",
            "reducer_semantic_sha256",
        ):
            _check_hash(getattr(self, label), label)
        for label in ("dive_commit", "upstream_commit"):
            _check_hash(getattr(self, label), label, commit=True)
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or any(type(item) is not str or not item for item in self.argv)
        ):
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID, "invalid argv"
            )
        if (
            type(self.working_directory) is not str
            or not Path(self.working_directory).is_absolute()
        ):
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID,
                "working_directory must be absolute",
            )
        for label in (
            "request_path",
            "result_path",
            "stdout_path",
            "stderr_path",
            "bulk_directory",
        ):
            value = getattr(self, label)
            if type(value) is not str or not Path(value).is_absolute():
                raise _protocol_error(
                    WorkerFailureReason.REQUEST_SCHEMA_INVALID,
                    f"{label} must be absolute",
                )
        if len({self.result_path, self.stdout_path, self.stderr_path}) != 3:
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID,
                "declared terminal outputs must be distinct",
            )
        if (
            type(self.timeout_seconds) is not float
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID, "invalid timeout_seconds"
            )
        if not isinstance(self.environment_allowlist, tuple):
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID,
                "environment_allowlist must be a tuple",
            )
        object.__setattr__(
            self, "artifacts", _freeze_artifacts(self.artifacts, "artifacts")
        )
        for label in ("worker_source", "worker_interpreter", "worker_environment"):
            identity = getattr(self, label)
            if not isinstance(identity, ArtifactIdentity):
                raise _protocol_error(
                    WorkerFailureReason.REQUEST_SCHEMA_INVALID, f"invalid {label}"
                )
            _identity_from_record(
                _identity_record(identity), label, absolute=label != "worker_source"
            )
        for label in (
            "worker_callable",
            "worker_schema_version",
            "result_schema",
            "reducer_identifier",
        ):
            if type(getattr(self, label)) is not str or not getattr(self, label):
                raise _protocol_error(
                    WorkerFailureReason.REQUEST_SCHEMA_INVALID, f"invalid {label}"
                )
        object.__setattr__(
            self, "frozen_ids", _freeze_string_vectors(self.frozen_ids, "frozen_ids")
        )
        object.__setattr__(self, "frozen_roles", _freeze_roles(self.frozen_roles))
        object.__setattr__(
            self,
            "environment",
            _freeze_environment(self.environment, self.environment_allowlist),
        )
        object.__setattr__(
            self, "resource_limits", _freeze_limits(self.resource_limits)
        )

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "family": self.family,
            "benchmark_contract_sha256": self.benchmark_contract_sha256,
            "registry_semantic_sha256": self.registry_semantic_sha256,
            "evaluator_semantic_sha256": self.evaluator_semantic_sha256,
            "dive_commit": self.dive_commit,
            "upstream_commit": self.upstream_commit,
            "artifacts": {
                name: _identity_record(item) for name, item in self.artifacts.items()
            },
            "worker_source": _identity_record(self.worker_source),
            "worker_interpreter": _identity_record(self.worker_interpreter),
            "worker_environment": _identity_record(self.worker_environment),
            "worker_callable": self.worker_callable,
            "worker_schema_version": self.worker_schema_version,
            "result_schema": self.result_schema,
            "reducer_identifier": self.reducer_identifier,
            "reducer_semantic_sha256": self.reducer_semantic_sha256,
            "frozen_ids": {
                name: list(items) for name, items in self.frozen_ids.items()
            },
            "frozen_roles": {
                name: list(item) if isinstance(item, tuple) else item
                for name, item in self.frozen_roles.items()
            },
            "argv": list(self.argv),
            "working_directory": self.working_directory,
            "environment": dict(self.environment),
            "environment_allowlist": list(self.environment_allowlist),
            "request_path": self.request_path,
            "result_path": self.result_path,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "bulk_directory": self.bulk_directory,
            "timeout_seconds": self.timeout_seconds,
            "resource_limits": dict(self.resource_limits),
        }

    @classmethod
    def from_record(cls, record: object) -> WorkerRequest:

        if not isinstance(record, Mapping) or set(record) != _REQUEST_KEYS:
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID,
                "request keys are incomplete or unknown",
            )
        try:
            artifacts = record["artifacts"]
            frozen_ids = record["frozen_ids"]
            frozen_roles = record["frozen_roles"]
            if not isinstance(artifacts, Mapping):
                raise TypeError("artifacts must be a mapping")
            if not isinstance(frozen_ids, Mapping) or not isinstance(
                frozen_roles, Mapping
            ):
                raise TypeError("frozen request values must be mappings")
            return cls(
                record["schema_version"],
                record["request_id"],
                record["run_id"],
                record["family"],
                record["benchmark_contract_sha256"],
                record["registry_semantic_sha256"],
                record["evaluator_semantic_sha256"],
                record["dive_commit"],
                record["upstream_commit"],
                {
                    name: _identity_from_record(identity, f"artifacts.{name}")
                    for name, identity in artifacts.items()
                },
                _identity_from_record(
                    record["worker_source"], "worker_source", absolute=False
                ),
                _identity_from_record(
                    record["worker_interpreter"], "worker_interpreter"
                ),
                _identity_from_record(
                    record["worker_environment"], "worker_environment"
                ),
                record["worker_callable"],
                record["worker_schema_version"],
                record["result_schema"],
                record["reducer_identifier"],
                record["reducer_semantic_sha256"],
                {
                    name: tuple(items) if isinstance(items, list) else items
                    for name, items in frozen_ids.items()
                },
                {
                    name: tuple(items) if isinstance(items, list) else items
                    for name, items in frozen_roles.items()
                },
                tuple(record["argv"])
                if isinstance(record["argv"], list)
                else record["argv"],
                record["working_directory"],
                record["environment"],
                tuple(record["environment_allowlist"])
                if isinstance(record["environment_allowlist"], list)
                else record["environment_allowlist"],
                record["request_path"],
                record["result_path"],
                record["stdout_path"],
                record["stderr_path"],
                record["bulk_directory"],
                record["timeout_seconds"],
                record["resource_limits"],
            )
        except WorkerProtocolError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise _protocol_error(
                WorkerFailureReason.REQUEST_SCHEMA_INVALID, str(error)
            ) from error

def _metric_record(value: MetricValue | MetricUnavailable) -> dict[str, object]:
    if isinstance(value, MetricValue):
        return {
            "kind": "value",
            "name": value.name,
            "value": value.value,
            "direction": value.direction,
            "evaluator_manifest_hash": value.evaluator_manifest_hash,
        }
    return {
        "kind": "unavailable",
        "name": value.name,
        "reason_code": value.reason_code,
        "detail": value.detail,
        "evaluator_manifest_hash": value.evaluator_manifest_hash,
        "status": value.status.value,
    }

def _metric_from_record(value: object) -> MetricValue | MetricUnavailable:
    if not isinstance(value, Mapping) or type(value.get("kind")) is not str:
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid primary record"
        )
    kind = value["kind"]
    if kind == "value":
        if set(value) != {
            "kind",
            "name",
            "value",
            "direction",
            "evaluator_manifest_hash",
        }:
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid primary record"
            )
        if (
            type(value["name"]) is not str
            or type(value["value"]) is not float
            or not math.isfinite(value["value"])
            or value["direction"] not in {"higher", "lower"}
            or not _SHA256.fullmatch(str(value["evaluator_manifest_hash"]))
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid primary record"
            )
        return MetricValue(
            value["name"],
            value["value"],
            value["direction"],
            value["evaluator_manifest_hash"],
        )
    if kind == "unavailable":
        if (
            set(value)
            != {
                "kind",
                "name",
                "reason_code",
                "detail",
                "evaluator_manifest_hash",
                "status",
            }
            or value["status"] != "unavailable"
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid unavailable primary"
            )
        if (
            type(value["name"]) is not str
            or type(value["reason_code"]) is not str
            or type(value["detail"]) is not str
            or not _SHA256.fullmatch(str(value["evaluator_manifest_hash"]))
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid unavailable primary"
            )
        return MetricUnavailable(
            value["name"],
            value["reason_code"],
            value["detail"],
            value["evaluator_manifest_hash"],
        )
    raise _protocol_error(
        WorkerFailureReason.RESULT_SCHEMA_INVALID, "unknown primary record kind"
    )

def _mapping_result(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, f"{label} must be a mapping"
        )
    return value

@dataclass(frozen=True, slots=True)
class WorkerResult:
    schema_version: str
    request_id: str
    run_id: str
    family: str
    evaluator_semantic_sha256: str
    status: str
    return_code: int
    duration_seconds: float
    failure_reason: WorkerFailureReason | None
    failure_detail: str | None
    stdout: ArtifactIdentity
    stderr: ArtifactIdentity
    produced_artifacts: Mapping[str, ArtifactIdentity]
    raw: Mapping[str, RawValue]
    primary: MetricValue | MetricUnavailable
    optional: Mapping[str, MetricValue | MetricUnavailable] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA or self.family not in {"binder", "ame"}:
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "unsupported result schema or family",
            )
        for label in ("request_id", "run_id"):
            _check_identifier(
                getattr(self, label), label, WorkerFailureReason.RESULT_SCHEMA_INVALID
            )
        _check_hash(
            self.evaluator_semantic_sha256,
            "evaluator_semantic_sha256",
            reason=WorkerFailureReason.RESULT_SCHEMA_INVALID,
        )
        if (
            self.status not in {"succeeded", "failed"}
            or type(self.return_code) is not int
            or isinstance(self.return_code, bool)
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid result status"
            )
        if (self.status == "succeeded" and self.return_code != 0) or (
            self.status == "failed" and self.return_code == 0
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "return code conflicts with result status",
            )
        if (
            type(self.duration_seconds) is not float
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise _protocol_error(
                WorkerFailureReason.NONFINITE_METRIC, "invalid result duration"
            )
        if (self.status == "succeeded") != (
            self.failure_reason is None and self.failure_detail is None
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "inconsistent result failure fields",
            )
        if self.status == "failed" and (
            not isinstance(self.failure_reason, WorkerFailureReason)
            or type(self.failure_detail) is not str
            or not self.failure_detail
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid failure result"
            )
        for label in ("stdout", "stderr"):
            value = getattr(self, label)
            if not isinstance(value, ArtifactIdentity):
                raise _protocol_error(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID,
                    f"invalid {label} identity",
                )
            _identity_from_record(_identity_record(value), label, result=True)
        if not isinstance(self.raw, Mapping):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "raw must be a mapping"
            )
        checked_raw: dict[str, RawValue] = {}
        for name, value in self.raw.items():
            if type(name) is not str or not name:
                raise _protocol_error(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid raw name"
                )
            try:
                checked_raw[name] = validate_raw_value(value)
            except ValueError as error:
                raise _protocol_error(
                    WorkerFailureReason.NONFINITE_METRIC, str(error)
                ) from error
            except TypeError as error:
                raise _protocol_error(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID, str(error)
                ) from error
        if not isinstance(self.primary, (MetricValue, MetricUnavailable)):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid primary"
            )
        if (
            self.status == "succeeded" and not isinstance(self.primary, MetricValue)
        ) or (
            self.status == "failed" and not isinstance(self.primary, MetricUnavailable)
        ):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "primary conflicts with result status",
            )
        if not isinstance(self.optional, Mapping):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "optional must be a mapping"
            )
        checked_optional: dict[str, MetricValue | MetricUnavailable] = {}
        for name, metric in self.optional.items():
            if (
                type(name) is not str
                or not name
                or not isinstance(metric, (MetricValue, MetricUnavailable))
            ):
                raise _protocol_error(
                    WorkerFailureReason.RESULT_SCHEMA_INVALID, "invalid optional metric"
                )
            checked_optional[name] = metric
        object.__setattr__(
            self,
            "produced_artifacts",
            _freeze_artifacts(self.produced_artifacts, "produced_artifacts"),
        )
        object.__setattr__(self, "raw", MappingProxyType(checked_raw))
        object.__setattr__(self, "optional", MappingProxyType(checked_optional))

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "family": self.family,
            "evaluator_semantic_sha256": self.evaluator_semantic_sha256,
            "status": self.status,
            "return_code": self.return_code,
            "duration_seconds": self.duration_seconds,
            "failure_reason": self.failure_reason.value
            if self.failure_reason is not None
            else None,
            "failure_detail": self.failure_detail,
            "stdout": _identity_record(self.stdout),
            "stderr": _identity_record(self.stderr),
            "produced_artifacts": {
                name: _identity_record(item)
                for name, item in self.produced_artifacts.items()
            },
            "raw": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in self.raw.items()
            },
            "primary": _metric_record(self.primary),
            "optional": {
                name: _metric_record(metric) for name, metric in self.optional.items()
            },
        }

    @classmethod
    def from_record(cls, record: object) -> WorkerResult:
        if not isinstance(record, Mapping):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "result must be a mapping"
            )
        unknown = set(record) - _RESULT_KEYS
        if unknown:
            raise _protocol_error(
                WorkerFailureReason.UNKNOWN_OUTPUT_FIELD,
                f"unknown result keys: {sorted(unknown)}",
            )
        if set(record) != _RESULT_KEYS:
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, "result keys are incomplete"
            )
        try:
            raw = record["raw"]
            if not isinstance(raw, Mapping):
                raise TypeError("raw must be a mapping")
            converted_raw = {
                name: tuple(value) if isinstance(value, list) else value
                for name, value in raw.items()
            }
            artifacts = record["produced_artifacts"]
            if not isinstance(artifacts, Mapping):
                raise TypeError("produced_artifacts must be a mapping")
            failure = record["failure_reason"]
            if failure is not None:
                failure = WorkerFailureReason(failure)
            return cls(
                record["schema_version"],
                record["request_id"],
                record["run_id"],
                record["family"],
                record["evaluator_semantic_sha256"],
                record["status"],
                record["return_code"],
                record["duration_seconds"],
                failure,
                record["failure_detail"],
                _identity_from_record(record["stdout"], "stdout", result=True),
                _identity_from_record(record["stderr"], "stderr", result=True),
                {
                    name: _identity_from_record(
                        item, f"produced_artifacts.{name}", result=True
                    )
                    for name, item in artifacts.items()
                },
                converted_raw,
                _metric_from_record(record["primary"]),
                {
                    name: _metric_from_record(metric)
                    for name, metric in _mapping_result(
                        record["optional"], "optional"
                    ).items()
                },
            )
        except WorkerProtocolError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID, str(error)
            ) from error

def write_request_create_new(
    request: WorkerRequest,
    path: Path,
    *,
    registry: object | None = None,
    expected_dive_commit: str | None = None,
    policy: DisposableEvidencePolicy | None = None,
) -> Path:
    evaluator = None
    if policy is None:
        if registry is None or expected_dive_commit is None:
            raise _protocol_error(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                "production writes require registry and expected commit",
            )
        evaluator = _validate_request_against_evaluator(
            request, registry, expected_dive_commit
        )
    preflight_output_paths(
        request, evaluator=evaluator, policy=policy, request_path=Path(path)
    )
    terminal_root, _, expected_gid, group_policy = resolve_evidence_roots(
        evaluator, policy
    )
    return write_create_new(
        Path(path),
        terminal_root,
        canonical_json_bytes(request.as_record()),
        expected_gid=expected_gid,
        group_policy=group_policy,
        policy=policy,
    )

def write_result_create_new(
    result: WorkerResult,
    path: Path,
    *,
    request: WorkerRequest,
    registry: object | None = None,
    expected_dive_commit: str | None = None,
    policy: DisposableEvidencePolicy | None = None,
) -> Path:
    if Path(path) != Path(request.result_path):
        raise _protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "result path differs from request",
        )
    evaluator = None
    if policy is None:
        if registry is None or expected_dive_commit is None:
            raise _protocol_error(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                "production writes require registry and expected commit",
            )
        evaluator = _validate_request_against_evaluator(
            request, registry, expected_dive_commit
        )
    preflight_output_paths(request, evaluator=evaluator, policy=policy)
    _validate_result_identity(result, request)
    terminal_root, _, expected_gid, group_policy = resolve_evidence_roots(
        evaluator, policy
    )
    return write_create_new(
        Path(path),
        terminal_root,
        canonical_json_bytes(result.as_record()),
        expected_gid=expected_gid,
        group_policy=group_policy,
        policy=policy,
    )

def _validate_result_identity(result: WorkerResult, request: WorkerRequest) -> None:
    if (
        result.request_id,
        result.run_id,
        result.family,
        result.evaluator_semantic_sha256,
    ) != (
        request.request_id,
        request.run_id,
        request.family,
        request.evaluator_semantic_sha256,
    ):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
            "result does not bind the request",
        )
    if Path(result.stdout.path) != Path(request.stdout_path) or Path(
        result.stderr.path
    ) != Path(request.stderr_path):
        raise _protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            "stdout or stderr path differs from request",
        )
    bulk = Path(request.bulk_directory)
    for identity in result.produced_artifacts.values():
        try:
            Path(identity.path).relative_to(bulk)
        except ValueError as error:
            raise _protocol_error(
                WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
                "artifact escapes request bulk directory",
            ) from error

def _registered_evaluator(request: WorkerRequest, registry: object) -> object:
    resolver = getattr(registry, "for_family", None)
    if not callable(resolver):
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "registry has no family authority",
        )
    try:
        evaluator = resolver(request.family)
    except Exception as error:
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "family is not registered",
        ) from error
    if getattr(evaluator, "family", None) != request.family:
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "registry family record drifted",
        )
    return evaluator

def _validate_request_against_evaluator(
    request: WorkerRequest, registry: object, expected_dive_commit: str
) -> object:
    _check_hash(
        expected_dive_commit,
        "expected_dive_commit",
        commit=True,
        reason=WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
    )
    evaluator = _registered_evaluator(request, registry)
    worker = getattr(evaluator, "worker", None)
    if worker is None or getattr(evaluator, "family", None) != request.family:
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "worker family is unauthenticated",
        )
    if (
        request.benchmark_contract_sha256
        != getattr(registry, "benchmark_contract_hash", None)
        or request.registry_semantic_sha256
        != getattr(registry, "semantic_sha256", None)
        or request.dive_commit != expected_dive_commit
        or request.evaluator_semantic_sha256
        != getattr(evaluator, "semantic_sha256", None)
        or request.upstream_commit != getattr(evaluator, "source_commit", None)
        or request.schema_version != getattr(worker, "request_schema", None)
        or request.result_schema != getattr(worker, "result_schema", None)
        or request.worker_schema_version != getattr(worker, "schema_version", None)
        or request.worker_source != getattr(worker, "source", None)
        or request.worker_interpreter
        != getattr(getattr(worker, "interpreter", None), "identity", None)
        or request.worker_environment
        != getattr(getattr(worker, "interpreter", None), "environment", None)
        or request.worker_callable != getattr(worker, "callable_name", None)
        or request.reducer_identifier
        != getattr(getattr(evaluator, "reducer", None), "identifier", None)
        or request.reducer_semantic_sha256
        != getattr(getattr(evaluator, "reducer", None), "semantic_sha256", None)
        or tuple(request.environment_allowlist)
        != tuple(getattr(worker, "environment_allowlist", ()))
        or dict(request.environment) != dict(getattr(worker, "environment", ()))
        or request.working_directory
        != getattr(
            worker,
            "resolved_working_directory",
            getattr(worker, "working_directory", None),
        )
    ):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
            "request differs from registered worker contract",
        )
    substitutions = {
        "{interpreter}": getattr(getattr(worker, "interpreter", None), "path", None),
        "{worker}": getattr(getattr(worker, "source", None), "path", None),
        "{request_path}": request.request_path,
    }
    expected_argv = tuple(
        substitutions.get(item, item) for item in getattr(worker, "argv_template", ())
    )
    if request.argv != expected_argv:
        raise _protocol_error(
            WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
            "argv differs from registered worker template",
        )
    return evaluator

def _validate_raw_against_evaluator(result: WorkerResult, evaluator: object) -> None:
    fields = getattr(evaluator, "raw_fields", ())
    if not isinstance(fields, tuple):
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "raw field contract is unauthenticated",
        )
    declared = {getattr(field, "name", None): field for field in fields}
    if None in declared or len(declared) != len(fields):
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "raw field contract is invalid",
        )
    unknown = set(result.raw) - set(declared)
    if unknown:
        raise _protocol_error(
            WorkerFailureReason.UNKNOWN_OUTPUT_FIELD,
            f"unknown raw fields: {sorted(unknown)}",
        )
    missing = [
        name
        for name, field in declared.items()
        if getattr(field, "required", False) and name not in result.raw
    ]
    if missing:
        raise _protocol_error(
            WorkerFailureReason.MISSING_REQUIRED_METRIC,
            f"missing raw fields: {sorted(missing)}",
        )
    expected_types = {"string": str, "boolean": bool, "float": float}
    for name, value in result.raw.items():
        expected = expected_types.get(getattr(declared[name], "value_type", None))
        values = value if isinstance(value, tuple) else (value,)
        if expected is None or any(type(item) is not expected for item in values):
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                f"raw field has wrong type: {name}",
            )

def _validate_metric_against_contract(
    metric: MetricValue | MetricUnavailable, contract: object, manifest_hash: str
) -> None:
    if (
        metric.name != getattr(contract, "name", None)
        or metric.evaluator_manifest_hash != manifest_hash
    ):
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "metric identity differs from evaluator",
        )
    if isinstance(metric, MetricValue) and metric.direction != getattr(
        contract, "direction", None
    ):
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "metric direction differs from evaluator",
        )

def _validate_metrics_against_evaluator(
    result: WorkerResult, request: WorkerRequest, evaluator: object
) -> None:
    _validate_metric_against_contract(
        result.primary,
        getattr(evaluator, "primary", None),
        request.evaluator_semantic_sha256,
    )
    contracts = {
        getattr(metric, "name", None): metric
        for metric in getattr(evaluator, "optional_metrics", ())
    }
    if None in contracts or set(result.optional) != set(contracts):
        raise _protocol_error(
            WorkerFailureReason.UNKNOWN_OUTPUT_FIELD,
            "optional metric set differs from evaluator",
        )
    for name, metric in result.optional.items():
        _validate_metric_against_contract(
            metric, contracts[name], request.evaluator_semantic_sha256
        )

def read_and_verify_result(
    path: Path,
    request: WorkerRequest,
    registry: object,
    reducer: Reducer
    | Callable[[WorkerResult, WorkerRequest, object], FamilyMetricRecord],
    *,
    expected_dive_commit: str,
    policy: DisposableEvidencePolicy | None = None,
) -> FamilyMetricRecord:

    evaluator = _validate_request_against_evaluator(
        request, registry, expected_dive_commit
    )
    terminal_root, bulk_root, expected_gid, group_policy = resolve_evidence_roots(
        evaluator, policy
    )
    if Path(path) != Path(request.result_path):
        raise _protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            "result path differs from request",
        )
    try:
        descriptor = open_read_fd(
            Path(path),
            terminal_root,
            expected_gid=expected_gid,
            group_policy=group_policy,
            policy=policy,
        )
    except WorkerProtocolError:
        raise
    try:
        payload = read_descriptor_bytes(descriptor, rewind=False)
    except OSError as error:
        raise _protocol_error(
            WorkerFailureReason.RESULT_MISSING, "result.json cannot be read"
        ) from error
    finally:
        close_descriptors([descriptor])
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except ValueError as error:
        reason = (
            WorkerFailureReason.NONFINITE_METRIC
            if str(error) in {"NaN", "Infinity", "-Infinity"}
            else WorkerFailureReason.RESULT_SCHEMA_INVALID
        )
        raise _protocol_error(reason, "result is not valid finite JSON") from error
    except UnicodeDecodeError as error:
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "result is not UTF-8"
        ) from error
    try:
        if canonical_json_bytes(decoded) != payload:
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "result bytes are not canonical",
            )
    except PreflightError as error:
        raise _protocol_error(
            WorkerFailureReason.NONFINITE_METRIC, "result is not finite JSON"
        ) from error
    result = WorkerResult.from_record(decoded)
    if result.schema_version != getattr(
        getattr(evaluator, "worker", None), "result_schema", None
    ):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
            "result schema differs from registered worker",
        )
    _validate_result_identity(result, request)
    _validate_raw_against_evaluator(result, evaluator)
    _validate_metrics_against_evaluator(result, request, evaluator)
    hash_identity(
        result.stdout,
        terminal_root,
        expected_gid=expected_gid,
        group_policy=group_policy,
        policy=policy,
    )
    hash_identity(
        result.stderr,
        terminal_root,
        expected_gid=expected_gid,
        group_policy=group_policy,
        policy=policy,
    )
    for identity in result.produced_artifacts.values():
        hash_identity(
            identity,
            bulk_root,
            expected_gid=expected_gid,
            group_policy=group_policy,
            policy=policy,
        )
    if result.status == "failed":
        assert result.failure_reason is not None
        assert result.failure_detail is not None
        raise _protocol_error(result.failure_reason, result.failure_detail)
    try:
        recomputed = reducer(result, request, evaluator)
    except WorkerProtocolError:
        raise
    except Exception as error:
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            f"reducer refused result: {error}",
        ) from error
    if (
        not isinstance(recomputed, FamilyMetricRecord)
        or recomputed.family != request.family
    ):
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID,
            "reducer returned an invalid family record",
        )
    if _metric_record(recomputed.primary) != _metric_record(result.primary) or {
        name: _metric_record(metric) for name, metric in recomputed.optional.items()
    } != {name: _metric_record(metric) for name, metric in result.optional.items()}:
        raise _protocol_error(
            WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            "worker primary differs from reducer recomputation",
        )
    return recomputed

def run_fixture_worker(
    request: WorkerRequest,
    *,
    expected_dive_commit: str | None = None,
    policy: DisposableEvidencePolicy | None = None,
) -> Path:

    if policy is None or expected_dive_commit is None:
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "fixture worker execution requires a commit and disposable policy",
        )
    return run_worker(
        request,
        expected_dive_commit=expected_dive_commit,
        policy=policy,
    )

def run_worker(
    request: WorkerRequest,
    *,
    expected_dive_commit: str,
    policy: DisposableEvidencePolicy | None = None,
) -> Path:

    from dive.benchmark.evaluators.worker_runner import run_worker as _run_worker

    return _run_worker(
        request,
        expected_dive_commit=expected_dive_commit,
        policy=policy,
    )
