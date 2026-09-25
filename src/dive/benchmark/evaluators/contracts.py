
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from dive.benchmark.contracts import ArtifactIdentity, MetricStatus, file_identity
from dive.signed_value.roots import (
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_ROOT,
)
from dive.training.preflight import canonical_json_bytes

class EvaluatorContractError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    value: float
    direction: str
    evaluator_manifest_hash: str

@dataclass(frozen=True, slots=True)
class MetricUnavailable:
    name: str
    reason_code: str
    detail: str
    evaluator_manifest_hash: str
    status: MetricStatus = MetricStatus.UNAVAILABLE

@dataclass(frozen=True, slots=True)
class MetricContract:
    name: str
    direction: str
    required: bool

@dataclass(frozen=True, slots=True)
class ModuleTreeIdentity:
    root: str
    sha256: str
    size_bytes: int
    file_count: int

@dataclass(frozen=True, slots=True)
class RuntimeDependency:
    kind: str
    module: str
    distribution: str | None
    version: str | None
    tree: ModuleTreeIdentity | None
    required: bool
    metrics: tuple[str, ...]
    reason_code: str | None = None
    provenance: str | None = None

@dataclass(frozen=True, slots=True)
class InterpreterContract:
    declared_path: str
    identity: ArtifactIdentity
    version: str
    environment: ArtifactIdentity

    @property
    def path(self) -> str:
        return self.declared_path

@dataclass(frozen=True, slots=True)
class RawFieldContract:
    name: str
    value_type: str
    unit: str
    required: bool

@dataclass(frozen=True, slots=True)
class ReducerContract:
    identifier: str
    semantic_sha256: str

@dataclass(frozen=True, slots=True)
class WorkerContract:
    schema_version: str
    source: ArtifactIdentity
    interpreter: InterpreterContract
    callable_name: str
    argv_template: tuple[str, ...]
    working_directory: str
    resolved_working_directory: str
    environment_allowlist: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    request_schema: str
    result_schema: str
    fixture_source: ArtifactIdentity
    fixture_result_sha256: str

@dataclass(frozen=True, slots=True)
class EvidenceContract:
    terminal_root: str
    bulk_root: str
    create_new: bool

@dataclass(frozen=True, slots=True)
class ExecutableRuntime:
    shebang: str
    entry_module: str
    interpreter: InterpreterContract
    package: RuntimeDependency
    dependencies: tuple[RuntimeDependency, ...]

@dataclass(frozen=True, slots=True)
class EvaluatorAsset:
    role: str
    kind: str
    identity: ArtifactIdentity | None
    required: bool
    checkpoint: str | None
    metric: str | None = None
    reason_code: str | None = None
    provenance: str | None = None
    runtime: ExecutableRuntime | None = None
    tool: str | None = None
    version: str | None = None

@dataclass(frozen=True, slots=True)
class EvaluatorSourceFile:
    metric: str
    entry_point: bool
    identity: ArtifactIdentity

@dataclass(frozen=True, slots=True)
class EvaluatorContract:
    family: str
    name: str
    semantic_version: str
    source_repository: str
    source_commit: str
    interpreter: InterpreterContract
    input_schema: str
    output_schema: str
    primary: MetricContract
    required_metrics: tuple[MetricContract, ...]
    optional_metrics: tuple[MetricContract, ...]
    assets: tuple[EvaluatorAsset, ...]
    source_files: tuple[EvaluatorSourceFile, ...]
    runtime_dependencies: tuple[RuntimeDependency, ...]
    provenance_status: str
    worker: WorkerContract | None = None
    raw_fields: tuple[RawFieldContract, ...] = ()
    reducer: ReducerContract | None = None
    evidence: EvidenceContract | None = None
    failure_reasons: tuple[str, ...] = ()

    @property
    def semantic_sha256(self) -> str:

        worker = self.worker
        reducer = self.reducer
        if worker is None or reducer is None:
            return hashlib.sha256(
                canonical_json_bytes({"family": self.family})
            ).hexdigest()
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "family": self.family,
                    "name": self.name,
                    "semantic_version": self.semantic_version,
                    "source_commit": self.source_commit,
                    "input_schema": self.input_schema,
                    "output_schema": self.output_schema,
                    "primary": {
                        "name": self.primary.name,
                        "direction": self.primary.direction,
                    },
                    "worker": {
                        "schema_version": worker.schema_version,
                        "source": {
                            "path": worker.source.path,
                            "sha256": worker.source.sha256,
                            "size_bytes": worker.source.size_bytes,
                        },
                        "interpreter": {
                            "path": worker.interpreter.identity.path,
                            "sha256": worker.interpreter.identity.sha256,
                            "size_bytes": worker.interpreter.identity.size_bytes,
                        },
                        "environment_identity": {
                            "path": worker.interpreter.environment.path,
                            "sha256": worker.interpreter.environment.sha256,
                            "size_bytes": worker.interpreter.environment.size_bytes,
                        },
                        "callable_name": worker.callable_name,
                        "argv_template": list(worker.argv_template),
                        "working_directory": worker.working_directory,
                        "environment": dict(worker.environment),
                        "request_schema": worker.request_schema,
                        "result_schema": worker.result_schema,
                    },
                    "reducer": {
                        "identifier": reducer.identifier,
                        "semantic_sha256": reducer.semantic_sha256,
                    },
                }
            )
        ).hexdigest()

@dataclass(frozen=True, slots=True)
class EvaluatorRegistry:
    benchmark_contract_hash: str
    evaluators: tuple[EvaluatorContract, ...]
    semantic_sha256: str

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(item.family for item in self.evaluators)

    def for_family(self, family: str) -> EvaluatorContract:
        for item in self.evaluators:
            if item.family == family:
                return item
        raise EvaluatorContractError(f"evaluator family is not registered: {family}")

@dataclass(frozen=True, slots=True)
class FamilyAvailability:
    family: str
    primary: MetricUnavailable | None
    optional: Mapping[str, MetricUnavailable]

    @property
    def status(self) -> MetricStatus:
        return (
            MetricStatus.UNAVAILABLE
            if self.primary is not None
            else MetricStatus.AVAILABLE
        )

@dataclass(frozen=True, slots=True)
class EvaluatorAvailability:
    families: tuple[FamilyAvailability, ...]

    @property
    def ready(self) -> bool:
        return all(item.primary is None for item in self.families)

    def for_family(self, family: str) -> FamilyAvailability:
        for item in self.families:
            if item.family == family:
                return item
        raise EvaluatorContractError(f"evaluator family is not registered: {family}")

_REGISTRY_SCHEMA = "dive-evaluator-registry-v1"
_FAMILIES = ("binder", "ame", "antibody")
_DIRECTIONS = frozenset(("higher", "lower"))
_SHA256 = frozenset("0123456789abcdef")
_AMENDED_FAMILIES = frozenset(("binder", "ame"))
_RAW_VALUE_TYPES = frozenset(("string", "boolean", "float"))
_RAW_UNITS = frozenset(("sequence", "boolean", "normalized", "angstrom", "raw_rf3"))
_WORKER_SCHEMA = "dive-evaluator-worker-v1"
_REQUEST_SCHEMA = "dive-evaluator-request-v1"
_RESULT_SCHEMA = "dive-evaluator-result-v1"
_PRODUCTION_WORKER_SOURCE = Path("tests/benchmark/evaluators/fixture_worker.py")
_PRODUCTION_WORKER_FIXTURES = {
    "binder": Path("tests/benchmark/evaluators/fixtures/binder-result-v1.json"),
    "ame": Path("tests/benchmark/evaluators/fixtures/ame-result-v1.json"),
}
_FAILURE_REASONS = (
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
_SOURCE_PREFIXES = (Path("src/proteinfoundation"),)
_SOURCE_MODULE_PREFIXES = tuple(
    prefix.relative_to("src") for prefix in _SOURCE_PREFIXES
)

def load_evaluator_registry(contract: object) -> EvaluatorRegistry:

    config_path = _config_path(contract)
    repository_root = _repository_root(config_path)
    allow_test_fixture = (
        config_path.resolve()
        != (repository_root / "configs" / "emergent" / "benchmark.yaml").resolve()
    )
    raw = _read_yaml(config_path)
    registry = _mapping(raw.get("evaluator_registry"), "evaluator_registry")
    _exact_keys(
        registry, {"schema_version", "interpreter", "evaluators"}, "evaluator_registry"
    )
    if registry["schema_version"] != _REGISTRY_SCHEMA:
        raise EvaluatorContractError("unsupported evaluator registry schema_version")
    interpreter = _interpreter(registry["interpreter"])
    evaluators_raw = _mapping(registry["evaluators"], "evaluator_registry.evaluators")
    _exact_keys(evaluators_raw, set(_FAMILIES), "evaluator_registry.evaluators")
    benchmark_evaluators = {
        item.family: item.primary for item in getattr(contract, "evaluators", ())
    }
    evaluators = tuple(
        _evaluator(
            family,
            evaluators_raw[family],
            contract,
            interpreter,
            benchmark_evaluators.get(family),
            repository_root,
            allow_test_fixture,
        )
        for family in _FAMILIES
    )
    _authenticate_runtime_dependencies(evaluators)
    _verify_production_checkout(contract)
    semantic_sha256 = hashlib.sha256(
        canonical_json_bytes(_registry_mapping(evaluators))
    ).hexdigest()
    return EvaluatorRegistry(
        getattr(contract, "semantic_sha256", "unbound"), evaluators, semantic_sha256
    )

def verify_evaluator_availability(registry: EvaluatorRegistry) -> EvaluatorAvailability:

    interpreters = {evaluator.interpreter for evaluator in registry.evaluators}
    interpreters.update(
        asset.runtime.interpreter
        for evaluator in registry.evaluators
        for asset in evaluator.assets
        if asset.runtime is not None
    )
    for interpreter in interpreters:
        _authenticate_interpreter(interpreter, "evaluator interpreter")
    dependency_availability = _authenticate_runtime_dependencies(registry.evaluators)
    families = []
    for evaluator in registry.evaluators:
        primary: MetricUnavailable | None = None
        optional: dict[str, MetricUnavailable] = {}
        for asset in evaluator.assets:
            unavailable = _asset_unavailable(asset, registry.semantic_sha256)
            if unavailable is None:
                continue
            if asset.required:
                if primary is None:
                    identity_path = (
                        asset.identity.path
                        if asset.identity is not None
                        else asset.provenance or "no authenticated asset"
                    )
                    primary = MetricUnavailable(
                        evaluator.primary.name,
                        "required_asset_missing",
                        f"required evaluator asset {asset.role}: {identity_path}",
                        registry.semantic_sha256,
                    )
            else:
                optional[unavailable.name] = unavailable
        dependencies = [
            (dependency, evaluator.interpreter)
            for dependency in evaluator.runtime_dependencies
        ]
        dependencies.extend(
            (dependency, asset.runtime.interpreter)
            for asset in evaluator.assets
            if asset.runtime is not None
            for dependency in (asset.runtime.package, *asset.runtime.dependencies)
        )
        for dependency, interpreter in dependencies:
            if dependency_availability[(interpreter.path, dependency.module)]:
                continue
            reason_code = dependency.reason_code or (
                "required_dependency_missing"
                if dependency.required
                else "optional_dependency_missing"
            )
            detail = dependency.provenance or (
                f"runtime module {dependency.module} is unavailable through "
                f"{interpreter.path}"
            )
            for metric in dependency.metrics:
                unavailable = MetricUnavailable(
                    metric,
                    reason_code,
                    detail,
                    registry.semantic_sha256,
                )
                if metric == evaluator.primary.name:
                    if primary is None:
                        primary = unavailable
                else:
                    optional[metric] = unavailable
        families.append(FamilyAvailability(evaluator.family, primary, optional))
    return EvaluatorAvailability(tuple(families))

def _evaluator(
    family: str,
    value: object,
    contract: object,
    interpreter: InterpreterContract,
    benchmark_primary: str | None,
    repository_root: Path,
    allow_test_fixture: bool,
) -> EvaluatorContract:
    entry = _mapping(value, f"evaluator_registry.evaluators.{family}")
    expected_keys = {
        "name",
        "semantic_version",
        "source_repository",
        "source_commit",
        "primary",
        "input_schema",
        "output_schema",
        "required_metrics",
        "optional_metrics",
        "assets",
        "source_files",
        "runtime_dependencies",
        "provenance_status",
    }
    if family in _AMENDED_FAMILIES:
        expected_keys.update(
            {"worker", "raw_fields", "reducer", "evidence", "failure_reasons"}
        )
    _exact_keys(entry, expected_keys, f"evaluator_registry.evaluators.{family}")
    for key in (
        "name",
        "semantic_version",
        "input_schema",
        "output_schema",
        "provenance_status",
    ):
        if type(entry[key]) is not str or not entry[key]:
            raise EvaluatorContractError(
                f"evaluator {family}.{key} must be a non-empty string"
            )
    if entry["name"] != family:
        raise EvaluatorContractError(f"evaluator {family}.name must equal its family")
    if entry["source_repository"] != "proteina-complexa":
        raise EvaluatorContractError(
            f"evaluator {family} has an unexpected source repository"
        )
    if entry["source_commit"] != getattr(contract, "upstream_commit", None):
        raise EvaluatorContractError(
            f"evaluator {family} has an unexpected upstream commit"
        )
    required = _metrics(entry["required_metrics"], family, required=True)
    optional = _metrics(entry["optional_metrics"], family, required=False)
    names = {metric.name for metric in required}
    if names & {metric.name for metric in optional}:
        raise EvaluatorContractError(
            f"evaluator {family} repeats a required metric as optional"
        )
    primary_name = entry["primary"]
    if type(primary_name) is not str or primary_name not in names:
        raise EvaluatorContractError(
            f"evaluator {family}.primary must be a required metric"
        )
    if benchmark_primary is not None and not _core_primary_matches(
        family, benchmark_primary, primary_name
    ):
        raise EvaluatorContractError(
            f"evaluator {family}.primary disagrees with benchmark contract"
        )
    primary = next(metric for metric in required if metric.name == primary_name)
    declared_metrics = {metric.name for metric in (*required, *optional)}
    assets = _assets(entry["assets"], family, contract, declared_metrics, primary_name)
    source_files = _source_files(
        entry["source_files"],
        family,
        contract,
        declared_metrics,
    )
    dependencies = _dependencies(
        entry["runtime_dependencies"], family, declared_metrics, primary_name
    )
    _assert_source_dependencies(source_files, dependencies, family)
    worker = raw_fields = reducer = evidence = None
    failure_reasons: tuple[str, ...] = ()
    if family in _AMENDED_FAMILIES:
        worker = _worker(entry["worker"], family, repository_root, allow_test_fixture)
        raw_fields = _raw_fields(entry["raw_fields"], family)
        reducer = _reducer(entry["reducer"], family)
        evidence = _evidence(entry["evidence"], family)
        failure_reasons = _failure_reasons(entry["failure_reasons"], family)
        if family == "ame" and primary_name != "ame_motif_ligand_success":
            raise EvaluatorContractError(
                "evaluator ame.primary must be ame_motif_ligand_success"
            )
    return EvaluatorContract(
        family,
        entry["name"],
        entry["semantic_version"],
        entry["source_repository"],
        entry["source_commit"],
        interpreter,
        entry["input_schema"],
        entry["output_schema"],
        primary,
        required,
        optional,
        assets,
        source_files,
        dependencies,
        entry["provenance_status"],
        worker,
        raw_fields or (),
        reducer,
        evidence,
        failure_reasons,
    )

def _worker(
    value: object, family: str, repository_root: Path, allow_test_fixture: bool
) -> WorkerContract:
    label = f"evaluator {family}.worker"
    entry = _mapping(value, label)
    _exact_keys(
        entry,
        {
            "schema_version",
            "test_fixture",
            "source",
            "interpreter",
            "callable_name",
            "argv_template",
            "working_directory",
            "environment_allowlist",
            "environment",
            "request_schema",
            "result_schema",
            "fixture_source",
            "fixture_result_sha256",
        },
        label,
    )
    test_fixture = entry["test_fixture"]
    if type(test_fixture) is not bool or (test_fixture and not allow_test_fixture):
        raise EvaluatorContractError(
            f"{label} has an invalid authenticated worker contract"
        )
    source = _worker_identity(
        entry["source"],
        f"{label}.source",
        repository_root,
        _PRODUCTION_WORKER_SOURCE,
        allow_test_fixture=test_fixture,
    )
    fixture_source = _worker_identity(
        entry["fixture_source"],
        f"{label}.fixture_source",
        repository_root,
        _PRODUCTION_WORKER_FIXTURES[family],
        allow_test_fixture=test_fixture,
    )
    interpreter = _interpreter(entry["interpreter"])
    callable_name = entry["callable_name"]
    argv_template = entry["argv_template"]
    working_directory = entry["working_directory"]
    environment_allowlist = entry["environment_allowlist"]
    environment = entry["environment"]
    fixture_result_sha256 = entry["fixture_result_sha256"]
    if (
        entry["schema_version"] != _WORKER_SCHEMA
        or type(callable_name) is not str
        or not callable_name
        or not isinstance(argv_template, list)
        or tuple(argv_template)
        != (
            "{interpreter}",
            "{worker}",
            "--request-fd",
            "{request_fd}",
            "--result-fd",
            "{result_fd}",
            "--fixture-fd",
            "{fixture_fd}",
        )
        or type(working_directory) is not str
        or working_directory not in (".", str(repository_root))
        or not isinstance(environment_allowlist, list)
        or tuple(environment_allowlist) != ("PYTHONPATH",)
        or not isinstance(environment, dict)
        or tuple(sorted(environment)) != tuple(environment_allowlist)
        or any(
            type(name) is not str or type(item) is not str
            for name, item in environment.items()
        )
        or entry["request_schema"] != _REQUEST_SCHEMA
        or entry["result_schema"] != _RESULT_SCHEMA
        or not _is_sha256(fixture_result_sha256)
        or fixture_result_sha256 != fixture_source.sha256
    ):
        raise EvaluatorContractError(
            f"{label} has invalid authenticated worker contract"
        )
    return WorkerContract(
        entry["schema_version"],
        source,
        interpreter,
        callable_name,
        tuple(argv_template),
        working_directory,
        str(repository_root.resolve())
        if working_directory == "."
        else str(Path(working_directory).resolve()),
        tuple(environment_allowlist),
        tuple(sorted(environment.items())),
        entry["request_schema"],
        entry["result_schema"],
        fixture_source,
        fixture_result_sha256,
    )

def _worker_identity(
    value: object,
    label: str,
    repository_root: Path,
    expected_path: Path,
    *,
    allow_test_fixture: bool,
) -> ArtifactIdentity:
    entry = _mapping(value, label)
    _exact_keys(entry, {"path", "sha256", "size_bytes"}, label)
    raw_path, digest, size = entry["path"], entry["sha256"], entry["size_bytes"]
    if not _is_sha256(digest) or type(size) is not int or size < 0:
        raise EvaluatorContractError(f"{label} has an invalid artifact identity")
    if type(raw_path) is not str or not raw_path:
        raise EvaluatorContractError(f"{label}.path must be a non-empty string")
    candidate = Path(raw_path)
    if allow_test_fixture:
        if not candidate.is_absolute():
            raise EvaluatorContractError(
                f"{label} test fixture must use an absolute path"
            )
        identity = ArtifactIdentity(str(candidate), digest, size)
        _assert_identity(identity, label)
        return identity
    if candidate != expected_path:
        raise EvaluatorContractError(f"{label} is not an approved repository artifact")
    resolved = (repository_root / candidate).resolve()
    if not resolved.is_relative_to(repository_root):
        raise EvaluatorContractError(f"{label} escapes the repository root")
    observed = file_identity(resolved)
    if observed.sha256 != digest or observed.size_bytes != size:
        raise EvaluatorContractError(f"{label} sha256 or size drifted")
    return ArtifactIdentity(candidate.as_posix(), digest, size)

def _raw_fields(value: object, family: str) -> tuple[RawFieldContract, ...]:
    label = f"evaluator {family}.raw_fields"
    if not isinstance(value, list) or not value:
        raise EvaluatorContractError(f"{label} must be a non-empty list")
    fields = []
    for index, raw in enumerate(value):
        entry = _mapping(raw, f"{label}[{index}]")
        _exact_keys(
            entry, {"name", "value_type", "unit", "required"}, f"{label}[{index}]"
        )
        name, value_type, unit, required = (
            entry["name"],
            entry["value_type"],
            entry["unit"],
            entry["required"],
        )
        if (
            type(name) is not str
            or not name
            or value_type not in _RAW_VALUE_TYPES
            or unit not in _RAW_UNITS
            or type(required) is not bool
        ):
            raise EvaluatorContractError(f"{label} has an invalid field contract")
        fields.append(RawFieldContract(name, value_type, unit, required))
    names = tuple(field.name for field in fields)
    if names != tuple(sorted(set(names))):
        raise EvaluatorContractError(f"{label} names must be sorted and unique")
    return tuple(fields)

def _reducer(value: object, family: str) -> ReducerContract:
    label = f"evaluator {family}.reducer"
    entry = _mapping(value, label)
    _exact_keys(entry, {"identifier", "semantic_sha256"}, label)
    identifier, semantic_sha256 = entry["identifier"], entry["semantic_sha256"]
    if type(identifier) is not str or not identifier or not _is_sha256(semantic_sha256):
        raise EvaluatorContractError(f"{label} has an invalid reducer contract")
    return ReducerContract(identifier, semantic_sha256)

def _evidence(value: object, family: str) -> EvidenceContract:
    label = f"evaluator {family}.evidence"
    entry = _mapping(value, label)
    _exact_keys(entry, {"terminal_root", "bulk_root", "create_new"}, label)
    terminal_root, bulk_root, create_new = (
        entry["terminal_root"],
        entry["bulk_root"],
        entry["create_new"],
    )
    terminal_boundary = EMERGENT_EVIDENCE_ROOT / "benchmark_ready"
    bulk_boundary = EMERGENT_BULK_ROOT / "benchmark_ready"
    if (
        not _approved_evidence_root(terminal_root, terminal_boundary)
        or not _approved_evidence_root(bulk_root, bulk_boundary)
        or create_new is not True
    ):
        raise EvaluatorContractError(f"{label} has an invalid evidence contract")
    return EvidenceContract(terminal_root, bulk_root, create_new)

def _approved_evidence_root(value: object, boundary: Path) -> bool:
    if type(value) is not str:
        return False
    declared = Path(value)
    if not declared.is_absolute():
        return False
    try:
        resolved = declared.resolve(strict=False)
        expected = boundary.resolve(strict=False)
    except OSError:
        return False
    return resolved == declared and (
        resolved == expected or expected in resolved.parents
    )

def _failure_reasons(value: object, family: str) -> tuple[str, ...]:
    label = f"evaluator {family}.failure_reasons"
    if not isinstance(value, list) or tuple(value) != _FAILURE_REASONS:
        raise EvaluatorContractError(
            f"{label} must match the frozen failure vocabulary"
        )
    return tuple(value)

def _interpreter(value: object) -> InterpreterContract:
    entry = _mapping(value, "evaluator_registry.interpreter")
    _exact_keys(
        entry,
        {"path", "sha256", "size_bytes", "version", "environment"},
        "evaluator_registry.interpreter",
    )
    version = entry["version"]
    declared_path = entry["path"]
    if type(version) is not str or not version:
        raise EvaluatorContractError("evaluator interpreter.version must be non-empty")
    if type(declared_path) is not str or not declared_path:
        raise EvaluatorContractError("evaluator interpreter.path must be non-empty")
    identity = _identity(entry, "evaluator_registry.interpreter")
    environment = _identity(
        _mapping(entry["environment"], "evaluator_registry.interpreter.environment"),
        "evaluator_registry.interpreter.environment",
    )
    interpreter = InterpreterContract(declared_path, identity, version, environment)
    _authenticate_interpreter(interpreter, "evaluator interpreter")
    return interpreter

def _authenticate_interpreter(interpreter: InterpreterContract, label: str) -> None:
    declared = Path(interpreter.path)
    if not declared.exists():
        raise EvaluatorContractError(f"{label} is missing")
    if declared.resolve() != Path(interpreter.identity.path):
        raise EvaluatorContractError(f"{label} resolved path drifted")
    _assert_identity(interpreter.identity, label)
    _assert_identity(interpreter.environment, f"{label} environment")
    completed = subprocess.run(
        [interpreter.path, "-I", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    observed = (completed.stdout or completed.stderr).strip().removeprefix("Python ")
    if completed.returncode != 0 or observed != interpreter.version:
        raise EvaluatorContractError(f"{label} version drifted")

def _metrics(
    value: object, family: str, *, required: bool
) -> tuple[MetricContract, ...]:
    entry = _mapping(value, f"evaluator {family} metrics")
    if not entry:
        if required:
            raise EvaluatorContractError(f"evaluator {family} has no required metrics")
        return ()
    metrics = []
    for name in sorted(entry):
        direction = entry[name]
        if type(name) is not str or not name or direction not in _DIRECTIONS:
            raise EvaluatorContractError(
                f"evaluator {family} has invalid metric direction"
            )
        metrics.append(MetricContract(name, direction, required))
    return tuple(metrics)

def _assets(
    value: object,
    family: str,
    contract: object,
    declared_metrics: set[str],
    primary: str,
) -> tuple[EvaluatorAsset, ...]:
    if not isinstance(value, list):
        raise EvaluatorContractError(f"evaluator {family}.assets must be a list")
    declared_checkpoints = dict(getattr(contract, "checkpoints", ()))
    assets, roles = [], set()
    for index, raw in enumerate(value):
        entry = _mapping(raw, f"evaluator {family}.assets[{index}]")
        if entry.get("kind") == "unavailable_requirement":
            _exact_keys(
                entry,
                {"role", "kind", "required", "metric", "reason_code", "provenance"},
                f"evaluator {family}.assets[{index}]",
            )
            role, required = entry["role"], entry["required"]
            metric, reason_code, provenance = (
                entry["metric"],
                entry["reason_code"],
                entry["provenance"],
            )
            if (
                type(role) is not str
                or not role
                or role in roles
                or required
                or type(metric) is not str
                or not metric
                or type(reason_code) is not str
                or not reason_code
                or type(provenance) is not str
                or not provenance
            ):
                raise EvaluatorContractError(
                    f"evaluator {family} has invalid unavailable requirement"
                )
            assets.append(
                EvaluatorAsset(
                    role,
                    "unavailable_requirement",
                    None,
                    False,
                    None,
                    metric,
                    reason_code,
                    provenance,
                )
            )
            roles.add(role)
            continue
        role, kind, required, checkpoint = (
            entry["role"],
            entry["kind"],
            entry["required"],
            entry["checkpoint"],
        )
        expected_keys = {
            "role",
            "kind",
            "required",
            "checkpoint",
            "path",
            "sha256",
            "size_bytes",
        }
        if kind == "executable":
            expected_keys.add("runtime")
        if kind == "benchmark_executable":
            expected_keys.update({"metric", "tool", "version"})
        _exact_keys(entry, expected_keys, f"evaluator {family}.assets[{index}]")
        if (
            type(role) is not str
            or not role
            or role in roles
            or kind
            not in {
                "benchmark_executable",
                "checkpoint",
                "configuration",
                "executable",
                "database",
                "reference_database",
                "source_file",
            }
            or type(required) is not bool
        ):
            raise EvaluatorContractError(
                f"evaluator {family} has invalid or repeated asset role"
            )
        if kind == "checkpoint" and (
            type(checkpoint) is not str or checkpoint not in declared_checkpoints
        ):
            raise EvaluatorContractError(
                f"evaluator {family} checkpoint asset {role} has undeclared checkpoint"
            )
        if kind != "checkpoint" and checkpoint is not None:
            raise EvaluatorContractError(
                f"evaluator {family} non-checkpoint asset {role} cannot bind checkpoint"
            )
        if checkpoint is not None and (
            type(checkpoint) is not str or checkpoint not in declared_checkpoints
        ):
            raise EvaluatorContractError(
                f"evaluator {family} asset {role} has undeclared checkpoint"
            )
        identity = _identity(entry, f"evaluator {family}.assets[{index}]")
        if checkpoint is not None and declared_checkpoints[checkpoint] != identity:
            raise EvaluatorContractError(
                f"evaluator {family} asset {role} disagrees with checkpoint identity"
            )
        metric = tool = version = None
        if kind == "benchmark_executable":
            metric, tool, version = entry["metric"], entry["tool"], entry["version"]
            foldseek = getattr(contract, "foldseek", None)
            if (
                required
                or type(metric) is not str
                or metric not in declared_metrics
                or tool != "foldseek"
                or type(version) is not str
                or not version
                or foldseek is None
                or identity != foldseek.identity
                or version != foldseek.version
            ):
                raise EvaluatorContractError(
                    f"evaluator {family} benchmark executable {role} disagrees with benchmark tool contract"
                )
        runtime = (
            _executable_runtime(
                entry["runtime"],
                Path(identity.path),
                family,
                role,
                declared_metrics,
                primary,
            )
            if kind == "executable"
            else None
        )
        asset = EvaluatorAsset(
            role,
            kind,
            identity,
            required,
            checkpoint,
            metric=metric,
            runtime=runtime,
            tool=tool,
            version=version,
        )

        if Path(asset.identity.path).exists():
            _assert_identity(asset.identity, f"evaluator {family} asset {role}")
            if kind in {"benchmark_executable", "executable"} and not os.access(
                asset.identity.path, os.X_OK
            ):
                raise EvaluatorContractError(
                    f"evaluator {family} executable asset {role} is not executable"
                )
        assets.append(asset)
        roles.add(role)
    return tuple(assets)

def _executable_runtime(
    value: object,
    executable: Path,
    family: str,
    role: str,
    declared_metrics: set[str],
    primary: str,
) -> ExecutableRuntime:
    entry = _mapping(value, f"evaluator {family} executable {role}.runtime")
    _exact_keys(
        entry,
        {"shebang", "entry_module", "interpreter", "package", "dependencies"},
        f"evaluator {family} executable {role}.runtime",
    )
    shebang = entry["shebang"]
    if type(shebang) is not str or not shebang or not Path(shebang).is_absolute():
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} shebang interpreter is invalid"
        )
    interpreter = _interpreter(entry["interpreter"])
    if shebang != interpreter.path:
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} shebang interpreter disagrees with runtime"
        )
    package = _dependency(
        entry["package"],
        family,
        declared_metrics,
        primary,
        f"evaluator {family} executable {role}.runtime.package",
    )
    if package.kind != "module":
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} package must have an authenticated module tree"
        )
    entry_module = entry["entry_module"]
    if (
        type(entry_module) is not str
        or not entry_module
        or not (
            entry_module == package.module
            or entry_module.startswith(f"{package.module}.")
        )
    ):
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} entry module is invalid"
        )
    dependencies = _dependencies(
        entry["dependencies"], family, declared_metrics, primary
    )
    if any(dependency.kind != "module" for dependency in dependencies):
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} direct runtime dependency must be authenticated"
        )
    if executable.exists():
        try:
            first_line = executable.open("rb").readline().decode("utf-8").rstrip("\r\n")
        except (OSError, UnicodeError) as error:
            raise EvaluatorContractError(
                f"evaluator {family} executable {role} shebang is unreadable"
            ) from error
        if first_line != f"#!{shebang}":
            raise EvaluatorContractError(
                f"evaluator {family} executable {role} shebang interpreter drifted"
            )
    _assert_executable_runtime_dependencies(
        executable,
        entry_module,
        package,
        dependencies,
        family,
        role,
    )
    return ExecutableRuntime(shebang, entry_module, interpreter, package, dependencies)

def _assert_executable_runtime_dependencies(
    executable: Path,
    entry_module: str,
    package: RuntimeDependency,
    dependencies: tuple[RuntimeDependency, ...],
    family: str,
    role: str,
) -> None:

    assert package.tree is not None
    package_root = Path(package.tree.root)
    entry_source = _runtime_module_source(package_root, package.module, entry_module)
    if entry_source is None:
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} entry module is missing"
        )
    closure: dict[str, Path] = {}
    pending: list[tuple[str, Path]] = []

    def add_internal(module: str) -> bool:
        source = _runtime_module_source(package_root, package.module, module)
        if source is None:
            return False
        if module not in closure:
            closure[module] = source
            pending.append((module, source))
        for initializer_module, initializer in _runtime_package_initializers(
            package_root, package.module, module
        ):
            if initializer_module not in closure:
                closure[initializer_module] = initializer
                pending.append((initializer_module, initializer))
        return True

    add_internal(entry_module)
    imported: set[str] = set()
    if executable.exists():
        wrapper_imports = _runtime_imports(executable, "")
        wrapper_internal = False
        for base, aliases in wrapper_imports:
            if base == entry_module:
                wrapper_internal = True
            elif any(f"{base}.{alias}" == entry_module for alias in aliases):
                wrapper_internal = True
            top = base.partition(".")[0]
            if (
                top
                and top != package.module
                and top not in sys.stdlib_module_names
                and top != "__future__"
            ):
                imported.add(top)
        if not wrapper_internal:
            raise EvaluatorContractError(
                f"evaluator {family} executable {role} does not import its entry module"
            )
    while pending:
        module, source = pending.pop()
        for base, aliases in _runtime_imports(source, module):
            if base == package.module or base.startswith(f"{package.module}."):
                add_internal(base)
                for alias in aliases:
                    add_internal(f"{base}.{alias}")
                continue
            top = base.partition(".")[0]
            if top and top not in sys.stdlib_module_names and top != "__future__":
                imported.add(top)
    declared = {dependency.module for dependency in dependencies}
    missing = sorted(imported - declared)
    extra = sorted(declared - imported)
    if missing:
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} has missing direct runtime dependencies {missing}"
        )
    if extra:
        raise EvaluatorContractError(
            f"evaluator {family} executable {role} has extra direct runtime dependencies {extra}"
        )

def _runtime_imports(
    path: Path, module: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise EvaluatorContractError(
            f"cannot parse executable runtime source {path}: {error}"
        ) from error
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_module(package, node.level, node.module)
            if base:
                imports.append((base, tuple(alias.name for alias in node.names)))
    return tuple(imports)

def _runtime_module_source(
    package_root: Path, package_module: str, module: str
) -> Path | None:
    if module == package_module:
        candidate = package_root
    elif module.startswith(f"{package_module}."):
        relative = module.removeprefix(f"{package_module}.")
        candidate = package_root.joinpath(*relative.split("."))
    else:
        return None
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path.resolve()
    return None

def _runtime_package_initializers(
    package_root: Path, package_module: str, module: str
) -> tuple[tuple[str, Path], ...]:
    suffix = module.removeprefix(package_module).removeprefix(".")
    parts = suffix.split(".") if suffix else []
    result = []
    for end in range(0, len(parts)):
        initializer_module = ".".join((package_module, *parts[:end]))
        initializer = package_root.joinpath(*parts[:end], "__init__.py")
        if initializer.is_file():
            result.append((initializer_module, initializer.resolve()))
    return tuple(result)

def _source_files(
    value: object, family: str, contract: object, declared_metrics: set[str]
) -> tuple[EvaluatorSourceFile, ...]:
    if not isinstance(value, list) or not value:
        raise EvaluatorContractError(
            f"evaluator {family}.source_files must be a non-empty list"
        )
    root = Path(getattr(contract, "upstream_root", "")).resolve()
    sources, paths = [], set()
    for index, raw in enumerate(value):
        entry = _mapping(raw, f"evaluator {family}.source_files[{index}]")
        _exact_keys(
            entry,
            {"metric", "entry_point", "path", "sha256", "size_bytes"},
            f"evaluator {family}.source_files[{index}]",
        )
        metric = entry["metric"]
        entry_point = entry["entry_point"]
        if type(metric) is not str or metric not in declared_metrics:
            raise EvaluatorContractError(
                f"evaluator {family} source owner is not a declared metric"
            )
        if type(entry_point) is not bool:
            raise EvaluatorContractError(
                f"evaluator {family} source entry_point is invalid"
            )
        identity = _identity(entry, f"evaluator {family}.source_files[{index}]")
        path = Path(identity.path).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise EvaluatorContractError(
                f"evaluator {family} source escapes upstream root"
            ) from error
        if not any(relative.is_relative_to(prefix) for prefix in _SOURCE_PREFIXES):
            raise EvaluatorContractError(
                f"evaluator {family} source is outside pinned evaluator directories"
            )
        if path in paths:
            raise EvaluatorContractError(f"evaluator {family} repeats a source file")
        _assert_identity(identity, f"evaluator {family} source file {path}")
        sources.append(EvaluatorSourceFile(metric, entry_point, identity))
        paths.add(path)
    result = tuple(sources)
    _assert_source_closure(result, family, root)
    return result

def _dependencies(
    value: object, family: str, declared_metrics: set[str], primary: str
) -> tuple[RuntimeDependency, ...]:
    if not isinstance(value, list):
        raise EvaluatorContractError(
            f"evaluator {family}.runtime_dependencies must be a list"
        )
    dependencies = tuple(
        _dependency(
            raw,
            family,
            declared_metrics,
            primary,
            f"evaluator {family}.runtime_dependencies[{index}]",
        )
        for index, raw in enumerate(value)
    )
    modules = [dependency.module for dependency in dependencies]
    if len(set(modules)) != len(modules):
        raise EvaluatorContractError(f"evaluator {family} repeats a runtime dependency")
    return dependencies

def _dependency(
    value: object,
    family: str,
    declared_metrics: set[str],
    primary: str,
    label: str,
) -> RuntimeDependency:
    entry = _mapping(value, label)
    kind = entry.get("kind")
    if kind == "unavailable_requirement":
        _exact_keys(
            entry,
            {
                "kind",
                "module",
                "required",
                "metrics",
                "reason_code",
                "provenance",
            },
            label,
        )
        reason_code, provenance = entry["reason_code"], entry["provenance"]
        tree = distribution = version = None
        if (
            type(reason_code) is not str
            or not reason_code
            or type(provenance) is not str
            or not provenance
        ):
            raise EvaluatorContractError(f"{label} has invalid unavailability evidence")
    elif kind == "module":
        _exact_keys(
            entry,
            {
                "kind",
                "module",
                "distribution",
                "version",
                "required",
                "metrics",
                "root",
                "sha256",
                "size_bytes",
                "file_count",
            },
            label,
        )
        distribution, version = entry["distribution"], entry["version"]
        if (distribution is None) != (version is None) or (
            distribution is not None
            and (
                type(distribution) is not str
                or not distribution
                or type(version) is not str
                or not version
            )
        ):
            raise EvaluatorContractError(f"{label} has invalid distribution version")
        tree = _module_tree_identity(entry, label)
        reason_code = provenance = None
    else:
        raise EvaluatorContractError(f"{label}.kind is invalid")
    module, required, metrics_value = (
        entry["module"],
        entry["required"],
        entry["metrics"],
    )
    if type(module) is not str or not module or type(required) is not bool:
        raise EvaluatorContractError(f"{label} has invalid module or required flag")
    if (
        not isinstance(metrics_value, list)
        or not metrics_value
        or not all(type(metric) is str for metric in metrics_value)
        or len(set(metrics_value)) != len(metrics_value)
        or not set(metrics_value) <= declared_metrics
    ):
        raise EvaluatorContractError(f"{label} has invalid metric owners")
    metrics = tuple(metrics_value)
    if required != (primary in metrics):
        raise EvaluatorContractError(
            f"{label} required flag disagrees with primary metric ownership"
        )
    return RuntimeDependency(
        kind,
        module,
        distribution,
        version,
        tree,
        required,
        metrics,
        reason_code,
        provenance,
    )

def _module_tree_identity(
    value: Mapping[str, object], label: str
) -> ModuleTreeIdentity:
    root, digest, size, count = (
        value["root"],
        value["sha256"],
        value["size_bytes"],
        value["file_count"],
    )
    if (
        type(root) is not str
        or not root
        or not Path(root).is_absolute()
        or type(digest) is not str
        or len(digest) != 64
        or not set(digest) <= _SHA256
        or type(size) is not int
        or size < 0
        or type(count) is not int
        or count < 1
    ):
        raise EvaluatorContractError(f"{label} has invalid module tree identity")
    return ModuleTreeIdentity(str(Path(root).resolve()), digest, size, count)

_MODULE_PROBE = r"""
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys

requests = json.loads(sys.argv[1])
result = {}
for request in requests:
    module = request["module"]
    declared_root = request.get("root")
    if declared_root:
        parent = str(pathlib.Path(declared_root).resolve().parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
    spec = importlib.util.find_spec(module)
    if spec is None:
        result[module] = None
        continue
    if spec.submodule_search_locations:
        root = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    else:
        if spec.origin is None:
            raise RuntimeError(f"module {module} has no file-backed origin")
        root = pathlib.Path(spec.origin).resolve()
        files = [root]
    records = []
    for path in files:
        data = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix() if root.is_dir() else path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    distribution = request["distribution"]
    result[module] = {
        "distribution": distribution,
        "version": (
            importlib.metadata.version(distribution)
            if distribution is not None
            else None
        ),
        "root": str(root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": sum(record["size_bytes"] for record in records),
        "file_count": len(records),
    }
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""

def _authenticate_runtime_dependencies(
    evaluators: tuple[EvaluatorContract, ...],
) -> dict[tuple[str, str], bool]:
    by_interpreter: dict[
        str, tuple[InterpreterContract, dict[str, RuntimeDependency]]
    ] = {}
    for evaluator in evaluators:
        groups = [(evaluator.interpreter, evaluator.runtime_dependencies)]
        groups.extend(
            (
                asset.runtime.interpreter,
                (asset.runtime.package, *asset.runtime.dependencies),
            )
            for asset in evaluator.assets
            if asset.runtime is not None
        )
        for interpreter, dependencies in groups:
            if interpreter.path not in by_interpreter:
                by_interpreter[interpreter.path] = (interpreter, {})
            known = by_interpreter[interpreter.path][1]
            for dependency in dependencies:
                prior = known.get(dependency.module)
                if prior is not None and _dependency_identity(
                    prior
                ) != _dependency_identity(dependency):
                    raise EvaluatorContractError(
                        f"runtime dependency {dependency.module} has conflicting identities"
                    )
                known[dependency.module] = dependency

    availability: dict[tuple[str, str], bool] = {}
    for interpreter, dependencies in by_interpreter.values():
        requests = [
            {
                "module": dependency.module,
                "distribution": dependency.distribution,
                "root": None if dependency.tree is None else dependency.tree.root,
            }
            for dependency in dependencies.values()
        ]
        completed = subprocess.run(
            [interpreter.path, "-I", "-c", _MODULE_PROBE, json.dumps(requests)],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            observed = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise EvaluatorContractError(
                "evaluator interpreter dependency probe failed"
            ) from error
        if completed.returncode != 0 or set(observed) != set(dependencies):
            raise EvaluatorContractError(
                "evaluator interpreter dependency probe failed"
            )
        for module, dependency in dependencies.items():
            record = observed[module]
            present = record is not None
            availability[(interpreter.path, module)] = present
            if dependency.kind == "unavailable_requirement":
                if present:
                    raise EvaluatorContractError(
                        f"runtime dependency {module} is present without an authenticated identity"
                    )
                continue
            if not present:
                continue
            assert dependency.tree is not None
            expected = {
                "distribution": dependency.distribution,
                "version": dependency.version,
                "root": dependency.tree.root,
                "sha256": dependency.tree.sha256,
                "size_bytes": dependency.tree.size_bytes,
                "file_count": dependency.tree.file_count,
            }
            if record != expected:
                differing = sorted(
                    key for key in expected if record.get(key) != expected[key]
                )
                detail = (
                    "module tree sha256"
                    if "sha256" in differing
                    else ", ".join(differing)
                )
                raise EvaluatorContractError(
                    f"runtime dependency {module} {detail} drifted"
                )
    return availability

def _dependency_identity(dependency: RuntimeDependency) -> tuple[object, ...]:
    return (
        dependency.kind,
        dependency.module,
        dependency.distribution,
        dependency.version,
        dependency.tree,
        dependency.reason_code,
        dependency.provenance,
    )

def _assert_source_dependencies(
    source_files: tuple[EvaluatorSourceFile, ...],
    dependencies: tuple[RuntimeDependency, ...],
    family: str,
) -> None:

    declared = {dependency.module for dependency in dependencies}
    imported: set[str] = set()
    for source in source_files:
        path = Path(source.identity.path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise EvaluatorContractError(
                f"evaluator {family} cannot parse pinned source {path}: {error}"
            ) from error
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.partition(".")[0])
    imported = {
        item
        for item in imported
        if item not in sys.stdlib_module_names and item != "proteinfoundation"
    }
    undeclared = sorted(imported - declared)
    extra = sorted(declared - imported)
    if undeclared:
        raise EvaluatorContractError(
            f"evaluator {family} source has undeclared runtime dependency {undeclared}"
        )
    if extra:
        raise EvaluatorContractError(
            f"evaluator {family} source has extra runtime dependency {extra}"
        )

def _assert_source_closure(
    source_files: tuple[EvaluatorSourceFile, ...], family: str, upstream_root: Path
) -> None:

    declared = {Path(source.identity.path).resolve() for source in source_files}
    entry_points = [
        Path(source.identity.path).resolve()
        for source in source_files
        if source.entry_point
    ]
    if not entry_points:
        raise EvaluatorContractError(f"evaluator {family} has no source entry point")
    source_root = upstream_root / "src"
    closure = _source_seed(entry_points, source_root)
    pending = list(closure)
    while pending:
        current = pending.pop()
        for dependency in _in_scope_imports(current, source_root):
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    if closure != declared:
        missing = sorted(str(path) for path in closure - declared)
        extra = sorted(str(path) for path in declared - closure)
        raise EvaluatorContractError(
            f"evaluator {family} source closure has missing {missing} or extra {extra} entries"
        )
    entry_points_by_metric: dict[str, list[Path]] = {}
    for source in source_files:
        if source.entry_point:
            entry_points_by_metric.setdefault(source.metric, []).append(
                Path(source.identity.path).resolve()
            )
    closure_by_metric: dict[str, set[Path]] = {}
    for metric, metric_entry_points in entry_points_by_metric.items():
        metric_closure = _source_seed(metric_entry_points, source_root)
        pending = list(metric_closure)
        while pending:
            current = pending.pop()
            for dependency in _in_scope_imports(current, source_root):
                if dependency not in metric_closure:
                    metric_closure.add(dependency)
                    pending.append(dependency)
        closure_by_metric[metric] = metric_closure
    wrong_owners = sorted(
        f"{source.identity.path} -> {source.metric}"
        for source in source_files
        if Path(source.identity.path).resolve()
        not in closure_by_metric.get(source.metric, set())
    )
    if wrong_owners:
        raise EvaluatorContractError(
            f"evaluator {family} source owner is outside its metric closure {wrong_owners}"
        )

def _source_seed(entry_points: list[Path], source_root: Path) -> set[Path]:
    result = set(entry_points)
    for entry_point in entry_points:
        module = ".".join(entry_point.relative_to(source_root).with_suffix("").parts)
        result.update(_package_initializers(source_root, module))
    return result

def _in_scope_imports(path: Path, source_root: Path) -> tuple[Path, ...]:

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise EvaluatorContractError(
            f"cannot parse evaluator source {path}: {error}"
        ) from error
    package = ".".join(path.relative_to(source_root).with_suffix("").parts[:-1])
    candidates: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _module_source(source_root, alias.name)
                if resolved is not None:
                    candidates.add(resolved)
                candidates.update(_package_initializers(source_root, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_module(package, node.level, node.module)
            if module is None:
                continue
            direct = _module_source(source_root, module)
            if direct is not None:
                candidates.add(direct)
            candidates.update(_package_initializers(source_root, module))
            for alias in node.names:
                resolved = _module_source(source_root, f"{module}.{alias.name}")
                if resolved is not None:
                    candidates.add(resolved)
                candidates.update(
                    _package_initializers(source_root, f"{module}.{alias.name}")
                )
    return tuple(
        path
        for path in candidates
        if any(
            path.is_relative_to(source_root / prefix)
            for prefix in _SOURCE_MODULE_PREFIXES
        )
    )

def _resolve_import_module(package: str, level: int, module: str | None) -> str | None:
    if level == 0:
        return module
    parts = package.split(".")
    if level > len(parts) + 1:
        return None
    base = parts[: len(parts) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(base)

def _module_source(source_root: Path, module: str) -> Path | None:
    if not module.startswith("proteinfoundation"):
        return None
    candidate = source_root.joinpath(*module.split("."))
    for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        if path.is_file():
            return path.resolve()
    return None

def _package_initializers(source_root: Path, module: str | None) -> set[Path]:
    if module is None or not module.startswith("proteinfoundation"):
        return set()
    result = set()
    parts = module.split(".")
    for end in range(1, len(parts)):
        initializer = source_root.joinpath(*parts[:end], "__init__.py")
        if initializer.is_file():
            result.add(initializer.resolve())
    return result

def _asset_unavailable(
    asset: EvaluatorAsset, manifest_hash: str
) -> MetricUnavailable | None:
    if asset.identity is None:
        return MetricUnavailable(
            asset.metric or asset.role,
            asset.reason_code or "optional_requirement_unavailable",
            asset.provenance or "no authenticated optional asset",
            manifest_hash,
        )
    if not Path(asset.identity.path).exists():
        reason = (
            "optional_asset_missing" if not asset.required else "required_asset_missing"
        )
        return MetricUnavailable(
            asset.metric or asset.role,
            reason,
            f"evaluator asset {asset.role} missing: {asset.identity.path}",
            manifest_hash,
        )
    _assert_identity(asset.identity, f"evaluator asset {asset.role}")
    if asset.kind in {"benchmark_executable", "executable"} and not os.access(
        asset.identity.path, os.X_OK
    ):
        raise EvaluatorContractError(
            f"evaluator executable asset {asset.role} is not executable"
        )
    return None

def _assert_identity(identity: ArtifactIdentity, label: str) -> None:
    try:
        observed = file_identity(Path(identity.path))
    except Exception as error:
        raise EvaluatorContractError(
            f"{label} missing or unreadable: {error}"
        ) from error
    if observed != identity:
        raise EvaluatorContractError(
            f"{label} sha256 or size drifted: expected {identity}, observed {observed}"
        )

def _verify_production_checkout(contract: object) -> None:
    root = Path(getattr(contract, "upstream_root", ""))
    if root != EMERGENT_UPSTREAM_ROOT:
        return
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        head.returncode != 0
        or head.stdout.strip() != getattr(contract, "upstream_commit", None)
        or dirty.returncode != 0
        or dirty.stdout
    ):
        raise EvaluatorContractError(
            "pinned upstream checkout HEAD or cleanliness drifted"
        )

def _registry_mapping(evaluators: tuple[EvaluatorContract, ...]) -> dict[str, object]:
    return {
        "schema_version": _REGISTRY_SCHEMA,
        "evaluators": [
            {
                "family": item.family,
                "name": item.name,
                "semantic_version": item.semantic_version,
                "source_repository": item.source_repository,
                "source_commit": item.source_commit,
                "interpreter": {
                    "path": item.interpreter.path,
                    "sha256": item.interpreter.identity.sha256,
                    "size_bytes": item.interpreter.identity.size_bytes,
                    "version": item.interpreter.version,
                    "environment": {
                        "path": item.interpreter.environment.path,
                        "sha256": item.interpreter.environment.sha256,
                        "size_bytes": item.interpreter.environment.size_bytes,
                    },
                },
                "input_schema": item.input_schema,
                "output_schema": item.output_schema,
                "primary": item.primary.name,
                "required_metrics": {
                    metric.name: metric.direction for metric in item.required_metrics
                },
                "optional_metrics": {
                    metric.name: metric.direction for metric in item.optional_metrics
                },
                "assets": [
                    (
                        {
                            "role": asset.role,
                            "kind": asset.kind,
                            "required": asset.required,
                            "metric": asset.metric,
                            "reason_code": asset.reason_code,
                            "provenance": asset.provenance,
                        }
                        if asset.identity is None
                        else {
                            "role": asset.role,
                            "kind": asset.kind,
                            "required": asset.required,
                            "checkpoint": asset.checkpoint,
                            "path": asset.identity.path,
                            "sha256": asset.identity.sha256,
                            "size_bytes": asset.identity.size_bytes,
                            **(
                                {
                                    "metric": asset.metric,
                                    "tool": asset.tool,
                                    "version": asset.version,
                                }
                                if asset.kind == "benchmark_executable"
                                else {}
                            ),
                            **(
                                {"runtime": _executable_runtime_mapping(asset.runtime)}
                                if asset.runtime is not None
                                else {}
                            ),
                        }
                    )
                    for asset in item.assets
                ],
                "source_files": [
                    {
                        "metric": source.metric,
                        "entry_point": source.entry_point,
                        "path": source.identity.path,
                        "sha256": source.identity.sha256,
                        "size_bytes": source.identity.size_bytes,
                    }
                    for source in item.source_files
                ],
                "runtime_dependencies": [
                    _dependency_mapping(dependency)
                    for dependency in item.runtime_dependencies
                ],
                "provenance_status": item.provenance_status,
                **(
                    {
                        "worker": _worker_mapping(item.worker),
                        "raw_fields": [
                            {
                                "name": field.name,
                                "value_type": field.value_type,
                                "unit": field.unit,
                                "required": field.required,
                            }
                            for field in item.raw_fields
                        ],
                        "reducer": {
                            "identifier": item.reducer.identifier,
                            "semantic_sha256": item.reducer.semantic_sha256,
                        },
                        "evidence": {
                            "terminal_root": item.evidence.terminal_root,
                            "bulk_root": item.evidence.bulk_root,
                            "create_new": item.evidence.create_new,
                        },
                        "failure_reasons": list(item.failure_reasons),
                    }
                    if item.family in _AMENDED_FAMILIES
                    else {}
                ),
            }
            for item in evaluators
        ],
    }

def _worker_mapping(worker: WorkerContract | None) -> dict[str, object]:
    if worker is None:
        raise EvaluatorContractError("amended evaluator is missing a worker contract")
    return {
        "schema_version": worker.schema_version,
        "source": {
            "path": worker.source.path,
            "sha256": worker.source.sha256,
            "size_bytes": worker.source.size_bytes,
        },
        "interpreter": {
            "path": worker.interpreter.path,
            "sha256": worker.interpreter.identity.sha256,
            "size_bytes": worker.interpreter.identity.size_bytes,
            "version": worker.interpreter.version,
            "environment": {
                "path": worker.interpreter.environment.path,
                "sha256": worker.interpreter.environment.sha256,
                "size_bytes": worker.interpreter.environment.size_bytes,
            },
        },
        "callable_name": worker.callable_name,
        "argv_template": list(worker.argv_template),
        "working_directory": worker.working_directory,
        "environment_allowlist": list(worker.environment_allowlist),
        "environment": dict(worker.environment),
        "request_schema": worker.request_schema,
        "result_schema": worker.result_schema,
        "fixture_source": {
            "path": worker.fixture_source.path,
            "sha256": worker.fixture_source.sha256,
            "size_bytes": worker.fixture_source.size_bytes,
        },
        "fixture_result_sha256": worker.fixture_result_sha256,
    }

def _executable_runtime_mapping(runtime: ExecutableRuntime) -> dict[str, object]:
    return {
        "shebang": runtime.shebang,
        "entry_module": runtime.entry_module,
        "interpreter": {
            "path": runtime.interpreter.path,
            "sha256": runtime.interpreter.identity.sha256,
            "size_bytes": runtime.interpreter.identity.size_bytes,
            "version": runtime.interpreter.version,
            "environment": {
                "path": runtime.interpreter.environment.path,
                "sha256": runtime.interpreter.environment.sha256,
                "size_bytes": runtime.interpreter.environment.size_bytes,
            },
        },
        "package": _dependency_mapping(runtime.package),
        "dependencies": [
            _dependency_mapping(dependency) for dependency in runtime.dependencies
        ],
    }

def _dependency_mapping(dependency: RuntimeDependency) -> dict[str, object]:
    common: dict[str, object] = {
        "kind": dependency.kind,
        "module": dependency.module,
        "required": dependency.required,
        "metrics": list(dependency.metrics),
    }
    if dependency.kind == "unavailable_requirement":
        return {
            **common,
            "reason_code": dependency.reason_code,
            "provenance": dependency.provenance,
        }
    assert dependency.tree is not None
    return {
        **common,
        "distribution": dependency.distribution,
        "version": dependency.version,
        "root": dependency.tree.root,
        "sha256": dependency.tree.sha256,
        "size_bytes": dependency.tree.size_bytes,
        "file_count": dependency.tree.file_count,
    }

def _core_primary_matches(family: str, core: str, primary: str) -> bool:
    return (family, core, primary) in {
        ("binder", "target_conditioned_success", "target_conditioned_success"),
        ("ame", "ame_motif_ligand_success", "ame_motif_ligand_success"),
        ("antibody", "cdr_h3_ca_rmsd", "cdr_h3_ca_rmsd"),
    }

def _config_path(contract: object) -> Path:
    path = getattr(contract, "source_path", None) or getattr(
        contract, "config_path", None
    )
    if not isinstance(path, Path):
        raise EvaluatorContractError(
            "benchmark contract does not retain its config path"
        )
    return path

def _repository_root(config_path: Path) -> Path:
    try:
        root = config_path.resolve().parents[2]
    except OSError as error:
        raise EvaluatorContractError(
            f"cannot resolve benchmark config path {config_path}: {error}"
        ) from error
    if (root / "configs" / "emergent" / "benchmark.yaml").is_file():
        return root
    return config_path.resolve().parent

def _read_yaml(path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluatorContractError(
            f"cannot read evaluator registry {path}: {error}"
        ) from error
    return _mapping(raw, "benchmark config")

def _identity(value: Mapping[str, object], label: str) -> ArtifactIdentity:
    path, digest, size = value["path"], value["sha256"], value["size_bytes"]
    if (
        type(path) is not str
        or not path
        or type(digest) is not str
        or len(digest) != 64
        or not set(digest) <= _SHA256
    ):
        raise EvaluatorContractError(
            f"{label}.sha256 must be a lowercase SHA-256 identity"
        )
    if type(size) is not int or size < 0:
        raise EvaluatorContractError(f"{label}.size_bytes must be non-negative")
    return ArtifactIdentity(str(Path(path).resolve()), digest, size)

def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA256

def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise EvaluatorContractError(f"{label} must be a string-keyed mapping")
    return value

def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    unknown, missing = sorted(set(value) - expected), sorted(expected - set(value))
    if unknown or missing:
        raise EvaluatorContractError(
            f"{label} has unknown {unknown} or missing {missing} field(s)"
        )
