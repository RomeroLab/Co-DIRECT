
from __future__ import annotations

from dive.codirect_paths import joined

import argparse
import base64
import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import dis
import gc
from enum import Enum
from functools import lru_cache
import grp
import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tarfile
from types import CodeType, FrameType, FunctionType, MappingProxyType, ModuleType
from typing import Any

from dive.evaluation.directional_confirmation import (
    DirectionalConfirmationError,
    EMERGENT_UPSTREAM_COMMIT,
    FAMILIES,
    HARD_ROUTES,
    STRICT_FLOW_PARENT_COUNTS,
    STRICT_FLOW_ROW_COUNTS,
    METRIC_SOURCE_PATHS,
    _equal_family_route_score,
    _validate_preblind,
    blind_verdict as _ORIGINAL_BLIND_VERDICT,
    metric_source_inventory,
)
from dive.signed_value.roots import (
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_ROOT,
    RUNTIME_PYTHON,
)

class SealError(RuntimeError):
    pass

FROZEN_GENERATION_SEEDS = (42001, 42002)
FROZEN_GENERATION_ARMS = ("adaptive", "best_frozen_fixed_control")
FROZEN_FLOW_ARMS = (
    "adaptive",
    "none",
    "z_to_x",
    "x_to_z",
    "bidirectional",
)

SEAL_SCHEMA_VERSION = 2

REQUIRED_SEAL_KEYS = (
    "schema_version",
    "execution_authentication",
    "repositories",
    "preblind",
    "file_bindings",
    "blind_inventory",
    "evaluation",
    "thresholds",
    "blind_seal_payload",
    "output_manifest",
)

EVIDENCE_GROUP = "romerolab"

EXECUTION_AUTHENTICATION_SCHEMA_VERSION = 1
WORKER_PROTOCOL_VERSION = 1
_EXECUTION_AUTHENTICATION_KEYS = frozenset(
    {
        "schema_version",
        "bootstrap",
        "git_object_reader",
        "source_catalog",
        "worker_protocol_version",
        "external_runtime_inventory",
    }
)
_IDENTITY_ONLY_KEYS = frozenset({"path", "size_bytes", "sha256"})
_READER_KEYS = _IDENTITY_ONLY_KEYS | frozenset({"version"})
_CATALOG_BINDING_KEYS = frozenset(
    {"schema_version", "dive_commit", "upstream_commit", "identity_sha256"}
)

EXTERNAL_RUNTIME_LEAVES = (
    "cuequivariance_torch",
    "einops",
    "gzip",
    "math",
    "openfold",
    "os",
    "platform",
    "random",
    "torch",
    "torch.nn",
    "torch.nn.functional",
)
_EXTERNAL_RUNTIME_INVENTORY_SCHEMA_VERSION = 1
_EXTERNAL_RUNTIME_INVENTORY_KEYS = frozenset(
    {"schema_version", "interpreter", "python_version", "modules"}
)
_EXTERNAL_RUNTIME_MODULE_KEYS = frozenset(
    {"package_root", "origin", "origin_sha256", "distribution", "version"}
)

MUTABLE_SOURCE_REALM_EXCLUSIONS: Mapping[str, type] = MappingProxyType({})

GUARDED_MUTABLE_REALM_MODULES = (
    "dive.evaluation.directional_checks",
    "dive.evaluation.directional_confirmation",
    "dive.evaluation.directional_runtime",
    "dive.evaluation.directional_seal",
    "dive.integrations.complexa_training",
    "dive.integrations.directional_v2",
    "dive.integrations.flow",
    "dive.leadership.directional_bridge",
)

def _assert_realm_mutable_state_registered() -> None:

    for name in GUARDED_MUTABLE_REALM_MODULES:
        _assert_registered_mutable_globals(importlib.import_module(name))

_REQUIRED_FILE_BINDINGS = (
    "dive_dirty_report",
    "upstream_dirty_report",
    "preblind_record",
    "strict_blind_loader_manifest",
)

_FROZEN_METRICS = ("normalized_flow_loss", "oracle_gap_recovery")
_DIVE_READ_ROOT = Path(__file__).resolve().parents[3]
_BULK_READ_ROOT = Path(joined('CODIRECT_DATA_ROOT'))
_AUTOENCODER_CHECKPOINT_SHA256 = (
    "35f8865efd269995eeaf1670e1c1085acfe2988c40abdeda8e09a0e15eb40816"
)
_AUTOENCODER_CHECKPOINT_SIZE_BYTES = 4_100_101_779

_BLIND_THRESHOLD_VALUES: Mapping[str, object] = MappingProxyType(
    {
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
    }
)

_BLIND_SEAL_KEYS = frozenset(
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

@dataclass(frozen=True, slots=True)
class ConfirmationSeal:

    path: Path
    record_sha256: str
    blind_seal_hash: str
    strict_inventory_hash: str
    generation_inventory_hash: str
    availability_manifest_hash: str
    record: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class ConfirmationResult:

    seal_sha256: str
    append_offset: int
    append_length: int
    append_sha256: str
    record: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class _TestOnlyEnvironment:

    evidence_root: Path
    expected_gid: int
    dive_repo_root: Path
    upstream_repo_root: Path
    repository_states: Mapping[str, tuple[str, bool]]
    bypass_selection_artifact_io: bool = True

@dataclass(frozen=True, slots=True)
class _SealEnvironment:
    evidence_root: Path
    expected_gid: int
    dive_repo_root: Path
    upstream_repo_root: Path
    repository_states: Mapping[str, tuple[str, bool]] | None
    bypass_selection_artifact_io: bool

    git_reader_path: str | None = None

    in_process_workers: bool = False

    dive_read_root: Path | None = None

    @property
    def metric_read_root(self) -> Path:
        return self.dive_read_root or _DIVE_READ_ROOT

@dataclass(frozen=True, slots=True)
class VerifiedBlindInputs:

    seal_sha256: str
    snapshots: Mapping[str, bytes]
    artifacts: Mapping[str, "VerifiedArtifact"]

    dive_read_root: Path | None = None

@dataclass(frozen=True, slots=True)
class VerifiedArtifact:

    name: str
    descriptor: int
    sha256: str
    size_bytes: int
    device: int
    inode: int

@dataclass(slots=True)
class _AppendState:

    phase: str = "before_write"
    offset: int | None = None
    length: int | None = None
    sha256: str | None = None
    bytes_written: int = 0
    durable: bool = False

def _append_can_publish_failure(state: _AppendState | None) -> bool:

    return state is None or state.bytes_written == 0

@dataclass(frozen=True, slots=True)
class GenerationCell:

    family: str
    parent_id: str
    seed: int
    arm: str
    cell_hash: str
    request: Mapping[str, object]

@dataclass(frozen=True, slots=True)
class FlowIdentity:

    family: str
    example_id: str
    parent_id: str
    cell_hash: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "family": self.family,
            "example_id": self.example_id,
            "parent_id": self.parent_id,
            "cell_hash": self.cell_hash,
        }

@dataclass(frozen=True, slots=True)
class RawFlowRow:

    family: str
    example_id: str
    parent_id: str
    cell_hash: str
    request: Mapping[str, object]

    @property
    def identity(self) -> FlowIdentity:
        return FlowIdentity(
            self.family, self.example_id, self.parent_id, self.cell_hash
        )

FlowRow = RawFlowRow

_OUTCOME_FIELD_NAMES = frozenset(
    {
        "aggregate",
        "blind_checks",
        "counterfactual_validity",
        "flow_examples",
        "flow_loss",
        "joint_preservation",
        "loss",
        "metric",
        "outcome",
        "passed",
        "performance",
        "prediction",
        "result",
        "route_losses",
        "score",
        "statistics",
        "verdict",
    }
)

@dataclass(frozen=True, slots=True)
class BlindEvaluationData:

    flow_rows: tuple[FlowRow, ...]
    generation_cells: tuple[GenerationCell, ...]

def _environment() -> _SealEnvironment:
    return _SealEnvironment(
        EMERGENT_EVIDENCE_ROOT,
        grp.getgrnam(EVIDENCE_GROUP).gr_gid,
        Path(__file__).resolve().parents[3],
        EMERGENT_UPSTREAM_ROOT,
        None,
        False,
    )

def _test_only_seal_environment(value: _TestOnlyEnvironment) -> _SealEnvironment:
    if not isinstance(value, _TestOnlyEnvironment):
        raise SealError("private test environment has the wrong type")
    return _SealEnvironment(
        value.evidence_root,
        value.expected_gid,
        value.dive_repo_root,
        value.upstream_repo_root,
        value.repository_states,
        value.bypass_selection_artifact_io,
        None,
        True,
        _DIVE_READ_ROOT,
    )

def _canonical_bytes(value: object) -> bytes:
    def json_shape(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): json_shape(nested) for key, nested in item.items()}
        if isinstance(item, tuple):
            return [json_shape(nested) for nested in item]
        if isinstance(item, list):
            return [json_shape(nested) for nested in item]
        return item

    try:
        return json.dumps(
            json_shape(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SealError(f"seal is not finite canonical JSON: {error}") from error

def canonical_record_hash(value: object) -> str:

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()

def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(nested) for nested in value)
    return value

def _require_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SealError(f"{name} must be a mapping")
    return value

def _require_exact_keys(
    name: str, value: Mapping[str, object], expected: Iterable[str]
) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    unknown = sorted(set(value) - expected_set)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise SealError(f"{name} has {' and '.join(detail)}")

def _require_hash(name: str, value: object, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SealError(f"{name} must be a lowercase {length}-character hex hash")
    return value

def _absolute_path(name: str, value: object) -> str:

    if type(value) is not str or not value:
        raise SealError(f"{name} must be an exact string path")
    path = Path(value)
    if not path.is_absolute() or value != os.path.normpath(value) or value == "/":
        raise SealError(f"{name} must be a normalized absolute path")
    return value

def _exact_int(name: str, value: object) -> int:

    if type(value) is not int or value < 0:
        raise SealError(f"{name} must be an exact non-negative integer")
    return value

def _validate_execution_authentication(value: object) -> Mapping[str, object]:

    record = _require_mapping("execution_authentication", value)
    _require_exact_keys(
        "execution_authentication", record, _EXECUTION_AUTHENTICATION_KEYS
    )
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != EXECUTION_AUTHENTICATION_SCHEMA_VERSION
    ):
        raise SealError("execution_authentication.schema_version must be exactly 1")
    if (
        type(record["worker_protocol_version"]) is not int
        or record["worker_protocol_version"] != WORKER_PROTOCOL_VERSION
    ):
        raise SealError("execution_authentication.worker_protocol_version must be 1")
    for name, keys in (
        ("bootstrap", _IDENTITY_ONLY_KEYS),
        ("git_object_reader", _READER_KEYS),
        ("external_runtime_inventory", _IDENTITY_ONLY_KEYS),
    ):
        entry = _require_mapping(f"execution_authentication.{name}", record[name])
        _require_exact_keys(f"execution_authentication.{name}", entry, keys)
        _absolute_path(f"execution_authentication.{name}.path", entry["path"])
        _exact_int(f"execution_authentication.{name}.size_bytes", entry["size_bytes"])
        _require_hash(f"execution_authentication.{name}.sha256", entry["sha256"])
        if name == "git_object_reader":
            version = entry["version"]
            if type(version) is not str or not version.startswith("git version "):
                raise SealError(
                    "execution_authentication.git_object_reader.version is not exact"
                )
    catalog = _require_mapping(
        "execution_authentication.source_catalog", record["source_catalog"]
    )
    _require_exact_keys(
        "execution_authentication.source_catalog", catalog, _CATALOG_BINDING_KEYS
    )
    if type(catalog["schema_version"]) is not int or catalog["schema_version"] != 1:
        raise SealError(
            "execution_authentication.source_catalog.schema_version must be exactly 1"
        )
    if catalog["upstream_commit"] != EMERGENT_UPSTREAM_COMMIT:
        raise SealError(
            "execution_authentication.source_catalog.upstream_commit differs from "
            "the emergent pin"
        )
    _require_hash(
        "execution_authentication.source_catalog.dive_commit",
        catalog["dive_commit"],
        length=40,
    )
    _require_hash(
        "execution_authentication.source_catalog.identity_sha256",
        catalog["identity_sha256"],
    )
    return MappingProxyType(dict(record))

def _pinned_file_bytes(binding: Mapping[str, object], *, name: str) -> bytes:

    path = Path(_absolute_path(f"{name}.path", binding["path"]))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SealError(f"{name} cannot be opened safely: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SealError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != metadata.st_size:
        raise SealError(f"{name} changed size while being read")
    if len(raw) != binding["size_bytes"]:
        raise SealError(f"{name} size differs from the sealed binding")
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise SealError(f"{name} digest differs from the sealed binding")
    return raw

WORKER_ROLE_VARIABLE = "DIVE_WORKER_ROLE"

def _assert_authenticated_request_roots(
    request: Mapping[str, object], authentication: Mapping[str, object]
) -> None:

    if request["in_process_workers"] is not False:
        raise SealError("a production confirmation never bypasses artifact identity")
    if request["repository_states"] is not None:
        raise SealError("a worker never takes the parent's repository state")
    bootstrap = _require_mapping(
        "execution_authentication.bootstrap", authentication["bootstrap"]
    )
    expected_dive_root = Path(str(bootstrap["path"])).resolve().parents[2]
    expected = {
        "evidence_root": str(EMERGENT_EVIDENCE_ROOT),
        "upstream_repo_root": str(EMERGENT_UPSTREAM_ROOT),
        "dive_repo_root": str(expected_dive_root),
        "dive_read_root": str(expected_dive_root),
    }
    for name, value in expected.items():
        if str(request[name]) != value:
            raise SealError(
                f"confirmation request {name} differs from the authenticated root"
            )
    try:
        expected_gid = grp.getgrnam(EVIDENCE_GROUP).gr_gid
    except KeyError as error:
        raise SealError(f"evidence group {EVIDENCE_GROUP} does not exist") from error
    if request["expected_gid"] != expected_gid:
        raise SealError("confirmation request expected_gid is not the evidence group")

def _verify_sealed_git_reader(record: Mapping[str, object]) -> str:

    reader = _require_mapping(
        "execution_authentication.git_object_reader", record["git_object_reader"]
    )
    _pinned_file_bytes(reader, name="git object reader")
    path = str(reader["path"])
    try:
        observed = subprocess.run(
            (path, "--version"), capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise SealError(f"cannot execute the sealed git object reader: {error}") from (
            error
        )
    if observed.returncode != 0 or observed.stdout.strip() != reader["version"]:
        raise SealError("git object reader version differs from the sealed binding")
    return path

def _verify_execution_authentication_files(
    record: Mapping[str, object],
) -> Mapping[str, bytes]:

    bootstrap = _require_mapping(
        "execution_authentication.bootstrap", record["bootstrap"]
    )
    reader = _require_mapping(
        "execution_authentication.git_object_reader", record["git_object_reader"]
    )
    del reader
    anchor = _pinned_file_bytes(bootstrap, name="directional bootstrap anchor")
    _verify_sealed_git_reader(record)
    return MappingProxyType({"bootstrap": anchor})

def _module_origin_identity(module: ModuleType) -> tuple[str, str, str]:
    origin = getattr(module, "__file__", None)
    if type(origin) is not str or not origin:
        raise SealError(
            f"external runtime inventory module {module.__name__} has no file origin"
        )
    resolved = Path(origin).resolve()
    package_path = getattr(module, "__path__", None)
    root = resolved.parent if package_path is not None else resolved
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise SealError(
            f"external runtime inventory module {module.__name__} is unreadable"
        ) from error
    return str(root), str(resolved), hashlib.sha256(raw).hexdigest()

def _distribution_identity(name: str) -> tuple[str | None, str | None]:

    import importlib.metadata

    top_level = name.split(".", 1)[0]
    try:
        distribution = importlib.metadata.distribution(top_level)
    except importlib.metadata.PackageNotFoundError:
        return None, None
    metadata_name = distribution.metadata["Name"]
    if type(metadata_name) is not str or not metadata_name:
        return None, None
    return metadata_name, str(distribution.version)

def build_external_runtime_inventory() -> dict[str, object]:

    modules: dict[str, object] = {}
    for name in EXTERNAL_RUNTIME_LEAVES:
        module = importlib.import_module(name)
        package_root, origin, origin_sha256 = _module_origin_identity(module)
        distribution, version = _distribution_identity(name)
        modules[name] = {
            "package_root": package_root,
            "origin": origin,
            "origin_sha256": origin_sha256,
            "distribution": distribution,
            "version": version,
        }
    return {
        "schema_version": _EXTERNAL_RUNTIME_INVENTORY_SCHEMA_VERSION,
        "interpreter": str(RUNTIME_PYTHON),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "modules": modules,
    }

def _verify_external_runtime_inventory(
    binding: Mapping[str, object], environment: _SealEnvironment
) -> Mapping[str, object]:

    path = Path(_absolute_path("external_runtime_inventory.path", binding["path"]))
    raw, _ = _safe_file_bytes(
        path, name="external runtime inventory", environment=environment
    )
    if len(raw) != binding["size_bytes"]:
        raise SealError("external runtime inventory size differs from the seal")
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise SealError("external runtime inventory digest differs from the seal")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("external runtime inventory is not canonical JSON") from error
    if raw != _canonical_bytes(record) + b"\n":
        raise SealError("external runtime inventory is not canonical JSON")
    record = _require_mapping("external runtime inventory", record)
    try:
        _require_exact_keys(
            "external runtime inventory", record, _EXTERNAL_RUNTIME_INVENTORY_KEYS
        )
    except SealError as error:
        raise SealError(f"external runtime inventory schema is not exact: {error}")
    if record["schema_version"] != _EXTERNAL_RUNTIME_INVENTORY_SCHEMA_VERSION:
        raise SealError("external runtime inventory schema_version must be exactly 1")
    if record["interpreter"] != str(RUNTIME_PYTHON):
        raise SealError("external runtime inventory names another interpreter")
    live_version = ".".join(str(part) for part in sys.version_info[:3])
    if record["python_version"] != live_version:
        raise SealError("external runtime inventory names another Python version")
    modules = _require_mapping("external runtime inventory modules", record["modules"])
    if tuple(sorted(modules)) != tuple(sorted(EXTERNAL_RUNTIME_LEAVES)):
        raise SealError(
            "external runtime inventory does not name exactly the admitted leaves"
        )
    for name in EXTERNAL_RUNTIME_LEAVES:
        entry = _require_mapping(f"external runtime inventory {name}", modules[name])
        try:
            _require_exact_keys(
                f"external runtime inventory {name}",
                entry,
                _EXTERNAL_RUNTIME_MODULE_KEYS,
            )
        except SealError as error:
            raise SealError(f"external runtime inventory entry is not exact: {error}")
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            raise SealError(
                f"external runtime inventory names unimportable module {name}"
            ) from error
        if module.__name__ != name:
            raise SealError(f"external runtime inventory module {name} is aliased")
        package_root, origin, origin_sha256 = _module_origin_identity(module)
        if (
            entry["package_root"] != package_root
            or entry["origin"] != origin
            or entry["origin_sha256"] != origin_sha256
        ):
            raise SealError(
                f"external runtime inventory package root for {name} is not what is "
                "importable"
            )
        distribution, version = _distribution_identity(name)
        if entry["distribution"] != distribution or entry["version"] != version:
            raise SealError(
                f"external runtime inventory distribution/version for {name} is not "
                "what is importable"
            )
    return MappingProxyType(record)

def _capture_repository(
    name: str, root: Path, environment: _SealEnvironment
) -> tuple[str, bool]:
    try:
        metadata = os.lstat(root)
    except OSError as error:
        raise SealError(f"{name} repository root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise SealError(f"{name} repository root must be a real directory")
    if environment.repository_states is not None:
        try:
            commit, clean = environment.repository_states[name]
        except KeyError as error:
            raise SealError(f"test repository state is missing {name}") from error
        _require_hash(f"{name} repository commit", commit, length=40)
        if type(clean) is not bool:
            raise SealError(f"{name} repository clean state must be boolean")
        return commit, clean
    reader = environment.git_reader_path or "git"
    commands = (
        (reader, "rev-parse", "HEAD"),
        (reader, "status", "--porcelain=v1", "--untracked-files=normal"),
    )
    results = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise SealError(f"cannot capture {name} repository state")
        results.append(result.stdout)
    commit = results[0].strip()
    _require_hash(f"{name} repository commit", commit, length=40)
    return commit, results[1] == ""

def _records(value: object) -> list[Mapping[str, object]]:
    if hasattr(value, "to_dict"):
        value = value.to_dict(orient="records")
    if isinstance(value, (str, bytes, Mapping)):
        raise SealError("blind inventory input must be an iterable of records")
    try:
        records = list(value)
    except TypeError as error:
        raise SealError(
            "blind inventory input must be an iterable of records"
        ) from error
    if any(not isinstance(record, Mapping) for record in records):
        raise SealError("blind inventory input contains a non-record value")
    return records

def build_blind_inventory(
    frame: object,
    *,
    deterministic_exclusions: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:

    strict: list[dict[str, str]] = []
    seen_examples: set[tuple[str, str]] = set()
    for index, raw in enumerate(_records(frame)):
        if raw.get("partition") != "test-blind" or raw.get("view_strict") is not True:
            continue
        family = raw.get("family")
        example_id = raw.get("example_id")
        parent_id = raw.get("parent_id")
        if family not in FAMILIES:
            raise SealError(f"blind row {index} has unknown family")
        if not isinstance(example_id, str) or not example_id:
            raise SealError(f"blind row {index} has empty example_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise SealError(f"blind row {index} has empty parent_id")
        identity_key = (family, example_id)
        if identity_key in seen_examples:
            raise SealError(
                f"blind inventory contains duplicate example_id {example_id}"
            )
        seen_examples.add(identity_key)
        identity = {
            "family": family,
            "example_id": example_id,
            "parent_id": parent_id,
        }
        strict.append({**identity, "cell_hash": canonical_record_hash(identity)})

    strict.sort(key=lambda row: (row["family"], row["example_id"]))
    row_counts = {
        family: sum(row["family"] == family for row in strict) for family in FAMILIES
    }
    if row_counts != dict(STRICT_FLOW_ROW_COUNTS):
        raise SealError(
            "strict row count must be exactly binder=198, ame=292, antibody=1106"
        )
    parent_counts = {
        family: len({row["parent_id"] for row in strict if row["family"] == family})
        for family in FAMILIES
    }
    if parent_counts != dict(STRICT_FLOW_PARENT_COUNTS):
        raise SealError(
            "strict parent count must be exactly binder=131, ame=120, antibody=454"
        )

    strict_flow_inventory = {
        "schema_version": 1,
        "rows": strict,
        "rows_hash": canonical_record_hash({"rows": strict}),
    }
    parents_by_family: dict[str, list[str]] = {}
    for family in FAMILIES:
        unique = {row["parent_id"] for row in strict if row["family"] == family}
        parents_by_family[family] = sorted(
            unique,
            key=lambda parent: (
                hashlib.sha256(f"{family}{parent}".encode()).hexdigest(),
                parent,
            ),
        )[:20]
        if len(parents_by_family[family]) != 20:
            raise SealError(f"{family} must select exactly 20 generation parents")

    generation_cells: list[dict[str, object]] = []
    for family in FAMILIES:
        for parent_id in parents_by_family[family]:
            for seed in FROZEN_GENERATION_SEEDS:
                for arm in FROZEN_GENERATION_ARMS:
                    identity = {
                        "family": family,
                        "parent_id": parent_id,
                        "seed": seed,
                        "arm": arm,
                    }
                    generation_cells.append(
                        {**identity, "cell_hash": canonical_record_hash(identity)}
                    )
    if len(generation_cells) != 240:
        raise SealError("generation inventory must contain exactly 240 planned cells")
    if not isinstance(deterministic_exclusions, Sequence) or isinstance(
        deterministic_exclusions, (str, bytes)
    ):
        raise SealError("deterministic_exclusions must be a sequence")
    known_cells = {str(cell["cell_hash"]): cell for cell in generation_cells}
    exclusions: list[dict[str, str]] = []
    seen_exclusions: set[str] = set()
    for index, raw in enumerate(deterministic_exclusions):
        record = _require_mapping(f"deterministic_exclusions[{index}]", raw)
        _require_exact_keys(
            f"deterministic_exclusions[{index}]", record, ("cell_hash", "reason")
        )
        cell_hash = _require_hash(
            f"deterministic_exclusions[{index}].cell_hash", record["cell_hash"]
        )
        reason = record["reason"]
        if not isinstance(reason, str) or not reason:
            raise SealError(f"deterministic_exclusions[{index}].reason is empty")
        if cell_hash not in known_cells or cell_hash in seen_exclusions:
            raise SealError("deterministic exclusion is duplicate or not preregistered")
        seen_exclusions.add(cell_hash)
        exclusions.append({"cell_hash": cell_hash, "reason": reason})
    excluded_by_parent: dict[tuple[str, str], int] = {}
    for cell_hash in seen_exclusions:
        cell = known_cells[cell_hash]
        key = (str(cell["family"]), str(cell["parent_id"]))
        excluded_by_parent[key] = excluded_by_parent.get(key, 0) + 1
    if any(count != 4 for count in excluded_by_parent.values()):
        raise SealError("deterministic exclusions must remove complete paired parents")
    availability_manifest = {
        "schema_version": 1,
        "deterministic_exclusions": exclusions,
    }
    generation_inventory = {
        "schema_version": 1,
        "parents_by_family": parents_by_family,
        "parent_inventory_hash": canonical_record_hash(
            {"parents_by_family": parents_by_family}
        ),
        "seeds": list(FROZEN_GENERATION_SEEDS),
        "arms": list(FROZEN_GENERATION_ARMS),
        "preregistered_cells": generation_cells,
        "preregistered_cells_hash": canonical_record_hash({"cells": generation_cells}),
        "availability_manifest_hash": canonical_record_hash(availability_manifest),
    }
    return {
        "schema_version": 1,
        "strict_row_counts": row_counts,
        "strict_parent_counts": parent_counts,
        "strict_flow_inventory": strict_flow_inventory,
        "strict_inventory_hash": canonical_record_hash(strict_flow_inventory),
        "generation_parent_ids": parents_by_family,
        "generation_seeds": list(FROZEN_GENERATION_SEEDS),
        "generation_arms": list(FROZEN_GENERATION_ARMS),
        "generation_cells": generation_cells,
        "planned_generation_cells": 240,
        "generation_inventory": generation_inventory,
        "generation_inventory_hash": canonical_record_hash(generation_inventory),
        "availability_manifest": availability_manifest,
        "availability_manifest_hash": canonical_record_hash(availability_manifest),
    }

def _open_terminal_parent(
    path: Path, environment: _SealEnvironment, *, create: bool
) -> tuple[int, str]:
    root = environment.evidence_root.absolute()
    destination = path.absolute()
    try:
        relative = destination.relative_to(root)
    except ValueError as error:
        raise SealError(
            f"terminal path must stay under {root}: {destination}"
        ) from error
    if destination == root or any(part in {"", ".", ".."} for part in relative.parts):
        raise SealError("terminal path schema is unsafe")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        current = os.open(root, directory_flags)
    except OSError as error:
        raise SealError("fixed terminal evidence root is unavailable") from error
    try:
        root_stat = os.fstat(current)
        if (
            root_stat.st_gid != environment.expected_gid
            or not root_stat.st_mode & stat.S_ISGID
            or not root_stat.st_mode & stat.S_IWGRP
        ):
            raise SealError(
                "terminal evidence root lacks expected gid/setgid/group-write"
            )
        for part in relative.parent.parts:
            try:
                following = os.open(part, directory_flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o2770, dir_fd=current)
                following = os.open(part, directory_flags, dir_fd=current)
                os.fchmod(following, 0o2770)
                os.fsync(current)
            metadata = os.fstat(following)
            if (
                metadata.st_gid != environment.expected_gid
                or not metadata.st_mode & stat.S_ISGID
                or not metadata.st_mode & stat.S_IWGRP
            ):
                os.close(following)
                raise SealError(
                    f"terminal parent {part!r} lacks expected gid/setgid/group-write"
                )
            os.close(current)
            current = following
        return current, relative.name
    except BaseException:
        os.close(current)
        raise

def _validate_terminal_destination(path: Path, environment: _SealEnvironment) -> None:
    root = environment.evidence_root.absolute()
    destination = path.absolute()
    try:
        relative = destination.relative_to(root)
    except ValueError as error:
        raise SealError(
            f"terminal path must stay under {root}: {destination}"
        ) from error
    if destination == root or any(part in {"", ".", ".."} for part in relative.parts):
        raise SealError("terminal path schema is unsafe")

def _safe_file_bytes(
    path: Path, *, name: str, environment: _SealEnvironment
) -> tuple[bytes, os.stat_result]:
    if not path.is_absolute() or path == Path("/"):
        raise SealError(f"{name}.path must be a normalized absolute path")
    if path != Path(os.path.normpath(path)):
        raise SealError(f"{name}.path must be a normalized absolute path")
    parent_fd, leaf = _open_terminal_parent(path, environment, create=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise SealError(f"{name} cannot be opened safely: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_gid != environment.expected_gid
            or metadata.st_mode & 0o660 != 0o660
            or metadata.st_mode & 0o002
        ):
            raise SealError(f"{name} must be a group-writable protected regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), metadata
    finally:
        os.close(descriptor)
        os.close(parent_fd)

def _read_regular_under_root(
    path: Path, root: Path, *, name: str
) -> tuple[bytes, os.stat_result]:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise SealError(f"{name}.path must be a normalized absolute path")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise SealError(f"{name} is outside its fixed read-only root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SealError(f"{name}.path is unsafe")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open(root, directory_flags)
    try:
        for part in relative.parent.parts:
            following = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = following
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(relative.name, flags, dir_fd=current)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise SealError(f"{name} must be a regular file")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks), metadata
        finally:
            os.close(descriptor)
    except OSError as error:
        raise SealError(f"{name} cannot be opened component-safely: {error}") from error
    finally:
        os.close(current)

def _metric_source_identity(read_root: Path) -> dict[str, object]:

    if not isinstance(read_root, Path) or not read_root.is_absolute():
        raise SealError("metric source read root must be an absolute path")
    identities: list[dict[str, object]] = []
    verified_files: list[dict[str, object]] = []
    for relative_name in METRIC_SOURCE_PATHS:
        path = (read_root / relative_name).resolve()
        raw, metadata = _read_regular_under_root(
            path, read_root, name=f"metric source {relative_name}"
        )
        source = {
            "path": relative_name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        identities.append(source)
        verified_files.append(
            {
                **source,
                "absolute_path": str(path),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
            }
        )
    try:
        inventory = metric_source_inventory(identities)
    except DirectionalConfirmationError as error:
        raise SealError(f"metric source inventory is invalid: {error}") from error
    return {
        "schema_version": inventory["schema_version"],
        "check_schema_version": inventory["check_schema_version"],
        "sources": [dict(source) for source in inventory["sources"]],
        "inventory_hash": inventory["inventory_hash"],
        "verified_files": verified_files,
    }

def _verify_metric_source_identity(
    value: object, read_root: Path
) -> Mapping[str, bytes]:
    identity = _require_mapping("metric source", value)
    _require_exact_keys(
        "metric source",
        identity,
        (
            "schema_version",
            "check_schema_version",
            "sources",
            "inventory_hash",
            "verified_files",
        ),
    )
    observed = _metric_source_identity(read_root)
    if _canonical_bytes(identity) != _canonical_bytes(observed):
        raise SealError("metric source identity changed")
    snapshots: dict[str, bytes] = {}
    for relative_name in METRIC_SOURCE_PATHS:
        raw, _ = _read_regular_under_root(
            (read_root / relative_name).resolve(),
            read_root,
            name=f"metric source {relative_name}",
        )
        snapshots[relative_name] = raw
    return MappingProxyType(snapshots)

def _metric_snapshot_module(
    *,
    logical_name: str,
    snapshot: bytes,
    import_overrides: Mapping[str, ModuleType],
) -> ModuleType:
    unique_name = f"_dive_authenticated_{logical_name.rsplit('.', 1)[-1]}"
    module = ModuleType(unique_name)
    module.__file__ = (
        str(_DIVE_READ_ROOT / "src" / Path(*logical_name.split("."))) + ".py"
    )
    module.__package__ = "dive.evaluation"
    original_import = builtins.__import__

    def authenticated_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 0 and name in import_overrides:
            return import_overrides[name]
        return original_import(name, globals, locals, fromlist, level)

    controlled_builtins = dict(vars(builtins))
    controlled_builtins["__import__"] = authenticated_import
    module.__dict__["__builtins__"] = controlled_builtins
    previous = sys.modules.get(unique_name)
    sys.modules[unique_name] = module
    try:
        code = compile(snapshot, module.__file__, "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException as error:
        raise SealError(
            f"authenticated execution snapshot {logical_name} cannot be compiled"
        ) from error
    finally:
        if previous is None:
            sys.modules.pop(unique_name, None)
        else:
            sys.modules[unique_name] = previous
    return module

def _execution_code_names(code: CodeType) -> frozenset[str]:
    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and type(instruction.argval) is str
    }
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(_execution_code_names(constant))
    return frozenset(names)

_NO_MUTABLE_EXCLUSIONS: Mapping[str, type] = MappingProxyType({})
_RUNTIME_MODULE_NAME = "dive.evaluation.directional_runtime"

def _mutable_source_realm_exclusions() -> Mapping[str, type]:

    module = sys.modules.get(_RUNTIME_MODULE_NAME)
    exclusions = getattr(module, "MUTABLE_SOURCE_REALM_EXCLUSIONS", None)
    if type(exclusions) is not MappingProxyType:
        return _NO_MUTABLE_EXCLUSIONS
    return exclusions

def _is_registered_mutable_global(owner_name: object, name: str) -> bool:

    if type(owner_name) is not str:
        return False
    return f"{owner_name}.{name}" in _mutable_source_realm_exclusions()

def _is_evaluator_owned_mutable(value: object) -> bool:

    value_type = type(value)
    for qualified_name, expected_type in _mutable_source_realm_exclusions().items():
        if value_type is not expected_type or type(qualified_name) is not str:
            continue
        module_name, separator, attribute = qualified_name.rpartition(".")
        if not separator:
            continue
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, attribute, None) is value:
            return True
    return False

def _mapping_proxy_backing_is_sole_owned(proxy: object) -> bool:

    referents = gc.get_referents(proxy)
    if len(referents) != 1 or type(referents[0]) is not dict:
        return False
    backing = referents[0]
    for referrer in gc.get_referrers(backing):

        if referrer is proxy or referrer is referents or referrer is backing:
            continue
        if isinstance(referrer, FrameType):
            continue
        return False
    return True

def _signature_recurses_into(value: object) -> bool:

    value_type = type(value)
    if value is None or value is Ellipsis:
        return False
    if value_type in (bool, int, float, complex, str, bytes):
        return False
    if isinstance(value, (Path, FunctionType, type, ModuleType)):
        return False
    return isinstance(value, (Mapping, Sequence, set, frozenset))

def _is_immutable_signature_value(value: object, active: set[int]) -> bool:

    if not _signature_recurses_into(value):

        return True
    identity = id(value)
    if identity in active:
        return True
    if isinstance(value, Enum):

        return _is_immutable_signature_value(value.value, active)
    value_type = type(value)
    active.add(identity)
    try:
        if value_type is tuple or value_type is frozenset:
            return all(_is_immutable_signature_value(n, active) for n in value)
        if value_type is MappingProxyType and _mapping_proxy_backing_is_sole_owned(
            value
        ):
            return all(
                _is_immutable_signature_value(key, active)
                and _is_immutable_signature_value(nested, active)
                for key, nested in value.items()
            )
        return False
    finally:
        active.discard(identity)

def _resolve_exclusion_target(qualified_name: str) -> tuple[object, str] | None:

    parts = qualified_name.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module = sys.modules.get(".".join(parts[:split]))
        if not isinstance(module, ModuleType):
            continue
        owner: object = module
        for step in parts[split:-1]:
            owner = getattr(owner, step, None)
            if owner is None:
                return None
        return owner, parts[-1]
    return None

def _validate_mutable_exclusions(exclusions: Mapping[str, type]) -> None:

    for qualified_name, expected_type in exclusions.items():
        if type(qualified_name) is not str or "." not in qualified_name:
            raise SealError(f"mutable exclusion {qualified_name!r} is not qualified")
        if type(expected_type) is not type:
            raise SealError(f"mutable exclusion {qualified_name!r} names a non-type")
        resolved = _resolve_exclusion_target(qualified_name)
        if resolved is None:
            raise SealError(f"mutable exclusion module for {qualified_name!r} is dead")
        owner, attribute = resolved
        if not hasattr(owner, attribute):
            raise SealError(f"mutable exclusion {qualified_name!r} does not exist")
        if type(getattr(owner, attribute)) is not expected_type:
            raise SealError(f"mutable exclusion {qualified_name!r} has the wrong type")

def _assert_registered_mutable_value(
    qualified_name: str,
    description: str,
    value: object,
    exclusions: Mapping[str, type],
) -> None:

    if not _signature_recurses_into(value):
        return
    if _is_immutable_signature_value(value, set()):
        return
    if qualified_name in exclusions:
        return
    raise SealError(
        f"unregistered mutable {description} is reachable from scientific execution"
    )

_PROJECT_SOURCE_ORIGIN_PREFIXES = ("<catalog:", "<retained:")

def _is_project_source_origin(origin: str) -> bool:

    if origin.startswith(_PROJECT_SOURCE_ORIGIN_PREFIXES):
        return True
    try:
        resolved = Path(origin).resolve()
    except (OSError, ValueError):
        return False
    return any(
        resolved.is_relative_to(root)
        for root in (
            _DIVE_READ_ROOT / "src" / "dive",
            EMERGENT_UPSTREAM_ROOT / "src" / "proteinfoundation",
        )
    )

def _content_bound_values(
    module: ModuleType, modules: Mapping[str, ModuleType] | None = None
) -> tuple[tuple[str, str, object, str | None], ...]:

    if modules is None:
        modules = {module.__name__: module}
    required_roots = frozenset(
        name
        for name, value in vars(module).items()
        if not name.startswith("__")
        and isinstance(value, (FunctionType, type))
        and getattr(value, "__module__", None) == module.__name__
    )
    collected: list[tuple[str, str, object, str | None]] = []
    _bounded_execution_graph_signature(
        modules,
        root_module=module.__name__,
        required_roots=required_roots,
        observer=lambda *reported: collected.append(reported),
    )
    return tuple(collected)

def _assert_registered_mutable_globals(
    module: ModuleType, modules: Mapping[str, ModuleType] | None = None
) -> None:

    exclusions = getattr(module, "MUTABLE_SOURCE_REALM_EXCLUSIONS", None)
    if type(exclusions) is not MappingProxyType:
        raise SealError(
            f"{module.__name__} has no immutable mutable-owner exclusion registry"
        )
    _validate_mutable_exclusions(exclusions)
    for name, value in vars(module).items():
        if name.startswith("__"):
            continue
        qualified_name = f"{module.__name__}.{name}"
        _assert_registered_mutable_value(
            qualified_name, f"global {qualified_name}", value, exclusions
        )
    for kind, qualified_name, value, origin in _content_bound_values(module, modules):
        if origin is not None and not _is_project_source_origin(origin):
            continue
        _assert_registered_mutable_value(
            qualified_name, f"{kind} {qualified_name}", value, exclusions
        )

def _execution_value_signature(
    value: object,
    logical_modules: Mapping[str, str],
    logical_objects: Mapping[int, tuple[str, str]] = MappingProxyType({}),
) -> tuple[object, ...]:
    value_type = type(value)
    if _is_evaluator_owned_mutable(value):

        return (
            "evaluator-owned-mutable-leaf",
            value_type.__module__,
            value_type.__qualname__,
        )
    if value is None or value is Ellipsis or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (float, value.hex())
    if value_type is complex:
        return (complex, value.real.hex(), value.imag.hex())
    if isinstance(value, Path):
        return ("path", value_type.__module__, value_type.__qualname__, str(value))
    if isinstance(value, Enum):
        logical = logical_modules.get(value_type.__module__)
        if logical is not None:
            return ("project-enum", logical, value_type.__qualname__, value.name)
    if value_type is object and id(value) in logical_objects:
        return ("project-object", *logical_objects[id(value)])
    if isinstance(value, FunctionType):
        logical = logical_modules.get(value.__module__)
        if logical is not None:
            return ("project-function", logical, value.__qualname__)
        return (
            "external-function-leaf",
            id(value),
            value.__module__,
            value.__qualname__,
        )
    if isinstance(value, type):
        logical = logical_modules.get(value.__module__)
        if logical is not None:
            return ("project-class", logical, value.__qualname__)
        return (
            "external-class-leaf",
            id(value),
            value.__module__,
            value.__qualname__,
        )
    if isinstance(value, ModuleType):
        logical = logical_modules.get(value.__name__)
        if logical is not None:
            return ("project-module", logical)
        return ("external-module-leaf", id(value), value.__name__)
    if isinstance(value, Mapping):
        entries = [
            (
                _execution_value_signature(key, logical_modules, logical_objects),
                _execution_value_signature(nested, logical_modules, logical_objects),
            )
            for key, nested in value.items()
        ]
        return ("mapping", value_type.__name__, tuple(sorted(entries, key=repr)))
    if isinstance(value, Sequence):
        return (
            "sequence",
            value_type.__name__,
            tuple(
                _execution_value_signature(nested, logical_modules, logical_objects)
                for nested in value
            ),
        )
    if isinstance(value, (set, frozenset)):
        entries = [
            _execution_value_signature(nested, logical_modules, logical_objects)
            for nested in value
        ]
        return ("set", value_type.__name__, tuple(sorted(entries, key=repr)))
    return (
        "external-identity-leaf",
        id(value),
        value_type.__module__,
        value_type.__qualname__,
    )

def _execution_function_signature(
    function: FunctionType,
    logical_modules: Mapping[str, str],
    logical_objects: Mapping[int, tuple[str, str]] = MappingProxyType({}),
) -> tuple[object, ...]:
    owner_name = function.__globals__.get("__name__")
    globals_signature = tuple(
        (
            name,
            _execution_value_signature(
                function.__globals__[name], logical_modules, logical_objects
            ),
        )
        for name in sorted(_execution_code_names(function.__code__))
        if name in function.__globals__
        and not _is_registered_mutable_global(owner_name, name)
    )
    closure = ()
    if function.__closure__ is not None:
        closure = tuple(
            (
                function.__code__.co_freevars[index],
                _execution_value_signature(
                    cell.cell_contents, logical_modules, logical_objects
                ),
            )
            for index, cell in enumerate(function.__closure__)
        )
    return (
        function.__code__,
        _execution_value_signature(
            function.__defaults__, logical_modules, logical_objects
        ),
        _execution_value_signature(
            function.__kwdefaults__, logical_modules, logical_objects
        ),
        _execution_value_signature(
            function.__annotations__, logical_modules, logical_objects
        ),
        globals_signature,
        closure,
    )

def _execution_class_signature(
    module_class: type,
    logical_modules: Mapping[str, str],
    logical_objects: Mapping[int, tuple[str, str]] = MappingProxyType({}),
) -> tuple[object, ...]:
    methods = []
    state = []
    for name, value in sorted(vars(module_class).items()):
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if isinstance(value, FunctionType):
            methods.append(
                (
                    name,
                    _execution_function_signature(
                        value, logical_modules, logical_objects
                    ),
                )
            )
        elif isinstance(value, property):
            methods.append(
                (
                    name,
                    tuple(
                        None
                        if method is None
                        else _execution_function_signature(
                            method, logical_modules, logical_objects
                        )
                        for method in (value.fget, value.fset, value.fdel)
                    ),
                )
            )
        elif (
            not name.startswith("__")
            and value is not None
            and type(value)
            in (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                tuple,
                list,
                dict,
                set,
                frozenset,
                MappingProxyType,
            )
        ):
            state.append(
                (
                    name,
                    _execution_value_signature(value, logical_modules, logical_objects),
                )
            )
    return tuple(methods), tuple(state)

def _metric_module_execution_signature(
    module: ModuleType, logical_modules: Mapping[str, str]
) -> tuple[object, ...]:
    logical_name = logical_modules.get(module.__name__)
    logical_objects = MappingProxyType(
        {
            id(value): (logical_name, name)
            for name, value in vars(module).items()
            if logical_name is not None and type(value) is object
        }
    )
    roots = []
    referenced_names: set[str] = set()
    for name, value in sorted(vars(module).items()):
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            roots.append(
                (
                    "function",
                    name,
                    _execution_function_signature(
                        value, logical_modules, logical_objects
                    ),
                )
            )
            referenced_names.update(_execution_code_names(value.__code__))
        elif isinstance(value, type) and value.__module__ == module.__name__:
            roots.append(
                (
                    "class",
                    name,
                    _execution_class_signature(value, logical_modules, logical_objects),
                )
            )
            for nested in vars(value).values():
                if isinstance(nested, (staticmethod, classmethod)):
                    nested = nested.__func__
                if isinstance(nested, FunctionType):
                    referenced_names.update(_execution_code_names(nested.__code__))
    bindings = tuple(
        (
            name,
            _execution_value_signature(
                vars(module)[name], logical_modules, logical_objects
            ),
        )
        for name in sorted(referenced_names)
        if name in vars(module)
        and not _is_registered_mutable_global(module.__name__, name)
    )
    return tuple(roots), bindings

def _authenticate_metric_execution(inputs: VerifiedBlindInputs) -> None:
    paths = {
        "dive.evaluation.directional_checks": (
            "src/dive/evaluation/directional_checks.py"
        ),
        "dive.evaluation.directional_confirmation": (
            "src/dive/evaluation/directional_confirmation.py"
        ),
    }
    try:
        checks_snapshot = inputs.snapshots[
            f"metric_source:{paths['dive.evaluation.directional_checks']}"
        ]
        confirmation_snapshot = inputs.snapshots[
            f"metric_source:{paths['dive.evaluation.directional_confirmation']}"
        ]
    except KeyError as error:
        raise SealError("authenticated execution source snapshot is missing") from error
    expected_checks = _metric_snapshot_module(
        logical_name="dive.evaluation.directional_checks",
        snapshot=checks_snapshot,
        import_overrides={},
    )
    expected_confirmation = _metric_snapshot_module(
        logical_name="dive.evaluation.directional_confirmation",
        snapshot=confirmation_snapshot,
        import_overrides={
            "dive.evaluation.directional_checks": expected_checks,
        },
    )
    live_modules = {name: sys.modules.get(name) for name in paths}
    if any(not isinstance(module, ModuleType) for module in live_modules.values()):
        raise SealError("authenticated metric execution module is missing")
    expected_modules = {
        "dive.evaluation.directional_checks": expected_checks,
        "dive.evaluation.directional_confirmation": expected_confirmation,
    }
    live_logical = {name: name for name in paths}
    expected_logical = {
        module.__name__: name for name, module in expected_modules.items()
    }
    for name in paths:
        live = live_modules[name]
        expected = expected_modules[name]
        assert isinstance(live, ModuleType)
        if _metric_module_execution_signature(live, live_logical) != (
            _metric_module_execution_signature(expected, expected_logical)
        ):
            raise SealError(f"authenticated metric execution graph for {name} changed")
    live_confirmation = live_modules["dive.evaluation.directional_confirmation"]
    assert isinstance(live_confirmation, ModuleType)
    if _ORIGINAL_BLIND_VERDICT is not vars(live_confirmation).get("blind_verdict"):
        raise SealError("authenticated blind verdict binding changed")

_CATALOG_SOURCE_ORIGIN_PREFIX = "<catalog:"
_CATALOG_ROOT_CLASSES = (
    ("dive", "src/dive/"),
    ("upstream", "src/proteinfoundation/"),
)

def _execution_source_snapshot_for_module(
    inputs: VerifiedBlindInputs, module: ModuleType
) -> tuple[bytes, Path]:

    source_name = getattr(module, "__file__", None)
    if type(source_name) is not str:
        raise SealError("reviewed execution module has no source file")
    if source_name.startswith(_CATALOG_SOURCE_ORIGIN_PREFIX) and source_name.endswith(
        ">"
    ):
        relative = source_name[len(_CATALOG_SOURCE_ORIGIN_PREFIX) : -1]
        for root_class, prefix in _CATALOG_ROOT_CLASSES:
            if not relative.startswith(prefix):
                continue
            key = f"execution_source:{root_class}:{relative}"
            try:
                return inputs.snapshots[key], Path(source_name)
            except KeyError as error:
                raise SealError(
                    "reviewed execution module escaped the authenticated source "
                    "catalog"
                ) from error
        raise SealError(
            "reviewed execution module is outside authenticated source roots"
        )
    source_path = Path(source_name).resolve()
    roots = (
        ("dive", inputs.dive_read_root or _DIVE_READ_ROOT),
        ("upstream", EMERGENT_UPSTREAM_ROOT),
    )
    for root_class, root in roots:
        try:
            relative = source_path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        key = f"execution_source:{root_class}:{relative}"
        try:
            return inputs.snapshots[key], source_path
        except KeyError as error:
            raise SealError(
                "reviewed execution module escaped the authenticated source catalog"
            ) from error
    raise SealError("reviewed execution module is outside authenticated source roots")

def _module_compiled_code_index(
    snapshot: bytes, source_path: Path
) -> Mapping[str, tuple[CodeType, ...]]:
    try:
        root = compile(snapshot, str(source_path), "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as error:
        raise SealError("reviewed execution source cannot be compiled") from error
    codes: dict[str, list[CodeType]] = {}

    def visit(code: CodeType) -> None:
        codes.setdefault(code.co_qualname, []).append(code)
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                visit(constant)

    visit(root)
    return MappingProxyType({name: tuple(values) for name, values in codes.items()})

def _module_execution_functions(module: ModuleType) -> tuple[FunctionType, ...]:
    functions: list[FunctionType] = []
    for value in vars(module).values():
        if isinstance(value, FunctionType) and value.__module__ == module.__name__:
            functions.append(inspect.unwrap(value))
        elif isinstance(value, type) and value.__module__ == module.__name__:
            for nested in vars(value).values():
                if isinstance(nested, (staticmethod, classmethod)):
                    nested = nested.__func__
                if isinstance(nested, FunctionType):
                    functions.append(inspect.unwrap(nested))
                elif isinstance(nested, property):
                    functions.extend(
                        method
                        for method in (nested.fget, nested.fset, nested.fdel)
                        if isinstance(method, FunctionType)
                    )
    return tuple(functions)

class _AuthenticatedExecutionGraph:

    def __init__(self, inputs: VerifiedBlindInputs) -> None:
        self._inputs = inputs
        self._modules: dict[str, ModuleType] = {}
        self._replaced: dict[str, ModuleType | None] = {}
        self._original_import = builtins.__import__

    @property
    def modules(self) -> Mapping[str, ModuleType]:
        return MappingProxyType(dict(self._modules))

    def _source(self, name: str) -> tuple[bytes, Path, bool]:
        if name == "dive" or name.startswith("dive."):
            root_class = "dive"
            root = _DIVE_READ_ROOT
            base = Path("src")
        elif name == "proteinfoundation" or name.startswith("proteinfoundation."):
            root_class = "upstream"
            root = EMERGENT_UPSTREAM_ROOT
            base = Path("src")
        else:
            raise KeyError(name)
        stem = base.joinpath(*name.split("."))
        candidates = ((stem.with_suffix(".py"), False), (stem / "__init__.py", True))
        for relative, is_package in candidates:
            key = f"execution_source:{root_class}:{relative.as_posix()}"
            if key in self._inputs.snapshots:
                return (
                    self._inputs.snapshots[key],
                    (root / relative).resolve(),
                    is_package,
                )
        raise SealError(
            f"project import {name!r} escaped the authenticated execution catalog"
        )

    def _has_source(self, name: str) -> bool:
        try:
            self._source(name)
        except KeyError:
            return False
        except SealError:
            return False
        return True

    def _authenticated_import(
        self,
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        del locals
        if level:
            package = "" if globals is None else str(globals.get("__package__", ""))
            absolute = importlib.util.resolve_name(f"{'.' * level}{name}", package)
        else:
            absolute = name
        if (
            absolute == "dive"
            or absolute.startswith("dive.")
            or (
                absolute == "proteinfoundation"
                or absolute.startswith("proteinfoundation.")
            )
        ):
            imported = self.load(absolute)
            if fromlist:
                for item in fromlist:
                    if item == "*" or hasattr(imported, item):
                        continue
                    child = f"{absolute}.{item}"
                    if self._has_source(child):
                        setattr(imported, item, self.load(child))
                return imported
            return self.load(absolute.split(".", 1)[0])
        return self._original_import(name, globals, None, fromlist, level)

    def load(self, name: str) -> ModuleType:
        cached = self._modules.get(name)
        if cached is not None:
            return cached
        snapshot, source_path, is_package = self._source(name)
        parent_name, separator, child_name = name.rpartition(".")
        parent = self.load(parent_name) if separator else None
        module = ModuleType(name)
        module.__file__ = str(source_path)
        module.__package__ = name if is_package else parent_name
        if is_package:
            module.__path__ = [str(source_path.parent)]
        controlled_builtins = dict(vars(builtins))
        controlled_builtins["__import__"] = self._authenticated_import
        module.__dict__["__builtins__"] = controlled_builtins
        self._modules[name] = module
        self._replaced[name] = sys.modules.get(name)
        sys.modules[name] = module
        if parent is not None:
            setattr(parent, child_name, module)
        try:
            code = compile(snapshot, str(source_path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
        except BaseException as error:
            raise SealError(
                f"authenticated project module {name!r} cannot execute"
            ) from error
        return module

    def close(self) -> None:
        for name in reversed(tuple(self._replaced)):
            previous = self._replaced[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

def _bounded_execution_graph_signature(
    modules: Mapping[str, ModuleType],
    *,
    root_module: str,
    required_roots: frozenset[str],
    observer: Callable[[str, str, object, str | None], None] | None = None,
) -> tuple[object, ...]:

    try:
        root = modules[root_module]
    except KeyError as error:
        raise SealError("authenticated execution root module is missing") from error
    logical_modules = {name: name for name in modules}
    seen: set[tuple[str, str, str]] = set()
    entries: list[tuple[object, ...]] = []

    def logical_objects(module: ModuleType) -> Mapping[int, tuple[str, str]]:
        return MappingProxyType(
            {
                id(value): (module.__name__, name)
                for name, value in vars(module).items()
                if type(value) is object
            }
        )

    def visit_nested(value: object, active: set[int]) -> None:
        identity = id(value)
        if _is_evaluator_owned_mutable(value):
            return
        if isinstance(value, FunctionType):
            if value.__module__ in modules:
                visit_function(value)
            return
        if isinstance(value, type):
            if value.__module__ in modules:
                visit_class(value)
            return
        if isinstance(value, ModuleType):
            if value.__name__ in modules:
                for nested in vars(modules[value.__name__]).values():
                    if isinstance(nested, (FunctionType, type)) and (
                        nested.__module__ == value.__name__
                    ):
                        visit_nested(nested, active)
            return
        if identity in active:
            return
        if isinstance(value, Mapping):
            active.add(identity)
            try:
                for key, nested in value.items():
                    visit_nested(key, active)
                    visit_nested(nested, active)
            finally:
                active.remove(identity)
        elif isinstance(value, (Sequence, set, frozenset)) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            active.add(identity)
            try:
                for nested in value:
                    visit_nested(nested, active)
            finally:
                active.remove(identity)

    def note(
        kind: str, qualified_name: str, value: object, origin: str | None = None
    ) -> None:

        if observer is not None:
            observer(kind, qualified_name, value, origin)

    def discover_function(function: FunctionType) -> None:
        owner_name = function.__globals__.get("__name__")
        label = f"{function.__module__}.{function.__qualname__}"
        for name in sorted(_execution_code_names(function.__code__)):
            if name in function.__globals__ and not _is_registered_mutable_global(
                owner_name, name
            ):
                note("global", f"{owner_name}.{name}", function.__globals__[name])
                visit_nested(function.__globals__[name], set())
        origin = function.__code__.co_filename
        for index, default in enumerate(function.__defaults__ or ()):
            note("function state", f"{label}.__defaults__[{index}]", default, origin)
        for slot, mapping in (
            ("__kwdefaults__", function.__kwdefaults__ or {}),
            ("__annotations__", function.__annotations__),
        ):
            for key, nested in mapping.items():
                note("function state", f"{label}.{slot}[{key!r}]", nested, origin)
        for value in (
            function.__defaults__,
            function.__kwdefaults__,
            function.__annotations__,
        ):
            visit_nested(value, set())
        if function.__closure__ is not None:
            for index, cell in enumerate(function.__closure__):
                try:
                    contents = cell.cell_contents
                except ValueError:
                    continue
                note(
                    "function state",
                    f"{label}.__closure__[{index}]",
                    contents,
                    origin,
                )
                visit_nested(contents, set())

    def visit_function(function: FunctionType) -> None:
        module_name = function.__module__
        key = (module_name, "function", function.__qualname__)
        if key in seen:
            return
        try:
            owner = modules[module_name]
        except KeyError as error:
            raise SealError(
                "project function escaped the bounded authenticated graph"
            ) from error
        seen.add(key)
        entries.append(
            (
                *key,
                _execution_function_signature(
                    function, logical_modules, logical_objects(owner)
                ),
            )
        )
        discover_function(function)

    def visit_class(module_class: type) -> None:
        module_name = module_class.__module__
        key = (module_name, "class", module_class.__qualname__)
        if key in seen:
            return
        try:
            owner = modules[module_name]
        except KeyError as error:
            raise SealError(
                "project class escaped the bounded authenticated graph"
            ) from error
        seen.add(key)
        entries.append(
            (
                *key,
                _execution_class_signature(
                    module_class, logical_modules, logical_objects(owner)
                ),
            )
        )
        for base in module_class.__bases__:
            if base.__module__ in modules:
                visit_class(base)
        for attribute, nested in vars(module_class).items():
            if not attribute.startswith("__"):
                note(
                    "class attribute",
                    f"{module_name}.{module_class.__qualname__}.{attribute}",
                    nested,
                )
        for nested in vars(module_class).values():
            if isinstance(nested, (staticmethod, classmethod)):
                nested = nested.__func__
            if isinstance(nested, FunctionType):
                discover_function(nested)
            elif isinstance(nested, property):
                for method in (nested.fget, nested.fset, nested.fdel):
                    if isinstance(method, FunctionType):
                        discover_function(method)

    roots = []
    for name in sorted(required_roots):
        value = vars(root).get(name)
        if not isinstance(value, (FunctionType, type)):
            raise SealError(f"reviewed execution module is missing root {name!r}")
        roots.append((name, value.__module__, value.__qualname__, type(value).__name__))
        visit_nested(value, set())
    return tuple(roots), tuple(sorted(entries, key=repr))

@lru_cache(maxsize=32)
def _cached_expected_execution_graph(
    snapshot_owner: int,
    catalog: tuple[tuple[str, bytes], ...],
    root_module: str,
    required_roots: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[object, ...]]:
    del snapshot_owner
    snapshots = MappingProxyType(dict(catalog))
    retained_inputs = VerifiedBlindInputs(
        seal_sha256="0" * 64,
        snapshots=snapshots,
        artifacts=MappingProxyType({}),
    )
    graph = _AuthenticatedExecutionGraph(retained_inputs)
    try:
        graph.load(root_module)
        expected_modules = dict(graph.modules)
    finally:
        graph.close()
    signature = _bounded_execution_graph_signature(
        expected_modules,
        root_module=root_module,
        required_roots=frozenset(required_roots),
    )
    return tuple(sorted(expected_modules)), signature

def _authenticate_reviewed_module_code(
    inputs: VerifiedBlindInputs,
    module: ModuleType,
    *,
    required_roots: frozenset[str],
) -> tuple[object, ...]:
    _execution_source_snapshot_for_module(inputs, module)
    catalog = tuple(
        sorted(
            (name, raw)
            for name, raw in inputs.snapshots.items()
            if name.startswith("execution_source:")
        )
    )
    expected_names, expected_signature = _cached_expected_execution_graph(
        id(inputs.snapshots),
        catalog,
        module.__name__,
        tuple(sorted(required_roots)),
    )
    live_modules: dict[str, ModuleType] = {}
    for name in expected_names:
        live_module = sys.modules.get(name)
        if not isinstance(live_module, ModuleType):
            raise SealError(
                f"authenticated project execution module {name!r} is not live"
            )
        _execution_source_snapshot_for_module(inputs, live_module)
        live_modules[name] = live_module
    live_signature = _bounded_execution_graph_signature(
        live_modules,
        root_module=module.__name__,
        required_roots=required_roots,
    )
    if live_signature != expected_signature:
        raise SealError(
            f"reviewed execution code/metadata/global graph changed: {module.__name__}"
        )
    return live_signature

def _revalidate_reviewed_module(
    inputs: VerifiedBlindInputs,
    module: ModuleType,
    *,
    required_roots: frozenset[str],
    expected_signature: tuple[object, ...],
) -> None:
    if (
        _authenticate_reviewed_module_code(
            inputs, module, required_roots=required_roots
        )
        != expected_signature
    ):
        raise SealError("reviewed execution callable/global signature changed")

def _selection_artifact_specs(
    selection: Mapping[str, object], selected_candidate: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    lineage = _require_mapping("selected Task 8 lineage", selected_candidate["lineage"])
    provenance = _require_mapping("selected Task 8 provenance", lineage["provenance"])
    config_identity = _require_mapping(
        "selected config identity", lineage["config_identity"]
    )
    specs: dict[str, dict[str, object]] = {
        "candidate_checkpoint": {
            "path": selection["checkpoint_path"],
            "sha256": selection["checkpoint_hash"],
            "size_bytes": selection["checkpoint_size_bytes"],
            "root_class": "bulk",
        },
        "parent_checkpoint": {
            "path": selection["parent_checkpoint_path"],
            "sha256": selection["parent_checkpoint_hash"],
            "size_bytes": selection["parent_checkpoint_size_bytes"],
            "root_class": "bulk",
        },
        "common_checkpoint": {
            "path": selection["common_checkpoint_path"],
            "sha256": selection["common_checkpoint_hash"],
            "size_bytes": selection["common_checkpoint_size_bytes"],
            "root_class": "upstream",
        },
        "autoencoder_checkpoint": {
            "path": provenance["autoencoder_checkpoint"],
            "sha256": provenance["autoencoder_checkpoint_hash"],
            "size_bytes": _AUTOENCODER_CHECKPOINT_SIZE_BYTES,
            "root_class": "upstream",
        },
        "config": {
            "path": selection["config_path"],
            "sha256": selection["config_hash"],
            "size_bytes": config_identity["size_bytes"],
            "root_class": "dive",
        },
    }
    if specs["autoencoder_checkpoint"]["sha256"] != _AUTOENCODER_CHECKPOINT_SHA256:
        raise SealError("selected autoencoder checkpoint differs from frozen identity")
    for name, prefix in (
        ("provenance", "provenance"),
        ("training_completion", "training_completion"),
        ("checkpoint_completion", "checkpoint_completion"),
        ("validation_record", "validation_record"),
    ):
        specs[name] = {
            "path": selection[f"{prefix}_path"],
            "sha256": selection[f"{prefix}_hash"],
            "size_bytes": selection[f"{prefix}_size_bytes"],
            "root_class": "evidence",
        }
    provenance = _require_mapping("selected provenance", lineage["provenance"])
    manifest_root = Path(str(provenance["split_manifest_root"]))
    for raw in provenance["loader_manifests"]:
        manifest = _require_mapping("selected loader manifest", raw)
        name = str(manifest["name"])
        specs[f"split_manifest:{name}"] = {
            "path": str(manifest_root / name),
            "sha256": manifest["sha256"],
            "size_bytes": manifest["size_bytes"],
            "root_class": "evidence",
        }
    return specs

def _classify_read_root(path: Path, environment: _SealEnvironment) -> str:
    for name, root in (
        ("dive", environment.dive_repo_root),
        ("upstream", environment.upstream_repo_root),
        ("bulk", _BULK_READ_ROOT),
        ("evidence", environment.evidence_root),
    ):
        try:
            path.relative_to(root)
            return name
        except ValueError:
            pass
    if environment.bypass_selection_artifact_io:
        if str(path).startswith("/bulk/"):
            return "bulk"
        if str(path).startswith("/evidence/"):
            return "evidence"
        if path.suffix in {".yaml", ".yml"}:
            return "dive"
        if str(path).startswith(str(EMERGENT_UPSTREAM_ROOT)):
            return "upstream"
    raise SealError(f"Task 8 file {path} is outside every fixed read-only root")

def _all_task8_file_specs(
    preblind: Mapping[str, object], environment: _SealEnvironment
) -> dict[str, dict[str, object]]:
    by_identity: dict[tuple[str, str, int], dict[str, object]] = {}

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if {"path", "sha256", "size_bytes"} <= set(value):
                path = Path(str(value["path"]))
                digest = _require_hash("Task 8 transitive file sha256", value["sha256"])
                size = value["size_bytes"]
                if type(size) is not int or size < 0:
                    raise SealError("Task 8 transitive file size is invalid")
                key = (str(path), digest, size)
                by_identity[key] = {
                    "path": str(path),
                    "sha256": digest,
                    "size_bytes": size,
                    "root_class": _classify_read_root(path, environment),
                }
            if "split_manifest_root" in value and "loader_manifests" in value:
                root = Path(str(value["split_manifest_root"]))
                manifests = value["loader_manifests"]
                if not isinstance(manifests, Sequence) or isinstance(
                    manifests, (str, bytes)
                ):
                    raise SealError("Task 8 loader manifests are malformed")
                for raw in manifests:
                    manifest = _require_mapping("Task 8 loader manifest", raw)
                    path = root / str(manifest["name"])
                    digest = _require_hash(
                        "Task 8 loader manifest sha256", manifest["sha256"]
                    )
                    size = manifest["size_bytes"]
                    if type(size) is not int or size < 0:
                        raise SealError("Task 8 loader manifest size is invalid")
                    by_identity[(str(path), digest, size)] = {
                        "path": str(path),
                        "sha256": digest,
                        "size_bytes": size,
                        "root_class": _classify_read_root(path, environment),
                    }
            for nested in value.values():
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                visit(nested)

    visit(preblind)
    return {
        f"task8:{index:03d}": spec
        for index, (_identity, spec) in enumerate(sorted(by_identity.items()))
    }

def _artifact_root(root_class: str, environment: _SealEnvironment) -> Path:
    roots = {
        "dive": environment.dive_repo_root,
        "upstream": environment.upstream_repo_root,
        "bulk": _BULK_READ_ROOT,
        "evidence": environment.evidence_root,
    }
    try:
        return roots[root_class]
    except KeyError as error:
        raise SealError("selection artifact has an unknown root policy") from error

def _open_selection_artifact(
    name: str, spec: Mapping[str, object], environment: _SealEnvironment
) -> VerifiedArtifact:
    path = Path(str(spec["path"]))
    expected_hash = _require_hash(f"selection_artifacts.{name}.sha256", spec["sha256"])
    expected_size = spec["size_bytes"]
    if type(expected_size) is not int or expected_size < 0:
        raise SealError(f"selection_artifacts.{name}.size_bytes is invalid")
    root_class = str(spec["root_class"])
    if environment.bypass_selection_artifact_io:
        return VerifiedArtifact(name, -1, expected_hash, expected_size, -1, -1)
    root = _artifact_root(root_class, environment)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise SealError(
            f"selection_artifacts.{name} is outside its {root_class} root"
        ) from error
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current = os.open(root, directory_flags)
    try:
        for part in relative.parent.parts:
            following = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = following
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current,
        )
    except OSError as error:
        raise SealError(
            f"selection_artifacts.{name} cannot be opened component-safely"
        ) from error
    finally:
        os.close(current)
    retain = False
    try:
        metadata = os.fstat(descriptor)
        digest = hashlib.sha256()
        observed_size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            observed_size += len(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or observed_size != expected_size
            or digest.hexdigest() != expected_hash
        ):
            raise SealError(
                f"selection_artifacts.{name} differs from its Task 8 identity"
            )
        retain = True
        return VerifiedArtifact(
            name,
            descriptor,
            expected_hash,
            expected_size,
            metadata.st_dev,
            metadata.st_ino,
        )
    finally:
        if not retain:
            os.close(descriptor)

def _validate_file_identity(
    name: str, value: object, environment: _SealEnvironment
) -> dict[str, object]:
    identity = _require_mapping(name, value)
    allowed = {"path", "sha256", "size_bytes"}
    if "device" in identity or "inode" in identity:
        allowed |= {"device", "inode"}
    _require_exact_keys(name, identity, allowed)
    raw_path = identity["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise SealError(f"{name}.path must be a nonempty absolute path")
    expected_hash = _require_hash(f"{name}.sha256", identity["sha256"])
    size = identity["size_bytes"]
    if type(size) is not int or size < 0:
        raise SealError(f"{name}.size_bytes must be a nonnegative integer")
    data, metadata = _safe_file_bytes(
        Path(raw_path), name=name, environment=environment
    )
    if metadata.st_size != size or len(data) != size:
        raise SealError(f"{name} size differs from the reviewed identity")
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise SealError(f"{name} hash differs from the reviewed identity")
    for field, observed in (("device", metadata.st_dev), ("inode", metadata.st_ino)):
        if field in identity and identity[field] != observed:
            raise SealError(f"{name} {field} differs from the verified descriptor")
    return {
        "path": raw_path,
        "sha256": expected_hash,
        "size_bytes": size,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }

def _validate_strict_blind_loader_manifest(
    raw: bytes, inventory: Mapping[str, object]
) -> dict[str, object]:

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("strict blind loader manifest is not JSON") from error
    if raw != _canonical_bytes(value) + b"\n":
        raise SealError("strict blind loader manifest is not canonical")
    manifest = _require_mapping("strict blind loader manifest", value)
    _require_exact_keys(
        "strict blind loader manifest",
        manifest,
        ("schema_version", "records", "semantic_sha256"),
    )
    if manifest["schema_version"] != 1:
        raise SealError("strict blind loader manifest schema_version must be one")
    records = manifest["records"]
    if not isinstance(records, list) or len(records) != 1_596:
        raise SealError("strict blind loader manifest must contain exactly 1,596 rows")
    parsed = []
    for index, raw_record in enumerate(records):
        name = f"strict blind loader manifest.records[{index}]"
        record = _require_mapping(name, raw_record)
        _require_exact_keys(
            name,
            record,
            (
                "family",
                "example_id",
                "parent_id",
                "cell_hash",
                "source_locator",
                "loader_row",
                "loader_spec",
            ),
        )
        identity = {
            key: record[key]
            for key in ("family", "example_id", "parent_id", "cell_hash")
        }
        if identity["family"] not in FAMILIES or any(
            type(identity[key]) is not str or not identity[key] for key in identity
        ):
            raise SealError(f"{name} identity is invalid")
        _require_hash(f"{name}.cell_hash", identity["cell_hash"])
        locator = _require_mapping(f"{name}.source_locator", record["source_locator"])
        _require_exact_keys(
            f"{name}.source_locator",
            locator,
            ("family", "root_class", "relative_path", "sha256", "size_bytes"),
        )
        relative = Path(str(locator["relative_path"]))
        if (
            locator["family"] != identity["family"]
            or locator["root_class"] != "bulk"
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != locator["relative_path"]
        ):
            raise SealError(f"{name}.source_locator is not normalized/family-tagged")
        _require_hash(f"{name}.source_locator.sha256", locator["sha256"])
        if type(locator["size_bytes"]) is not int or locator["size_bytes"] <= 0:
            raise SealError(f"{name}.source_locator.size_bytes is invalid")
        loader_row = _require_mapping(f"{name}.loader_row", record["loader_row"])
        _require_exact_keys(
            f"{name}.loader_row",
            loader_row,
            (
                "example_id",
                "relative_path",
                "generated",
                "context",
                "target",
                "sha256",
                "size_bytes",
            ),
        )
        if (
            loader_row["example_id"] != identity["example_id"]
            or loader_row["relative_path"] != locator["relative_path"]
            or loader_row["sha256"] != locator["sha256"]
            or loader_row["size_bytes"] != locator["size_bytes"]
            or type(loader_row["generated"]) is not str
            or not loader_row["generated"]
            or type(loader_row["context"]) is not str
            or not loader_row["context"]
            or type(loader_row["target"]) is not str
        ):
            raise SealError(f"{name}.loader_row is not exact or locator-bound")
        loader = _require_mapping(f"{name}.loader_spec", record["loader_spec"])
        _require_exact_keys(
            f"{name}.loader_spec",
            loader,
            (
                "schema",
                "version",
                "family",
                "crop_size",
                "motif_seed",
                "family_loader_kwargs",
                "transform_chain_sources",
                "collate_identity",
                "batch_size",
                "shuffle",
                "num_workers",
            ),
        )
        if (
            loader["schema"] != "dive-directional-family-loader-v1"
            or loader["version"] != 1
            or loader["family"] != identity["family"]
            or type(loader["crop_size"]) is not int
            or loader["crop_size"] <= 0
            or loader["batch_size"] != 1
            or loader["shuffle"] is not False
            or loader["num_workers"] != 0
        ):
            raise SealError(f"{name}.loader_spec is not exact/family-tagged")
        motif = _require_mapping(f"{name}.loader_spec.motif_seed", loader["motif_seed"])
        _require_exact_keys(
            f"{name}.loader_spec.motif_seed", motif, ("policy", "value")
        )
        expected_policy = "fixed-per-record" if identity["family"] == "ame" else "none"
        if (
            motif["policy"] != expected_policy
            or (expected_policy == "none" and motif["value"] is not None)
            or (
                expected_policy == "fixed-per-record"
                and (type(motif["value"]) is not int or motif["value"] < 0)
            )
        ):
            raise SealError(f"{name}.loader_spec.motif_seed is invalid")
        kwargs = _require_mapping(
            f"{name}.loader_spec.family_loader_kwargs",
            loader["family_loader_kwargs"],
        )
        if dict(kwargs) != {}:
            raise SealError(f"{name}.loader_spec.family_loader_kwargs drifted")
        sources = loader["transform_chain_sources"]
        if type(sources) is not list or not sources:
            raise SealError(f"{name}.loader_spec transform sources are invalid")

        def validate_source(source_name: str, raw_source: object) -> None:
            source = _require_mapping(source_name, raw_source)
            _require_exact_keys(
                source_name,
                source,
                ("root_class", "relative_path", "sha256", "size_bytes"),
            )
            source_relative = Path(str(source["relative_path"]))
            if (
                source["root_class"] not in {"dive", "upstream"}
                or source_relative.is_absolute()
                or not source_relative.parts
                or any(part in {"", ".", ".."} for part in source_relative.parts)
                or source_relative.as_posix() != source["relative_path"]
            ):
                raise SealError(f"{source_name} is not a normalized fixed-root source")
            _require_hash(f"{source_name}.sha256", source["sha256"])
            if type(source["size_bytes"]) is not int or source["size_bytes"] <= 0:
                raise SealError(f"{source_name}.size_bytes is invalid")

        for source_index, raw_source in enumerate(sources):
            validate_source(
                f"{name}.loader_spec.transform_chain_sources[{source_index}]",
                raw_source,
            )
        validate_source(
            f"{name}.loader_spec.collate_identity", loader["collate_identity"]
        )
        _reject_raw_outcome_fields(record)
        parsed.append(dict(record))
    strict = _require_mapping(
        "blind_inventory.strict_flow_inventory", inventory["strict_flow_inventory"]
    )
    expected = strict["rows"]
    observed = [
        {key: record[key] for key in ("family", "example_id", "parent_id", "cell_hash")}
        for record in parsed
    ]
    if _canonical_bytes(observed) != _canonical_bytes(expected):
        raise SealError(
            "strict blind loader manifest is not bijective with sealed strict inventory"
        )
    semantic = {"schema_version": 1, "records": parsed}
    expected_semantic = _require_hash(
        "strict blind loader manifest.semantic_sha256", manifest["semantic_sha256"]
    )
    if canonical_record_hash(semantic) != expected_semantic:
        raise SealError("strict blind loader manifest semantic hash is inconsistent")
    return dict(manifest)

def _validate_payload(
    value: Mapping[str, object], environment: _SealEnvironment
) -> dict[str, object]:
    allowed_seal_keys = set(REQUIRED_SEAL_KEYS)
    for derived_key in ("metric_source", "selection_artifacts"):
        if derived_key in value:
            allowed_seal_keys.add(derived_key)
    _require_exact_keys("seal", value, allowed_seal_keys)
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != SEAL_SCHEMA_VERSION
    ):
        raise SealError(f"schema_version must be exactly {SEAL_SCHEMA_VERSION}")
    execution_authentication = _validate_execution_authentication(
        value["execution_authentication"]
    )

    repositories = _require_mapping("repositories", value["repositories"])
    _require_exact_keys("repositories", repositories, ("dive", "upstream"))
    for name, expected_commit in (
        ("dive", None),
        ("upstream", EMERGENT_UPSTREAM_COMMIT),
    ):
        repository = _require_mapping(f"repositories.{name}", repositories[name])
        _require_exact_keys(f"repositories.{name}", repository, ("commit", "dirty"))
        commit = _require_hash(
            f"repositories.{name}.commit", repository["commit"], length=40
        )
        if expected_commit is not None and commit != expected_commit:
            raise SealError(f"repositories.{name}.commit differs from the emergent pin")
        if repository["dirty"] is not False:
            raise SealError(f"repositories.{name}.dirty must be false")
    catalog_binding = _require_mapping(
        "execution_authentication.source_catalog",
        execution_authentication["source_catalog"],
    )
    for name in ("dive", "upstream"):
        repository = _require_mapping(f"repositories.{name}", repositories[name])
        if catalog_binding[f"{name}_commit"] != repository["commit"]:
            raise SealError(
                f"execution_authentication.source_catalog.{name}_commit differs "
                f"from repositories.{name}.commit"
            )
    for name, root in (
        ("dive", environment.dive_repo_root),
        ("upstream", environment.upstream_repo_root),
    ):
        observed_commit, observed_clean = _capture_repository(name, root, environment)
        repository = _require_mapping(f"repositories.{name}", repositories[name])
        if observed_commit != repository["commit"] or not observed_clean:
            raise SealError(
                f"{name} live repository state differs from the exact clean seal"
            )

    bindings = _require_mapping("file_bindings", value["file_bindings"])
    _require_exact_keys("file_bindings", bindings, _REQUIRED_FILE_BINDINGS)
    verified_bindings = {
        name: _validate_file_identity(
            f"file_bindings.{name}", bindings[name], environment
        )
        for name in _REQUIRED_FILE_BINDINGS
    }
    for repository_name, binding_name in (
        ("dive", "dive_dirty_report"),
        ("upstream", "upstream_dirty_report"),
    ):
        repository = _require_mapping(
            f"repositories.{repository_name}", repositories[repository_name]
        )
        identity = verified_bindings[binding_name]
        report_bytes, _ = _safe_file_bytes(
            Path(str(identity["path"])),
            name=f"file_bindings.{binding_name}",
            environment=environment,
        )
        try:
            report = json.loads(report_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SealError(f"{binding_name} is not canonical JSON") from error
        expected_report = {
            "schema_version": 1,
            "commit": repository["commit"],
            "dirty": False,
            "porcelain": "",
        }
        if (
            report_bytes != _canonical_bytes(report) + b"\n"
            or report != expected_report
        ):
            raise SealError(
                f"{binding_name} must prove the exact commit with dirty false"
            )

    preblind = _require_mapping("preblind", value["preblind"])
    preblind_identity = verified_bindings["preblind_record"]
    preblind_bytes, _ = _safe_file_bytes(
        Path(str(preblind_identity["path"])),
        name="file_bindings.preblind_record",
        environment=environment,
    )
    if preblind_bytes != _canonical_bytes(preblind) + b"\n":
        raise SealError("preblind differs from the exact canonical Task 8 record")
    try:
        selection, preblind_passed, _, validated_blind_payload = _validate_preblind(
            preblind
        )
    except DirectionalConfirmationError as error:
        raise SealError(
            f"preblind canonical Task 8 validation failed: {error}"
        ) from error
    if not preblind_passed:
        raise SealError(
            "preblind canonical Task 8 development/rerun gates did not pass"
        )
    candidate = selection.to_mapping()
    if candidate["repo_commit"] != repositories["dive"]["commit"]:
        raise SealError("Task 8 selection repo_commit differs from sealed repository")
    if candidate["upstream_commit"] != EMERGENT_UPSTREAM_COMMIT:
        raise SealError("Task 8 selection upstream_commit differs from emergent pin")

    inventory = _require_mapping("blind_inventory", value["blind_inventory"])
    required_inventory = {
        "schema_version",
        "strict_row_counts",
        "strict_parent_counts",
        "strict_flow_inventory",
        "strict_inventory_hash",
        "generation_parent_ids",
        "generation_seeds",
        "generation_arms",
        "generation_cells",
        "planned_generation_cells",
        "generation_inventory",
        "generation_inventory_hash",
        "availability_manifest",
        "availability_manifest_hash",
    }
    _require_exact_keys("blind_inventory", inventory, required_inventory)
    nested_strict = _require_mapping(
        "blind_inventory.strict_flow_inventory", inventory["strict_flow_inventory"]
    )
    nested_rows = nested_strict.get("rows")
    if not isinstance(nested_rows, Sequence) or isinstance(nested_rows, (str, bytes)):
        raise SealError("blind_inventory.strict_flow_inventory.rows must be a sequence")
    metadata_rows = [
        {
            "family": _require_mapping("strict metadata row", row).get("family"),
            "example_id": _require_mapping("strict metadata row", row).get(
                "example_id"
            ),
            "parent_id": _require_mapping("strict metadata row", row).get("parent_id"),
            "partition": "test-blind",
            "view_strict": True,
        }
        for row in nested_rows
    ]
    availability = _require_mapping(
        "blind_inventory.availability_manifest", inventory["availability_manifest"]
    )
    rebuilt_inventory = build_blind_inventory(
        metadata_rows,
        deterministic_exclusions=availability.get("deterministic_exclusions", ()),
    )
    if _canonical_bytes(inventory) != _canonical_bytes(rebuilt_inventory):
        raise SealError(
            "blind_inventory is disconnected from its canonical strict metadata"
        )
    if inventory["strict_row_counts"] != dict(STRICT_FLOW_ROW_COUNTS):
        raise SealError("blind_inventory strict row counts drifted")
    if inventory["strict_parent_counts"] != dict(STRICT_FLOW_PARENT_COUNTS):
        raise SealError("blind_inventory strict parent counts drifted")
    if inventory["generation_seeds"] != list(FROZEN_GENERATION_SEEDS):
        raise SealError("blind_inventory generation seeds drifted")
    if inventory["generation_arms"] != list(FROZEN_GENERATION_ARMS):
        raise SealError("blind_inventory generation arms drifted")
    if (
        inventory["planned_generation_cells"] != 240
        or len(inventory["generation_cells"]) != 240
    ):
        raise SealError(
            "blind_inventory must bind exactly 240 planned generation cells"
        )
    for record_name, hash_name in (
        ("strict_flow_inventory", "strict_inventory_hash"),
        ("generation_inventory", "generation_inventory_hash"),
        ("availability_manifest", "availability_manifest_hash"),
    ):
        expected = _require_hash(f"blind_inventory.{hash_name}", inventory[hash_name])
        if canonical_record_hash(inventory[record_name]) != expected:
            raise SealError(f"blind_inventory.{hash_name} is inconsistent")
    loader_identity = verified_bindings["strict_blind_loader_manifest"]
    loader_raw, _ = _safe_file_bytes(
        Path(str(loader_identity["path"])),
        name="file_bindings.strict_blind_loader_manifest",
        environment=environment,
    )
    _validate_strict_blind_loader_manifest(loader_raw, inventory)

    evaluation = _require_mapping("evaluation", value["evaluation"])
    required_evaluation = (
        "flow_arms",
        "generation_arms",
        "generation_seeds",
        "corruption_seed",
        "bootstrap_seed",
        "bootstrap_resamples",
        "flow_sample_count",
        "flow_passes_per_row",
        "strict_flow_passes",
        "planned_generation_cells",
        "metrics",
        "fixed_generation_route",
    )
    _require_exact_keys("evaluation", evaluation, required_evaluation)
    raw_flow_arms = evaluation["flow_arms"]
    if not isinstance(raw_flow_arms, Sequence) or isinstance(
        raw_flow_arms, (str, bytes)
    ):
        raise SealError("evaluation.flow_arms must be a sequence")
    names = []
    for index, raw in enumerate(raw_flow_arms):
        arm = _require_mapping(f"evaluation.flow_arms[{index}]", raw)
        _require_exact_keys(
            f"evaluation.flow_arms[{index}]", arm, ("name", "compute_equivalent_passes")
        )
        names.append(arm["name"])
        expected_passes = 1.0 if arm["name"] == "none" else 1.75
        if (
            type(arm["compute_equivalent_passes"]) is not float
            or arm["compute_equivalent_passes"] != expected_passes
        ):
            raise SealError(
                f"evaluation.flow_arms[{index}] compute-equivalent passes drifted"
            )
    if tuple(names) != FROZEN_FLOW_ARMS:
        raise SealError("evaluation.flow_arms must bind every frozen arm in order")
    if evaluation["generation_arms"] != list(FROZEN_GENERATION_ARMS):
        raise SealError("evaluation.generation_arms drifted")
    if evaluation["generation_seeds"] != list(FROZEN_GENERATION_SEEDS):
        raise SealError("evaluation.generation_seeds drifted")
    for name, expected in (
        ("corruption_seed", 42),
        ("bootstrap_seed", 42),
        ("bootstrap_resamples", 2000),
        ("flow_sample_count", 1596),
        ("strict_flow_passes", 14_364),
        ("planned_generation_cells", 240),
    ):
        if type(evaluation[name]) is not int or evaluation[name] != expected:
            raise SealError(f"evaluation.{name} drifted")
    if (
        type(evaluation["flow_passes_per_row"]) is not float
        or evaluation["flow_passes_per_row"] != 9.0
    ):
        raise SealError("evaluation.flow_passes_per_row drifted")
    if evaluation["fixed_generation_route"] not in (
        "z_to_x",
        "x_to_z",
        "bidirectional",
    ):
        raise SealError(
            "evaluation.fixed_generation_route must be a sealed non-none route"
        )
    development_report = _require_mapping(
        "preblind.development_report", preblind["development_report"]
    )
    candidates = development_report["candidates"]
    selected_candidate = next(
        _require_mapping("selected Task 8 candidate", item)
        for item in candidates
        if _require_mapping("Task 8 candidate", item).get("seed") == selection.seed
    )
    example_statistics = selected_candidate["example_statistics"]
    exact_fixed_route = min(
        HARD_ROUTES,
        key=lambda route: (
            _equal_family_route_score(example_statistics, route),
            route,
        ),
    )
    if evaluation["fixed_generation_route"] != exact_fixed_route:
        raise SealError(
            "evaluation.fixed_generation_route differs from the exact Task 8 best fixed route"
        )
    artifact_specs = _selection_artifact_specs(candidate, selected_candidate)
    artifact_specs.update(_all_task8_file_specs(preblind, environment))
    verified_artifacts: dict[str, dict[str, object]] = {}
    for artifact_name, spec in artifact_specs.items():
        artifact = _open_selection_artifact(artifact_name, spec, environment)
        verified_artifacts[artifact_name] = {
            **spec,
            "device": artifact.device,
            "inode": artifact.inode,
        }
        if artifact.descriptor >= 0:
            os.close(artifact.descriptor)
    if (
        "selection_artifacts" in value
        and value["selection_artifacts"] != verified_artifacts
    ):
        raise SealError("selection artifacts differ from exact Task 8 live bindings")
    metrics = evaluation["metrics"]
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        raise SealError("evaluation.metrics must be the exact frozen metric inventory")
    if tuple(metrics) != _FROZEN_METRICS:
        raise SealError(
            "evaluation.metrics differs from the exact frozen metric inventory"
        )

    thresholds = _require_mapping("thresholds", value["thresholds"])
    _require_exact_keys("thresholds", thresholds, _BLIND_THRESHOLD_VALUES)
    for name, expected in _BLIND_THRESHOLD_VALUES.items():
        if type(thresholds[name]) is not type(expected) or thresholds[name] != expected:
            raise SealError(f"thresholds.{name} drifted")

    blind_payload = _require_mapping("blind_seal_payload", value["blind_seal_payload"])
    _require_exact_keys("blind_seal_payload", blind_payload, _BLIND_SEAL_KEYS)
    if blind_payload["schema_version"] != 1:
        raise SealError("blind_seal_payload.schema_version must be exactly 1")
    for key in _BLIND_SEAL_KEYS - {"schema_version"}:
        _require_hash(f"blind_seal_payload.{key}", blind_payload[key])
    for seal_name, inventory_name in (
        ("strict_flow_inventory_hash", "strict_inventory_hash"),
        ("generation_inventory_hash", "generation_inventory_hash"),
        ("availability_manifest_hash", "availability_manifest_hash"),
    ):
        if blind_payload[seal_name] != inventory[inventory_name]:
            raise SealError(f"blind_seal_payload.{seal_name} differs from inventory")
    if _canonical_bytes(blind_payload) != _canonical_bytes(validated_blind_payload):
        raise SealError("blind_seal_payload differs from canonical Task 8 preblind")

    output = _require_mapping("output_manifest", value["output_manifest"])
    output_keys = {"path", "sha256", "size_bytes", "contract"}
    if "device" in output or "inode" in output:
        output_keys |= {"device", "inode"}
    _require_exact_keys("output_manifest", output, output_keys)
    if output["contract"] != "append-only-jsonl-v1":
        raise SealError("output_manifest.contract must be append-only-jsonl-v1")
    verified_output = _validate_file_identity(
        "output_manifest",
        {key: output[key] for key in output if key != "contract"},
        environment,
    )

    record = json.loads(_canonical_bytes(value))
    record["file_bindings"] = verified_bindings
    metric_source = _metric_source_identity(environment.metric_read_root)
    if "metric_source" in value and value["metric_source"] != metric_source:
        raise SealError("metric source differs from the fixed implementation")
    record["metric_source"] = metric_source
    record["selection_artifacts"] = verified_artifacts
    record["output_manifest"] = {**verified_output, "contract": output["contract"]}
    return record

def _seal_from_record(
    path: Path, record_hash: str, record: Mapping[str, object]
) -> ConfirmationSeal:
    blind_payload = _require_mapping("blind_seal_payload", record["blind_seal_payload"])
    inventory = _require_mapping("blind_inventory", record["blind_inventory"])
    return ConfirmationSeal(
        path=path,
        record_sha256=record_hash,
        blind_seal_hash=canonical_record_hash(blind_payload),
        strict_inventory_hash=str(inventory["strict_inventory_hash"]),
        generation_inventory_hash=str(inventory["generation_inventory_hash"]),
        availability_manifest_hash=str(inventory["availability_manifest_hash"]),
        record=_freeze_json(record),
    )

def _write_exclusive(path: Path, data: bytes, environment: _SealEnvironment) -> None:
    parent, leaf = _open_terminal_parent(path, environment, create=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(leaf, flags, 0o660, dir_fd=parent)
        os.fchmod(descriptor, 0o660)
        metadata = os.fstat(descriptor)
        if (
            metadata.st_gid != environment.expected_gid
            or metadata.st_mode & 0o660 != 0o660
        ):
            raise SealError("created terminal file lacks exact group-write protection")
        os.fsync(parent)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SealError("exclusive terminal write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fsync(parent)
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            os.close(parent)

def _create_confirmation_seal(
    payload: Mapping[str, object],
    destination: Path | str,
    environment: _SealEnvironment,
) -> ConfirmationSeal:
    path = Path(destination)
    _validate_terminal_destination(path, environment)
    record = _validate_payload(_require_mapping("seal", payload), environment)
    raw = _canonical_bytes(record) + b"\n"
    record_hash = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    reservation = path.with_name(path.name + ".publication.reservation.json")
    success = path.with_name(path.name + ".publication.success.json")
    failure = path.with_name(path.name + ".publication.failure.json")
    reservation_payload = {
        "schema_version": 1,
        "seal_path": str(path),
        "seal_sha256": record_hash,
        "sidecar_path": str(sidecar),
    }
    reservation_raw = _canonical_bytes(reservation_payload) + b"\n"
    _write_exclusive(reservation, reservation_raw, environment)
    try:
        _write_exclusive(path, raw, environment)
        _write_exclusive(sidecar, f"{record_hash}  {path.name}\n".encode(), environment)
        success_payload = {
            "schema_version": 1,
            "status": "success",
            "reservation_sha256": hashlib.sha256(reservation_raw).hexdigest(),
            "seal_sha256": record_hash,
            "sidecar_sha256": hashlib.sha256(
                f"{record_hash}  {path.name}\n".encode()
            ).hexdigest(),
        }
        _write_exclusive(
            success, _canonical_bytes(success_payload) + b"\n", environment
        )
    except BaseException as error:
        failure_payload = {
            "schema_version": 1,
            "status": "failure",
            "reservation_sha256": hashlib.sha256(reservation_raw).hexdigest(),
            "error": f"{type(error).__name__}: {error}",
        }
        try:
            _write_exclusive(
                failure, _canonical_bytes(failure_payload) + b"\n", environment
            )
        except BaseException:
            pass
        raise
    return _seal_from_record(path, record_hash, record)

def _verify_seal_publication(
    path: Path, expected_hash: str, environment: _SealEnvironment
) -> None:
    reservation = path.with_name(path.name + ".publication.reservation.json")
    success = path.with_name(path.name + ".publication.success.json")
    try:
        reservation_raw, _ = _safe_file_bytes(
            reservation, name="seal publication reservation", environment=environment
        )
        reservation_record = json.loads(reservation_raw)
        success_raw, _ = _safe_file_bytes(
            success, name="seal publication success", environment=environment
        )
        record = json.loads(success_raw)
    except (OSError, SealError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("seal publication is incomplete") from error
    expected_reservation = {
        "schema_version": 1,
        "seal_path": str(path),
        "seal_sha256": expected_hash,
        "sidecar_path": str(path.with_name(path.name + ".sha256")),
    }
    if (
        reservation_raw != _canonical_bytes(reservation_record) + b"\n"
        or reservation_record != expected_reservation
    ):
        raise SealError("seal publication reservation is inconsistent")
    expected = {
        "schema_version": 1,
        "status": "success",
        "reservation_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        "seal_sha256": expected_hash,
        "sidecar_sha256": hashlib.sha256(
            f"{expected_hash}  {path.name}\n".encode()
        ).hexdigest(),
    }
    if success_raw != _canonical_bytes(record) + b"\n" or record != expected:
        raise SealError("seal publication success disposition is inconsistent")

def create_confirmation_seal(
    payload: Mapping[str, object], destination: Path | str
) -> ConfirmationSeal:

    return _create_confirmation_seal(payload, destination, _environment())

def _create_confirmation_seal_test_only(
    payload: Mapping[str, object],
    destination: Path | str,
    test_environment: _TestOnlyEnvironment,
) -> ConfirmationSeal:
    return _create_confirmation_seal(
        payload, destination, _test_only_seal_environment(test_environment)
    )

def _verify_confirmation_seal(
    path: Path | str,
    expected_sha256: str,
    environment: _SealEnvironment,
) -> ConfirmationSeal:
    expected = _require_hash("expected seal sha256", expected_sha256)
    seal_path = Path(path)
    _verify_seal_publication(seal_path, expected, environment)
    raw, _ = _safe_file_bytes(seal_path, name="seal", environment=environment)
    if not raw.endswith(b"\n"):
        raise SealError("seal must end with one canonical newline")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError(f"seal is not valid JSON: {error}") from error
    canonical = _canonical_bytes(record) + b"\n"
    if raw != canonical:
        raise SealError("seal bytes are not canonical")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise SealError("seal hash differs from the explicitly authorized hash")
    sidecar_path = seal_path.with_name(seal_path.name + ".sha256")
    sidecar, _ = _safe_file_bytes(
        sidecar_path, name="seal sidecar", environment=environment
    )
    if sidecar != f"{observed}  {seal_path.name}\n".encode():
        raise SealError("seal sidecar differs from the canonical seal")
    validated = _validate_payload(_require_mapping("seal", record), environment)
    if _canonical_bytes(validated) != _canonical_bytes(record):
        raise SealError("seal changed while its bindings were verified")
    return _seal_from_record(seal_path, observed, validated)

def verify_confirmation_seal(
    path: Path | str, expected_sha256: str
) -> ConfirmationSeal:

    return _verify_confirmation_seal(path, expected_sha256, _environment())

def _verify_confirmation_seal_test_only(
    path: Path | str,
    expected_sha256: str,
    test_environment: _TestOnlyEnvironment,
) -> ConfirmationSeal:
    return _verify_confirmation_seal(
        path, expected_sha256, _test_only_seal_environment(test_environment)
    )

def execute_generation_arm(
    model: object,
    *,
    arm: str,
    fixed_route: str,
    sample: Callable[[], Any],
) -> Any:

    if arm == "adaptive":
        route = "adaptive"
    elif arm == "best_frozen_fixed_control":
        route = fixed_route
        if route not in ("z_to_x", "x_to_z", "bidirectional"):
            raise SealError(
                "sealed fixed generation route must be a non-none hard route"
            )
    else:
        raise SealError(f"unknown generation arm {arm!r}")
    nn = getattr(model, "nn", None)
    if nn is None or not callable(getattr(nn, "route_context", None)):
        raise SealError("model.nn does not expose the directional route_context")
    with nn.route_context(route):
        return sample()

def _selected_generation_record(
    records: Sequence[Mapping[str, object]], family: str, parent_id: str
) -> Mapping[str, object]:
    candidates = [
        record
        for record in records
        if record.get("family") == family and record.get("parent_id") == parent_id
    ]
    if not candidates:
        raise SealError("generation parent has no strict loader representative")
    candidates.sort(
        key=lambda record: (
            str(record.get("cell_hash")),
            str(record.get("example_id")),
        )
    )
    key = (
        str(candidates[0].get("cell_hash")),
        str(candidates[0].get("example_id")),
    )
    if (
        sum(
            (str(record.get("cell_hash")), str(record.get("example_id"))) == key
            for record in candidates
        )
        != 1
    ):
        raise SealError("generation representative must be exactly selected")
    return candidates[0]

def _validate_generation_payload(
    cell: GenerationCell,
    value: object,
    loader_records: Sequence[Mapping[str, object]] | None,
) -> None:
    payload = _require_mapping("generation request payload", value)
    _require_exact_keys(
        "generation request payload",
        payload,
        ("representative_record", "generation_policy"),
    )
    representative = _require_mapping(
        "generation representative record", payload["representative_record"]
    )
    if (
        representative.get("family") != cell.family
        or representative.get("parent_id") != cell.parent_id
    ):
        raise SealError("generation representative identity differs from its cell")
    if loader_records is not None:
        expected = _selected_generation_record(
            loader_records, cell.family, cell.parent_id
        )
        if _canonical_bytes(representative) != _canonical_bytes(expected):
            raise SealError(
                "generation representative differs from deterministic manifest minimum"
            )
    policy = _require_mapping(
        "generation request payload.generation_policy",
        payload["generation_policy"],
    )
    _require_exact_keys(
        "generation request payload.generation_policy",
        policy,
        (
            "schema",
            "version",
            "nsteps",
            "guidance_w",
            "ag_ratio",
            "n_recycle",
            "search_algorithm",
            "self_cond",
            "model",
        ),
    )
    if (
        policy["schema"] != "dive-directional-generation-policy-v1"
        or policy["version"] != 1
        or type(policy["version"]) is not int
        or policy["nsteps"] != 100
        or type(policy["nsteps"]) is not int
        or policy["guidance_w"] != 1.0
        or type(policy["guidance_w"]) is not float
        or policy["ag_ratio"] != 0.0
        or type(policy["ag_ratio"]) is not float
        or policy["n_recycle"] != 0
        or type(policy["n_recycle"]) is not int
        or policy["search_algorithm"] != "single-pass"
        or type(policy["self_cond"]) is not bool
        or not isinstance(policy["model"], Mapping)
        or not policy["model"]
    ):
        raise SealError("generation request policy differs from the frozen contract")
    _reject_raw_outcome_fields(payload)

def _validate_generation_execution(
    cells: Sequence[GenerationCell],
    inventory: Mapping[str, object],
    loader_records: Sequence[Mapping[str, object]] | None = None,
) -> None:
    def reject_callable(value: object) -> None:
        if callable(value):
            raise SealError("generation execution data cannot contain a callable")
        if isinstance(value, Mapping):
            for key, nested in value.items():
                reject_callable(key)
                reject_callable(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                reject_callable(nested)

    def reject_outcomes(value: object) -> None:
        forbidden = {
            "generation_cells",
            "generation_results",
            "loss",
            "non_finite_count",
            "non_finite_events",
            "outcome",
            "statistics",
        }
        if isinstance(value, Mapping):
            if forbidden & {str(key) for key in value}:
                raise SealError(
                    "generation raw payload contains a forbidden outcome field"
                )
            for nested in value.values():
                reject_outcomes(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                reject_outcomes(nested)

    planned = inventory.get("generation_cells")
    availability = _require_mapping(
        "blind_inventory.availability_manifest", inventory.get("availability_manifest")
    )
    exclusions = availability.get("deterministic_exclusions")
    if not isinstance(planned, Sequence) or isinstance(planned, (str, bytes)):
        raise SealError("generation execution inventory is malformed")
    if not isinstance(exclusions, Sequence) or isinstance(exclusions, (str, bytes)):
        raise SealError("generation availability exclusions are malformed")
    excluded = {
        str(_require_mapping("generation exclusion", item)["cell_hash"])
        for item in exclusions
    }
    expected = {
        str(_require_mapping("planned generation cell", item)["cell_hash"]): dict(item)
        for item in planned
        if str(_require_mapping("planned generation cell", item)["cell_hash"])
        not in excluded
    }
    observed: dict[str, dict[str, object]] = {}
    for cell in cells:
        if type(cell) is not GenerationCell or not isinstance(cell.request, Mapping):
            raise SealError(
                "generation execution must contain data-only GenerationCell records"
            )
        reject_callable(cell.request)
        identity = {
            "family": cell.family,
            "parent_id": cell.parent_id,
            "seed": cell.seed,
            "arm": cell.arm,
            "cell_hash": cell.cell_hash,
        }
        _require_exact_keys(
            "generation request", cell.request, (*identity.keys(), "payload")
        )
        if {key: cell.request[key] for key in identity} != identity:
            raise SealError("generation request identity differs from its sealed cell")
        if not isinstance(cell.request["payload"], Mapping):
            raise SealError("generation request payload must be a data-only mapping")
        reject_outcomes(cell.request["payload"])
        _validate_generation_payload(cell, cell.request["payload"], loader_records)
        if cell.cell_hash in observed or expected.get(cell.cell_hash) != identity:
            raise SealError(
                "generation execution must form a bijection with available sealed cells"
            )
        observed[cell.cell_hash] = identity
    if observed != expected:
        raise SealError(
            "generation execution must form a bijection with available sealed cells"
        )

_EXECUTION_SOURCE_PREFIXES = MappingProxyType(
    {
        "dive": "src/dive",
        "upstream": "src/proteinfoundation",
    }
)

def _archive_python_sources(
    root: Path,
    commit: str,
    prefix: str,
    *,
    repository_name: str,
    git_reader_path: str | None = None,
) -> Mapping[str, bytes]:

    reader = git_reader_path or "git"
    try:
        result = subprocess.run(
            (reader, "archive", "--format=tar", commit, prefix),
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise SealError(
            f"cannot execute the git object reader for {repository_name}: {error}"
        ) from error
    if result.returncode != 0:
        raise SealError(
            f"cannot retain authenticated {repository_name} execution sources"
        )
    sources: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            for member in archive.getmembers():
                relative = Path(member.name)
                if not member.isfile() or relative.suffix != ".py":
                    continue
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or relative.as_posix() != member.name
                    or not relative.is_relative_to(Path(prefix))
                ):
                    raise SealError(
                        f"authenticated {repository_name} source archive is unsafe"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise SealError(
                        f"authenticated {repository_name} source archive is incomplete"
                    )
                raw = stream.read()
                if not raw and relative.name != "__init__.py":
                    raise SealError(f"authenticated {repository_name} source is empty")
                sources[relative.as_posix()] = raw
    except (OSError, tarfile.TarError) as error:
        raise SealError(
            f"authenticated {repository_name} source archive is invalid"
        ) from error
    if not sources:
        raise SealError(f"authenticated {repository_name} source catalog is empty")
    return MappingProxyType(sources)

def _test_python_sources(root: Path, prefix: str) -> Mapping[str, bytes]:
    sources = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / prefix).rglob("*.py"))
        if path.is_file()
    }
    if not sources:
        raise SealError("test execution source catalog is incomplete")
    return MappingProxyType(sources)

def _authenticated_repository_source_catalog(
    seal: ConfirmationSeal, environment: _SealEnvironment
) -> Mapping[str, bytes]:
    repositories = _require_mapping("repositories", seal.record["repositories"])
    snapshots: dict[str, bytes] = {}
    for repository_name, prefix in _EXECUTION_SOURCE_PREFIXES.items():
        repository = _require_mapping(
            f"repositories.{repository_name}", repositories[repository_name]
        )
        if environment.repository_states is None:
            root = (
                environment.dive_repo_root
                if repository_name == "dive"
                else environment.upstream_repo_root
            )
            sources = _archive_python_sources(
                root,
                str(repository["commit"]),
                prefix,
                repository_name=repository_name,
                git_reader_path=environment.git_reader_path,
            )
        else:
            root = (
                _DIVE_READ_ROOT if repository_name == "dive" else EMERGENT_UPSTREAM_ROOT
            )
            sources = _test_python_sources(root, prefix)
        for relative_name, raw in sources.items():
            snapshots[f"execution_source:{repository_name}:{relative_name}"] = raw
    return MappingProxyType(snapshots)

def _verified_blind_inputs(
    seal: ConfirmationSeal, environment: _SealEnvironment
) -> VerifiedBlindInputs:
    metric_sources = _verify_metric_source_identity(
        seal.record["metric_source"], environment.metric_read_root
    )
    bindings = _require_mapping("file_bindings", seal.record["file_bindings"])
    snapshots: dict[str, bytes] = {
        f"metric_source:{name}": raw for name, raw in metric_sources.items()
    }
    execution_sources = _authenticated_repository_source_catalog(seal, environment)
    snapshots.update(execution_sources)
    for name, raw in metric_sources.items():
        if execution_sources.get(f"execution_source:dive:{name}") != raw:
            raise SealError(
                "metric source differs from the authenticated repository catalog"
            )
    inventory = _require_mapping("blind_inventory", seal.record["blind_inventory"])
    snapshots["sealed_strict_flow_inventory"] = (
        _canonical_bytes(inventory["strict_flow_inventory"]) + b"\n"
    )
    snapshots["sealed_generation_cells"] = (
        _canonical_bytes(inventory["generation_cells"]) + b"\n"
    )
    snapshots["sealed_availability_manifest"] = (
        _canonical_bytes(inventory["availability_manifest"]) + b"\n"
    )
    for name, raw_identity in bindings.items():
        identity = _require_mapping(f"file_bindings.{name}", raw_identity)
        data, metadata = _safe_file_bytes(
            Path(str(identity["path"])),
            name=f"file_bindings.{name}",
            environment=environment,
        )
        if (
            metadata.st_dev != identity["device"]
            or metadata.st_ino != identity["inode"]
            or len(data) != identity["size_bytes"]
            or hashlib.sha256(data).hexdigest() != identity["sha256"]
        ):
            raise SealError(f"file_bindings.{name} changed at loader boundary")
        snapshots[str(name)] = data
    artifacts: dict[str, VerifiedArtifact] = {}
    artifact_records = _require_mapping(
        "selection_artifacts", seal.record["selection_artifacts"]
    )
    try:
        for name, raw_spec in artifact_records.items():
            spec = _require_mapping(f"selection_artifacts.{name}", raw_spec)
            artifact = _open_selection_artifact(str(name), spec, environment)
            if artifact.device != spec["device"] or artifact.inode != spec["inode"]:
                if artifact.descriptor >= 0:
                    os.close(artifact.descriptor)
                raise SealError(
                    f"selection_artifacts.{name} changed at loader boundary"
                )
            artifacts[str(name)] = artifact
    except BaseException:
        for artifact in artifacts.values():
            if artifact.descriptor >= 0:
                os.close(artifact.descriptor)
        raise
    return VerifiedBlindInputs(
        seal.record_sha256,
        MappingProxyType(snapshots),
        MappingProxyType(artifacts),
        environment.metric_read_root,
    )

def _append_output_record(
    seal: ConfirmationSeal,
    result: Mapping[str, object],
    environment: _SealEnvironment,
    state: _AppendState,
) -> dict[str, object]:

    output = _require_mapping("output_manifest", seal.record["output_manifest"])
    path = Path(str(output["path"]))
    flags = (
        os.O_RDWR
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd, leaf = _open_terminal_parent(path, environment, create=False)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent_fd)
        except OSError as error:
            raise SealError(
                f"output_manifest cannot be opened safely: {error}"
            ) from error
        metadata = os.fstat(descriptor)
        expected_size = int(output["size_bytes"])
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != output["device"]
            or metadata.st_ino != output["inode"]
            or metadata.st_size != expected_size
            or metadata.st_gid != environment.expected_gid
        ):
            raise SealError("output_manifest no longer has its sealed append prefix")
        prefix = os.pread(descriptor, expected_size, 0)
        if (
            len(prefix) != expected_size
            or hashlib.sha256(prefix).hexdigest() != output["sha256"]
        ):
            raise SealError("output_manifest prefix differs from its sealed identity")
        record = {
            "schema_version": 1,
            "seal_sha256": seal.record_sha256,
            "blind_seal_hash": seal.blind_seal_hash,
            "strict_inventory_hash": seal.strict_inventory_hash,
            "generation_inventory_hash": seal.generation_inventory_hash,
            "availability_manifest_hash": seal.availability_manifest_hash,
            "flow_examples": result["flow_examples"],
            "blind_checks": result["blind_checks"],
            "generation_cells": result["generation_cells"],
            "verdict": result["verdict"],
        }
        raw = _canonical_bytes(record) + b"\n"
        offset = expected_size
        state.offset = offset
        state.length = len(raw)
        state.sha256 = hashlib.sha256(raw).hexdigest()
        state.phase = "writing"
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SealError("output append made no forward progress")
            state.bytes_written += written
            view = view[written:]
        state.phase = "written"
        os.fsync(descriptor)
        state.durable = True
        state.phase = "durable"
        os.fsync(parent_fd)
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            os.close(parent_fd)
    return {
        "offset": offset,
        "length": len(raw),
        "sha256": state.sha256,
    }

def _reserve_verdict_publication(
    seal: ConfirmationSeal, environment: _SealEnvironment
) -> Path:
    output = _require_mapping("output_manifest", seal.record["output_manifest"])
    path = Path(str(output["path"]))
    reservation = path.with_name(path.name + ".reservation.json")
    payload = {
        "schema_version": 1,
        "seal_sha256": seal.record_sha256,
        "blind_seal_hash": seal.blind_seal_hash,
        "output_device": output["device"],
        "output_inode": output["inode"],
        "output_size_bytes": output["size_bytes"],
        "output_sha256": output["sha256"],
    }
    _write_exclusive(reservation, _canonical_bytes(payload) + b"\n", environment)
    return reservation

def _publish_verdict_disposition(
    reservation: Path,
    *,
    status: str,
    seal: ConfirmationSeal,
    environment: _SealEnvironment,
    error: str | None = None,
    append: Mapping[str, object] | None = None,
) -> None:
    if status not in {"success", "failure"}:
        raise SealError("verdict disposition status is invalid")
    destination = reservation.with_name(
        reservation.name.removesuffix(".reservation.json") + f".{status}.json"
    )
    reservation_raw, _ = _safe_file_bytes(
        reservation, name="verdict reservation", environment=environment
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "seal_sha256": seal.record_sha256,
        "reservation": str(reservation),
        "reservation_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        "error": error,
        "append": dict(append) if append is not None else None,
    }
    if status == "success" and append is None:
        raise SealError("success disposition requires the exact append identity")
    if status == "failure" and append is not None:
        raise SealError("failure disposition cannot bind an append")
    _write_exclusive(destination, _canonical_bytes(payload) + b"\n", environment)

def _verify_confirmation_result(
    seal: ConfirmationSeal, environment: _SealEnvironment
) -> ConfirmationResult:
    _verify_seal_publication(seal.path, seal.record_sha256, environment)
    seal_raw, _ = _safe_file_bytes(
        seal.path, name="confirmation result seal", environment=environment
    )
    if (
        hashlib.sha256(seal_raw).hexdigest() != seal.record_sha256
        or seal_raw != _canonical_bytes(seal.record) + b"\n"
    ):
        raise SealError("confirmation result seal object differs from canonical bytes")
    sidecar, _ = _safe_file_bytes(
        seal.path.with_name(seal.path.name + ".sha256"),
        name="confirmation result seal sidecar",
        environment=environment,
    )
    if sidecar != f"{seal.record_sha256}  {seal.path.name}\n".encode():
        raise SealError("confirmation result seal sidecar is inconsistent")
    rebound = _seal_from_record(seal.path, seal.record_sha256, seal.record)
    if (
        rebound.blind_seal_hash != seal.blind_seal_hash
        or rebound.strict_inventory_hash != seal.strict_inventory_hash
        or rebound.generation_inventory_hash != seal.generation_inventory_hash
        or rebound.availability_manifest_hash != seal.availability_manifest_hash
    ):
        raise SealError("confirmation result seal trust roots are inconsistent")
    output = _require_mapping("output_manifest", seal.record["output_manifest"])
    output_path = Path(str(output["path"]))
    reservation_path = output_path.with_name(output_path.name + ".reservation.json")
    success_path = output_path.with_name(output_path.name + ".success.json")
    failure_path = output_path.with_name(output_path.name + ".failure.json")
    failure_parent, failure_leaf = _open_terminal_parent(
        failure_path, environment, create=False
    )
    try:
        try:
            failure_fd = os.open(
                failure_leaf,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=failure_parent,
            )
        except FileNotFoundError:
            failure_fd = None
        if failure_fd is not None:
            os.close(failure_fd)
            raise SealError(
                "confirmation result has a contradictory failure disposition"
            )
    finally:
        os.close(failure_parent)
    try:
        reservation_raw, _ = _safe_file_bytes(
            reservation_path, name="verdict reservation", environment=environment
        )
        success_raw, _ = _safe_file_bytes(
            success_path, name="verdict success", environment=environment
        )
    except SealError as error:
        raise SealError(
            "confirmation result success transaction is incomplete"
        ) from error
    try:
        reservation = json.loads(reservation_raw)
        success = json.loads(success_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError(
            "confirmation result disposition is not canonical JSON"
        ) from error
    if reservation_raw != _canonical_bytes(reservation) + b"\n":
        raise SealError("confirmation result reservation is not canonical")
    expected_reservation = {
        "schema_version": 1,
        "seal_sha256": seal.record_sha256,
        "blind_seal_hash": seal.blind_seal_hash,
        "output_device": output["device"],
        "output_inode": output["inode"],
        "output_size_bytes": output["size_bytes"],
        "output_sha256": output["sha256"],
    }
    if reservation != expected_reservation:
        raise SealError("confirmation result reservation differs from the seal")
    if success_raw != _canonical_bytes(success) + b"\n":
        raise SealError("confirmation result success disposition is not canonical")
    append = _require_mapping("confirmation result append", success.get("append"))
    _require_exact_keys(
        "confirmation result append", append, ("offset", "length", "sha256")
    )
    offset = append["offset"]
    length = append["length"]
    digest = _require_hash("confirmation result append sha256", append["sha256"])
    if (
        type(offset) is not int
        or offset != output["size_bytes"]
        or type(length) is not int
        or length <= 0
    ):
        raise SealError(
            "confirmation result append bounds differ from the sealed prefix"
        )
    expected_success = {
        "schema_version": 1,
        "status": "success",
        "seal_sha256": seal.record_sha256,
        "reservation": str(reservation_path),
        "reservation_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        "error": None,
        "append": dict(append),
    }
    if success != expected_success:
        raise SealError("confirmation result success disposition is inconsistent")
    output_raw, metadata = _safe_file_bytes(
        output_path, name="confirmation output", environment=environment
    )
    if metadata.st_dev != output["device"] or metadata.st_ino != output["inode"]:
        raise SealError("confirmation output identity changed")
    prefix = output_raw[:offset]
    segment = output_raw[offset : offset + length]
    if (
        len(output_raw) != offset + length
        or hashlib.sha256(prefix).hexdigest() != output["sha256"]
        or hashlib.sha256(segment).hexdigest() != digest
    ):
        raise SealError("confirmation output segment differs from success disposition")
    try:
        record = json.loads(segment)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("confirmation output segment is not JSON") from error
    if segment != _canonical_bytes(record) + b"\n":
        raise SealError("confirmation output segment is not canonical")
    parsed_record = _require_mapping("confirmation output segment", record)
    _require_exact_keys(
        "confirmation output segment",
        parsed_record,
        (
            "schema_version",
            "seal_sha256",
            "blind_seal_hash",
            "strict_inventory_hash",
            "generation_inventory_hash",
            "availability_manifest_hash",
            "flow_examples",
            "blind_checks",
            "generation_cells",
            "verdict",
        ),
    )
    if parsed_record["schema_version"] != 1:
        raise SealError("confirmation output schema_version must be exactly one")
    if parsed_record.get("seal_sha256") != seal.record_sha256:
        raise SealError("confirmation output segment differs from the exact seal")
    for field, expected in (
        ("blind_seal_hash", seal.blind_seal_hash),
        ("strict_inventory_hash", seal.strict_inventory_hash),
        ("generation_inventory_hash", seal.generation_inventory_hash),
        ("availability_manifest_hash", seal.availability_manifest_hash),
    ):
        if parsed_record[field] != expected:
            raise SealError(f"confirmation output {field} differs from the seal")
    metric_sources = _verify_metric_source_identity(
        seal.record["metric_source"], environment.metric_read_root
    )
    metric_inputs = VerifiedBlindInputs(
        seal.record_sha256,
        MappingProxyType(
            {f"metric_source:{name}": raw for name, raw in metric_sources.items()}
        ),
        MappingProxyType({}),
        environment.metric_read_root,
    )
    _authenticate_metric_execution(metric_inputs)
    report = _base_blind_report(seal)
    for field in ("flow_examples", "blind_checks", "generation_cells"):
        report[field] = json.loads(_canonical_bytes(parsed_record[field]))
    try:
        replayed = _ORIGINAL_BLIND_VERDICT(
            report,
            expected_seal_hash=seal.blind_seal_hash,
            expected_strict_inventory_hash=seal.strict_inventory_hash,
            expected_generation_inventory_hash=seal.generation_inventory_hash,
            expected_availability_manifest_hash=seal.availability_manifest_hash,
        )
    except DirectionalConfirmationError as error:
        raise SealError(
            "confirmation output raw evidence does not revalidate"
        ) from error
    _authenticate_metric_execution(metric_inputs)
    expected_verdict = _verdict_record(replayed)
    if parsed_record["verdict"] != expected_verdict:
        raise SealError("confirmation output verdict differs from raw-evidence replay")
    return ConfirmationResult(
        seal.record_sha256,
        offset,
        length,
        digest,
        _freeze_json(parsed_record),
    )

def verify_confirmation_result(seal: ConfirmationSeal) -> ConfirmationResult:

    if not isinstance(seal, ConfirmationSeal):
        raise SealError("confirmation result requires an exact verified seal")
    return _verify_confirmation_result(seal, _environment())

def _verify_confirmation_result_test_only(
    seal: ConfirmationSeal, test_environment: _TestOnlyEnvironment
) -> ConfirmationResult:
    return _verify_confirmation_result(
        seal, _test_only_seal_environment(test_environment)
    )

def _verify_later_authorization(
    path: Path,
    expected_hash: str,
    seal: ConfirmationSeal,
    environment: _SealEnvironment,
) -> None:
    expected = _require_hash("authorization sha256", expected_hash)
    raw, _ = _safe_file_bytes(
        path, name="authorization record", environment=environment
    )
    if not raw.endswith(b"\n") or hashlib.sha256(raw).hexdigest() != expected:
        raise SealError("authorization record hash is not exact")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("authorization record is not valid JSON") from error
    if raw != _canonical_bytes(record) + b"\n":
        raise SealError("authorization record is not canonical")
    expected_record = {
        "schema_version": 1,
        "action": "open-directional-test-blind",
        "authorized": True,
        "seal_path": str(seal.path),
        "seal_sha256": seal.record_sha256,
        "blind_seal_hash": seal.blind_seal_hash,
    }
    if record != expected_record:
        raise SealError("authorization record does not authorize this exact seal")

def build_blind_loader(_inputs: VerifiedBlindInputs) -> BlindEvaluationData:

    runtime = importlib.import_module("dive.evaluation.directional_runtime")
    return runtime.construct_reviewed_data(_inputs)

def _load_reviewed_data(inputs: VerifiedBlindInputs) -> BlindEvaluationData:

    return build_blind_loader(inputs)

def _load_reviewed_model(
    inputs: VerifiedBlindInputs, data: BlindEvaluationData
) -> object:
    runtime = importlib.import_module("dive.evaluation.directional_runtime")
    return runtime.construct_reviewed_runtime(inputs, data)

def _construct_reviewed_components(
    inputs: VerifiedBlindInputs,
    *,
    load_data: Callable[[VerifiedBlindInputs], object],
    load_model: Callable[[VerifiedBlindInputs, BlindEvaluationData], object],
    seal_data: Callable[[BlindEvaluationData], BlindEvaluationData] | None = None,
) -> tuple[BlindEvaluationData, object]:

    try:
        data = load_data(inputs)
        if type(data) is not BlindEvaluationData:
            raise SealError(
                "blind loader must return exact data-only BlindEvaluationData"
            )
        if seal_data is not None:
            data = seal_data(data)
        model = load_model(inputs, data)
        return data, model
    finally:
        for artifact in inputs.artifacts.values():
            if artifact.descriptor >= 0:
                os.close(artifact.descriptor)

def _construct_reviewed_components_test_only(
    inputs: VerifiedBlindInputs,
    *,
    load_data: Callable[[VerifiedBlindInputs], object],
    load_model: Callable[[VerifiedBlindInputs, BlindEvaluationData], object],
) -> tuple[BlindEvaluationData, object]:
    return _construct_reviewed_components(
        inputs, load_data=load_data, load_model=load_model
    )

def _generation_result(cell: GenerationCell, value: object) -> dict[str, object]:
    result = _require_mapping("generation sampler result", value)
    _require_exact_keys(
        "generation sampler result",
        result,
        ("loss", "non_finite_count", "non_finite_events"),
    )
    loss = _require_mapping("generation sampler result.loss", result["loss"])
    _require_exact_keys(
        "generation sampler result.loss", loss, ("numerator", "denominator")
    )
    numerator = loss["numerator"]
    denominator = loss["denominator"]
    if (
        type(numerator) not in (int, float)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise SealError("generation sampler loss is invalid")
    count = result["non_finite_count"]
    events = result["non_finite_events"]
    if (
        type(count) is not int
        or count < 0
        or not isinstance(events, Sequence)
        or isinstance(events, (str, bytes))
    ):
        raise SealError("generation sampler non-finite outcome is invalid")
    if count != len(events) or any(
        not isinstance(event, str) or not event for event in events
    ):
        raise SealError("generation sampler non-finite outcome is inconsistent")
    return {
        "family": cell.family,
        "parent_id": cell.parent_id,
        "seed": cell.seed,
        "arm": cell.arm,
        "cell_hash": cell.cell_hash,
        "loss": dict(loss),
        "non_finite_count": count,
        "non_finite_events": list(events),
    }

def _reject_raw_outcome_fields(value: object) -> None:
    if isinstance(value, Mapping):
        illegal = _OUTCOME_FIELD_NAMES & {str(key) for key in value}
        if illegal:
            raise SealError(
                f"exact raw flow schema contains outcome fields {sorted(illegal)}"
            )
        for nested in value.values():
            _reject_raw_outcome_fields(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _reject_raw_outcome_fields(nested)
    elif callable(value) or isinstance(value, Path):
        raise SealError("exact raw flow schema contains an executable or path")

def _validate_raw_flow_envelope(row: RawFlowRow) -> None:

    if type(row) is not RawFlowRow or type(row.request) not in (dict, MappingProxyType):
        raise SealError("exact raw flow schema requires an exact RawFlowRow")
    identity = row.identity.to_mapping()
    _require_exact_keys(
        "exact raw flow schema",
        row.request,
        (*identity.keys(), "corruption_seed", "payload"),
    )
    if {key: row.request[key] for key in identity} != identity:
        raise SealError("exact raw flow schema identity differs from the sealed row")
    if (
        type(row.request["corruption_seed"]) is not int
        or row.request["corruption_seed"] != 42
    ):
        raise SealError("exact raw flow schema corruption seed differs from the seal")
    payload = row.request["payload"]
    if type(payload) not in (dict, MappingProxyType):
        raise SealError("exact raw flow schema payload must be an exact mapping")
    _require_exact_keys("exact raw flow schema payload", payload, ("batch",))
    batch = payload["batch"]
    if type(batch) not in (dict, MappingProxyType):
        raise SealError("exact raw flow schema batch must be an exact mapping")
    _reject_raw_outcome_fields(row.request)

def _seal_loader_data(
    data: BlindEvaluationData,
    inventory: Mapping[str, object],
    loader_records: Sequence[Mapping[str, object]] | None = None,
) -> BlindEvaluationData:
    strict = _require_mapping(
        "blind_inventory.strict_flow_inventory", inventory["strict_flow_inventory"]
    )
    raw_rows = strict["rows"]
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise SealError("strict flow inventory rows are malformed")
    expected = {
        str(_require_mapping("strict row", row)["cell_hash"]): dict(row)
        for row in raw_rows
    }
    observed: dict[str, dict[str, object]] = {}
    for row in data.flow_rows:
        _validate_raw_flow_envelope(row)
        identity = {
            "family": row.family,
            "example_id": row.example_id,
            "parent_id": row.parent_id,
            "cell_hash": row.cell_hash,
        }
        if row.cell_hash in observed or expected.get(row.cell_hash) != identity:
            raise SealError("flow rows must form a bijection with sealed strict rows")
        observed[row.cell_hash] = identity
    if observed != expected:
        raise SealError("flow rows must form a bijection with sealed strict rows")
    _validate_generation_execution(
        data.generation_cells, inventory, loader_records=loader_records
    )
    flow_rows = tuple(data.flow_rows)
    cells = tuple(
        GenerationCell(
            cell.family,
            cell.parent_id,
            cell.seed,
            cell.arm,
            cell.cell_hash,
            _freeze_json(json.loads(_canonical_bytes(cell.request))),
        )
        for cell in data.generation_cells
    )
    return BlindEvaluationData(flow_rows, cells)

def _validate_flow_arm_schedule(arms: Sequence[str]) -> None:
    if tuple(arms) != FROZEN_FLOW_ARMS or len(set(arms)) != len(FROZEN_FLOW_ARMS):
        raise SealError("flow arm schedule must contain each exact sealed arm once")

def _base_blind_report(seal: ConfirmationSeal) -> dict[str, object]:
    preblind = json.loads(_canonical_bytes(seal.record["preblind"]))
    selection = _require_mapping("preblind.selection", preblind["selection"])
    development = _require_mapping(
        "preblind.development_report", preblind["development_report"]
    )
    candidates = development["candidates"]
    selected = next(
        _require_mapping("selected candidate", candidate)
        for candidate in candidates
        if _require_mapping("candidate", candidate).get("seed") == selection["seed"]
    )
    inventory = _require_mapping("blind_inventory", seal.record["blind_inventory"])
    return {
        "schema_version": 1,
        "confirmation": selected["confirmation"],
        "seal_payload": json.loads(_canonical_bytes(seal.record["blind_seal_payload"])),
        "seal_hash": seal.blind_seal_hash,
        "preblind": preblind,
        "strict_flow_inventory": json.loads(
            _canonical_bytes(inventory["strict_flow_inventory"])
        ),
        "generation_inventory": json.loads(
            _canonical_bytes(inventory["generation_inventory"])
        ),
        "availability_manifest": json.loads(
            _canonical_bytes(inventory["availability_manifest"])
        ),
    }

def _verdict_record(verdict: object) -> dict[str, object]:

    diagnostics = _require_mapping(
        "blind verdict diagnostics", getattr(verdict, "diagnostics")
    )
    _require_exact_keys("blind verdict diagnostics", diagnostics, ("blind_checks",))
    blind_checks = _require_mapping(
        "blind verdict diagnostics.blind_checks", diagnostics["blind_checks"]
    )
    _require_exact_keys(
        "blind verdict diagnostics.blind_checks",
        blind_checks,
        (
            "validity_passed",
            "preservation_passed",
            "equal_family_ratio",
            "family_ratios",
        ),
    )
    if (
        type(blind_checks["validity_passed"]) is not bool
        or type(blind_checks["preservation_passed"]) is not bool
        or type(blind_checks["equal_family_ratio"]) is not float
    ):
        raise SealError("blind verdict diagnostics have noncanonical scalar types")
    family_ratios = blind_checks["family_ratios"]
    if (
        not isinstance(family_ratios, Sequence)
        or isinstance(family_ratios, (str, bytes))
        or tuple(item[0] for item in family_ratios) != FAMILIES
        or any(
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not float
            for item in family_ratios
        )
    ):
        raise SealError("blind verdict family diagnostics are not exact")
    canonical_diagnostics = json.loads(_canonical_bytes(diagnostics))
    return {
        "check_schema_version": 2,
        "passed": bool(getattr(verdict, "passed")),
        "terminal": bool(getattr(verdict, "terminal")),
        "claim_label": str(getattr(verdict, "claim_label")),
        "reasons": list(getattr(verdict, "reasons")),
        "conditions": dict(getattr(verdict, "conditions")),
        "diagnostics": canonical_diagnostics,
    }

def _render_confirmed_command(args: argparse.Namespace) -> str:
    pieces = [
        str(RUNTIME_PYTHON),
        "scripts/emergent/evaluate_directional.py",
        "--seal",
        str(args.seal),
        "--seal-sha256",
        args.seal_sha256,
        "--confirm-blind",
        "--authorization",
        "/ABSOLUTE/PATH/TO/LATER-AUTHORIZATION.json",
        "--authorization-sha256",
        "LATER_AUTHORIZATION_SHA256",
    ]
    return shlex.join(pieces)

_MAX_WORKER_FRAME_BYTES = 1 << 30
_WORKER_ENVELOPE_KEYS = frozenset({"protocol_version", "payload"})

_RAW_RESULT_KEYS = frozenset(
    {
        "kind",
        "catalog_identity",
        "meta_path_additions",
        "schema",
        "length",
        "sha256",
        "raw_base64",
    }
)
_REPLAY_VERDICT_KEYS = frozenset(
    {"kind", "catalog_identity", "meta_path_additions", "raw_sha256", "verdict"}
)
RAW_RESULT_SCHEMA = "dive-directional-raw-v2"

def _decode_worker_frame(raw: bytes) -> bytes:

    if type(raw) is not bytes or len(raw) < 8:
        raise SealError("worker frame is shorter than its length header")
    length = int.from_bytes(raw[:8], "big")
    if length <= 0 or length > _MAX_WORKER_FRAME_BYTES:
        raise SealError("worker frame length is not exact")
    if len(raw) < 8 + length:
        raise SealError("worker frame body is shorter than declared")
    if len(raw) > 8 + length:
        raise SealError("worker stream carries extra bytes after one frame")
    return raw[8 : 8 + length]

def _encode_worker_body(payload: Mapping[str, object]) -> bytes:

    return json.dumps(
        {"protocol_version": WORKER_PROTOCOL_VERSION, "payload": dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

def _worker_payload(body: bytes, *, kind: str) -> Mapping[str, object]:

    try:
        envelope = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("worker frame is not UTF-8 canonical JSON") from error
    if type(envelope) is not dict or set(envelope) != _WORKER_ENVELOPE_KEYS:
        raise SealError("worker frame keys are not exact")
    if envelope["protocol_version"] != WORKER_PROTOCOL_VERSION:
        raise SealError("worker protocol version differs")
    payload = envelope["payload"]
    if type(payload) is not dict:
        raise SealError("worker payload is not an exact mapping")

    if body != _encode_worker_body(payload):
        raise SealError("worker frame is not canonical: duplicate or reordered keys")
    if payload.get("kind") != kind:
        raise SealError(f"worker payload kind is not exactly {kind}")
    expected = _RAW_RESULT_KEYS if kind == "raw_result" else _REPLAY_VERDICT_KEYS
    _require_exact_keys(f"worker {kind} payload", payload, expected)
    return payload

def _worker_raw_bytes(payload: Mapping[str, object], *, catalog_identity: str) -> bytes:

    if payload["catalog_identity"] != catalog_identity:
        raise SealError("execution worker reported another catalog identity")
    if payload["schema"] != RAW_RESULT_SCHEMA:
        raise SealError("execution worker raw schema is not exact")
    encoded = payload["raw_base64"]
    if type(encoded) is not str:
        raise SealError("execution worker raw payload is not exact")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise SealError("execution worker raw payload is not exact base64") from error
    if _exact_int("worker raw length", payload["length"]) != len(raw):
        raise SealError("execution worker raw length differs from its bytes")
    if _require_hash("worker raw sha256", payload["sha256"]) != (
        hashlib.sha256(raw).hexdigest()
    ):
        raise SealError("execution worker raw digest differs from its bytes")
    return raw

_CONFIRMATION_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "seal_path",
        "seal_sha256",
        "authorization_path",
        "authorization_sha256",
        "evidence_root",
        "expected_gid",
        "dive_repo_root",
        "upstream_repo_root",
        "git_reader_path",
        "catalog_identity",
        "in_process_workers",
        "repository_states",
        "dive_read_root",
    }
)

@dataclass(frozen=True, slots=True)
class _ExecutionOverrides:

    data: "BlindEvaluationData | None" = None
    runtime: object | None = None
    generation_sampler: Callable[[object, Mapping[str, object]], object] | None = None

def _confirmation_request_bytes(
    args: argparse.Namespace,
    seal: ConfirmationSeal,
    environment: _SealEnvironment,
) -> bytes:
    authentication = _require_mapping(
        "execution_authentication", seal.record["execution_authentication"]
    )
    reader = _require_mapping(
        "execution_authentication.git_object_reader",
        authentication["git_object_reader"],
    )
    catalog = _require_mapping(
        "execution_authentication.source_catalog", authentication["source_catalog"]
    )
    request = {
        "schema_version": 1,
        "seal_path": str(Path(args.seal).resolve()),
        "seal_sha256": seal.record_sha256,
        "authorization_path": str(Path(args.authorization).resolve()),
        "authorization_sha256": str(args.authorization_sha256),
        "evidence_root": str(environment.evidence_root),
        "expected_gid": int(environment.expected_gid),
        "dive_repo_root": str(environment.dive_repo_root),
        "dive_read_root": str(environment.metric_read_root),
        "upstream_repo_root": str(environment.upstream_repo_root),
        "git_reader_path": str(reader["path"]),
        "catalog_identity": str(catalog["identity_sha256"]),
        "in_process_workers": bool(environment.in_process_workers),

        "repository_states": (
            None
            if environment.repository_states is None
            else {
                name: list(state)
                for name, state in sorted(environment.repository_states.items())
            }
        ),
    }
    return _canonical_bytes(request) + b"\n"

def _realm_seal_and_environment(
    seal_bytes: bytes, request_bytes: bytes
) -> tuple[ConfirmationSeal, _SealEnvironment, Mapping[str, object]]:

    if type(seal_bytes) is not bytes or type(request_bytes) is not bytes:
        raise SealError("realm inputs must be exact bytes")
    try:
        request = json.loads(request_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("confirmation request is not canonical JSON") from error
    request = _require_mapping("confirmation request", request)
    _require_exact_keys("confirmation request", request, _CONFIRMATION_REQUEST_KEYS)
    if request["schema_version"] != 1:
        raise SealError("confirmation request schema_version must be exactly 1")
    if request_bytes != _canonical_bytes(request) + b"\n":
        raise SealError("confirmation request is not canonical JSON")
    in_process = bool(request["in_process_workers"])
    states = request["repository_states"]
    if states is not None and not in_process:
        raise SealError("a worker never takes the parent's repository state")
    if hashlib.sha256(seal_bytes).hexdigest() != request["seal_sha256"]:
        raise SealError("seal bytes differ from the request's exact seal digest")
    if states is not None:
        states = MappingProxyType(
            {str(name): (str(row[0]), bool(row[1])) for name, row in states.items()}
        )
    try:
        parsed = json.loads(seal_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("seal bytes are not canonical JSON") from error
    parsed = _require_mapping("seal", parsed)

    authentication = _validate_execution_authentication(
        parsed.get("execution_authentication")
    )
    git_reader_path = _verify_sealed_git_reader(authentication)
    if str(request["git_reader_path"]) != git_reader_path:
        raise SealError("request git reader differs from the sealed binding")
    if not in_process:
        _assert_authenticated_request_roots(request, authentication)
    if os.environ.get(WORKER_ROLE_VARIABLE) and in_process:
        raise SealError("a worker realm never accepts the in-process test seam")
    environment = _SealEnvironment(
        Path(str(request["evidence_root"])),
        int(request["expected_gid"]),
        Path(str(request["dive_repo_root"])),
        Path(str(request["upstream_repo_root"])),
        states,
        in_process,
        git_reader_path,
        in_process,
        Path(str(request["dive_read_root"])),
    )
    record = _validate_payload(parsed, environment)
    if _canonical_bytes(record) + b"\n" != seal_bytes:
        raise SealError("seal record differs from its exact canonical bytes")
    catalog = _require_mapping(
        "execution_authentication.source_catalog",
        _require_mapping(
            "execution_authentication", record["execution_authentication"]
        )["source_catalog"],
    )
    if catalog["identity_sha256"] != request["catalog_identity"]:
        raise SealError("request catalog identity differs from the sealed binding")
    seal = _seal_from_record(
        Path(str(request["seal_path"])), str(request["seal_sha256"]), record
    )
    return seal, environment, MappingProxyType(dict(request))

def _execute_confirmation(
    seal: ConfirmationSeal,
    environment: _SealEnvironment,
    *,
    artifact_descriptors: Mapping[str, int] = MappingProxyType({}),
    overrides: _ExecutionOverrides | None = None,
) -> bytes:

    if overrides is None:
        overrides = _ExecutionOverrides()
    if type(overrides) is not _ExecutionOverrides:
        raise SealError("execution overrides are not the private exact type")
    if type(artifact_descriptors) is not MappingProxyType and not isinstance(
        artifact_descriptors, Mapping
    ):
        raise SealError("artifact descriptors are not an exact mapping")
    authentication = _require_mapping(
        "execution_authentication", seal.record["execution_authentication"]
    )
    _verify_external_runtime_inventory(
        _require_mapping(
            "execution_authentication.external_runtime_inventory",
            authentication["external_runtime_inventory"],
        ),
        environment,
    )
    catalog_binding = _require_mapping(
        "execution_authentication.source_catalog", authentication["source_catalog"]
    )
    runtime_module = importlib.import_module("dive.evaluation.directional_runtime")

    cache = runtime_module.RetainedModuleCache(
        catalog_identity=str(catalog_binding["identity_sha256"])
    )
    runtime_module._RETAINED_MODULE_CACHE_OWNER = cache

    _assert_realm_mutable_state_registered()
    inputs = _verified_blind_inputs(seal, environment)
    _authenticate_metric_execution(inputs)
    inventory = _require_mapping("blind_inventory", seal.record["blind_inventory"])
    loader_manifest = _validate_strict_blind_loader_manifest(
        inputs.snapshots["strict_blind_loader_manifest"], inventory
    )
    loader_records = loader_manifest["records"]
    if type(loader_records) is not list:
        raise SealError("strict blind loader records are not exact")

    def seal_loaded_data(loaded: BlindEvaluationData) -> BlindEvaluationData:
        inventory_bound = _seal_loader_data(
            loaded, inventory, loader_records=loader_records
        )
        return runtime_module.seal_loaded_data(inventory_bound)

    factory = (
        (lambda _inputs: overrides.data)
        if overrides.data is not None
        else _load_reviewed_data
    )
    data, runtime = _construct_reviewed_components(
        inputs,
        load_data=factory,
        load_model=(
            (lambda _inputs, _data: overrides.runtime)
            if overrides.runtime is not None
            else _load_reviewed_model
        ),
        seal_data=seal_loaded_data,
    )
    if type(runtime) is not runtime_module.DirectionalEvaluationRuntime:
        raise SealError("reviewed model loader must return an exact frozen runtime")
    evaluation = _require_mapping("evaluation", seal.record["evaluation"])
    fixed_route = str(evaluation["fixed_generation_route"])
    _validate_flow_arm_schedule(FROZEN_FLOW_ARMS)
    flow_examples, blind_checks = runtime_module.flow_records(runtime, data.flow_rows)
    generation_outputs = []
    for cell in data.generation_cells:
        result = execute_generation_arm(
            runtime.candidate_model,
            arm=cell.arm,
            fixed_route=fixed_route,
            sample=(
                (
                    lambda cell=cell: overrides.generation_sampler(
                        runtime.candidate_model, cell.request
                    )
                )
                if overrides.generation_sampler is not None
                else (
                    lambda cell=cell: runtime_module.sample_reviewed_generation(
                        runtime, cell
                    )
                )
            ),
        )
        generation_outputs.append(_generation_result(cell, result))
    report = _base_blind_report(seal)
    report["flow_examples"] = flow_examples
    report["blind_checks"] = blind_checks
    report["generation_cells"] = generation_outputs
    _authenticate_metric_execution(inputs)
    verdict = _ORIGINAL_BLIND_VERDICT(
        report,
        expected_seal_hash=seal.blind_seal_hash,
        expected_strict_inventory_hash=seal.strict_inventory_hash,
        expected_generation_inventory_hash=seal.generation_inventory_hash,
        expected_availability_manifest_hash=seal.availability_manifest_hash,
    )
    _authenticate_metric_execution(inputs)
    return (
        _canonical_bytes(
            {
                "flow_examples": flow_examples,
                "blind_checks": blind_checks,
                "generation_cells": generation_outputs,
                "verdict": _verdict_record(verdict),
            }
        )
        + b"\n"
    )

def _raw_result_mapping(raw_bytes: bytes) -> Mapping[str, object]:
    if type(raw_bytes) is not bytes or not raw_bytes:
        raise SealError("canonical raw result bytes are not exact")
    try:
        parsed = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("canonical raw result is not canonical JSON") from error
    output = _require_mapping("canonical raw result", parsed)
    _require_exact_keys(
        "canonical raw result",
        output,
        ("flow_examples", "blind_checks", "generation_cells", "verdict"),
    )
    if raw_bytes != _canonical_bytes(output) + b"\n":
        raise SealError("canonical raw result is not canonical JSON")
    return output

def _replay_confirmation(
    seal: ConfirmationSeal, raw_bytes: bytes
) -> Mapping[str, object]:

    output = _raw_result_mapping(raw_bytes)
    report = _base_blind_report(seal)
    report["flow_examples"] = output["flow_examples"]
    report["blind_checks"] = output["blind_checks"]
    report["generation_cells"] = output["generation_cells"]
    verdict = _ORIGINAL_BLIND_VERDICT(
        report,
        expected_seal_hash=seal.blind_seal_hash,
        expected_strict_inventory_hash=seal.strict_inventory_hash,
        expected_generation_inventory_hash=seal.generation_inventory_hash,
        expected_availability_manifest_hash=seal.availability_manifest_hash,
    )
    return MappingProxyType(
        {
            "schema": RAW_RESULT_SCHEMA,
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "verdict": _verdict_record(verdict),
        }
    )

def execute_confirmation_in_realm(
    *,
    seal_bytes: bytes,
    request_bytes: bytes,
    artifact_descriptors: Mapping[str, int],
) -> bytes:

    seal, environment, _request = _realm_seal_and_environment(seal_bytes, request_bytes)
    return _execute_confirmation(
        seal, environment, artifact_descriptors=artifact_descriptors
    )

def replay_confirmation_in_realm(
    *, seal_bytes: bytes, request_bytes: bytes, raw_bytes: bytes
) -> Mapping[str, object]:

    seal, _environment, _request = _realm_seal_and_environment(
        seal_bytes, request_bytes
    )
    return _replay_confirmation(seal, raw_bytes)

_REALM_ENTRYPOINT_NAMES = (
    "_authenticate_metric_execution",
    "_drive_confirmation",
    "_execute_confirmation",
    "_replay_confirmation",
    "drive_confirmation_in_realm",
    "execute_confirmation_in_realm",
    "replay_confirmation_in_realm",
)

def _realm_entrypoint_codes() -> tuple[tuple[str, CodeType], ...]:
    module = sys.modules[__name__]
    return tuple(
        (name, vars(module)[name].__code__) for name in _REALM_ENTRYPOINT_NAMES
    )

def _assert_realm_entrypoints_authentic() -> None:

    module = sys.modules[__name__]
    for name, code in _REALM_ENTRYPOINT_BASELINE:
        live = vars(module).get(name)
        if not isinstance(live, FunctionType) or live.__code__ is not code:
            raise SealError(f"realm entrypoint {name} is not the authenticated one")

def _verify_appended_record(
    seal: ConfirmationSeal,
    environment: _SealEnvironment,
    append: Mapping[str, object],
    staged_raw: bytes,
) -> None:

    output = _require_mapping("output_manifest", seal.record["output_manifest"])
    path = Path(str(output["path"]))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd, leaf = _open_terminal_parent(path, environment, create=False)
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        raise SealError(f"appended record cannot be re-read safely: {error}") from error
    try:
        offset = int(append["offset"])
        length = int(append["length"])
        raw = os.pread(descriptor, length, offset)
    finally:
        os.close(descriptor)
        os.close(parent_fd)
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != append["sha256"]:
        raise SealError("appended record does not read back byte for byte")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealError("appended record is not canonical JSON") from error
    staged = _raw_result_mapping(staged_raw)
    for field in ("flow_examples", "blind_checks", "generation_cells", "verdict"):
        if _canonical_bytes(record.get(field)) != _canonical_bytes(staged[field]):
            raise SealError(
                f"appended record {field} differs from the staged raw result"
            )

class _InProcessWorkerDriver:

    __slots__ = ("_catalog_identity", "_overrides")

    def __init__(
        self, *, catalog_identity: str, overrides: _ExecutionOverrides
    ) -> None:
        self._catalog_identity = catalog_identity
        self._overrides = overrides

    def execute(self, *, seal_bytes: bytes, request_bytes: bytes) -> bytes:
        seal, environment, _request = _realm_seal_and_environment(
            seal_bytes, request_bytes
        )
        raw = _execute_confirmation(seal, environment, overrides=self._overrides)
        return _decode_worker_frame(
            _frame_worker_body(
                _encode_worker_body(
                    {
                        "kind": "raw_result",
                        "catalog_identity": self._catalog_identity,

                        "meta_path_additions": [],
                        "schema": RAW_RESULT_SCHEMA,
                        "length": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "raw_base64": base64.b64encode(raw).decode("ascii"),
                    }
                )
            )
        )

    def stage(self, raw_bytes: bytes) -> bytes:
        return raw_bytes

    def replay(
        self, *, seal_bytes: bytes, request_bytes: bytes, staged: bytes
    ) -> bytes:
        verdict = replay_confirmation_in_realm(
            seal_bytes=seal_bytes, request_bytes=request_bytes, raw_bytes=staged
        )
        return _decode_worker_frame(
            _frame_worker_body(
                _encode_worker_body(
                    {
                        "kind": "replay_verdict",
                        "catalog_identity": self._catalog_identity,
                        "meta_path_additions": [],
                        "raw_sha256": hashlib.sha256(staged).hexdigest(),
                        "verdict": dict(verdict),
                    }
                )
            )
        )

def _frame_worker_body(body: bytes) -> bytes:
    return len(body).to_bytes(8, "big") + body

class _ProcessWorkerDriver:

    __slots__ = (
        "_anchor",
        "_catalog_identity",
        "_catalog_fd",
        "_bootstrap_fd",
        "_staged_fd",
    )

    _CATALOG_TARGET = 3
    _REQUEST_TARGET = 4
    _SEAL_TARGET = 5
    _RAW_TARGET = 6

    def __init__(
        self,
        *,
        anchor: ModuleType,
        catalog_identity: str,
        catalog_fd: int,
        bootstrap_fd: int,
    ) -> None:
        self._anchor = anchor
        self._catalog_identity = catalog_identity
        self._staged_fd: int | None = None

        self._catalog_fd = catalog_fd
        self._bootstrap_fd = bootstrap_fd

    def _run(self, role: str, descriptors: dict[int, int]) -> bytes:
        handle = self._anchor.spawn_worker(
            bootstrap_fd=self._bootstrap_fd, role=role, descriptors=descriptors
        )
        try:
            return self._anchor.collect_worker(handle)
        except Exception as error:
            raise SealError(f"{role} worker refused: {error}") from error

    def execute(self, *, seal_bytes: bytes, request_bytes: bytes) -> bytes:
        seal_fd = self._anchor.seal_memory_bytes("seal", seal_bytes)
        request_fd = self._anchor.seal_memory_bytes("request", request_bytes)
        try:
            return self._run(
                "execute",
                {
                    self._CATALOG_TARGET: self._catalog_fd,
                    self._REQUEST_TARGET: request_fd,
                    self._SEAL_TARGET: seal_fd,
                },
            )
        finally:
            os.close(request_fd)
            os.close(seal_fd)

    def stage(self, raw_bytes: bytes) -> int:

        if self._staged_fd is not None:
            raise SealError("the staging descriptor is created exactly once")
        self._staged_fd = self._anchor.seal_memory_bytes("raw", raw_bytes)
        return self._staged_fd

    def replay(self, *, seal_bytes: bytes, request_bytes: bytes, staged: int) -> bytes:
        seal_fd = self._anchor.seal_memory_bytes("seal", seal_bytes)
        request_fd = self._anchor.seal_memory_bytes("request", request_bytes)
        try:
            return self._run(
                "replay",
                {
                    self._CATALOG_TARGET: self._catalog_fd,
                    self._REQUEST_TARGET: request_fd,
                    self._SEAL_TARGET: seal_fd,
                    self._RAW_TARGET: staged,
                },
            )
        finally:
            os.close(request_fd)
            os.close(seal_fd)

    def close(self) -> None:

        if self._staged_fd is not None:
            try:
                os.close(self._staged_fd)
            except OSError:
                pass
            self._staged_fd = None

def _load_bootstrap_anchor(
    record: Mapping[str, object], *, bootstrap_bytes: bytes | None = None
) -> ModuleType:

    bootstrap = _require_mapping(
        "execution_authentication.bootstrap", record["bootstrap"]
    )
    if bootstrap_bytes is None:
        verified = dict(_verify_execution_authentication_files(record))
    else:
        if len(bootstrap_bytes) != bootstrap["size_bytes"]:
            raise SealError("inherited bootstrap anchor size differs from the seal")
        if hashlib.sha256(bootstrap_bytes).hexdigest() != bootstrap["sha256"]:
            raise SealError("inherited bootstrap anchor digest differs from the seal")
        verified = {"bootstrap": bootstrap_bytes}
    name = "_dive_directional_bootstrap_anchor"
    module = ModuleType(name)
    module.__file__ = str(bootstrap["path"])

    sys.modules[name] = module
    code = compile(
        verified["bootstrap"], str(bootstrap["path"]), "exec", dont_inherit=True
    )
    exec(code, module.__dict__)
    for name in ("spawn_worker", "collect_worker", "seal_memory_bytes"):
        if not callable(vars(module).get(name)):
            raise SealError(f"bootstrap anchor does not expose {name}")
    return module

def _drive_confirmation(
    args: argparse.Namespace,
    environment: _SealEnvironment,
    *,
    driver: object | None = None,
    overrides: _ExecutionOverrides | None = None,
) -> str:

    seal = _verify_confirmation_seal(args.seal, args.seal_sha256, environment)
    if args.authorization is None or args.authorization_sha256 is None:
        raise SealError("explicit later authorization record and hash are required")
    _verify_later_authorization(
        args.authorization, args.authorization_sha256, seal, environment
    )
    seal = _verify_confirmation_seal(args.seal, args.seal_sha256, environment)
    _assert_realm_entrypoints_authentic()
    authentication = _require_mapping(
        "execution_authentication", seal.record["execution_authentication"]
    )
    catalog_identity = str(
        _require_mapping(
            "execution_authentication.source_catalog",
            authentication["source_catalog"],
        )["identity_sha256"]
    )
    seal_bytes, _ = _safe_file_bytes(
        Path(args.seal), name="confirmation seal", environment=environment
    )
    if hashlib.sha256(seal_bytes).hexdigest() != seal.record_sha256:
        raise SealError("confirmation seal changed after verification")
    request_bytes = _confirmation_request_bytes(args, seal, environment)
    if driver is None:
        driver = _InProcessWorkerDriver(
            catalog_identity=catalog_identity,
            overrides=overrides or _ExecutionOverrides(),
        )
    reservation = _reserve_verdict_publication(seal, environment)
    append_state: _AppendState | None = None
    try:
        execution = _worker_payload(
            driver.execute(seal_bytes=seal_bytes, request_bytes=request_bytes),
            kind="raw_result",
        )
        raw_bytes = _worker_raw_bytes(execution, catalog_identity=catalog_identity)
        staged = driver.stage(raw_bytes)
        replay = _worker_payload(
            driver.replay(
                seal_bytes=seal_bytes, request_bytes=request_bytes, staged=staged
            ),
            kind="replay_verdict",
        )
        if replay["catalog_identity"] != execution["catalog_identity"]:
            raise SealError(
                "execution and replay workers reported different catalog identity"
            )
        if replay["catalog_identity"] != catalog_identity:
            raise SealError("replay worker catalog identity differs from the seal")
        staged_digest = hashlib.sha256(raw_bytes).hexdigest()
        if replay["raw_sha256"] != staged_digest:
            raise SealError("replay worker read other staged raw bytes")
        output = _raw_result_mapping(raw_bytes)
        replayed = _require_mapping("replay verdict", replay["verdict"])
        _require_exact_keys(
            "replay verdict", replayed, ("schema", "raw_sha256", "verdict")
        )
        if replayed["schema"] != RAW_RESULT_SCHEMA:
            raise SealError("replay verdict schema is not exact")
        if replayed["raw_sha256"] != staged_digest:
            raise SealError("replay verdict binds other staged raw bytes")
        if _canonical_bytes(replayed["verdict"]) != _canonical_bytes(output["verdict"]):
            raise SealError("replay verdict differs from the executed verdict")
        append_state = _AppendState()
        append_identity = _append_output_record(seal, output, environment, append_state)
        _verify_appended_record(seal, environment, append_identity, raw_bytes)
        _publish_verdict_disposition(
            reservation,
            status="success",
            seal=seal,
            environment=environment,
            append=append_identity,
        )
        verified_result = _verify_confirmation_result(seal, environment)
    except BaseException as error:
        if _append_can_publish_failure(append_state):
            _publish_verdict_disposition(
                reservation,
                status="failure",
                seal=seal,
                environment=environment,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    finally:
        closer = getattr(driver, "close", None)
        if callable(closer):
            closer()
    return _canonical_bytes(verified_result.record).decode("utf-8")

def confirmation_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Evaluate the sealed directional blind confirmation"
    )
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--seal-sha256", required=True)
    parser.add_argument("--confirm-blind", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    return parser

def drive_confirmation_in_realm(
    *, seal_bytes: bytes, request_bytes: bytes, bootstrap_bytes: bytes, descriptors
) -> Mapping[str, object]:

    seal, environment, request = _realm_seal_and_environment(seal_bytes, request_bytes)
    authentication = _require_mapping(
        "execution_authentication", seal.record["execution_authentication"]
    )
    anchor = _load_bootstrap_anchor(authentication, bootstrap_bytes=bootstrap_bytes)
    driver = _ProcessWorkerDriver(
        anchor=anchor,
        catalog_identity=str(request["catalog_identity"]),
        catalog_fd=int(descriptors["catalog"]),
        bootstrap_fd=int(descriptors["bootstrap"]),
    )
    args = confirmation_argument_parser().parse_args(
        [
            "--seal",
            str(request["seal_path"]),
            "--seal-sha256",
            str(request["seal_sha256"]),
            "--confirm-blind",
            "--authorization",
            str(request["authorization_path"]),
            "--authorization-sha256",
            str(request["authorization_sha256"]),
        ]
    )
    record_text = _drive_confirmation(args, environment, driver=driver)
    return MappingProxyType({"kind": "confirmation", "record_text": record_text})

def _evaluate_main(
    argv: list[str] | None,
    environment: _SealEnvironment,
    *,
    test_only_data: BlindEvaluationData | None = None,
    test_only_runtime: object | None = None,
    test_only_generation_sampler: Callable[[object, Mapping[str, object]], object]
    | None = None,
    test_only_driver: object | None = None,
) -> int:

    args = confirmation_argument_parser().parse_args(argv)
    if not args.confirm_blind:
        print(_render_confirmed_command(args))
        print("nothing loaded")
        return 0
    if not environment.in_process_workers:
        raise SealError(
            "the confirmation entrypoint is scripts/emergent/evaluate_directional.py, "
            "which runs the seal-bound trust anchor; this module never orchestrates "
            "a confirmation from unauthenticated driver code"
        )
    print(
        _drive_confirmation(
            args,
            environment,
            driver=test_only_driver,
            overrides=_ExecutionOverrides(
                data=test_only_data,
                runtime=test_only_runtime,
                generation_sampler=test_only_generation_sampler,
            ),
        )
    )
    return 0

def _evaluate_main_test_only(
    argv: list[str] | None,
    test_environment: _TestOnlyEnvironment,
    *,
    blind_data: BlindEvaluationData | None = None,
    runtime: object | None = None,
    generation_sampler: Callable[[object, Mapping[str, object]], object] | None = None,
    driver: object | None = None,
) -> int:
    return _evaluate_main(
        argv,
        _test_only_seal_environment(test_environment),
        test_only_data=blind_data,
        test_only_runtime=runtime,
        test_only_generation_sampler=generation_sampler,
        test_only_driver=driver,
    )

_REALM_ENTRYPOINT_BASELINE = _realm_entrypoint_codes()

__all__ = [
    "BlindEvaluationData",
    "ConfirmationResult",
    "ConfirmationSeal",
    "FROZEN_GENERATION_ARMS",
    "FROZEN_GENERATION_SEEDS",
    "FlowRow",
    "GenerationCell",
    "EXTERNAL_RUNTIME_LEAVES",
    "GUARDED_MUTABLE_REALM_MODULES",
    "REQUIRED_SEAL_KEYS",
    "SEAL_SCHEMA_VERSION",
    "SealError",
    "VerifiedBlindInputs",
    "build_blind_inventory",
    "build_blind_loader",
    "canonical_record_hash",
    "create_confirmation_seal",
    "build_external_runtime_inventory",
    "drive_confirmation_in_realm",
    "execute_confirmation_in_realm",
    "execute_generation_arm",
    "replay_confirmation_in_realm",
    "verify_confirmation_seal",
    "verify_confirmation_result",
]
