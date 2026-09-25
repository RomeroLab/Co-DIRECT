
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from urllib.parse import parse_qsl, urlsplit

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.external_catalog import (
    CandidateDeclaration,
    ExternalCandidateId,
    ExternalCatalog,
    load_external_catalog,
)
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    RUNTIME_PYTHON,
)
from dive.training.preflight import canonical_json_bytes

class ExternalProvenanceError(RuntimeError):
    pass

class AuditDisposition(StrEnum):
    ELIGIBLE_FOR_ADAPTER = "eligible_for_adapter"
    ELIGIBLE_FOR_WEIGHT_ACQUISITION = "eligible_for_weight_acquisition"
    TASK_CONTRACT_UNREPRESENTABLE = "task_contract_unrepresentable"
    WEIGHTS_UNAVAILABLE = "weights_unavailable"
    LICENSE_UNAVAILABLE = "license_unavailable"
    SOURCE_UNAVAILABLE = "source_unavailable"
    PROVENANCE_FAILED = "provenance_failed"
    REVIEW_REQUIRED = "review_required"

class WeightContentClass(StrEnum):
    NOT_RETAINED = "not_retained"
    PYTORCH_ZIP = "pytorch_zip"
    PICKLE_STREAM = "pickle_stream"
    SAFETENSORS = "safetensors"
    HDF5 = "hdf5"
    HTML = "html"
    TEXT = "text"
    UNRECOGNIZED_BINARY = "unrecognized_binary"

class WeightLicenseStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    RESEARCH_EVALUATION_ALLOWED = "research_evaluation_allowed"
    UNAVAILABLE = "unavailable"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CITED_URL = re.compile(r"https?://[^\s<>\"'`]+")
_SECRET_KEY_FRAGMENTS = (
    "access_key",
    "api_key",
    "auth",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "header",
    "password",
    "secret",
    "signature",
    "signed",
    "token",
)
_BENIGN_WEIGHT_QUERY_KEYS = frozenset(
    {"confirm", "download", "export", "filename", "id", "raw"}
)
_LICENSE_REVIEW_CLASSES = frozenset(
    {
        "research_evaluation_allowed",
        "restricted_review_required",
        "unavailable",
    }
)
_ATTEMPT_SCHEMA = "dive-external-audit-attempt-v1"
_LEGACY_ATTEMPT_CATALOG_SCHEMA = "dive-external-baseline-candidates-v1"
_OBSERVATION_SCHEMA = "dive-external-candidate-observation-v1"
_COMPLETION_SCHEMA = "dive-external-audit-completion-v1"
_OBSERVATION_SEAL_SCHEMA = "dive-external-candidate-observation-seal-v1"
_SOURCE_RETRIEVAL_SCHEMA = "dive-external-source-retrieval-v1"
_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_PLAUSIBLE_WEIGHT_CLASSES = frozenset(
    {
        WeightContentClass.PYTORCH_ZIP,
        WeightContentClass.PICKLE_STREAM,
        WeightContentClass.SAFETENSORS,
        WeightContentClass.HDF5,
    }
)
_RFANTIBODY_WEIGHT_COMPONENTS = (
    (
        "rfdiffusion_ab",
        "https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt",
    ),
    (
        "proteinmpnn",
        "https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt",
    ),
    ("rf2_ab", "https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt"),
)
_WEIGHT_COMPONENTS_BY_CANDIDATE = MappingProxyType(
    {ExternalCandidateId.RFANTIBODY_H3_V1: _RFANTIBODY_WEIGHT_COMPONENTS}
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PINNED_CATALOG_PATHS = MappingProxyType(
    {
        "dive-external-baseline-candidates-v1": (
            _REPOSITORY_ROOT
            / "configs"
            / "emergent"
            / "external_baseline_candidates.yaml"
        ),
        "dive-external-baseline-candidates-v2": (
            _REPOSITORY_ROOT
            / "configs"
            / "emergent"
            / "external_baseline_candidates_v2.yaml"
        ),
    }
)
_GIT_BINARY = Path("/usr/bin/git")
_GIT_TIMEOUT_SECONDS = 300
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
_SAFE_COMMANDS = frozenset(
    {
        ("audit", "observe"),
        ("audit", "decide"),
        ("audit", "build-registry"),
    }
)

_TEST_ROOTS: tuple[Path, Path] | None = None

def _require_nonempty(label: str, value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ExternalProvenanceError(f"{label} must be a nonempty string")
    return value

def _require_sha256(label: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ExternalProvenanceError(f"{label} must be a lowercase sha256")
    return value

def _require_commit(label: str, value: object) -> str:
    if type(value) is not str or _GIT_COMMIT.fullmatch(value) is None:
        raise ExternalProvenanceError(f"{label} must be an exact 40-hex commit")
    return value

def _require_timestamp(label: str, value: object) -> str:
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        raise ExternalProvenanceError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ExternalProvenanceError(
            f"{label} must be an RFC3339 UTC timestamp"
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ExternalProvenanceError(f"{label} timestamp is not canonical")
    return value

def _path_is_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents

def _require_absolute_safe_path(label: str, value: object) -> Path:
    if type(value) is not str or not value:
        raise ExternalProvenanceError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExternalProvenanceError(f"{label} must be an absolute normalized path")
    normalized = Path(os.path.abspath(path))
    test_roots = _TEST_ROOTS
    if test_roots is not None and any(
        _path_is_below(normalized, Path(root)) for root in test_roots
    ):
        return normalized
    for denied in DENIED_PREFIXES:
        denied_path = Path(denied).resolve(strict=False)
        if _path_is_below(normalized, denied_path):
            raise ExternalProvenanceError(f"{label} is under a denied root")
    return normalized

def _validate_artifact(label: str, artifact: object) -> ArtifactIdentity:
    if not isinstance(artifact, ArtifactIdentity):
        raise ExternalProvenanceError(f"{label} must be an ArtifactIdentity")
    _require_absolute_safe_path(f"{label} path", artifact.path)
    _require_sha256(f"{label} sha256", artifact.sha256)
    if type(artifact.size_bytes) is not int or artifact.size_bytes < 0:
        raise ExternalProvenanceError(f"{label} size must be nonnegative")
    return artifact

def _validate_https_url(label: str, value: object) -> str:
    url = _require_nonempty(label, value)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ExternalProvenanceError(f"{label} must be a credential-free HTTPS URL")
    return url

def _validate_weight_url(label: str, value: object) -> str:
    url = _require_nonempty(label, value)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ExternalProvenanceError(f"{label} must be a credential-free HTTPS URL")
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered_key = key.lower()
        lowered_value = value.lower()
        if lowered_key not in _BENIGN_WEIGHT_QUERY_KEYS or any(
            fragment in lowered_key or fragment in lowered_value
            for fragment in _SECRET_KEY_FRAGMENTS
        ):
            raise ExternalProvenanceError(
                f"{label} contains a secret or unsupported query parameter"
            )
    return url

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ExternalProvenanceError(f"{label} has invalid keys")
    if any(type(key) is not str for key in value):
        raise ExternalProvenanceError(f"{label} keys must be strings")
    return value

def _artifact_from_mapping(value: object, label: str) -> ArtifactIdentity:
    payload = _exact_mapping(value, frozenset({"path", "sha256", "size_bytes"}), label)
    artifact = ArtifactIdentity(
        path=payload["path"],
        sha256=payload["sha256"],
        size_bytes=payload["size_bytes"],
    )
    return _validate_artifact(label, artifact)

def _optional_artifact_from_mapping(
    value: object, label: str
) -> ArtifactIdentity | None:
    return None if value is None else _artifact_from_mapping(value, label)

def _tuple_from_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ExternalProvenanceError(f"{label} must be a list")
    return _validate_string_tuple(label, tuple(value))

def _validate_tracked_paths(label: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ExternalProvenanceError(f"{label} must be a tuple")
    previous: str | None = None
    for item in value:
        if type(item) is not str or not item or "\x00" in item:
            raise ExternalProvenanceError(f"{label} contains an invalid path")
        path = Path(item)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != item
        ):
            raise ExternalProvenanceError(f"{label} contains an unsafe path")
        if previous is not None and item <= previous:
            raise ExternalProvenanceError(
                f"{label} must be strictly ordered without duplicates"
            )
        previous = item
    return value

def _tracked_paths_from_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ExternalProvenanceError(f"{label} must be a list")
    return _validate_tracked_paths(label, tuple(value))

def _validate_stat_identity(
    label: str, value: object
) -> tuple[int, int, int, int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 6
        or any(type(item) is not int or item < 0 for item in value)
        or value[1] < 1
    ):
        raise ExternalProvenanceError(f"{label} is invalid")
    return value

def _parse_enum(enum_type, value: object, label: str):
    if type(value) is not str:
        raise ExternalProvenanceError(f"{label} is invalid")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ExternalProvenanceError(f"{label} is invalid") from error

@dataclass(frozen=True, slots=True)
class SourceObservation:
    url: str
    observed_ref: str
    commit: str
    checkout_path: str
    tracked_tree_sha256: str
    retrieval_log: ArtifactIdentity
    observed_at: str
    checkout_device: int
    checkout_inode: int
    git_binary: ArtifactIdentity
    git_binary_stat: tuple[int, int, int, int, int, int]
    git_lfs_pointer_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_https_url("source URL", self.url)
        _require_nonempty("source observed ref", self.observed_ref)
        _require_commit("source commit", self.commit)
        _require_absolute_safe_path("source checkout path", self.checkout_path)
        _require_sha256("source tracked-tree hash", self.tracked_tree_sha256)
        _validate_artifact("source retrieval log", self.retrieval_log)
        _require_timestamp("source observation", self.observed_at)
        if (
            type(self.checkout_device) is not int
            or self.checkout_device < 0
            or type(self.checkout_inode) is not int
            or self.checkout_inode < 1
        ):
            raise ExternalProvenanceError("source checkout identity is invalid")
        _validate_artifact("source git binary", self.git_binary)
        _validate_stat_identity("source git binary stat", self.git_binary_stat)
        _validate_tracked_paths(
            "source Git-LFS pointer paths", self.git_lfs_pointer_paths
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "url": self.url,
            "observed_ref": self.observed_ref,
            "commit": self.commit,
            "checkout_path": self.checkout_path,
            "tracked_tree_sha256": self.tracked_tree_sha256,
            "retrieval_log": _identity_mapping(self.retrieval_log),
            "observed_at": self.observed_at,
            "checkout_identity": {
                "device": self.checkout_device,
                "inode": self.checkout_inode,
            },
            "git_binary": _identity_mapping(self.git_binary),
            "git_binary_stat": list(self.git_binary_stat),
            "git_lfs_pointer_paths": list(self.git_lfs_pointer_paths),
        }

    @classmethod
    def from_mapping(cls, value: object) -> SourceObservation:
        payload = _exact_mapping(
            value,
            frozenset(
                {
                    "url",
                    "observed_ref",
                    "commit",
                    "checkout_path",
                    "tracked_tree_sha256",
                    "retrieval_log",
                    "observed_at",
                    "checkout_identity",
                    "git_binary",
                    "git_binary_stat",
                    "git_lfs_pointer_paths",
                }
            ),
            "source observation",
        )
        checkout_identity = _exact_mapping(
            payload["checkout_identity"],
            frozenset({"device", "inode"}),
            "source checkout identity",
        )
        raw_git_stat = payload["git_binary_stat"]
        if not isinstance(raw_git_stat, list):
            raise ExternalProvenanceError("source git binary stat must be a list")
        return cls(
            url=payload["url"],
            observed_ref=payload["observed_ref"],
            commit=payload["commit"],
            checkout_path=payload["checkout_path"],
            tracked_tree_sha256=payload["tracked_tree_sha256"],
            retrieval_log=_artifact_from_mapping(
                payload["retrieval_log"], "source retrieval log"
            ),
            observed_at=payload["observed_at"],
            checkout_device=checkout_identity["device"],
            checkout_inode=checkout_identity["inode"],
            git_binary=_artifact_from_mapping(
                payload["git_binary"], "source git binary"
            ),
            git_binary_stat=_validate_stat_identity(
                "source git binary stat", tuple(raw_git_stat)
            ),
            git_lfs_pointer_paths=_tracked_paths_from_list(
                payload["git_lfs_pointer_paths"], "source Git-LFS pointer paths"
            ),
        )

@dataclass(frozen=True, slots=True)
class LicenseObservation:
    artifact: ArtifactIdentity | None
    review_class: str
    observed_at: str

    def __post_init__(self) -> None:
        if self.artifact is not None:
            _validate_artifact("license artifact", self.artifact)
        if self.review_class not in _LICENSE_REVIEW_CLASSES:
            raise ExternalProvenanceError("license review class is unknown")
        if self.artifact is None and self.review_class != "unavailable":
            raise ExternalProvenanceError(
                "missing license artifact must be classified unavailable"
            )
        _require_timestamp("license observation", self.observed_at)

    def as_mapping(self) -> dict[str, object]:
        return {
            "artifact": (
                None if self.artifact is None else _identity_mapping(self.artifact)
            ),
            "review_class": self.review_class,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_mapping(cls, value: object) -> LicenseObservation:
        payload = _exact_mapping(
            value,
            frozenset({"artifact", "review_class", "observed_at"}),
            "license observation",
        )
        return cls(
            artifact=_optional_artifact_from_mapping(
                payload["artifact"], "license artifact"
            ),
            review_class=payload["review_class"],
            observed_at=payload["observed_at"],
        )

@dataclass(frozen=True, slots=True)
class WeightReference:
    url: str
    source_artifact: ArtifactIdentity
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        _validate_weight_url("weight reference URL", self.url)
        _validate_artifact("weight reference source", self.source_artifact)
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ExternalProvenanceError("weight reference line span is invalid")

    @property
    def line_number(self) -> int:
        return self.start_line

    def as_mapping(self) -> dict[str, object]:
        return {
            "url": self.url,
            "source_artifact": _identity_mapping(self.source_artifact),
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_mapping(cls, value: object) -> WeightReference:
        payload = _exact_mapping(
            value,
            frozenset({"url", "source_artifact", "start_line", "end_line"}),
            "weight reference",
        )
        return cls(
            url=payload["url"],
            source_artifact=_artifact_from_mapping(
                payload["source_artifact"], "weight reference source"
            ),
            start_line=payload["start_line"],
            end_line=payload["end_line"],
        )

@dataclass(frozen=True, slots=True)
class WeightObservation:
    official_url: str | None
    artifact: ArtifactIdentity | None
    unavailable_reason: str | None
    observed_at: str
    citation: WeightReference | None
    content_class: WeightContentClass

    def __post_init__(self) -> None:
        if self.official_url is not None:
            _validate_weight_url("weight official URL", self.official_url)
            if self.citation is None or self.citation.url != self.official_url:
                raise ExternalProvenanceError(
                    "weight official URL requires its exact source citation"
                )
        elif self.citation is not None:
            raise ExternalProvenanceError("weight citation requires an official URL")
        if not isinstance(self.content_class, WeightContentClass):
            raise ExternalProvenanceError("weight content class is invalid")
        if self.artifact is not None:
            _validate_artifact("weight artifact", self.artifact)
            if self.official_url is None:
                raise ExternalProvenanceError(
                    "weight artifact requires an official URL"
                )
            if self.content_class in _PLAUSIBLE_WEIGHT_CLASSES:
                if self.unavailable_reason is not None:
                    raise ExternalProvenanceError(
                        "plausible checkpoint bytes cannot be unavailable"
                    )
            elif self.unavailable_reason is None:
                raise ExternalProvenanceError(
                    "non-checkpoint weight bytes require an unavailable reason"
                )
        elif (
            self.unavailable_reason is None
            or self.content_class is not WeightContentClass.NOT_RETAINED
        ):
            raise ExternalProvenanceError(
                "unretained weight requires typed unavailability and not_retained content"
            )
        if self.unavailable_reason is not None:
            reason = _require_nonempty(
                "weight unavailable reason", self.unavailable_reason
            )
            if not re.fullmatch(r"[a-z0-9][a-z0-9_]{1,63}", reason):
                raise ExternalProvenanceError(
                    "weight unavailable reason must be a typed reason code"
                )
        _require_timestamp("weight observation", self.observed_at)

    def as_mapping(self) -> dict[str, object]:
        return {
            "official_url": self.official_url,
            "artifact": (
                None if self.artifact is None else _identity_mapping(self.artifact)
            ),
            "unavailable_reason": self.unavailable_reason,
            "observed_at": self.observed_at,
            "citation": None if self.citation is None else self.citation.as_mapping(),
            "content_class": self.content_class.value,
        }

    @classmethod
    def from_mapping(cls, value: object) -> WeightObservation:
        payload = _exact_mapping(
            value,
            frozenset(
                {
                    "official_url",
                    "artifact",
                    "unavailable_reason",
                    "observed_at",
                    "citation",
                    "content_class",
                }
            ),
            "weight observation",
        )
        return cls(
            official_url=payload["official_url"],
            artifact=_optional_artifact_from_mapping(
                payload["artifact"], "weight artifact"
            ),
            unavailable_reason=payload["unavailable_reason"],
            observed_at=payload["observed_at"],
            citation=(
                None
                if payload["citation"] is None
                else WeightReference.from_mapping(payload["citation"])
            ),
            content_class=_parse_enum(
                WeightContentClass, payload["content_class"], "weight content class"
            ),
        )

@dataclass(frozen=True, slots=True)
class SourceSpan:
    source_artifact: ArtifactIdentity
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        _validate_artifact("source span artifact", self.source_artifact)
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ExternalProvenanceError("source span line range is invalid")

    def as_mapping(self) -> dict[str, object]:
        return {
            "source_artifact": _identity_mapping(self.source_artifact),
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_mapping(cls, value: object) -> SourceSpan:
        payload = _exact_mapping(
            value,
            frozenset({"source_artifact", "start_line", "end_line"}),
            "source span",
        )
        return cls(
            source_artifact=_artifact_from_mapping(
                payload["source_artifact"], "source span artifact"
            ),
            start_line=payload["start_line"],
            end_line=payload["end_line"],
        )

@dataclass(frozen=True, slots=True)
class WeightComponentObservation:
    component_id: str
    official_url: str
    artifact: ArtifactIdentity | None
    unavailable_reason: str | None
    observed_at: str
    citation: WeightReference | None
    content_class: WeightContentClass
    license_status: WeightLicenseStatus
    license_evidence: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        _require_nonempty("weight component id", self.component_id)
        _validate_weight_url("weight component official URL", self.official_url)
        if self.citation is not None:
            if not isinstance(self.citation, WeightReference):
                raise ExternalProvenanceError("weight component citation is invalid")
            if self.citation.url != self.official_url:
                raise ExternalProvenanceError(
                    "weight component citation does not bind its official URL"
                )
        if not isinstance(self.content_class, WeightContentClass):
            raise ExternalProvenanceError("weight component content class is invalid")
        if self.artifact is not None:
            _validate_artifact("weight component artifact", self.artifact)
            if self.content_class in _PLAUSIBLE_WEIGHT_CLASSES:
                if self.unavailable_reason is not None:
                    raise ExternalProvenanceError(
                        "plausible component bytes cannot be unavailable"
                    )
            elif self.unavailable_reason is None:
                raise ExternalProvenanceError(
                    "non-checkpoint component bytes require an unavailable reason"
                )
        elif (
            self.unavailable_reason is None
            or self.content_class is not WeightContentClass.NOT_RETAINED
        ):
            raise ExternalProvenanceError(
                "unretained component requires typed unavailability and not_retained content"
            )
        if self.unavailable_reason is not None:
            reason = _require_nonempty(
                "weight component unavailable reason", self.unavailable_reason
            )
            if not re.fullmatch(r"[a-z0-9][a-z0-9_]{1,63}", reason):
                raise ExternalProvenanceError(
                    "weight component unavailable reason must be a typed reason code"
                )
        if not isinstance(self.license_status, WeightLicenseStatus):
            raise ExternalProvenanceError("weight component license status is invalid")
        if not isinstance(self.license_evidence, tuple) or any(
            not isinstance(item, SourceSpan) for item in self.license_evidence
        ):
            raise ExternalProvenanceError(
                "weight component license evidence is invalid"
            )
        _require_timestamp("weight component observation", self.observed_at)

    def as_mapping(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "official_url": self.official_url,
            "artifact": (
                None if self.artifact is None else _identity_mapping(self.artifact)
            ),
            "unavailable_reason": self.unavailable_reason,
            "observed_at": self.observed_at,
            "citation": None if self.citation is None else self.citation.as_mapping(),
            "content_class": self.content_class.value,
            "license_status": self.license_status.value,
            "license_evidence": [item.as_mapping() for item in self.license_evidence],
        }

    @classmethod
    def from_mapping(cls, value: object) -> WeightComponentObservation:
        payload = _exact_mapping(
            value,
            frozenset(
                {
                    "component_id",
                    "official_url",
                    "artifact",
                    "unavailable_reason",
                    "observed_at",
                    "citation",
                    "content_class",
                    "license_status",
                    "license_evidence",
                }
            ),
            "weight component observation",
        )
        if not isinstance(payload["license_evidence"], list):
            raise ExternalProvenanceError(
                "weight component license evidence is invalid"
            )
        return cls(
            component_id=payload["component_id"],
            official_url=payload["official_url"],
            artifact=_optional_artifact_from_mapping(
                payload["artifact"], "weight component artifact"
            ),
            unavailable_reason=payload["unavailable_reason"],
            observed_at=payload["observed_at"],
            citation=(
                None
                if payload["citation"] is None
                else WeightReference.from_mapping(payload["citation"])
            ),
            content_class=_parse_enum(
                WeightContentClass,
                payload["content_class"],
                "weight component content class",
            ),
            license_status=_parse_enum(
                WeightLicenseStatus,
                payload["license_status"],
                "weight component license status",
            ),
            license_evidence=tuple(
                SourceSpan.from_mapping(item) for item in payload["license_evidence"]
            ),
        )

@dataclass(frozen=True, slots=True)
class WeightBundleObservation:
    components: tuple[WeightComponentObservation, ...]
    non_task_references: tuple[WeightReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple) or any(
            not isinstance(item, WeightComponentObservation) for item in self.components
        ):
            raise ExternalProvenanceError("weight bundle components are invalid")
        if (
            tuple((item.component_id, item.official_url) for item in self.components)
            != _RFANTIBODY_WEIGHT_COMPONENTS
        ):
            raise ExternalProvenanceError(
                "weight bundle components are not the reviewed RFantibody bundle"
            )
        if not isinstance(self.non_task_references, tuple) or any(
            not isinstance(item, WeightReference) for item in self.non_task_references
        ):
            raise ExternalProvenanceError(
                "weight bundle non-task references are invalid"
            )
        component_urls = {item.official_url for item in self.components}
        if any(item.url in component_urls for item in self.non_task_references):
            raise ExternalProvenanceError(
                "weight bundle required citations cannot be non-task references"
            )

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "dive-external-weight-bundle-v1",
            "components": [item.as_mapping() for item in self.components],
            "non_task_references": [
                item.as_mapping() for item in self.non_task_references
            ],
        }

    @classmethod
    def from_mapping(cls, value: object) -> WeightBundleObservation:
        payload = _exact_mapping(
            value,
            frozenset({"schema_version", "components", "non_task_references"}),
            "weight bundle observation",
        )
        if payload["schema_version"] != "dive-external-weight-bundle-v1":
            raise ExternalProvenanceError("weight bundle observation schema is invalid")
        if not isinstance(payload["components"], list) or not isinstance(
            payload["non_task_references"], list
        ):
            raise ExternalProvenanceError("weight bundle observation lists are invalid")
        return cls(
            components=tuple(
                WeightComponentObservation.from_mapping(item)
                for item in payload["components"]
            ),
            non_task_references=tuple(
                WeightReference.from_mapping(item)
                for item in payload["non_task_references"]
            ),
        )

WeightEvidence = WeightObservation | WeightBundleObservation

@dataclass(frozen=True, slots=True)
class CapabilityJudgment:
    capability: str
    supported: bool
    source_artifact: ArtifactIdentity
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        _validate_string_tuple("capability judgment", (self.capability,))
        if type(self.supported) is not bool:
            raise ExternalProvenanceError("capability judgment status is invalid")
        _validate_artifact("capability judgment source", self.source_artifact)
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ExternalProvenanceError("capability judgment line span is invalid")

    def as_mapping(self) -> dict[str, object]:
        return {
            "capability": self.capability,
            "supported": self.supported,
            "source_artifact": _identity_mapping(self.source_artifact),
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_mapping(cls, value: object) -> CapabilityJudgment:
        payload = _exact_mapping(
            value,
            frozenset(
                {
                    "capability",
                    "supported",
                    "source_artifact",
                    "start_line",
                    "end_line",
                }
            ),
            "capability judgment",
        )
        return cls(
            capability=payload["capability"],
            supported=payload["supported"],
            source_artifact=_artifact_from_mapping(
                payload["source_artifact"], "capability judgment source"
            ),
            start_line=payload["start_line"],
            end_line=payload["end_line"],
        )

@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    source_commit: str
    source_tree_sha256: str
    judgments: tuple[CapabilityJudgment, ...]
    reviewed_at: str

    def __post_init__(self) -> None:
        _require_commit("capability source commit", self.source_commit)
        _require_sha256("capability source tree", self.source_tree_sha256)
        if not isinstance(self.judgments, tuple) or not self.judgments:
            raise ExternalProvenanceError("capability decision is empty")
        if any(not isinstance(item, CapabilityJudgment) for item in self.judgments):
            raise ExternalProvenanceError("capability decision judgment is invalid")
        if len({item.capability for item in self.judgments}) != len(self.judgments):
            raise ExternalProvenanceError("capability decision contains duplicates")
        _require_timestamp("capability review", self.reviewed_at)

    @property
    def supported_capabilities(self) -> tuple[str, ...]:
        return tuple(item.capability for item in self.judgments if item.supported)

    @property
    def unsupported_capabilities(self) -> tuple[str, ...]:
        return tuple(item.capability for item in self.judgments if not item.supported)

    @property
    def review_evidence(self) -> tuple[ArtifactIdentity, ...]:
        return tuple(item.source_artifact for item in self.judgments)

    def as_mapping(self) -> dict[str, object]:
        return {
            "source_commit": self.source_commit,
            "source_tree_sha256": self.source_tree_sha256,
            "judgments": [item.as_mapping() for item in self.judgments],
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_mapping(cls, value: object) -> CapabilityDecision:
        payload = _exact_mapping(
            value,
            frozenset(
                {
                    "source_commit",
                    "source_tree_sha256",
                    "judgments",
                    "reviewed_at",
                }
            ),
            "capability decision",
        )
        raw_judgments = payload["judgments"]
        if not isinstance(raw_judgments, list):
            raise ExternalProvenanceError("capability judgments must be a list")
        return cls(
            source_commit=payload["source_commit"],
            source_tree_sha256=payload["source_tree_sha256"],
            judgments=tuple(
                CapabilityJudgment.from_mapping(item) for item in raw_judgments
            ),
            reviewed_at=payload["reviewed_at"],
        )

def _validate_string_tuple(label: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ExternalProvenanceError(f"{label} must be a tuple")
    for item in value:
        if (
            type(item) is not str
            or not item
            or not re.fullmatch(r"[a-z0-9][a-z0-9_]{1,63}", item)
        ):
            raise ExternalProvenanceError(f"{label} contains an invalid value")
    if len(set(value)) != len(value):
        raise ExternalProvenanceError(f"{label} contains duplicates")
    return value

def classify_weight_content(raw: bytes) -> WeightContentClass:

    if not isinstance(raw, bytes) or not raw:
        return WeightContentClass.TEXT
    stripped = raw.lstrip()
    lower = stripped[:64].lower()
    if lower.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return WeightContentClass.HTML
    if raw.startswith(b"PK\x03\x04"):
        return WeightContentClass.PYTORCH_ZIP
    if raw.startswith(b"\x80") and len(raw) >= 2 and 2 <= raw[1] <= 5:
        return WeightContentClass.PICKLE_STREAM
    if raw.startswith(b"\x89HDF\r\n\x1a\n"):
        return WeightContentClass.HDF5
    if len(raw) >= 10:
        header_size = int.from_bytes(raw[:8], "little")
        if 2 <= header_size <= min(len(raw) - 8, 100_000_000):
            header = raw[8 : 8 + header_size]
            try:
                parsed = json.loads(header)
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            else:
                if isinstance(parsed, Mapping):
                    return WeightContentClass.SAFETENSORS
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return WeightContentClass.UNRECOGNIZED_BINARY
    if text.isprintable() or all(
        character.isprintable() or character.isspace() for character in text
    ):
        return WeightContentClass.TEXT
    return WeightContentClass.UNRECOGNIZED_BINARY

def _validate_source_citation(
    source: SourceObservation,
    artifact: ArtifactIdentity,
    start_line: int,
    end_line: int,
    expected_url: str | None,
    *,
    label: str,
) -> None:
    checkout = Path(source.checkout_path)
    artifact_path = Path(artifact.path)
    if not _path_is_below(artifact_path, checkout):
        raise ExternalProvenanceError(f"{label} is outside retained source")
    relative = artifact_path.relative_to(checkout)
    relative_text = relative.as_posix()
    if relative_text in source.git_lfs_pointer_paths:
        raise ExternalProvenanceError(f"{label} cannot cite a Git-LFS pointer")
    git_binary = Path(source.git_binary.path)
    stages = _tracked_source_stages(checkout, git_binary, source.git_binary_stat)
    if relative_text not in stages:
        raise ExternalProvenanceError(f"{label} must cite a tracked source file")
    raw = _read_regular_bytes_at(checkout, relative, label)
    actual = ArtifactIdentity(
        str(artifact_path), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    if actual != artifact:
        raise ExternalProvenanceError(f"{label} artifact identity drifted")
    expected_blob = stages[relative_text][1]
    actual_blob = hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw,
        usedforsecurity=False,
    ).hexdigest()
    if actual_blob != expected_blob:
        raise ExternalProvenanceError(f"{label} differs from retained source HEAD")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ExternalProvenanceError(f"{label} must cite UTF-8 text") from error
    if end_line > len(lines) or not any(
        line.strip() for line in lines[start_line - 1 : end_line]
    ):
        raise ExternalProvenanceError(f"{label} line span is invalid")
    if expected_url is not None:
        span = "\n".join(lines[start_line - 1 : end_line])
        normalized = re.sub(r"\\\s*\n\s*", "", span)
        normalized = re.sub(r"([\"'])\s*\1", "", normalized)
        cited_urls = tuple(
            _trim_cited_url(match.group(0)) for match in _CITED_URL.finditer(normalized)
        )
        if expected_url not in cited_urls:
            raise ExternalProvenanceError(f"{label} does not contain its exact URL")

def _trim_cited_url(url: str) -> str:
    url = url.rstrip(".,;:!?")
    for closing, opening in ((")", "("), ("]", "["), ("}", "{")):
        while url.endswith(closing) and url.count(closing) > url.count(opening):
            url = url[:-1]
    return url

@dataclass(frozen=True, slots=True)
class CandidateObservation:
    candidate_id: ExternalCandidateId
    source: SourceObservation | None
    license: LicenseObservation | None
    weight: WeightEvidence
    capability: CapabilityDecision | None
    disposition: AuditDisposition
    observed_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, ExternalCandidateId):
            raise ExternalProvenanceError("candidate observation id is invalid")
        if self.source is not None and not isinstance(self.source, SourceObservation):
            raise ExternalProvenanceError("candidate source observation is invalid")
        if self.license is not None and not isinstance(
            self.license, LicenseObservation
        ):
            raise ExternalProvenanceError("candidate license observation is invalid")
        if not isinstance(self.weight, (WeightObservation, WeightBundleObservation)):
            raise ExternalProvenanceError("candidate weight observation is invalid")
        if self.candidate_id is ExternalCandidateId.RFANTIBODY_H3_V1:
            if not isinstance(self.weight, WeightBundleObservation):
                raise ExternalProvenanceError(
                    "RFantibody candidate requires a weight bundle observation"
                )
        elif not isinstance(self.weight, WeightObservation):
            raise ExternalProvenanceError(
                "only RFantibody may use a weight bundle observation"
            )
        if self.capability is not None and not isinstance(
            self.capability, CapabilityDecision
        ):
            raise ExternalProvenanceError("candidate capability decision is invalid")
        if not isinstance(self.disposition, AuditDisposition):
            raise ExternalProvenanceError("candidate disposition is invalid")
        _require_timestamp("candidate observation", self.observed_at)

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": _OBSERVATION_SCHEMA,
            "candidate_id": self.candidate_id.value,
            "source": None if self.source is None else self.source.as_mapping(),
            "license": None if self.license is None else self.license.as_mapping(),
            "weight": self.weight.as_mapping(),
            "capability": (
                None if self.capability is None else self.capability.as_mapping()
            ),
            "disposition": self.disposition.value,
            "observed_at": self.observed_at,
        }

    @classmethod
    def from_mapping(cls, value: object) -> CandidateObservation:
        payload = _exact_mapping(
            value,
            frozenset(
                {
                    "schema_version",
                    "candidate_id",
                    "source",
                    "license",
                    "weight",
                    "capability",
                    "disposition",
                    "observed_at",
                }
            ),
            "candidate observation",
        )
        if payload["schema_version"] != _OBSERVATION_SCHEMA:
            raise ExternalProvenanceError("candidate observation schema is invalid")
        candidate_id = _parse_enum(
            ExternalCandidateId, payload["candidate_id"], "candidate id"
        )
        weight_payload = payload["weight"]
        if not isinstance(weight_payload, Mapping):
            raise ExternalProvenanceError("candidate weight observation is invalid")
        weight_keys = frozenset(weight_payload)
        singleton_keys = frozenset(
            {
                "official_url",
                "artifact",
                "unavailable_reason",
                "observed_at",
                "citation",
                "content_class",
            }
        )
        bundle_keys = frozenset({"schema_version", "components", "non_task_references"})
        if weight_keys == singleton_keys:
            weight: WeightEvidence = WeightObservation.from_mapping(weight_payload)
        elif weight_keys == bundle_keys:
            weight = WeightBundleObservation.from_mapping(weight_payload)
        else:
            raise ExternalProvenanceError(
                "candidate weight observation has invalid keys"
            )
        return cls(
            candidate_id=candidate_id,
            source=(
                None
                if payload["source"] is None
                else SourceObservation.from_mapping(payload["source"])
            ),
            license=(
                None
                if payload["license"] is None
                else LicenseObservation.from_mapping(payload["license"])
            ),
            weight=weight,
            capability=(
                None
                if payload["capability"] is None
                else CapabilityDecision.from_mapping(payload["capability"])
            ),
            disposition=_parse_enum(
                AuditDisposition, payload["disposition"], "audit disposition"
            ),
            observed_at=payload["observed_at"],
        )

@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int

    def as_mapping(self) -> dict[str, int]:
        return {"device": self.device, "inode": self.inode}

    @classmethod
    def from_mapping(cls, value: object, label: str) -> _DirectoryIdentity:
        payload = _exact_mapping(value, frozenset({"device", "inode"}), label)
        device = payload["device"]
        inode = payload["inode"]
        if type(device) is not int or device < 0 or type(inode) is not int or inode < 1:
            raise ExternalProvenanceError(f"{label} is invalid")
        return cls(device=device, inode=inode)

@dataclass(frozen=True, slots=True, weakref_slot=True)
class ExternalAuditRun:
    run_id: str
    evidence_dir: Path
    bulk_dir: Path
    sources_dir: Path
    attempt: ArtifactIdentity
    catalog_schema_version: str
    catalog_sha256: str
    candidate_ids: tuple[ExternalCandidateId, ...]
    declarations: tuple[CandidateDeclaration, ...]
    repo_commit: str
    argv: tuple[str, ...]
    directory_identities: tuple[tuple[str, _DirectoryIdentity], ...]
    observations: tuple[CandidateObservation, ...] = field(
        default=(), init=False, compare=False
    )
    completion: ArtifactIdentity | None = field(default=None, init=False, compare=False)

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        catalog: ExternalCatalog,
        attempt: ArtifactIdentity,
        evidence_dir: Path,
        bulk_dir: Path,
    ) -> ExternalAuditRun:
        _validate_closed_catalog(catalog)
        attempt_keys = frozenset(
            {
                "schema_version",
                "status",
                "run_id",
                "argv",
                "repo_commit",
                "observed_refs",
                "catalog_sha256",
                "runtime_python",
                "directories",
                "evidence_path",
                "bulk_path",
                "sources_path",
            }
        )
        if isinstance(value, Mapping) and "catalog_schema_version" in value:
            payload = _exact_mapping(
                value,
                attempt_keys | {"catalog_schema_version"},
                "audit attempt",
            )
            attempt_catalog_schema = _require_nonempty(
                "audit attempt catalog schema", payload["catalog_schema_version"]
            )
        else:
            payload = _exact_mapping(value, attempt_keys, "audit attempt")
            if catalog.schema_version != _LEGACY_ATTEMPT_CATALOG_SCHEMA:
                raise ExternalProvenanceError(
                    "audit attempt lacks an authenticated catalog schema"
                )
            attempt_catalog_schema = catalog.schema_version
        run_id = payload["run_id"]
        _validate_run_id(run_id)
        argv = payload["argv"]
        if not isinstance(argv, list):
            raise ExternalProvenanceError("audit attempt argv must be a list")
        command = _validate_argv(argv)
        sources_dir = evidence_dir / "sources"
        observed_refs = {
            item.candidate_id.value: item.observed_ref for item in catalog.candidates
        }
        raw_directories = _exact_mapping(
            payload["directories"],
            frozenset({"evidence", "bulk", "sources"}),
            "audit directory identities",
        )
        directory_identities = tuple(
            (
                name,
                _DirectoryIdentity.from_mapping(
                    raw_directories[name], f"{name} directory identity"
                ),
            )
            for name in ("evidence", "bulk", "sources")
        )
        if (
            payload["schema_version"] != _ATTEMPT_SCHEMA
            or payload["status"] != "BUILDING"
            or attempt_catalog_schema != catalog.schema_version
            or payload["observed_refs"] != observed_refs
            or payload["catalog_sha256"] != catalog.semantic_sha256
            or payload["runtime_python"] != str(Path(RUNTIME_PYTHON).resolve())
            or payload["evidence_path"] != str(evidence_dir)
            or payload["bulk_path"] != str(bulk_dir)
            or payload["sources_path"] != str(sources_dir)
            or attempt.path != str(evidence_dir / "attempt.json")
        ):
            raise ExternalProvenanceError("audit attempt does not bind its fixed run")
        repo_commit = _require_commit("repository commit", payload["repo_commit"])
        return cls(
            run_id=run_id,
            evidence_dir=evidence_dir,
            bulk_dir=bulk_dir,
            sources_dir=sources_dir,
            attempt=attempt,
            catalog_schema_version=catalog.schema_version,
            catalog_sha256=catalog.semantic_sha256,
            candidate_ids=tuple(item.candidate_id for item in catalog.candidates),
            declarations=catalog.candidates,
            repo_commit=repo_commit,
            argv=command,
            directory_identities=directory_identities,
        )

@dataclass(frozen=True, slots=True)
class _LiveAuditState:
    observation_seals: tuple[
        tuple[ExternalCandidateId, ArtifactIdentity, ArtifactIdentity], ...
    ] = ()
    completion: ArtifactIdentity | None = None

_LIVE_AUDIT_STATES: dict[
    int, tuple[weakref.ReferenceType[ExternalAuditRun], _LiveAuditState]
] = {}

def _release_live_audit_state(
    identity: int, released: weakref.ReferenceType[ExternalAuditRun]
) -> None:
    current = _LIVE_AUDIT_STATES.get(identity)
    if current is not None and current[0] is released:
        _LIVE_AUDIT_STATES.pop(identity, None)

def _register_live_audit_state(run: ExternalAuditRun, state: _LiveAuditState) -> None:
    identity = id(run)
    current = _LIVE_AUDIT_STATES.get(identity)
    if current is not None and current[0]() is not None:
        raise ExternalProvenanceError("audit controller identity collision")
    reference = weakref.ref(
        run,
        lambda released, identity=identity: _release_live_audit_state(
            identity, released
        ),
    )
    _LIVE_AUDIT_STATES[identity] = (reference, state)

def _live_audit_state(run: ExternalAuditRun) -> _LiveAuditState:
    current = _LIVE_AUDIT_STATES.get(id(run))
    if current is None or current[0]() is not run:
        raise ExternalProvenanceError(
            "audit controller has no live state for this exact run object"
        )
    return current[1]

def _replace_live_audit_state(run: ExternalAuditRun, state: _LiveAuditState) -> None:
    current = _LIVE_AUDIT_STATES.get(id(run))
    if current is None or current[0]() is not run:
        raise ExternalProvenanceError(
            "audit controller has no live state for this exact run object"
        )
    _LIVE_AUDIT_STATES[id(run)] = (current[0], state)

def claim_external_audit(
    run_id: str, argv: Sequence[str], catalog: ExternalCatalog
) -> ExternalAuditRun:

    _validate_run_id(run_id)
    command = _validate_argv(argv)
    _validate_closed_catalog(catalog)
    runtime_python = _require_runtime_python()
    evidence_root, bulk_root = _validate_fixed_roots()
    repo_commit = _current_clean_commit()
    candidate_ids = tuple(item.candidate_id for item in catalog.candidates)
    if len(candidate_ids) != 6 or len(set(candidate_ids)) != 6:
        raise ExternalProvenanceError(
            "audit catalog must contain exactly six candidates"
        )

    evidence_parent = _ensure_named_directory(evidence_root, "external_baselines")
    bulk_parent = _ensure_named_directory(bulk_root, "external_baselines")
    evidence_dir = evidence_parent / run_id
    bulk_dir = bulk_parent / run_id
    try:
        _mkdir_create_new(evidence_dir, "audit evidence run", root=evidence_root)
    except FileExistsError as error:
        raise ExternalProvenanceError(
            f"audit destination already exists: {error}"
        ) from error
    try:
        _mkdir_create_new(bulk_dir, "audit bulk run", root=bulk_root)
        candidates_dir = _mkdir_create_new(
            evidence_dir / "candidates",
            "audit candidates directory",
            root=evidence_root,
        )
        sources_dir = _mkdir_create_new(
            evidence_dir / "sources",
            "audit retained-source directory",
            root=evidence_root,
        )
        for candidate_id in candidate_ids:
            _mkdir_create_new(
                candidates_dir / candidate_id.value,
                f"candidate evidence directory {candidate_id.value}",
                root=evidence_root,
            )
    except FileExistsError as error:
        raise ExternalProvenanceError(
            f"audit destination already exists: {error}"
        ) from error
    except OSError as error:
        raise ExternalProvenanceError(f"cannot create audit run: {error}") from error

    directory_identities = (
        (
            "evidence",
            _directory_identity_at(evidence_root, Path("external_baselines") / run_id),
        ),
        (
            "bulk",
            _directory_identity_at(bulk_root, Path("external_baselines") / run_id),
        ),
        (
            "sources",
            _directory_identity_at(
                evidence_root, Path("external_baselines") / run_id / "sources"
            ),
        ),
    )

    attempt_record = {
        "schema_version": _ATTEMPT_SCHEMA,
        "status": "BUILDING",
        "run_id": run_id,
        "argv": list(command),
        "repo_commit": repo_commit,
        "observed_refs": {
            item.candidate_id.value: item.observed_ref for item in catalog.candidates
        },
        "catalog_sha256": catalog.semantic_sha256,
        "catalog_schema_version": catalog.schema_version,
        "runtime_python": str(runtime_python),
        "directories": {
            name: identity.as_mapping() for name, identity in directory_identities
        },
        "evidence_path": str(evidence_dir),
        "bulk_path": str(bulk_dir),
        "sources_path": str(sources_dir),
    }
    attempt_path = evidence_dir / "attempt.json"
    try:
        _recheck_execution_binding(repo_commit)
        attempt = _write_new_json(
            attempt_path,
            attempt_record,
            root=evidence_root,
        )
    except FileExistsError as error:
        raise ExternalProvenanceError("audit attempt already exists") from error
    except OSError as error:
        raise ExternalProvenanceError(
            f"cannot publish audit attempt; claimed run remains occupied: {error}"
        ) from error
    if (
        _read_identity_at(
            evidence_root,
            Path("external_baselines") / run_id / "attempt.json",
            "audit attempt",
        )
        != attempt
    ):
        raise ExternalProvenanceError("attempt record changed while it was written")
    run = ExternalAuditRun(
        run_id=run_id,
        evidence_dir=evidence_dir,
        bulk_dir=bulk_dir,
        sources_dir=sources_dir,
        attempt=attempt,
        catalog_schema_version=catalog.schema_version,
        catalog_sha256=catalog.semantic_sha256,
        candidate_ids=candidate_ids,
        declarations=catalog.candidates,
        repo_commit=repo_commit,
        argv=command,
        directory_identities=directory_identities,
    )
    _register_live_audit_state(run, _LiveAuditState())
    return run

def resolve_official_source(
    candidate: CandidateDeclaration,
    run: ExternalAuditRun,
    *,
    git_binary: Path,
) -> SourceObservation:

    if not isinstance(candidate, CandidateDeclaration):
        raise ExternalProvenanceError("source resolution requires a declaration")
    if candidate.observed_ref != "HEAD":
        raise ExternalProvenanceError("source observed_ref must be literal HEAD")
    _validate_https_url("source URL", candidate.url)
    _live_audit_state(run)
    _authenticate_run(run, require_building=True)
    declaration = _declaration_for_run(run, candidate.candidate_id)
    if candidate != declaration:
        raise ExternalProvenanceError(
            "source declaration differs from the claimed catalog"
        )
    binary_identity = _validate_git_binary(git_binary)
    binary_artifact = _git_binary_artifact(git_binary, binary_identity)
    checkout_relative = Path("sources") / candidate.candidate_id.value
    log_relative = Path("candidates") / candidate.candidate_id.value / "retrieval.log"
    if _entry_exists_at(run.evidence_dir, checkout_relative) or _entry_exists_at(
        run.evidence_dir, log_relative
    ):
        raise ExternalProvenanceError("source resolution slot is already occupied")

    commit = _resolve_remote_head(candidate.url, git_binary, binary_identity)
    checkout = run.sources_dir / candidate.candidate_id.value
    _run_source_git(
        git_binary,
        binary_identity,
        "clone source",
        "clone",
        "--no-checkout",
        "--filter=blob:none",
        candidate.url,
        str(checkout),
    )
    checkout_identity = _directory_identity_at(run.evidence_dir, checkout_relative)
    _run_source_git(
        git_binary,
        binary_identity,
        "detach source checkout",
        "-C",
        str(checkout),
        "checkout",
        "--detach",
        commit,
    )
    _verify_source_binding(
        candidate,
        checkout,
        commit,
        git_binary,
        binary_identity,
        checkout_identity,
        run,
    )
    tracked_tree_sha256, lfs_pointer_paths = _hash_retained_source_tree(
        checkout, git_binary, binary_identity
    )
    _verify_source_binding(
        candidate,
        checkout,
        commit,
        git_binary,
        binary_identity,
        checkout_identity,
        run,
    )
    if _resolve_remote_head(candidate.url, git_binary, binary_identity) != commit:
        raise ExternalProvenanceError("official source remote HEAD drifted")
    if (
        _validate_git_binary(git_binary) != binary_identity
        or _git_binary_artifact(git_binary, binary_identity) != binary_artifact
    ):
        raise ExternalProvenanceError("git binary identity drifted")

    observed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_payload = {
        "schema_version": _SOURCE_RETRIEVAL_SCHEMA,
        "candidate_id": candidate.candidate_id.value,
        "url": candidate.url,
        "observed_ref": candidate.observed_ref,
        "commit": commit,
        "checkout_path": str(checkout),
        "tracked_tree_sha256": tracked_tree_sha256,
        "git_lfs_pointer_paths": lfs_pointer_paths,
        "checkout_identity": checkout_identity.as_mapping(),
        "git_binary": _identity_mapping(binary_artifact),
        "git_binary_stat": list(binary_identity),
        "observed_at": observed_at,
    }
    try:
        _recheck_execution_binding(run.repo_commit)
        retrieval_log = _write_new_json(
            run.evidence_dir / log_relative,
            log_payload,
            root=run.evidence_dir,
        )
    except FileExistsError as error:
        raise ExternalProvenanceError(
            "source retrieval log slot is already occupied"
        ) from error
    if (
        _read_identity_at(run.evidence_dir, log_relative, "source retrieval log")
        != retrieval_log
    ):
        raise ExternalProvenanceError("source retrieval log drifted after write")
    if _directory_identity_at(run.evidence_dir, checkout_relative) != checkout_identity:
        raise ExternalProvenanceError("retained source directory identity drifted")
    return SourceObservation(
        url=candidate.url,
        observed_ref=candidate.observed_ref,
        commit=commit,
        checkout_path=str(checkout),
        tracked_tree_sha256=tracked_tree_sha256,
        retrieval_log=retrieval_log,
        observed_at=observed_at,
        checkout_device=checkout_identity.device,
        checkout_inode=checkout_identity.inode,
        git_binary=binary_artifact,
        git_binary_stat=binary_identity,
        git_lfs_pointer_paths=tuple(lfs_pointer_paths),
    )

def write_candidate_observation(
    run: ExternalAuditRun, observation: CandidateObservation
) -> ArtifactIdentity:

    _authenticate_run(run, require_building=True)
    live_state = _live_audit_state(run)
    if not isinstance(observation, CandidateObservation):
        raise ExternalProvenanceError("expected a CandidateObservation")
    if observation.candidate_id not in run.candidate_ids:
        raise ExternalProvenanceError("candidate observation is not in the catalog")
    if observation.candidate_id in {
        candidate_id for candidate_id, _, _ in live_state.observation_seals
    }:
        raise ExternalProvenanceError("duplicate candidate observation")
    declaration = _declaration_for_run(run, observation.candidate_id)
    _validate_observation_binding(run, declaration, observation)
    payload = observation.as_mapping()
    _reject_secret_like_keys(payload)
    destination = (
        run.evidence_dir
        / "candidates"
        / observation.candidate_id.value
        / "observation.json"
    )
    try:
        _recheck_execution_binding(run.repo_commit)
        identity = _write_new_json(destination, payload, root=run.evidence_dir)
    except FileExistsError as error:
        raise ExternalProvenanceError(
            f"candidate observation already exists: {observation.candidate_id.value}"
        ) from error
    observation_relative = destination.relative_to(run.evidence_dir)
    if (
        _read_identity_at(
            run.evidence_dir, observation_relative, "candidate observation"
        )
        != identity
    ):
        raise ExternalProvenanceError("candidate observation drifted after write")
    seal_path = destination.with_name("observation.identity.json")
    seal_payload = _observation_seal_payload(run, observation.candidate_id, identity)
    try:
        _recheck_execution_binding(run.repo_commit)
        seal_identity = _write_new_json(seal_path, seal_payload, root=run.evidence_dir)
    except FileExistsError as error:
        raise ExternalProvenanceError(
            f"candidate observation seal already exists: {observation.candidate_id.value}"
        ) from error
    if (
        _read_identity_at(
            run.evidence_dir,
            seal_path.relative_to(run.evidence_dir),
            "candidate observation seal",
        )
        != seal_identity
    ):
        raise ExternalProvenanceError("candidate observation seal drifted after write")
    _replace_live_audit_state(
        run,
        _LiveAuditState(
            observation_seals=live_state.observation_seals
            + ((observation.candidate_id, identity, seal_identity),),
            completion=live_state.completion,
        ),
    )
    return identity

def complete_external_audit(run: ExternalAuditRun) -> ArtifactIdentity:

    _authenticate_run(run, require_building=True)
    live_state = _live_audit_state(run)
    _validate_candidate_directory_membership(run)
    missing = [
        candidate_id.value
        for candidate_id in run.candidate_ids
        if not _entry_exists_at(
            run.evidence_dir,
            Path("candidates") / candidate_id.value / "observation.json",
        )
        or not _entry_exists_at(
            run.evidence_dir,
            Path("candidates") / candidate_id.value / "observation.identity.json",
        )
    ]
    if missing:
        raise ExternalProvenanceError(
            f"completion requires exactly six observations; missing={missing}"
        )
    in_process_seals = {
        candidate_id: (observation, seal)
        for candidate_id, observation, seal in live_state.observation_seals
    }
    identities: dict[str, dict[str, object]] = {}
    observations: list[CandidateObservation] = []
    for candidate_id in run.candidate_ids:
        if candidate_id not in in_process_seals:
            raise ExternalProvenanceError(
                f"completion is missing originally published identity: {candidate_id.value}"
            )
        expected_observation, expected_seal = in_process_seals[candidate_id]
        observation, actual_identity, seal_identity = _read_observation_slot(
            run,
            candidate_id,
            expected_observation=expected_observation,
            expected_seal=expected_seal,
        )
        observations.append(observation)
        identities[candidate_id.value] = _identity_mapping(actual_identity)
    completion = {
        "schema_version": _COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "run_id": run.run_id,
        "catalog_sha256": run.catalog_sha256,
        "attempt": _identity_mapping(run.attempt),
        "observations": identities,
    }
    _reject_secret_like_keys(completion)
    try:
        _recheck_execution_binding(run.repo_commit)
        identity = _write_new_json(
            run.evidence_dir / "completion.json",
            completion,
            root=run.evidence_dir,
        )
    except FileExistsError as error:
        raise ExternalProvenanceError("external audit is already complete") from error
    if (
        _read_identity_at(run.evidence_dir, Path("completion.json"), "audit completion")
        != identity
    ):
        raise ExternalProvenanceError("completion record drifted after write")
    object.__setattr__(run, "observations", tuple(observations))
    object.__setattr__(run, "completion", identity)
    _replace_live_audit_state(
        run,
        _LiveAuditState(
            observation_seals=live_state.observation_seals,
            completion=identity,
        ),
    )
    return identity

def load_external_audit(
    run_id: str,
    catalog: ExternalCatalog,
    *,
    expected_completion: ArtifactIdentity,
) -> ExternalAuditRun:

    _validate_run_id(run_id)
    _validate_closed_catalog(catalog)
    evidence_root, bulk_root = _validate_fixed_roots()
    evidence_dir = evidence_root / "external_baselines" / run_id
    bulk_dir = bulk_root / "external_baselines" / run_id
    completion_relative = Path("completion.json")
    completion_payload = _read_authenticated_canonical_mapping_at(
        evidence_dir,
        completion_relative,
        "audit completion",
        expected_completion,
    )
    completion_payload = _exact_mapping(
        completion_payload,
        frozenset(
            {
                "schema_version",
                "status",
                "run_id",
                "catalog_sha256",
                "attempt",
                "observations",
            }
        ),
        "audit completion",
    )
    completed_attempt = _artifact_from_mapping(
        completion_payload["attempt"], "completed attempt"
    )
    attempt_relative = Path("external_baselines") / run_id / "attempt.json"
    attempt_payload = _read_authenticated_canonical_mapping_at(
        evidence_root,
        attempt_relative,
        "audit attempt",
        completed_attempt,
    )
    run = ExternalAuditRun.from_mapping(
        attempt_payload,
        catalog=catalog,
        attempt=completed_attempt,
        evidence_dir=evidence_dir,
        bulk_dir=bulk_dir,
    )
    _authenticate_run(
        run,
        require_building=False,
        recheck_execution=False,
    )
    raw_observations = _exact_mapping(
        completion_payload["observations"],
        frozenset(item.value for item in run.candidate_ids),
        "completed observations",
    )
    if (
        completion_payload["schema_version"] != _COMPLETION_SCHEMA
        or completion_payload["status"] != "COMPLETE"
        or completion_payload["run_id"] != run_id
        or completion_payload["catalog_sha256"] != run.catalog_sha256
        or completed_attempt != run.attempt
    ):
        raise ExternalProvenanceError("audit completion does not bind its fixed run")
    _validate_candidate_directory_membership(run)
    observations: list[CandidateObservation] = []
    seals: list[tuple[ExternalCandidateId, ArtifactIdentity, ArtifactIdentity]] = []
    for candidate_id in run.candidate_ids:
        declared_identity = _artifact_from_mapping(
            raw_observations[candidate_id.value],
            f"completed observation {candidate_id.value}",
        )
        expected_path = (
            run.evidence_dir / "candidates" / candidate_id.value / "observation.json"
        )
        if declared_identity.path != str(expected_path):
            raise ExternalProvenanceError(
                f"completed observation is outside its fixed slot: {candidate_id.value}"
            )
        observation, actual_identity, seal_identity = _read_observation_slot(
            run,
            candidate_id,
            expected_observation=declared_identity,
        )
        observations.append(observation)
        seals.append((candidate_id, actual_identity, seal_identity))
    object.__setattr__(run, "observations", tuple(observations))
    object.__setattr__(run, "completion", expected_completion)
    _register_live_audit_state(
        run,
        _LiveAuditState(
            observation_seals=tuple(seals),
            completion=expected_completion,
        ),
    )
    return run

def _declaration_for_run(
    run: ExternalAuditRun, candidate_id: ExternalCandidateId
) -> CandidateDeclaration:
    for declaration in run.declarations:
        if declaration.candidate_id is candidate_id:
            return declaration
    raise ExternalProvenanceError("candidate declaration is absent from the run")

def _validate_observation_binding(
    run: ExternalAuditRun,
    declaration: CandidateDeclaration,
    observation: CandidateObservation,
) -> None:
    source = observation.source
    expected_checkout = run.sources_dir / observation.candidate_id.value

    if source is not None:
        if (
            source.url != declaration.url
            or source.observed_ref != declaration.observed_ref
        ):
            raise ExternalProvenanceError(
                "source observation does not match the catalog"
            )
        if Path(source.checkout_path) != expected_checkout:
            raise ExternalProvenanceError(
                "source checkout path must use the candidate evidence source slot"
            )
        _directory_identity_at(
            run.evidence_dir,
            Path("sources") / observation.candidate_id.value,
        )
        _authenticate_artifact_in_slot(
            source.retrieval_log,
            "source retrieval log",
            run.evidence_dir,
            Path("candidates") / observation.candidate_id.value / "retrieval.log",
            exact=True,
        )
        _reauthenticate_source_observation(run, declaration, source)
    if observation.license is not None and observation.license.artifact is not None:
        license_path = Path(observation.license.artifact.path)
        allowed_license_paths = tuple(
            expected_checkout / relative
            for relative in declaration.license_file_candidates
        )
        if license_path not in allowed_license_paths:
            raise ExternalProvenanceError(
                "license artifact is outside its declared retained-source slot"
            )
        _authenticate_artifact_in_slot(
            observation.license.artifact,
            "license artifact",
            run.evidence_dir,
            license_path.relative_to(run.evidence_dir),
            exact=True,
        )
    _authenticate_weight_evidence(run, observation, source)
    capability = observation.capability
    if capability is not None:
        if source is not None and (
            capability.source_commit != source.commit
            or capability.source_tree_sha256 != source.tracked_tree_sha256
        ):
            raise ExternalProvenanceError("capability decision source identity drifted")
        decided = set(capability.supported_capabilities).union(
            capability.unsupported_capabilities
        )
        if decided != set(declaration.required_capabilities):
            raise ExternalProvenanceError(
                "capability decision must cover the catalog requirements"
            )
        if source is None:
            raise ExternalProvenanceError(
                "capability decision requires retained source"
            )
        for judgment in capability.judgments:
            _validate_source_citation(
                source,
                judgment.source_artifact,
                judgment.start_line,
                judgment.end_line,
                None,
                label=f"capability judgment {judgment.capability}",
            )
    if source is not None:
        _reauthenticate_source_observation(run, declaration, source)

    _validate_disposition_state(observation, declaration)

def _authenticate_weight_evidence(
    run: ExternalAuditRun,
    observation: CandidateObservation,
    source: SourceObservation | None,
) -> None:
    def authenticate_citation(citation: WeightReference, label: str) -> None:
        if source is None:
            raise ExternalProvenanceError("weight citation requires retained source")
        _validate_source_citation(
            source,
            citation.source_artifact,
            citation.start_line,
            citation.end_line,
            citation.url,
            label=label,
        )

    def authenticate_span(span: SourceSpan, label: str) -> None:
        if source is None:
            raise ExternalProvenanceError(
                "weight license evidence requires retained source"
            )
        _validate_source_citation(
            source,
            span.source_artifact,
            span.start_line,
            span.end_line,
            None,
            label=label,
        )

    def authenticate_artifact(
        artifact: ArtifactIdentity,
        content_class: WeightContentClass,
        label: str,
    ) -> None:
        weight_bytes = _authenticate_artifact_in_slot(
            artifact,
            label,
            run.evidence_dir,
            Path("candidates") / observation.candidate_id.value / "weights",
            exact=False,
        )
        actual_content_class = classify_weight_content(weight_bytes)
        if weight_bytes.startswith(_GIT_LFS_POINTER_PREFIX):
            raise ExternalProvenanceError(
                "Git-LFS pointer bytes cannot satisfy a weight checkpoint identity"
            )
        if actual_content_class is not content_class:
            raise ExternalProvenanceError("weight content classification drifted")

    weight = observation.weight
    if isinstance(weight, WeightObservation):
        if weight.citation is not None:
            authenticate_citation(weight.citation, "weight citation")
        if weight.artifact is not None:
            authenticate_artifact(
                weight.artifact, weight.content_class, "weight artifact"
            )
        return
    for component in weight.components:
        if component.citation is not None:
            authenticate_citation(
                component.citation,
                f"weight component citation {component.component_id}",
            )
        for index, span in enumerate(component.license_evidence, start=1):
            authenticate_span(
                span,
                f"weight component license evidence {component.component_id}:{index}",
            )
        if component.artifact is not None:
            authenticate_artifact(
                component.artifact,
                component.content_class,
                f"weight component artifact {component.component_id}",
            )
    for index, reference in enumerate(weight.non_task_references, start=1):
        authenticate_citation(reference, f"non-task weight reference {index}")

def _reauthenticate_source_observation(
    run: ExternalAuditRun,
    declaration: CandidateDeclaration,
    source: SourceObservation,
) -> None:
    checkout = Path(source.checkout_path)
    checkout_identity = _DirectoryIdentity(
        source.checkout_device, source.checkout_inode
    )
    relative = Path("sources") / declaration.candidate_id.value
    if _directory_identity_at(run.evidence_dir, relative) != checkout_identity:
        raise ExternalProvenanceError("retained source checkout identity drifted")
    git_binary = Path(source.git_binary.path)
    git_binary_stat = _validate_git_binary(git_binary)
    if git_binary_stat != source.git_binary_stat:
        raise ExternalProvenanceError("retained source git binary stat drifted")
    if _git_binary_artifact(git_binary, git_binary_stat) != source.git_binary:
        raise ExternalProvenanceError("retained source git binary identity drifted")
    _verify_source_binding(
        declaration,
        checkout,
        source.commit,
        git_binary,
        git_binary_stat,
        checkout_identity,
        run,
    )
    tree_sha256, lfs_pointer_paths = _hash_retained_source_tree(
        checkout, git_binary, git_binary_stat
    )
    _verify_source_binding(
        declaration,
        checkout,
        source.commit,
        git_binary,
        git_binary_stat,
        checkout_identity,
        run,
    )
    if (
        tree_sha256 != source.tracked_tree_sha256
        or lfs_pointer_paths != source.git_lfs_pointer_paths
    ):
        raise ExternalProvenanceError("retained source tree identity drifted")
    log_relative = Path("candidates") / declaration.candidate_id.value / "retrieval.log"
    log = _read_canonical_mapping_at(
        run.evidence_dir, log_relative, "source retrieval log"
    )
    expected_log = {
        "schema_version": _SOURCE_RETRIEVAL_SCHEMA,
        "candidate_id": declaration.candidate_id.value,
        "url": source.url,
        "observed_ref": source.observed_ref,
        "commit": source.commit,
        "checkout_path": source.checkout_path,
        "tracked_tree_sha256": source.tracked_tree_sha256,
        "git_lfs_pointer_paths": list(source.git_lfs_pointer_paths),
        "checkout_identity": checkout_identity.as_mapping(),
        "git_binary": _identity_mapping(source.git_binary),
        "git_binary_stat": list(source.git_binary_stat),
        "observed_at": source.observed_at,
    }
    if dict(log) != expected_log:
        raise ExternalProvenanceError(
            "source retrieval log does not bind the retained source"
        )

def _validate_disposition_state(
    observation: CandidateObservation, declaration: CandidateDeclaration
) -> None:
    source = observation.source
    license_observation = observation.license
    weight = observation.weight
    capability = observation.capability
    disposition = observation.disposition
    all_supported = (
        capability is not None
        and capability.supported_capabilities == declaration.required_capabilities
        and not capability.unsupported_capabilities
    )
    research_license = (
        license_observation is not None
        and license_observation.review_class == "research_evaluation_allowed"
    )
    usable_weight = _weight_is_usable(weight)
    unavailable_weight = not usable_weight
    has_official_reference = _weight_has_official_reference(weight)

    if disposition is AuditDisposition.ELIGIBLE_FOR_ADAPTER:
        valid = (
            source is not None
            and research_license
            and usable_weight
            and capability is not None
            and all_supported
        )
    elif disposition is AuditDisposition.ELIGIBLE_FOR_WEIGHT_ACQUISITION:
        valid = (
            source is not None
            and research_license
            and unavailable_weight
            and has_official_reference
            and _weight_can_advance_to_acquisition(weight)
            and capability is not None
            and all_supported
        )
    elif disposition is AuditDisposition.TASK_CONTRACT_UNREPRESENTABLE:
        valid = (
            source is not None
            and research_license
            and unavailable_weight
            and capability is not None
            and bool(capability.unsupported_capabilities)
        )
    elif disposition is AuditDisposition.WEIGHTS_UNAVAILABLE:
        valid = (
            source is not None
            and research_license
            and unavailable_weight
            and _weight_can_be_unavailable(weight)
            and capability is not None
            and all_supported
        )
    elif disposition is AuditDisposition.LICENSE_UNAVAILABLE:
        valid = (
            source is not None
            and (
                license_observation is None
                or license_observation.review_class == "unavailable"
                or _weight_license_blocks_progress(weight)
            )
            and unavailable_weight
            and (
                capability is None
                or (
                    set(capability.supported_capabilities).union(
                        capability.unsupported_capabilities
                    )
                    == set(declaration.required_capabilities)
                )
            )
        )
    elif disposition is AuditDisposition.SOURCE_UNAVAILABLE:
        valid = (
            source is None
            and license_observation is None
            and unavailable_weight
            and not has_official_reference
            and capability is None
        )
    elif disposition is AuditDisposition.PROVENANCE_FAILED:
        valid = (
            source is None
            and license_observation is None
            and unavailable_weight
            and not has_official_reference
            and capability is None
        )
    elif disposition is AuditDisposition.REVIEW_REQUIRED:
        valid = (
            source is not None
            and license_observation is not None
            and license_observation.artifact is not None
            and license_observation.review_class
            in {"research_evaluation_allowed", "restricted_review_required"}
            and unavailable_weight
            and (
                capability is None
                or (
                    set(capability.supported_capabilities).union(
                        capability.unsupported_capabilities
                    )
                    == set(declaration.required_capabilities)
                )
            )
        )
    else:
        raise ExternalProvenanceError("audit disposition is outside the closed table")
    if not valid:
        raise ExternalProvenanceError(
            f"{disposition.value} disposition has inconsistent source, license, "
            "weight, or capability evidence"
        )

def _weight_is_usable(weight: WeightEvidence) -> bool:
    if isinstance(weight, WeightObservation):
        return (
            weight.artifact is not None
            and weight.content_class in _PLAUSIBLE_WEIGHT_CLASSES
            and weight.unavailable_reason is None
        )
    return all(
        component.artifact is not None
        and component.content_class in _PLAUSIBLE_WEIGHT_CLASSES
        and component.unavailable_reason is None
        and component.license_status is WeightLicenseStatus.RESEARCH_EVALUATION_ALLOWED
        for component in weight.components
    )

def _weight_has_official_reference(weight: WeightEvidence) -> bool:
    if isinstance(weight, WeightObservation):
        return weight.official_url is not None and weight.citation is not None
    return all(component.citation is not None for component in weight.components)

def _weight_can_advance_to_acquisition(weight: WeightEvidence) -> bool:
    if isinstance(weight, WeightObservation):
        return True
    return all(
        component.artifact is None
        and component.license_status is WeightLicenseStatus.RESEARCH_EVALUATION_ALLOWED
        for component in weight.components
    )

def _weight_can_be_unavailable(weight: WeightEvidence) -> bool:
    if isinstance(weight, WeightObservation):
        return weight.official_url is None or weight.artifact is not None
    return any(
        component.citation is None
        or component.artifact is not None
        or component.license_status is WeightLicenseStatus.UNAVAILABLE
        for component in weight.components
    )

def _weight_license_blocks_progress(weight: WeightEvidence) -> bool:
    return isinstance(weight, WeightBundleObservation) and any(
        component.license_status is not WeightLicenseStatus.RESEARCH_EVALUATION_ALLOWED
        for component in weight.components
    )

def _authenticate_artifact_in_slot(
    identity: ArtifactIdentity,
    label: str,
    root: Path,
    slot: Path,
    *,
    exact: bool,
) -> bytes:
    path = Path(identity.path)
    expected = root / slot
    if (exact and path != expected) or (
        not exact and (path == expected or expected not in path.parents)
    ):
        raise ExternalProvenanceError(f"{label} path is outside its approved slot")
    relative = path.relative_to(root)
    raw = _read_regular_bytes_at(root, relative, label)
    actual = ArtifactIdentity(
        str(root / relative), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    if actual != identity:
        raise ExternalProvenanceError(f"{label} identity drifted")
    return raw

def _validate_run_id(run_id: object) -> None:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise ExternalProvenanceError("run id is invalid")

def _validate_closed_catalog(catalog: object) -> ExternalCatalog:
    if not isinstance(catalog, ExternalCatalog):
        raise ExternalProvenanceError("claim requires an ExternalCatalog")
    try:
        path = _PINNED_CATALOG_PATHS[catalog.schema_version]
    except (KeyError, TypeError) as error:
        raise ExternalProvenanceError(
            "catalog schema version is unsupported"
        ) from error
    try:
        pinned = load_external_catalog(path)
    except Exception as error:
        raise ExternalProvenanceError(
            f"cannot authenticate the pinned closed catalog: {error}"
        ) from error
    if (
        catalog.schema_version != pinned.schema_version
        or tuple(item.as_mapping() for item in catalog.candidates)
        != tuple(item.as_mapping() for item in pinned.candidates)
        or tuple(item.observed_ref for item in catalog.candidates)
        != tuple(item.observed_ref for item in pinned.candidates)
        or any(item.observed_ref != "HEAD" for item in catalog.candidates)
    ):
        raise ExternalProvenanceError(
            "catalog declarations drifted from the closed reviewed vocabulary"
        )
    return catalog

def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise ExternalProvenanceError("argv must be a nonempty string sequence")
    command = tuple(argv)
    if not all(type(item) is str and item and "\x00" not in item for item in command):
        raise ExternalProvenanceError("argv must contain nonempty strings")
    for item in command:
        key = item.lstrip("-").split("=", 1)[0].lower().replace("-", "_")
        if any(fragment in key for fragment in _SECRET_KEY_FRAGMENTS):
            raise ExternalProvenanceError("argv contains a secret-like key")
    if command not in _SAFE_COMMANDS:
        raise ExternalProvenanceError(
            "argv does not match the closed safe audit command schema"
        )
    return command

def _require_runtime_python() -> Path:
    actual = Path(sys.executable).resolve()
    expected = Path(RUNTIME_PYTHON).resolve()
    if actual != expected or sys.version_info[:2] != (3, 12):
        raise ExternalProvenanceError("audit claim requires the pinned interpreter")
    return actual

def _current_clean_commit() -> str:
    try:
        status = subprocess.run(
            (
                str(_GIT_BINARY),
                "-C",
                str(_REPOSITORY_ROOT),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            check=True,
            capture_output=True,
            text=True,
            env=_GIT_ENVIRONMENT,
        ).stdout
        if status:
            raise ExternalProvenanceError("current repository is dirty")
        commit = subprocess.run(
            (
                str(_GIT_BINARY),
                "-C",
                str(_REPOSITORY_ROOT),
                "rev-parse",
                "HEAD^{commit}",
            ),
            check=True,
            capture_output=True,
            text=True,
            env=_GIT_ENVIRONMENT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExternalProvenanceError(
            f"cannot authenticate current repository: {error}"
        ) from error
    return _require_commit("repository commit", commit)

def _recheck_execution_binding(expected_commit: str) -> None:
    _require_runtime_python()
    if _current_clean_commit() != expected_commit:
        raise ExternalProvenanceError(
            "repository HEAD commit drifted after audit claim"
        )

def _validate_git_binary(git_binary: object) -> tuple[int, int, int, int, int, int]:
    if not isinstance(git_binary, Path) or not git_binary.is_absolute():
        raise ExternalProvenanceError("git binary must be an absolute Path")
    if Path(os.path.abspath(git_binary)) != git_binary:
        raise ExternalProvenanceError("git binary path must be normalized")
    if _TEST_ROOTS is None and git_binary != _GIT_BINARY:
        raise ExternalProvenanceError(
            "production source resolution requires canonical /usr/bin/git"
        )
    try:
        info = os.stat(git_binary, follow_symlinks=False)
    except OSError as error:
        raise ExternalProvenanceError("git binary is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ExternalProvenanceError("git binary must be a plain regular file")
    if info.st_mode & 0o111 == 0:
        raise ExternalProvenanceError("git binary is not executable")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ExternalProvenanceError("git binary must not be group/world writable")
    if _TEST_ROOTS is None:
        _validate_trusted_production_git_path(git_binary)
    return _stat_identity(info)

def _git_binary_artifact(
    git_binary: Path,
    expected_stat: tuple[int, int, int, int, int, int],
) -> ArtifactIdentity:
    relative = Path(*git_binary.parts[1:])
    raw = _read_regular_bytes_at(Path("/"), relative, "git binary")
    try:
        after = os.stat(git_binary, follow_symlinks=False)
    except OSError as error:
        raise ExternalProvenanceError("git binary identity drifted") from error
    if _stat_identity(after) != expected_stat:
        raise ExternalProvenanceError("git binary identity drifted")
    return ArtifactIdentity(str(git_binary), hashlib.sha256(raw).hexdigest(), len(raw))

def _validate_trusted_production_git_path(git_binary: Path) -> None:
    paths = [Path("/")]
    for component in git_binary.parts[1:]:
        paths.append(paths[-1] / component)
    for index, current in enumerate(paths):
        try:
            info = os.stat(current, follow_symlinks=False)
        except OSError as error:
            raise ExternalProvenanceError(
                "canonical git path is unavailable"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise ExternalProvenanceError("canonical git path contains a symlink")
        if index < len(paths) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ExternalProvenanceError("canonical git parent is not a directory")
        if index == len(paths) - 1 and not stat.S_ISREG(info.st_mode):
            raise ExternalProvenanceError("canonical git is not a regular file")
        if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ExternalProvenanceError(
                "canonical git path is not root-owned and non-writable"
            )

def _run_source_git(
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
    operation: str,
    *arguments: str,
    binary_output: bool = False,
) -> str | bytes:
    if _validate_git_binary(git_binary) != binary_identity:
        raise ExternalProvenanceError("git binary identity drifted")
    environment = dict(_GIT_ENVIRONMENT)
    environment.update(
        {
            "GIT_ASKPASS": "/bin/false",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "SSH_ASKPASS": "/bin/false",
        }
    )
    try:
        result = subprocess.run(
            (str(git_binary), *arguments),
            check=False,
            capture_output=True,
            text=not binary_output,
            env=environment,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ExternalProvenanceError(f"{operation} failed") from error
    if _validate_git_binary(git_binary) != binary_identity:
        raise ExternalProvenanceError("git binary identity drifted")
    if result.returncode != 0:
        raise ExternalProvenanceError(f"{operation} failed")
    return result.stdout

def _resolve_remote_head(
    url: str,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
) -> str:
    output = _run_source_git(
        git_binary,
        binary_identity,
        "resolve official source HEAD",
        "ls-remote",
        "--exit-code",
        url,
        "HEAD",
    )
    if not isinstance(output, str):
        raise ExternalProvenanceError("official source HEAD response is invalid")
    lines = output.splitlines()
    if len(lines) != 1:
        raise ExternalProvenanceError("official source HEAD response is ambiguous")
    fields = lines[0].split("\t")
    if len(fields) != 2 or fields[1] != "HEAD":
        raise ExternalProvenanceError("official source HEAD response is malformed")
    return _require_commit("official source HEAD", fields[0])

def _source_git_text(
    checkout: Path,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
    operation: str,
    *arguments: str,
) -> str:
    output = _run_source_git(
        git_binary,
        binary_identity,
        operation,
        "-C",
        str(checkout),
        *arguments,
    )
    if not isinstance(output, str):
        raise ExternalProvenanceError(f"{operation} output is invalid")
    return output.strip()

def _tracked_source_paths(
    checkout: Path,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
) -> tuple[str, ...]:
    output = _run_source_git(
        git_binary,
        binary_identity,
        "list tracked source paths",
        "-C",
        str(checkout),
        "ls-files",
        "-z",
        binary_output=True,
    )
    if not isinstance(output, bytes) or not output.endswith(b"\x00"):
        raise ExternalProvenanceError("tracked source path inventory is malformed")
    raw_paths = output[:-1].split(b"\x00")
    try:
        paths = tuple(item.decode("utf-8") for item in raw_paths)
    except UnicodeDecodeError as error:
        raise ExternalProvenanceError(
            "tracked source paths must be valid UTF-8"
        ) from error
    return _validate_tracked_paths("tracked source paths", paths)

def _tracked_source_stages(
    checkout: Path,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
) -> dict[str, tuple[str, str]]:
    output = _run_source_git(
        git_binary,
        binary_identity,
        "list tracked source modes",
        "-C",
        str(checkout),
        "ls-files",
        "--stage",
        "-z",
        binary_output=True,
    )
    if not isinstance(output, bytes) or not output.endswith(b"\x00"):
        raise ExternalProvenanceError("tracked source mode inventory is malformed")
    stages: dict[str, tuple[str, str]] = {}
    for raw_entry in output[:-1].split(b"\x00"):
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
            relative_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ExternalProvenanceError(
                "tracked source mode inventory is malformed"
            ) from error
        _require_sha1_object_id(object_id)
        if stage != "0" or relative_path in stages:
            raise ExternalProvenanceError(
                "tracked source index has duplicate or unmerged entries"
            )
        _validate_tracked_paths("tracked source stage path", (relative_path,))
        stages[relative_path] = (mode, object_id)
    return stages

def _tracked_head_tree(
    checkout: Path,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
) -> dict[str, tuple[str, str]]:
    output = _run_source_git(
        git_binary,
        binary_identity,
        "list retained source HEAD tree",
        "-C",
        str(checkout),
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        "HEAD",
        binary_output=True,
    )
    if not isinstance(output, bytes) or not output.endswith(b"\x00"):
        raise ExternalProvenanceError("retained source HEAD tree is malformed")
    tree: dict[str, tuple[str, str]] = {}
    for raw_entry in output[:-1].split(b"\x00"):
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            relative_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ExternalProvenanceError(
                "retained source HEAD tree is malformed"
            ) from error
        _require_sha1_object_id(object_id)
        if object_type not in {"blob", "commit"} or relative_path in tree:
            raise ExternalProvenanceError(
                "retained source HEAD tree has an invalid entry"
            )
        _validate_tracked_paths("retained source HEAD path", (relative_path,))
        tree[relative_path] = (mode, object_id)
    return tree

def _tracked_source_flags(
    checkout: Path,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
) -> dict[str, str]:
    output = _run_source_git(
        git_binary,
        binary_identity,
        "list retained source index flags",
        "-C",
        str(checkout),
        "ls-files",
        "-v",
        "-z",
        binary_output=True,
    )
    if not isinstance(output, bytes) or not output.endswith(b"\x00"):
        raise ExternalProvenanceError("retained source index flags are malformed")
    flags: dict[str, str] = {}
    for raw_entry in output[:-1].split(b"\x00"):
        try:
            tag, raw_path = raw_entry[:1].decode("ascii"), raw_entry[2:]
            if raw_entry[1:2] != b" ":
                raise ValueError
            relative_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ExternalProvenanceError(
                "retained source index flags are malformed"
            ) from error
        if relative_path in flags:
            raise ExternalProvenanceError("retained source index flags are duplicated")
        _validate_tracked_paths("retained source index-flag path", (relative_path,))
        flags[relative_path] = tag
    return flags

def _source_object_format(
    checkout: Path,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
) -> str:
    value = _source_git_text(
        checkout,
        git_binary,
        binary_identity,
        "read retained source object format",
        "rev-parse",
        "--show-object-format",
    )
    if value not in {"sha1", "sha256"}:
        raise ExternalProvenanceError("retained source object format is invalid")
    return value

def _hash_retained_source_tree(
    checkout: Path,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
) -> tuple[str, tuple[str, ...]]:
    object_format = _source_object_format(checkout, git_binary, binary_identity)
    if object_format != "sha1":
        raise ExternalProvenanceError(
            f"retained source object format is unsupported: {object_format}"
        )
    paths_before = _tracked_source_paths(checkout, git_binary, binary_identity)
    stages_before = _tracked_source_stages(checkout, git_binary, binary_identity)
    head_tree_before = _tracked_head_tree(checkout, git_binary, binary_identity)
    flags_before = _tracked_source_flags(checkout, git_binary, binary_identity)
    if tuple(stages_before) != paths_before:
        raise ExternalProvenanceError(
            "tracked source path and stage inventories differ"
        )
    if stages_before != head_tree_before:
        raise ExternalProvenanceError("retained source HEAD and index differ")
    if flags_before != {path: "H" for path in paths_before}:
        raise ExternalProvenanceError(
            "retained source index has assume-unchanged or skip-worktree flags"
        )
    tree_entries: list[dict[str, object]] = []
    lfs_pointer_paths: list[str] = []
    for relative_path in paths_before:
        mode, expected_blob = stages_before[relative_path]
        if mode == "160000":
            raise ExternalProvenanceError(
                f"tracked source contains forbidden submodule mode 160000: "
                f"{relative_path}"
            )
        if mode not in {"100644", "100755"}:
            raise ExternalProvenanceError(
                f"tracked source entry is not a regular file: {relative_path}"
            )
        raw = _read_regular_bytes_at(
            checkout, Path(relative_path), f"tracked source file {relative_path}"
        )
        file_sha256 = hashlib.sha256(raw).hexdigest()
        actual_blob = hashlib.sha1(
            f"blob {len(raw)}\0".encode("ascii") + raw,
            usedforsecurity=False,
        ).hexdigest()
        if actual_blob != expected_blob:
            raise ExternalProvenanceError(
                f"tracked source worktree blob differs from HEAD: {relative_path}"
            )
        tree_entries.append(
            {
                "path": relative_path,
                "mode": mode,
                "size_bytes": len(raw),
                "sha256": file_sha256,
            }
        )
        if raw.startswith(_GIT_LFS_POINTER_PREFIX):
            lfs_pointer_paths.append(relative_path)
    tracked_tree_sha256 = hashlib.sha256(
        canonical_json_bytes({"entries": tree_entries})
    ).hexdigest()
    paths_after = _tracked_source_paths(checkout, git_binary, binary_identity)
    stages_after = _tracked_source_stages(checkout, git_binary, binary_identity)
    head_tree_after = _tracked_head_tree(checkout, git_binary, binary_identity)
    flags_after = _tracked_source_flags(checkout, git_binary, binary_identity)
    if (
        paths_after != paths_before
        or stages_after != stages_before
        or head_tree_after != head_tree_before
        or flags_after != flags_before
    ):
        raise ExternalProvenanceError("tracked source inventory drifted while hashing")
    return tracked_tree_sha256, tuple(lfs_pointer_paths)

def _require_sha1_object_id(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ExternalProvenanceError("tracked source object id is invalid")
    return value

def _verify_source_binding(
    candidate: CandidateDeclaration,
    checkout: Path,
    commit: str,
    git_binary: Path,
    binary_identity: tuple[int, int, int, int, int, int],
    checkout_identity: _DirectoryIdentity,
    run: ExternalAuditRun,
) -> None:
    relative = Path("sources") / candidate.candidate_id.value
    if _directory_identity_at(run.evidence_dir, relative) != checkout_identity:
        raise ExternalProvenanceError("retained source directory identity drifted")
    origin = _source_git_text(
        checkout,
        git_binary,
        binary_identity,
        "read retained source origin",
        "remote",
        "get-url",
        "origin",
    )
    if origin != candidate.url:
        raise ExternalProvenanceError("retained source origin remote drifted")
    head = _source_git_text(
        checkout,
        git_binary,
        binary_identity,
        "read retained source HEAD",
        "rev-parse",
        "HEAD^{commit}",
    )
    if head != commit:
        raise ExternalProvenanceError("retained source HEAD drifted")
    head_name = _source_git_text(
        checkout,
        git_binary,
        binary_identity,
        "read retained source HEAD attachment",
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )
    if head_name != "HEAD":
        raise ExternalProvenanceError("retained source HEAD is not detached")
    status = _source_git_text(
        checkout,
        git_binary,
        binary_identity,
        "read retained source status",
        "status",
        "--porcelain=v1",
    )
    if status:
        raise ExternalProvenanceError("retained source checkout is dirty")

def _validate_fixed_roots() -> tuple[Path, Path]:
    evidence_root, bulk_root = _TEST_ROOTS or (
        EMERGENT_EVIDENCE_ROOT,
        EMERGENT_BULK_ROOT,
    )
    evidence_root = Path(evidence_root)
    bulk_root = Path(bulk_root)
    if _TEST_ROOTS is None:
        for root in (evidence_root, bulk_root):
            absolute = Path(os.path.abspath(root))
            for denied in DENIED_PREFIXES:
                if _path_is_below(absolute, Path(denied).resolve(strict=False)):
                    raise ExternalProvenanceError(f"fixed root is denied: {root}")
    if evidence_root == bulk_root:
        raise ExternalProvenanceError("evidence and bulk roots must be distinct")
    for label, root in (("evidence", evidence_root), ("bulk", bulk_root)):
        try:
            info = os.stat(root, follow_symlinks=False)
        except OSError as error:
            raise ExternalProvenanceError(
                f"{label} root is unavailable: {error}"
            ) from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ExternalProvenanceError(f"{label} root must be a plain directory")
    return evidence_root, bulk_root

def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

def _validate_relative_path(relative: Path, label: str) -> None:
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ExternalProvenanceError(f"{label} has an unsafe relative path")

def _open_directory_at_path(root: Path, relative: Path, label: str) -> int:
    _validate_relative_path(relative, label)
    try:
        current = os.open(root, _directory_flags())
    except OSError as error:
        raise ExternalProvenanceError(f"cannot open {label} root: {error}") from error
    try:
        for component in relative.parts:
            try:
                next_fd = os.open(component, _directory_flags(), dir_fd=current)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ExternalProvenanceError(
                        f"{label} symlink traversal is forbidden"
                    ) from error
                raise ExternalProvenanceError(
                    f"cannot open {label} directory component: {error}"
                ) from error
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise

def _directory_identity_at(root: Path, relative: Path) -> _DirectoryIdentity:
    descriptor = _open_directory_at_path(root, relative, "retained directory")
    try:
        info = os.fstat(descriptor)
        return _DirectoryIdentity(info.st_dev, info.st_ino)
    finally:
        os.close(descriptor)

def _read_regular_bytes_at(root: Path, relative: Path, label: str) -> bytes:
    _validate_relative_path(relative, label)
    parent_relative = relative.parent
    if parent_relative == Path("."):
        try:
            parent_fd = os.open(root, _directory_flags())
        except OSError as error:
            raise ExternalProvenanceError(
                f"cannot open {label} root: {error}"
            ) from error
    else:
        parent_fd = _open_directory_at_path(root, parent_relative, label)
    descriptor: int | None = None
    try:
        try:
            path_before = os.stat(
                relative.name, dir_fd=parent_fd, follow_symlinks=False
            )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(relative.name, flags, dir_fd=parent_fd)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ExternalProvenanceError(
                    f"{label} symlink traversal is forbidden"
                ) from error
            raise ExternalProvenanceError(f"cannot open {label}: {error}") from error
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExternalProvenanceError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as error:
            raise ExternalProvenanceError(f"{label} identity drifted") from error
        if (
            _stat_identity(path_before) != _stat_identity(before)
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(path_before) != _stat_identity(path_after)
        ):
            raise ExternalProvenanceError(f"{label} identity drifted while reading")
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)

def _read_identity_at(root: Path, relative: Path, label: str) -> ArtifactIdentity:
    raw = _read_regular_bytes_at(root, relative, label)
    return ArtifactIdentity(
        str(root / relative), hashlib.sha256(raw).hexdigest(), len(raw)
    )

def _read_canonical_mapping_at(
    root: Path, relative: Path, label: str
) -> Mapping[str, object]:
    raw = _read_regular_bytes_at(root, relative, label)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalProvenanceError(f"{label} is malformed JSON") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise ExternalProvenanceError(f"{label} is not canonical JSON")
    return payload

def _read_authenticated_canonical_mapping_at(
    root: Path,
    relative: Path,
    label: str,
    expected: ArtifactIdentity,
) -> Mapping[str, object]:
    _validate_artifact(f"expected {label}", expected)
    if expected.path != str(root / relative):
        raise ExternalProvenanceError(f"expected {label} is outside its fixed slot")
    raw = _read_regular_bytes_at(root, relative, label)
    actual = ArtifactIdentity(
        path=str(root / relative),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )
    if actual != expected:
        raise ExternalProvenanceError(f"{label} identity drifted")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalProvenanceError(f"{label} is malformed JSON") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise ExternalProvenanceError(f"{label} is not canonical JSON")
    return payload

def _ensure_named_directory(root: Path, name: str) -> Path:
    root_fd = os.open(root, _directory_flags())
    try:
        try:
            os.mkdir(name, 0o2775, dir_fd=root_fd)
            os.fsync(root_fd)
        except FileExistsError:
            pass
        child_fd = os.open(name, _directory_flags(), dir_fd=root_fd)
        try:
            if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
                raise ExternalProvenanceError(f"{name} must be a plain directory")
        finally:
            os.close(child_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ExternalProvenanceError(
                f"{name} symlink traversal is forbidden"
            ) from error
        raise ExternalProvenanceError(f"cannot prepare {name}: {error}") from error
    finally:
        os.close(root_fd)
    return root / name

def _mkdir_create_new(path: Path, label: str, *, root: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ExternalProvenanceError(f"{label} is outside its fixed root") from error
    _validate_relative_path(relative, label)
    if relative.parent == Path("."):
        parent_fd = os.open(root, _directory_flags())
    else:
        parent_fd = _open_directory_at_path(root, relative.parent, label)
    try:
        try:
            os.mkdir(path.name, 0o2775, dir_fd=parent_fd)
        except FileExistsError:
            raise FileExistsError(f"{label} already exists: {path}") from None
        os.fsync(parent_fd)
        child_fd = os.open(path.name, _directory_flags(), dir_fd=parent_fd)
        try:
            os.fchmod(child_fd, 0o2775)
            os.fsync(child_fd)
        finally:
            os.close(child_fd)
    finally:
        os.close(parent_fd)
    return path

def _write_new_json(
    path: Path, payload: Mapping[str, object], *, root: Path
) -> ArtifactIdentity:
    _reject_secret_like_keys(payload)
    raw = canonical_json_bytes(payload)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ExternalProvenanceError(
            "create-new JSON path is outside its fixed root"
        ) from error
    _validate_relative_path(relative, "create-new JSON")
    try:
        if relative.parent == Path("."):
            parent_fd = os.open(root, _directory_flags())
        else:
            parent_fd = _open_directory_at_path(
                root, relative.parent, "create-new JSON parent"
            )
    except OSError as error:
        raise ExternalProvenanceError(
            f"cannot open create-new JSON parent: {error}"
        ) from error
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(relative.name, flags, 0o664, dir_fd=parent_fd)
        os.fsync(parent_fd)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("create-new write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.fsync(parent_fd)
    except FileExistsError:
        raise
    except OSError as error:
        raise ExternalProvenanceError(
            f"create-new JSON publication failed; occupied partial claim is preserved: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return ArtifactIdentity(str(path), hashlib.sha256(raw).hexdigest(), len(raw))

def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )

def _entry_exists_at(root: Path, relative: Path) -> bool:
    _validate_relative_path(relative, "fixed slot")
    parent = relative.parent
    if parent == Path("."):
        descriptor = os.open(root, _directory_flags())
    else:
        descriptor = _open_directory_at_path(root, parent, "fixed slot parent")
    try:
        try:
            os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(descriptor)

def _validate_candidate_directory_membership(run: ExternalAuditRun) -> None:
    descriptor = _open_directory_at_path(
        run.evidence_dir, Path("candidates"), "candidate directory"
    )
    try:
        observed = set(os.listdir(descriptor))
    finally:
        os.close(descriptor)
    expected = {candidate_id.value for candidate_id in run.candidate_ids}
    if observed != expected:
        raise ExternalProvenanceError(
            "candidate directory membership differs from the closed catalog"
        )

def _observation_seal_payload(
    run: ExternalAuditRun,
    candidate_id: ExternalCandidateId,
    observation: ArtifactIdentity,
) -> dict[str, object]:
    return {
        "schema_version": _OBSERVATION_SEAL_SCHEMA,
        "candidate_id": candidate_id.value,
        "catalog_sha256": run.catalog_sha256,
        "attempt": _identity_mapping(run.attempt),
        "observation": _identity_mapping(observation),
    }

def _canonical_mapping_identity(
    path: Path, payload: Mapping[str, object]
) -> ArtifactIdentity:
    raw = canonical_json_bytes(payload)
    return ArtifactIdentity(str(path), hashlib.sha256(raw).hexdigest(), len(raw))

def _read_observation_slot(
    run: ExternalAuditRun,
    candidate_id: ExternalCandidateId,
    *,
    expected_observation: ArtifactIdentity,
    expected_seal: ArtifactIdentity | None = None,
) -> tuple[CandidateObservation, ArtifactIdentity, ArtifactIdentity]:
    candidate_relative = Path("candidates") / candidate_id.value
    observation_relative = candidate_relative / "observation.json"
    seal_relative = candidate_relative / "observation.identity.json"
    expected_observation_path = run.evidence_dir / observation_relative
    if expected_observation.path != str(expected_observation_path):
        raise ExternalProvenanceError(
            f"candidate observation is outside its fixed slot: {candidate_id.value}"
        )
    expected_seal_payload = _observation_seal_payload(
        run, candidate_id, expected_observation
    )
    seal_path = run.evidence_dir / seal_relative
    derived_seal = _canonical_mapping_identity(seal_path, expected_seal_payload)
    if expected_seal is None:
        expected_seal = derived_seal
    elif expected_seal != derived_seal:
        raise ExternalProvenanceError(
            "candidate observation seal identity does not bind its declared content: "
            f"{candidate_id.value}"
        )
    seal = _read_authenticated_canonical_mapping_at(
        run.evidence_dir,
        seal_relative,
        "candidate observation seal",
        expected_seal,
    )
    seal = _exact_mapping(
        seal,
        frozenset(
            {
                "schema_version",
                "candidate_id",
                "catalog_sha256",
                "attempt",
                "observation",
            }
        ),
        "candidate observation seal",
    )
    if dict(seal) != expected_seal_payload:
        raise ExternalProvenanceError(
            "candidate observation identity seal does not bind its fixed slot: "
            f"{candidate_id.value}"
        )
    payload = _read_authenticated_canonical_mapping_at(
        run.evidence_dir,
        observation_relative,
        "candidate observation",
        expected_observation,
    )
    observation = CandidateObservation.from_mapping(payload)
    if observation.candidate_id is not candidate_id:
        raise ExternalProvenanceError(
            f"candidate observation is in the wrong fixed slot: {candidate_id.value}"
        )
    declaration = _declaration_for_run(run, candidate_id)
    _validate_observation_binding(run, declaration, observation)
    return observation, expected_observation, expected_seal

def _authenticate_run(
    run: ExternalAuditRun,
    *,
    require_building: bool,
    recheck_execution: bool = True,
) -> None:
    if not isinstance(run, ExternalAuditRun):
        raise ExternalProvenanceError("operation requires an ExternalAuditRun")
    _validate_run_id(run.run_id)
    if not isinstance(run.declarations, tuple) or not all(
        isinstance(item, CandidateDeclaration) for item in run.declarations
    ):
        raise ExternalProvenanceError("audit run catalog declarations are invalid")
    rebound_catalog = ExternalCatalog(
        schema_version=run.catalog_schema_version,
        candidates=run.declarations,
    )
    _validate_closed_catalog(rebound_catalog)
    declaration_ids = tuple(item.candidate_id for item in run.declarations)
    if declaration_ids != run.candidate_ids or len(set(declaration_ids)) != 6:
        raise ExternalProvenanceError("audit run catalog membership drifted")
    if rebound_catalog.semantic_sha256 != run.catalog_sha256:
        raise ExternalProvenanceError("audit run catalog identity drifted")
    if recheck_execution:
        _recheck_execution_binding(run.repo_commit)
    evidence_root, bulk_root = _validate_fixed_roots()
    if (
        run.evidence_dir != evidence_root / "external_baselines" / run.run_id
        or run.bulk_dir != bulk_root / "external_baselines" / run.run_id
        or run.sources_dir != run.evidence_dir / "sources"
    ):
        raise ExternalProvenanceError("audit run paths violate the fixed-root contract")
    expected_directories = {
        "evidence": _directory_identity_at(
            evidence_root, Path("external_baselines") / run.run_id
        ),
        "bulk": _directory_identity_at(
            bulk_root, Path("external_baselines") / run.run_id
        ),
        "sources": _directory_identity_at(
            evidence_root,
            Path("external_baselines") / run.run_id / "sources",
        ),
    }
    if dict(run.directory_identities) != expected_directories:
        raise ExternalProvenanceError("retained audit directory identity drifted")
    expected_attempt_path = run.evidence_dir / "attempt.json"
    if run.attempt.path != str(expected_attempt_path):
        raise ExternalProvenanceError("attempt path is outside its fixed slot")
    payload = _read_authenticated_canonical_mapping_at(
        run.evidence_dir,
        Path("attempt.json"),
        "attempt",
        run.attempt,
    )
    observed_refs = {
        item.candidate_id.value: item.observed_ref for item in run.declarations
    }
    if payload.get("observed_refs") != observed_refs:
        raise ExternalProvenanceError("audit run catalog discovery input drifted")
    expected = {
        "schema_version": _ATTEMPT_SCHEMA,
        "status": "BUILDING",
        "run_id": run.run_id,
        "argv": list(run.argv),
        "repo_commit": run.repo_commit,
        "observed_refs": observed_refs,
        "catalog_sha256": run.catalog_sha256,
        "catalog_schema_version": run.catalog_schema_version,
        "runtime_python": str(Path(RUNTIME_PYTHON).resolve()),
        "directories": {
            name: identity.as_mapping() for name, identity in run.directory_identities
        },
        "evidence_path": str(run.evidence_dir),
        "bulk_path": str(run.bulk_dir),
        "sources_path": str(run.sources_dir),
    }
    if "catalog_schema_version" not in payload:
        if run.catalog_schema_version != _LEGACY_ATTEMPT_CATALOG_SCHEMA:
            raise ExternalProvenanceError(
                "audit attempt lacks an authenticated catalog schema"
            )
        expected.pop("catalog_schema_version")
    if dict(payload) != expected:
        raise ExternalProvenanceError("attempt does not bind this audit run")
    if require_building and _entry_exists_at(run.evidence_dir, Path("completion.json")):
        raise ExternalProvenanceError("external audit is already complete")

def _reject_secret_like_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if type(key) is not str:
                raise ExternalProvenanceError("evidence mapping keys must be strings")
            normalized = key.lower().replace("-", "_")
            if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
                raise ExternalProvenanceError("evidence contains a secret-like key")
            _reject_secret_like_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_secret_like_keys(nested)
