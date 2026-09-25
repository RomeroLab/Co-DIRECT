
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

class InventoryError(RuntimeError):
    pass

class AssetStatus(StrEnum):
    REUSE_EXACT = "REUSE_EXACT"
    EXTEND = "EXTEND"
    REPLACE_WITH_REASON = "REPLACE_WITH_REASON"
    DO_NOT_TOUCH = "DO_NOT_TOUCH"
    READY_EXACT = "READY_EXACT"
    READY_PATCHED = "READY_PATCHED"
    MISSING_CODE = "MISSING_CODE"
    MISSING_WEIGHTS = "MISSING_WEIGHTS"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    LICENSE_BLOCKED = "LICENSE_BLOCKED"
    UNSUPPORTED_CONTRACT = "UNSUPPORTED_CONTRACT"
    SMOKE_FAILED = "SMOKE_FAILED"
    PLANNED_ONLY = "PLANNED_ONLY"

@dataclass(frozen=True, slots=True)
class RepositoryRecord:
    path: str
    url: str
    commit: str
    branch: str
    dirty: bool
    owner: str
    status: AssetStatus

@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    path: str
    size_bytes: int
    sha256: str | None
    architecture: str
    has_lora: bool
    official: bool
    status: AssetStatus
    reason: str

@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    path: str
    interpreter: str
    status: AssetStatus

@dataclass(frozen=True, slots=True)
class AssetInventory:
    repositories: tuple[RepositoryRecord, ...]
    checkpoints: tuple[CheckpointRecord, ...]
    environments: tuple[EnvironmentRecord, ...]
    prior_targets: tuple[Mapping[str, str], ...]

def classify_repository(
    *,
    path: str,
    url: str,
    commit: str,
    branch: str,
    dirty: bool,
    owner: str,
) -> RepositoryRecord:
    if owner in {"emergent", "signed-value", "canonical-dirty-do-not-touch"}:
        status = AssetStatus.DO_NOT_TOUCH
    elif owner == "codirect-benchmark":
        status = AssetStatus.EXTEND
    elif dirty:
        status = AssetStatus.DO_NOT_TOUCH
    else:
        status = AssetStatus.REUSE_EXACT
    return RepositoryRecord(
        path=path,
        url=url,
        commit=commit,
        branch=branch,
        dirty=dirty,
        owner=owner,
        status=status,
    )

def classify_checkpoint(
    *,
    path: Path,
    expected_sha256: str | None,
    expected_architecture: str,
    has_lora: bool,
    official: bool,
    expected_size_bytes: int | None = None,
) -> CheckpointRecord:
    candidate = Path(path)
    if not candidate.is_file():
        return CheckpointRecord(
            path=str(candidate),
            size_bytes=0,
            sha256=None,
            architecture=expected_architecture,
            has_lora=has_lora,
            official=official,
            status=AssetStatus.MISSING_WEIGHTS,
            reason="path_missing",
        )
    size = candidate.stat().st_size
    if size == 0:
        return CheckpointRecord(
            path=str(candidate),
            size_bytes=0,
            sha256=None,
            architecture=expected_architecture,
            has_lora=has_lora,
            official=official,
            status=AssetStatus.MISSING_WEIGHTS,
            reason="empty_file_is_not_a_weight",
        )
    digest = None
    if size <= 32 * 1024 * 1024:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if expected_size_bytes is not None and size != expected_size_bytes:
        return CheckpointRecord(
            path=str(candidate),
            size_bytes=size,
            sha256=digest,
            architecture=expected_architecture,
            has_lora=has_lora,
            official=official,
            status=AssetStatus.REPLACE_WITH_REASON,
            reason="size_does_not_match_authenticated_identity",
        )
    if digest is not None and expected_sha256 is not None and digest != expected_sha256:
        return CheckpointRecord(
            path=str(candidate),
            size_bytes=size,
            sha256=digest,
            architecture=expected_architecture,
            has_lora=has_lora,
            official=official,
            status=AssetStatus.REPLACE_WITH_REASON,
            reason="sha256_does_not_match_authenticated_identity",
        )
    if not official:
        return CheckpointRecord(
            path=str(candidate),
            size_bytes=size,
            sha256=digest or expected_sha256,
            architecture=expected_architecture,
            has_lora=has_lora,
            official=False,
            status=AssetStatus.MISSING_WEIGHTS,
            reason="unofficial_or_unauthenticated_bytes",
        )
    return CheckpointRecord(
        path=str(candidate),
        size_bytes=size,
        sha256=digest or expected_sha256,
        architecture=expected_architecture,
        has_lora=has_lora,
        official=official,
        status=AssetStatus.REUSE_EXACT,
        reason="size_and_declared_identity_match",
    )

def inventory_from_observations(
    *,
    repositories: Sequence[RepositoryRecord],
    checkpoints: Sequence[CheckpointRecord],
    environments: Sequence[EnvironmentRecord],
    prior_targets: Sequence[Mapping[str, str]],
) -> AssetInventory:
    labeled = []
    for row in prior_targets:
        payload = dict(row)
        payload["partition"] = "legacy-dev"
        labeled.append(payload)
    return AssetInventory(
        repositories=tuple(repositories),
        checkpoints=tuple(checkpoints),
        environments=tuple(environments),
        prior_targets=tuple(labeled),
    )
