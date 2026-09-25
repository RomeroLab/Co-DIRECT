
from __future__ import annotations

import hashlib
import json
import math
import os
import stat as stat_module
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
)

TRAINABLE_PARTITIONS = frozenset({"train", "validation"})

_BLIND_NAME_MARKERS = ("test-blind", "test_blind")

APPROVED_DIRECTIONAL_MANIFESTS = (
    "binder_train.parquet",
    "binder_validation.parquet",
    "ame_train.parquet",
    "ame_validation.parquet",
    "antibody_train.parquet",
    "antibody_validation.parquet",
)
_KNOWN_DIRECTIONAL_MANIFEST_METADATA = frozenset({"coverage.json"})

class PreflightError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ResumeState:

    stage: str
    split_hash: str
    config_hash: str
    upstream_commit: str
    checkpoint_hash: str

    def as_provenance(self) -> dict[str, str]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class ManifestIdentity:

    name: str
    family: str
    partition: str
    sha256: str
    size_bytes: int
    row_count: int
    example_id_hash: str

    def as_provenance(self) -> dict[str, object]:
        return asdict(self)

_DIRECTIONAL_RESUME_SCHEMA_VERSION = 3
_DIRECTIONAL_STAGES = frozenset({"bridge_warmup", "joint_adaptation"})

_TERMINAL_EVIDENCE_ROOT = EMERGENT_EVIDENCE_ROOT
_TERMINAL_BULK_ROOT = EMERGENT_BULK_ROOT
_TERMINAL_DENIED_PREFIXES = DENIED_PREFIXES

_DIRECTIONAL_MANIFEST_SPECS = None

@dataclass(frozen=True, slots=True)
class DirectionalResumeState:

    stage: str
    seed: int
    repo_commit: str
    upstream_commit: str
    common_checkpoint_hash: str
    parent_checkpoint_hash: str | None
    split_hash: str
    config_hash: str
    trainable_name_hash: str
    route_equivalent_passes: float
    dropout: float
    ranking_mode: str
    rank_override: float | None
    optimizer_state_hash: str
    parent_projection_hash: str
    parent_projection_file_hash: str
    parent_projection_spec_hash: str

    def __post_init__(self) -> None:
        if self.stage not in _DIRECTIONAL_STAGES:
            raise PreflightError(
                f"stage must be one of {sorted(_DIRECTIONAL_STAGES)}, got "
                f"{self.stage!r}"
            )
        if type(self.seed) is not int or self.seed not in (42, 314159):
            raise PreflightError(
                f"seed must be one of the frozen directional seeds, got {self.seed!r}"
            )
        for field, value in (
            ("repo_commit", self.repo_commit),
            ("upstream_commit", self.upstream_commit),
        ):
            _require_hex_identity(field, value, length=40)
        for field, value in (
            ("common_checkpoint_hash", self.common_checkpoint_hash),
            ("split_hash", self.split_hash),
            ("config_hash", self.config_hash),
            ("trainable_name_hash", self.trainable_name_hash),
            ("optimizer_state_hash", self.optimizer_state_hash),
            ("parent_projection_hash", self.parent_projection_hash),
            ("parent_projection_file_hash", self.parent_projection_file_hash),
            ("parent_projection_spec_hash", self.parent_projection_spec_hash),
        ):
            _require_hex_identity(field, value, length=64)
        if self.parent_checkpoint_hash is not None:
            _require_hex_identity(
                "parent_checkpoint_hash", self.parent_checkpoint_hash, length=64
            )
        elif self.stage == "joint_adaptation":
            raise PreflightError(
                "parent_checkpoint_hash is required for joint_adaptation"
            )
        if (
            isinstance(self.route_equivalent_passes, bool)
            or not isinstance(self.route_equivalent_passes, (int, float))
            or float(self.route_equivalent_passes) != 8.0
        ):
            raise PreflightError("route_equivalent_passes must be the frozen value 8.0")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or float(self.dropout) != 0.0
        ):
            raise PreflightError("dropout must be exactly 0.0")
        _require_frozen_directional_ranking(
            self.ranking_mode,
            self.rank_override,
        )

    def as_provenance(self) -> dict[str, object]:
        return {
            "schema_version": _DIRECTIONAL_RESUME_SCHEMA_VERSION,
            **asdict(self),
        }

    @classmethod
    def from_provenance(cls, payload: Mapping[str, object]) -> "DirectionalResumeState":
        if not isinstance(payload, Mapping):
            raise PreflightError("directional resume state must be a mapping")
        expected = {"schema_version", *cls.__dataclass_fields__}
        missing = sorted(expected - set(payload))
        unknown = sorted(set(payload) - expected)
        if missing:
            raise PreflightError(
                f"directional resume state is missing field(s) {missing}"
            )
        if unknown:
            raise PreflightError(
                f"directional resume state has unknown field(s) {unknown}"
            )
        if type(payload["schema_version"]) is not int or (
            payload["schema_version"] != _DIRECTIONAL_RESUME_SCHEMA_VERSION
        ):
            raise PreflightError(
                "directional resume state schema_version must be exactly "
                f"{_DIRECTIONAL_RESUME_SCHEMA_VERSION}"
            )
        values = {key: payload[key] for key in cls.__dataclass_fields__}
        return cls(**values)

def _require_hex_identity(name: str, value: object, *, length: int) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreflightError(
            f"{name} must be a lowercase {length}-character hexadecimal identity"
        )

def _require_frozen_directional_ranking(
    ranking_mode: object,
    rank_override: object,
) -> None:
    matched = type(ranking_mode) is str and ranking_mode == "matched"
    no_ranking = type(ranking_mode) is str and ranking_mode == "no_ranking"
    positive_float_zero = (
        type(rank_override) is float
        and rank_override == 0.0
        and math.copysign(1.0, rank_override) == 1.0
    )
    if not (
        (matched and rank_override is None) or (no_ranking and positive_float_zero)
    ):
        raise PreflightError(
            "ranking_mode/rank_override must be exactly matched/None or "
            "no_ranking/positive built-in float 0.0"
        )

def assert_manifest_is_trainable(path: Path, frame=None) -> None:

    name = Path(path).name.lower()
    if any(marker in name for marker in _BLIND_NAME_MARKERS):
        raise PreflightError(
            f"test access is disabled during training; refusing manifest {path}"
        )

    if frame is None:
        return
    if "partition" not in getattr(frame, "columns", []):
        raise PreflightError(f"{path} has no `partition` column to verify")

    present = {str(value) for value in frame["partition"].unique()}
    forbidden = sorted(present - TRAINABLE_PARTITIONS)
    if forbidden:
        raise PreflightError(
            f"test access is disabled during training; {path} contains "
            f"partition(s) {forbidden}, expected only {sorted(TRAINABLE_PARTITIONS)}"
        )

def inspect_approved_manifest_inventory(root: Path) -> tuple[ManifestIdentity, ...]:

    from dive.training.parent_projection import verify_manifest_inventory

    specs = _DIRECTIONAL_MANIFEST_SPECS
    if specs is None:
        from dive.training.parent_projection import PRODUCTION_CONTRACT

        specs = PRODUCTION_CONTRACT.manifests
    verified = verify_manifest_inventory(Path(root), specs)
    return tuple(ManifestIdentity(**item.spec.as_provenance()) for item in verified)

def assert_manifest_inventory_unchanged(
    root: Path, expected: tuple[ManifestIdentity, ...]
) -> None:

    if type(expected) is not tuple or not all(
        isinstance(item, ManifestIdentity) for item in expected
    ):
        raise PreflightError("expected manifest inventory must be an exact tuple")
    observed = inspect_approved_manifest_inventory(root)
    if observed != expected:
        raise PreflightError(
            "approved manifest identity changed after provenance preflight"
        )

def assert_resume_compatible(previous: ResumeState, current: ResumeState) -> None:

    mismatches = []
    for field in ("split_hash", "config_hash", "upstream_commit", "checkpoint_hash"):
        before, after = getattr(previous, field), getattr(current, field)
        if before != after:
            mismatches.append(f"{field}: {before!r} -> {after!r}")
    if mismatches:
        raise PreflightError(
            "refusing to resume: the run's immutable identity changed, so its "
            "recorded history would not describe it. " + "; ".join(mismatches)
        )

def assert_directional_resume_compatible(
    previous: DirectionalResumeState,
    current: DirectionalResumeState,
    *,
    observed_parent_checkpoint_hash: str,
    observed_router_optimizer_state_hash: str,
) -> None:

    if not isinstance(previous, DirectionalResumeState) or not isinstance(
        current, DirectionalResumeState
    ):
        raise PreflightError(
            "directional resume compatibility requires DirectionalResumeState"
        )
    _require_hex_identity(
        "observed_parent_checkpoint_hash",
        observed_parent_checkpoint_hash,
        length=64,
    )
    _require_hex_identity(
        "observed_router_optimizer_state_hash",
        observed_router_optimizer_state_hash,
        length=64,
    )

    same_stage = previous.stage == current.stage
    warmup_to_joint = (
        previous.stage == "bridge_warmup" and current.stage == "joint_adaptation"
    )
    if not (same_stage or warmup_to_joint):
        raise PreflightError(
            "refusing directional resume: stage transition must stay within one "
            "stage or be exactly bridge_warmup -> joint_adaptation"
        )

    mismatches = []
    for field in (
        "seed",
        "repo_commit",
        "upstream_commit",
        "common_checkpoint_hash",
        "split_hash",
        "parent_projection_hash",
        "parent_projection_file_hash",
        "parent_projection_spec_hash",
        "route_equivalent_passes",
        "dropout",
        "ranking_mode",
        "rank_override",
    ):
        before, after = getattr(previous, field), getattr(current, field)
        if before != after:
            mismatches.append(f"{field}: {before!r} -> {after!r}")

    if same_stage:
        for field in ("config_hash", "trainable_name_hash"):
            before, after = getattr(previous, field), getattr(current, field)
            if before != after:
                mismatches.append(f"{field}: {before!r} -> {after!r}")

    if current.parent_checkpoint_hash != observed_parent_checkpoint_hash:
        mismatches.append(
            "parent_checkpoint_hash: "
            f"recorded {current.parent_checkpoint_hash!r}, observed "
            f"{observed_parent_checkpoint_hash!r}"
        )
    if previous.optimizer_state_hash != observed_router_optimizer_state_hash:
        mismatches.append(
            "optimizer_state_hash: parent record "
            f"{previous.optimizer_state_hash!r}, observed router state "
            f"{observed_router_optimizer_state_hash!r}"
        )
    if current.optimizer_state_hash != observed_router_optimizer_state_hash:
        mismatches.append(
            "optimizer_state_hash: current launch "
            f"{current.optimizer_state_hash!r}, observed router state "
            f"{observed_router_optimizer_state_hash!r}"
        )
    if mismatches:
        raise PreflightError(
            "refusing directional resume: immutable identity or optimizer "
            "continuity changed. " + "; ".join(mismatches)
        )

def assert_device_budget(*, world_size: int, visible: int, cap: int) -> None:

    if world_size < 1:
        raise PreflightError(f"world size must be at least 1, got {world_size}")
    if world_size > cap:
        raise PreflightError(
            f"this host permits at most {cap} concurrent devices, and {world_size} "
            f"were requested. {visible} are visible and they are usually idle, but "
            f"idleness is not authorization on a shared machine"
        )
    if world_size > visible:
        raise PreflightError(
            f"{world_size} ranks requested but only {visible} device(s) are visible"
        )

def config_hash(config: Mapping) -> str:

    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()

def directory_hash(root: Path, pattern: str = "*.parquet") -> str:

    digest = hashlib.sha256()
    for path in sorted(Path(root).glob(pattern)):
        digest.update(path.name.encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest()

def write_atomic_json(path: Path, payload: Mapping) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)

def write_create_new_json(path: Path, payload: Mapping) -> None:

    write_create_new_bytes(path, canonical_json_bytes(payload))

def write_create_new_bytes(path: Path, content: bytes) -> None:

    if not isinstance(content, bytes):
        raise PreflightError("create-new terminal evidence must be bytes")

    destination, root = _validated_terminal_destination(Path(path))
    relative = destination.relative_to(root)
    parent_fd = _open_safe_parent(root, relative.parent)
    file_fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(relative.name, flags, 0o664, dir_fd=parent_fd)
        except FileExistsError as error:
            raise PreflightError(
                f"create-new terminal evidence already exists: {destination}"
            ) from error

        os.fsync(parent_fd)
        view = memoryview(content)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("write_create_new_bytes made no forward progress")
            view = view[written:]
        os.fsync(file_fd)
        os.fsync(parent_fd)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)

def canonical_json_bytes(payload: Mapping) -> bytes:

    if not isinstance(payload, Mapping):
        raise PreflightError("create-new JSON payload must be a mapping")
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PreflightError(
            f"create-new JSON payload is not canonicalizable: {error}"
        ) from error

def read_terminal_bytes(path: Path) -> bytes:

    destination, root = _validated_terminal_destination(Path(path))
    relative = destination.relative_to(root)
    try:
        parent_fd = _open_existing_safe_parent(root, relative.parent)
    except OSError as error:
        raise PreflightError(
            f"cannot safely read terminal evidence {destination}: {error}"
        ) from error
    file_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(relative.name, flags, dir_fd=parent_fd)
        stat = os.fstat(file_fd)
        if not stat_module.S_ISREG(stat.st_mode):
            raise PreflightError(
                f"terminal evidence is not a regular file: {destination}"
            )
        chunks = []
        while chunk := os.read(file_fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as error:
        raise PreflightError(
            f"cannot safely read terminal evidence {destination}: {error}"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)

def _validated_terminal_destination(path: Path) -> tuple[Path, Path]:
    root_path = Path(_TERMINAL_EVIDENCE_ROOT)
    try:
        root = root_path.resolve(strict=True)
    except OSError as error:
        raise PreflightError(
            f"fixed terminal evidence root is unavailable: {root_path}: {error}"
        ) from error
    if not root.is_dir():
        raise PreflightError(f"fixed terminal evidence root is not a directory: {root}")

    destination = Path(os.path.abspath(Path(path)))
    denied = tuple(
        Path(prefix).resolve(strict=False) for prefix in _TERMINAL_DENIED_PREFIXES
    )
    bulk = Path(_TERMINAL_BULK_ROOT).resolve(strict=False)
    forbidden = [*denied, bulk]
    if any(
        destination == prefix or prefix in destination.parents for prefix in forbidden
    ):
        raise PreflightError(
            f"terminal evidence path is under a denied or bulk root: {destination}"
        )
    if destination == root or root not in destination.parents:
        raise PreflightError(
            f"terminal evidence path must stay under fixed group evidence root "
            f"{root}: {destination}"
        )
    if not destination.name or destination.name in {".", ".."}:
        raise PreflightError(f"terminal evidence path has no filename: {destination}")
    return destination, root

def _open_safe_parent(root: Path, relative_parent: Path) -> int:

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(root, directory_flags)
    try:
        for part in relative_parent.parts:
            if part in {"", ".", ".."}:
                raise PreflightError(
                    f"unsafe terminal evidence parent component {part!r}"
                )
            try:
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o2775, dir_fd=current_fd)
                except FileExistsError:

                    pass
                os.fsync(current_fd)
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except OSError as error:
                raise PreflightError(
                    f"cannot safely open terminal evidence parent {part!r}: {error}"
                ) from error
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise

def _open_existing_safe_parent(root: Path, relative_parent: Path) -> int:

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(root, directory_flags)
    try:
        for part in relative_parent.parts:
            if part in {"", ".", ".."}:
                raise PreflightError(
                    f"unsafe terminal evidence parent component {part!r}"
                )
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise
