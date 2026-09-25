
from __future__ import annotations

import sys

import errno
import grp
import hashlib
import json
import math
import os
import re
import stat as stat_module
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_COMMIT,
    EMERGENT_UPSTREAM_ROOT,
    UPSTREAM_ROOT,
)
from dive.training.preflight import PreflightError, canonical_json_bytes

SECONDS_PER_GPU_DAY = 86_400
AMENDMENT_CEILING = 32 * SECONDS_PER_GPU_DAY
PROJECT_PLANNED_CAP = 84 * SECONDS_PER_GPU_DAY
PROJECT_TOTAL = 120 * SECONDS_PER_GPU_DAY
MIN_RESERVE = 36 * SECONDS_PER_GPU_DAY
OBSERVED_PROJECT_GPU_SECONDS = 220_279
L40S_TOTAL_MEMORY_BYTES = 47_695_921_152
SIGNED_VALUE_UPSTREAM_COMMIT = "916eaaedce5b07c205efb6ef32370c01d366591e"

FIXED_CAMPAIGN_DEVICE_ISLANDS = ((0, 1, 2, 3), (4, 5, 6, 7))

FIXED_CAMPAIGN_DEVICES = (4, 5, 6, 7)

_MIN_FREE_BYTES = 16 * 1024**3
FULL_CHAIN_GPU_SECONDS = 440_640
DIVE_REQUIRED_ANCESTOR = "107fb0b44e5e6995ab9a40abd1d87b53ec76a09a"
PARENT_PROJECTION_COMPLETION_SHA256 = (
    "6c1f171023716d568917e4d4f6341b638b6d4da2ba95c6e1dfaf1202fa92bd6a"
)

PARENT_PROJECTION_SEMANTIC_SHA256 = (
    "354be8923a6a31982a305f3fd4ca96c7a2d3b0d2fd2cf1d52de075d1f36183c6"
)
PARENT_PROJECTION_FILE_SHA256 = (
    "5c0d44946d7a295360d9ec91a2f614e70379dd3007e2d31e27783ce7ae511ed4"
)
PARENT_PROJECTION_SPEC_SHA256 = (
    "64658c5f0ec1b3365e18a2d59b9c79891d580f6dcce0e286f93ed0c9ff041bf7"
)

COMMON_CHECKPOINT_SHA256 = (
    "589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9"
)
COMMON_CHECKPOINT_SIZE_BYTES = 2_934_289_381
AUTOENCODER_CHECKPOINT_SHA256 = (
    "35f8865efd269995eeaf1670e1c1085acfe2988c40abdeda8e09a0e15eb40816"
)
AUTOENCODER_CHECKPOINT_SIZE_BYTES = 4_100_101_779

_TERMINAL_EVIDENCE_ROOT = EMERGENT_EVIDENCE_ROOT
_TERMINAL_BULK_ROOT = EMERGENT_BULK_ROOT
_TERMINAL_DENIED_PREFIXES = DENIED_PREFIXES
_EXECUTING_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIGNED_UPSTREAM_ROOT = UPSTREAM_ROOT
_EMERGENT_UPSTREAM_ROOT = EMERGENT_UPSTREAM_ROOT
_EXPECTED_EVIDENCE_GID = grp.getgrnam("romerolab").gr_gid

_attempt_admission_validated_test_hook: Callable[[], None] | None = None

_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

_PROVENANCE_KEYS = frozenset(
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
_TRAINING_KEYS = frozenset(
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
_COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint",
        "step_completed",
        "resume_state",
        "ownership",
        "validation_record",
    }
)
_RESUME_KEYS = frozenset(
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
_OWNERSHIP_KEYS = frozenset(
    {
        "stage",
        "trainable_names",
        "trainable_name_count",
        "trainable_parameters",
        "groups",
    }
)
_ATTEMPT_START_KEYS = frozenset(
    {
        "schema_version",
        "attempt_id",
        "run_id",
        "preflight",
        "preparation_success",
        "action_profile",
        "seed",
        "resume",
        "reserved_log",
        "started_monotonic_ns",
    }
)
_ATTEMPT_TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "attempt_id",
        "run_id",
        "start_sha256",
        "status",
        "elapsed_seconds",
        "allocated_gpu_count",
        "completed_monotonic_ns",
        "error",
    }
)

class CampaignError(RuntimeError):
    pass

def _assert_launch_policy_devices(devices: object) -> tuple[int, ...]:

    if not isinstance(devices, list):
        raise CampaignError("launch policy devices are not a frozen GPU island")
    return _assert_device_island(tuple(devices))

def _assert_device_island(devices: object) -> tuple[int, ...]:

    if devices not in FIXED_CAMPAIGN_DEVICE_ISLANDS:
        raise CampaignError(
            f"campaign devices {devices!r} are not exactly one frozen GPU island"
        )
    assert isinstance(devices, tuple)
    return devices

class TornRecordRead(CampaignError):
    pass

_STAGING_NAME = re.compile(r"\.[A-Za-z0-9_.-]+\.staging\.[0-9a-f]{32}")

@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    returncode: int
    stdout: str
    stderr: str

def _subprocess_command_adapter(
    argv: tuple[str, ...], cwd: Path | None
) -> DiscoveryResult:
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return DiscoveryResult(result.returncode, result.stdout, result.stderr)

_COMMAND_ADAPTER = _subprocess_command_adapter

_GPU_INVENTORY_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.free,uuid",
    "--format=csv,noheader,nounits",
)
_GPU_PROCESS_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits",
)

def _discovery_command_is_allowed(argv: tuple[str, ...]) -> bool:
    if argv in {_GPU_INVENTORY_QUERY, _GPU_PROCESS_QUERY}:
        return True
    if argv in {
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
    }:
        return True
    return (
        len(argv) == 5
        and argv[:3] == ("git", "merge-base", "--is-ancestor")
        and _HEX40.fullmatch(argv[3]) is not None
        and argv[4] == "HEAD"
    )

def _run_discovery(argv: tuple[str, ...], *, cwd: Path | None) -> DiscoveryResult:
    if not isinstance(argv, tuple) or not _discovery_command_is_allowed(argv):
        raise CampaignError(
            f"discovery command is outside the exact allowlist: {argv!r}"
        )
    return _COMMAND_ADAPTER(argv, cwd)

@dataclass(frozen=True, slots=True)
class ActionProfile:
    profile_id: str
    mode: str
    stage: str
    config_path: Path
    config_sha256: str
    ranking_mode: str
    rank_override: float | None
    devices: tuple[int, ...]
    max_steps: int
    worst_case_wall_seconds: int
    worst_case_gpu_seconds: int

    def as_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "mode": self.mode,
            "stage": self.stage,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "ranking_mode": self.ranking_mode,
            "rank_override": self.rank_override,
            "devices": list(self.devices),
            "max_steps": self.max_steps,
            "worst_case_wall_seconds": self.worst_case_wall_seconds,
            "worst_case_gpu_seconds": self.worst_case_gpu_seconds,
        }

_ACTION_PROFILE_SPECS = {

    "configs/emergent/directional_bridge_smoke.yaml": (
        "directional-bridge-smoke-real-v1",
        "9121aaaeb3fe7bf57d94080566ca925ebd25cf4be9e9f77998ffd8f04049ee20",
        "bridge_warmup",
        "matched",
        None,
        6,
        3_600,
        14_400,

        FIXED_CAMPAIGN_DEVICE_ISLANDS[0],
    ),
    "configs/emergent/directional_bridge_warmup.yaml": (
        "directional-warmup-real-v1",
        "3b132b616b37ba55c45ce52abd4404d0a43f1efc6df0f8b2f2e43f91feeccbaf",
        "bridge_warmup",
        "matched",
        None,
        10_000,
        36_720,
        146_880,
        FIXED_CAMPAIGN_DEVICES,
    ),
    "configs/emergent/directional_bridge_joint.yaml": (
        "directional-joint-real-v1",
        "f7b10ac2e7c390347c9d1b94461a28eacf911af3f0d47229ef139084abd1076e",
        "joint_adaptation",
        "matched",
        None,
        20_000,
        73_440,
        293_760,
        FIXED_CAMPAIGN_DEVICES,
    ),
    "configs/emergent/directional_bridge_no_ranking_warmup.yaml": (
        "directional-no-ranking-warmup-real-v1",
        "be7401bf0a3acd633ef0486f782b56d22aa7c812d1c6a746082ae354d1846b59",
        "bridge_warmup",
        "no_ranking",
        0.0,
        10_000,
        36_720,
        146_880,
        FIXED_CAMPAIGN_DEVICES,
    ),
    "configs/emergent/directional_bridge_no_ranking_joint.yaml": (
        "directional-no-ranking-joint-real-v1",
        "a5d5c8afac99294d9bb81f95d37e41764f72cf194c070f679b6afe4c0515e724",
        "joint_adaptation",
        "no_ranking",
        0.0,
        20_000,
        73_440,
        293_760,
        FIXED_CAMPAIGN_DEVICES,
    ),
}

def action_profile_for(config_path: Path) -> ActionProfile:
    path = _absolute(config_path)
    repo_root = _fixed_directory(_EXECUTING_REPO_ROOT, "executing repository")
    try:
        relative = path.relative_to(repo_root).as_posix()
        values = _ACTION_PROFILE_SPECS[relative]
    except (ValueError, KeyError) as error:
        raise CampaignError("config has no frozen real action profile") from error
    _capture_file_identity(
        path,
        label="frozen action profile config",
        expected_sha256=values[1],
    )
    return ActionProfile(
        profile_id=values[0],
        mode="real",
        stage=values[2],
        config_path=path,
        config_sha256=values[1],
        ranking_mode=values[3],
        rank_override=values[4],
        devices=_assert_device_island(values[8]),
        max_steps=values[5],
        worst_case_wall_seconds=values[6],
        worst_case_gpu_seconds=values[7],
    )

_WALL_RESERVE_FRACTION = 0.05
_MIN_WALL_RESERVE_SECONDS = 120.0
_MAX_WALL_RESERVE_SECONDS = 900.0

def attempt_wall_reserve_seconds(worst_case_wall_seconds: object) -> float:

    seconds = _require_positive_integer(
        "worst case wall seconds", worst_case_wall_seconds
    )
    return max(
        _MIN_WALL_RESERVE_SECONDS,
        min(_MAX_WALL_RESERVE_SECONDS, _WALL_RESERVE_FRACTION * seconds),
    )

_WATCHDOG_MARGIN_SECONDS = 45
_WATCHDOG_KILL_AFTER_SECONDS = 20

_LAUNCH_STARTUP_ALLOWANCE_SECONDS = 60.0

def attempt_watchdog_seconds(worst_case_wall_seconds: object) -> tuple[int, int]:

    seconds = _require_positive_integer(
        "worst case wall seconds", worst_case_wall_seconds
    )
    reserve = attempt_wall_reserve_seconds(seconds)
    terminate = seconds - _WATCHDOG_MARGIN_SECONDS
    then_kill = _WATCHDOG_KILL_AFTER_SECONDS
    if terminate <= 0 or terminate + then_kill >= seconds:
        raise CampaignError("watchdog budget does not fit inside the frozen wall cap")
    if reserve - _WATCHDOG_MARGIN_SECONDS < _LAUNCH_STARTUP_ALLOWANCE_SECONDS:
        raise CampaignError("watchdog margin leaves no room for launch startup skew")
    return terminate, then_kill

def _capture_repository(
    root: Path,
    *,
    label: str,
    expected_commit: str | None,
    required_ancestor: str,
) -> RepositorySnapshot:
    fixed_root = _absolute(root)
    observed_root = os.lstat(fixed_root)
    if not stat_module.S_ISDIR(observed_root.st_mode) or stat_module.S_ISLNK(
        observed_root.st_mode
    ):
        raise CampaignError(f"{label} repository root is not a real directory")
    head = _run_discovery(("git", "rev-parse", "HEAD"), cwd=fixed_root)
    commit = head.stdout.strip()
    if head.returncode != 0 or _HEX40.fullmatch(commit) is None:
        raise CampaignError(f"cannot capture {label} repository HEAD")
    status = _run_discovery(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=fixed_root,
    )
    if status.returncode != 0 or status.stdout:
        raise CampaignError(f"{label} repository is dirty or status failed")
    ancestry = _run_discovery(
        ("git", "merge-base", "--is-ancestor", required_ancestor, "HEAD"),
        cwd=fixed_root,
    )
    if ancestry.returncode != 0:
        raise CampaignError(f"{label} repository ancestry is invalid")
    if expected_commit is not None and commit != expected_commit:
        raise CampaignError(f"{label} repository is at the wrong exact pin")
    return RepositorySnapshot(fixed_root, commit, True)

def _classify_compute_process(pid: int) -> tuple[str, bool]:
    _require_positive_integer("compute pid", pid)
    proc_root = Path("/proc") / str(pid)
    try:
        raw = (proc_root / "cmdline").read_bytes()
        tokens = tuple(
            item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item
        )
        cwd = os.readlink(proc_root / "cwd")
        executable = os.readlink(proc_root / "exe")
    except (OSError, UnicodeDecodeError) as error:
        raise CampaignError(f"cannot safely classify compute PID {pid}") from error
    if not tokens:
        raise CampaignError(f"compute PID {pid} has an empty command")
    command = " ".join(tokens)
    repo = str(_fixed_directory(_EXECUTING_REPO_ROOT, "executing repository"))
    belongs = any(
        marker in value
        for value in (command, cwd, executable)
        for marker in (repo, "scripts/emergent/train_directional.py")
    )
    return command, belongs

def _capture_gpus() -> tuple[GpuSnapshot, ...]:
    inventory_result = _run_discovery(_GPU_INVENTORY_QUERY, cwd=None)
    if inventory_result.returncode != 0:
        raise CampaignError("cannot capture GPU inventory")
    raw_gpus: list[tuple[int, str, int, str]] = []
    for line in inventory_result.stdout.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 5:
            raise CampaignError("GPU inventory row schema is not exact")
        try:
            index = int(fields[0])
            total_bytes = int(fields[2]) * 1024 * 1024
            int(fields[3])
        except ValueError as error:
            raise CampaignError("GPU inventory numeric field is invalid") from error
        if not fields[4] or any(item[3] == fields[4] for item in raw_gpus):
            raise CampaignError("GPU inventory UUID is absent or duplicate")
        raw_gpus.append((index, fields[1], total_bytes, fields[4]))
    process_result = _run_discovery(_GPU_PROCESS_QUERY, cwd=None)
    if process_result.returncode != 0:
        raise CampaignError("cannot capture GPU compute processes")
    by_uuid: dict[str, list[ComputeProcess]] = {item[3]: [] for item in raw_gpus}
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != 3 or fields[0] not in by_uuid:
            raise CampaignError("GPU process row schema or UUID is invalid")
        try:
            pid = int(fields[1])
            int(fields[2])
        except ValueError as error:
            raise CampaignError("GPU process numeric field is invalid") from error
        command, belongs = _classify_compute_process(pid)
        by_uuid[fields[0]].append(ComputeProcess(pid, command, belongs))
    return tuple(
        GpuSnapshot(index, name, total_bytes, tuple(by_uuid[uuid]))
        for index, name, total_bytes, uuid in raw_gpus
    )

def _capture_file_identity(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> FileIdentity:
    destination = _absolute(path)
    try:
        descriptor = os.open(
            destination,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise CampaignError(f"cannot capture {label}: {destination}") from error
    try:
        identity = _capture_open_file_identity(
            descriptor, destination=destination, label=label
        )
    finally:
        os.close(descriptor)
    if expected_sha256 is not None and identity.sha256 != expected_sha256:
        raise CampaignError(f"{label} hash differs from reviewed identity")
    if expected_size_bytes is not None and identity.size_bytes != expected_size_bytes:
        raise CampaignError(f"{label} size differs from reviewed identity")
    return identity

def _capture_open_file_identity(
    descriptor: int, *, destination: Path, label: str
) -> FileIdentity:
    before = os.fstat(descriptor)
    if not stat_module.S_ISREG(before.st_mode):
        raise CampaignError(f"{label} is not a regular file")
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, 1 << 20):
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != before.st_size:
        raise CampaignError(f"{label} changed while hashing")
    return FileIdentity(destination, digest.hexdigest(), size)

def _capture_bulk_file_identity(relative: Path, *, label: str) -> FileIdentity:

    root = _fixed_directory(_TERMINAL_BULK_ROOT, "bulk")
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise CampaignError("bulk file path schema is unsafe")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd: int | None = None
    file_fd: int | None = None
    try:
        current_fd = os.open(root, directory_flags)
        for part in relative.parent.parts:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(relative.name, file_flags, dir_fd=current_fd)
        return _capture_open_file_identity(
            file_fd, destination=root / relative, label=label
        )
    except OSError as error:
        raise CampaignError(f"cannot safely traverse bulk path for {label}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if current_fd is not None:
            os.close(current_fd)

def _capture_storage(path: Path, *, label: str) -> StorageSnapshot:
    root = _fixed_directory(path, label)
    try:
        filesystem = os.statvfs(root)
    except OSError as error:
        raise CampaignError(f"cannot capture {label} free space") from error
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    if not os.access(root, os.W_OK | os.X_OK):
        raise CampaignError(f"{label} is not writable/traversable")
    return StorageSnapshot(root, True, free_bytes)

def _capture_parent_identities(
    resume: Path | None,
) -> tuple[
    FileIdentity | None,
    FileIdentity | None,
    FileIdentity | None,
    FileIdentity | None,
]:
    if resume is None:
        return None, None, None, None
    bulk_root = _fixed_directory(_TERMINAL_BULK_ROOT, "bulk")
    checkpoint = _absolute(resume)
    try:
        relative = checkpoint.relative_to(bulk_root)
    except ValueError as error:
        raise CampaignError("parent checkpoint is outside fixed bulk root") from error
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != "runs"
        or parts[2] != "checkpoints"
        or not parts[3].endswith(".ckpt")
        or _RUN_ID.fullmatch(parts[1]) is None
    ):
        raise CampaignError("parent checkpoint path schema is not exact")
    parent = _capture_bulk_file_identity(relative, label="parent checkpoint")
    evidence_root = _fixed_evidence_root()
    completion_path = (
        evidence_root
        / "runs"
        / parts[1]
        / "checkpoints"
        / f"{Path(parts[3]).stem}.json"
    )
    training_path = evidence_root / "runs" / parts[1] / "training.json"
    provenance_path = evidence_root / "runs" / parts[1] / "provenance.json"
    return (
        parent,
        _capture_file_identity(completion_path, label="parent completion"),
        _capture_file_identity(training_path, label="parent training"),
        _capture_file_identity(provenance_path, label="parent provenance"),
    )

def capture_live_state(
    *, config_path: Path, resume: Path | None, log_path: Path
) -> LiveCapture:

    profile = action_profile_for(config_path)
    config = _capture_file_identity(
        profile.config_path,
        label="directional config",
        expected_sha256=profile.config_sha256,
    )
    repo = _capture_repository(
        _EXECUTING_REPO_ROOT,
        label="DIVE",
        expected_commit=None,
        required_ancestor=DIVE_REQUIRED_ANCESTOR,
    )
    signed = _capture_repository(
        _SIGNED_UPSTREAM_ROOT,
        label="signed-value upstream",
        expected_commit=SIGNED_VALUE_UPSTREAM_COMMIT,
        required_ancestor=SIGNED_VALUE_UPSTREAM_COMMIT,
    )
    emergent = _capture_repository(
        _EMERGENT_UPSTREAM_ROOT,
        label="emergent upstream",
        expected_commit=EMERGENT_UPSTREAM_COMMIT,
        required_ancestor=EMERGENT_UPSTREAM_COMMIT,
    )
    common = _capture_file_identity(
        emergent.root / "ckpts/complexa.ckpt",
        label="common checkpoint",
        expected_sha256=COMMON_CHECKPOINT_SHA256,
        expected_size_bytes=COMMON_CHECKPOINT_SIZE_BYTES,
    )
    autoencoder = _capture_file_identity(
        emergent.root / "ckpts/complexa_ae.ckpt",
        label="autoencoder checkpoint",
        expected_sha256=AUTOENCODER_CHECKPOINT_SHA256,
        expected_size_bytes=AUTOENCODER_CHECKPOINT_SIZE_BYTES,
    )
    projection_artifact = _capture_file_identity(
        _fixed_evidence_root() / "directional_parent_projections/v1.json",
        label="parent projection artifact",
        expected_sha256=PARENT_PROJECTION_FILE_SHA256,
    )
    projection_completion = _capture_file_identity(
        _fixed_evidence_root() / "directional_parent_projections/v1.completion.json",
        label="parent projection completion",
        expected_sha256=PARENT_PROJECTION_COMPLETION_SHA256,
    )
    projection_spec = _capture_file_identity(
        repo.root / "configs/emergent/directional_parent_projection_v1.json",
        label="parent projection reviewed spec",
        expected_sha256=PARENT_PROJECTION_SPEC_SHA256,
    )
    parent, completion, training, provenance = _capture_parent_identities(resume)
    proposed_log = validate_terminal_evidence_path(log_path)
    snapshot = SystemSnapshot(
        repo=repo,
        signed_value_upstream=signed,
        emergent_upstream=emergent,
        common_checkpoint=common,
        autoencoder_checkpoint=autoencoder,
        parent_checkpoint=parent,
        checkpoint_completion=completion,
        gpus=_capture_gpus(),
        evidence_storage=_capture_storage(
            _TERMINAL_EVIDENCE_ROOT, label="terminal evidence"
        ),
        bulk_storage=_capture_storage(_TERMINAL_BULK_ROOT, label="bulk"),
        projection=ProjectionIdentity(
            PARENT_PROJECTION_SEMANTIC_SHA256,
            PARENT_PROJECTION_FILE_SHA256,
            PARENT_PROJECTION_SPEC_SHA256,
        ),
        reserved_log=proposed_log,
        projection_files=(
            projection_artifact,
            projection_completion,
            projection_spec,
        ),
        parent_training=training,
        parent_provenance=provenance,
    )
    return LiveCapture(profile, config, CampaignLedger.from_evidence(), snapshot)

def validate_preparation_identity(*, run_id: str, seed: int) -> None:

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise CampaignError("run id is invalid")
    if type(seed) is not int or seed not in (42, 314159):
        raise CampaignError("seed is not one of the two frozen directional seeds")
    evidence = _fixed_evidence_root()
    bulk = _fixed_directory(_TERMINAL_BULK_ROOT, "bulk")
    candidates = (
        evidence / "runs" / run_id,
        evidence / "preflights" / f"{run_id}.json",
        evidence / "preparation_dispositions" / f"{run_id}.json",
        evidence / "attempts" / run_id,
        bulk / "runs" / run_id,
    )
    if any(os.path.lexists(path) for path in candidates):
        raise CampaignError(f"create-new run id already exists: {run_id}")

def validate_run_candidate(
    *,
    run_id: str,
    seed: int,
    profile: ActionProfile,
    config: FileIdentity,
    ledger: CampaignLedger,
    snapshot: SystemSnapshot,
) -> None:

    validate_preparation_identity(run_id=run_id, seed=seed)
    if (
        not isinstance(profile, ActionProfile)
        or not isinstance(config, FileIdentity)
        or not isinstance(ledger, CampaignLedger)
        or not isinstance(snapshot, SystemSnapshot)
    ):
        raise CampaignError("preflight candidate inputs have the wrong type")
    if (
        _absolute(config.path) != profile.config_path
        or config.sha256 != profile.config_sha256
    ):
        raise CampaignError("config identity differs from frozen action profile")
    _require_ranking(profile.ranking_mode, profile.rank_override)
    repo_root = _fixed_directory(_EXECUTING_REPO_ROOT, "executing repository")
    if _absolute(snapshot.repo.root) != repo_root or not snapshot.repo.clean:
        raise CampaignError("executing DIVE repository snapshot is dirty or mismatched")
    signed_root = _fixed_directory(_SIGNED_UPSTREAM_ROOT, "signed-value upstream")
    emergent_root = _fixed_directory(_EMERGENT_UPSTREAM_ROOT, "emergent upstream")
    _assert_repository_snapshot(
        snapshot.signed_value_upstream,
        root=signed_root,
        commit=SIGNED_VALUE_UPSTREAM_COMMIT,
        label="signed-value upstream",
    )
    _assert_repository_snapshot(
        snapshot.emergent_upstream,
        root=emergent_root,
        commit=EMERGENT_UPSTREAM_COMMIT,
        label="emergent upstream",
    )
    if (
        _absolute(config.path) == repo_root
        or repo_root not in _absolute(config.path).parents
    ):
        raise CampaignError("config identity must stay below the executing repository")
    _assert_frozen_checkpoint(
        snapshot.common_checkpoint,
        path=emergent_root / "ckpts/complexa.ckpt",
        sha256=COMMON_CHECKPOINT_SHA256,
        size_bytes=COMMON_CHECKPOINT_SIZE_BYTES,
        label="common checkpoint",
    )
    _assert_frozen_checkpoint(
        snapshot.autoencoder_checkpoint,
        path=emergent_root / "ckpts/complexa_ae.ckpt",
        sha256=AUTOENCODER_CHECKPOINT_SHA256,
        size_bytes=AUTOENCODER_CHECKPOINT_SIZE_BYTES,
        label="autoencoder checkpoint",
    )
    _assert_parent_contract(
        snapshot,
        ledger=ledger,
        stage=profile.stage,
        seed=seed,
        ranking_mode=profile.ranking_mode,
        rank_override=profile.rank_override,
    )
    _assert_projection(snapshot.projection)
    _validate_gpu_inventory(snapshot.gpus, profile.devices)
    evidence_root = _fixed_evidence_root()
    bulk_root = _fixed_directory(_TERMINAL_BULK_ROOT, "bulk")
    _assert_storage(snapshot.evidence_storage, evidence_root, "evidence")
    _assert_storage(snapshot.bulk_storage, bulk_root, "bulk")
    authorize_projection(ledger, profile.worst_case_gpu_seconds)

def _require_nonnegative_integer(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise CampaignError(f"{name} must be a literal nonnegative integer")
    return value

def _require_positive_integer(name: str, value: object) -> int:
    result = _require_nonnegative_integer(name, value)
    if result == 0:
        raise CampaignError(f"{name} must be positive")
    return result

def _require_hex(name: str, value: object, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise CampaignError(f"{name} has an invalid lowercase hexadecimal identity")
    return value

def _require_finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if (
        not math.isfinite(number)
        or number < 0
        or (number == 0.0 and math.copysign(1.0, number) < 0)
    ):
        raise CampaignError(f"{name} must be a finite nonnegative number")
    return number

def _require_ranking(ranking_mode: object, rank_override: object) -> None:
    matched = type(ranking_mode) is str and ranking_mode == "matched"
    no_ranking = (
        type(ranking_mode) is str
        and ranking_mode == "no_ranking"
        and type(rank_override) is float
        and rank_override == 0.0
        and math.copysign(1.0, rank_override) == 1.0
    )
    if not ((matched and rank_override is None) or no_ranking):
        raise CampaignError(
            "ranking mode must be exactly matched/None or no_ranking/+0.0"
        )

@dataclass(frozen=True, slots=True)
class RunDebit:
    run_id: str
    elapsed_seconds: float
    allocated_gpu_count: int
    gpu_seconds: int
    provenance_sha256: str
    training_sha256: str

    def as_record(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class AttemptDebit:
    attempt_id: str
    run_id: str
    status: str
    elapsed_seconds: float
    allocated_gpu_count: int
    gpu_seconds: int
    start_sha256: str
    terminal_sha256: str | None

    def as_record(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class ImmutableRunRecord:
    provenance_path: Path
    provenance_sha256: str
    training_path: Path
    training_sha256: str

    def __post_init__(self) -> None:
        _require_hex("provenance_sha256", self.provenance_sha256, _HEX64)
        _require_hex("training_sha256", self.training_sha256, _HEX64)

    def as_record(self) -> dict[str, object]:
        return {
            "provenance": {
                "path": str(self.provenance_path),
                "sha256": self.provenance_sha256,
            },
            "training": {
                "path": str(self.training_path),
                "sha256": self.training_sha256,
            },
        }

@dataclass(frozen=True, slots=True)
class CampaignProjection:
    observed_gpu_seconds: int
    prior_amendment_gpu_seconds: int
    next_gpu_seconds: int
    amendment_gpu_seconds: int
    project_gpu_seconds: int
    reserve_gpu_seconds: int

    def as_record(self) -> dict[str, int]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class CampaignLedger:

    observed_gpu_seconds: int
    amendment_gpu_seconds: int
    run_debits: tuple[AttemptDebit, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative_integer("observed_gpu_seconds", self.observed_gpu_seconds)
        _require_nonnegative_integer(
            "amendment_gpu_seconds", self.amendment_gpu_seconds
        )
        if self.run_debits:
            run_ids = [item.run_id for item in self.run_debits]
            if len(run_ids) != len(set(run_ids)):
                raise CampaignError("duplicate run debit identity")
            if sum(item.gpu_seconds for item in self.run_debits) != (
                self.amendment_gpu_seconds
            ):
                raise CampaignError(
                    "amendment debit differs from reconstructed immutable runs"
                )

    @classmethod
    def from_evidence(cls) -> "CampaignLedger":
        root = _fixed_evidence_root()
        attempts = root / "attempts"
        debits: list[AttemptDebit] = []
        run_ids: set[str] = set()
        if os.path.lexists(attempts):
            observed_attempts = os.lstat(attempts)
            if not stat_module.S_ISDIR(
                observed_attempts.st_mode
            ) or stat_module.S_ISLNK(observed_attempts.st_mode):
                raise CampaignError("attempt evidence root is not a real directory")
            attempt_directories = sorted(attempts.iterdir(), key=lambda item: item.name)
        else:
            attempt_directories = []
        for directory in attempt_directories:
            observed_directory = os.lstat(directory)
            if (
                not stat_module.S_ISDIR(observed_directory.st_mode)
                or stat_module.S_ISLNK(observed_directory.st_mode)
                or _RUN_ID.fullmatch(directory.name) is None
            ):
                raise CampaignError("unknown attempt artifact at fixed evidence root")
            names = {item.name for item in directory.iterdir()}

            names -= {name for name in names if _STAGING_NAME.fullmatch(name)}
            if not names <= {"start.json", "terminal.json"}:
                raise CampaignError("unknown attempt artifact in attempt directory")
            if "start.json" not in names:
                raise CampaignError("orphan attempt terminal has no start record")
            debit = _debit_from_attempt(directory)
            if debit.run_id in run_ids:
                raise CampaignError("duplicate directional attempt run id")
            run_ids.add(debit.run_id)
            debits.append(debit)
        by_run = {item.run_id: item for item in debits}
        reconciled_successes: set[str] = set()
        runs = root / "runs"
        if os.path.lexists(runs):
            observed_runs = os.lstat(runs)
            if not stat_module.S_ISDIR(observed_runs.st_mode) or stat_module.S_ISLNK(
                observed_runs.st_mode
            ):
                raise CampaignError(
                    "directional run evidence root is not a real directory"
                )
            for run_directory in sorted(runs.iterdir(), key=lambda item: item.name):
                observed_run = os.lstat(run_directory)
                if not stat_module.S_ISDIR(observed_run.st_mode) or stat_module.S_ISLNK(
                    observed_run.st_mode
                ):
                    raise CampaignError("unknown run artifact at fixed evidence root")
                provenance_path = run_directory / "provenance.json"
                training_path = run_directory / "training.json"
                if not os.path.lexists(provenance_path):
                    if os.path.lexists(training_path):
                        label = "unexplained run training"
                        training_only, _, training_only_canonical = (
                            _read_unpinned_classifiable_json(training_path, label=label)
                        )
                        if _is_directional_record(training_only):
                            if not training_only_canonical:
                                raise CampaignError(f"{label} is not canonical JSON")
                            raise CampaignError(
                                "directional training has no provenance"
                            )
                    continue
                provenance, provenance_sha, provenance_canonical = (
                    _read_unpinned_classifiable_json(
                        provenance_path, label="run provenance"
                    )
                )
                directional = _is_directional_record(provenance)
                if not directional:
                    if os.path.lexists(training_path):
                        label = "unexplained run training"
                        unexplained_training, _, unexplained_canonical = (
                            _read_unpinned_classifiable_json(training_path, label=label)
                        )
                        if _is_directional_record(unexplained_training):
                            if not unexplained_canonical:
                                raise CampaignError(f"{label} is not canonical JSON")
                            raise CampaignError(
                                "directional training has unrelated provenance"
                            )
                    continue
                if not provenance_canonical:
                    raise CampaignError("run provenance is not canonical JSON")
                run_id = provenance.get("run_id")
                if run_id != run_directory.name or run_id not in by_run:
                    raise CampaignError(
                        "real directional run has no accounted attempt start"
                    )
                debit = by_run[run_id]
                if debit.status == "success":
                    if not os.path.lexists(training_path):
                        raise CampaignError(
                            "successful attempt has no canonical training record"
                        )
                    _, training_sha = _read_unpinned_canonical_json(
                        training_path, label="run training"
                    )
                    run_debit = _debit_from_record(
                        ImmutableRunRecord(
                            provenance_path,
                            provenance_sha,
                            training_path,
                            training_sha,
                        )
                    )
                    _reconcile_successful_attempt(
                        attempts / run_id,
                        debit=debit,
                        attempts_by_run=by_run,
                        run_debit=run_debit,
                        provenance_path=provenance_path,
                        training_path=training_path,
                    )
                    reconciled_successes.add(run_id)
                elif os.path.lexists(training_path):
                    raise CampaignError(
                        "partial or failed attempt has an unaccounted training record"
                    )
        missing_successes = sorted(
            item.run_id
            for item in debits
            if item.status == "success" and item.run_id not in reconciled_successes
        )
        if missing_successes:
            raise CampaignError(
                f"successful attempt has no exact run artifacts: {missing_successes}"
            )
        return cls(
            OBSERVED_PROJECT_GPU_SECONDS,
            sum(item.gpu_seconds for item in debits),
            tuple(debits),
        )

    def as_record(self) -> dict[str, object]:
        return {
            "observed_gpu_seconds": self.observed_gpu_seconds,
            "amendment_gpu_seconds": self.amendment_gpu_seconds,
            "run_debits": [item.as_record() for item in self.run_debits],
        }

def authorize_projection(
    ledger: CampaignLedger, next_gpu_seconds: int
) -> CampaignProjection:
    if not isinstance(ledger, CampaignLedger):
        raise CampaignError("ledger must be a CampaignLedger")
    requested = _require_nonnegative_integer("next_gpu_seconds", next_gpu_seconds)
    amendment = ledger.amendment_gpu_seconds + requested
    project = ledger.observed_gpu_seconds + amendment
    reserve = PROJECT_TOTAL - project
    if amendment > AMENDMENT_CEILING:
        raise CampaignError("projected directional campaign exceeds 32.0 GPU-days")
    if project > PROJECT_PLANNED_CAP or reserve < MIN_RESERVE:
        raise CampaignError(
            "projected campaign exceeds the 84 GPU-day cap or violates the "
            "36 GPU-day reserve"
        )
    return CampaignProjection(
        observed_gpu_seconds=ledger.observed_gpu_seconds,
        prior_amendment_gpu_seconds=ledger.amendment_gpu_seconds,
        next_gpu_seconds=requested,
        amendment_gpu_seconds=amendment,
        project_gpu_seconds=project,
        reserve_gpu_seconds=reserve,
    )

def _action_profile_from_record(value: object) -> ActionProfile:
    if not isinstance(value, Mapping) or set(value) != {
        "profile_id",
        "mode",
        "stage",
        "config_path",
        "config_sha256",
        "ranking_mode",
        "rank_override",
        "devices",
        "max_steps",
        "worst_case_wall_seconds",
        "worst_case_gpu_seconds",
    }:
        raise CampaignError("attempt action profile schema is not exact")
    config_path = _absolute(Path(str(value["config_path"])))
    repo_root = _fixed_directory(_EXECUTING_REPO_ROOT, "executing repository")
    try:
        relative = config_path.relative_to(repo_root).as_posix()
        spec = _ACTION_PROFILE_SPECS[relative]
    except (ValueError, KeyError) as error:
        raise CampaignError("attempt action profile config is not frozen") from error
    profile = ActionProfile(
        profile_id=value["profile_id"],
        mode=value["mode"],
        stage=value["stage"],
        config_path=config_path,
        config_sha256=value["config_sha256"],
        ranking_mode=value["ranking_mode"],
        rank_override=value["rank_override"],
        devices=tuple(value["devices"]) if isinstance(value["devices"], list) else (),
        max_steps=value["max_steps"],
        worst_case_wall_seconds=value["worst_case_wall_seconds"],
        worst_case_gpu_seconds=value["worst_case_gpu_seconds"],
    )
    expected = ActionProfile(
        profile_id=spec[0],
        mode="real",
        stage=spec[2],
        config_path=config_path,
        config_sha256=spec[1],
        ranking_mode=spec[3],
        rank_override=spec[4],
        devices=_assert_device_island(spec[8]),
        max_steps=spec[5],
        worst_case_wall_seconds=spec[6],
        worst_case_gpu_seconds=spec[7],
    )
    if profile != expected:
        raise CampaignError("attempt action profile differs from frozen design basis")
    return profile

def _file_identity_from_record(value: object, *, label: str) -> FileIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise CampaignError(f"{label} identity schema is not exact")
    return FileIdentity(Path(str(value["path"])), value["sha256"], value["size_bytes"])

def _validate_run_preflight_record(
    record: Mapping[str, object], *, expected_run_id: str
) -> ActionProfile:
    expected_keys = set(RunPreflight.__dataclass_fields__) | {
        "schema_version",
        "launch_authorized",
    }
    if set(record) != expected_keys:
        raise CampaignError("attempt preflight schema is not exact and complete")
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["launch_authorized"] is not False
        or record["run_id"] != expected_run_id
        or type(record["seed"]) is not int
        or record["seed"] not in (42, 314159)
    ):
        raise CampaignError("attempt preflight top-level identity is invalid")
    profile = _action_profile_from_record(record["action_profile"])
    _require_ranking(record["ranking_mode"], record["rank_override"])
    if (
        record["stage"] != profile.stage
        or record["ranking_mode"] != profile.ranking_mode
        or record["rank_override"] != profile.rank_override
    ):
        raise CampaignError("attempt preflight profile/stage/ranking binding differs")
    config = _file_identity_from_record(record["config"], label="preflight config")
    if (
        _absolute(config.path) != profile.config_path
        or config.sha256 != profile.config_sha256
    ):
        raise CampaignError("attempt preflight config differs from frozen profile")

    projection = record["ledger_projection"]
    if not isinstance(projection, Mapping) or set(projection) != {
        "observed_gpu_seconds",
        "prior_amendment_gpu_seconds",
        "next_gpu_seconds",
        "amendment_gpu_seconds",
        "project_gpu_seconds",
        "reserve_gpu_seconds",
    }:
        raise CampaignError("attempt preflight ledger projection schema is not exact")
    for key, value in projection.items():
        _require_nonnegative_integer(f"preflight projection {key}", value)
    if (
        projection["observed_gpu_seconds"] != OBSERVED_PROJECT_GPU_SECONDS
        or projection["next_gpu_seconds"] != profile.worst_case_gpu_seconds
        or projection["amendment_gpu_seconds"]
        != projection["prior_amendment_gpu_seconds"] + profile.worst_case_gpu_seconds
        or projection["project_gpu_seconds"]
        != projection["observed_gpu_seconds"] + projection["amendment_gpu_seconds"]
        or projection["reserve_gpu_seconds"]
        != PROJECT_TOTAL - projection["project_gpu_seconds"]
    ):
        raise CampaignError("attempt preflight ledger projection arithmetic differs")

    repositories = record["repositories"]
    if not isinstance(repositories, list) or len(repositories) != 3:
        raise CampaignError("attempt preflight repository inventory is not exact")
    parsed_repositories: list[RepositorySnapshot] = []
    for value in repositories:
        if not isinstance(value, Mapping) or set(value) != {"root", "commit", "clean"}:
            raise CampaignError("attempt preflight repository schema is not exact")
        parsed_repositories.append(
            RepositorySnapshot(
                Path(str(value["root"])), value["commit"], value["clean"]
            )
        )
    expected_repositories = (
        (_absolute(_EXECUTING_REPO_ROOT), None),
        (_absolute(_SIGNED_UPSTREAM_ROOT), SIGNED_VALUE_UPSTREAM_COMMIT),
        (_absolute(_EMERGENT_UPSTREAM_ROOT), EMERGENT_UPSTREAM_COMMIT),
    )
    for observed, (root, commit) in zip(
        parsed_repositories, expected_repositories, strict=True
    ):
        if (
            _absolute(observed.root) != root
            or not observed.clean
            or (commit is not None and observed.commit != commit)
        ):
            raise CampaignError("attempt preflight repository binding differs")

    checkpoints = record["checkpoints"]
    expected_count = 2 if profile.stage == "bridge_warmup" else 3
    if not isinstance(checkpoints, list) or len(checkpoints) != expected_count:
        raise CampaignError("attempt preflight checkpoint inventory is not exact")
    parsed_checkpoints = tuple(
        _file_identity_from_record(value, label="preflight checkpoint")
        for value in checkpoints
    )
    _assert_frozen_checkpoint(
        parsed_checkpoints[0],
        path=_absolute(_EMERGENT_UPSTREAM_ROOT) / "ckpts/complexa.ckpt",
        sha256=COMMON_CHECKPOINT_SHA256,
        size_bytes=COMMON_CHECKPOINT_SIZE_BYTES,
        label="preflight common checkpoint",
    )
    _assert_frozen_checkpoint(
        parsed_checkpoints[1],
        path=_absolute(_EMERGENT_UPSTREAM_ROOT) / "ckpts/complexa_ae.ckpt",
        sha256=AUTOENCODER_CHECKPOINT_SHA256,
        size_bytes=AUTOENCODER_CHECKPOINT_SIZE_BYTES,
        label="preflight autoencoder checkpoint",
    )
    completion = record["checkpoint_completion"]
    if (profile.stage == "bridge_warmup") != (completion is None):
        raise CampaignError("attempt preflight completion differs from stage")
    if completion is not None:
        _file_identity_from_record(completion, label="preflight completion")
    parent_provenance = record["parent_provenance"]
    if (profile.stage == "bridge_warmup") != (parent_provenance is None):
        raise CampaignError("attempt preflight parent provenance differs from stage")
    if parent_provenance is not None:
        _file_identity_from_record(
            parent_provenance, label="preflight parent provenance"
        )
    parent_training = record["parent_training"]
    if (profile.stage == "bridge_warmup") != (parent_training is None):
        raise CampaignError("attempt preflight parent training differs from stage")
    if parent_training is not None:
        _file_identity_from_record(parent_training, label="preflight parent training")

    gpu_inventory = record["gpu_inventory"]
    if not isinstance(gpu_inventory, list):
        raise CampaignError("attempt preflight GPU inventory is not a list")
    parsed_gpus: list[GpuSnapshot] = []
    for value in gpu_inventory:
        if not isinstance(value, Mapping) or set(value) != {
            "index",
            "name",
            "total_memory_bytes",
            "processes",
        }:
            raise CampaignError("attempt preflight GPU schema is not exact")
        processes = value["processes"]
        if not isinstance(processes, list):
            raise CampaignError("attempt preflight GPU process inventory is invalid")
        parsed_processes: list[ComputeProcess] = []
        for process in processes:
            if not isinstance(process, Mapping) or set(process) != {
                "pid",
                "command",
                "belongs_to_dive",
            }:
                raise CampaignError("attempt preflight GPU process schema is not exact")
            parsed_processes.append(
                ComputeProcess(
                    process["pid"], process["command"], process["belongs_to_dive"]
                )
            )
        parsed_gpus.append(
            GpuSnapshot(
                value["index"],
                value["name"],
                value["total_memory_bytes"],
                tuple(parsed_processes),
            )
        )
    eligible = _validate_gpu_inventory(tuple(parsed_gpus), profile.devices)
    if record["eligible_idle_devices"] != list(eligible):
        raise CampaignError("attempt preflight eligible devices differ")

    storage = record["storage"]
    if not isinstance(storage, list) or len(storage) != 2:
        raise CampaignError("attempt preflight storage inventory is not exact")
    for value, root, label in zip(
        storage,
        (_TERMINAL_EVIDENCE_ROOT, _TERMINAL_BULK_ROOT),
        ("evidence", "bulk"),
        strict=True,
    ):
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "writable",
            "free_bytes",
        }:
            raise CampaignError("attempt preflight storage schema is not exact")
        _assert_storage(
            StorageSnapshot(
                Path(str(value["path"])), value["writable"], value["free_bytes"]
            ),
            _absolute(root),
            label,
        )
    projection_record = record["projection"]
    if not isinstance(projection_record, Mapping) or set(projection_record) != {
        "semantic_sha256",
        "file_sha256",
        "spec_sha256",
    }:
        raise CampaignError("attempt preflight projection schema is not exact")
    _assert_projection(
        ProjectionIdentity(
            projection_record["semantic_sha256"],
            projection_record["file_sha256"],
            projection_record["spec_sha256"],
        )
    )
    projection_files = record["projection_files"]
    if not isinstance(projection_files, list) or len(projection_files) != 3:
        raise CampaignError("attempt preflight projection files are not exact")
    parsed_projection_files = tuple(
        _file_identity_from_record(value, label="preflight projection file")
        for value in projection_files
    )
    expected_projection_files = (
        (
            _fixed_evidence_root() / "directional_parent_projections/v1.json",
            PARENT_PROJECTION_FILE_SHA256,
        ),
        (
            _fixed_evidence_root()
            / "directional_parent_projections/v1.completion.json",
            PARENT_PROJECTION_COMPLETION_SHA256,
        ),
        (
            _absolute(_EXECUTING_REPO_ROOT)
            / "configs/emergent/directional_parent_projection_v1.json",
            PARENT_PROJECTION_SPEC_SHA256,
        ),
    )
    if any(
        _absolute(observed.path) != expected_path or observed.sha256 != expected_hash
        for observed, (expected_path, expected_hash) in zip(
            parsed_projection_files, expected_projection_files, strict=True
        )
    ):
        raise CampaignError("attempt preflight projection file identity differs")
    log = record["reserved_log"]
    if not isinstance(log, Mapping) or set(log) != {"path", "device", "inode"}:
        raise CampaignError("attempt preflight reserved log schema is not exact")
    reserved_log = ReservedLog(Path(str(log["path"])), log["device"], log["inode"])
    success = _file_identity_from_record(
        record["preparation_success"], label="preparation success"
    )
    expected_success = _expected_preparation_success(
        run_id=expected_run_id,
        seed=record["seed"],
        profile=profile,
        reserved_log=reserved_log,
    )
    if success != expected_success:
        raise CampaignError("preparation success identity differs from preflight")
    return profile

def _debit_from_attempt(directory: Path) -> AttemptDebit:
    start, start_sha256 = _read_unpinned_canonical_json(
        directory / "start.json", label="attempt start"
    )
    if set(start) != _ATTEMPT_START_KEYS:
        raise CampaignError("attempt start schema is not exact")
    if type(start["schema_version"]) is not int or start["schema_version"] != 1:
        raise CampaignError("attempt start schema_version is not exact")
    attempt_id = start["attempt_id"]
    run_id = start["run_id"]
    if (
        type(attempt_id) is not str
        or attempt_id != directory.name
        or _RUN_ID.fullmatch(attempt_id) is None
        or type(run_id) is not str
        or _RUN_ID.fullmatch(run_id) is None
    ):
        raise CampaignError("attempt start identity is invalid")
    if type(start["seed"]) is not int or start["seed"] not in (42, 314159):
        raise CampaignError("attempt start seed is not frozen")
    if start["resume"] is not None and (
        type(start["resume"]) is not str or not start["resume"]
    ):
        raise CampaignError("attempt start resume identity is invalid")
    started_ns = _require_positive_integer(
        "attempt monotonic origin", start["started_monotonic_ns"]
    )
    profile = _action_profile_from_record(start["action_profile"])
    preflight = start["preflight"]
    if not isinstance(preflight, Mapping) or set(preflight) != {"path", "sha256"}:
        raise CampaignError("attempt preflight identity is not exact")
    preflight_path = Path(str(preflight["path"]))
    expected_preflight = _fixed_evidence_root() / "preflights" / f"{run_id}.json"
    if _absolute(preflight_path) != expected_preflight:
        raise CampaignError("attempt preflight path is not the fixed run identity")
    preflight_record = _read_canonical_terminal_json(
        preflight_path,
        expected_sha256=preflight["sha256"],
        label="attempt preflight",
    )
    preflight_profile = _validate_run_preflight_record(
        preflight_record, expected_run_id=run_id
    )
    preparation_success = _validate_preparation_success(
        preflight_record, expected_run_id=run_id
    )
    if start["preparation_success"] != preparation_success.as_record():
        raise CampaignError("attempt preparation success binding differs")
    log = start["reserved_log"]
    if not isinstance(log, Mapping) or set(log) != {"path", "device", "inode"}:
        raise CampaignError("attempt reserved log identity is not exact")
    log_path = validate_terminal_evidence_path(Path(str(log["path"])))
    observed_log = os.lstat(log_path)
    if (
        not stat_module.S_ISREG(observed_log.st_mode)
        or stat_module.S_ISLNK(observed_log.st_mode)
        or type(log["device"]) is not int
        or type(log["inode"]) is not int
        or (observed_log.st_dev, observed_log.st_ino) != (log["device"], log["inode"])
    ):
        raise CampaignError("attempt reserved log identity changed")
    if (
        preflight_record.get("run_id") != run_id
        or preflight_record.get("seed") != start["seed"]
        or preflight_profile != profile
        or preflight_record.get("reserved_log") != log
    ):
        raise CampaignError("attempt start differs from its bound preflight")
    if (profile.stage == "bridge_warmup") != (start["resume"] is None):
        raise CampaignError("attempt resume does not match its frozen stage")
    terminal_path = directory / "terminal.json"
    if not os.path.lexists(terminal_path):
        return AttemptDebit(
            attempt_id,
            run_id,
            "partial",
            float(profile.worst_case_wall_seconds),
            len(profile.devices),
            profile.worst_case_gpu_seconds,
            start_sha256,
            None,
        )
    terminal, terminal_sha256 = _read_unpinned_canonical_json(
        terminal_path, label="attempt terminal"
    )
    if set(terminal) != _ATTEMPT_TERMINAL_KEYS:
        raise CampaignError("attempt terminal schema is not exact")
    if (
        type(terminal["schema_version"]) is not int
        or terminal["schema_version"] != 1
        or terminal["attempt_id"] != attempt_id
        or terminal["run_id"] != run_id
        or terminal["start_sha256"] != start_sha256
    ):
        raise CampaignError("attempt terminal identity differs from start")
    status = terminal["status"]
    if status not in {"success", "failure"} or type(status) is not str:
        raise CampaignError("attempt terminal status is invalid")
    elapsed = _require_finite_nonnegative(
        "attempt terminal elapsed_seconds", terminal["elapsed_seconds"]
    )
    allocated = _require_positive_integer(
        "attempt terminal allocated_gpu_count", terminal["allocated_gpu_count"]
    )
    completed_ns = _require_positive_integer(
        "attempt terminal monotonic completion", terminal["completed_monotonic_ns"]
    )
    if allocated != len(profile.devices) or elapsed > profile.worst_case_wall_seconds:
        raise CampaignError("attempt terminal exceeds its frozen action profile")
    monotonic_elapsed = (completed_ns - started_ns) / 1_000_000_000
    if completed_ns < started_ns or abs(monotonic_elapsed - elapsed) > 1e-6:
        raise CampaignError("attempt terminal elapsed differs from monotonic interval")
    error = terminal["error"]
    if status == "success":
        if error is not None:
            raise CampaignError("successful attempt terminal unexpectedly has error")
    elif (
        not isinstance(error, Mapping)
        or set(error) != {"type", "message"}
        or any(type(error[key]) is not str for key in error)
    ):
        raise CampaignError("failed attempt terminal error schema is not exact")
    gpu_seconds = math.ceil(elapsed * allocated)
    if gpu_seconds > profile.worst_case_gpu_seconds:
        raise CampaignError("attempt terminal debit exceeds its frozen profile")
    return AttemptDebit(
        attempt_id,
        run_id,
        status,
        elapsed,
        allocated,
        gpu_seconds,
        start_sha256,
        terminal_sha256,
    )

def _debit_from_record(record: ImmutableRunRecord) -> RunDebit:
    provenance = _read_canonical_terminal_json(
        record.provenance_path,
        expected_sha256=record.provenance_sha256,
        label="provenance",
    )
    training = _read_canonical_terminal_json(
        record.training_path,
        expected_sha256=record.training_sha256,
        label="training",
    )
    if set(provenance) != _PROVENANCE_KEYS:
        raise CampaignError("Task 7 provenance schema is not exact")
    if set(training) != _TRAINING_KEYS:
        raise CampaignError("Task 7 training schema is not exact")
    if (
        type(provenance["schema_version"]) is not int
        or provenance["schema_version"] != 1
    ):
        raise CampaignError("Task 7 provenance schema_version must be exactly 1")
    if type(training["schema_version"]) is not int or training["schema_version"] != 1:
        raise CampaignError("Task 7 training schema_version must be exactly 1")
    run_id = provenance["run_id"]
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise CampaignError("Task 7 provenance run id is invalid")
    expected_parent = record.provenance_path.parent
    if (
        record.provenance_path.name != "provenance.json"
        or record.training_path.name != "training.json"
        or record.training_path.parent != expected_parent
        or expected_parent.name != run_id
    ):
        raise CampaignError("Task 7 record paths do not bind the same run id")
    if provenance["mode"] != "real" or training["mode"] != "real":
        raise CampaignError("only completed real Task 7 runs are campaign debits")
    stage = provenance["stage"]
    if type(stage) is not str or stage not in {"bridge_warmup", "joint_adaptation"}:
        raise CampaignError("Task 7 stage is not exact")
    seed = provenance["seed"]
    if type(seed) is not int or seed not in (42, 314159):
        raise CampaignError("Task 7 seed is not frozen")
    for field in ("repo_commit", "upstream_commit"):
        _require_hex(f"Task 7 {field}", provenance[field], _HEX40)
    if provenance["upstream_commit"] != EMERGENT_UPSTREAM_COMMIT:
        raise CampaignError("Task 7 emergent upstream pin changed")
    for field in (
        "config_hash",
        "split_hash",
        "common_checkpoint_hash",
        "autoencoder_checkpoint_hash",
    ):
        _require_hex(f"Task 7 {field}", provenance[field], _HEX64)
    if (
        provenance["common_checkpoint_hash"] != COMMON_CHECKPOINT_SHA256
        or provenance["autoencoder_checkpoint_hash"] != AUTOENCODER_CHECKPOINT_SHA256
    ):
        raise CampaignError("Task 7 frozen checkpoint identity changed")
    _assert_projection(
        ProjectionIdentity(
            provenance["parent_projection_hash"],
            provenance["parent_projection_file_hash"],
            provenance["parent_projection_spec_hash"],
        )
    )
    if (
        _require_finite_nonnegative(
            "route_equivalent_passes", provenance["route_equivalent_passes"]
        )
        != 8.0
        or _require_finite_nonnegative("dropout", provenance["dropout"]) != 0.0
    ):
        raise CampaignError("Task 7 route passes or dropout changed")
    parent = provenance["parent_checkpoint"]
    if (stage == "bridge_warmup" and parent is not None) or (
        stage == "joint_adaptation" and (type(parent) is not str or not parent)
    ):
        raise CampaignError("Task 7 parent checkpoint does not match its stage")
    launch_policy = provenance["launch_command_policy"]
    if not isinstance(launch_policy, Mapping) or set(launch_policy) != {
        "rendering_only",
        "requires_pre_reserved_log",
        "log_redirection",
        "interpreter",
        "devices",
        "nproc_per_node",
    }:
        raise CampaignError("Task 7 launch rendering policy changed")

    if {
        key: value for key, value in launch_policy.items() if key != "devices"
    } != {
        "rendering_only": True,
        "requires_pre_reserved_log": True,
        "log_redirection": "append",
        "interpreter": sys.executable,
        "nproc_per_node": 4,
    }:
        raise CampaignError("Task 7 launch rendering policy changed")
    _assert_launch_policy_devices(launch_policy["devices"])
    if (
        provenance["uses_gpu"] is not True
        or training["uses_gpu"] is not True
        or provenance["loads_blind_data"] is not False
        or training["loads_blind_data"] is not False
        or training["uses_synthetic_data"] is not False
    ):
        raise CampaignError("Task 7 run mode flags are not an exact real GPU run")
    for field in ("stage", "seed", "ranking_mode", "rank_override"):
        if provenance[field] != training[field]:
            raise CampaignError(f"Task 7 provenance/training {field} differs")
    _require_ranking(training["ranking_mode"], training["rank_override"])
    _assert_training_resume(provenance, training)
    allocated = _require_positive_integer(
        "allocated_gpu_count", training["allocated_gpu_count"]
    )
    if allocated != 4 or provenance["declared_devices"] != allocated:
        raise CampaignError("Task 7 allocated GPU count differs from four-device claim")
    elapsed = _require_finite_nonnegative(
        "elapsed_seconds", training["elapsed_seconds"]
    )
    _require_positive_integer("Task 7 steps", training["steps"])
    _require_nonnegative_integer("Task 7 start_step", training["start_step"])
    _require_hex("Task 7 checkpoint_hash", training["checkpoint_hash"], _HEX64)
    for field in ("checkpoint", "checkpoint_completion"):
        if type(training[field]) is not str or not training[field]:
            raise CampaignError(f"Task 7 {field} is not an exact path string")
    gpu_seconds = math.ceil(elapsed * allocated)
    return RunDebit(
        run_id=run_id,
        elapsed_seconds=elapsed,
        allocated_gpu_count=allocated,
        gpu_seconds=gpu_seconds,
        provenance_sha256=record.provenance_sha256,
        training_sha256=record.training_sha256,
    )

def _reconcile_successful_attempt(
    directory: Path,
    *,
    debit: AttemptDebit,
    attempts_by_run: Mapping[str, AttemptDebit],
    run_debit: RunDebit,
    provenance_path: Path,
    training_path: Path,
) -> None:
    start, _ = _read_unpinned_canonical_json(
        directory / "start.json", label="successful attempt start"
    )
    preflight_identity = start["preflight"]
    assert isinstance(preflight_identity, Mapping)
    preflight = _read_canonical_terminal_json(
        Path(str(preflight_identity["path"])),
        expected_sha256=preflight_identity["sha256"],
        label="successful attempt preflight",
    )
    profile = _validate_run_preflight_record(preflight, expected_run_id=debit.run_id)
    provenance, _ = _read_unpinned_canonical_json(
        provenance_path, label="successful run provenance"
    )
    training, _ = _read_unpinned_canonical_json(
        training_path, label="successful run training"
    )
    config = preflight["config"]
    if not isinstance(config, Mapping):
        raise CampaignError("successful preflight config is invalid")
    expected_resume = provenance["parent_checkpoint"]
    if expected_resume is not None:
        expected_resume = str(_absolute(Path(str(expected_resume))))
    observed_resume = start["resume"]
    if observed_resume is not None:
        observed_resume = str(_absolute(Path(str(observed_resume))))
    repositories = preflight["repositories"]
    checkpoints = preflight["checkpoints"]
    projection = preflight["projection"]
    assert isinstance(repositories, list)
    assert isinstance(checkpoints, list)
    assert isinstance(projection, Mapping)
    parent_checkpoint = None if profile.stage == "bridge_warmup" else checkpoints[2]
    training_resume = training["resume_state"]
    assert isinstance(training_resume, Mapping)
    if any(
        condition
        for condition in (
            run_debit.run_id != debit.run_id,
            debit.elapsed_seconds < run_debit.elapsed_seconds,
            debit.allocated_gpu_count != run_debit.allocated_gpu_count,
            provenance["seed"] != start["seed"],
            training["seed"] != start["seed"],
            provenance["stage"] != profile.stage,
            training["stage"] != profile.stage,
            provenance["config_path"] != str(profile.config_path),
            provenance["config_hash"] != profile.config_sha256,
            config.get("sha256") != profile.config_sha256,
            provenance["ranking_mode"] != profile.ranking_mode,
            training["ranking_mode"] != profile.ranking_mode,
            provenance["rank_override"] != profile.rank_override,
            training["rank_override"] != profile.rank_override,
            provenance["declared_devices"] != len(profile.devices),
            training["allocated_gpu_count"] != len(profile.devices),
            observed_resume != expected_resume,
            repositories[0]["commit"] != provenance["repo_commit"],
            repositories[2]["commit"] != provenance["upstream_commit"],
            checkpoints[0]["sha256"] != provenance["common_checkpoint_hash"],
            checkpoints[1]["sha256"] != provenance["autoencoder_checkpoint_hash"],
            projection["semantic_sha256"] != provenance["parent_projection_hash"],
            projection["file_sha256"] != provenance["parent_projection_file_hash"],
            projection["spec_sha256"] != provenance["parent_projection_spec_hash"],
            parent_checkpoint is not None
            and str(_absolute(Path(str(parent_checkpoint["path"])))) != observed_resume,
            parent_checkpoint is not None
            and parent_checkpoint["sha256"]
            != training_resume["parent_checkpoint_hash"],
            training["start_step"] != 0,
            training["steps"] != profile.max_steps,
        )
    ):
        raise CampaignError(
            "successful attempt profile/config/stage/ranking/resume/allocation "
            "or elapsed differs from Task 7 run"
        )
    if profile.stage == "joint_adaptation":
        assert isinstance(parent_checkpoint, Mapping)
        parent_path = _absolute(Path(str(parent_checkpoint["path"])))
        bulk_root = _absolute(_TERMINAL_BULK_ROOT)
        try:
            relative_parent = parent_path.relative_to(bulk_root)
        except ValueError as error:
            raise CampaignError("joint attempt parent is outside fixed bulk") from error
        if (
            len(relative_parent.parts) != 4
            or relative_parent.parts[0] != "runs"
            or relative_parent.parts[2] != "checkpoints"
        ):
            raise CampaignError("joint attempt parent path schema is not exact")
        parent_run_id = relative_parent.parts[1]
        parent_attempt = attempts_by_run.get(parent_run_id)
        if parent_attempt is None or parent_attempt.status != "success":
            raise CampaignError("joint run parent has no successful accounted attempt")
        immutable_parent_attempt = _debit_from_attempt(
            _fixed_evidence_root() / "attempts" / parent_run_id
        )
        if immutable_parent_attempt != parent_attempt:
            raise CampaignError("joint run parent attempt debit changed")
        parent_provenance = _file_identity_from_record(
            preflight["parent_provenance"], label="joint parent provenance"
        )
        parent_training = _file_identity_from_record(
            preflight["parent_training"], label="joint parent training"
        )
        parent_completion = _file_identity_from_record(
            preflight["checkpoint_completion"], label="joint parent completion"
        )
        expected_parent_root = _fixed_evidence_root() / "runs" / parent_run_id
        if (
            _absolute(parent_provenance.path)
            != expected_parent_root / "provenance.json"
            or _absolute(parent_training.path) != expected_parent_root / "training.json"
            or _absolute(parent_completion.path).parent
            != expected_parent_root / "checkpoints"
        ):
            raise CampaignError("joint parent artifact paths are unrelated")
        parent_run_debit = _debit_from_record(
            ImmutableRunRecord(
                parent_provenance.path,
                parent_provenance.sha256,
                parent_training.path,
                parent_training.sha256,
            )
        )
        parent_training_record = _read_canonical_terminal_json(
            parent_training.path,
            expected_sha256=parent_training.sha256,
            label="joint parent training",
        )
        if (
            parent_run_debit.run_id != parent_run_id
            or parent_attempt.elapsed_seconds < parent_run_debit.elapsed_seconds
            or parent_training_record["mode"] != "real"
            or parent_training_record["stage"] != "bridge_warmup"
            or parent_training_record["seed"] != start["seed"]
            or parent_training_record["ranking_mode"] != profile.ranking_mode
            or parent_training_record["rank_override"] != profile.rank_override
            or parent_training_record["start_step"] != 0
            or parent_training_record["steps"] != 10_000
            or parent_training_record["checkpoint"] != str(parent_path)
            or parent_training_record["checkpoint_hash"] != parent_checkpoint["sha256"]
            or parent_training_record["checkpoint_completion"]
            != str(parent_completion.path)
        ):
            raise CampaignError("joint run parent lineage is not exact real warmup")
        _read_canonical_terminal_json(
            parent_completion.path,
            expected_sha256=parent_completion.sha256,
            label="joint parent completion",
        )

def _assert_training_resume(
    provenance: Mapping[str, object], training: Mapping[str, object]
) -> None:
    resume = training["resume_state"]
    if not isinstance(resume, Mapping) or set(resume) != _RESUME_KEYS:
        raise CampaignError("Task 7 training resume v3 schema is not exact")
    if type(resume["schema_version"]) is not int or resume["schema_version"] != 3:
        raise CampaignError("Task 7 training resume schema_version is not exact v3")
    expected = {
        "stage": provenance["stage"],
        "seed": provenance["seed"],
        "repo_commit": provenance["repo_commit"],
        "upstream_commit": provenance["upstream_commit"],
        "common_checkpoint_hash": provenance["common_checkpoint_hash"],
        "split_hash": provenance["split_hash"],
        "config_hash": provenance["config_hash"],
        "route_equivalent_passes": provenance["route_equivalent_passes"],
        "dropout": provenance["dropout"],
        "ranking_mode": provenance["ranking_mode"],
        "rank_override": provenance["rank_override"],
        "parent_projection_hash": provenance["parent_projection_hash"],
        "parent_projection_file_hash": provenance["parent_projection_file_hash"],
        "parent_projection_spec_hash": provenance["parent_projection_spec_hash"],
    }
    if any(resume[key] != value for key, value in expected.items()):
        raise CampaignError("Task 7 training resume projection or identity differs")
    for key in ("trainable_name_hash", "optimizer_state_hash"):
        _require_hex(f"Task 7 resume {key}", resume[key], _HEX64)
    parent_hash = resume["parent_checkpoint_hash"]
    if provenance["stage"] == "bridge_warmup":
        if parent_hash is not None:
            raise CampaignError("Task 7 warm-up resume unexpectedly has a parent hash")
    else:
        _require_hex("Task 7 resume parent checkpoint hash", parent_hash, _HEX64)

@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    root: Path
    commit: str
    clean: bool

    def __post_init__(self) -> None:
        _require_hex("repository commit", self.commit, _HEX40)
        if type(self.clean) is not bool:
            raise CampaignError("repository clean state must be boolean")

    def as_record(self) -> dict[str, object]:
        return {"root": str(self.root), "commit": self.commit, "clean": self.clean}

@dataclass(frozen=True, slots=True)
class FileIdentity:
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _require_hex("file sha256", self.sha256, _HEX64)
        _require_positive_integer("file size_bytes", self.size_bytes)

    def as_record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

@dataclass(frozen=True, slots=True)
class _BoundPreparationSuccess:
    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int

    def __post_init__(self) -> None:
        FileIdentity(self.path, self.sha256, self.size_bytes)
        _require_nonnegative_integer("preparation success device", self.device)
        _require_positive_integer("preparation success inode", self.inode)

    def as_record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "device": self.device,
            "inode": self.inode,
        }

@dataclass(frozen=True, slots=True)
class ReservedLog:
    path: Path
    device: int
    inode: int

    def __post_init__(self) -> None:
        _require_nonnegative_integer("reserved log device", self.device)
        _require_positive_integer("reserved log inode", self.inode)

    def as_record(self) -> dict[str, object]:
        return {"path": str(self.path), "device": self.device, "inode": self.inode}

@dataclass(frozen=True, slots=True)
class ComputeProcess:
    pid: int
    command: str
    belongs_to_dive: bool

    def __post_init__(self) -> None:
        _require_positive_integer("compute pid", self.pid)
        if type(self.command) is not str or not self.command:
            raise CampaignError("compute command must be a nonempty string")
        if type(self.belongs_to_dive) is not bool:
            raise CampaignError("compute DIVE classification must be boolean")

    def as_record(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class BoundAttempt:

    attempt_id: str
    run_id: str
    start_path: Path
    start_sha256: str
    started_monotonic_ns: int
    allocated_gpu_count: int

@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    index: int
    name: str
    total_memory_bytes: int
    processes: tuple[ComputeProcess, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer("GPU index", self.index)
        _require_positive_integer("GPU total memory", self.total_memory_bytes)
        if type(self.name) is not str or not self.name:
            raise CampaignError("GPU name must be a nonempty string")
        if not isinstance(self.processes, tuple) or any(
            not isinstance(item, ComputeProcess) for item in self.processes
        ):
            raise CampaignError("GPU processes must be a tuple of ComputeProcess")

    def as_record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
            "processes": [item.as_record() for item in self.processes],
        }

@dataclass(frozen=True, slots=True)
class StorageSnapshot:
    path: Path
    writable: bool
    free_bytes: int

    def __post_init__(self) -> None:
        if type(self.writable) is not bool:
            raise CampaignError("storage writable state must be boolean")
        _require_positive_integer("storage free_bytes", self.free_bytes)

    def as_record(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "writable": self.writable,
            "free_bytes": self.free_bytes,
        }

@dataclass(frozen=True, slots=True)
class ProjectionIdentity:
    semantic_sha256: str
    file_sha256: str
    spec_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_hex(name, value, _HEX64)

    def as_record(self) -> dict[str, str]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    repo: RepositorySnapshot
    signed_value_upstream: RepositorySnapshot
    emergent_upstream: RepositorySnapshot
    common_checkpoint: FileIdentity
    autoencoder_checkpoint: FileIdentity
    parent_checkpoint: FileIdentity | None
    checkpoint_completion: FileIdentity | None
    gpus: tuple[GpuSnapshot, ...]
    evidence_storage: StorageSnapshot
    bulk_storage: StorageSnapshot
    projection: ProjectionIdentity
    reserved_log: ReservedLog | Path
    projection_files: tuple[FileIdentity, ...] = ()
    parent_training: FileIdentity | None = None
    parent_provenance: FileIdentity | None = None

@dataclass(frozen=True, slots=True)
class LiveCapture:
    profile: ActionProfile
    config: FileIdentity
    ledger: CampaignLedger
    system: SystemSnapshot

@dataclass(frozen=True, slots=True)
class RunPreflight:
    run_id: str
    seed: int
    stage: str
    action_profile: ActionProfile
    config: FileIdentity
    ranking_mode: str
    rank_override: float | None
    ledger_projection: CampaignProjection
    repositories: tuple[RepositorySnapshot, ...]
    checkpoints: tuple[FileIdentity, ...]
    checkpoint_completion: FileIdentity | None
    gpu_inventory: tuple[GpuSnapshot, ...]
    eligible_idle_devices: tuple[int, ...]
    storage: tuple[StorageSnapshot, StorageSnapshot]
    projection: ProjectionIdentity
    projection_files: tuple[FileIdentity, ...]
    reserved_log: ReservedLog
    parent_provenance: FileIdentity | None = None
    parent_training: FileIdentity | None = None
    preparation_success: FileIdentity | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "seed": self.seed,
            "stage": self.stage,
            "action_profile": self.action_profile.as_record(),
            "config": self.config.as_record(),
            "ranking_mode": self.ranking_mode,
            "rank_override": self.rank_override,
            "ledger_projection": self.ledger_projection.as_record(),
            "repositories": [item.as_record() for item in self.repositories],
            "checkpoints": [item.as_record() for item in self.checkpoints],
            "checkpoint_completion": (
                None
                if self.checkpoint_completion is None
                else self.checkpoint_completion.as_record()
            ),
            "gpu_inventory": [item.as_record() for item in self.gpu_inventory],
            "eligible_idle_devices": list(self.eligible_idle_devices),
            "storage": [item.as_record() for item in self.storage],
            "projection": self.projection.as_record(),
            "projection_files": [item.as_record() for item in self.projection_files],
            "reserved_log": self.reserved_log.as_record(),
            "parent_provenance": (
                None
                if self.parent_provenance is None
                else self.parent_provenance.as_record()
            ),
            "parent_training": (
                None
                if self.parent_training is None
                else self.parent_training.as_record()
            ),
            "preparation_success": (
                None
                if self.preparation_success is None
                else self.preparation_success.as_record()
            ),
            "launch_authorized": False,
        }

def _preparation_success_payload(
    *,
    run_id: str,
    seed: int,
    profile: ActionProfile,
    reserved_log: ReservedLog,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "success",
        "seed": seed,
        "preflight_path": str(_fixed_evidence_root() / "preflights" / f"{run_id}.json"),
        "action_profile": profile.as_record(),
        "reserved_log": reserved_log.as_record(),
    }

def _expected_preparation_success(
    *,
    run_id: str,
    seed: int,
    profile: ActionProfile,
    reserved_log: ReservedLog,
) -> FileIdentity:
    payload = _preparation_success_payload(
        run_id=run_id,
        seed=seed,
        profile=profile,
        reserved_log=reserved_log,
    )
    content = canonical_json_bytes(payload)
    return FileIdentity(
        _fixed_evidence_root() / "preparation_dispositions" / f"{run_id}.json",
        hashlib.sha256(content).hexdigest(),
        len(content),
    )

def create_run_preflight(
    *,
    run_id: str,
    seed: int,
    profile: ActionProfile,
    config: FileIdentity,
    ledger: CampaignLedger,
    snapshot: SystemSnapshot,
) -> RunPreflight:

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise CampaignError("run id is invalid")
    if type(seed) is not int or seed not in (42, 314159):
        raise CampaignError("seed is not one of the two frozen directional seeds")
    if (
        not isinstance(profile, ActionProfile)
        or not isinstance(config, FileIdentity)
        or not isinstance(snapshot, SystemSnapshot)
    ):
        raise CampaignError("preflight inputs have the wrong type")
    stage = profile.stage
    ranking_mode = profile.ranking_mode
    rank_override = profile.rank_override
    if (
        _absolute(config.path) != profile.config_path
        or config.sha256 != profile.config_sha256
    ):
        raise CampaignError("config identity differs from frozen action profile")
    _require_ranking(ranking_mode, rank_override)
    repo_root = _fixed_directory(_EXECUTING_REPO_ROOT, "executing repository")
    if _absolute(snapshot.repo.root) != repo_root or not snapshot.repo.clean:
        raise CampaignError("executing DIVE repository snapshot is dirty or mismatched")
    signed_root = _fixed_directory(_SIGNED_UPSTREAM_ROOT, "signed-value upstream")
    emergent_root = _fixed_directory(_EMERGENT_UPSTREAM_ROOT, "emergent upstream")
    _assert_repository_snapshot(
        snapshot.signed_value_upstream,
        root=signed_root,
        commit=SIGNED_VALUE_UPSTREAM_COMMIT,
        label="signed-value upstream",
    )
    _assert_repository_snapshot(
        snapshot.emergent_upstream,
        root=emergent_root,
        commit=EMERGENT_UPSTREAM_COMMIT,
        label="emergent upstream",
    )
    if (
        _absolute(config.path) == repo_root
        or repo_root not in _absolute(config.path).parents
    ):
        raise CampaignError("config identity must stay below the executing repository")
    _assert_frozen_checkpoint(
        snapshot.common_checkpoint,
        path=emergent_root / "ckpts/complexa.ckpt",
        sha256=COMMON_CHECKPOINT_SHA256,
        size_bytes=COMMON_CHECKPOINT_SIZE_BYTES,
        label="common checkpoint",
    )
    _assert_frozen_checkpoint(
        snapshot.autoencoder_checkpoint,
        path=emergent_root / "ckpts/complexa_ae.ckpt",
        sha256=AUTOENCODER_CHECKPOINT_SHA256,
        size_bytes=AUTOENCODER_CHECKPOINT_SIZE_BYTES,
        label="autoencoder checkpoint",
    )
    _assert_parent_contract(
        snapshot,
        ledger=ledger,
        stage=stage,
        seed=seed,
        ranking_mode=ranking_mode,
        rank_override=rank_override,
    )
    _assert_projection(snapshot.projection)
    eligible = _validate_gpu_inventory(snapshot.gpus, profile.devices)
    evidence_root = _fixed_evidence_root()
    bulk_root = _fixed_directory(_TERMINAL_BULK_ROOT, "bulk")
    _assert_storage(snapshot.evidence_storage, evidence_root, "evidence")
    _assert_storage(snapshot.bulk_storage, bulk_root, "bulk")
    evidence_run_root = evidence_root / "runs" / run_id
    bulk_run_root = bulk_root / "runs" / run_id
    preflight_path = evidence_root / "preflights" / f"{run_id}.json"
    if any(
        os.path.lexists(path)
        for path in (evidence_run_root, bulk_run_root, preflight_path)
    ):
        raise CampaignError(f"create-new run id already exists: {run_id}")
    if not isinstance(snapshot.reserved_log, ReservedLog):
        raise CampaignError("preflight requires a captured reserved log identity")
    reserved_log = snapshot.reserved_log
    reserved_path = validate_terminal_evidence_path(reserved_log.path)
    observed_log = os.lstat(reserved_path)
    if (observed_log.st_dev, observed_log.st_ino) != (
        reserved_log.device,
        reserved_log.inode,
    ):
        raise CampaignError("pre-reserved terminal log identity changed")
    if _read_terminal_bytes(reserved_path):
        raise CampaignError("pre-reserved terminal log is no longer empty")
    projection = authorize_projection(ledger, profile.worst_case_gpu_seconds)
    checkpoints = [snapshot.common_checkpoint, snapshot.autoencoder_checkpoint]
    if snapshot.parent_checkpoint is not None:
        checkpoints.append(snapshot.parent_checkpoint)
    return RunPreflight(
        run_id=run_id,
        seed=seed,
        stage=stage,
        action_profile=profile,
        config=config,
        ranking_mode=ranking_mode,
        rank_override=rank_override,
        ledger_projection=projection,
        repositories=(
            snapshot.repo,
            snapshot.signed_value_upstream,
            snapshot.emergent_upstream,
        ),
        checkpoints=tuple(checkpoints),
        checkpoint_completion=snapshot.checkpoint_completion,
        gpu_inventory=snapshot.gpus,
        eligible_idle_devices=eligible,
        storage=(snapshot.evidence_storage, snapshot.bulk_storage),
        projection=snapshot.projection,
        projection_files=snapshot.projection_files,
        reserved_log=reserved_log,
        parent_provenance=snapshot.parent_provenance,
        parent_training=snapshot.parent_training,
        preparation_success=_expected_preparation_success(
            run_id=run_id,
            seed=seed,
            profile=profile,
            reserved_log=reserved_log,
        ),
    )

def _assert_parent_contract(
    snapshot: SystemSnapshot,
    *,
    ledger: CampaignLedger,
    stage: str,
    seed: int,
    ranking_mode: str,
    rank_override: float | None,
) -> None:
    parent = snapshot.parent_checkpoint
    completion = snapshot.checkpoint_completion
    training = snapshot.parent_training
    provenance = snapshot.parent_provenance
    if stage == "bridge_warmup":
        if any(item is not None for item in (parent, completion, training, provenance)):
            raise CampaignError("bridge warm-up refuses a parent or completion")
        return
    if parent is None or completion is None or training is None or provenance is None:
        raise CampaignError(
            "joint adaptation requires parent provenance, completion, and training"
        )
    bulk_root = _fixed_directory(_TERMINAL_BULK_ROOT, "bulk")
    parent_path = _absolute(parent.path)
    if parent_path == bulk_root or bulk_root not in parent_path.parents:
        raise CampaignError("parent checkpoint is outside fixed bulk storage")
    relative_parent = parent_path.relative_to(bulk_root)
    if (
        len(relative_parent.parts) != 4
        or relative_parent.parts[0] != "runs"
        or _RUN_ID.fullmatch(relative_parent.parts[1]) is None
        or relative_parent.parts[2] != "checkpoints"
        or not relative_parent.parts[3].endswith(".ckpt")
    ):
        raise CampaignError("parent checkpoint path schema is not exact")
    parent_run_id = relative_parent.parts[1]
    evidence_root = _fixed_evidence_root()
    expected_run_root = evidence_root / "runs" / parent_run_id
    if (
        _absolute(provenance.path) != expected_run_root / "provenance.json"
        or _absolute(training.path) != expected_run_root / "training.json"
        or _absolute(completion.path).parent != expected_run_root / "checkpoints"
    ):
        raise CampaignError("parent provenance/training/completion paths are unrelated")
    parent_debit = next(
        (item for item in ledger.run_debits if item.run_id == parent_run_id), None
    )
    immutable_parent_debit = _debit_from_attempt(
        evidence_root / "attempts" / parent_run_id
    )
    if (
        parent_debit is None
        or parent_debit.status != "success"
        or immutable_parent_debit != parent_debit
    ):
        raise CampaignError("joint parent has no successful accounted Task 9 attempt")
    run_debit = _debit_from_record(
        ImmutableRunRecord(
            provenance.path,
            provenance.sha256,
            training.path,
            training.sha256,
        )
    )
    if (
        run_debit.run_id != parent_run_id
        or parent_debit.elapsed_seconds < run_debit.elapsed_seconds
    ):
        raise CampaignError(
            "joint parent run is not authenticated by its attempt debit"
        )
    record = _read_canonical_terminal_json(
        completion.path,
        expected_sha256=completion.sha256,
        expected_size_bytes=completion.size_bytes,
        label="parent checkpoint completion",
    )
    if (
        set(record) != _COMPLETION_KEYS
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
    ):
        raise CampaignError("parent checkpoint completion schema is not exact")
    if record["checkpoint"] != parent.as_record():
        raise CampaignError("parent checkpoint hash/size/path differs from completion")
    if record["step_completed"] != 10_000:
        raise CampaignError("joint adaptation requires exact warm-up step 10000")
    resume = record["resume_state"]
    if not isinstance(resume, Mapping) or set(resume) != _RESUME_KEYS:
        raise CampaignError("parent completion resume v3 schema is not exact")
    _assert_parent_resume(
        resume,
        snapshot=snapshot,
        seed=seed,
        ranking_mode=ranking_mode,
        rank_override=rank_override,
    )
    ownership = record["ownership"]
    if not isinstance(ownership, Mapping) or set(ownership) != _OWNERSHIP_KEYS:
        raise CampaignError("parent completion ownership schema is not exact")
    if ownership["stage"] != "bridge_warmup":
        raise CampaignError("parent completion ownership is not bridge warm-up")
    _assert_completion_ownership(ownership)
    validation = record["validation_record"]
    if not isinstance(validation, Mapping) or set(validation) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise CampaignError("parent completion validation identity is not exact")
    validation_identity = FileIdentity(
        Path(str(validation["path"])),
        validation["sha256"],
        validation["size_bytes"],
    )
    _read_canonical_terminal_json(
        validation_identity.path,
        expected_sha256=validation_identity.sha256,
        expected_size_bytes=validation_identity.size_bytes,
        label="parent validation record",
    )
    training_record = _read_canonical_terminal_json(
        training.path,
        expected_sha256=training.sha256,
        expected_size_bytes=training.size_bytes,
        label="parent training record",
    )
    if set(training_record) != _TRAINING_KEYS:
        raise CampaignError("parent training record schema is not exact")
    expected_training = {
        "schema_version": 1,
        "mode": "real",
        "stage": "bridge_warmup",
        "seed": seed,
        "ranking_mode": ranking_mode,
        "rank_override": rank_override,
        "steps": 10_000,
        "start_step": 0,
        "checkpoint": str(parent.path),
        "checkpoint_hash": parent.sha256,
        "checkpoint_completion": str(completion.path),
        "uses_gpu": True,
        "loads_blind_data": False,
        "uses_synthetic_data": False,
        "allocated_gpu_count": 4,
    }
    if any(training_record[key] != value for key, value in expected_training.items()):
        raise CampaignError(
            "joint parent is not an exact final successful real warm-up"
        )
    if training_record["resume_state"] != resume:
        raise CampaignError("parent training and completion resume lineage differ")

def _assert_completion_ownership(ownership: Mapping[str, object]) -> None:
    names = ownership["trainable_names"]
    if (
        not isinstance(names, list)
        or not names
        or any(type(name) is not str or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise CampaignError("parent completion ownership names are invalid")
    if _require_positive_integer(
        "parent ownership name count", ownership["trainable_name_count"]
    ) != len(names):
        raise CampaignError("parent completion ownership name count differs")
    _require_positive_integer(
        "parent ownership parameter count", ownership["trainable_parameters"]
    )
    groups = ownership["groups"]
    if not isinstance(groups, Mapping) or not groups:
        raise CampaignError("parent completion ownership groups are invalid")
    grouped: list[object] = []
    for group, group_names in groups.items():
        if (
            type(group) is not str
            or not isinstance(group_names, list)
            or any(type(name) is not str for name in group_names)
        ):
            raise CampaignError("parent completion ownership groups are invalid")
        grouped.extend(group_names)
    if sorted(grouped) != sorted(names) or len(grouped) != len(set(grouped)):
        raise CampaignError("parent completion ownership groups differ from names")

def _assert_parent_resume(
    resume: Mapping[str, object],
    *,
    snapshot: SystemSnapshot,
    seed: int,
    ranking_mode: str,
    rank_override: float | None,
) -> None:
    if resume["schema_version"] != 3 or resume["stage"] != "bridge_warmup":
        raise CampaignError("parent completion resume identity is not warm-up v3")
    warmup_config = (
        "configs/emergent/directional_bridge_warmup.yaml"
        if ranking_mode == "matched"
        else "configs/emergent/directional_bridge_no_ranking_warmup.yaml"
    )
    expected = {
        "seed": seed,
        "repo_commit": snapshot.repo.commit,
        "upstream_commit": EMERGENT_UPSTREAM_COMMIT,
        "common_checkpoint_hash": COMMON_CHECKPOINT_SHA256,
        "parent_checkpoint_hash": None,
        "route_equivalent_passes": 8.0,
        "dropout": 0.0,
        "ranking_mode": ranking_mode,
        "rank_override": rank_override,
        "config_hash": _ACTION_PROFILE_SPECS[warmup_config][1],
        "parent_projection_hash": PARENT_PROJECTION_SEMANTIC_SHA256,
        "parent_projection_file_hash": PARENT_PROJECTION_FILE_SHA256,
        "parent_projection_spec_hash": PARENT_PROJECTION_SPEC_SHA256,
    }
    if any(resume[key] != value for key, value in expected.items()):
        raise CampaignError(
            "parent completion resume projection or run identity changed"
        )
    for key in (
        "split_hash",
        "trainable_name_hash",
        "optimizer_state_hash",
    ):
        _require_hex(f"parent resume {key}", resume[key], _HEX64)

def _assert_repository_snapshot(
    snapshot: RepositorySnapshot, *, root: Path, commit: str, label: str
) -> None:
    if (
        _absolute(snapshot.root) != root
        or snapshot.commit != commit
        or not snapshot.clean
    ):
        raise CampaignError(f"{label} snapshot is dirty, moved, or at the wrong pin")

def _assert_frozen_checkpoint(
    identity: FileIdentity,
    *,
    path: Path,
    sha256: str,
    size_bytes: int,
    label: str,
) -> None:
    if (
        _absolute(identity.path) != _absolute(path)
        or identity.sha256 != sha256
        or identity.size_bytes != size_bytes
    ):
        raise CampaignError(f"{label} hash, size, or path changed")

def _assert_projection(identity: ProjectionIdentity) -> None:
    if identity != ProjectionIdentity(
        PARENT_PROJECTION_SEMANTIC_SHA256,
        PARENT_PROJECTION_FILE_SHA256,
        PARENT_PROJECTION_SPEC_SHA256,
    ):
        raise CampaignError("Task 7 parent projection identities changed")

def _validate_gpu_inventory(
    gpus: Sequence[GpuSnapshot], devices: Sequence[int]
) -> tuple[int, ...]:
    if not isinstance(gpus, tuple) or len(gpus) != 8:
        raise CampaignError("GPU inventory must contain exactly eight devices")
    if tuple(item.index for item in gpus) != tuple(range(8)):
        raise CampaignError("GPU inventory indices must be exactly 0 through 7")
    seen_processes: dict[int, tuple[str, bool]] = {}
    for gpu in gpus:
        if gpu.name != "NVIDIA L40S":
            raise CampaignError(f"GPU {gpu.index} is not exact NVIDIA L40S")
        if gpu.total_memory_bytes < L40S_TOTAL_MEMORY_BYTES:
            raise CampaignError(f"GPU {gpu.index} is below the L40S VRAM floor")
        for process in gpu.processes:
            observed = (process.command, process.belongs_to_dive)
            prior = seen_processes.setdefault(process.pid, observed)
            if prior != observed:
                raise CampaignError(
                    "GPU process inventory gives one PID two identities"
                )
            if process.belongs_to_dive:
                raise CampaignError(f"stale DIVE compute PID {process.pid} remains")
    island = _assert_device_island(tuple(devices))
    by_index = {item.index: item for item in gpus}
    if any(by_index[index].processes for index in island):
        raise CampaignError("the four fixed campaign devices are not all eligible idle")
    return island

def _storage_identity(entry: Mapping[str, object]) -> dict[str, object]:

    return {"path": entry["path"], "writable": entry["writable"]}

def _assert_storage(snapshot: StorageSnapshot, root: Path, label: str) -> None:
    if _absolute(snapshot.path) != root or not snapshot.writable:
        raise CampaignError(f"{label} storage snapshot is moved or not writable")
    if snapshot.free_bytes < _MIN_FREE_BYTES:
        raise CampaignError(
            f"{label} storage has insufficient free space: "
            f"{snapshot.free_bytes} < {_MIN_FREE_BYTES}"
        )

def validate_terminal_evidence_path(path: Path) -> Path:

    destination = _absolute(path)
    if not destination.name or destination.name in {".", ".."}:
        raise CampaignError("path has no safe final name")
    return destination

def intended_preflight_destination(run_id: str) -> Path:

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise CampaignError("preflight destination run id is invalid")
    destination = _fixed_evidence_root() / "preflights" / f"{run_id}.json"
    return validate_terminal_evidence_path(destination)

def reserve_terminal_log(path: Path) -> ReservedLog:

    destination = validate_terminal_evidence_path(path)
    root = _fixed_evidence_root()
    relative = destination.relative_to(root)
    parent_fd = _open_terminal_parent(relative.parent, create=True)
    assert parent_fd is not None
    file_fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(relative.name, flags, 0o664, dir_fd=parent_fd)
        except FileExistsError as error:
            raise CampaignError(
                f"create-new terminal log already exists: {destination}"
            ) from error
        os.fchmod(file_fd, 0o664)
        observed = os.fstat(file_fd)
        if not stat_module.S_ISREG(observed.st_mode) or observed.st_size != 0:
            raise CampaignError("reserved terminal log is not an empty regular file")
        os.fsync(file_fd)
        os.fsync(parent_fd)
        return ReservedLog(destination, observed.st_dev, observed.st_ino)
    except OSError as error:
        raise CampaignError(f"cannot safely reserve terminal log: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)

def write_preflight_create_new(path: Path, preflight: RunPreflight) -> Path:

    if not isinstance(preflight, RunPreflight):
        raise CampaignError("preflight must be a RunPreflight")
    content = canonical_json_bytes(preflight.as_record())
    destination = validate_terminal_evidence_path(path)
    root = _fixed_evidence_root()
    relative = destination.relative_to(root)
    parent_fd = _open_terminal_parent(relative.parent, create=True)
    assert parent_fd is not None
    file_fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(relative.name, flags, 0o664, dir_fd=parent_fd)
        except FileExistsError as error:
            raise CampaignError(
                f"create-new preflight already exists: {destination}"
            ) from error
        os.fchmod(file_fd, 0o664)
        os.fsync(parent_fd)
        view = memoryview(content)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError("preflight write made no forward progress")
            view = view[written:]
        os.fsync(file_fd)
        os.fsync(parent_fd)
        return destination
    except OSError as error:
        raise CampaignError(f"cannot safely publish preflight: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)

def revalidate_bound_preflight(
    *,
    preflight_path: Path,
    preflight_sha256: str,
    preparation_disposition_path: Path,
    preparation_disposition_sha256: str,
    run_id: str,
    seed: int,
    config_path: Path,
    resume: Path | None,
    reserved_log: ReservedLog,
    action_profile_id: str,
    campaign_stage: str,
    campaign_mode: str,
    campaign_devices: str,
    worst_case_wall_seconds: int,
) -> Mapping[str, object]:

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise CampaignError("bound preflight run id is invalid")
    evidence_root = _fixed_evidence_root()
    expected_path = evidence_root / "preflights" / f"{run_id}.json"
    if _absolute(preflight_path) != expected_path:
        raise CampaignError("bound preflight path is not the fixed run identity")
    record = _read_canonical_terminal_json(
        expected_path,
        expected_sha256=preflight_sha256,
        label="bound campaign preflight",
    )
    expected_keys = set(RunPreflight.__dataclass_fields__) | {
        "schema_version",
        "launch_authorized",
    }
    if set(record) != expected_keys:
        raise CampaignError("bound campaign preflight schema is not exact")
    if (
        record["schema_version"] != 1
        or record["launch_authorized"] is not False
        or record["run_id"] != run_id
        or record["seed"] != seed
    ):
        raise CampaignError("bound campaign preflight identity changed")
    if record["reserved_log"] != reserved_log.as_record():
        raise CampaignError("bound campaign preflight log identity changed")
    success_identity = _file_identity_from_record(
        record["preparation_success"], label="bound preparation success"
    )
    if (
        _absolute(preparation_disposition_path) != success_identity.path
        or preparation_disposition_sha256 != success_identity.sha256
    ):
        raise CampaignError("command preparation disposition binding changed")
    _validate_preparation_success(record, expected_run_id=run_id)
    bound_action = record["action_profile"]
    if not isinstance(bound_action, Mapping) or any(
        observed != expected
        for observed, expected in (
            (action_profile_id, bound_action.get("profile_id")),
            (campaign_stage, bound_action.get("stage")),
            (campaign_mode, bound_action.get("mode")),
            (
                campaign_devices,
                ",".join(str(item) for item in bound_action.get("devices", ())),
            ),
            (worst_case_wall_seconds, bound_action.get("worst_case_wall_seconds")),
        )
    ):
        raise CampaignError("command action binding differs from campaign preflight")

    captured = capture_live_state(
        config_path=config_path,
        resume=resume,
        log_path=reserved_log.path,
    )
    expected_projection = authorize_projection(
        captured.ledger, captured.profile.worst_case_gpu_seconds
    )
    live_fields: dict[str, object] = {
        "stage": captured.profile.stage,
        "action_profile": captured.profile.as_record(),
        "config": captured.config.as_record(),
        "ranking_mode": captured.profile.ranking_mode,
        "rank_override": captured.profile.rank_override,
        "ledger_projection": expected_projection.as_record(),
        "repositories": [
            item.as_record()
            for item in (
                captured.system.repo,
                captured.system.signed_value_upstream,
                captured.system.emergent_upstream,
            )
        ],
        "checkpoints": [
            item.as_record()
            for item in (
                captured.system.common_checkpoint,
                captured.system.autoencoder_checkpoint,
                *(
                    ()
                    if captured.system.parent_checkpoint is None
                    else (captured.system.parent_checkpoint,)
                ),
            )
        ],
        "checkpoint_completion": (
            None
            if captured.system.checkpoint_completion is None
            else captured.system.checkpoint_completion.as_record()
        ),
        "parent_provenance": (
            None
            if captured.system.parent_provenance is None
            else captured.system.parent_provenance.as_record()
        ),
        "parent_training": (
            None
            if captured.system.parent_training is None
            else captured.system.parent_training.as_record()
        ),
        "gpu_inventory": [item.as_record() for item in captured.system.gpus],
        "eligible_idle_devices": list(
            _validate_gpu_inventory(captured.system.gpus, captured.profile.devices)
        ),

        "storage": [
            _storage_identity(item.as_record())
            for item in (
                captured.system.evidence_storage,
                captured.system.bulk_storage,
            )
        ],
        "projection": captured.system.projection.as_record(),
        "projection_files": [
            item.as_record() for item in captured.system.projection_files
        ],
    }
    recorded_fields = dict(record)
    recorded_fields["storage"] = [
        _storage_identity(item) for item in record["storage"]
    ]
    changed = sorted(
        key for key, value in live_fields.items() if recorded_fields[key] != value
    )
    if changed:
        raise CampaignError(
            f"bound campaign preflight is stale or live state drifted: {changed}"
        )

    _assert_storage(
        captured.system.evidence_storage, _fixed_evidence_root(), "evidence"
    )
    _assert_storage(
        captured.system.bulk_storage,
        _fixed_directory(_TERMINAL_BULK_ROOT, "bulk"),
        "bulk",
    )
    observed_log = os.lstat(reserved_log.path)
    if (
        not stat_module.S_ISREG(observed_log.st_mode)
        or stat_module.S_ISLNK(observed_log.st_mode)
        or observed_log.st_size != 0
        or (observed_log.st_dev, observed_log.st_ino)
        != (reserved_log.device, reserved_log.inode)
    ):
        raise CampaignError("bound reserved log changed before launch")
    return record

def attach_reserved_log(reserved_log: ReservedLog) -> None:

    if not isinstance(reserved_log, ReservedLog):
        raise CampaignError("reserved log identity has the wrong type")
    destination = validate_terminal_evidence_path(reserved_log.path)
    root = _fixed_evidence_root()
    relative = destination.relative_to(root)
    parent_fd = _open_terminal_parent(relative.parent, create=False)
    assert parent_fd is not None
    file_fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(relative.name, flags, dir_fd=parent_fd)
        observed = os.fstat(file_fd)
        if not stat_module.S_ISREG(observed.st_mode) or (
            observed.st_dev,
            observed.st_ino,
        ) != (reserved_log.device, reserved_log.inode):
            raise CampaignError("reserved log inode changed before attachment")
        os.dup2(file_fd, 1)
        os.dup2(file_fd, 2)
    except OSError as error:
        raise CampaignError(f"cannot safely attach reserved log: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)

def write_attempt_start(
    *,
    run_id: str,
    preflight_path: Path,
    preflight_sha256: str,
    preflight: Mapping[str, object],
    resume: Path | None,
    reserved_log: ReservedLog,
) -> BoundAttempt:

    if preflight.get("run_id") != run_id:
        raise CampaignError("attempt and preflight run identities differ")
    profile = _action_profile_from_record(preflight.get("action_profile"))
    preparation_success = _validate_preparation_success(
        preflight, expected_run_id=run_id
    )
    hook = _attempt_admission_validated_test_hook
    if hook is not None:
        hook()
    started_ns = time.monotonic_ns()
    payload = {
        "schema_version": 1,
        "attempt_id": run_id,
        "run_id": run_id,
        "preflight": {"path": str(preflight_path), "sha256": preflight_sha256},
        "preparation_success": preparation_success.as_record(),
        "action_profile": profile.as_record(),
        "seed": preflight.get("seed"),
        "resume": None if resume is None else str(_absolute(resume)),
        "reserved_log": reserved_log.as_record(),
        "started_monotonic_ns": started_ns,
    }
    destination = _fixed_evidence_root() / "attempts" / run_id / "start.json"
    _write_canonical_create_new(destination, payload, label="attempt start")
    attempt = BoundAttempt(
        run_id,
        run_id,
        destination,
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        started_ns,
        len(profile.devices),
    )
    try:
        revalidated_success = _validate_preparation_success(
            preflight, expected_run_id=run_id
        )
        if revalidated_success != preparation_success:
            raise CampaignError(
                "preparation success identity changed during attempt admission"
            )
    except BaseException as error:
        try:
            write_attempt_terminal(attempt, status="failure", error=error)
        except BaseException as terminal_error:
            raise CampaignError(
                "attempt admission failed and its terminal refusal could not be "
                "published"
            ) from terminal_error
        if isinstance(error, CampaignError):
            raise
        raise CampaignError("attempt admission success revalidation failed") from error
    return attempt

def _validate_preparation_success(
    preflight: Mapping[str, object], *, expected_run_id: str
) -> _BoundPreparationSuccess:
    profile = _action_profile_from_record(preflight.get("action_profile"))
    reserved_record = preflight.get("reserved_log")
    if not isinstance(reserved_record, Mapping) or set(reserved_record) != {
        "path",
        "device",
        "inode",
    }:
        raise CampaignError("preparation success log binding is invalid")
    reserved_log = ReservedLog(
        Path(str(reserved_record["path"])),
        reserved_record["device"],
        reserved_record["inode"],
    )
    expected = _expected_preparation_success(
        run_id=expected_run_id,
        seed=preflight.get("seed"),
        profile=profile,
        reserved_log=reserved_log,
    )
    observed = _file_identity_from_record(
        preflight.get("preparation_success"), label="preparation success"
    )
    if observed != expected:
        raise CampaignError("preparation success identity is not exact")
    raw, descriptor = _read_terminal_snapshot(observed.path)
    if len(raw) != observed.size_bytes:
        raise CampaignError("preparation success disposition size changed")
    if hashlib.sha256(raw).hexdigest() != observed.sha256:
        raise CampaignError("preparation success disposition hash changed")
    record = _decode_canonical_mapping(raw, label="preparation success disposition")
    expected_record = _preparation_success_payload(
        run_id=expected_run_id,
        seed=preflight.get("seed"),
        profile=profile,
        reserved_log=reserved_log,
    )
    if record != expected_record:
        raise CampaignError("preparation disposition is not exact success")
    return _BoundPreparationSuccess(
        observed.path,
        observed.sha256,
        observed.size_bytes,
        descriptor.st_dev,
        descriptor.st_ino,
    )

def write_preparation_success(preflight: RunPreflight) -> Path:

    if not isinstance(preflight, RunPreflight):
        raise CampaignError("preparation success requires a RunPreflight")
    expected = _expected_preparation_success(
        run_id=preflight.run_id,
        seed=preflight.seed,
        profile=preflight.action_profile,
        reserved_log=preflight.reserved_log,
    )
    if preflight.preparation_success != expected:
        raise CampaignError("preparation success differs from preflight binding")
    return _write_canonical_create_new(
        expected.path,
        _preparation_success_payload(
            run_id=preflight.run_id,
            seed=preflight.seed,
            profile=preflight.action_profile,
            reserved_log=preflight.reserved_log,
        ),
        label="preparation disposition",
    )

def write_preparation_failure(
    *,
    run_id: str,
    reserved_log: ReservedLog,
    error: BaseException,
    preflight_path: Path,
    preflight_sha256: str | None = None,
) -> Path:

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise CampaignError("preparation failure run id is invalid")
    if not isinstance(reserved_log, ReservedLog):
        raise CampaignError("preparation failure log identity has the wrong type")
    expected_path = _fixed_evidence_root() / "preflights" / f"{run_id}.json"
    if _absolute(preflight_path) != expected_path:
        raise CampaignError("preparation failure preflight path is not fixed")
    expected_sha256 = None
    if preflight_sha256 is not None:
        expected_sha256 = _require_hex(
            "preparation failure preflight sha256", preflight_sha256, _HEX64
        )
    publication_state, observed = _observe_preflight_publication(
        expected_path, expected_sha256=expected_sha256
    )
    preflight_identity = {
        "path": str(expected_path),
        "expected_sha256": expected_sha256,
        "publication_state": publication_state,
        "observed": observed,
    }
    destination = _fixed_evidence_root() / "preparation_dispositions" / f"{run_id}.json"
    return _write_canonical_create_new(
        destination,
        {
            "schema_version": 1,
            "run_id": run_id,
            "reserved_log": reserved_log.as_record(),
            "preflight": preflight_identity,
            "status": "failure",
            "phase": "preflight-publication",
            "error": {"type": type(error).__name__, "message": str(error)},
        },
        label="preparation disposition",
    )

def write_attempt_terminal(
    attempt: BoundAttempt, *, status: str, error: BaseException | None = None
) -> Path:

    if not isinstance(attempt, BoundAttempt) or status not in {"success", "failure"}:
        raise CampaignError("attempt terminal identity or status is invalid")
    if (status == "success") != (error is None):
        raise CampaignError("attempt terminal status and error differ")
    completed_ns = time.monotonic_ns()
    payload = {
        "schema_version": 1,
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "start_sha256": attempt.start_sha256,
        "status": status,
        "elapsed_seconds": (completed_ns - attempt.started_monotonic_ns)
        / 1_000_000_000,
        "allocated_gpu_count": attempt.allocated_gpu_count,
        "completed_monotonic_ns": completed_ns,
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    destination = attempt.start_path.with_name("terminal.json")
    _write_canonical_create_new(destination, payload, label="attempt terminal")
    return destination

def load_attempt_start(
    run_id: str, *, timeout_seconds: float = 30.0, poll_seconds: float = 0.02
) -> BoundAttempt:

    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise CampaignError("attempt run id is invalid")
    path = _fixed_evidence_root() / "attempts" / run_id / "start.json"
    limit = time.monotonic() + timeout_seconds
    torn: CampaignError | None = None
    while True:
        if os.path.lexists(path):
            try:
                return _bound_attempt_from_start(run_id, path)
            except TornRecordRead as error:
                torn = error
        if time.monotonic() >= limit:
            raise CampaignError("timed out waiting for attempt start") from torn
        time.sleep(poll_seconds)

def _read_attempt_start_record(path: Path) -> tuple[Mapping[str, object], str]:

    raw = _read_terminal_bytes(path)
    if not raw:
        raise TornRecordRead("attempt start is still zero bytes")
    identity = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:

        raise TornRecordRead("attempt start is not yet complete JSON") from error
    if not isinstance(payload, Mapping):
        raise CampaignError("attempt start must be a JSON mapping")
    try:
        canonical = canonical_json_bytes(payload)
    except PreflightError as error:
        raise CampaignError("attempt start is not canonical JSON") from error
    if raw != canonical:
        if canonical.startswith(raw):

            raise TornRecordRead("attempt start is a canonical prefix")
        raise CampaignError("attempt start is not canonical JSON")
    return payload, identity

def _bound_attempt_from_start(run_id: str, path: Path) -> BoundAttempt:
    record, identity = _read_attempt_start_record(path)
    if set(record) != _ATTEMPT_START_KEYS or record.get("run_id") != run_id:
        raise CampaignError("attempt start schema or run identity changed")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("attempt_id") != run_id
    ):
        raise CampaignError("attempt start identity changed")
    preflight_identity = record.get("preflight")
    if not isinstance(preflight_identity, Mapping) or set(preflight_identity) != {
        "path",
        "sha256",
    }:
        raise CampaignError("attempt preflight identity is not exact")
    preflight_path = Path(str(preflight_identity["path"]))
    expected_preflight_path = _fixed_evidence_root() / "preflights" / f"{run_id}.json"
    if _absolute(preflight_path) != expected_preflight_path:
        raise CampaignError("attempt preflight path is not fixed")
    preflight = _read_canonical_terminal_json(
        preflight_path,
        expected_sha256=preflight_identity["sha256"],
        label="attempt preflight",
    )
    preflight_profile = _validate_run_preflight_record(
        preflight, expected_run_id=run_id
    )
    preparation_success = _validate_preparation_success(
        preflight, expected_run_id=run_id
    )
    if record.get("preparation_success") != preparation_success.as_record():
        raise CampaignError("attempt preparation success identity changed")
    profile = _action_profile_from_record(record.get("action_profile"))
    reserved_log = record.get("reserved_log")
    resume = record.get("resume")
    if (
        profile != preflight_profile
        or record.get("seed") != preflight.get("seed")
        or reserved_log != preflight.get("reserved_log")
        or (profile.stage == "bridge_warmup") != (resume is None)
        or (resume is not None and (type(resume) is not str or not resume))
    ):
        raise CampaignError("attempt start differs from its bound preflight")
    started_ns = _require_positive_integer(
        "attempt monotonic origin", record.get("started_monotonic_ns")
    )
    return BoundAttempt(
        run_id,
        run_id,
        path,
        identity,
        started_ns,
        len(profile.devices),
    )

def _write_canonical_create_new(
    path: Path, payload: Mapping[str, object], *, label: str
) -> Path:

    content = canonical_json_bytes(payload)
    destination = validate_terminal_evidence_path(path)
    root = _fixed_evidence_root()
    relative = destination.relative_to(root)
    parent_fd = _open_terminal_parent(relative.parent, create=True)
    assert parent_fd is not None
    staging = f".{relative.name}.staging.{uuid.uuid4().hex}"
    file_fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(staging, flags, 0o664, dir_fd=parent_fd)
        os.fchmod(file_fd, 0o664)
        view = memoryview(content)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:
                raise OSError(f"{label} write made no forward progress")
            view = view[written:]
        os.fsync(file_fd)
        os.link(
            staging,
            relative.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
        return destination
    except FileExistsError as error:
        raise CampaignError(
            f"create-new {label} already exists: {destination}"
        ) from error
    except OSError as error:
        raise CampaignError(f"cannot safely publish {label}: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)

        try:
            os.unlink(staging, dir_fd=parent_fd)
        except OSError:
            pass
        os.close(parent_fd)

def _read_canonical_terminal_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    expected_size_bytes: int | None = None,
) -> Mapping[str, object]:
    _require_hex(f"{label} hash", expected_sha256, _HEX64)
    raw = _read_terminal_bytes(path)
    if expected_size_bytes is not None:
        expected_size = _require_positive_integer(
            f"{label} size_bytes", expected_size_bytes
        )
        if len(raw) != expected_size:
            raise CampaignError(f"{label} size changed: {len(raw)} != {expected_size}")
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_sha256:
        raise CampaignError(f"{label} hash changed: {observed} != {expected_sha256}")
    return _decode_canonical_mapping(raw, label=label)

def _decode_canonical_mapping(raw: bytes, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise CampaignError(f"{label} must be a JSON mapping")
    try:
        canonical = canonical_json_bytes(payload)
    except PreflightError as error:
        raise CampaignError(f"{label} is not canonical JSON") from error
    if canonical != raw:
        raise CampaignError(f"{label} is not canonical JSON")
    return payload

def _read_unpinned_canonical_json(
    path: Path, *, label: str
) -> tuple[Mapping[str, object], str]:
    raw = _read_terminal_bytes(path)
    identity = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise CampaignError(f"{label} must be a JSON mapping")
    try:
        canonical = canonical_json_bytes(payload)
    except PreflightError as error:
        raise CampaignError(f"{label} is not canonical JSON") from error
    if raw != canonical:
        raise CampaignError(f"{label} is not canonical JSON")
    return payload, identity

def _is_directional_record(record: Mapping[str, object]) -> bool:

    return record.get("mode") == "real" and record.get("stage") in {
        "bridge_warmup",
        "joint_adaptation",
    }

def _read_unpinned_classifiable_json(
    path: Path, *, label: str
) -> tuple[Mapping[str, object], str, bool]:

    raw = _read_terminal_bytes(path)
    identity = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise CampaignError(f"{label} must be a JSON mapping")
    try:
        canonical = canonical_json_bytes(payload)
    except PreflightError:
        return payload, identity, False
    return payload, identity, raw == canonical

def load_canonical_terminal_json(
    path: Path, *, expected_sha256: str, label: str = "snapshot"
) -> Mapping[str, object]:

    return _read_canonical_terminal_json(
        path,
        expected_sha256=expected_sha256,
        label=label,
    )

def _observe_preflight_publication(
    path: Path, *, expected_sha256: str | None
) -> tuple[str, dict[str, object] | None]:

    root = _fixed_evidence_root()
    relative = _absolute(path).relative_to(root)
    try:
        parent_fd = _open_terminal_parent(
            relative.parent, create=False, allow_missing=True
        )
    except (CampaignError, OSError):
        return "unsafe-or-unreadable", None
    if parent_fd is None:
        return "absent", None
    file_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(relative.name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return "absent", None
        except OSError:
            return "unsafe-or-unreadable", None
        before = os.fstat(file_fd)
        if not stat_module.S_ISREG(before.st_mode):
            return "unsafe-or-unreadable", None
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(file_fd, 1 << 20):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(file_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or size != before.st_size:
            return "unsafe-or-unreadable", None
        observed_sha256 = digest.hexdigest()
        observed = {
            "device": before.st_dev,
            "inode": before.st_ino,
            "size_bytes": size,
            "sha256": observed_sha256,
        }
        if size == 0:
            state = "claimed-empty"
        elif expected_sha256 is not None and observed_sha256 == expected_sha256:
            state = "complete"
        else:
            state = "partial-or-mismatched"
        return state, observed
    except OSError:
        return "unsafe-or-unreadable", None
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)

def _read_terminal_bytes(path: Path) -> bytes:
    raw, _ = _read_terminal_snapshot(path)
    return raw

def _read_terminal_snapshot(path: Path) -> tuple[bytes, os.stat_result]:
    destination = validate_terminal_evidence_path(path)
    root = _fixed_evidence_root()
    relative = destination.relative_to(root)
    try:
        parent_fd = _open_terminal_parent(relative.parent, create=False)
    except OSError as error:
        raise CampaignError(f"cannot safely open terminal parent: {error}") from error
    assert parent_fd is not None
    file_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(relative.name, flags, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat_module.S_ISREG(before.st_mode):
            raise CampaignError("terminal record is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(file_fd, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise TornRecordRead("terminal record changed while reading")
        named = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat_module.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise CampaignError("terminal record path identity changed while reading")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise TornRecordRead("terminal record length differs from descriptor size")
        return raw, before
    except OSError as error:
        if error.errno == errno.ENOENT:
            raise TornRecordRead(
                f"terminal record is not readable yet: {error}"
            ) from error
        raise CampaignError(f"cannot safely read terminal record: {error}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)

def _fixed_directory(path: Path, label: str) -> Path:
    absolute = _absolute(path)
    try:
        observed = os.lstat(absolute)
    except OSError as error:
        raise CampaignError(f"fixed {label} root is unavailable: {absolute}") from error
    if not stat_module.S_ISDIR(observed.st_mode) or stat_module.S_ISLNK(
        observed.st_mode
    ):
        raise CampaignError(f"fixed {label} root is not a real directory: {absolute}")
    return absolute

def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))

def _open_terminal_parent(
    relative_parent: Path,
    *,
    create: bool,
    allow_missing: bool = False,
) -> int | None:
    root = _fixed_evidence_root()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open(root, flags)
    try:
        _verify_evidence_directory(os.fstat(current_fd), label="evidence root")
        for part in relative_parent.parts:
            if part in {"", ".", ".."}:
                raise CampaignError(f"unsafe terminal parent component {part!r}")
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    if allow_missing:
                        os.close(current_fd)
                        return None
                    raise
                created = False
                try:
                    os.mkdir(part, 0o2775, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                os.fsync(current_fd)
                next_fd = os.open(part, flags, dir_fd=current_fd)
                if created:
                    os.fchmod(next_fd, 0o2775)
            except OSError as error:
                raise CampaignError(
                    f"cannot safely traverse terminal parent {part!r}: {error}"
                ) from error
            _verify_evidence_directory(
                os.fstat(next_fd), label=f"terminal parent {part!r}"
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise

def _fixed_evidence_root() -> Path:
    root = _fixed_directory(_TERMINAL_EVIDENCE_ROOT, "terminal evidence")
    _verify_evidence_directory(os.lstat(root), label="evidence root")
    return root

def _verify_evidence_directory(observed: os.stat_result, *, label: str) -> None:
    if observed.st_gid != _EXPECTED_EVIDENCE_GID:
        raise CampaignError(
            f"{label} has gid {observed.st_gid}, expected romerolab "
            f"gid {_EXPECTED_EVIDENCE_GID}"
        )
    if not observed.st_mode & stat_module.S_ISGID:
        raise CampaignError(f"{label} is missing required setgid mode")

__all__ = [
    "AMENDMENT_CEILING",
    "BoundAttempt",
    "CampaignError",
    "CampaignLedger",
    "CampaignProjection",
    "ComputeProcess",
    "FileIdentity",
    "GpuSnapshot",
    "ImmutableRunRecord",
    "ProjectionIdentity",
    "RepositorySnapshot",
    "RunPreflight",
    "StorageSnapshot",
    "SystemSnapshot",
    "attach_reserved_log",
    "action_profile_for",
    "authorize_projection",
    "capture_live_state",
    "create_run_preflight",
    "intended_preflight_destination",
    "load_canonical_terminal_json",
    "load_attempt_start",
    "revalidate_bound_preflight",
    "replace",
    "reserve_terminal_log",
    "validate_terminal_evidence_path",
    "validate_preparation_identity",
    "validate_run_candidate",
    "write_attempt_start",
    "write_attempt_terminal",
    "write_preflight_create_new",
    "write_preparation_failure",
    "write_preparation_success",
]
