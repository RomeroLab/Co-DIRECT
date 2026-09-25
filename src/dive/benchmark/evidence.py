
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dive.benchmark.contracts import (
    ArtifactIdentity,
    BenchmarkContract,
    file_identity,
)
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    RUNTIME_PYTHON,
)
from dive.training import preflight
from dive.training.preflight import (
    canonical_json_bytes,
    read_terminal_bytes,
    write_create_new_bytes,
    write_create_new_json,
)

class BenchmarkEvidenceError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    run_id: str
    evidence_dir: Path
    bulk_dir: Path
    attempt: ArtifactIdentity
    contract_sha256: str

_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_TEST_ROOTS: tuple[Path, Path] | None = None

def claim_benchmark_run(
    run_id: str, argv: Sequence[str], contract: BenchmarkContract
) -> BenchmarkRun:

    _validate_run_id(run_id)
    command = _argv(argv)
    runtime_python = _require_runtime_python()
    evidence_root, bulk_root = _validate_roots(contract)
    evidence_dir = evidence_root / "benchmark_ready" / run_id
    bulk_dir = bulk_root / "benchmark_ready" / run_id
    repo_commit = _current_clean_commit()
    _create_plain_directory(evidence_dir.parent)
    try:
        os.mkdir(evidence_dir, 0o2775)
    except FileExistsError:
        raise
    except OSError as error:
        raise BenchmarkEvidenceError(f"cannot claim benchmark evidence directory {evidence_dir}: {error}") from error
    try:
        _create_plain_directory(bulk_dir.parent)
        os.mkdir(bulk_dir, 0o2775)
    except FileExistsError as error:
        raise BenchmarkEvidenceError(f"bulk destination already exists: {bulk_dir}") from error
    except OSError as error:
        raise BenchmarkEvidenceError(f"cannot claim benchmark bulk directory {bulk_dir}: {error}") from error

    record = {
        "schema_version": "dive-benchmark-attempt-v1",
        "status": "BUILDING",
        "run_id": run_id,
        "argv": list(command),
        "argv_shell": shlex.join(command),
        "repo_commit": repo_commit,
        "contract_sha256": contract.semantic_sha256,
        "upstream_commit": contract.upstream_commit,
        "runtime_python": str(runtime_python),
        "host": socket.gethostname(),
        "evidence_path": str(evidence_dir),
        "bulk_path": str(bulk_dir),
    }
    attempt_path = evidence_dir / "attempt.json"
    _write_terminal_json(attempt_path, record, evidence_root, bulk_root)
    observed = _read_terminal_bytes(attempt_path, evidence_root, bulk_root)
    if observed != canonical_json_bytes(record):
        raise BenchmarkEvidenceError("attempt record changed while it was being claimed")
    return BenchmarkRun(
        run_id=run_id,
        evidence_dir=evidence_dir,
        bulk_dir=bulk_dir,
        attempt=file_identity(attempt_path),
        contract_sha256=contract.semantic_sha256,
    )

def complete_benchmark_run(run: BenchmarkRun, payload: Mapping[str, object]) -> ArtifactIdentity:

    if not isinstance(run, BenchmarkRun):
        raise BenchmarkEvidenceError("complete_benchmark_run requires a BenchmarkRun")
    if not isinstance(payload, Mapping):
        raise BenchmarkEvidenceError("completion payload must be a mapping")
    evidence_root, bulk_root = _validate_run_paths(run)
    _validate_building_attempt(run, evidence_root, bulk_root)
    completion = {
        "schema_version": "dive-benchmark-completion-v1",
        "status": "COMPLETE",
        "run_id": run.run_id,
        "contract_sha256": run.contract_sha256,
        "attempt": {"path": run.attempt.path, "sha256": run.attempt.sha256, "size_bytes": run.attempt.size_bytes},
        "payload": dict(payload),
    }
    destination = run.evidence_dir / "completion.json"
    _write_terminal_json(destination, completion, evidence_root, bulk_root)
    observed = _read_terminal_bytes(destination, evidence_root, bulk_root)
    if observed != canonical_json_bytes(completion):
        raise BenchmarkEvidenceError("completion record changed while it was being written")
    return file_identity(destination)

def _validate_run_id(run_id: object) -> None:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise BenchmarkEvidenceError("run id must match [a-z0-9][a-z0-9-]{2,79}")

def _argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
        raise BenchmarkEvidenceError("argv must be a non-empty sequence of strings")
    if not all(type(item) is str and item for item in argv):
        raise BenchmarkEvidenceError("argv must contain non-empty strings")
    return tuple(argv)

def _require_runtime_python() -> Path:
    actual = Path(sys.executable).resolve()
    expected = RUNTIME_PYTHON.resolve()
    if actual != expected:
        raise BenchmarkEvidenceError("benchmark claim requires the pinned runtime interpreter")
    return actual

def _validate_roots(contract: BenchmarkContract) -> tuple[Path, Path]:
    if not isinstance(contract, BenchmarkContract):
        raise BenchmarkEvidenceError("claim_benchmark_run requires a BenchmarkContract")
    expected = _TEST_ROOTS or (EMERGENT_EVIDENCE_ROOT, EMERGENT_BULK_ROOT)
    evidence_root, bulk_root = (Path(contract.evidence_root), Path(contract.bulk_root))
    if evidence_root != expected[0] or bulk_root != expected[1]:
        raise BenchmarkEvidenceError("evidence or bulk root is denied by the fixed production contract")
    if _TEST_ROOTS is None:
        for root in (evidence_root, bulk_root):
            if any(root == denied or denied in root.parents for denied in DENIED_PREFIXES):
                raise BenchmarkEvidenceError(f"root is denied: {root}")
    if evidence_root == bulk_root:
        raise BenchmarkEvidenceError("terminal evidence and bulk roots must be distinct")
    if not evidence_root.is_dir() or evidence_root.is_symlink():
        raise BenchmarkEvidenceError(f"terminal evidence root is unavailable or not plain: {evidence_root}")
    return evidence_root, bulk_root

def _validate_run_paths(run: BenchmarkRun) -> tuple[Path, Path]:
    _validate_run_id(run.run_id)
    expected = _TEST_ROOTS or (EMERGENT_EVIDENCE_ROOT, EMERGENT_BULK_ROOT)
    evidence_root, bulk_root = expected
    expected_evidence = evidence_root / "benchmark_ready" / run.run_id
    expected_bulk = bulk_root / "benchmark_ready" / run.run_id
    if run.evidence_dir != expected_evidence or run.bulk_dir != expected_bulk:
        raise BenchmarkEvidenceError("run paths are denied by the fixed production contract")
    if not run.evidence_dir.is_dir() or run.evidence_dir.is_symlink():
        raise BenchmarkEvidenceError("claimed evidence directory is not a plain directory")
    if not run.bulk_dir.is_dir() or run.bulk_dir.is_symlink():
        raise BenchmarkEvidenceError("claimed bulk directory is not a plain directory")
    return evidence_root, bulk_root

def _validate_building_attempt(
    run: BenchmarkRun, evidence_root: Path, bulk_root: Path
) -> None:

    attempt_path = run.evidence_dir / "attempt.json"
    try:
        raw = _read_terminal_bytes(attempt_path, evidence_root, bulk_root)
        payload = json.loads(raw)
    except (
        BenchmarkEvidenceError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise BenchmarkEvidenceError(f"cannot authenticate BUILDING attempt: {error}") from error
    observed_identity = ArtifactIdentity(
        str(attempt_path), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    if observed_identity != run.attempt:
        raise BenchmarkEvidenceError("BUILDING attempt identity does not match the claimed run")
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise BenchmarkEvidenceError("BUILDING attempt is not canonical JSON")
    expected_keys = {
        "schema_version",
        "status",
        "run_id",
        "argv",
        "argv_shell",
        "repo_commit",
        "contract_sha256",
        "upstream_commit",
        "runtime_python",
        "host",
        "evidence_path",
        "bulk_path",
    }
    if set(payload) != expected_keys:
        raise BenchmarkEvidenceError("BUILDING attempt has an invalid schema")
    if (
        payload["schema_version"] != "dive-benchmark-attempt-v1"
        or payload["status"] != "BUILDING"
        or payload["run_id"] != run.run_id
        or payload["contract_sha256"] != run.contract_sha256
        or payload["evidence_path"] != str(run.evidence_dir)
        or payload["bulk_path"] != str(run.bulk_dir)
    ):
        raise BenchmarkEvidenceError("BUILDING attempt does not bind this benchmark run")

def _create_plain_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o2775, parents=True, exist_ok=True)
    except OSError as error:
        raise BenchmarkEvidenceError(f"cannot create directory {path}: {error}") from error
    if not path.is_dir() or path.is_symlink():
        raise BenchmarkEvidenceError(f"destination must be a plain directory: {path}")

@contextmanager
def _preflight_roots(evidence_root: Path, bulk_root: Path):

    original = (
        preflight._TERMINAL_EVIDENCE_ROOT,
        preflight._TERMINAL_BULK_ROOT,
        preflight._TERMINAL_DENIED_PREFIXES,
    )
    preflight._TERMINAL_EVIDENCE_ROOT = evidence_root
    preflight._TERMINAL_BULK_ROOT = bulk_root
    if _TEST_ROOTS is not None:
        preflight._TERMINAL_DENIED_PREFIXES = ()
    try:
        yield
    finally:
        (
            preflight._TERMINAL_EVIDENCE_ROOT,
            preflight._TERMINAL_BULK_ROOT,
            preflight._TERMINAL_DENIED_PREFIXES,
        ) = original

def _write_terminal_json(path: Path, payload: Mapping[str, object], evidence_root: Path, bulk_root: Path) -> None:
    try:
        with _preflight_roots(evidence_root, bulk_root):
            write_create_new_json(path, payload)
    except preflight.PreflightError as error:
        raise BenchmarkEvidenceError(str(error)) from error

def write_terminal_bytes(
    path: Path, content: bytes, evidence_root: Path, bulk_root: Path
) -> None:

    try:
        with _preflight_roots(evidence_root, bulk_root):
            write_create_new_bytes(path, content)
    except preflight.PreflightError as error:
        raise BenchmarkEvidenceError(str(error)) from error

def _read_terminal_bytes(path: Path, evidence_root: Path, bulk_root: Path) -> bytes:
    try:
        with _preflight_roots(evidence_root, bulk_root):
            return read_terminal_bytes(path)
    except preflight.PreflightError as error:
        raise BenchmarkEvidenceError(str(error)) from error

def _current_clean_commit() -> str:
    try:
        status = subprocess.run(
            ("git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise BenchmarkEvidenceError("current repository is not clean; refusing benchmark claim")
        return subprocess.run(
            ("git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise BenchmarkEvidenceError(f"cannot determine current clean commit: {error}") from error
