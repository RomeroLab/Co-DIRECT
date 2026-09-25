
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

_HASH_CHUNK_BYTES = 8 * 1024 * 1024

class ProvenanceError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class FileEvidence:

    path: Path
    size_bytes: int
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> FileEvidence:

        resolved_path = Path(path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(resolved_path)
        if not resolved_path.is_file():
            raise ProvenanceError(f"provenance path is not a regular file: {resolved_path}")

        digest = hashlib.sha256()
        with resolved_path.open("rb") as artifact:
            before = os.fstat(artifact.fileno())
            while chunk := artifact.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
            after = os.fstat(artifact.fileno())
        try:
            current = resolved_path.stat()
        except OSError as error:
            raise ProvenanceError(f"file changed while hashing: {resolved_path}") from error
        if not (
            _stable_file_identity(before) == _stable_file_identity(after)
            == _stable_file_identity(current)
        ):
            raise ProvenanceError(f"file changed while hashing: {resolved_path}")
        return cls(
            path=resolved_path,
            size_bytes=before.st_size,
            sha256=digest.hexdigest(),
        )

    def to_dict(self) -> dict[str, int | str]:

        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

@dataclass(frozen=True, slots=True)
class GitEvidence:

    path: Path
    commit: str
    clean: bool

    def to_dict(self) -> dict[str, bool | str]:

        return {"path": str(self.path), "commit": self.commit, "clean": self.clean}

def verify_git_checkout(path: Path, expected_commit: str) -> GitEvidence:

    checkout = Path(path).resolve()
    try:
        head = _git_output(checkout, "rev-parse", "HEAD")
        status = _git_output(checkout, "status", "--short")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(f"could not inspect git checkout: {checkout}") from error

    if head != expected_commit:
        raise ProvenanceError(
            f"git commit mismatch for {checkout}: expected {expected_commit}, observed {head}"
        )
    if status:
        raise ProvenanceError(f"git checkout is dirty: {checkout}")
    return GitEvidence(path=checkout, commit=head, clean=True)

def _git_output(checkout: Path, *args: str) -> str:

    result = subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()

def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
