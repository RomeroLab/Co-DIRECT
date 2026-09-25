
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_CHUNK = 1 << 22

class CompletionError(RuntimeError):
    pass

def checkpoint_identity(path) -> dict[str, Any]:

    path = Path(path)
    if not path.exists():
        raise CompletionError(
            f"{path} does not exist; a run cannot describe a checkpoint it did "
            f"not write, and recording it as absent would let a later stage "
            f"resume from nothing"
        )
    size = path.stat().st_size
    if size == 0:
        raise CompletionError(
            f"{path} is empty; a zero-byte file is a failed save, not a checkpoint"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}

def completion_record(
    *, final, best, stage: str, steps: int, extra: dict[str, Any] | None = None
) -> dict[str, Any]:

    return {
        "stage": str(stage),
        "steps": int(steps),
        "artifact_complete": True,
        "checkpoints": {
            "final": checkpoint_identity(final),
            "best": None if best is None else checkpoint_identity(best),
        },
        **(extra or {}),
    }
