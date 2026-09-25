
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from dive.signed_value.roots import DENIED_PREFIXES
from dive.training.preflight import canonical_json_bytes

class SealedAccessError(RuntimeError):
    pass

_SEALED_MARKERS = (
    "sealed",
    "test-blind",
    "test_blind",
    "ame-sealed",
    "binder-sealed",
    "antibody-sealed",
)

_PUBLISHED_VIEW_SEMANTIC = (
    "f5e54f8e0d91fe857f3d36135fb603f921523fa16563eb83d2b099e34ac8d253"
)
_PUBLISHED_VIEW_PARTITION = (
    "527c359aefa234ea3015f645a8d128b587ad22c7e0b1aa263491c6a54a11fc03"
)

@dataclass(frozen=True, slots=True)
class SealedEnvelope:
    family: str
    view_semantic_hash: str
    view_partition_hash: str
    metric_hash: str
    threshold_hash: str
    seed: int
    budget_gpu_seconds: int
    envelope_sha256: str
    opened: bool = False

    def as_mapping(self) -> dict[str, object]:
        return {
            "budget_gpu_seconds": self.budget_gpu_seconds,
            "envelope_sha256": self.envelope_sha256,
            "family": self.family,
            "metric_hash": self.metric_hash,
            "opened": self.opened,
            "seed": self.seed,
            "status": "sealed",
            "threshold_hash": self.threshold_hash,
            "view_partition_hash": self.view_partition_hash,
            "view_semantic_hash": self.view_semantic_hash,
        }

def bind_sealed_envelope(
    *,
    view_semantic_hash: str,
    view_partition_hash: str,
    family: str,
    metric_hash: str,
    threshold_hash: str,
    seed: int,
    budget_gpu_seconds: int,
) -> SealedEnvelope:

    if view_semantic_hash != _PUBLISHED_VIEW_SEMANTIC:
        raise SealedAccessError("sealed envelope must bind the published view hash")
    if view_partition_hash != _PUBLISHED_VIEW_PARTITION:
        raise SealedAccessError(
            "sealed envelope must bind the published partition hash"
        )
    if budget_gpu_seconds < 0:
        raise SealedAccessError("budget_gpu_seconds must be non-negative")
    payload = {
        "budget_gpu_seconds": budget_gpu_seconds,
        "family": family,
        "metric_hash": metric_hash,
        "seed": seed,
        "threshold_hash": threshold_hash,
        "view_partition_hash": view_partition_hash,
        "view_semantic_hash": view_semantic_hash,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return SealedEnvelope(
        family=family,
        view_semantic_hash=view_semantic_hash,
        view_partition_hash=view_partition_hash,
        metric_hash=metric_hash,
        threshold_hash=threshold_hash,
        seed=seed,
        budget_gpu_seconds=budget_gpu_seconds,
        envelope_sha256=digest,
        opened=False,
    )

def open_sealed(envelope: SealedEnvelope) -> None:

    raise SealedAccessError(
        f"refusing to open sealed envelope {envelope.envelope_sha256}"
    )

def refuse_sealed_path(path: str | Path) -> None:

    candidate = Path(path)
    text = os.path.abspath(str(candidate)).lower()
    for prefix in DENIED_PREFIXES:
        if text.startswith(str(prefix).lower()):
            raise SealedAccessError(f"denied evidence prefix: {candidate}")
    if any(marker in text for marker in _SEALED_MARKERS):
        raise SealedAccessError(f"sealed or blind path refused: {candidate}")
