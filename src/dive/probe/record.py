
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from dive.provenance import FileEvidence

_STACKED_FIELDS: tuple[str, ...] = (
    "valid",
    "a_x",
    "a_z",
    "b_zero_z_to_x",
    "b_zero_x_to_z",
    "b_sham_z_to_x",
    "b_sham_x_to_z",
    "x_t_bb_ca",
)

class RecordError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class StepRecord:

    target: str
    seed: int
    step: int
    t_bb_ca: float
    t_local_latents: float
    valid: Tensor
    a_x: Tensor
    a_z: Tensor
    b_zero_z_to_x: Tensor
    b_zero_x_to_z: Tensor
    b_sham_z_to_x: Tensor
    b_sham_x_to_z: Tensor
    x_t_bb_ca: Tensor
    x_target: Tensor

class ShardWriter:

    def __init__(self, root: Path | str, run_id: str, target: str, seed: int) -> None:
        self._path = Path(root) / run_id / f"{target}__seed{seed}.pt"
        self._records: list[StepRecord] = []
        self._target = target
        self._seed = seed

    def append(self, record: StepRecord) -> None:

        self._records.append(record)

    def close(self) -> dict[str, object]:

        if not self._records:
            raise RecordError(f"refusing to write an empty shard: {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "target": self._target,
            "seed": self._seed,
            "step": torch.tensor([r.step for r in self._records]),
            "t_bb_ca": torch.tensor([r.t_bb_ca for r in self._records]),
            "t_local_latents": torch.tensor([r.t_local_latents for r in self._records]),
        }
        for field in _STACKED_FIELDS:
            payload[field] = torch.stack(
                [getattr(record, field).detach().cpu() for record in self._records]
            )

        payload["x_target"] = self._agreed_target()
        torch.save(payload, self._path)
        evidence = FileEvidence.from_path(self._path)
        return {
            "path": str(self._path),
            "target": self._target,
            "seed": self._seed,
            "steps": len(self._records),
            "size_bytes": evidence.size_bytes,
            "sha256": evidence.sha256,
        }

    def _agreed_target(self) -> Tensor:

        first = self._records[0].x_target.detach().cpu()
        for record in self._records[1:]:
            other = record.x_target.detach().cpu()
            if other.shape != first.shape or not torch.equal(other, first):
                raise RecordError(
                    "x_target changed between steps of one shard: "
                    f"step {record.step} disagrees with step {self._records[0].step}"
                )
        return first

def write_manifest(manifest_root: Path | str, run_id: str, payload: dict) -> Path:

    root = Path(manifest_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run_id}.json"
    path.write_text(json.dumps({"run_id": run_id, **payload}, indent=2, sort_keys=True) + "\n")
    return path
