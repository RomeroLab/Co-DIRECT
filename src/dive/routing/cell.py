
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from dive.routing.schedule import AMPLITUDES

ARM_KINDS: tuple[str, ...] = ("routed", "baseline")

FORWARDS_PER_ROUTED_STEP = 3

REQUIRED_BINDER_METRICS: tuple[str, ...] = (
    "self_complex_i_pAE",
    "self_complex_pLDDT",
    "self_binder_scRMSD_ca",
)

_METRICS_WHERE_INFINITY_COUNTS_AS_MEASURED: frozenset[str] = frozenset(
    {"self_binder_scRMSD_ca"}
)

CELLS_DIRNAME = "cells"

FRESH = "FRESH"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"

class CellError(RuntimeError):
    pass

def _normalized_amplitude(value: float, *, name: str) -> float:

    amplitude = float(value)
    if amplitude == 0.0:
        amplitude = 0.0
    if not any(amplitude == candidate for candidate in AMPLITUDES):
        raise CellError(
            f"{name}={value!r} is not in the pre-registered grid {AMPLITUDES}"
        )
    return amplitude

@dataclass(frozen=True, slots=True)
class ArmSpec:

    kind: str
    amplitude_x: float
    amplitude_z: float
    steps: int

    def __post_init__(self) -> None:
        if self.kind not in ARM_KINDS:
            raise CellError(f"unknown arm kind {self.kind!r}; expected one of {ARM_KINDS}")
        if int(self.steps) <= 0:
            raise CellError(f"steps must be positive, observed {self.steps!r}")
        amplitude_x = _normalized_amplitude(self.amplitude_x, name="amplitude_x")
        amplitude_z = _normalized_amplitude(self.amplitude_z, name="amplitude_z")
        if self.kind == "baseline" and (amplitude_x != 0.0 or amplitude_z != 0.0):
            raise CellError(
                "a baseline arm runs the untouched model and cannot carry an "
                f"amplitude; observed ({self.amplitude_x!r}, {self.amplitude_z!r})"
            )
        object.__setattr__(self, "amplitude_x", amplitude_x)
        object.__setattr__(self, "amplitude_z", amplitude_z)
        object.__setattr__(self, "steps", int(self.steps))

    @property
    def arm_id(self) -> str:

        if self.kind == "baseline":
            return f"baseline-s{self.steps}"
        return f"ax{self.amplitude_x:+.1f}_az{self.amplitude_z:+.1f}"

    @property
    def expected_denoiser_calls(self) -> int:

        if self.kind == "baseline":
            return self.steps
        return FORWARDS_PER_ROUTED_STEP * self.steps

    def to_dict(self) -> dict[str, object]:

        return {
            "kind": self.kind,
            "arm_id": self.arm_id,
            "amplitude_x": self.amplitude_x,
            "amplitude_z": self.amplitude_z,
            "steps": self.steps,
            "expected_denoiser_calls": self.expected_denoiser_calls,
        }

@dataclass(frozen=True, slots=True)
class CellSpec:

    run_id: str
    target: str
    seed: int
    arm: ArmSpec
    evidence_dir: Path
    generation_dir: Path
    evaluation_base: Path
    run_name: str

    @property
    def cell_id(self) -> str:

        return f"{self.target}-seed{self.seed}"

    @property
    def evidence_path(self) -> Path:

        return self.evidence_dir / f"{self.cell_id}.json"

    def to_dict(self) -> dict[str, object]:

        return {
            "run_id": self.run_id,
            "target": self.target,
            "seed": self.seed,
            "cell_id": self.cell_id,
            "arm": self.arm.to_dict(),
            "evidence_dir": str(self.evidence_dir),
            "evidence_path": str(self.evidence_path),
            "generation_dir": str(self.generation_dir),
            "evaluation_base": str(self.evaluation_base),
            "run_name": self.run_name,
        }

def cell_spec(
    run_id: str,
    target: str,
    seed: int,
    arm: ArmSpec,
    *,
    evidence_root: Path,
    bulk_root: Path,
) -> CellSpec:

    if not run_id or "/" in run_id:
        raise CellError(f"run_id must be a non-empty path component, observed {run_id!r}")
    if not target or "/" in target:
        raise CellError(f"target must be a non-empty path component, observed {target!r}")

    arm_id = arm.arm_id
    cell_id = f"{target}-seed{seed}"
    bulk_dir = Path(bulk_root) / CELLS_DIRNAME / run_id / arm_id / cell_id
    return CellSpec(
        run_id=run_id,
        target=target,
        seed=int(seed),
        arm=arm,
        evidence_dir=Path(evidence_root) / CELLS_DIRNAME / run_id / arm_id,
        generation_dir=bulk_dir / "generation",
        evaluation_base=bulk_dir / "evaluation",

        run_name=f"dive-routing-{run_id}-{arm_id}-{cell_id}",
    )

def cell_state(spec: CellSpec) -> str:

    manifest = spec.evidence_path
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict) and _describes_this_cell(payload, spec):
            return COMPLETE

    try:
        if spec.generation_dir.is_dir() and any(spec.generation_dir.iterdir()):
            return PARTIAL
    except OSError:

        return PARTIAL
    if manifest.exists():

        return PARTIAL
    return FRESH

def _describes_this_cell(payload: dict, spec: CellSpec) -> bool:

    if str(payload.get("status")) != "complete":
        return False
    if str(payload.get("target")) != spec.target:
        return False
    if payload.get("seed") != spec.seed:
        return False
    if payload.get("steps") != spec.arm.steps:
        return False
    if str(payload.get("arm_id")) != spec.arm.arm_id:
        return False
    return all(
        metric in payload and _is_measured(payload[metric], metric)
        for metric in REQUIRED_BINDER_METRICS
    )

def _is_measured(value: object, metric: str) -> bool:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if math.isnan(value):
        return False
    if math.isinf(value):
        return metric in _METRICS_WHERE_INFINITY_COUNTS_AS_MEASURED
    return True

def assert_removable(path: Path, spec: CellSpec, *, bulk_root: Path) -> Path:

    resolved = Path(path).expanduser().resolve()
    root = Path(bulk_root).expanduser().resolve()

    if resolved == root:
        raise CellError(f"refusing to remove the bulk root itself: {resolved}")
    if not resolved.is_relative_to(root):
        raise CellError(f"refusing to remove a path that is not under the bulk root: {resolved}")

    identity = (CELLS_DIRNAME, spec.run_id, spec.arm.arm_id, spec.cell_id)
    relative_parts = resolved.relative_to(root).parts
    if relative_parts[: len(identity)] != identity:
        raise CellError(
            f"refusing to remove {resolved}: it does not follow the canonical "
            f"{'/'.join(identity)} layout of this cell under the bulk root"
        )
    return resolved

def prepare_cell(spec: CellSpec, *, bulk_root: Path) -> str:

    state = cell_state(spec)
    if state != PARTIAL:
        return state

    bulk_dir = spec.generation_dir.parent
    if bulk_dir.exists():
        shutil.rmtree(assert_removable(bulk_dir, spec, bulk_root=bulk_root))
    return state
