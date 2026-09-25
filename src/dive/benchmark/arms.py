
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from dive.training.directional_campaign import (
    MIN_RESERVE,
    OBSERVED_PROJECT_GPU_SECONDS,
    PROJECT_PLANNED_CAP,
    PROJECT_TOTAL,
)
from dive.training.preflight import canonical_json_bytes

class BenchmarkArmError(RuntimeError):
    pass

class EvaluationArm(StrEnum):
    BASE = "base"
    LORA_JOINT = "lora_joint"
    FIXED_SELF = "fixed_self"
    FIXED_BACKBONE_LED = "fixed_backbone_led"
    FIXED_LOCAL_LED = "fixed_local_led"
    FIXED_JOINT = "fixed_joint"
    CONTINUOUS_FIXED = "continuous_fixed"
    TIME_ROUTER = "time_router"
    SAMPLE_ROUTER = "sample_router"
    RESIDUE_ROUTER = "residue_router"
    TASK_ID_UPPER_BOUND = "task_id_upper_bound"
    GENERIC_ADAPTER = "generic_adapter"
    PASS_MATCHED = "pass_matched"
    SHUFFLED_GATES = "shuffled_gates"
    ADAPTIVE = "adaptive"

DEFAULT_PAIRING_SEEDS = (20260826,)
DEFAULT_PAIRING_SAMPLE_INDICES = (0,)
DEFAULT_PAIRING_STEPS = 100
_FAMILIES = frozenset({"binder", "ame", "antibody"})
_PARTITIONS = frozenset({"validation", "test-blind"})
_VIEW_NAMES = frozenset({"standard", "strict", "temporal-clean"})
_FIXED_CORNER_GATES = MappingProxyType(
    {
        EvaluationArm.FIXED_SELF: (0.0, 0.0),
        EvaluationArm.FIXED_BACKBONE_LED: (0.0, 1.0),
        EvaluationArm.FIXED_LOCAL_LED: (1.0, 0.0),
        EvaluationArm.FIXED_JOINT: (1.0, 1.0),
    }
)
_SHA256 = frozenset("0123456789abcdef")

@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    parent_id: str
    family: str
    partition: str

@dataclass(frozen=True, slots=True)
class EvaluationViewSpec:
    name: str
    view_contract: Mapping[str, object]
    targets: tuple[EvaluationTarget, ...]
    seeds: tuple[int, ...] = DEFAULT_PAIRING_SEEDS
    sample_indices: tuple[int, ...] = DEFAULT_PAIRING_SAMPLE_INDICES
    steps: int = DEFAULT_PAIRING_STEPS
    code_commit: str = ""
    planned_gpu_seconds: int = 0

@dataclass(frozen=True, slots=True)
class ComputeAttempt:
    gpu_seconds: float
    wall_seconds: float
    passed: bool
    peak_bytes: int = 0
    disposition: str = "failed"
    denoiser_calls: int = 0

@dataclass(frozen=True, slots=True)
class ComputeRecord:
    requested_samples: int
    completed_samples: int
    denoiser_calls: int
    condition_count: int
    denoiser_equivalent_passes: int
    attempts: int
    failures: int
    wall_seconds: float
    gpu_seconds: float
    peak_bytes: int
    dispositions: tuple[str, ...]
    planned_gpu_seconds: int = 0

@dataclass(frozen=True, slots=True)
class EvaluationCell:
    cell_id: str
    view_contract: Mapping[str, object]
    family: str
    partition: str
    parent_id: str
    checkpoint: str
    arm: EvaluationArm
    fixed_gate_hash: str
    seed: int
    sample_index: int
    steps: int
    condition_count: int
    denoiser_equivalent_passes: int
    evaluator_hashes: Mapping[str, str]
    code_commit: str
    compute: ComputeRecord
    filled: bool
    selection_evidence_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "view_contract", MappingProxyType(dict(self.view_contract))
        )
        object.__setattr__(
            self,
            "evaluator_hashes",
            MappingProxyType(dict(self.evaluator_hashes)),
        )

def failed_attempt(gpu_seconds: float, *, denoiser_calls: int = 0) -> ComputeAttempt:
    seconds = _nonnegative_number("gpu_seconds", gpu_seconds)
    calls = _nonnegative_int("denoiser_calls", denoiser_calls)
    return ComputeAttempt(seconds, seconds, False, 0, "failed", calls)

def passed_attempt(gpu_seconds: float, *, denoiser_calls: int = 0) -> ComputeAttempt:
    seconds = _nonnegative_number("gpu_seconds", gpu_seconds)
    calls = _nonnegative_int("denoiser_calls", denoiser_calls)
    return ComputeAttempt(seconds, seconds, True, 0, "passed", calls)

def record_attempt(compute: ComputeRecord, attempt: ComputeAttempt) -> ComputeRecord:

    if not isinstance(compute, ComputeRecord):
        raise BenchmarkArmError("record_attempt requires a ComputeRecord")
    if not isinstance(attempt, ComputeAttempt):
        raise BenchmarkArmError("record_attempt requires a ComputeAttempt")
    return replace(
        compute,
        gpu_seconds=compute.gpu_seconds + attempt.gpu_seconds,
        wall_seconds=compute.wall_seconds + attempt.wall_seconds,
        denoiser_calls=compute.denoiser_calls + attempt.denoiser_calls,
        attempts=compute.attempts + 1,
        failures=compute.failures + (0 if attempt.passed else 1),
        completed_samples=compute.completed_samples + (1 if attempt.passed else 0),
        peak_bytes=max(compute.peak_bytes, attempt.peak_bytes),
        dispositions=compute.dispositions + (attempt.disposition,),
    )

def cell_identity(
    *,
    view_contract: Mapping[str, object],
    family: str,
    partition: str,
    parent_id: str,
    checkpoint: str,
    arm: EvaluationArm | str,
    fixed_gate_hash: str,
    seed: int,
    sample_index: int,
    steps: int,
    condition_count: int,
    denoiser_equivalent_passes: int,
    evaluator_hashes: Mapping[str, str],
    code_commit: str,
) -> str:

    payload = {
        "arm": arm.value if isinstance(arm, EvaluationArm) else arm,
        "checkpoint": checkpoint,
        "code_commit": code_commit,
        "condition_count": condition_count,
        "denoiser_equivalent_passes": denoiser_equivalent_passes,
        "evaluator_hashes": dict(evaluator_hashes),
        "family": family,
        "fixed_gate_hash": fixed_gate_hash,
        "parent_id": parent_id,
        "partition": partition,
        "sample_index": sample_index,
        "seed": seed,
        "steps": steps,
        "view_contract": dict(view_contract),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def build_evaluation_plan(view, checkpoint, evaluators, contract):

    spec = _require_view(view)
    checkpoint_hash = _require_checkpoint(checkpoint)
    evaluator_hashes = _require_evaluator_hashes(evaluators)
    _require_contract_arms(contract)
    view_contract = dict(spec.view_contract)
    if "name" not in view_contract:
        view_contract["name"] = spec.name
    cells_by_arm: dict[EvaluationArm, list[EvaluationCell]] = {
        arm: [] for arm in EvaluationArm
    }
    targets = _sorted_targets(spec.targets)
    for target in targets:
        _require_target(target)
        for seed in spec.seeds:
            _require_seed(seed)
            for sample_index in spec.sample_indices:
                _require_sample_index(sample_index)
                for arm in EvaluationArm:
                    cells_by_arm[arm].append(
                        _build_cell(
                            view_contract=view_contract,
                            target=target,
                            checkpoint=checkpoint_hash,
                            arm=arm,
                            seed=seed,
                            sample_index=sample_index,
                            steps=spec.steps,
                            evaluator_hashes=evaluator_hashes,
                            code_commit=spec.code_commit,
                        )
                    )
    plan = MappingProxyType({arm: tuple(cells) for arm, cells in cells_by_arm.items()})
    assert_paired_plan(plan)
    assert_compute_budget(spec.planned_gpu_seconds + _planned_gpu_seconds(plan))
    return plan

def assert_paired_plan(plan) -> None:

    if not isinstance(plan, Mapping) or not plan:
        raise BenchmarkArmError("paired plan must map every arm to cells")
    if set(plan) != set(EvaluationArm):
        raise BenchmarkArmError("paired plan is missing evaluation arms")
    keys = {arm: _pairing_keys(cells) for arm, cells in plan.items()}
    if len(set(keys.values())) != 1:
        raise BenchmarkArmError(
            "paired arms must share identical parent/seed/sample tuples"
        )
    shared = next(iter(keys.values()))
    if not shared:
        raise BenchmarkArmError("paired plan has no parent/seed/sample tuples")

def assert_compute_budget(
    planned_gpu_seconds: int,
    observed_gpu_seconds: int = OBSERVED_PROJECT_GPU_SECONDS,
) -> None:

    planned = _nonnegative_int("planned_gpu_seconds", planned_gpu_seconds)
    observed = _nonnegative_int("observed_gpu_seconds", observed_gpu_seconds)
    if observed != OBSERVED_PROJECT_GPU_SECONDS:
        raise BenchmarkArmError(
            "compute accounting must read the existing observed GPU-seconds ledger"
        )
    project = observed + planned
    reserve = PROJECT_TOTAL - project
    if project > PROJECT_PLANNED_CAP or reserve < MIN_RESERVE:
        raise BenchmarkArmError(
            "planned GPU-days would exceed the 84 GPU-day cap or violate the "
            "36 GPU-day reserve"
        )

def fill_continuous_fixed(
    cell: EvaluationCell,
    *,
    gate_values: Sequence[float] | None = None,
    selection_evidence_hash: str | None = None,
) -> EvaluationCell:

    del cell, gate_values, selection_evidence_hash
    raise BenchmarkArmError(
        "continuous_fixed cannot be filled before validation selection"
    )

def render_plan_identity(plan) -> str:

    if not isinstance(plan, Mapping):
        raise BenchmarkArmError("render_plan_identity requires a paired plan")
    payload = {
        "arms": [arm.value for arm in EvaluationArm],
        "cells": [cell.cell_id for arm in EvaluationArm for cell in plan[arm]],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def _build_cell(
    *,
    view_contract: Mapping[str, object],
    target: EvaluationTarget,
    checkpoint: str,
    arm: EvaluationArm,
    seed: int,
    sample_index: int,
    steps: int,
    evaluator_hashes: Mapping[str, str],
    code_commit: str,
) -> EvaluationCell:
    condition_count, passes = _arm_accounting(arm)
    gate_hash = _fixed_gate_hash(arm)
    filled = arm is not EvaluationArm.CONTINUOUS_FIXED
    identity = cell_identity(
        view_contract=view_contract,
        family=target.family,
        partition=target.partition,
        parent_id=target.parent_id,
        checkpoint=checkpoint,
        arm=arm,
        fixed_gate_hash=gate_hash,
        seed=seed,
        sample_index=sample_index,
        steps=steps,
        condition_count=condition_count,
        denoiser_equivalent_passes=passes,
        evaluator_hashes=evaluator_hashes,
        code_commit=code_commit,
    )
    return EvaluationCell(
        cell_id=identity,
        view_contract=view_contract,
        family=target.family,
        partition=target.partition,
        parent_id=target.parent_id,
        checkpoint=checkpoint,
        arm=arm,
        fixed_gate_hash=gate_hash,
        seed=seed,
        sample_index=sample_index,
        steps=steps,
        condition_count=condition_count,
        denoiser_equivalent_passes=passes,
        evaluator_hashes=evaluator_hashes,
        code_commit=code_commit,
        compute=ComputeRecord(
            requested_samples=1,
            completed_samples=0,
            denoiser_calls=0,
            condition_count=condition_count,
            denoiser_equivalent_passes=passes,
            attempts=0,
            failures=0,
            wall_seconds=0.0,
            gpu_seconds=0.0,
            peak_bytes=0,
            dispositions=(),
            planned_gpu_seconds=0,
        ),
        filled=filled,
        selection_evidence_hash=None,
    )

def _fixed_gate_hash(arm: EvaluationArm) -> str:
    if arm is EvaluationArm.CONTINUOUS_FIXED:
        payload = {
            "arm": EvaluationArm.CONTINUOUS_FIXED.value,
            "selection_evidence_hash": None,
            "status": "unfilled",
        }
    elif arm in _FIXED_CORNER_GATES:
        payload = {"arm": arm.value, "gates": list(_FIXED_CORNER_GATES[arm])}
    else:
        payload = {"arm": arm.value, "gates": None}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def _require_view(view) -> EvaluationViewSpec:
    if not isinstance(view, EvaluationViewSpec):
        raise BenchmarkArmError("build_evaluation_plan requires an EvaluationViewSpec")
    if view.name not in _VIEW_NAMES:
        raise BenchmarkArmError(f"unsupported benchmark view name {view.name!r}")
    if not isinstance(view.view_contract, Mapping):
        raise BenchmarkArmError("view_contract must be a mapping")
    if not view.targets:
        raise BenchmarkArmError("evaluation view has no fixture targets")
    if not view.seeds or not view.sample_indices:
        raise BenchmarkArmError("evaluation view requires seeds and samples")
    _require_positive_int("steps", view.steps)
    _nonnegative_int("planned_gpu_seconds", view.planned_gpu_seconds)
    _require_commit(view.code_commit)
    seen: set[tuple[str, str, str]] = set()
    for target in view.targets:
        key = (target.parent_id, target.family, target.partition)
        if key in seen:
            raise BenchmarkArmError(f"duplicate evaluation target {key}")
        seen.add(key)
    return view

def _require_target(target: EvaluationTarget) -> None:
    if not isinstance(target, EvaluationTarget):
        raise BenchmarkArmError("evaluation targets must be EvaluationTarget records")
    if type(target.parent_id) is not str or not target.parent_id:
        raise BenchmarkArmError("parent_id must be a non-empty string")
    if target.family not in _FAMILIES:
        raise BenchmarkArmError(f"unsupported evaluation family {target.family!r}")
    if target.partition not in _PARTITIONS:
        raise BenchmarkArmError(
            f"evaluation partition must be validation or test-blind, not {target.partition!r}"
        )

def _require_contract_arms(contract) -> None:
    arms = tuple(getattr(contract, "arms", ()))
    expected = tuple(arm.value for arm in EvaluationArm)
    if arms != expected:
        raise BenchmarkArmError("arms must match the frozen evaluation-arm vocabulary")

def _require_checkpoint(checkpoint) -> str:
    if type(checkpoint) is str:
        digest = checkpoint
    else:
        digest = getattr(checkpoint, "sha256", None)
        if digest is None and isinstance(checkpoint, Mapping):
            digest = checkpoint.get("sha256")
    _require_sha256("checkpoint", digest)
    return digest

def _require_evaluator_hashes(evaluators) -> dict[str, str]:
    if isinstance(evaluators, Mapping):
        hashes = {str(key): str(value) for key, value in evaluators.items()}
    else:
        registry = getattr(evaluators, "semantic_sha256", None)
        families = getattr(evaluators, "evaluators", None)
        if type(registry) is not str or families is None:
            raise BenchmarkArmError("evaluators must supply registry and family hashes")
        hashes = {"registry": registry}
        for item in families:
            hashes[item.family] = item.semantic_sha256
    required = {"registry", *_FAMILIES}
    if required - hashes.keys():
        raise BenchmarkArmError(
            "evaluator hashes must include registry and each family"
        )
    for name, digest in hashes.items():
        _require_sha256(f"evaluator hash {name}", digest)
    return hashes

def _arm_accounting(arm: EvaluationArm) -> tuple[int, int]:

    if arm is EvaluationArm.ADAPTIVE:
        return 3, 3
    if arm is EvaluationArm.PASS_MATCHED:
        return 1, 3
    return 1, 1

def _pairing_keys(cells: Sequence[EvaluationCell]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (cell.family, cell.partition, cell.parent_id, cell.seed, cell.sample_index)
            for cell in cells
        )
    )

def _sorted_targets(
    targets: Sequence[EvaluationTarget],
) -> tuple[EvaluationTarget, ...]:
    return tuple(
        sorted(targets, key=lambda item: (item.family, item.partition, item.parent_id))
    )

def _planned_gpu_seconds(plan: Mapping[EvaluationArm, Sequence[EvaluationCell]]) -> int:
    return sum(
        cell.compute.planned_gpu_seconds for cells in plan.values() for cell in cells
    )

def _require_seed(seed: int) -> None:
    _require_positive_int("seed", seed)

def _require_sample_index(sample_index: int) -> None:
    if (
        type(sample_index) is not int
        or isinstance(sample_index, bool)
        or sample_index < 0
    ):
        raise BenchmarkArmError("sample_index must be a nonnegative integer")

def _require_commit(value: object) -> None:
    if type(value) is not str or len(value) != 40 or set(value) - _SHA256:
        raise BenchmarkArmError("code_commit must be a 40-character lowercase SHA-1")

def _require_sha256(label: str, value: object) -> None:
    if type(value) is not str or len(value) != 64 or set(value) - _SHA256:
        raise BenchmarkArmError(f"{label} must be a lowercase SHA-256")

def _require_positive_int(label: str, value: object) -> int:
    number = _nonnegative_int(label, value)
    if number < 1:
        raise BenchmarkArmError(f"{label} must be a positive integer")
    return number

def _nonnegative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise BenchmarkArmError(f"{label} must be a nonnegative integer")
    return value

def _nonnegative_number(label: str, value: object) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise BenchmarkArmError(f"{label} must be a nonnegative number")
    number = float(value)
    if number < 0 or not math.isfinite(number):
        raise BenchmarkArmError(f"{label} must be a nonnegative number")
    return number
