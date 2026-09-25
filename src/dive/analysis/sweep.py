
from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

BASELINE_ARM_ID = "baseline-s1200"

EQUAL_STEPS_ARM_ID = "ax+0.0_az+0.0"
I_PAE_KEY = "self_complex_i_pAE"
I_PAE_NORMALIZATION = 31.0

SENTINEL_NORMALIZED = 1.0

HEADROOM_MAX_NORMALIZED = 26.0 / I_PAE_NORMALIZATION
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260812

class SweepAnalysisError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class CellResult:

    arm_id: str
    target: str
    seed: int
    amplitude_x: float
    amplitude_z: float
    i_pAE: float
    pLDDT: float
    scRMSD: float
    alpha: float
    beta: float
    coil: float
    denoiser_calls_match: bool

def load_cells(evidence_root: Path | str, run_id: str) -> tuple[CellResult, ...]:

    root = Path(evidence_root) / "cells" / run_id
    if not root.is_dir():
        raise SweepAnalysisError(f"no cell tree at {root}")

    results: list[CellResult] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or str(payload.get("status")) != "complete":
            continue
        arm = payload.get("arm") or {}
        try:
            results.append(
                CellResult(
                    arm_id=str(payload["arm_id"]),
                    target=str(payload["target"]),
                    seed=int(payload["seed"]),
                    amplitude_x=float(arm.get("amplitude_x", 0.0)),
                    amplitude_z=float(arm.get("amplitude_z", 0.0)),
                    i_pAE=float(payload[I_PAE_KEY]),
                    pLDDT=float(payload["self_complex_pLDDT"]),
                    scRMSD=float(payload["self_binder_scRMSD_ca"]),
                    alpha=float(payload["alpha"]),
                    beta=float(payload["beta"]),
                    coil=float(payload["coil"]),
                    denoiser_calls_match=bool(payload["denoiser_calls_match"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(results)

def missing_cells(
    cells: Iterable[CellResult],
    targets: Sequence[str],
    seeds: Sequence[int],
    arm_ids: Sequence[str],
) -> tuple[str, ...]:

    present = {(cell.arm_id, cell.target, cell.seed) for cell in cells}
    return tuple(
        f"{arm_id}/{target}-seed{seed}"
        for arm_id in arm_ids
        for target in targets
        for seed in seeds
        if (arm_id, target, seed) not in present
    )

def unplanned_cells(
    cells: Iterable[CellResult],
    targets: Sequence[str],
    seeds: Sequence[int],
    arm_ids: Sequence[str],
) -> tuple[str, ...]:

    planned = {
        (arm_id, target, seed)
        for arm_id in arm_ids
        for target in targets
        for seed in seeds
    }
    present = {(cell.arm_id, cell.target, cell.seed) for cell in cells}
    return tuple(
        f"{arm_id}/{target}-seed{seed}" for arm_id, target, seed in sorted(present - planned)
    )

def per_target_deltas(
    cells: Iterable[CellResult], arm_id: str, *, reference: str = BASELINE_ARM_ID
) -> dict[str, float]:

    cells = tuple(cells)
    against = {
        (cell.target, cell.seed): cell.i_pAE
        for cell in cells
        if cell.arm_id == reference
    }
    by_target: dict[str, list[float]] = {}
    for cell in cells:
        if cell.arm_id != arm_id:
            continue
        key = (cell.target, cell.seed)
        if key not in against:
            raise SweepAnalysisError(
                f"{arm_id} has a cell at {cell.target} seed {cell.seed} with no "
                f"baseline partner in {reference}; an unpaired cell changes what "
                "the mean is over"
            )
        by_target.setdefault(cell.target, []).append(cell.i_pAE - against[key])
    return {
        target: math.fsum(values) / len(values) for target, values in sorted(by_target.items())
    }

def secondary_summary(cells: Iterable[CellResult]) -> dict[str, dict[str, float | None]]:

    grouped: dict[str, list[CellResult]] = {}
    for cell in cells:
        grouped.setdefault(cell.arm_id, []).append(cell)

    summary: dict[str, dict[str, float | None]] = {}
    for arm_id, arm_cells in sorted(grouped.items()):
        finite = [cell.scRMSD for cell in arm_cells if math.isfinite(cell.scRMSD)]
        summary[arm_id] = {
            "self_complex_pLDDT": math.fsum(c.pLDDT for c in arm_cells) / len(arm_cells),
            "self_binder_scRMSD_ca": (
                math.fsum(finite) / len(finite) if finite else None
            ),
            "n_nonfinite_scRMSD": len(arm_cells) - len(finite),
            "alpha": math.fsum(c.alpha for c in arm_cells) / len(arm_cells),
            "beta": math.fsum(c.beta for c in arm_cells) / len(arm_cells),
            "coil": math.fsum(c.coil for c in arm_cells) / len(arm_cells),
        }
    return summary

def arm_score(per_target: Mapping[str, float]) -> float:

    if not per_target:
        raise SweepAnalysisError("no paired cells for this arm; refusing to average nothing")
    return math.fsum(per_target.values()) / len(per_target)

def bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:

    if not values:
        raise SweepAnalysisError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        math.fsum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[min(int(0.975 * resamples), resamples - 1)]

def select_winner(scores: Mapping[str, tuple[float, float, float]]) -> str:

    candidates = {arm: value for arm, value in scores.items() if arm != BASELINE_ARM_ID}
    if not candidates:
        raise SweepAnalysisError("no routed arm to select from")
    return min(
        sorted(candidates),
        key=lambda arm: (
            candidates[arm][0],
            abs(candidates[arm][1]) + abs(candidates[arm][2]),
            candidates[arm][1],
            candidates[arm][2],
        ),
    )

def headroom_subset(cells: Iterable[CellResult]) -> tuple[str, ...]:

    by_target: dict[str, list[float]] = {}
    for cell in cells:
        if cell.arm_id == BASELINE_ARM_ID:
            by_target.setdefault(cell.target, []).append(cell.i_pAE)
    return tuple(
        target
        for target, values in sorted(by_target.items())
        if math.fsum(values) / len(values) <= HEADROOM_MAX_NORMALIZED
    )

def sentinel_counts(cells: Iterable[CellResult]) -> dict[str, int]:

    counts: dict[str, int] = {}
    for cell in cells:
        counts.setdefault(cell.arm_id, 0)
        if cell.i_pAE == SENTINEL_NORMALIZED:
            counts[cell.arm_id] += 1
    return counts

def reduce_sweep(
    cells: Sequence[CellResult],
    targets: Sequence[str],
    seeds: Sequence[int],
    *,
    arm_ids: Sequence[str],
) -> dict[str, object]:

    missing = missing_cells(cells, targets, seeds, arm_ids)
    unplanned = unplanned_cells(cells, targets, seeds, arm_ids)
    mismatched = [
        f"{cell.arm_id}/{cell.target}-seed{cell.seed}"
        for cell in cells
        if not cell.denoiser_calls_match
    ]
    if missing or unplanned or mismatched:
        return {
            "verdict": "BLOCKED",
            "label": "exploration -- the development sweep is not a result",
            "missing_cells": list(missing),
            "unplanned_cells": list(unplanned),
            "denoiser_call_mismatches": sorted(mismatched),
        }

    subset = headroom_subset(cells)
    arms: dict[str, object] = {}
    scores: dict[str, tuple[float, float, float]] = {}
    for arm_id in arm_ids:
        if arm_id == BASELINE_ARM_ID:
            continue
        per_target = per_target_deltas(cells, arm_id)
        score = arm_score(per_target)
        amplitudes = next(
            (cell.amplitude_x, cell.amplitude_z) for cell in cells if cell.arm_id == arm_id
        )
        low, high = bootstrap_interval(list(per_target.values()))
        subset_values = [per_target[t] for t in subset if t in per_target]

        has_equal_steps = any(cell.arm_id == EQUAL_STEPS_ARM_ID for cell in cells)
        equal_steps = (
            per_target_deltas(cells, arm_id, reference=EQUAL_STEPS_ARM_ID)
            if has_equal_steps
            else {}
        )
        arms[arm_id] = {
            "score": score,
            "bootstrap_95": [low, high],
            "per_target": per_target,
            "headroom_subset_score": (
                math.fsum(subset_values) / len(subset_values) if subset_values else None
            ),
            "equal_steps_context_score": arm_score(equal_steps) if equal_steps else None,
        }
        scores[arm_id] = (score, *amplitudes)

    winner = select_winner(scores)
    return {
        "verdict": "COMPLETE",
        "label": "exploration -- the development sweep is not a result",
        "primary_metric": I_PAE_KEY,
        "direction": "minimize -- a negative delta is an improvement",
        "compared_against": BASELINE_ARM_ID,
        "arms": arms,
        "winner": winner,
        "winner_score": scores[winner][0],
        "winner_beats_matched_compute": scores[winner][0] < 0.0,
        "headroom_subset": list(subset),
        "sentinel_counts": sentinel_counts(cells),
        "secondary": secondary_summary(cells),
        "equal_steps_context_note": (
            "each arm against the (0,0) equal-steps arm. Context only; a win "
            "here is not evidence for clause 3."
        ),
    }
