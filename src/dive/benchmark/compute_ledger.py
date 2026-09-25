
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dive.signed_value.roots import DENIED_PREFIXES

class LedgerError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class LedgerRow:
    run_id: str
    method_id: str
    family: str
    target_cluster_id: str
    comparison: str
    generated_trajectories: int
    final_candidates: int
    sequences_per_structure: int
    refold_samples_per_sequence: int
    reward_evaluations: int
    backward_calls: int
    filter_calls: int
    gpu_hours: float
    wall_seconds: float
    timeouts: int
    ooms: int
    invalids: int
    model_calls: int
    predictor_draws: int
    passed: bool
    reason_code: str
    exception: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "generated_trajectories",
            "final_candidates",
            "sequences_per_structure",
            "refold_samples_per_sequence",
            "reward_evaluations",
            "backward_calls",
            "filter_calls",
            "timeouts",
            "ooms",
            "invalids",
            "model_calls",
            "predictor_draws",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise LedgerError(f"{name} must be a non-negative int")
        if type(self.gpu_hours) is not float or self.gpu_hours < 0:
            raise LedgerError("gpu_hours must be a non-negative float")
        if type(self.wall_seconds) is not float or self.wall_seconds < 0:
            raise LedgerError("wall_seconds must be a non-negative float")

def append_ledger_row(
    rows: Sequence[LedgerRow], row: LedgerRow
) -> tuple[LedgerRow, ...]:
    if not isinstance(row, LedgerRow):
        raise LedgerError("append_ledger_row requires a LedgerRow")
    return tuple(rows) + (row,)

def success_with_failures_in_denominator(
    rows: Sequence[LedgerRow],
    *,
    drop_reason_codes: Sequence[str] | None = None,
) -> float:
    if drop_reason_codes:
        raise LedgerError("failures must remain in the denominator")
    if not rows:
        raise LedgerError("cannot score an empty ledger")
    return sum(1 for row in rows if row.passed) / len(rows)

def _refuse_denied(path: Path) -> None:
    text = os.path.abspath(str(path))
    for prefix in DENIED_PREFIXES:
        if text.startswith(str(prefix)):
            raise LedgerError(f"denied evidence prefix: {path}")

def write_ledger(path: Path, rows: Sequence[LedgerRow]) -> Path:
    destination = Path(path)
    _refuse_denied(destination)
    if not rows:
        raise LedgerError("refusing to write an empty compute ledger")
    table = pa.table(
        {
            "run_id": [row.run_id for row in rows],
            "method_id": [row.method_id for row in rows],
            "family": [row.family for row in rows],
            "target_cluster_id": [row.target_cluster_id for row in rows],
            "comparison": [row.comparison for row in rows],
            "generated_trajectories": [row.generated_trajectories for row in rows],
            "final_candidates": [row.final_candidates for row in rows],
            "sequences_per_structure": [row.sequences_per_structure for row in rows],
            "refold_samples_per_sequence": [
                row.refold_samples_per_sequence for row in rows
            ],
            "reward_evaluations": [row.reward_evaluations for row in rows],
            "backward_calls": [row.backward_calls for row in rows],
            "filter_calls": [row.filter_calls for row in rows],
            "gpu_hours": [row.gpu_hours for row in rows],
            "wall_seconds": [row.wall_seconds for row in rows],
            "timeouts": [row.timeouts for row in rows],
            "ooms": [row.ooms for row in rows],
            "invalids": [row.invalids for row in rows],
            "model_calls": [row.model_calls for row in rows],
            "predictor_draws": [row.predictor_draws for row in rows],
            "passed": [row.passed for row in rows],
            "reason_code": [row.reason_code for row in rows],
            "exception": [row.exception for row in rows],
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    return destination

def with_exception(row: LedgerRow, exception: str) -> LedgerRow:
    return replace(row, exception=exception, passed=False)
