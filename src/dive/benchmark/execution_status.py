
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

class ExecutionStatusError(RuntimeError):
    pass

class ResultOrigin(StrEnum):
    FRESH_RUN = "fresh_run"
    HISTORICAL_REUSE = "historical_reuse"
    FIXTURE = "fixture"
    NOT_RUN = "not_run"

@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    baseline_id: str
    family: str
    task_type: str
    pipeline: str
    official_or_custom_contract: str
    code_present: bool
    environment_present: bool
    weights_present: bool
    checkpoint_load_exact: bool | None
    adapter_fixture_ok: bool
    generation_executed: bool
    sequence_design_executed: bool
    refold_or_eval_executed: bool
    multi_target_pilot: bool
    result_origin: ResultOrigin
    blocker: str | None = None

    def __post_init__(self) -> None:
        if self.result_origin is ResultOrigin.FIXTURE and self.generation_executed:
            raise ExecutionStatusError("fixture rows cannot claim generation executed")
        if self.result_origin is ResultOrigin.NOT_RUN and (
            self.generation_executed or self.sequence_design_executed
        ):
            raise ExecutionStatusError("not_run cannot claim executed subprocesses")

def scientific_table_admitted(row: ExecutionEvidence) -> bool:

    if row.result_origin in {ResultOrigin.FIXTURE, ResultOrigin.NOT_RUN}:
        return False
    if not row.generation_executed:
        return False
    return True
