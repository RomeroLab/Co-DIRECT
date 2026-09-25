
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

class RunState(Enum):
    COMPLETED = "COMPLETED"
    PREEMPTED = "PREEMPTED"
    FAILED = "FAILED"

    @property
    def may_chain(self) -> bool:

        return self is RunState.COMPLETED

def classify_run(
    *,
    verdict: Mapping[str, Any] | None,
    expected_steps: int,
    checkpoint_loads: bool,
    stage: str | None,
    expected_stage: str,
) -> RunState:

    if expected_steps <= 0:
        raise ValueError(f"expected_steps must be positive, got {expected_steps}")

    if not checkpoint_loads:
        return RunState.FAILED
    if stage != expected_stage:
        return RunState.FAILED
    if verdict is None:
        return RunState.PREEMPTED
    if not verdict.get("passed"):
        return RunState.FAILED

    selected = verdict.get("selected_step")
    if selected is None or int(selected) < expected_steps - 1:
        return RunState.PREEMPTED
    return RunState.COMPLETED
