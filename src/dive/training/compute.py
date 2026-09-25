
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dive.training.stages import TrainingStage, contract_for

DENOISER_FORWARDS_PER_STEP = 3

FORWARD_ONLY = 1.0
FORWARD_AND_BACKWARD = 3.0

class ComputeError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class StageCost:

    stage: str
    steps: int
    accumulate: int
    forwards: int
    backward_multiplier: float
    teacher_forwards: float
    denoiser_equivalents: float

    def as_provenance(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "steps": self.steps,
            "accumulate": self.accumulate,
            "forwards": self.forwards,
            "backward_multiplier": self.backward_multiplier,
            "teacher_forwards": self.teacher_forwards,
            "denoiser_equivalents": self.denoiser_equivalents,
            "denoiser_forwards_per_step": DENOISER_FORWARDS_PER_STEP,
        }

def stage_cost(stage, *, steps: int, accumulate: int) -> StageCost:

    if steps <= 0 or accumulate <= 0:
        raise ComputeError(
            f"steps and accumulate must be positive, got steps={steps} "
            f"accumulate={accumulate}"
        )
    contract = contract_for(stage)
    forwards = steps * accumulate * DENOISER_FORWARDS_PER_STEP
    multiplier = FORWARD_AND_BACKWARD if contract.train_lora else FORWARD_ONLY

    teacher = 0.0
    every = int(contract.joint_preservation_every or 0)
    if every > 0 and float(contract.joint_preservation_weight) > 0.0:
        teacher = (steps / every) * DENOISER_FORWARDS_PER_STEP

    return StageCost(
        stage=str(stage),
        steps=steps,
        accumulate=accumulate,
        forwards=forwards,
        backward_multiplier=multiplier,
        teacher_forwards=teacher,
        denoiser_equivalents=forwards * multiplier + teacher,
    )

def matched_step_count(candidate, *, control_stage, control_accumulate: int) -> int:

    total = sum(
        stage_cost(stage, steps=steps, accumulate=accumulate).denoiser_equivalents
        for stage, steps, accumulate in candidate
    )
    per_step = stage_cost(
        control_stage, steps=1, accumulate=control_accumulate
    ).denoiser_equivalents
    if per_step <= 0:
        raise ComputeError(f"control stage {control_stage} has no per-step cost")
    steps = int(total // per_step)
    if steps <= 0:
        raise ComputeError(
            f"the candidate's budget buys no complete control step; total "
            f"{total}, per step {per_step}"
        )
    return steps
