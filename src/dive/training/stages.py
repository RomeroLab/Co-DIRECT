
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

class StageError(RuntimeError):
    pass

class TrainingStage(StrEnum):

    NULL_WARMUP = "null_warmup"
    ROUTER_WARMUP = "router_warmup"
    JOINT_LORA = "joint_lora"

@dataclass(frozen=True, slots=True)
class StageContract:

    stage: TrainingStage
    train_lora: bool
    train_router: bool
    train_presence: bool

    train_autoencoder: bool = False

    fixed_corner: str | None = None

    utilization_weight: float = 0.0
    utilization_decay_fraction: float = 0.20

    joint_preservation_every: int = 0

    def utilization_at(self, *, step: int, total_steps: int) -> float:

        if self.utilization_weight == 0.0:
            return 0.0
        if total_steps <= 0:
            raise StageError("total_steps must be positive to schedule the decay")
        horizon = self.utilization_decay_fraction * total_steps
        if horizon <= 0 or step >= horizon:
            return 0.0
        return self.utilization_weight * (1.0 - step / horizon)

STAGE_CONTRACTS: dict[TrainingStage, StageContract] = {
    TrainingStage.NULL_WARMUP: StageContract(
        stage=TrainingStage.NULL_WARMUP,
        train_lora=True,
        train_router=False,
        train_presence=True,
        fixed_corner="joint",
        joint_preservation_every=8,
    ),
    TrainingStage.ROUTER_WARMUP: StageContract(
        stage=TrainingStage.ROUTER_WARMUP,
        train_lora=False,
        train_router=True,
        train_presence=True,
        utilization_weight=0.01,
    ),
    TrainingStage.JOINT_LORA: StageContract(
        stage=TrainingStage.JOINT_LORA,
        train_lora=True,
        train_router=True,
        train_presence=True,
    ),
}

def contract_for(stage: TrainingStage) -> StageContract:
    try:
        return STAGE_CONTRACTS[TrainingStage(stage)]
    except (KeyError, ValueError) as error:
        raise StageError(
            f"unknown stage {stage!r}; expected one of {[s.value for s in TrainingStage]}"
        ) from error
