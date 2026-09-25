
from __future__ import annotations

from pathlib import Path

from dive.training.stages import TrainingStage

class CheckpointStageError(RuntimeError):
    pass

def stage_of_checkpoint(path, *, map_location: str = "cpu") -> TrainingStage:

    import torch

    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    raw = payload.get("stage")
    if raw is None:
        raise CheckpointStageError(
            f"{path} records no stage; defaulting would silently build the model "
            f"under ownership the checkpoint never trained with, and a stage "
            f"that pins the gates would bypass the router without saying so"
        )
    try:
        return TrainingStage(str(raw))
    except ValueError as error:
        raise CheckpointStageError(
            f"{path} records unknown stage {raw!r}; expected one of "
            f"{[s.value for s in TrainingStage]}"
        ) from error
