
from __future__ import annotations

from collections.abc import Mapping

import torch

FORBIDDEN_LABEL_KEYS = (
    "j_final",
    "c_refold",
    "refold",
    "binding",
    "affinity",
    "function_score",
    "boltz",
    "af2",
    "predictor",
)

def refuse_outcome_labels(payload: Mapping | None) -> None:

    if not payload:
        return
    for key in payload:
        lowered = str(key).lower()
        if any(token in lowered for token in FORBIDDEN_LABEL_KEYS):
            raise ValueError("J_final / predictor / function scores are not training labels")

def relation_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target relation features must match")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be bool")
    diff = (predicted[mask] - target[mask]).square()
    if diff.numel() == 0:
        return predicted.new_zeros(())
    return diff.mean()

def swapped_relation_contrast(
    predicted: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 0.05,
    example_id=None,
    negative_example_id=None,
) -> tuple[torch.Tensor, dict]:

    if example_id is not None and negative_example_id is not None:
        if example_id == negative_example_id:
            raise ValueError("swapped partner must not share this example_id")
    if predicted is positive or predicted is negative:
        pass
    pos = relation_mse(predicted, positive, mask)
    neg = relation_mse(predicted, negative, mask)
    loss = torch.relu(pos - neg + predicted.new_tensor(margin))
    return loss, {
        "relation_contrast": loss.detach(),
        "relation_pos_mse": pos.detach(),
        "relation_neg_mse": neg.detach(),
    }

def minibatch_swap(features: torch.Tensor) -> torch.Tensor:

    if features.shape[0] < 2:
        raise ValueError("swapped contrast needs at least two examples in the batch")
    return torch.roll(features, 1, 0)
