
from __future__ import annotations

import torch
from torch import Tensor

RESTYPES = "ARNDCQEGHILKMFPSTWYV"
GLYCINE = RESTYPES.index("G")
ALANINE = RESTYPES.index("A")
DECLINE_CODES = (GLYCINE, ALANINE)

def decline_probability(seq_logits: Tensor, designed: Tensor) -> Tensor | None:

    if seq_logits.dim() != 3:
        raise ValueError(f"seq_logits must be [B, N, 20], got {tuple(seq_logits.shape)}")
    if designed.shape != seq_logits.shape[:2]:
        raise ValueError(
            f"designed must be [B, N] matching the logits, got "
            f"{tuple(designed.shape)} against {tuple(seq_logits.shape[:2])}")

    keep = designed.bool()
    if not bool(keep.any()):
        return None

    probabilities = seq_logits.softmax(dim=-1)
    refusal = probabilities[..., list(DECLINE_CODES)].sum(dim=-1)
    weight = keep.to(refusal.dtype)
    return (refusal * weight).sum() / weight.sum()
