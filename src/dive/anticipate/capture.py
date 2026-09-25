
from __future__ import annotations

from contextlib import contextmanager

import torch

@contextmanager
def capture_last_forward(model):

    original = model.call_nn
    had_instance_binding = "call_nn" in model.__dict__
    held: dict = {}

    def call(batch, *args, **kwargs):
        out = original(batch, *args, **kwargs)
        held["batch"] = batch
        held["out"] = out
        return out

    model.call_nn = call
    try:
        yield held
    finally:
        if had_instance_binding:
            model.call_nn = original
        else:
            del model.call_nn

def designed_from_batch(batch) -> torch.Tensor:

    mask = batch["mask"].bool()
    motif = batch.get("motif_mask")
    if motif is None:
        return mask
    if motif.shape[:2] != mask.shape:
        return mask
    conditioned = motif.bool()
    while conditioned.dim() > mask.dim():
        conditioned = conditioned.any(dim=-1)
    return mask & ~conditioned
