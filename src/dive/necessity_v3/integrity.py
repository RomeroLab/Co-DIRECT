
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

_CHUNK = 1 << 20

class ParametersChanged(RuntimeError):
    pass

class NotFrozen(RuntimeError):
    pass

def tensor_hash(tensor: torch.Tensor) -> str:

    detached = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode())
    digest.update(str(tuple(detached.shape)).encode())
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()

def parameter_hashes(
    model: nn.Module, *, include_buffers: bool = False
) -> dict[str, str]:

    hashes = {name: tensor_hash(p) for name, p in model.named_parameters()}
    if include_buffers:
        hashes.update({name: tensor_hash(b) for name, b in model.named_buffers()})
    return hashes

def assert_parameters_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> None:

    if not before or not after:
        raise ParametersChanged(
            "refusing a vacuous integrity check: one side of the comparison "
            f"is empty (before={len(before)}, after={len(after)})"
        )
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    if missing or added or changed:
        raise ParametersChanged(
            "frozen-parameter integrity failed. "
            f"changed={changed} missing={missing} added={added}"
        )

def assert_frozen(model: nn.Module) -> None:

    training = [n or "<root>" for n, m in model.named_modules() if m.training]
    if training:
        raise NotFrozen(
            f"model must be in eval mode; still training: {training[:5]}"
        )
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    if with_grad:
        raise NotFrozen(
            f"parameters carry a populated gradient: {with_grad[:5]}"
        )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable:
        raise NotFrozen(
            f"parameters still have requires_grad set: {trainable[:5]}"
        )

def sha256_file(path: str | Path) -> str:

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
