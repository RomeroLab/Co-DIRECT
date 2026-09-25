
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

WRAPPER_KEYS = frozenset({"nn", "step", "args"})

class TrunkFormatError(RuntimeError):
    pass

def save_trunk(module, *, step: int, args: Mapping[str, Any], path: Path) -> Path:

    import torch

    state = module.state_dict()
    if not state:
        raise TrunkFormatError(
            "refusing to save an empty state dict; an empty trunk loads "
            "without complaint under a loose load and is indistinguishable "
            "from a clean save"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    staging = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {"nn": {k: v.detach().cpu() for k, v in state.items()},
         "step": int(step),
         "args": dict(args)},
        staging,
    )
    staging.replace(path)
    return path

def read_trunk(path: Path) -> dict:

    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TrunkFormatError(f"trunk must be a dict wrapper, got {type(payload)!r}")
    keys = set(payload)
    if keys != WRAPPER_KEYS:
        missing = sorted(WRAPPER_KEYS - keys)
        extra = sorted(keys - WRAPPER_KEYS)
        raise TrunkFormatError(
            f"trunk wrapper keys are {sorted(keys)}; expected "
            f"{sorted(WRAPPER_KEYS)} (missing={missing} unexpected={extra}). "
            "A bare state dict is not a trunk file."
        )
    if not isinstance(payload["nn"], Mapping) or not payload["nn"]:
        raise TrunkFormatError("trunk carries no weights under 'nn'")
    return payload

def load_trunk_into(module, path: Path) -> dict:

    payload = read_trunk(path)
    state = payload["nn"]
    destination = module.state_dict()

    missing = sorted(set(destination) - set(state))
    unexpected = sorted(set(state) - set(destination))
    mismatched = sorted(
        f"{key}: file {tuple(state[key].shape)} vs graph {tuple(destination[key].shape)}"
        for key in set(state) & set(destination)
        if tuple(state[key].shape) != tuple(destination[key].shape)
    )
    if missing or unexpected or mismatched:
        raise TrunkFormatError(
            "trunk does not match this graph key-for-key; refusing a partial "
            "load.\n"
            f"  missing from file ({len(missing)}): {missing[:6]}\n"
            f"  not in graph ({len(unexpected)}): {unexpected[:6]}\n"
            f"  shape mismatches ({len(mismatched)}): {mismatched[:6]}"
        )
    module.load_state_dict(state, strict=True)
    return payload

def hash_module_parameters(module) -> str:

    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().contiguous()
        digest.update(repr(tuple(array.shape)).encode())
        digest.update(bytes(array.numpy()))
    return digest.hexdigest()
