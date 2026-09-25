
from __future__ import annotations

import torch

N_RESIDUE_TYPES = 20

CONTINUOUS_FIELDS = ("bb_ca", "local_latents")

class BridgeError(RuntimeError):
    pass

def linear_schedule(tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    return tau, torch.ones_like(tau)

def _check(source: dict, native: dict, tau: torch.Tensor) -> None:
    if torch.is_tensor(tau) and (float(tau.min()) < 0.0 or float(tau.max()) > 1.0):
        raise BridgeError(
            f"tau must lie in [0, 1]; got [{float(tau.min())}, {float(tau.max())}]")
    for field in (*CONTINUOUS_FIELDS, "residue_type"):
        if field not in source or field not in native:
            raise BridgeError(f"both states must carry {field!r}")
        if source[field].shape != native[field].shape:
            raise BridgeError(
                f"shape disagreement on {field!r}: source "
                f"{tuple(source[field].shape)} against native "
                f"{tuple(native[field].shape)}. Designs vary in residue count by "
                f"seed, and pairing across a mismatch aligns residue i of one "
                f"protein with residue i of a different one")

def _broadcast(tau: torch.Tensor, like: torch.Tensor) -> torch.Tensor:

    return tau.reshape(-1, *([1] * (like.dim() - 1))).to(like.dtype)

def _one_hot(index: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        index.long(), num_classes=N_RESIDUE_TYPES).to(torch.float32)

def bridge_state(source: dict, native: dict, tau: torch.Tensor) -> dict:

    _check(source, native, tau)
    alpha, _ = linear_schedule(tau)
    state: dict = {}
    for field in CONTINUOUS_FIELDS:
        a = _broadcast(alpha, source[field])
        state[field] = (1.0 - a) * source[field] + a * native[field]
    a = _broadcast(alpha, source["residue_type"].unsqueeze(-1).float())
    state["residue_type_probs"] = (
        (1.0 - a) * _one_hot(source["residue_type"])
        + a * _one_hot(native["residue_type"])
    )
    return state

def bridge_target(source: dict, native: dict, tau: torch.Tensor) -> dict:

    _check(source, native, tau)
    _, alpha_prime = linear_schedule(tau)
    target: dict = {}
    for field in CONTINUOUS_FIELDS:
        a = _broadcast(alpha_prime, source[field])
        target[field] = a * (native[field] - source[field])
    a = _broadcast(alpha_prime, source["residue_type"].unsqueeze(-1).float())
    target["residue_type_probs"] = a * (
        _one_hot(native["residue_type"]) - _one_hot(source["residue_type"]))
    return target

def assert_condition_is_stationary(
    source: dict, native: dict, fixed: torch.Tensor, *, atol: float = 1e-6
) -> None:

    for field in CONTINUOUS_FIELDS:
        if field not in source:
            continue
        rows = fixed
        while rows.dim() < source[field].dim():
            rows = rows.unsqueeze(-1)
        rows = rows.expand_as(source[field])
        gap = (source[field][rows] - native[field][rows]).abs()
        if gap.numel() and float(gap.max()) > atol:
            raise BridgeError(
                f"the supplied condition differs between source and native on "
                f"{field!r} by up to {float(gap.max()):.4g}; the source did not "
                f"honour its condition, and bridging would move a fixed term")
    differing = int((source["residue_type"][fixed] != native["residue_type"][fixed]).sum())
    if differing:
        raise BridgeError(
            f"{differing} fixed residues carry a different identity in the "
            f"source than in the native; the source did not honour its condition")
