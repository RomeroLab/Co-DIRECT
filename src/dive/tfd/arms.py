
from __future__ import annotations

from typing import Mapping

import torch

MODE_X = "bb_ca"

MODE_Z = "local_latents"

ARMS: tuple[str, ...] = (
    "released",
    "full_refresh",
    "x_to_z_fresh",
    "x_to_z_stale",
    "z_to_x_fresh",
    "z_to_x_stale",
)

_REFRESH: dict[str, tuple[str, ...]] = {
    "full_refresh": (MODE_X, MODE_Z),
    "x_to_z_fresh": (MODE_X,),
    "x_to_z_stale": (),
    "z_to_x_fresh": (MODE_Z,),
    "z_to_x_stale": (),
}

_ADOPT: dict[str, dict[str, int]] = {
    "released": {MODE_X: 1, MODE_Z: 1},
    "full_refresh": {MODE_X: 2, MODE_Z: 2},
    "x_to_z_fresh": {MODE_X: 1, MODE_Z: 2},
    "x_to_z_stale": {MODE_X: 1, MODE_Z: 2},
    "z_to_x_fresh": {MODE_X: 2, MODE_Z: 1},
    "z_to_x_stale": {MODE_X: 2, MODE_Z: 1},
}

def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

def adopt_from(arm: str) -> dict[str, int]:

    _check_arm(arm)
    return dict(_ADOPT[arm])

def second_call_self_conditioning(
    arm: str,
    sc_old: Mapping[str, torch.Tensor],
    x1_pred: Mapping[str, torch.Tensor],
    *,
    region_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:

    _check_arm(arm)
    if arm == "released":
        return None
    missing = [m for m in (MODE_X, MODE_Z) if m not in sc_old or m not in x1_pred]
    if missing:
        raise KeyError(f"self-conditioning is missing modalities {missing}")
    if region_mask is not None:
        expected = tuple(sc_old[MODE_X].shape[:2])
        if tuple(region_mask.shape) != expected:
            raise ValueError(
                f"region_mask {tuple(region_mask.shape)} does not match state {expected}"
            )
    refresh = _REFRESH[arm]
    out: dict[str, torch.Tensor] = {}
    for mode in (MODE_X, MODE_Z):
        old = sc_old[mode]
        if mode not in refresh:
            out[mode] = old.clone()
            continue
        new = x1_pred[mode]
        if region_mask is None:
            out[mode] = new.clone()
        else:
            broadcast = region_mask.reshape(
                region_mask.shape + (1,) * (new.dim() - 2)
            ).to(new.device)
            out[mode] = torch.where(broadcast, new, old).clone()
    return out

def merge_nn_out(
    arm: str,
    first: Mapping[str, dict],
    second: Mapping[str, dict] | None,
) -> Mapping[str, dict]:

    _check_arm(arm)
    table = _ADOPT[arm]
    if second is None:
        if any(v == 2 for v in table.values()):
            raise ValueError(f"arm {arm!r} adopts from the second pass, which was not made")
        return first
    out: dict[str, dict] = {}
    for mode, which in table.items():
        source = first if which == 1 else second
        if mode not in source:
            raise KeyError(f"pass {which} produced no output for modality {mode!r}")
        out[mode] = source[mode]
    for mode in set(first) | set(second):
        if mode not in out:
            out[mode] = first[mode]
    return out
