
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

_MODALITIES: tuple[str, ...] = ("bb_ca", "local_latents")

class GateError(RuntimeError):
    pass

def apply_gate(joint: Tensor, self_only: Tensor, gate: float) -> Tensor:

    if joint.shape != self_only.shape:
        raise GateError(
            f"shape mismatch: {tuple(joint.shape)} != {tuple(self_only.shape)}"
        )
    if not (torch.isfinite(joint).all() and torch.isfinite(self_only).all()):
        raise GateError("non-finite value in a prediction")
    if float(gate) == 1.0:

        return joint
    if not math.isfinite(gate):
        raise GateError(f"non-finite gate: {gate}")
    return self_only + float(gate) * (joint - self_only)

def gated_nn_out(
    nn_out: Mapping,
    self_only: Mapping[str, Tensor],
    gates: Mapping[str, float],
    output_parameterization: Mapping[str, str],
) -> dict:

    gated: dict[str, dict[str, Tensor]] = {}
    for modality in _MODALITIES:
        if modality not in gates:
            raise GateError(f"no gate supplied for '{modality}'")
        if modality not in nn_out:
            raise GateError(f"nn output is missing modality '{modality}'")
        if modality not in self_only:
            raise GateError(f"no self-only prediction for '{modality}'")

        key = output_parameterization[modality]
        entry = nn_out[modality]
        if key not in entry:
            raise GateError(
                f"nn output for '{modality}' lacks configured key '{key}', "
                f"observed {sorted(entry)}"
            )
        gated[modality] = {
            key: apply_gate(entry[key], self_only[modality], gates[modality])
        }

    for modality, entry in nn_out.items():
        if modality not in gated:
            gated[modality] = dict(entry)
    return gated
