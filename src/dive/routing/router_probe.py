
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Generator

from dive.counterfactual.builder import build_conditions
from dive.counterfactual.registry import FeatureRegistry
from dive.routing.gate import gated_nn_out
from dive.routing.schedule import GateSchedule

_SELF_ONLY_CONDITION: Mapping[str, str] = {
    "bb_ca": "without_latent",
    "local_latents": "without_backbone",
}
_GATE_DIRECTION: Mapping[str, str] = {
    "bb_ca": "z_to_x",
    "local_latents": "x_to_z",
}

class RouterError(RuntimeError):
    pass

def wrap_predict_for_sampling_routing(
    real_fn: Callable[..., Mapping],
    *,
    registry: FeatureRegistry,
    extract_velocities: Callable[[Mapping], dict],
    recompute_pair_features: Callable[[dict], dict],
    schedules: Mapping[str, GateSchedule],
    output_parameterization: Mapping[str, str],
    generator: Generator,
) -> Callable[..., Mapping]:

    for direction in ("z_to_x", "x_to_z"):
        if direction not in schedules:
            raise RouterError(f"no schedule supplied for direction '{direction}'")

    def routed(batch: dict, mode: str = "full", n_recycle: int = 0) -> Mapping:
        joint_output = real_fn(batch, mode=mode, n_recycle=n_recycle)
        if mode != "full":
            return joint_output

        time = _scalar_time(batch)
        gates = {
            modality: schedules[_GATE_DIRECTION[modality]].gate_at(time)
            for modality in _SELF_ONLY_CONDITION
        }

        conditions = build_conditions(
            batch, registry, recompute_pair_features, generator=generator
        )

        self_only: dict[str, torch.Tensor] = {}
        for modality, condition in _SELF_ONLY_CONDITION.items():
            if condition not in conditions.batches:
                raise RouterError(f"condition set is missing '{condition}'")
            raw = real_fn(conditions.batches[condition], mode="full", n_recycle=n_recycle)
            velocities = extract_velocities(raw)
            if modality not in velocities:
                raise RouterError(f"'{condition}' output is missing '{modality}'")
            self_only[modality] = velocities[modality]

        return gated_nn_out(joint_output, self_only, gates, output_parameterization)

    return routed

def _scalar_time(batch: Mapping) -> float:

    time = batch.get("t")
    if isinstance(time, Mapping):
        time = time.get("bb_ca")
    if isinstance(time, torch.Tensor):
        return float(time.flatten()[0].item())
    if isinstance(time, (int, float)):
        return float(time)
    raise RouterError("batch does not carry a readable corruption time 't'")
