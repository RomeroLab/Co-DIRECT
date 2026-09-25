
from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import Generator

from dive.counterfactual.builder import build_conditions
from dive.counterfactual.registry import FeatureRegistry
from dive.counterfactual.runner import run_conditions
from dive.probe.record import StepRecord

_ATOM37 = 37
_CA_INDEX = 1

class ProbeError(RuntimeError):
    pass

def wrap_predict_for_sampling(
    real_fn: Callable[..., Mapping],
    *,
    registry: FeatureRegistry,
    extract_velocities: Callable[[Mapping], dict],
    recompute_pair_features: Callable[[dict], dict],
    recorder,
    generator: Generator,
    target: str,
    seed: int,
) -> Callable[..., Mapping]:

    state = {"step": 0}

    def probed(batch: dict, mode: str = "full", n_recycle: int = 0) -> Mapping:
        joint_output = real_fn(batch, mode=mode, n_recycle=n_recycle)

        if mode != "full":
            return joint_output

        conditions = build_conditions(
            batch, registry, recompute_pair_features, generator=generator
        )

        def denoiser(sibling: dict) -> Mapping:
            return real_fn(sibling, mode="full", n_recycle=n_recycle)

        predictions = run_conditions(denoiser, conditions, joint_output, extract_velocities)

        t_bb_ca, t_local_latents = _modality_times(batch)
        recorder.append(
            StepRecord(
                target=target,
                seed=seed,
                step=state["step"],
                t_bb_ca=t_bb_ca,
                t_local_latents=t_local_latents,
                valid=batch["mask"].detach().cpu(),
                a_x=predictions.v_x_self_zero.detach().cpu(),
                a_z=predictions.v_z_self_zero.detach().cpu(),
                b_zero_z_to_x=predictions.b_zero_z_to_x.detach().cpu(),
                b_zero_x_to_z=predictions.b_zero_x_to_z.detach().cpu(),
                b_sham_z_to_x=predictions.b_sham_z_to_x.detach().cpu(),
                b_sham_x_to_z=predictions.b_sham_x_to_z.detach().cpu(),
                x_t_bb_ca=batch["x_t"]["bb_ca"].detach().cpu(),
                x_target=_target_ca(batch),
            )
        )
        state["step"] += 1
        return joint_output

    return probed

def _target_ca(batch: Mapping) -> torch.Tensor:

    value = batch.get("x_target")
    if not isinstance(value, torch.Tensor):
        raise ProbeError("batch does not carry a readable 'x_target' tensor")
    if value.dim() == 4:
        if value.shape[-2] != _ATOM37 or value.shape[-1] != 3:
            raise ProbeError(f"unreadable 'x_target' shape {tuple(value.shape)}")
        return value[:, :, _CA_INDEX, :].detach().cpu()
    if value.dim() == 3 and value.shape[-1] == 3:
        return value.detach().cpu()
    raise ProbeError(f"unreadable 'x_target' shape {tuple(value.shape)}")

def _modality_times(batch: Mapping) -> tuple[float, float]:

    time = batch.get("t")
    if isinstance(time, Mapping):
        return _scalar(time, "bb_ca"), _scalar(time, "local_latents")
    value = _scalar_value(time)
    return value, value

def _scalar(time: Mapping, modality: str) -> float:

    if modality not in time:
        raise ProbeError(f"batch's per-modality 't' is missing '{modality}'")
    return _scalar_value(time[modality])

def _scalar_value(time: object) -> float:

    if isinstance(time, torch.Tensor):
        return float(time.flatten()[0].item())
    if isinstance(time, (int, float)):
        return float(time)
    raise ProbeError("batch does not carry a readable corruption time 't'")
