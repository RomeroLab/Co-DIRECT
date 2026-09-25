
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from dive.counterfactual.builder import CONDITION_NAMES, Conditions, tree_sha256

_COUNTERFACTUAL_ORDER: tuple[str, ...] = (
    "without_latent",
    "without_backbone",
    "sham_latent",
    "sham_backbone",
)
_MODALITIES: tuple[str, ...] = ("bb_ca", "local_latents")

class RunnerError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class DirectionalPredictions:

    v_x_joint: Tensor
    v_z_joint: Tensor
    v_x_self_zero: Tensor
    v_z_self_zero: Tensor
    v_x_self_sham: Tensor
    v_z_self_sham: Tensor
    b_zero_z_to_x: Tensor
    b_zero_x_to_z: Tensor
    b_sham_z_to_x: Tensor
    b_sham_x_to_z: Tensor
    calls: int

def run_conditions(
    denoiser: Callable[[dict], Mapping],
    conditions: Conditions,
    joint_output: Mapping,
    extract_velocities: Callable[[Mapping], dict[str, Tensor]],
) -> DirectionalPredictions:

    joint = _checked(_extract(extract_velocities, joint_output, "joint"), "joint", reference=None)

    outputs: dict[str, dict[str, Tensor]] = {}
    calls = 0
    for name in _COUNTERFACTUAL_ORDER:
        if name not in conditions.batches:
            raise RunnerError(f"condition set is missing '{name}'")
        batch = conditions.batches[name]
        before = tree_sha256(batch)
        raw = denoiser(batch)
        calls += 1
        if tree_sha256(batch) != before:
            raise RunnerError(f"denoiser mutated the '{name}' condition batch")
        outputs[name] = _checked(_extract(extract_velocities, raw, name), name, reference=joint)

    expected_calls = len(CONDITION_NAMES) - 1
    if calls != expected_calls:
        raise RunnerError(f"expected {expected_calls} denoiser calls, observed {calls}")

    v_x_joint, v_z_joint = joint["bb_ca"], joint["local_latents"]
    v_x_self_zero = outputs["without_latent"]["bb_ca"]
    v_z_self_zero = outputs["without_backbone"]["local_latents"]
    v_x_self_sham = outputs["sham_latent"]["bb_ca"]
    v_z_self_sham = outputs["sham_backbone"]["local_latents"]

    return DirectionalPredictions(
        v_x_joint=v_x_joint,
        v_z_joint=v_z_joint,
        v_x_self_zero=v_x_self_zero,
        v_z_self_zero=v_z_self_zero,
        v_x_self_sham=v_x_self_sham,
        v_z_self_sham=v_z_self_sham,
        b_zero_z_to_x=v_x_joint - v_x_self_zero,
        b_zero_x_to_z=v_z_joint - v_z_self_zero,
        b_sham_z_to_x=v_x_joint - v_x_self_sham,
        b_sham_x_to_z=v_z_joint - v_z_self_sham,
        calls=calls,
    )

def _extract(
    extract_velocities: Callable[[Mapping], dict[str, Tensor]], nn_out: Mapping, name: str
) -> dict[str, Tensor]:

    try:
        return extract_velocities(nn_out)
    except (KeyError, IndexError, TypeError) as exc:
        raise RunnerError(f"'{name}' output could not be parsed: {exc}") from exc

def _checked(
    velocities: Mapping[str, Tensor], name: str, reference: Mapping[str, Tensor] | None
) -> dict[str, Tensor]:

    missing = [modality for modality in _MODALITIES if modality not in velocities]
    if missing:
        raise RunnerError(f"'{name}' output is missing {', '.join(missing)}")
    checked: dict[str, Tensor] = {}
    for modality in _MODALITIES:
        tensor = velocities[modality]
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
            raise RunnerError(f"'{name}' {modality} is not a floating tensor")
        if not torch.isfinite(tensor).all():
            raise RunnerError(f"'{name}' {modality} contains non-finite values")
        if reference is not None:
            expected = reference[modality]
            if tensor.shape != expected.shape:
                raise RunnerError(
                    f"'{name}' {modality} shape {tuple(tensor.shape)} "
                    f"!= joint {tuple(expected.shape)}"
                )
            if tensor.dtype != expected.dtype or tensor.device != expected.device:
                raise RunnerError(f"'{name}' {modality} dtype/device differs from joint")
        checked[modality] = tensor
    return checked
