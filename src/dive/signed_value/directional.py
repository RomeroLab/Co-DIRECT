
from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from dive.signed_value.basis import BasisResult

class DirectionalError(RuntimeError):
    pass

Decoder = Callable[..., tuple[Tensor, Tensor]]

@dataclass(frozen=True, slots=True)
class DirectionalState:

    x_t: Mapping[str, Tensor]
    joint_velocities: Mapping[str, Tensor]
    t_x: float
    t_z: float
    mask: Tensor

@dataclass(frozen=True, slots=True)
class DirectionalBundle:

    x1: Tensor
    z1: Tensor
    joint_logits: Tensor
    joint_atom37: Tensor
    h0: tuple[float, float]
    c_z: tuple[Tensor, Tensor]
    perturbed_logits: Mapping[str, Tensor]
    tangents: Mapping[str, Tensor]
    tangent_metrics: Mapping[str, float]
    decoder_calls: int

LADDER = (0.5, 1.0, 2.0, 4.0)

DECODER_CALLS = 1 + 2 * len(LADDER) * 2

def scale_token(scale: float) -> str:

    if scale <= 0.0 or not math.isfinite(scale):
        raise DirectionalError(f"ladder rung must be finite and positive: {scale!r}")
    if float(scale).is_integer():
        return f"scale{int(scale)}"
    return "scale" + f"{scale:g}".replace(".", "p")

def masked_values(value: Tensor, mask: Tensor) -> Tensor:

    _require_floating_tensor(value, "value")
    if (
        not isinstance(mask, Tensor)
        or mask.dtype is not torch.bool
        or mask.device != value.device
    ):
        raise DirectionalError(
            "masked operation requires a non-empty boolean mask on the value device"
        )
    _require_boolean_mask(mask)
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    try:
        expanded = expanded.expand_as(value)
    except RuntimeError as error:
        raise DirectionalError("mask cannot be expanded to the value shape") from error
    if expanded.shape != value.shape:
        raise DirectionalError("mask cannot be expanded to the value shape")
    return value[expanded]

def masked_rms(value: Tensor, mask: Tensor) -> float:

    selected = masked_values(value, mask)
    result = torch.sqrt(torch.mean(selected.to(torch.float64).square())).item()
    if not math.isfinite(result):
        raise DirectionalError("masked_rms produced a non-finite value")
    return result

def clean_endpoints(
    x_t: Tensor,
    z_t: Tensor,
    v_x: Tensor,
    v_z: Tensor,
    *,
    t_x: float,
    t_z: float,
) -> tuple[Tensor, Tensor]:

    _require_matching_tensors(x_t, v_x, "x_t", "v_x")
    _require_matching_tensors(z_t, v_z, "z_t", "v_z")
    resolved_t_x = _require_unit_time(t_x)
    resolved_t_z = _require_unit_time(t_z)
    x1 = x_t + (1.0 - resolved_t_x) * v_x
    z1 = z_t + (1.0 - resolved_t_z) * v_z
    if not bool(torch.isfinite(x1).all() and torch.isfinite(z1).all()):
        raise DirectionalError("clean endpoint is non-finite")
    return x1, z1

def finite_difference_h0(z1: Tensor, c_z: Tensor, mask: Tensor) -> float:

    _require_matching_tensors(z1, c_z, "z1", "c_z")
    result = max(1.0, masked_rms(z1, mask)) / (
        256.0 * max(masked_rms(c_z, mask), 1e-12)
    )
    if not math.isfinite(result) or result <= 0.0:
        raise DirectionalError("finite-difference h0 is non-finite")
    return result

def tangent(plus: Tensor, minus: Tensor, h: float) -> Tensor:

    _require_matching_tensors(plus, minus, "plus", "minus")
    step = _require_positive_finite(h, "h")
    result = (plus - minus) / (2.0 * step)
    if not bool(torch.isfinite(result).all()):
        raise DirectionalError("tangent is non-finite")
    return result

def cosine_and_relative_norm(
    left: Tensor, right: Tensor, mask: Tensor
) -> tuple[float, float]:

    _require_matching_tensors(left, right, "left tangent", "right tangent")
    left_flat = masked_values(left, mask).to(torch.float64)
    right_flat = masked_values(right, mask).to(torch.float64)
    left_norm = torch.linalg.vector_norm(left_flat).item()
    right_norm = torch.linalg.vector_norm(right_flat).item()
    if not math.isfinite(left_norm) or not math.isfinite(right_norm):
        raise DirectionalError("tangent norm is non-finite")
    if left_norm == 0.0 or right_norm == 0.0:
        raise DirectionalError("tangent norm is zero")
    cosine = torch.dot(left_flat, right_flat).item() / (left_norm * right_norm)
    norm_change = abs(left_norm - right_norm) / max(left_norm, right_norm)
    if not math.isfinite(cosine) or not math.isfinite(norm_change):
        raise DirectionalError("tangent comparison is non-finite")
    return cosine, norm_change

def symmetric_relative_error(left: float, right: float) -> float:

    resolved_left = _require_finite_number(left, "left")
    resolved_right = _require_finite_number(right, "right")
    result = abs(resolved_left - resolved_right) / max(
        abs(resolved_left), abs(resolved_right), 1e-6
    )
    if not math.isfinite(result):
        raise DirectionalError("symmetric relative error is non-finite")
    return result

def build_directional_bundle(
    decoder: Decoder, state: DirectionalState, basis: BasisResult
) -> DirectionalBundle:

    if not callable(decoder):
        raise DirectionalError("decoder must be callable")
    if not isinstance(state, DirectionalState):
        raise DirectionalError("state must be a DirectionalState")
    x_t, z_t, v_x, v_z = _validated_state_tensors(state)
    if getattr(basis, "active", None) is not True:
        raise DirectionalError("directional bundle refuses an inactive basis")
    bases = _validated_bases(basis, z_t)
    x1, z1 = clean_endpoints(x_t, z_t, v_x, v_z, t_x=state.t_x, t_z=state.t_z)
    t_z = _require_unit_time(state.t_z)
    c_z = tuple((1.0 - t_z) * item for item in bases)
    if not all(bool(torch.isfinite(item).all()) for item in c_z):
        raise DirectionalError("decoder direction is non-finite")
    h0 = tuple(finite_difference_h0(z1, item, state.mask) for item in c_z)

    joint_logits, joint_atom37 = _decode(decoder, z1, x1, state.mask, "joint")
    perturbed_logits: dict[str, Tensor] = {}
    tangent_pairs: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}
    tangent_records: dict[str, Tensor] = {}
    decoder_calls = 1
    for prior, (direction, step) in enumerate(zip(c_z, h0, strict=True)):
        for scale in LADDER:
            token = scale_token(scale)
            scaled_step = scale * step
            minus_logits, minus_atom37 = _decode(
                decoder,
                z1 - scaled_step * direction,
                x1,
                state.mask,
                f"prior{prior}.{token}.minus",
            )
            plus_logits, plus_atom37 = _decode(
                decoder,
                z1 + scaled_step * direction,
                x1,
                state.mask,
                f"prior{prior}.{token}.plus",
            )
            decoder_calls += 2
            perturbed_logits[f"prior{prior}.{token}.minus"] = minus_logits
            perturbed_logits[f"prior{prior}.{token}.plus"] = plus_logits
            logit_tangent = tangent(plus_logits, minus_logits, scaled_step)

            atom37_tangent = tangent(plus_atom37, minus_atom37, scaled_step)
            tangent_pairs[(prior, scale)] = (logit_tangent, atom37_tangent)
            tangent_records[f"prior{prior}.{token}.logits"] = logit_tangent
            tangent_records[f"prior{prior}.{token}.atom37"] = atom37_tangent

    if decoder_calls != DECODER_CALLS:
        raise DirectionalError("directional decoder call ledger drifted")
    metrics: dict[str, float] = {}
    for prior in (0, 1):
        for output_index, output_name in enumerate(("logits", "atom37")):
            at_h0 = tangent_pairs[(prior, 1.0)][output_index]
            at_2h0 = tangent_pairs[(prior, 2.0)][output_index]
            cosine, relative_norm_change = cosine_and_relative_norm(
                at_h0, at_2h0, state.mask
            )
            metrics[f"prior{prior}.{output_name}.cosine"] = cosine
            metrics[f"prior{prior}.{output_name}.relative_norm_change"] = (
                relative_norm_change
            )
    return DirectionalBundle(
        x1=x1,
        z1=z1,
        joint_logits=joint_logits,
        joint_atom37=joint_atom37,
        h0=(h0[0], h0[1]),
        c_z=(c_z[0], c_z[1]),
        perturbed_logits=perturbed_logits,
        tangents=tangent_records,
        tangent_metrics=metrics,
        decoder_calls=decoder_calls,
    )

def _decode(
    decoder: Decoder, z_latent: Tensor, ca_coors_nm: Tensor, mask: Tensor, label: str
) -> tuple[Tensor, Tensor]:
    try:
        result = decoder(z_latent=z_latent, ca_coors_nm=ca_coors_nm, mask=mask)
    except Exception as error:
        raise DirectionalError(f"{label} decoder call failed: {error}") from error
    if not isinstance(result, tuple) or len(result) != 2:
        raise DirectionalError(f"{label} decoder must return (logits, atom37)")
    logits, atom37 = result
    _require_floating_tensor(logits, f"{label} logits")
    _require_floating_tensor(atom37, f"{label} atom37")
    length = z_latent.shape[1]
    if logits.shape != (1, length, 20):
        raise DirectionalError(f"{label} decoder logits shape drift")
    if atom37.shape != (1, length, 37, 3):
        raise DirectionalError(f"{label} decoder atom37 shape drift")
    if logits.device != z_latent.device or atom37.device != z_latent.device:
        raise DirectionalError(f"{label} decoder device drift")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(atom37).all()):
        raise DirectionalError(f"{label} decoder output is non-finite")
    return logits, atom37

def _validated_state_tensors(
    state: DirectionalState,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if set(state.x_t) != {"bb_ca", "local_latents"}:
        raise DirectionalError("state x_t must carry exactly both modalities")
    if set(state.joint_velocities) != {"bb_ca", "local_latents"}:
        raise DirectionalError("joint velocities must carry exactly both modalities")
    x_t = state.x_t["bb_ca"]
    z_t = state.x_t["local_latents"]
    v_x = state.joint_velocities["bb_ca"]
    v_z = state.joint_velocities["local_latents"]
    _require_matching_tensors(x_t, v_x, "x_t.bb_ca", "joint bb_ca velocity")
    _require_matching_tensors(
        z_t, v_z, "x_t.local_latents", "joint local_latents velocity"
    )
    if x_t.ndim != 3 or x_t.shape[0] != 1 or x_t.shape[-1] != 3:
        raise DirectionalError("x_t.bb_ca must have shape [1, L, 3]")
    if z_t.ndim != 3 or z_t.shape[0] != 1 or z_t.shape[:2] != x_t.shape[:2]:
        raise DirectionalError("x_t.local_latents must have shape [1, L, D]")
    _require_boolean_mask(state.mask)
    if state.mask.shape != x_t.shape[:2] or state.mask.device != x_t.device:
        raise DirectionalError("binder mask must match the endpoint batch and length")
    return x_t, z_t, v_x, v_z

def _validated_bases(basis: BasisResult, z_t: Tensor) -> tuple[Tensor, Tensor]:
    values = getattr(basis, "bases", None)
    if not isinstance(values, tuple) or len(values) != 2:
        raise DirectionalError("basis must carry exactly two prior directions")
    for index, value in enumerate(values):
        _require_matching_tensors(z_t, value, "z_t", f"basis {index}")
    return values

def _require_boolean_mask(mask: Tensor) -> None:
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
        raise DirectionalError("masked operation requires a non-empty boolean mask")
    if mask.numel() == 0 or not bool(mask.any()):
        raise DirectionalError("masked operation requires a non-empty boolean mask")

def _require_floating_tensor(value: Tensor, label: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise DirectionalError(f"{label} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise DirectionalError(f"{label} is non-finite")

def _require_matching_tensors(
    left: Tensor, right: Tensor, left_label: str, right_label: str
) -> None:
    if not isinstance(left, Tensor) or not left.is_floating_point():
        raise DirectionalError(f"{left_label} must be a floating tensor")
    if not isinstance(right, Tensor) or not right.is_floating_point():
        raise DirectionalError(f"{right_label} must be a floating tensor")
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
    ):
        raise DirectionalError(
            f"{left_label} and {right_label} shape/dtype/device mismatch"
        )
    _require_floating_tensor(left, left_label)
    _require_floating_tensor(right, right_label)

def _require_unit_time(value: float) -> float:
    resolved = _require_finite_number(value, "modality time")
    if not 0.0 <= resolved <= 1.0:
        raise DirectionalError("modality times must be finite values in [0, 1]")
    return resolved

def _require_positive_finite(value: float, label: str) -> float:
    resolved = _require_finite_number(value, label)
    if resolved <= 0.0:
        raise DirectionalError(f"{label} must be a positive finite number")
    return resolved

def _require_finite_number(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise DirectionalError(f"{label} must be a finite numeric value")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise DirectionalError(f"{label} must be a finite numeric value")
    return resolved
