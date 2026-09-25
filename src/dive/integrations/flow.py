
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import Tensor, nn

from dive.integrations.directional_v2 import DirectionalForward
from dive.leadership.capture import capture_module_output
from dive.leadership.directional_bridge import RouteState

class FlowBindingError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ProteinaBindings:

    call_denoiser: Callable[[nn.Module, dict], Mapping]
    corrupt: Callable[[dict], dict]
    flow_loss: Callable[[Tensor, Tensor, Mapping], Tensor]
    per_residue_flow_loss: Callable[[Tensor, Tensor, Mapping], Tensor]
    output_key: str
    trunk_layers: int

    per_modality_residue_flow_loss: Callable[
        [Tensor, Tensor, Mapping], "dict[str, Tensor]"
    ] | None = None

@dataclass(frozen=True, slots=True)
class DirectionalProteinaBindings:

    corrupt: Callable[[dict], dict]
    call_route: Callable[[nn.Module, dict, RouteState], DirectionalForward]
    flow_loss: Callable[[Tensor, Tensor, Mapping], Tensor]
    per_residue_flow_loss: Callable[[Tensor, Tensor, Mapping], Tensor]
    output_key: str

    per_modality_residue_flow_loss: Callable[
        [Tensor, Tensor, Mapping], "dict[str, Tensor]"
    ] | None = None

def proteina_bindings(model, *, n_recycle: int = 0) -> ProteinaBindings:

    if n_recycle != 0:
        raise FlowBindingError(
            "recycling calls the trunk more than once, and the hidden-state "
            "capture refuses that rather than silently taking the last call"
        )

    layers = getattr(getattr(model, "nn", None), "transformer_layers", None)
    if layers is None or len(layers) == 0:
        raise FlowBindingError("model.nn has no transformer_layers to capture from")
    trunk_final = layers[-1]

    (output_key, corrupt, flow_loss, per_residue_flow_loss,
     per_modality_residue_flow_loss) = _flow_callables(model)

    def call_denoiser(bound_model, packed_batch: dict) -> dict:
        with capture_module_output(trunk_final) as captured:
            nn_out = bound_model.call_nn(packed_batch, n_recycle=0)

        hidden = _trim_to_design(captured.tensor, packed_batch)

        return {**nn_out, "hidden": hidden}

    return ProteinaBindings(
        call_denoiser=call_denoiser,
        corrupt=corrupt,
        flow_loss=flow_loss,
        per_residue_flow_loss=per_residue_flow_loss,
        per_modality_residue_flow_loss=per_modality_residue_flow_loss,
        output_key=output_key,
        trunk_layers=len(layers),
    )

def directional_proteina_bindings(
    model, *, n_recycle: int = 0
) -> DirectionalProteinaBindings:

    if n_recycle != 0:
        raise FlowBindingError("directional confirmation pins n_recycle to zero")
    denoiser = getattr(model, "nn", None)
    if not callable(getattr(denoiser, "forward_route", None)):
        raise FlowBindingError("model.nn has no directional forward_route")

    (output_key, corrupt, flow_loss, per_residue_flow_loss,
     per_modality_residue_flow_loss) = _flow_callables(model)

    def call_route(
        bound_model: nn.Module, batch: dict, route: RouteState
    ) -> DirectionalForward:
        return bound_model.nn.forward_route(batch, route)

    return DirectionalProteinaBindings(
        corrupt=corrupt,
        call_route=call_route,
        flow_loss=flow_loss,
        per_residue_flow_loss=per_residue_flow_loss,
        per_modality_residue_flow_loss=per_modality_residue_flow_loss,
        output_key=output_key,
    )

def _flow_callables(model):

    output_key = _output_key(model)

    def corrupt(batch: dict) -> dict:

        from proteinfoundation.utils.sample_utils import add_clean_samples

        batch = add_clean_samples(
            batch,
            model.cfg_exp.product_flowmatcher,
            getattr(model, "autoencoder", None),
        )
        return model.fm.corrupt_batch(batch)

    def flow_loss(v_x: Tensor, v_z: Tensor, batch: Mapping) -> Tensor:

        routed = {
            "bb_ca": {output_key: v_x},
            "local_latents": {output_key: v_z},
        }
        losses = model.fm.compute_loss(batch=dict(batch), nn_out=routed)
        terms = [value.mean() for key, value in losses.items() if "_justlog" not in key]
        if not terms:
            raise FlowBindingError("compute_loss returned no trainable term")
        return sum(terms)

    def per_modality_residue_flow_loss(
        v_x: Tensor, v_z: Tensor, batch: Mapping
    ) -> dict[str, Tensor]:

        routed = {"bb_ca": v_x, "local_latents": v_z}
        mask = batch["mask"]
        nres = mask.sum(dim=-1)
        terms: dict[str, Tensor] = {}
        for mode, velocity in routed.items():
            matcher = model.fm.base_flow_matchers[mode]
            prediction = matcher.nn_out_add_clean_sample_prediction(
                x_t=batch["x_t"][mode],
                t=batch["t"][mode],
                mask=mask,
                nn_out={output_key: velocity},
            )["x_1"]
            error = (batch["x_1"][mode] - prediction) * mask[..., None]
            time_weight = 1.0 / ((1.0 - batch["t"][mode]) ** 2 + 1e-5)
            terms[mode] = (
                error.square().sum(dim=-1) / nres[..., None] * time_weight[..., None]
            )
        return terms

    def per_residue_flow_loss(v_x: Tensor, v_z: Tensor, batch: Mapping) -> Tensor:

        return sum(per_modality_residue_flow_loss(v_x, v_z, batch).values())

    return (
        output_key,
        corrupt,
        flow_loss,
        per_residue_flow_loss,
        per_modality_residue_flow_loss,
    )

def _output_key(model) -> str:

    try:
        parameterization = model.cfg_exp.nn.output_parameterization
    except AttributeError as error:
        raise FlowBindingError(
            "model config carries no nn.output_parameterization"
        ) from error

    keys = {str(parameterization[m]) for m in ("bb_ca", "local_latents")}
    if len(keys) != 1:
        raise FlowBindingError(
            f"the two modalities use different output parameterizations {keys}; "
            f"the routed field would combine unlike quantities"
        )
    return keys.pop()

def _trim_to_design(hidden: Tensor, batch: Mapping) -> Tensor:

    mask = batch.get("mask")
    if mask is None:
        raise FlowBindingError("batch has no `mask`, so the design length is unknown")
    n_orig = int(mask.shape[1])

    if hidden.shape[1] == n_orig:
        return hidden
    if hidden.shape[1] < n_orig:
        raise FlowBindingError(
            f"the trunk emitted {hidden.shape[1]} tokens, fewer than the "
            f"{n_orig} of the design sequence"
        )
    return hidden[:, :n_orig, :]

MUTABLE_SOURCE_REALM_EXCLUSIONS: Mapping[str, type] = MappingProxyType({})
