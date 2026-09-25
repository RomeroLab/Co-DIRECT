
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn

from dive.leadership.directional_bridge import (
    BridgeTelemetry,
    DirectionalBridge,
    FourWayRouter,
    RouteProbabilities,
    RouteState,
    hard_route_probabilities,
)

PRESENCE_KEY = "dive_modality_presence"

class DirectionalV2Error(RuntimeError):
    pass

def tap_indices(nlayers: int) -> tuple[int, int, int]:

    if not isinstance(nlayers, int) or isinstance(nlayers, bool) or nlayers < 4:
        raise DirectionalV2Error(
            f"three distinct taps require at least four transformer blocks, got {nlayers!r}"
        )

    used: set[int] = set()
    taps: list[int] = []
    for fraction in (0.25, 0.50, 0.75):
        index = round(fraction * (nlayers - 1))
        while index in used and index < nlayers:
            index += 1
        if index >= nlayers:
            raise DirectionalV2Error(
                f"cannot place three distinct taps in a {nlayers}-block trunk"
            )
        taps.append(index)
        used.add(index)
    return tuple(taps)

@dataclass(frozen=True, slots=True)
class DirectionalForward:

    nn_out: Mapping[str, Mapping[str, Tensor]]
    telemetry: tuple[BridgeTelemetry, ...]

@dataclass(frozen=True, slots=True)
class _ReplayState:
    seqs: Tensor
    pair_rep: Tensor
    cond: Tensor
    mask: Tensor
    orig_mask: Tensor
    n_orig: int
    n_concat: int
    target_rep: Tensor | None = None
    target_mask: Tensor | None = None

    def clone(self) -> "_ReplayState":

        return replace(self, seqs=self.seqs.clone(), pair_rep=self.pair_rep.clone())

class DirectionalV2Adapter(nn.Module):

    @classmethod
    def from_loaded(
        cls,
        base: nn.Module,
        *,
        router: FourWayRouter,
        rank: int = 8,
        n_recycle: int = 0,
    ) -> "DirectionalV2Adapter":

        try:
            hidden_dim = int(base.token_dim)
            nlayers = int(base.nlayers)
        except (AttributeError, TypeError, ValueError) as error:
            raise DirectionalV2Error(
                "loaded base must expose integer token_dim and nlayers"
            ) from error
        if hidden_dim <= 0:
            raise DirectionalV2Error("loaded base token_dim must be positive")
        bridges = {
            str(index): DirectionalBridge(hidden_dim=hidden_dim, rank=rank)
            for index in tap_indices(nlayers)
        }
        return cls(
            base,
            router=router,
            bridges=bridges,
            n_recycle=n_recycle,
        )

    def __init__(
        self,
        base: nn.Module,
        *,
        router: FourWayRouter,
        bridges: Mapping[str, DirectionalBridge],
        n_recycle: int = 0,
    ) -> None:
        super().__init__()
        if n_recycle != 0:
            raise DirectionalV2Error(
                "directional v2 replay requires recycling to be zero"
            )
        for name in (
            "transformer_layers",
            "pair_repr_builder",
            "ca_linear",
            "local_latents_linear",
            "cond_factory",
            "transition_c_1",
            "transition_c_2",
            "init_repr_factory",
        ):
            if not hasattr(base, name):
                raise DirectionalV2Error(
                    f"base denoiser is not the pinned v2 shape: missing {name}"
                )
        try:
            nlayers = int(base.nlayers)
        except (AttributeError, TypeError, ValueError) as error:
            raise DirectionalV2Error(
                "base denoiser is not the pinned v2 shape: invalid nlayers"
            ) from error

        selectors: dict[str, bool] = {}
        for name in (
            "use_concat",
            "use_advanced_pair",
            "use_target_cross_attn",
            "update_pair_repr",
        ):
            value = getattr(base, name, None)
            if type(value) is not bool:
                raise DirectionalV2Error(
                    f"base graph selector {name} must exist and be boolean"
                )
            selectors[name] = value

        if selectors["use_advanced_pair"] and not selectors["use_concat"]:
            raise DirectionalV2Error(
                "base use_advanced_pair requires use_concat in pinned v2"
            )
        if selectors["use_target_cross_attn"] and selectors["use_concat"]:
            raise DirectionalV2Error(
                "base use_target_cross_attn requires use_concat to be false"
            )

        def require_module(name: str) -> None:
            if not isinstance(getattr(base, name, None), nn.Module):
                raise DirectionalV2Error(
                    f"selected pinned-v2 graph requires module {name}"
                )

        def require_module_list(
            name: str, expected_length: int, *, allow_none: bool
        ) -> None:
            value = getattr(base, name, None)
            if not isinstance(value, nn.ModuleList) or len(value) != expected_length:
                raise DirectionalV2Error(
                    f"selected pinned-v2 graph requires {name} ModuleList "
                    f"of length {expected_length}"
                )
            if any(
                item is not None and not isinstance(item, nn.Module) for item in value
            ) or (not allow_none and any(item is None for item in value)):
                raise DirectionalV2Error(
                    f"selected pinned-v2 graph has invalid modules in {name}"
                )

        if selectors["use_concat"]:
            require_module("concat_factory")
        if selectors["use_advanced_pair"]:
            require_module("concat_pair_factory")
        if selectors["use_target_cross_attn"]:
            require_module("target_factory")
            require_module_list(
                "target2binder_cross_attention_layer", nlayers, allow_none=False
            )
        if selectors["update_pair_repr"]:
            require_module_list("pair_update_layers", nlayers - 1, allow_none=True)

        taps = tap_indices(nlayers)
        expected_bridge_keys = {str(index) for index in taps}
        if set(bridges) != expected_bridge_keys:
            raise DirectionalV2Error(
                f"bridges must be keyed by tap indices {sorted(expected_bridge_keys)}, "
                f"got {sorted(bridges)}"
            )

        self.base = base
        self.router = router
        self.bridges = nn.ModuleDict(dict(bridges))
        self.nlayers = nlayers
        self.tap_indices = taps
        if len(base.transformer_layers) != nlayers:
            raise DirectionalV2Error(
                "base denoiser transformer_layers length disagrees with nlayers"
            )
        output_param = getattr(base, "output_param", None)
        if not isinstance(output_param, Mapping):
            raise DirectionalV2Error(
                "base denoiser is not the pinned v2 shape: invalid output_param"
            )
        try:
            self.output_key_x = str(output_param["bb_ca"])
            self.output_key_z = str(output_param["local_latents"])
        except KeyError as error:
            raise DirectionalV2Error(
                "base output_param must name bb_ca and local_latents heads"
            ) from error
        self._active_route: ContextVar[RouteState] = ContextVar(
            f"dive_directional_route_{id(self)}", default=RouteState.NONE
        )

    def forward(self, batch: Mapping) -> Mapping[str, Mapping[str, Tensor]]:
        return self.forward_route(batch, self._active_route.get()).nn_out

    @contextmanager
    def route_context(self, route: RouteState) -> Iterator[None]:

        try:
            selected = RouteState(route)
        except ValueError as error:
            raise DirectionalV2Error(f"unknown directional route {route!r}") from error
        token = self._active_route.set(selected)
        try:
            yield
        finally:
            self._active_route.reset(token)

    def forward_route(self, batch: Mapping, route: RouteState) -> DirectionalForward:
        try:
            route = RouteState(route)
        except ValueError as error:
            raise DirectionalV2Error(f"unknown directional route {route!r}") from error
        if route is RouteState.NONE:
            return DirectionalForward(nn_out=self.base(dict(batch)), telemetry=())
        return self._dual_stream_forward(dict(batch), route)

    def _dual_stream_forward(
        self, batch: dict, route: RouteState
    ) -> DirectionalForward:
        presence = self._structural_presence(batch)
        telemetry: list[BridgeTelemetry] = []
        state = self._initial_state(batch)
        x_state, z_state = self._shared_prefix(state, batch, route, presence, telemetry)
        x_state, z_state = self._stream_suffix(
            x_state, z_state, batch, route, presence, telemetry
        )
        return DirectionalForward(
            nn_out=self._heads(x_state, z_state), telemetry=tuple(telemetry)
        )

    def _initial_state(self, batch: dict) -> _ReplayState:

        mask = batch.get("mask")
        if not isinstance(mask, Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
            raise DirectionalV2Error(
                "batch mask must be a boolean tensor with shape [B, N]"
            )
        orig_mask = mask.clone()

        cond = self.base.cond_factory(batch)
        cond = self.base.transition_c_2(
            self.base.transition_c_1(cond, mask),
            mask,
        )
        seqs = self.base.init_repr_factory(batch) * mask[..., None]
        if seqs.ndim != 3 or seqs.shape[:2] != mask.shape:
            raise DirectionalV2Error(
                "base init_repr_factory did not return pinned-v2 [B, N, D] state"
            )
        batch_size, n_orig, _ = seqs.shape

        if bool(getattr(self.base, "use_concat", False)):
            concat_factory = getattr(self.base, "concat_factory", None)
            if concat_factory is None:
                raise DirectionalV2Error(
                    "base use_concat is true but concat_factory is missing"
                )
            seqs, mask = concat_factory(batch, seqs, mask)
            n_concat = int(seqs.shape[1] - n_orig)
            if n_concat < 0 or mask.shape != seqs.shape[:2]:
                raise DirectionalV2Error(
                    "concat_factory returned an invalid pinned-v2 shape"
                )
            if n_concat:
                zero_cond = torch.zeros(
                    batch_size, n_concat, cond.shape[-1], device=seqs.device
                )
                cond = torch.cat((cond, zero_cond), dim=1)
        else:
            n_concat = 0

        pair_rep = self.base.pair_repr_builder(batch)
        if bool(getattr(self.base, "use_concat", False)) and bool(
            getattr(self.base, "use_advanced_pair", False)
        ):
            concat_pair_factory = getattr(self.base, "concat_pair_factory", None)
            if concat_pair_factory is None:
                raise DirectionalV2Error(
                    "base advanced concat pair mode is missing concat_pair_factory"
                )
            pair_rep = concat_pair_factory(batch, pair_rep, orig_mask)
        elif n_concat:
            pair_dim = pair_rep.shape[-1]
            zero_pad_1 = torch.zeros(
                batch_size, n_concat, n_orig, pair_dim, device=seqs.device
            )
            pair_rep = torch.cat((pair_rep, zero_pad_1), dim=1)
            zero_pad_2 = torch.zeros(
                batch_size, seqs.shape[1], n_concat, pair_dim, device=seqs.device
            )
            pair_rep = torch.cat((pair_rep, zero_pad_2), dim=2)

        extended = seqs.shape[1]
        if pair_rep.shape[:3] != (batch_size, extended, extended):
            raise DirectionalV2Error(
                "pair representation shape does not match the pinned-v2 sequence state"
            )

        target_rep = target_mask = None
        if bool(getattr(self.base, "use_target_cross_attn", False)):
            target_factory = getattr(self.base, "target_factory", None)
            cross_layers = getattr(
                self.base, "target2binder_cross_attention_layer", None
            )
            if (
                target_factory is None
                or cross_layers is None
                or len(cross_layers) != self.nlayers
            ):
                raise DirectionalV2Error(
                    "base target cross-attention modules do not match the pinned-v2 shape"
                )
            target_rep = target_factory(batch)
            target_mask = batch.get("seq_target_mask")
            if (
                not isinstance(target_mask, Tensor)
                or target_mask.shape != target_rep.shape[:2]
            ):
                raise DirectionalV2Error(
                    "seq_target_mask does not match target representation"
                )
            target_rep = target_rep * target_mask[..., None]

        return _ReplayState(
            seqs=seqs,
            pair_rep=pair_rep,
            cond=cond,
            mask=mask,
            orig_mask=orig_mask,
            n_orig=n_orig,
            n_concat=n_concat,
            target_rep=target_rep,
            target_mask=target_mask,
        )

    def _shared_prefix(
        self,
        state: _ReplayState,
        batch: Mapping,
        route: RouteState,
        presence: Tensor,
        telemetry: list[BridgeTelemetry],
    ) -> tuple[_ReplayState, _ReplayState]:
        first_tap = self.tap_indices[0]
        for index in range(first_tap + 1):
            state = self._block(index, state)
        x_state, z_state = state.clone(), state.clone()
        x_state, z_state = self._apply_bridge(
            first_tap, x_state, z_state, batch, route, presence, telemetry
        )
        return x_state, z_state

    def _stream_suffix(
        self,
        x_state: _ReplayState,
        z_state: _ReplayState,
        batch: Mapping,
        route: RouteState,
        presence: Tensor,
        telemetry: list[BridgeTelemetry],
    ) -> tuple[_ReplayState, _ReplayState]:
        for index in range(self.tap_indices[0] + 1, self.nlayers):
            x_state = self._block(index, x_state)
            z_state = self._block(index, z_state)
            if index in self.tap_indices:
                x_state, z_state = self._apply_bridge(
                    index, x_state, z_state, batch, route, presence, telemetry
                )
        return x_state, z_state

    def _block(self, index: int, state: _ReplayState) -> _ReplayState:
        seqs = state.seqs
        if bool(getattr(self.base, "use_target_cross_attn", False)):
            assert state.target_rep is not None and state.target_mask is not None
            seqs = self.base.target2binder_cross_attention_layer[index](
                seqs,
                state.target_rep,
                state.cond,
                state.mask,
                state.target_mask,
            )
        seqs = self.base.transformer_layers[index](
            seqs, state.pair_rep, state.cond, state.mask
        )
        pair_rep = state.pair_rep
        if (
            bool(getattr(self.base, "update_pair_repr", False))
            and index < self.nlayers - 1
        ):
            pair_update_layers = getattr(self.base, "pair_update_layers", None)
            if (
                pair_update_layers is None
                or len(pair_update_layers) != self.nlayers - 1
            ):
                raise DirectionalV2Error(
                    "base pair_update_layers do not match the pinned-v2 shape"
                )
            update = pair_update_layers[index]
            if update is not None:
                pair_rep = update(seqs, pair_rep, state.mask)
        return replace(state, seqs=seqs, pair_rep=pair_rep)

    def _apply_bridge(
        self,
        index: int,
        x_state: _ReplayState,
        z_state: _ReplayState,
        batch: Mapping,
        route: RouteState,
        presence: Tensor,
        telemetry: list[BridgeTelemetry],
    ) -> tuple[_ReplayState, _ReplayState]:
        n_orig = x_state.n_orig
        h_x = x_state.seqs[:, :n_orig]
        h_z = z_state.seqs[:, :n_orig]
        batch_size, residues, _ = h_x.shape
        if route is RouteState.ADAPTIVE:
            times = batch.get("t")
            if not isinstance(times, Mapping):
                raise DirectionalV2Error(
                    "adaptive routing requires modality time tensors"
                )
            try:
                t_backbone = times["bb_ca"]
                t_latent = times["local_latents"]
                generated_mask = batch["generated_mask"]
                fixed_mask = batch["fixed_mask"]
            except KeyError as error:
                raise DirectionalV2Error(
                    f"adaptive routing batch is missing {error.args[0]!r}"
                ) from error
            probabilities = self.router(
                h_x=h_x,
                h_z=h_z,
                t_backbone=t_backbone,
                t_latent=t_latent,
                generated_mask=generated_mask,
                fixed_mask=fixed_mask,
                presence=presence,
            )
        else:
            probabilities = RouteProbabilities(
                hard_route_probabilities(
                    route,
                    batch=batch_size,
                    residues=residues,
                    device=h_x.device,
                )
            )

        bridged = self.bridges[str(index)](
            h_x, h_z, probabilities, x_state.orig_mask, presence
        )
        x_seqs = torch.cat((bridged.h_x, x_state.seqs[:, n_orig:]), dim=1)
        z_seqs = torch.cat((bridged.h_z, z_state.seqs[:, n_orig:]), dim=1)
        telemetry.append(bridged.telemetry)
        return replace(x_state, seqs=x_seqs), replace(z_state, seqs=z_seqs)

    def _heads(
        self, x_state: _ReplayState, z_state: _ReplayState
    ) -> Mapping[str, Mapping[str, Tensor]]:
        local_latents = (
            self.base.local_latents_linear(z_state.seqs) * z_state.mask[..., None]
        )
        bb_ca = self.base.ca_linear(x_state.seqs) * x_state.mask[..., None]
        if x_state.n_concat:
            local_latents = (
                local_latents[:, : x_state.n_orig] * z_state.orig_mask[..., None]
            )
            bb_ca = bb_ca[:, : x_state.n_orig] * x_state.orig_mask[..., None]
        return {
            "bb_ca": {self.output_key_x: bb_ca},
            "local_latents": {self.output_key_z: local_latents},
        }

    def _structural_presence(self, batch: Mapping) -> Tensor:
        mask = batch.get("mask")
        if not isinstance(mask, Tensor) or mask.ndim != 2:
            raise DirectionalV2Error("cannot derive presence without a [B, N] mask")
        x_t = batch.get("x_t")
        x_t = x_t if isinstance(x_t, Mapping) else {}
        row = [
            int(isinstance(x_t.get("bb_ca"), Tensor)),
            int(isinstance(x_t.get("local_latents"), Tensor)),
        ]
        structural = torch.tensor(
            [row] * int(mask.shape[0]), dtype=torch.long, device=mask.device
        )
        claimed = batch.get(PRESENCE_KEY)
        if claimed is not None:
            if not isinstance(claimed, Tensor) or claimed.shape != structural.shape:
                shape = tuple(claimed.shape) if isinstance(claimed, Tensor) else None
                raise DirectionalV2Error(
                    f"{PRESENCE_KEY} must have shape {tuple(structural.shape)}, got {shape}"
                )
            if not torch.all((claimed == 0) | (claimed == 1)):
                raise DirectionalV2Error(f"{PRESENCE_KEY} values must be binary")
            if not torch.equal(
                claimed.to(device=mask.device, dtype=torch.long), structural
            ):
                raise DirectionalV2Error(
                    f"{PRESENCE_KEY} disagrees with structural x_t modality presence"
                )
        return structural

MUTABLE_SOURCE_REALM_EXCLUSIONS: Mapping[str, type] = MappingProxyType({})
