
from __future__ import annotations

import torch
from torch import nn

from dive.sdx.cavity import SDX_CAVITY, SDX_IDENTITY
from dive.sdx.exchange import N_FEATURES
from dive.sdx.heads import IdentityEmbed, RelationHead, TokenCapture

SDX_MESSAGE = "sdx_message"

class SdxFactory(nn.Module):

    def __init__(self, inner: nn.Module, *, token_dim: int, hidden: int = 128):
        super().__init__()
        self.inner = inner
        self.token_dim = int(token_dim)
        self.identity = IdentityEmbed(token_dim, hidden=hidden)
        final = nn.Linear(hidden, token_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.projection = nn.Sequential(
            nn.LayerNorm(N_FEATURES),
            nn.Linear(N_FEATURES, hidden),
            nn.GELU(),
            final,
        )

        self._live: list = [None]

        self._inject: bool = True

        self._trajectory_length: int | None = None

    def attach_live(self, exchange) -> None:
        self._live[0] = exchange

    def set_trajectory_length(self, length: int | None) -> None:

        self._trajectory_length = int(length) if length else None

    def set_injection(self, inject: bool) -> None:

        self._inject = bool(inject)

    @property
    def injects(self) -> bool:
        return self._inject

    @property
    def live(self):
        return self._live[0]

    def forward(self, batch: dict) -> torch.Tensor:
        representation = self.inner(batch)
        mask = batch.get("mask")

        if batch.get(SDX_CAVITY, False):

            identity = batch.get(SDX_IDENTITY)
            if identity is not None:
                keep = mask if mask is not None else torch.ones_like(identity, dtype=torch.bool)
                representation = representation + self.identity(identity, keep)
            return representation

        message = batch.get(SDX_MESSAGE)
        if message is None and self._live[0] is not None:

            message = self._live[0].message_for(batch)
        if message is None:
            return representation
        if message.shape[-1] != N_FEATURES:
            raise ValueError(
                f"{SDX_MESSAGE} has width {message.shape[-1]}, expected {N_FEATURES}; "
                "refusing to broadcast a mismatched message"
            )
        delta = self.projection(message.to(representation.dtype))
        if mask is not None:
            delta = delta * mask[..., None].to(delta.dtype)

        self._record(representation, delta, mask)
        if not self._inject:

            return representation
        return representation + delta

    def _record(self, representation, delta, mask) -> None:
        exchange = self._live[0]
        stats = getattr(exchange, "stats", None)
        if stats is None:
            return
        with torch.no_grad():
            if mask is not None:
                keep = mask[..., None].to(delta.dtype)
                n = float(keep.sum().clamp(min=1.0)) * delta.shape[-1]
                delta_rms = float(((delta * keep) ** 2).sum() / n) ** 0.5
                repr_rms = float(((representation * keep) ** 2).sum() / n) ** 0.5
            else:
                delta_rms = float((delta**2).mean()) ** 0.5
                repr_rms = float((representation**2).mean()) ** 0.5
        stats["injections_measured"] = stats.get("injections_measured", 0) + 1
        total = self._trajectory_length
        if total:
            position = max(int(stats.get("calls", 1)) - 1, 0)
            decile = min(position * 10 // total, 9)
            for key, value in (("injections_by_decile", 1),
                               ("sum_delta_rms_by_decile", delta_rms),
                               ("sum_delta_over_repr_by_decile",
                                delta_rms / repr_rms if repr_rms > 0 else 0.0)):
                bucket = stats.setdefault(key, [0] * 10 if value == 1 else [0.0] * 10)
                bucket[decile] += value
        stats["sum_delta_rms"] = stats.get("sum_delta_rms", 0.0) + delta_rms
        stats["sum_repr_rms"] = stats.get("sum_repr_rms", 0.0) + repr_rms
        stats["max_delta_rms"] = max(stats.get("max_delta_rms", 0.0), delta_rms)
        ratio = delta_rms / repr_rms if repr_rms > 0 else 0.0
        stats["sum_delta_over_repr"] = stats.get("sum_delta_over_repr", 0.0) + ratio
        stats["max_delta_over_repr"] = max(stats.get("max_delta_over_repr", 0.0), ratio)

class SdxAdapter(nn.Module):

    def __init__(self, factory: SdxFactory, capture: TokenCapture,
                 relation: RelationHead, *, autoencoder=None):
        super().__init__()
        self.factory = factory
        self.capture = capture
        self.relation = relation

        self._autoencoder = [autoencoder]

    @property
    def autoencoder(self):
        return self._autoencoder[0]

    @property
    def identity(self) -> IdentityEmbed:
        return self.factory.identity

    @property
    def projection(self) -> nn.Module:
        return self.factory.projection

    def new_parameters(self):

        yield from self.factory.identity.parameters()
        yield from self.factory.projection.parameters()
        yield from self.relation.parameters()

    def set_trainable(self, flag: bool = True) -> int:
        count = 0
        for parameter in self.new_parameters():
            parameter.requires_grad_(flag)
            count += parameter.numel()
        return count

def install_sdx(model, *, token_dim: int | None = None, hidden: int = 128) -> SdxAdapter:

    trunk = getattr(model, "nn", None)
    if trunk is None or not hasattr(trunk, "init_repr_factory"):
        raise ValueError("model has no `nn.init_repr_factory` to adapt")
    if isinstance(trunk.init_repr_factory, SdxFactory):
        raise ValueError("the operator is already installed")
    if isinstance(getattr(trunk, "local_latents_linear", None), TokenCapture):
        raise ValueError("the operator is already installed")
    if getattr(trunk, "sdx_relation", None) is not None:
        raise ValueError("the operator is already installed")

    width = int(token_dim if token_dim is not None else model.cfg_exp.nn.token_dim)
    factory = SdxFactory(trunk.init_repr_factory, token_dim=width, hidden=hidden)
    capture = TokenCapture(trunk.local_latents_linear)
    relation = RelationHead(width, hidden=hidden)

    reference = next((p for p in trunk.parameters()), None)
    if reference is not None:
        factory = factory.to(device=reference.device, dtype=reference.dtype)
        relation = relation.to(device=reference.device, dtype=reference.dtype)

    trunk.init_repr_factory = factory
    trunk.local_latents_linear = capture

    trunk.sdx_relation = relation
    return SdxAdapter(factory, capture, relation,
                      autoencoder=getattr(model, "autoencoder", None))
