
from __future__ import annotations

import torch
from torch import nn

from dive.ccr.message import FEATURE_NAMES

N_MESSAGE_FEATURES = len(FEATURE_NAMES)
MESSAGE_KEY = "ccr_message"

class MessageAdapter(nn.Module):

    def __init__(self, inner: nn.Module, *, token_dim: int,
                 hidden: int | None = None, n_features: int = N_MESSAGE_FEATURES):
        super().__init__()
        self.inner = inner
        self.token_dim = int(token_dim)
        self.n_features = int(n_features)
        width = int(hidden or max(4 * n_features, 64))
        final = nn.Linear(width, self.token_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, width),
            nn.GELU(),
            final,
        )

    def forward(self, batch: dict) -> torch.Tensor:
        representation = self.inner(batch)
        message = batch.get(MESSAGE_KEY)
        if message is None:
            return representation
        if message.shape[-1] != self.n_features:
            raise ValueError(
                f"{MESSAGE_KEY} has width {message.shape[-1]}, expected "
                f"{self.n_features}; refusing to broadcast a mismatched message"
            )
        message = message.to(representation.dtype)
        delta = self.projection(message)
        mask = batch.get("mask")
        if mask is not None:
            delta = delta * mask[..., None].to(delta.dtype)
        return representation + delta

def install_message_adapter(model, *, token_dim: int | None = None) -> MessageAdapter:

    trunk = getattr(model, "nn", None)
    if trunk is None or not hasattr(trunk, "init_repr_factory"):
        raise ValueError("model has no `nn.init_repr_factory` to adapt")
    if isinstance(trunk.init_repr_factory, MessageAdapter):
        raise ValueError("a MessageAdapter is already installed")
    width = int(token_dim if token_dim is not None else model.cfg_exp.nn.token_dim)
    adapter = MessageAdapter(trunk.init_repr_factory, token_dim=width)
    trunk.init_repr_factory = adapter
    return adapter
