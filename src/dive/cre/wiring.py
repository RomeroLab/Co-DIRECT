
from __future__ import annotations

import torch

from dive.ccr.trunk import MESSAGE_KEY, MessageAdapter
from dive.cre.exchange import N_FEATURES

class SelfConditionedResponseAdapter(MessageAdapter):

    def __init__(self, inner, *, token_dim: int, n_features: int,
                 exchange=None, autoencoder=None, stride: int = 1):
        super().__init__(inner, token_dim=token_dim, n_features=n_features)
        self.exchange = exchange

        self._autoencoder = [autoencoder]
        self.stride = int(stride)
        self._calls = 0
        self._cached = None
        self._cached_shape = None

    @property
    def autoencoder(self):
        return self._autoencoder[0]

    def forward(self, batch: dict):
        if MESSAGE_KEY not in batch and self.exchange is not None:
            clean = batch.get("x_sc")
            mask = batch.get("mask")
            if clean is not None and mask is not None:
                shape = tuple(mask.shape)
                due = (self._calls % self.stride == 0) or self._cached_shape != shape
                if due:
                    message = self.exchange(
                        batch, clean, autoencoder=self.autoencoder
                    )
                    self._cached, self._cached_shape = message, shape
                self._calls += 1
                if self._cached is not None and self._cached_shape == shape:
                    batch = {**batch, MESSAGE_KEY: self._cached}
        return super().forward(batch)

def install_response_adapter(
    model, *, token_dim: int | None = None, exchange=None, stride: int = 1
) -> MessageAdapter:

    trunk = getattr(model, "nn", None)
    if trunk is None or not hasattr(trunk, "init_repr_factory"):
        raise ValueError("model has no `nn.init_repr_factory` to adapt")
    if isinstance(trunk.init_repr_factory, MessageAdapter):
        raise ValueError("a MessageAdapter is already installed")
    width = int(token_dim if token_dim is not None else model.cfg_exp.nn.token_dim)
    adapter = SelfConditionedResponseAdapter(
        trunk.init_repr_factory, token_dim=width, n_features=N_FEATURES,
        exchange=exchange, autoencoder=getattr(model, "autoencoder", None),
        stride=stride,
    )

    reference = next((p for p in trunk.parameters()), None)
    if reference is not None:
        adapter = adapter.to(device=reference.device, dtype=reference.dtype)
    trunk.init_repr_factory = adapter
    return adapter

def message_tensor(per_residue, *, n_residues: int, device=None) -> torch.Tensor:

    out = torch.zeros(1, n_residues, N_FEATURES, dtype=torch.float32, device=device)
    for index, features in per_residue.items():
        if 0 <= index < n_residues:
            out[0, index] = torch.as_tensor(features, dtype=torch.float32, device=device)
    return out

VARIANTS: dict[str, tuple[int, ...]] = {

    "full": tuple(range(N_FEATURES)),

    "no_response": (0, 1, 2, 3, 7),

    "feasibility_only": (3,),

    "response_only": (4, 5, 6),
}

def apply_variant(message: torch.Tensor, variant: str) -> torch.Tensor:

    try:
        keep = VARIANTS[variant]
    except KeyError as error:
        raise KeyError(f"unknown variant {variant!r}; expected {sorted(VARIANTS)}") from error
    out = torch.zeros_like(message)
    for column in keep:
        out[..., column] = message[..., column]
    return out
