
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from dive.leadership.conditions import PRESENCE_KEY

INIT_NOISE_STD = 0.02

JOINT_CODE = 0b11

class PresenceError(RuntimeError):
    pass

class PresenceConditionedFactory(nn.Module):

    def __init__(self, factory: nn.Module, *, token_dim: int) -> None:
        super().__init__()
        self.factory = factory
        self.token_dim = token_dim

        self.presence_embedding = nn.Embedding(4, token_dim)
        with torch.no_grad():
            self.presence_embedding.weight.normal_(0.0, INIT_NOISE_STD)

            self.presence_embedding.weight[JOINT_CODE].zero_()

    def forward(self, batch: Mapping) -> Tensor:
        representation = self.factory(batch)
        codes = _codes(batch)

        offset = self.presence_embedding(codes.to(representation.device))

        out = representation + offset.unsqueeze(1)

        mask = batch.get("mask")
        if isinstance(mask, Tensor):

            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out

def _codes(batch: Mapping) -> Tensor:

    presence = batch.get(PRESENCE_KEY)
    if not isinstance(presence, Tensor):
        raise PresenceError(
            f"batch carries no {PRESENCE_KEY!r}; defaulting to joint would make "
            f"every null condition a silent no-op"
        )
    if presence.dim() != 2 or presence.shape[1] != 2:
        raise PresenceError(
            f"{PRESENCE_KEY} must have shape [batch, 2], got {tuple(presence.shape)}"
        )
    if not torch.isin(presence, torch.tensor([0, 1], device=presence.device)).all():
        raise PresenceError(f"{PRESENCE_KEY} entries must be 0 or 1, got {presence.tolist()}")

    backbone, latent = presence[:, 0].long(), presence[:, 1].long()
    return backbone * 2 + latent
