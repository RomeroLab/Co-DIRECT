
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

_ATOM37 = 37

class EncodeError(RuntimeError):
    pass

class Autoencoder(Protocol):

    def encode(self, batch: dict) -> Tensor:
        ...

@dataclass(frozen=True, slots=True)
class GroundTruthSample:

    bb_ca: Tensor
    local_latents: Tensor
    mask: Tensor

def encode_binder(
    atom37: Tensor,
    mask: Tensor,
    *,
    autoencoder: Autoencoder,
    ca_index: int = 1,
) -> GroundTruthSample:

    if atom37.dim() != 4 or atom37.shape[2] != _ATOM37 or atom37.shape[3] != 3:
        raise EncodeError(
            f"coordinates must be atom37 with shape [b, n, 37, 3], observed "
            f"{tuple(atom37.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise EncodeError(f"mask must be bool, observed {mask.dtype}")
    if mask.shape != atom37.shape[:2]:
        raise EncodeError(
            f"mask shape {tuple(mask.shape)} != {tuple(atom37.shape[:2])}"
        )
    if not bool(mask.any()):
        raise EncodeError("chain has no valid residue")
    if not torch.isfinite(atom37[mask]).all():
        raise EncodeError("non-finite coordinate at a residue marked valid")
    if not 0 <= ca_index < _ATOM37:
        raise EncodeError(f"ca_index {ca_index} is outside atom37")

    latents = autoencoder.encode({"coords": atom37, "mask": mask})
    if latents.dim() != 3 or latents.shape[1] != atom37.shape[1]:
        raise EncodeError(
            f"encoded latent length {tuple(latents.shape)} does not match the "
            f"{atom37.shape[1]} residues supplied"
        )

    expanded = mask.unsqueeze(-1)
    bb_ca = torch.where(expanded, atom37[:, :, ca_index, :], torch.zeros_like(atom37[:, :, ca_index, :]))
    local_latents = torch.where(expanded, latents, torch.zeros_like(latents))
    return GroundTruthSample(bb_ca=bb_ca, local_latents=local_latents, mask=mask)
