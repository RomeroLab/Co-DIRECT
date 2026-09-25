
from __future__ import annotations

import torch
from torch import nn

from dive.sdx.relations import ANCHOR_FEATURES, N_BINS

UNKNOWN_RESIDUE = 20

class IdentityEmbed(nn.Module):

    def __init__(self, token_dim: int, *, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(UNKNOWN_RESIDUE + 1, hidden)
        final = nn.Linear(hidden, token_dim)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.project = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), final)

    def forward(self, residue_type: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe = torch.where(
            (residue_type >= 0) & (residue_type < UNKNOWN_RESIDUE) & mask,
            residue_type,
            torch.full_like(residue_type, UNKNOWN_RESIDUE),
        )
        out = self.project(self.embed(safe))
        return out * mask[..., None].to(out.dtype)

class RelationHead(nn.Module):

    def __init__(self, token_dim: int, *, hidden: int = 128, n_bins: int = N_BINS):
        super().__init__()
        self.hidden = int(hidden)
        self.query = nn.Linear(token_dim, hidden)

        self.anchor = nn.Sequential(
            nn.Linear(ANCHOR_FEATURES, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(3 * hidden),
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        anchor_features: torch.Tensor,
        anchor_valid: torch.Tensor,
    ) -> torch.Tensor:
        b, n, _ = tokens.shape
        m = anchor_features.shape[1]
        q = self.query(tokens)
        k = self.anchor(anchor_features.to(q.dtype))

        q_e = q[:, :, None, :].expand(b, n, m, self.hidden)
        k_e = k[:, None, :, :].expand(b, n, m, self.hidden)
        features = torch.cat([q_e, k_e, q_e * k_e], dim=-1)
        logits = self.mlp(features)
        return logits * anchor_valid[:, None, :, None].to(logits.dtype)

class TokenCapture(nn.Module):

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._captured: list[torch.Tensor | None] = [None]

    @property
    def captured(self) -> torch.Tensor | None:
        return self._captured[0]

    def clear(self) -> None:
        self._captured[0] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._captured[0] = x
        return self.inner(x)
