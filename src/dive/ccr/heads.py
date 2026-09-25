
from __future__ import annotations

import torch
from torch import nn

N_AMINO_ACIDS = 20
MESSAGE_KEY = "ccr_message"
CANDIDATE_INDEX_KEY = "ccr_cand_index"
CANDIDATE_FEATURE_KEY = "ccr_cand_feat"

class CandidateAdapter(nn.Module):

    def __init__(
        self,
        inner: nn.Module,
        *,
        token_dim: int,
        n_slots: int,
        n_residue_features: int,
        n_candidate_features: int,
        width: int | None = None,
    ):
        super().__init__()
        self.inner = inner
        self.token_dim = int(token_dim)
        self.n_slots = int(n_slots)
        self.n_residue_features = int(n_residue_features)
        self.n_candidate_features = int(n_candidate_features)
        d = int(width or max(64, 4 * n_candidate_features))
        self.width = d

        self.candidate_embedding = nn.Embedding(N_AMINO_ACIDS, d)
        self.candidate_projection = nn.Sequential(
            nn.Linear(self.n_candidate_features, d), nn.GELU(), nn.Linear(d, d)
        )
        self.candidate_norm = nn.LayerNorm(d)

        self.residue_projection = nn.Sequential(
            nn.LayerNorm(self.n_residue_features), nn.Linear(self.n_residue_features, d),
            nn.GELU(), nn.Linear(d, d),
        )
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)

        self.choice_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        output = nn.Linear(d, self.token_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.output = nn.Sequential(nn.LayerNorm(d), nn.GELU(), output)

        self.last_choice_logits: torch.Tensor | None = None

    def candidate_tokens(
        self, index: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        present = index >= 0
        safe = index.clamp(min=0)
        tokens = self.candidate_embedding(safe) + self.candidate_projection(
            features.to(self.candidate_embedding.weight.dtype)
        )
        tokens = self.candidate_norm(tokens) * present[..., None].to(tokens.dtype)
        return tokens, present

    def choice_logits(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        logits = self.choice_head(tokens)[..., 0]
        return logits.masked_fill(~present, float("-inf"))

    def pooled(self, tokens, present, message) -> torch.Tensor:

        context = self.residue_projection(message.to(tokens.dtype))
        q = self.query(context)[..., None, :]
        k = self.key(tokens)
        v = self.value(tokens)
        scale = float(self.width) ** 0.5
        scores = (q * k).sum(-1) / scale
        scores = scores.masked_fill(~present, float("-inf"))

        any_present = present.any(-1, keepdim=True)
        scores = torch.where(any_present, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1) * any_present.to(scores.dtype)
        return context + (weights[..., None] * v).sum(-2)

    def forward(self, batch: dict) -> torch.Tensor:
        representation = self.inner(batch)
        message = batch.get(MESSAGE_KEY)
        index = batch.get(CANDIDATE_INDEX_KEY)
        features = batch.get(CANDIDATE_FEATURE_KEY)
        if message is None:
            self.last_choice_logits = None
            return representation
        if message.shape[-1] != self.n_residue_features:
            raise ValueError(
                f"{MESSAGE_KEY} has width {message.shape[-1]}, expected "
                f"{self.n_residue_features}; refusing to broadcast a mismatched message"
            )

        b, n = message.shape[0], message.shape[1]
        device, dtype = representation.device, representation.dtype
        if index is None or features is None:
            index = torch.full((b, n, self.n_slots), -1, dtype=torch.long, device=device)
            features = torch.zeros(b, n, self.n_slots, self.n_candidate_features,
                                   device=device)
        if features.shape[-1] != self.n_candidate_features:
            raise ValueError(
                f"{CANDIDATE_FEATURE_KEY} has width {features.shape[-1]}, expected "
                f"{self.n_candidate_features}"
            )

        tokens, present = self.candidate_tokens(index, features)
        self.last_choice_logits = self.choice_logits(tokens, present)
        delta = self.output(self.pooled(tokens, present, message)).to(dtype)

        mask = batch.get("mask")
        if mask is not None:
            delta = delta * mask[..., None].to(delta.dtype)
        return representation + delta

def install_candidate_adapter(
    model, *, n_slots: int, n_residue_features: int, n_candidate_features: int,
    token_dim: int | None = None,
) -> CandidateAdapter:

    trunk = getattr(model, "nn", None)
    if trunk is None or not hasattr(trunk, "init_repr_factory"):
        raise ValueError("model has no `nn.init_repr_factory` to adapt")
    if isinstance(trunk.init_repr_factory, CandidateAdapter):
        raise ValueError("a CandidateAdapter is already installed")
    width = int(token_dim if token_dim is not None else model.cfg_exp.nn.token_dim)
    adapter = CandidateAdapter(
        trunk.init_repr_factory, token_dim=width, n_slots=n_slots,
        n_residue_features=n_residue_features,
        n_candidate_features=n_candidate_features,
    )
    trunk.init_repr_factory = adapter
    return adapter
