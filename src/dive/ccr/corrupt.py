
from __future__ import annotations

from dataclasses import dataclass

import torch

CORRUPTION_KINDS: tuple[str, ...] = (
    "rotamer", "backbone", "identity_free", "identity_fixed", "joint",
)

LATENT_SIGMA = 1.4
CA_SIGMA_NM = 0.25
MAX_SPAN = 8

@dataclass(frozen=True)
class RepairBatch:

    state: dict[str, torch.Tensor]
    repair_mask: torch.Tensor
    score_sequence: torch.Tensor
    score_geometry: torch.Tensor
    identity_given: torch.Tensor
    kinds: tuple[str, ...]

def apply_corruptions(
    clean: dict[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    kinds: tuple[str, ...] | None = None,
) -> RepairBatch:

    ca = clean["bb_ca"]
    latents = clean["local_latents"]
    b, n = mask.shape
    device = ca.device

    def rand(*shape):
        return torch.randn(*shape, generator=generator, device="cpu").to(device)

    def randint(high, shape=()):
        return torch.randint(0, int(high), shape, generator=generator,
                             device="cpu")

    state = {"bb_ca": ca.clone(), "local_latents": latents.clone()}
    repair = torch.zeros(b, n, dtype=torch.bool, device=device)
    given = torch.zeros(b, n, dtype=torch.bool, device=device)
    chosen: list[str] = []

    latent_scale = float(latents[mask].std()) if bool(mask.any()) else 1.0
    menu = tuple(kinds) if kinds else CORRUPTION_KINDS

    for i in range(b):
        valid = torch.nonzero(mask[i], as_tuple=False).flatten()
        if valid.numel() == 0:
            chosen.append("none")
            continue
        kind = menu[int(randint(len(menu)))]
        chosen.append(kind)

        span = (int(1 + randint(2))
                if kind in ("rotamer", "identity_free", "identity_fixed")
                else int(2 + randint(MAX_SPAN - 1)))
        span = min(span, int(valid.numel()))
        start = int(randint(max(int(valid.numel()) - span + 1, 1)))
        rows = valid[start: start + span]
        repair[i, rows] = True

        if kind in ("rotamer", "joint"):
            state["local_latents"][i, rows] += (
                LATENT_SIGMA * latent_scale * rand(rows.numel(), latents.shape[-1])
            )
        if kind in ("backbone", "joint", "identity_fixed"):
            state["bb_ca"][i, rows] += CA_SIGMA_NM * rand(rows.numel(), 3)
        if kind == "identity_free":

            donor = valid[int(randint(valid.numel()))]
            state["local_latents"][i, rows] = latents[i, donor][None, :]
        if kind == "identity_fixed":
            given[i, rows] = True

    score_geometry = repair.clone()
    score_sequence = repair & ~given
    return RepairBatch(state, repair, score_sequence, score_geometry, given,
                       tuple(chosen))

def repair_losses(
    *,
    seq_logits: torch.Tensor,
    native_aatype: torch.Tensor,
    predicted_ca: torch.Tensor,
    native_ca: torch.Tensor,
    score_sequence: torch.Tensor,
    score_geometry: torch.Tensor,
    sequence_weight: float = 1.0,
    geometry_weight: float = 1.0,
    predicted_atoms: torch.Tensor | None = None,
    native_atoms: torch.Tensor | None = None,
    atom_score: torch.Tensor | None = None,
) -> dict:

    device = seq_logits.device
    n_sequence = int(score_sequence.sum())
    n_geometry = int(score_geometry.sum())

    zero = seq_logits.sum() * 0.0
    sequence = zero
    if n_sequence:
        flat = seq_logits[score_sequence]
        sequence = torch.nn.functional.cross_entropy(
            flat.float(), native_aatype[score_sequence].long(), reduction="mean"
        )

    geometry = zero
    if n_geometry:
        geometry = torch.nn.functional.smooth_l1_loss(
            predicted_ca[score_geometry].float(),
            native_ca[score_geometry].float(),
            beta=0.05, reduction="mean",
        )

    atoms = zero
    n_atoms = 0
    if predicted_atoms is not None and native_atoms is not None and atom_score is not None:
        n_atoms = int(atom_score.sum())
        if n_atoms:
            atoms = torch.nn.functional.smooth_l1_loss(
                predicted_atoms[atom_score].float(), native_atoms[atom_score].float(),
                beta=0.05, reduction="mean",
            )

    total = sequence_weight * sequence + geometry_weight * (geometry + atoms)
    return {
        "total": total,
        "sequence": sequence,
        "geometry": geometry,
        "atoms": atoms,
        "n_sequence": n_sequence,
        "n_geometry": n_geometry,
        "n_atoms": n_atoms,
    }
