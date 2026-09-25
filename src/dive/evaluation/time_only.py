
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True, slots=True)
class TimeOnlyHoldout:

    per_example: tuple[float, ...]

    selected_by_fold: tuple[dict[int, str], ...]

    fallback_examples: int

def time_only_holdout(
    per_residue_losses: Mapping[str, Tensor],
    *,
    mask: Tensor,
    time_bins: Tensor,
    groups: Sequence[str],
) -> TimeOnlyHoldout:

    if not per_residue_losses:
        raise ValueError("at least one fixed regime is required")
    names = tuple(sorted(per_residue_losses))
    shape = tuple(mask.shape)
    n_examples = shape[0]
    if len(groups) != n_examples:
        raise ValueError(
            f"{len(groups)} group labels for {n_examples} examples; every example "
            "needs a parent or the held-out split is not what it claims"
        )
    if tuple(time_bins.shape) != (n_examples,):
        raise ValueError("time_bins must carry one bin per example")

    tensors = []
    for name in names:
        value = per_residue_losses[name]
        if tuple(value.shape) != shape:
            raise ValueError(
                f"regime {name!r} has shape {tuple(value.shape)}, expected {shape}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"regime {name!r} contains non-finite loss terms")
        tensors.append(value * mask)
    example_losses = torch.stack(tensors, dim=0).sum(dim=-1)

    parents = sorted(set(groups))
    if len(parents) < 2:
        raise ValueError(
            f"{len(parents)} parent(s); a held-out schedule needs at least 2, "
            "otherwise selection and scoring share the same targets"
        )

    device = example_losses.device
    scored = torch.empty(n_examples, dtype=example_losses.dtype, device=device)
    chosen_per_fold: list[dict[int, str]] = []
    fallbacks = 0
    index = torch.arange(n_examples, device=device)

    for parent in parents:
        held = torch.tensor([g == parent for g in groups], device=device)
        select = ~held

        fallback = int(example_losses[:, select].mean(dim=1).argmin())
        chosen: dict[int, str] = {}
        for raw_bin in torch.unique(time_bins[select], sorted=True):
            in_bin = select & (time_bins == raw_bin)
            chosen[int(raw_bin)] = names[
                int(example_losses[:, in_bin].mean(dim=1).argmin())
            ]
        chosen_per_fold.append(chosen)

        for position in index[held].tolist():
            bin_id = int(time_bins[position])
            if bin_id in chosen:
                regime = names.index(chosen[bin_id])
            else:
                regime = fallback
                fallbacks += 1
            scored[position] = example_losses[regime, position]

    return TimeOnlyHoldout(
        per_example=tuple(float(v) for v in scored),
        selected_by_fold=tuple(chosen_per_fold),
        fallback_examples=fallbacks,
    )
