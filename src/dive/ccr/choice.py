
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dive.ccr.candidate import build_candidate_atoms, rotamer_samples
from dive.ccr.constraints import linearise
from dive.ccr.exchange import _cap_least_slack, _present_atoms
from dive.ccr.obstruction_batch import obstruction_dual_batch
from dive.ccr.operator import N_ATOM37, build_constraints, constraints_touching

FEASIBLE = "F"
INFEASIBLE = "I"
UNDETERMINED = "U"

FEASIBLE_TOLERANCE = 1e-6

@dataclass(frozen=True)
class ChoiceLabels:

    candidates: tuple[str, ...]
    label: np.ndarray
    obstruction: np.ndarray
    residual_after: np.ndarray
    verified_move_norm: np.ndarray
    scope: dict = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {k: int((self.label == k).sum())
                for k in (FEASIBLE, INFEASIBLE, UNDETERMINED)}

def _true_residual(coors, residue, chain_index, mask, cap,
                   max_clash_constraints=4000,
                   identity_rows_only: bool = False) -> float:

    constraints = build_constraints(coors, mask=mask, chain_index=chain_index,
                                    max_clash_constraints=max_clash_constraints)
    local, _ = constraints_touching(constraints, {residue})
    if local.n_constraints == 0:
        return 0.0
    if identity_rows_only:
        from dive.ccr.exchange import _identity_rows

        rows = _identity_rows(local, residue)
        if rows.size == 0:
            return 0.0
        return float(np.maximum(0.0, -local.values[rows]).max())
    return float(np.maximum(0.0, -local.values).max())

def label_candidates(
    coors: np.ndarray,
    *,
    residue: int,
    current: str,
    candidates: tuple[str, ...],
    radius: float,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    n_rotamers: int = 2,
    solver_iterations: int = 700,
    max_local_constraints: int = 64,
) -> ChoiceLabels:

    from dive.ccr.exchange import scan_candidates

    _report, labels = scan_candidates(
        coors, residue=residue, current=current, candidates=candidates,
        radius=radius, mask=mask, chain_index=chain_index, n_rotamers=n_rotamers,
        solver_iterations=solver_iterations,
        max_local_constraints=max_local_constraints, with_labels=True,
    )
    return labels

def choice_loss(
    logits: torch.Tensor, labels: np.ndarray, *, reduction: str = "mean"
) -> tuple[torch.Tensor, int]:

    labels = np.asarray(labels)
    if labels.shape != tuple(logits.shape):
        raise ValueError(f"labels {labels.shape} must match logits {tuple(logits.shape)}")

    feasible = torch.as_tensor(labels == FEASIBLE, device=logits.device)
    infeasible = torch.as_tensor(labels == INFEASIBLE, device=logits.device)
    scored = feasible.any(-1) & infeasible.any(-1)
    n = int(scored.sum())
    if n == 0:

        safe = torch.where(torch.isfinite(logits), logits,
                           torch.zeros_like(logits))
        return safe.sum() * 0.0, 0

    considered = feasible | infeasible

    safe = torch.where(considered, logits, torch.zeros_like(logits))
    neg_inf = torch.finfo(safe.dtype).min
    all_terms = torch.where(considered, safe, torch.full_like(safe, neg_inf))
    f_terms = torch.where(feasible, safe, torch.full_like(safe, neg_inf))

    loss_per_residue = torch.logsumexp(all_terms, dim=-1) - torch.logsumexp(f_terms, dim=-1)
    total = (loss_per_residue * scored.to(loss_per_residue.dtype)).sum()
    return (total / n if reduction == "mean" else total), n
