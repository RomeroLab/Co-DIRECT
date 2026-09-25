
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dive.ccr.exchange import (
    CANDIDATE_FEATURES,
    ccr_candidate_features,
    control_candidate_features,
    scan_candidates,
)
from dive.ccr.message import FEATURE_NAMES, MessageMode, residue_message
from dive.ccr.patch import local_patch

RESTYPES: tuple[str, ...] = tuple("ARNDCQEGHILKMFPSTWYV")

CHEMICAL_ANCHORS: tuple[str, ...] = ("G", "W")

def candidate_slots(seq_logits_row: np.ndarray, *, n_slots: int) -> np.ndarray:

    order = list(np.argsort(seq_logits_row)[::-1])
    anchors = [RESTYPES.index(code) for code in CHEMICAL_ANCHORS]
    chosen = [int(order[0])]
    for index in anchors:
        if index not in chosen and len(chosen) < n_slots:
            chosen.append(int(index))
    for index in order[1:]:
        if len(chosen) >= n_slots:
            break
        if int(index) not in chosen:
            chosen.append(int(index))

    head = [chosen[0]] + [c for c in chosen if c not in anchors and c != chosen[0]]
    tail = [c for c in chosen if c in anchors and c != chosen[0]]
    return np.array((head + tail)[:n_slots], dtype=int)

@dataclass
class ExchangeInputs:

    message: np.ndarray
    cand_index: np.ndarray
    cand_feat: np.ndarray
    choice_labels: np.ndarray
    stats: dict

    def to_torch(self, device="cpu"):
        import torch

        return {
            "ccr_message": torch.as_tensor(self.message, dtype=torch.float32,
                                           device=device)[None],
            "ccr_cand_index": torch.as_tensor(self.cand_index, dtype=torch.long,
                                              device=device)[None],
            "ccr_cand_feat": torch.as_tensor(self.cand_feat, dtype=torch.float32,
                                             device=device)[None],
        }

def residue_exchange_inputs(
    coors: np.ndarray,
    *,
    seq_logits: np.ndarray,
    residues: np.ndarray,
    arm: str,
    radius: float,
    time_fraction: float,
    n_slots: int = 6,
    chain_index: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    constraints=None,
    constraint_index=None,
    n_design: int | None = None,
    message_mode: MessageMode = MessageMode.FULL,
    n_rotamers: int = 2,
    solver_iterations: int = 700,
    max_local_constraints: int = 64,
    patch_margin: float = 8.0,
    patch_max_residues: int = 48,
    max_clash_constraints: int = 4000,
    time_budget: float | None = 60.0,
    with_choice_labels: bool = True,
) -> ExchangeInputs:

    if arm not in ("ccr", "control"):
        raise ValueError(f"arm must be 'ccr' or 'control', got {arm!r}")
    coors = np.asarray(coors, dtype=np.float64)
    seq_logits = np.asarray(seq_logits, dtype=np.float64)
    rows = n_design if n_design is not None else int(coors.shape[0])

    message = np.zeros((rows, len(FEATURE_NAMES)))
    cand_index = np.full((rows, n_slots), -1, dtype=np.int64)
    cand_feat = np.zeros((rows, n_slots, len(CANDIDATE_FEATURES)))
    labels = np.full((rows, n_slots), "U", dtype="<U1")
    stats = {
        "n_residues_messaged": 0, "n_residues_scanned": 0,
        "n_candidates": 0, "counts": {"F": 0, "I": 0, "U": 0},
        "arm": arm, "radius": float(radius), "n_slots": int(n_slots),
        "n_rotamers": int(n_rotamers), "solver_iterations": int(solver_iterations),
        "max_local_constraints": int(max_local_constraints),
        "patch_margin": float(patch_margin),
        "patch_max_residues": int(patch_max_residues),
        "with_choice_labels": bool(with_choice_labels),
        "time_budget": time_budget,
        "budget_exhausted": False,
        "n_patches_truncated": 0,
        "scan_seconds": 0.0,
    }
    started = time.perf_counter()

    residues = np.asarray(residues, dtype=int).ravel()
    if residues.size == 0:
        return ExchangeInputs(message, cand_index, cand_feat, labels, stats)

    build = ccr_candidate_features if arm == "ccr" else control_candidate_features

    for raw in residues:
        residue = int(raw)
        if residue < 0 or residue >= rows:
            continue

        if time_budget is not None and (time.perf_counter() - started) > time_budget:
            stats["budget_exhausted"] = True
            break
        message[residue] = residue_message(
            coors, residue=residue, radius=radius,
            time_fraction=time_fraction, mode=message_mode,
            mask=mask, chain_index=chain_index, constraints=constraints,
            constraint_index=constraint_index,
            solver_iterations=solver_iterations,
            max_local_constraints=max_local_constraints,
        )
        stats["n_residues_messaged"] += 1

        order = candidate_slots(seq_logits[residue], n_slots=n_slots)
        codes = tuple(RESTYPES[int(a)] for a in order)
        current = codes[0]

        patch = local_patch(coors, residue=residue, margin=patch_margin,
                            chain_index=chain_index,
                            max_residues=patch_max_residues)
        stats["n_patches_truncated"] += int(patch.truncated)
        report, verdict = scan_candidates(
            patch.coors, residue=patch.centre, current=current, candidates=codes,
            radius=radius, chain_index=patch.chain_index, n_rotamers=n_rotamers,
            solver_iterations=solver_iterations,
            max_local_constraints=max_local_constraints,
            max_clash_constraints=max_clash_constraints,
            with_labels=with_choice_labels,
        )
        cand_index[residue, : len(codes)] = order
        cand_feat[residue] = build(report, n_slots=n_slots)
        stats["n_residues_scanned"] += 1
        stats["n_candidates"] += len(codes)

        if verdict is not None:
            labels[residue, : len(codes)] = verdict.label
            for key in ("F", "I", "U"):
                stats["counts"][key] += int((verdict.label == key).sum())

    stats["scan_seconds"] = time.perf_counter() - started
    return ExchangeInputs(message, cand_index, cand_feat, labels, stats)
