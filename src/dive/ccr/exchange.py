
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dive.ccr.candidate import (
    build_candidate_atoms,
    rotamer_samples,
    substitution_effect,
)
import time

from dive.ccr.constraints import ConstraintSet, linearise
from dive.ccr.obstruction_batch import obstruction_dual_batch
from dive.ccr.operator import (
    BACKBONE_SLOTS,
    N_ATOM37,
    build_constraints,
    constraints_touching,
    sidechain_slots,
)

CANDIDATE_FEATURES: tuple[str, ...] = (
    "mech_0", "mech_1", "mech_2", "mech_3", "mech_4", "mech_5",
    "atoms_added", "atoms_removed", "is_truncation",
    "n_local_constraints", "present",
)

FEASIBLE_TOLERANCE = 1e-6

@dataclass(frozen=True)
class ExchangeReport:

    residue: int
    current: str
    candidates: tuple[str, ...]
    obstruction: np.ndarray
    obstruction_rotamer: np.ndarray
    obstruction_backbone: np.ndarray
    atoms_added: np.ndarray
    atoms_removed: np.ndarray
    is_truncation: np.ndarray
    n_local_constraints: np.ndarray

    clash_pairs: np.ndarray
    overlap_depth: np.ndarray
    worst_violation: np.ndarray

    witness_bond_fraction: np.ndarray
    witness_clash_fraction: np.ndarray
    n_rotamers_searched: int
    radius: float
    max_local_constraints: int

    @property
    def feasible(self) -> np.ndarray:
        return self.obstruction <= FEASIBLE_TOLERANCE

    @property
    def best_margin(self) -> float:

        if self.obstruction.size < 2:
            return 0.0
        try:
            here = self.candidates.index(self.current)
        except ValueError:
            return 0.0
        others = np.delete(self.obstruction, here)
        return float(self.obstruction[here] - others.min())

def _squash(value, scale: float = 0.5):
    return np.tanh(np.asarray(value, dtype=np.float64) / scale)

def _cap_least_slack(local: ConstraintSet, cap: int) -> ConstraintSet:

    if cap is None or local.n_constraints <= cap:
        return local
    order = np.argsort(local.values)[:cap]
    return ConstraintSet(
        tuple(local.names[i] for i in order),
        local.values[order],
        tuple(local.atom_pairs[i] for i in order),
        local.signs[order],
        local.thresholds[order],
        tuple(local.family[i] for i in order) if local.family else (),
    )

def _present_atoms(coors: np.ndarray, residue: int, slots) -> list[tuple[int, int]]:
    return [
        (residue, a) for a in slots
        if a < coors.shape[1] and bool(np.isfinite(coors[residue, a]).all())
    ]

def _raw_physics(local: ConstraintSet) -> tuple[float, float, float]:

    if local.n_constraints == 0:
        return 0.0, 0.0, 0.0
    deficit = np.maximum(0.0, -local.values)
    clash_rows = np.array(
        [f == "clash" for f in local.family] if local.family
        else [n.startswith("clash") for n in local.names]
    )
    pairs = float((deficit[clash_rows] > 0).sum()) if clash_rows.any() else 0.0
    return pairs, float(deficit.sum()), float(deficit.max())

def candidate_exchange(coors: np.ndarray, **kwargs) -> ExchangeReport:

    kwargs.pop("with_labels", None)
    report, _labels = scan_candidates(coors, with_labels=False, **kwargs)
    return report

def scan_candidates(
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
    max_clash_constraints: int | None = None,
    with_labels: bool = False,
):

    from dive.ccr.choice import (
        FEASIBLE, FEASIBLE_TOLERANCE as LABEL_TOLERANCE, INFEASIBLE,
        UNDETERMINED, ChoiceLabels, _true_residual,
    )

    coors = np.asarray(coors, dtype=np.float64)
    n_cand = len(candidates)
    if n_cand == 0:
        empty = np.zeros(0)
        report = ExchangeReport(
            residue, current, (), empty, empty, empty, empty, empty,
            np.zeros(0, dtype=bool), np.zeros(0, dtype=int),
            empty, empty, empty, empty, empty, 0, float(radius),
            int(max_local_constraints),
        )
        labels = ChoiceLabels((), np.zeros(0, dtype="<U1"), empty, empty, empty,
                              _label_scope(radius, n_rotamers, max_local_constraints))
        return report, (labels if with_labels else None)

    trial_clash_cap = (4000 if max_clash_constraints is None
                       else int(max_clash_constraints))

    action_names = ("rotamer", "backbone", "joint")
    problems: list[tuple[np.ndarray, np.ndarray]] = []
    keys: list[tuple[int, int, str]] = []
    per_trial_local: dict[tuple[int, int], ConstraintSet] = {}
    per_trial_coors: dict[tuple[int, int], np.ndarray] = {}
    per_trial_movable: dict[tuple[int, int], list] = {}

    searched = 0
    for ci, code in enumerate(candidates):
        for ri, chi in enumerate(rotamer_samples(code, n=n_rotamers)):
            trial = coors.copy()
            trial[residue] = build_candidate_atoms(coors[residue], code, chi=chi)
            constraints = build_constraints(
                trial, mask=mask, chain_index=chain_index,
                max_clash_constraints=trial_clash_cap,
            )
            local, _ = constraints_touching(constraints, {residue})
            local = _cap_least_slack(local, max_local_constraints)
            per_trial_local[(ci, ri)] = local
            per_trial_coors[(ci, ri)] = trial
            searched += 1
            plans = {
                "rotamer": _present_atoms(trial, residue, sidechain_slots()),
                "backbone": _present_atoms(trial, residue, BACKBONE_SLOTS),
                "joint": _present_atoms(trial, residue, range(N_ATOM37)),
            }
            per_trial_movable[(ci, ri)] = plans["joint"]
            for action in action_names:
                movable = plans[action]
                if local.n_constraints == 0 or not movable:
                    problems.append((np.zeros((0, 3)), np.zeros(0)))
                else:
                    problems.append(linearise(local, trial, movable))
                keys.append((ci, ri, action))

    values, witnesses = _solve_padded(problems, radius=radius,
                                      solver_iterations=solver_iterations)

    best = {a: np.full(n_cand, np.inf) for a in action_names}
    best_rotamer = np.zeros(n_cand, dtype=int)
    bond_fraction = np.zeros(n_cand)
    clash_fraction = np.zeros(n_cand)
    joint_row: dict[tuple[int, int], int] = {}
    for index, (ci, ri, action) in enumerate(keys):
        if action == "joint":
            joint_row[(ci, ri)] = index
        if values[index] < best[action][ci]:
            best[action][ci] = values[index]
            if action == "joint":
                best_rotamer[ci] = ri
                bond_fraction[ci], clash_fraction[ci] = _witness_families(
                    per_trial_local[(ci, ri)], witnesses[index]
                )

    added = np.zeros(n_cand)
    removed = np.zeros(n_cand)
    truncation = np.zeros(n_cand, dtype=bool)
    counts = np.zeros(n_cand, dtype=int)
    pairs = np.zeros(n_cand)
    depth = np.zeros(n_cand)
    worst = np.zeros(n_cand)
    for ci, code in enumerate(candidates):
        effect = substitution_effect(current, code)
        added[ci] = effect["atoms_added"]
        removed[ci] = effect["atoms_removed"]
        truncation[ci] = effect["is_truncation"]
        local = per_trial_local[(ci, int(best_rotamer[ci]))]
        counts[ci] = local.n_constraints
        pairs[ci], depth[ci], worst[ci] = _raw_physics(local)

    for action in action_names:
        best[action] = np.where(np.isfinite(best[action]), best[action], 0.0)

    report = ExchangeReport(
        residue=residue, current=current, candidates=tuple(candidates),
        obstruction=best["joint"],
        obstruction_rotamer=best["rotamer"],
        obstruction_backbone=best["backbone"],
        atoms_added=added, atoms_removed=removed, is_truncation=truncation,
        n_local_constraints=counts,
        clash_pairs=pairs, overlap_depth=depth, worst_violation=worst,
        witness_bond_fraction=bond_fraction, witness_clash_fraction=clash_fraction,
        n_rotamers_searched=searched, radius=float(radius),
        max_local_constraints=int(max_local_constraints),
    )
    if not with_labels:
        return report, None

    label = np.full(n_cand, UNDETERMINED, dtype="<U1")
    residual = np.full(n_cand, np.inf)
    move_norm = np.zeros(n_cand)
    testable = np.zeros(n_cand, dtype=bool)
    identity_obstruction = np.zeros(n_cand)
    n_rotamer_slots = max(
        (ri for (_ci, ri) in per_trial_local), default=0
    ) + 1
    for ci in range(n_cand):
        for ri in range(n_rotamer_slots):
            if (ci, ri) not in per_trial_local:
                continue
            local = per_trial_local[(ci, ri)]
            trial = per_trial_coors[(ci, ri)]
            movable = per_trial_movable[(ci, ri)]

            rows = _identity_rows(local, residue)
            if rows.size == 0:

                continue
            testable[ci] = True
            rest = float(np.maximum(0.0, -local.values[rows]).max())
            if rest < residual[ci]:
                residual[ci], move_norm[ci] = rest, 0.0
            if rest <= LABEL_TOLERANCE:
                label[ci] = FEASIBLE
                break

            if local.n_constraints == 0 or not movable:
                continue

            A_all, b_all = linearise(local, trial, movable)
            A_id, b_id = A_all[rows], b_all[rows]
            certified = obstruction_dual_batch(
                A_id[None], b_id[None], np.ones((1, rows.size), dtype=bool),
                radius=radius, max_iter=solver_iterations,
            )
            identity_obstruction[ci] = max(
                identity_obstruction[ci], float(certified.value[0]))
            direction = A_id.T @ certified.witness[0]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            step = radius * direction / norm
            for move in (step, 0.5 * step):
                moved = trial.copy()
                for k, (r, a) in enumerate(movable):
                    moved[r, a] = trial[r, a] + move[3 * k: 3 * k + 3]
                got = _true_residual(moved, residue, chain_index, mask,
                                     max_local_constraints,
                                     max_clash_constraints=trial_clash_cap,
                                     identity_rows_only=True)
                if got < residual[ci]:
                    residual[ci] = got
                    move_norm[ci] = float(np.linalg.norm(move))
                if got <= LABEL_TOLERANCE:
                    label[ci] = FEASIBLE
                    break
            if label[ci] == FEASIBLE:
                break

    for ci in range(n_cand):
        if label[ci] == FEASIBLE:
            continue
        if not testable[ci]:
            label[ci] = UNDETERMINED
            continue

        label[ci] = (INFEASIBLE if identity_obstruction[ci] > LABEL_TOLERANCE
                     else UNDETERMINED)

    labels = ChoiceLabels(
        tuple(candidates), label, identity_obstruction,
        np.where(np.isfinite(residual), residual, 0.0), move_norm,
        _label_scope(radius, n_rotamers, max_local_constraints),
    )
    return report, labels

def _label_scope(radius, n_rotamers, max_local_constraints) -> dict:
    return {
        "action_set": "joint",
        "linearised": True,
        "rows": "identity_attributable",
        "radius": float(radius),
        "n_rotamers_searched": int(n_rotamers),
        "max_local_constraints": int(max_local_constraints),
        "note": "I is infeasibility in this linearisation over this action set, "
                "restricted to constraint rows an atom of the candidate's own "
                "side chain participates in. Rows among BACKBONE atoms alone are "
                "identical for every candidate -- on a real training state at "
                "CA-CA validity 0.616 they made all 48 candidates infeasible at "
                "all 8 scanned residues -- so they are carried by the geometry "
                "message, not attributed to the identity choice. Not a claim of "
                "global or physical impossibility.",
    }

def _identity_rows(local: ConstraintSet, residue: int) -> np.ndarray:

    if local.n_constraints == 0:
        return np.zeros(0, dtype=int)
    side = set(sidechain_slots())
    keep = [
        k for k, (a, b) in enumerate(local.atom_pairs)
        if (a[0] == residue and a[1] in side) or (b[0] == residue and b[1] in side)
    ]
    return np.array(keep, dtype=int)

def _witness_families(local: ConstraintSet, witness: np.ndarray) -> tuple[float, float]:

    if local.n_constraints == 0 or witness.size == 0:
        return 0.0, 0.0
    active = np.flatnonzero(witness[: local.n_constraints] > 1e-6)
    if active.size == 0:
        return 0.0, 0.0
    families = [local.names[i].split("[")[0] for i in active]
    total = len(families)
    clash = sum(1 for f in families if f == "clash")
    return (total - clash) / total, clash / total

def _solve_padded(problems, *, radius: float, solver_iterations: int):

    if not problems:
        return np.zeros(0), np.zeros((0, 0))
    m = max(int(b.size) for _A, b in problems)
    n = max(int(A.shape[1]) if A.ndim == 2 else 0 for A, _b in problems)
    k = len(problems)
    if m == 0 or n == 0:
        return np.zeros(k), np.zeros((k, max(m, 1)))
    A_pad = np.zeros((k, m, n))
    b_pad = np.zeros((k, m))
    valid = np.zeros((k, m), dtype=bool)
    for i, (A, b) in enumerate(problems):
        if b.size == 0 or A.size == 0:
            continue
        A_pad[i, : A.shape[0], : A.shape[1]] = A
        b_pad[i, : b.size] = b
        valid[i, : b.size] = True
    result = obstruction_dual_batch(A_pad, b_pad, valid, radius=radius,
                                    max_iter=solver_iterations)
    return result.value, result.witness

def _candidate_obstruction_reference(
    coors: np.ndarray, *, residue: int, candidates: tuple[str, ...], radius: float,
    n_rotamers: int = 2, max_local_constraints: int = 64,
    solver_iterations: int = 700,
) -> np.ndarray:

    from dive.ccr.obstruction import obstruction_dual

    coors = np.asarray(coors, dtype=np.float64)
    out = []
    for code in candidates:
        best = np.inf
        for chi in rotamer_samples(code, n=n_rotamers):
            trial = coors.copy()
            trial[residue] = build_candidate_atoms(coors[residue], code, chi=chi)
            local, _ = constraints_touching(build_constraints(trial), {residue})
            local = _cap_least_slack(local, max_local_constraints)
            movable = _present_atoms(trial, residue, range(N_ATOM37))
            if local.n_constraints == 0 or not movable:
                best = min(best, 0.0)
                continue
            A, b = linearise(local, trial, movable)
            value = obstruction_dual(A, b, radius=radius, check_primal=False,
                                     max_iter=solver_iterations).value
            best = min(best, float(value))
        out.append(best if np.isfinite(best) else 0.0)
    return np.array(out)

def _pack(report: ExchangeReport, mechanism: np.ndarray, n_slots: int) -> np.ndarray:

    block = np.zeros((n_slots, len(CANDIDATE_FEATURES)))
    n = min(len(report.candidates), n_slots)
    if n == 0:
        return block
    block[:n, 0:6] = mechanism[:n]
    block[:n, 6] = report.atoms_added[:n] / 10.0
    block[:n, 7] = report.atoms_removed[:n] / 10.0
    block[:n, 8] = report.is_truncation[:n].astype(float)
    block[:n, 9] = np.tanh(report.n_local_constraints[:n] / 64.0)
    block[:n, 10] = 1.0
    return block

def ccr_candidate_features(report: ExchangeReport, *, n_slots: int = 6) -> np.ndarray:

    n = min(len(report.candidates), n_slots)
    mechanism = np.zeros((max(n, 0), 6))
    if n:
        joint = report.obstruction[:n]
        mechanism[:, 0] = _squash(joint)
        mechanism[:, 1] = _squash(report.obstruction_rotamer[:n] - joint)
        mechanism[:, 2] = _squash(report.obstruction_backbone[:n] - joint)
        mechanism[:, 3] = (joint <= FEASIBLE_TOLERANCE).astype(float)
        mechanism[:, 4] = report.witness_bond_fraction[:n]
        mechanism[:, 5] = report.witness_clash_fraction[:n]
    return _pack(report, mechanism, n_slots)

def control_candidate_features(report: ExchangeReport, *, n_slots: int = 6) -> np.ndarray:

    n = min(len(report.candidates), n_slots)
    mechanism = np.zeros((max(n, 0), 6))
    if n:
        mechanism[:, 0] = np.tanh(report.clash_pairs[:n] / 8.0)
        mechanism[:, 1] = _squash(report.overlap_depth[:n], scale=2.0)
        mechanism[:, 2] = _squash(report.worst_violation[:n])
        mechanism[:, 3] = (report.worst_violation[:n] <= FEASIBLE_TOLERANCE).astype(float)
        mechanism[:, 4] = np.tanh(report.n_local_constraints[:n] / 32.0)
        mechanism[:, 5] = (report.clash_pairs[:n] > 0).astype(float)
    return _pack(report, mechanism, n_slots)
