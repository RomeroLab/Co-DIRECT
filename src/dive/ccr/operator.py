
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dive.ccr.constraints import (
    ATOM37_C,
    ATOM37_CA,
    ATOM37_N,
    ATOM37_O,
    ConstraintSet,
    bond_constraints,
    clash_constraints,
    linearise,
)
from dive.ccr.obstruction import obstruction_dual

BACKBONE_SLOTS: tuple[int, ...] = (ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O)
N_ATOM37 = 37

def sidechain_slots() -> tuple[int, ...]:

    return tuple(a for a in range(N_ATOM37) if a not in BACKBONE_SLOTS)

@dataclass(frozen=True)
class ActionSet:

    name: str
    value: float
    movable: tuple[tuple[int, int], ...]
    active_constraint_names: tuple[str, ...]
    witness: np.ndarray
    n_constraints: int
    radius: float
    converged: bool
    duality_gap: float | None

def _present_atoms(coors: np.ndarray, residue: int, slots) -> list[tuple[int, int]]:
    return [
        (residue, a) for a in slots
        if a < coors.shape[1] and bool(np.isfinite(coors[residue, a]).all())
    ]

def index_by_residue(constraints: ConstraintSet) -> dict[int, list[int]]:

    out: dict[int, list[int]] = {}
    for row, (a, b) in enumerate(constraints.atom_pairs):
        out.setdefault(a[0], []).append(row)
        if b[0] != a[0]:
            out.setdefault(b[0], []).append(row)
    return out

def constraints_touching(
    constraints: ConstraintSet,
    residues: set[int],
    index: dict[int, list[int]] | None = None,
) -> tuple[ConstraintSet, np.ndarray]:

    if index is not None:
        rows = sorted({r for residue in residues for r in index.get(residue, ())})
        if not rows:
            return ConstraintSet.empty(), np.zeros(0, dtype=int)
        idx = np.array(rows, dtype=int)
    else:
        keep = np.array(
            [a[0] in residues or b[0] in residues for a, b in constraints.atom_pairs],
            dtype=bool,
        )
        if not keep.any():
            return ConstraintSet.empty(), np.zeros(0, dtype=int)
        idx = np.flatnonzero(keep)
    subset = ConstraintSet(
        tuple(constraints.names[i] for i in idx),
        constraints.values[idx],
        tuple(constraints.atom_pairs[i] for i in idx),
        constraints.signs[idx],
        constraints.thresholds[idx],
        tuple(constraints.family[i] for i in idx) if constraints.family else (),
    )
    return subset, idx

def build_constraints(
    coors: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    max_clash_constraints: int | None = None,
) -> ConstraintSet:

    return bond_constraints(coors, mask=mask, chain_index=chain_index).concat(
        clash_constraints(coors, mask=mask, chain_index=chain_index,
                          max_constraints=max_clash_constraints)
    )

def residue_obstructions(
    coors: np.ndarray,
    *,
    residue: int,
    radius: float,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    constraints: ConstraintSet | None = None,
    constraint_index: dict[int, list[int]] | None = None,
    check_primal: bool = True,
    solver_iterations: int = 4000,
    max_local_constraints: int | None = None,
    partners: tuple[int, ...] = (),
) -> dict[str, ActionSet]:

    coors = np.asarray(coors, dtype=np.float64)
    if constraints is None:
        constraints = build_constraints(coors, mask=mask, chain_index=chain_index)

    local, _ = constraints_touching(constraints, {residue}, constraint_index)
    if max_local_constraints is not None and local.n_constraints > max_local_constraints:

        order = np.argsort(local.values)[:max_local_constraints]
        local = ConstraintSet(
            tuple(local.names[i] for i in order),
            local.values[order],
            tuple(local.atom_pairs[i] for i in order),
            local.signs[order],
            local.thresholds[order],
            tuple(local.family[i] for i in order) if local.family else (),
        )
    movers = (residue,) + tuple(partners)

    plans = {
        "rotamer": [a for r in movers for a in _present_atoms(coors, r, sidechain_slots())],
        "backbone": [a for r in movers for a in _present_atoms(coors, r, BACKBONE_SLOTS)],
        "joint": [a for r in movers for a in _present_atoms(coors, r, range(N_ATOM37))],
    }

    if local.n_constraints and float(local.values.min()) >= 0.0:
        return {
            name: ActionSet(name, 0.0, tuple(movable), (),
                            np.zeros(local.n_constraints), local.n_constraints,
                            radius, True, 0.0)
            for name, movable in plans.items()
        }

    out: dict[str, ActionSet] = {}
    for name, movable in plans.items():
        if local.n_constraints == 0 or not movable:
            out[name] = ActionSet(
                name, 0.0 if local.n_constraints == 0 else float(max(0.0, -local.values.min())),
                tuple(movable), (), np.zeros(local.n_constraints),
                local.n_constraints, radius, True, 0.0,
            )
            continue
        A, b = linearise(local, coors, movable)
        result = obstruction_dual(A, b, radius=radius, check_primal=check_primal,
                                  max_iter=solver_iterations)
        active = tuple(local.names[i] for i in result.active_constraints)
        out[name] = ActionSet(
            name, float(result.value), tuple(movable), active, result.witness,
            local.n_constraints, radius, result.converged, result.duality_gap,
        )
    return out

def candidate_obstructions(
    coors: np.ndarray,
    *,
    residue: int,
    current: str,
    candidates: tuple[str, ...],
    radius: float,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    n_rotamers: int = 3,
    solver_iterations: int = 700,
    max_local_constraints: int | None = 256,
) -> dict[str, dict]:

    from dive.ccr.candidate import (
        build_candidate_atoms,
        rotamer_samples,
        substitution_effect,
    )

    coors = np.asarray(coors, dtype=np.float64)
    out: dict[str, dict] = {}
    for code in candidates:
        best, best_chi, searched = float("inf"), None, 0
        for chi in rotamer_samples(code, n=n_rotamers):
            trial = coors.copy()
            trial[residue] = build_candidate_atoms(coors[residue], code, chi=chi)
            constraints = build_constraints(trial, mask=mask, chain_index=chain_index)
            actions = residue_obstructions(
                trial, residue=residue, radius=radius, mask=mask,
                chain_index=chain_index, constraints=constraints,
                check_primal=False, solver_iterations=solver_iterations,
                max_local_constraints=max_local_constraints,
            )
            searched += 1
            value = float(actions["joint"].value)
            if value < best:
                best, best_chi = value, chi
        effect = substitution_effect(current, code)
        out[code] = {
            "obstruction": best if np.isfinite(best) else None,
            "chi": best_chi,
            "n_rotamers_searched": searched,
            **effect,
        }
    return out
