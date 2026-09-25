
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dive.ccr.constraints import bond_constraints, clash_constraints, linearise

@dataclass(frozen=True)
class RepairResult:
    coors: np.ndarray
    max_violation_before: float
    max_violation_after: float
    total_violation_before: float
    total_violation_after: float
    n_constraints_before: int
    n_constraints_after: int
    n_violated_before: int
    n_violated_after: int
    rmsd_moved: float
    max_atom_displacement: float
    iterations: int
    converged: bool
    n_immovable_rows_dropped: int = 0

def fixed_chain_indices(*, n_design: int, n_fixed: int) -> np.ndarray:

    return np.concatenate([
        np.zeros(n_design, dtype=int),
        np.arange(1, n_fixed + 1, dtype=int),
    ])

def restrict_fixed_neighbourhood(
    design: np.ndarray, fixed: np.ndarray, *, margin: float = 12.0
) -> tuple[np.ndarray, np.ndarray]:

    design_points = design.reshape(-1, 3)
    design_points = design_points[np.isfinite(design_points).all(axis=-1)]
    keep = np.zeros(fixed.shape[0], dtype=bool)
    if design_points.size == 0:
        return fixed[keep], keep

    from scipy.spatial import cKDTree

    flat = fixed.reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=-1)
    owner = np.repeat(np.arange(fixed.shape[0]), fixed.shape[1])[finite]
    points = flat[finite]
    if points.size == 0:
        return fixed[keep], keep

    tree = cKDTree(design_points)
    near = tree.query_ball_point(points, r=margin, return_length=True) > 0
    keep[np.unique(owner[near])] = True
    return fixed[keep], keep

def _revalue(constraints, coors):

    values = np.array([
        constraints.signs[k] * (
            float(np.linalg.norm(coors[b[0], b[1]] - coors[a[0], a[1]]))
            - constraints.thresholds[k]
        )
        for k, (a, b) in enumerate(constraints.atom_pairs)
    ]) if constraints.n_constraints else np.zeros(0)
    return type(constraints)(
        constraints.names, values, constraints.atom_pairs,
        constraints.signs, constraints.thresholds, constraints.family,
    )

def _constraints(coors, mask, chain_index):
    return bond_constraints(coors, mask=mask, chain_index=chain_index).concat(
        clash_constraints(coors, mask=mask, chain_index=chain_index)
    )

def _drop_immovable(constraints, movable):

    if constraints.n_constraints == 0:
        return constraints, 0
    keep = np.array([
        bool(movable[a[0]] or movable[b[0]]) for a, b in constraints.atom_pairs
    ])
    dropped = int((~keep).sum())
    if dropped == 0:
        return constraints, 0
    idx = np.flatnonzero(keep)
    return type(constraints)(
        tuple(constraints.names[i] for i in idx),
        constraints.values[idx],
        tuple(constraints.atom_pairs[i] for i in idx),
        constraints.signs[idx],
        constraints.thresholds[idx],
        tuple(constraints.family[i] for i in idx) if constraints.family else (),
    ), dropped

def _summary(constraints) -> tuple[float, float, int, int]:
    if constraints.n_constraints == 0:
        return 0.0, 0.0, 0, 0
    deficit = np.maximum(0.0, -constraints.values)
    return (
        float(deficit.max()),
        float(deficit.sum()),
        int(constraints.n_constraints),
        int((deficit > 0).sum()),
    )

def project_structure(
    coors: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    movable_residues: np.ndarray | None = None,
    step: float = 0.05,
    max_iterations: int = 500,
    max_displacement: float | None = None,
    tolerance: float = 1e-9,
    rebuild_every: int = 1,
) -> RepairResult:

    coors = np.array(coors, dtype=np.float64, copy=True)
    original = coors.copy()
    n = int(coors.shape[0])
    movable = (
        np.ones(n, dtype=bool) if movable_residues is None
        else np.asarray(movable_residues, dtype=bool)
    )

    before_all = _constraints(coors, mask, chain_index)
    before, dropped_rows = _drop_immovable(before_all, movable)
    max_before, total_before, count_before, violated_before = _summary(before)

    best_coors = coors.copy()
    best_max = max_before
    iterations, converged, stalled = 0, False, 0
    current = before
    for iterations in range(1, max_iterations + 1):
        if (iterations - 1) % max(rebuild_every, 1) == 0:
            current, _ = _drop_immovable(_constraints(coors, mask, chain_index), movable)
        else:

            current = _revalue(current, coors)
        if current.n_constraints == 0:
            converged = True
            break
        deficit = np.maximum(0.0, -current.values)
        worst = float(deficit.max())
        if worst < best_max:
            best_max, best_coors = worst, coors.copy()
            stalled = 0
        else:
            stalled += 1
        if worst <= tolerance:
            converged = True
            break
        if stalled > 60:
            break

        atoms = sorted({
            (r, a)
            for pair in current.atom_pairs for (r, a) in pair
            if movable[r]
        })
        if not atoms:
            break
        A, _ = linearise(current, coors, atoms)

        gradient = -2.0 * (A.T @ deficit)
        norm = float(np.linalg.norm(gradient))
        if norm <= 1e-12:
            converged = True
            break
        decay = 1.0 / (1.0 + 0.01 * iterations)
        move = -step * decay * worst * gradient / norm
        for k, (r, a) in enumerate(atoms):
            coors[r, a] += move[3 * k: 3 * k + 3]

        if max_displacement is not None:
            delta = coors - original
            lengths = np.linalg.norm(delta, axis=-1)
            over = lengths > max_displacement
            if over.any():
                scale = np.ones_like(lengths)
                scale[over] = max_displacement / lengths[over]
                coors = original + delta * scale[..., None]

    coors = best_coors

    after, _ = _drop_immovable(_constraints(coors, mask, chain_index), movable)
    max_after, total_after, count_after, violated_after = _summary(after)

    moved = coors - original
    finite = np.isfinite(moved).all(axis=-1)
    lengths = np.linalg.norm(np.where(finite[..., None], moved, 0.0), axis=-1)
    n_atoms = int(finite.sum())
    return RepairResult(
        coors=coors,
        max_violation_before=max_before,
        max_violation_after=max_after,
        total_violation_before=total_before,
        total_violation_after=total_after,
        n_constraints_before=count_before,
        n_constraints_after=count_after,
        n_violated_before=violated_before,
        n_violated_after=violated_after,
        rmsd_moved=float(np.sqrt((lengths[finite] ** 2).mean())) if n_atoms else 0.0,
        max_atom_displacement=float(lengths[finite].max()) if n_atoms else 0.0,
        iterations=iterations,
        converged=converged,
        n_immovable_rows_dropped=dropped_rows,
    )
