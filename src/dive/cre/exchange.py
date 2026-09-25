
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dive.cre.response import QuadraticCandidate

CLASH_WEIGHT = 1.0

OWN_RIDGE = 1e-2

CHI_STEP = 1e-3

D_POLAR, D_APOLAR = 2.8, 3.4

BASIN_OFFSETS_DEG = (0.0, 120.0, -120.0)

def anchored_basins(measured_chi) -> list[tuple[float, ...]]:

    import numpy as np

    measured = tuple(float(x) for x in measured_chi)
    if not measured:
        return [()]
    out = []
    for offset in BASIN_OFFSETS_DEG:
        shifted = list(measured)
        shifted[0] = measured[0] + np.deg2rad(offset)
        out.append(tuple(shifted))
    return out

@dataclass
class PatchCandidate:

    name: str
    chi_centre: tuple[float, ...]
    quadratic: QuadraticCandidate
    value_at_zero: float
    gradient_at_zero: np.ndarray
    curvature: np.ndarray
    best_atom_deviation: float
    clash_count: int
    feasible: bool

def _clash_limit(element: str) -> float:
    return D_POLAR if element in ("N", "O") else D_APOLAR

def build_patch_candidates(
    *,
    residue_xyz: np.ndarray,
    code: str,
    required: list[tuple[int, np.ndarray]],
    environment: np.ndarray,
    environment_elements: list[str],
    chi_centres: list[tuple[float, ...]],
    clash_weight: float = CLASH_WEIGHT,
    own_ridge: float = OWN_RIDGE,
) -> list[PatchCandidate]:

    from dive.ccr.candidate import build_candidate_atoms

    out: list[PatchCandidate] = []
    limits = np.asarray([_clash_limit(e) for e in environment_elements]) if len(environment) else np.zeros(0)

    for index, centre in enumerate(chi_centres):
        base = build_candidate_atoms(residue_xyz, code, tuple(centre))
        slots = [slot for slot, _target in required]
        if any(not np.isfinite(base[slot]).all() for slot in slots):
            continue
        x0 = np.asarray([base[slot] for slot in slots])
        targets = np.asarray([target for _slot, target in required])

        n_chi = len(centre)
        jacobian = np.zeros((len(slots), 3, n_chi))
        for j in range(n_chi):
            stepped = list(centre)
            stepped[j] = centre[j] + CHI_STEP
            moved = build_candidate_atoms(residue_xyz, code, tuple(stepped))
            for a, slot in enumerate(slots):
                jacobian[a, :, j] = (moved[slot] - base[slot]) / CHI_STEP

        residual = x0 - targets
        n_atoms = len(slots)
        A = 2.0 * n_atoms * np.eye(3)
        B = 2.0 * jacobian.sum(axis=0)
        C = 2.0 * np.einsum("akj,akl->jl", jacobian, jacobian)
        a_lin = 2.0 * residual.sum(axis=0)
        d_lin = 2.0 * np.einsum("ak,akj->j", residual, jacobian)
        k_const = float((residual ** 2).sum())

        clash_count = 0
        side_chain = np.asarray([base[slot] for slot in slots])
        if len(environment):
            offsets = side_chain[:, None, :] - environment[None, :, :]
            distances = np.linalg.norm(offsets, axis=-1)
            violating = (distances < limits[None, :]) & (distances > 1e-6)
            clash_count = int(violating.sum())
            if clash_count:
                atom_index, env_index = np.nonzero(violating)
                unit = offsets[atom_index, env_index] / distances[atom_index, env_index][:, None]
                depth = limits[env_index] - distances[atom_index, env_index]
                grad_b = -unit
                grad_u = -np.einsum("pk,pkj->pj", unit, jacobian[atom_index])
                A += 2.0 * clash_weight * (grad_b.T @ grad_b)
                B += 2.0 * clash_weight * (grad_b.T @ grad_u)
                C += 2.0 * clash_weight * (grad_u.T @ grad_u)
                a_lin += 2.0 * clash_weight * (grad_b.T @ depth)
                d_lin += 2.0 * clash_weight * (grad_u.T @ depth)
                k_const += clash_weight * float((depth ** 2).sum())

        C = C + 2.0 * own_ridge * np.eye(n_chi) if n_chi else np.zeros((0, 0))
        if n_chi == 0:

            C = np.array([[1.0]])
            B = np.zeros((3, 1))
            d_lin = np.zeros(1)
        quadratic = QuadraticCandidate(A=A, B=B, C=C, a=a_lin, d=d_lin, k=k_const)
        S, q, _kappa = quadratic.value_quadratic()
        out.append(PatchCandidate(
            name=f"rot{index}",
            chi_centre=tuple(centre),
            quadratic=quadratic,
            value_at_zero=quadratic.value(np.zeros(3)),
            gradient_at_zero=q,
            curvature=S,
            best_atom_deviation=float(np.linalg.norm(residual, axis=1).max()),
            clash_count=clash_count,
            feasible=clash_count == 0,
        ))
    return out

FEATURE_NAMES: tuple[str, ...] = (
    "n_candidates",
    "best_value",
    "value_spread",
    "feasible_fraction",
    "best_response_sensitivity",
    "direction_agreement",
    "predicted_gain",
    "best_atom_deviation",
)
N_FEATURES = len(FEATURE_NAMES)

def _squash(value: float) -> float:

    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(value))

def patch_message(candidates: list[PatchCandidate], *, trust_radius: float = 0.5) -> np.ndarray:

    if not candidates:
        return np.zeros(N_FEATURES)
    values = np.asarray([c.value_at_zero for c in candidates])
    order = np.argsort(values)
    best = candidates[int(order[0])]
    gradients = np.asarray([c.gradient_at_zero for c in candidates])
    best_gradient = gradients[int(order[0])]
    if len(order) > 1:
        second_gradient = gradients[int(order[1])]
        denominator = np.linalg.norm(best_gradient) * np.linalg.norm(second_gradient)
        agreement = float(best_gradient @ second_gradient / denominator) if denominator > 1e-9 else 0.0
    else:
        agreement = 0.0

    step = -best_gradient
    norm = float(np.linalg.norm(step))
    if norm > trust_radius:
        step = step * (trust_radius / norm)
    gain = best.value_at_zero - best.quadratic.value(step)
    return np.asarray([
        _squash(len(candidates) / 10.0),
        _squash(float(values.min()) / 10.0),
        _squash(float(values.max() - values.min()) / 10.0),
        float(np.mean([c.feasible for c in candidates])),
        _squash(float(np.linalg.norm(best_gradient)) / 10.0),
        agreement,
        _squash(float(gain) / 10.0),
        _squash(float(best.best_atom_deviation)),
    ])
