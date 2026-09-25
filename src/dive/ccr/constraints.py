
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_CB, ATOM37_O = 0, 1, 2, 3, 4

BOND_SPECS: tuple[tuple[str, int, int, float, float, bool], ...] = (
    ("N_CA", ATOM37_N, ATOM37_CA, 1.35, 1.55, False),
    ("CA_C", ATOM37_CA, ATOM37_C, 1.45, 1.65, False),
    ("C_O", ATOM37_C, ATOM37_O, 1.15, 1.35, False),
    ("CA_CB", ATOM37_CA, ATOM37_CB, 1.43, 1.63, False),
    ("C_N", ATOM37_C, ATOM37_N, 1.20, 1.45, True),
    ("CA_CA", ATOM37_CA, ATOM37_CA, 3.50, 4.20, True),
)

_ATOM37_ELEMENTS = (
    "N", "C", "C", "C", "O", "C", "C", "C", "O", "O", "S", "C", "C", "C",
    "N", "N", "O", "O", "S", "C", "C", "C", "C", "N", "N", "N", "O", "O",
    "C", "N", "N", "O", "C", "C", "C", "N", "O",
)
_VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

CLASH_TOLERANCE = 0.40

CLASH_CUTOFF = 5.0

CLASH_MIN_SEQUENCE_SEPARATION = 1

@dataclass(frozen=True)
class ConstraintSet:

    names: tuple[str, ...]
    values: np.ndarray
    atom_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    signs: np.ndarray
    thresholds: np.ndarray
    family: tuple[str, ...] = ()

    @property
    def n_constraints(self) -> int:
        return int(self.values.size)

    @property
    def violated(self) -> np.ndarray:
        return self.values < 0.0

    @classmethod
    def empty(cls) -> "ConstraintSet":
        return cls((), np.zeros(0), (), np.zeros(0), np.zeros(0), ())

    def concat(self, other: "ConstraintSet") -> "ConstraintSet":
        return ConstraintSet(
            self.names + other.names,
            np.concatenate([self.values, other.values]),
            self.atom_pairs + other.atom_pairs,
            np.concatenate([self.signs, other.signs]),
            np.concatenate([self.thresholds, other.thresholds]),
            self.family + other.family,
        )

def _distance(coors, a, b) -> float:
    return float(np.linalg.norm(coors[a[0], a[1]] - coors[b[0], b[1]]))

def _present(coors, residue, atom) -> bool:
    return bool(np.isfinite(coors[residue, atom]).all())

def bond_constraints(
    coors: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
) -> ConstraintSet:

    coors = np.asarray(coors, dtype=np.float64)
    n = int(coors.shape[0])
    keep = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    chains = None if chain_index is None else np.asarray(chain_index)

    names, values, pairs, signs, thresholds, families = [], [], [], [], [], []
    for name, atom_a, atom_b, low, high, inter in BOND_SPECS:
        for i in range(n):
            j = i + 1 if inter else i
            if j >= n or not keep[i] or not keep[j]:
                continue
            if inter and chains is not None and chains[i] != chains[j]:
                continue
            if not (_present(coors, i, atom_a) and _present(coors, j, atom_b)):
                continue
            a, b = (i, atom_a), (j, atom_b)
            d = _distance(coors, a, b)
            for sign, threshold in ((+1.0, low), (-1.0, high)):
                names.append(f"{name}[{i}]{'lo' if sign > 0 else 'hi'}")

                values.append(sign * (d - threshold))
                pairs.append((a, b))
                signs.append(sign)
                thresholds.append(threshold)
                families.append("bond")
    if not values:
        return ConstraintSet.empty()
    return ConstraintSet(tuple(names), np.array(values), tuple(pairs),
                         np.array(signs), np.array(thresholds), tuple(families))

def _bonded_pairs(n: int, chains=None) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    out = set()
    for name, atom_a, atom_b, _low, _high, inter in BOND_SPECS:
        for i in range(n):
            j = i + 1 if inter else i
            if j >= n:
                continue
            if inter and chains is not None and chains[i] != chains[j]:
                continue
            out.add(tuple(sorted(((i, atom_a), (j, atom_b)))))
    return out

def clash_constraints(
    coors: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    cutoff: float = CLASH_CUTOFF,
    tolerance: float = CLASH_TOLERANCE,
    max_constraints: int | None = None,
) -> ConstraintSet:

    coors = np.asarray(coors, dtype=np.float64)
    n = int(coors.shape[0])
    keep = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    chains = None if chain_index is None else np.asarray(chain_index)

    present = [
        (i, a) for i in range(n) if keep[i]
        for a in range(min(37, coors.shape[1])) if _present(coors, i, a)
    ]
    if not present:
        return ConstraintSet.empty()

    positions = np.array([coors[i, a] for i, a in present])
    bonded = _bonded_pairs(n, chains)

    names, values, pairs, signs, thresholds, families = [], [], [], [], [], []

    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    candidate_pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    if candidate_pairs.size == 0:
        return ConstraintSet.empty()

    left = positions[candidate_pairs[:, 0]]
    right = positions[candidate_pairs[:, 1]]
    distances = np.linalg.norm(left - right, axis=-1)

    if max_constraints is not None and candidate_pairs.shape[0] > max_constraints:

        keep_n = min(candidate_pairs.shape[0], max(4 * max_constraints, max_constraints))
        order = np.argsort(distances)[:keep_n]
        candidate_pairs = candidate_pairs[order]
        distances = distances[order]

    for (u, v), distance in zip(candidate_pairs, distances):
        if distance <= 0.0:
            continue
        a, b = present[int(u)], present[int(v)]
        if a[0] == b[0]:
            continue
        same_chain = chains is None or chains[a[0]] == chains[b[0]]
        if (same_chain and abs(a[0] - b[0]) <= CLASH_MIN_SEQUENCE_SEPARATION
                and tuple(sorted((a, b))) in bonded):
            continue
        if same_chain and abs(a[0] - b[0]) <= CLASH_MIN_SEQUENCE_SEPARATION and (
            a[1] in (ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O)
            and b[1] in (ATOM37_N, ATOM37_CA, ATOM37_C, ATOM37_O)
        ):
            continue
        radius = (
            _VDW.get(_ATOM37_ELEMENTS[a[1]], 1.7)
            + _VDW.get(_ATOM37_ELEMENTS[b[1]], 1.7)
            - tolerance
        )
        names.append(f"clash[{a[0]}.{a[1]}|{b[0]}.{b[1]}]")
        values.append(float(distance) - radius)
        pairs.append((a, b))
        signs.append(+1.0)
        thresholds.append(radius)
        families.append("clash")

    if not values:
        return ConstraintSet.empty()
    if max_constraints is not None and len(values) > max_constraints:
        order = np.argsort(np.asarray(values))[:max_constraints]
        names = [names[i] for i in order]
        pairs = [pairs[i] for i in order]
        signs = [signs[i] for i in order]
        thresholds = [thresholds[i] for i in order]
        families = [families[i] for i in order]
        values = [values[i] for i in order]
    return ConstraintSet(tuple(names), np.array(values), tuple(pairs),
                         np.array(signs), np.array(thresholds), tuple(families))

def linearise(
    constraints: ConstraintSet,
    coors: np.ndarray,
    movable: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:

    coors = np.asarray(coors, dtype=np.float64)
    index = {key: k for k, key in enumerate(movable)}
    m, n = constraints.n_constraints, 3 * len(movable)
    A = np.zeros((m, n))
    if m == 0:
        return A, np.zeros(0)

    for row, ((a, b), sign) in enumerate(zip(constraints.atom_pairs, constraints.signs)):
        pa, pb = coors[a[0], a[1]], coors[b[0], b[1]]
        delta = pa - pb
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            continue
        unit = delta / distance
        if a in index:
            A[row, 3 * index[a]: 3 * index[a] + 3] = sign * unit
        if b in index:
            A[row, 3 * index[b]: 3 * index[b] + 3] = -sign * unit
    return A, -constraints.values.copy()
