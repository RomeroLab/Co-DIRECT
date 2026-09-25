
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ATOM37_N, ATOM37_CA, ATOM37_C = 0, 1, 2

C_N_RANGE = (1.2, 1.45)
CA_CA_RANGE = (3.5, 4.2)
GATE_FRACTION = 0.95

@dataclass(frozen=True)
class BondSeries:

    name: str
    values: np.ndarray
    pair_index: list[tuple[int, int]]
    low: float
    high: float
    n: int
    fraction: float | None
    mean: float | None
    valid: bool | None

    @property
    def in_range(self) -> np.ndarray:
        return (self.values > self.low) & (self.values < self.high)

@dataclass(frozen=True)
class BackboneBonds:

    n_residues: int
    adjacent_pairs: int
    c_n: BondSeries
    ca_ca: BondSeries
    valid: bool | None
    dropped_by_mask: int = 0
    dropped_by_chain: int = 0
    notes: dict = field(default_factory=dict)

def _conjunction(*gates: bool | None) -> bool | None:

    if any(g is False for g in gates):
        return False
    if any(g is None for g in gates):
        return None
    return True

def _series(name, coors, pairs, a_idx, b_idx, low, high) -> BondSeries:
    values = np.array(
        [
            float(np.linalg.norm(coors[b, b_idx] - coors[a, a_idx]))
            for a, b in pairs
        ],
        dtype=np.float64,
    )
    n = int(values.size)
    finite = bool(np.isfinite(values).all()) if n else True
    if n == 0 or not finite:
        return BondSeries(name, values, list(pairs), low, high, n, None, None, None)
    in_range = (values > low) & (values < high)
    fraction = float(in_range.mean())
    return BondSeries(
        name,
        values,
        list(pairs),
        low,
        high,
        n,
        fraction,
        float(values.mean()),
        bool(fraction >= GATE_FRACTION),
    )

def atom37_backbone_bonds(
    coors: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    residue_index: np.ndarray | None = None,
) -> BackboneBonds:

    coors = np.asarray(coors, dtype=np.float64)
    if coors.ndim != 3 or coors.shape[1] < 3 or coors.shape[2] != 3:
        raise ValueError(f"expected [n, >=3, 3] atom37 coordinates, got {coors.shape}")

    n_in = int(coors.shape[0])
    keep = np.ones(n_in, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if keep.shape != (n_in,):
        raise ValueError("mask must be one flag per residue")

    kept = np.flatnonzero(keep)
    dropped_by_mask = n_in - int(kept.size)

    chains = None if chain_index is None else np.asarray(chain_index)
    resids = None if residue_index is None else np.asarray(residue_index)

    pairs: list[tuple[int, int]] = []
    dropped_by_chain = 0
    for a, b in zip(kept, kept[1:]):

        if int(b) - int(a) != 1:
            continue
        if chains is not None and chains[a] != chains[b]:
            dropped_by_chain += 1
            continue
        if resids is not None and int(resids[b]) - int(resids[a]) != 1:
            dropped_by_chain += 1
            continue
        pairs.append((int(a), int(b)))

    c_n = _series("C_N", coors, pairs, ATOM37_C, ATOM37_N, *C_N_RANGE)
    ca_ca = _series("CA_CA", coors, pairs, ATOM37_CA, ATOM37_CA, *CA_CA_RANGE)

    return BackboneBonds(
        n_residues=int(kept.size),
        adjacent_pairs=len(pairs),
        c_n=c_n,
        ca_ca=ca_ca,
        valid=_conjunction(c_n.valid, ca_ca.valid),
        dropped_by_mask=dropped_by_mask,
        dropped_by_chain=dropped_by_chain,
    )
