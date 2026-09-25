
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MARGIN = 8.0

DEFAULT_MAX_RESIDUES = 48

@dataclass(frozen=True)
class Patch:
    coors: np.ndarray
    chain_index: np.ndarray
    rows: np.ndarray
    centre: int
    movable: np.ndarray
    truncated: bool = False

def local_patch(
    coors: np.ndarray,
    *,
    residue: int,
    margin: float = DEFAULT_MARGIN,
    chain_index: np.ndarray | None = None,
    max_residues: int | None = DEFAULT_MAX_RESIDUES,
) -> Patch:

    coors = np.asarray(coors, dtype=np.float64)
    n = int(coors.shape[0])
    keep = np.zeros(n, dtype=bool)
    keep[residue] = True
    for neighbour in (residue - 1, residue + 1):
        if 0 <= neighbour < n:
            keep[neighbour] = True

    centre_points = coors[residue].reshape(-1, 3)
    centre_points = centre_points[np.isfinite(centre_points).all(axis=-1)]
    if centre_points.size:
        from scipy.spatial import cKDTree

        flat = coors.reshape(-1, 3)
        finite = np.isfinite(flat).all(axis=-1)
        owner = np.repeat(np.arange(n), coors.shape[1])[finite]
        points = flat[finite]
        if points.size:
            tree = cKDTree(centre_points)
            near = tree.query_ball_point(points, r=float(margin), return_length=True) > 0
            keep[np.unique(owner[near])] = True

    rows = np.flatnonzero(keep)
    truncated = False
    if max_residues is not None and rows.size > max_residues:

        distance = np.full(rows.size, np.inf)
        centre_atoms = coors[residue].reshape(-1, 3)
        centre_atoms = centre_atoms[np.isfinite(centre_atoms).all(axis=-1)]
        for k, r in enumerate(rows):
            atoms = coors[r].reshape(-1, 3)
            atoms = atoms[np.isfinite(atoms).all(axis=-1)]
            if atoms.size and centre_atoms.size:
                distance[k] = float(
                    np.sqrt(((atoms[:, None, :] - centre_atoms[None, :, :]) ** 2)
                            .sum(-1)).min()
                )
        must = np.isin(rows, [residue - 1, residue, residue + 1])
        distance[must] = -1.0
        rows = np.sort(rows[np.argsort(distance)[:max_residues]])
        truncated = True

    breaks = np.ones(rows.size, dtype=bool)
    breaks[1:] = rows[1:] != rows[:-1] + 1
    if chain_index is not None:
        source = np.asarray(chain_index)
        breaks[1:] |= source[rows[1:]] != source[rows[:-1]]
    patch_chain = np.cumsum(breaks) - 1

    centre = int(np.flatnonzero(rows == residue)[0])
    movable = np.zeros(rows.size, dtype=bool)
    movable[centre] = True
    return Patch(coors[rows].copy(), patch_chain.astype(int), rows, centre, movable,
                 truncated)
