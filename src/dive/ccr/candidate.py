
from __future__ import annotations

import numpy as np

ATOM37_NAMES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
)
SLOT_OF_NAME = {name: i for i, name in enumerate(ATOM37_NAMES)}
BACKBONE_NAMES = ("N", "CA", "C", "O")

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

def _constants():
    from alphafold.common import residue_constants

    return residue_constants

def _residue_atoms() -> dict[str, tuple[str, ...]]:
    rc = _constants()
    out = {}
    for code, three in ONE_TO_THREE.items():
        names = tuple(
            name for name, _group, _pos in rc.rigid_group_atom_positions[three]
            if name in SLOT_OF_NAME
        )
        out[code] = names
    return out

RESIDUE_ATOMS: dict[str, tuple[str, ...]] = _residue_atoms()

def candidate_atom_names(code: str) -> tuple[str, ...]:
    try:
        return RESIDUE_ATOMS[code]
    except KeyError as error:
        raise KeyError(f"unknown residue code {code!r}") from error

def n_chi(code: str) -> int:
    rc = _constants()
    return int(sum(rc.chi_angles_mask[rc.restype_order[code]]))

def rotamer_samples(code: str, n: int = 3) -> list[tuple[float, ...]]:

    count = n_chi(code)
    if count == 0:
        return [()]
    base = np.deg2rad([-60.0, 60.0, 180.0][: max(2, min(n, 3))])

    grid = []
    if count == 1:
        return [(float(chi1),) for chi1 in base][: max(n, 2)]
    for chi2 in base:
        for chi1 in base:
            rest = tuple(float(np.deg2rad(180.0)) for _ in range(count - 2))
            grid.append((float(chi1), float(chi2)) + rest)
    return grid[: max(n, 2)] if n < len(grid) else grid

def _frame_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray):

    x = c - b
    x = x / max(float(np.linalg.norm(x)), 1e-8)
    v = a - b
    v = v - x * float(v @ x)
    y = v / max(float(np.linalg.norm(v)), 1e-8)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1), b

def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])

def build_candidate_atoms(
    residue: np.ndarray, code: str, chi: tuple[float, ...] | None = None
) -> np.ndarray:

    rc = _constants()
    three = ONE_TO_THREE.get(code)
    if three is None:
        raise KeyError(f"unknown residue code {code!r}")

    residue = np.asarray(residue, dtype=np.float64)
    out = np.full((37, 3), np.nan)
    for name in BACKBONE_NAMES:
        slot = SLOT_OF_NAME[name]
        if np.isfinite(residue[slot]).all():
            out[slot] = residue[slot]

    n, ca, c = (out[SLOT_OF_NAME[k]] for k in ("N", "CA", "C"))
    if not (np.isfinite(n).all() and np.isfinite(ca).all() and np.isfinite(c).all()):
        return out

    rotation, origin = _frame_from_three(n, ca, c)
    frames = {0: (rotation, origin)}

    positions = rc.rigid_group_atom_positions[three]
    chi_atoms = rc.chi_angles_atoms[three]
    angles = list(chi if chi is not None else [np.deg2rad(180.0)] * len(chi_atoms))
    angles += [np.deg2rad(180.0)] * (len(chi_atoms) - len(angles))

    placed: dict[str, np.ndarray] = {"N": n, "CA": ca, "C": c}
    for name, group, local in positions:
        if group == 0 and name in SLOT_OF_NAME:
            point = rotation @ np.array(local, dtype=np.float64) + origin
            if name not in BACKBONE_NAMES:
                out[SLOT_OF_NAME[name]] = point
            placed[name] = point

    for index, quartet in enumerate(chi_atoms):
        group = 4 + index
        a, b, cc, _d = quartet
        if not all(k in placed for k in (a, b, cc)):
            break
        axis = placed[cc] - placed[b]
        base_rotation, base_origin = _frame_from_three(placed[a], placed[b], placed[cc])
        turn = _rotation_about(axis, angles[index])
        rotation_g = turn @ base_rotation
        origin_g = placed[cc]
        frames[group] = (rotation_g, origin_g)

        for name, atom_group, local in positions:
            if atom_group != group or name not in SLOT_OF_NAME:
                continue
            local = np.array(local, dtype=np.float64)

            point = rotation_g @ local + origin_g
            out[SLOT_OF_NAME[name]] = point
            placed[name] = point

    return out

def substitution_effect(current: str, candidate: str) -> dict:

    have = set(candidate_atom_names(current))
    want = set(candidate_atom_names(candidate))
    added = sorted(want - have)
    removed = sorted(have - want)
    return {
        "current": current,
        "candidate": candidate,
        "atoms_added": len(added),
        "atoms_removed": len(removed),
        "added": tuple(added),
        "removed": tuple(removed),
        "is_truncation": bool(removed and not added),
    }
