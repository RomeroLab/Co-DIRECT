
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ALTLOC_KEEP = frozenset({"", " ", ".", "A"})

BACKBONE_ATOMS = ("N", "CA", "C", "O")

SYMMETRIC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    "ARG": (("NH1", "NH2"),),
    "VAL": (("CG1", "CG2"),),
    "LEU": (("CD1", "CD2"),),
}

@dataclass
class RefChain:

    res_ids: np.ndarray
    res_names: list[str]
    coords37: np.ndarray
    mask37: np.ndarray
    n_atoms_unresolved: int
    chain: str

    def __len__(self) -> int:
        return len(self.res_ids)

    def subset(self, start: int, stop: int) -> "RefChain":
        return RefChain(res_ids=self.res_ids[start:stop],
                        res_names=self.res_names[start:stop],
                        coords37=self.coords37[start:stop],
                        mask37=self.mask37[start:stop],
                        n_atoms_unresolved=self.n_atoms_unresolved,
                        chain=self.chain)

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "RefChain":
        xyz = self.coords37 @ R.T + t
        xyz[~self.mask37] = 0.0
        return RefChain(self.res_ids, self.res_names, xyz, self.mask37,
                        self.n_atoms_unresolved, self.chain)

def kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc

def select_window(ref: RefChain, motif_res_ids, max_len: int = 180):

    ids = np.asarray(ref.res_ids)
    L = len(ids)
    want = sorted(int(r) for r in motif_res_ids)
    pos = [int(np.where(ids == r)[0][0]) for r in want if np.any(ids == r)]
    if len(pos) != len(want):
        return None
    lo, hi = min(pos), max(pos)

    if L <= max_len:
        start, stop = 0, L
    else:
        span = hi - lo + 1
        if span > max_len:
            return None
        pad = (max_len - span) // 2
        start = max(0, lo - pad)
        stop = start + max_len
        if stop > L:
            stop, start = L, L - max_len

    seg = ids[start:stop]
    if len(seg) > 1 and not np.array_equal(np.diff(seg), np.ones(len(seg) - 1, dtype=seg.dtype)):
        return None
    return int(start), int(stop)

def parse_reference(pdb_path: Path, chain: str) -> RefChain:

    import biotite.structure.io.pdb as pdbio
    from biotite.structure import filter_amino_acids
    from openfold.np.residue_constants import (atom_order, restype_1to3,
                                               restype_3to1, restype_name_to_atom14_names)

    arr = pdbio.PDBFile.read(str(pdb_path)).get_structure(model=1, altloc="all")
    arr = arr[filter_amino_acids(arr)]
    arr = arr[arr.chain_id == chain]
    if arr.array_length() == 0:
        raise ValueError(f"chain {chain!r} absent from {pdb_path.name}")

    altloc = getattr(arr, "altloc_id", None)
    if altloc is not None:
        arr = arr[np.isin(np.asarray(altloc, dtype=object), list(ALTLOC_KEEP))]
    ins = getattr(arr, "ins_code", None)
    if ins is not None:
        codes = np.asarray(ins, dtype=object)
        if np.any(~np.isin(codes, ["", " ", None])):
            raise ValueError(f"{pdb_path.name} chain {chain} carries insertion codes")

    res_ids = np.unique(arr.res_id)
    L = len(res_ids)
    coords = np.zeros((L, 37, 3), dtype=np.float64)
    mask = np.zeros((L, 37), dtype=bool)
    names: list[str] = []
    unresolved = 0

    for i, rid in enumerate(res_ids):
        sel = arr[arr.res_id == rid]
        rname = str(sel.res_name[0])
        names.append(rname)
        for aname, xyz in zip(sel.atom_name, sel.coord):
            a = atom_order.get(str(aname))
            if a is None or mask[i, a]:
                continue
            coords[i, a] = np.asarray(xyz, dtype=np.float64) / 10.0
            mask[i, a] = True
        one = restype_3to1.get(rname)
        if one is not None:
            expected = {n for n in restype_name_to_atom14_names[restype_1to3[one]] if n}
            unresolved += sum(1 for n in expected if not mask[i, atom_order[n]])

    return RefChain(res_ids=res_ids, res_names=names, coords37=coords,
                    mask37=mask, n_atoms_unresolved=unresolved, chain=chain)

def map_to_generation_frame(ref: RefChain, x_motif: np.ndarray, motif_spec):

    from openfold.np.residue_constants import atom_order

    P, Q = [], []
    for k, (_chain, rid, anames) in enumerate(motif_spec):
        hits = np.where(ref.res_ids == int(rid))[0]
        if len(hits) == 0:
            continue
        i = int(hits[0])
        for nm in anames:
            a = atom_order.get(nm)
            if a is None or not ref.mask37[i, a]:
                continue
            P.append(ref.coords37[i, a])
            Q.append(np.asarray(x_motif[k, a], dtype=np.float64))
    if len(P) < 3:
        raise ValueError("fewer than 3 motif atoms available for superposition")
    P, Q = np.asarray(P), np.asarray(Q)
    R, t = kabsch(P, Q)
    resid = np.linalg.norm(P @ R.T + t - Q, axis=1)
    return ref.transformed(R, t), float(np.sqrt((resid ** 2).mean()))
