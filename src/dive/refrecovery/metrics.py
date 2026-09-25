
from __future__ import annotations

import numpy as np
from openfold.np.residue_constants import atom_order

from dive.refrecovery.reference import SYMMETRIC_PAIRS, RefChain

BACKBONE_IDX: tuple[int, ...] = tuple(atom_order[n] for n in ("N", "CA", "C", "O"))
CA_IDX: int = atom_order["CA"]

F1_MOTIF, F2_POCKET, F3_FAR = 1, 2, 3

LDDT_INCLUSION_NM = 1.5
LDDT_THRESHOLDS_NM = (0.05, 0.10, 0.20, 0.40)

def lddt(gen_ca: np.ndarray, ref_ca: np.ndarray, mask: np.ndarray,
         inclusion: float = LDDT_INCLUSION_NM) -> np.ndarray:

    gen_ca = np.asarray(gen_ca, float)
    ref_ca = np.asarray(ref_ca, float)
    mask = np.asarray(mask, bool)
    L = len(ref_ca)

    dref = np.linalg.norm(ref_ca[:, None] - ref_ca[None, :], axis=-1)
    dgen = np.linalg.norm(gen_ca[:, None] - gen_ca[None, :], axis=-1)

    pair = (dref < inclusion) & (~np.eye(L, dtype=bool))
    pair &= mask[:, None] & mask[None, :]

    diff = np.abs(dgen - dref)
    score = np.zeros((L, L))
    for th in LDDT_THRESHOLDS_NM:
        score += (diff < th)
    score /= len(LDDT_THRESHOLDS_NM)

    n = pair.sum(axis=1)
    out = np.full(L, np.nan)
    ok = n > 0
    out[ok] = (score * pair).sum(axis=1)[ok] / n[ok]
    return out

def _rmsd_over(idx, gen37, gen_mask, ref37, ref_mask) -> float:
    idx = np.asarray(idx, int)
    both = gen_mask[idx] & ref_mask[idx]
    if not both.any():
        return np.nan
    d = gen37[idx][both] - ref37[idx][both]
    return float(np.sqrt((d ** 2).sum(-1).mean()))

def backbone_deviation(gen37, gen_mask, ref37, ref_mask) -> np.ndarray:

    gen37, ref37 = np.asarray(gen37, float), np.asarray(ref37, float)
    gen_mask, ref_mask = np.asarray(gen_mask, bool), np.asarray(ref_mask, bool)
    return np.array([_rmsd_over(BACKBONE_IDX, gen37[i], gen_mask[i], ref37[i], ref_mask[i])
                     for i in range(len(gen37))])

def sidechain_deviation(gen37, gen_mask, gen_restype, ref37, ref_mask, ref_restype):

    gen37, ref37 = np.asarray(gen37, float), np.asarray(ref37, float)
    gen_mask, ref_mask = np.asarray(gen_mask, bool), np.asarray(ref_mask, bool)
    L = len(gen37)
    kept = np.array([str(gen_restype[i]) == str(ref_restype[i]) for i in range(L)])
    dev = np.full(L, np.nan)
    side = np.array([a for a in range(37) if a not in BACKBONE_IDX])
    for i in np.where(kept)[0]:
        ref_i = resolve_symmetry(str(ref_restype[i]), gen37[i], ref37[i],
                                 gen_mask[i] & ref_mask[i])
        dev[i] = _rmsd_over(side, gen37[i], gen_mask[i], ref_i, ref_mask[i])
    return dev, kept

def resolve_symmetry(res_name: str, gen37_res, ref37_res, mask_res) -> np.ndarray:

    out = np.array(ref37_res, dtype=float, copy=True)
    for a_name, b_name in SYMMETRIC_PAIRS.get(str(res_name).upper(), ()):
        a, b = atom_order[a_name], atom_order[b_name]
        if not (mask_res[a] and mask_res[b]):
            continue
        keep = (np.linalg.norm(gen37_res[a] - out[a]) ** 2
                + np.linalg.norm(gen37_res[b] - out[b]) ** 2)
        swap = (np.linalg.norm(gen37_res[a] - out[b]) ** 2
                + np.linalg.norm(gen37_res[b] - out[a]) ** 2)
        if swap < keep:
            out[[a, b]] = out[[b, a]]
    return out

def assign_regions(ref: RefChain, motif_res_ids, ligand_xyz,
                   near: float = 0.5, far: float = 1.0, flank: int = 2) -> np.ndarray:

    L = len(ref)
    ids = np.asarray(ref.res_ids)
    want = {int(r) for r in motif_res_ids}
    is_f1 = np.array([int(r) in want for r in ids])

    lig = np.asarray(ligand_xyz, float).reshape(-1, 3)
    heavy = ref.coords37.copy()
    dmin = np.full(L, np.inf)
    for i in range(L):
        m = ref.mask37[i]
        if not m.any():
            continue
        dmin[i] = np.linalg.norm(heavy[i][m][:, None] - lig[None], axis=-1).min()

    f1_pos = np.where(is_f1)[0]
    flanking = np.zeros(L, bool)
    for p in f1_pos:
        flanking[max(0, p - flank):min(L, p + flank + 1)] = True

    reg = np.full(L, F2_POCKET, dtype=int)
    reg[(dmin > far) & ~flanking] = F3_FAR
    reg[is_f1] = F1_MOTIF
    return reg

def tm_score(gen_ca: np.ndarray, ref_ca: np.ndarray, mask: np.ndarray) -> float:

    m = np.asarray(mask, bool)
    L = int(m.sum())
    if L < 3:
        return np.nan
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8 if L > 15 else 0.5
    d0 = max(d0, 0.5) / 10.0
    d = np.linalg.norm(np.asarray(gen_ca)[m] - np.asarray(ref_ca)[m], axis=-1)
    return float((1.0 / (1.0 + (d / d0) ** 2)).mean())
