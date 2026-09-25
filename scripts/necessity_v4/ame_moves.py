
from __future__ import annotations
import numpy as np
import torch
from openfold.np.residue_constants import (
    atom_order, atom_types, chi_angles_atoms, chi_angles_mask,
    restype_3to1, restype_atom37_to_rigid_group, restype_order, restypes,
)

RESTYPE_1TO3 = {v: k for k, v in restype_3to1.items()}

CHI1_GROUP = 4

def res3(rt: int) -> str:
    return RESTYPE_1TO3.get(restypes[rt], "UNK") if rt < len(restypes) else "UNK"

RING_CLOSED = {"PRO"}

def chi_moving_atoms(rt: int, chi_idx: int) -> np.ndarray:

    groups = restype_atom37_to_rigid_group[rt]
    return np.nonzero(groups >= CHI1_GROUP + chi_idx)[0]

def chi_axis_atoms(rt: int, chi_idx: int) -> tuple[int, int] | None:

    name = res3(rt)
    if name not in chi_angles_atoms or chi_idx >= len(chi_angles_atoms[name]):
        return None
    a = chi_angles_atoms[name][chi_idx]
    return atom_order[a[1]], atom_order[a[2]]

def n_chis(rt: int) -> int:
    return int(sum(chi_angles_mask[rt])) if rt < len(chi_angles_mask) else 0

def rotate_about_axis(points: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor,
                      angle: float) -> torch.Tensor:

    axis = p1 - p0
    n = axis.norm()
    if float(n) < 1e-8:
        return points
    k = axis / n
    v = points - p0
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return p0 + v * c + torch.cross(k.expand_as(v), v, dim=-1) * s           + k * (v @ k).unsqueeze(-1) * (1.0 - c)

def apply_chi(coors: torch.Tensor, atom_mask: torch.Tensor, residue_type: torch.Tensor,
              res_idx: int, chi_idx: int, angle: float) -> torch.Tensor:

    rt = int(residue_type[res_idx])
    if res3(rt) in RING_CLOSED:
        return coors
    ax = chi_axis_atoms(rt, chi_idx)
    if ax is None:
        return coors
    a2, a3 = ax
    if not (bool(atom_mask[res_idx, a2]) and bool(atom_mask[res_idx, a3])):
        return coors
    mov = [a for a in chi_moving_atoms(rt, chi_idx) if bool(atom_mask[res_idx, a])]
    if not mov:
        return coors
    out = coors.clone()
    out[res_idx, mov] = rotate_about_axis(
        coors[res_idx, mov], coors[res_idx, a2], coors[res_idx, a3], angle)
    return out

def backrub_allowed(residue_type, res_idx: int, n: int) -> bool:

    if not (0 < res_idx < n - 1):
        return False
    return res3(int(residue_type[res_idx + 1])) not in RING_CLOSED

def apply_backrub(coors: torch.Tensor, atom_mask: torch.Tensor,
                  res_idx: int, angle: float) -> torch.Tensor:

    ca, n_i, c_i, o_i = atom_order["CA"], atom_order["N"], atom_order["C"], atom_order["O"]
    i0, i1 = res_idx - 1, res_idx + 1
    if i0 < 0 or i1 >= coors.shape[0]:
        return coors
    p0, p1 = coors[i0, ca], coors[i1, ca]
    out = coors.clone()
    segments = [(i0, [a for a in (c_i, o_i) if bool(atom_mask[i0, a])]),
                (res_idx, [a for a in atom_mask[res_idx].nonzero(as_tuple=False)
                           .flatten().tolist()]),
                (i1, [a for a in (n_i,) if bool(atom_mask[i1, a])])]
    for r, sel in segments:
        if sel:
            out[r, sel] = rotate_about_axis(coors[r, sel], p0, p1, angle)
    return out

def _angle(a, b, c):
    u, v = a - b, c - b
    cos = (u @ v) / (u.norm() * v.norm()).clamp(min=1e-9)
    return float(torch.arccos(cos.clamp(-1.0, 1.0)))

def backbone_angle_drift(coors, ref_coors, atom_mask, res_indices) -> float:

    n_i, ca, c_i = atom_order["N"], atom_order["CA"], atom_order["C"]
    worst = 0.0
    for r in res_indices:
        if not (bool(atom_mask[r, n_i]) and bool(atom_mask[r, ca]) and bool(atom_mask[r, c_i])):
            continue
        a0 = _angle(ref_coors[r, n_i], ref_coors[r, ca], ref_coors[r, c_i])
        a1 = _angle(coors[r, n_i], coors[r, ca], coors[r, c_i])
        worst = max(worst, abs(a1 - a0) * 180.0 / np.pi)
    return worst

_BONDS, _VIRT, _ANGLES = None, None, None

def _tables():
    global _BONDS, _VIRT, _ANGLES
    if _BONDS is None:
        from openfold.np.residue_constants import load_stereo_chemical_props
        _BONDS, _VIRT, _ANGLES = load_stereo_chemical_props()
    return _BONDS, _ANGLES

def bond_length_drift(coors, ref_coors, atom_mask, res_mask, residue_type) -> float:

    bonds, _ = _tables()
    worst = 0.0
    live = res_mask.nonzero(as_tuple=False).flatten().tolist()
    for r in live:
        name = res3(int(residue_type[r]))
        for bd in bonds.get(name, []):
            i, j = atom_order.get(bd.atom1_name), atom_order.get(bd.atom2_name)
            if i is None or j is None or not (bool(atom_mask[r, i]) and bool(atom_mask[r, j])):
                continue
            d0 = float((ref_coors[r, i] - ref_coors[r, j]).norm())
            d1 = float((coors[r, i] - coors[r, j]).norm())
            worst = max(worst, abs(d1 - d0))
    if len(live) < 2:
        return worst
    n_i, c_i = atom_order["N"], atom_order["C"]
    idx = torch.tensor(live)
    cn0 = (ref_coors[idx[:-1], c_i] - ref_coors[idx[1:], n_i]).norm(dim=-1)
    cn1 = (coors[idx[:-1], c_i] - coors[idx[1:], n_i]).norm(dim=-1)
    return max(worst, float((cn1 - cn0).abs().max()))

def angle_drift_sigma(coors, ref_coors, atom_mask, residue_type, res_indices,
                      return_worst=False):

    _, angles = _tables()
    worst, where = 0.0, None
    for r in res_indices:
        if r < 0 or r >= coors.shape[0]:
            continue
        name = res3(int(residue_type[r]))
        for an in angles.get(name, []):
            i = atom_order.get(an.atom1_name); j = atom_order.get(an.atom2_name)
            k = atom_order.get(an.atom3name)
            if None in (i, j, k) or not (bool(atom_mask[r, i]) and bool(atom_mask[r, j])
                                         and bool(atom_mask[r, k])):
                continue
            d = abs(_angle(coors[r, i], coors[r, j], coors[r, k])
                    - _angle(ref_coors[r, i], ref_coors[r, j], ref_coors[r, k]))
            ratio = d / max(float(an.stddev), 1e-6)
            if ratio > worst:
                worst, where = ratio, (name, r, f"{an.atom1_name}-{an.atom2_name}-{an.atom3name}",
                                       d * 180.0 / np.pi)
    return (worst, where) if return_worst else worst

_CA_ANGLES = (("N", "CA", "C"), ("N", "CA", "CB"), ("CB", "CA", "C"))
_CA_SIGMA: dict[tuple[str, tuple[str, str, str]], float] = {}

def ca_angle_drift_sigma(coors, ref_coors, atom_mask, residue_type, res_indices) -> float:

    _, angles = _tables()
    if not _CA_SIGMA:
        for name, lst in angles.items():
            for an in lst:
                key = (an.atom1_name, an.atom2_name, an.atom3name)
                if key in _CA_ANGLES or key[::-1] in _CA_ANGLES:
                    _CA_SIGMA[(name, key)] = float(an.stddev)
    worst = 0.0
    for r in res_indices:
        if r < 0 or r >= coors.shape[0]:
            continue
        name = res3(int(residue_type[r]))
        for trio in _CA_ANGLES:
            sd = _CA_SIGMA.get((name, trio)) or _CA_SIGMA.get((name, trio[::-1]))
            if sd is None:
                continue
            i, j, k = (atom_order[t] for t in trio)
            if not (bool(atom_mask[r, i]) and bool(atom_mask[r, j]) and bool(atom_mask[r, k])):
                continue
            d = abs(_angle(coors[r, i], coors[r, j], coors[r, k])
                    - _angle(ref_coors[r, i], ref_coors[r, j], ref_coors[r, k]))
            worst = max(worst, d / max(sd, 1e-6))
    return worst

def bond_angle_drift_deg(coors, ref_coors, atom_mask, res_mask, residue_type) -> float:

    _, angles = _tables()
    worst = 0.0
    for r in res_mask.nonzero(as_tuple=False).flatten().tolist():
        name = res3(int(residue_type[r]))
        for an in angles.get(name, []):
            i = atom_order.get(an.atom1_name); j = atom_order.get(an.atom2_name)
            k = atom_order.get(an.atom3name)
            if None in (i, j, k) or not (bool(atom_mask[r, i]) and bool(atom_mask[r, j])
                                         and bool(atom_mask[r, k])):
                continue
            a0 = _angle(ref_coors[r, i], ref_coors[r, j], ref_coors[r, k])
            a1 = _angle(coors[r, i], coors[r, j], coors[r, k])
            worst = max(worst, abs(a1 - a0) * 180.0 / np.pi)
    return worst

def rigid_invariance_drift(coors, atom_mask, ref_coors, residue_type, res_indices):

    worst = 0.0
    for r in res_indices:
        rt = int(residue_type[r])
        if rt >= len(restypes):
            continue
        groups = restype_atom37_to_rigid_group[rt]
        live = atom_mask[r].nonzero(as_tuple=False).flatten().tolist()
        for g in set(int(groups[a]) for a in live):
            sel = [a for a in live if int(groups[a]) == g]
            if len(sel) < 2:
                continue
            a = torch.cdist(coors[r, sel][None], coors[r, sel][None])[0]
            b = torch.cdist(ref_coors[r, sel][None], ref_coors[r, sel][None])[0]
            worst = max(worst, float((a - b).abs().max()))
    return worst

def axis_distance_drift(coors, ref_coors, atom_mask, res_idx, a2, a3):

    live = atom_mask[res_idx].nonzero(as_tuple=False).flatten().tolist()
    w = 0.0
    for ax in (a2, a3):
        d0 = (ref_coors[res_idx, live] - ref_coors[res_idx, ax]).norm(dim=-1)
        d1 = (coors[res_idx, live] - coors[res_idx, ax]).norm(dim=-1)
        w = max(w, float((d1 - d0).abs().max()))
    return w

def chain_geometry(coors, atom_mask, res_mask):

    n_i, ca_i, c_i = atom_order["N"], atom_order["CA"], atom_order["C"]
    live = res_mask.nonzero(as_tuple=False).flatten()
    ca = coors[live, ca_i]
    cn = (coors[live[:-1], c_i] - coors[live[1:], n_i]).norm(dim=-1)
    d = (ca[1:] - ca[:-1]).norm(dim=-1)
    n = ca.shape[0]
    far = ~torch.eye(n, dtype=torch.bool)
    for k in (1, 2):
        far &= ~torch.diag(torch.ones(n - k, dtype=torch.bool), k)
        far &= ~torch.diag(torch.ones(n - k, dtype=torch.bool), -k)
    pair = torch.cdist(ca[None], ca[None])[0]
    return {"ca_ca_mean_nm": float(d.mean()),
            "ca_ca_valid_frac": float(((d > 0.34) & (d < 0.42)).float().mean()),
            "ca_ca_min_nm": float(d.min()), "ca_ca_max_nm": float(d.max()),
            "peptide_cn_mean_nm": float(cn.mean()),
            "peptide_cn_max_nm": float(cn.max()),
            "peptide_cn_valid_frac": float(((cn > 0.12) & (cn < 0.15)).float().mean()),
            "ca_self_clash_frac": float((pair[far] < 0.30).float().mean())}
