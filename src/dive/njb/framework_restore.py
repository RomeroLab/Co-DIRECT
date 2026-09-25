
from __future__ import annotations

import torch

def restore_fixed_framework(
    *,
    coors_nm: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_type: torch.Tensor,
    motif_mask: torch.Tensor,
    x_motif: torch.Tensor,
    seq_motif: torch.Tensor,
    seq_motif_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if coors_nm.shape != x_motif.shape:
        raise ValueError("x_motif is not protein-aligned to coors_nm")
    coors = coors_nm.clone()
    atoms = atom_mask.clone()
    seq = residue_type.clone()
    framework = motif_mask.any(-1) & residue_mask
    copy_aa = seq_motif_mask.bool() & framework
    seq = torch.where(copy_aa, seq_motif.to(seq.dtype), seq)
    copy_xyz = motif_mask.bool() & residue_mask[..., None]
    coors = torch.where(copy_xyz[..., None], x_motif, coors)
    atoms = atoms | copy_xyz
    return coors, atoms, seq

def sample_designed_residue_type(
    residue_type: torch.Tensor,
    designed_mask: torch.Tensor,
    probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

    if probs.shape != (20,) or float(probs.sum()) <= 0:
        raise ValueError("designed AA probs must be a length-20 positive histogram")
    p = probs.to(dtype=torch.float64)
    p = p / p.sum()
    out = residue_type.clone()
    n = int(designed_mask.sum())
    if n == 0:
        return out
    draws = torch.multinomial(p, n, replacement=True, generator=generator)
    out = out.clone()
    out[designed_mask] = draws.to(out.dtype)
    return out

def fill_designed_coil_ca(
    ca: torch.Tensor,
    *,
    framework: torch.Tensor,
    residue_mask: torch.Tensor,
    bond_nm: float = 0.38,
) -> torch.Tensor:

    if ca.shape[:2] != framework.shape or framework.shape != residue_mask.shape:
        raise ValueError("coil CA/framework/mask shape mismatch")
    out = ca.clone()
    bsz, nres, _ = ca.shape
    for b in range(bsz):
        fw = framework[b] & residue_mask[b]
        des = (~framework[b]) & residue_mask[b]
        if not des.any():
            continue
        idx = torch.where(des)[0]
        starts = [int(idx[0])]
        ends: list[int] = []
        for a, c in zip(idx[:-1].tolist(), idx[1:].tolist()):
            if c != a + 1:
                ends.append(a)
                starts.append(c)
        ends.append(int(idx[-1]))
        for i0, i1 in zip(starts, ends):
            left = i0 - 1 if i0 > 0 and bool(fw[i0 - 1]) else None
            right = i1 + 1 if i1 + 1 < nres and bool(fw[i1 + 1]) else None
            start = out[b, left] if left is not None else out[b, i0]
            end = out[b, right] if right is not None else out[b, i1]
            n_des = i1 - i0 + 1
            pts = _coil_points(start, end, n_des, bond_nm)
            for k, i in enumerate(range(i0, i1 + 1)):
                out[b, i] = pts[k]
    return out

def _coil_points(start, end, n, bond):
    dvec = end - start
    dist = float(dvec.norm())
    s = bond * (n + 1)
    if dist < 1e-4:
        perp = start.new_tensor([0.0, 1.0, 0.0])
        return [start + (k + 1) * bond * perp for k in range(n)]
    if s <= dist + 1e-3:
        return [start + ((k + 1) / (n + 1)) * dvec for k in range(n)]
    unit = dvec / dist
    ratio = dist / s
    lo, hi = 1e-3, 3.14159
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        val = float(torch.sin(torch.tensor(mid)) / mid)
        if val > ratio:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    radius = s / (2.0 * theta)
    perp = torch.linalg.cross(unit, start.new_tensor([0.0, 0.0, 1.0]))
    if float(perp.norm()) < 1e-3:
        perp = torch.linalg.cross(unit, start.new_tensor([0.0, 1.0, 0.0]))
    perp = perp / perp.norm().clamp(min=1e-4)
    midpt = 0.5 * (start + end)
    half = dist / 2.0
    offset = max(radius ** 2 - half ** 2, 0.0) ** 0.5
    center = midpt + offset * perp
    v0 = start - center
    axis = torch.linalg.cross(v0, end - center)
    axis = axis / axis.norm().clamp(min=1e-4)
    phi = 2.0 * theta / (n + 1)
    pts = []
    for k in range(n):
        ang = phi * (k + 1)
        cos_a = torch.cos(torch.tensor(ang, device=start.device))
        sin_a = torch.sin(torch.tensor(ang, device=start.device))
        rot = v0 * cos_a + torch.linalg.cross(axis, v0) * sin_a + axis * torch.dot(axis, v0) * (1 - cos_a)
        pts.append(center + rot)
    return pts

def rigid_place_designed_span(
    ca: torch.Tensor,
    *,
    framework: torch.Tensor,
    residue_mask: torch.Tensor,
    designed: torch.Tensor,
) -> torch.Tensor:

    if ca.shape[:2] != framework.shape or framework.shape != residue_mask.shape:
        raise ValueError("place: CA/framework/mask shape mismatch")
    if designed.shape != framework.shape:
        raise ValueError("place: designed mask must match the residue axis")
    out = ca.clone()
    target = fill_designed_coil_ca(ca, framework=framework,
                                   residue_mask=residue_mask)
    for b in range(ca.shape[0]):
        rows = designed[b] & residue_mask[b]
        if int(rows.sum()) < 3:

            continue
        idx = torch.where(rows)[0]
        P = ca[b, idx].double()
        Q = target[b, idx].double()
        Pc = P - P.mean(0)
        Qc = Q - Q.mean(0)
        U, _, Vt = torch.linalg.svd(Pc.T @ Qc)
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64))
        R = Vt.T @ D @ U.T
        out[b, idx] = ((R @ Pc.T).T + Q.mean(0)).to(ca.dtype)
    return out
