
from __future__ import annotations
import re
from dataclasses import dataclass

import numpy as np
import torch
from openfold.np.residue_constants import atom_order, atom_types, restype_3to1, restype_order, restypes
from scipy.optimize import linear_sum_assignment

TOL_NM = 0.15
TOL_SENSITIVITY_NM = (0.10, 0.15, 0.20)

D_POLAR, D_APOLAR = 0.28, 0.34
CONTACT_MAX = 0.45

ATOM37_ELEMENT = [a[0] for a in atom_types]
RESTYPE_1TO3 = {v: k for k, v in restype_3to1.items()}

def parse_contig_atoms(spec: str) -> list[tuple[str, int, list[str]]]:

    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([A-Za-z0-9]+?)(-?\d+)\s*:\s*\[(.*)\]$", part)
        if not m:
            raise ValueError(f"unparsable contig_atoms fragment: {part!r}")
        chain, res_id, names = m.group(1), int(m.group(2)), m.group(3)
        out.append((chain, res_id, [n.strip() for n in names.split(",") if n.strip()]))
    merged: dict[tuple[str, int], list[str]] = {}
    for c, r, ns in out:
        merged.setdefault((c, r), []).extend(ns)
    return [(c, r, ns) for (c, r), ns in merged.items()]

@dataclass(frozen=True, slots=True)
class Requirement:

    motif_res: int
    chain: str
    res_id: int
    res_name: str
    atom_name: str
    atom37: int
    xyz_nm: tuple[float, float, float]
    d_ligand_nm: float
    ligand_partner: int
    is_polar: bool
    is_sidechain: bool

def build_requirements(spec: str, x_motif, motif_mask, seq_motif, x_lig) -> list[Requirement]:

    parsed = parse_contig_atoms(spec)
    xm = x_motif[0].detach().cpu()
    mm = motif_mask[0].detach().cpu().bool()
    sm = seq_motif[0].detach().cpu()
    lig = x_lig.detach().cpu()

    if len(parsed) != mm.shape[0]:
        raise ValueError(f"spec has {len(parsed)} residues, tensor has {mm.shape[0]}")
    if int(mm.sum()) != sum(len(ns) for _, _, ns in parsed):
        raise ValueError(f"spec names {sum(len(ns) for _,_,ns in parsed)} atoms, "
                         f"mask has {int(mm.sum())}")

    reqs = []
    for k, (chain, res_id, names) in enumerate(parsed):
        want = sorted(atom_order[n] for n in names if n in atom_order)
        got = sorted(mm[k].nonzero(as_tuple=False).flatten().tolist())
        if want != got:
            raise ValueError(f"residue {chain}{res_id}: spec atom37 {want} != mask {got}")
        rt = int(sm[k])
        res_name = RESTYPE_1TO3.get(restypes[rt], "UNK") if rt < len(restypes) else "UNK"
        for a in got:
            xyz = xm[k, a]
            d = torch.cdist(xyz[None, None], lig[None])[0, 0]
            reqs.append(Requirement(
                motif_res=k, chain=chain, res_id=res_id, res_name=res_name,
                atom_name=atom_types[a], atom37=a,
                xyz_nm=tuple(float(v) for v in xyz),
                d_ligand_nm=float(d.min()), ligand_partner=int(d.argmin()),
                is_polar=ATOM37_ELEMENT[a] in ("N", "O"),
                is_sidechain=a >= 5))
    return reqs

def input_sanity(reqs: list[Requirement], x_lig) -> dict:

    P = np.array([r.xyz_nm for r in reqs])
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    lig = x_lig.detach().cpu().numpy()
    dl = np.linalg.norm(P[:, None] - lig[None], axis=-1).min(axis=1)
    intra = []
    for k in {r.motif_res for r in reqs}:
        idx = [i for i, r in enumerate(reqs) if r.motif_res == k]
        if len(idx) > 1:
            sub = D[np.ix_(idx, idx)]
            intra.append(float(sub[np.isfinite(sub)].max()))
    return {"n_requirements": len(reqs),
            "min_pairwise_nm": float(D.min()),
            "atoms_overlapping_input": int((D.min(axis=1) < 0.05).sum()),
            "min_atom_ligand_nm": float(dl.min()),
            "atoms_inside_ligand_input": int((dl < 0.12).sum()),
            "max_intra_residue_span_nm": max(intra) if intra else 0.0,

            "input_contradiction": bool((D.min() < 0.05) or (dl.min() < 0.12))}

def assign_and_score(reqs, coors_nm, atom_mask, residue_type, res_mask, tol_nm=TOL_NM):

    C = coors_nm[0].detach().cpu()
    A = atom_mask[0].detach().cpu().bool()
    RT = residue_type[0].detach().cpu()
    M = res_mask[0].detach().cpu().bool()
    live = M.nonzero(as_tuple=False).flatten()
    kres = sorted({r.motif_res for r in reqs})
    CAP = 1.0

    cost = np.full((len(kres), len(live)), 0.0)
    for ki, k in enumerate(kres):
        rs = [r for r in reqs if r.motif_res == k]
        want = torch.tensor([r.atom37 for r in rs])
        ref = torch.tensor([r.xyz_nm for r in rs])
        sub = C[live][:, want, :]
        d = (sub - ref[None]).norm(dim=-1)
        present = A[live][:, want]
        d = torch.where(present, d, torch.full_like(d, CAP)).clamp(max=CAP)
        cost[ki] = d.sum(dim=1).numpy()
    ri, ci = linear_sum_assignment(cost)

    rows = []
    for ki, cj in zip(ri, ci):
        k, j = kres[ki], int(live[cj])
        for r in [x for x in reqs if x.motif_res == k]:
            present = bool(A[j, r.atom37])
            dev = float((C[j, r.atom37] - torch.tensor(r.xyz_nm)).norm()) if present else float("nan")
            rt = int(RT[j])
            gen_res = RESTYPE_1TO3.get(restypes[rt], "UNK") if rt < len(restypes) else "UNK"
            rows.append({"motif_res": k, "gen_res_index": j,
                         "chain": r.chain, "res_id": r.res_id,
                         "req_res_name": r.res_name, "gen_res_name": gen_res,
                         "res_identity_match": gen_res == r.res_name,
                         "atom_name": r.atom_name, "atom37": r.atom37,
                         "is_sidechain": r.is_sidechain, "is_polar": r.is_polar,
                         "atom_present": present, "deviation_nm": dev,
                         "placed": bool(present and dev <= tol_nm),
                         "d_ligand_ref_nm": r.d_ligand_nm})
    return rows

def scaffold_and_ligand_metrics(coors_nm, atom_mask, res_mask, bb_ca, x_lig, motif_gen_idx):

    C = coors_nm[0].detach().cpu()
    A = atom_mask[0].detach().cpu().bool() & res_mask[0].detach().cpu().bool()[:, None]
    ca = bb_ca[0].detach().cpu()[res_mask[0].detach().cpu().bool()]
    lig = x_lig.detach().cpu()

    idx = A.nonzero(as_tuple=False)
    pts = C[idx[:, 0], idx[:, 1]]
    elem = [ATOM37_ELEMENT[int(a)] for a in idx[:, 1]]
    pol = torch.tensor([e in ("N", "O") for e in elem])
    dmin = torch.where(pol, torch.tensor(D_POLAR), torch.tensor(D_APOLAR))
    d = torch.cdist(pts[None], lig[None])[0]
    nearest = d.min(dim=1).values
    is_motif_atom = torch.tensor([int(i) in set(motif_gen_idx) for i in idx[:, 0]])

    bond = (ca[1:] - ca[:-1]).norm(dim=-1)
    n = ca.shape[0]
    far = ~torch.eye(n, dtype=torch.bool)
    for k in (1, 2):
        far &= ~torch.diag(torch.ones(n - k, dtype=torch.bool), k)
        far &= ~torch.diag(torch.ones(n - k, dtype=torch.bool), -k)
    pair = torch.cdist(ca[None], ca[None])[0]
    return {"ca_bond_mean_nm": float(bond.mean()),
            "ca_bond_valid_frac": float(((bond > 0.34) & (bond < 0.42)).float().mean()),
            "ca_self_clash_frac": float((pair[far] < 0.30).float().mean()),
            "radius_gyration_nm": float((ca - ca.mean(0)).norm(dim=-1).pow(2).mean().sqrt()),
            "ligand_min_dist_nm": float(nearest.min()),
            "clashes_total": int((nearest < dmin).sum()),
            "clashes_scaffold_added": int(((nearest < dmin) & ~is_motif_atom).sum()),
            "clash_depth_sum_nm": float(torch.relu(dmin - nearest).sum()),
            "ligand_contact_atoms": int(((nearest >= dmin) & (nearest < CONTACT_MAX)).sum()),
            "ligand_atoms_uncovered": int((d.min(dim=0).values > CONTACT_MAX).sum()),
            "n_ligand_atoms": int(lig.shape[0])}
