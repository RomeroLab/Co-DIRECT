
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import ame_moves as MV
from openfold.np.residue_constants import atom_order

D_POLAR, D_APOLAR = 0.28, 0.34

POCKET_NM = 0.80

SIGMA_CHI, SIGMA_BACKRUB, BACKRUB_MAX = 0.35, 0.10, 0.35

ANGLE_SIGMA_MAX = 2.0

LAM_LIG, MU_SELF = 1.0, 1.0

class Scorer:

    def __init__(self, coors, atom_mask, res_mask, lig, targets, dev):
        self.am = atom_mask & res_mask[:, None]
        self.lig = lig.to(dev)
        self.dev = dev
        idx = self.am.nonzero(as_tuple=False)
        self.ai, self.aj = idx[:, 0], idx[:, 1]
        elem = [MV.atom_types[int(a)][0] for a in self.aj]
        pol = torch.tensor([e in ("N", "O") for e in elem], device=dev)
        self.dmin = torch.where(pol, torch.tensor(D_POLAR, device=dev),
                                torch.tensor(D_APOLAR, device=dev))
        self.t_res = torch.tensor([t[0] for t in targets], device=dev)
        self.t_atom = torch.tensor([t[1] for t in targets], device=dev)
        self.t_xyz = torch.stack([t[2] for t in targets]).to(dev)
        self.ca = res_mask.nonzero(as_tuple=False).flatten()
        n = self.ca.shape[0]
        far = ~torch.eye(n, dtype=torch.bool, device=dev)
        for k in (1, 2):
            far &= ~torch.diag(torch.ones(n - k, dtype=torch.bool, device=dev), k)
            far &= ~torch.diag(torch.ones(n - k, dtype=torch.bool, device=dev), -k)
        self.far = far
        self.ca_i = atom_order["CA"]

    def __call__(self, coors):
        dev2 = (coors[self.t_res, self.t_atom] - self.t_xyz).pow(2).sum()
        pts = coors[self.ai, self.aj]
        d = torch.cdist(pts[None], self.lig[None])[0]
        lig_pen = torch.relu(self.dmin[:, None] - d).pow(2).sum()
        cav = coors[self.ca, self.ca_i]
        p = torch.cdist(cav[None], cav[None])[0]
        self_pen = torch.relu(0.30 - p[self.far]).pow(2).sum() * 0.5
        return float(dev2 + LAM_LIG * lig_pen + MU_SELF * self_pen)

def anneal(coors0, atom_mask, residue_type, res_mask, scorer, chi_dof, br_dof,
           n_eval, seed, tau_ref, tau_res):

    g = torch.Generator(device="cpu").manual_seed(seed)
    cur = coors0.clone()
    cur_e = scorer(cur)
    best, best_e = cur.clone(), cur_e
    dofs = [("chi", r, c) for (r, c) in chi_dof] + [("br", r, None) for r in br_dof]
    if not dofs:
        return best, best_e, {"n_eval": 0, "accepted": 0, "rejected_tau": 0}
    T0, T1 = 0.05, 1e-4
    acc = rej_tau = 0
    for step in range(n_eval):
        T = T0 * (T1 / T0) ** (step / max(n_eval - 1, 1))
        kind, r, c = dofs[int(torch.randint(len(dofs), (1,), generator=g))]
        if kind == "chi":
            a = float(torch.randn(1, generator=g)) * SIGMA_CHI
            cand = MV.apply_chi(cur, atom_mask, residue_type, r, c, a)
        else:
            a = float(torch.randn(1, generator=g)) * SIGMA_BACKRUB
            cand = MV.apply_backrub(cur, atom_mask, r, a)

            if MV.ca_angle_drift_sigma(cand, tau_ref, atom_mask, residue_type,
                                       (r - 1, r, r + 1)) > ANGLE_SIGMA_MAX:
                rej_tau += 1
                continue
        e = scorer(cand)
        if e <= cur_e or float(torch.rand(1, generator=g)) < np.exp(-(e - cur_e) / T):
            cur, cur_e = cand, e
            acc += 1
            if e < best_e:
                best, best_e = cand.clone(), e
    return best, best_e, {"n_eval": n_eval, "accepted": acc, "rejected_tau": rej_tau}

def pocket_residues(coors, atom_mask, res_mask, lig, dev):
    am = atom_mask & res_mask[:, None]
    idx = am.nonzero(as_tuple=False)
    d = torch.cdist(coors[idx[:, 0], idx[:, 1]][None], lig[None])[0].min(dim=1).values
    return sorted(set(int(i) for i in idx[d < POCKET_NM, 0]))

def build_dofs(arm, n, atom_mask, residue_type, res_mask, residues):

    chi_dof = []
    for r in sorted(set(residues)):
        rt = int(residue_type[r])
        if MV.res3(rt) in MV.RING_CLOSED:
            continue
        for c in range(MV.n_chis(rt)):
            ax = MV.chi_axis_atoms(rt, c)
            if ax and bool(atom_mask[r, ax[0]]) and bool(atom_mask[r, ax[1]]):
                chi_dof.append((r, c))
    br_dof = []
    if arm == "flexible":
        br_dof = [r for r in sorted(set(residues))
                  if 0 < r < n - 1 and bool(res_mask[r - 1]) and bool(res_mask[r + 1])
                  and MV.backrub_allowed(residue_type, r, n)]
    return chi_dof, br_dof

def run_arm(arm, coors0, atom_mask, residue_type, res_mask, scorer, motif_res,
            pocket, n_eval, restarts, seed0, dev):
    chi_dof, br_dof = build_dofs(arm, coors0.shape[0], atom_mask, residue_type,
                                 res_mask, set(motif_res) | set(pocket))
    tau_res = sorted({r for d in br_dof for r in (d - 1, d + 1)}) or [0]

    t0 = time.time()
    best, best_e, stats = None, float("inf"), {"accepted": 0, "rejected_tau": 0, "n_eval": 0}
    per = max(n_eval // restarts, 1)
    for k in range(restarts):
        b, e, st = anneal(coors0, atom_mask, residue_type, res_mask, scorer,
                          chi_dof, br_dof, per, seed0 + 1000 * k, coors0, tau_res)
        for kk in stats:
            stats[kk] += st.get(kk, 0)
        if e < best_e:
            best, best_e = b, e
    return best, best_e, {**stats, "n_chi_dof": len(chi_dof), "n_backrub_dof": len(br_dof),
                          "restarts": restarts, "wall_seconds": time.time() - t0}
