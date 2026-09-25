
from __future__ import annotations

from dive.codirect_paths import joined
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import ame_moves as MV
import ame_search as S

def displacement(a, b, atom_mask, res_mask):
    m = atom_mask.bool() & res_mask.bool()[:, None]
    return float((a[m] - b[m]).pow(2).sum(-1).mean().sqrt())

def matched_random(native, atom_mask, residue_type, res_mask, chi_dof, br_dof,
                   target_rmsd, seed, tau_res, tol=0.05, max_iter=400):

    g = torch.Generator().manual_seed(seed)
    dofs = [("chi", r, c) for (r, c) in chi_dof] + [("br", r, None) for r in br_dof]
    if not dofs or target_rmsd <= 0:
        return None, {"reason": "no dofs or zero target"}

    def draw(scale):
        cur = native.clone()
        for _ in range(len(dofs)):
            kind, r, c = dofs[int(torch.randint(len(dofs), (1,), generator=g))]
            if kind == "chi":
                cand = MV.apply_chi(cur, atom_mask, residue_type, r, c,
                                    float(torch.randn(1, generator=g)) * S.SIGMA_CHI * scale)
            else:
                cand = MV.apply_backrub(cur, atom_mask, r,
                                        float(torch.randn(1, generator=g)) * S.SIGMA_BACKRUB * scale)
                if MV.ca_angle_drift_sigma(cand, native, atom_mask, residue_type,
                                           (r - 1, r, r + 1)) > S.ANGLE_SIGMA_MAX:
                    continue
            cur = cand
        return cur

    lo, hi = 0.0, 4.0
    best, best_err = None, float("inf")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = draw(mid)
        r = displacement(p, native, atom_mask, res_mask)
        err = abs(r - target_rmsd) / max(target_rmsd, 1e-9)
        if err < best_err:
            best, best_err = p, err
        if err < tol:
            break
        if r < target_rmsd:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            lo, hi = 0.0, hi * 2 if r < target_rmsd else hi
            if hi > 64:
                break
    return best, {"achieved_rel_error": best_err, "target_rmsd_nm": target_rmsd}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", nargs="+", required=True, help="dirs of task__stem__arm.pt")
    ap.add_argument("--arm", default="flexible")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pt in sorted(q for d in a.poses for q in Path(d).glob(f"*__{a.arm}.pt")):
        task, stem, arm = pt.stem.split("__")
        s = torch.load(pt, map_location="cpu", weights_only=False)
        native_path = Path(joined('CODIRECT_STRUCTURE_ROOT', 'ame_native')) / task / f"{stem}.pt"
        nat = torch.load(native_path, map_location="cpu", weights_only=False)
        native = nat["coors_nm"][0]
        improved = s["coors_nm"][0]
        am = nat["atom_mask"][0].bool(); rt = nat["residue_type"][0]; rm = nat["mask"][0].bool()
        lig = nat["x_target"][0][nat["target_mask"][0].bool()].reshape(-1, 3)
        target = displacement(improved, native, am, rm)
        pocket = S.pocket_residues(native, am, rm, lig, torch.device("cpu"))
        motif_res = sorted({int(i) for i in s.get("motif_res", [])}) if "motif_res" in s else []
        chi_dof, br_dof = S.build_dofs(arm, native.shape[0], am, rt, rm,
                                       set(pocket) | set(motif_res))
        tau_res = sorted({x for r in br_dof for x in (r - 1, r + 1)}) or [0]
        pose, info = matched_random(native, am, rt, rm, chi_dof, br_dof, target,
                                    abs(hash(pt.stem)) % 99991, tau_res)
        if pose is None:
            rows.append({"task": task, "file": pt.name, "ok": False, **info}); continue
        bl = MV.bond_length_drift(pose, native, am, rm, rt)
        td = MV.angle_drift_sigma(pose, native, am, rt,
                                  rm.nonzero(as_tuple=False).flatten().tolist())
        torch.save({**nat, "coors_nm": pose[None], "arm": "random_matched"},
                   out / f"{task}__{stem}__randommatched.pt")
        rows.append({"task": task, "file": pt.name, "ok": True,
                     "target_rmsd_nm": target,
                     "achieved_rmsd_nm": displacement(pose, native, am, rm),
                     "rel_error": info["achieved_rel_error"],
                     "bond_drift_nm": bl, "angle_sigma": td,
                     "gates_pass": bool(bl < 1e-4 and td <= S.ANGLE_SIGMA_MAX + 1e-6)})
        print(f"{task} {stem} target={target:.4f} got={rows[-1]['achieved_rmsd_nm']:.4f} "
              f"err={info['achieved_rel_error']:.3f}", flush=True)
    (out / "control_report.json").write_text(json.dumps(rows, indent=2, default=str))
    ok = sum(1 for r in rows if r.get("ok"))
    print(json.dumps({"controls": ok, "attempted": len(rows)}, indent=2))

if __name__ == "__main__":
    main()
