
from __future__ import annotations

import argparse, json, sys, time, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, yaml
import ame_requirements as Q, ame_moves as MV, ame_search as S, ame_policies as P

def gates(pose, native, am, rm, rt):
    bl = MV.bond_length_drift(pose, native, am, rm, rt)
    sg = MV.angle_drift_sigma(pose, native, am, rt,
                              rm.nonzero(as_tuple=False).flatten().tolist())
    ch = MV.chain_geometry(pose, am, rm); c0 = MV.chain_geometry(native, am, rm)
    return {"bond_drift_nm": bl, "angle_sigma": sg,
            "ca_valid_delta": ch["ca_ca_valid_frac"] - c0["ca_ca_valid_frac"],
            "cn_valid_delta": ch["peptide_cn_valid_frac"] - c0["peptide_cn_valid_frac"],
            "self_clash_delta": ch["ca_self_clash_frac"] - c0["ca_self_clash_frac"],
            "passes": bool(bl < 1e-4 and sg <= S.ANGLE_SIGMA_MAX + 1e-6
                           and ch["ca_ca_valid_frac"] - c0["ca_ca_valid_frac"] >= -0.01
                           and ch["peptide_cn_valid_frac"] - c0["peptide_cn_valid_frac"] >= -0.01
                           and ch["ca_self_clash_frac"] - c0["ca_self_clash_frac"] <= 0.005)}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", type=int, default=32000)
    ap.add_argument("--n-cand", type=int, default=4)
    ap.add_argument("--lead-eval", type=int, default=200)
    ap.add_argument("--follow-eval", type=int, default=200)
    ap.add_argument("--block", type=int, default=800)
    ap.add_argument("--policies", default="joint,alternating,local_led,backbone_led,fixed_only")
    ap.add_argument("--save-poses", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    want = set(a.tasks.split(",")) if a.tasks else None
    rows, errs = [], []
    t_all = time.time()

    for d in sorted(Path(a.root).iterdir()):
        if not d.is_dir() or (want and d.name not in want):
            continue
        task = d.name
        try:
            spec = td["motif_target_dict_cfg"][task]["contig_atoms"]
        except KeyError:
            errs.append({"task": task, "error": "not in ame_dict_v2"}); continue
        for pt in sorted(d.glob("native_seed*.pt")):
            try:
                s = torch.load(pt, map_location="cpu", weights_only=False)
                lig = s["x_target"][0][s["target_mask"][0].bool()].reshape(-1, 3)
                reqs = Q.build_requirements(spec, s["x_motif"], s["motif_mask"],
                                            s["seq_motif"], lig)
                nat_rows = Q.assign_and_score(reqs, s["coors_nm"], s["atom_mask"],
                                              s["residue_type"], s["mask"])
                coors = s["coors_nm"][0]; am = s["atom_mask"][0].bool()
                rt = s["residue_type"][0]; rm = s["mask"][0].bool()
                dev = torch.device("cpu")
                targets = [(r["gen_res_index"], r["atom37"],
                            torch.tensor([q.xyz_nm for q in reqs
                                          if q.motif_res == r["motif_res"]
                                          and q.atom37 == r["atom37"]][0]))
                           for r in nat_rows if r["atom_present"]]
                if not targets:
                    errs.append({"task": task, "file": pt.name, "error": "no targets"}); continue
                motif_res = sorted({r["gen_res_index"] for r in nat_rows})
                pocket = S.pocket_residues(coors, am, rm, lig, dev)
                scorer = S.Scorer(coors, am, rm, lig, targets, dev)
                chi_dof, br_dof = S.build_dofs("flexible", coors.shape[0], am, rt, rm,
                                               set(motif_res) | set(pocket))
                nat_sm = Q.scaffold_and_ligand_metrics(s["coors_nm"], s["atom_mask"],
                                                       s["mask"], s["bb_ca"], lig,
                                                       {r["gen_res_index"] for r in nat_rows})
                base = {"task": task, "stem": pt.stem,
                        "native_placed": int(sum(r["placed"] for r in nat_rows)),
                        "n_req": len(nat_rows),
                        "native_all_placed": bool(all(r["placed"] for r in nat_rows)),
                        "native_clashes": nat_sm["clashes_total"],
                        "native_contacts": nat_sm["ligand_contact_atoms"],
                        "native_objective": scorer(coors),
                        "n_chi_dof": len(chi_dof), "n_backrub_dof": len(br_dof)}
                for name in a.policies.split(","):
                    rng = np.random.default_rng(abs(hash((task, pt.stem, name))) % (2**32))
                    budget = P.Budget(scorer, a.evals)
                    t0 = time.time()
                    pose, obj, info = P.POLICIES[name](
                        coors, am, rt, rm, budget, chi_dof, br_dof, rng, coors,
                        n_cand=a.n_cand, lead_eval=a.lead_eval,
                        follow_eval=a.follow_eval, block=a.block)
                    wall = time.time() - t0
                    gg = gates(pose, coors, am, rm, rt)
                    ev = Q.assign_and_score(reqs, pose[None], s["atom_mask"],
                                            s["residue_type"], s["mask"])
                    sm = Q.scaffold_and_ligand_metrics(pose[None], s["atom_mask"],
                                                       s["mask"], pose[None][:, :, 1, :],
                                                       lig, {r["gen_res_index"] for r in ev})
                    ok = np.array([r["placed"] for r in ev])
                    disp = float((pose[am & rm[:, None]] - coors[am & rm[:, None]])
                                 .pow(2).sum(-1).mean().sqrt())
                    rows.append({**base, "policy": name, "objective": obj,
                                 "evals_used": budget.used, "wall_s": wall,
                                 "outer_iters": info["outer"], "rejected": info["rejected"],
                                 "placed": int(ok.sum()), "all_placed": bool(ok.all()),
                                 "clashes": sm["clashes_total"],
                                 "clash_depth_nm": sm["clash_depth_sum_nm"],
                                 "contacts": sm["ligand_contact_atoms"],
                                 "lig_min_nm": sm["ligand_min_dist_nm"],
                                 "displacement_nm": disp, **gg})
                    if a.save_poses and gg["passes"]:
                        torch.save({**s, "coors_nm": pose[None], "policy": name},
                                   out / f"{task}__{pt.stem}__{name}.pt")
                print(f"{task} {pt.stem} " + " ".join(
                    f"{r['policy']}={r['placed']}/{r['clashes']}"
                    for r in rows[-len(a.policies.split(',')):])
                    + f" native={base['native_placed']}/{base['native_clashes']}"
                    + f" {(time.time()-t_all)/60:.1f}min", flush=True)
            except Exception as exc:
                import traceback
                errs.append({"task": task, "file": pt.name,
                             "error": traceback.format_exc()[-400:]})

    pd.DataFrame(rows).to_parquet(out / "policy_results.parquet", index=False)
    (out / "policy_errors.json").write_text(json.dumps(errs, indent=2, default=str))
    print(json.dumps({"rows": len(rows), "errors": len(errs),
                      "minutes": round((time.time()-t_all)/60, 1)}, indent=2))

if __name__ == "__main__":
    main()
