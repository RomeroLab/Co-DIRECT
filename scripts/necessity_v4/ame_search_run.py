
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, yaml
import ame_requirements as Q, ame_moves as MV, ame_search as S

def load_spec(task):
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    return td["motif_target_dict_cfg"][task]["contig_atoms"]

def gates(coors, native, atom_mask, res_mask, residue_type, tau_res):

    g = MV.chain_geometry(coors, atom_mask, res_mask)
    g0 = MV.chain_geometry(native, atom_mask, res_mask)
    live = res_mask.nonzero(as_tuple=False).flatten().tolist()

    g["bond_length_drift_nm"] = MV.bond_length_drift(
        coors, native, atom_mask, res_mask, residue_type)
    g["bond_angle_drift_deg"] = MV.bond_angle_drift_deg(
        coors, native, atom_mask, res_mask, residue_type)
    g["rigid_group_drift_nm"] = MV.rigid_invariance_drift(
        coors, atom_mask, native, residue_type, live)
    g["tau_drift_deg"] = MV.backbone_angle_drift(coors, native, atom_mask, tau_res)
    g["angle_drift_sigma"] = MV.angle_drift_sigma(
        coors, native, atom_mask, residue_type,
        res_mask.nonzero(as_tuple=False).flatten().tolist())
    g["ca_bond_valid_delta"] = g["ca_ca_valid_frac"] - g0["ca_ca_valid_frac"]
    g["peptide_valid_delta"] = g["peptide_cn_valid_frac"] - g0["peptide_cn_valid_frac"]
    g["self_clash_delta"] = g["ca_self_clash_frac"] - g0["ca_self_clash_frac"]
    g["passes"] = bool(g["bond_length_drift_nm"] < 1e-4
                       and g["angle_drift_sigma"] <= S.ANGLE_SIGMA_MAX + 1e-6
                       and g["ca_bond_valid_delta"] >= -0.01
                       and g["peptide_valid_delta"] >= -0.01
                       and g["self_clash_delta"] <= 0.005)
    return g

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", type=int, default=20000)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-poses", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    want = set(a.tasks.split(",")) if a.tasks else None
    rows, atomrows, errs = [], [], []
    t_start = time.time()

    for td in sorted(Path(a.root).iterdir()):
        if not td.is_dir() or (want and td.name not in want):
            continue
        task = td.name
        try:
            spec = load_spec(task)
        except KeyError:
            errs.append({"task": task, "error": "not in ame_dict_v2"}); continue
        for pt in sorted(td.glob("native_seed*.pt")):
            try:
                s = torch.load(pt, map_location="cpu", weights_only=False)
                lig = s["x_target"][0][s["target_mask"][0].bool()].reshape(-1, 3)
                reqs = Q.build_requirements(spec, s["x_motif"], s["motif_mask"],
                                            s["seq_motif"], lig)
                nat_rows = Q.assign_and_score(reqs, s["coors_nm"], s["atom_mask"],
                                              s["residue_type"], s["mask"])
            except Exception as exc:
                errs.append({"task": task, "file": pt.name,
                             "error": f"{type(exc).__name__}: {exc}"[:300]}); continue

            coors = s["coors_nm"][0].to(dev)
            am = s["atom_mask"][0].bool().to(dev)
            rt = s["residue_type"][0].to(dev)
            rm = s["mask"][0].bool().to(dev)
            ligd = lig.to(dev)

            targets = [(r["gen_res_index"], r["atom37"],
                        torch.tensor([q.xyz_nm for q in reqs
                                      if q.motif_res == r["motif_res"]
                                      and q.atom37 == r["atom37"]][0]))
                       for r in nat_rows if r["atom_present"]]
            if not targets:
                errs.append({"task": task, "file": pt.name, "error": "no present targets"})
                continue
            motif_res = sorted({r["gen_res_index"] for r in nat_rows})
            pocket = S.pocket_residues(coors, am, rm, ligd, dev)
            scorer = S.Scorer(coors, am, rm, ligd, targets, dev)
            base_e = scorer(coors)
            seed = abs(hash((task, pt.name))) % 100000

            per_arm = {}
            for arm in ("fixed", "flexible"):
                best, best_e, st = S.run_arm(arm, coors, am, rt, rm, scorer, motif_res,
                                             pocket, a.evals, a.restarts, seed, dev)
                tau_res = sorted({x for r in (motif_res + pocket) for x in (r - 1, r + 1)
                                  if 0 <= x < coors.shape[0]})
                gg = gates(best, coors, am, rm, rt, tau_res)
                s2 = dict(s)
                s2["coors_nm"] = best[None].cpu()
                ev_rows = Q.assign_and_score(reqs, s2["coors_nm"], s["atom_mask"],
                                             s["residue_type"], s["mask"])
                sm = Q.scaffold_and_ligand_metrics(s2["coors_nm"], s["atom_mask"],
                                                   s["mask"], s["bb_ca"], lig,
                                                   {r["gen_res_index"] for r in ev_rows})
                per_arm[arm] = (best, best_e, st, gg, ev_rows, sm)
                if a.save_poses:
                    torch.save({**{k: v for k, v in s.items()},
                                "coors_nm": best[None].cpu(), "arm": arm,
                                "objective": best_e, "gates": gg},
                               out / f"{task}__{pt.stem}__{arm}.pt")

            def summarise(rws, sm):
                ok = np.array([r["placed"] for r in rws])
                dv = np.array([r["deviation_nm"] for r in rws], float)
                return {"placed": int(ok.sum()), "n_req": len(rws),
                        "all_placed": bool(ok.all()),
                        "rmsd_nm": float(np.sqrt(np.nanmean(dv ** 2))),
                        "max_dev_nm": float(np.nanmax(dv)) if np.isfinite(dv).any() else np.nan,
                        "clashes": sm["clashes_total"],
                        "clash_depth_nm": sm["clash_depth_sum_nm"],
                        "contacts": sm["ligand_contact_atoms"],
                        "lig_uncovered": sm["ligand_atoms_uncovered"],
                        "lig_min_nm": sm["ligand_min_dist_nm"]}

            nat_sm = Q.scaffold_and_ligand_metrics(s["coors_nm"], s["atom_mask"], s["mask"],
                                                   s["bb_ca"], lig,
                                                   {r["gen_res_index"] for r in nat_rows})
            row = {"task": task, "file": pt.name, "base_objective": base_e,
                   "n_motif_res": len(motif_res), "n_pocket": len(pocket),
                   "atoms_absent": int(sum(not r["atom_present"] for r in nat_rows))}
            for k, v in summarise(nat_rows, nat_sm).items():
                row[f"native_{k}"] = v
            for arm in ("fixed", "flexible"):
                best, best_e, st, gg, ev, sm = per_arm[arm]
                for k, v in summarise(ev, sm).items():
                    row[f"{arm}_{k}"] = v
                row[f"{arm}_objective"] = best_e
                row[f"{arm}_gates_pass"] = gg["passes"]
                row[f"{arm}_tau_drift_deg"] = gg["tau_drift_deg"]
                row[f"{arm}_rigid_drift_nm"] = gg["rigid_group_drift_nm"]
                row[f"{arm}_bond_drift_nm"] = gg["bond_length_drift_nm"]
                row[f"{arm}_angle_drift_deg"] = gg["bond_angle_drift_deg"]
                row[f"{arm}_angle_sigma"] = gg["angle_drift_sigma"]
                row[f"{arm}_wall_s"] = st["wall_seconds"]
                row[f"{arm}_accepted"] = st["accepted"]
                row[f"{arm}_n_chi_dof"] = st["n_chi_dof"]
                row[f"{arm}_n_backrub_dof"] = st["n_backrub_dof"]
                row[f"{arm}_rejected_tau"] = st["rejected_tau"]
                for r0, r1 in zip(nat_rows, ev):
                    atomrows.append({"task": task, "file": pt.name, "arm": arm,
                                     "atom_name": r1["atom_name"], "res_id": r1["res_id"],
                                     "is_sidechain": r1["is_sidechain"],
                                     "atom_present": r1["atom_present"],
                                     "native_dev_nm": r0["deviation_nm"],
                                     "arm_dev_nm": r1["deviation_nm"],
                                     "native_placed": r0["placed"], "arm_placed": r1["placed"]})
            rows.append(row)
            print(f"{task} {pt.stem} native={row['native_placed']}/{row['native_n_req']} "
                  f"fixed={row['fixed_placed']} flex={row['flexible_placed']} "
                  f"gates f/x={row['fixed_gates_pass']}/{row['flexible_gates_pass']} "
                  f"{(time.time()-t_start)/60:.1f}min", flush=True)

    pd.DataFrame(rows).to_parquet(out / "search_results.parquet", index=False)
    pd.DataFrame(atomrows).to_parquet(out / "search_atoms.parquet", index=False)
    (out / "search_errors.json").write_text(json.dumps(errs, indent=2, default=str))
    print(json.dumps({"designs": len(rows), "errors": len(errs),
                      "minutes": round((time.time()-t_start)/60, 1)}, indent=2))

if __name__ == "__main__":
    main()
