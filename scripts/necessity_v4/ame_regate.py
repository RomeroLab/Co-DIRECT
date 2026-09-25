
from __future__ import annotations
import argparse, json, sys, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, yaml
import ame_requirements as Q, ame_moves as MV, ame_search as S

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", nargs="+", required=True)
    ap.add_argument("--native-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    rows, errs = [], []
    files = [p for d in a.poses for p in sorted(Path(d).glob("*__*__*.pt"))]

    for pt in files:
        try:
            task, stem, arm = pt.stem.split("__")
            spec = td["motif_target_dict_cfg"][task]["contig_atoms"]
            nat = torch.load(Path(a.native_root) / task / f"{stem}.pt",
                             map_location="cpu", weights_only=False)
            s = torch.load(pt, map_location="cpu", weights_only=False)
            native = nat["coors_nm"][0]; pose = s["coors_nm"][0]
            am = nat["atom_mask"][0].bool(); rt = nat["residue_type"][0]
            rm = nat["mask"][0].bool()
            lig = nat["x_target"][0][nat["target_mask"][0].bool()].reshape(-1, 3)

            bl = MV.bond_length_drift(pose, native, am, rm, rt)
            ba = MV.bond_angle_drift_deg(pose, native, am, rm, rt)
            bs_ = MV.angle_drift_sigma(pose, native, am, rt,
                                       rm.nonzero(as_tuple=False).flatten().tolist())
            ch = MV.chain_geometry(pose, am, rm)
            ch0 = MV.chain_geometry(native, am, rm)
            passes = bool(bl < 1e-4 and bs_ <= S.ANGLE_SIGMA_MAX + 1e-6
                          and ch["ca_ca_valid_frac"] - ch0["ca_ca_valid_frac"] >= -0.01
                          and ch["peptide_cn_valid_frac"] - ch0["peptide_cn_valid_frac"] >= -0.01
                          and ch["ca_self_clash_frac"] - ch0["ca_self_clash_frac"] <= 0.005)

            reqs = Q.build_requirements(spec, nat["x_motif"], nat["motif_mask"],
                                        nat["seq_motif"], lig)
            ev = Q.assign_and_score(reqs, pose[None], nat["atom_mask"],
                                    nat["residue_type"], nat["mask"])
            sm = Q.scaffold_and_ligand_metrics(pose[None], nat["atom_mask"], nat["mask"],
                                               pose[None][:, :, 1, :], lig,
                                               {r["gen_res_index"] for r in ev})
            nrows = Q.assign_and_score(reqs, nat["coors_nm"], nat["atom_mask"],
                                       nat["residue_type"], nat["mask"])
            nsm = Q.scaffold_and_ligand_metrics(nat["coors_nm"], nat["atom_mask"], nat["mask"],
                                                nat["bb_ca"], lig,
                                                {r["gen_res_index"] for r in nrows})
            ok = np.array([r["placed"] for r in ev])
            nok = np.array([r["placed"] for r in nrows])
            disp = float((pose[am & rm[:, None]] - native[am & rm[:, None]])
                         .pow(2).sum(-1).mean().sqrt())
            rows.append({"task": task, "stem": stem, "arm": arm, "file": pt.name,
                         "gates_pass": passes, "bond_drift_nm": bl,
                         "angle_drift_deg": ba, "angle_sigma": bs_, "displacement_nm": disp,
                         "objective": float(s.get("objective", np.nan)),
                         "n_req": len(ev),
                         "placed": int(ok.sum()), "native_placed": int(nok.sum()),
                         "all_placed": bool(ok.all()), "native_all_placed": bool(nok.all()),
                         "clashes": sm["clashes_total"], "native_clashes": nsm["clashes_total"],
                         "clash_depth_nm": sm["clash_depth_sum_nm"],
                         "native_clash_depth_nm": nsm["clash_depth_sum_nm"],
                         "contacts": sm["ligand_contact_atoms"],
                         "native_contacts": nsm["ligand_contact_atoms"],
                         "lig_min_nm": sm["ligand_min_dist_nm"],
                         "native_lig_min_nm": nsm["ligand_min_dist_nm"],
                         "ca_valid": ch["ca_ca_valid_frac"], "native_ca_valid": ch0["ca_ca_valid_frac"],
                         "cn_valid": ch["peptide_cn_valid_frac"],
                         "native_cn_valid": ch0["peptide_cn_valid_frac"],
                         "self_clash": ch["ca_self_clash_frac"],
                         "native_self_clash": ch0["ca_self_clash_frac"]})
        except Exception as exc:
            errs.append({"file": pt.name, "error": f"{type(exc).__name__}: {exc}"[:300]})

    pd.DataFrame(rows).to_parquet(out / "regated.parquet", index=False)
    (out / "regate_errors.json").write_text(json.dumps(errs, indent=2, default=str))
    print(json.dumps({"poses": len(rows), "errors": len(errs)}, indent=2))

if __name__ == "__main__":
    main()
