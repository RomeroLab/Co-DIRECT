
from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, yaml
import ame_requirements as Q

def spec_for(task: str) -> str:
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    return td["motif_target_dict_cfg"][task]["contig_atoms"]

def evaluate_state(task, spec, s, tol_nm=Q.TOL_NM):
    lig = s["x_target"][0][s["target_mask"][0].bool()] if s["target_mask"].dim() == 2          else s["x_target"][0]
    lig = lig.reshape(-1, 3)
    reqs = Q.build_requirements(spec, s["x_motif"], s["motif_mask"], s["seq_motif"], lig)
    rows = Q.assign_and_score(reqs, s["coors_nm"], s["atom_mask"], s["residue_type"],
                              s["mask"], tol_nm=tol_nm)
    gen_idx = {r["gen_res_index"] for r in rows}
    sm = Q.scaffold_and_ligand_metrics(s["coors_nm"], s["atom_mask"], s["mask"],
                                       s["bb_ca"], lig, gen_idx)
    dev = np.array([r["deviation_nm"] for r in rows], dtype=float)
    ok = np.array([r["placed"] for r in rows])
    sc = np.array([r["is_sidechain"] for r in rows])
    summary = {"task": task, "n_req": len(rows),
               "placed_frac": float(ok.mean()),
               "placed_frac_sidechain": float(ok[sc].mean()) if sc.any() else np.nan,
               "placed_frac_backbone": float(ok[~sc].mean()) if (~sc).any() else np.nan,
               "n_failed": int((~ok).sum()),
               "atoms_absent": int(sum(not r["atom_present"] for r in rows)),
               "res_identity_frac": float(np.mean([r["res_identity_match"] for r in rows])),
               "motif_rmsd_nm": float(np.sqrt(np.nanmean(dev ** 2))),
               "max_deviation_nm": float(np.nanmax(dev)) if np.isfinite(dev).any() else np.nan,
               "all_placed": bool(ok.all()), **sm}
    for t in Q.TOL_SENSITIVITY_NM:
        summary[f"placed_frac_tol{int(t*100)}"] = float(
            np.mean([bool(r["atom_present"]) and r["deviation_nm"] <= t for r in rows]))
    return reqs, rows, summary

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--glob", default="native_seed*.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    atom_rows, design_rows, sanity, errs = [], [], {}, []

    for td in sorted(Path(a.root).iterdir()):
        if not td.is_dir():
            continue
        task = td.name
        try:
            spec = spec_for(task)
        except KeyError:
            errs.append({"task": task, "error": "not in ame_dict_v2"}); continue
        for pt in sorted(td.glob(a.glob)):
            try:
                s = torch.load(pt, map_location="cpu", weights_only=False)
                reqs, rows, summ = evaluate_state(task, spec, s)
            except Exception as exc:
                errs.append({"task": task, "file": pt.name,
                             "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            if task not in sanity:
                lig = s["x_target"][0][s["target_mask"][0].bool()].reshape(-1, 3)
                sanity[task] = Q.input_sanity(reqs, lig)
                sanity[task]["contig_atoms"] = spec
                sanity[task]["bond_graph_available"] = bool(
                    int(s["target_bond_mask"].sum()) > 0)
            summ["file"] = pt.name
            summ["seed"] = int("".join(c for c in pt.stem if c.isdigit())[-1:] or 0)
            design_rows.append(summ)
            for r in rows:
                atom_rows.append({"task": task, "file": pt.name, **r})

    pd.DataFrame(atom_rows).to_parquet(out / "requirement_atoms.parquet", index=False)
    pd.DataFrame(design_rows).to_parquet(out / "design_summary.parquet", index=False)
    (out / "input_sanity.json").write_text(json.dumps(sanity, indent=2, default=str))
    (out / "eval_errors.json").write_text(json.dumps(errs, indent=2, default=str))
    print(json.dumps({"designs": len(design_rows), "atom_rows": len(atom_rows),
                      "tasks": len({d['task'] for d in design_rows}),
                      "errors": len(errs)}, indent=2))

if __name__ == "__main__":
    main()
