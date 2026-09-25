
from __future__ import annotations

import argparse, json, sys, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, yaml, os
import ame_requirements as Q
os.chdir(R.UPSTREAM)

def ae_batch(coors, atom_mask, residue_type, res_mask, ca, dev):

    am = atom_mask.bool()
    n = coors.shape[1]
    return {"coords_nm": coors, "coords": coors * 10.0, "coord_mask": am,
            "mask": res_mask.bool(), "residue_type": residue_type, "atom_mask": am,
            "ca_coors_nm": ca,
            "residue_pdb_idx": torch.arange(1, n + 1, device=dev,
                                            dtype=torch.float32)[None],
            "mask_dict": {"coords": am[..., None].expand(-1, -1, -1, 3),
                          "residue_type": res_mask.bool()}}

def round_trip(ae, coors, atom_mask, residue_type, res_mask, ca, dev, seed=0):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    b = ae_batch(coors, atom_mask, residue_type, res_mask, ca, dev)
    with torch.no_grad():
        enc = ae.encode(dict(b))
        out_z = ae.decode(enc["z_latent"], ca, res_mask.bool())
        out_m = ae.decode(ae.encoder.ln_z(enc["mean"]) * res_mask.bool()[..., None],
                          ca, res_mask.bool())
    return enc, out_z, out_m

def compare(spec, s, coors, ca, out, tag):

    lig = s["x_target"][0][s["target_mask"][0].bool()].reshape(-1, 3)
    reqs = Q.build_requirements(spec, s["x_motif"], s["motif_mask"], s["seq_motif"], lig)
    rows = Q.assign_and_score(reqs, out["coors_nm"].cpu(), out["atom_mask"].cpu(),
                              out["residue_type"].cpu(), s["mask"])
    ok = np.array([r["placed"] for r in rows])
    dv = np.array([r["deviation_nm"] for r in rows], float)
    sm = Q.scaffold_and_ligand_metrics(out["coors_nm"].cpu(), out["atom_mask"].cpu(),
                                       s["mask"], ca.cpu(), lig,
                                       {r["gen_res_index"] for r in rows})
    am_in = (s["atom_mask"][0].bool() & s["mask"][0].bool()[:, None])
    am_out = (out["atom_mask"][0].cpu().bool() & s["mask"][0].bool()[:, None])
    live = s["mask"][0].bool()
    return {f"{tag}_placed": int(ok.sum()), f"{tag}_n_req": len(rows),
            f"{tag}_all_placed": bool(ok.all()),
            f"{tag}_rmsd_nm": float(np.sqrt(np.nanmean(dv ** 2))),
            f"{tag}_clashes": sm["clashes_total"],
            f"{tag}_contacts": sm["ligand_contact_atoms"],
            f"{tag}_seq_identity_kept": float(
                (out["residue_type"][0].cpu()[live] == s["residue_type"][0][live]).float().mean()),
            f"{tag}_atom_mask_kept": float((am_in == am_out).float().mean()),
            f"{tag}_coord_rmsd_vs_input_nm": float(
                (out["coors_nm"][0].cpu()[am_in] - coors[0].cpu()[am_in]).pow(2)
                .sum(-1).mean().sqrt())}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-root", required=True)
    ap.add_argument("--pose-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model, linfo = R.load(a.device); ae = model.autoencoder
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    rows, errs = [], []

    jobs = []
    for d in sorted(Path(a.native_root).iterdir()):
        if d.is_dir():
            for pt in sorted(d.glob("native_seed*.pt")):
                jobs.append((d.name, "native", pt))
    if a.pose_root:
        for pt in sorted(Path(a.pose_root).glob("*.pt")):
            task, stem, arm = pt.stem.split("__")
            jobs.append((task, arm, pt))

    for task, arm, pt in jobs:
        try:
            spec = td["motif_target_dict_cfg"][task]["contig_atoms"]
            s = torch.load(pt, map_location="cpu", weights_only=False)
            coors = s["coors_nm"].to(dev)
            ca = s["bb_ca"].to(dev) if arm == "native" else                 s["coors_nm"][:, :, 1, :].to(dev)
            am = s["atom_mask"].to(dev); rt = s["residue_type"].to(dev)
            rm = s["mask"].to(dev)
            enc, oz, om = round_trip(ae, coors, am, rt, rm, ca, dev)
            row = {"task": task, "arm": arm, "file": pt.name,
                   "latent_std": float(enc["z_latent"].std()),
                   "log_scale_mean": float(enc["log_scale"].mean())}
            row.update(compare(spec, s, coors, ca, oz, "rt"))
            row.update(compare(spec, s, coors, ca, om, "rtmean"))

            lig = s["x_target"][0][s["target_mask"][0].bool()].reshape(-1, 3)
            reqs = Q.build_requirements(spec, s["x_motif"], s["motif_mask"],
                                        s["seq_motif"], lig)
            pre = Q.assign_and_score(reqs, s["coors_nm"], s["atom_mask"],
                                     s["residue_type"], s["mask"])
            row["pre_placed"] = int(sum(r["placed"] for r in pre))
            row["pre_n_req"] = len(pre)
            rows.append(row)
            print(f"{task} {arm} {pt.stem[:22]} pre={row['pre_placed']}/{row['pre_n_req']} "
                  f"rt={row['rt_placed']} rtmean={row['rtmean_placed']} "
                  f"coordRMSD={row['rt_coord_rmsd_vs_input_nm']:.3f}nm "
                  f"seq={row['rt_seq_identity_kept']:.3f}", flush=True)
        except Exception as exc:
            errs.append({"task": task, "arm": arm, "file": pt.name,
                         "error": f"{type(exc).__name__}: {exc}"[:300]})

    pd.DataFrame(rows).to_parquet(out / "roundtrip.parquet", index=False)
    (out / "roundtrip_errors.json").write_text(json.dumps(errs, indent=2, default=str))
    print(json.dumps({"rows": len(rows), "errors": len(errs)}, indent=2))

if __name__ == "__main__":
    main()
