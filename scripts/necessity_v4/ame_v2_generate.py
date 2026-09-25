
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, hydra
os.chdir(R.UPSTREAM)

MATCH_NM, CLASH_NM, CONTACT_NM = 0.15, 0.30, 0.45
ELEMENTS = {0: "N", 1: "C", 2: "C", 3: "O"}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="M0024_1nzy_v3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--nsamples", type=int, default=1)
    ap.add_argument("--motif-subset", type=int, default=0,
                    help="drop this many trailing motif residues (valid alternative spec)")
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    gen, cmeta = R.config(args.task, nsteps=args.nsteps, nsamples=args.nsamples)
    model, linfo = R.load(args.device)
    model.configure_inference(gen, nn_ag=None)
    batch0 = next(iter(hydra.utils.instantiate(gen.dataloader)))
    batch0 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch0.items()}

    if args.motif_subset:

        keep = batch0["motif_mask"].shape[1] - args.motif_subset
        for k in ("motif_mask", "x_motif", "seq_motif", "seq_motif_mask"):
            if k in batch0 and torch.is_tensor(batch0[k]):
                batch0[k] = batch0[k][:, :keep].contiguous()

    mask = batch0["mask"]; bs = int(mask.shape[0])
    mm = batch0["motif_mask"].bool()
    xm = batch0["x_motif"]
    lig = batch0["x_target"]; lmask = batch0["target_mask"].bool()
    motif_pts = xm[0][mm[0]].reshape(-1, 3)
    lig_pts = lig[0][lmask[0]] if lmask.dim() == 2 else lig[0]
    lig_pts = lig_pts.reshape(-1, 3)

    rows, meta = [], {"config": cmeta, "load": linfo, "nsteps": args.nsteps,
                      "tag": args.tag, "motif_subset_dropped": args.motif_subset,
                      "match_nm": MATCH_NM, "clash_nm": CLASH_NM,
                      "n_motif_atoms": int(motif_pts.shape[0]),
                      "n_ligand_atoms": int(lig_pts.shape[0]), "failures": []}
    t0 = time.time()

    for seed in range(args.seeds):
        torch.manual_seed(20260903 + seed); torch.cuda.manual_seed(20260903 + seed)
        try:
            with torch.no_grad():
                sample = model.generate(dict(batch0))
        except Exception as exc:
            meta["failures"].append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            print(f"seed {seed} FAILED {type(exc).__name__}: {exc}", flush=True); continue
        ca, z = sample["bb_ca"], sample["local_latents"]
        with torch.no_grad():
            dec = model.autoencoder.decode(z, ca, mask)
        coords = dec["coors_nm"][0]
        amask = dec["atom_mask"][0].bool() & mask[0][:, None]
        pts = coords[amask]
        finite = bool(torch.isfinite(ca).all() and torch.isfinite(coords[amask]).all())
        cav = ca[0][mask[0].bool()]
        bond = (cav[1:] - cav[:-1]).norm(dim=-1)
        pair = torch.cdist(cav[None], cav[None])[0]
        n = cav.shape[0]
        far = ~torch.eye(n, dtype=torch.bool, device=cav.device)
        for k in (1, 2):
            far &= ~torch.diag(torch.ones(n-k, dtype=torch.bool, device=cav.device), k)
            far &= ~torch.diag(torch.ones(n-k, dtype=torch.bool, device=cav.device), -k)
        d_lig = torch.cdist(pts[None], lig_pts[None])[0].min(dim=1).values
        d_motif = torch.cdist(motif_pts[None], pts[None])[0].min(dim=1).values
        rows.append({
            "task": args.task, "tag": args.tag, "seed": seed, "finite": finite,
            "n_residues": int(mask[0].sum()), "n_atoms": int(pts.shape[0]),
            "ca_ca_mean_nm": float(bond.mean()),
            "ca_ca_valid_frac": float(((bond > 0.34) & (bond < 0.42)).float().mean()),
            "self_clash_frac": float((pair[far] < 0.30).float().mean()),
            "radius_gyration_nm": float((cav - cav.mean(0)).norm(dim=-1).pow(2).mean().sqrt()),
            "ligand_clash_atoms": float((d_lig < CLASH_NM).sum()),
            "ligand_contact_atoms": float(((d_lig >= CLASH_NM) & (d_lig < CONTACT_NM)).sum()),
            "ligand_min_dist_nm": float(d_lig.min()),
            "motif_atoms": int(motif_pts.shape[0]),
            "motif_placed_frac": float((d_motif < MATCH_NM).float().mean()),
            "motif_nearest_median_nm": float(d_motif.median()),
            "motif_nearest_min_nm": float(d_motif.min()),
        })
        torch.save({"bb_ca": ca.cpu(), "local_latents": z.cpu(),
                    "coors_nm": dec["coors_nm"].cpu(),
                    "atom_mask": dec["atom_mask"].cpu(),
                    "residue_type": dec["residue_type"].cpu(),
                    "x_motif": xm.cpu(), "motif_mask": mm.cpu(),
                    "x_target": lig.cpu(), "target_mask": lmask.cpu()},
                   out / f"sample_{args.tag}_seed{seed}.pt")
        print(f"seed {seed} ok  ca_valid={rows[-1]['ca_ca_valid_frac']:.3f} "
              f"lig_clash={rows[-1]['ligand_clash_atoms']:.0f} "
              f"lig_min={rows[-1]['ligand_min_dist_nm']:.3f} "
              f"motif_placed={rows[-1]['motif_placed_frac']:.2f} "
              f"{(time.time()-t0)/60:.1f}min", flush=True)

    meta["runtime_minutes"] = (time.time() - t0) / 60.0
    meta["n_completed"] = len(rows)
    pd.DataFrame(rows).to_parquet(out / f"metrics_{args.tag}.parquet", index=False)
    (out / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({"completed": len(rows), "failures": len(meta["failures"]),
                      "runtime_minutes": round(meta["runtime_minutes"], 2)}, indent=2))

if __name__ == "__main__":
    main()
