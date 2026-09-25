
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, hydra
os.chdir(R.UPSTREAM)

D_POLAR, D_APOLAR, CONTACT_MAX = 0.28, 0.34, 0.45
ATOM37_ELEM = ["N", "C", "C", "O"] + ["C"] * 33

def elements_for(atom_mask: torch.Tensor) -> list[str]:
    idx = atom_mask.nonzero(as_tuple=False)[:, 1].tolist()
    return [ATOM37_ELEM[i] if i < len(ATOM37_ELEM) else "C" for i in idx]

def pair_dmin(prot_elem, lig_polar, device):
    p = torch.tensor([e in ("N", "O") for e in prot_elem], device=device)
    m = p[:, None] | lig_polar[None, :]
    return torch.where(m, torch.tensor(D_POLAR, device=device),
                       torch.tensor(D_APOLAR, device=device))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--correct-last", type=int, default=60)
    ap.add_argument("--rms-frac", type=float, default=0.015)
    ap.add_argument("--w-motif", type=float, default=4.0)
    ap.add_argument("--w-contact", type=float, default=4.0)
    ap.add_argument("--arms", default="native,local_only,backbone_only,joint,local_led,backbone_led")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    model, linfo = R.load(args.device)
    fm = model.fm
    rows, meta = [], {"load": linfo, "nsteps": args.nsteps,
                      "correct_last": args.correct_last, "rms_frac": args.rms_frac,
                      "d_polar": D_POLAR, "d_apolar": D_APOLAR,
                      "contact_max": CONTACT_MAX, "arms": args.arms.split(","),
                      "w_motif": args.w_motif, "w_contact": args.w_contact,
                      "objective": "steric_overlap + w_contact*ligand_left_uncovered + w_motif*motif_drift",
                      "failures": []}
    t0 = time.time()

    for task in args.tasks.split(","):
        gen, cmeta = R.config(task, nsteps=args.nsteps, nsamples=1)
        model.configure_inference(gen, nn_ag=None)
        b0 = next(iter(hydra.utils.instantiate(gen.dataloader)))
        b0 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b0.items()}
        mask = b0["mask"]; bs = int(mask.shape[0])
        mm = b0["motif_mask"].bool(); xm = b0["x_motif"]
        motif_pts = xm[0][mm[0]].reshape(-1, 3)
        lmask = b0["target_mask"].bool()
        lig_pts = (b0["x_target"][0][lmask[0]] if lmask.dim() == 2 else b0["x_target"][0]).reshape(-1, 3)
        lig_elem_idx = b0["seq_target"][0].argmax(-1) if "seq_target" in b0 else None

        lig_polar = torch.zeros(lig_pts.shape[0], dtype=torch.bool, device=dev)
        SA = {m: gen.model[m] for m in ("bb_ca", "local_latents")}
        ts, gts = fm.sample_schedule(args.nsteps, SA)
        SP = {m: dict(SA[m]["simulation_step_params"]) for m in SA}
        start = args.nsteps - args.correct_last

        for seed in range(args.seeds):
            for arm in args.arms.split(","):
                torch.manual_seed(20260903 + seed); torch.cuda.manual_seed(20260903 + seed)
                st = fm.sample_noise(int(mask.shape[1]), shape=(bs,), device=dev, mask=mask)
                st = {m: st[m] for m in ("bb_ca", "local_latents")}
                nfe = nbwd = ndec = 0
                disp = {"bb_ca": 0.0, "local_latents": 0.0}

                def decode(z, ca):
                    nonlocal ndec
                    ndec += 1
                    return model.autoencoder.decode(z, ca, mask)

                def objective(z, ca, motif_ref):
                    d = decode(z, ca)
                    coords = d["coors_nm"][0]
                    am = d["atom_mask"][0].bool() & mask[0][:, None]
                    pts = coords[am]
                    elems = elements_for(am)
                    dmin = pair_dmin(elems, lig_polar, dev)
                    dist = torch.cdist(pts[None], lig_pts[None])[0]
                    steric = torch.relu(dmin - dist).pow(2).sum()

                    lig_nearest = dist.min(dim=0).values
                    away = torch.relu(lig_nearest - CONTACT_MAX).pow(2).sum()

                    dm = torch.cdist(motif_pts[None], pts[None])[0].min(dim=1).values
                    motif = torch.relu(dm - motif_ref).pow(2).sum()
                    return (steric + args.w_contact * away + args.w_motif * motif,
                            float(steric), float(motif))

                def motif_ref_now(z, ca):
                    with torch.no_grad():
                        d = decode(z, ca)
                        am = d["atom_mask"][0].bool() & mask[0][:, None]
                        pts = d["coors_nm"][0][am]
                        return torch.cdist(motif_pts[None], pts[None])[0].min(dim=1).values.detach()

                def apply_grad(state, movers, ref):

                    nonlocal nbwd
                    leaves = {m: state[m].detach().clone().requires_grad_(True) for m in movers}
                    z = leaves.get("local_latents", state["local_latents"])
                    ca = leaves.get("bb_ca", state["bb_ca"])
                    loss, _, _ = objective(z, ca, ref)
                    if float(loss) <= 0:
                        return state
                    g = torch.autograd.grad(loss, list(leaves.values()), allow_unused=True)
                    nbwd += 1
                    newst = dict(state)
                    for (m, _), gr in zip(leaves.items(), g):
                        if gr is None:
                            continue
                        scale = state[m].detach().pow(2).mean().sqrt()
                        gn = gr.pow(2).mean().sqrt().clamp(min=1e-12)
                        step = (args.rms_frac * scale / gn) * gr
                        disp[m] += float(step.pow(2).mean().sqrt())
                        newst[m] = (state[m] - step).detach()
                    return newst

                for i in range(args.nsteps):
                    bb = dict(b0)
                    bb["x_t"] = {m: st[m] for m in st}
                    bb["t"] = {m: torch.full((bs,), float(ts[m][i]), device=dev) for m in st}
                    bb["mask"] = mask
                    with torch.no_grad():
                        o = model.predict_for_sampling(bb, mode="full", n_recycle=0)
                    nfe += 1
                    o = fm.nn_out_add_simulation_tensor(batch=bb, nn_out=o)
                    for m in st:
                        o[m] = dict(o[m]); o[m]["v_guided"] = o[m]["v"]

                    st = {m: fm.base_flow_matchers[m].simulation_step(
                        x_t=st[m], nn_out=o[m],
                        t=torch.full((bs,), float(ts[m][i]), device=dev),
                        dt=ts[m][i + 1] - ts[m][i], gt=gts[m][i], mask=mask,
                        simulation_step_params=SP[m]) for m in st}

                    if arm != "native" and i >= start:
                        ref = motif_ref_now(st["local_latents"], st["bb_ca"])
                        if arm == "local_only":
                            st = apply_grad(st, ("local_latents",), ref)
                        elif arm == "backbone_only":
                            st = apply_grad(st, ("bb_ca",), ref)
                        elif arm == "joint":
                            st = apply_grad(st, ("local_latents", "bb_ca"), ref)
                        elif arm == "local_led":

                            st = apply_grad(st, ("local_latents",), ref)
                            st = apply_grad(st, ("bb_ca",), ref)
                        elif arm == "backbone_led":
                            st = apply_grad(st, ("bb_ca",), ref)
                            st = apply_grad(st, ("local_latents",), ref)

                with torch.no_grad():
                    d = decode(st["local_latents"], st["bb_ca"])
                rows.append({"task": task, "seed": seed, "arm": arm,
                             "nfe": nfe, "backward": nbwd, "decoder_calls": ndec,
                             "disp_bb_ca": disp["bb_ca"], "disp_local": disp["local_latents"],
                             **evaluate(d, mask, motif_pts, lig_pts, lig_polar, st, dev,
                                        b0, mm)})
                print(f"{task} s{seed} {arm:13s} " +
                      " ".join(f"{k}={rows[-1][k]:.3f}" for k in
                               ("ca_ca_valid", "lig_clash_atoms", "valid_contacts",
                                "motif_matched_frac", "motif_rmsd_nm")) +
                      f"  bwd={nbwd} {(time.time()-t0)/60:.1f}min", flush=True)

    meta["runtime_minutes"] = (time.time() - t0) / 60.0
    pd.DataFrame(rows).to_parquet(out / "leadership.parquet", index=False)
    (out / "leadership_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("done", len(rows), "rows")

def evaluate(d, mask, motif_pts, lig_pts, lig_polar, st, dev, b0, mm):

    coords = d["coors_nm"][0]
    am = d["atom_mask"][0].bool() & mask[0][:, None]
    pts = coords[am]
    elems = elements_for(am)
    dmin = pair_dmin(elems, lig_polar, dev)
    dist = torch.cdist(pts[None], lig_pts[None])[0]
    near = dist.min(dim=1).values
    per_pair_min = dmin.min(dim=1).values
    clash = float((near < per_pair_min).sum())
    valid = float(((dist >= dmin) & (dist < CONTACT_MAX)).any(dim=1).sum())

    dmat = torch.cdist(motif_pts[None], pts[None])[0].clone()
    matched, dists = 0, []
    for _ in range(motif_pts.shape[0]):
        v, idx = dmat.view(-1).min(0)
        if not torch.isfinite(v) or float(v) > 0.15:
            break
        r, c = int(idx // dmat.shape[1]), int(idx % dmat.shape[1])
        matched += 1; dists.append(float(v))
        dmat[r, :] = float("inf"); dmat[:, c] = float("inf")
    ca = st["bb_ca"][0][mask[0].bool()]
    bond = (ca[1:] - ca[:-1]).norm(dim=-1)
    pair = torch.cdist(ca[None], ca[None])[0]
    n = ca.shape[0]
    far = ~torch.eye(n, dtype=torch.bool, device=ca.device)
    for k in (1, 2):
        far &= ~torch.diag(torch.ones(n-k, dtype=torch.bool, device=ca.device), k)
        far &= ~torch.diag(torch.ones(n-k, dtype=torch.bool, device=ca.device), -k)
    return {"n_atoms": int(pts.shape[0]),
            "lig_clash_atoms": clash,
            "valid_contacts": valid,
            "lig_min_dist_nm": float(near.min()),
            "motif_matched_frac": matched / max(motif_pts.shape[0], 1),
            "motif_rmsd_nm": float(np.sqrt(np.mean(np.square(dists)))) if dists else float("nan"),
            "ca_ca_valid": float(((bond > 0.34) & (bond < 0.42)).float().mean()),
            "ca_ca_mean_nm": float(bond.mean()),
            "self_clash_frac": float((pair[far] < 0.30).float().mean()),
            "radius_gyration_nm": float((ca - ca.mean(0)).norm(dim=-1).pow(2).mean().sqrt()),
            "finite": bool(torch.isfinite(coords[am]).all())}

if __name__ == "__main__":
    main()
