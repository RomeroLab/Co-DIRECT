
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lig_rt
_lig_rt.bootstrap()
import numpy as np, pandas as pd, torch, hydra

os.chdir(_lig_rt.UPSTREAM)
CLASH_NM, CONTACT_NM, TARGET_NM = 0.30, 0.45, 0.35

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(_lig_rt.TASKS))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--nsteps", type=int, default=100)
    ap.add_argument("--correct-last", type=int, default=25)
    ap.add_argument("--rms-frac", type=float, default=0.02)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    model, info = _lig_rt.load_model(args.device)
    fm = model.fm

    def clash_loss(z, ca, mask, lig, lmask):
        d = model.autoencoder.decode(z, ca, mask)
        coords = d["coors_nm"][0]
        amask = d["atom_mask"][0].bool() & mask[0][:, None]
        pts = coords[amask]
        L = lig[0][lmask[0].bool()]
        dist = torch.cdist(pts[None], L[None])[0]
        return (torch.relu(TARGET_NM - dist.min(dim=1).values) ** 2).sum()

    def geometry(z, ca, mask, lig, lmask):
        with torch.no_grad():
            d = model.autoencoder.decode(z, ca, mask)
            coords = d["coors_nm"][0]
            amask = d["atom_mask"][0].bool() & mask[0][:, None]
            pts = coords[amask]
            L = lig[0][lmask[0].bool()]
            near = torch.cdist(pts[None], L[None])[0].min(dim=1).values
            ca_ = ca[0][mask[0].bool()]
            bond = (ca_[1:] - ca_[:-1]).norm(dim=-1)
            return dict(clash=float((near < CLASH_NM).sum()),
                        contact=float(((near >= CLASH_NM) & (near < CONTACT_NM)).sum()),
                        min_dist=float(near.min()),
                        ca_ca_valid=float(((bond > 0.34) & (bond < 0.42)).float().mean()),
                        n_atoms=int(pts.shape[0]))

    rows, meta = [], {"load": info, "nsteps": args.nsteps,
                      "correct_last": args.correct_last, "rms_frac": args.rms_frac,
                      "target_nm": TARGET_NM, "cost": {}, "failures": []}
    t0 = time.time()

    for task in args.tasks.split(","):
        gen = _lig_rt.config(task, nsteps=args.nsteps, nsamples=1)
        model.configure_inference(gen, nn_ag=None)
        b0 = next(iter(hydra.utils.instantiate(gen.dataloader)))
        b0 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b0.items()}
        mask = b0["mask"]; bs, n_res = mask.shape
        lig, lmask = b0["x_target"], b0["target_mask"]
        SA = {m: gen.model[m] for m in ("bb_ca", "local_latents")}
        ts, gts = fm.sample_schedule(args.nsteps, SA)
        SP = {m: dict(SA[m]["simulation_step_params"]) for m in SA}
        start = args.nsteps - args.correct_last

        for seed in range(args.seeds):
            for arm in ("native", "z_only", "x_only", "both"):
                torch.manual_seed(20260903 + seed); torch.cuda.manual_seed(20260903 + seed)
                st = fm.sample_noise(n_res, shape=(bs,), device=dev, mask=mask)
                st = {m: st[m] for m in ("bb_ca", "local_latents")}
                nfe = nbwd = 0
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
                        movers = {"z_only": ("local_latents",), "x_only": ("bb_ca",),
                                  "both": ("local_latents", "bb_ca")}[arm]
                        leaves = {m: st[m].detach().clone().requires_grad_(True) for m in movers}
                        z = leaves.get("local_latents", st["local_latents"])
                        ca = leaves.get("bb_ca", st["bb_ca"])
                        loss = clash_loss(z, ca, mask, lig, lmask)
                        if float(loss) > 0:
                            g = torch.autograd.grad(loss, list(leaves.values()),
                                                    allow_unused=True)
                            nbwd += 1
                            for (m, leaf), gr in zip(leaves.items(), g):
                                if gr is None:
                                    continue

                                scale = st[m].detach().pow(2).mean().sqrt()
                                gn = gr.pow(2).mean().sqrt().clamp(min=1e-12)
                                st[m] = (st[m] - (args.rms_frac * scale / gn) * gr).detach()
                        del leaves
                geo = geometry(st["local_latents"], st["bb_ca"], mask, lig, lmask)
                rows.append({"task": task, "seed": seed, "arm": arm, "nfe": nfe,
                             "backward": nbwd, "decoder_calls": nbwd + 1, **geo})
                print(f"{task} s{seed} {arm:7s} clash={geo['clash']:.0f} "
                      f"contact={geo['contact']:.0f} min={geo['min_dist']:.3f} "
                      f"ca_valid={geo['ca_ca_valid']:.3f} nfe={nfe} bwd={nbwd} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)

    meta["runtime_minutes"] = (time.time() - t0) / 60.0
    pd.DataFrame(rows).to_parquet(out / "recover.parquet", index=False)
    (out / "recover_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("done")

if __name__ == "__main__":
    main()
