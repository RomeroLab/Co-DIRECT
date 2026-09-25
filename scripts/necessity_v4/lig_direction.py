
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lig_rt
_lig_rt.bootstrap()
import numpy as np, pandas as pd, torch, hydra

os.chdir(_lig_rt.UPSTREAM)
CLASH_NM, CONTACT_NM, POCKET_NM = 0.30, 0.45, 0.80

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(_lig_rt.TASKS))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--nsteps", type=int, default=100)
    ap.add_argument("--branch-fracs", default="0.50,0.75")
    ap.add_argument("--h-steps", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    model, info = _lig_rt.load_model(args.device)
    fm = model.fm
    hashes = {k: float(v.double().abs().sum()) for k, v in
              list(model.state_dict().items())[:50]}

    def decode_all_atom(z, ca, mask):
        with torch.no_grad():
            return model.autoencoder.decode(z, ca, mask)

    def ligand_geometry(z, ca, mask, lig_xyz, lig_mask):

        d = decode_all_atom(z, ca, mask)
        coords = d["coors_nm"][0]
        amask = d["atom_mask"][0].bool() & mask[0][:, None]
        pts = coords[amask]
        lig = lig_xyz[0][lig_mask[0].bool()]
        if pts.numel() == 0 or lig.numel() == 0:
            return dict(clash=float("nan"), contact=float("nan"), score=float("nan"),
                        min_dist=float("nan"))
        dist = torch.cdist(pts[None], lig[None])[0]
        nearest = dist.min(dim=1).values
        clash = float((nearest < CLASH_NM).float().sum())
        contact = float(((nearest >= CLASH_NM) & (nearest < CONTACT_NM)).float().sum())
        return dict(clash=clash, contact=contact,
                    score=contact - 5.0 * clash, min_dist=float(nearest.min()))

    rows, meta = [], {"checkpoint": str(_lig_rt.CKPT), "autoencoder": str(_lig_rt.AE),
                      "load": info, "nsteps": args.nsteps,
                      "clash_nm": CLASH_NM, "contact_nm": CONTACT_NM,
                      "pocket_nm": POCKET_NM, "forwards": 0, "failures": []}
    fracs = [float(v) for v in args.branch_fracs.split(",")]
    t0 = time.time()

    for task in args.tasks.split(","):
        gen = _lig_rt.config(task, nsteps=args.nsteps, nsamples=1)
        model.configure_inference(gen, nn_ag=None)
        batch0 = next(iter(hydra.utils.instantiate(gen.dataloader)))
        batch0 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch0.items()}
        mask = batch0["mask"]; bs, n_res = mask.shape
        lig_xyz, lig_mask = batch0["x_target"], batch0["target_mask"]
        SA = {m: gen.model[m] for m in ("bb_ca", "local_latents")}
        ts, gts = fm.sample_schedule(args.nsteps, SA)
        SP = {m: dict(SA[m]["simulation_step_params"]) for m in SA}

        for seed in range(args.seeds):
            torch.manual_seed(20260903 + seed); torch.cuda.manual_seed(20260903 + seed)
            st = fm.sample_noise(n_res, shape=(bs,), device=dev, mask=mask)
            st = {m: st[m] for m in ("bb_ca", "local_latents")}

            def fwd(state, i):
                b = dict(batch0)
                b["x_t"] = {m: state[m] for m in state}
                b["t"] = {m: torch.full((bs,), float(ts[m][i]), device=dev) for m in state}
                b["mask"] = mask
                with torch.no_grad():
                    o = model.predict_for_sampling(b, mode="full", n_recycle=0)
                meta["forwards"] += 1
                o = fm.nn_out_add_simulation_tensor(batch=b, nn_out=o)

                for m in state:
                    o[m] = dict(o[m]); o[m]["v_guided"] = o[m]["v"]
                return {m: o[m] for m in state}

            def step(m, x, nn_out, i, scale=1.0):
                dt = (ts[m][i + 1] - ts[m][i]) * scale
                return fm.base_flow_matchers[m].simulation_step(
                    x_t=x, nn_out=nn_out, t=torch.full((bs,), float(ts[m][i]), device=dev),
                    dt=dt, gt=gts[m][i], mask=mask, simulation_step_params=SP[m])

            for i in range(args.nsteps):
                if i in {int(round(f * args.nsteps)) for f in fracs}:
                    o = fwd(st, i)
                    base = ligand_geometry(st["local_latents"], st["bb_ca"], mask,
                                           lig_xyz, lig_mask)

                    for mover, label in (("bb_ca", "x_only"), ("local_latents", "z_only"),
                                         ("both", "joint")):
                        s2 = {k: v.clone() for k, v in st.items()}
                        if mover == "both":
                            for mm in ("bb_ca", "local_latents"):
                                s2[mm] = step(mm, st[mm], o[mm], i, scale=float(args.h_steps))
                        else:
                            s2[mover] = step(mover, st[mover], o[mover], i,
                                             scale=float(args.h_steps))
                            held = "bb_ca" if mover != "bb_ca" else "local_latents"
                            assert torch.equal(s2[held], st[held])
                        g = ligand_geometry(s2["local_latents"], s2["bb_ca"], mask,
                                            lig_xyz, lig_mask)
                        rows.append({"task": task, "seed": seed, "branch_step": i,
                                     "branch_frac": i / args.nsteps, "mover": label,
                                     "base_score": base["score"], "new_score": g["score"],
                                     "d_score": g["score"] - base["score"],
                                     "base_clash": base["clash"], "new_clash": g["clash"],
                                     "base_contact": base["contact"], "new_contact": g["contact"],
                                     "base_min_dist": base["min_dist"],
                                     "new_min_dist": g["min_dist"],
                                     "n_ligand_atoms": int(lig_mask[0].sum())})
                    o_use = o
                else:
                    o_use = fwd(st, i)
                st = {m: step(m, st[m], o_use[m], i) for m in st}

            final = ligand_geometry(st["local_latents"], st["bb_ca"], mask, lig_xyz, lig_mask)
            rows.append({"task": task, "seed": seed, "branch_step": -1, "branch_frac": 1.0,
                         "mover": "final", "base_score": final["score"],
                         "new_score": final["score"], "d_score": 0.0,
                         "base_clash": final["clash"], "new_clash": final["clash"],
                         "base_contact": final["contact"], "new_contact": final["contact"],
                         "base_min_dist": final["min_dist"], "new_min_dist": final["min_dist"],
                         "n_ligand_atoms": int(lig_mask[0].sum())})
            print(f"{task} seed {seed} done  forwards={meta['forwards']} "
                  f"{(time.time()-t0)/60:.1f}min  final score={final['score']:.1f} "
                  f"clash={final['clash']:.0f} contact={final['contact']:.0f}", flush=True)

    after = {k: float(v.double().abs().sum()) for k, v in
             list(model.state_dict().items())[:50]}
    meta["weights_unchanged"] = all(abs(hashes[k] - after[k]) < 1e-9 for k in hashes)
    meta["runtime_minutes"] = (time.time() - t0) / 60.0
    pd.DataFrame(rows).to_parquet(out / "lig_direction.parquet", index=False)
    (out / "lig_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("forwards", "runtime_minutes",
                                           "weights_unchanged")}, indent=2))

if __name__ == "__main__":
    main()
