
from __future__ import annotations

import argparse, json, sys, time, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, yaml, hydra, os
import ame_requirements as Q, ame_moves as MV
os.chdir(R.UPSTREAM)

BB, LT = "bb_ca", "local_latents"
EPS_T = 1e-4

def pocket_mask(coors, atom_mask, res_mask, lig, cutoff=0.80):
    am = atom_mask.bool() & res_mask.bool()[:, None]
    idx = am.nonzero(as_tuple=False)
    d = torch.cdist(coors[idx[:, 0], idx[:, 1]][None], lig[None])[0].min(dim=1).values
    m = torch.zeros(coors.shape[0], dtype=torch.bool, device=coors.device)
    m[idx[d < cutoff, 0]] = True
    return m

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--lo", type=float, default=0.55)
    ap.add_argument("--hi", type=float, default=0.95)
    ap.add_argument("--lag-steps", type=int, default=1,
                    help="how many steps back the sender state is taken from")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="multiplier on the modulation; 1.0 = full clamp for beta")
    ap.add_argument("--arms", default="natural,beta1,gamma1,beta1_perm,gamma1_perm,beta1_far,gamma1_far")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model, linfo = R.load(a.device); fm = model.fm; ae = model.autoencoder
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    want = set(a.tasks.split(",")) if a.tasks else None
    rows, errs = [], []
    natural_final = {}
    t_all = time.time()
    tasks = [d.name for d in sorted(Path(a.root).iterdir())
             if d.is_dir() and (not want or d.name in want)]

    for task in tasks:
        try:
            spec = td["motif_target_dict_cfg"][task]["contig_atoms"]
            gen, _ = R.config(task, nsteps=a.nsteps, nsamples=1)
            model.configure_inference(gen, nn_ag=None)
            b0 = next(iter(hydra.utils.instantiate(gen.dataloader)))
            b0 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b0.items()}
            mask = b0["mask"]; bs = int(mask.shape[0]); nres = int(mask.shape[1])
            SA = {m: gen.model[m] for m in (BB, LT)}
            ts, gts = fm.sample_schedule(a.nsteps, SA)
            SP = {m: dict(SA[m]["simulation_step_params"]) for m in SA}
            lo, hi = int(a.lo * a.nsteps), int(a.hi * a.nsteps)
            lig = b0["x_target"][0][b0["target_mask"][0].bool()].reshape(-1, 3)
            live = mask[0].bool()
        except Exception as exc:
            errs.append({"task": task, "error": f"{type(exc).__name__}: {exc}"[:300]}); continue

        for seed in range(a.seeds):
            arms = sorted(a.arms.split(","), key=lambda x: (x != "natural",))
            for arm in arms:
                try:
                    torch.manual_seed(70707 + seed); torch.cuda.manual_seed(70707 + seed)
                    st = fm.sample_noise(nres, shape=(bs,), device=dev, mask=mask)
                    st = {m: st[m] for m in (BB, LT)}
                    hist = {m: [st[m].clone()] for m in st}
                    beta = 1.0 if arm.startswith("beta") else 0.0
                    gamma = 1.0 if arm.startswith("gamma") else 0.0
                    perm_ctrl = arm.endswith("_perm"); far_ctrl = arm.endswith("_far")
                    g = torch.Generator(device="cpu").manual_seed(1234 + seed)
                    perm = torch.randperm(nres, generator=g).to(dev)
                    lag_mag, n_mod = [], 0
                    pmask = None
                    for i in range(a.nsteps):
                        bb = dict(b0); bb["mask"] = mask
                        bb["x_t"] = {m: st[m] for m in st}
                        bb["t"] = {m: torch.full((bs,), float(ts[m][i]), device=dev) for m in st}
                        with torch.no_grad():
                            o = model.predict_for_sampling(bb, mode="full", n_recycle=0)
                        o = fm.nn_out_add_clean_sample_prediction(batch=bb, nn_out=o)
                        o = fm.nn_out_add_simulation_tensor(batch=bb, nn_out=o)
                        o = {m: dict(o[m]) for m in st}

                        if (beta or gamma) and lo <= i < hi:
                            sender, receiver = (BB, LT) if beta else (LT, BB)
                            if pmask is None:
                                with torch.no_grad():
                                    dec0 = ae.decode(o[LT]["x_1"], o[BB]["x_1"], mask)
                                pk = pocket_mask(dec0["coors_nm"][0],
                                                 dec0["atom_mask"][0], live, lig)
                                far = (~pk) & live
                                k = int(pk.sum())
                                fidx = far.nonzero(as_tuple=False).flatten()[:k]
                                fm_ = torch.zeros_like(pk); fm_[fidx] = True
                                pmask = fm_ if far_ctrl else pk
                            bl = dict(bb)
                            back = hist[sender][max(0, len(hist[sender]) - 1 - a.lag_steps)]
                            bl["x_t"] = {m: (back if m == sender else st[m]) for m in st}
                            with torch.no_grad():
                                ol = model.predict_for_sampling(bl, mode="full", n_recycle=0)
                            ol = fm.nn_out_add_clean_sample_prediction(batch=bl, nn_out=ol)
                            diff = ol[receiver]["x_1"] - o[receiver]["x_1"]
                            lag_mag.append(float(diff.norm() / o[receiver]["x_1"].norm().clamp(min=1e-9)))
                            if perm_ctrl:
                                diff = diff[:, perm, :]
                            sel = pmask[None, :, None].to(diff.dtype)
                            x1 = o[receiver]["x_1"] + a.strength * (beta - gamma) * diff * sel
                            tt = float(ts[receiver][i])
                            if 1.0 - tt > EPS_T:
                                v = (x1 - st[receiver]) / (1.0 - tt)
                                o[receiver]["v"] = v
                                o[receiver]["x_1"] = x1
                                n_mod += 1
                        st = {m: fm.base_flow_matchers[m].simulation_step(
                                  x_t=st[m], nn_out={**o[m], "v_guided": o[m]["v"]},
                                  t=torch.full((bs,), float(ts[m][i]), device=dev),
                                  dt=ts[m][i+1]-ts[m][i], gt=gts[m][i], mask=mask,
                                  simulation_step_params=SP[m]) for m in st}
                        for m in st:
                            hist[m].append(st[m].clone())
                            if len(hist[m]) > a.lag_steps + 2:
                                hist[m].pop(0)
                    with torch.no_grad():
                        dec = ae.decode(st[LT], st[BB], mask)
                    s0 = {"x_motif": b0["x_motif"].cpu(), "motif_mask": b0["motif_mask"].cpu(),
                          "seq_motif": b0["seq_motif"].cpu(), "x_target": b0["x_target"].cpu(),
                          "target_mask": b0["target_mask"].cpu(), "mask": mask.cpu()}
                    reqs = Q.build_requirements(spec, s0["x_motif"], s0["motif_mask"],
                                                s0["seq_motif"], lig.cpu())
                    ev = Q.assign_and_score(reqs, dec["coors_nm"].cpu(),
                                            dec["atom_mask"].cpu(),
                                            dec["residue_type"].cpu(), s0["mask"])
                    sm = Q.scaffold_and_ligand_metrics(dec["coors_nm"].cpu(),
                                                       dec["atom_mask"].cpu(), s0["mask"],
                                                       st[BB].cpu(), lig.cpu(),
                                                       {r["gen_res_index"] for r in ev})
                    ch = MV.chain_geometry(dec["coors_nm"][0].cpu(),
                                           dec["atom_mask"][0].cpu().bool(), live.cpu())
                    ok = np.array([r["placed"] for r in ev])
                    fin = dec["coors_nm"][0].cpu()
                    amv = dec["atom_mask"][0].cpu().bool() & live.cpu()[:, None]
                    if arm == "natural":
                        natural_final[(task, seed)] = fin
                    ref = natural_final.get((task, seed))

                    rmsd_nat = (float((fin[amv] - ref[amv]).pow(2).sum(-1).mean().sqrt())
                                if ref is not None else float("nan"))
                    rows.append({"task": task, "seed": seed, "arm": arm,
                                 "placed": int(ok.sum()), "n_req": len(ev),
                                 "all_placed": bool(ok.all()),
                                 "clashes": sm["clashes_total"],
                                 "clash_depth_nm": sm["clash_depth_sum_nm"],
                                 "contacts": sm["ligand_contact_atoms"],
                                 "lig_min_nm": sm["ligand_min_dist_nm"],
                                 "ca_valid": ch["ca_ca_valid_frac"],
                                 "self_clash": ch["ca_self_clash_frac"],
                                 "rg_nm": MV.chain_geometry(dec["coors_nm"][0].cpu(),
                                     dec["atom_mask"][0].cpu().bool(), live.cpu())["ca_ca_mean_nm"],
                                 "steps_modulated": n_mod, "lag_steps": a.lag_steps,
                                 "strength": a.strength,
                                 "lag_rel_mean": float(np.mean(lag_mag)) if lag_mag else 0.0,
                                 "n_modulated_res": int(pmask.sum()) if pmask is not None else 0,
                                 "rmsd_vs_natural_nm": rmsd_nat})
                    print(f"{task} s{seed} {arm:14s} placed={rows[-1]['placed']}/{rows[-1]['n_req']} "
                          f"clash={rows[-1]['clashes']} caok={rows[-1]['ca_valid']:.3f} "
                          f"lag={rows[-1]['lag_rel_mean']:.4f} "
                          f"dNat={rmsd_nat:.4f} {(time.time()-t_all)/60:.1f}min", flush=True)
                except Exception as exc:
                    import traceback
                    errs.append({"task": task, "seed": seed, "arm": arm,
                                 "error": traceback.format_exc()[-400:]})
    pd.DataFrame(rows).to_parquet(out / "modulation.parquet", index=False)
    (out / "modulation_errors.json").write_text(json.dumps(errs, indent=2, default=str))
    print(json.dumps({"rows": len(rows), "errors": len(errs),
                      "minutes": round((time.time()-t_all)/60, 1)}, indent=2))

if __name__ == "__main__":
    main()
