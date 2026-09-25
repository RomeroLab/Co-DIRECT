
from __future__ import annotations

import argparse, json, sys, warnings, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, yaml, hydra, os
import ame_requirements as Q
from ame_roundtrip import ae_batch
os.chdir(R.UPSTREAM)

BB, LT = "bb_ca", "local_latents"

def clean_state(ae, s, coors, dev):

    ca = coors[:, :, 1, :].contiguous()
    b = ae_batch(coors, s["atom_mask"].to(dev), s["residue_type"].to(dev),
                 s["mask"].to(dev), ca, dev)
    with torch.no_grad():
        enc = ae.encode(dict(b))
    return {BB: ca, LT: enc["z_latent"]}

def evaluate(spec, s, coors, atom_mask, residue_type, bb_ca):
    lig = s["x_target"][0][s["target_mask"][0].bool()].reshape(-1, 3)
    reqs = Q.build_requirements(spec, s["x_motif"], s["motif_mask"], s["seq_motif"], lig)
    rows = Q.assign_and_score(reqs, coors, atom_mask, residue_type, s["mask"])
    sm = Q.scaffold_and_ligand_metrics(coors, atom_mask, s["mask"], bb_ca, lig,
                                       {r["gen_res_index"] for r in rows})
    ok = np.array([r["placed"] for r in rows])
    return {"placed": int(ok.sum()), "n_req": len(rows), "all_placed": bool(ok.all()),
            "clashes": sm["clashes_total"], "clash_depth_nm": sm["clash_depth_sum_nm"],
            "contacts": sm["ligand_contact_atoms"],
            "lig_min_nm": sm["ligand_min_dist_nm"]}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-root", required=True)
    ap.add_argument("--poses", nargs="+", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", default="0.55,0.75,0.90")
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model, linfo = R.load(a.device); ae = model.autoencoder; fm = model.fm
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    want = set(a.tasks.split(",")) if a.tasks else None
    fracs = [float(v) for v in a.fracs.split(",")]
    rows, errs = [], []

    jobs = {}
    for pt in sorted(q for d in a.poses for q in Path(d).glob("*__flexible.pt")):
        task, stem, _ = pt.stem.split("__")
        if want and task not in want:
            continue
        c = Path(a.controls) / f"{task}__{stem}__randommatched.pt"
        n = Path(a.native_root) / task / f"{stem}.pt"
        fx = pt.parent / f"{task}__{stem}__fixed.pt"
        if c.exists() and n.exists():
            j = {"improved": pt, "random_matched": c, "native": n}
            if fx.exists():
                j["fixed"] = fx
            jobs[(task, stem)] = j

    cache = {}
    for (task, stem), paths in jobs.items():
        try:
            spec = td["motif_target_dict_cfg"][task]["contig_atoms"]
            if task not in cache:
                gen, _ = R.config(task, nsteps=a.nsteps, nsamples=1)
                model.configure_inference(gen, nn_ag=None)
                b0 = next(iter(hydra.utils.instantiate(gen.dataloader)))
                b0 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b0.items()}
                SA = {m: gen.model[m] for m in (BB, LT)}
                ts, gts = fm.sample_schedule(a.nsteps, SA)
                SP = {m: dict(SA[m]["simulation_step_params"]) for m in SA}
                cache[task] = (b0, ts, gts, SP)
            b0, ts, gts, SP = cache[task]
            mask = b0["mask"]; bs = int(mask.shape[0])

            states = {}
            for name, p in paths.items():
                s = torch.load(p, map_location="cpu", weights_only=False)
                states[name] = (s, clean_state(ae, s, s["coors_nm"].to(dev), dev))
            s_nat = states["native"][0]

            for frac in fracs:
                i0 = int(round(frac * a.nsteps))
                torch.manual_seed(4242); torch.cuda.manual_seed(4242)
                x0 = fm.sample_noise(int(mask.shape[1]), shape=(bs,), device=dev, mask=mask)
                x0 = {m: x0[m] for m in (BB, LT)}
                for name, (s, x1) in states.items():
                    t = {m: torch.full((bs,), float(ts[m][i0]), device=dev) for m in (BB, LT)}
                    st = fm.interpolate(x0, x1, t, mask)
                    st = {m: st[m] for m in (BB, LT)}
                    torch.manual_seed(999); torch.cuda.manual_seed(999)
                    for i in range(i0, a.nsteps):
                        bb = dict(b0)
                        bb["x_t"] = {m: st[m] for m in st}
                        bb["t"] = {m: torch.full((bs,), float(ts[m][i]), device=dev)
                                   for m in st}
                        bb["mask"] = mask
                        with torch.no_grad():
                            o = model.predict_for_sampling(bb, mode="full", n_recycle=0)
                        o = fm.nn_out_add_clean_sample_prediction(batch=bb, nn_out=o)
                        o = fm.nn_out_add_simulation_tensor(batch=bb, nn_out=o)
                        st = {m: fm.base_flow_matchers[m].simulation_step(
                                  x_t=st[m], nn_out={**o[m], "v_guided": o[m]["v"]},
                                  t=torch.full((bs,), float(ts[m][i]), device=dev),
                                  dt=ts[m][i+1]-ts[m][i], gt=gts[m][i], mask=mask,
                                  simulation_step_params=SP[m]) for m in st}
                    with torch.no_grad():
                        dec = ae.decode(st[LT], st[BB], mask.bool())
                    ev = evaluate(spec, s_nat, dec["coors_nm"].cpu(),
                                  dec["atom_mask"].cpu(), dec["residue_type"].cpu(),
                                  st[BB].cpu())
                    am = (s_nat["atom_mask"][0].bool() & s_nat["mask"][0].bool()[:, None])
                    fin = dec["coors_nm"][0].cpu()
                    d_start = float((fin[am] - s["coors_nm"][0][am]).pow(2).sum(-1).mean().sqrt())
                    d_nat = float((fin[am] - s_nat["coors_nm"][0][am]).pow(2).sum(-1).mean().sqrt())
                    start_ev = evaluate(spec, s_nat, s["coors_nm"], s_nat["atom_mask"],
                                        s_nat["residue_type"], s["coors_nm"][:, :, 1, :])
                    rows.append({"task": task, "stem": stem, "start": name, "frac": frac,
                                 "start_placed": start_ev["placed"],
                                 "start_clashes": start_ev["clashes"],
                                 "n_req": ev["n_req"],
                                 **{f"end_{k}": v for k, v in ev.items()},
                                 "rmsd_to_start_nm": d_start, "rmsd_to_native_nm": d_nat})
                    print(f"{task} {stem} f={frac} {name:14s} "
                          f"start={start_ev['placed']}/{ev['n_req']} clash={start_ev['clashes']} "
                          f"-> end={ev['placed']} clash={ev['clashes']} "
                          f"d_start={d_start:.3f} d_nat={d_nat:.3f}", flush=True)
        except Exception as exc:
            import traceback
            errs.append({"task": task, "stem": stem,
                         "error": traceback.format_exc()[-500:]})

    pd.DataFrame(rows).to_parquet(out / "dynamics.parquet", index=False)
    (out / "dynamics_errors.json").write_text(json.dumps(errs, indent=2, default=str))
    print(json.dumps({"rows": len(rows), "errors": len(errs)}, indent=2))

if __name__ == "__main__":
    main()
