
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, hydra
os.chdir(R.UPSTREAM)
from ame_response import Probe, BB, LT

def advance(fm, st, outs, ts, gts, SP, i, mask, bs, dev):
    nx = {}
    for m in st:
        o = dict(outs[m]); o["v_guided"] = o["v"]
        nx[m] = fm.base_flow_matchers[m].simulation_step(
            x_t=st[m], nn_out=o, t=torch.full((bs,), float(ts[m][i]), device=dev),
            dt=ts[m][i+1]-ts[m][i], gt=gts[m][i], mask=mask,
            simulation_step_params=SP[m])
    return nx

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--probe-frac", type=float, default=0.75)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model, linfo = R.load(a.device); fm = model.fm
    pstep = int(round(a.probe_frac * a.nsteps))
    rows, meta = [], {"load": linfo, "probe_step": pstep, "eps": a.eps,
                      "nsteps": a.nsteps, "identity_checks": []}
    t0 = time.time()

    for task in a.tasks.split(","):
        gen, _ = R.config(task, nsteps=a.nsteps, nsamples=1)
        model.configure_inference(gen, nn_ag=None)
        b0 = next(iter(hydra.utils.instantiate(gen.dataloader)))
        b0 = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b0.items()}
        mask = b0["mask"]; bs = int(mask.shape[0])
        mm = b0["motif_mask"].bool()
        motif_pts = b0["x_motif"][0][mm[0]].reshape(-1, 3)
        lmask = b0["target_mask"].bool()
        lig_pts = (b0["x_target"][0][lmask[0]] if lmask.dim() == 2
                   else b0["x_target"][0]).reshape(-1, 3)
        P = Probe(model, b0, mask, motif_pts, lig_pts, dev)
        SA = {m: gen.model[m] for m in (BB, LT)}
        ts, gts = fm.sample_schedule(a.nsteps, SA)
        SP = {m: dict(SA[m]["simulation_step_params"]) for m in SA}

        for seed in range(a.seeds):
            sd = 20260903 + seed
            torch.manual_seed(sd); torch.cuda.manual_seed(sd)
            st = fm.sample_noise(int(mask.shape[1]), shape=(bs,), device=dev, mask=mask)
            st = {m: st[m] for m in (BB, LT)}
            for i in range(pstep):
                o, _ = P.forward(st, ts, i)
                st = advance(fm, st, o, ts, gts, SP, i, mask, bs, dev)

            out0, _ = P.forward(st, ts, pstep)
            cl0 = P.clean(out0)
            E0 = P.task_error(cl0[LT], cl0[BB])
            branches = {"native": (st, out0, None)}
            for sender, receiver, dn in ((BB, LT, "x_to_z"), (LT, BB, "z_to_x")):
                d = P.sender_direction(st, out0, sender, a.eps)
                if d is None:
                    meta.setdefault("excluded", []).append([task, seed, dn]); continue
                st1 = {m: st[m].clone() for m in st}
                st1[sender] = st[sender] + float(ts[sender][pstep]) * d
                out1, _ = P.forward(st1, ts, pstep)

                oc = {sender: out1[sender], receiver: out0[receiver]}
                meta["identity_checks"].append(
                    {"task": task, "seed": seed, "dir": dn, "sender_output_identical":
                     bool(torch.equal(out1[sender]["v"], oc[sender]["v"]))})
                branches[f"allowed_{dn}"] = (st1, out1, receiver)
                branches[f"clamped_{dn}"] = (st1, oc, receiver)

            for name, (s0, o0, _rec) in branches.items():
                torch.manual_seed(sd + 777); torch.cuda.manual_seed(sd + 777)
                s = advance(fm, s0, o0, ts, gts, SP, pstep, mask, bs, dev)
                for i in range(pstep + 1, a.nsteps):
                    oo, _ = P.forward(s, ts, i)
                    s = advance(fm, s, oo, ts, gts, SP, i, mask, bs, dev)
                fo, _ = P.forward(s, ts, a.nsteps - 1)
                fc = P.clean(fo)
                E = P.task_error(fc[LT], fc[BB])
                rows.append({"task": task, "seed": seed, "arm": name,
                             "E_probe_base": E0["E"], **{f"final_{k}": v for k, v in E.items()}})
                print(f"{task} s{seed} {name} E={E['E']:.4f} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)

    pd.DataFrame(rows).to_parquet(out / "rollout.parquet", index=False)
    (out / "rollout_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({"rows": len(rows)}, indent=2))

if __name__ == "__main__":
    main()
