
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, pandas as pd, torch, hydra
os.chdir(R.UPSTREAM)

BB, LT = "bb_ca", "local_latents"
D_POLAR, D_APOLAR, CONTACT_MAX, MATCH_NM = 0.28, 0.34, 0.45, 0.15
ATOM37_ELEM = ["N", "C", "C", "O"] + ["C"] * 33

def elements_for(am):
    return [ATOM37_ELEM[i] if i < len(ATOM37_ELEM) else "C"
            for i in am.nonzero(as_tuple=False)[:, 1].tolist()]

class Probe:
    def __init__(self, model, b0, mask, motif_pts, lig_pts, dev):
        self.m, self.b0, self.mask, self.dev = model, b0, mask, dev
        self.motif, self.lig = motif_pts, lig_pts
        self.fm = model.fm

    def forward(self, st, ts, i):
        b = dict(self.b0)
        b["x_t"] = {k: st[k] for k in st}
        b["t"] = {k: torch.full((self.mask.shape[0],), float(ts[k][i]), device=self.dev)
                  for k in st}
        b["mask"] = self.mask
        with torch.no_grad():
            o = self.m.predict_for_sampling(b, mode="full", n_recycle=0)
        o = self.fm.nn_out_add_clean_sample_prediction(batch=b, nn_out=o)
        o = self.fm.nn_out_add_simulation_tensor(batch=b, nn_out=o)
        return {k: {kk: vv for kk, vv in o[k].items()} for k in st}, b

    def clean(self, out):
        return {k: out[k]["x_1"] for k in (BB, LT)}

    def task_error(self, z_clean, ca_clean):

        with torch.no_grad():
            d = self.m.autoencoder.decode(z_clean, ca_clean, self.mask)
            am = d["atom_mask"][0].bool() & self.mask[0][:, None]
            pts = d["coors_nm"][0][am]
            el = elements_for(am)
            pol = torch.tensor([e in ("N", "O") for e in el], device=self.dev)
            dmin = torch.where(pol[:, None], torch.tensor(D_POLAR, device=self.dev),
                               torch.tensor(D_APOLAR, device=self.dev)).expand(-1, self.lig.shape[0])
            dist = torch.cdist(pts[None], self.lig[None])[0]
            overlap = float(torch.relu(dmin - dist).pow(2).sum())
            uncovered = float(torch.relu(dist.min(dim=0).values - CONTACT_MAX).pow(2).sum())
            dm = torch.cdist(self.motif[None], pts[None])[0]
            motif_err = float(dm.min(dim=1).values.pow(2).sum())
            clash = float((dist.min(dim=1).values < dmin.min(dim=1).values).sum())
            contacts = float(((dist >= dmin) & (dist < CONTACT_MAX)).any(dim=1).sum())
            placed = float((dm.min(dim=1).values < MATCH_NM).float().mean())
        return {"E": overlap + uncovered + motif_err, "overlap": overlap,
                "uncovered": uncovered, "motif_err": motif_err,
                "clash": clash, "contacts": contacts, "placed": placed}

    def sender_direction(self, st, out, mode, eps):

        cl = self.clean(out)
        z = cl[LT].detach().clone().requires_grad_(mode == LT)
        ca = cl[BB].detach().clone().requires_grad_(mode == BB)
        d = self.m.autoencoder.decode(z, ca, self.mask)
        am = d["atom_mask"][0].bool() & self.mask[0][:, None]
        pts = d["coors_nm"][0][am]
        el = elements_for(am)
        pol = torch.tensor([e in ("N", "O") for e in el], device=self.dev)
        dmin = torch.where(pol[:, None], torch.tensor(D_POLAR, device=self.dev),
                           torch.tensor(D_APOLAR, device=self.dev)).expand(-1, self.lig.shape[0])
        dist = torch.cdist(pts[None], self.lig[None])[0]
        loss = (torch.relu(dmin - dist).pow(2).sum()
                + torch.relu(dist.min(dim=0).values - CONTACT_MAX).pow(2).sum()
                + torch.cdist(self.motif[None], pts[None])[0].min(dim=1).values.pow(2).sum())
        leaf = z if mode == LT else ca
        g, = torch.autograd.grad(loss, [leaf], allow_unused=True)
        if g is None or float(g.abs().sum()) == 0:
            return None
        scale = cl[mode].detach().pow(2).mean().sqrt()
        gn = g.pow(2).mean().sqrt().clamp(min=1e-12)
        return (-(eps * scale / gn) * g).detach()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--probe-fracs", default="0.55,0.75,0.90")
    ap.add_argument("--eps", default="0.01,0.02")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    model, linfo = R.load(args.device)
    fm = model.fm
    fracs = [float(v) for v in args.probe_fracs.split(",")]
    epss = [float(v) for v in args.eps.split(",")]
    rows, sanity, meta = [], {}, {"load": linfo, "nsteps": args.nsteps,
                                 "probe_fracs": fracs, "eps": epss,
                                 "excluded": 0, "attempted": 0, "failures": []}
    t0 = time.time()

    for task in args.tasks.split(","):
        gen, cmeta = R.config(task, nsteps=args.nsteps, nsamples=1)
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
        ts, gts = fm.sample_schedule(args.nsteps, SA)
        SP = {m: dict(SA[m]["simulation_step_params"]) for m in SA}
        probe_steps = {int(round(f * args.nsteps)) for f in fracs}

        for seed in range(args.seeds):
            torch.manual_seed(20260903 + seed); torch.cuda.manual_seed(20260903 + seed)
            st = fm.sample_noise(int(mask.shape[1]), shape=(bs,), device=dev, mask=mask)
            st = {m: st[m] for m in (BB, LT)}
            for i in range(args.nsteps):
                out0, b = P.forward(st, ts, i)
                if i in probe_steps:
                    cl0 = P.clean(out0)
                    E0 = P.task_error(cl0[LT], cl0[BB])
                    for sender, receiver, dname in ((LT, BB, "z_to_x"), (BB, LT, "x_to_z")):
                        for eps in epss:
                            base_d = P.sender_direction(st, out0, sender, eps)
                            if base_d is None:
                                meta["excluded"] += 1; continue
                            for kind in ("task", "random", "opposite", "zero"):
                                meta["attempted"] += 1
                                if kind == "zero":
                                    dclean = torch.zeros_like(base_d)
                                elif kind == "random":
                                    g = torch.randn_like(base_d)
                                    dclean = g * (base_d.norm() / g.norm().clamp(min=1e-12))
                                elif kind == "opposite":
                                    dclean = -base_d
                                else:
                                    dclean = base_d

                                st1 = {m: st[m].clone() for m in st}
                                st1[sender] = st[sender] + float(ts[sender][i]) * dclean
                                out1, _ = P.forward(st1, ts, i)
                                cl1 = P.clean(out1)

                                same_sender = True
                                allowed = {LT: cl1[LT], BB: cl1[BB]}
                                clamped = dict(allowed); clamped[receiver] = cl0[receiver]
                                Ea = P.task_error(allowed[LT], allowed[BB])
                                Ec = P.task_error(clamped[LT], clamped[BB])
                                drec = (cl1[receiver] - cl0[receiver])
                                rows.append({
                                    "task": task, "seed": seed, "step": i,
                                    "frac": i / args.nsteps, "direction": dname,
                                    "eps": eps, "kind": kind,
                                    "t_sender": float(ts[sender][i]),
                                    "t_receiver": float(ts[receiver][i]),
                                    "influence_rel": float(drec.norm() / cl0[receiver].norm()),
                                    "influence_abs": float(drec.norm()),
                                    "sender_delta_rel": float((cl1[sender]-cl0[sender]).norm()
                                                              / cl0[sender].norm()),
                                    "E_base": E0["E"], "E_allowed": Ea["E"], "E_clamped": Ec["E"],
                                    "U": Ec["E"] - Ea["E"],
                                    "sender_effect": E0["E"] - Ec["E"],
                                    "clash_allowed": Ea["clash"], "clash_clamped": Ec["clash"],
                                    "contacts_allowed": Ea["contacts"], "contacts_clamped": Ec["contacts"],
                                    "placed_allowed": Ea["placed"], "placed_clamped": Ec["placed"],
                                    "motif_allowed": Ea["motif_err"], "motif_clamped": Ec["motif_err"],
                                    "same_sender_output": same_sender})
                                if kind == "zero" and "zero_check" not in sanity:
                                    sanity["zero_check"] = {
                                        "U": Ec["E"] - Ea["E"],
                                        "influence_rel": float(drec.norm() / cl0[receiver].norm())}
                for m in st:
                    out0[m] = dict(out0[m]); out0[m]["v_guided"] = out0[m]["v"]
                st = {m: fm.base_flow_matchers[m].simulation_step(
                    x_t=st[m], nn_out=out0[m],
                    t=torch.full((bs,), float(ts[m][i]), device=dev),
                    dt=ts[m][i+1]-ts[m][i], gt=gts[m][i], mask=mask,
                    simulation_step_params=SP[m]) for m in st}
            print(f"{task} seed {seed} done rows={len(rows)} {(time.time()-t0)/60:.1f}min",
                  flush=True)

    meta["sanity"] = sanity
    meta["runtime_minutes"] = (time.time()-t0)/60.0
    pd.DataFrame(rows).to_parquet(out / "response.parquet", index=False)
    (out / "response_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({"rows": len(rows), "sanity": sanity,
                      "excluded": meta["excluded"]}, indent=2, default=str))

if __name__ == "__main__":
    main()
