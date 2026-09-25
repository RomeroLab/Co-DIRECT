
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import torch, hydra
os.chdir(R.UPSTREAM)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--nsteps", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model, linfo = R.load(a.device)
    report = {"load": linfo, "nsteps": a.nsteps, "seeds": a.seeds, "tasks": {}}
    t0 = time.time()

    for task in a.tasks.split(","):
        rec = {"status": "pending"}
        try:
            gen, cmeta = R.config(task, nsteps=a.nsteps, nsamples=1)
            model.configure_inference(gen, nn_ag=None)
            b = next(iter(hydra.utils.instantiate(gen.dataloader)))
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            mask = b["mask"]
            rec.update(cmeta)
            rec["n_motif_res"] = int(b["motif_mask"].shape[1])
            rec["n_motif_atoms"] = int(b["motif_mask"].sum())
            rec["n_ligand_atoms"] = int(b["target_mask"].sum())
            rec["n_res"] = int(mask.sum())
        except Exception as exc:
            rec.update({"status": "batch_failed",
                        "error": f"{type(exc).__name__}: {exc}"[:400]})
            report["tasks"][task] = rec
            print(f"{task} BATCH FAILED {type(exc).__name__}", flush=True)
            continue

        td = out / task; td.mkdir(exist_ok=True)
        done, fails = 0, []
        for seed in range(a.seeds):
            torch.manual_seed(20260903 + seed); torch.cuda.manual_seed(20260903 + seed)
            try:
                with torch.no_grad():
                    s = model.generate(dict(b))
                    dec = model.autoencoder.decode(s["local_latents"], s["bb_ca"], mask)
                if not (torch.isfinite(s["bb_ca"]).all()
                        and torch.isfinite(dec["coors_nm"]).all()):
                    raise RuntimeError("non-finite output")
                torch.save({"bb_ca": s["bb_ca"].cpu(),
                            "local_latents": s["local_latents"].cpu(),
                            "coors_nm": dec["coors_nm"].cpu(),
                            "atom_mask": dec["atom_mask"].cpu(),
                            "residue_type": dec["residue_type"].cpu(),
                            "seq_logits": dec["seq_logits"].cpu(),
                            "mask": mask.cpu(),
                            "x_motif": b["x_motif"].cpu(),
                            "motif_mask": b["motif_mask"].cpu(),
                            "seq_motif": b["seq_motif"].cpu(),
                            "x_target": b["x_target"].cpu(),
                            "target_mask": b["target_mask"].cpu(),
                            "seq_target": b["seq_target"].cpu(),
                            "target_atom_name": b["target_atom_name"].cpu(),
                            "target_bond_mask": b["target_bond_mask"].cpu(),
                            "target_bond_order": b["target_bond_order"].cpu()},
                           td / f"native_seed{seed}.pt")
                done += 1
            except Exception as exc:
                fails.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"[:300]})
        rec.update({"status": "ok" if done else "all_seeds_failed",
                    "completed": done, "seed_failures": fails})
        report["tasks"][task] = rec
        print(f"{task} {rec['status']} {done}/{a.seeds} "
              f"motif={rec['n_motif_atoms']}atoms lig={rec['n_ligand_atoms']} "
              f"{(time.time()-t0)/60:.1f}min", flush=True)

    report["runtime_minutes"] = (time.time() - t0) / 60.0
    (out / f"native_report_{os.getpid()}.json").write_text(json.dumps(report, indent=2, default=str))
    ok = sum(1 for v in report["tasks"].values() if v["status"] == "ok")
    print(json.dumps({"ok": ok, "total": len(report["tasks"])}, indent=2))

if __name__ == "__main__":
    main()
