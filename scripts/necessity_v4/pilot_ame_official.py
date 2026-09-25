
from __future__ import annotations
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_rt
_ame_rt.bootstrap()
import torch, hydra

os.chdir(_ame_rt.UPSTREAM)
TASK = sys.argv[1]; NSTEPS = int(sys.argv[2]); NSAMP = int(sys.argv[3])
gen = _ame_rt.ame_config(TASK, nsteps=NSTEPS, nsamples=NSAMP)
print("task", TASK, "nsteps", NSTEPS, "nsamples", NSAMP, flush=True)

model = _ame_rt.load_ame_model("cuda:0")
print("arch:", type(model.nn).__name__, flush=True)
cf = model.cfg_exp.nn.get("concat_features")
print("concat_features:", dict(cf) if cf is not None else None, flush=True)
model.configure_inference(gen, nn_ag=None)

loader = hydra.utils.instantiate(gen.dataloader)
batch = next(iter(loader))
batch = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in batch.items()}
print("BATCH", flush=True)
for k in sorted(batch):
    v = batch[k]
    print("  %-28s %s" % (k, tuple(v.shape) if torch.is_tensor(v) else type(v).__name__), flush=True)

torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(batch)
torch.cuda.synchronize()
print(f"GENERATED in {time.perf_counter()-t0:.1f}s peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
for k, v in out.items():
    print("  out", k, tuple(v.shape) if torch.is_tensor(v) else type(v),
          "finite", bool(torch.isfinite(v).all()) if torch.is_tensor(v) else "-")
