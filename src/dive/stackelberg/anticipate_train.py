
from __future__ import annotations

from contextlib import contextmanager

import torch

from dive.stackelberg.joint_update import (
    DEFAULT_CLIP, PRIOR_SCALE_NM, anticipated_clean_state,
    anticipated_partner_state, blend)

MODALITIES = ("bb_ca", "local_latents")

N_STEPS = 400

def sampler_dt(batch: dict, n_steps: int = N_STEPS) -> float:

    tdict = batch.get("t") or {}
    tvals = [tdict[m] for m in MODALITIES if m in tdict]
    if not tvals:
        return 0.0
    tmean = sum(t.float().mean() for t in tvals) / len(tvals)
    remain = (1.0 - tmean).clamp(min=0.0)
    steps = max(int(n_steps), 1)
    return float(remain / steps)

def l2_to_init(module, init_cpu: dict):

    total = None
    n = 0
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or name not in init_cpu:
            continue
        ref = init_cpu[name].to(device=parameter.device, dtype=parameter.dtype)
        term = (parameter - ref).pow(2).sum()
        total = term if total is None else total + term
        n += parameter.numel()
    if total is None or n == 0:
        raise ValueError("l2_to_init: no overlapping trainable parameters")
    return total / n

def designed_rows_for_train(batch: dict):

    if "generated_mask" in batch:
        return batch["generated_mask"].bool()
    return None

def _alpha_weights(alpha, alpha_latent):
    if alpha_latent is not None:
        return {"bb_ca": float(alpha), "local_latents": float(alpha_latent)}
    if isinstance(alpha, dict):
        missing = [m for m in MODALITIES if m not in alpha]
        if missing:
            raise ValueError(f"alpha mapping missing {missing}")
        return {m: float(alpha[m]) for m in MODALITIES}
    a = float(alpha)
    return {"bb_ca": a, "local_latents": a}

@contextmanager
def train_under_anticipation(model, *, alpha=1.0, alpha_latent=None,
                             clip=DEFAULT_CLIP, router=None, partner_only=False,
                             n_steps=N_STEPS, clean_hat=False):

    weights = _alpha_weights(alpha, alpha_latent)
    original = model.call_nn

    def call(batch, *args, **kwargs):
        output = original(batch, *args, **kwargs)
        if max(weights.values()) == 0.0:
            return output
        if "x_t" not in batch or "t" not in batch:
            return output
        dt = sampler_dt(batch, n_steps=n_steps)
        if dt <= 0.0:
            return output

        advanced = {m: anticipated_partner_state(batch["x_t"][m], output[m]["v"], dt)
                    for m in MODALITIES
                    if m in batch["x_t"] and m in output and weights.get(m, 0.0) != 0.0}
        keep = designed_rows_for_train(batch)
        if keep is not None:
            xt0 = batch["x_t"]["bb_ca"]
            if keep.shape[:2] != xt0.shape[:2]:
                raise ValueError(
                    f"generated_mask has shape {tuple(keep.shape)}; expected "
                    f"[b, n]={tuple(xt0.shape[:2])}")
            for m in list(advanced):
                k = keep
                while k.dim() < advanced[m].dim():
                    k = k[..., None]
                advanced[m] = torch.where(k, advanced[m], batch["x_t"][m])

        def probe_once(hold=()):
            probe = dict(batch)
            probe["x_t"] = {**batch["x_t"],
                            **{m: s for m, s in advanced.items() if m not in hold}}
            if clean_hat:
                hats = {}
                for m in MODALITIES:
                    if m not in output or "v" not in output[m]:
                        continue
                    if m not in batch["x_t"] or m not in batch.get("t", {}):
                        continue
                    xt = batch["x_t"][m]
                    hats[m] = anticipated_clean_state(
                        xt, output[m]["v"], batch["t"][m].to(xt.dtype),
                        max_step_nm=PRIOR_SCALE_NM)
                if hats:
                    probe["x_sc"] = hats
            return original(probe, *args, **kwargs)

        targets = [m for m in MODALITIES if m in output and m in advanced]
        responded_v = {}
        if partner_only:
            for m in targets:
                out_m = probe_once(hold=(m,))
                if m in out_m:
                    responded_v[m] = out_m[m]["v"]
        else:
            shared = probe_once()
            for m in targets:
                if m in shared:
                    responded_v[m] = shared[m]["v"]

        model._last_jacobi_v = {m: output[m]["v"].detach()
                                for m in MODALITIES if m in output and "v" in output[m]}
        model._last_extra_v = {m: responded_v[m].detach() for m in responded_v}
        model._last_xt = {m: batch["x_t"][m].detach()
                          for m in batch["x_t"] if torch.is_tensor(batch["x_t"][m])}
        model._last_t = batch.get("t")

        routed = None
        if router is not None:
            routed = router(batch, output)
            missing = [m for m in MODALITIES if m not in routed]
            if missing:
                raise ValueError(f"router returned no weight for {missing}")

        revised = {m: dict(v) for m, v in output.items()}
        for m in MODALITIES:
            if m not in revised or m not in responded_v:
                continue
            if weights.get(m, 0.0) == 0.0:
                continue
            base = output[m]["v"]
            if routed is None:
                new = blend(base, responded_v[m], weights[m], clip=clip)
            else:
                w = routed[m].to(dtype=base.dtype)
                while w.dim() < base.dim():
                    w = w[..., None]
                unit = blend(base, responded_v[m], 1.0, clip=clip)
                new = base + w * (unit - base)
            if keep is not None:
                k = keep
                while k.dim() < new.dim():
                    k = k[..., None]
                new = torch.where(k, new, base)
            revised[m]["v"] = new
            for key in ("v_guided", "score", "score_guided", "guided_v", "guided_score"):
                revised[m].pop(key, None)
        return revised

    model.call_nn = call
    try:
        yield
    finally:
        model.call_nn = original
