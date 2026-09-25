
from __future__ import annotations

import torch

from dive.anticipate.capture import designed_from_batch
from dive.stackelberg.extra_pass_gate import native_velocity
from dive.stackelberg.joint_update import (
    intervention_rows, partner_tensors, snap_anticipation_probe)

def snap_anticipation_loss(model, batch, jacobi_v, weights, *,
                           dt: float, contact_nm: float = 0.5,
                           teacher=None) -> torch.Tensor:
    if jacobi_v is None or "bb_ca" not in jacobi_v or "local_latents" not in jacobi_v:
        raise ValueError("snap anticipation needs both Jacobi velocities")
    if teacher is None and "x_1" not in batch:
        raise ValueError("snap anticipation needs native x_1")
    partner, keep = partner_tensors(batch)
    rows = intervention_rows(designed_from_batch(batch), batch.get("mask"))
    detached = {m: jacobi_v[m].detach() for m in ("bb_ca", "local_latents")}
    probe_x, follower = snap_anticipation_probe(
        batch["x_t"], detached, batch["t"], weights, partner, keep,
        dt=dt, contact_nm=contact_nm, valid=batch.get("mask"),
        row_mask=rows)
    probe = dict(batch)
    probe["x_t"] = dict(batch["x_t"])
    probe["x_t"].update(probe_x)
    taught = None
    if teacher is not None:
        held = dict(probe)
        held["x_t"] = {k: (v.detach() if torch.is_tensor(v) else v)
                       for k, v in probe["x_t"].items()}
        with torch.no_grad():
            taught = teacher.call_nn(held)
    out = model.call_nn(probe)
    total = None
    for key in ("bb_ca", "local_latents"):
        if key not in out or "v" not in out[key]:
            raise ValueError(f"snap anticipation probe returned no {key} velocity")
        if teacher is None:
            if key not in batch["x_1"]:
                raise ValueError(f"snap anticipation needs native {key}")
            target = native_velocity(
                batch["x_t"][key], batch["x_1"][key], batch["t"][key])
        else:
            if taught is None or key not in taught or "v" not in taught[key]:
                raise ValueError(
                    f"frozen reaction teacher returned no {key} velocity")
            target = taught[key]["v"].detach()
        err = (out[key]["v"] - target).pow(2).sum(dim=-1)
        weight = follower[key].to(dtype=err.dtype)
        term = (err * weight).sum() / weight.sum().clamp_min(1.0)
        total = term if total is None else total + term
    return total
