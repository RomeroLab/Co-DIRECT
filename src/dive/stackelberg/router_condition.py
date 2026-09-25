
from __future__ import annotations

from contextlib import contextmanager

import torch

from dive.stackelberg.joint_update import PRIOR_SCALE_NM

MODALITIES = ("bb_ca", "local_latents")

def router_gated_x_sc(x_hat: dict, weights: dict) -> dict:

    missing = [m for m in MODALITIES if m not in weights]
    if missing:
        raise ValueError(
            f"router weights missing {missing}; a missing modality would leave "
            "that channel unconditioned while the label still claims a wrap")
    missing_hat = [m for m in MODALITIES if m not in x_hat]
    if missing_hat:
        raise ValueError(
            f"hat missing {missing_hat}; zeros would fake a leadership map")
    out = {}
    for m in MODALITIES:
        hat = x_hat[m]
        w = weights[m].to(device=hat.device, dtype=hat.dtype)
        while w.dim() < hat.dim():
            w = w[..., None]
        if w.shape[:2] != hat.shape[:2]:
            raise ValueError(
                f"weight for {m} has shape {tuple(weights[m].shape)}; expected "
                f"[b, n]={tuple(hat.shape[:2])}. Broadcasting it would condition "
                "one residue with another's leadership")
        out[m] = w * hat
    return out

def add_persist_residual(x_sc: dict, persist_clean: dict, x_t: dict,
                         *, max_step_nm=PRIOR_SCALE_NM) -> dict:

    missing = [m for m in MODALITIES if m not in persist_clean or m not in x_t]
    if missing:
        raise ValueError(
            f"persist residual missing {missing}; dropping a channel would "
            "condition one modality and leave the other as router-only")
    out = {}
    for m in MODALITIES:
        gated = x_sc[m]
        residual = persist_clean[m].to(device=gated.device, dtype=gated.dtype) - (
            x_t[m].to(device=gated.device, dtype=gated.dtype))
        if max_step_nm is not None:
            norm = residual.norm(dim=-1, keepdim=True)
            residual = residual * (float(max_step_nm) / norm.clamp_min(1e-12)).clamp(max=1.0)
        out[m] = gated + residual
    return out

def _hat_from_batch(batch: dict, *, missing: str = "x_t") -> dict:

    sc = batch.get("x_sc")
    if isinstance(sc, dict) and all(m in sc for m in MODALITIES):
        return sc
    xt = batch.get("x_t")
    if not isinstance(xt, dict) or any(m not in xt for m in MODALITIES):
        raise ValueError(
            "need x_t with bb_ca and local_latents to build the router "
            "condition")
    if missing == "zeros":
        return {m: torch.zeros_like(xt[m]) for m in MODALITIES}
    if missing != "x_t":
        raise ValueError(f"unknown hat fallback {missing!r}; use x_t or zeros")
    return xt

def _trunk_cond(net, batch):
    if not getattr(net, "use_trunk_condition", False):
        return None
    from dive.stackelberg.trunk_condition import trunk_condition_features
    return trunk_condition_features(batch)

def live_or_native_router(net):

    def router(batch, output):
        from dive.stackelberg.learned_router import (
            native_contact_labels, native_from_train_batch)
        native = native_from_train_batch(batch)
        xt = batch.get("x_t")
        if isinstance(xt, dict) and all(m in xt for m in MODALITIES):
            ca = xt["bb_ca"]
            lat = xt["local_latents"]
            partner = native["partner"].to(device=ca.device, dtype=ca.dtype)
            dist = torch.cdist(ca, partner).min(dim=-1).values
            if getattr(net, "use_residue_type", True):
                restype = native["residue_type"].to(device=ca.device)
            else:
                restype = torch.zeros(ca.shape[:2], dtype=torch.long, device=ca.device)
            des = native["designed"].to(device=ca.device)
            sc = None
            if getattr(net, "use_sc_dist", False):
                atom37 = native.get("atom37")
                if atom37 is None:
                    raise ValueError(
                        "use_sc_dist router needs atom37 on the train batch")
                sc = native_contact_labels(
                    ca, partner, atom37=atom37.to(device=ca.device, dtype=ca.dtype)
                )["sc_dist"]
            out = net.forward(dist, restype, lat, des, sc_dist=sc,
                              t=batch.get("t"), cond=_trunk_cond(net, batch))
            return {m: out[m] for m in MODALITIES}
        dist = torch.cdist(native["bb_ca"], native["partner"]).min(dim=-1).values
        sc = None
        if getattr(net, "use_sc_dist", False):
            labels = native_contact_labels(
                native["bb_ca"], native["partner"], atom37=native.get("atom37"))
            sc = labels["sc_dist"]
        if getattr(net, "use_residue_type", True):
            restype = native["residue_type"]
        else:
            restype = torch.zeros(
                native["bb_ca"].shape[:2], dtype=torch.long,
                device=native["bb_ca"].device)
        out = net.forward(
            dist, restype, native["local_latents"],
            native["designed"], sc_dist=sc, t=batch.get("t"),
            cond=_trunk_cond(net, batch))
        return {m: out[m] for m in MODALITIES}
    return router

@contextmanager
def trunk_reads_router_condition(model, router, *, missing: str = "x_t"):

    original = model.call_nn

    def call(batch, *args, **kwargs):
        batch = dict(batch)
        xt = batch.get("x_t")
        dummy = {}
        if isinstance(xt, dict):
            dummy = {m: {"v": xt[m]} for m in xt}
        weights = router(batch, dummy)
        gated = router_gated_x_sc(_hat_from_batch(batch, missing=missing), weights)
        persist = batch.get("_persist_clean")
        if isinstance(persist, dict) and isinstance(xt, dict):
            gated = add_persist_residual(gated, persist, xt)
        batch["x_sc"] = gated
        return original(batch, *args, **kwargs)

    model.call_nn = call
    had = True
    try:
        yield
    finally:
        if had:
            model.call_nn = original
