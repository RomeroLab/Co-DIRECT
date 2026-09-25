
from __future__ import annotations

import torch

from dive.stackelberg.anticipate_train import sampler_dt
from dive.stackelberg.extra_pass_gate import native_velocity
from dive.stackelberg.interface_fm import native_interface_mask, partner_near_mask
from dive.stackelberg.joint_update import (
    DEFAULT_CLIP, PRIOR_SCALE_NM, anticipated_clean_state,
    anticipated_partner_state, residue_commitment, routed_extra_pass_velocity)
from dive.training.sequence_recovery import refuse_outcome_labels

MODALITIES = ("bb_ca", "local_latents")

def native_ward_state(x_t: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor,
                      dt: float) -> torch.Tensor:

    u = native_velocity(x_t, x_1, t)
    return x_t + float(dt) * u

def _designed(batch: dict, n: int, device) -> torch.Tensor:
    if "generated_mask" in batch:
        d = batch["generated_mask"].bool()
    elif "mask" in batch:
        d = batch["mask"].bool()
    else:
        d = torch.ones(1, n, dtype=torch.bool, device=device)
    return d.to(device=device)

def _schedule_t(tdict: dict) -> float:

    clock = tdict["bb_ca"] if "bb_ca" in tdict else next(iter(tdict.values()))
    return float(clock.detach().reshape(-1)[0].clamp(0, 1))

def anticipation_fm_loss(model, batch, *, n_steps: int = 400,
                         leadership: str = "xz", router=None,
                         followers: tuple[str, ...] = MODALITIES,
                         partner_advance: str = "native",
                         jacobi_v: dict | None = None,
                         advance_frac: float = 1.0,
                         objective: str = "velocity") -> torch.Tensor:

    refuse_outcome_labels(batch)
    banned = ("iptm", "ipae", "plddt", "ptm", "motif_rmsd")
    for key in batch:
        low = str(key).lower()
        if any(tok in low for tok in banned):
            raise ValueError(
                "predictor / ESMFold2 / MotifRMSD scores are not training labels")
    if "x_1" not in batch or not isinstance(batch["x_1"], dict):
        raise ValueError(
            "anticipation FM needs native x_1; there is no decoded or "
            "ESMFold2 structure in this loss")
    missing = [m for m in MODALITIES if m not in batch["x_t"] or m not in batch["x_1"]]
    if missing:
        raise ValueError(f"anticipation FM missing modalities {missing}")
    dt = sampler_dt(batch, n_steps=n_steps)
    if dt <= 0.0:
        return batch["x_t"]["bb_ca"].new_zeros(())
    if partner_advance not in ("native", "jacobi", "commit", "residue"):
        raise ValueError(
            f"unknown partner_advance {partner_advance!r}; expected "
            "native, jacobi, commit, or residue")
    if objective not in ("velocity", "applied"):
        raise ValueError(
            f"unknown anticipation objective {objective!r}; expected "
            "velocity (raw extra-pass vs native u) or applied "
            "(generate blend vs native u)")
    if objective == "applied" and partner_advance != "jacobi":
        raise ValueError(
            "applied anticipation loss is the generate operator, so the "
            "partner must be the Jacobi step, not the native-ward oracle")
    step = float(dt) * float(advance_frac)
    if step <= 0.0:
        raise ValueError(
            "advance_frac must be > 0; 0 would skip the partner move and "
            "the extra-pass would see the Jacobi input")

    if leadership not in ("designed", "xz", "interface"):
        raise ValueError(
            f"unknown extra-pass leadership {leadership!r}; expected "
            "xz/designed (x↔z on generated rows) or interface (x↔z on "
            "native-contact rows)")
    xt, x1, tdict = batch["x_t"], batch["x_1"], batch["t"]
    nres = xt["bb_ca"].shape[1]
    des = _designed(batch, nres, xt["bb_ca"].device)
    if leadership == "interface":
        if "x_target" not in batch:
            raise ValueError(
                "interface x↔z extra-pass FM needs x_target to mark the "
                "exchange site; the partner is not a flow leader")
        site = native_interface_mask(batch)
        if int(site.sum()) == 0:
            site = des
    else:
        site = des
    missing_f = [m for m in followers if m not in MODALITIES]
    if missing_f or not followers:
        raise ValueError(
            f"followers must be a non-empty subset of {MODALITIES}; got {followers!r}")
    routed = None
    if router is not None:
        routed = router(batch, None)
        missing_r = [m for m in MODALITIES if m not in routed]
        if missing_r:
            raise ValueError(
                f"router returned no follower weight for {missing_r}; "
                "per-residue x↔z leadership would skip a channel")
    live_jacobi = jacobi_v
    if partner_advance in ("jacobi", "commit", "residue") and live_jacobi is None:
        jacobi_out = model.call_nn(batch)
        live_jacobi = {}
        for m in MODALITIES:
            if m not in jacobi_out or "v" not in jacobi_out[m]:
                raise ValueError(
                    f"jacobi partner-advance needs v for {m}; the extra-pass "
                    "cannot invent a partner step")
            live_jacobi[m] = jacobi_out[m]["v"]
    if partner_advance in ("jacobi", "commit", "residue"):
        missing_j = [m for m in MODALITIES if m not in live_jacobi]
        if missing_j:
            raise ValueError(
                f"jacobi partner-advance missing v for {missing_j}")
    if partner_advance == "residue":
        if routed is None:
            raise ValueError(
                "residue commitment needs a router so who answers can differ "
                "by residue; without weights leadership would be global")
        vel = {m: live_jacobi[m].detach() for m in MODALITIES}
        probe_x, follower = residue_commitment(
            xt, vel, tdict, routed, dt, row_mask=site)
        probe = dict(batch)
        probe["x_t"] = {**xt, **probe_x}
        extra = model.call_nn(probe)
        terms = []
        for m in followers:
            if m not in extra or "v" not in extra[m]:
                raise ValueError(f"extra-pass returned no v for {m}")
            u = native_velocity(xt[m], x1[m], tdict[m])
            err = (extra[m]["v"] - u).pow(2).sum(dim=-1)
            keep = follower[m] & site

            if "x_target" not in batch:
                raise ValueError(
                    "residue commitment keeps partner-near rows in the loss, "
                    "which needs x_target")
            near = partner_near_mask(batch)
            keep = keep | near
            w = keep.to(dtype=err.dtype)
            terms.append((err * w).sum() / w.sum().clamp_min(1.0))
        return sum(terms) / len(terms)
    terms = []
    for m in followers:
        probe_xt = dict(xt)
        for other in MODALITIES:
            if other == m:
                continue
            if partner_advance in ("jacobi", "commit"):
                partner_v = live_jacobi[other]
                if objective == "applied" or partner_advance == "commit":
                    partner_v = partner_v.detach()
                if partner_advance == "commit":

                    moved = anticipated_clean_state(
                        xt[other], partner_v, tdict[other],
                        max_step_nm=PRIOR_SCALE_NM)
                else:
                    moved = anticipated_partner_state(xt[other], partner_v, step)
                lead = site.to(dtype=moved.dtype)
            else:
                moved = native_ward_state(
                    xt[other], x1[other], tdict[other], step)
                if routed is None:
                    lead = site.to(dtype=moved.dtype)
                else:
                    w_m = routed[m].to(device=moved.device, dtype=moved.dtype)
                    w_o = routed[other].to(device=moved.device, dtype=moved.dtype)
                    lead = w_m / (w_m + w_o).clamp_min(1e-6)
                    lead = lead * site.to(dtype=lead.dtype)
            while lead.dim() < moved.dim():
                lead = lead[..., None]
            probe_xt[other] = xt[other] + lead * (moved - xt[other])
        probe = dict(batch)
        probe["x_t"] = probe_xt
        if partner_advance == "commit":
            probe_t = dict(tdict)
            for other in MODALITIES:
                if other == m or other not in probe_t:
                    continue
                probe_t[other] = torch.ones_like(tdict[other])
            probe["t"] = probe_t
        extra = model.call_nn(probe)
        if m not in extra or "v" not in extra[m]:
            raise ValueError(f"extra-pass returned no v for {m}")
        u = native_velocity(xt[m], x1[m], tdict[m])
        pred = extra[m]["v"]
        if objective == "applied":
            follow = site.to(dtype=pred.dtype)
            if routed is not None:
                follow = follow * routed[m].to(device=pred.device, dtype=pred.dtype)
            follow = follow * _schedule_t(tdict)
            pred = routed_extra_pass_velocity(
                live_jacobi[m].detach(), pred, follow, clip=DEFAULT_CLIP)
        err = (pred - u).pow(2).sum(dim=-1)
        w = site.to(dtype=err.dtype)

        if routed is not None and partner_advance != "commit":
            w = w * routed[m].to(device=err.device, dtype=err.dtype)
        denom = w.sum().clamp_min(1.0)
        terms.append((err * w).sum() / denom)
    return sum(terms) / len(terms)
