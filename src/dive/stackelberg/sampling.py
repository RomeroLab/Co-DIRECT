
from __future__ import annotations

from contextlib import contextmanager

import torch

from dive.stackelberg.joint_update import (
    DEFAULT_CLIP, PRIOR_SCALE_NM, anticipated_clean_state,
    anticipated_partner_state, blend, intervention_rows, partner_tensors,
    residue_commitment, partner_near_rows, rigid_contact_shift,
    routed_extra_pass_velocity, snap_anticipation_probe, snap_ca_to_partner)

MODES = ("off", "respond", "anticipate")
MODALITIES = ("bb_ca", "local_latents")

def follower_fraction(weight: torch.Tensor) -> torch.Tensor:

    peak = weight.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (weight / peak).clamp(0.0, 1.0)

def invert_leadership(routed: dict, designed: torch.Tensor) -> dict:

    des = designed.bool()
    out = {}
    for m, w in routed.items():
        if des.shape != w.shape[:2]:
            raise ValueError(
                f"invert designed has shape {tuple(des.shape)}; expected "
                f"{tuple(w.shape[:2])} to match router weight for {m}")
        peak = w.amax(dim=-1, keepdim=True)
        mask = des.to(device=w.device)
        out[m] = torch.where(mask, peak - w, w)
    return out

class Telemetry(list):
    def summary(self) -> dict:
        fired = [r for r in self if r["fired"]]
        if not fired:
            return {"steps": len(self), "fired": 0, "trunk_calls_extra": 0}
        import statistics as st

        def mean_of(key):

            vals = [r[key] for r in fired if r.get(key) is not None]
            return st.fmean(vals) if vals else None

        return {
            "steps": len(self),
            "fired": len(fired),
            "trunk_calls_extra": sum(r["extra_trunk_calls"] for r in self),
            "dt_median": st.median([r["dt"] for r in fired]),
            "relative_change_bb_ca": mean_of("rel_bb_ca"),
            "relative_change_local_latents": mean_of("rel_local_latents"),
        }

@contextmanager
def anticipating_sampler(model, *, mode="anticipate", alpha=1.0,
                         clip=DEFAULT_CLIP, telemetry=None, designed=None,
                         correct=MODALITIES, probe_context=None,
                         partner_only=False, committed=None, commitment=None,
                         router=None, clean_hat=False, response_router=False,
                         advance_frac=1.0, alpha_schedule=None,
                         extra_pass_gate=None, leader_commit=False,
                         invert_router=False, recycle_k=0, persist_sc=False,
                         commit_clean=False, residue_commit=False,
                         contact_snap=False, contact_rigid=False,
                         contact_clean=False, contact_reciprocal=False,
                         contact_snap_far_clean=False,
                         snap_anticipation=False):

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    if (committed is None) != (commitment is None):
        raise ValueError(
            "committed and commitment must be given together; a mask with no "
            "destination, or a destination with no rows to apply it to, would "
            "silently run the released probe under a label that claims "
            "otherwise")

    if isinstance(alpha, dict):
        missing = [m for m in MODALITIES if m not in alpha]
        if missing:
            raise ValueError(
                f"alpha mapping is missing {missing}; a modality with no weight "
                "would silently fall back to a different correction than the label")
        weights = {m: float(alpha[m]) for m in MODALITIES}
    else:
        weights = {m: float(alpha) for m in MODALITIES}
    if mode == "off" and any(w != 0.0 for w in weights.values()):
        raise ValueError("mode 'off' must carry alpha 0.0, or the label lies")
    if mode == "respond":
        weights = {m: 1.0 for m in MODALITIES}
    elif mode == "off":
        weights = {m: 0.0 for m in MODALITIES}
    effective = max(weights.values())
    if partner_only and effective != 0.0:
        nonzero = [m for m, w in weights.items() if w != 0.0]
        if len(nonzero) < 2:
            raise ValueError(
                "partner_only extra-pass holds the corrected modality and "
                "advances the others; with only "
                f"{nonzero} nonzero the probe is the Jacobi input and the "
                "extra pass is a no-op. Drop partner_only (shared probe) or "
                "give the partner a nonzero alpha")
    step_frac = float(advance_frac)
    if step_frac <= 0.0:
        raise ValueError(
            "advance_frac must be > 0; 0 would skip the extra-pass move and "
            "1.0 is the Euler partner state already used")
    if alpha_schedule not in (None, "t", "one_minus_t"):
        raise ValueError(
            f"unknown alpha_schedule {alpha_schedule!r}; expected t, "
            "one_minus_t, or None")
    if residue_commit and router is None:
        raise ValueError(
            "residue_commit needs a router; without per-residue weights the "
            "who would be the same on every residue")
    if residue_commit and (partner_only or commit_clean):
        raise ValueError(
            "residue_commit is one assembled probe; partner_only and "
            "commit_clean each replace a whole channel")
    if commit_clean and leader_commit:
        raise ValueError(
            "commit_clean shows the leader endpoint; leader_commit would "
            "turn that endpoint back into a per-residue velocity scale")
    if leader_commit and router is None:
        raise ValueError(
            "leader_commit needs a router; without weights the probe cannot "
            "tell leaders from followers and would silently keep advancing "
            "every residue")
    if invert_router and router is None:
        raise ValueError(
            "invert_router needs a router; there are no weights to swap")
    if invert_router and designed is None:
        raise ValueError(
            "invert_router needs designed; otherwise undesigned rows with "
            "weight 0 become the peak and receive extra-pass")
    n_recycle = int(recycle_k)
    if n_recycle < 0:
        raise ValueError(
            "recycle_k must be >= 0; a negative count would silently skip "
            "the Disco/AF clean re-evaluation")

    original = model.call_nn
    had_binding = "call_nn" in model.__dict__
    previous_t: dict = {}
    persist_stash: dict = {}

    def _stash_clean(batch, out):
        if not persist_sc:
            return
        hats = {}
        for m in MODALITIES:
            if m not in out or "v" not in out[m]:
                continue
            if m not in batch.get("x_t", {}) or m not in batch.get("t", {}):
                continue
            xt = batch["x_t"][m]
            hats[m] = anticipated_clean_state(
                xt, out[m]["v"], batch["t"][m].to(xt.dtype),
                max_step_nm=PRIOR_SCALE_NM)
        persist_stash.clear()
        persist_stash.update(hats)

    def call(batch, *args, **kwargs):
        batch_use = batch
        if persist_sc and persist_stash:
            batch_use = dict(batch)
            batch_use["_persist_clean"] = persist_stash
        output = original(batch_use, *args, **kwargs)
        row = {"fired": False, "extra_trunk_calls": 0, "dt": None,
               "rel_bb_ca": None, "rel_local_latents": None}
        recycled_sc = None
        for _ in range(n_recycle):
            hats = {}
            for m in MODALITIES:
                if m not in output or "v" not in output[m]:
                    continue
                if m not in batch.get("x_t", {}) or m not in batch.get("t", {}):
                    continue
                xt = batch["x_t"][m]
                hats[m] = anticipated_clean_state(
                    xt, output[m]["v"], batch["t"][m].to(xt.dtype),
                    max_step_nm=PRIOR_SCALE_NM)
            if not hats:
                break
            rec = dict(batch)
            rec["x_sc"] = hats
            recycled_sc = hats
            output = original(rec, *args, **kwargs)
            for m in output:
                if isinstance(output[m], dict):
                    for key in ("v_guided", "score", "score_guided",
                                "guided_v", "guided_score"):
                        output[m].pop(key, None)
            row["extra_trunk_calls"] += 1

        t_now = {m: float(batch["t"][m].mean()) for m in MODALITIES if m in batch["t"]}
        dt = None
        if previous_t:
            deltas = [t_now[m] - previous_t[m] for m in t_now if m in previous_t]
            if deltas:
                dt = sum(deltas) / len(deltas)
        previous_t.update(t_now)

        if effective == 0.0 or (
                (dt is None or dt <= 0.0)
                and not commit_clean and not contact_snap and not contact_rigid
                and not contact_clean and not contact_reciprocal
                and not contact_snap_far_clean and not snap_anticipation):
            if telemetry is not None:
                row["dt"] = dt
                telemetry.append(row)
            _stash_clean(batch, output)
            return output

        step_w = dict(weights)
        if alpha_schedule is not None:
            tmap = batch.get("t") or {}
            clock = tmap["bb_ca"] if "bb_ca" in tmap else next(iter(tmap.values()))
            scale = float(clock.reshape(-1)[0].clamp(0, 1))
            if alpha_schedule == "one_minus_t":
                scale = 1.0 - scale
            step_w = {m: w * scale for m, w in step_w.items()}

        routed = None
        if snap_anticipation and router is None:
            raise ValueError(
                "snap anticipation needs a router; who leads is per residue")
        if router is not None:
            routed = router(batch, output)
            missing = [m for m in MODALITIES if m not in routed]
            if missing:
                raise ValueError(
                    f"router returned no weight for {missing}; a modality with "
                    "no routed weight would silently fall back to a different "
                    "leadership direction than the label claims")
            nres = next(iter(batch["x_t"].values())).shape[1]
            for m in MODALITIES:
                w = routed[m]
                if w.shape[:2] != (batch["x_t"][m].shape[0], nres):
                    raise ValueError(
                        f"router weight for {m} has shape {tuple(w.shape)}; "
                        f"expected [b, {nres}]. Broadcasting it would route "
                        "another residue's leadership")
            if invert_router:
                routed = invert_leadership(routed, designed)

        if snap_anticipation:
            if routed is None:
                raise ValueError("snap anticipation needs router weights")
            if "x_target" not in batch:
                raise ValueError("snap anticipation needs x_target")
            partner, keep = partner_tensors(batch)
            vel = {m: output[m]["v"] for m in MODALITIES
                   if m in output and "v" in output[m]}
            rows = intervention_rows(designed, batch.get("mask"))
            probe_x, follower = snap_anticipation_probe(
                batch["x_t"], vel, batch["t"], routed, partner, keep,
                dt=dt, valid=batch.get("mask"), row_mask=rows)
            probe = dict(batch)
            probe["x_t"] = {**batch["x_t"], **probe_x}
            extra = original(probe, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            revised = {m: dict(v) for m, v in output.items()}
            for m in MODALITIES:
                if m not in revised or m not in extra or "v" not in extra.get(m, {}):
                    continue
                base = output[m]["v"]
                answer = extra[m]["v"]
                fol = follower[m]
                while fol.dim() < base.dim():
                    fol = fol[..., None]
                revised[m]["v"] = torch.where(fol, answer, base)
            if telemetry is not None:
                row["dt"] = dt
                telemetry.append(row)
            _stash_clean(batch, revised)
            return revised

        if contact_reciprocal:
            if designed is None:
                raise ValueError(
                    "contact_reciprocal needs designed; motif rows must stay "
                    "on the Jacobi backbone")
            if "bb_ca" not in output or "v" not in output["bb_ca"]:
                raise ValueError("contact_reciprocal needs the Jacobi backbone velocity")
            if "local_latents" not in output or "v" not in output["local_latents"]:
                raise ValueError("contact_reciprocal needs the Jacobi latent velocity")
            if "x_target" not in batch:
                raise ValueError("contact_reciprocal needs x_target")
            ca = batch["x_t"]["bb_ca"]
            z = batch["x_t"]["local_latents"]
            partner, keep = partner_tensors(batch)
            probe_ca, near = snap_ca_to_partner(
                ca, partner, keep, valid=batch.get("mask"))
            probe = dict(batch)
            probe["x_t"] = dict(batch["x_t"])
            probe["x_t"]["bb_ca"] = probe_ca
            extra = original(probe, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            revised = {m: dict(v) for m, v in output.items()}
            base = output["local_latents"]["v"]
            answer = extra["local_latents"]["v"]
            fol = near
            while fol.dim() < base.dim():
                fol = fol[..., None]
            revised["local_latents"]["v"] = torch.where(fol, answer, base)
            shipped = revised["local_latents"]["v"].detach()
            z_clean = anticipated_clean_state(
                z, shipped, batch["t"]["local_latents"].to(dtype=z.dtype),
                max_step_nm=PRIOR_SCALE_NM)
            heard = dict(batch)
            heard["x_t"] = dict(batch["x_t"])
            heard["x_t"]["local_latents"] = z_clean
            back = original(heard, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            des = designed.bool()
            while des.dim() > near.dim():
                des = des.any(dim=-1)
            if (des.dim() == 1 and near.dim() == 2
                    and des.shape[0] == near.shape[-1]):
                des = des.view(1, -1).expand(near.shape[0], -1)
            if des.shape != near.shape:
                raise ValueError(
                    f"designed shape {tuple(des.shape)} != near {tuple(near.shape)}")
            take = near & des
            base_ca = output["bb_ca"]["v"]
            ans_ca = back["bb_ca"]["v"]
            mask = take
            while mask.dim() < base_ca.dim():
                mask = mask[..., None]
            revised["bb_ca"]["v"] = torch.where(mask, ans_ca, base_ca)
            if telemetry is not None:
                row["dt"] = dt
                telemetry.append(row)
            _stash_clean(batch, revised)
            return revised

        if contact_clean:
            if "bb_ca" not in output or "v" not in output["bb_ca"]:
                raise ValueError("contact_clean needs the Jacobi Cα velocity")
            if "x_target" not in batch:
                raise ValueError("contact_clean needs x_target to choose rows")
            ca = batch["x_t"]["bb_ca"]
            clean = anticipated_clean_state(
                ca, output["bb_ca"]["v"], batch["t"]["bb_ca"].to(dtype=ca.dtype),
                max_step_nm=PRIOR_SCALE_NM)
            partner, keep = partner_tensors(batch)
            near = partner_near_rows(
                ca, partner, keep, valid=batch.get("mask"))
            probe = dict(batch)
            probe["x_t"] = dict(batch["x_t"])
            probe["x_t"]["bb_ca"] = clean
            extra = original(probe, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            revised = {m: dict(v) for m, v in output.items()}
            if "local_latents" in revised and "local_latents" in extra:
                base = output["local_latents"]["v"]
                answer = extra["local_latents"]["v"]
                fol = near
                while fol.dim() < base.dim():
                    fol = fol[..., None]
                revised["local_latents"]["v"] = torch.where(fol, answer, base)
            if telemetry is not None:
                row["dt"] = dt
                telemetry.append(row)
            _stash_clean(batch, revised)
            return revised

        if contact_snap_far_clean:
            if "bb_ca" not in output or "v" not in output["bb_ca"]:
                raise ValueError("split commitment needs the Jacobi backbone velocity")
            if "local_latents" not in output or "v" not in output["local_latents"]:
                raise ValueError("split commitment needs the Jacobi latent velocity")
            if "x_target" not in batch:
                raise ValueError("split commitment needs x_target")
            ca = batch["x_t"]["bb_ca"]
            z = batch["x_t"]["local_latents"]
            partner, keep = partner_tensors(batch)
            probe_ca, near = snap_ca_to_partner(
                ca, partner, keep, valid=batch.get("mask"))
            probe = dict(batch)
            probe["x_t"] = dict(batch["x_t"])
            probe["x_t"]["bb_ca"] = probe_ca
            extra = original(probe, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            revised = {m: dict(v) for m, v in output.items()}
            base = output["local_latents"]["v"]
            answer = extra["local_latents"]["v"]
            fol = near
            while fol.dim() < base.dim():
                fol = fol[..., None]
            revised["local_latents"]["v"] = torch.where(fol, answer, base)
            clean = anticipated_clean_state(
                ca, output["bb_ca"]["v"].detach(),
                batch["t"]["bb_ca"].to(dtype=ca.dtype),
                max_step_nm=PRIOR_SCALE_NM)
            heard = dict(batch)
            heard["x_t"] = dict(batch["x_t"])
            heard["x_t"]["bb_ca"] = clean
            back = original(heard, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            valid = batch.get("mask")
            far = ~near if valid is None else valid.bool() & ~near
            fol = far
            while fol.dim() < base.dim():
                fol = fol[..., None]
            revised["local_latents"]["v"] = torch.where(
                fol, back["local_latents"]["v"], revised["local_latents"]["v"])
            if telemetry is not None:
                row["dt"] = dt
                telemetry.append(row)
            _stash_clean(batch, revised)
            return revised

        if contact_rigid or contact_snap:
            if "x_target" not in batch:
                raise ValueError(
                    "contact placement needs x_target; the message is not a velocity")
            partner, keep = partner_tensors(batch)
            place = rigid_contact_shift if contact_rigid else snap_ca_to_partner
            probe_ca, near = place(
                batch["x_t"]["bb_ca"], partner, keep,
                valid=batch.get("mask"))
            probe = dict(batch)
            probe["x_t"] = dict(batch["x_t"])
            probe["x_t"]["bb_ca"] = probe_ca
            extra = original(probe, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            revised = {m: dict(v) for m, v in output.items()}
            if "local_latents" in revised and "local_latents" in extra:
                base = output["local_latents"]["v"]
                answer = extra["local_latents"]["v"]
                fol = near
                while fol.dim() < base.dim():
                    fol = fol[..., None]
                revised["local_latents"]["v"] = torch.where(fol, answer, base)
            if telemetry is not None:
                row["dt"] = dt
                telemetry.append(row)
            _stash_clean(batch, revised)
            return revised

        if residue_commit:
            if routed is None:
                raise ValueError(
                    "residue_commit needs a router at this step")
            vel = {m: output[m]["v"] for m in MODALITIES
                   if m in output and "v" in output[m]}
            row_mask = None
            if designed is not None:
                row_mask = designed.bool()
                while row_mask.dim() > 2:
                    row_mask = row_mask.any(dim=-1)
            probe_x, follower = residue_commitment(
                batch["x_t"], vel, batch["t"], routed, dt, row_mask=row_mask)
            probe = dict(batch)
            probe["x_t"] = {**batch["x_t"], **probe_x}
            extra = original(probe, *args, **kwargs)
            row["extra_trunk_calls"] += 1
            revised = {m: dict(v) for m, v in output.items()}
            for m in MODALITIES:
                if m not in revised or m not in extra or "v" not in extra.get(m, {}):
                    continue
                base = output[m]["v"]
                answer = extra[m]["v"]
                fol = follower[m]
                while fol.dim() < base.dim():
                    fol = fol[..., None]
                revised[m]["v"] = torch.where(fol, answer, base)
            if telemetry is not None:
                row["dt"] = dt
                telemetry.append(row)
            _stash_clean(batch, revised)
            return revised

        if commit_clean:
            advanced = {}
            for m in MODALITIES:
                if (m not in batch["x_t"] or m not in output
                        or "v" not in output[m] or weights.get(m, 0.0) == 0.0):
                    continue
                xt = batch["x_t"][m]
                tm = batch["t"][m].to(dtype=xt.dtype)
                advanced[m] = anticipated_clean_state(
                    xt, output[m]["v"], tm, max_step_nm=PRIOR_SCALE_NM)
        else:
            advanced = {m: anticipated_partner_state(
                            batch["x_t"][m], output[m]["v"], dt * step_frac)
                        for m in MODALITIES
                        if m in batch["x_t"] and m in output
                        and weights.get(m, 0.0) != 0.0}
        if leader_commit:
            if routed is None:
                raise ValueError(
                    "leader_commit needs router weights before the probe")
            for m in list(advanced):
                frac = follower_fraction(routed[m].to(advanced[m].dtype))
                lead = 1.0 - frac
                while lead.dim() < advanced[m].dim():
                    lead = lead[..., None]
                xt = batch["x_t"][m]
                advanced[m] = xt + lead * (advanced[m] - xt)
        if designed is not None:
            keep_adv = designed.bool()
            xt0 = batch["x_t"]["bb_ca"]
            if keep_adv.shape[:2] != xt0.shape[:2]:
                raise ValueError(
                    f"designed has shape {tuple(keep_adv.shape)}; expected "
                    f"[b, n]={tuple(xt0.shape[:2])}. Advancing frozen catalyst "
                    "/ framework rows with dt·v is how Motif-fit left 8.70")
            for m in list(advanced):
                k = keep_adv
                while k.dim() < advanced[m].dim():
                    k = k[..., None]
                advanced[m] = torch.where(k, advanced[m], batch["x_t"][m])
        if committed is not None and "bb_ca" in advanced:
            state = batch["x_t"]["bb_ca"]
            if commitment.shape != state.shape:
                raise ValueError(
                    f"commitment has shape {tuple(commitment.shape)}; expected "
                    f"{tuple(state.shape)}. Broadcasting it would place rows at "
                    "another residue's destination")
            rows = committed.to(torch.bool)
            while rows.dim() < state.dim():
                rows = rows[..., None]

            t_ca = batch.get("t", {}).get("bb_ca")
            if t_ca is None:
                raise ValueError(
                    "a commitment needs t to be expressed in the current time "
                    "frame; without it the only options are placing the rows at "
                    "a t=1 position, which is measured to derail, or ignoring "
                    "the commitment while still claiming it")
            tt = t_ca.to(state.dtype)
            while tt.dim() < state.dim():
                tt = tt[..., None]

            frac = (dt / (1.0 - tt).clamp_min(1e-6)).clamp(max=1.0)
            advanced["bb_ca"] = torch.where(
                rows, state + frac * (commitment.to(state.dtype) - state),
                advanced["bb_ca"])

        def probe_once(hold=()):
            probe = dict(batch)
            if recycled_sc is not None:
                probe["x_sc"] = recycled_sc
            probe["x_t"] = {**batch["x_t"],
                            **{m: s for m, s in advanced.items() if m not in hold}}
            if commit_clean:
                tprobe = {k: val for k, val in batch.get("t", {}).items()}
                for m in advanced:
                    if m in hold or m not in tprobe:
                        continue
                    tprobe[m] = torch.ones_like(tprobe[m])
                probe["t"] = tprobe
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
            if probe_context is None:
                return original(probe, *args, **kwargs)

            with probe_context(batch, output):
                return original(probe, *args, **kwargs)

        targets = [m for m in MODALITIES
                   if m in output and m in correct and m in advanced]
        responded_v = {}
        if partner_only:

            for m in targets:
                out_m = probe_once(hold=(m,))
                if m in out_m:
                    responded_v[m] = out_m[m]["v"]
            row["extra_trunk_calls"] += len(targets)
        else:
            shared = probe_once()
            for m in targets:
                if m in shared:
                    responded_v[m] = shared[m]["v"]
            row["extra_trunk_calls"] += 1

        if response_router:
            nres = next(iter(batch["x_t"].values())).shape[1]
            bsz = next(iter(batch["x_t"].values())).shape[0]
            resp = {}
            for m in MODALITIES:
                if m in output and m in responded_v:
                    base = output[m]["v"]
                    delta = responded_v[m] - base
                    denom = base.norm(dim=-1).clamp_min(1e-6)
                    resp[m] = delta.norm(dim=-1) / denom
                else:
                    resp[m] = torch.zeros(bsz, nres)
            if routed is None:
                routed = resp
            else:
                routed = {m: routed[m].to(dtype=resp[m].dtype) * resp[m]
                          for m in MODALITIES}
        if extra_pass_gate is not None:
            gated = extra_pass_gate(batch, output, responded_v)
            if routed is None:
                routed = gated
            else:
                routed = {m: routed[m].to(dtype=gated[m].dtype) * gated[m]
                          for m in MODALITIES}

        keep = None
        if designed is not None:
            keep = designed.bool()
            while keep.dim() > 2:
                keep = keep.any(dim=-1)

        revised = {m: dict(v) for m, v in output.items()}
        for m in MODALITIES:
            if m not in revised or m not in responded_v or m not in correct:
                continue
            if weights.get(m, 0.0) == 0.0:
                continue
            base = output[m]["v"]
            sched = step_w[m] / weights[m] if weights[m] else 0.0
            if commit_clean:

                new = responded_v[m]
            elif routed is None:
                new = blend(base, responded_v[m], step_w[m], clip=clip)
            else:

                w = routed[m].to(base.dtype) * sched
                new = routed_extra_pass_velocity(
                    base, responded_v[m], w, clip=clip)
            if keep is not None and keep.shape == base.shape[:2]:
                new = torch.where(keep[..., None], new, base)
            revised[m]["v"] = new

            for key in ("v_guided", "score", "score_guided", "guided_v", "guided_score"):
                revised[m].pop(key, None)
            denom = base.norm().clamp_min(1e-12)
            row[f"rel_{m}"] = float((new - base).norm() / denom)
        row["fired"] = True
        row["dt"] = dt
        if telemetry is not None:
            telemetry.append(row)
        _stash_clean(batch, revised)
        return revised

    model.call_nn = call
    try:
        yield telemetry
    finally:
        if had_binding:
            model.call_nn = original
        else:
            del model.call_nn
