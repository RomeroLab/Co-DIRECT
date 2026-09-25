
from __future__ import annotations

import torch

from dive.anticipate.capture import capture_last_forward, designed_from_batch
from dive.training.sequence_recovery import refuse_outcome_labels

MODALITIES = ("bb_ca", "local_latents")
NATIVE_CONTACT_NM = 0.5

def partner_coords_padded(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:

    from dive.stackelberg.joint_update import partner_tensors
    return partner_tensors(batch)

def _native_ca(batch: dict) -> torch.Tensor:
    if "x_1" in batch and isinstance(batch["x_1"], dict) and "bb_ca" in batch["x_1"]:
        return batch["x_1"]["bb_ca"]
    if "bb_ca" in batch:
        return batch["bb_ca"]
    raise ValueError("need native bb_ca (x_1['bb_ca'] or bb_ca) to label contacts")

def partner_near_mask(batch: dict, *, contact_nm: float = NATIVE_CONTACT_NM
                      ) -> torch.Tensor:

    ca = _native_ca(batch)
    partner, keep = partner_coords_padded(batch)
    dist = torch.cdist(ca, partner)
    dist = dist.masked_fill(~keep[:, None, :], 1e6)
    near = dist.min(dim=-1).values <= float(contact_nm)
    if "mask" in batch:
        near = near & batch["mask"].bool()
    return near

def live_contact_probe(batch: dict, *, contact_nm: float = NATIVE_CONTACT_NM
                      ) -> tuple[torch.Tensor, torch.Tensor]:

    if "x_t" not in batch or "bb_ca" not in batch["x_t"]:
        raise ValueError(
            "contact misalignment needs the flow Cα; a velocity is not the message")
    from dive.stackelberg.joint_update import snap_ca_to_partner
    ca = batch["x_t"]["bb_ca"]
    partner, keep = partner_coords_padded(batch)
    valid = batch["mask"] if "mask" in batch else None
    return snap_ca_to_partner(
        ca, partner, keep, contact_nm=contact_nm, valid=valid)

def reciprocal_contact_loss(model, batch, jacobi_v, *,
                            contact_nm: float = NATIVE_CONTACT_NM
                            ) -> torch.Tensor:

    from dive.stackelberg.extra_pass_gate import native_velocity
    from dive.stackelberg.joint_update import (
        PRIOR_SCALE_NM, anticipated_clean_state)
    _refuse_predictor(batch)
    if jacobi_v is None or "local_latents" not in jacobi_v:
        raise ValueError("reciprocal contact needs the Jacobi latent velocity")
    if "x_1" not in batch or "local_latents" not in batch["x_1"]:
        raise ValueError("reciprocal contact needs native local_latents")
    if "bb_ca" not in batch["x_1"]:
        raise ValueError("reciprocal contact needs native bb_ca")
    probe_ca, near = live_contact_probe(batch, contact_nm=contact_nm)
    probe = dict(batch)
    probe["x_t"] = dict(batch["x_t"])
    probe["x_t"]["bb_ca"] = probe_ca
    snap_out = model.call_nn(probe)
    if "local_latents" not in snap_out or "v" not in snap_out["local_latents"]:
        raise ValueError("trunk returned no latent velocity for the snap")
    u_lat = native_velocity(
        batch["x_t"]["local_latents"], batch["x_1"]["local_latents"],
        batch["t"]["local_latents"])
    err_lat = (snap_out["local_latents"]["v"] - u_lat).pow(2).sum(dim=-1)
    w_lat = near.to(dtype=err_lat.dtype)
    lat_loss = (err_lat * w_lat).sum() / w_lat.sum().clamp_min(1.0)

    answered = snap_out["local_latents"]["v"].detach()
    base = jacobi_v["local_latents"].detach()
    fol = near
    while fol.dim() < answered.dim():
        fol = fol[..., None]
    shipped = torch.where(fol, answered, base.to(dtype=answered.dtype))
    z = batch["x_t"]["local_latents"]
    z_clean = anticipated_clean_state(
        z, shipped, batch["t"]["local_latents"].to(dtype=z.dtype),
        max_step_nm=PRIOR_SCALE_NM)
    heard = dict(batch)
    heard["x_t"] = dict(batch["x_t"])
    heard["x_t"]["local_latents"] = z_clean
    back = model.call_nn(heard)
    if "bb_ca" not in back or "v" not in back["bb_ca"]:
        raise ValueError("trunk returned no backbone velocity for the committed latent")
    u_ca = native_velocity(
        batch["x_t"]["bb_ca"], batch["x_1"]["bb_ca"], batch["t"]["bb_ca"])
    designed = designed_from_batch(batch)
    while designed.dim() > near.dim():
        designed = designed.any(dim=-1)
    take = near & designed
    err_ca = (back["bb_ca"]["v"] - u_ca).pow(2).sum(dim=-1)
    w_ca = take.to(dtype=err_ca.dtype)
    ca_loss = (err_ca * w_ca).sum() / w_ca.sum().clamp_min(1.0)
    return lat_loss + ca_loss

def clean_backbone_latent_loss(model, batch, jacobi_v, *,
                               contact_nm: float = NATIVE_CONTACT_NM
                               ) -> torch.Tensor:

    from dive.stackelberg.extra_pass_gate import native_velocity
    from dive.stackelberg.joint_update import (
        PRIOR_SCALE_NM, anticipated_clean_state, partner_near_rows)
    _refuse_predictor(batch)
    if jacobi_v is None or "bb_ca" not in jacobi_v:
        raise ValueError("clean-backbone latent target needs the Jacobi Cα velocity")
    if "x_1" not in batch or "local_latents" not in batch["x_1"]:
        raise ValueError("clean-backbone latent target needs native local_latents")
    ca = batch["x_t"]["bb_ca"]
    clean = anticipated_clean_state(
        ca, jacobi_v["bb_ca"].detach(), batch["t"]["bb_ca"].to(dtype=ca.dtype),
        max_step_nm=PRIOR_SCALE_NM)
    partner, keep = partner_coords_padded(batch)
    near = partner_near_rows(
        ca, partner, keep, contact_nm=contact_nm,
        valid=batch.get("mask"))
    probe = dict(batch)
    probe["x_t"] = dict(batch["x_t"])
    probe["x_t"]["bb_ca"] = clean
    out = model.call_nn(probe)
    if "local_latents" not in out or "v" not in out["local_latents"]:
        raise ValueError("trunk returned no latent velocity for the clean backbone")
    u = native_velocity(
        batch["x_t"]["local_latents"], batch["x_1"]["local_latents"],
        batch["t"]["local_latents"])
    err = (out["local_latents"]["v"] - u).pow(2).sum(dim=-1)
    weight = near.to(dtype=err.dtype)
    return (err * weight).sum() / weight.sum().clamp_min(1.0)

def far_clean_latent_loss(model, batch, jacobi_v, *,
                          contact_nm: float = NATIVE_CONTACT_NM
                          ) -> torch.Tensor:

    from dive.stackelberg.extra_pass_gate import native_velocity
    from dive.stackelberg.joint_update import (
        PRIOR_SCALE_NM, anticipated_clean_state, partner_near_rows)
    _refuse_predictor(batch)
    if jacobi_v is None or "bb_ca" not in jacobi_v:
        raise ValueError("far clean latent needs the Jacobi Cα velocity")
    if "x_1" not in batch or "local_latents" not in batch["x_1"]:
        raise ValueError("far clean latent needs native local_latents")
    ca = batch["x_t"]["bb_ca"]
    clean = anticipated_clean_state(
        ca, jacobi_v["bb_ca"].detach(), batch["t"]["bb_ca"].to(dtype=ca.dtype),
        max_step_nm=PRIOR_SCALE_NM)
    partner, keep = partner_coords_padded(batch)
    near = partner_near_rows(
        ca, partner, keep, contact_nm=contact_nm, valid=batch.get("mask"))
    mask = batch["mask"].bool() if "mask" in batch else torch.ones_like(near)
    far = mask & ~near
    probe = dict(batch)
    probe["x_t"] = dict(batch["x_t"])
    probe["x_t"]["bb_ca"] = clean
    out = model.call_nn(probe)
    if "local_latents" not in out or "v" not in out["local_latents"]:
        raise ValueError("trunk returned no latent velocity for the far clean backbone")
    u = native_velocity(
        batch["x_t"]["local_latents"], batch["x_1"]["local_latents"],
        batch["t"]["local_latents"])
    err = (out["local_latents"]["v"] - u).pow(2).sum(dim=-1)
    weight = far.to(dtype=err.dtype)
    return (err * weight).sum() / weight.sum().clamp_min(1.0)

def rigid_contact_latent_loss(model, batch, *,
                              contact_nm: float = NATIVE_CONTACT_NM
                              ) -> torch.Tensor:

    from dive.stackelberg.extra_pass_gate import native_velocity
    from dive.stackelberg.joint_update import rigid_contact_shift
    _refuse_predictor(batch)
    if "x_t" not in batch or "bb_ca" not in batch["x_t"]:
        raise ValueError("rigid contact message needs the flow Cα")
    if "x_1" not in batch or "local_latents" not in batch["x_1"]:
        raise ValueError("rigid contact latent target needs native local_latents")
    ca = batch["x_t"]["bb_ca"]
    partner, keep = partner_coords_padded(batch)
    valid = batch["mask"] if "mask" in batch else None
    probe_ca, near = rigid_contact_shift(
        ca, partner, keep, contact_nm=contact_nm, valid=valid)
    probe = dict(batch)
    probe["x_t"] = dict(batch["x_t"])
    probe["x_t"]["bb_ca"] = probe_ca
    out = model.call_nn(probe)
    if "local_latents" not in out or "v" not in out["local_latents"]:
        raise ValueError("trunk returned no latent velocity for the rigid contact")
    u = native_velocity(
        batch["x_t"]["local_latents"], batch["x_1"]["local_latents"],
        batch["t"]["local_latents"])
    err = (out["local_latents"]["v"] - u).pow(2).sum(dim=-1)
    weight = near.to(dtype=err.dtype)
    return (err * weight).sum() / weight.sum().clamp_min(1.0)

def contact_misalignment_latent_loss(model, batch, *,
                                    contact_nm: float = NATIVE_CONTACT_NM
                                    ) -> torch.Tensor:

    from dive.stackelberg.extra_pass_gate import native_velocity
    _refuse_predictor(batch)
    if "x_1" not in batch or "local_latents" not in batch["x_1"]:
        raise ValueError("misalignment latent target needs native local_latents")
    probe_ca, near = live_contact_probe(batch, contact_nm=contact_nm)
    probe = dict(batch)
    probe["x_t"] = dict(batch["x_t"])
    probe["x_t"]["bb_ca"] = probe_ca
    out = model.call_nn(probe)
    if "local_latents" not in out or "v" not in out["local_latents"]:
        raise ValueError("trunk returned no latent velocity for the misalignment")
    u = native_velocity(
        batch["x_t"]["local_latents"], batch["x_1"]["local_latents"],
        batch["t"]["local_latents"])
    err = (out["local_latents"]["v"] - u).pow(2).sum(dim=-1)
    weight = near.to(dtype=err.dtype)
    return (err * weight).sum() / weight.sum().clamp_min(1.0)

def conditioned_residue_mask(batch: dict) -> torch.Tensor:

    mask = batch["mask"].bool()
    motif = batch.get("motif_mask")
    if motif is None:
        return torch.zeros_like(mask)
    cond = motif.bool()
    while cond.dim() > mask.dim():
        cond = cond.any(dim=-1)
    if cond.shape != mask.shape:
        return torch.zeros_like(mask)
    return mask & cond

def conditioned_ca_loss(model, batch, nn_out) -> torch.Tensor:

    _refuse_predictor(batch)
    rows = conditioned_residue_mask(batch)
    if "x_1" not in batch or "bb_ca" not in batch["x_1"]:
        raise ValueError("conditioned Cα loss needs native bb_ca")
    if int(rows.sum()) == 0:
        return batch["x_1"]["bb_ca"].new_zeros(())
    pred = _clean_pred(model.fm.base_flow_matchers["bb_ca"], batch, nn_out, "bb_ca")
    per = unreduced_clean_error(batch["x_1"]["bb_ca"], pred, batch["mask"].bool())
    t = batch["t"]["bb_ca"]
    while t.dim() < per.dim():
        t = t.unsqueeze(-1)
    per = per / ((1.0 - t) ** 2 + 1e-5)
    return interface_mean(per, rows)

def native_interface_mask(batch: dict, *, contact_nm: float = NATIVE_CONTACT_NM
                          ) -> torch.Tensor:

    near = partner_near_mask(batch, contact_nm=contact_nm)
    return near & designed_from_batch(batch)

def unreduced_clean_error(x_1: torch.Tensor, x_1_pred: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:

    err = (x_1 - x_1_pred) * mask[..., None].to(dtype=x_1.dtype)
    return (err ** 2).sum(dim=-1)

def interface_mean(per_residue: torch.Tensor, iface: torch.Tensor) -> torch.Tensor:

    w = iface.to(dtype=per_residue.dtype)
    denom = w.sum().clamp_min(1.0)
    return (per_residue * w).sum() / denom

def _clean_pred(matcher, batch, nn_out, modality: str) -> torch.Tensor:
    out = matcher.nn_out_add_clean_sample_prediction(
        x_t=batch["x_t"][modality],
        t=batch["t"][modality],
        mask=batch["mask"],
        nn_out=nn_out[modality],
    )
    if "x_1" not in out:
        raise ValueError(f"{modality} nn_out has no x_1 after clean-sample prediction")
    return out["x_1"]

def _refuse_predictor(payload) -> None:
    refuse_outcome_labels(payload)
    if not payload:
        return
    banned = ("iptm", "ipae", "plddt", "ptm")
    for key in payload:
        low = str(key).lower()
        if any(tok in low for tok in banned):
            raise ValueError("predictor scores are not training labels")

def interface_fm_extra(model, held: dict, *, contact_nm: float = NATIVE_CONTACT_NM,
                       modalities: tuple[str, ...] = MODALITIES) -> torch.Tensor:

    _refuse_predictor(held.get("batch"))
    batch = held["batch"]
    nn_out = held["out"]
    iface = native_interface_mask(batch, contact_nm=contact_nm)
    extras = []
    for m in modalities:
        if m not in batch.get("x_1", {}):
            continue
        pred = _clean_pred(model.fm.base_flow_matchers[m], batch, nn_out, m)
        per = unreduced_clean_error(batch["x_1"][m], pred, batch["mask"].bool())
        t = batch["t"][m]
        while t.dim() < per.dim():
            t = t.unsqueeze(-1)
        per = per / ((1.0 - t) ** 2 + 1e-5)
        extras.append(interface_mean(per, iface))
    if not extras:
        return batch["mask"].new_zeros(())
    return sum(extras) / len(extras)

def _rotation_angle(pred: torch.Tensor, native: torch.Tensor) -> torch.Tensor:

    if pred.shape[0] < 3:
        raise ValueError("a rotation needs at least 3 points")
    pc = pred - pred.mean(dim=0, keepdim=True)
    nc = native - native.mean(dim=0, keepdim=True)
    h = pc.transpose(0, 1) @ nc
    u, _, vh = torch.linalg.svd(h)
    eye = torch.ones(3, device=pred.device, dtype=pred.dtype)
    eye[-1] = torch.det(vh.transpose(0, 1) @ u.transpose(0, 1)).sign()
    rot = vh.transpose(0, 1) @ torch.diag(eye) @ u.transpose(0, 1)
    cosine = ((rot.diagonal().sum() - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)

def rigid_placement_loss(pred: torch.Tensor, native: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:

    if pred.shape != native.shape or mask.shape[:2] != pred.shape[:2]:
        raise ValueError(
            f"pred {tuple(pred.shape)}, native {tuple(native.shape)}, "
            f"mask {tuple(mask.shape)} do not describe one patch")
    weight = mask.to(dtype=pred.dtype)
    count = weight.sum(dim=-1)
    has = count > 0
    denom = count.clamp_min(1.0).unsqueeze(-1)
    centroid_pred = (pred * weight.unsqueeze(-1)).sum(dim=1) / denom
    centroid_native = (native * weight.unsqueeze(-1)).sum(dim=1) / denom
    centroid = (centroid_pred - centroid_native).pow(2).sum(dim=-1)
    centroid = torch.where(has, centroid, torch.zeros_like(centroid))
    rotation = pred.new_zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        if int(mask[i].sum()) < 3:
            continue
        rotation[i] = _rotation_angle(pred[i, mask[i]], native[i, mask[i]])
    total = centroid + rotation.pow(2)
    if not torch.isfinite(total).all():

        return pred.new_zeros(())
    return total.sum() / has.to(dtype=pred.dtype).sum().clamp_min(1.0)

def interface_frame_loss(model, held: dict, *,
                         contact_nm: float = NATIVE_CONTACT_NM) -> torch.Tensor:

    _refuse_predictor(held.get("batch"))
    batch = held["batch"]
    iface = native_interface_mask(batch, contact_nm=contact_nm)
    pred = _clean_pred(model.fm.base_flow_matchers["bb_ca"], batch, held["out"], "bb_ca")
    return rigid_placement_loss(pred, batch["x_1"]["bb_ca"], iface)

def native_plus_interface_step(model, batch, *, lambda_iface: float,
                               contact_nm: float = NATIVE_CONTACT_NM,
                               batch_idx: int = 0,
                               modalities: tuple[str, ...] = MODALITIES,
                               lambda_frame: float = 0.0):

    from dive.cre.train import native_training_step

    _refuse_predictor(batch)
    with capture_last_forward(model) as held:
        loss, parts = native_training_step(model, batch, batch_idx=batch_idx)
        extra = None
        frame = None
        if float(lambda_iface) != 0.0:
            extra = interface_fm_extra(
                model, held, contact_nm=contact_nm, modalities=modalities)
            loss = loss + float(lambda_iface) * extra
        if float(lambda_frame) != 0.0:
            frame = interface_frame_loss(model, held, contact_nm=contact_nm)
            loss = loss + float(lambda_frame) * frame
        if held.get("batch") is not None:

            model._last_fm_batch = held["batch"]
        if held.get("out") is not None:
            out = held["out"]
            model._last_jacobi_v = {
                m: out[m]["v"] for m in MODALITIES
                if isinstance(out, dict) and m in out
                and isinstance(out[m], dict) and "v" in out[m]
            }
            model._last_out = out
    parts = dict(parts)
    parts["interface_fm"] = float(extra.detach()) if extra is not None else 0.0
    parts["lambda_iface"] = float(lambda_iface)
    parts["interface_frame"] = float(frame.detach()) if frame is not None else 0.0
    parts["lambda_frame"] = float(lambda_frame)
    return loss, parts
