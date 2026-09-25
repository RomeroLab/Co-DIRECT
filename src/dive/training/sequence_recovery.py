
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

_RESIDUE_FEAT_TYPES = ("OptionalResidueTypeSeqFeat", "ResidueTypeSeqFeat")

def refuse_outcome_labels(payload: Mapping | None) -> None:
    if not payload:
        return

    from dive.training.relation_contrast import FORBIDDEN_LABEL_KEYS

    for key in payload:
        lowered = str(key).lower()
        if any(token in lowered for token in FORBIDDEN_LABEL_KEYS):
            raise ValueError("J_final / predictor / function scores are not training labels")

def generation_hard_sequence(seq_logits: torch.Tensor) -> torch.Tensor:

    if seq_logits.ndim < 2:
        raise ValueError("sequence logits must include a residue-type axis")
    return seq_logits.argmax(dim=-1)

def straight_through_residue_onehot(
    seq_logits: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:

    hard = generation_hard_sequence(seq_logits)
    n_types = seq_logits.shape[-1]
    hard_oh = F.one_hot(hard, num_classes=n_types).to(dtype=seq_logits.dtype)
    soft = F.softmax(seq_logits, dim=-1)
    ste = hard_oh + soft - soft.detach()
    if mask is not None:
        ste = ste * mask[..., None].to(dtype=ste.dtype)
    return ste

def softmax_residue_onehot(
    seq_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:

    scale = float(temperature) if float(temperature) > 0 else 1.0
    soft = F.softmax(seq_logits / scale, dim=-1)
    if mask is not None:
        soft = soft * mask[..., None].to(dtype=soft.dtype)
    return soft

def second_pass_conditional_kind(
    *,
    keep_latents: bool = False,
    sequence_latents: bool = False,
) -> str:

    if keep_latents and sequence_latents:
        raise ValueError("keep_latents and sequence_latents are mutually exclusive")
    if keep_latents:
        return "fixed_z"
    if sequence_latents:
        return "sequence_inferred_z"
    return "prior_noise_z"

def fixed_z_conditional_mean(z: float) -> float:

    return float(z)

def marginal_sequence_conditional_mean(a: float, alpha: float) -> float:

    if not 0.0 < float(alpha) < 0.5:
        raise ValueError("alpha must be in (0, 1/2)")
    return (1.0 - 2.0 * float(alpha)) * float(a)

def apply_recovered_sequence_inputs(
    inputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    seq_logits: torch.Tensor,
    *,
    blank_target: bool = False,
    xt_noise: Mapping[str, Any] | None = None,
    residue_feature: str = "ste",
    temperature: float = 1.0,
    mix_t: float = 1.0,
    clean_xt: Mapping[str, Any] | None = None,
    keep_latents: bool = False,
    sequence_latents: bool = False,
    inferred_latents: torch.Tensor | None = None,
) -> dict:

    refuse_outcome_labels(inputs)
    refuse_outcome_labels(batch)
    if residue_feature not in {"ste", "soft"}:
        raise ValueError("residue_feature must be ste or soft")
    if keep_latents and sequence_latents:
        raise ValueError("keep_latents and sequence_latents are mutually exclusive")
    if sequence_latents and inferred_latents is None:
        raise ValueError("sequence_latents requires inferred_latents from A")
    scale = float(mix_t)
    if scale <= 0.0:
        raise ValueError("mix_t must be > 0 (t=0 zeros are the origin)")
    mask = batch["mask"]
    hard = generation_hard_sequence(seq_logits)
    feat = (
        straight_through_residue_onehot(seq_logits, mask)
        if residue_feature == "ste"
        else softmax_residue_onehot(seq_logits, mask, temperature=temperature)
    )
    second = dict(inputs)
    if "x_motif" in second and torch.is_tensor(second["x_motif"]):
        second["x_motif"] = torch.zeros_like(second["x_motif"])
    if blank_target and "x_target" in second and torch.is_tensor(second["x_target"]):
        second["x_target"] = torch.zeros_like(second["x_target"])
    if xt_noise is not None and isinstance(xt_noise, Mapping):
        second["x_t"] = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in xt_noise.items()
        }
    elif "x_t" in second and isinstance(second["x_t"], Mapping):
        noise = {
            key: torch.randn_like(value.detach()) if torch.is_tensor(value) else value
            for key, value in second["x_t"].items()
        }
        if scale < 1.0 and clean_xt is not None:
            mixed = {}
            for key, nval in noise.items():
                if torch.is_tensor(nval) and key in clean_xt and torch.is_tensor(clean_xt[key]):
                    mixed[key] = (1.0 - scale) * clean_xt[key].detach() + scale * nval
                else:
                    mixed[key] = nval
            second["x_t"] = mixed
        else:
            second["x_t"] = noise
        if keep_latents:
            if clean_xt is None or "local_latents" not in clean_xt:
                raise ValueError("keep_latents requires clean_xt local_latents")
            xt = dict(second["x_t"])
            xt["local_latents"] = clean_xt["local_latents"].detach()
            second["x_t"] = xt
        if sequence_latents:
            xt = dict(second["x_t"])
            xt["local_latents"] = inferred_latents
            second["x_t"] = xt
    if sequence_latents and isinstance(second.get("x_t"), Mapping):
        xt = dict(second["x_t"])
        xt["local_latents"] = inferred_latents
        second["x_t"] = xt
    if "x_sc" in second and isinstance(second["x_sc"], Mapping):
        second["x_sc"] = {
            key: torch.zeros_like(value) if torch.is_tensor(value) else value
            for key, value in second["x_sc"].items()
        }
    if "t" in second:
        times = second["t"]
        if isinstance(times, Mapping):
            second["t"] = {
                key: torch.ones_like(value) * min(scale, 1.0) if torch.is_tensor(value) else value
                for key, value in times.items()
            }
        elif torch.is_tensor(times):
            second["t"] = torch.ones_like(times) * min(scale, 1.0)
        if (keep_latents or sequence_latents) and isinstance(second.get("t"), Mapping) and "local_latents" in second["t"]:

            lat_t = second["t"]["local_latents"]
            if torch.is_tensor(lat_t):
                second["t"] = dict(second["t"])
                second["t"]["local_latents"] = torch.full_like(lat_t, 1e-3)
    second["residue_type"] = hard
    second["residue_type_onehot"] = feat
    second["use_residue_type_feature"] = True
    second["use_ca_coors_nm_feature"] = False
    mask_dict = dict(second.get("mask_dict") or {})
    mask_dict["residue_type"] = mask
    second["mask_dict"] = mask_dict
    return second

def infer_latents_from_sequence(
    model: Any,
    inputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    seq_logits: torch.Tensor,
    *,
    residue_feature: str = "ste",
    temperature: float = 1.0,
    native_residue_type: bool = False,
    xt_noise: Mapping[str, Any] | None = None,
) -> torch.Tensor:

    probe = apply_recovered_sequence_inputs(
        inputs, batch, seq_logits, xt_noise=xt_noise,
        residue_feature=residue_feature, temperature=temperature,
        keep_latents=False, sequence_latents=False,
    )
    out = (
        model.call_nn(probe)
        if native_residue_type
        else call_nn_with_residue_onehot(model, probe)
    )
    clean = model.fm.nn_out_to_clean_sample_prediction(probe, out)
    return clean["local_latents"]

def recompute_clean_from_sequence(
    model: Any,
    batch: Mapping[str, Any],
    sample: Mapping[str, Any],
    seq_logits: torch.Tensor,
    *,
    native_residue_type: bool = True,
    residue_feature: str = "ste",
) -> dict:

    refuse_outcome_labels(batch)
    refuse_outcome_labels(sample)
    before_ca = sample["bb_ca"].detach().clone()
    inputs = second_pass_inputs_from_sample(batch, sample)
    second = apply_recovered_sequence_inputs(
        inputs, batch, seq_logits, mix_t=1.0, residue_feature=residue_feature,
    )
    out = (
        model.call_nn(second)
        if native_residue_type
        else call_nn_with_residue_onehot(model, second)
    )
    clean = model.fm.nn_out_to_clean_sample_prediction(second, out)
    if not torch.equal(sample["bb_ca"], before_ca):
        raise RuntimeError("designed-span CA was rewritten")
    return {"bb_ca": clean["bb_ca"], "local_latents": clean["local_latents"]}

def attach_residue_ste_features(
    inputs: Mapping[str, Any],
    seq_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    residue_feature: str = "ste",
    temperature: float = 1.0,
) -> dict:

    refuse_outcome_labels(inputs)
    if residue_feature not in {"ste", "soft"}:
        raise ValueError("residue_feature must be ste or soft")
    hard = generation_hard_sequence(seq_logits)
    feat = (
        straight_through_residue_onehot(seq_logits, mask)
        if residue_feature == "ste"
        else softmax_residue_onehot(seq_logits, mask, temperature=temperature)
    )
    out = dict(inputs)
    if "x_t" in inputs and isinstance(inputs["x_t"], Mapping):
        out["x_t"] = dict(inputs["x_t"])
    out["residue_type"] = hard
    out["residue_type_onehot"] = feat
    out["use_residue_type_feature"] = True
    out["use_ca_coors_nm_feature"] = False
    mask_dict = dict(out.get("mask_dict") or {})
    mask_dict["residue_type"] = mask
    out["mask_dict"] = mask_dict
    return out

def apply_first_pass_residue_ste_nn(
    model: Any,
    inputs: Mapping[str, Any],
    unfeatured_out: Mapping[str, Any],
    *,
    nn_call: Callable | None = None,
) -> Any:

    refuse_outcome_labels(inputs)
    refuse_outcome_labels(unfeatured_out)
    nn_call = nn_call or model.call_nn
    mask = inputs["mask"]
    xt = None
    if "x_t" in inputs and isinstance(inputs["x_t"], Mapping):
        xt = inputs["x_t"].get("bb_ca")
        if torch.is_tensor(xt):
            xt = xt.detach().clone()
    clean = model.fm.nn_out_to_clean_sample_prediction(inputs, unfeatured_out)
    decoded = model.autoencoder.decode(
        clean["local_latents"], clean["bb_ca"], mask
    )
    featured = attach_residue_ste_features(
        inputs, decoded["seq_logits"], mask, residue_feature="ste"
    )
    if xt is not None and not torch.equal(featured["x_t"]["bb_ca"], xt):
        raise RuntimeError("designed-span CA was rewritten")
    with _inject_residue_onehot(model, featured.get("residue_type_onehot")):
        return nn_call(featured)

@contextmanager
def first_pass_residue_ste_sampling(model: Any, *, enabled: bool = True):

    if not enabled:
        yield
        return
    original = model.call_nn
    had_instance_binding = "call_nn" in model.__dict__

    def call(batch, *args, **kwargs):
        out = original(batch, *args, **kwargs)
        return apply_first_pass_residue_ste_nn(
            model, batch, out, nn_call=original
        )

    model.call_nn = call
    try:
        yield
    finally:
        if had_instance_binding:
            model.call_nn = original
        else:
            delattr(model, "call_nn")

class LastNNInputs:

    def __init__(self) -> None:
        self.first_xt: dict[str, Any] | None = None
        self.first_t: dict[str, Any] | None = None
        self.xt: dict[str, Any] | None = None
        self.t: dict[str, Any] | None = None

def _clone_state_map(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in payload.items()
        }
    if torch.is_tensor(payload):
        return payload.detach().clone()
    return payload

@contextmanager
def capture_last_nn_inputs(model: Any):

    bucket = LastNNInputs()
    original = model.call_nn
    had_instance_binding = "call_nn" in model.__dict__

    def call(batch, *args, **kwargs):
        if isinstance(batch, Mapping):
            if "x_t" in batch:
                cloned = _clone_state_map(batch["x_t"])
                if bucket.first_xt is None:
                    bucket.first_xt = cloned
                bucket.xt = cloned
            if "t" in batch:
                cloned_t = _clone_state_map(batch["t"])
                if bucket.first_t is None:
                    bucket.first_t = cloned_t
                bucket.t = cloned_t
        return original(batch, *args, **kwargs)

    model.call_nn = call
    try:
        yield bucket
    finally:
        if had_instance_binding:
            model.call_nn = original
        else:
            delattr(model, "call_nn")

def designed_span_mask(batch: Mapping[str, Any]) -> torch.Tensor:

    mask = batch["mask"]
    motif = batch.get("motif_residue_mask")
    if motif is None:
        atom_motif = batch.get("motif_mask")
        if atom_motif is not None and torch.is_tensor(atom_motif) and atom_motif.shape[:2] == mask.shape:
            motif = atom_motif.any(dim=-1) if atom_motif.dim() > 2 else atom_motif
    if motif is not None and torch.is_tensor(motif) and motif.shape == mask.shape and bool(motif.any()):
        free = mask & ~motif.to(dtype=torch.bool)
        if bool(free.any()):
            return free
    return mask

def required_span_mask(batch: Mapping[str, Any]) -> torch.Tensor:

    refuse_outcome_labels(batch)
    mask = batch["mask"]
    motif = batch.get("motif_residue_mask")
    if motif is None:
        atom_motif = batch.get("motif_mask")
        if atom_motif is not None and torch.is_tensor(atom_motif):
            if atom_motif.dim() > 2 and atom_motif.shape[:2] == mask.shape:
                motif = atom_motif.any(dim=-1)
            elif atom_motif.shape == mask.shape:
                motif = atom_motif
    if motif is None or not torch.is_tensor(motif) or motif.shape != mask.shape:
        return torch.zeros_like(mask)
    return mask & motif.to(dtype=torch.bool)

def masked_kabsch_params(
    mobile: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if mobile.shape != target.shape or mobile.shape[:2] != mask.shape:
        raise ValueError("kabsch CA/mask shape mismatch")
    tgt = target.detach()
    w = mask.to(dtype=mobile.dtype).unsqueeze(-1)
    n = w.sum(dim=1).clamp_min(1.0)
    mu_m = (mobile * w).sum(dim=1) / n
    mu_t = (tgt * w).sum(dim=1) / n
    m = (mobile - mu_m.unsqueeze(1)) * w
    t = (tgt - mu_t.unsqueeze(1)) * w
    cov = m.transpose(1, 2) @ t
    u, _, vh = torch.linalg.svd(cov)
    det = torch.det(u @ vh)
    diag = torch.ones(mobile.shape[0], 3, device=mobile.device, dtype=mobile.dtype)
    diag[:, -1] = torch.where(det < 0, det.new_tensor(-1.0), det.new_tensor(1.0))
    rot = u @ torch.diag_embed(diag) @ vh
    return rot, mu_m, mu_t

def masked_kabsch_align(
    mobile: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    rot, mu_m, mu_t = masked_kabsch_params(mobile, target, mask)
    return (mobile - mu_m.unsqueeze(1)) @ rot + mu_t.unsqueeze(1)

def required_span_aligned_cycle_loss(
    pred_ca: torch.Tensor,
    delivered_ca: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:

    refuse_outcome_labels(batch)
    designed = designed_span_mask(batch)
    required = required_span_mask(batch)
    if not bool(required.any()):
        return pred_ca.new_zeros(())
    n_des = designed.to(dtype=pred_ca.dtype).sum(dim=1)
    if not bool((n_des >= 3).all()):
        return pred_ca.new_zeros(())
    rot, mu_m, mu_t = masked_kabsch_params(pred_ca.detach(), delivered_ca, designed)
    aligned = (pred_ca - mu_m.unsqueeze(1)) @ rot + mu_t.unsqueeze(1)
    return recovered_sequence_ca_loss(aligned, delivered_ca, required)

def required_atom_aligned_cycle_loss(
    pred_ca: torch.Tensor,
    delivered_ca: torch.Tensor,
    coors_nm: torch.Tensor,
    atom_mask: torch.Tensor | None,
    batch: Mapping[str, Any],
) -> torch.Tensor:

    refuse_outcome_labels(batch)
    designed = designed_span_mask(batch)
    required = required_span_mask(batch)
    target = batch.get("coords_nm")
    if target is None or not torch.is_tensor(target):
        return pred_ca.new_zeros(())
    if not bool(required.any()):
        return pred_ca.new_zeros(())
    n_des = designed.to(dtype=pred_ca.dtype).sum(dim=1)
    if not bool((n_des >= 3).all()):
        return pred_ca.new_zeros(())
    if coors_nm.shape[:2] != required.shape or target.shape != coors_nm.shape:
        raise ValueError("required-atom coors/mask shape mismatch")
    rot, mu_m, mu_t = masked_kabsch_params(pred_ca.detach(), delivered_ca, designed)
    aligned = (coors_nm - mu_m[:, None, None, :]) @ rot + mu_t[:, None, None, :]
    atom_ok = required.unsqueeze(-1).expand_as(aligned[:, :, :, 0])
    if atom_mask is not None:
        atom_ok = atom_ok & atom_mask.to(dtype=torch.bool)
    tgt_ok = batch.get("coord_mask")
    if tgt_ok is not None and torch.is_tensor(tgt_ok) and tgt_ok.shape == atom_ok.shape:
        atom_ok = atom_ok & tgt_ok.to(dtype=torch.bool)
    if not bool(atom_ok.any()):
        return pred_ca.new_zeros(())
    return (aligned[atom_ok] - target[atom_ok].detach()).square().mean()

def first_pass_required_atom_loss(
    generated_ca: torch.Tensor,
    reference_ca: torch.Tensor,
    coors_nm: torch.Tensor,
    atom_mask: torch.Tensor | None,
    batch: Mapping[str, Any],
) -> torch.Tensor:

    refuse_outcome_labels(batch)
    designed = designed_span_mask(batch)
    required = required_span_mask(batch)
    target = batch.get("coords_nm")
    if target is None or not torch.is_tensor(target):
        return generated_ca.new_zeros(())
    if not bool(required.any()):
        return generated_ca.new_zeros(())
    n_des = designed.to(dtype=generated_ca.dtype).sum(dim=1)
    if not bool((n_des >= 3).all()):
        return generated_ca.new_zeros(())
    if coors_nm.shape[:2] != required.shape or target.shape != coors_nm.shape:
        raise ValueError("first-pass required-atom coors/target shape mismatch")
    rot, mu_m, mu_t = masked_kabsch_params(
        generated_ca.detach(), reference_ca.detach(), designed)
    aligned = (coors_nm - mu_m[:, None, None, :]) @ rot + mu_t[:, None, None, :]
    atom_ok = required.unsqueeze(-1).expand_as(aligned[:, :, :, 0])
    if atom_mask is not None:
        atom_ok = atom_ok & atom_mask.to(dtype=torch.bool)
    tgt_ok = batch.get("coord_mask")
    if tgt_ok is not None and torch.is_tensor(tgt_ok) and tgt_ok.shape == atom_ok.shape:
        atom_ok = atom_ok & tgt_ok.to(dtype=torch.bool)
    if not bool(atom_ok.any()):
        return generated_ca.new_zeros(())
    return (aligned[atom_ok] - target[atom_ok].detach()).square().mean()

_RESTYPE_ORDER = "ARNDCQEGHILKMFPSTWYV"

_HYDROPHOBIC = frozenset("AILMFWVC")
_HYDROPHOBIC_INDEX = tuple(i for i, a in enumerate(_RESTYPE_ORDER) if a in _HYDROPHOBIC)

def hydrophobic_burial_loss(
    ca: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    radius_nm: float = 1.0,
    temperature_nm: float = 0.15,
) -> torch.Tensor:

    refuse_outcome_labels(batch)
    mask = batch["mask"].bool()
    residue = batch["residue_type"]
    if ca.shape[:2] != mask.shape or residue.shape[:2] != mask.shape:
        raise ValueError("burial ca/mask/residue shape mismatch")
    if not bool(mask.any()):
        return ca.new_zeros(())
    phobic_ids = torch.tensor(_HYDROPHOBIC_INDEX, device=residue.device)
    is_phobic = (residue.unsqueeze(-1) == phobic_ids).any(dim=-1) & mask
    is_polar = (~is_phobic) & mask
    total = ca.new_zeros(())
    counted = 0
    for b in range(ca.shape[0]):
        sel = mask[b]
        if not bool(is_phobic[b].any()) or not bool(is_polar[b].any()):
            continue
        points = ca[b][sel]
        dist = torch.cdist(points, points)
        near = torch.sigmoid((radius_nm - dist) / temperature_nm)
        burial = near.sum(dim=-1) - torch.sigmoid(
            torch.tensor(radius_nm / temperature_nm, device=ca.device, dtype=ca.dtype))
        phobic_here = is_phobic[b][sel]
        total = total + burial[~phobic_here].mean() - burial[phobic_here].mean()
        counted += 1
    if counted == 0:
        return ca.new_zeros(())
    return total / counted

def permute_designed_span_logits(
    seq_logits: torch.Tensor,
    designed_mask: torch.Tensor,
) -> torch.Tensor:

    if seq_logits.shape[:2] != designed_mask.shape:
        raise ValueError("permute logits/mask shape mismatch")
    out = seq_logits.clone()
    for batch_i in range(seq_logits.shape[0]):
        idx = designed_mask[batch_i].nonzero(as_tuple=False).flatten()
        if int(idx.numel()) < 2:
            continue
        perm = idx[torch.randperm(idx.numel(), device=seq_logits.device)]
        out[batch_i, idx] = seq_logits[batch_i, perm].detach()
    return out

def sequence_dependence_hinge(
    pred_ca: torch.Tensor,
    control_ca: torch.Tensor,
    target_ca: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:

    err = recovered_sequence_ca_loss(pred_ca, target_ca, mask)
    err_c = recovered_sequence_ca_loss(control_ca, target_ca, mask).detach()
    return torch.relu(err - err_c + float(margin))

def designed_span_ca_bond_loss(
    ca: torch.Tensor,
    designed_mask: torch.Tensor,
    *,
    target_nm: float = 0.38,
) -> torch.Tensor:

    if ca.shape[:2] != designed_mask.shape or ca.shape[-1] != 3:
        raise ValueError("designed-span bond CA/mask shape mismatch")
    pair = designed_mask[:, 1:] & designed_mask[:, :-1]
    if not bool(pair.any()):
        return ca.new_zeros(())
    dist = (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)
    return (dist[pair] - ca.new_tensor(float(target_nm))).square().mean()

def designed_span_peptide_bond_loss(
    coors_nm: torch.Tensor,
    atom_mask: torch.Tensor | None,
    designed_mask: torch.Tensor,
) -> torch.Tensor:

    if coors_nm.shape[:2] != designed_mask.shape:
        raise ValueError("designed-span peptide coors/mask mismatch")
    pair = designed_mask[:, 1:] & designed_mask[:, :-1]
    if not bool(pair.any()):
        return coors_nm.new_zeros(())
    n_next = coors_nm[:, 1:, 0]
    c_prev = coors_nm[:, :-1, 2]
    ca_next = coors_nm[:, 1:, 1]
    ca_prev = coors_nm[:, :-1, 1]
    if atom_mask is not None:
        pair = (
            pair & atom_mask[:, :-1, 2] & atom_mask[:, 1:, 0]
            & atom_mask[:, :-1, 1] & atom_mask[:, 1:, 1]
        )
    if not bool(pair.any()):
        return coors_nm.new_zeros(())
    cn = (c_prev - n_next).norm(dim=-1)
    caca = (ca_prev - ca_next).norm(dim=-1)
    t_cn = coors_nm.new_tensor(0.133)
    t_ca = coors_nm.new_tensor(0.38)
    return (cn[pair] - t_cn).square().mean() + (caca[pair] - t_ca).square().mean()

def designed_span_g_bond_loss(
    coors_nm: torch.Tensor,
    atom_mask: torch.Tensor | None,
    designed_mask: torch.Tensor,
) -> torch.Tensor:

    if coors_nm.shape[:2] != designed_mask.shape:
        raise ValueError("designed-span G-bond coors/mask mismatch")
    pair = designed_mask[:, 1:] & designed_mask[:, :-1]
    if not bool(pair.any()):
        return coors_nm.new_zeros(())
    n_next = coors_nm[:, 1:, 0]
    c_prev = coors_nm[:, :-1, 2]
    ca_next = coors_nm[:, 1:, 1]
    ca_prev = coors_nm[:, :-1, 1]
    if atom_mask is not None:
        pair = (
            pair & atom_mask[:, :-1, 2] & atom_mask[:, 1:, 0]
            & atom_mask[:, :-1, 1] & atom_mask[:, 1:, 1]
        )
    if not bool(pair.any()):
        return coors_nm.new_zeros(())
    cn = (c_prev - n_next).norm(dim=-1)
    caca = (ca_prev - ca_next).norm(dim=-1)

    cn_lo, cn_hi = coors_nm.new_tensor(0.12), coors_nm.new_tensor(0.145)
    ca_lo, ca_hi = coors_nm.new_tensor(0.35), coors_nm.new_tensor(0.42)
    cn_h = torch.relu(cn_lo - cn[pair]).square() + torch.relu(cn[pair] - cn_hi).square()
    ca_h = torch.relu(ca_lo - caca[pair]).square() + torch.relu(caca[pair] - ca_hi).square()
    return cn_h.mean() + ca_h.mean()

def recovered_sequence_ca_loss(
    pred_ca: torch.Tensor, target_ca: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:

    if pred_ca.shape[:2] != mask.shape:
        raise ValueError("recovery CA/mask shape mismatch")
    if not mask.any():
        raise ValueError("empty recovery mask")
    return (pred_ca[mask] - target_ca[mask].detach()).square().mean()

def marginalized_mean_prediction(
    draws: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    grad_draws: int | None = None,
) -> torch.Tensor:

    items = list(draws)
    if not items:
        raise ValueError("marginalized mean needs at least one draw")
    for value in items:
        if not torch.is_tensor(value):
            raise ValueError("marginalized mean takes tensors")
        if value.shape != items[0].shape:
            raise ValueError("marginalized draw shape mismatch")
    total = len(items)
    grad_n = total if grad_draws is None else int(grad_draws)
    if grad_n < 1 or grad_n > total:
        raise ValueError("grad_draws must be in 1..len(draws)")
    if total == 1 and grad_n == 1:
        return items[0]
    mean = torch.stack([value.detach() for value in items], dim=0).mean(dim=0)
    carried = items[:grad_n]
    surrogate = torch.stack(
        [value - value.detach() for value in carried], dim=0
    ).mean(dim=0)
    return mean + surrogate

def marginalized_second_pass_ca(
    build_second: Any,
    run_pass: Any,
    *,
    draws: int = 1,
    grad_draws: int = 1,
) -> torch.Tensor:

    total = int(draws)
    grad_n = int(grad_draws)
    if total < 1:
        raise ValueError("marginal cycle needs at least one draw")
    if grad_n < 1 or grad_n > total:
        raise ValueError("grad_draws must be in 1..draws")
    predictions = []
    for index in range(total):
        if index < grad_n:
            predictions.append(run_pass(build_second()))
        else:
            with torch.no_grad():
                predictions.append(run_pass(build_second()))
    return marginalized_mean_prediction(predictions, grad_draws=grad_n)

def designed_span_entropy(
    seq_logits: torch.Tensor,
    designed_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:

    if seq_logits.shape[:2] != designed_mask.shape:
        raise ValueError("entropy logits/mask shape mismatch")
    if not bool(designed_mask.any()):
        return seq_logits.new_zeros(())
    scale = float(temperature) if float(temperature) > 0 else 1.0
    probs = F.softmax(seq_logits / scale, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy[designed_mask].mean()

def designed_span_logit_gap(
    seq_logits: torch.Tensor,
    designed_mask: torch.Tensor,
) -> torch.Tensor:

    if seq_logits.shape[:2] != designed_mask.shape:
        raise ValueError("logit-gap logits/mask shape mismatch")
    if not bool(designed_mask.any()):
        return seq_logits.new_zeros(())
    top2 = seq_logits[designed_mask].topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).mean()

def recovered_interface_ca_loss(
    pred_ca: torch.Tensor,
    delivered_ca: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    thresh_nm: float = 0.8,
) -> torch.Tensor:

    refuse_outcome_labels(batch)
    x_target = batch.get("x_target")
    tmask = batch.get("target_mask")
    mask = batch["mask"]
    if x_target is None or not torch.is_tensor(x_target):
        return pred_ca.new_zeros(())
    if x_target.dim() == 4:
        tca = x_target[:, :, 1, :]
        t_ok = tmask.any(-1) if tmask is not None else torch.ones(
            tca.shape[:2], dtype=torch.bool, device=tca.device)
    elif x_target.dim() == 3:
        tca = x_target
        t_ok = tmask if tmask is not None else torch.ones(
            tca.shape[:2], dtype=torch.bool, device=tca.device)
    else:
        raise ValueError("x_target rank must be 3 or 4")
    delivered_d = torch.cdist(delivered_ca.detach(), tca.detach())
    pred_d = torch.cdist(pred_ca, tca.detach())
    contacts = (delivered_d < thresh_nm) & mask[:, :, None] & t_ok[:, None, :]
    if not contacts.any():
        return pred_ca.new_zeros(())
    return (pred_d[contacts] - delivered_d[contacts]).square().mean()

def recovery_through_second_pass(
    seq_logits: torch.Tensor,
    mask: torch.Tensor,
    target_ca: torch.Tensor,
    second_pass: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:

    hard = generation_hard_sequence(seq_logits)
    ste = straight_through_residue_onehot(seq_logits, mask)
    pred_ca = second_pass(hard, ste)
    return recovered_sequence_ca_loss(pred_ca, target_ca, mask)

def invert_designed_span_logits(
    seq_logits: torch.Tensor,
    delivered_ca: torch.Tensor,
    designed_mask: torch.Tensor,
    predict_ca: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int = 8,
    lr: float = 2.0,
    flatten_designed: bool = False,
    extra_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    init_temperature: float = 1.0,
) -> torch.Tensor:

    refuse_outcome_labels({"seq_logits": seq_logits})
    if delivered_ca.shape[:2] != designed_mask.shape:
        raise ValueError("delivered CA/mask shape mismatch")
    if seq_logits.shape[:2] != designed_mask.shape:
        raise ValueError("sequence logits/mask shape mismatch")
    if not bool(designed_mask.any()):
        return generation_hard_sequence(seq_logits.detach())
    logits = seq_logits.detach().clone()
    if flatten_designed:

        logits = logits.masked_fill(designed_mask[..., None], 0.0)
    elif float(init_temperature) != 1.0:
        scale = float(init_temperature) if float(init_temperature) > 0 else 1.0
        logits = torch.where(designed_mask[..., None], logits / scale, logits)
    logits = logits.requires_grad_(True)
    opt = torch.optim.SGD([logits], lr=float(lr))
    target = delivered_ca.detach()
    gate = designed_mask[..., None].to(dtype=logits.dtype)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pred = predict_ca(logits)
        loss = recovered_sequence_ca_loss(pred, target, designed_mask)
        if extra_loss is not None:
            loss = loss + extra_loss(logits, pred)

        if not loss.requires_grad:
            break
        loss.backward()
        if logits.grad is None:
            break
        logits.grad = logits.grad * gate
        opt.step()
    return generation_hard_sequence(logits.detach())

def replace_designed_span_residue_type(
    residue_type: torch.Tensor,
    recovered: torch.Tensor,
    designed_mask: torch.Tensor,
) -> torch.Tensor:

    if residue_type.shape != recovered.shape or residue_type.shape != designed_mask.shape:
        raise ValueError("residue_type/recovered/mask shape mismatch")
    return torch.where(designed_mask, recovered.to(dtype=residue_type.dtype), residue_type)

def predict_ca_from_sequence_logits(
    model: Any,
    inputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    seq_logits: torch.Tensor,
    *,
    xt_noise: Mapping[str, Any] | None = None,
    residue_feature: str = "ste",
    temperature: float = 1.0,
    native_residue_type: bool = False,
    sequence_latents: bool = False,
    inferred_latents: torch.Tensor | None = None,
) -> torch.Tensor:

    refuse_outcome_labels(inputs)
    refuse_outcome_labels(batch)
    second = apply_recovered_sequence_inputs(
        inputs, batch, seq_logits, xt_noise=xt_noise,
        residue_feature=residue_feature, temperature=temperature,
        sequence_latents=sequence_latents, inferred_latents=inferred_latents,
    )
    out2 = (
        model.call_nn(second)
        if native_residue_type
        else call_nn_with_residue_onehot(model, second)
    )
    clean2 = model.fm.nn_out_to_clean_sample_prediction(second, out2)
    return clean2["bb_ca"]

def second_pass_inputs_from_sample(batch: Mapping[str, Any], sample: Mapping[str, Any]) -> dict:

    refuse_outcome_labels(batch)
    refuse_outcome_labels(sample)
    delivered = sample["bb_ca"]
    b = delivered.shape[0]
    device = delivered.device
    inputs = dict(batch)
    inputs["x_t"] = {
        "bb_ca": delivered.detach(),
        "local_latents": sample["local_latents"].detach(),
    }
    inputs["t"] = {
        "bb_ca": torch.ones(b, device=device, dtype=delivered.dtype),
        "local_latents": torch.ones(b, device=device, dtype=sample["local_latents"].dtype),
    }
    inputs["mask"] = batch["mask"]
    return inputs

def recover_generated_designed_span(
    model: Any,
    batch: Mapping[str, Any],
    sample: Mapping[str, Any],
    decoded: Mapping[str, Any],
    *,
    steps: int = 20,
    lr: float = 4.0,
    invert_model: Any | None = None,
    residue_feature: str = "ste",
    feature_temperature: float = 1.0,
    entropy_weight: float = 0.0,
    interface_weight: float = 0.0,
    init_temperature: float = 1.0,
    mix_t: float = 1.0,
    keep_latents: bool = False,
    native_residue_type: bool = False,
    sequence_latents: bool = False,
    last_xt: Mapping[str, Any] | None = None,
    last_t: Mapping[str, Any] | Any = None,
) -> dict:

    refuse_outcome_labels(batch)
    refuse_outcome_labels(decoded)
    delivered = sample["bb_ca"]
    before = delivered.detach().clone()
    designed = designed_span_mask(batch)
    inputs = second_pass_inputs_from_sample(batch, sample)
    decoder = invert_model if invert_model is not None else model
    if last_xt is not None:
        if last_t is None:
            raise ValueError("last_xt requires last_t")
        frozen_xt = _clone_state_map(last_xt)
        frozen_t = _clone_state_map(last_t)

        def predict_ca(seq_logits: torch.Tensor) -> torch.Tensor:
            featured = attach_residue_ste_features(
                {**inputs, "x_t": frozen_xt, "t": frozen_t},
                seq_logits,
                batch["mask"],
                residue_feature=residue_feature,
                temperature=feature_temperature,
            )
            if not torch.equal(featured["x_t"]["bb_ca"], frozen_xt["bb_ca"]):
                raise RuntimeError("designed-span CA was rewritten")
            out2 = (
                decoder.call_nn(featured)
                if native_residue_type
                else call_nn_with_residue_onehot(decoder, featured)
            )
            return decoder.fm.nn_out_to_clean_sample_prediction(featured, out2)["bb_ca"]
    else:
        clean_xt = {
            "bb_ca": delivered.detach(),
            "local_latents": sample["local_latents"].detach(),
        }
        frozen = apply_recovered_sequence_inputs(
            inputs, batch, decoded["seq_logits"],
            residue_feature=residue_feature, temperature=feature_temperature,
            mix_t=float(mix_t), clean_xt=clean_xt, keep_latents=keep_latents,
        )["x_t"]

        def predict_ca(seq_logits: torch.Tensor) -> torch.Tensor:
            inferred = None
            if sequence_latents:
                inferred = infer_latents_from_sequence(
                    decoder, inputs, batch, seq_logits,
                    residue_feature=residue_feature, temperature=feature_temperature,
                    native_residue_type=native_residue_type,
                )
            return predict_ca_from_sequence_logits(
                decoder, inputs, batch, seq_logits, xt_noise=frozen,
                residue_feature=residue_feature, temperature=feature_temperature,
                native_residue_type=native_residue_type,
                sequence_latents=sequence_latents, inferred_latents=inferred,
            )

    def extra_loss(seq_logits: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        extra = pred.new_zeros(())
        if float(entropy_weight) != 0.0:
            extra = extra - float(entropy_weight) * designed_span_entropy(
                seq_logits, designed
            )
        if float(interface_weight) != 0.0:
            extra = extra + float(interface_weight) * recovered_interface_ca_loss(
                pred, delivered, batch
            )
        return extra

    recovered = invert_designed_span_logits(
        decoded["seq_logits"], delivered, designed, predict_ca,
        steps=steps, lr=lr, flatten_designed=False,
        extra_loss=extra_loss if (float(entropy_weight) or float(interface_weight)) else None,
        init_temperature=init_temperature,
    )
    out = dict(decoded)
    out["residue_type"] = replace_designed_span_residue_type(
        decoded["residue_type"], recovered, designed
    )
    if not torch.equal(sample["bb_ca"], before):
        raise RuntimeError("designed-span CA was rewritten")
    return out

@contextmanager
def _inject_residue_onehot(model: Any, onehot: torch.Tensor | None):
    patched: list[tuple[Any, Any]] = []
    if onehot is None or not hasattr(model, "modules"):
        yield
        return

    def _forward(original):
        def forward(batch, *args, **kwargs):

            mask = (batch.get("mask_dict") or {}).get("residue_type", batch.get("mask"))
            if mask is None or tuple(onehot.shape[:2]) != tuple(mask.shape[:2]):
                return original(batch, *args, **kwargs)
            return onehot * mask[..., None].to(dtype=onehot.dtype)

        return forward

    for module in model.modules():
        if module.__class__.__name__ not in _RESIDUE_FEAT_TYPES:
            continue
        original = module.forward
        module.forward = _forward(original)
        patched.append((module, original))
    try:
        yield
    finally:
        for module, original in patched:
            module.forward = original

def call_nn_with_residue_onehot(model: Any, inputs: Mapping[str, Any]):

    with _inject_residue_onehot(model, inputs.get("residue_type_onehot")):
        return model.call_nn(inputs)
