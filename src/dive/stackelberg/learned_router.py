
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("bb_ca", "local_latents")

NATIVE_CONTACT_NM = 0.5
N_AA = 20
ATOM37_CA = 1

ATOM37_BACKBONE = (0, 1, 2, 4)

def atom37_from_batch(batch: dict) -> torch.Tensor | None:

    for key in ("coords_nm", "coors_nm", "atom37"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() == 4 and value.shape[-2] == 37:
            return value
    x1 = batch.get("x_1")
    if isinstance(x1, dict):
        for key in ("coords_nm", "coors_nm", "atom37"):
            value = x1.get(key)
            if torch.is_tensor(value) and value.dim() == 4 and value.shape[-2] == 37:
                return value
    return None

def native_contact_labels(bb_ca: torch.Tensor, partner_coords: torch.Tensor,
                          *, atom37: torch.Tensor | None = None,
                          contact_nm: float = NATIVE_CONTACT_NM) -> dict:

    if bb_ca.dim() != 3 or partner_coords.dim() != 3:
        raise ValueError("bb_ca and partner_coords must be [b, n, 3] and [b, k, 3]")
    ca_d = torch.cdist(bb_ca, partner_coords).min(dim=-1).values
    iface = ca_d <= float(contact_nm)
    sc_iface = torch.zeros_like(iface)
    has_sidechain = torch.zeros_like(iface)
    sc_min = ca_d.clone()
    if atom37 is not None:
        b, n, a, _ = atom37.shape
        xyz = atom37.reshape(b, n * a, 3)
        d = torch.cdist(xyz, partner_coords).reshape(b, n, a, -1).min(dim=-1).values
        sidechain = atom37.abs().sum(-1) > 0
        for slot in ATOM37_BACKBONE:
            sidechain[..., slot] = False
        sc_d = d.masked_fill(~sidechain, 1e6)
        sc_min = sc_d.min(dim=-1).values
        has_sidechain = sidechain.any(dim=-1)
        sc_iface = (sc_min <= float(contact_nm)) & has_sidechain
    return {"iface": iface, "sidechain_iface": sc_iface,
            "ca_dist": ca_d, "sc_dist": sc_min, "has_sidechain": has_sidechain}

class LearnedLeadershipRouter(nn.Module):

    def __init__(self, *, d_latent: int = 8, d_hidden: int = 64, n_aa: int = N_AA,
                 use_sc_dist: bool = False, use_time: bool = False,
                 use_trunk_condition: bool = False,
                 use_residue_type: bool = True):
        super().__init__()
        self.d_latent = int(d_latent)
        self.n_aa = int(n_aa)
        self.use_residue_type = bool(use_residue_type)
        self.aa_dim = self.n_aa if self.use_residue_type else 0
        self.use_sc_dist = bool(use_sc_dist)
        self.use_time = bool(use_time)
        self.use_trunk_condition = bool(use_trunk_condition)
        self.cond_dim = 0
        if self.use_trunk_condition:
            from dive.stackelberg.trunk_condition import trunk_condition_dim
            self.cond_dim = int(trunk_condition_dim())
        n_dist = 2 if self.use_sc_dist else 1
        n_time = 1 if self.use_time else 0
        self.mlp = nn.Sequential(
            nn.Linear(n_dist + n_time + self.aa_dim + self.d_latent + self.cond_dim,
                      d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 3),
        )

    def _pack(self, dist_nm, residue_type, local_latents, sc_dist=None, t=None,
              cond=None):
        b, n = dist_nm.shape
        lat = local_latents
        if lat.shape[-1] != self.d_latent:
            if lat.shape[-1] > self.d_latent:
                lat = lat[..., :self.d_latent]
            else:
                lat = F.pad(lat, (0, self.d_latent - lat.shape[-1]))
        parts = [dist_nm[..., None]]
        if self.use_sc_dist:
            extra = sc_dist if sc_dist is not None else dist_nm
            parts.append(extra[..., None])
        if self.use_time:
            if t is None:
                raise ValueError(
                    "use_time router needs t; dropping it would make the extra "
                    "input a constant and leadership would not depend on noise")
            time = t
            if isinstance(time, dict):
                time = time.get("bb_ca", next(iter(time.values())))
            time = time.to(device=dist_nm.device, dtype=dist_nm.dtype)
            while time.dim() < 2:
                time = time.view(*time.shape, *([1] * (2 - time.dim())))
            if time.shape != (b, n):
                time = time.reshape(b, -1)[:, :1].expand(b, n)
            parts.append(time[..., None])
        if self.use_residue_type:
            oh = F.one_hot(residue_type.clamp(0, self.n_aa - 1).long(),
                           num_classes=self.n_aa).to(dtype=dist_nm.dtype)
            parts.append(oh)
        parts.append(lat)
        if self.use_trunk_condition:
            if cond is None:
                raise ValueError(
                    "use_trunk_condition router needs the trunk condition "
                    "features; dropping them would train leadership that does "
                    "not see the motif, ligand, or target the trunk sees")
            extra = cond.to(device=dist_nm.device, dtype=dist_nm.dtype)
            if extra.dim() == 2:
                extra = extra[:, None, :].expand(b, n, -1)
            elif extra.dim() == 3 and extra.shape[1] == 1 and n != 1:
                extra = extra.expand(b, n, -1)
            elif extra.dim() != 3 or extra.shape[1] != n:
                raise ValueError(
                    f"trunk condition has shape {tuple(cond.shape)}; expected "
                    f"[b, {self.cond_dim}] or [b, {n}, {self.cond_dim}]")
            if extra.shape[-1] != self.cond_dim:
                raise ValueError(
                    f"trunk condition width {extra.shape[-1]} != {self.cond_dim}")
            parts.append(extra)
        return torch.cat(parts, dim=-1)

    def forward(self, dist_nm, residue_type, local_latents, designed,
                sc_dist=None, t=None, cond=None):
        x = self._pack(dist_nm, residue_type, local_latents, sc_dist=sc_dist,
                       t=t, cond=cond)
        raw = self.mlp(x)
        logit = raw[..., 0]
        w_ca = F.softplus(raw[..., 1])
        w_lat = F.softplus(raw[..., 2])
        des = designed.to(dtype=w_ca.dtype)
        return {"logit_iface": logit, "bb_ca": w_ca * des,
                "local_latents": w_lat * des}

    def weights(self, dist_nm, residue_type, local_latents, designed, cond=None):
        out = self.forward(dist_nm, residue_type, local_latents, designed,
                           cond=cond)
        return {m: out[m] for m in MODALITIES}

    def as_router(self, *, partner_coords: torch.Tensor, designed: torch.Tensor,
                  atom37: torch.Tensor | None = None, split_roles: bool = False):

        net = self
        held_atom37 = atom37
        use_roles = bool(split_roles)

        def router(batch, output):
            ca = batch["x_t"]["bb_ca"]
            lat = batch["x_t"]["local_latents"]
            part = partner_coords.to(device=ca.device, dtype=ca.dtype)
            dist = torch.cdist(ca, part).min(dim=-1).values
            if net.use_residue_type and "residue_type" in batch:
                restype = batch["residue_type"]
                if restype.dim() == 3:
                    restype = restype.argmax(dim=-1)
            else:
                restype = torch.zeros(ca.shape[:2], dtype=torch.long, device=ca.device)
            des = designed.to(device=ca.device)
            sc = None
            if net.use_sc_dist:
                atom = atom37_from_batch(batch)
                if atom is None:
                    atom = held_atom37
                if atom is None:
                    raise ValueError(
                        "use_sc_dist router needs batch coors_nm/atom37; "
                        "copying CA dist would make the extra input a no-op")
                sc = native_contact_labels(
                    ca, part, atom37=atom.to(device=ca.device, dtype=ca.dtype)
                )["sc_dist"]
            t = None
            if net.use_time:
                t = batch.get("t")
                if t is None:
                    raise ValueError(
                        "use_time router needs batch t; a missing clock would "
                        "make leadership independent of remaining noise")
            extra = {}
            if net.use_trunk_condition:
                from dive.stackelberg.trunk_condition import trunk_condition_features
                extra["cond"] = trunk_condition_features(batch)
            out = net.forward(dist, restype, lat, des, sc_dist=sc, t=t, **extra)
            if use_roles:
                return cross_modal_role_weights(out["logit_iface"], des)
            return {m: out[m] for m in MODALITIES}

        return router

def cross_modal_role_weights(logit_iface: torch.Tensor, designed: torch.Tensor) -> dict:

    p = torch.sigmoid(logit_iface)
    des = designed.to(device=p.device, dtype=p.dtype)
    return {"bb_ca": (1.0 - p) * des, "local_latents": p * des}

def load_learned_router(path) -> LearnedLeadershipRouter:

    from pathlib import Path
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            "learned-router checkpoint must be a dict with 'state_dict'; "
            "a raw state_dict cannot carry the architecture knobs")
    cfg = payload.get("config") or {}
    kw = {k: cfg[k] for k in ("d_latent", "d_hidden", "n_aa", "use_sc_dist",
                              "use_time", "use_trunk_condition",
                              "use_residue_type") if k in cfg}
    net = LearnedLeadershipRouter(**kw)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net

def make_generation_router(args, batch):

    from dive.stackelberg.interface_router import (
        interface_router, partner_atoms_from_batch)
    learned = getattr(args, "learned_router", None)
    hand = bool(getattr(args, "interface_router", False))
    if learned and hand:
        raise ValueError(
            "pass --learned-router or --interface-router, not both; mixing a "
            "fitted net with hand iface-alpha would not be the labelled arm")
    if not learned and not hand:
        return None
    partner = partner_atoms_from_batch(batch)
    designed = batch.get("generated_mask", batch["mask"]).bool()
    if learned:
        net = load_learned_router(learned)
        net.to(device=batch["mask"].device if torch.is_tensor(batch["mask"]) else "cpu")
        return net.as_router(partner_coords=partner, designed=designed,
                             atom37=atom37_from_batch(batch),
                             split_roles=bool(getattr(args, "role_router", False)))
    return interface_router(
        partner_coords=partner, designed=designed,
        cutoff_nm=args.iface_cutoff,
        interface={"bb_ca": args.iface_alpha_ca,
                   "local_latents": args.iface_alpha_lat},
        bulk={"bb_ca": args.bulk_alpha_ca,
              "local_latents": args.bulk_alpha_lat})

def native_from_train_batch(batch: dict, *, trunk_condition: bool = False) -> dict:

    from dive.training.sequence_recovery import refuse_outcome_labels
    refuse_outcome_labels(batch)
    banned = ("iptm", "ipae", "plddt", "ptm")
    for key in batch:
        low = str(key).lower()
        if any(tok in low for tok in banned):
            raise ValueError("predictor scores are not training labels for the router")
    from dive.anticipate.capture import designed_from_batch
    from dive.stackelberg.interface_fm import partner_coords_padded

    lat = None
    if "x_1" in batch and isinstance(batch["x_1"], dict) and "bb_ca" in batch["x_1"]:
        ca = batch["x_1"]["bb_ca"]
        lat = batch["x_1"].get("local_latents")
    elif "bb_ca" in batch:
        ca = batch["bb_ca"]
        lat = batch.get("local_latents")
    elif "ca_coors_nm" in batch:
        ca = batch["ca_coors_nm"]
        lat = batch.get("local_latents")
    elif "coords_nm" in batch and batch["coords_nm"].dim() == 4:
        ca = batch["coords_nm"][:, :, ATOM37_CA, :]
        lat = batch.get("local_latents")
    elif "coors_nm" in batch and batch["coors_nm"].dim() == 4:
        ca = batch["coors_nm"][:, :, ATOM37_CA, :]
        lat = batch.get("local_latents")
    else:
        raise ValueError("train batch has no native bb_ca")
    partner, keep = partner_coords_padded(batch)
    far = partner.clone()
    far = far.masked_fill(~keep[..., None], 1.0e3)
    restype = batch.get("residue_type")
    if restype is None:
        restype = torch.zeros(ca.shape[:2], dtype=torch.long, device=ca.device)
    elif restype.dim() == 3:
        restype = restype.argmax(dim=-1)
    designed = designed_from_batch(batch)
    if lat is None:
        lat = torch.zeros(*ca.shape[:2], 8, dtype=ca.dtype, device=ca.device)
    packed = {
        "bb_ca": ca,
        "partner": far,
        "residue_type": restype,
        "designed": designed,
        "local_latents": lat,
        "atom37": atom37_from_batch(batch),
        "t": batch.get("t"),
        "mask": batch["mask"].bool(),
    }
    if trunk_condition:
        from dive.stackelberg.trunk_condition import trunk_condition_features
        packed["trunk_condition"] = trunk_condition_features(batch)
    return packed

def contact_target(labels: dict, label: str) -> torch.Tensor:

    if label == "ca":
        return labels["iface"]
    if label == "sidechain":
        return labels["sidechain_iface"]
    if label == "union":
        return labels["iface"] | labels["sidechain_iface"]
    raise ValueError(f"unknown contact label {label!r}; use ca, sidechain, or union")

def _rows_like(tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    rows = tensor.bool()
    while rows.dim() > like.dim():
        rows = rows.any(dim=-1)
    if rows.shape != like.shape:
        raise ValueError(
            f"row mask shape {tuple(rows.shape)} != {tuple(like.shape)}")
    return rows

def _supervision_rows(native: dict, like: torch.Tensor) -> torch.Tensor:

    designed = _rows_like(native["designed"], like)
    mask = native.get("mask")
    if mask is None:
        return designed
    return designed & _rows_like(mask, like)

def _masked_mean(values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    keep = rows.to(dtype=values.dtype)
    return (values * keep).sum() / keep.sum().clamp_min(1.0)

def native_router_loss(net: LearnedLeadershipRouter, native: dict, *,
                       extra_payload: dict | None = None,
                       label: str = "ca") -> dict:

    from dive.training.sequence_recovery import refuse_outcome_labels
    refuse_outcome_labels(extra_payload)
    refuse_outcome_labels(native)
    device = next(net.parameters()).device
    native = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in native.items()}
    banned = ("iptm", "ipae", "plddt", "ptm")
    for payload in (extra_payload, native):
        if not payload:
            continue
        for key in payload:
            low = str(key).lower()
            if any(tok in low for tok in banned):
                raise ValueError(
                    "predictor scores are not training labels for the router")
    labels = native_contact_labels(native["bb_ca"], native["partner"],
                                   atom37=native.get("atom37"))
    dist = labels["ca_dist"]
    lat = native.get("local_latents")
    if lat is None:
        lat = torch.zeros(*dist.shape, net.d_latent, dtype=dist.dtype,
                          device=dist.device)
    t = native.get("t")
    if net.use_time and t is None:
        t = torch.rand(dist.shape[0], 1, device=dist.device, dtype=dist.dtype)
    cond = None
    if net.use_trunk_condition:
        cond = native.get("trunk_condition")
        if cond is None:
            raise ValueError(
                "use_trunk_condition router loss needs native['trunk_condition']; "
                "native_from_train_batch(..., trunk_condition=True) builds it "
                "from the same batch the trunk reads")
    pred = net(dist, native["residue_type"], lat, native["designed"],
               sc_dist=(labels["sc_dist"] if net.use_sc_dist else None),
               t=t, cond=cond)
    target = contact_target(labels, label).to(dtype=pred["logit_iface"].dtype)
    eligible = _supervision_rows(native, pred["logit_iface"])
    bce_map = F.binary_cross_entropy_with_logits(
        pred["logit_iface"], target, reduction="none")
    bce = _masked_mean(bce_map, eligible)

    gap = (labels["ca_dist"] - labels["sc_dist"]).clamp(-1.0, 1.0)
    rel = pred["local_latents"] - pred["bb_ca"]
    measurable = labels["has_sidechain"].to(device=eligible.device) & eligible
    rel_loss = _masked_mean((rel - gap * 4.0).pow(2), measurable)
    loss = bce + rel_loss
    return {"loss": loss, "bce": bce.detach(), "rel": rel_loss.detach(),
            "logit_iface": pred["logit_iface"], "bb_ca": pred["bb_ca"],
            "local_latents": pred["local_latents"]}
