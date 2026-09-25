
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dive.stackelberg.learned_router import MODALITIES, N_AA

def native_velocity(x_t: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

    clock = t
    if isinstance(clock, dict):
        clock = clock.get("bb_ca", next(iter(clock.values())))
    clock = clock.to(device=x_t.device, dtype=x_t.dtype)
    while clock.dim() < x_t.dim():
        clock = clock.view(*clock.shape, *([1] * (x_t.dim() - clock.dim())))
    return (x_1 - x_t) / (1.0 - clock).clamp_min(1e-6)

def extra_pass_wins(v_jacobi: torch.Tensor, v_extra: torch.Tensor,
                    v_native: torch.Tensor) -> torch.Tensor:

    err_j = (v_jacobi - v_native).norm(dim=-1)
    err_e = (v_extra - v_native).norm(dim=-1)
    return (err_e < err_j).to(dtype=v_jacobi.dtype)

class ExtraPassGate(nn.Module):

    def __init__(self, *, d_latent: int = 8, d_hidden: int = 64, n_aa: int = N_AA):
        super().__init__()
        self.d_latent = int(d_latent)
        self.n_aa = int(n_aa)

        self.mlp = nn.Sequential(
            nn.Linear(3 + self.n_aa + self.d_latent, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

    def _pack(self, dist_nm, residue_type, local_latents, t, delta_nm):
        b, n = dist_nm.shape
        oh = F.one_hot(residue_type.clamp(0, self.n_aa - 1).long(),
                       num_classes=self.n_aa).to(dtype=dist_nm.dtype)
        lat = local_latents
        if lat.shape[-1] != self.d_latent:
            if lat.shape[-1] > self.d_latent:
                lat = lat[..., :self.d_latent]
            else:
                lat = F.pad(lat, (0, self.d_latent - lat.shape[-1]))
        clock = t
        if isinstance(clock, dict):
            clock = clock.get("bb_ca", next(iter(clock.values())))
        clock = clock.to(device=dist_nm.device, dtype=dist_nm.dtype)
        while clock.dim() < 2:
            clock = clock.view(*clock.shape, *([1] * (2 - clock.dim())))
        if clock.shape != (b, n):
            clock = clock.reshape(b, -1)[:, :1].expand(b, n)
        return torch.cat(
            [dist_nm[..., None], delta_nm[..., None], clock[..., None], oh, lat],
            dim=-1)

    def forward(self, dist_nm, residue_type, local_latents, designed, t, delta_nm):
        x = self._pack(dist_nm, residue_type, local_latents, t, delta_nm)
        raw = self.mlp(x)
        logit = raw[..., 0]
        des = designed.to(dtype=dist_nm.dtype)
        mix = torch.sigmoid(logit) * des
        return {"logit": logit, "bb_ca": mix, "local_latents": mix}

    def as_gate(self, *, partner_coords: torch.Tensor, designed: torch.Tensor):

        net = self

        def gate(batch, jacobi_out, extra_v):
            ca = batch["x_t"]["bb_ca"]
            lat = batch["x_t"]["local_latents"]
            part = partner_coords.to(device=ca.device, dtype=ca.dtype)
            dist = torch.cdist(ca, part).min(dim=-1).values
            if "residue_type" in batch:
                restype = batch["residue_type"]
                if restype.dim() == 3:
                    restype = restype.argmax(dim=-1)
            else:
                restype = torch.zeros(ca.shape[:2], dtype=torch.long, device=ca.device)
            des = designed.to(device=ca.device)
            t = batch.get("t")
            if t is None:
                raise ValueError("extra-pass gate needs batch t")
            delta = {}
            for m in MODALITIES:
                if m in jacobi_out and m in extra_v and "v" in jacobi_out[m]:
                    delta[m] = (extra_v[m] - jacobi_out[m]["v"]).norm(dim=-1)
                else:
                    delta[m] = torch.zeros_like(dist)
            out = net.forward(dist, restype, lat, des, t, delta["bb_ca"])
            return {m: out[m] for m in MODALITIES}

        return gate

def load_extra_pass_gate(path) -> ExtraPassGate:
    from pathlib import Path
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("extra-pass gate checkpoint must be a dict with state_dict")
    cfg = payload.get("config") or {}
    kw = {k: cfg[k] for k in ("d_latent", "d_hidden", "n_aa") if k in cfg}
    net = ExtraPassGate(**kw)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net

def extra_pass_gate_loss(net: ExtraPassGate, payload: dict) -> dict:
    from dive.training.sequence_recovery import refuse_outcome_labels
    refuse_outcome_labels(payload)
    banned = ("iptm", "ipae", "plddt", "ptm")
    for key in payload:
        if any(tok in str(key).lower() for tok in banned):
            raise ValueError(
                "predictor scores are not training labels for the extra-pass gate")
    device = next(net.parameters()).device
    def _to(x):
        return x.to(device) if torch.is_tensor(x) else x
    payload = {k: _to(v) for k, v in payload.items()}
    if "dist_nm" not in payload and "partner" in payload:
        payload["dist_nm"] = torch.cdist(
            payload["x_t"], payload["partner"]).min(dim=-1).values
    v_nat = native_velocity(payload["x_t"], payload["x_1"], payload["t"])
    win = extra_pass_wins(payload["v_jacobi"], payload["v_extra"], v_nat)
    delta = (payload["v_extra"] - payload["v_jacobi"]).norm(dim=-1)
    pred = net.forward(
        payload["dist_nm"], payload["residue_type"], payload["local_latents"],
        payload["designed"], payload["t"], delta)
    des = payload["designed"].to(dtype=win.dtype)
    bce = F.binary_cross_entropy_with_logits(
        pred["logit"], win, weight=des)
    return {"loss": bce, "bce": bce.detach(), "bb_ca": pred["bb_ca"],
            "local_latents": pred["local_latents"], "win": win}
