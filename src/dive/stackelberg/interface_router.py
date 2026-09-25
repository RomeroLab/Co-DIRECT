
from __future__ import annotations

import torch

MODALITIES = ("bb_ca", "local_latents")

def interface_router(*, partner_coords: torch.Tensor, designed: torch.Tensor,
                     cutoff_nm: float, interface: dict, bulk: dict,
                     max_pairs: int = 4_000_000):

    for name, table in (("interface", interface), ("bulk", bulk)):
        missing = [m for m in MODALITIES if m not in table]
        if missing:
            raise ValueError(
                f"{name} weights are missing {missing}; a modality with no "
                "weight would silently keep a different leadership direction "
                "than the label claims")
    if partner_coords.dim() != 3 or partner_coords.shape[-1] != 3:
        raise ValueError(
            f"partner_coords has shape {tuple(partner_coords.shape)}; expected [b, k, 3]")
    if partner_coords.shape[1] == 0:
        raise ValueError(
            "the partner has no atoms, so every row would route as bulk and the "
            "arm would be mislabelled as interface-routed")
    if designed.dim() != 2:
        raise ValueError(f"designed has shape {tuple(designed.shape)}; expected [b, n]")
    if float(cutoff_nm) <= 0:
        raise ValueError("cutoff_nm must be positive")

    def router(batch, output):
        ca = batch["x_t"]["bb_ca"]
        b, n = ca.shape[0], ca.shape[1]
        k = partner_coords.shape[1]
        if n * k > max_pairs:
            raise ValueError(
                f"interface distance matrix would be {n}x{k}; raise max_pairs "
                "deliberately rather than allocating this mid-sampling")
        part = partner_coords.to(ca.device, ca.dtype)

        d = torch.cdist(ca, part)
        near = (d.min(dim=-1).values <= float(cutoff_nm))
        des = designed.to(ca.device).bool()
        if des.shape != (b, n):
            raise ValueError(
                f"designed has shape {tuple(des.shape)} against a residue axis "
                f"of {(b, n)}; routing the wrong rows would move the framework")
        out = {}
        for m in MODALITIES:
            w = torch.full((b, n), float(bulk[m]), device=ca.device, dtype=ca.dtype)
            w = torch.where(near, torch.full_like(w, float(interface[m])), w)
            out[m] = torch.where(des, w, torch.zeros_like(w))
        return out

    return router

def partner_atoms_from_batch(batch) -> "torch.Tensor":

    if "x_target" not in batch:
        raise ValueError(
            "this batch carries no x_target, so there is no partner to route "
            "against; an interface-routed arm here would be mislabelled")
    x = batch["x_target"]
    m = batch.get("target_mask")
    if x.dim() == 4:
        b = x.shape[0]
        flat = x.reshape(b, -1, 3)
        keep = (m.reshape(b, -1).bool() if m is not None
                else torch.ones(flat.shape[:2], dtype=torch.bool, device=x.device))
    elif x.dim() == 3:
        flat = x
        keep = (m.bool() if m is not None
                else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device))
    else:
        raise ValueError(f"x_target has shape {tuple(x.shape)}; expected [b, k, 3] "
                         "or [b, n, a, 3]")
    if flat.shape[0] != 1:
        raise ValueError("partner extraction assumes batch size 1; a ragged "
                         "batch would need per-row padding, not a flat select")
    sel = flat[0][keep[0]]
    if sel.numel() == 0:
        raise ValueError("every partner atom is masked out; routing would put "
                         "the whole chain in bulk under an interface label")
    return sel[None]
