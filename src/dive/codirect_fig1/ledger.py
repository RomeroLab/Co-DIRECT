
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

MODALITIES = ("bb_ca", "local_latents")

class LedgerError(ValueError):
    pass

def _check(values: Mapping[str, Tensor], mask: Tensor, what: str) -> None:
    missing = [m for m in MODALITIES if m not in values]
    if missing:
        raise LedgerError(
            f"{what} is missing {missing}; a modality dropped from the ledger "
            "would leave G describing a different objective than the label"
        )
    for mode in MODALITIES:
        if values[mode].shape != mask.shape:
            raise LedgerError(
                f"{what}[{mode}] has shape {tuple(values[mode].shape)}, expected "
                f"[b, n] = {tuple(mask.shape)}"
            )

def _assert_padding_is_zero(values: Mapping[str, Tensor], mask: Tensor) -> None:
    pad = ~mask.bool()
    if not pad.any():
        return
    for mode in MODALITIES:
        leaked = values[mode].masked_select(pad)
        if leaked.numel() and bool((leaked != 0).any()):
            worst = float(leaked.abs().max())
            raise LedgerError(
                f"{mode} carries a nonzero loss contribution of {worst:.3e} on a "
                "padded residue. Upstream masks the error before squaring, so "
                "this means the loss was computed against the wrong mask and "
                "every sum below would include padding."
            )

def total_loss(values: Mapping[str, Tensor], mask: Tensor) -> Tensor:

    _check(values, mask, "loss")
    _assert_padding_is_zero(values, mask)
    keep = mask.bool()
    total = torch.zeros(mask.shape[0], dtype=torch.float64, device=mask.device)
    for mode in MODALITIES:
        total = total + (values[mode].double() * keep).sum(dim=-1)
    return total

def signed_delta(
    base: Mapping[str, Tensor], corrected: Mapping[str, Tensor], mask: Tensor
) -> dict[str, Tensor]:

    _check(base, mask, "base")
    _check(corrected, mask, "corrected")
    _assert_padding_is_zero(base, mask)
    _assert_padding_is_zero(corrected, mask)
    keep = mask.bool()
    return {
        mode: ((base[mode].double() - corrected[mode].double()) * keep).to(
            base[mode].dtype if base[mode].dtype == torch.float64 else torch.float64
        )
        for mode in MODALITIES
    }

def _split(values: Tensor) -> tuple[Tensor, Tensor]:
    return values.clamp(min=0.0), (-values).clamp(min=0.0)

def block_ledger(delta: Mapping[str, Tensor], mask: Tensor) -> dict[str, Tensor]:

    _check(delta, mask, "delta")
    keep = mask.bool()
    out: dict[str, Tensor] = {}
    b_total = torch.zeros(mask.shape[0], dtype=torch.float64, device=mask.device)
    h_total = torch.zeros_like(b_total)
    for mode in MODALITIES:
        benefit, harm = _split(delta[mode].double() * keep)
        out[f"B_{mode}"] = benefit.sum(dim=-1)
        out[f"H_{mode}"] = harm.sum(dim=-1)
        out[f"G_{mode}"] = out[f"B_{mode}"] - out[f"H_{mode}"]
        b_total = b_total + out[f"B_{mode}"]
        h_total = h_total + out[f"H_{mode}"]
    out["B"], out["H"], out["G"] = b_total, h_total, b_total - h_total
    out["n_residues"] = keep.sum(dim=-1)
    out["n_terms"] = keep.sum(dim=-1) * len(MODALITIES)
    out["n_benefit_terms"] = sum(
        ((delta[m].double() * keep) > 0).sum(dim=-1) for m in MODALITIES
    )
    out["n_harm_terms"] = sum(
        ((delta[m].double() * keep) < 0).sum(dim=-1) for m in MODALITIES
    )
    out["n_zero_terms"] = out["n_terms"] - out["n_benefit_terms"] - out["n_harm_terms"]
    return out

def residue_ledger(delta: Mapping[str, Tensor], mask: Tensor) -> dict[str, Tensor]:

    _check(delta, mask, "delta")
    keep = mask.bool()
    per_residue = torch.zeros(mask.shape, dtype=torch.float64, device=mask.device)
    for mode in MODALITIES:
        per_residue = per_residue + delta[mode].double()
    per_residue = per_residue * keep
    benefit, harm = _split(per_residue)
    return {
        "B": benefit.sum(dim=-1),
        "H": harm.sum(dim=-1),
        "G": benefit.sum(dim=-1) - harm.sum(dim=-1),
        "n_residues": keep.sum(dim=-1),
        "n_benefit_residues": (per_residue > 0).sum(dim=-1),
        "n_harm_residues": (per_residue < 0).sum(dim=-1),
        "n_zero_residues": keep.sum(dim=-1)
        - (per_residue > 0).sum(dim=-1)
        - (per_residue < 0).sum(dim=-1),
    }

def assert_correction_respects_mask(
    correction: Mapping[str, Tensor], designed: Tensor
) -> None:

    off = ~designed.bool()
    if not off.any():
        return
    for mode, value in correction.items():
        rows = off
        while rows.dim() < value.dim():
            rows = rows[..., None]
        leaked = value.masked_select(rows.expand_as(value))
        if leaked.numel() and bool((leaked != 0).any()):
            worst = float(leaked.abs().max())
            raise LedgerError(
                f"the {mode} correction is {worst:.3e} on a residue outside the "
                "design mask; padding and fixed context must receive exactly "
                "zero, or the measured effect includes rows the run never designs"
            )

def apply_alpha(
    velocity: Mapping[str, Tensor], correction: Mapping[str, Tensor], alpha: float
) -> dict[str, Tensor]:

    if float(alpha) == 0.0:
        return {mode: velocity[mode] for mode in velocity}
    return {
        mode: velocity[mode] + float(alpha) * correction[mode] for mode in velocity
    }
