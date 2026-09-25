
from __future__ import annotations

import torch

CONDITION_KEYS = (

    "x_motif", "motif_mask", "seq_motif", "seq_motif_mask",

    "x_target", "target_mask", "seq_target", "seq_target_mask",
    "target_atom_name", "target_bond_mask", "target_bond_order",
    "target_charge", "target_padding_mask", "target_hotspot_mask",
    "target_laplacian_pe", "target_residue_mask", "n_target",

    "mask", "nres", "nsamples",

    "chain_breaks_per_residue", "chain_id", "seq_pos",
)

FORCED_OFF = ("use_ca_coors_nm_feature", "use_residue_type_feature")

REQUIRED_ANY = ("x_motif", "motif_mask", "x_target", "target_mask")

class ConditionError(RuntimeError):
    pass

def condition_only(native: dict) -> dict:

    if "mask" not in native:
        raise ConditionError(
            "the native batch carries no mask, so the source's length is unknown "
            "and generating at a guessed length would break the pairing")
    if not any(key in native for key in REQUIRED_ANY):
        raise ConditionError(
            f"the native batch carries no condition: none of {REQUIRED_ANY} is "
            f"present, so this would be unconditional generation with a "
            f"conditional label")

    out = {key: native[key] for key in CONDITION_KEYS if key in native}
    for flag in FORCED_OFF:
        out[flag] = False
    mask = out["mask"]
    out["nres"] = int(mask.sum()) if mask.dim() == 1 else int(mask[0].sum())
    if not isinstance(out.get("nsamples"), int):
        out["nsamples"] = int(mask.shape[0]) if isinstance(mask, torch.Tensor) else 1
    return out
