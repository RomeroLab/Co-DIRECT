
from __future__ import annotations

import numpy as np

def mask_absent_atoms(coors, atom_mask, residue_mask=None):

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.is_tensor(coors):
        keep = atom_mask.to(torch.bool)
        if residue_mask is not None:
            keep = keep & residue_mask.to(torch.bool)[..., None]
        out = coors.clone().to(torch.float32)
        out[~keep] = float("nan")
        return out

    coors = np.array(coors, dtype=np.float64, copy=True)
    keep = np.asarray(atom_mask, dtype=bool)
    if residue_mask is not None:
        keep = keep & np.asarray(residue_mask, dtype=bool)[..., None]
    out = coors
    out[~keep] = np.nan
    return out
