
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from dive.codirect_paths import proteina_root, proteina_src

_CREATORS = None
_NAMES = ("motif", "ligand", "target")

def _ensure_proteina() -> None:
    src = proteina_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ.setdefault("DATA_PATH", str(proteina_root() / "assets"))

def _creators():
    global _CREATORS
    if _CREATORS is None:
        _ensure_proteina()
        from proteinfoundation.nn.feature_factory.ligand_feats import (
            LigandConcatSeqFeat)
        from proteinfoundation.nn.feature_factory.motif_feats import (
            MotifConcatSeqFeat)
        from proteinfoundation.nn.feature_factory.target_feats import (
            TargetConcatSeqFeat)
        _CREATORS = (
            MotifConcatSeqFeat(),
            LigandConcatSeqFeat(),
            TargetConcatSeqFeat(),
        )
    return _CREATORS

def trunk_condition_layout() -> tuple[tuple[str, int], ...]:

    return tuple((name, int(creator.dim))
                 for name, creator in zip(_NAMES, _creators(), strict=True))

def trunk_condition_dim() -> int:
    return sum(width for _, width in trunk_condition_layout())

def _batch_size_device(batch: dict) -> tuple[int, torch.device, torch.dtype]:
    for key in ("mask", "x_motif", "x_target", "bb_ca"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.dim() >= 1:
            dtype = value.dtype if value.is_floating_point() else torch.float32
            return int(value.shape[0]), value.device, dtype
    xt = batch.get("x_1")
    if isinstance(xt, dict):
        for value in xt.values():
            if torch.is_tensor(value) and value.dim() >= 1:
                dtype = value.dtype if value.is_floating_point() else torch.float32
                return int(value.shape[0]), value.device, dtype
    raise ValueError(
        "trunk condition needs mask, x_motif, x_target, or bb_ca to size the batch")

def _mean_rows(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

    if feats.dim() != 3:
        raise ValueError(
            f"concat condition features must be [b, n, d], got {tuple(feats.shape)}")
    if mask.shape[:2] != feats.shape[:2]:
        raise ValueError(
            f"condition mask {tuple(mask.shape)} does not match features "
            f"{tuple(feats.shape)}")
    width = int(feats.shape[-1])
    if feats.shape[1] == 0:
        return feats.new_zeros(feats.shape[0], width)
    weight = mask.to(dtype=feats.dtype)
    while weight.dim() > 2:
        weight = weight.amax(dim=-1)
    weight = weight[..., None]
    total = weight.sum(dim=1)
    summed = (feats * weight).sum(dim=1)
    out = feats.new_zeros(feats.shape[0], width)
    present = total.squeeze(-1) > 0
    out[present] = summed[present] / total[present]
    return out.detach()

def _zeros(n: int, width: int, *, device, dtype) -> torch.Tensor:
    return torch.zeros(n, width, device=device, dtype=dtype)

def _motif_block(batch: dict, n: int, device, dtype) -> torch.Tensor:
    creator = _creators()[0]
    if "x_motif" not in batch:
        return _zeros(n, int(creator.dim), device=device, dtype=dtype)
    coords = batch["x_motif"]
    if not torch.is_tensor(coords) or coords.dim() != 4 or coords.shape[-2:] != (37, 3):
        raise ValueError(
            "x_motif must be [b, n_motif, 37, 3], the tensor MotifConcatSeqFeat "
            f"reads; got {None if not torch.is_tensor(coords) else tuple(coords.shape)}")
    if int(coords.shape[0]) != n:
        raise ValueError(
            f"x_motif batch {int(coords.shape[0])} != condition batch {n}")
    feats, mask = creator(batch)
    return _mean_rows(feats, mask).to(device=device, dtype=dtype)

def _partner_blocks(batch: dict, n: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    ligand_creator, target_creator = _creators()[1], _creators()[2]
    ligand_w, target_w = int(ligand_creator.dim), int(target_creator.dim)
    coords = batch.get("x_target")
    if coords is None:
        return (_zeros(n, ligand_w, device=device, dtype=dtype),
                _zeros(n, target_w, device=device, dtype=dtype))
    if not torch.is_tensor(coords):
        raise ValueError("x_target must be a tensor")
    if int(coords.shape[0]) != n:
        raise ValueError(
            f"x_target batch {int(coords.shape[0])} != condition batch {n}")
    if coords.dim() == 3 and int(coords.shape[-1]) == 3:
        feats, mask = ligand_creator(batch)
        ligand = _mean_rows(feats, mask).to(device=device, dtype=dtype)
        return ligand, _zeros(n, target_w, device=device, dtype=dtype)
    if coords.dim() == 4 and coords.shape[-2:] == (37, 3):
        feats, mask = target_creator(batch)
        target = _mean_rows(feats, mask).to(device=device, dtype=dtype)
        return _zeros(n, ligand_w, device=device, dtype=dtype), target
    raise ValueError(
        f"x_target shape {tuple(coords.shape)} is neither ligand [b, n, 3] nor "
        "protein [b, n, 37, 3]; those are the two tensors the trunk's concat "
        "factory reads")

def trunk_condition_features(batch: dict) -> torch.Tensor:

    if not isinstance(batch, dict):
        raise TypeError("trunk condition features are read from the batch dict")
    n, device, dtype = _batch_size_device(batch)
    motif = _motif_block(batch, n, device, dtype)
    ligand, target = _partner_blocks(batch, n, device, dtype)
    return torch.cat([motif, ligand, target], dim=-1)
