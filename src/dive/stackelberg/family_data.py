
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from dive.codirect_paths import benchv2_root, proteina_root, proteina_src

FAMILIES = ("binder", "antibody")

def _prepare_import() -> None:
    src = proteina_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ.setdefault("DATA_PATH", str(proteina_root() / "assets"))

def manifest_path(family: str, split: str) -> Path:
    if family not in FAMILIES:
        raise ValueError(f"{family} is not a role-manifest family")
    if split not in ("train", "validation"):
        raise ValueError(f"split must be train or validation, got {split!r}")
    path = benchv2_root() / "splits" / "v6" / f"{family}_{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def family_role_dataloader(
    family: str,
    split: str,
    *,
    batch_size: int = 2,
    seed: int = 1401,
    crop_size: int = 256,
    target_budget: int = 96,
    shuffle: bool | None = None,
):

    _prepare_import()
    import pandas as pd
    from proteinfoundation.datasets.structure_data import structure_collate_fn

    from dive.data.crop import RoleAwareCrop
    from dive.data.dataset import RoleAwareStructureDataset
    from dive.data.pipeline import family_transforms
    from dive.training.runtime import assert_manifest_is_trainable

    import proteinfoundation.patches.atomworks_patches

    from dive.training.runtime import partition_from_manifest_name

    path = manifest_path(family, split)
    frame = pd.read_parquet(path)
    if "partition" not in frame.columns:
        frame = frame.copy()
        frame["partition"] = partition_from_manifest_name(path.name)
    assert_manifest_is_trainable(path, frame)
    chain = family_transforms(family, crop_size=crop_size)
    chain = [
        RoleAwareCrop(crop_size=crop_size, target_budget=target_budget)
        if isinstance(step, RoleAwareCrop) else step
        for step in chain
    ]
    dataset = RoleAwareStructureDataset(frame, transforms=chain)

    def collate(samples):
        usable = [s for s in samples if s is not None]
        if not usable:
            return None
        return structure_collate_fn(
            usable, pad_max_total_tokens=None,
            pad_group_priority=["target", "motif"],
        )

    if shuffle is None:
        shuffle = split == "train"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
