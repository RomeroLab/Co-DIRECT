
from __future__ import annotations

import os
from pathlib import Path

from dive.training.preflight import (
    PreflightError,
    assert_manifest_is_trainable,
)

class GenerationContractError(RuntimeError):
    pass

def normalize_generation_masks(collated):

    import torch

    design = collated.get("design_mask")
    mask = collated.get("mask")
    if design is None or mask is None:
        missing = "design_mask" if design is None else "mask"
        raise GenerationContractError(
            f"collated batch is missing {missing!r}; the generation contract "
            f"cannot be established"
        )
    if not isinstance(design, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise GenerationContractError("design_mask and mask must be tensors")
    if design.ndim != 2 or mask.ndim != 2:
        raise GenerationContractError(
            f"design_mask and mask must be [B, N]; got {tuple(design.shape)} "
            f"and {tuple(mask.shape)}"
        )
    if design.shape != mask.shape:
        raise GenerationContractError(
            f"design_mask {tuple(design.shape)} does not match mask "
            f"{tuple(mask.shape)}"
        )

    generated = design.bool()
    valid = mask.bool()
    if bool((generated & ~valid).any()):
        raise GenerationContractError(
            "design selects residues outside the valid mask; generating a "
            "padding position is a corrupt batch, not a recoverable skip"
        )
    per_example = generated.sum(dim=-1)
    if bool((per_example <= 0).any()):
        empty = [index for index, count in enumerate(per_example.tolist()) if count <= 0]
        raise GenerationContractError(
            f"examples {empty} have no generated residue; an all-false "
            f"generated mask trains to predict nothing"
        )

    fixed = valid & ~generated

    if bool((generated & fixed).any()):
        raise GenerationContractError("generated and fixed masks overlap")
    if not bool(torch.equal(generated | fixed, valid)):
        raise GenerationContractError(
            "generated and fixed masks do not partition the valid mask"
        )

    collated["generated_mask"] = generated
    collated["fixed_mask"] = fixed

    del collated["design_mask"]
    return collated

def _family_dataloader(
    family,
    manifest_path,
    *,
    batch_size,
    crop_size,
    seed,
    motif_seed=None,
    num_workers,
    limit,
    world_size=1,
    rank=0,
    shuffle=True,
):

    import torch
    from proteinfoundation.datasets.structure_data import structure_collate_fn

    from dive.data.dataset import RoleAwareStructureDataset
    from dive.data.loaders import assert_loader_manifest_is_clean
    from dive.data.pipeline import family_loader_kwargs, family_transforms

    from dive.training.parent_projection import VerifiedManifest

    if isinstance(manifest_path, VerifiedManifest):
        frame = manifest_path.frame()
        source_path = Path(manifest_path.spec.name)
        expected_partition = manifest_path.spec.partition
    else:
        import pandas as pd

        frame = pd.read_parquet(manifest_path)
        source_path = Path(manifest_path)
        expected_partition = (
            "validation"
            if source_path.name.endswith("_validation.parquet")
            else "train"
        )
    inspected = (
        frame
        if "partition" in frame.columns
        else frame.assign(partition=expected_partition)
    )
    assert_manifest_is_trainable(source_path, inspected)
    assert_loader_manifest_is_clean(
        frame.drop(columns=["parent_id"])
        if isinstance(manifest_path, VerifiedManifest) and "parent_id" in frame.columns
        else frame
    )
    parent_by_example = (
        dict(zip(frame["example_id"].astype(str), frame["parent_id"], strict=True))
        if "parent_id" in frame.columns
        else None
    )
    if limit:
        frame = frame.head(limit)

    dataset = RoleAwareStructureDataset(
        frame,
        transforms=family_transforms(family, crop_size=crop_size, seed=motif_seed),
        **family_loader_kwargs(family),
    )

    def collate(samples):

        usable = [sample for sample in samples if sample is not None]
        if not usable:
            return None
        try:
            collated = structure_collate_fn(usable)
        except Exception:
            return None

        if isinstance(collated, dict):
            normalize_generation_masks(collated)
            if parent_by_example is not None:
                ids = collated.get("id", ())
                if isinstance(ids, str):
                    ids = (ids,)
                collated["parent_id"] = tuple(
                    parent_by_example[str(example_id)] for example_id in ids
                )
        return collated

    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )

def _validation_loaders(
    manifests,
    *,
    batch_size,
    crop_size,
    seed,
    motif_seed=None,
    num_workers,
    max_batches,
    world_size,
    rank,
):

    loaders = {}
    verified_by_name = None
    if not isinstance(manifests, (str, os.PathLike, Path)):
        verified_by_name = {item.spec.name: item for item in manifests}
    for family in ("binder", "ame", "antibody"):
        name = f"{family}_validation.parquet"
        if verified_by_name is None:
            path = Path(manifests) / name
            if not path.exists():
                raise PreflightError(f"no validation manifest for {family} at {path}")
        else:
            path = verified_by_name.get(name)
            if path is None:
                raise PreflightError(f"no verified validation manifest for {family}")
        loaders[family] = _family_dataloader(
            family,
            path,
            batch_size=batch_size,
            crop_size=crop_size,
            seed=seed,
            motif_seed=motif_seed,
            num_workers=num_workers,

            limit=max_batches * batch_size * world_size * 2,
            world_size=world_size,
            rank=rank,
            shuffle=False,
        )
    return loaders

class _FamilyStream:

    def __init__(self, family, loader):
        self.family = family
        self.loader = loader
        self.sampler = getattr(loader, "sampler", None)

    def __iter__(self):
        while True:
            for batch in self.loader:
                if batch is None:
                    continue
                record = dict(batch)

                record["example_ids"] = tuple(record.pop("id", ("<unknown>",)))

                record.pop("parent_id", None)
                yield record

def _configure_allocator() -> None:

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _bind_local_device() -> None:

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else None
    if devices and len(devices) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(local_rank)].strip()
    elif not devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
