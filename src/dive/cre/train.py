
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from dive.codirect_paths import cache_dir, proteina_root, proteina_src, v6_data

UPSTREAM = proteina_root()
MANIFESTS = cache_dir("loader_manifests_v5")

def _ensure_paths() -> None:
    for path in (proteina_src(),):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

AME_DATA = cache_dir("candidate_response_exchange", "cre-20260909a", "data-filtered")
CACHE_DIR = cache_dir("candidate_response_exchange", "cre-20260909a", "parse-cache")

def _ligand_element_transforms():

    from proteinfoundation.datasets import transforms as T

    class KeepLigandElement(T.AtomworksLigandFeaturesTransform):
        def __call__(self, data):
            atomworks_data = getattr(data, "_atomworks_data", None)
            element = None
            if atomworks_data is not None:
                from atomworks.ml.utils.token import get_af3_token_representative_idxs

                atom_array = atomworks_data.get("atom_array", None)
                atom_element = atomworks_data.get("atom_element", None)
                representative = atomworks_data.get("ground_truth", {}).get(
                    "rep_atom_idxs",
                    get_af3_token_representative_idxs(atom_array) if atom_array is not None else None,
                )
                if atom_element is not None and representative is not None:
                    element = atom_element[representative]
            data = super().__call__(data)
            if element is not None:
                data.target_residue_element = element
            return data

    class ExtractTargetWithElement(T.ExtractTargetCoordinatesTransform):
        def __call__(self, graph):

            selection = graph.target_mask.sum(dim=-1).bool()
            element = getattr(graph, "target_residue_element", None)
            graph = super().__call__(graph)
            if element is not None and selection.any():
                graph.seq_target = element[selection].long()
            return graph

    return KeepLigandElement, ExtractTargetWithElement

class LigandOneHotElement:

    def __call__(self, data):
        import torch
        import torch.nn.functional as F

        if hasattr(data, "seq_target") and data.seq_target.ndim == 1:
            values = data.seq_target.long().clamp(min=0, max=127)
            data.seq_target = F.one_hot(values, num_classes=128).to(torch.float32)
        return data

class LigandTargetMaskSqueeze:

    def __call__(self, data):
        if hasattr(data, "target_mask") and data.target_mask.ndim == 2:
            data.target_mask = data.target_mask[:, 1]
        return data

def _official_ame_atom37_transforms():
    from proteinfoundation.datasets import transforms as T

    keep_element, extract_with_element = _ligand_element_transforms()
    return [
        keep_element(remove_atomworks_data=True),
        T.CoordsToNanometers(),
        T.GlobalRotationTransform(),
        T.MotifMaskTransform(
            atom_selection_mode="ame",
            residue_selection_mode="absolute_number",
            motif_max_pct_res=0.3,
            motif_min_pct_res=0.05,
            motif_prob=1.0,
            motif_min_n_seg=1,
            motif_max_n_seg=4,
            motif_min_n_res=1,
            motif_max_n_res=8,
        ),
        T.ChainBreakPerResidueTransform(),
        T.CenteringTransform(center_mode="target", data_mode="ligand_atom37"),
        extract_with_element(compact_mode=True),
        T.FilterTargetResiduesTransform(),
        T.LigandAtom37SqueezeTransform(),
        LigandTargetMaskSqueeze(),
        LigandOneHotElement(),
        T.ExtractMotifCoordinatesTransform(compact_mode=True),
    ]

V6_DATA = v6_data()

def _resolve_structure_manifest(csv_path):

    import csv
    from pathlib import Path

    from dive.codirect_paths import structure_root

    root = structure_root()
    out = cache_dir("manifests", Path(csv_path).name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open(newline="") as src, out.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            path = Path(row["path"])
            if not path.is_absolute():
                row["path"] = str(root / path)
            writer.writerow(row)
    return out

def _v6_ame_atom37_transforms(stats=None):

    from proteinfoundation.datasets import transforms as T

    from dive.v6.motif import LigandContactMotifTransform

    keep_element, extract_with_element = _ligand_element_transforms()
    return [
        keep_element(remove_atomworks_data=True),
        T.CoordsToNanometers(),
        T.GlobalRotationTransform(),
        LigandContactMotifTransform(stats=stats),
        T.ChainBreakPerResidueTransform(),
        T.CenteringTransform(center_mode="target", data_mode="ligand_atom37"),
        extract_with_element(compact_mode=True),
        T.FilterTargetResiduesTransform(),
        T.LigandAtom37SqueezeTransform(),
        LigandTargetMaskSqueeze(),
        LigandOneHotElement(),
        T.ExtractMotifCoordinatesTransform(compact_mode=True),
    ]

def ame_datamodule(*, batch_size: int = 2, num_workers: int = 0, seed: int = 42,
                   v6: bool = False, v6_motif: bool | None = None, motif_stats=None):

    _ensure_paths()

    import proteinfoundation.patches.atomworks_patches
    import torch

    from proteinfoundation.datasets.structure_data import StructureDataModule

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if v6_motif is None:
        v6_motif = v6
    train_csv = (V6_DATA / "ame_train_v6.csv") if v6 else (AME_DATA / "ame_train.csv")
    val_csv = (V6_DATA / "ame_validation_v6.csv") if v6 else (AME_DATA / "ame_validation.csv")
    if v6:
        train_csv = _resolve_structure_manifest(train_csv)
        val_csv = _resolve_structure_manifest(val_csv)
    module = StructureDataModule(
        metadata_file=str(train_csv),
        val_metadata_file=str(val_csv),
        batch_size=batch_size,
        num_workers=num_workers,
        id_column="example_id",
        path_column="path",
        pin_memory=False,
        use_parse=True,
        parser_args={
            "add_missing_atoms": True,
            "fix_formal_charges": True,
            "cache_dir": str(CACHE_DIR),
            "save_to_cache": True,
            "load_from_cache": True,
            "add_id_and_entity_annotations": True,
        },
        pipeline_target=(
            "proteinfoundation.datasets.pipelines.complexa_ligand"
            ".build_protein_ligand_transform_pipeline"
        ),
        atom37_transforms=(_v6_ame_atom37_transforms(stats=motif_stats) if v6_motif
                           else _official_ame_atom37_transforms()),
        pad_group_priority=["target", "motif"],
        columns_to_load=["example_id", "complex_name", "path"],
    )
    module.setup("fit")
    return module

REQUIRED_GROUPS = frozenset({"motif", "target"})

def _homogeneous_collate(pad_group_priority=("target", "motif")):

    from proteinfoundation.datasets.structure_data import structure_collate_fn

    def collate(samples):
        usable = [
            sample for sample in samples
            if sample is not None
            and REQUIRED_GROUPS <= set(getattr(sample, "_token_groups", {}) or {})
        ]
        if not usable:
            return None
        return structure_collate_fn(
            usable, pad_max_total_tokens=None,
            pad_group_priority=list(pad_group_priority),
        )

    return collate

def ame_dataloader(
    split: str, *, batch_size: int = 2, num_workers: int = 0, seed: int = 42,
    shuffle: bool | None = None, v6: bool = False, v6_motif: bool | None = None,
    motif_stats=None,
):
    import torch

    module = ame_datamodule(batch_size=batch_size, num_workers=num_workers,
                            seed=seed, v6=v6, v6_motif=v6_motif,
                            motif_stats=motif_stats)
    dataset = module.train_dataset if split == "train" else module.val_dataset
    if shuffle is None:
        shuffle = split == "train"
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=_homogeneous_collate(),
        generator=torch.Generator().manual_seed(seed),
    )

@contextmanager
def _stubbed_logger(model):

    from types import SimpleNamespace

    original_log = model.__dict__.get("log", None)
    original_trainer = model.__dict__.get("_trainer", None)
    model.log = lambda *args, **kwargs: None
    if original_trainer is None:
        model._trainer = SimpleNamespace(world_size=1, global_step=0, current_epoch=0)
    try:
        yield
    finally:
        if original_log is None:
            model.__dict__.pop("log", None)
        else:
            model.log = original_log
        if original_trainer is None:
            model.__dict__["_trainer"] = None
        else:
            model.__dict__["_trainer"] = original_trainer

def native_training_step(model, batch, *, batch_idx: int = 0, device: str | None = None):

    import torch

    if device is None:
        device = next(model.parameters()).device
    batch = {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    captured: dict[str, float] = {}
    original_log_losses = model.log_losses

    def capture(bs, losses, log_prefix, batch):
        for key, value in losses.items():
            captured[key] = float(torch.mean(value).detach())
        return None

    model.log_losses = capture
    try:
        with _stubbed_logger(model):
            loss = model.training_step(batch, batch_idx)
    finally:
        model.log_losses = original_log_losses
    return loss, captured

def _rng_state(device):
    import torch

    return (torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None)

def _restore_rng(state, device):
    import torch

    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)

def two_pass_step(model, batch, *, exchange=None, batch_idx: int = 0, device=None):

    import torch

    from proteinfoundation.utils.sample_utils import add_clean_samples
    from proteinfoundation.utils.training_handlers import handle_batch_conditioning

    from dive.ccr.trunk import MESSAGE_KEY

    if device is None:
        device = next(model.parameters()).device
    batch = {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    batch = add_clean_samples(
        batch, model.cfg_exp.product_flowmatcher, getattr(model, "autoencoder", None)
    )
    batch = model.fm.corrupt_batch(batch)
    batch_size = batch["mask"].shape[0]
    batch, n_recycle = handle_batch_conditioning(
        batch, batch_size, model.cfg_exp.training, model.call_nn, model.fm
    )

    message = None
    if exchange is not None:
        state = _rng_state(device)
        batch.pop(MESSAGE_KEY, None)
        with torch.no_grad():
            nn_out = model.call_nn(batch, n_recycle=n_recycle)
            clean = model.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
        _restore_rng(state, device)
        message = exchange(batch, clean, model)

    if message is not None:
        batch[MESSAGE_KEY] = message.to(device)
    else:
        batch.pop(MESSAGE_KEY, None)

    nn_out = model.call_nn(batch, n_recycle=n_recycle)
    losses = model.fm.compute_loss(batch=batch, nn_out=nn_out)
    loss = sum(torch.mean(losses[k]) for k in losses if "_justlog" not in k)
    parts = {k: float(torch.mean(v).detach()) for k, v in losses.items()}
    return loss, parts
