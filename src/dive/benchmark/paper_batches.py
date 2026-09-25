
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import ArtifactIdentity, file_identity
from dive.benchmark.paper_inputs import PaperTargetRecord
from dive.data.pipeline import FAMILIES
from dive.signed_value.roots import EMERGENT_UPSTREAM_ROOT

class PaperBatchError(RuntimeError):
    pass

class PaperRequestedCellFailure(PaperBatchError):

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code

@dataclass(frozen=True, slots=True)
class PaperBatch:
    target: PaperTargetRecord
    crop_size: int
    seed: int
    tensors: Mapping[str, object]
    structure: ArtifactIdentity
    ligand_loaded: bool

def load_paper_batch(
    target,
    crop_size=256,
    seed=42,
    *,
    max_design_fraction=None,
    target_budget=None,
) -> PaperBatch:

    if not isinstance(target, PaperTargetRecord):
        raise PaperBatchError("target must be a PaperTargetRecord")
    _reject_blind(
        target.source_path,
        target.partition,
        target.example_id,
        target.parent_id,
        target.family,
    )
    if target.partition != "validation" or not target.strict:
        raise PaperBatchError("paper batches retain strict validation only")
    if target.family not in FAMILIES:
        raise PaperBatchError(f"unknown family {target.family!r}")

    path = Path(target.source_path)
    _reject_blind(path)
    structure = file_identity(path)
    _ensure_upstream()

    import pandas as pd
    from proteinfoundation.datasets.structure_data import structure_collate_fn

    from dive.data.dataset import RoleAwareStructureDataset
    from dive.data.loaders import LOADER_COLUMNS, assert_loader_manifest_is_clean
    from dive.data.pipeline import family_loader_kwargs, family_transforms
    from dive.training.runtime import normalize_generation_masks

    try:
        roles = {
            "generated": str(target.role_payload["generated"]),
            "context": str(target.role_payload["context"]),
            "target": str(target.role_payload["target"]),
        }
    except KeyError as error:
        raise PaperBatchError("role payload is missing a required selector") from error
    frame = pd.DataFrame(
        [
            {
                "example_id": target.example_id,
                "path": str(path),
                **roles,
            }
        ],
        columns=list(LOADER_COLUMNS),
    )
    assert_loader_manifest_is_clean(frame)
    kwargs = family_loader_kwargs(target.family)
    dataset = RoleAwareStructureDataset(
        frame,
        transforms=family_transforms(
            target.family,
            crop_size=crop_size,
            seed=seed,
            **(
                {}
                if max_design_fraction is None
                else {"max_design_fraction": max_design_fraction}
            ),
            **({} if target_budget is None else {"target_budget": target_budget}),
        ),
        **kwargs,
    )
    sample = dataset[0]
    if sample is None:
        raise PaperRequestedCellFailure(
            f"loader exclusion for example {target.example_id!r}",
            reason_code="loader_exclusion",
        )
    try:
        collated = structure_collate_fn([sample])
    except Exception as error:
        raise PaperRequestedCellFailure(
            f"loader exclusion for example {target.example_id!r}",
            reason_code="loader_exclusion",
        ) from error
    if not isinstance(collated, dict):
        raise PaperRequestedCellFailure(
            f"loader exclusion for example {target.example_id!r}",
            reason_code="loader_exclusion",
        )
    normalize_generation_masks(collated)
    after = file_identity(path)
    if after.sha256 != structure.sha256 or after.size_bytes != structure.size_bytes:
        raise PaperBatchError("source identity drifted")
    batch = PaperBatch(
        target,
        crop_size,
        seed,
        MappingProxyType(collated),
        structure,
        bool(kwargs.get("atomize_non_polymers") or kwargs.get("use_parse")),
    )
    assert_batch_matches_target(batch, target)
    return batch

def assert_batch_matches_target(batch, target) -> None:

    if not isinstance(batch, PaperBatch) or not isinstance(target, PaperTargetRecord):
        raise PaperBatchError("batch and target identities are required")
    if (
        batch.target.parent_id != target.parent_id
        or batch.target.example_id != target.example_id
        or batch.target.family != target.family
        or batch.target.partition != target.partition
    ):
        raise PaperBatchError("identity mismatch")
    live = file_identity(Path(target.source_path))
    if (
        live.sha256 != batch.structure.sha256
        or live.size_bytes != batch.structure.size_bytes
    ):
        raise PaperBatchError("source identity drifted")
    x_target = batch.tensors.get("x_target")
    if x_target is None or int(getattr(x_target, "shape", (0, 0))[1]) <= 0:
        raise PaperBatchError("empty target")
    generated = batch.tensors.get("generated_mask")
    fixed = batch.tensors.get("fixed_mask")
    if generated is None or not bool(generated.any()):
        raise PaperBatchError("generated mask is empty")
    if fixed is None:
        raise PaperBatchError("fixed mask is missing")
    if "target_hotspot_mask" not in batch.tensors:
        raise PaperBatchError("target_hotspot_mask is missing")
    if target.family == "ame":
        motif = batch.tensors.get("motif_mask")
        if motif is None or not bool(motif.any()):
            raise PaperBatchError("missing motif")
        if batch.ligand_loaded:
            raise PaperBatchError("AME ligand provenance forbids a loaded ligand")
    if target.family == "antibody":
        antigen = str(target.role_payload.get("target") or "")
        tensor_target = batch.tensors.get("target")
        tensor_text = ""
        if isinstance(tensor_target, list) and tensor_target:
            tensor_text = str(tensor_target[0])
        if not antigen.strip() or not tensor_text.strip():
            raise PaperBatchError("missing antigen")
        generated_role = str(target.role_payload.get("generated") or "")
        if ":" not in generated_role:
            raise PaperBatchError("antibody roles require a CDR range")
    for key in ("generated", "context", "target"):
        expected = str(target.role_payload[key])
        value = batch.tensors.get(key)
        if value != [expected]:
            raise PaperBatchError("role mismatch")

def _ensure_upstream() -> None:
    source = str(EMERGENT_UPSTREAM_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    os.environ.setdefault("USE_V2_COMPLEXA_ARCH", "True")
    import proteinfoundation.patches.atomworks_patches

def _reject_blind(*values: object) -> None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        lowered = text.lower()
        if lowered in {"blind", "test-blind", "test_blind"}:
            raise PaperBatchError("blind aliases are forbidden")
        if "test-blind" in lowered or "test_blind" in lowered:
            raise PaperBatchError("blind aliases are forbidden")
        if (
            isinstance(value, Path)
            or "/" in text
            or text.endswith((".parquet", ".cif", ".pdb", ".gz", ".json"))
        ):
            if any("blind" in part.lower() for part in Path(text).parts):
                raise PaperBatchError("blind aliases are forbidden")
