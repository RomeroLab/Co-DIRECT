
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

_MOTIF_MARKERS = ("concat_factory.linear_out.", "concat_pair_factory.linear_out.")

class SpliceMigrationError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class Migration:

    source: str
    destinations: tuple[str, ...]
    rationale: str

TARGET_CONDITIONING_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        source="nn.concat_factory.linear_out.weight",
        destinations=("nn.concat_factory.linear_out_target.weight",),
        rationale=(
            "v1 ConcatFeaturesFactory creates a single linear_out over the active "
            "branch; with enable_target=True and enable_motif=False that layer is "
            "the target projection. v2 renames it to linear_out_target and reuses "
            "the name linear_out for the motif branch only. Identical shape."
        ),
    ),
    Migration(
        source="nn.concat_pair_factory.linear_out.weight",
        destinations=(
            "nn.concat_pair_factory.linear_out_target.weight",
            "nn.concat_pair_factory.linear_out_lower_right_target.weight",
        ),
        rationale=(
            "v1's target branch applies the SAME linear_out to the upper-right "
            "(sample-to-target) and lower-right (target-to-target) blocks, which "
            "have identical feature composition and input dimension. v2 split that "
            "shared layer into two independent projections, so restoring v1's "
            "function exactly requires copying the one source into both."
        ),
    ),
)

CRITICAL_TARGET_PARAMETERS: frozenset[str] = frozenset(
    destination
    for migration in TARGET_CONDITIONING_MIGRATIONS
    for destination in migration.destinations
)

def tensor_sha256(tensor: Tensor) -> str:

    return hashlib.sha256(
        tensor.detach().cpu().contiguous().float().numpy().tobytes()
    ).hexdigest()

def is_target_conditioning_parameter(key: str) -> bool:

    return (
        ("concat_factory" in key or "concat_pair_factory" in key)
        and "target" in key
        and "lora_" not in key
    )

def _is_motif_destination(key: str) -> bool:

    return any(marker in key for marker in _MOTIF_MARKERS)

def apply_semantic_migrations(
    model_state: Mapping[str, Tensor],
    checkpoint_state: Mapping[str, Tensor],
    *,
    migrations: Sequence[Migration] = TARGET_CONDITIONING_MIGRATIONS,
    into: Mapping[str, Tensor] | None = None,
) -> tuple[dict[str, Tensor], dict]:

    if not model_state:
        raise SpliceMigrationError("model state is empty; nothing to migrate into")
    if not checkpoint_state:
        raise SpliceMigrationError("checkpoint state is empty; nothing to migrate from")

    spliced = dict(model_state if into is None else into)
    rows: list[dict] = []
    declared: set[str] = set()

    for migration in migrations:

        if migration.source in model_state and _is_motif_destination(migration.source):
            raise SpliceMigrationError(
                f"source {migration.source!r} collides with a motif parameter that "
                f"exists in this model: v2 reuses the name linear_out for the motif "
                f"branch, so a name-matching splice would write target-projection "
                f"weights into motif. Disambiguate the source before loading."
            )
        if migration.source not in checkpoint_state:
            raise SpliceMigrationError(
                f"declared migration source {migration.source!r} is absent from the "
                f"checkpoint; the pretrained target projection cannot be restored"
            )
        source = checkpoint_state[migration.source]
        for destination in migration.destinations:
            if _is_motif_destination(destination):
                raise SpliceMigrationError(
                    f"refusing to write target-projection weights into the motif "
                    f"parameter {destination!r}: v2 reuses the name linear_out for "
                    f"the motif branch and the semantics differ"
                )
            if destination not in model_state:
                raise SpliceMigrationError(
                    f"migration destination {destination!r} is absent from the model; "
                    f"the graph does not match the declared architecture"
                )
            target = model_state[destination]
            if tuple(source.shape) != tuple(target.shape):
                raise SpliceMigrationError(
                    f"shape {tuple(source.shape)} of {migration.source!r} does not "
                    f"match {tuple(target.shape)} of {destination!r}; refusing to "
                    f"resize a semantically distinct tensor"
                )
            spliced[destination] = source.detach().clone()
            declared.add(destination)
            rows.append(
                {
                    "source": migration.source,
                    "destination": destination,
                    "shape": list(source.shape),
                    "classification": "semantic_rename",
                    "exact_copy": True,
                    "source_sha256": tensor_sha256(source),
                    "rationale": migration.rationale,
                }
            )

    observed = {k for k in model_state if is_target_conditioning_parameter(k)}
    unmapped = sorted(observed - declared)
    if unmapped:
        candidates = {
            key: sorted(
                source
                for source, tensor in checkpoint_state.items()
                if tuple(tensor.shape) == tuple(model_state[key].shape)
                and source not in model_state
            )
            for key in unmapped
        }
        raise SpliceMigrationError(
            f"unmapped or ambiguous target-conditioning parameters {unmapped}; "
            f"same-shape checkpoint candidates {candidates}. Declare an explicit "
            f"Migration or prove the parameter has no pretrained counterpart."
        )

    intentionally_new = sorted(
        k
        for k in model_state
        if ("concat_factory" in k or "concat_pair_factory" in k)
        and "target" in k
        and "lora_" in k
    )
    audit = {
        "migrations": rows,
        "intentionally_new_lora_adapters": intentionally_new,
        "critical_parameters": sorted(CRITICAL_TARGET_PARAMETERS),
        "unmapped_target_parameters": [],

        "bias_tensors_required": [],
    }
    return spliced, audit

def assert_target_conditioning_restored(
    loaded_state: Mapping[str, Tensor],
    checkpoint_state: Mapping[str, Tensor],
    *,
    migrations: Sequence[Migration] = TARGET_CONDITIONING_MIGRATIONS,
) -> None:

    problems: list[str] = []
    for migration in migrations:
        if migration.source not in checkpoint_state:
            problems.append(f"{migration.source}: absent from checkpoint")
            continue
        source = checkpoint_state[migration.source]
        for destination in migration.destinations:
            if destination not in loaded_state:
                problems.append(f"{destination}: absent from the loaded model")
                continue
            live = loaded_state[destination]
            if tuple(live.shape) != tuple(source.shape):
                problems.append(
                    f"{destination}: shape {tuple(live.shape)} != "
                    f"{tuple(source.shape)}"
                )
            elif not torch.equal(live.detach().cpu().float(), source.detach().cpu().float()):
                problems.append(
                    f"{destination}: not exactly equal to {migration.source} "
                    f"(still random or overwritten)"
                )
    if problems:
        raise SpliceMigrationError(
            "target conditioning was not restored: " + "; ".join(problems)
        )
    return None

__all__ = [
    "CRITICAL_TARGET_PARAMETERS",
    "TARGET_CONDITIONING_MIGRATIONS",
    "Migration",
    "SpliceMigrationError",
    "apply_semantic_migrations",
    "assert_target_conditioning_restored",
    "is_target_conditioning_parameter",
    "tensor_sha256",
]
