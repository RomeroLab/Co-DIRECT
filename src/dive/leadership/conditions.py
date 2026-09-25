
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Generator, Tensor

from dive.counterfactual.builder import Conditions, _build_conditions
from dive.counterfactual.registry import FeatureRegistry

DIRECTIONAL_CONDITION_NAMES: tuple[str, ...] = (
    "joint",
    "without_latent",
    "without_backbone",
)

PRESENCE_BY_CONDITION: dict[str, tuple[int, int]] = {
    "joint": (1, 1),
    "without_latent": (1, 0),
    "without_backbone": (0, 1),
}

PRESENCE_KEY = "dive_modality_presence"

class ConditionPackingError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class DirectionalConditions:

    batches: Mapping[str, dict]
    metadata: Mapping[str, str]

@dataclass(frozen=True, slots=True)
class PackedConditions:

    batch: dict
    slices: Mapping[str, slice]
    batch_size: int

    @property
    def packed_size(self) -> int:
        return self.batch_size * len(self.slices)

def build_directional_conditions(
    source: Mapping,
    registry: FeatureRegistry,
    recompute_pair_features: Callable[[dict], dict],
    *,
    generator: Generator,
) -> DirectionalConditions:

    conditions: Conditions = _build_conditions(
        source,
        DIRECTIONAL_CONDITION_NAMES,
        registry,
        recompute_pair_features,
        generator=generator,
    )
    return DirectionalConditions(
        batches=conditions.batches, metadata=conditions.metadata
    )

def pack_conditions(conditions: DirectionalConditions) -> PackedConditions:

    names = tuple(conditions.batches)
    missing = [n for n in DIRECTIONAL_CONDITION_NAMES if n not in names]
    if missing:
        raise ConditionPackingError(f"missing condition(s) for packing: {missing}")

    ordered = [conditions.batches[name] for name in DIRECTIONAL_CONDITION_NAMES]
    batch_size = _batch_size(ordered[0])

    packed = _pack_tree(ordered, batch_size, path="")

    presence = torch.tensor(
        [
            list(PRESENCE_BY_CONDITION[name])
            for name in DIRECTIONAL_CONDITION_NAMES
            for _ in range(batch_size)
        ],
        dtype=torch.long,
    )
    packed[PRESENCE_KEY] = presence

    slices = {
        name: slice(index * batch_size, (index + 1) * batch_size)
        for index, name in enumerate(DIRECTIONAL_CONDITION_NAMES)
    }
    return PackedConditions(batch=packed, slices=slices, batch_size=batch_size)

def unpack_tensor(tensor: Tensor, packed: PackedConditions, name: str) -> Tensor:

    if name not in packed.slices:
        raise ConditionPackingError(
            f"unknown condition {name!r}; packed conditions are {sorted(packed.slices)}"
        )
    if tensor.shape[0] != packed.packed_size:
        raise ConditionPackingError(
            f"leading dimension {tensor.shape[0]} does not match the packed size "
            f"{packed.packed_size}; this tensor was not produced by the packed batch"
        )
    return tensor[packed.slices[name]]

def _batch_size(batch: Mapping) -> int:
    mask = batch.get("mask")
    if not isinstance(mask, Tensor) or mask.dim() < 1:
        raise ConditionPackingError("cannot determine batch size: no 2-D boolean 'mask'")
    return int(mask.shape[0])

def _pack_tree(nodes: list, batch_size: int, path: str) -> dict:

    first = nodes[0]

    if isinstance(first, Mapping):
        keys = set(first)
        for other in nodes[1:]:
            if set(other) != keys:
                raise ConditionPackingError(
                    f"conditions disagree on the keys at {path or '<root>'!r}: "
                    f"{sorted(keys ^ set(other))}"
                )
        return {
            key: _pack_tree([node[key] for node in nodes], batch_size, f"{path}.{key}")
            for key in sorted(keys)
        }

    if isinstance(first, Tensor):
        for other in nodes[1:]:
            if other.shape[1:] != first.shape[1:] or other.dtype != first.dtype:
                raise ConditionPackingError(
                    f"conditions disagree on shape/dtype at {path!r}: "
                    f"{tuple(first.shape)}/{first.dtype} vs "
                    f"{tuple(other.shape)}/{other.dtype}"
                )
        if first.shape and first.shape[0] == batch_size:

            return torch.cat([node.clone() for node in nodes], dim=0)

        for other in nodes[1:]:
            if not torch.equal(other, first):
                raise ConditionPackingError(
                    f"non-batch tensor at {path!r} varies between conditions; there is "
                    f"no unambiguous way to pack it"
                )
        return first.clone()

    if isinstance(first, (list, tuple)):
        return type(first)(
            _pack_tree([node[i] for node in nodes], batch_size, f"{path}.{i}")
            for i in range(len(first))
        )

    for other in nodes[1:]:
        if other != first:
            raise ConditionPackingError(
                f"non-tensor input at {path!r} varies between conditions "
                f"({first!r} vs {other!r}); packing would silently apply one "
                f"condition's setting to all three"
            )
    return first
