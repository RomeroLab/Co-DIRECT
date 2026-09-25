
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

FAMILY_ORDER: tuple[str, ...] = ("binder", "ame", "antibody")

FORBIDDEN_MODEL_KEYS = frozenset(
    {"family", "example_ids", "cdr_h3", "catalytic", "interface", "task_id"}
)

class MixtureError(RuntimeError):
    pass

@dataclass(slots=True)
class FamilyBatch:

    family: str
    example_ids: tuple[str, ...]
    model_batch: dict[str, Any]

@dataclass(frozen=True, slots=True)
class BatchMetadata:

    family: str
    example_ids: tuple[str, ...]

def unwrap_family_batch(batch: FamilyBatch) -> tuple[BatchMetadata, dict[str, Any]]:

    present = sorted(FORBIDDEN_MODEL_KEYS & set(batch.model_batch))
    if present:
        raise MixtureError(
            f"semantic key(s) {present} in the model batch; family and region "
            f"identity travel beside the batch, never inside it"
        )
    return (
        BatchMetadata(family=batch.family, example_ids=batch.example_ids),
        dict(batch.model_batch),
    )

class BalancedFamilyLoader:

    def __init__(
        self,
        loaders: Mapping[str, Any],
        *,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
    ) -> None:
        missing = [family for family in FAMILY_ORDER if family not in loaders]
        if missing:
            raise MixtureError(
                f"missing loader(s) for {missing}; a two-family mixture is a different "
                f"experiment and must be declared, not arrived at"
            )
        unknown = sorted(set(loaders) - set(FAMILY_ORDER))
        if unknown:
            raise MixtureError(f"unknown family loader(s): {unknown}")

        self._loaders = dict(loaders)
        self._seed = seed
        self._rank = rank
        self._world_size = world_size
        self._epoch = epoch

    def set_epoch(self, epoch: int) -> None:

        self._epoch = epoch
        for loader in self._loaders.values():
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[FamilyBatch]:

        offset = self._epoch % len(FAMILY_ORDER)
        order = FAMILY_ORDER[offset:] + FAMILY_ORDER[:offset]

        streams = {family: iter(self._loaders[family]) for family in order}
        step = 0
        while True:
            family = order[step % len(order)]
            try:
                record = next(streams[family])
            except StopIteration:

                streams[family] = iter(self._loaders[family])
                try:
                    record = next(streams[family])
                except StopIteration:
                    raise MixtureError(f"loader for {family!r} yields nothing") from None
            step += 1
            yield _to_family_batch(family, record)

def _to_family_batch(family: str, record: Mapping[str, Any]) -> FamilyBatch:

    payload = dict(record)
    example_ids = payload.pop("example_id", None) or payload.pop("example_ids", None)
    if example_ids is None:
        raise MixtureError(f"{family!r} loader produced a record with no example id")
    if isinstance(example_ids, str):
        example_ids = (example_ids,)
    return FamilyBatch(
        family=family, example_ids=tuple(example_ids), model_batch=payload
    )
