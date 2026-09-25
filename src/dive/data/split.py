
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from dive.data.contracts import Family, Partition
from dive.data.similarity import ExactGroup, NeighborIndex, SimilarityEdge

class SplitError(RuntimeError):
    pass

class UnderpoweredSplitError(SplitError):
    pass

SCORED_PARTITIONS = frozenset(
    {Partition.TRAIN, Partition.VALIDATION, Partition.TEST_BLIND}
)

@dataclass(frozen=True, slots=True)
class SplitBundle:

    assignments: dict[str, Partition]
    reasons: dict[str, str]
    views: dict[str, dict[str, bool]]
    families: dict[str, Family]
    seed: int
    excluded_edges: tuple[SimilarityEdge, ...] = field(default=())

    def partition_of(self, parent_id: str) -> Partition:
        try:
            return self.assignments[parent_id]
        except KeyError:
            raise SplitError(f"{parent_id!r} was never assigned") from None

    def view_of(self, parent_id: str) -> dict[str, bool]:
        return self.views[parent_id]

    def partitions(self, partition: Partition) -> frozenset[str]:
        return frozenset(k for k, v in self.assignments.items() if v is partition)

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for parent_id, partition in self.assignments.items():
            family = str(self.families[parent_id])
            out.setdefault(family, {}).setdefault(str(partition), 0)
            out[family][str(partition)] += 1
        return out

def assign_splits(
    groups: Sequence[ExactGroup],
    edges: Sequence[SimilarityEdge],
    *,
    legacy: Iterable[str] = (),
    outer_test: Iterable[str] = (),
    outer_train: Iterable[str] = (),
    requested_test: Iterable[str] = (),
    seed: int = 20260819,
    validation_fraction: float = 0.10,
    min_strict_groups: int = 20,
    temporal_clean: Iterable[str] = (),
) -> SplitBundle:

    by_id = {group.parent_id: group for group in groups}
    if len(by_id) != len(groups):
        raise SplitError("duplicate parent_id among the input groups")

    legacy = set(legacy)
    outer_test = set(outer_test)
    outer_train = set(outer_train)
    requested_test = set(requested_test)
    temporal_clean = set(temporal_clean)

    both = outer_test & outer_train
    if both:
        raise SplitError(f"group(s) in both official sets: {sorted(both)[:5]}")

    unknown = (legacy | outer_test | outer_train | requested_test) - set(by_id)
    if unknown:
        raise SplitError(f"unknown parent group(s): {sorted(unknown)[:5]}")

    edges = list(edges)

    index = NeighborIndex(edges)

    illegal = requested_test & outer_train
    if illegal:
        raise SplitError(
            f"official train group(s) requested as blind test: {sorted(illegal)[:5]}"
        )
    illegal = requested_test & legacy
    if illegal:
        raise SplitError(f"legacy group(s) requested as blind test: {sorted(illegal)[:5]}")

    for candidate in sorted(requested_test):
        touching = index.neighbors(candidate) & legacy
        if touching:
            raise SplitError(
                f"{candidate!r} is a legacy neighbor of {sorted(touching)[:3]} and cannot "
                f"be a blind-test anchor"
            )

    blind = set(outer_test) | set(requested_test)

    assignments: dict[str, Partition] = {}
    reasons: dict[str, str] = {}

    def place(parent_id: str, partition: Partition, reason: str) -> None:
        if parent_id in assignments:
            return
        assignments[parent_id] = partition
        reasons[parent_id] = reason

    for parent_id in sorted(blind):
        origin = "official outer test" if parent_id in outer_test else "newly selected anchor"
        place(parent_id, Partition.TEST_BLIND, f"blind anchor ({origin})")

    for parent_id in sorted(legacy):
        place(parent_id, Partition.LEGACY_DEV, "previously inspected during project development")

    blind_neighbors = set(index.neighbors_of_any(blind))
    for parent_id in sorted(blind_neighbors - blind):
        place(
            parent_id,
            Partition.EXCLUDED_TEST_NEIGHBOR,
            "directly conflicts with a blind-test anchor",
        )

    remaining = [g for g in sorted(by_id) if g not in assignments]
    by_family: dict[Family, list[str]] = {}
    for parent_id in remaining:
        by_family.setdefault(by_id[parent_id].family, []).append(parent_id)

    rng = random.Random(seed)
    validation: set[str] = set()
    for family in sorted(by_family, key=str):
        candidates = sorted(by_family[family])
        wanted = int(round(len(candidates) * validation_fraction))

        chosen = rng.sample(candidates, min(wanted, len(candidates))) if wanted else []
        validation.update(chosen)

    for parent_id in sorted(validation):
        place(parent_id, Partition.VALIDATION, "seeded, family-stratified validation anchor")

    validation_neighbors = set(index.neighbors_of_any(validation))
    for parent_id in sorted(validation_neighbors - validation):
        place(
            parent_id,
            Partition.EXCLUDED_VAL_NEIGHBOR,
            "directly conflicts with a validation anchor",
        )

    for parent_id in sorted(by_id):
        place(parent_id, Partition.TRAIN, "no conflict with any anchor")

    views = _views(by_id, assignments, index, legacy, temporal_clean)
    families = {parent_id: group.family for parent_id, group in by_id.items()}

    bundle = SplitBundle(
        assignments=assignments,
        reasons=reasons,
        views=views,
        families=families,
        seed=seed,
        excluded_edges=tuple(sorted(edges)),
    )

    if min_strict_groups:
        _assert_powered(bundle, min_strict_groups)
    return bundle

def _views(
    by_id: dict[str, ExactGroup],
    assignments: dict[str, Partition],
    index: NeighborIndex,
    legacy: set[str],
    temporal_clean: set[str],
) -> dict[str, dict[str, bool]]:

    views: dict[str, dict[str, bool]] = {}
    for parent_id in by_id:
        partition = assignments[parent_id]
        in_blind = partition is Partition.TEST_BLIND
        touches_legacy = bool(index.neighbors(parent_id) & legacy)
        strict = in_blind and not touches_legacy and parent_id not in legacy
        views[parent_id] = {
            "standard": in_blind,
            "strict": strict,

            "temporal_clean": strict and parent_id in temporal_clean,
        }
    return views

def _assert_powered(bundle: SplitBundle, minimum: int) -> None:

    per_family: dict[str, dict[str, int]] = {}
    for parent_id, partition in bundle.assignments.items():
        family = str(bundle.families[parent_id])
        stats = per_family.setdefault(family, {"validation": 0, "strict_blind": 0})
        if partition is Partition.VALIDATION:
            stats["validation"] += 1
        if bundle.views[parent_id]["strict"]:
            stats["strict_blind"] += 1

    shortfalls = [
        f"{family}: {stats['validation']} validation, {stats['strict_blind']} strict blind"
        for family, stats in sorted(per_family.items())
        if stats["validation"] < minimum or stats["strict_blind"] < minimum
    ]
    if shortfalls:
        raise UnderpoweredSplitError(
            f"below the {minimum}-group floor -- repair data or narrow the family claim, "
            f"never lower the floor: " + "; ".join(shortfalls)
        )
