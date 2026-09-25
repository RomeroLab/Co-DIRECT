
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

EXCLUDED_TARGETS: tuple[str, ...] = (
    "31_IL7RA_FIX",
    "32_PDL1_ALPHA_FIX",
    "38_TNFalpha_FIX",
)

class SplitError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class Split:

    dev_groups: tuple[str, ...]
    heldout_groups: tuple[str, ...]
    dev_targets: tuple[str, ...]
    heldout_targets: tuple[str, ...]
    seed: int

    def to_dict(self) -> dict[str, object]:

        return {
            "seed": self.seed,
            "dev_groups": list(self.dev_groups),
            "heldout_groups": list(self.heldout_groups),
            "dev_targets": list(self.dev_targets),
            "heldout_targets": list(self.heldout_targets),
            "grouped_by": "pdb_id (falling back to target_filename)",
        }

def target_groups(targets: Mapping[str, Mapping]) -> dict[str, tuple[str, ...]]:

    groups: dict[str, list[str]] = {}
    for name in sorted(targets):
        if name in EXCLUDED_TARGETS:
            continue
        key = targets[name].get("pdb_id") or targets[name].get("target_filename")
        if not key:
            raise SplitError(f"{name} carries neither pdb_id nor target_filename")
        groups.setdefault(str(key), []).append(name)
    return {key: tuple(value) for key, value in sorted(groups.items())}

def split_groups(
    groups: Mapping[str, Sequence[str]], *, seed: int, dev_fraction: float = 2.0 / 3.0
) -> Split:

    names = sorted(groups)
    if len(names) < 2:
        raise SplitError("a split needs at least two protein groups")
    if not 0.0 < dev_fraction < 1.0:
        raise SplitError(f"dev_fraction must be in (0, 1), observed {dev_fraction}")

    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, round(dev_fraction * len(shuffled))))
    dev, heldout = sorted(shuffled[:cut]), sorted(shuffled[cut:])

    return Split(
        dev_groups=tuple(dev),
        heldout_groups=tuple(heldout),
        dev_targets=tuple(t for g in dev for t in groups[g]),
        heldout_targets=tuple(t for g in heldout for t in groups[g]),
        seed=int(seed),
    )

def representative_targets(
    dev_targets: Sequence[str], groups: Mapping[str, Sequence[str]], count: int
) -> tuple[str, ...]:

    if count <= 0:
        raise SplitError(f"count must be positive, observed {count}")
    group_of = {
        target: group for group, members in groups.items() for target in members
    }
    selected: list[str] = []
    seen: set[str] = set()
    for target in dev_targets:
        group = group_of.get(str(target))
        if group is None or group in seen:
            continue
        seen.add(group)
        selected.append(str(target))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise SplitError(
            f"count={count} exceeds the {len(selected)} distinct protein "
            "groups present in the given development targets"
        )
    return tuple(selected)
