
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

def split_parents(
    groups: Sequence[str], *, seed: int
) -> tuple[list[bool], list[bool]]:

    parents = sorted(set(groups))
    if len(parents) < 2:
        raise ValueError(
            f"{len(parents)} parent(s); a selection/assessment split needs at "
            "least 2, or the two sides share targets"
        )

    def side(parent: str) -> bool:
        digest = hashlib.sha256(f"{seed}:{parent}".encode()).digest()
        return digest[0] % 2 == 0

    chosen = {p: side(p) for p in parents}

    if all(chosen.values()) or not any(chosen.values()):
        for index, parent in enumerate(parents):
            chosen[parent] = index % 2 == 0

    selection = [chosen[g] for g in groups]
    assessment = [not s for s in selection]
    return selection, assessment

def best_regime_on(
    regime_losses: Mapping[str, Sequence[float]], mask: Sequence[bool]
) -> str:

    if not any(mask):
        raise ValueError("empty mask; no rows to select a regime on")
    means = {}
    for name, values in regime_losses.items():
        rows = [v for v, keep in zip(values, mask) if keep]
        means[name] = sum(rows) / len(rows)
    return min(sorted(means), key=lambda name: means[name])
