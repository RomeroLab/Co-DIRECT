
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

class PerExampleError(ValueError):
    pass

def group_by_parent(
    example_ids: Sequence[str], parents: Mapping[str, str]
) -> list[str]:

    missing = sorted({str(e) for e in example_ids if str(e) not in parents})
    if missing:
        raise PerExampleError(
            f"{len(missing)} example(s) have no parent in the manifest "
            f"({missing[:3]}...); falling back to the example id would invent "
            f"independent clusters and shrink every interval"
        )
    return [str(parents[str(e)]) for e in example_ids]

def equal_family_relative(
    numerators: Mapping[str, Sequence[float]],
    denominators: Mapping[str, Sequence[float]],
) -> float:

    ratios = []
    for family in sorted(numerators):
        top, bottom = numerators[family], denominators[family]
        if not top or not bottom:
            raise PerExampleError(
                f"family {family!r} contributed no examples; an equal-family "
                f"score with a family missing is not the declared metric"
            )
        total = float(sum(bottom))
        if total <= 0:
            raise PerExampleError(f"family {family!r} has a non-positive denominator")
        ratios.append(float(sum(top)) / total)
    return sum(ratios) / len(ratios)

def per_family_bootstrap(
    routed: Sequence[float],
    reference: Sequence[float],
    *,
    parents: Sequence[str],
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:

    from dive.evaluation.fixed_regimes import target_bootstrap_ratio_interval

    if not (len(routed) == len(reference) == len(parents)) or not routed:
        raise PerExampleError(
            f"routed, reference and parents must share a non-zero length; got "
            f"{len(routed)}, {len(reference)}, {len(parents)}"
        )

    total_reference = float(sum(reference))
    if total_reference <= 0:
        raise PerExampleError("reference total must be positive to form a ratio")

    point = 1.0 - float(sum(routed)) / total_reference
    low, high = target_bootstrap_ratio_interval(
        routed, reference, target_ids=parents, resamples=resamples, seed=seed
    )
    return {
        "point": point,
        "low": low,
        "high": high,
        "parents": len(set(map(str, parents))),
        "examples": len(routed),
        "resamples": resamples,
    }

def parent_map(evidence_root, families) -> dict[str, str]:

    import pandas as pd

    mapping: dict[str, str] = {}
    for family in families:
        path = evidence_root / "manifests" / f"{family}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no manifest for family {family!r} at {path}")
        frame = pd.read_parquet(path, columns=["example_id", "parent_id"])
        mapping.update(
            {str(a): str(b) for a, b in zip(frame.example_id, frame.parent_id)}
        )
    return mapping
