
from __future__ import annotations

import math
import random
from typing import Sequence

__all__ = [
    "StatisticsError",
    "bootstrap_paired_difference",
    "success_rate",
    "wilson_interval",
]

MIN_INFORMATIVE_PAIRS = 2

DEFAULT_RESAMPLES = 10000

class StatisticsError(ValueError):
    pass

def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):

    if n <= 0:
        raise StatisticsError("cannot form an interval on an empty cohort")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def success_rate(outcomes: Sequence[bool | None]):

    if not outcomes:
        raise StatisticsError("cannot compute a success rate over an empty cohort")
    n = len(outcomes)
    successes = sum(1 for o in outcomes if o is True)
    unevaluable = sum(1 for o in outcomes if o is None)
    low, high = wilson_interval(successes, n)
    return {
        "n": n,
        "successes": successes,
        "n_unevaluable": unevaluable,
        "rate": successes / n,
        "ci_low": low,
        "ci_high": high,
        "ci_method": "wilson_95",
    }

def bootstrap_paired_difference(
    arm_a: Sequence[float | None],
    arm_b: Sequence[float | None],
    *,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
):

    if len(arm_a) != len(arm_b):
        raise StatisticsError(
            f"arms must be aligned target by target; got {len(arm_a)} and {len(arm_b)}"
        )
    pairs = [
        (a, b) for a, b in zip(arm_a, arm_b) if a is not None and b is not None
    ]
    dropped = len(arm_a) - len(pairs)
    if not pairs:
        raise StatisticsError("no target has a value in both arms")

    diffs = [a - b for a, b in pairs]
    observed = sum(diffs) / len(diffs)

    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]

    return {
        "n_pairs": n,
        "n_dropped": dropped,
        "observed": observed,
        "ci_low": lo,
        "ci_high": hi,
        "ci_method": f"paired_bootstrap_{resamples}",
        "seed": seed,
        "inconclusive": n < MIN_INFORMATIVE_PAIRS,
    }
