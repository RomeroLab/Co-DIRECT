
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dive.phase3_verify.q2_use import InterventionKind, Q2Error, Q2Reading

BOOTSTRAP = 2000
MIN_TASKS = 5
SEED = 1401

def _interval(values: Sequence[float]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    arr = np.asarray(values, dtype=float)
    draws = np.median(rng.choice(arr, size=(BOOTSTRAP, len(arr))), axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

def intervention_reading(
    family: str,
    *,
    ratios: Sequence[float],
    nulls: Sequence[float],
    total_steps: int,
) -> Q2Reading:

    if not nulls:
        raise Q2Error(
            "no null arm was run; an absent null is not an inert one"
        )
    if len(ratios) < MIN_TASKS:
        raise Q2Error(
            f"{len(ratios)} tasks is too few to bootstrap an interval; "
            f"at least {MIN_TASKS} are needed"
        )
    low, high = _interval(ratios)
    if len(nulls) == 1 or len(set(nulls)) == 1:
        n_low = n_high = float(nulls[0])
    else:
        n_low, n_high = _interval(nulls)
    return Q2Reading(
        family=family,
        kind=InterventionKind.MAGNITUDE_PRESERVING_SCRAMBLE,

        read_steps=tuple(range(total_steps)),
        total_steps=total_steps,
        effect_ci_low=low,
        effect_ci_high=high,
        null_ci_low=n_low,
        null_ci_high=n_high,
    )
