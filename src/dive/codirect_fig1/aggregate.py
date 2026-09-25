
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

LADDER = ("parent_id", "example_id", "schedule_index", "draw")

N_BOOT = 10_000
SEED = 20260921

class AggregateError(ValueError):
    pass

def _require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise AggregateError(
            f"missing column(s) {missing}; walking the ladder without them "
            "would silently collapse a stage and reweight every parent"
        )

def parent_means(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:

    cols = list(cols)
    _require(frame, [*LADDER, *cols])
    group = dict(observed=True, sort=False)
    at_state = frame.groupby(list(LADDER), **group)[cols].mean()
    at_index = at_state.groupby(level=["parent_id", "example_id", "schedule_index"],
                                observed=True).mean()
    at_example = at_index.groupby(level=["parent_id", "example_id"],
                                  observed=True).mean()
    at_parent = at_example.groupby(level="parent_id", observed=True).mean()
    if at_parent.isna().any().any():
        bad = at_parent[at_parent.isna().any(axis=1)].index.tolist()
        raise AggregateError(
            f"parent(s) {bad[:3]} came out NaN; empty groups leaked in, so "
            "check that observed=True survived and that no parent is missing "
            "every value"
        )
    return at_parent

def widen_by_policy(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:

    cols = list(cols)
    _require(frame, [*LADDER, "alpha", *cols])
    index = list(LADDER)
    wide = frame.pivot_table(index=index, columns="alpha", values=cols,
                             observed=True, aggfunc="mean")
    wide.columns = [f"{stat}@{alpha:.2f}" for stat, alpha in wide.columns]
    return wide

def paired_parent_bootstrap(
    frame: pd.DataFrame,
    cols: Sequence[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:

    cols = list(cols)
    parents = parent_means(frame, cols)
    values = parents[cols].to_numpy(dtype=float)
    n_parents = values.shape[0]
    point = values.mean(axis=0)
    out: dict = {
        "_parents": list(parents.index),
        "_n_parents": n_parents,
        "_n_boot": n_boot,
        "_seed": seed,
    }
    if n_parents < 2:
        for i, col in enumerate(cols):
            out[col] = {"point": float(point[i]), "low": float("nan"),
                        "high": float("nan"), "n_parents": n_parents}
        out["_replicates"] = {col: np.full(n_boot, np.nan) for col in cols}
        return out

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_parents, size=(n_boot, n_parents))
    reps = values[draws].mean(axis=1)
    low = np.quantile(reps, alpha / 2.0, axis=0)
    high = np.quantile(reps, 1.0 - alpha / 2.0, axis=0)
    for i, col in enumerate(cols):
        out[col] = {"point": float(point[i]), "low": float(low[i]),
                    "high": float(high[i]), "n_parents": n_parents}
    out["_replicates"] = {col: reps[:, i] for i, col in enumerate(cols)}
    return out

def interval_for_contrast(
    result: dict, left: str, right: str, *, alpha: float = 0.05
) -> dict:

    reps = result["_replicates"]
    for name in (left, right):
        if name not in reps:
            raise AggregateError(f"{name!r} was not among the bootstrapped columns")
    diff = reps[left] - reps[right]
    return {
        "point": result[left]["point"] - result[right]["point"],
        "low": float(np.quantile(diff, alpha / 2.0)),
        "high": float(np.quantile(diff, 1.0 - alpha / 2.0)),
        "n_parents": result["_n_parents"],
    }
