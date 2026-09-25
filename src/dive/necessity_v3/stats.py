
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

@dataclass(frozen=True)
class BootstrapInterval:

    point: float
    low: float
    high: float
    n_clusters: int
    n_rows: int

def _clean(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).ravel()
    clusters = np.asarray(clusters).ravel()
    keep = np.isfinite(values)
    return values[keep], clusters[keep]

def cluster_bootstrap(
    *,
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapInterval:

    values, clusters = _clean(values, clusters)
    if values.size == 0:
        return BootstrapInterval(np.nan, np.nan, np.nan, 0, 0)

    unique, index = np.unique(clusters, return_inverse=True)
    n_clusters = unique.size
    point = float(values.mean())
    if n_clusters < 2:
        return BootstrapInterval(point, np.nan, np.nan, n_clusters, values.size)

    sums = np.bincount(index, weights=values, minlength=n_clusters)
    counts = np.bincount(index, minlength=n_clusters).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapInterval(point, float(low), float(high), n_clusters, values.size)

def harmful_fraction(values: np.ndarray) -> float:

    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float((finite < 0).sum() / finite.size)

def weighted_cell_mean(
    values: np.ndarray,
    cells: np.ndarray,
    *,
    weights: Mapping[int, float] | None,
    counts: np.ndarray | None = None,
) -> float:

    values = np.asarray(values, dtype=float).ravel()
    cells = np.asarray(cells).ravel()
    keep = np.isfinite(values)
    values, cells = values[keep], cells[keep]
    if values.size == 0:
        return float("nan")

    unique = np.unique(cells)
    means = np.array([values[cells == c].mean() for c in unique], dtype=float)
    if weights is None:
        return float(means.mean())
    w = np.array([float(weights.get(int(c), 0.0)) for c in unique], dtype=float)
    total = w.sum()
    if total <= 0:
        return float(means.mean())
    return float((means * w).sum() / total)

def leave_one_out(*, values: np.ndarray, clusters: np.ndarray) -> dict:

    values, clusters = _clean(values, clusters)
    if values.size == 0:
        return {}

    unique, index = np.unique(clusters, return_inverse=True)
    sums = np.bincount(index, weights=values, minlength=unique.size)
    counts = np.bincount(index, minlength=unique.size)
    total_sum, total_n = float(values.sum()), int(values.size)
    out = {}
    for k, cluster in enumerate(unique):
        remaining = total_n - int(counts[k])
        key = cluster.item() if hasattr(cluster, "item") else cluster
        out[key] = ((total_sum - float(sums[k])) / remaining
                    if remaining > 0 else float("nan"))
    return out

def most_influential_cluster(*, values: np.ndarray, clusters: np.ndarray):

    values, clusters = _clean(values, clusters)
    if values.size == 0:
        return None
    full = values.mean()
    deltas = leave_one_out(values=values, clusters=clusters)
    if not deltas:
        return None
    return max(deltas, key=lambda k: abs(deltas[k] - full))

def sign_survives_leave_one_out(*, values: np.ndarray, clusters: np.ndarray) -> bool:

    values, clusters = _clean(values, clusters)
    if values.size == 0:
        return False
    if np.unique(clusters).size < 2:
        return False
    full = values.mean()
    if full == 0.0:
        return False
    return all(
        np.sign(v) == np.sign(full)
        for v in leave_one_out(values=values, clusters=clusters).values()
    )

@dataclass(frozen=True)
class MatchedDifference:

    difference: float
    n_strata_used: int
    n_strata_dropped: int
    n_rows: int

def decile_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:

    values = np.asarray(values, dtype=float).ravel()
    out = np.full(values.shape, -1, dtype=int)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    good = values[finite]
    edges = np.unique(np.quantile(good, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        out[finite] = 0
        return out
    out[finite] = np.clip(np.searchsorted(edges, good, side="right") - 1,
                          0, edges.size - 2)
    return out

def matched_difference(
    *, values: np.ndarray, group: np.ndarray, strata: np.ndarray
) -> MatchedDifference:

    values = np.asarray(values, dtype=float).ravel()
    group = np.asarray(group).ravel().astype(bool)
    strata = np.asarray(strata).ravel()
    keep = np.isfinite(values)
    values, group, strata = values[keep], group[keep], strata[keep]

    if values.size == 0:
        return MatchedDifference(float("nan"), 0, 0, 0)

    _, codes = np.unique(strata, return_inverse=True)
    n_strata = int(codes.max()) + 1 if codes.size else 0
    slot = codes * 2 + group.astype(np.int64)
    sums = np.bincount(slot, weights=values, minlength=2 * n_strata)
    counts = np.bincount(slot, minlength=2 * n_strata)
    control_n, treated_n = counts[0::2], counts[1::2]
    control_s, treated_s = sums[0::2], sums[1::2]

    usable = (control_n > 0) & (treated_n > 0)
    dropped = int((~usable).sum())
    diffs = (treated_s[usable] / treated_n[usable]
             - control_s[usable] / control_n[usable])
    weights = np.minimum(control_n[usable], treated_n[usable]).astype(float)

    if diffs.size == 0:
        return MatchedDifference(float("nan"), 0, dropped, int(values.size))
    return MatchedDifference(
        float((diffs * weights).sum() / weights.sum()),
        int(diffs.size), dropped, int(values.size),
    )
