
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260912

@dataclass(frozen=True)
class Primary:

    key: str
    direction: int
    unit: str
    source: str

    container: str | None = None

FAMILY_PRIMARY: Mapping[str, Primary] = {
    "binder": Primary(
        key="self_binder_scRMSD_ca",
        direction=-1,
        unit="angstrom",
        source="own-sequence refold vs designed binder backbone (C_fold quantity)",
        container="metrics",
    ),
    "ame": Primary(
        key="refold_required_atom_rmsd_in_scaffold",
        direction=-1,
        unit="angstrom",
        source="required-atom RMSD in the scaffold frame (C_final quantity)",
    ),
    "antibody": Primary(
        key="self_consistency_ca_rmsd",
        direction=-1,
        unit="angstrom",
        source="own-sequence refold vs designed CA trace (S quantity)",
    ),
}

SECONDARY = {
    "binder": {
        "self_complex_i_pAE_angstrom": -1,
        "self_complex_pLDDT": +1,
        "self_complex_i_pTM": +1,
        "self_complex_scRMSD_ca": -1,
    },
    "ame": {
        "refold_required_atom_rmsd_released": -1,
        "self_consistency_ca_rmsd": -1,
        "refold_plddt": +1,
    },
    "antibody": {
        "refold_plddt": +1,
        "framework_rmsd_refold_A": -1,
        "cn_fraction": +1,
        "caca_fraction": +1,
    },
}

RATIO_UNSUITABLE = {
    "refold_plddt": "bounded in [0,1]",
    "self_complex_pLDDT": "bounded in [0,1]",
    "self_complex_i_pTM": "bounded in [0,1]",
    "cn_fraction": "bounded in [0,1]",
    "caca_fraction": "bounded in [0,1]",
    "self_complex_i_pAE_angstrom": "capped at 31 A; baseline already near the cap",
    "designed_invalid_excess": "can be zero or negative; ratio undefined",
}

def _finite(value) -> bool:

    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float, np.floating, np.integer)):
        return False
    return math.isfinite(float(value))

def per_parent_means(slots: Iterable[Mapping]) -> dict[str, float]:

    buckets: dict[str, list[float]] = {}
    for row in slots:
        if not _finite(row.get("value")):
            continue
        buckets.setdefault(str(row["parent_id"]), []).append(float(row["value"]))
    return {parent: float(np.mean(vals)) for parent, vals in buckets.items()}

def arm_mean(slots: Iterable[Mapping]) -> float | None:

    means = per_parent_means(slots)
    if not means:
        return None
    return float(np.mean(list(means.values())))

def unscorable(slots: Iterable[Mapping]) -> int:
    return sum(1 for row in slots if not _finite(row.get("value")))

def relative_improvement(
    baseline_mean: float | None, candidate_mean: float | None, direction: int
) -> float | None:

    if baseline_mean is None or candidate_mean is None:
        return None
    if not math.isfinite(baseline_mean) or not math.isfinite(candidate_mean):
        return None
    if baseline_mean == 0.0:
        return None
    if direction < 0:
        return float((baseline_mean - candidate_mean) / baseline_mean)
    return float((candidate_mean - baseline_mean) / baseline_mean)

def _key(row: Mapping) -> tuple:
    return (str(row["task"]), int(row["seed"]))

def paired_relative_improvement(
    baseline: Sequence[Mapping],
    candidate: Sequence[Mapping],
    *,
    direction: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:

    base_by_key = {_key(r): r for r in baseline}
    cand_by_key = {_key(r): r for r in candidate}
    shared = sorted(set(base_by_key) & set(cand_by_key))
    paired_base, paired_cand = [], []
    dropped = 0
    for key in shared:
        b, c = base_by_key[key], cand_by_key[key]
        if _finite(b.get("value")) and _finite(c.get("value")):
            paired_base.append(b)
            paired_cand.append(c)
        else:
            dropped += 1
    dropped += len(set(base_by_key) ^ set(cand_by_key))

    bm = per_parent_means(paired_base)
    cm = per_parent_means(paired_cand)
    parents = sorted(set(bm) & set(cm))
    baseline_mean = float(np.mean([bm[p] for p in parents])) if parents else None
    candidate_mean = float(np.mean([cm[p] for p in parents])) if parents else None
    point = relative_improvement(baseline_mean, candidate_mean, direction)

    ci: list[float | None] = [None, None]
    if parents and point is not None:
        b_arr = np.array([bm[p] for p in parents], dtype=float)
        c_arr = np.array([cm[p] for p in parents], dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(len(parents), size=(int(replicates), len(parents)))
        b_boot = b_arr[idx].mean(axis=1)
        c_boot = c_arr[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            if direction < 0:
                r_boot = (b_boot - c_boot) / b_boot
            else:
                r_boot = (c_boot - b_boot) / b_boot
        r_boot = r_boot[np.isfinite(r_boot)]
        if r_boot.size:
            lo, hi = np.quantile(r_boot, [0.025, 0.975])
            ci = [float(lo), float(hi)]

    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "relative_improvement": point,
        "ci95": ci,
        "n_paired_slots": len(paired_base),
        "n_parents": len(parents),
        "dropped_slots": dropped,
        "baseline_unscorable": unscorable(baseline),
        "candidate_unscorable": unscorable(candidate),
        "baseline_own_set_mean": arm_mean(baseline),
        "candidate_own_set_mean": arm_mean(candidate),
        "bootstrap": {"unit": "parent", "replicates": int(replicates), "seed": int(seed)},
    }
