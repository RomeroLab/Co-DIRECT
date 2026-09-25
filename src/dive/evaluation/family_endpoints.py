
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

FAMILIES = ("ame", "binder", "antibody")

C_FINAL_ROLE = {
    "ame": "C_refold_scaffold",
    "binder": ("S_binder", "C_fold", "C_iface"),
    "antibody": "C_framework_refold",
}

PRIMARY_METRIC = "J_final"
PRESERVATION_METRICS = ("S", "J_gen")

def _as_bool(value):
    if value is None:
        return None
    if value is True or value is False:
        return value
    raise ValueError(f"endpoint must be bool or None, got {type(value)!r}")

def conjunction(*values):

    parsed = [_as_bool(v) for v in values]
    if any(v is None for v in parsed):
        return None
    return all(parsed)

def j_gen_f(row: Mapping, family: str):

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if family == "ame":
        return conjunction(row.get("G"), row.get("S"), row.get("C_gen"))
    if family == "binder":
        return conjunction(row.get("G"), row.get("C_epitope_generated"), row.get("C_clash_generated"))
    return conjunction(row.get("G"), row.get("C_gen"))

def j_final_f(row: Mapping, family: str):

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if family == "ame":
        c_final = row.get("C_final", row.get("C_refold_scaffold"))
        return conjunction(row.get("G"), row.get("S"), c_final)
    if family == "binder":
        return conjunction(row.get("G"), row.get("S"), row.get("C_fold"), row.get("C_iface"))
    c_final = row.get("C_final", row.get("C_framework_refold"))
    return conjunction(row.get("G"), row.get("S"), c_final)

def rate(rows: Sequence[Mapping], metric_values: Sequence) -> dict:
    n = len(rows)
    if n != len(metric_values):
        raise ValueError("rate requires one value per requested slot")
    known = sum(v is not None for v in metric_values)
    success = sum(v is True for v in metric_values)
    unknown = n - known
    return {
        "successes": success,
        "requested": n,
        "known": known,
        "unknown": unknown,
        "confirmed_rate": success / n if n else None,
        "dropped": 0,
    }

def reject_pooled_primary(claim: Mapping) -> None:

    primary = claim.get("primary")
    if primary in {"pooled", "pooled_three_family", "three_family_mean"}:
        raise ValueError("pooled three-family scalar is not a primary endpoint")
    if claim.get("families_pooled_as_primary") is True:
        raise ValueError("pooled three-family scalar is not a primary endpoint")
    if primary is not None and primary not in {PRIMARY_METRIC, "J_final_f"}:
        if primary in PRESERVATION_METRICS or primary in {"J_gen", "S_f", "J_gen_f"}:
            raise ValueError(f"{primary} cannot satisfy the primary")

def parent_resampled_effect(
    candidate_rows: Sequence[Mapping],
    control_rows: Sequence[Mapping],
    values_candidate: Sequence,
    values_control: Sequence,
    *,
    replicates: int = 10000,
    seed: int = 20260912,
) -> dict:

    if len(candidate_rows) != len(control_rows) or len(candidate_rows) != len(values_candidate):
        raise ValueError("paired contrast requires identical requested slot sets")
    if len(values_control) != len(values_candidate):
        raise ValueError("paired contrast requires identical requested slot sets")
    grouped = defaultdict(list)
    unknown = 0
    for cand, ctrl, av, bv in zip(candidate_rows, control_rows, values_candidate, values_control):
        pa, pb = cand.get("parent_id"), ctrl.get("parent_id")
        if not pa or pa != pb:
            raise ValueError("audited parent_id required for parent uncertainty")
        if (cand.get("task"), cand.get("seed")) != (ctrl.get("task"), ctrl.get("seed")):
            raise ValueError("paired slots mismatch")
        delta = float(av is True) - float(bv is True)
        grouped[pa].append(delta)
        if av is None or bv is None:
            unknown += 1
    means = np.array([np.mean(v) for v in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(len(means), size=(replicates, len(means)))].mean(1)
    lo, hi = np.quantile(boot, [0.025, 0.975]).tolist()
    return {
        "effect": float(means.mean()) if len(means) else None,
        "ci95": [lo, hi] if len(means) else [None, None],
        "independent_units": int(len(means)),
        "paired_slots": int(len(candidate_rows)),
        "unknown_pair_slots": int(unknown),
        "bootstrap_seed": seed,
        "replicates": replicates,
        "target_supported": bool(len(means) and lo >= 0.10),
        "preservation_supported": None,
    }

def preservation_supported(s_ci_lo: float, j_gen_ci_lo: float) -> bool:
    return s_ci_lo >= -0.05 and j_gen_ci_lo >= -0.05
