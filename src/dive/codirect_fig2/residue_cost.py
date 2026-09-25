
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

EVAL_KEY = ("family", "parent_id", "example_id", "schedule_index", "draw")

RESIDUE_ID = ("example_id", "chain", "residue_pdb_idx")

BLOCKS = ("bb_ca", "local_latents")

THRESHOLDS = (0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2)
THRESHOLD_LABELS = ("0%", "0.01%", "0.1%", "0.5%", "1%", "5%")

NUMERICAL_ZERO = 1e-12

PLANNED_EVALUATIONS = 32

PATTERNS = ("both_better", "both_worse", "backbone_worse_latent_better",
            "backbone_better_latent_worse", "no_change")

class CostError(ValueError):
    pass

def _need(frame: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise CostError(f"{what} is missing {missing}")

def residue_net(rows: pd.DataFrame, *, designed_only: bool = True,
                valid_only: bool = True) -> pd.DataFrame:

    _need(rows, [*EVAL_KEY, "modality", "residue_index", "chain",
                 "residue_pdb_idx", "designed", "delta", "ell_base"],
          "residue rows")
    work = rows
    if valid_only and "valid" in work.columns:
        work = work[work["valid"].astype(bool)]
    if designed_only:
        work = work[work["designed"].astype(bool)]
    if work.empty:
        raise CostError("no designed, valid residue rows to count")

    key = [*EVAL_KEY, "residue_index"]
    dup = work.duplicated(subset=[*key, "modality"]).sum()
    if dup:
        raise CostError(
            f"{dup} (evaluation, residue, component) pairs have more than one "
            "row; the cache key is not unique and every sum would double-count")

    g = work.groupby(key, observed=True, sort=False)
    net = g.agg(
        delta_net=("delta", "sum"),
        ell_base_net=("ell_base", "sum"),
        n_components=("modality", "nunique"),
        chain=("chain", "first"),
        residue_pdb_idx=("residue_pdb_idx", "first"),
        designed=("designed", "first"),
        _chain_n=("chain", "nunique"),
        _pdb_n=("residue_pdb_idx", "nunique"),
    ).reset_index()

    bad = net[(net._chain_n != 1) | (net._pdb_n != 1)]
    if len(bad):
        raise CostError(
            f"{len(bad)} residues disagree on their own identity across "
            "components; the chain or source residue number is not constant, "
            "so the join is not one-to-one")
    return net.drop(columns=["_chain_n", "_pdb_n"])

def state_scale(net: pd.DataFrame) -> pd.DataFrame:

    _need(net, [*EVAL_KEY, "ell_base_net"], "net frame")
    out = net.groupby(list(EVAL_KEY), observed=True, sort=False).agg(
        baseline_total=("ell_base_net", "sum"),
        n_designed=("residue_index", "size"),
    ).reset_index()
    out["mean_residue_error"] = np.where(
        out.n_designed > 0, out.baseline_total / out.n_designed, np.nan)
    out["scale_defined"] = (np.isfinite(out.mean_residue_error)
                            & (out.mean_residue_error > 0))
    out.loc[~out.scale_defined, "mean_residue_error"] = np.nan
    return out

def flag_increases(net: pd.DataFrame, scale: pd.DataFrame, *,
                   thresholds: Sequence[float] = THRESHOLDS,
                   labels: Sequence[str] = THRESHOLD_LABELS) -> pd.DataFrame:

    if len(thresholds) != len(labels):
        raise CostError("thresholds and labels differ in length")
    merged = net.merge(
        scale[[*EVAL_KEY, "mean_residue_error", "scale_defined", "n_designed"]],
        on=list(EVAL_KEY), how="left", validate="many_to_one")
    if merged.mean_residue_error.isna().all() and scale.scale_defined.any():
        raise CostError("the scale did not join to the residue rows")

    merged["magnitude"] = (-merged.delta_net).clip(lower=0.0)
    merged["increase"] = merged.delta_net < 0.0
    merged["improvement"] = merged.delta_net > 0.0
    merged["unchanged"] = merged.delta_net == 0.0
    ok = merged.scale_defined.fillna(False).to_numpy(dtype=bool)
    merged["relative_magnitude"] = np.where(
        ok, merged.magnitude / merged.mean_residue_error.where(ok, 1.0), np.nan)
    for thr, lab in zip(thresholds, labels, strict=True):
        col = f"increase@{lab}"
        if thr == 0.0:

            merged[col] = merged.increase
        else:
            merged[col] = pd.array(
                np.where(ok, merged.relative_magnitude > thr, None),
                dtype="boolean")
    return merged

def component_pattern(rows: pd.DataFrame) -> pd.DataFrame:

    _need(rows, [*EVAL_KEY, "modality", "residue_index", "delta"],
          "residue rows")
    work = rows
    if "valid" in work.columns:
        work = work[work["valid"].astype(bool)]
    work = work[work["designed"].astype(bool)]
    key = [*EVAL_KEY, "residue_index"]
    wide = work.pivot_table(index=key, columns="modality", values="delta",
                            observed=True, aggfunc="sum")
    for block in BLOCKS:
        if block not in wide.columns:
            wide[block] = 0.0
    wide = wide.reset_index()
    bb, lt = wide[BLOCKS[0]], wide[BLOCKS[1]]
    wide["pattern"] = np.select(
        [(bb > 0) & (lt > 0), (bb < 0) & (lt < 0),
         (bb < 0) & (lt >= 0), (bb >= 0) & (lt < 0)],
        list(PATTERNS[:4]), default=PATTERNS[4])
    wide["any_component_worse"] = (bb < 0) | (lt < 0)
    wide["both_components_worse"] = (bb < 0) & (lt < 0)
    wide["backbone_worse"] = bb < 0
    wide["latent_worse"] = lt < 0
    return wide

def _count_block(frame: pd.DataFrame, group: list[str],
                 labels: Sequence[str]) -> pd.DataFrame:
    agg = {"n_designed": ("residue_index", "size"),
           "n_improvement": ("improvement", "sum"),
           "n_unchanged": ("unchanged", "sum"),
           "n_undefined_scale": ("relative_magnitude", lambda s: int(s.isna().sum())),
           "magnitude_total": ("magnitude", "sum")}
    for lab in labels:
        agg[f"n_increase@{lab}"] = (f"increase@{lab}", "sum")
    out = frame.groupby(group, observed=True, sort=False).agg(**agg).reset_index()

    out["n_scale_defined"] = out.n_designed - out.n_undefined_scale
    for lab in labels:
        col = f"n_increase@{lab}"
        out[col] = out[col].astype("Int64")
        denom = out.n_designed if lab == "0%" else out.n_scale_defined
        out[f"rate@{lab}"] = np.where(
            denom > 0, out[col].astype(float) / denom, np.nan)
    return out

def state_counts(flagged: pd.DataFrame, *,
                 labels: Sequence[str] = THRESHOLD_LABELS) -> pd.DataFrame:

    return _count_block(flagged, list(EVAL_KEY), labels)

def region_counts(flagged: pd.DataFrame, labels_frame: pd.DataFrame, *,
                  strata: Sequence[str] | None = None,
                  labels: Sequence[str] = THRESHOLD_LABELS) -> pd.DataFrame:

    _need(labels_frame, ["example_id", "residue_index", "stratum"], "labels")
    joined = flagged.merge(
        labels_frame[["example_id", "residue_index", "stratum"]],
        on=["example_id", "residue_index"], how="left", validate="many_to_one")
    unlabelled = int(joined.stratum.isna().sum())
    if unlabelled:
        raise CostError(
            f"{unlabelled} residue evaluations have no region label; the "
            "annotation and the cached rows describe different residues")
    per = _count_block([*EVAL_KEY, "stratum"], labels)
    want = list(strata) if strata is not None else sorted(per.stratum.unique())
    grid = (per[list(EVAL_KEY)].drop_duplicates()
            .merge(pd.DataFrame({"stratum": want}), how="cross"))
    full = grid.merge(per, on=[*EVAL_KEY, "stratum"], how="left")
    count_cols = ["n_designed", "n_improvement", "n_unchanged",
                  "n_undefined_scale", *[f"n_increase@{x}" for x in labels]]
    for c in count_cols:
        full[c] = full[c].fillna(0)
    full["magnitude_total"] = full["magnitude_total"].fillna(0.0)
    for lab in labels:
        full[f"rate@{lab}"] = np.where(
            full.n_designed > 0, full[f"rate@{lab}"], np.nan)
    return full

def paired_states(per_region: pd.DataFrame, focus: str, other: str, *,
                  labels: Sequence[str] = THRESHOLD_LABELS) -> pd.DataFrame:

    left = per_region[per_region.stratum == focus].set_index(list(EVAL_KEY))
    right = per_region[per_region.stratum == other].set_index(list(EVAL_KEY))
    common = left.index.intersection(right.index)
    left, right = left.loc[common], right.loc[common]
    keep = (left.n_designed > 0) & (right.n_designed > 0)
    out = pd.DataFrame(index=left.index[keep])
    out[f"n_designed_{focus}"] = left.n_designed[keep]
    out[f"n_designed_{other}"] = right.n_designed[keep]
    for lab in labels:
        lo = left[f"rate@{lab}"][keep].astype(float)
        ro = right[f"rate@{lab}"][keep].astype(float)
        out[f"rate_{focus}@{lab}"] = lo
        out[f"rate_{other}@{lab}"] = ro
        out[f"rate_diff@{lab}"] = lo - ro
    out[f"mean_magnitude_{focus}"] = np.where(
        left["n_increase@0%"][keep].astype(float) > 0,
        left.magnitude_total[keep] / left["n_increase@0%"][keep].astype(float),
        np.nan)
    out[f"mean_magnitude_{other}"] = np.where(
        right["n_increase@0%"][keep].astype(float) > 0,
        right.magnitude_total[keep] / right["n_increase@0%"][keep].astype(float),
        np.nan)
    return out.reset_index()

def residue_frequency(flagged: pd.DataFrame, *,
                      planned_evaluations: int = PLANNED_EVALUATIONS,
                      labels: Sequence[str] = THRESHOLD_LABELS) -> pd.DataFrame:

    _need(flagged, [*RESIDUE_ID, "schedule_index", "draw", "increase"],
          "flagged frame")
    agg = {"n_evaluations": ("increase", "size"),
           "n_increase": ("increase", "sum"),
           "n_improvement": ("improvement", "sum"),
           "n_unchanged": ("unchanged", "sum"),
           "n_timepoints": ("schedule_index", "nunique"),
           "n_draws": ("draw", "nunique"),
           "parent_id": ("parent_id", "first"),
           "family": ("family", "first"),
           "residue_index": ("residue_index", "first"),
           "magnitude_mean_over_increases": (
               "magnitude", lambda s: float(s[s > 0].mean()) if (s > 0).any()
               else np.nan),
           "magnitude_max": ("magnitude", "max")}
    for lab in labels:
        agg[f"n_increase@{lab}"] = (f"increase@{lab}", "sum")
    out = (flagged.groupby(list(RESIDUE_ID), observed=True, sort=False)
           .agg(**agg).reset_index())
    out["frequency"] = out.n_increase / out.n_evaluations
    out["planned_evaluations"] = planned_evaluations
    out["coverage"] = out.n_evaluations / planned_evaluations
    for lab in labels:
        out[f"frequency@{lab}"] = (out[f"n_increase@{lab}"].astype(float)
                                   / out.n_evaluations)
    return out

def frequency_by_timepoint(flagged: pd.DataFrame) -> pd.DataFrame:

    _need(flagged, [*RESIDUE_ID, "schedule_index", "increase"], "flagged frame")
    out = (flagged.groupby([*RESIDUE_ID, "schedule_index"], observed=True,
                           sort=False)
           .agg(n_draws=("increase", "size"), n_increase=("increase", "sum"),
                parent_id=("parent_id", "first"))
           .reset_index())
    out["frequency"] = out.n_increase / out.n_draws
    out["all_draws_increase"] = out.n_increase == out.n_draws
    out["no_draw_increases"] = out.n_increase == 0
    return out
