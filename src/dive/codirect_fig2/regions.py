
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from dive.codirect_fig1 import aggregate as A

LADDER = list(A.LADDER)

STRATA = {
    "binder": ("interface", "other_designed", "unknown"),
    "antibody": ("antigen_contact", "other_designed", "unknown"),
    "ame": ("ligand_proximal", "other_designed", "unknown"),
}

FOCUS = {"binder": "interface", "antibody": "antigen_contact",
         "ame": "ligand_proximal"}

BLOCKS = ("bb_ca", "local_latents")

class RegionError(ValueError):
    pass

def classify(manifest: pd.DataFrame, family: str, cutoff_nm: float) -> pd.DataFrame:

    if family not in STRATA:
        raise RegionError(f"unknown family {family!r}")
    frame = manifest[manifest["family"] == family].copy()
    if frame.empty:
        raise RegionError(f"no annotation rows for {family!r}")
    designed = frame["designed"].to_numpy(dtype=bool)
    dist = frame["partner_distance_nm"].to_numpy(dtype=float)
    complete = frame["partner_complete"].to_numpy(dtype=bool)
    focus = FOCUS[family]

    label = np.full(len(frame), "not_designed", dtype=object)
    have = designed & np.isfinite(dist)
    label[designed & ~np.isfinite(dist)] = "unknown"
    label[have & (dist <= cutoff_nm)] = focus
    far = have & (dist > cutoff_nm)
    label[far & complete] = "other_designed"
    label[far & ~complete] = "unknown"
    frame["stratum"] = label
    frame["unknown_cause"] = np.where(
        label != "unknown", "",
        np.where(~np.isfinite(dist), "no_partner_distance",
                 "non_contact_unverifiable_truncated_partner"))
    bad = set(frame.loc[designed, "stratum"]) - set(STRATA[family])
    if bad:
        raise RegionError(f"designed residues fell outside the vocabulary: {bad}")
    return frame

def state_sums(rows: pd.DataFrame, labels: pd.DataFrame, *,
               per_block: bool = False) -> pd.DataFrame:

    need = {"example_id", "residue_index", "stratum"}
    if not need <= set(labels.columns):
        raise RegionError(f"labels need {sorted(need)}")
    join = rows.merge(labels[["example_id", "residue_index", "stratum"]],
                      on=["example_id", "residue_index"], how="left",
                      validate="many_to_one")
    unlabelled = int(join["stratum"].isna().sum())
    if unlabelled:
        raise RegionError(
            f"{unlabelled} cached rows have no annotation row; the region "
            "manifest and the cached rows describe different residues")
    join = join[join["stratum"] != "not_designed"]

    join["b"] = join["delta"].clip(lower=0.0)
    join["h"] = (-join["delta"]).clip(lower=0.0)
    keys = [*LADDER, "stratum"] + (["modality"] if per_block else [])
    out = join.groupby(keys, observed=True).agg(
        B=("b", "sum"), H=("h", "sum"), L0=("ell_base", "sum"),
        n_terms=("delta", "size"),
    ).reset_index()
    out["G"] = out["B"] - out["H"]
    return out

def reconcile(region: pd.DataFrame, rows: pd.DataFrame) -> dict:

    total = region.groupby(LADDER, observed=True)[["B", "H", "G", "L0"]].sum()
    designed = rows.copy()
    designed = designed[designed["designed"].astype(bool)]
    designed["b"] = designed["delta"].clip(lower=0.0)
    designed["h"] = (-designed["delta"]).clip(lower=0.0)
    ref = designed.groupby(LADDER, observed=True).agg(
        B=("b", "sum"), H=("h", "sum"), L0=("ell_base", "sum")).reset_index()
    ref["G"] = ref["B"] - ref["H"]
    ref = ref.set_index(LADDER)
    common = total.index.intersection(ref.index)
    err = (total.loc[common] - ref.loc[common, ["B", "H", "G", "L0"]]).abs()
    return {
        "states_compared": int(len(common)),
        "states_only_in_strata": int(len(total.index.difference(ref.index))),
        "states_only_in_design_mask": int(len(ref.index.difference(total.index))),
        "max_abs_error": {c: float(err[c].max()) for c in err.columns},
    }

def parent_table(region: pd.DataFrame, strata: Sequence[str]) -> pd.DataFrame:

    grid = (region[LADDER].drop_duplicates()
            .merge(pd.DataFrame({"stratum": list(strata)}), how="cross"))
    full = grid.merge(region, on=[*LADDER, "stratum"], how="left")
    for col in ("B", "H", "G", "L0", "n_terms"):
        full[col] = full[col].fillna(0.0)
    wide = full.pivot_table(index=LADDER, columns="stratum",
                            values=["B", "H", "G", "L0", "n_terms"],
                            observed=True, aggfunc="sum")
    wide.columns = [f"{stat}|{stratum}" for stat, stratum in wide.columns]
    return A.parent_means(wide.reset_index(), list(wide.columns))

def bootstrap_regions(
    parents: pd.DataFrame,
    strata: Sequence[str],
    *,
    contrasts: Sequence[tuple[str, str]] = (),
    n_boot: int = A.N_BOOT,
    seed: int = A.SEED,
    alpha: float = 0.05,
) -> dict:

    cols = [c for c in parents.columns if "|" in c]
    values = parents[cols].to_numpy(dtype=float)
    index = {c: i for i, c in enumerate(cols)}
    n_parents = values.shape[0]
    rng = np.random.default_rng(seed)
    draws = (rng.integers(0, n_parents, size=(n_boot, n_parents))
             if n_parents >= 2 else None)
    reps = (values[draws].mean(axis=1) if draws is not None else None)
    point = values.mean(axis=0)

    def interval(pt, series):
        if series is None or not np.isfinite(series).any():
            return {"point": (float(pt) if pt is not None and np.isfinite(pt)
                              else None),
                    "low": None, "high": None,
                    "n_parents": int(n_parents)}
        good = series[np.isfinite(series)]
        return {"point": float(pt), "low": float(np.quantile(good, alpha / 2)),
                "high": float(np.quantile(good, 1 - alpha / 2)),
                "n_parents": int(n_parents),
                "replicates_undefined": int(len(series) - len(good))}

    out: dict = {"n_parents": int(n_parents), "n_boot": int(n_boot),
                 "seed": int(seed), "strata": list(strata), "absolute": {},
                 "relative": {}, "contrasts": {}}
    rel_series: dict[str, np.ndarray] = {}
    rel_point: dict[str, float] = {}
    for stratum in strata:
        base_i = index.get(f"L0|{stratum}")
        for stat in ("B", "H", "G", "L0", "n_terms"):
            key = f"{stat}|{stratum}"
            if key not in index:
                continue
            i = index[key]
            out["absolute"][key] = interval(
                point[i], reps[:, i] if reps is not None else None)
        gross_pt = (point[index[f"B|{stratum}"]] + point[index[f"H|{stratum}"]]
                    if f"B|{stratum}" in index else None)
        gross_rep = (reps[:, index[f"B|{stratum}"]] + reps[:, index[f"H|{stratum}"]]
                     if reps is not None and f"B|{stratum}" in index else None)
        if gross_pt is not None:
            out["absolute"][f"gross|{stratum}"] = interval(gross_pt, gross_rep)
        if base_i is None:
            continue
        for stat, num_pt, num_rep in (
            ("B", point[index[f"B|{stratum}"]],
             reps[:, index[f"B|{stratum}"]] if reps is not None else None),
            ("H", point[index[f"H|{stratum}"]],
             reps[:, index[f"H|{stratum}"]] if reps is not None else None),
            ("G", point[index[f"G|{stratum}"]],
             reps[:, index[f"G|{stratum}"]] if reps is not None else None),
            ("gross", gross_pt, gross_rep),
        ):
            name = f"relative_{stat}|{stratum}"
            denom_pt = point[base_i]
            if denom_pt == 0:
                out["relative"][name] = {
                    "point": None, "low": None, "high": None,
                    "undefined_reason": "baseline aggregate is exactly zero",
                    "n_parents": int(n_parents)}
                continue
            pt = 100.0 * num_pt / denom_pt
            series = None
            if reps is not None:
                den = reps[:, base_i]
                series = np.where(den != 0, 100.0 * num_rep / np.where(den == 0, 1, den),
                                  np.nan)
            out["relative"][name] = interval(pt, series)
            rel_point[name] = pt
            if series is not None:
                rel_series[name] = series

    for left, right in contrasts:
        for stat in ("G", "H", "B", "gross"):
            lk, rk = f"relative_{stat}|{left}", f"relative_{stat}|{right}"
            if lk not in rel_point or rk not in rel_point:
                continue
            pt = rel_point[lk] - rel_point[rk]
            series = (rel_series[lk] - rel_series[rk]
                      if lk in rel_series and rk in rel_series else None)
            out["contrasts"][f"{stat}: {left} - {right}"] = interval(pt, series)
    return out

def common_support(region: pd.DataFrame, left: str, right: str) -> pd.DataFrame:

    counts = (region.groupby(["example_id", "stratum"], observed=True)["n_terms"]
              .sum().unstack(fill_value=0))
    for col in (left, right):
        if col not in counts.columns:
            return region.iloc[0:0]
    keep = counts.index[(counts[left] > 0) & (counts[right] > 0)]
    return region[region["example_id"].isin(set(keep))]
