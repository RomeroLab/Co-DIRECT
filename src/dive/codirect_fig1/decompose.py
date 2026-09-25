
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from dive.codirect_fig1 import aggregate as A

DECOMPOSITIONS = {
    "block": ("B_block", "H_block", "G"),
    "residue": ("B_residue", "H_residue", "G_residue"),
}

PATTERNS = ("all_positive", "all_negative", "positive_with_zero",
            "negative_with_zero", "mixed_no_zero", "mixed_with_zero",
            "all_zero")

RESIDUE_KEY = ["family", "parent_id", "example_id", "schedule_index",
               "modality", "residue_index"]

class DecomposeError(ValueError):
    pass

def _interval(point: float, reps: np.ndarray, alpha: float = 0.05) -> dict:
    if reps is None or not np.isfinite(reps).any():
        return {"point": float(point), "low": float("nan"),
                "high": float("nan")}
    return {"point": float(point),
            "low": float(np.quantile(reps, alpha / 2.0)),
            "high": float(np.quantile(reps, 1.0 - alpha / 2.0))}

def state_terms(frame: pd.DataFrame, decomposition: str = "block") -> pd.DataFrame:

    if decomposition not in DECOMPOSITIONS:
        raise DecomposeError(
            f"unknown decomposition {decomposition!r}; expected one of "
            f"{sorted(DECOMPOSITIONS)}")
    b, h, g = DECOMPOSITIONS[decomposition]
    missing = [c for c in (b, h, g) if c not in frame.columns]
    if missing:
        raise DecomposeError(f"state frame is missing {missing}")
    out = frame.copy()
    out["gross"] = out[b] + out[h]
    out["abs_G"] = out[g].abs()
    out["two_min"] = 2.0 * np.minimum(out[b], out[h])
    out["coexist"] = ((out[b] > 0) & (out[h] > 0)).astype(float)
    out["B_s"], out["H_s"], out["G_s"] = out[b], out[h], out[g]
    return out

def cancellation(
    frame: pd.DataFrame,
    *,
    decomposition: str = "block",
    n_boot: int = A.N_BOOT,
    seed: int = A.SEED,
) -> dict:

    terms = state_terms(frame, decomposition)
    cols = ["gross", "abs_G", "two_min", "G_s", "coexist", "B_s", "H_s"]
    res = A.paired_parent_bootstrap(terms, cols, n_boot=n_boot, seed=seed)
    rep = res["_replicates"]

    def pt(name):
        return res[name]["point"]

    w_pt = pt("two_min")
    t_pt = pt("gross") - abs(pt("G_s"))
    a_pt = pt("abs_G") - abs(pt("G_s"))
    w_rep = rep["two_min"]
    t_rep = rep["gross"] - np.abs(rep["G_s"])
    a_rep = rep["abs_G"] - np.abs(rep["G_s"])

    alt = pt("gross") - pt("abs_G")
    if abs(alt - w_pt) > 1e-9 * max(1.0, abs(w_pt)):
        raise DecomposeError(
            f"W as 2*E[min(B,H)] is {w_pt!r} but as E[B+H-|G|] is {alt!r}; the "
            "two must agree exactly, so the aggregation is inconsistent")

    out: dict = {
        "decomposition": decomposition,
        "n_parents": res["_n_parents"],
        "n_state_draws": int(len(terms)),
        "W": _interval(w_pt, w_rep),
        "A": _interval(a_pt, a_rep),
        "T": _interval(t_pt, t_rep),
        "W_half": _interval(w_pt / 2.0, w_rep / 2.0),
        "gross": _interval(pt("gross"), rep["gross"]),
        "abs_G": _interval(pt("abs_G"), rep["abs_G"]),
        "net_G": _interval(pt("G_s"), rep["G_s"]),
        "B": _interval(pt("B_s"), rep["B_s"]),
        "H": _interval(pt("H_s"), rep["H_s"]),
        "coexist_share": _interval(pt("coexist"), rep["coexist"]),
        "identity_max_abs_error": float(
            np.abs((w_rep + a_rep) - t_rep).max()),
        "identity_point_abs_error": float(abs((w_pt + a_pt) - t_pt)),
        "T_zero": bool(t_pt == 0.0),
        "_replicates": {"W": w_rep, "A": a_rep, "T": t_rep,
                        "gross": rep["gross"], "abs_G": rep["abs_G"],
                        "G_s": rep["G_s"]},
    }

    if t_pt == 0.0:

        undefined = {"point": None, "low": None, "high": None,
                     "undefined_reason": "T is exactly zero: nothing cancelled"}
        out["W_over_T"] = dict(undefined)
        out["A_over_T"] = dict(undefined)
    else:
        ok = t_rep != 0.0
        out["W_over_T"] = _interval(w_pt / t_pt,
                                    (w_rep[ok] / t_rep[ok]) if ok.any() else None)
        out["A_over_T"] = _interval(a_pt / t_pt,
                                    (a_rep[ok] / t_rep[ok]) if ok.any() else None)
        out["replicates_with_T_zero"] = int((~ok).sum())
    return out

def cancellation_by_index(
    frame: pd.DataFrame, *, decomposition: str = "block",
    n_boot: int = 2000, seed: int = A.SEED,
) -> dict:

    out = {}
    for index, part in frame.groupby("schedule_index", observed=True):
        out[str(int(index))] = cancellation(
            part, decomposition=decomposition, n_boot=n_boot, seed=seed)
    return out

def state_distribution(frame: pd.DataFrame, *, decomposition: str = "block",
                       quantiles: Sequence[float] = (
                           0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:

    terms = state_terms(frame, decomposition)
    per_parent = terms.groupby("parent_id", observed=True).size()
    n_parents = len(per_parent)
    weight = terms["parent_id"].map(1.0 / per_parent) / n_parents
    out: dict = {"n_parents": int(n_parents), "n_state_draws": int(len(terms)),
                 "weighting": "each parent contributes total weight 1/n_parents"}
    for col in ("B_s", "H_s", "G_s", "two_min", "gross"):
        values = terms[col].to_numpy(dtype=float)
        order = np.argsort(values)
        v, cw = values[order], np.cumsum(weight.to_numpy()[order])
        cw = cw / cw[-1]
        out[col] = {
            "weighted_mean": float((values * weight).sum() / weight.sum()),
            "quantiles": {str(q): float(np.interp(q, cw, v)) for q in quantiles},
        }
    return out

def draw_agreement(frame: pd.DataFrame, *, n_draws: int = 4) -> dict:

    need = [*RESIDUE_KEY, "draw", "delta", "residue_pdb_idx", "chain",
            "designed"]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise DecomposeError(f"residue frame is missing {missing}")

    work = frame[need].copy()
    work["pos"] = (work.delta > 0).astype(np.int8)
    work["neg"] = (work.delta < 0).astype(np.int8)
    work["zero"] = (work.delta == 0).astype(np.int8)
    g = work.groupby(RESIDUE_KEY, observed=True, sort=False)
    agg = g.agg(
        n_draw=("draw", "size"),
        n_draw_unique=("draw", "nunique"),
        delta_mean=("delta", "mean"),
        delta_sd=("delta", "std"),
        delta_min=("delta", "min"),
        delta_max=("delta", "max"),
        n_positive=("pos", "sum"),
        n_negative=("neg", "sum"),
        n_zero=("zero", "sum"),
        pdb_nunique=("residue_pdb_idx", "nunique"),
        chain_nunique=("chain", "nunique"),
        designed_nunique=("designed", "nunique"),
        designed=("designed", "first"),
    ).reset_index()

    bad_count = (agg.n_draw != n_draws) | (agg.n_draw_unique != n_draws)
    bad_ident = ~bad_count & ((agg.pdb_nunique != 1) | (agg.chain_nunique != 1))
    bad_mask = ~bad_count & ~bad_ident & (agg.designed_nunique != 1)
    unalignable = {
        "wrong_draw_count": int(bad_count.sum()),
        "identity_disagrees": int(bad_ident.sum()),
        "mask_disagrees": int(bad_mask.sum()),
        "total_groups": int(len(agg)),
    }
    keep = agg[~(bad_count | bad_ident | bad_mask)].copy()

    pos, neg, zero = keep.n_positive, keep.n_negative, keep.n_zero
    keep["pattern"] = np.select(
        [
            pos == n_draws,
            neg == n_draws,
            zero == n_draws,
            (pos > 0) & (neg > 0) & (zero == 0),
            (pos > 0) & (neg > 0) & (zero > 0),
            (pos > 0) & (neg == 0) & (zero > 0),
            (neg > 0) & (pos == 0) & (zero > 0),
        ],
        ["all_positive", "all_negative", "all_zero", "mixed_no_zero",
         "mixed_with_zero", "positive_with_zero", "negative_with_zero"],
        default="UNCLASSIFIED",
    )
    if (keep["pattern"] == "UNCLASSIFIED").any():
        raise DecomposeError(
            "a residue fell into no sign-pattern category; the categories must "
            "be exhaustive or the shares below do not sum to one")
    keep = keep.drop(columns=["n_draw", "n_draw_unique", "pdb_nunique",
                              "chain_nunique", "designed_nunique"])
    shares = (keep["pattern"].value_counts(normalize=True).to_dict()
              if len(keep) else {})
    return {
        "per_residue": keep,
        "n_residues_aligned": int(len(keep)),
        "unalignable": unalignable,
        "pattern_shares": {p: float(shares.get(p, 0.0)) for p in PATTERNS},
        "pattern_counts": {p: int((keep["pattern"] == p).sum())
                           for p in PATTERNS},
    }

def draw_agreement_by_parent(agreement: dict) -> pd.DataFrame:

    per = agreement["per_residue"]
    if not len(per):
        return pd.DataFrame()
    out = per.groupby("parent_id", observed=True).agg(
        n_residue_slots=("delta_mean", "size"),
        mean_abs_delta_mean=("delta_mean", lambda s: s.abs().mean()),
        mean_delta_sd=("delta_sd", "mean"),
    )
    for pattern in PATTERNS:
        out[f"share_{pattern}"] = (
            per.assign(hit=(per["pattern"] == pattern).astype(float))
               .groupby("parent_id", observed=True)["hit"].mean())
    return out.reset_index()

def design_region_loss(frame: pd.DataFrame) -> pd.DataFrame:

    need = ["parent_id", "example_id", "schedule_index", "draw", "designed",
            "ell_base"]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise DecomposeError(f"residue frame is missing {missing}")
    key = ["parent_id", "example_id", "schedule_index", "draw"]
    work = frame[[*key, "designed", "ell_base"]].copy()
    work["ell_design"] = work.ell_base.where(work.designed.astype(bool), 0.0)
    out = work.groupby(key, observed=True, sort=False).agg(
        L_full_base=("ell_base", "sum"),
        L_design_base=("ell_design", "sum"),
    ).reset_index()
    out["L_context_base"] = out.L_full_base - out.L_design_base
    return out

def relative_effect(
    frame: pd.DataFrame,
    *,
    numerators: Sequence[str],
    denominators: Sequence[str],
    n_boot: int = A.N_BOOT,
    seed: int = A.SEED,
) -> dict:

    cols = list(dict.fromkeys([*numerators, *denominators]))
    res = A.paired_parent_bootstrap(frame, cols, n_boot=n_boot, seed=seed)
    rep = res["_replicates"]
    out: dict = {"n_parents": res["_n_parents"]}
    for den in denominators:
        out[f"E[{den}]"] = {k: res[den][k] for k in ("point", "low", "high")}
    for num in numerators:
        out[f"E[{num}]"] = {k: res[num][k] for k in ("point", "low", "high")}
        for den in denominators:
            name = f"{num}/{den}"
            d_pt = res[den]["point"]
            if d_pt == 0:
                out[name] = {"point": None, "low": None, "high": None,
                             "undefined_reason": "denominator is zero"}
                continue
            ok = rep[den] != 0
            ratio = (rep[num][ok] / rep[den][ok]) if ok.any() else None
            out[name] = _interval(res[num]["point"] / d_pt, ratio)
            out[name]["replicates_with_zero_denominator"] = int((~ok).sum())
    return out

def leave_one_index_out(
    frame: pd.DataFrame, *, column: str = "G",
    n_boot: int = 2000, seed: int = A.SEED,
) -> dict:

    indices = sorted(frame["schedule_index"].unique(), key=int)
    full = A.paired_parent_bootstrap(frame, [column], n_boot=n_boot, seed=seed)
    out = {"all_indices": {k: full[column][k]
                           for k in ("point", "low", "high")},
           "dropped": {}}
    for index in indices:
        part = frame[frame.schedule_index != index]
        one = A.paired_parent_bootstrap(part, [column], n_boot=n_boot, seed=seed)
        out["dropped"][str(int(index))] = {
            **{k: one[column][k] for k in ("point", "low", "high")},
            "shift_from_all": (one[column]["point"] - full[column]["point"]),
        }
    return out
