
from __future__ import annotations

import numpy as np

N_BOOT = 10_000

def cluster_bootstrap(values_by_cluster, n_boot: int = N_BOOT, seed: int = 0):

    keys = sorted(values_by_cluster)
    per = np.array([np.nanmean(values_by_cluster[k]) for k in keys], float)
    per = per[~np.isnan(per)]
    if len(per) == 0:
        return {"n_clusters": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(per), size=(n_boot, len(per)))
    draws = per[idx].mean(axis=1)
    return {"n_clusters": int(len(per)), "mean": float(per.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "sd_clusters": float(per.std(ddof=1)) if len(per) > 1 else np.nan}

def paired_effect(df, arm, ref_arm, value, cluster="task", keys=("seed", "frac")):

    a = df[df.arm == arm].set_index([cluster, *keys])[value]
    b = df[df.arm == ref_arm].set_index([cluster, *keys])[value]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"n_clusters": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "n_pairs": 0}
    diff = (a.loc[common] - b.loc[common])
    by = {}
    for (cl, *_rest), v in diff.items():
        by.setdefault(cl, []).append(v)
    out = cluster_bootstrap(by)
    out["n_pairs"] = int(len(common))
    return out

def verdict(effect, *, floor: float, lower_is_better: bool = True):

    if not np.isfinite(effect.get("mean", np.nan)):
        return "undetermined"
    m, lo, hi = effect["mean"], effect["lo"], effect["hi"]
    gain = -m if lower_is_better else m
    glo, ghi = (-hi, -lo) if lower_is_better else (lo, hi)
    if glo > 0 and gain >= floor:
        return "improves"
    if ghi < 0 and -gain >= floor:
        return "worsens"
    if max(abs(glo), abs(ghi)) < floor:
        return "practical effect excluded"
    return "not distinguishable"

def non_inferior(effect, margin: float, lower_is_better: bool = True):

    if not np.isfinite(effect.get("hi", np.nan)):
        return False
    return (effect["hi"] < margin) if lower_is_better else (effect["lo"] > -margin)
