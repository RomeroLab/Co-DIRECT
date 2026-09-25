
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd

POOL = {"M0024_1nzy_og": "M0024_1nzy", "M0024_1nzy_v3": "M0024_1nzy"}
REF = "joint"
MIN_ATOMS, MIN_CLASHES = 1, 2

def boot(cluster, value, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"c": np.asarray(cluster), "v": np.asarray(value, float)})
    g = [x.values for _, x in t.groupby("c")["v"]]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    return (np.mean([x.mean() for x in g]),
            *np.percentile([np.mean([g[i].mean() for i in rng.integers(0, len(g), len(g))])
                            for _ in range(B)], [2.5, 97.5]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(p) for p in glob.glob(a.glob)])
    d["cluster"] = d.task.replace(POOL)
    out = ["# Path II — do leader/follower policies beat joint and alternating?\n",
           "All policies: same native start, same ligand and theozyme information, same",
           "objective, same **objective-evaluation** budget. The follower is an external",
           "structure-preserving solver, never the model, so nothing here is evidence about",
           "leadership inside the frozen network.\n"]

    out.append("## Budget and cost\n")
    out.append("| policy | evals used | wall s/design | outer iterations | gates pass | displacement (nm) |")
    out.append("|---|---|---|---|---|---|")
    for pol, g in d.groupby("policy"):
        out.append(f"| {pol} | {g.evals_used.mean():.0f} | {g.wall_s.mean():.1f} | "
                   f"{g.outer_iters.mean():.1f} | {int(g.passes.sum())}/{len(g)} | "
                   f"{g.displacement_nm.mean():.4f} |")
    out.append("\nEvaluation counts are matched by construction; wall time is **not** matched "
               "and is reported separately rather than equated with it.\n")

    p = d.pivot_table(index=["cluster", "task", "stem"], columns="policy",
                      values=["placed", "clashes", "objective", "wall_s", "passes"]).reset_index()
    p.columns = [f"{b}_{a}" if b else a for a, b in p.columns]
    n = (d[d.policy == REF].set_index(["task", "stem"])
         [["native_placed", "native_clashes", "n_req", "native_all_placed"]])
    p = p.set_index(["task", "stem"]).join(n).reset_index()
    pols = sorted(d.policy.unique())
    valid = p[np.all([p[f"{q}_passes"] == 1 for q in pols], axis=0)]

    out.append(f"## Result (n={len(valid)} designs, {valid.cluster.nunique()} target clusters, "
               f"all policies valid)\n")
    out.append(f"native: **{valid.native_placed.mean():.3f}** placed of "
               f"{valid.n_req.mean():.1f} required, **{valid.native_clashes.mean():.2f}** clashes\n")
    out.append("| policy | placed | vs joint | 95% CI | clashes | vs joint | 95% CI |")
    out.append("|---|---|---|---|---|---|---|")
    for q in pols:
        dp = valid[f"{q}_placed"] - valid[f"{REF}_placed"]
        dc = valid[f"{REF}_clashes"] - valid[f"{q}_clashes"]
        mp, lp, hp = boot(valid.cluster, dp)
        mc, lc, hc = boot(valid.cluster, dc)
        tag = " *(reference)*" if q == REF else ""
        out.append(f"| {q}{tag} | {valid[f'{q}_placed'].mean():.3f} | {mp:+.3f} | "
                   f"[{lp:+.3f},{hp:+.3f}] | {valid[f'{q}_clashes'].mean():.2f} | "
                   f"{mc:+.2f} | [{lc:+.2f},{hc:+.2f}] |")

    out.append("\n### Against the pre-fixed practical minimum "
               f"(>= {MIN_ATOMS} atom or >= {MIN_CLASHES} clashes better than `{REF}`)\n")
    out.append("| policy | designs meeting it | verdict |")
    out.append("|---|---|---|")
    for q in pols:
        if q == REF:
            continue
        dp = valid[f"{q}_placed"] - valid[f"{REF}_placed"]
        dc = valid[f"{REF}_clashes"] - valid[f"{q}_clashes"]
        hit = int(((dp >= MIN_ATOMS) | (dc >= MIN_CLASHES)).sum())
        mp, lp, hp = boot(valid.cluster, dp)
        mc, lc, hc = boot(valid.cluster, dc)

        sig = (lp > 0) or (lc > 0)
        meaningful = (mp >= MIN_ATOMS) or (mc >= MIN_CLASHES)
        excl = (hp < MIN_ATOMS) and (hc < MIN_CLASHES)
        v = ("beats joint AND clears the practical minimum" if (sig and meaningful) else
             "distinguishable from joint but BELOW the practical minimum" if sig else
             "practical effect EXCLUDED" if excl else
             "not significant, practical effect not excluded")
        out.append(f"| {q} | {hit}/{len(valid)} | {v} |")

    out.append("\n## Per cluster\n")
    out.append("| cluster | n | native placed | " + " | ".join(f"{q} placed" for q in pols) + " |")
    out.append("|" + "---|" * (3 + len(pols)))
    for c, g in valid.groupby("cluster"):
        out.append(f"| `{c}` | {len(g)} | {g.native_placed.mean():.2f} | "
                   + " | ".join(f"{g[f'{q}_placed'].mean():.2f}" for q in pols) + " |")
    out.append("\n| cluster | n | native clashes | " + " | ".join(f"{q}" for q in pols) + " |")
    out.append("|" + "---|" * (3 + len(pols)))
    for c, g in valid.groupby("cluster"):
        out.append(f"| `{c}` | {len(g)} | {g.native_clashes.mean():.1f} | "
                   + " | ".join(f"{g[f'{q}_clashes'].mean():.1f}" for q in pols) + " |")

    Path(a.out).write_text("\n".join(out) + "\n")
    valid.to_parquet(Path(a.out).parent / "policy_paired.parquet", index=False)
    print("\n".join(out))

if __name__ == "__main__":
    main()
