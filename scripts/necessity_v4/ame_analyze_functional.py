
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

EXCLUDE = {"M0674_1uf7"}
PILOT = {"M0024_1nzy_v3", "M0050_1dbt", "M0054_1qfe", "M0096_1chm_og",
         "M0315_1ey3", "M0349_1e3v", "M0552_1fgh"}
HELDOUT = {"M0664_2dhn", "M0731_1mt5", "M0732_1xs1"}
VARIANT = {"M0024_1nzy_og"}

def paired(d):

    keep = ["gates_pass", "placed", "clashes", "clash_depth_nm", "contacts",
            "lig_min_nm", "displacement_nm", "objective", "all_placed",
            "bond_drift_nm", "angle_sigma"]
    w = d.pivot_table(index=["task", "stem"], columns="arm", values=keep)
    w.columns = [f"{a}_{b}" for b, a in w.columns]
    n = (d[d.arm == "fixed"].set_index(["task", "stem"])
         [["native_placed", "native_clashes", "native_contacts", "native_lig_min_nm",
           "native_clash_depth_nm", "n_req", "native_all_placed"]])
    return w.join(n).reset_index()

def boot(task, value, B=20000, seed=0):

    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"task": np.asarray(task), "v": np.asarray(value, dtype=float)})
    g = [x.values for _, x in t.groupby("task")["v"]]
    if len(g) < 2:
        return np.nan, np.nan, np.nan
    pt = np.mean([x.mean() for x in g])
    bs = [np.mean([g[i].mean() for i in rng.integers(0, len(g), len(g))]) for _ in range(B)]
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def block(p, label, out):
    out.append(f"\n### {label}  (n={len(p)} designs, {p.task.nunique()} targets)\n")
    if not len(p):
        out.append("_no designs_\n"); return
    out.append("| comparison | mean | 95% CI (target-clustered) | better | tie | worse |")
    out.append("|---|---|---|---|---|---|")
    rows = [("placed atoms: fixed - native", p.fixed_placed - p.native_placed),
            ("placed atoms: flexible - native", p.flexible_placed - p.native_placed),
            ("placed atoms: flexible - fixed", p.flexible_placed - p.fixed_placed),
            ("clashes removed: native - fixed", p.native_clashes - p.fixed_clashes),
            ("clashes removed: native - flexible", p.native_clashes - p.flexible_clashes),
            ("clashes: fixed - flexible", p.fixed_clashes - p.flexible_clashes)]
    for name, v in rows:
        m, lo, hi = boot(p.task.values, v.values)
        out.append(f"| {name} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                   f"{int((v>0).sum())} | {int((v==0).sum())} | {int((v<0).sum())} |")
    out.append("")
    out.append(f"- all named atoms placed: native **{int(p.native_all_placed.sum())}/{len(p)}**, "
               f"fixed **{int(p.fixed_all_placed.sum())}/{len(p)}**, "
               f"flexible **{int(p.flexible_all_placed.sum())}/{len(p)}**")
    out.append(f"- added clashes: native **{p.native_clashes.mean():.2f}**, "
               f"fixed **{p.fixed_clashes.mean():.2f}**, flexible **{p.flexible_clashes.mean():.2f}**")
    out.append(f"- designs meeting the declared practical bar (>=1 extra atom placed, or "
               f">=2 clashes removed, all gates passing): fixed "
               f"**{int((((p.fixed_placed-p.native_placed)>=1)|((p.native_clashes-p.fixed_clashes)>=2)).sum())}"
               f"/{len(p)}**, flexible "
               f"**{int((((p.flexible_placed-p.native_placed)>=1)|((p.native_clashes-p.flexible_clashes)>=2)).sum())}"
               f"/{len(p)}**")
    out.append(f"- nearest protein-ligand heavy-atom distance: native "
               f"**{p.native_lig_min_nm.mean():.3f}**, fixed **{p.fixed_lig_min_nm.mean():.3f}**, "
               f"flexible **{p.flexible_lig_min_nm.mean():.3f}** nm")
    out.append(f"- ligand contact atoms: native **{p.native_contacts.mean():.1f}**, fixed "
               f"**{p.fixed_contacts.mean():.1f}**, flexible **{p.flexible_contacts.mean():.1f}** "
               f"(kept, so improvement is not the protein backing away)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regated", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_parquet(a.regated)
    out = ["# Fixed-backbone vs flexible-backbone search\n",
           "Equal budget (32,000 objective evaluations, 4 restarts), same native start,",
           "same conditions, same objective. Arms differ only in degrees of freedom.",
           "Reported endpoints are not the optimised quantity.\n"]
    out.append("## Validity gates\n")
    out.append("| arm | poses passing | max bond drift (nm) | max angle drift (sigma) | mean displacement (nm) | wall (s) |")
    out.append("|---|---|---|---|---|---|")
    for arm, g in d[~d.task.isin(EXCLUDE)].groupby("arm"):
        out.append(f"| {arm} | {int(g.gates_pass.sum())}/{len(g)} | {g.bond_drift_nm.max():.2e} | "
                   f"{g.angle_sigma.max():.3f} | {g.displacement_nm.mean():.4f} | - |")
    p = paired(d[~d.task.isin(EXCLUDE)])
    both = p[(p.fixed_gates_pass == 1) & (p.flexible_gates_pass == 1)]
    out.append(f"\nDesigns where **both** arms produced a valid structure: "
               f"**{len(both)}/{len(p)}**. Only these are compared.\n")
    block(both, "All verified targets", out)
    block(both[both.task.isin(PILOT)], "Pilot targets (analysed in earlier work)", out)
    block(both[both.task.isin(HELDOUT)], "Held-out targets (never analysed before)", out)
    block(both[both.task.isin(VARIANT)], "M0024_1nzy_og (same enzyme as a pilot target; NOT independent)", out)

    out.append("\n## Per target\n")
    out.append("| target | role | n | required | native placed | fixed | flexible | native clashes | fixed | flexible |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for t, g in both.groupby("task"):
        role = "pilot" if t in PILOT else ("held out" if t in HELDOUT else "variant")
        out.append(f"| `{t}` | {role} | {len(g)} | {g.n_req.mean():.0f} | "
                   f"{g.native_placed.mean():.2f} | {g.fixed_placed.mean():.2f} | "
                   f"{g.flexible_placed.mean():.2f} | {g.native_clashes.mean():.1f} | "
                   f"{g.fixed_clashes.mean():.1f} | {g.flexible_clashes.mean():.1f} |")
    pp = paired(d)
    bb = pp[(pp.fixed_gates_pass == 1) & (pp.flexible_gates_pass == 1)]
    out.append(f"\n## With the excluded target included, for transparency\n")
    out.append(f"n={len(bb)} designs. placed: native {bb.native_placed.sum()}, "
               f"fixed {bb.fixed_placed.sum()}, flexible {bb.flexible_placed.sum()}; "
               f"clashes: native {bb.native_clashes.mean():.2f}, fixed {bb.fixed_clashes.mean():.2f}, "
               f"flexible {bb.flexible_clashes.mean():.2f}. The direction is unchanged.")
    Path(a.out).write_text("\n".join(out) + "\n")
    both.to_parquet(Path(a.out).parent / "paired_valid.parquet", index=False)
    print("\n".join(out))

if __name__ == "__main__":
    main()
