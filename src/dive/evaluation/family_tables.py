
from __future__ import annotations

import json
from pathlib import Path

from dive.evaluation.family_endpoints import (
    FAMILIES,
    conjunction,
    j_final_f,
    j_gen_f,
    parent_resampled_effect,
    preservation_supported,
    rate,
    reject_pooled_primary,
)

AME_FIELD = {
    "G": "G",
    "S": "S",
    "C_gen": "C_generated",
    "C_final": "C_refold_in_scaffold",
    "J_gen_recorded": "J_gen",
    "J_final_recorded": "J_refold",
}

BLOCKED_TERMS = {
    "ame": ("C_final",),
    "binder": ("C_fold", "C_iface"),
    "antibody": ("G", "S"),
}

COMPONENT_TERMS = {
    "ame": ("C_gen", "C_final"),
    "binder": (
        "C_fold",
        "C_iface",
        "C_fold_and_iface",
        "C_epitope_generated",
        "C_clash_generated",
    ),
    "antibody": ("C_gen", "C_final"),
}

BASE_METRICS = ("J_final", "J_gen", "S", "G")

def term_values(rows: list[dict], family: str, term: str) -> list:

    if term == "C_fold_and_iface":
        return [conjunction(r.get("C_fold"), r.get("C_iface")) for r in rows]
    return [r.get(term) for r in rows]

def blocked_term_verdict(family: str, tables: dict) -> dict:

    blocked = list(BLOCKED_TERMS[family])
    per_term = {}
    for term in blocked:
        ci = tables[term]["delta"]["ci95"]
        per_term[term] = {
            "effect": tables[term]["delta"]["effect"],
            "ci95_lower": ci[0],
            "ci95_upper": ci[1],
            "supported": bool(ci[0] is not None and ci[0] >= 0.10),
        }
    others = [m for m in tables if m not in blocked and m != "J_final"]
    other_terms = {}
    for term in sorted(others):
        lo = tables[term]["delta"]["ci95"][0]
        other_terms[term] = {
            "ci95_lower": lo,
            "preserved": bool(lo is not None and lo >= -0.05),
        }
    return {
        "terms": blocked,
        "per_term": per_term,
        "supported": bool(per_term and all(v["supported"] for v in per_term.values())),
        "any_term_supported": bool(any(v["supported"] for v in per_term.values())),
        "other_terms": other_terms,
        "other_terms_preserved": bool(
            all(v["preserved"] for v in other_terms.values())
        ),
        "conjunction_is_not_the_term": True,
    }

def load_parent_map(path: Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text())
    mapping = {}
    for row in payload["rows"]:
        task = row["task"]
        parent = row["source_parent_id"]
        if not parent:
            raise ValueError(f"empty audited parent for {task}")
        if task in mapping and mapping[task] != parent:
            raise ValueError(f"audited parent conflict for {task}")
        mapping[task] = parent
    return mapping

def normalize_binder_row(row: dict) -> dict:

    gen = row.get("generated_interface") or {}
    clash_d = gen.get("min_interchain_distance")
    c_clash = None if clash_d is None else bool(clash_d >= 2.0)
    out = dict(row)
    out["G"] = row.get("G_binder")
    out["S"] = row.get("S_binder_complex_slice")
    out["C_fold"] = row.get("C_fold")
    out["C_iface"] = row.get("C_iface")
    out["C_epitope_generated"] = gen.get("C_epitope")
    out["C_clash_generated"] = c_clash
    out["family"] = "binder"
    if not out.get("parent_id"):
        raise ValueError("binder slot missing parent_id")
    return out

def load_antibody_designs(path: Path, build: Path) -> list[dict]:
    designs = json.loads(Path(path).read_text())
    built = json.loads(Path(build).read_text())
    parents = {t["task_name"]: t["parent_id"] for t in built["tasks"]}
    rows = []
    for row in designs:
        task = row["task"]
        if task not in parents:
            raise ValueError(f"antibody task missing parent: {task}")
        rows.append(normalize_antibody_row(row, parents[task]))
    return rows

def normalize_antibody_row(row: dict, parent_id: str) -> dict:

    out = dict(row)
    out["parent_id"] = parent_id
    out["family"] = "antibody"
    out["C_gen"] = row.get("C_gen")
    out.setdefault("C_final", row.get("C_framework_refold"))
    out.setdefault("S", row.get("S"))
    return out

_UNKNOWN_ENDPOINT_KEYS = (
    "G",
    "S",
    "C_gen",
    "C_final",
    "C_framework_refold",
    "C_generated",
    "C_fold",
    "C_iface",
    "C_epitope_generated",
    "C_clash_generated",
    "J_final",
    "J_gen",
)

def unknown_requested_rows(control_rows: list[dict]) -> list[dict]:

    out = []
    for row in control_rows:
        task = row.get("task")
        seed = row.get("seed")
        parent = row.get("parent_id")
        if not task or seed is None or not parent:
            raise ValueError("unknown rows require task, seed, and parent_id")
        cand = {
            "task": task,
            "seed": seed,
            "parent_id": parent,
            "family": row.get("family"),
        }
        for key in _UNKNOWN_ENDPOINT_KEYS:
            cand[key] = None
        out.append(cand)
    return out

def load_binder_combined(root: Path) -> list[dict]:
    rows = []
    for report in sorted(Path(root).glob("*/evaluation_report.json")):
        payload = json.loads(report.read_text())
        if payload.get("family") not in (None, "binder"):
            raise ValueError(f"not a binder report: {report}")
        for slot in payload["slots"]:
            rows.append(normalize_binder_row(slot))
    return rows

def normalize_ame_row(row: dict, parent_map: dict[str, str]) -> dict:
    task = row["task"]
    if task not in parent_map:
        raise ValueError(f"audited parent missing: {task}")
    out = dict(row)
    out["parent_id"] = parent_map[task]
    out["C_gen"] = row.get("C_generated")
    out["C_final"] = row.get("C_refold_in_scaffold")
    out["family"] = "ame"
    return out

def scored_values(rows: list[dict], family: str) -> dict[str, list]:
    j_final = [j_final_f(r, family) for r in rows]
    j_gen = [j_gen_f(r, family) for r in rows]
    s = [r.get("S") for r in rows]
    g = [r.get("G") for r in rows]
    out = {"J_final": j_final, "J_gen": j_gen, "S": s, "G": g}
    for term in COMPONENT_TERMS[family]:
        out[term] = term_values(rows, family, term)
    return out

def pair_rows(candidate: list[dict], control: list[dict]) -> tuple[list[dict], list[dict]]:
    idx_c = {(r["task"], r["seed"]): r for r in candidate}
    idx_k = {(r["task"], r["seed"]): r for r in control}
    if set(idx_c) != set(idx_k):
        raise ValueError("paired requested slots differ")
    keys = sorted(idx_c)
    return [idx_c[k] for k in keys], [idx_k[k] for k in keys]

def family_contrast(
    candidate_rows: list[dict],
    control_rows: list[dict],
    *,
    family: str,
    candidate_arm: str,
    control_arm: str,
) -> dict:
    reject_pooled_primary({"primary": "J_final"})
    cand, ctrl = pair_rows(candidate_rows, control_rows)
    cv = scored_values(cand, family)
    kv = scored_values(ctrl, family)
    tables = {}
    for metric in (*BASE_METRICS, *COMPONENT_TERMS[family]):
        tables[metric] = {
            "candidate": rate(cand, cv[metric]),
            "control": rate(ctrl, kv[metric]),
            "delta": parent_resampled_effect(
                cand, ctrl, cv[metric], kv[metric], replicates=10000, seed=20260912
            ),
        }
    j_lo = tables["J_final"]["delta"]["ci95"][0]
    s_lo = tables["S"]["delta"]["ci95"][0]
    g_lo = tables["J_gen"]["delta"]["ci95"][0]
    return {
        "family": family,
        "candidate_arm": candidate_arm,
        "control_arm": control_arm,
        "primary": "J_final",
        "pooled": False,
        "confirmation": "not_opened",
        "n_requested": len(cand),
        "n_parents": tables["J_final"]["delta"]["independent_units"],
        "metrics": tables,
        "target_supported": bool(j_lo is not None and j_lo >= 0.10),
        "preservation_supported": preservation_supported(s_lo, g_lo),
        "blocked_term_verdict": blocked_term_verdict(family, tables),
        "second_seed_is_not_confirmation": True,
    }

def second_seed_sign(first: dict, second: dict) -> dict:

    e1 = first["metrics"]["J_final"]["delta"]["effect"]
    e2 = second["metrics"]["J_final"]["delta"]["effect"]
    if e1 is None or e2 is None:
        raise ValueError("missing J_final effects")
    return {
        "effect_first": e1,
        "effect_second": e2,
        "sign_reversed": bool((e1 > 0 and e2 < 0) or (e1 < 0 and e2 > 0)),
        "not_confirmation": True,
    }

def three_family_summary(tables: dict[str, dict]) -> dict:

    missing = [f for f in ("ame", "binder", "antibody") if f not in tables]
    if missing:
        raise ValueError(f"missing family tables: {missing}")
    for family, table in tables.items():
        if table.get("pooled") is True:
            raise ValueError(f"{family} table is pooled")
        if table.get("primary") != "J_final":
            raise ValueError(f"{family} primary is not J_final")
    return {
        "primary": "per_family_J_final",
        "pooled_three_family_mean": None,
        "pooled_forbidden": True,
        "confirmation": {
            "ame": "none_remaining",
            "binder": "none_remaining",
            "antibody": "test-blind_unopened",
        },
        "families": {name: tables[name] for name in ("ame", "binder", "antibody")},
        "any_family_target_supported": any(
            tables[name].get("target_supported") for name in ("ame", "binder", "antibody")
        ),
    }

def write_family_table(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
