
from __future__ import annotations

from dive.codirect_paths import cache_dir

import json
import math
from pathlib import Path

GROUP = Path(cache_dir('emergent'))
RCE = GROUP / "receiver_conditioned_corepair/rce-20260910a"
TFRC = GROUP / "three_family_relation_contrast"
NJB = GROUP / "native_joint_bridge/njb-20260910a"

AME_PARENT_EVAL = RCE / "ame-parent-final-1401/evaluation.json"
AME_PARENT_MAP = RCE / "confirmation-eligibility-002/dev-task-parent-map.json"
BINDER_PARENT_DIR = RCE / "binder-parent-combined-001"
ANTIBODY_PARENT_DESIGNS = TFRC / "tfrc-20260914e/antibody-parent_caecho-final/designs.json"
ANTIBODY_PARENT_GENERATE = TFRC / "tfrc-20260914e/antibody-parent_caecho-generate"
ANTIBODY_BUILD = NJB / "antibody/dev/build.json"

def _num(value):

    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None

def _parent_map(path: Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    for row in payload["rows"]:
        parent = row["source_parent_id"]
        if not parent:
            raise ValueError(f"empty audited parent for {row['task']}")
        out[row["task"]] = parent
    return out

def _derive_antibody(row: dict) -> dict:

    bond = row.get("bond_geometry") or {}
    cn, ca = bond.get("C_N") or {}, bond.get("CA_CA") or {}
    out = dict(row)
    out["cn_fraction"] = cn.get("fraction")
    out["caca_fraction"] = ca.get("fraction")
    n = cn.get("n")
    if n and cn.get("fraction") is not None and ca.get("fraction") is not None:
        bad = max(round((1 - cn["fraction"]) * n), round((1 - ca["fraction"]) * n))
        out["designed_invalid_excess"] = bad - (n - math.ceil(0.95 * n))
    return out

def ame_slots(evaluation: Path = AME_PARENT_EVAL, *, metric: str,
              parent_map: Path = AME_PARENT_MAP,
              generations: Path | None = None) -> list[dict]:
    mapping = _parent_map(parent_map)
    root = Path(evaluation).parent if generations is None else Path(generations)
    index = {(p.parent.name, int(p.stem.removeprefix("native_seed"))): p
             for p in root.rglob("native_seed*.pt")}
    out = []
    for row in json.loads(Path(evaluation).read_text())["rows"]:
        task, seed = row["task"], int(row["seed"])
        if task not in mapping:
            raise ValueError(f"audited parent missing for AME task {task}")
        out.append({"task": task, "seed": seed, "parent_id": mapping[task],
                    "value": _num(row.get(metric)), "design": index.get((task, seed)),
                    "row": row})
    return out

def binder_slots(combined: Path = BINDER_PARENT_DIR, *, metric: str) -> list[dict]:
    out = []
    reports = sorted(Path(combined).glob("*/evaluation_report.json"))
    if not reports:
        raise FileNotFoundError(f"no evaluation_report.json under {combined}")
    for report in reports:
        payload = json.loads(report.read_text())
        if payload.get("family") not in (None, "binder"):
            raise ValueError(f"not a binder report: {report}")
        for slot in payload["slots"]:
            if not slot.get("parent_id"):
                raise ValueError(f"binder slot missing parent_id in {report}")
            metrics = slot.get("metrics") or {}
            value = metrics.get(metric, slot.get(metric))
            design = slot.get("path")
            out.append({"task": slot["task"], "seed": int(slot["seed"]),
                        "parent_id": slot["parent_id"], "value": _num(value),
                        "design": Path(design) if design else None,
                        "row": {**slot, **metrics}})
    return out

def antibody_slots(designs: Path = ANTIBODY_PARENT_DESIGNS, *, metric: str,
                   build: Path = ANTIBODY_BUILD,
                   generations: Path | None = None) -> list[dict]:
    built = json.loads(Path(build).read_text())
    parents = {t["task_name"]: t["parent_id"] for t in built["tasks"]}
    root = Path(generations) if generations else ANTIBODY_PARENT_GENERATE
    out = []
    for raw in json.loads(Path(designs).read_text()):
        row = _derive_antibody(raw)
        task, seed = row["task"], int(row["seed"])
        if task not in parents:
            raise ValueError(f"antibody task missing parent: {task}")
        design = root / task / f"seed-{seed}.pt"
        out.append({"task": task, "seed": seed, "parent_id": parents[task],
                    "value": _num(row.get(metric)),
                    "design": design if design.exists() else None, "row": row})
    return out

LOADERS = {"ame": ame_slots, "binder": binder_slots, "antibody": antibody_slots}

def load_parent_slots(family: str, *, metric: str) -> list[dict]:

    return LOADERS[family](metric=metric)
