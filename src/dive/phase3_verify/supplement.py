
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

EXPECTED_TOTAL_ROWS = 72_205
EXPECTED_AXES = 8

class CoverageDefinition(Enum):

    UNRECONCILED = "unreconciled"
    RECONCILED = "reconciled"

@dataclass(frozen=True, slots=True)
class Reconstruction:
    total_rows: int
    primary_rows: int
    unique_entities: int
    unique_sources: int
    unique_extracted: int
    axis_counts: dict[tuple[str, str], tuple[int, int]]

@dataclass(frozen=True, slots=True)
class Terminal:
    status: str
    error_message: str

@dataclass(frozen=True, slots=True)
class BindDecision:
    allowed: bool
    refusals: list[str] = field(default_factory=list)

def reconstruct_axes(manifest_path: Path) -> Reconstruction:

    frame = pd.read_parquet(manifest_path)
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for (family, axis), group in frame.groupby(["family", "axis"], sort=True):
        partition = group["partition"]
        counts[(str(family), str(axis))] = (
            int((partition == "train").sum()),
            int((partition == "validation").sum()),
        )
    return Reconstruction(
        total_rows=int(len(frame)),
        primary_rows=int((frame["decision_role"] == "primary").sum()),
        unique_entities=int(frame["entity_id"].nunique()),
        unique_sources=int(frame["source_sha256"].nunique()),
        unique_extracted=int(frame["extracted_sha256"].nunique()),
        axis_counts=counts,
    )

def check_plan_agreement(manifest_path: Path, plan_path: Path) -> list[str]:

    reconstruction = reconstruct_axes(manifest_path)
    plan = json.loads(Path(plan_path).read_text())
    searches = plan.get("searches", [])
    mismatches: list[str] = []

    planned = {(str(s["family"]), str(s["axis"])) for s in searches}
    rebuilt = set(reconstruction.axis_counts)
    for missing in sorted(rebuilt - planned):
        mismatches.append(f"{missing[0]}/{missing[1]}: axis has no planned search")
    for extra in sorted(planned - rebuilt):
        mismatches.append(f"{extra[0]}/{extra[1]}: planned search has no axis")

    for search in searches:
        key = (str(search["family"]), str(search["axis"]))
        if key not in reconstruction.axis_counts:
            continue
        train, validation = reconstruction.axis_counts[key]
        if int(search["query_entity_count"]) != validation:
            mismatches.append(
                f"{key[0]}/{key[1]}: plan queries {search['query_entity_count']}"
                f" but manifest has {validation} validation entities"
            )
        if int(search["target_entity_count"]) != train:
            mismatches.append(
                f"{key[0]}/{key[1]}: plan targets {search['target_entity_count']}"
                f" but manifest has {train} train entities"
            )
    return mismatches

def read_terminal(evidence_root: Path) -> Terminal | None:

    path = Path(evidence_root) / "terminal-status.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    error = payload.get("error") or {}
    message = error.get("message", "") if isinstance(error, dict) else str(error)
    return Terminal(status=str(payload.get("status", "")), error_message=message)

def bind_decision(
    evidence_root: Path, *, coverage: CoverageDefinition
) -> BindDecision:

    root = Path(evidence_root)
    refusals: list[str] = []

    terminal = read_terminal(root)
    if terminal is None:
        refusals.append("no terminal status")
    elif terminal.status != "COMPLETE":
        refusals.append(f"terminal status is {terminal.status}")

    if coverage is not CoverageDefinition.RECONCILED:
        refusals.append(
            "coverage definition is not reconciled with the approved specification"
        )

    manifest = root / "entity_manifest.parquet"
    plan = root / "foldseek_plan.json"
    if not manifest.is_file():
        refusals.append("entity manifest is missing")
    else:
        reconstruction = reconstruct_axes(manifest)
        if reconstruction.total_rows != EXPECTED_TOTAL_ROWS:
            refusals.append(
                f"row count is {reconstruction.total_rows},"
                f" expected {EXPECTED_TOTAL_ROWS}"
            )
        if len(reconstruction.axis_counts) != EXPECTED_AXES:
            refusals.append(
                f"axis count is {len(reconstruction.axis_counts)},"
                f" expected {EXPECTED_AXES}"
            )
        if plan.is_file():
            refusals.extend(check_plan_agreement(manifest, plan))
        else:
            refusals.append("foldseek plan is missing")

    return BindDecision(allowed=not refusals, refusals=refusals)

__all__ = (
    "BindDecision",
    "CoverageDefinition",
    "Reconstruction",
    "Terminal",
    "bind_decision",
    "check_plan_agreement",
    "read_terminal",
    "reconstruct_axes",
)
