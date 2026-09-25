
from __future__ import annotations

from dive.codirect_paths import cache_dir

import csv
import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from dive.benchmark.sealed_guard import SealedAccessError, refuse_sealed_path
from dive.benchmark.task_cards import (
    ContaminationStatus,
    LigandGraph,
    ManifestPartition,
    SequencePolicy,
    binder_task_card,
    enzyme_task_card,
    task_card_from_canonical,
)
from dive.data.contracts import CanonicalExample, ChainRecord, Family, LigandRecord
from dive.signed_value.roots import DENIED_PREFIXES, EMERGENT_EVIDENCE_ROOT
from dive.training.preflight import canonical_json_bytes

class MaterializeError(RuntimeError):
    pass

VIEW_DIR = Path(
    cache_dir('emergent', 'benchmark_ready', 'benchmark-views-20260828a')
)
LOADER_DIR = Path(
    cache_dir('emergent', 'loader_manifests_v5')
)
SUITE_V2 = EMERGENT_EVIDENCE_ROOT / "benchmarks" / "codirect-public-dev-v2"

LEGACY_EXAMPLE_IDS = {
    "binder": "binder:4osp:A|B+C+D",
    "ame": "ame:7cnu__1__1.A__1.C",
    "antibody": "pdb_00009ivj_H_L",
}

PUBLIC_N = 10
CALIBRATION_N = 2

@dataclass(frozen=True, slots=True)
class SelectedCohort:
    public: tuple[dict[str, object], ...]
    calibration: tuple[dict[str, object], ...]
    legacy_dev: tuple[dict[str, object], ...]
    eligible_parents: int
    source_rows: int

@dataclass(frozen=True, slots=True)
class MaterializedTask:
    family: str
    example_id: str
    parent_id: str
    partition: str
    source_path: str
    source_sha256: str
    generated: str
    context: str
    target: str
    method_input_path: str | None
    evaluator_hidden_path: str | None
    hotspots: tuple[str, ...]
    exclusion_reason: str | None = None

def _refuse(path: Path) -> Path:
    text = os.path.abspath(str(path))
    for prefix in DENIED_PREFIXES:
        if text.startswith(str(prefix)):
            raise MaterializeError(f"denied path {path}")
    refuse_sealed_path(path)
    return Path(text)

def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()

def _read_validation_table(path: Path) -> object:
    _refuse(path)
    if "test-blind" in str(path).lower() or "sealed" in path.name.lower():
        raise SealedAccessError(f"refusing {path}")
    return pq.read_table(path)

def load_joined_validation(family: str) -> tuple[dict[str, object], ...]:
    if family not in {"binder", "ame", "antibody"}:
        raise MaterializeError(f"unknown family {family}")
    view = _read_validation_table(VIEW_DIR / f"{family}_validation_examples.parquet")
    loader = _read_validation_table(LOADER_DIR / f"{family}_validation.parquet")
    view_rows = {row["example_id"]: row for row in view.to_pylist()}
    out: list[dict[str, object]] = []
    for row in loader.to_pylist():
        example_id = str(row["example_id"])
        meta = view_rows.get(example_id)
        if meta is None:
            continue
        if str(meta["partition"]) != "validation":
            raise MaterializeError(f"{example_id} is not validation")
        if "test-blind" in example_id or "test-blind" in str(row["path"]):
            raise SealedAccessError(f"blind identifier leaked: {example_id}")
        out.append(
            {
                "example_id": example_id,
                "parent_id": str(meta["parent_id"]),
                "family": family,
                "partition": "validation",
                "strict": bool(meta["strict"]),
                "path": str(row["path"]),
                "generated": str(row["generated"]),
                "context": str(row["context"]),
                "target": str(row["target"]),
            }
        )
    return tuple(out)

def select_cohort(
    rows: Sequence[Mapping[str, object]],
    *,
    family: str,
    public_n: int = PUBLIC_N,
    calibration_n: int = CALIBRATION_N,
) -> SelectedCohort:

    by_parent: dict[str, dict[str, object]] = {}
    for row in rows:
        parent = str(row["parent_id"])
        example = str(row["example_id"])
        previous = by_parent.get(parent)
        if previous is None or example < str(previous["example_id"]):
            by_parent[parent] = dict(row)
    legacy_id = LEGACY_EXAMPLE_IDS[family]
    legacy_rows = [dict(row) for row in rows if str(row["example_id"]) == legacy_id]
    legacy_parents = {str(item["parent_id"]) for item in legacy_rows}
    legacy = tuple(legacy_rows[:1])
    remaining = sorted(
        (
            item
            for item in by_parent.values()
            if str(item["parent_id"]) not in legacy_parents and bool(item["strict"])
        ),
        key=lambda item: (str(item["parent_id"]), str(item["example_id"])),
    )
    if len(remaining) < public_n + calibration_n:
        raise MaterializeError(f"{family} has only {len(remaining)} non-legacy parents")
    return SelectedCohort(
        public=tuple(remaining[:public_n]),
        calibration=tuple(remaining[public_n : public_n + calibration_n]),
        legacy_dev=legacy,
        eligible_parents=len(by_parent),
        source_rows=len(rows),
    )

def _chain_records(row: Mapping[str, object]) -> tuple[ChainRecord, ...]:
    family = str(row["family"])
    generated = str(row["generated"]).split(":")[0]
    target = [part.strip() for part in str(row["target"]).split(",") if part.strip()]
    if family == "binder":
        chains = [
            ChainRecord(
                generated[-1] if "." in generated else generated, "designed", "X" * 16
            )
        ]
        for chain_id in target or ["B"]:
            chains.append(ChainRecord(chain_id, "target", "Y" * 16))
        return tuple(chains)
    if family == "ame":
        chain_id = generated.split(".")[-1] if "." in generated else generated
        return (ChainRecord(chain_id or "A", "designed", "X" * 16),)
    antibody_chain = generated.split(":")[0]
    antigen = target[0] if target else "A"
    return (
        ChainRecord(antibody_chain, "antibody", "EVQLVESGGGLVQPGGSLRL"),
        ChainRecord(antigen, "antigen", "Y" * 16),
    )

def task_card_for(row: Mapping[str, object], *, partition: ManifestPartition):
    family = str(row["family"])
    example = CanonicalExample(
        example_id=str(row["example_id"]),
        parent_id=str(row["parent_id"]),
        family=Family(family),
        source_release="validation-v5",
        pdb_id=None,
        assembly_id=None,
        deposition_date=None,
        chains=_chain_records(row),
        ligands=(LigandRecord("LIG", "C", None, ""),) if family == "ame" else (),
        metadata={"source_path": str(row["path"])},
    )
    if family == "binder":
        hotspots = tuple(row.get("hotspots") or ("target-site",))
        return binder_task_card(example, hotspot_residues=hotspots, partition=partition)
    if family == "ame":
        graph = LigandGraph(
            ligand_id="LIG",
            atom_names=("C1",),
            elements=("C",),
            bonds=(),
            formal_charges=(0,),
            stereochemistry="unknown",
        )
        return enzyme_task_card(
            example,
            ligand_graph=graph,
            catalytic_atoms=("named-atom-pending-from-cif",),
            partition=partition,
        )
    cdr = str(row["generated"])
    length = 10
    if ":" in cdr and "-" in cdr.split(":")[-1]:
        start, end = cdr.split(":")[-1].split("-")
        try:
            length = int(end) - int(start) + 1
        except ValueError:
            length = 10

    return task_card_from_canonical(
        example,
        task_type="antibody_fixed_framework_cdr_h3",
        design_regions=("CDR-H3",),
        generated_chains=(example.chains[0].chain_id,),
        conditioning_chains=(example.chains[1].chain_id,),
        hidden_reference=(
            "native_cdr_sequence",
            "native_cdr_conformation",
            "native_antibody_antigen_dock",
        ),
        motif_or_epitope=("antigen-site",),
        design_length=max(length, 1),
        partition=partition,
        metadata={
            "framework_id": "sabdab2-row-framework",
            "format": "paired_or_single",
        },
    )

def write_cohort(root: Path = SUITE_V2) -> dict[str, object]:
    destination = _refuse(root / "cohort.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": "dive-codirect-public-dev-v2",
        "families": {},
        "legacy_example_ids": dict(LEGACY_EXAMPLE_IDS),
        "sealed_opened": False,
    }
    manifest_rows: list[dict[str, str]] = []
    for family in ("binder", "ame", "antibody"):
        joined = load_joined_validation(family)
        selected = select_cohort(family=family)
        family_payload = {
            "source_rows": selected.source_rows,
            "eligible_parents": selected.eligible_parents,
            "public": len(selected.public),
            "calibration": len(selected.calibration),
            "legacy_dev": len(selected.legacy_dev),
            "excluded_unselected": selected.eligible_parents
            - len(selected.public)
            - len(selected.calibration)
            - len(selected.legacy_dev),
        }
        tasks = []
        mapping = (
            (selected.public, ManifestPartition.PUBLIC),
            (selected.calibration, ManifestPartition.CALIBRATION),
            (selected.legacy_dev, ManifestPartition.LEGACY_DEV),
        )
        for group, partition in mapping:
            for row in group:
                path = Path(str(row["path"]))
                sha = _hash_file(path) if path.is_file() else "missing"
                card = task_card_for(row, partition=partition)
                tasks.append(
                    {
                        "example_id": row["example_id"],
                        "parent_id": row["parent_id"],
                        "partition": partition.value,
                        "path": str(path),
                        "source_sha256": sha,
                        "generated": row["generated"],
                        "context": row["context"],
                        "target": row["target"],
                        "task_type": card.task_type,
                        "contamination": ContaminationStatus.UNKNOWN.value,
                        "sequence_policy": SequencePolicy.SAMPLED.value,
                        "task_card_sha256": card.semantic_sha256(),
                    }
                )
                manifest_rows.append(
                    {
                        "family": family,
                        "example_id": str(row["example_id"]),
                        "parent_id": str(row["parent_id"]),
                        "partition": partition.value,
                        "path": str(path),
                        "source_sha256": sha,
                    }
                )
        family_payload["tasks"] = tasks
        summary["families"][family] = family_payload
    payload = canonical_json_bytes(summary)
    destination.write_bytes(payload)
    csv_path = root / "task_manifest.csv"
    _refuse(csv_path)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "example_id",
                "parent_id",
                "partition",
                "path",
                "source_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary["cohort_sha256"] = hashlib.sha256(payload).hexdigest()
    summary["manifest_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return summary
