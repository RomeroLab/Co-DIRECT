
from __future__ import annotations

from dive.codirect_paths import ROOT, joined

import csv
import hashlib
import os
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dive.benchmark.baselines import (
    condition_matrix_rows,
    load_codirect_baseline_registry,
)
from dive.benchmark.compute_ledger import write_ledger
from dive.benchmark.evaluators.controls import evaluate_control_suite
from dive.benchmark.inventory import (
    AssetStatus,
    CheckpointRecord,
    EnvironmentRecord,
    classify_checkpoint,
    classify_repository,
    inventory_from_observations,
)
from dive.benchmark.manifests import build_family_manifests
from dive.benchmark.pilot import run_public_dev_pilot
from dive.benchmark.preregistration import freeze_preregistration
from dive.benchmark.sealed_guard import bind_sealed_envelope, refuse_sealed_path
from dive.benchmark.task_cards import (
    LigandGraph,
    ManifestPartition,
    binder_task_card,
    enzyme_task_card,
    vhh_cdr_h3_task_card,
)
from dive.data.contracts import CanonicalExample, ChainRecord, Family, LigandRecord
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_COMMIT,
    EMERGENT_UPSTREAM_ROOT,
    RUNTIME_PYTHON,
    SIGNED_VALUE_UPSTREAM_COMMIT,
    UPSTREAM_ROOT,
)
from dive.training.preflight import canonical_json_bytes

class SuiteError(RuntimeError):
    pass

SUITE_RUN_ID = "codirect-canonical-v1"
SUITE_ROOT = EMERGENT_EVIDENCE_ROOT / "benchmarks" / SUITE_RUN_ID

_AUTHENTICATED_CHECKPOINTS = (
    (
        Path(joined('PROTEINA_COMPLEXA_ROOT', 'ckpts', 'complexa.ckpt')),
        "589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9",
        2934289381,
        "proteina_complexa_v1",
        False,
    ),
    (
        Path(joined('PROTEINA_COMPLEXA_ROOT', 'ckpts', 'complexa_ae.ckpt')),
        "35f8865efd269995eeaf1670e1c1085acfe2988c40abdeda8e09a0e15eb40816",
        4100101779,
        "proteina_complexa_ae",
        False,
    ),
    (
        Path(
            joined('PROTEINA_COMPLEXA_ROOT', 'community_models', 'ProteinMPNN', 'ca_model_weights', 'v_48_020.pt')
        ),
        "f28f40170e21858c5ff31ef50b6e63414ff76dc331b19f85aa8586a12031744a",
        6624011,
        "proteinmpnn_ca_v48_020",
        False,
    ),
    (
        Path(
            joined('PROTEINA_COMPLEXA_ROOT', 'community_models', 'LigandMPNN', 'model_params', 'ligandmpnn_v_32_010_25.pt')
        ),
        "161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970",
        10541943,
        "ligandmpnn_v32_010_25",
        False,
    ),
    (
        Path(
            joined('PROTEINA_COMPLEXA_ROOT', 'community_models', 'ckpts', 'RF3', 'rf3_foundry_01_24_latest_remapped.ckpt')
        ),
        "364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0",
        3038876446,
        "rf3_foundry",
        False,
    ),
)

def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SuiteError(f"git {' '.join(args)} failed in {path}: {result.stderr}")
    return result.stdout.strip()

def _refuse_destination(path: Path) -> None:
    text = os.path.abspath(str(path))
    for prefix in DENIED_PREFIXES:
        if text.startswith(str(prefix)):
            raise SuiteError(f"denied evidence prefix: {path}")
    refuse_sealed_path(path)

def _write_json(path: Path, payload: Mapping[str, object]) -> str:
    _refuse_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def _calibration_examples() -> dict[Family, CanonicalExample]:
    binder = CanonicalExample(
        example_id="cal-binder-1",
        parent_id="cal-binder-parent",
        family=Family.BINDER,
        source_release="2024-06",
        pdb_id="1abc",
        assembly_id="1",
        deposition_date=None,
        chains=(
            ChainRecord("A", "target", "ACDE"),
            ChainRecord("B", "designed", "FGHI"),
        ),
        ligands=(),
    )
    ame = CanonicalExample(
        example_id="cal-ame-1",
        parent_id="cal-ame-parent",
        family=Family.AME,
        source_release="2024-06",
        pdb_id="2enz",
        assembly_id="1",
        deposition_date=None,
        chains=(ChainRecord("A", "designed", "ACDEFGHIK"),),
        ligands=(LigandRecord("LIG", "CCO", None, ""),),
    )
    antibody = CanonicalExample(
        example_id="cal-ab-1",
        parent_id="cal-ab-parent",
        family=Family.ANTIBODY,
        source_release="2024-06",
        pdb_id="3ab1",
        assembly_id="1",
        deposition_date=None,
        chains=(
            ChainRecord("H", "antibody", "EVQLVESGG"),
            ChainRecord("A", "antigen", "ACDE"),
        ),
        ligands=(),
    )
    return {Family.BINDER: binder, Family.AME: ame, Family.ANTIBODY: antibody}

def _public_examples() -> dict[Family, CanonicalExample]:

    binder = CanonicalExample(
        example_id="pub-binder-1",
        parent_id="pub-binder-parent",
        family=Family.BINDER,
        source_release="2024-06",
        pdb_id="1pub",
        assembly_id="1",
        deposition_date=None,
        chains=(
            ChainRecord("A", "target", "ACDEFG"),
            ChainRecord("B", "designed", "HIKLMN"),
        ),
        ligands=(),
    )
    ame = CanonicalExample(
        example_id="pub-ame-1",
        parent_id="pub-ame-parent",
        family=Family.AME,
        source_release="2024-06",
        pdb_id="1ame",
        assembly_id="1",
        deposition_date=None,
        chains=(ChainRecord("A", "designed", "ACDEFGHIKLMN"),),
        ligands=(LigandRecord("LIG", "CCN", None, ""),),
    )
    antibody = CanonicalExample(
        example_id="pub-ab-1",
        parent_id="pub-ab-parent",
        family=Family.ANTIBODY,
        source_release="2024-06",
        pdb_id="1vhh",
        assembly_id="1",
        deposition_date=None,
        chains=(
            ChainRecord("H", "antibody", "EVQLVESGGG"),
            ChainRecord("A", "antigen", "ACDEFG"),
        ),
        ligands=(),
    )
    return {Family.BINDER: binder, Family.AME: ame, Family.ANTIBODY: antibody}

def _legacy_examples() -> dict[Family, CanonicalExample]:
    binder = CanonicalExample(
        example_id="paper-smoke-real-20260829q-binder",
        parent_id="legacy-binder-29q",
        family=Family.BINDER,
        source_release="2026-08-31",
        pdb_id=None,
        assembly_id=None,
        deposition_date=None,
        chains=(
            ChainRecord("A", "target", "ACDE"),
            ChainRecord("B", "designed", "FGHI"),
        ),
        ligands=(),
        metadata={"seen_in": "paper-smoke-real-20260829q"},
    )
    ame = CanonicalExample(
        example_id="paper-smoke-real-20260829q-ame",
        parent_id="legacy-ame-29q",
        family=Family.AME,
        source_release="2026-08-31",
        pdb_id=None,
        assembly_id=None,
        deposition_date=None,
        chains=(ChainRecord("A", "designed", "ACDEFGHIK"),),
        ligands=(LigandRecord("LIG", "CCO", None, ""),),
        metadata={"seen_in": "paper-smoke-real-20260829q"},
    )
    antibody = CanonicalExample(
        example_id="paper-smoke-real-20260829q-antibody",
        parent_id="legacy-ab-29q",
        family=Family.ANTIBODY,
        source_release="2026-08-31",
        pdb_id=None,
        assembly_id=None,
        deposition_date=None,
        chains=(
            ChainRecord("H", "antibody", "EVQLVESGG"),
            ChainRecord("A", "antigen", "ACDE"),
        ),
        ligands=(),
        metadata={"seen_in": "paper-smoke-real-20260829q"},
    )
    return {Family.BINDER: binder, Family.AME: ame, Family.ANTIBODY: antibody}

def build_live_inventory() -> dict[str, object]:
    worktree = Path(__file__).resolve().parents[3]
    dive_canonical = Path(ROOT)
    repos = [
        classify_repository(
            path=str(worktree),
            url="local-dive-worktree",
            commit=_git(worktree, "rev-parse", "HEAD"),
            branch=_git(worktree, "rev-parse", "--abbrev-ref", "HEAD"),
            dirty=bool(_git(worktree, "status", "--porcelain")),
            owner="codirect-benchmark",
        ),
        classify_repository(
            path=str(EMERGENT_UPSTREAM_ROOT),
            url="https://github.com/NVIDIA-Digital-Bio/proteina-complexa.git",
            commit=_git(EMERGENT_UPSTREAM_ROOT, "rev-parse", "HEAD"),
            branch=_git(EMERGENT_UPSTREAM_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
            dirty=bool(_git(EMERGENT_UPSTREAM_ROOT, "status", "--porcelain")),
            owner="emergent",
        ),
        classify_repository(
            path=str(UPSTREAM_ROOT),
            url="https://github.com/NVIDIA-Digital-Bio/proteina-complexa.git",
            commit=_git(UPSTREAM_ROOT, "rev-parse", "HEAD"),
            branch=_git(UPSTREAM_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
            dirty=bool(_git(UPSTREAM_ROOT, "status", "--porcelain")),
            owner="signed-value",
        ),
        classify_repository(
            path=str(dive_canonical),
            url="local-dive-canonical",
            commit=_git(dive_canonical, "rev-parse", "HEAD"),
            branch=_git(dive_canonical, "rev-parse", "--abbrev-ref", "HEAD"),
            dirty=bool(_git(dive_canonical, "status", "--porcelain")),
            owner="canonical-dirty-do-not-touch",
        ),
    ]
    if repos[1].commit != EMERGENT_UPSTREAM_COMMIT:
        raise SuiteError("emergent Proteina-Complexa pin drifted")
    if repos[2].commit != SIGNED_VALUE_UPSTREAM_COMMIT:
        raise SuiteError("signed-value Proteina-Complexa pin drifted")
    checkpoints: list[CheckpointRecord] = []
    for path, sha, size, architecture, lora in _AUTHENTICATED_CHECKPOINTS:
        checkpoints.append(
            classify_checkpoint(
                path=path,
                expected_sha256=sha,
                expected_size_bytes=size,
                expected_architecture=architecture,
                has_lora=lora,
                official=True,
            )
        )
    environments = (
        EnvironmentRecord(
            path=str(RUNTIME_PYTHON.parent.parent),
            interpreter=str(RUNTIME_PYTHON),
            status=AssetStatus.REUSE_EXACT,
        ),
    )
    prior = (
        {
            "example_id": "paper-smoke-real-20260829q-binder",
            "parent_id": "legacy-binder-29q",
            "family": "binder",
            "seen_in": "paper-smoke-real-20260829q",
        },
        {
            "example_id": "paper-smoke-real-20260829q-ame",
            "parent_id": "legacy-ame-29q",
            "family": "ame",
            "seen_in": "paper-smoke-real-20260829q",
        },
        {
            "example_id": "paper-smoke-real-20260829q-antibody",
            "parent_id": "legacy-ab-29q",
            "family": "antibody",
            "seen_in": "paper-smoke-real-20260829q",
        },
        {
            "example_id": "paper-baseline-reduced-b-20260903",
            "parent_id": "reduced-b-campaign",
            "family": "all",
            "seen_in": "paper-baseline-reduced-b-20260903",
        },
    )
    inventory = inventory_from_observations(
        repositories=repos,
        checkpoints=checkpoints,
        environments=environments,
        prior_targets=prior,
    )

    def _record(item: object) -> dict[str, object]:
        payload = asdict(item)
        if "status" in payload:
            payload["status"] = str(payload["status"])
        return payload

    return {
        "checkpoints": [_record(item) for item in inventory.checkpoints],
        "environments": [_record(item) for item in inventory.environments],
        "prior_targets": [dict(item) for item in inventory.prior_targets],
        "repositories": [_record(item) for item in inventory.repositories],
        "schema_version": "dive-codirect-inventory-v1",
        "upstream_pins": {
            "emergent": EMERGENT_UPSTREAM_COMMIT,
            "signed_value": SIGNED_VALUE_UPSTREAM_COMMIT,
        },
    }

def materialize_suite(root: Path = SUITE_ROOT) -> dict[str, str]:

    _refuse_destination(root / "MANIFEST.json")
    hashes: dict[str, str] = {}
    inventory = build_live_inventory()
    hashes["inventory/assets.json"] = _write_json(
        root / "inventory" / "assets.json", inventory
    )
    hashes["inventory/checkpoints.json"] = _write_json(
        root / "inventory" / "checkpoints.json",
        {"checkpoints": inventory["checkpoints"]},
    )
    hashes["inventory/environments.json"] = _write_json(
        root / "inventory" / "environments.json",
        {"environments": inventory["environments"]},
    )
    prior_path = root / "inventory" / "prior-targets.csv"
    _refuse_destination(prior_path)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    with prior_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["example_id", "parent_id", "family", "seen_in", "partition"],
        )
        writer.writeheader()
        writer.writerows(inventory["prior_targets"])
    hashes["inventory/prior-targets.csv"] = hashlib.sha256(
        prior_path.read_bytes()
    ).hexdigest()

    calibration = _calibration_examples()
    public = _public_examples()
    legacy = _legacy_examples()
    ligand = LigandGraph(
        ligand_id="LIG",
        atom_names=("C1", "O1"),
        elements=("C", "O"),
        bonds=((0, 1, 1),),
        formal_charges=(0, 0),
        stereochemistry="none",
    )
    cards = {
        "binder": binder_task_card(
            public[Family.BINDER], hotspot_residues=("A-1",)
        ).as_mapping(),
        "ame": enzyme_task_card(
            public[Family.AME],
            ligand_graph=ligand,
            catalytic_atoms=("A-57-NE2",),
        ).as_mapping(),
        "antibody": vhh_cdr_h3_task_card(
            public[Family.ANTIBODY],
            epitope_hotspots=("A-4",),
            cdr_h3_length=7,
            framework_id="vhh-fixed-1",
        ).as_mapping(),
    }
    hashes["task_cards/public.json"] = _write_json(
        root / "task_cards" / "public.json", cards
    )

    manifest_counts: dict[str, object] = {}
    inspected = frozenset(example.parent_id for example in legacy.values()) | frozenset(
        {"reduced-b-campaign"}
    )
    for family in Family:
        bundle = build_family_manifests(
            family=family,
            public=(public[family],),
            legacy_dev=(legacy[family],),
            calibration=(calibration[family],),
            sealed_parent_ids=(),
            inspected_parent_ids=inspected,
        )
        manifest_counts[str(family)] = {
            "calibration": bundle.counts[ManifestPartition.CALIBRATION],
            "legacy_dev": bundle.counts[ManifestPartition.LEGACY_DEV],
            "manifest_sha256": bundle.manifest_sha256,
            "parse_success_rate": bundle.parse_success_rate,
            "public": bundle.counts[ManifestPartition.PUBLIC],
            "sealed_bound": bundle.sealed_bound,
            "sealed_parent_count": bundle.sealed_parent_count,
        }
    hashes["manifests/family_counts.json"] = _write_json(
        root / "manifests" / "family_counts.json", manifest_counts
    )

    envelopes = {}
    for family in ("binder", "ame", "antibody"):
        envelope = bind_sealed_envelope(
            view_semantic_hash="f5e54f8e0d91fe857f3d36135fb603f921523fa16563eb83d2b099e34ac8d253",
            view_partition_hash="527c359aefa234ea3015f645a8d128b587ad22c7e0b1aa263491c6a54a11fc03",
            family=family,
            metric_hash="aa" * 32,
            threshold_hash="bb" * 32,
            seed=20260826,
            budget_gpu_seconds=0,
        )
        envelopes[family] = envelope.as_mapping()
        hashes[f"manifests/{family}_envelope.json"] = _write_json(
            root / "manifests" / f"{family}_envelope.json",
            envelope.as_mapping(),
        )

    registry = load_codirect_baseline_registry()
    hashes["baselines/registry.json"] = _write_json(
        root / "baselines" / "registry.json",
        {
            "baselines": [
                {
                    "baseline_id": row.baseline_id,
                    "commit": row.commit,
                    "common_redesign": row.common_redesign,
                    "families": list(row.families),
                    "native_pipeline": row.native_pipeline,
                    "official_url": row.official_url,
                    "readiness": row.readiness.value,
                    "reason": row.reason,
                    "role": row.role,
                    "weight_sha256": row.weight_sha256,
                }
                for row in registry.baselines
            ],
            "schema_version": registry.schema_version,
        },
    )
    matrix_path = root / "baseline_status.csv"
    _refuse_destination(matrix_path)
    with matrix_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["baseline_id", "families", "role", "readiness", "reason"],
        )
        writer.writeheader()
        for row in registry.baselines:
            writer.writerow(
                {
                    "baseline_id": row.baseline_id,
                    "families": ",".join(row.families),
                    "role": row.role,
                    "readiness": row.readiness.value,
                    "reason": row.reason,
                }
            )
    hashes["baseline_status.csv"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()

    condition_path = root / "method_condition_matrix.csv"
    _refuse_destination(condition_path)
    with condition_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["baseline_id", "family", "condition", "consumption"],
        )
        writer.writeheader()
        for row in condition_matrix_rows(registry):
            writer.writerow(asdict(row))
    hashes["method_condition_matrix.csv"] = hashlib.sha256(
        condition_path.read_bytes()
    ).hexdigest()

    report = run_public_dev_pilot(registry)
    hashes["public_dev/pilot.json"] = _write_json(
        root / "public_dev" / "pilot.json",
        {
            "common_redesign_rows": len(report.common_redesign),
            "failures": len(report.failures_unsupported_unavailable),
            "family_smoke": {
                family: [
                    {
                        "baseline_id": item.baseline_id,
                        "passed": item.passed,
                        "readiness": item.readiness.value,
                        "roundtrip_ok": item.roundtrip_ok,
                        "score": item.score,
                    }
                    for item in results
                ]
                for family, results in report.family_smoke.items()
            },
            "native_pipeline_rows": len(report.native_pipeline),
            "sealed_opened": report.sealed_opened,
        },
    )
    write_ledger(root / "compute_ledger.parquet", report.compute_ledger)
    hashes["compute_ledger.parquet"] = hashlib.sha256(
        (root / "compute_ledger.parquet").read_bytes()
    ).hexdigest()

    hashes["smoke/controls.json"] = _write_json(
        root / "smoke" / "controls.json",
        {
            family: {
                "agreement": evaluate_control_suite(
                    family
                ).locked_primary_agrees_with_secondary,
                "results": [
                    {
                        "kind": item.kind.value,
                        "passed": item.passed,
                        "status": item.status,
                    }
                    for item in evaluate_control_suite(family).results
                ],
            }
            for family in ("binder", "ame", "antibody")
        },
    )

    frozen = freeze_preregistration(
        primary_population="full_sealed_population",
        statistical_unit="target_cluster",
    )
    hashes["provenance/preregistration.json"] = _write_json(
        root / "provenance" / "preregistration.json", frozen.as_mapping()
    )
    hashes["provenance/bound_envelopes.json"] = _write_json(
        root / "provenance" / "bound_envelopes.json", envelopes
    )

    created = datetime.now(UTC).isoformat()
    hashes["MANIFEST.json"] = _write_json(
        root / "MANIFEST.json",
        {
            "created_at": created,
            "hashes": hashes,
            "run_id": SUITE_RUN_ID,
            "schema_version": "dive-codirect-suite-v1",
            "sealed_opened": False,
        },
    )
    sums_path = root / "SHA256SUMS"
    _refuse_destination(sums_path)
    lines = [f"{digest}  {name}\n" for name, digest in sorted(hashes.items())]
    sums_path.write_text("".join(lines), encoding="utf-8")
    hashes["SHA256SUMS"] = hashlib.sha256(sums_path.read_bytes()).hexdigest()
    return hashes
