
from __future__ import annotations

from dive.codirect_paths import cache_dir

import json
import time
from pathlib import Path

from dive.benchmark.paper_baseline import PaperArm, family_steps
from dive.benchmark.paper_batches import (
    PaperBatchError,
    PaperRequestedCellFailure,
    load_paper_batch,
)
from dive.benchmark.paper_export import export_generated_sample
from dive.benchmark.paper_generation import (
    build_model_for_arm,
    generate_paper_sample,
)
from dive.benchmark.paper_inputs import (
    PaperTargetRecord,
    _ame_ligand_identity,
    _load_resolved_ligand_smiles,
)
from dive.benchmark.public_dev_materialize import SUITE_V2
from dive.signed_value.roots import DENIED_PREFIXES, EMERGENT_BULK_ROOT
from dive.training.preflight import canonical_json_bytes

class ProteinaExecuteError(RuntimeError):
    pass

def _refuse(path: Path) -> Path:
    text = str(path.resolve())
    for prefix in DENIED_PREFIXES:
        if text.startswith(str(prefix)):
            raise ProteinaExecuteError(f"denied {path}")
    return Path(text)

_RESOLVED_SMILES: object | None = None

def target_from_task(task: dict[str, object], *, family: str) -> PaperTargetRecord:
    from dive.benchmark.contracts import file_identity

    source = Path(str(task["path"]))
    identity = file_identity(source)
    loader = cache_dir("emergent", "loader_manifests_v5",
        f"{family}_validation.parquet",
    )
    token = None
    ligand_identity = None
    smiles = None
    if family == "ame":
        global _RESOLVED_SMILES
        if _RESOLVED_SMILES is None:
            _RESOLVED_SMILES = _load_resolved_ligand_smiles()
        token, ligand_identity, smiles = _ame_ligand_identity(
            family, str(task["example_id"]), _RESOLVED_SMILES
        )
    return PaperTargetRecord(
        parent_id=str(task["parent_id"]),
        example_id=str(task["example_id"]),
        family=family,
        partition="validation",
        strict=True,
        source_path=str(source),
        source_sha256=identity.sha256,
        role_payload={
            "generated": str(task["generated"]),
            "context": str(task["context"]),
            "target": str(task["target"]),
        },
        loader_manifest_sha256=file_identity(loader).sha256,
        canonical_component_token=token,
        ligand_identity=ligand_identity,
        ligand_smiles=smiles,
    )

def run_selected(
    *,
    families: tuple[str, ...] = ("binder", "ame"),
    n_public: int = 2,
    seed: int = 42001,
    device: str = "cuda",
    example_ids: tuple[str, ...] | None = None,
    max_design_fraction: float | None = None,
    target_budget: int | None = None,
    evidence_subdir: str = "proteina",
    shipped_protocol: bool = False,
    arm: PaperArm = PaperArm.BASE_COMMON_EXACT_V1,
) -> dict[str, object]:

    cohort_path = SUITE_V2 / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    evidence = _refuse(SUITE_V2 / evidence_subdir)
    bulk = _refuse(EMERGENT_BULK_ROOT / "codirect-public-dev-v2" / evidence_subdir)
    evidence.mkdir(parents=True, exist_ok=True)
    bulk.mkdir(parents=True, exist_ok=True)
    from pathlib import Path as P

    from dive.emergent_contract import EmergentContract

    contract = EmergentContract.from_yaml(P("configs/emergent/resources.yaml"))
    store = bulk / "store"
    store.mkdir(parents=True, exist_ok=True)
    model, report = build_model_for_arm(arm, contract, store_dir=store)
    model = model.to(device).eval()
    rows: list[dict[str, object]] = []
    for family in families:
        tasks = [
            task
            for task in cohort["families"][family]["tasks"]
            if (
                example_ids is not None and task["example_id"] in example_ids
            )
            or (
                example_ids is None and task["partition"] == "public"
            )
        ][:n_public]
        for task in tasks:
            started = time.perf_counter()
            record: dict[str, object] = {
                "family": family,
                "example_id": task["example_id"],
                "seed": seed,
                "result_origin": "fresh_run",
            }
            try:
                target = target_from_task(task, family=family)
                batch = load_paper_batch(
                    target,
                    seed=seed,
                    max_design_fraction=max_design_fraction,
                    target_budget=target_budget,
                )
                tensors = {
                    key: (value.to(device) if hasattr(value, "to") else value)
                    for key, value in batch.tensors.items()
                }
                sample = generate_paper_sample(
                    model, tensors, family=family, seed=seed,
                    shipped_protocol=shipped_protocol,
                )
                out_dir = evidence / family / str(task["example_id"]).replace("/", "_")
                out_dir.mkdir(parents=True, exist_ok=True)
                import torch

                pt_path = out_dir / f"sample_seed{seed}.pt"
                torch.save(
                    {
                        k: (v.detach().cpu() if hasattr(v, "detach") else v)
                        for k, v in sample.items()
                    },
                    pt_path,
                )
                record["sample_pt"] = str(pt_path)
                try:
                    exported = export_generated_sample(
                        sample=sample, batch=batch, output_dir=out_dir
                    )
                except Exception as export_error:
                    record.update(
                        {
                            "status": "generated_export_failed",
                            "export_error": type(export_error).__name__,
                            "export_detail": str(export_error)[:1000],
                            "steps": family_steps(
                                family, panel="A", arm=arm,
                                shipped_protocol=shipped_protocol,
                            ),
                            "wall_seconds": time.perf_counter() - started,
                            "export_dir": str(out_dir),
                            "live_parameter_sha256": report.live_parameter_sha256,
                        }
                    )
                    rows.append(record)
                    with (evidence / "progress.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    continue
                record.update(
                    {
                        "status": "complete",
                        "steps": family_steps(
                            family, panel="A", arm=arm,
                            shipped_protocol=shipped_protocol,
                        ),
                        "wall_seconds": time.perf_counter() - started,
                        "export_dir": str(out_dir),
                        "complex_pdb": exported.complex_pdb.path,
                        "complex_pdb_sha256": exported.complex_pdb.sha256,
                        "live_parameter_sha256": report.live_parameter_sha256,
                    }
                )
            except (PaperRequestedCellFailure, PaperBatchError, Exception) as error:
                record.update(
                    {
                        "status": "failed",
                        "reason": type(error).__name__,
                        "detail": str(error)[:2000],
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
            rows.append(record)
            with (evidence / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
    payload = {
        "arm": arm.value,
        "live_parameter_sha256": report.live_parameter_sha256,
        "rows": rows,
        "seed": seed,
    }
    (evidence / "generation.json").write_bytes(canonical_json_bytes(payload))
    return payload

def reexport_ame_samples(*, seed: int = 42001) -> tuple[dict[str, object], ...]:

    import torch

    cohort = json.loads((SUITE_V2 / "cohort.json").read_text(encoding="utf-8"))
    evidence = _refuse(SUITE_V2 / "proteina" / "ame")
    rows: list[dict[str, object]] = []
    tasks = {
        str(task["example_id"]): task for task in cohort["families"]["ame"]["tasks"]
    }
    for pt_path in sorted(evidence.glob(f"*/sample_seed{seed}.pt")):
        example_id = pt_path.parent.name
        task = tasks.get(example_id)
        record: dict[str, object] = {
            "example_id": example_id,
            "seed": seed,
            "result_origin": "fresh_run",
            "sample_pt": str(pt_path),
        }
        if task is None:
            record.update({"status": "failed", "reason": "missing_cohort_row"})
            rows.append(record)
            continue
        target = target_from_task(task, family="ame")
        batch = load_paper_batch(target, seed=seed)
        sample = torch.load(pt_path, map_location="cpu", weights_only=False)
        exported = export_generated_sample(
            sample=sample, batch=batch, output_dir=pt_path.parent
        )
        record.update(
            {
                "status": "complete",
                "complex_pdb": exported.complex_pdb.path,
                "complex_pdb_sha256": exported.complex_pdb.sha256,
            }
        )
        rows.append(record)
    payload = {"rows": rows, "seed": seed}
    out = SUITE_V2 / "proteina" / "ame_reexport.json"
    _refuse(out).write_bytes(canonical_json_bytes(payload))
    return tuple(rows)
