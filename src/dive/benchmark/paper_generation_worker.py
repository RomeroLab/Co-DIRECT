
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dive.benchmark import paper_baseline as paper_baseline_mod
from dive.benchmark.contracts import ArtifactIdentity, file_identity
from dive.benchmark.paper_baseline import (
    panel_arm,
    PAPER_SEEDS,
    PaperArm,
    PaperBaselineError,
    PaperCell,
    PaperPlanManifest,
    family_steps,
)
from dive.benchmark.paper_batches import PaperBatch
from dive.benchmark.paper_generation import (
    SpliceReport,
    generate_paper_sample,
    hash_live_parameters,
    hash_splice_report,
)
from dive.signed_value.roots import DENIED_PREFIXES

@dataclass(frozen=True, slots=True)
class GeneratedSampleRecord:
    cell_id: str
    status: str
    family: str
    seed: int
    steps: int
    output_path: str
    output_sha256: str
    output_size_bytes: int
    denoiser_calls: int
    wall_seconds: float
    peak_memory_bytes: int
    splice_report_sha256: str
    live_parameter_sha256: str

@dataclass(frozen=True, slots=True)
class GenerationShardRequest:
    plan: PaperPlanManifest
    cells: tuple[PaperCell, ...]
    batches: Mapping[str, PaperBatch]
    evidence_dir: Path
    bulk_dir: Path
    model: object | None = None
    splice_report: SpliceReport | None = None
    store_dir: Path | None = None

def run_generation_shard(
    request: GenerationShardRequest,
    shipped_protocol: bool = False,
) -> tuple[GeneratedSampleRecord, ...]:
    evidence_dir = _refuse_denied(Path(request.evidence_dir))
    bulk_dir = _refuse_denied(Path(request.bulk_dir))
    _verify_git_and_identities(request)
    model, splice = _load_model_once(request)
    live_sha = _recheck_live_hashes(model, splice, request.plan)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    bulk_dir.mkdir(parents=True, exist_ok=True)
    progress = evidence_dir / "progress.jsonl"
    completed = _completed_records(progress)
    records: list[GeneratedSampleRecord] = []
    for cell in request.cells:
        existing = completed.get(cell.cell_id)
        output = bulk_dir / cell.family / f"{cell.cell_id}.pt"
        _refuse_partials(output, cell.cell_id)
        if existing is not None:
            records.append(_resume_or_refuse(existing, output))
            continue
        batch = _batch_for_cell(request.batches, cell)
        started = time.perf_counter()
        calls = 0

        def count_call(_module, _args):
            nonlocal calls
            calls += 1

        hook = model.nn.register_forward_pre_hook(count_call)
        try:
            sample = generate_paper_sample(
                model,
                _clone_tensors(batch.tensors, device=_inference_device()),
                family=cell.family,
                seed=cell.seed,
                shipped_protocol=shipped_protocol,
            )
        finally:
            hook.remove()
        if calls != cell.steps:
            raise PaperBaselineError(
                f"single-pass generation made {calls} denoiser calls"
            )
        if sample["coors"].shape[0] != 1:
            raise PaperBaselineError("generation requires exactly one result")
        payload = {
            "schema": "dive.paper.generated_sample.v1",
            "cell_id": cell.cell_id,
            "family": cell.family,
            "seed": cell.seed,
            "sample": _to_cpu(sample),
            "chain_names": deepcopy(batch.tensors.get("chain_names")),
            "residue_pdb_idx": _to_cpu_value(batch.tensors.get("residue_pdb_idx")),
            "arm": cell.arm,
            "panel": cell.panel,
        }
        identity = _publish_tensor(payload, output)
        record = GeneratedSampleRecord(
            cell_id=cell.cell_id,
            status="complete",
            family=cell.family,
            seed=cell.seed,
            steps=cell.steps,
            output_path=str(output),
            output_sha256=identity.sha256,
            output_size_bytes=identity.size_bytes,
            denoiser_calls=calls,
            wall_seconds=time.perf_counter() - started,
            peak_memory_bytes=_peak_memory_bytes(),
            splice_report_sha256=request.plan.splice_report_sha256,
            live_parameter_sha256=live_sha,
        )
        _append_progress(progress, record)
        records.append(record)
    return tuple(records)

def _refuse_denied(path: Path) -> Path:
    resolved = Path(path).resolve(strict=False)
    for denied in DENIED_PREFIXES:
        prefix = Path(denied).resolve(strict=False)
        if resolved == prefix or prefix in resolved.parents:
            raise PaperBaselineError(f"root is denied: {resolved}")
    return resolved

def _verify_git_and_identities(request: GenerationShardRequest) -> None:
    observed = paper_baseline_mod._clean_repo_commit()
    plan = request.plan
    if plan.code_commit != observed:
        raise PaperBaselineError("bad code commit")
    if plan.arm != panel_arm("A"):
        raise PaperBaselineError(f"panel A requires {panel_arm('A').value}")
    if not request.cells:
        raise PaperBaselineError("generation shard has no cells")
    for cell in request.cells:
        if cell.code_commit != plan.code_commit:
            raise PaperBaselineError("bad code commit")
        if cell.splice_report_sha256 != plan.splice_report_sha256:
            raise PaperBaselineError("splice report hash drifted")
        if cell.checkpoint != plan.checkpoint_sha256:
            raise PaperBaselineError("checkpoint identity drifted")
        if cell.arm != panel_arm("A"):
            raise PaperBaselineError(f"panel A requires {panel_arm('A').value}")
        if cell.seed not in PAPER_SEEDS:
            raise PaperBaselineError("generation seed must come from the plan cell")
        expected = family_steps(
            cell.family,
            panel="A",
            arm=panel_arm("A"),
            steps=cell.steps,
        )
        if cell.steps != expected:
            raise PaperBaselineError("sampling contract drifted")
        _batch_for_cell(request.batches, cell)

def _batch_for_cell(batches: Mapping[str, PaperBatch], cell: PaperCell) -> PaperBatch:
    try:
        batch = batches[cell.example_id]
    except KeyError as error:
        raise PaperBaselineError(
            f"missing batch for example {cell.example_id!r}"
        ) from error
    if batch.target.example_id != cell.example_id:
        raise PaperBaselineError("identity mismatch")
    if batch.target.family != cell.family:
        raise PaperBaselineError("identity mismatch")
    if batch.target.parent_id != cell.parent_id:
        raise PaperBaselineError("identity mismatch")
    return batch

def _load_model_once(request: GenerationShardRequest):
    if request.model is not None:
        if request.splice_report is None:
            raise PaperBaselineError("injected model requires a splice report")
        return _prepare_model_for_inference(request.model), request.splice_report
    from dive.benchmark.paper_generation import (
        assert_load_matches_arm,
        build_model_for_arm,
    )
    from dive.emergent_contract import EmergentContract

    store_dir = request.store_dir or (Path(request.evidence_dir) / "store")
    contract = EmergentContract.from_yaml(Path("configs/emergent/resources.yaml"))

    arm = request.plan.arm
    model, splice = build_model_for_arm(arm, contract, store_dir=store_dir)
    assert_load_matches_arm(arm, splice)
    return _prepare_model_for_inference(model), splice

def _prepare_model_for_inference(model):
    if os.environ.get("DIVE_PAPER_REQUIRE_MODEL_EVAL") != "1":
        return model
    model.eval()
    device = _inference_device()
    if device.type == "cuda":
        return model.cuda()
    return model

def _inference_device():
    import torch

    if (
        os.environ.get("DIVE_PAPER_REQUIRE_MODEL_EVAL") == "1"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")
    return torch.device("cpu")

def _recheck_live_hashes(model, splice: SpliceReport, plan: PaperPlanManifest) -> str:
    live = hash_live_parameters(model)
    if splice.live_parameter_sha256 != live:
        raise PaperBaselineError("live parameter hash drifted")
    observed = hash_splice_report(splice)
    if observed != plan.splice_report_sha256:
        raise PaperBaselineError("splice report hash drifted")
    return live

def _completed_records(progress: Path) -> dict[str, GeneratedSampleRecord]:
    if not progress.exists():
        return {}
    completed: dict[str, GeneratedSampleRecord] = {}
    for line in progress.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("status") != "complete":
            continue
        completed[str(payload["cell_id"])] = _record_from_mapping(payload)
    return completed

def _record_from_mapping(payload: Mapping[str, Any]) -> GeneratedSampleRecord:
    return GeneratedSampleRecord(
        cell_id=str(payload["cell_id"]),
        status=str(payload["status"]),
        family=str(payload["family"]),
        seed=int(payload["seed"]),
        steps=int(payload["steps"]),
        output_path=str(payload["output_path"]),
        output_sha256=str(payload["output_sha256"]),
        output_size_bytes=int(payload["output_size_bytes"]),
        denoiser_calls=int(payload["denoiser_calls"]),
        wall_seconds=float(payload["wall_seconds"]),
        peak_memory_bytes=int(payload["peak_memory_bytes"]),
        splice_report_sha256=str(payload["splice_report_sha256"]),
        live_parameter_sha256=str(payload["live_parameter_sha256"]),
    )

def _resume_or_refuse(
    record: GeneratedSampleRecord, output: Path
) -> GeneratedSampleRecord:
    if not output.is_file() or output.is_symlink():
        raise PaperBaselineError("completed cell is missing its output")
    live = file_identity(output)
    if live.sha256 != record.output_sha256:
        raise PaperBaselineError("output hash drifted")
    if live.size_bytes != record.output_size_bytes:
        raise PaperBaselineError("output hash drifted")
    return record

def _refuse_partials(output: Path, cell_id: str) -> None:
    parent = output.parent
    if not parent.exists():
        return
    leftover = list(parent.glob(f".{cell_id}.pt.partial.*"))
    if leftover:
        raise PaperBaselineError("partial generation output exists")

def _publish_tensor(payload: Mapping[str, object], output: Path) -> ArtifactIdentity:
    import torch

    output = _refuse_denied(output)
    if output.exists() or output.is_symlink():
        raise PaperBaselineError("refusing to overwrite generation output")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise PaperBaselineError("partial generation output exists")
    try:
        torch.save(payload, partial)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    return file_identity(output)

def _append_progress(path: Path, record: GeneratedSampleRecord) -> None:
    path = _refuse_denied(path)
    payload = {
        "cell_id": record.cell_id,
        "denoiser_calls": record.denoiser_calls,
        "family": record.family,
        "live_parameter_sha256": record.live_parameter_sha256,
        "output_path": record.output_path,
        "output_sha256": record.output_sha256,
        "output_size_bytes": record.output_size_bytes,
        "peak_memory_bytes": record.peak_memory_bytes,
        "seed": record.seed,
        "splice_report_sha256": record.splice_report_sha256,
        "status": record.status,
        "steps": record.steps,
        "wall_seconds": record.wall_seconds,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o664)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _clone_tensors(
    tensors: Mapping[str, object], *, device: object | None = None
) -> dict[str, object]:
    import torch

    cloned: dict[str, object] = {}
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            item = value.clone()
            if device is not None and item.device != device:
                item = item.to(device)
            cloned[key] = item
        else:
            cloned[key] = value
    return cloned

def _to_cpu(sample: Mapping[str, object]) -> dict[str, object]:
    return {key: _to_cpu_value(value) for key, value in sample.items()}

def _to_cpu_value(value: object) -> object:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value

def _peak_memory_bytes() -> int:
    import torch

    if torch.cuda.is_initialized():
        return int(torch.cuda.max_memory_allocated())
    return 0
