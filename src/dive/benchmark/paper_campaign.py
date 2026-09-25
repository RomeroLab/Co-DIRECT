
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dive.benchmark.contracts import ArtifactIdentity, file_identity
from dive.benchmark.evaluators.contracts import MetricUnavailable, MetricValue
from dive.benchmark.paper_baseline import (
    PaperArm,
    DIVE_ARM_NAMES,
    FIXTURE_EVALUATOR_REGISTRY_SHA256,
    PRODUCTION_EVALUATOR_REGISTRY_SHA256,
    PaperBaselineError,
    PaperCell,
    PaperPlanManifest,
    PaperRun,
    _cell_mapping,
    _clean_repo_commit,
    claim_paper_baseline_run,
    complete_paper_baseline_run,
)
from dive.benchmark.paper_batches import (
    PaperBatch,
    PaperBatchError,
    assert_batch_matches_target,
    load_paper_batch,
)
from dive.benchmark.paper_evaluation import PaperEvaluationRequest
from dive.benchmark.paper_export import (
    PaperExportResult,
    _authenticate_generated_roles_against_batch,
    _authenticate_generated_rows_against_pdb,
    _frozen_antibody_roles,
    _generated_authority_rows,
    _label_to_auth_mapping,
    _native_atom_records,
    _predicted_atom_records,
    _read_authenticated_generated_sample,
    _residue_rows,
    ame_ligand_transport_from_mapping,
    binder_backbone_normalization_from_mapping,
)
from dive.benchmark.paper_pdb_auth import PdbAuthenticationError, read_authenticated_pdb
from dive.benchmark.paper_inputs import load_paper_inputs
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_COMMIT,
    EMERGENT_UPSTREAM_ROOT,
    RUNTIME_PYTHON,
)
from dive.training.directional_campaign import (
    MIN_RESERVE,
    OBSERVED_PROJECT_GPU_SECONDS,
    PROJECT_PLANNED_CAP,
    PROJECT_TOTAL,
)
from dive.training.preflight import canonical_json_bytes

class PaperCampaignError(PaperBaselineError):
    pass

class CampaignStage(StrEnum):
    PLANNED = "PLANNED"
    GENERATING = "GENERATING"
    GENERATED = "GENERATED"
    EXPORTING = "EXPORTING"
    EXPORTED = "EXPORTED"
    EVALUATING = "EVALUATING"
    EVALUATED = "EVALUATED"
    AGGREGATED = "AGGREGATED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

CAMPAIGN_STAGES: tuple[CampaignStage, ...] = (
    CampaignStage.PLANNED,
    CampaignStage.GENERATING,
    CampaignStage.GENERATED,
    CampaignStage.EXPORTING,
    CampaignStage.EXPORTED,
    CampaignStage.EVALUATING,
    CampaignStage.EVALUATED,
    CampaignStage.AGGREGATED,
    CampaignStage.COMPLETE,
)
CAMPAIGN_CLI_ACTIONS = frozenset(
    {"generate", "export", "evaluate", "reconcile", "report", "render", "ledger"}
)
_IN_PROGRESS = frozenset(
    {
        CampaignStage.GENERATING,
        CampaignStage.EXPORTING,
        CampaignStage.EVALUATING,
    }
)
_START_REQUIRES = {
    CampaignStage.GENERATING: CampaignStage.PLANNED,
    CampaignStage.EXPORTING: CampaignStage.GENERATED,
    CampaignStage.EVALUATING: CampaignStage.EXPORTED,
}
_RECONCILE_IN_PROGRESS = {
    CampaignStage.GENERATED: CampaignStage.GENERATING,
    CampaignStage.EXPORTED: CampaignStage.EXPORTING,
    CampaignStage.EVALUATED: CampaignStage.EVALUATING,
}
_SHA256 = frozenset("0123456789abcdef")
_WORKER = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "emergent"
    / "paper_baseline_production_worker.py"
)
_RUN_SCRIPT = "scripts/emergent/run_paper_baseline.py"
if OBSERVED_PROJECT_GPU_SECONDS != 220_279:
    raise RuntimeError("observed GPU-seconds ledger must remain 220279")

@dataclass(frozen=True, slots=True)
class GpuLedgerRecord:
    observed_gpu_seconds: int
    planned_gpu_seconds: int
    project_gpu_seconds: int
    reserve_gpu_seconds: int
    planned_cap_gpu_seconds: int
    min_reserve_gpu_seconds: int
    sampled: bool = False
    tuned: bool = False

@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: CampaignStage
    status: str
    cell_ids: tuple[str, ...]
    identities: Mapping[str, ArtifactIdentity]
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))

@dataclass(frozen=True, slots=True)
class CampaignShard:
    shard_id: str
    stage: CampaignStage
    cell_ids: tuple[str, ...]
    gpu: int
    directory: Path
    request_path: Path
    progress_path: Path
    terminal_path: Path
    manifest_path: Path
    log_path: Path

@dataclass(frozen=True, slots=True)
class _RetainedSourcePath:

    descriptor: int
    original_name: str

    @property
    def name(self) -> str:
        return self.original_name

    def __str__(self) -> str:
        return f"/proc/self/fd/{self.descriptor}"

    def open(self, *args, **kwargs):
        return Path(str(self)).open(*args, **kwargs)

class PaperCampaignController:

    def __init__(
        self,
        *,
        run: PaperRun,
        plan: PaperPlanManifest,
        occupied_gpus: frozenset[int] | None,
        pid_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self.run = run
        self.plan = plan
        self._occupied_gpus = occupied_gpus
        self._pid_alive = pid_alive or _pid_alive

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        plan: PaperPlanManifest,
        evidence_root: Path,
        bulk_root: Path,
        argv: Sequence[str],
        occupied_gpus: frozenset[int] | None = None,
        pid_alive: Callable[[int], bool] | None = None,
    ) -> PaperCampaignController:
        _validate_run_id(run_id)
        _refuse_denied(Path(evidence_root))
        _refuse_denied(Path(bulk_root))
        try:
            commit = _clean_repo_commit()
        except PaperBaselineError as error:
            raise PaperCampaignError(str(error)) from error
        upstream = _require_emergent_upstream()
        if upstream != EMERGENT_UPSTREAM_COMMIT:
            raise PaperCampaignError(f"wrong upstream {upstream}")
        _validate_plan(plan, commit)
        try:
            run = claim_paper_baseline_run(
                run_id,
                tuple(argv),
                evidence_root=Path(evidence_root),
                bulk_root=Path(bulk_root),
                repo_commit=commit,
            )
        except FileExistsError as error:
            raise PaperCampaignError(f"run id already exists: {run_id}") from error
        except PaperBaselineError as error:
            raise PaperCampaignError(str(error)) from error
        controller = cls(
            run=run,
            plan=plan,
            occupied_gpus=occupied_gpus,
            pid_alive=pid_alive,
        )
        controller._persist_plan()
        controller._write_stage_terminal(
            CampaignStage.PLANNED,
            status="COMPLETE",
            identities={},
            cell_ids=tuple(cell.cell_id for cell in plan.cells),
        )
        return controller

    @classmethod
    def open(
        cls,
        *,
        run_id: str,
        evidence_root: Path,
        bulk_root: Path,
        occupied_gpus: frozenset[int] | None = None,
        pid_alive: Callable[[int], bool] | None = None,
        expect_commit: str | None = None,
    ) -> PaperCampaignController:
        _validate_run_id(run_id)
        _refuse_denied(Path(evidence_root))
        _refuse_denied(Path(bulk_root))
        try:
            commit = _clean_repo_commit()
        except PaperBaselineError as error:
            raise PaperCampaignError(str(error)) from error
        upstream = _require_emergent_upstream()
        if upstream != EMERGENT_UPSTREAM_COMMIT:
            raise PaperCampaignError(f"wrong upstream {upstream}")
        evidence_dir = Path(evidence_root) / "paper_quality_baseline" / run_id
        bulk_dir = Path(bulk_root) / "paper_quality_baseline" / run_id
        if not evidence_dir.is_dir() or evidence_dir.is_symlink():
            raise PaperCampaignError(f"run id does not exist: {run_id}")
        attempt_path = evidence_dir / "attempt.json"
        if not attempt_path.is_file():
            raise PaperCampaignError("missing attempt record")
        raw = attempt_path.read_bytes()
        plan = _load_plan(evidence_dir / "plan.json")

        _validate_plan(plan, expect_commit or commit)
        run = PaperRun(
            run_id=run_id,
            evidence_dir=evidence_dir,
            bulk_dir=bulk_dir,
            attempt=ArtifactIdentity(
                str(attempt_path), hashlib.sha256(raw).hexdigest(), len(raw)
            ),
        )
        return cls(
            run=run,
            plan=plan,
            occupied_gpus=occupied_gpus,
            pid_alive=pid_alive,
        )

    @property
    def stage(self) -> CampaignStage:
        failed = self._stage_dir(CampaignStage.FAILED) / "terminal.json"
        if failed.is_file():
            return CampaignStage.FAILED
        if (self.run.evidence_dir / "completion.json").is_file():
            return CampaignStage.COMPLETE
        last = CampaignStage.PLANNED
        for stage in CAMPAIGN_STAGES:
            directory = self._stage_dir(stage)
            if (directory / "terminal.json").is_file() or (
                directory / "status.json"
            ).is_file():
                last = stage
        return last

    def start_stage(self, stage: CampaignStage) -> None:
        self._refuse_failed()
        if stage not in _START_REQUIRES:
            raise PaperCampaignError(f"cannot start stage {stage}")
        required = _START_REQUIRES[stage]
        self._require_complete(required)
        status_path = self._stage_dir(stage) / "status.json"
        self._stage_dir(stage).mkdir(parents=True, exist_ok=True)
        _write_create_new(
            status_path,
            canonical_json_bytes(
                {
                    "schema_version": "dive-paper-stage-status-v1",
                    "stage": stage.value,
                    "status": "IN_PROGRESS",
                    "run_id": self.run.run_id,
                }
            ),
        )

    def fail_stage(self, reason: str) -> StageRecord:
        record = self._write_stage_terminal(
            CampaignStage.FAILED,
            status="FAILED",
            identities={},
            cell_ids=(),
            reason=reason,
        )
        return record

    def create_shards(
        self,
        stage: CampaignStage,
        *,
        assignments: Sequence[tuple[str, Sequence[str], int]],
    ) -> tuple[CampaignShard, ...]:
        self._refuse_failed()
        if self.stage != stage or stage not in _IN_PROGRESS:
            raise PaperCampaignError(f"cannot create shards for {stage}")
        planned = {cell.cell_id for cell in self.plan.cells}
        seen: set[str] = set()
        prepared: list[tuple[str, tuple[str, ...], int]] = []
        gpus: list[int] = []
        for shard_id, cell_ids, gpu in assignments:
            if type(gpu) is not int:
                raise PaperCampaignError("each shard requires one explicit GPU")
            gpus.append(gpu)
            ids = tuple(cell_ids)
            if len(ids) != len(set(ids)):
                raise PaperCampaignError("duplicate cells")
            unplanned = [cell_id for cell_id in ids if cell_id not in planned]
            if unplanned:
                raise PaperCampaignError(f"unplanned cells {unplanned}")
            overlap = seen.intersection(ids)
            if overlap:
                raise PaperCampaignError(f"duplicate cells {sorted(overlap)}")
            seen.update(ids)
            prepared.append((shard_id, ids, gpu))
        _require_free_gpus(gpus, self._resolved_occupied(None))
        shards: list[CampaignShard] = []
        for shard_id, ids, gpu in prepared:
            shard = self._shard_for(stage, shard_id, ids, gpu)
            shard.directory.mkdir(parents=True, exist_ok=True)
            _write_create_new(
                shard.directory / "assignment.json",
                canonical_json_bytes(
                    {
                        "cell_ids": list(ids),
                        "gpu": gpu,
                        "shard_id": shard_id,
                        "stage": stage.value,
                    }
                ),
            )
            _write_create_new(shard.log_path, b"")
            shards.append(shard)
        return tuple(shards)

    def write_shard_request(self, shard: CampaignShard) -> ArtifactIdentity:
        payload = {
            "schema_version": "dive-paper-shard-request-v1",
            "arm": self.plan.arm,
            "cell_ids": list(shard.cell_ids),
            "cells": [
                _cell_mapping(cell)
                for cell in self.plan.cells
                if cell.cell_id in set(shard.cell_ids)
            ],
            "code_commit": self.plan.code_commit,
            "evaluator_registry_sha256": self.plan.evaluator_registry_sha256,
            "gpu": shard.gpu,
            "run_id": self.run.run_id,
            "shard_id": shard.shard_id,
            "splice_report_sha256": self.plan.splice_report_sha256,
            "stage": shard.stage.value,
        }
        return _write_create_new(shard.request_path, canonical_json_bytes(payload))

    def append_shard_progress(
        self, shard: CampaignShard, record: Mapping[str, object]
    ) -> None:
        if not isinstance(record, Mapping):
            raise PaperCampaignError("progress record must be a mapping")
        _append_jsonl(shard.progress_path, dict(record))

    def record_shard_start(self, shard: CampaignShard, *, pid: int) -> None:
        self.append_shard_progress(shard, {"event": "start", "pid": int(pid)})

    def record_shard_exit(
        self,
        shard: CampaignShard,
        *,
        exit_code: int | None = None,
        signal: int | None = None,
    ) -> None:
        payload: dict[str, object] = {"event": "exit"}
        if exit_code is not None:
            payload["exit_code"] = int(exit_code)
        if signal is not None:
            payload["signal"] = int(signal)
            payload["status"] = "failed"
        self.append_shard_progress(shard, payload)

    def publish_shard_terminal(
        self, shard: CampaignShard, payload: Mapping[str, object]
    ) -> ArtifactIdentity:
        progress = _read_progress(shard.progress_path)
        cells: dict[str, dict[str, object]] = {}
        for row in progress:
            if row.get("status") != "complete":
                continue
            cell_id = str(row["cell_id"])
            cells[cell_id] = {
                "path": row["output_path"],
                "sha256": row["output_sha256"],
                "size_bytes": row["output_size_bytes"],
            }
        manifest = {
            "schema_version": "dive-paper-shard-manifest-v1",
            "cells": cells,
            "run_id": self.run.run_id,
            "shard_id": shard.shard_id,
            "stage": shard.stage.value,
        }
        _write_create_new(shard.manifest_path, canonical_json_bytes(manifest))
        terminal = {
            "schema_version": "dive-paper-shard-terminal-v1",
            "payload": dict(payload),
            "run_id": self.run.run_id,
            "shard_id": shard.shard_id,
            "stage": shard.stage.value,
        }
        return _write_create_new(shard.terminal_path, canonical_json_bytes(terminal))

    def evaluate_cell(
        self,
        cell: PaperCell,
        export: PaperExportResult,
        work_dir: Path,
        *,
        expected_batch: PaperBatch | None = None,
        generated_sample: ArtifactIdentity | None = None,
    ):
        _validate_plan(self.plan, self.plan.code_commit)
        planned = {item.cell_id for item in self.plan.cells}
        if cell.cell_id not in planned:
            raise PaperCampaignError("unplanned cell")
        evaluation_work_dir = Path(work_dir)
        evaluation_work_dir.mkdir(parents=True, exist_ok=True)
        request = PaperEvaluationRequest(
            cell=cell,
            export=export,
            registry_sha256=self.plan.evaluator_registry_sha256,
            work_dir=evaluation_work_dir,
            expected_batch=expected_batch,
            generated_sample=generated_sample,
        )
        return self._evaluate_family(cell.family, request)

    def _evaluate_family(self, family: str, request: PaperEvaluationRequest):
        return _load_evaluate_family()(family, request)

    def execute_generation_shard(
        self,
        shard: CampaignShard,
        batches: Mapping[str, object],
        *,
        model: object | None = None,
        splice_report: object | None = None,
    ):
        from dive.benchmark.paper_generation_worker import (
            GenerationShardRequest,
            run_generation_shard,
        )

        cells = tuple(
            cell for cell in self.plan.cells if cell.cell_id in set(shard.cell_ids)
        )
        request = GenerationShardRequest(
            plan=self.plan,
            cells=cells,
            batches=batches,
            evidence_dir=shard.directory,
            bulk_dir=Path(self.run.bulk_dir) / "generation" / shard.shard_id,
            model=model,
            splice_report=splice_report,
        )
        return run_generation_shard(request)

    def execute_export_cell(
        self, *, sample: Mapping[str, object], batch, output_dir: Path
    ):
        from dive.benchmark.paper_export import export_generated_sample

        return export_generated_sample(
            sample=sample, batch=batch, output_dir=output_dir
        )

    def shard_by_id(self, shard_id: str) -> CampaignShard:
        for stage in _IN_PROGRESS:
            for shard in self._list_shards(stage):
                if shard.shard_id == shard_id:
                    return shard
        raise PaperCampaignError(f"unknown shard {shard_id}")

    def _persist_plan(self) -> ArtifactIdentity:
        payload = {
            "arm": self.plan.arm,
            "autoencoder_sha256": self.plan.autoencoder_sha256,
            "blind_opened": self.plan.blind_opened,
            "cells": [_cell_mapping(cell) for cell in self.plan.cells],
            "checkpoint_sha256": self.plan.checkpoint_sha256,
            "code_commit": self.plan.code_commit,
            "evaluator_registry_sha256": self.plan.evaluator_registry_sha256,
            "example_counts": dict(self.plan.example_counts),
            "executed_dive_arms": list(self.plan.executed_dive_arms),
            "family_cells": dict(self.plan.family_cells),
            "foldseek_executed": self.plan.foldseek_executed,
            "input_sources": [
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.plan.input_sources
            ],
            "panel": self.plan.panel,
            "parent_counts": dict(self.plan.parent_counts),
            "sampling_contract": _jsonable(self.plan.sampling_contract),
            "schema_version": "dive-paper-campaign-plan-v1",
            "splice_report_sha256": self.plan.splice_report_sha256,
            "view_partition_hash": self.plan.view_partition_hash,
            "view_semantic_hash": self.plan.view_semantic_hash,
        }
        return _write_create_new(
            Path(self.run.evidence_dir) / "plan.json", canonical_json_bytes(payload)
        )

    def _write_stage_terminal(
        self,
        stage: CampaignStage,
        *,
        status: str,
        identities: Mapping[str, ArtifactIdentity],
        cell_ids: Sequence[str],
        reason: str | None = None,
    ) -> StageRecord:
        directory = self._stage_dir(stage)
        directory.mkdir(parents=True, exist_ok=True)
        serialized = {
            cell_id: {
                "path": identity.path,
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            }
            for cell_id, identity in identities.items()
        }
        payload = {
            "schema_version": "dive-paper-stage-terminal-v1",
            "cell_ids": list(cell_ids),
            "identities": serialized,
            "reason": reason,
            "run_id": self.run.run_id,
            "stage": stage.value,
            "status": status,
        }
        _write_create_new(directory / "terminal.json", canonical_json_bytes(payload))
        return StageRecord(
            stage=stage,
            status=status,
            cell_ids=tuple(cell_ids),
            identities=dict(identities),
            reason=reason,
        )

    def _require_complete(self, stage: CampaignStage) -> dict[str, Any]:
        path = self._stage_dir(stage) / "terminal.json"
        if not path.is_file():
            raise PaperCampaignError(
                f"later stage refuses until predecessor {stage.value} "
                "authenticates every requested identity"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise PaperCampaignError(f"predecessor {stage.value} is not COMPLETE")
        observed = set(payload.get("cell_ids") or ())
        planned = {cell.cell_id for cell in self.plan.cells}
        if observed != planned:
            raise PaperCampaignError(
                f"predecessor {stage.value} did not authenticate every requested identity"
            )
        return payload

    def _refuse_failed(self) -> None:
        if self.stage == CampaignStage.FAILED:
            raise PaperCampaignError("FAILED is terminal")

    def _stage_dir(self, stage: CampaignStage) -> Path:
        return Path(self.run.evidence_dir) / "stages" / stage.value

    def _shard_for(
        self,
        stage: CampaignStage,
        shard_id: str,
        cell_ids: tuple[str, ...],
        gpu: int,
    ) -> CampaignShard:
        directory = self._stage_dir(stage) / "shards" / shard_id
        return CampaignShard(
            shard_id=shard_id,
            stage=stage,
            cell_ids=cell_ids,
            gpu=gpu,
            directory=directory,
            request_path=directory / "request.json",
            progress_path=directory / "progress.jsonl",
            terminal_path=directory / "terminal.json",
            manifest_path=directory / "manifest.json",
            log_path=directory / "worker.log",
        )

    def _list_shards(self, stage: CampaignStage) -> tuple[CampaignShard, ...]:
        root = self._stage_dir(stage) / "shards"
        if not root.is_dir():
            return ()
        shards: list[CampaignShard] = []
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            assignment = json.loads(
                (directory / "assignment.json").read_text(encoding="utf-8")
            )
            shards.append(
                self._shard_for(
                    stage,
                    str(assignment["shard_id"]),
                    tuple(assignment["cell_ids"]),
                    int(assignment["gpu"]),
                )
            )
        return tuple(shards)

    def _resolved_occupied(self, occupied: frozenset[int] | None) -> frozenset[int]:
        if occupied is not None:
            return frozenset(occupied)
        if self._occupied_gpus is not None:
            return frozenset(self._occupied_gpus)
        return _live_occupied_gpus()

    def _reconcile_sharded(self, stage: CampaignStage) -> StageRecord:
        in_progress = _RECONCILE_IN_PROGRESS[stage]
        if self.stage == CampaignStage.FAILED:
            raise PaperCampaignError("FAILED is terminal")
        if self.stage != in_progress:
            raise PaperCampaignError(
                f"predecessor {in_progress.value} is not in progress from "
                f"{self.stage.value}"
            )
        shards = self._list_shards(in_progress)
        identities: dict[str, ArtifactIdentity] = {}
        for shard in shards:
            progress = _read_progress(shard.progress_path)
            interrupt = _progress_failure(progress, pid_alive=self._pid_alive)
            if interrupt == "signal_or_exit":
                return self.fail_stage("nonzero exit or SIGTERM")
            if interrupt == "stale":
                raise PaperCampaignError("stale PID")
            if not shard.terminal_path.is_file():
                raise PaperCampaignError("missing terminal")
            if not shard.request_path.is_file() or not shard.manifest_path.is_file():
                raise PaperCampaignError("missing shard request or hash manifest")
            request = json.loads(shard.request_path.read_text(encoding="utf-8"))
            request_cells = tuple(request.get("cell_ids") or ())
            if request_cells != shard.cell_ids:
                raise PaperCampaignError("identity mismatch")
            live_request = hashlib.sha256(shard.request_path.read_bytes()).hexdigest()
            expected = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
            if live_request != expected:
                raise PaperCampaignError("hash drift")
            manifest = json.loads(shard.manifest_path.read_text(encoding="utf-8"))
            manifest_cells = manifest.get("cells")
            if not isinstance(manifest_cells, Mapping):
                raise PaperCampaignError("hash manifest is invalid")
            complete_rows = [row for row in progress if row.get("status") == "complete"]
            seen_progress: set[str] = set()
            for row in complete_rows:
                cell_id = str(row.get("cell_id") or "")
                if cell_id in seen_progress:
                    raise PaperCampaignError("duplicate cells")
                seen_progress.add(cell_id)
                if cell_id not in request_cells:
                    raise PaperCampaignError("identity mismatch")
                sha = row.get("output_sha256")
                if type(sha) is not str or len(sha) != 64 or set(sha) - _SHA256:
                    raise PaperCampaignError("invalid hash")
                path = Path(str(row["output_path"]))
                live = file_identity(path)
                if live.sha256 != sha or live.size_bytes != int(
                    row["output_size_bytes"]
                ):
                    raise PaperCampaignError("hash drift")
                recorded = manifest_cells.get(cell_id)
                if not isinstance(recorded, Mapping) or recorded.get("sha256") != sha:
                    raise PaperCampaignError("hash drift")
                if cell_id in identities:
                    raise PaperCampaignError("duplicate cells")
                identities[cell_id] = ArtifactIdentity(
                    live.path, live.sha256, live.size_bytes
                )
        planned = [cell.cell_id for cell in self.plan.cells]
        if set(identities) != set(planned):
            missing = sorted(set(planned) - set(identities))
            extra = sorted(set(identities) - set(planned))
            if extra:
                raise PaperCampaignError(f"unplanned cells {extra}")
            raise PaperCampaignError(f"missing cells {missing}; count gap")
        if len(identities) != len(planned):
            raise PaperCampaignError("count gap")
        return self._write_stage_terminal(
            stage,
            status="COMPLETE",
            identities=identities,
            cell_ids=tuple(planned),
        )

    def _reconcile_aggregated(self) -> StageRecord:
        payload = self._require_complete(CampaignStage.EVALUATED)
        identities = _identities_from_payload(payload)
        return self._write_stage_terminal(
            CampaignStage.AGGREGATED,
            status="COMPLETE",
            identities=identities,
            cell_ids=tuple(payload["cell_ids"]),
        )

    def _reconcile_complete(self) -> StageRecord:
        payload = self._require_complete(CampaignStage.AGGREGATED)
        identities = _identities_from_payload(payload)
        completion = complete_paper_baseline_run(
            self.run,
            {
                "stage": CampaignStage.COMPLETE.value,
                "cell_ids": list(payload["cell_ids"]),
                "evaluator_registry_sha256": self.plan.evaluator_registry_sha256,
            },
        )
        record = self._write_stage_terminal(
            CampaignStage.COMPLETE,
            status="COMPLETE",
            identities=identities,
            cell_ids=tuple(payload["cell_ids"]),
        )
        del completion
        return record

def reconcile_stage(
    controller: PaperCampaignController, stage: CampaignStage
) -> StageRecord:
    controller._refuse_failed()
    if stage in _RECONCILE_IN_PROGRESS:
        return controller._reconcile_sharded(stage)
    if stage is CampaignStage.AGGREGATED:
        return controller._reconcile_aggregated()
    if stage is CampaignStage.COMPLETE:
        return controller._reconcile_complete()
    raise PaperCampaignError(f"cannot reconcile stage {stage}")

def render_detached_commands(
    controller: PaperCampaignController,
    shards: Sequence[CampaignShard],
    *,
    gpus: Sequence[int] | None = None,
    occupied_gpus: frozenset[int] | None = None,
) -> tuple[str, ...]:
    requested = list(gpus or ())
    requested.extend(shard.gpu for shard in shards)
    _require_free_gpus(requested, controller._resolved_occupied(occupied_gpus))
    rendered: list[str] = []
    for shard in shards:
        action = {
            CampaignStage.GENERATING: "generate",
            CampaignStage.EXPORTING: "export",
            CampaignStage.EVALUATING: "evaluate",
        }[shard.stage]
        command = (
            "setsid nohup env "
            f"CUDA_VISIBLE_DEVICES={shard.gpu} "
            "DIVE_PAPER_REQUIRE_MODEL_EVAL=1 "
            "PYTHONPATH=src "
            f"{RUNTIME_PYTHON} {_RUN_SCRIPT} {action} "
            f"--run-id {controller.run.run_id} --shard {shard.shard_id} "
            f"> {shard.log_path} 2>&1 < /dev/null &"
        )
        rendered.append(command)
    return tuple(rendered)

def assert_gpu_ledger_safe(
    planned_gpu_seconds: int,
    observed_gpu_seconds: int = OBSERVED_PROJECT_GPU_SECONDS,
) -> GpuLedgerRecord:
    if observed_gpu_seconds != OBSERVED_PROJECT_GPU_SECONDS:
        raise PaperCampaignError("observed GPU-seconds ledger must remain 220279")
    if type(planned_gpu_seconds) is not int or planned_gpu_seconds < 0:
        raise PaperCampaignError("planned_gpu_seconds must be a nonnegative int")
    project = observed_gpu_seconds + planned_gpu_seconds
    reserve = PROJECT_TOTAL - project
    if project > PROJECT_PLANNED_CAP or reserve < MIN_RESERVE:
        raise PaperCampaignError(
            "planned GPU-days would exceed the 84 GPU-day cap or violate the "
            "36 GPU-day reserve"
        )
    return GpuLedgerRecord(
        observed_gpu_seconds=observed_gpu_seconds,
        planned_gpu_seconds=planned_gpu_seconds,
        project_gpu_seconds=project,
        reserve_gpu_seconds=reserve,
        planned_cap_gpu_seconds=PROJECT_PLANNED_CAP,
        min_reserve_gpu_seconds=MIN_RESERVE,
        sampled=False,
        tuned=False,
    )

def load_planned_batches(cells: Sequence[PaperCell]) -> dict[str, PaperBatch]:
    wanted = {cell.example_id for cell in cells}
    if not wanted:
        raise PaperCampaignError("generation shard has no cells")
    view_run = EMERGENT_EVIDENCE_ROOT / "benchmark_ready" / "benchmark-views-20260828a"
    input_run = (
        EMERGENT_EVIDENCE_ROOT / "benchmark_ready" / "benchmark-inputs-20260826c"
    )
    bundle = load_paper_inputs(view_run=view_run, input_run=input_run)
    by_id = {row.example_id: row for row in bundle.targets if row.example_id in wanted}
    missing = wanted - set(by_id)
    if missing:
        raise PaperCampaignError(f"missing batch for example {sorted(missing)[0]!r}")
    return {
        example_id: load_paper_batch(target) for example_id, target in by_id.items()
    }

def _shard_cells(
    controller: PaperCampaignController, shard: CampaignShard
) -> tuple[PaperCell, ...]:
    wanted = set(shard.cell_ids)
    return tuple(cell for cell in controller.plan.cells if cell.cell_id in wanted)

def _cli_generate(controller: PaperCampaignController, shard: CampaignShard) -> None:
    cells = _shard_cells(controller, shard)
    batches = load_planned_batches(cells)
    records = controller.execute_generation_shard(shard, batches)
    controller.record_shard_exit(shard, exit_code=0)
    controller.publish_shard_terminal(
        shard,
        {
            "cells": [record.cell_id for record in records],
            "exit_code": 0,
        },
    )

CANARY_SUBSET_KIND = "technical_canary_subset"

def canary_reuse_provenance(
    *,
    generation_run_id: str,
    generation_commit: str,
    evaluation_commit: str,
    selection: Sequence[str],
    generated_identities: Mapping[str, object],
    registry_sha256: str,
) -> dict[str, object]:

    artifact_sha256 = {
        cell: str(entry["sha256"])
        for cell, entry in generated_identities.items()
        if isinstance(entry, Mapping) and "sha256" in entry
    }
    return {
        "kind": CANARY_SUBSET_KIND,
        "generation": {
            "run_id": generation_run_id,
            "commit": generation_commit,
            "selection": list(selection),
            "n_cells": len(tuple(selection)),
            "artifact_sha256": artifact_sha256,
        },
        "evaluation": {
            "commit": evaluation_commit,
            "registry_sha256": registry_sha256,
        },
        "commits_differ": generation_commit != evaluation_commit,
        "campaign_complete": False,
        "eligible_for_paper_aggregation": False,
    }

def assert_canary_selection_in_plan(
    planned_cell_ids: Sequence[str], selection: Sequence[str]
) -> tuple[str, ...]:

    chosen = tuple(selection)
    if len(chosen) != len(set(chosen)):
        raise PaperCampaignError("canary selection has duplicate cells")
    planned = set(planned_cell_ids)
    outside = sorted(cell for cell in chosen if cell not in planned)
    if outside:
        raise PaperCampaignError(f"canary selection has cells outside the plan {outside}")
    return chosen

def assert_canary_subset_manifest(
    manifest_cells: Mapping[str, object], selection: Sequence[str]
) -> Mapping[str, object]:

    chosen = tuple(selection)
    if len(chosen) != len(set(chosen)):
        raise PaperCampaignError("canary selection has duplicate cells")
    have = set(manifest_cells)
    want = set(chosen)
    missing = sorted(want - have)
    if missing:
        raise PaperCampaignError(f"canary subset is missing cells {missing}")
    outside = sorted(have - want)
    if outside:
        raise PaperCampaignError(
            f"canary subset carries artifacts outside the selection {outside}"
        )
    return {cell: manifest_cells[cell] for cell in chosen}

def canary_terminal_payload(cell_ids: Sequence[str], *, stage: str) -> dict[str, object]:

    return {
        "kind": CANARY_SUBSET_KIND,
        "stage": stage,
        "cells": list(cell_ids),
        "n_cells": len(tuple(cell_ids)),
        "exit_code": 0,
        "campaign_complete": False,
        "eligible_for_paper_aggregation": False,
        "note": (
            "Technical canary on a subset of the frozen plan. Not a campaign "
            "completion and not admissible to any full-plan aggregate."
        ),
    }

def _cli_export(
    controller: PaperCampaignController,
    shard: CampaignShard,
    *,
    identities: Mapping[str, object] | None = None,
) -> None:
    if identities is None:
        predecessor = controller._require_complete(CampaignStage.GENERATED)
        identities = predecessor.get("identities")
    if not isinstance(identities, Mapping):
        raise PaperCampaignError(
            "export worker requires authenticated generated samples"
        )
    cells = _shard_cells(controller, shard)
    batches = load_planned_batches(cells)
    import torch

    for cell in cells:
        recorded = identities.get(cell.cell_id)
        if not isinstance(recorded, Mapping):
            raise PaperCampaignError(
                "export worker requires authenticated generated samples"
            )
        path = Path(str(recorded["path"]))
        live = file_identity(path)
        if live.sha256 != recorded["sha256"] or live.size_bytes != int(
            recorded["size_bytes"]
        ):
            raise PaperCampaignError("export identity drifted")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "sample" not in payload:
            raise PaperCampaignError(
                "export worker requires authenticated generated samples"
            )
        try:
            batch = batches[cell.example_id]
        except KeyError as error:
            raise PaperCampaignError(
                f"missing batch for example {cell.example_id!r}"
            ) from error
        output_dir = (
            Path(controller.run.bulk_dir) / "export" / shard.shard_id / cell.cell_id
        )
        result = controller.execute_export_cell(
            sample=payload["sample"], batch=batch, output_dir=output_dir
        )
        envelope = _write_export_envelope(output_dir, result)
        controller.append_shard_progress(
            shard,
            {
                "cell_id": cell.cell_id,
                "status": "complete",
                "output_path": envelope.path,
                "output_sha256": envelope.sha256,
                "output_size_bytes": envelope.size_bytes,
            },
        )
    controller.record_shard_exit(shard, exit_code=0)
    controller.publish_shard_terminal(
        shard, {"cells": [cell.cell_id for cell in cells], "exit_code": 0}
    )

def evaluate_work_dir(
    controller: PaperCampaignController, shard_id: str, cell_id: str
) -> Path:

    return Path(controller.run.evidence_dir) / "evaluate" / shard_id / cell_id

def _cli_evaluate(
    controller: PaperCampaignController,
    shard: CampaignShard,
    *,
    identities: Mapping[str, object] | None = None,
    generated_identities: Mapping[str, object] | None = None,
) -> None:
    if identities is None:
        predecessor = controller._require_complete(CampaignStage.EXPORTED)
        identities = predecessor.get("identities")
    if not isinstance(identities, Mapping):
        raise PaperCampaignError("evaluate worker requires authenticated exports")
    if generated_identities is None:
        generated_predecessor = controller._require_complete(CampaignStage.GENERATED)
        generated_identities = generated_predecessor.get("identities")
    if not isinstance(generated_identities, Mapping):
        raise PaperCampaignError(
            "evaluate worker requires authenticated generated samples"
        )
    cells = _shard_cells(controller, shard)
    batches = load_planned_batches(cells)
    for cell in cells:
        recorded = identities.get(cell.cell_id)
        if not isinstance(recorded, Mapping):
            raise PaperCampaignError("evaluate worker requires authenticated exports")
        export = _load_export_envelope(recorded)
        work_dir = evaluate_work_dir(controller, shard.shard_id, cell.cell_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            expected_batch = batches[cell.example_id]
        except KeyError as error:
            raise PaperCampaignError(
                f"missing batch for example {cell.example_id!r}"
            ) from error
        generated_sample = _load_stage_artifact_identity(
            generated_identities.get(cell.cell_id),
            label="generated sample",
        )
        if cell.family == "antibody":
            export = _restore_antibody_export_for_evaluation(
                export,
                cell=cell,
                expected_batch=expected_batch,
                generated_sample=generated_sample,
            )
        result = controller.evaluate_cell(
            cell,
            export,
            work_dir,
            expected_batch=expected_batch,
            generated_sample=generated_sample,
        )
        output = work_dir / "result.json"
        identity = _write_create_new(
            output, canonical_json_bytes(_jsonable(result.to_mapping()))
        )
        controller.append_shard_progress(
            shard,
            {
                "cell_id": cell.cell_id,
                "status": "complete",
                "output_path": identity.path,
                "output_sha256": identity.sha256,
                "output_size_bytes": identity.size_bytes,
            },
        )
    controller.record_shard_exit(shard, exit_code=0)
    controller.publish_shard_terminal(
        shard, {"cells": [cell.cell_id for cell in cells], "exit_code": 0}
    )

def _load_stage_artifact_identity(value: object, *, label: str) -> ArtifactIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise PaperCampaignError(f"evaluate worker requires authenticated {label}")
    identity = ArtifactIdentity(
        str(value["path"]), str(value["sha256"]), int(value["size_bytes"])
    )
    live = file_identity(Path(identity.path))
    if live != identity:
        raise PaperCampaignError(f"{label} identity drifted")
    return identity

def _write_export_envelope(
    output_dir: Path, result: PaperExportResult
) -> ArtifactIdentity:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def dump(identity: ArtifactIdentity | None) -> dict[str, object] | None:
        if identity is None:
            return None
        return {
            "path": identity.path,
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }

    payload = {
        "family": result.family,
        "complex_pdb": dump(result.complex_pdb),
        "manifest": dump(result.manifest),
        "native_target": dump(result.native_target),
        "motif_positions": dump(result.motif_positions),
        "native_motif": dump(result.native_motif),
        "binder_backbone_normalization": (
            None
            if result.binder_backbone_normalization is None
            else result.binder_backbone_normalization.to_mapping()
        ),
        "ame_ligand_transport": (
            None
            if result.ame_ligand_transport is None
            else result.ame_ligand_transport.to_mapping()
        ),
        "schema": "dive.paper.export_envelope.v1",
    }
    return _write_create_new(
        Path(output_dir) / "envelope.json", canonical_json_bytes(payload)
    )

def _load_export_envelope(recorded: Mapping[str, object]) -> PaperExportResult:
    path = Path(str(recorded["path"]))
    live = file_identity(path)
    if live.sha256 != recorded["sha256"] or live.size_bytes != int(
        recorded["size_bytes"]
    ):
        raise PaperCampaignError("export identity drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PaperCampaignError("evaluate worker requires authenticated exports")

    def load_identity(key: str) -> ArtifactIdentity | None:
        item = payload.get(key)
        if item is None:
            return None
        if not isinstance(item, Mapping):
            raise PaperCampaignError("evaluate worker requires authenticated exports")
        identity = ArtifactIdentity(
            str(item["path"]), str(item["sha256"]), int(item["size_bytes"])
        )
        observed = file_identity(Path(identity.path))
        if (
            observed.sha256 != identity.sha256
            or observed.size_bytes != identity.size_bytes
        ):
            raise PaperCampaignError("export identity drifted")
        return identity

    family = str(payload.get("family") or "")
    complex_pdb = load_identity("complex_pdb")
    manifest = load_identity("manifest")
    if complex_pdb is None or manifest is None:
        raise PaperCampaignError("evaluate worker requires authenticated exports")
    binder_normalization = None
    if family == "binder":
        try:
            binder_normalization = binder_backbone_normalization_from_mapping(
                payload.get("binder_backbone_normalization")
            )
        except PaperBaselineError as error:
            raise PaperCampaignError(
                "evaluate worker requires authenticated binder normalization"
            ) from error
        manifest_payload = json.loads(Path(manifest.path).read_text(encoding="utf-8"))
        if (
            not isinstance(manifest_payload, Mapping)
            or manifest_payload.get("binder_backbone_normalization")
            != binder_normalization.to_mapping()
        ):
            raise PaperCampaignError("binder normalization manifest drifted")
    ame_transport = None
    if family == "ame":
        try:
            ame_transport = ame_ligand_transport_from_mapping(
                payload.get("ame_ligand_transport")
            )
        except PaperBaselineError as error:
            raise PaperCampaignError(
                "evaluate worker requires authenticated AME ligand transport"
            ) from error
        manifest_payload = json.loads(Path(manifest.path).read_text(encoding="utf-8"))
        if (
            not isinstance(manifest_payload, Mapping)
            or manifest_payload.get("ame_ligand_transport")
            != ame_transport.to_mapping()
        ):
            raise PaperCampaignError("AME ligand transport manifest drifted")
    return PaperExportResult(
        family=family,
        complex_pdb=complex_pdb,
        manifest=manifest,
        native_target=load_identity("native_target"),
        motif_positions=load_identity("motif_positions"),
        native_motif=load_identity("native_motif"),
        binder_backbone_normalization=binder_normalization,
        ame_ligand_transport=ame_transport,
    )

def _restore_antibody_export_for_evaluation(
    export: PaperExportResult,
    *,
    cell: PaperCell,
    expected_batch: PaperBatch,
    generated_sample: ArtifactIdentity,
) -> PaperExportResult:

    if type(export) is not PaperExportResult or export.family != "antibody":
        raise PaperCampaignError("evaluate worker requires an antibody export")
    if type(cell) is not PaperCell or cell.family != "antibody":
        raise PaperCampaignError("evaluate worker requires an antibody cell")
    if type(expected_batch) is not PaperBatch:
        raise PaperCampaignError("antibody evaluate requires an authenticated batch")
    target = expected_batch.target
    if (
        target.family != "antibody"
        or target.example_id != cell.example_id
        or target.parent_id != cell.parent_id
    ):
        raise PaperCampaignError("antibody expected batch identity drifted")
    try:
        assert_batch_matches_target(expected_batch, target)
        raw = _read_authenticated_generated_sample(generated_sample)
        import torch

        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except (OSError, PaperBaselineError, PaperBatchError) as error:
        raise PaperCampaignError(
            "antibody evaluate requires authenticated generation and batch inputs"
        ) from error
    required = {
        "arm",
        "cell_id",
        "chain_names",
        "family",
        "panel",
        "residue_pdb_idx",
        "sample",
        "schema",
        "seed",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PaperCampaignError("antibody generated sample schema drifted")
    if (
        payload.get("schema") != "dive.paper.generated_sample.v1"
        or payload.get("cell_id") != cell.cell_id
        or payload.get("family") != "antibody"
        or payload.get("seed") != cell.seed
        or payload.get("arm") != cell.arm
        or payload.get("panel") != cell.panel
    ):
        raise PaperCampaignError("antibody generated sample identity drifted")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping):
        raise PaperCampaignError("antibody generated sample schema drifted")
    try:
        authority_rows = _generated_authority_rows(
            sample,
            chain_names=payload.get("chain_names"),
            residue_pdb_idx=payload.get("residue_pdb_idx"),
        )
        _authenticate_generated_roles_against_batch(
            authority_rows,
            sample=sample,
            chain_names=payload.get("chain_names"),
            residue_pdb_idx=payload.get("residue_pdb_idx"),
            expected_batch=expected_batch,
        )
        complex_pdb = read_authenticated_pdb(
            export.complex_pdb, label="antibody complex PDB"
        )
        _authenticate_generated_rows_against_pdb(authority_rows, complex_pdb)
        rows = _residue_rows(sample, expected_batch)
        with _retained_authenticated_antibody_source(expected_batch) as source:
            mapping = _label_to_auth_mapping(source)
            roles = _frozen_antibody_roles(target.role_payload, rows, mapping)
            native_atoms = _native_atom_records(source, mapping)
        predicted_atoms = _predicted_atom_records(rows, mapping)
    except PaperCampaignError:
        raise
    except (KeyError, PdbAuthenticationError, PaperBaselineError, ValueError) as error:
        raise PaperCampaignError(
            "antibody evaluation inputs cannot be reconstructed"
        ) from error
    try:
        native_h3_ca = {
            atom.identity
            for atom in native_atoms
            if atom.atom == "CA" and atom.residue_key in roles.cdr_h3
        }
        predicted_h3_ca = {
            atom.identity
            for atom in predicted_atoms
            if atom.atom == "CA" and atom.residue_key in roles.cdr_h3
        }
    except (TypeError, ValueError) as error:
        raise PaperCampaignError("antibody atom identity is malformed") from error
    if len(native_h3_ca & predicted_h3_ca) < 3:
        raise PaperCampaignError(
            "antibody H3 has fewer than three common C-alpha atoms"
        )
    return PaperExportResult(
        family=export.family,
        complex_pdb=export.complex_pdb,
        manifest=export.manifest,
        native_target=export.native_target,
        motif_positions=export.motif_positions,
        native_motif=export.native_motif,
        predicted_atoms=predicted_atoms,
        native_atoms=native_atoms,
        antibody_roles=roles,
        binder_backbone_normalization=export.binder_backbone_normalization,
        ame_ligand_transport=export.ame_ligand_transport,
    )

@contextmanager
def _retained_authenticated_antibody_source(expected_batch: PaperBatch):

    source = Path(expected_batch.target.source_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, flags)
    except OSError as error:
        raise PaperCampaignError("antibody native source cannot be opened") from error
    try:
        try:
            metadata = os.fstat(source_descriptor)
        except OSError as error:
            raise PaperCampaignError(
                "antibody native source cannot be inspected"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise PaperCampaignError("antibody native source is not regular")
        try:
            snapshot_descriptor = _sealed_memfd("dive-antibody-source")
        except OSError as error:
            raise PaperCampaignError(
                "antibody native source snapshot cannot be created"
            ) from error
        digest = hashlib.sha256()
        size = 0
        try:
            try:
                while True:
                    chunk = os.read(source_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    _write_all(snapshot_descriptor, chunk)
            except OSError as error:
                raise PaperCampaignError(
                    "antibody native source snapshot cannot be copied"
                ) from error
            if (
                digest.hexdigest() != expected_batch.structure.sha256
                or size != expected_batch.structure.size_bytes
            ):
                raise PaperCampaignError("antibody native source identity drifted")
            try:
                os.lseek(snapshot_descriptor, 0, os.SEEK_SET)
            except OSError as error:
                raise PaperCampaignError(
                    "antibody native source snapshot cannot be rewound"
                ) from error
            try:
                _seal_memfd(snapshot_descriptor)
            except OSError as error:
                raise PaperCampaignError(
                    "antibody native source snapshot cannot be sealed"
                ) from error
            yield _RetainedSourcePath(snapshot_descriptor, source.name)
        finally:
            os.close(snapshot_descriptor)
    finally:
        os.close(source_descriptor)

def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PaperCampaignError("antibody native source snapshot write failed")
        view = view[written:]

def _sealed_memfd(name: str) -> int:

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        create = libc.memfd_create
    except AttributeError as error:
        raise OSError("memfd_create is unavailable") from error
    create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    create.restype = ctypes.c_int
    descriptor = create(name.encode("utf-8"), 0x0001 | 0x0002)
    if descriptor < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return descriptor

def _seal_memfd(descriptor: int) -> None:

    libc = ctypes.CDLL(None, use_errno=True)
    fcntl_call = libc.fcntl
    fcntl_call.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
    fcntl_call.restype = ctypes.c_int

    if fcntl_call(descriptor, 1033, 0x0001 | 0x0002 | 0x0004 | 0x0008) < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))

def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_paper_baseline")
    parser.add_argument("action", nargs="?", default=None)
    parser.add_argument("--foldseek", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--panel", default="A")
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--planned-gpu-seconds", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--shard")
    parser.add_argument("--gpus")
    parser.add_argument("--stage")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.foldseek:
            raise PaperCampaignError("Foldseek is forbidden")
        if args.partition != "validation" or "blind" in str(args.partition):
            raise PaperCampaignError("blind partitions are forbidden")
        if args.arm in DIVE_ARM_NAMES or args.arm == "base":
            raise PaperCampaignError(f"arm {args.arm} is excluded")
        if args.action == "ledger":
            if args.planned_gpu_seconds is None:
                raise PaperCampaignError("ledger requires --planned-gpu-seconds")
            record = assert_gpu_ledger_safe(args.planned_gpu_seconds)
            print(json.dumps(asdict(record), sort_keys=True))
            return 0
        if args.action in CAMPAIGN_CLI_ACTIONS:
            if not args.run_id:
                raise PaperCampaignError("campaign action requires --run-id")
            controller = PaperCampaignController.open(
                run_id=args.run_id,
                evidence_root=EMERGENT_EVIDENCE_ROOT,
                bulk_root=EMERGENT_BULK_ROOT,
            )
            if args.action == "report":
                print(controller.stage.value)
                return 0
            if args.action == "render":
                shards: list[CampaignShard] = []
                for stage in _IN_PROGRESS:
                    if controller.stage is stage:
                        shards.extend(controller._list_shards(stage))
                for command in render_detached_commands(controller, shards):
                    print(command)
                return 0
            if args.action == "reconcile":
                if not args.stage:
                    raise PaperCampaignError("reconcile requires --stage")
                record = reconcile_stage(controller, CampaignStage(args.stage))
                print(record.status)
                return 0
            if args.action in {"generate", "export", "evaluate"}:
                if not args.shard:
                    raise PaperCampaignError(f"{args.action} requires --shard")
                shard = controller.shard_by_id(args.shard)
                if not shard.request_path.is_file():
                    raise PaperCampaignError("missing shard request")
                try:
                    if args.action == "generate":
                        _cli_generate(controller, shard)
                    elif args.action == "export":
                        _cli_export(controller, shard)
                    else:
                        _cli_evaluate(controller, shard)
                except (PaperCampaignError, PaperBaselineError) as error:
                    controller.fail_stage(str(error))
                    raise
                return 0
            return 0
        if args.action not in {None, *CAMPAIGN_CLI_ACTIONS}:
            raise PaperCampaignError(f"unknown action {args.action}")
    except (PaperCampaignError, PaperBaselineError, ValueError):
        return 2
    return 0

def _load_evaluate_family():
    spec = importlib.util.spec_from_file_location(
        "dive_paper_baseline_production_worker", _WORKER
    )
    if spec is None or spec.loader is None:
        raise PaperCampaignError("cannot import evaluate_family")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate_family = getattr(module, "evaluate_family", None)
    if evaluate_family is None:
        raise PaperCampaignError("evaluate_family is missing")
    return evaluate_family

def _validate_run_id(run_id: str) -> None:
    if type(run_id) is not str or not (
        run_id.startswith("paper-smoke-") or run_id.startswith("paper-baseline-")
    ):
        raise PaperCampaignError(
            "run id must start with paper-smoke- or paper-baseline-"
        )

def _validate_plan(plan: PaperPlanManifest, commit: str) -> None:
    if not isinstance(plan, PaperPlanManifest):
        raise PaperCampaignError("campaign requires a PaperPlanManifest")
    if plan.foldseek_executed:
        raise PaperCampaignError("Foldseek is forbidden")
    if plan.blind_opened:
        raise PaperCampaignError("blind partitions are forbidden")
    if plan.executed_dive_arms:
        raise PaperCampaignError(f"excluded arms {tuple(plan.executed_dive_arms)}")
    if plan.arm in DIVE_ARM_NAMES or plan.arm == "base":
        raise PaperCampaignError(f"arm {plan.arm} is excluded")
    if plan.arm != PaperArm.BASE_COMMON_EXACT_V1.value:
        raise PaperCampaignError(f"arm {plan.arm} is excluded")
    if plan.evaluator_registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256:
        raise PaperCampaignError("fixture hash cannot authenticate as production")
    if plan.evaluator_registry_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256:
        raise PaperCampaignError("production evaluator registry hash drifted")
    if plan.code_commit != commit:
        raise PaperCampaignError("bad code commit")
    seen: set[str] = set()
    for cell in plan.cells:
        if cell.cell_id in seen:
            raise PaperCampaignError("duplicate cells")
        seen.add(cell.cell_id)
        if cell.evaluator_registry_sha256 != plan.evaluator_registry_sha256:
            raise PaperCampaignError("cell evaluator registry hash drifted")
        if cell.arm in DIVE_ARM_NAMES or cell.arm == "base" or cell.arm != plan.arm:
            raise PaperCampaignError(f"arm {cell.arm} is excluded")
        if cell.partition != "validation" or "blind" in cell.partition:
            raise PaperCampaignError("blind partitions are forbidden")
        if cell.evaluator_registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256:
            raise PaperCampaignError("fixture hash cannot authenticate as production")
        if cell.family == "binder" and cell.steps != 400:
            raise PaperCampaignError("classified_protocol_mismatch")

def _require_emergent_upstream() -> str:
    completed = subprocess.run(
        ("git", "-C", str(EMERGENT_UPSTREAM_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != EMERGENT_UPSTREAM_COMMIT:
        raise PaperCampaignError(f"wrong upstream {commit}")
    return commit

def _refuse_denied(path: Path) -> Path:
    resolved = Path(path).resolve(strict=False)
    for denied in DENIED_PREFIXES:
        prefix = Path(denied).resolve(strict=False)
        if resolved == prefix or prefix in resolved.parents:
            raise PaperCampaignError(f"root is denied: {resolved}")
    return resolved

def _require_free_gpus(requested: Sequence[int], occupied: frozenset[int]) -> None:
    overlap = sorted(set(requested) & set(occupied))
    if overlap:
        raise PaperCampaignError(f"occupied devices {overlap}")

def _live_occupied_gpus() -> frozenset[int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise PaperCampaignError("GPU inventory is unavailable")
    mapping = _gpu_uuid_to_index()
    occupied: set[int] = set()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        gpu_uuid = line.split(",", 1)[0].strip()
        if gpu_uuid in mapping:
            occupied.add(mapping[gpu_uuid])
    return frozenset(occupied)

def _gpu_uuid_to_index() -> dict[str, int]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise PaperCampaignError("GPU inventory is unavailable")
    mapping: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        index_text, uuid = [part.strip() for part in line.split(",", 1)]
        mapping[uuid] = int(index_text)
    return mapping

def _pid_alive(pid: int) -> bool:
    return Path(f"/proc/{int(pid)}").exists()

def _write_create_new(path: Path, content: bytes) -> ArtifactIdentity:
    destination = _refuse_denied(Path(path))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o664)
    except FileExistsError as error:
        raise PaperCampaignError(
            f"create-new artifact already exists: {destination}"
        ) from error
    except OSError as error:
        raise PaperCampaignError(
            f"cannot write create-new artifact: {error}"
        ) from error
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PaperCampaignError("create-new write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(content).hexdigest()
    return ArtifactIdentity(str(destination), digest, len(content))

def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    destination = _refuse_denied(Path(path))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o664)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _read_progress(path: Path) -> list[dict[str, Any]]:
    destination = Path(path)
    if not destination.exists():
        return []
    text = destination.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise PaperCampaignError("truncated progress")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise PaperCampaignError("truncated progress") from error
        if not isinstance(payload, dict):
            raise PaperCampaignError("truncated progress")
        rows.append(payload)
    return rows

def _progress_failure(
    progress: Sequence[Mapping[str, Any]], *, pid_alive: Callable[[int], bool]
) -> str | None:
    pids: list[int] = []
    exited = False
    for row in progress:
        if row.get("event") == "start" and "pid" in row:
            pids.append(int(row["pid"]))
        if row.get("signal") is not None:
            return "signal_or_exit"
        if row.get("event") == "exit" and row.get("exit_code") not in (None, 0):
            return "signal_or_exit"
        if row.get("event") == "exit":
            exited = True
    if pids and not exited:
        if not any(pid_alive(pid) for pid in pids):
            return "stale"
    return None

def _jsonable(value: object) -> object:
    if isinstance(value, MetricValue):
        return {
            "direction": value.direction,
            "evaluator_manifest_hash": value.evaluator_manifest_hash,
            "name": value.name,
            "value": value.value,
        }
    if isinstance(value, MetricUnavailable):
        return {
            "detail": value.detail,
            "evaluator_manifest_hash": value.evaluator_manifest_hash,
            "name": value.name,
            "reason_code": value.reason_code,
            "status": value.status.value,
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value

def _load_plan(path: Path) -> PaperPlanManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PaperCampaignError("campaign plan is invalid")
    cells = []
    for row in payload.get("cells") or ():
        if not isinstance(row, Mapping):
            raise PaperCampaignError("campaign plan is invalid")
        cells.append(
            PaperCell(
                cell_id=str(row["cell_id"]),
                panel=str(row["panel"]),
                arm=str(row["arm"]),
                family=str(row["family"]),
                partition=str(row["partition"]),
                parent_id=str(row["parent_id"]),
                example_id=str(row["example_id"]),
                seed=int(row["seed"]),
                sample_index=int(row["sample_index"]),
                generation_replicas=int(row["generation_replicas"]),
                steps=int(row["steps"]),
                condition_count=int(row["condition_count"]),
                denoiser_equivalent_passes=int(row["denoiser_equivalent_passes"]),
                checkpoint=str(row["checkpoint"]),
                splice_report_sha256=str(row["splice_report_sha256"]),
                evaluator_registry_sha256=str(row["evaluator_registry_sha256"]),
                view_semantic_hash=str(row["view_semantic_hash"]),
                code_commit=str(row["code_commit"]),
            )
        )
    sources = []
    for item in payload.get("input_sources") or ():
        if not isinstance(item, Mapping):
            raise PaperCampaignError("campaign plan is invalid")
        sources.append(
            ArtifactIdentity(
                str(item["path"]), str(item["sha256"]), int(item["size_bytes"])
            )
        )
    sampling = payload.get("sampling_contract")
    if not isinstance(sampling, Mapping):
        sampling = {}
    return PaperPlanManifest(
        panel=str(payload["panel"]),
        arm=str(payload["arm"]),
        code_commit=str(payload["code_commit"]),
        view_semantic_hash=str(payload["view_semantic_hash"]),
        view_partition_hash=str(payload["view_partition_hash"]),
        evaluator_registry_sha256=str(payload["evaluator_registry_sha256"]),
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        autoencoder_sha256=str(payload["autoencoder_sha256"]),
        splice_report_sha256=str(payload["splice_report_sha256"]),
        sampling_contract=MappingProxyType(dict(sampling)),
        parent_counts=MappingProxyType(dict(payload.get("parent_counts") or {})),
        example_counts=MappingProxyType(dict(payload.get("example_counts") or {})),
        family_cells=MappingProxyType(dict(payload.get("family_cells") or {})),
        cells=tuple(cells),
        input_sources=tuple(sources),
        blind_opened=bool(payload.get("blind_opened")),
        foldseek_executed=bool(payload.get("foldseek_executed")),
        executed_dive_arms=tuple(payload.get("executed_dive_arms") or ()),
    )

def _identities_from_payload(payload: Mapping[str, Any]) -> dict[str, ArtifactIdentity]:
    raw = payload.get("identities") or {}
    identities: dict[str, ArtifactIdentity] = {}
    if not isinstance(raw, Mapping):
        raise PaperCampaignError("identity mismatch")
    for cell_id, item in raw.items():
        if not isinstance(item, Mapping):
            raise PaperCampaignError("identity mismatch")
        identities[str(cell_id)] = ArtifactIdentity(
            str(item["path"]), str(item["sha256"]), int(item["size_bytes"])
        )
    return identities
