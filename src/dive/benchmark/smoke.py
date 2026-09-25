
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from dive.benchmark.contracts import (
    ArtifactIdentity,
    BenchmarkContract,
    file_identity,
    load_benchmark_contract,
)
from dive.benchmark.evaluators.ame import evaluate_ame
from dive.benchmark.evaluators.antibody import (
    AntibodyComplex,
    AntibodyEvaluation,
    AtomRecord,
    FrozenAntibodyRoles,
    evaluate_antibody,
)
from dive.benchmark.evaluators.binder import evaluate_binder
from dive.benchmark.evaluators.contracts import (
    MetricUnavailable,
    MetricValue,
    load_evaluator_registry,
    verify_evaluator_availability,
)
from dive.benchmark.evaluators.records import FamilyMetricRecord
from dive.benchmark.evaluators.worker_io import DisposableEvidencePolicy
from dive.benchmark.evaluators.worker_protocol import WorkerRequest, run_worker
from dive.benchmark.evidence import (
    BenchmarkEvidenceError,
    BenchmarkRun,
    _current_clean_commit,
    _validate_roots,
    _validate_run_paths,
    claim_benchmark_run,
    complete_benchmark_run,
    write_terminal_bytes,
)
from dive.benchmark.inputs import FrozenBenchmarkInputs
from dive.emergent_contract import AUTHORIZED_LORA_RANK, EmergentContract
from dive.integrations.complexa_training import build_common_v2_model
from dive.signed_value.roots import RUNTIME_PYTHON
from dive.training.preflight import canonical_json_bytes

class SmokeError(RuntimeError):
    pass

SMOKE_SEED = 20260826
SMOKE_SAMPLE_COUNT = 1
SMOKE_SAMPLE_INDEX = 0
SMOKE_STEPS = 2
SMOKE_ARM = "base"
SMOKE_VERIFICATIONS = (
    "load_splice",
    "schema",
    "evaluator_invocation",
    "identity",
    "evidence_paths",
)
_FAMILIES = ("binder", "ame", "antibody")
_PERMITTED_DEVICES = tuple(range(8))
_SELECTION_TOKENS = ("select", "winner", "rank")
_SMOKE_INPUT_RUN = "benchmark-inputs-20260826c"
_SMOKE_NRES = 32
_SMOKE_TARGET_NRES = 8
_FIXTURE_REQUEST_IDS = {
    "binder": "fixture-binder-ok",
    "ame": "fixture-ame-ok",
}
_SMOKE_SAMPLING_MODEL = {
    "bb_ca": {
        "schedule": {"mode": "log", "p": 2.0},
        "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": "sc",
            "sc_scale_noise": 0.1,
            "sc_scale_score": 1.0,
            "t_lim_ode": 0.98,
            "t_lim_ode_below": 0.02,
            "tsr_k": 1.0,
            "tsr_sigma": 1.0,
            "center_every_step": False,
        },
    },
    "local_latents": {
        "schedule": {"mode": "power", "p": 2.0},
        "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
        "simulation_step_params": {
            "sampling_mode": "sc",
            "sc_scale_noise": 0.1,
            "sc_scale_score": 1.0,
            "t_lim_ode": 0.98,
            "t_lim_ode_below": 0.02,
            "tsr_k": 1.0,
            "tsr_sigma": 1.0,
            "center_every_step": False,
        },
    },
}

@dataclass(frozen=True, slots=True)
class SmokeParent:
    parent_id: str
    family: str
    partition: str
    standard: bool = True

@dataclass(frozen=True, slots=True)
class SmokeInputs:
    parents: tuple[object, ...]
    blind_parent_ids: frozenset[str]
    examples: tuple[object, ...] = ()

@dataclass(frozen=True, slots=True)
class SmokeCell:
    family: str
    partition: str
    parent_id: str
    checkpoint: str
    autoencoder: str
    seed: int
    sample_index: int
    steps: int
    cell_id: str
    arm: str = SMOKE_ARM

@dataclass(frozen=True, slots=True)
class SmokePlan:
    cells: tuple[SmokeCell, ...]
    seed: int = SMOKE_SEED
    sample_count: int = SMOKE_SAMPLE_COUNT
    steps: int = SMOKE_STEPS
    verifications: tuple[str, ...] = SMOKE_VERIFICATIONS

    def __iter__(self):
        return iter(self.cells)

    def __len__(self) -> int:
        return len(self.cells)

    def __bool__(self) -> bool:
        return bool(self.cells)

    def as_mapping(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "sample_count": self.sample_count,
            "steps": self.steps,
            "verifications": list(self.verifications),
            "cells": [_cell_mapping(cell) for cell in self.cells],
        }

@dataclass(frozen=True, slots=True)
class SmokePreflight:
    device: int
    dive_commit: str

def build_smoke_plan(inputs, registry, contract) -> SmokePlan:

    _require_train_only_contract(contract)
    families = _registry_families(registry)
    blind = _blind_parent_ids(inputs)
    checkpoints = dict(getattr(contract, "checkpoints", ()))
    if "common" not in checkpoints or "autoencoder" not in checkpoints:
        raise SmokeError("smoke plan requires common and autoencoder checkpoints")
    common = checkpoints["common"].sha256
    autoencoder = checkpoints["autoencoder"].sha256
    cells = []
    for family in families:
        parent_id = _lex_first_train_parent(inputs, family, blind)
        payload = {
            "arm": SMOKE_ARM,
            "autoencoder": autoencoder,
            "checkpoint": common,
            "family": family,
            "parent_id": parent_id,
            "partition": "train",
            "sample_index": SMOKE_SAMPLE_INDEX,
            "seed": SMOKE_SEED,
            "steps": SMOKE_STEPS,
        }
        cells.append(
            SmokeCell(
                family=family,
                partition="train",
                parent_id=parent_id,
                checkpoint=common,
                autoencoder=autoencoder,
                seed=SMOKE_SEED,
                sample_index=SMOKE_SAMPLE_INDEX,
                steps=SMOKE_STEPS,
                cell_id=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
            )
        )
    plan = SmokePlan(tuple(cells))
    if {cell.parent_id for cell in plan} & blind:
        raise SmokeError("smoke plan included a blind parent id")
    return plan

def validate_smoke_result(result, plan, registry) -> None:

    mapping = _result_mapping(result)
    _assert_no_selection_fields(mapping)
    metrics = mapping.get("metrics")
    if not isinstance(metrics, Mapping):
        raise SmokeError("typed metric records are required")
    families = [cell.family for cell in plan]
    for family in families:
        if family not in metrics:
            raise SmokeError("missing required primary")
        primary = _typed_primary(family, metrics[family])
        expected = _primary_name(registry, family)
        if isinstance(primary, MetricUnavailable):
            raise SmokeError("missing required primary")
        if not isinstance(primary, MetricValue) or primary.name != expected:
            raise SmokeError("missing required primary")
        if type(primary.value) is not float or not math.isfinite(primary.value):
            raise SmokeError(
                "nonfinite required primary must be a typed metric unavailability"
            )
    extra = set(metrics) - set(families)
    if extra:
        raise SmokeError(f"smoke result has unexpected metric families {sorted(extra)}")

def preflight_smoke_execution(
    contract,
    *,
    occupied_gpus: frozenset[int] | None = None,
    smoke_pids: tuple[int, ...] | None = None,
    preferred_device: int | None = None,
) -> SmokePreflight:

    if not isinstance(contract, BenchmarkContract):
        raise SmokeError("smoke preflight requires a BenchmarkContract")
    dive_commit = _require_clean_dive_commit()
    _require_clean_upstream(contract)
    _authenticate_smoke_checkpoints(contract)
    _require_evidence_paths(contract)
    pids = _smoke_process_pids() if smoke_pids is None else smoke_pids
    if pids:
        raise SmokeError(f"existing smoke process {pids} must exit before a new smoke")
    occupied = (
        _occupied_gpu_indices() if occupied_gpus is None else frozenset(occupied_gpus)
    )
    device = _select_device(occupied, preferred_device)
    return SmokePreflight(device, dive_commit)

def claim_smoke_run(
    run_id: str, argv: Sequence[str], contract: BenchmarkContract
) -> BenchmarkRun:

    run = claim_benchmark_run(run_id, argv, contract)
    evidence_root = Path(contract.evidence_root)
    bulk_root = Path(contract.bulk_root)
    claim = {
        "schema_version": "dive-benchmark-smoke-claim-v1",
        "run_id": run.run_id,
        "attempt": {
            "path": run.attempt.path,
            "sha256": run.attempt.sha256,
            "size_bytes": run.attempt.size_bytes,
        },
        "contract_sha256": run.contract_sha256,
        "evidence_path": str(run.evidence_dir),
        "bulk_path": str(run.bulk_dir),
    }
    write_terminal_bytes(
        run.evidence_dir / "smoke.claim.json",
        canonical_json_bytes(claim),
        evidence_root,
        bulk_root,
    )
    write_terminal_bytes(
        run.evidence_dir / "controller.log",
        b"",
        evidence_root,
        bulk_root,
    )
    return run

def resume_smoke_claim(
    run_id: str, argv: Sequence[str], contract: BenchmarkContract
) -> BenchmarkRun:

    del argv
    if not isinstance(contract, BenchmarkContract):
        raise SmokeError("resume-claim requires a BenchmarkContract")
    _require_evidence_paths(contract)
    evidence_dir = Path(contract.evidence_root) / "benchmark_ready" / run_id
    bulk_dir = Path(contract.bulk_root) / "benchmark_ready" / run_id
    claim_path = evidence_dir / "smoke.claim.json"
    attempt_path = evidence_dir / "attempt.json"
    if not claim_path.is_file() or not attempt_path.is_file():
        raise SmokeError("resume-claim requires the exact claimed attempt identity")
    try:
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"cannot read smoke claim: {error}") from error
    if not isinstance(claim, Mapping):
        raise SmokeError("smoke claim is not a mapping")
    try:
        live = file_identity(attempt_path)
    except Exception as error:
        raise SmokeError(f"cannot authenticate attempt identity: {error}") from error
    recorded = claim.get("attempt")
    if (
        not isinstance(recorded, Mapping)
        or claim.get("run_id") != run_id
        or claim.get("contract_sha256") != contract.semantic_sha256
        or recorded.get("sha256") != live.sha256
        or recorded.get("size_bytes") != live.size_bytes
        or recorded.get("path") != live.path
    ):
        raise SmokeError("attempt identity does not match the claimed smoke run")
    return BenchmarkRun(
        run_id=run_id,
        evidence_dir=evidence_dir,
        bulk_dir=bulk_dir,
        attempt=live,
        contract_sha256=contract.semantic_sha256,
    )

def execute_smoke(
    inputs,
    plan,
    registry,
    contract,
    *,
    model_constructor: Callable[[], object] | None = None,
    evaluator_runner: Callable[[SmokeCell, object], object] | None = None,
    preflight: Callable[[], object] | None = None,
) -> dict[str, object]:

    _refuse_blind_parents(inputs, plan)
    _require_evidence_paths(contract)
    if preflight is not None:
        preflight()
    else:
        preflight_smoke_execution(contract)
    constructor = (
        model_constructor
        if model_constructor is not None
        else lambda: _load_base_checkpoint(contract)
    )
    model = constructor()
    if evaluator_runner is None:
        raise SmokeError(
            "evaluator invocation is required; refusing unvalidated empty metrics"
        )
    metrics = {cell.family: evaluator_runner(cell, model) for cell in plan}
    result = {
        "schema_version": "dive-benchmark-smoke-result-v1",
        "metrics": metrics,
        "cells": [cell.cell_id for cell in plan],
        "verifications": list(plan.verifications),
    }
    validate_smoke_result(result, plan, registry)
    return result

def write_smoke_completion(run, result, plan, registry) -> ArtifactIdentity:

    validate_smoke_result(result, plan, registry)
    if not isinstance(run, BenchmarkRun):
        raise SmokeError("write_smoke_completion requires a BenchmarkRun")
    mapping = _result_mapping(result)
    metrics = mapping["metrics"]
    if not isinstance(metrics, Mapping):
        raise SmokeError("typed metric records are required")
    cells_payload = []
    metrics_payload = {}
    for cell in plan:
        record = _metric_record_mapping(metrics[cell.family])
        cells_payload.append({**_cell_mapping(cell), "metrics": record})
        metrics_payload[cell.family] = record
    payload = {
        "schema_version": "dive-benchmark-smoke-completion-v1",
        "plan": plan.as_mapping(),
        "cells": cells_payload,
        "metrics": metrics_payload,
        "verifications": list(plan.verifications),
        "purpose": "engineering_validation_not_performance_baseline",
        "blind": False,
        "partition": "train",
    }
    try:
        evidence_root, bulk_root = _validate_run_paths(run)
    except BenchmarkEvidenceError as error:
        raise SmokeError(str(error)) from error
    write_terminal_bytes(
        run.evidence_dir / "smoke.completion.json",
        canonical_json_bytes(payload),
        evidence_root,
        bulk_root,
    )
    return complete_benchmark_run(run, payload)

def cli_main(argv: Sequence[str] | None = None) -> int:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--claim-only", action="store_true")
    parser.add_argument("--resume-claim", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if Path(sys.executable).resolve() != Path(RUNTIME_PYTHON).resolve():
        raise RuntimeError("smoke_benchmark_evaluators requires the pinned interpreter")
    if args.claim_only == args.resume_claim:
        raise SmokeError("exactly one of --claim-only or --resume-claim is required")
    contract = load_benchmark_contract(args.config)
    command = tuple(
        sys.argv if argv is None else ["smoke_benchmark_evaluators.py", *argv]
    )
    if args.claim_only:
        run = claim_smoke_run(args.run_id, command, contract)
        print(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "status": "BUILDING",
                    "claim_only": True,
                    "attempt": run.attempt.sha256,
                    "evidence_path": str(run.evidence_dir),
                    "controller_log": str(run.evidence_dir / "controller.log"),
                },
                sort_keys=True,
            )
        )
        return 0
    run = resume_smoke_claim(args.run_id, command, contract)
    preflight_smoke_execution(contract)
    inputs, registry, plan = _smoke_execution_context(contract)
    result = execute_smoke(
        inputs,
        plan,
        registry,
        contract,
        model_constructor=lambda: _load_base_checkpoint(
            contract, store_dir=run.bulk_dir / "store"
        ),
        evaluator_runner=_smoke_evaluator_runner(run, contract, inputs),
        preflight=lambda: None,
    )
    completion = write_smoke_completion(run, result, plan, registry)
    print(
        json.dumps(
            {
                "run_id": run.run_id,
                "status": "COMPLETE",
                "resume_claim": True,
                "attempt": run.attempt.sha256,
                "evidence_path": str(run.evidence_dir),
                "completion": completion.sha256,
            },
            sort_keys=True,
        )
    )
    return 0

def _load_base_checkpoint(contract, *, store_dir: Path | None = None):

    if not isinstance(contract, BenchmarkContract):
        raise SmokeError("base-checkpoint load requires a BenchmarkContract")
    _authenticate_smoke_checkpoints(contract)
    _require_clean_upstream(contract)
    emergent = _emergent_contract_from_benchmark(contract)
    destination = Path(store_dir or Path(contract.bulk_root) / "smoke_store")
    try:
        model, _report = build_common_v2_model(
            emergent,
            store_dir=destination,
            upstream_root=Path(contract.upstream_root),
            install_presence=False,
            lora_dropout=0.0,
        )
    except Exception as error:
        raise SmokeError(f"cannot load spliced base checkpoint: {error}") from error
    freeze = getattr(model, "eval", None)
    if callable(freeze):
        freeze()
    named = getattr(model, "named_parameters", None)
    if callable(named):
        for _name, parameter in named():
            requires_grad = getattr(parameter, "requires_grad_", None)
            if callable(requires_grad):
                requires_grad(False)
    return _maybe_cuda(model)

def _maybe_cuda(model):

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return model
    import torch

    if not torch.cuda.is_available():
        raise SmokeError("CUDA_VISIBLE_DEVICES is set but CUDA is unavailable")
    to_device = getattr(model, "to", None)
    if not callable(to_device):
        raise SmokeError("loaded model cannot move to CUDA")
    return to_device("cuda")

def _smoke_execution_context(contract, *, frozen_inputs=None, registry=None):

    if registry is None:
        registry = _planning_registry(contract)
    if frozen_inputs is None:
        frozen_inputs = _load_smoke_frozen_inputs(contract)
    inputs = _smoke_inputs_from_frozen(frozen_inputs)
    return inputs, registry, build_smoke_plan(inputs, registry, contract)

def _planning_registry(contract):
    evaluators = tuple(getattr(contract, "evaluators", ()) or ())
    families = tuple(item.family for item in evaluators)
    if not families:
        raise SmokeError("evaluator registry has no families")

    def for_family(family: str):
        for item in evaluators:
            if item.family == family:
                return item
        raise SmokeError(f"evaluator family is not registered: {family}")

    return SimpleNamespace(
        families=families,
        for_family=for_family,
        semantic_sha256=getattr(contract, "semantic_sha256", "unbound"),
    )

def _load_smoke_frozen_inputs(contract) -> FrozenBenchmarkInputs:

    from dive.benchmark.views import (
        BenchmarkViewError,
        _ACCEPTED_INPUT_RUN,
        _authenticate_input_run,
        _extend_with_canonical_blind,
        _inputs_from_inventory,
    )

    if _ACCEPTED_INPUT_RUN != _SMOKE_INPUT_RUN:
        raise SmokeError("smoke input run is not the accepted Task 2 inventory")
    evidence_root = Path(contract.evidence_root) / "benchmark_ready"
    try:
        _completion, inventory = _authenticate_input_run(
            evidence_root / _SMOKE_INPUT_RUN
        )
        frozen = _inputs_from_inventory(inventory)
        return _extend_with_canonical_blind(frozen)
    except (BenchmarkViewError, OSError, TypeError, ValueError) as error:
        raise SmokeError(f"cannot load frozen train parents: {error}") from error

def _smoke_inputs_from_frozen(frozen) -> SmokeInputs:
    parents_raw = getattr(frozen, "parents", None)
    if not isinstance(parents_raw, Sequence) or isinstance(parents_raw, (str, bytes)):
        raise SmokeError("frozen inputs must provide parents")
    train_parents = []
    blind: set[str] = set()
    for parent in parents_raw:
        parent_id, family, partition, _standard = _parent_fields(parent)
        if partition == "test-blind":
            blind.add(parent_id)
            continue
        if partition != "train":
            continue

        train_parents.append(SmokeParent(parent_id, family, "train", True))
    examples = tuple(getattr(frozen, "examples", ()) or ())
    return SmokeInputs(tuple(train_parents), frozenset(blind), examples)

def _emergent_contract_from_benchmark(contract: BenchmarkContract) -> EmergentContract:
    from datetime import date

    checkpoints = dict(contract.checkpoints)
    if "common" not in checkpoints or "autoencoder" not in checkpoints:
        raise SmokeError("smoke load requires common and autoencoder checkpoints")
    return EmergentContract(
        upstream_root=Path(contract.upstream_root),
        upstream_commit=str(contract.upstream_commit),
        common_checkpoint=Path(checkpoints["common"].path),
        autoencoder_checkpoint=Path(checkpoints["autoencoder"].path),
        architecture_v2=True,
        lora_rank=AUTHORIZED_LORA_RANK,
        gpu_count=1,
        evidence_root=Path(contract.evidence_root),
        bulk_root=Path(contract.bulk_root),
        backup_deadline=date(2026, 8, 27),
    )

def _smoke_evaluator_runner(run: BenchmarkRun, contract, inputs):
    policy = DisposableEvidencePolicy(run.evidence_dir, run.bulk_dir)

    def runner(cell: SmokeCell, model):
        sample = _generate_smoke_sample(
            model, cell, example=_example_for_cell(inputs, cell), contract=contract
        )
        return _evaluate_smoke_cell(
            cell,
            model,
            sample=sample,
            run=run,
            contract=contract,
            inputs=inputs,
            policy=policy,
        )

    return runner

def _generate_smoke_sample(model, cell, example=None, contract=None):

    del example, contract
    if not isinstance(cell, SmokeCell):
        raise SmokeError("smoke generation requires a SmokeCell")
    if cell.steps != SMOKE_STEPS:
        raise SmokeError("smoke generation must use the frozen 2-step contract")
    if cell.seed != SMOKE_SEED:
        raise SmokeError("smoke generation must use the frozen seed")
    if cell.sample_index != SMOKE_SAMPLE_INDEX:
        raise SmokeError("smoke generation must use sample index 0")
    if cell.partition != "train" or cell.arm != SMOKE_ARM:
        raise SmokeError("smoke generation is train-only base arm")
    _seed_smoke_rng(cell.seed)
    inf_cfg = _smoke_inference_config(cell)
    configure = getattr(model, "configure_inference", None)
    if callable(configure):
        configure(inf_cfg, None)
    eval_model = getattr(model, "eval", None)
    if callable(eval_model):
        eval_model()
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise SmokeError("loaded model cannot generate samples")
    batch = _smoke_generation_batch(model)
    generated = generate(batch)
    _assert_finite_generated(generated)
    return generated

def _seed_smoke_rng(seed: int) -> None:
    try:
        import lightning as L

        L.seed_everything(seed, workers=True)
    except Exception:
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)

def _smoke_inference_config(cell: SmokeCell):
    args = _Cfg(
        nsteps=cell.steps,
        self_cond=False,
        guidance_w=1.0,
        ag_ratio=0.0,
    )
    return _Cfg(args=args, model=_SMOKE_SAMPLING_MODEL, n_recycle=0)

def _smoke_generation_batch(model) -> dict[str, object]:

    import torch

    device = _model_device(model)
    batch = SMOKE_SAMPLE_COUNT
    generated = _SMOKE_NRES
    target = _SMOKE_TARGET_NRES
    mask = torch.ones(batch, generated, dtype=torch.bool, device=device)
    x_target = torch.zeros(batch, target, 37, 3, device=device)
    target_mask = torch.zeros(batch, target, 37, dtype=torch.bool, device=device)
    target_mask[:, :, :4] = True
    seq_target_mask = torch.ones(batch, target, dtype=torch.bool, device=device)
    zeros_target = torch.zeros(batch, target, dtype=torch.bool, device=device)
    return {
        "mask": mask,
        "x_target": x_target,
        "target_mask": target_mask,
        "seq_target": torch.zeros(batch, target, dtype=torch.long, device=device),
        "seq_target_mask": seq_target_mask,
        "target_chains": torch.zeros(batch, target, dtype=torch.long, device=device),
        "target_hotspot_mask": zeros_target,
        "hotspot_mask": torch.zeros(batch, generated, dtype=torch.bool, device=device),
        "chains": torch.ones(batch, generated, dtype=torch.long, device=device),
        "prepend_target": False,
        "atomistic_target": False,
    }

def _model_device(model):
    import torch

    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter = next(parameters())
        except StopIteration:
            parameter = None
        if parameter is not None and hasattr(parameter, "device"):
            return parameter.device
    return torch.device("cpu")

def _assert_finite_generated(generated) -> None:
    import torch

    values = generated.values() if isinstance(generated, Mapping) else (generated,)
    for value in values:
        if not hasattr(value, "isfinite"):
            continue
        try:
            finite = bool(torch.isfinite(value).all())
        except Exception:
            continue
        if not finite:
            raise SmokeError("nonfinite decoder output")

def _evaluate_smoke_cell(
    cell,
    model,
    *,
    sample,
    run,
    contract,
    inputs,
    policy=None,
    registry=None,
    availability=None,
):

    del model
    if not isinstance(cell, SmokeCell):
        raise SmokeError("evaluator invocation requires a SmokeCell")
    if policy is None:
        if not isinstance(run, BenchmarkRun):
            raise SmokeError("evaluator invocation requires a BenchmarkRun")
        policy = DisposableEvidencePolicy(run.evidence_dir, run.bulk_dir)
    if registry is None:
        registry = load_evaluator_registry(contract)
    if availability is None:
        availability = verify_evaluator_availability(registry)
    if cell.family in {"binder", "ame"}:
        request = _smoke_worker_request(
            cell=cell, run=run, registry=registry, contract=contract
        )
        evaluate = evaluate_binder if cell.family == "binder" else evaluate_ame
        return evaluate(
            request,
            registry,
            availability,
            runner=run_worker,
            policy=policy,
        )
    if cell.family == "antibody":
        complex_record = _antibody_complex_for_smoke(
            cell=cell,
            sample=sample,
            inputs=inputs,
            registry=registry,
        )
        return evaluate_antibody(complex_record)
    raise SmokeError(f"unsupported smoke family {cell.family}")

def _smoke_worker_request(*, cell, run, registry, contract) -> WorkerRequest:
    del contract
    evaluator = registry.for_family(cell.family)
    worker = getattr(evaluator, "worker", None)
    if worker is None:
        raise SmokeError(f"registry worker is missing for {cell.family}")
    request_id = _FIXTURE_REQUEST_IDS[cell.family]
    evidence_dir = Path(run.evidence_dir) / "evaluators" / cell.family
    bulk_dir = Path(run.bulk_dir) / "evaluators" / cell.family
    substitutions = {
        "{interpreter}": worker.interpreter.path,
        "{worker}": worker.source.path,
        "{request_path}": str(evidence_dir / "request.json"),
    }
    dive_commit = _require_clean_dive_commit()
    return WorkerRequest(
        schema_version=worker.request_schema,
        request_id=request_id,
        run_id=_request_run_id(run.run_id, cell.family),
        family=cell.family,
        benchmark_contract_sha256=registry.benchmark_contract_hash,
        registry_semantic_sha256=registry.semantic_sha256,
        evaluator_semantic_sha256=evaluator.semantic_sha256,
        dive_commit=dive_commit,
        upstream_commit=evaluator.source_commit,
        artifacts={},
        worker_source=worker.source,
        worker_interpreter=worker.interpreter.identity,
        worker_environment=worker.interpreter.environment,
        worker_callable=worker.callable_name,
        worker_schema_version=worker.schema_version,
        result_schema=worker.result_schema,
        reducer_identifier=evaluator.reducer.identifier,
        reducer_semantic_sha256=evaluator.reducer.semantic_sha256,
        frozen_ids={"parent_id": (cell.parent_id,)},
        frozen_roles={"target_chains": ("A",)},
        argv=tuple(substitutions.get(item, item) for item in worker.argv_template),
        working_directory=worker.resolved_working_directory,
        environment=dict(worker.environment),
        environment_allowlist=worker.environment_allowlist,
        request_path=str(evidence_dir / "request.json"),
        result_path=str(evidence_dir / "outputs" / "result.json"),
        stdout_path=str(evidence_dir / "outputs" / "stdout.log"),
        stderr_path=str(evidence_dir / "outputs" / "stderr.log"),
        bulk_directory=str(bulk_dir),
        timeout_seconds=30.0,
        resource_limits={"memory_mb": 1024},
    )

def _request_run_id(run_id: str, family: str) -> str:
    candidate = f"{run_id}-{family}"
    if all(ch.isalnum() or ch in "._-" for ch in candidate) and candidate[0].isalnum():
        return candidate
    return f"smoke-{family}"

def _example_for_cell(inputs, cell: SmokeCell):
    for example in getattr(inputs, "examples", ()) or ():
        example_parent = getattr(example, "parent_id", None)
        example_family = getattr(example, "family", None)
        if example_parent == cell.parent_id and example_family == cell.family:
            return example
    return None

def _antibody_complex_for_smoke(*, cell, sample, inputs, registry) -> AntibodyComplex:
    evaluator = registry.for_family(cell.family)
    manifest_hash = getattr(evaluator, "semantic_sha256", None)
    if type(manifest_hash) is not str or len(manifest_hash) != 64:
        manifest_hash = hashlib.sha256(b"antibody").hexdigest()
    example = _example_for_cell(inputs, cell)
    roles = _frozen_antibody_roles(example)
    native = _native_ca_atoms(getattr(example, "path", None), roles)
    predicted = _predicted_ca_atoms(sample, roles, native)
    return AntibodyComplex(native, predicted, roles, manifest_hash)

def _frozen_antibody_roles(example) -> FrozenAntibodyRoles:
    roles = getattr(example, "roles", {}) if example is not None else {}
    generated = str(roles.get("generated", "") or "")
    target = str(roles.get("target", "") or "")
    heavy, residues = _parse_generated_span(generated)
    antigen = tuple(part for part in target.split(",") if part)
    if not heavy or len(residues) < 3 or not antigen:
        raise SmokeError("antibody train parent is missing frozen H3 or antigen roles")
    return FrozenAntibodyRoles(heavy, residues, antigen)

def _parse_generated_span(value: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    if ":" not in value:
        return "", ()
    chain, span = value.split(":", 1)
    chain = chain.strip()
    if "-" not in span or not chain:
        return chain, ()
    start_text, end_text = span.split("-", 1)
    try:
        start, end = int(start_text), int(end_text)
    except ValueError:
        return chain, ()
    if end < start:
        return chain, ()
    residues = tuple((chain, str(index)) for index in range(start, end + 1))
    return chain, residues

def _native_ca_atoms(path, roles: FrozenAntibodyRoles) -> tuple[AtomRecord, ...]:
    if type(path) is not str or not path:
        raise SmokeError("antibody train parent is missing a native structure path")
    try:
        from Bio.PDB import MMCIFParser, PDBParser
    except ImportError as error:
        raise SmokeError("antibody native parse requires Biopython") from error
    source = Path(path)
    try:
        if source.suffix == ".gz":
            import gzip

            handle = gzip.open(source, "rt")
        else:
            handle = source.open("rt")
        with handle as stream:
            if ".cif" in source.name:
                structure = MMCIFParser(QUIET=True).get_structure("native", stream)
            else:
                structure = PDBParser(QUIET=True).get_structure("native", stream)
    except (OSError, ValueError) as error:
        raise SmokeError(f"cannot parse antibody native structure: {error}") from error
    wanted = {(chain, residue) for chain, residue in roles.cdr_h3}
    wanted.update(
        (chain, residue) for chain in roles.antigen_chains for residue in ("1",)
    )
    atoms = []
    for atom in structure.get_atoms():
        if atom.get_name() != "CA":
            continue
        residue = atom.get_parent()
        chain = residue.get_parent()
        residue_id = str(residue.get_id()[1])
        if (chain.id, residue_id) not in wanted:
            continue
        coord = tuple(float(value) for value in atom.get_coord())
        atoms.append(
            AtomRecord(
                chain.id, residue_id, "CA", coord, element="C", residue_name="GLY"
            )
        )
    if len([atom for atom in atoms if (atom.chain, atom.residue) in roles.cdr_h3]) < 3:
        raise SmokeError("native antibody structure is missing frozen H3 C-alpha atoms")
    return tuple(atoms)

def _predicted_ca_atoms(
    sample, roles: FrozenAntibodyRoles, native: tuple[AtomRecord, ...]
) -> tuple[AtomRecord, ...]:
    coords = _ca_coords_from_generated(sample)
    predicted = []
    for index, (chain, residue) in enumerate(roles.cdr_h3):
        if index >= len(coords):
            break
        predicted.append(
            AtomRecord(
                chain,
                residue,
                "CA",
                coords[index],
                element="C",
                residue_name="GLY",
            )
        )
    if len(predicted) < 3:
        raise SmokeError("generated sample is missing C-alpha coordinates")
    antigen_native = [
        atom for atom in native if atom.chain in set(roles.antigen_chains)
    ]
    return tuple(predicted) + tuple(antigen_native)

def _ca_coords_from_generated(sample) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(sample, Mapping):
        return ()
    value = sample.get("bb_ca", sample.get("coors"))
    if value is None:
        return ()
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return ()
    first = value[0]
    rows = (
        first
        if isinstance(first, (list, tuple))
        and first
        and isinstance(first[0], (list, tuple))
        else value
    )
    coords = []
    for row in rows:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        if isinstance(row[0], (list, tuple)):
            row = row[1] if len(row) > 1 else row[0]
        try:
            coords.append((float(row[0]), float(row[1]), float(row[2])))
        except (TypeError, ValueError):
            continue
    return tuple(coords)

def _require_train_only_contract(contract) -> None:
    scope = getattr(contract, "smoke_scope", None)
    if scope != "train_only":
        raise SmokeError("smoke.scope must be train_only; blind access remains sealed")

def _registry_families(registry) -> tuple[str, ...]:
    families = tuple(getattr(registry, "families", ()) or ())
    if not families:
        raise SmokeError("evaluator registry has no families")
    unknown = [family for family in families if family not in _FAMILIES]
    if unknown:
        raise SmokeError(f"registry contains unsupported families {unknown}")
    return families

def _blind_parent_ids(inputs) -> frozenset[str]:
    value = getattr(inputs, "blind_parent_ids", None)
    if value is None:
        raise SmokeError("smoke inputs must declare blind_parent_ids")
    return frozenset(value)

def _lex_first_train_parent(inputs, family: str, blind: frozenset[str]) -> str:
    parents = getattr(inputs, "parents", None)
    if not isinstance(parents, Sequence) or isinstance(parents, (str, bytes)):
        raise SmokeError("smoke inputs must provide parents")
    eligible = []
    for parent in parents:
        parent_id, parent_family, partition, standard = _parent_fields(parent)
        if (
            parent_family == family
            and partition == "train"
            and standard
            and parent_id not in blind
        ):
            eligible.append(parent_id)
    if not eligible:
        raise SmokeError(f"no eligible train parent for family {family}")
    ordered = sorted(eligible)
    if family != "antibody" or not getattr(inputs, "examples", ()):
        return ordered[0]
    for parent_id in ordered:
        example = _example_for_cell(
            inputs, SmokeParent(parent_id, family, "train", True)
        )
        try:
            roles = _frozen_antibody_roles(example)
            _native_ca_atoms(getattr(example, "path", None), roles)
        except SmokeError:
            continue
        return parent_id
    return ordered[0]

def _parent_fields(parent) -> tuple[str, str, str, bool]:
    parent_id = getattr(parent, "parent_id", None)
    family = getattr(parent, "family", None)
    partition = getattr(parent, "partition", None)
    if type(parent_id) is not str or not parent_id:
        raise SmokeError("parent_id must be a non-empty string")
    if type(family) is not str or not family:
        raise SmokeError("parent family must be a non-empty string")
    if type(partition) is not str or not partition:
        raise SmokeError("parent partition must be a non-empty string")
    if hasattr(parent, "standard"):
        standard = bool(parent.standard)
    elif hasattr(parent, "view_standard"):
        standard = bool(parent.view_standard)
    else:
        raise SmokeError("parent is missing a standard-view flag")
    return parent_id, family, partition, standard

def _result_mapping(result) -> Mapping[str, object]:
    if isinstance(result, Mapping):
        return result
    as_mapping = getattr(result, "as_mapping", None)
    if callable(as_mapping):
        mapping = as_mapping()
        if isinstance(mapping, Mapping):
            return mapping
    raise SmokeError("smoke result must be a mapping")

def _assert_no_selection_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _SELECTION_TOKENS) or lowered in {
                "best.ckpt",
                "best_ckpt",
                "best_checkpoint",
            }:
                raise SmokeError("selection fields are forbidden in smoke results")
            _assert_no_selection_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_selection_fields(item)

def _typed_primary(family: str, value: object):
    if type(value) is int or type(value) is float:
        raise SmokeError("typed metric records are required")
    if isinstance(value, FamilyMetricRecord):
        if value.family != family:
            raise SmokeError("typed metric family mismatch")
        return value.primary
    if isinstance(value, AntibodyEvaluation):
        return value.primary
    if isinstance(value, (MetricValue, MetricUnavailable)):
        return value
    raise SmokeError("typed metric records are required")

def _primary_name(registry, family: str) -> str:
    evaluator = registry.for_family(family)
    primary = getattr(evaluator, "primary", None)
    name = getattr(primary, "name", primary if type(primary) is str else None)
    if type(name) is not str or not name:
        raise SmokeError(f"registry primary is missing for {family}")
    return name

def _require_clean_dive_commit() -> str:
    try:
        return _current_clean_commit()
    except BenchmarkEvidenceError as error:
        raise SmokeError(str(error)) from error

def _require_clean_upstream(contract) -> None:
    root = Path(getattr(contract, "upstream_root", ""))
    expected = getattr(contract, "upstream_commit", None)
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        head.returncode != 0
        or head.stdout.strip() != expected
        or dirty.returncode != 0
        or dirty.stdout
    ):
        raise SmokeError("upstream checkout HEAD or cleanliness drifted")

def _authenticate_smoke_checkpoints(contract) -> None:
    checkpoints = dict(contract.checkpoints)
    for name in ("common", "autoencoder"):
        if name not in checkpoints:
            raise SmokeError(f"missing {name} checkpoint identity")
        expected = checkpoints[name]
        try:
            observed = file_identity(Path(expected.path))
        except Exception as error:
            raise SmokeError(
                f"cannot authenticate {name} checkpoint: {error}"
            ) from error
        if (
            observed.sha256 != expected.sha256
            or observed.size_bytes != expected.size_bytes
        ):
            raise SmokeError(f"checkpoint identity drifted for {name}")

def _require_evidence_paths(contract) -> None:
    try:
        _validate_roots(contract)
    except BenchmarkEvidenceError as error:
        raise SmokeError(str(error)) from error

def _occupied_gpu_indices() -> frozenset[int]:
    uuid_to_index = _gpu_uuid_to_index()
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
        raise SmokeError("GPU inventory is unavailable")
    occupied: set[int] = set()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        gpu_uuid = parts[0]
        if gpu_uuid in uuid_to_index:
            occupied.add(uuid_to_index[gpu_uuid])
    return frozenset(occupied)

def _gpu_uuid_to_index() -> dict[str, int]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SmokeError("GPU inventory is unavailable")
    mapping: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        index_text, uuid = [part.strip() for part in line.split(",", 1)]
        mapping[uuid] = int(index_text)
    return mapping

def _cmdline_is_smoke_python(raw: bytes) -> bool:

    parts = [part.decode("utf-8", "replace") for part in raw.split(b"\x00") if part]
    if not parts:
        return False
    executable = Path(parts[0]).name
    if executable in {"bash", "sh", "dash", "zsh"}:
        return False
    return any(Path(part).name == "smoke_benchmark_evaluators.py" for part in parts)

def _smoke_process_pids() -> tuple[int, ...]:
    current = os.getpid()
    root = Path("/proc")
    if not root.is_dir():
        return ()
    pids: list[int] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if _cmdline_is_smoke_python(raw):
            pids.append(pid)
    return tuple(pids)

def _select_device(occupied: frozenset[int], preferred: int | None) -> int:
    chosen = preferred
    if chosen is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            first = visible.split(",", 1)[0].strip()
            chosen = int(first) if first.isdigit() else 0
        else:
            chosen = 0
    if chosen in _PERMITTED_DEVICES and chosen not in occupied:
        return chosen
    for device in _PERMITTED_DEVICES:
        if device not in occupied:
            return device
    raise SmokeError("no available GPU for the smoke")

def _refuse_blind_parents(inputs, plan) -> None:
    blind = _blind_parent_ids(inputs)
    overlap = {cell.parent_id for cell in plan} & blind
    if overlap:
        raise SmokeError(
            f"blind parent {sorted(overlap)} excluded before model construction"
        )
    if any(cell.partition != "train" for cell in plan):
        raise SmokeError("blind or non-train parent excluded before model construction")

def _cell_mapping(cell: SmokeCell) -> dict[str, object]:
    return {
        "family": cell.family,
        "partition": cell.partition,
        "parent_id": cell.parent_id,
        "checkpoint": cell.checkpoint,
        "autoencoder": cell.autoencoder,
        "seed": cell.seed,
        "sample_index": cell.sample_index,
        "steps": cell.steps,
        "cell_id": cell.cell_id,
        "arm": cell.arm,
    }

def _metric_record_mapping(value: object) -> dict[str, object]:
    if isinstance(value, FamilyMetricRecord):
        return {
            "kind": "family_metric_record",
            "family": value.family,
            "primary": _metric_leaf_mapping(value.primary),
            "raw": {name: _jsonable_raw(raw) for name, raw in value.raw.items()},
            "optional": {
                name: _metric_leaf_mapping(item)
                for name, item in value.optional.items()
            },
        }
    if isinstance(value, AntibodyEvaluation):
        return {
            "kind": "antibody_evaluation",
            "primary": _metric_leaf_mapping(value.primary),
            "secondary": {
                name: _metric_leaf_mapping(item)
                for name, item in value.secondary.items()
            },
        }
    return _metric_leaf_mapping(value)

def _metric_leaf_mapping(value: object) -> dict[str, object]:
    if isinstance(value, MetricValue):
        return {
            "kind": "metric_value",
            "name": value.name,
            "value": value.value,
            "direction": value.direction,
            "evaluator_manifest_hash": value.evaluator_manifest_hash,
            "status": "available",
        }
    if isinstance(value, MetricUnavailable):
        status = value.status
        return {
            "kind": "metric_unavailable",
            "name": value.name,
            "reason_code": value.reason_code,
            "detail": value.detail,
            "evaluator_manifest_hash": value.evaluator_manifest_hash,
            "status": status.value if hasattr(status, "value") else str(status),
        }
    raise SmokeError("typed metric records are required")

def _jsonable_raw(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_raw(item) for item in value]
    return value
