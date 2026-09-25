
from __future__ import annotations

import os
import time
from pathlib import Path

from dive.benchmark.paper_baseline import (
    PaperArm,
    PaperBaselineError,
    claim_paper_baseline_run,
    complete_paper_baseline_run,
    family_steps,
)
from dive.benchmark.paper_generation import build_lora_free_v2_model, hash_splice_report
from dive.signed_value.roots import EMERGENT_BULK_ROOT, EMERGENT_EVIDENCE_ROOT

def assert_single_visible_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        raise PaperBaselineError("CUDA_VISIBLE_DEVICES is required")
    parts = [part.strip() for part in visible.split(",") if part.strip()]
    if len(parts) != 1:
        raise PaperBaselineError("exactly one GPU must be visible")
    return parts[0]

def run_gpu_smoke(
    *,
    run_id: str,
    repo_commit: str,
    generate: bool,
    argv: tuple[str, ...],
) -> dict[str, object]:
    visible = assert_single_visible_gpu()
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PaperBaselineError("smoke requires exactly one CUDA device")
    from dive.emergent_contract import EmergentContract

    binder_steps = family_steps(
        "binder", panel="A", arm=PaperArm.BASE_COMMON_EXACT_V1
    )
    run = claim_paper_baseline_run(
        run_id,
        argv,
        evidence_root=EMERGENT_EVIDENCE_ROOT,
        bulk_root=EMERGENT_BULK_ROOT,
        repo_commit=repo_commit,
    )
    store_dir = run.bulk_dir / "store"
    store_dir.mkdir(mode=0o2775)
    payload: dict[str, object] = {
        "stage": "starting",
        "visible_gpu": visible,
        "binder_steps": binder_steps,
        "generate": generate,
    }
    try:
        contract = EmergentContract.from_yaml(Path("configs/emergent/resources.yaml"))
        started = time.perf_counter()
        model, splice = build_lora_free_v2_model(contract, store_dir=store_dir)
        model = model.to("cuda").eval()
        payload.update(
            {
                "stage": "load",
                "cuda_name": torch.cuda.get_device_name(0),
                "load_seconds": time.perf_counter() - started,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "splice_report_sha256": hash_splice_report(splice),
                "missing_keys": list(splice.missing_keys),
                "unexpected_keys": list(splice.unexpected_keys),
                "resized_keys": [list(item) for item in splice.resized_keys],
                "live_parameter_sha256": splice.live_parameter_sha256,
            }
        )
        if generate:
            gen_started = time.perf_counter()
            _configure_inference(model, steps=binder_steps)
            batch = _dummy_batch(model)
            with torch.inference_mode():
                generated = model.generate(batch)
            payload["generate_seconds"] = time.perf_counter() - gen_started
            payload["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            payload["generated_type"] = type(generated).__name__
            payload["stage"] = "generate"
        complete_paper_baseline_run(run, payload)
    except Exception as error:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
        try:
            complete_paper_baseline_run(run, payload)
        except Exception:
            pass
        raise
    payload["run_id"] = run.run_id
    payload["evidence_dir"] = str(run.evidence_dir)
    return payload

def _configure_inference(model, *, steps: int) -> None:
    from copy import deepcopy

    from omegaconf import open_dict

    generation = deepcopy(model.cfg_exp.generation)
    with open_dict(generation):
        generation.args.nsteps = steps
        generation.args.guidance_w = 1.0
        generation.args.ag_ratio = 0.0
        generation.args.self_cond = False
        generation.n_recycle = 0
        generation.model = deepcopy(generation.model.ode)
    model.configure_inference(generation, nn_ag=None)

def _dummy_batch(model) -> dict[str, object]:
    import torch

    device = next(model.parameters()).device
    generated = 32
    target = 8
    mask = torch.ones(1, generated, dtype=torch.bool, device=device)
    x_target = torch.zeros(1, target, 37, 3, device=device)
    target_mask = torch.zeros(1, target, 37, dtype=torch.bool, device=device)
    target_mask[:, :, :4] = True
    zeros_target = torch.zeros(1, target, dtype=torch.bool, device=device)
    return {
        "mask": mask,
        "x_target": x_target,
        "target_mask": target_mask,
        "seq_target": torch.zeros(1, target, dtype=torch.long, device=device),
        "seq_target_mask": torch.ones(1, target, dtype=torch.bool, device=device),
        "target_chains": torch.zeros(1, target, dtype=torch.long, device=device),
        "target_hotspot_mask": zeros_target,
        "hotspot_mask": torch.zeros(1, generated, dtype=torch.bool, device=device),
        "chains": torch.ones(1, generated, dtype=torch.long, device=device),
        "prepend_target": False,
        "atomistic_target": False,
        "fixed_sequence_mask": torch.zeros(
            1, generated, dtype=torch.bool, device=device
        ),
        "fixed_structure_mask": torch.zeros(
            1, generated, dtype=torch.bool, device=device
        ),
        "fixed_sequence_target_mask": zeros_target,
        "fixed_structure_target_mask": zeros_target,
    }
