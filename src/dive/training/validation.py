
from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager

from dive.data.mixture import FamilyBatch

from dive.training.stages import contract_for
from dive.training.module import (
    MAX_LEAKAGE_PROBE_GAIN,
    HealthVerdict,
    equal_family_relative_loss,
)

ABSENT_IN = {"bb_ca": "without_backbone", "local_latents": "without_latent"}

_PROBE_EXCLUDED_PREFIX = "t."

class ValidationError(RuntimeError):
    pass

@contextmanager
def frozen_base(module):

    import torch

    saved = {}
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "lora_B" in name or "presence" in name:
                saved[name] = parameter.detach().clone()
                parameter.zero_()
    try:
        yield
    finally:
        lookup = dict(module.named_parameters())
        with torch.no_grad():
            for name, value in saved.items():
                lookup[name].copy_(value)

def validate(
    module,
    loaders: Mapping[str, object],
    *,
    corrupt,
    device,
    max_batches: int = 16,
    probe_batches: int = 2,
    seed: int = 0,
) -> dict:

    import torch

    routed: dict[str, float] = {}
    joint: dict[str, float] = {}
    base: dict[str, float] = {}
    counts: dict[str, int] = {}
    probe_totals: dict[str, list[float]] = {m: [] for m in ABSENT_IN}
    non_finite = 0

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            for family, loader in sorted(loaders.items()):
                totals = [0.0, 0.0, 0.0]
                seen = 0
                for batch in loader:
                    if batch is None:
                        continue
                    if seen >= max_batches:
                        break
                    model_batch = corrupt(_to_device(batch, device))

                    joint_loss = float(module.loss_at_corner(model_batch, "joint"))
                    corner = contract_for(module.stage).fixed_corner
                    routed_loss = (
                        joint_loss
                        if corner == "joint"
                        else float(
                            module.training_step(
                                FamilyBatch(
                                    family=family,
                                    example_ids=("validation",),
                                    model_batch=model_batch,
                                ),
                                -1,
                            )
                        )
                    )
                    with frozen_base(module):
                        reference = float(module.loss_at_corner(model_batch, "joint"))

                    if seen < probe_batches:
                        for modality in ABSENT_IN:
                            probe_totals[modality].append(
                                leakage_gain(
                                    module, model_batch, modality, seed=seed + seen
                                )
                            )

                    totals[0] += routed_loss
                    totals[1] += joint_loss
                    totals[2] += reference
                    non_finite += sum(
                        1
                        for value in (routed_loss, joint_loss, reference)
                        if not math.isfinite(value)
                    )
                    seen += 1

                if seen == 0:
                    raise ValidationError(
                        f"{family} produced no usable validation batch; an "
                        f"equal-family score with a family missing is not the "
                        f"declared metric"
                    )
                routed[family] = totals[0] / seen
                joint[family] = totals[1] / seen
                base[family] = totals[2] / seen
                counts[family] = seen
    finally:
        module.train(was_training)

    gains = {
        modality: (sum(values) / len(values) if values else float("nan"))
        for modality, values in probe_totals.items()
    }
    return {
        "routed_val_loss": routed,
        "joint_val_loss": joint,
        "frozen_base_val_loss": base,
        "batches": counts,
        "non_finite": non_finite,
        "equal_family_relative_val_loss": equal_family_relative_loss(routed, base),
        "joint_relative_val_loss": equal_family_relative_loss(joint, base),
        "leakage_probe_gains": gains,
    }

def leakage_gain(module, model_batch: dict, modality: str, *, seed: int = 0) -> float:

    import torch

    if modality not in ABSENT_IN:
        raise ValidationError(f"no ablating condition for modality {modality!r}")
    null_condition = ABSENT_IN[modality]

    perturbed = _resample_modality(model_batch, modality, seed=seed)
    if perturbed is None:
        raise ValidationError(
            f"batch carries no tensor for modality {modality!r}; the probe would "
            f"report a perfect zero having perturbed nothing"
        )

    with torch.no_grad():
        before = module.condition_velocities(model_batch)
        after = module.condition_velocities(perturbed)

    moved = lambda name: float(
        (after[name][modality] - before[name][modality]).abs().sum()
    )
    denominator = moved("joint")
    if denominator == 0.0:
        return float("inf")
    return moved(null_condition) / denominator

def health_verdict(report: Mapping, *, consecutive_finite_steps: int) -> HealthVerdict:

    return HealthVerdict.from_record(
        {
            "joint_relative_val_loss": report["joint_relative_val_loss"],
            "consecutive_finite_steps": consecutive_finite_steps,
            "leakage_probe_gains": dict(report["leakage_probe_gains"]),
        }
    )

def reduce_across_ranks(report: dict) -> dict:

    import torch

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return report

    world = torch.distributed.get_world_size()
    merged = dict(report)
    for key in ("routed_val_loss", "joint_val_loss", "frozen_base_val_loss"):
        merged[key] = {
            family: _all_reduce_mean(value, world)
            for family, value in sorted(report[key].items())
        }
    merged["leakage_probe_gains"] = {
        modality: _all_reduce_mean(value, world)
        for modality, value in sorted(report["leakage_probe_gains"].items())
    }
    merged["equal_family_relative_val_loss"] = equal_family_relative_loss(
        merged["routed_val_loss"], merged["frozen_base_val_loss"]
    )
    merged["joint_relative_val_loss"] = equal_family_relative_loss(
        merged["joint_val_loss"], merged["frozen_base_val_loss"]
    )
    merged["world_size"] = world
    return merged

def _all_reduce_mean(value: float, world: int) -> float:
    import torch

    tensor = torch.tensor([float(value)], dtype=torch.float64)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item()) / world

def _resample_modality(model_batch: dict, modality: str, *, seed: int):

    import torch

    from dive.counterfactual.registry import flatten_paths

    generator = torch.Generator(device="cpu").manual_seed(seed)
    targets = [
        path
        for path, leaf in flatten_paths(model_batch).items()
        if torch.is_tensor(leaf)
        and leaf.is_floating_point()
        and not path.startswith(_PROBE_EXCLUDED_PREFIX)
        and (path == modality or path.endswith(f".{modality}"))
    ]
    if not targets:
        return None

    perturbed = _deep_copy(model_batch)
    for path in targets:
        node, key = _resolve(perturbed, path)
        original = node[key]
        node[key] = torch.randn(
            original.shape, generator=generator, dtype=torch.float32
        ).to(device=original.device, dtype=original.dtype)
    return perturbed

def _deep_copy(node):

    if isinstance(node, dict):
        return {key: _deep_copy(value) for key, value in node.items()}
    return node

def _resolve(batch: dict, path: str):
    node = batch
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    return node, parts[-1]

def _to_device(batch: Mapping, device) -> dict:
    import torch

    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }

__all__ = [
    "ABSENT_IN",
    "MAX_LEAKAGE_PROBE_GAIN",
    "ValidationError",
    "frozen_base",
    "health_verdict",
    "leakage_gain",
    "reduce_across_ranks",
    "validate",
]
