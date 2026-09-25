
from __future__ import annotations

import hashlib
import math
import pickle
import random
import struct
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from dive.counterfactual.builder import clone_tree, tree_sha256
from dive.counterfactual.registry import (
    CleanFeatureLeakageError,
    FeatureClass,
    FeatureRegistry,
    UnknownFeaturePathError,
    flatten_paths,
)
from dive.integrations.complexa import ComplexaAdapter, UpstreamContractError

class BasisError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class BasisContext:

    real_fn: Callable[[dict, str, int], Mapping]
    model: object
    batch: Mapping
    joint_output: Mapping
    adapter: ComplexaAdapter
    registry: FeatureRegistry
    run_id: str
    target: str
    generation_seed: int
    checkpoint_index: int
    mode: str = "full"
    n_recycle: int = 0

@dataclass(frozen=True, slots=True)
class BasisResult:

    bases: tuple[Tensor, Tensor]
    reference_velocity: Tensor
    prior_velocities: tuple[Tensor, Tensor]
    prior_seeds: tuple[int, int]
    source_hash: str
    rng_before: Mapping[str, str]
    rng_after: Mapping[str, str]
    mc_ratio: float
    transport_ratio: float
    additional_denoiser_calls: int
    sampler_calls: tuple[str, str]
    active: bool
    joint_fallback: Mapping

def canonical_prior_seed(
    run_id: str,
    target: str,
    generation_seed: int,
    checkpoint_index: int,
    prior_index: int,
) -> int:

    if (
        type(run_id) is not str
        or type(target) is not str
        or any(
            type(value) is not int
            for value in (generation_seed, checkpoint_index, prior_index)
        )
    ):
        raise BasisError("invalid canonical prior seed fields")
    try:
        run = unicodedata.normalize("NFC", run_id).encode("utf-8")
        name = unicodedata.normalize("NFC", target).encode("utf-8")
        payload = b"signed-value-prior-v1\0"
        payload += struct.pack(">I", len(run)) + run
        payload += struct.pack(">I", len(name)) + name
        payload += struct.pack(">qII", generation_seed, checkpoint_index, prior_index)
    except (TypeError, UnicodeError, struct.error) as error:
        raise BasisError("invalid canonical prior seed fields") from error
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)

def snapshot_rng_state() -> dict[str, str]:

    return {
        "torch_cpu": _tensor_sha256(torch.random.get_rng_state()),
        "torch_cuda_0": _tensor_sha256(torch.cuda.get_rng_state(0)),
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
    }

def build_x_to_z_basis(context: BasisContext) -> BasisResult:

    _assert_fixed_call(context)
    raw_rng = _capture_rng_state()
    rng_hashes = snapshot_rng_state()
    try:
        return _build_x_to_z_basis(context)
    except BaseException:
        if snapshot_rng_state() != rng_hashes:
            _restore_rng_state(raw_rng)
        raise

def _build_x_to_z_basis(context: BasisContext) -> BasisResult:

    _assert_source_contract(context)
    source_hash = tree_sha256(context.batch)
    rng_before = snapshot_rng_state()

    joint = _extract_checked(
        context.adapter,
        context.joint_output,
        label="joint",
        expected=context.batch["x_t"],
    )

    reference_batch = _branch(context.batch, context.registry, prior=False)
    reference_output = _call_branch(context, reference_batch, label="reference")
    reference = _extract_checked(
        context.adapter,
        reference_output,
        label="reference",
        expected=joint,
    )

    prior_velocities: list[Tensor] = []
    prior_seeds: list[int] = []
    for prior_index in (0, 1):
        seed = canonical_prior_seed(
            context.run_id,
            context.target,
            context.generation_seed,
            context.checkpoint_index,
            prior_index,
        )
        prior_batch = _branch(context.batch, context.registry, prior=True)
        sampled = _sample_isolated(context, prior_batch, seed)
        if not bool(torch.isfinite(sampled).all()):
            raise BasisError("bb_ca prior contains non-finite values")
        prior_batch["x_t"]["bb_ca"] = sampled
        prior_batch["t"]["bb_ca"] = torch.zeros_like(prior_batch["t"]["bb_ca"])
        try:
            prior_batch = context.adapter.recompute_pair_features(prior_batch)
        except UpstreamContractError as error:
            raise BasisError(str(error)) from error
        _assert_branch_identity(
            context.batch, prior_batch, context.registry, prior=True
        )
        prior_output = _call_branch(context, prior_batch, label=f"prior {prior_index}")
        velocities = _extract_checked(
            context.adapter,
            prior_output,
            label=f"prior {prior_index}",
            expected=joint,
        )
        prior_velocities.append(velocities["local_latents"])
        prior_seeds.append(seed)

    if tree_sha256(context.batch) != source_hash:
        raise BasisError("basis construction mutated the source batch")
    rng_after = snapshot_rng_state()
    if rng_after != rng_before:
        raise BasisError("basis construction changed ambient RNG state")

    v_ref = reference["local_latents"]
    v_prior = (prior_velocities[0], prior_velocities[1])
    bases = (v_ref - v_prior[0], v_ref - v_prior[1])
    mask = context.batch["mask"]
    mc_ratio = _masked_rms(bases[0] - bases[1], mask) / (
        _masked_rms(bases[0] + bases[1], mask) + 1e-12
    )
    transport_ratio = _masked_rms(joint["local_latents"] - v_ref, mask) / (
        _masked_rms(joint["local_latents"], mask) + 1e-12
    )
    if not math.isfinite(mc_ratio) or not math.isfinite(transport_ratio):
        raise BasisError("basis ratio is non-finite")

    return BasisResult(
        bases=bases,
        reference_velocity=v_ref,
        prior_velocities=v_prior,
        prior_seeds=(prior_seeds[0], prior_seeds[1]),
        source_hash=source_hash,
        rng_before=rng_before,
        rng_after=rng_after,
        mc_ratio=mc_ratio,
        transport_ratio=transport_ratio,
        additional_denoiser_calls=3,
        sampler_calls=("bb_ca", "bb_ca"),
        active=mc_ratio <= 1.0,
        joint_fallback=context.joint_output,
    )

def _assert_fixed_call(context: BasisContext) -> None:
    if context.mode != "full":
        raise BasisError("basis requires production mode='full'")
    if isinstance(context.n_recycle, bool) or context.n_recycle != 0:
        raise BasisError("basis requires generator n_recycle=0")
    if isinstance(context.checkpoint_index, bool) or context.checkpoint_index != 200:
        raise BasisError("basis is restricted to production checkpoint 200")
    if dict(context.adapter.output_parameterization) != {
        "bb_ca": "v",
        "local_latents": "v",
    }:
        raise BasisError("both modalities require velocity parameterization 'v'")
    if torch.cuda.device_count() != 1:
        raise BasisError("basis requires exactly one visible CUDA device")

def _assert_source_contract(context: BasisContext) -> None:
    source = context.batch
    if "x_recycle" in source:
        raise BasisError("generated x_recycle paths are forbidden")
    if "x_sc" not in source:
        raise BasisError("x_sc is required by the exact 403-call feasibility contract")
    x_sc = source["x_sc"]
    if not isinstance(x_sc, Mapping) or set(x_sc) != {"bb_ca", "local_latents"}:
        raise BasisError("basis requires the complete production x_sc mapping")
    try:
        context.registry.assert_complete(source)
        context.registry.assert_denoiser_safe(source)
    except (UnknownFeaturePathError, CleanFeatureLeakageError) as error:
        raise BasisError(str(error)) from error
    x_t = source.get("x_t")
    if not isinstance(x_t, Mapping) or set(x_t) != {"bb_ca", "local_latents"}:
        raise BasisError("basis requires exactly both x_t modalities")
    for modality in ("bb_ca", "local_latents"):
        current = x_t[modality]
        self_conditioning = x_sc[modality]
        if not isinstance(current, Tensor) or not current.is_floating_point():
            raise BasisError(f"x_t.{modality} must be a floating tensor")
        if (
            not isinstance(self_conditioning, Tensor)
            or self_conditioning.shape != current.shape
            or self_conditioning.dtype != current.dtype
            or self_conditioning.device != current.device
        ):
            raise BasisError(f"complete production x_sc.{modality} does not match x_t")
        if not bool(
            torch.isfinite(current).all() and torch.isfinite(self_conditioning).all()
        ):
            raise BasisError(f"source {modality} contains non-finite values")
    mask = source.get("mask")
    if (
        not isinstance(mask, Tensor)
        or mask.dtype is not torch.bool
        or mask.shape != x_t["bb_ca"].shape[:2]
        or not bool(mask.any())
    ):
        raise BasisError("basis requires a non-empty boolean binder mask")
    times = source.get("t")
    if not isinstance(times, Mapping) or set(times) != {"bb_ca", "local_latents"}:
        raise BasisError("basis requires separate t.bb_ca and t.local_latents clocks")
    for modality in ("bb_ca", "local_latents"):
        value = times[modality]
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise BasisError(f"t.{modality} must be a finite floating tensor")

def _branch(source: Mapping, registry: FeatureRegistry, *, prior: bool) -> dict:
    branch = clone_tree(source)
    if not isinstance(branch, dict):
        raise BasisError("cloned batch is not a mutable mapping")
    branch.pop("x_sc")
    _assert_branch_identity(source, branch, registry, prior=prior)
    return branch

def _assert_branch_identity(
    source: Mapping, branch: Mapping, registry: FeatureRegistry, *, prior: bool
) -> None:
    if "x_sc" in branch:
        raise BasisError("basis branches must remove the entire x_sc mapping")
    try:
        registry.assert_complete(branch)
        registry.assert_denoiser_safe(branch)
    except (UnknownFeaturePathError, CleanFeatureLeakageError) as error:
        raise BasisError(str(error)) from error
    allowed = {path for path in flatten_paths(source) if path.startswith("x_sc.")}
    if prior:
        allowed.add("t.bb_ca")
        allowed.update(registry.paths_of(FeatureClass.BACKBONE_DERIVED))
    source_flat = flatten_paths(source)
    branch_flat = flatten_paths(branch)
    for path, source_value in source_flat.items():
        if path in allowed:
            continue
        if path not in branch_flat or _value_sha256(branch_flat[path]) != _value_sha256(
            source_value
        ):
            raise BasisError(
                f"basis branch changed preserved/conditioning path '{path}'"
            )

def _sample_isolated(context: BasisContext, batch: Mapping, seed: int) -> Tensor:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    python_before = hashlib.sha256(pickle.dumps(python_state)).hexdigest()
    numpy_before = hashlib.sha256(pickle.dumps(numpy_state)).hexdigest()
    changed = False
    try:
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            try:
                sampled = context.adapter.sample_backbone_prior(context.model, batch)
            except (
                AttributeError,
                KeyError,
                TypeError,
                UpstreamContractError,
            ) as error:
                raise BasisError(
                    f"bb_ca prior sampler contract failed: {error}"
                ) from error
        changed = (
            hashlib.sha256(pickle.dumps(random.getstate())).hexdigest() != python_before
            or hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()
            != numpy_before
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    if changed:
        raise BasisError("bb_ca prior sampler changed Python/NumPy RNG state")
    return sampled

def _call_branch(context: BasisContext, batch: dict, *, label: str) -> Mapping:
    before = tree_sha256(batch)
    output = context.real_fn(batch, "full", 0)
    if tree_sha256(batch) != before:
        raise BasisError(f"denoiser mutated the {label} branch")
    if not isinstance(output, Mapping):
        raise BasisError(f"{label} denoiser output is not a mapping")
    return output

def _extract_checked(
    adapter: ComplexaAdapter,
    output: Mapping,
    *,
    label: str,
    expected: Mapping[str, Any],
) -> dict[str, Tensor]:
    try:
        velocities = adapter.extract_velocities(output)
    except UpstreamContractError as error:
        raise BasisError(f"{label} velocity contract failed: {error}") from error
    for modality in ("bb_ca", "local_latents"):
        value = velocities[modality]
        reference = expected[modality]
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise BasisError(f"{label} {modality} velocity is not floating")
        if not isinstance(reference, Tensor) or (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
        ):
            raise BasisError(f"{label} {modality} velocity shape/dtype/device drift")
        if not bool(torch.isfinite(value).all()):
            raise BasisError(f"{label} {modality} velocity contains non-finite values")
    return velocities

def _masked_rms(value: Tensor, mask: Tensor) -> float:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    try:
        selected = value[expanded.expand_as(value)]
    except RuntimeError as error:
        raise BasisError("binder mask cannot expand to basis shape") from error
    if selected.numel() == 0:
        raise BasisError("basis RMS requires a non-empty binder mask")
    result = torch.sqrt(torch.mean(selected.to(torch.float64).square())).item()
    if not math.isfinite(result):
        raise BasisError("basis RMS is non-finite")
    return result

def _tensor_sha256(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    ).hexdigest()

def _capture_rng_state() -> dict[str, object]:
    return {
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda_0": torch.cuda.get_rng_state(0).clone(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }

def _restore_rng_state(state: Mapping[str, object]) -> None:
    torch.random.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda_0"], 0)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

def _value_sha256(value: object) -> str:
    return tree_sha256({"value": value})
