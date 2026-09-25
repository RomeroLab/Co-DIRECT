
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Generator, Tensor

from dive.counterfactual.registry import FeatureClass, FeatureRegistry

CONDITION_NAMES: tuple[str, ...] = (
    "joint",
    "without_latent",
    "without_backbone",
    "sham_latent",
    "sham_backbone",
)

_LATENT_CONDITIONS = frozenset({"without_latent", "sham_latent"})
_BACKBONE_CONDITIONS = frozenset({"without_backbone", "sham_backbone"})
_SHAM_CONDITIONS = frozenset({"sham_latent", "sham_backbone"})

class BuilderError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class Conditions:

    batches: Mapping[str, dict]
    permutation: Tensor
    metadata: Mapping[str, str]

def clone_tree(node: object) -> object:

    if isinstance(node, Tensor):
        return node.clone()
    if isinstance(node, Mapping):
        return {key: clone_tree(value) for key, value in node.items()}
    if isinstance(node, list):
        return [clone_tree(value) for value in node]
    if isinstance(node, tuple):
        return tuple(clone_tree(value) for value in node)
    return node

def tree_sha256(node: object) -> str:

    digest = hashlib.sha256()
    _absorb(node, prefix="", digest=digest)
    return digest.hexdigest()

def _absorb(node: object, prefix: str, digest: "hashlib._Hash") -> None:

    if isinstance(node, Mapping):
        for key in sorted(node, key=str):
            _absorb(node[key], f"{prefix}.{key}", digest)
        return
    if isinstance(node, (list, tuple)):
        for index, child in enumerate(node):
            _absorb(child, f"{prefix}.{index}", digest)
        return
    digest.update(prefix.encode())
    if isinstance(node, Tensor):
        digest.update(str(tuple(node.shape)).encode())
        digest.update(str(node.dtype).encode())

        digest.update(
            node.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    else:
        digest.update(repr(node).encode())

def build_conditions(
    source: Mapping,
    registry: FeatureRegistry,
    recompute_pair_features: Callable[[dict], dict],
    *,
    generator: Generator,
) -> Conditions:

    return _build_conditions(
        source,
        CONDITION_NAMES,
        registry,
        recompute_pair_features,
        generator=generator,
    )

def _build_conditions(
    source: Mapping,
    names: tuple[str, ...],
    registry: FeatureRegistry,
    recompute_pair_features: Callable[[dict], dict],
    *,
    generator: Generator,
) -> Conditions:

    unknown = [name for name in names if name not in CONDITION_NAMES]
    if unknown:
        raise BuilderError(f"unknown condition(s): {unknown}")
    if not names or names[0] != "joint":
        raise BuilderError(f"'joint' must be built first, got {names}")

    registry.assert_complete(source)
    registry.assert_denoiser_safe(source)

    valid = _valid_residues(source)
    permutation, identity_permutations = _sham_permutation(valid, generator)

    latent_paths = registry.paths_of(FeatureClass.LATENT_DERIVED)
    backbone_paths = registry.paths_of(FeatureClass.BACKBONE_DERIVED)

    per_condition_metadata: dict[str, str] = {}
    batches: dict[str, dict] = {}
    joint_pre_recompute: dict[str, Tensor] = {}
    joint_delivered: dict = {}
    for name in names:
        sibling = clone_tree(source)

        padded_paths: set[str] = set()
        padded_paths |= _zero_padding_for_paths(sibling, latent_paths, valid)
        padded_paths |= _zero_padding_for_paths(sibling, backbone_paths, valid)

        permuted_paths: set[str] = set()
        if name in _LATENT_CONDITIONS:
            _apply(sibling, latent_paths, name, valid, permutation, permuted_paths)
        if name in _BACKBONE_CONDITIONS:
            _apply(sibling, backbone_paths, name, valid, permutation, permuted_paths)

        pre_recompute = _snapshot(sibling, padded_paths | permuted_paths)
        if name == "joint":
            joint_pre_recompute = pre_recompute

        sibling = recompute_pair_features(sibling)

        _assert_no_ablation_leak(sibling, name, registry)
        _assert_padding_preserved(sibling, name, padded_paths, pre_recompute, valid)

        if name != "joint":

            changed_from_joint = {
                path
                for path, value in pre_recompute.items()
                if path in joint_pre_recompute
                and not torch.equal(value, joint_pre_recompute[path])
            }
            _assert_perturbation_survived(sibling, name, changed_from_joint, joint_delivered)

        if name in _SHAM_CONDITIONS:

            delivered = _unchanged_since(sibling, permuted_paths, pre_recompute)
            per_condition_metadata[f"permuted_paths.{name}"] = ",".join(sorted(delivered))

        batches[name] = sibling
        if name == "joint":
            joint_delivered = sibling

    return Conditions(
        batches=batches,
        permutation=permutation,
        metadata={
            "latent_null": "naive_null_debug",
            "backbone_null": "zero_coordinates",
            "sham": "within_example_valid_residue_permutation",
            "registry_sha256": registry.sha256(),
            "identity_permutations": str(identity_permutations),
            **per_condition_metadata,
        },
    )

def _valid_residues(source: Mapping) -> Tensor:

    mask = source.get("mask")
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
        raise BuilderError("source batch must carry a boolean 'mask'")
    if mask.dim() != 2:
        raise BuilderError(
            f"source 'mask' must be 2-D (batch, residues), got shape {tuple(mask.shape)}"
        )
    if not bool(mask.any()):
        raise BuilderError("source batch has no valid residue to permute")
    return mask

def _sham_permutation(valid: Tensor, generator: Generator) -> tuple[Tensor, int]:

    batch_size, residues = valid.shape

    permutation = torch.arange(residues, device=valid.device).unsqueeze(0).repeat(batch_size, 1)
    identity_count = 0
    for example in range(batch_size):
        positions = torch.nonzero(valid[example], as_tuple=False).flatten()
        if positions.numel() < 2:
            identity_count += 1
            continue

        draw = torch.randperm(positions.numel(), generator=generator).to(positions.device)
        shuffled = positions[draw]
        if torch.equal(shuffled, positions):
            identity_count += 1
        permutation[example, positions] = shuffled
    return permutation, identity_count

def _apply(
    sibling: dict,
    paths: tuple[str, ...],
    condition: str,
    valid: Tensor,
    permutation: Tensor,
    permuted_paths: set[str],
) -> None:

    for path in paths:
        holder, key = _resolve(sibling, path)
        if holder is None:
            continue
        tensor = holder[key]
        if not isinstance(tensor, Tensor):
            continue
        if condition in _SHAM_CONDITIONS:
            holder[key] = _permute_rows(tensor, permutation, path)
            permuted_paths.add(path)
        else:
            holder[key] = torch.zeros_like(tensor)
        _zero_padding(holder, key, valid)

def _resolve(sibling: dict, path: str) -> tuple[dict | None, str]:

    node: object = sibling
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(node, Mapping) or part not in node:
            return None, parts[-1]
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return None, parts[-1]
    return node, parts[-1]

def _permute_rows(tensor: Tensor, permutation: Tensor, path: str) -> Tensor:

    if tensor.dim() < 2 or tensor.shape[:2] != permutation.shape:
        raise BuilderError(
            f"cannot permute {path!r} for a sham condition: tensor shape "
            f"{tuple(tensor.shape)} does not match permutation shape "
            f"{tuple(permutation.shape)}"
        )
    index = permutation
    while index.dim() < tensor.dim():
        index = index.unsqueeze(-1)
    return torch.gather(tensor, 1, index.expand_as(tensor))

def _zero_padding(holder: dict, key: str, valid: Tensor) -> bool:

    tensor = holder[key]
    if tensor.dim() >= 2 and tensor.shape[:2] == valid.shape:
        expanded = valid
        while expanded.dim() < tensor.dim():
            expanded = expanded.unsqueeze(-1)
        holder[key] = torch.where(expanded, tensor, torch.zeros_like(tensor))
        return True
    return False

def _zero_padding_for_paths(sibling: dict, paths: tuple[str, ...], valid: Tensor) -> set[str]:

    touched: set[str] = set()
    for path in paths:
        holder, key = _resolve(sibling, path)
        if holder is None:
            continue
        if not isinstance(holder[key], Tensor):
            continue
        if _zero_padding(holder, key, valid):
            touched.add(path)
    return touched

def _snapshot(sibling: dict, paths: set[str]) -> dict[str, Tensor]:

    snapshot: dict[str, Tensor] = {}
    for path in paths:
        holder, key = _resolve(sibling, path)
        if holder is None:
            continue
        tensor = holder[key]
        if isinstance(tensor, Tensor):
            snapshot[path] = tensor.clone()
    return snapshot

def _unchanged_since(sibling: dict, paths: set[str], snapshot: dict[str, Tensor]) -> set[str]:

    unchanged: set[str] = set()
    for path in paths:
        holder, key = _resolve(sibling, path)
        if holder is None or path not in snapshot:
            continue
        tensor = holder[key]
        if isinstance(tensor, Tensor) and torch.equal(tensor, snapshot[path]):
            unchanged.add(path)
    return unchanged

def _assert_no_ablation_leak(sibling: dict, name: str, registry: FeatureRegistry) -> None:

    if name == "without_latent":
        _assert_all_zero(sibling, registry.paths_of(FeatureClass.LATENT_DERIVED), name)
    if name == "without_backbone":
        _assert_all_zero(sibling, registry.paths_of(FeatureClass.BACKBONE_DERIVED), name)

def _assert_all_zero(sibling: dict, paths: tuple[str, ...], name: str) -> None:

    for path in paths:
        holder, key = _resolve(sibling, path)
        if holder is None:
            continue
        tensor = holder[key]
        if isinstance(tensor, Tensor) and torch.count_nonzero(tensor) != 0:
            raise BuilderError(
                f"recompute_pair_features reintroduced non-zero data into "
                f"{path!r} for condition {name!r}; the ablated modality must "
                "stay zeroed"
            )

def _assert_padding_preserved(
    sibling: dict,
    name: str,
    padded_paths: set[str],
    pre_recompute: dict[str, Tensor],
    valid: Tensor,
) -> None:

    padded = ~valid
    for path in padded_paths:
        holder, key = _resolve(sibling, path)
        if holder is None:
            continue
        tensor = holder[key]
        if not isinstance(tensor, Tensor):
            continue
        if path not in pre_recompute or not torch.equal(tensor, pre_recompute[path]):
            continue
        expanded = padded
        while expanded.dim() < tensor.dim():
            expanded = expanded.unsqueeze(-1)
        leaked = torch.where(expanded, tensor, torch.zeros_like(tensor))
        if torch.count_nonzero(leaked) != 0:
            raise BuilderError(f"padded rows of {path!r} are not zero for condition {name!r}")

def _assert_perturbation_survived(
    sibling: dict, name: str, changed_paths: set[str], joint_delivered: dict
) -> None:

    for path in changed_paths:
        holder, key = _resolve(sibling, path)
        joint_holder, joint_key = _resolve(joint_delivered, path)
        if holder is None or joint_holder is None:
            continue
        tensor = holder[key]
        joint_tensor = joint_holder[joint_key]
        if not isinstance(tensor, Tensor) or not isinstance(joint_tensor, Tensor):
            continue
        if tensor.shape != joint_tensor.shape:
            continue
        if torch.equal(tensor, joint_tensor):
            raise BuilderError(
                f"{path!r} for condition {name!r} is identical to joint's "
                "delivered value, even though the builder perturbed it away "
                "from joint before recompute_pair_features ran; the "
                "callback appears to have erased that perturbation"
            )
