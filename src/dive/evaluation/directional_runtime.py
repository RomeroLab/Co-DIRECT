
from __future__ import annotations

from dive.codirect_paths import joined

import ast
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from enum import Enum
from functools import partial
import gzip
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import io
import json
import marshal
import math
import os
from pathlib import Path
import platform
import random
import stat
import struct
import sys
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CodeType,
    FunctionType,
    MappingProxyType,
    MethodType,
    ModuleType,
)
import numpy as np
import torch
from loguru import logger as _LOGURU_LOGGER
from torch import Tensor, nn

from dive.evaluation.directional_catalog import PROJECT_PREFIXES
from dive.evaluation.directional_checks import CHECK_SCHEMA_VERSION
from dive.evaluation.directional_confirmation import TIME_BIN_EDGES, TIME_BIN_SCHEME
from dive.evaluation.directional_seal import (
    BlindEvaluationData,
    FlowIdentity,
    FlowRow,
    GenerationCell,
    SealError,
    VerifiedBlindInputs,
    _validate_strict_blind_loader_manifest,
)
from dive.integrations.directional_v2 import DirectionalForward, DirectionalV2Adapter
from dive.integrations.flow import (
    DirectionalProteinaBindings,
    directional_proteina_bindings,
)
from dive.leadership.directional_bridge import RouteState
from dive.signed_value.roots import (
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_COMMIT,
    EMERGENT_UPSTREAM_ROOT,
)

class RuntimeContractError(SealError):
    pass

EXECUTABLE_ROUTE_STATES = (
    RouteState.ADAPTIVE,
    RouteState.NONE,
    RouteState.Z_TO_X,
    RouteState.X_TO_Z,
    RouteState.BIDIRECTIONAL,
)

_MODALITIES = ("bb_ca", "local_latents")
_CLEAN_REQUIRED_KEYS = frozenset(
    {"coords_nm", "coords", "mask", "generated_mask", "fixed_mask"}
)
_CLEAN_OPTIONAL_TENSOR_KEYS = frozenset(
    {
        "dive_modality_presence",
        "initial",
        "cond",
        "pair",
        "concat",
        "concat_mask",
        "extended_pair",
        "target",
        "target_mask",
        "seq_target_mask",
        "x_target",
        "seq_target",
        "motif_mask",
        "x_motif",
        "seq",
        "residue_type",
        "chain_index",
        "chains",
        "chain_breaks_per_residue",
        "coord_mask",
        "residue_pdb_idx",
        "seq_pos",
        "target_chains",
        "target_hotspot_mask",
        "target_padding_mask",
        "target_pdb_idx",
        "target_residue_mask",
    }
)
_CLEAN_OPTIONAL_MAPPING_KEYS = frozenset({"mask_dict"})
_COLLATE_METADATA_KEYS = frozenset(
    {
        "chain_id",
        "chain_names",
        "context",
        "database",
        "generated",
        "id",
        "n_target",
        "nres",
        "nsamples",
        "residues",
        "target",
    }
)
_COLLATE_TENSOR_KEYS = frozenset(
    {
        "chain_breaks_per_residue",
        "chains",
        "coord_mask",
        "coords",
        "coords_nm",
        "design_mask",
        "mask",
        "motif_mask",
        "residue_pdb_idx",
        "residue_type",
        "seq_pos",
        "seq_target",
        "seq_target_mask",
        "target_chains",
        "target_hotspot_mask",
        "target_mask",
        "target_padding_mask",
        "target_pdb_idx",
        "target_residue_mask",
        "x_target",
    }
)
_COLLATE_KEYS_BY_FAMILY = MappingProxyType(
    {
        "binder": _COLLATE_METADATA_KEYS | _COLLATE_TENSOR_KEYS,
        "ame": _COLLATE_METADATA_KEYS | _COLLATE_TENSOR_KEYS,
        "antibody": _COLLATE_METADATA_KEYS | _COLLATE_TENSOR_KEYS,
    }
)
_CORRUPTION_KEYS = frozenset({"x_0", "x_t", "x_1", "t"})
_OUTCOME_NAMES = frozenset(
    {
        "aggregate",
        "blind_checks",
        "check",
        "counterfactual_validity",
        "flow_examples",
        "flow_loss",
        "joint_preservation",
        "loss",
        "metric",
        "outcome",
        "passed",
        "performance",
        "prediction",
        "result",
        "route_losses",
        "score",
        "statistics",
        "verdict",
    }
)

def _exact_mapping(name: str, value: object) -> Mapping[str, object]:
    if type(value) not in (dict, MappingProxyType):
        raise RuntimeContractError(f"{name} requires an exact built-in mapping")
    if any(type(key) is not str for key in value):
        raise RuntimeContractError(f"{name} keys must be exact strings")
    return value

def _exact_keys(name: str, value: Mapping[str, object], expected: set[str]) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise RuntimeContractError(
            f"{name} schema has missing {missing} and unknown {unknown}"
        )

def _reject_outcomes(name: str, value: object) -> None:
    if type(value) in (dict, MappingProxyType):
        mapping = _exact_mapping(name, value)
        illegal = sorted(set(mapping) & _OUTCOME_NAMES)
        if illegal:
            raise RuntimeContractError(f"{name} contains outcome fields {illegal}")
        for key, nested in mapping.items():
            _reject_outcomes(f"{name}.{key}", nested)
    elif type(value) in (tuple, list):
        for index, nested in enumerate(value):
            _reject_outcomes(f"{name}[{index}]", nested)
    elif callable(value) or isinstance(value, Path):
        raise RuntimeContractError(f"{name} contains an executable or path")

def _clone_tensor(value: object, *, name: str) -> Tensor:
    if type(value) is not Tensor:
        raise RuntimeContractError(f"{name} must be an exact Tensor")
    return value.detach().clone()

def _freeze_clean_batch(value: object) -> Mapping[str, object]:
    batch = _exact_mapping("clean batch", value)
    allowed = (
        _CLEAN_REQUIRED_KEYS
        | _CLEAN_OPTIONAL_TENSOR_KEYS
        | _CLEAN_OPTIONAL_MAPPING_KEYS
    )
    missing = sorted(_CLEAN_REQUIRED_KEYS - set(batch))
    unknown = sorted(set(batch) - allowed)
    if missing or unknown:
        raise RuntimeContractError(
            f"clean batch schema has missing {missing} and unknown {unknown}"
        )
    result: dict[str, object] = {}
    for key, raw in batch.items():
        if key == "mask_dict":
            nested = _exact_mapping("clean batch.mask_dict", raw)
            _exact_keys("clean batch.mask_dict", nested, {"coords"})
            result[key] = MappingProxyType(
                {"coords": _clone_tensor(nested["coords"], name="mask_dict.coords")}
            )
        else:
            result[key] = _clone_tensor(raw, name=f"clean batch.{key}")
    frozen = MappingProxyType(result)
    _validate_clean_shapes(frozen)
    _require_finite_tree("clean batch", frozen)
    return frozen

def _validate_clean_shapes(batch: Mapping[str, object]) -> None:
    coords_nm = batch["coords_nm"]
    coords = batch["coords"]
    assert isinstance(coords_nm, Tensor) and isinstance(coords, Tensor)
    if (
        coords_nm.ndim != 4
        or coords_nm.shape[0] != 1
        or coords_nm.shape[2] != 37
        or coords_nm.shape[-1] != 3
        or tuple(coords.shape) != tuple(coords_nm.shape)
        or coords.dtype != coords_nm.dtype
        or not torch.is_floating_point(coords_nm)
    ):
        raise RuntimeContractError(
            "clean batch coords_nm/coords must share finite [1,N,37,3] shape/dtype"
        )
    residues = int(coords_nm.shape[1])
    masks = []
    for key in ("mask", "generated_mask", "fixed_mask"):
        value = batch[key]
        assert isinstance(value, Tensor)
        if value.dtype is not torch.bool or tuple(value.shape) != (1, residues):
            raise RuntimeContractError(f"clean batch.{key} must be bool [1,N]")
        masks.append(value)
    mask, generated, fixed = masks
    if torch.any(generated & ~mask) or torch.any(fixed & ~mask):
        raise RuntimeContractError("generated/fixed masks must be subsets of mask")
    if torch.any(generated & fixed):
        raise RuntimeContractError("generated and fixed masks must be disjoint")
    if int((generated & mask).sum().item()) <= 0:
        raise RuntimeContractError("clean batch has no generated valid residue")
    if not torch.equal(generated | fixed, mask):
        raise RuntimeContractError("generated/fixed masks must exactly partition mask")
    initial = batch.get("initial")
    cond = batch.get("cond")
    for key, value in (("initial", initial), ("cond", cond)):
        if value is not None and (
            type(value) is not Tensor
            or value.ndim != 3
            or tuple(value.shape[:2]) != (1, residues)
            or value.shape[-1] <= 0
            or not torch.is_floating_point(value)
        ):
            raise RuntimeContractError(f"clean batch.{key} must be floating [1,N,D]")
    pair = batch.get("pair")
    if pair is not None and (
        type(pair) is not Tensor
        or pair.ndim != 4
        or tuple(pair.shape[:3]) != (1, residues, residues)
        or pair.shape[-1] <= 0
        or not torch.is_floating_point(pair)
    ):
        raise RuntimeContractError("clean batch.pair must be floating [1,N,N,D]")
    mask_dict = batch.get("mask_dict")
    if mask_dict is not None:
        nested = _exact_mapping("clean batch.mask_dict", mask_dict)
        coords_mask = nested["coords"]
        if (
            type(coords_mask) is not Tensor
            or coords_mask.dtype is not torch.bool
            or tuple(coords_mask.shape) != tuple(coords_nm.shape[:-1])
        ):
            raise RuntimeContractError(
                "clean batch.mask_dict.coords must be bool [1,N,A]"
            )
    exact_residue_shapes = {
        "chain_breaks_per_residue": ((1, residues), torch.bool),
        "coord_mask": ((1, residues, 37), torch.bool),
        "motif_mask": ((1, residues, 37), torch.bool),
        "residue_type": ((1, residues), torch.int64),
        "chain_index": ((1, residues), torch.int64),
        "chains": ((1, residues), torch.int64),
        "residue_pdb_idx": ((1, residues), torch.int64),
        "seq_pos": ((1, residues, 1), torch.int64),
        "target_residue_mask": ((1, residues), torch.bool),
        "seq": ((1, residues), torch.int64),
        "dive_modality_presence": ((1, 2), torch.int64),
    }
    for key, (shape, dtype) in exact_residue_shapes.items():
        value = batch.get(key)
        if value is not None and (
            type(value) is not Tensor
            or tuple(value.shape) != shape
            or (dtype is not None and value.dtype is not dtype)
        ):
            raise RuntimeContractError(
                f"clean batch.{key} has wrong residue shape/dtype"
            )
    presence = batch.get("dive_modality_presence")
    if type(presence) is Tensor and not torch.all((presence == 0) | (presence == 1)):
        raise RuntimeContractError(
            "clean batch.dive_modality_presence values must be binary"
        )
    target_lengths = {
        int(value.shape[1])
        for key in (
            "x_target",
            "target_mask",
            "seq_target",
            "seq_target_mask",
            "target_chains",
            "target_hotspot_mask",
            "target_padding_mask",
            "target_pdb_idx",
            "target",
        )
        if type(value := batch.get(key)) is Tensor and value.ndim >= 2
    }
    if len(target_lengths) > 1:
        raise RuntimeContractError("clean batch target tensors disagree on M")
    if target_lengths:
        targets = next(iter(target_lengths))
        target_shapes = {
            "x_target": ((1, targets, 37, 3), "float"),
            "target_mask": ((1, targets, 37), torch.bool),
            "seq_target": ((1, targets), torch.int64),
            "seq_target_mask": ((1, targets), torch.bool),
            "target_chains": ((1, targets), torch.int64),
            "target_hotspot_mask": ((1, targets), torch.bool),
            "target_padding_mask": ((1, targets), torch.bool),
            "target_pdb_idx": ((1, targets), torch.int64),
        }
        for key, (shape, dtype) in target_shapes.items():
            value = batch.get(key)
            if value is not None and (
                type(value) is not Tensor
                or tuple(value.shape) != shape
                or (dtype == "float" and not torch.is_floating_point(value))
                or (dtype != "float" and value.dtype is not dtype)
            ):
                raise RuntimeContractError(
                    f"clean batch.{key} has wrong target shape/dtype"
                )
        target_value = batch.get("target")
        if target_value is not None and (
            type(target_value) is not Tensor
            or target_value.ndim != 3
            or tuple(target_value.shape[:2]) != (1, targets)
            or not torch.is_floating_point(target_value)
        ):
            raise RuntimeContractError("clean batch.target must be floating [1,M,D]")
    motif = batch.get("x_motif")
    if motif is not None and (
        type(motif) is not Tensor
        or motif.ndim != 4
        or motif.shape[0] != 1
        or motif.shape[1] > residues
        or tuple(motif.shape[2:]) != (37, 3)
        or not torch.is_floating_point(motif)
    ):
        raise RuntimeContractError("clean batch.x_motif must be floating [1,K<=N,37,3]")
    concat = batch.get("concat")
    concat_mask = batch.get("concat_mask")
    if (concat is None) != (concat_mask is None):
        raise RuntimeContractError("clean batch concat and concat_mask must coexist")
    if concat is not None:
        assert concat_mask is not None
        if (
            type(concat) is not Tensor
            or concat.ndim != 3
            or concat.shape[0] != 1
            or concat.shape[-1] <= 0
            or not torch.is_floating_point(concat)
            or type(concat_mask) is not Tensor
            or concat_mask.dtype is not torch.bool
            or tuple(concat_mask.shape) != tuple(concat.shape[:2])
        ):
            raise RuntimeContractError("clean batch concat tensors are invalid")
    extended_pair = batch.get("extended_pair")
    if extended_pair is not None:
        extended = residues + (int(concat.shape[1]) if type(concat) is Tensor else 0)
        if (
            type(extended_pair) is not Tensor
            or extended_pair.ndim != 4
            or tuple(extended_pair.shape[:3]) != (1, extended, extended)
            or extended_pair.shape[-1] <= 0
            or not torch.is_floating_point(extended_pair)
        ):
            raise RuntimeContractError(
                "clean batch.extended_pair has wrong extended shape/dtype"
            )
    for key, value in batch.items():
        if key == "mask_dict":
            continue
        assert isinstance(value, Tensor)
        if value.ndim < 1 or value.shape[0] != 1:
            raise RuntimeContractError(f"clean batch.{key} must have batch size one")

def _clone_tree(value: object) -> object:
    if type(value) is Tensor:
        return value.detach().clone()
    if type(value) in (dict, MappingProxyType):
        return {key: _clone_tree(nested) for key, nested in value.items()}
    raise RuntimeContractError("verified tensor schema contains a non-tensor leaf")

def _freeze_tree(value: object) -> object:
    if type(value) is Tensor:
        return value.detach().clone()
    if type(value) in (dict, MappingProxyType):
        return MappingProxyType(
            {key: _freeze_tree(nested) for key, nested in value.items()}
        )
    raise RuntimeContractError("verified tensor schema contains a non-tensor leaf")

def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + raw).hexdigest()

def _tree_digest(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if type(item) is Tensor:
            digest.update(b"T")
            digest.update(_tensor_digest(item).encode())
            return
        if type(item) in (dict, MappingProxyType):
            digest.update(b"M")
            for key in sorted(item):
                digest.update(key.encode())
                visit(item[key])
            return
        raise RuntimeContractError("verified tensor schema contains a non-tensor leaf")

    visit(value)
    return digest.hexdigest()

def _tree_elements(value: object) -> int:
    if type(value) is Tensor:
        return int(value.numel())
    if type(value) in (dict, MappingProxyType):
        return sum(_tree_elements(nested) for nested in value.values())
    raise RuntimeContractError("verified tensor schema contains a non-tensor leaf")

def _finite_events(name: str, value: object) -> tuple[str, ...]:
    events: list[str] = []

    def visit(path: str, item: object) -> None:
        if type(item) is Tensor:
            count = int((~torch.isfinite(item)).sum().item())
            if count:
                events.append(f"{path}:nonfinite={count}")
            return
        if type(item) in (dict, MappingProxyType):
            for key in sorted(item):
                visit(f"{path}.{key}", item[key])
            return
        raise RuntimeContractError(f"{path} contains a non-tensor leaf")

    visit(name, value)
    return tuple(events)

def _require_finite_tree(name: str, value: object) -> None:
    events = _finite_events(name, value)
    if events:
        raise RuntimeContractError(
            f"{name} contains non-finite tensor values: {events}"
        )

def _finite_events_residue_tensor(
    name: str, value: Tensor, valid: Tensor
) -> tuple[str, ...]:
    if value.ndim < 2 or tuple(value.shape[:2]) != tuple(valid.shape):
        raise RuntimeContractError(f"{name} does not share the exact residue mask")
    return _finite_events(name, value)

def _masked_squared_sum(value: Tensor, valid: Tensor) -> float:
    if value.ndim < 2 or tuple(value.shape[:2]) != tuple(valid.shape):
        raise RuntimeContractError("tensor does not share the exact residue mask")
    expanded = valid
    for _ in range(value.ndim - 2):
        expanded = expanded.unsqueeze(-1)
    selected = torch.nan_to_num(value).masked_select(expanded.expand_as(value))
    return float(selected.double().square().sum().item())

def validate_raw_row(row: FlowRow) -> FlowRow:

    if type(row) is not FlowRow:
        raise RuntimeContractError("raw row must be the exact FlowRow type")
    request = _exact_mapping("raw row request", row.request)
    expected = {
        "family",
        "example_id",
        "parent_id",
        "cell_hash",
        "corruption_seed",
        "payload",
    }
    _exact_keys("raw row request", request, expected)
    identity = row.identity.to_mapping()
    if {key: request[key] for key in identity} != identity:
        raise RuntimeContractError("raw row identity differs from its request")
    if type(request["corruption_seed"]) is not int or request["corruption_seed"] != 42:
        raise RuntimeContractError("raw row corruption_seed must be exactly 42")
    payload = _exact_mapping("raw row payload", request["payload"])
    _exact_keys("raw row payload", payload, {"batch"})
    _reject_outcomes("raw row request", request)
    batch = _freeze_clean_batch(payload["batch"])
    return FlowRow(
        row.family,
        row.example_id,
        row.parent_id,
        row.cell_hash,
        MappingProxyType(
            {
                **identity,
                "corruption_seed": 42,
                "payload": MappingProxyType({"batch": batch}),
            }
        ),
    )

def seal_loaded_data(data: BlindEvaluationData) -> BlindEvaluationData:

    if type(data) is not BlindEvaluationData:
        raise RuntimeContractError("blind loader returned the wrong data type")
    return BlindEvaluationData(
        flow_rows=tuple(validate_raw_row(row) for row in data.flow_rows),
        generation_cells=data.generation_cells,
    )

class VerifiedTensorTree:

    __slots__ = ("__batch", "__clean_keys")

    def __init__(
        self, batch: Mapping[str, object], clean_keys: tuple[str, ...]
    ) -> None:
        self.__batch = _freeze_tree(batch)
        self.__clean_keys = clean_keys

    @classmethod
    def from_corruption(
        cls, clean: Mapping[str, object], corrupted: object
    ) -> "VerifiedTensorTree":
        value = _exact_mapping("corrupted batch", corrupted)
        expected = set(clean) | set(_CORRUPTION_KEYS)
        _exact_keys("corrupted batch", value, expected)
        for key in clean:
            if _tree_digest(value[key]) != _tree_digest(clean[key]):
                raise RuntimeContractError(
                    f"corruption changed clean input field {key!r}"
                )
        for name in ("x_0", "x_t", "x_1", "t"):
            nested = _exact_mapping(f"corrupted batch.{name}", value[name])
            _exact_keys(f"corrupted batch.{name}", nested, set(_MODALITIES))
            if any(type(nested[mode]) is not Tensor for mode in _MODALITIES):
                raise RuntimeContractError(
                    f"corrupted batch.{name} has non-tensor fields"
                )
        masks = value["mask"]
        assert isinstance(masks, Tensor)
        batch_size, residues = masks.shape
        if batch_size != 1:
            raise RuntimeContractError("each raw strict row must have batch size one")
        for name in ("x_0", "x_t", "x_1"):
            nested = value[name]
            assert isinstance(nested, Mapping)
            for mode in _MODALITIES:
                tensor = nested[mode]
                assert isinstance(tensor, Tensor)
                if tensor.ndim != 3 or tuple(tensor.shape[:2]) != (1, residues):
                    raise RuntimeContractError(
                        f"corrupted batch.{name}.{mode} must be [1,N,D]"
                    )
        for mode in _MODALITIES:
            x0 = value["x_0"][mode]
            xt = value["x_t"][mode]
            x1 = value["x_1"][mode]
            if tuple(x0.shape) != tuple(xt.shape) or tuple(xt.shape) != tuple(x1.shape):
                raise RuntimeContractError(
                    f"corrupted {mode} product-space shapes differ"
                )
            if x0.dtype != xt.dtype or xt.dtype != x1.dtype:
                raise RuntimeContractError(
                    f"corrupted {mode} product-space dtypes differ"
                )
            time = value["t"][mode]
            if time.ndim != 1 or tuple(time.shape) != (1,):
                raise RuntimeContractError(f"corrupted batch.t.{mode} must be [1]")
        coords_nm = value["coords_nm"]
        if not torch.equal(
            value["x_1"]["bb_ca"],
            coords_nm[:, :, 1, :],
        ):
            raise RuntimeContractError("corrupted x_1.bb_ca differs from coords_nm CA")
        _require_finite_tree("corrupted batch", value)
        return cls(value, tuple(sorted(clean)))

    def clone_for_forward(self) -> dict[str, object]:
        return _clone_tree(self.__batch)

    def digest(self) -> str:
        return _tree_digest(self.__batch)

    def field_digest(self, name: str) -> str:
        return _tree_digest(self.__batch[name])

    def mask_digest(self) -> str:
        return _tree_digest(
            MappingProxyType(
                {
                    key: self.__batch[key]
                    for key in ("mask", "generated_mask", "fixed_mask")
                }
            )
        )

    def valid_mask(self) -> Tensor:
        return (self.__batch["mask"] & self.__batch["generated_mask"]).clone()

    def times(self) -> tuple[float, float]:
        values = self.__batch["t"]
        return tuple(float(values[mode].item()) for mode in _MODALITIES)

    def conditioning_digests(self) -> dict[str, str]:
        return {key: _tree_digest(self.__batch[key]) for key in self.__clean_keys}

    def conditioning_elements(self) -> int:
        return sum(_tree_elements(self.__batch[key]) for key in self.__clean_keys)

@dataclass(slots=True)
class _ExecutionLedger:
    corruptions: int = 0
    routes: tuple[RouteState, ...] = ()
    parent_forwards: int = 0

@dataclass(frozen=True, slots=True)
class CorruptedFlowInput:
    identity: FlowIdentity
    corruption_seed: int
    corruption_digest: str
    input_digest: str
    time_digest: str
    target_digest: str
    mask_digest: str
    _snapshot: VerifiedTensorTree
    _ledger: _ExecutionLedger

@dataclass(frozen=True, slots=True)
class TelemetryEvidence:
    probability_numerators: tuple[float, float, float, float]
    z_to_x_squared_sum: float
    x_to_z_squared_sum: float

@dataclass(frozen=True, slots=True)
class RouteFlowResult:
    requested_route: RouteState
    observed_route: RouteState
    corruption_digest: str
    input_digest: str
    time_digest: str
    target_digest: str
    mask_digest: str
    prediction_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    target_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    prediction_dtypes: tuple[str, str]
    target_dtypes: tuple[str, str]
    loss_sum: float
    valid_generated_residues: int
    non_finite_events: tuple[str, ...]
    per_residue_loss: tuple[float, ...]
    prediction_x_squared_sum: float
    prediction_z_squared_sum: float
    telemetry: tuple[TelemetryEvidence, ...]
    adaptive_simplex_violation_count: int
    hard_one_hot_violation_count: int
    none_bridge_residual_nonzero_count: int
    conditioning_mutation_count: int

    @property
    def non_finite_count(self) -> int:
        return len(self.non_finite_events)

@dataclass(frozen=True, slots=True)
class FrozenParentResult:
    corruption_digest: str
    input_digest: str
    time_digest: str
    target_digest: str
    mask_digest: str
    prediction_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    target_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    prediction_dtypes: tuple[str, str]
    target_dtypes: tuple[str, str]
    loss_sum: float
    valid_generated_residues: int
    non_finite_events: tuple[str, ...]
    conditioning_mutation_count: int
    forward_count: int

    @property
    def non_finite_count(self) -> int:
        return len(self.non_finite_events)

@dataclass(frozen=True, slots=True)
class RowFlowEvidence:
    identity: FlowIdentity
    corruption_seed: int
    corruption_digest: str
    input_digest: str
    time_digest: str
    target_digest: str
    mask_digest: str
    valid_mask: tuple[bool, ...]
    times: tuple[float, float]
    routes: tuple[RouteFlowResult, ...]
    frozen_parent: FrozenParentResult
    corruption_count: int
    route_calls: tuple[RouteState, ...]
    conditioning_elements: int

_CONSTRUCTION_TOKEN = object()
_DATA_CONSTRUCTION_TOKEN = object()
_PARENT_DESCRIPTOR_TOKEN = object()
_PARENT_TEST_TOKEN = object()

_PINNED_PARENT_MODULE = "proteinfoundation.nn.local_latents_transformer_v2"
_PINNED_PARENT_QUALNAME = "LocalLatentsTransformer"
_PINNED_PARENT_RELATIVE_PATH = (
    "src/proteinfoundation/nn/local_latents_transformer_v2.py"
)

def _assert_authenticated_catalog(source_catalog: object) -> None:

    from dive.evaluation.directional_catalog import (
        CATALOG_SCHEMA_VERSION,
        _catalog_identity,
    )

    schema_version = getattr(source_catalog, "schema_version", None)
    dive_commit = getattr(source_catalog, "dive_commit", None)
    upstream_commit = getattr(source_catalog, "upstream_commit", None)
    modules = getattr(source_catalog, "modules", None)
    identity = getattr(source_catalog, "identity_sha256", None)
    if (
        schema_version != CATALOG_SCHEMA_VERSION
        or type(dive_commit) is not str
        or not isinstance(modules, Mapping)
        or type(identity) is not str
    ):
        raise RuntimeContractError(
            "frozen parent authenticated source catalog is not exact"
        )
    if upstream_commit != EMERGENT_UPSTREAM_COMMIT:
        raise RuntimeContractError(
            "frozen parent source catalog is not the pinned emergent upstream commit"
        )
    if identity != _catalog_identity(
        schema_version, dive_commit, upstream_commit, modules
    ):
        raise RuntimeContractError(
            "frozen parent source catalog identity does not match its own contents"
        )

def _pinned_parent_class(realm=None):

    from proteinfoundation.nn.local_latents_transformer_v2 import (
        LocalLatentsTransformer,
    )

    if LocalLatentsTransformer.__module__ != _PINNED_PARENT_MODULE:
        raise RuntimeContractError(
            "frozen parent class does not belong to the pinned denoiser module"
        )
    if realm is None:
        expected = (
            EMERGENT_UPSTREAM_ROOT
            / "src"
            / "proteinfoundation"
            / "nn"
            / "local_latents_transformer_v2.py"
        ).resolve()
        source = inspect.getsourcefile(LocalLatentsTransformer)
        if (
            source is None
            or Path(source).resolve() != expected
            or Path(LocalLatentsTransformer.forward.__code__.co_filename).resolve()
            != expected
        ):
            raise RuntimeContractError(
                "frozen parent class source differs from the pinned checkout"
            )
        return LocalLatentsTransformer
    module = importlib.import_module(_PINNED_PARENT_MODULE)
    if (
        not isinstance(module, ModuleType)
        or vars(module).get(_PINNED_PARENT_QUALNAME) is not LocalLatentsTransformer
    ):
        raise RuntimeContractError(
            "the pinned parent module does not define the pinned parent class"
        )
    _assert_realm_owned_module(realm, _PINNED_PARENT_MODULE, module)
    source_catalog = getattr(realm, "catalog", None)
    _assert_authenticated_catalog(source_catalog)
    record = source_catalog.modules.get(_PINNED_PARENT_MODULE)
    if (
        record is None
        or record.root_class != "upstream"
        or record.relative_path != _PINNED_PARENT_RELATIVE_PATH
    ):
        raise RuntimeContractError(
            "the pinned parent module is not the pinned upstream catalog member"
        )
    if hashlib.sha256(record.source_bytes).hexdigest() != record.sha256:
        raise RuntimeContractError(
            "catalog source bytes for the pinned parent differ from their digest"
        )
    origin = f"<catalog:{record.relative_path}>"
    if (
        getattr(module, "__file__", None) != origin
        or LocalLatentsTransformer.forward.__code__.co_filename != origin
    ):
        raise RuntimeContractError(
            "the pinned parent class was not compiled from its catalog member"
        )
    return LocalLatentsTransformer

_MODULE_HOOK_REGISTRIES = frozenset(
    {
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
    }
)
_MODULE_BOUND_BOOKKEEPING = (
    frozenset(
        {
            "_buffers",
            "_is_full_backward_hook",
            "_modules",
            "_non_persistent_buffers_set",
            "_parameters",
        }
    )
    | _MODULE_HOOK_REGISTRIES
)

def _logical_tensor_signature(value: Tensor) -> tuple[object, ...]:
    if value.layout is not torch.strided:
        raise RuntimeContractError(
            "frozen parent tensor state must use the exact strided layout"
        )
    logical = value.detach().cpu().contiguous()
    raw = logical.reshape(-1).view(torch.uint8).numpy().tobytes()
    return (
        str(value.dtype),
        tuple(value.shape),
        str(value.device),
        hashlib.sha256(raw).hexdigest(),
    )

def _callable_code_signature(
    value: FunctionType, path: str, module_names: Mapping[int, str], active: set[int]
) -> tuple[object, ...]:
    closure = None
    if value.__closure__ is not None:
        closure = tuple(
            _bind_forward_state(
                cell.cell_contents,
                f"{path}.__closure__[{index}]",
                module_names,
                active,
            )
            for index, cell in enumerate(value.__closure__)
        )
    referenced_globals = tuple(
        (
            name,
            _bind_forward_global(
                value.__globals__[name],
                f"{path}.__globals__[{name!r}]",
                module_names,
                active,
            ),
        )
        for name in sorted(set(value.__code__.co_names))
        if name in value.__globals__
    )
    return (
        "function",
        id(value),
        value.__module__,
        value.__qualname__,
        id(value.__code__),
        hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
        _bind_forward_state(
            value.__defaults__, f"{path}.__defaults__", module_names, active
        ),
        _bind_forward_state(
            value.__kwdefaults__, f"{path}.__kwdefaults__", module_names, active
        ),
        _forward_annotation_signature(value.__annotations__),
        referenced_globals,
        closure,
    )

def _optional_external_module(name: str) -> ModuleType | None:

    module = sys.modules.get(name)
    if module is not None:
        return module
    try:
        return importlib.import_module(name)
    except Exception:
        return None

def _optional_external_attribute(module_name: str, attribute: str) -> object | None:

    module = _optional_external_module(module_name)
    return None if module is None else getattr(module, attribute, None)

_PARENT_FORWARD_EXTERNAL_MODULE_LEAVES = MappingProxyType(
    {
        "gzip": gzip,
        "math": math,
        "os": os,
        "platform": platform,
        "random": random,
        "torch": torch,
        "torch.nn": torch.nn,
        "torch.nn.functional": torch.nn.functional,
    }
)

_OPTIONAL_EXTERNAL_MODULE_LEAF_NAMES = ("einops", "openfold.data.data_transforms")

def _external_module_leaf(name: str) -> ModuleType | None:

    expected = _PARENT_FORWARD_EXTERNAL_MODULE_LEAVES.get(name)
    if expected is not None:
        return expected
    if name in _OPTIONAL_EXTERNAL_MODULE_LEAF_NAMES:
        return _optional_external_module(name)
    return None

_PARENT_FORWARD_EXTERNAL_IDENTITY_LEAVES = MappingProxyType(
    {
        "loguru.logger": _LOGURU_LOGGER,
        "torch.einsum": torch.einsum,
        "torch.nn.utils.rnn.pad_sequence": torch.nn.utils.rnn.pad_sequence,
        "torch.utils.checkpoint.checkpoint": _optional_external_attribute(
            "torch.utils.checkpoint", "checkpoint"
        ),
    }
)
_OPTIONAL_EXTERNAL_IDENTITY_LEAVES = (
    ("einops.rearrange", "einops", "rearrange"),
    (
        "cuequivariance_torch.attention_pair_bias",
        "cuequivariance_torch",
        "attention_pair_bias",
    ),
)

def _optional_external_identity_leaf(value: object) -> str | None:

    for label, module_name, attribute in _OPTIONAL_EXTERNAL_IDENTITY_LEAVES:
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, attribute, None) is value:
            return label
    return None

def _forward_annotation_signature(value: object) -> tuple[object, ...]:
    value_type = type(value)
    if value is None or value_type in (bool, int, float, complex, str, bytes):
        return (value_type, value)
    if isinstance(value, type):
        return ("annotation-type", id(value), value.__module__, value.__qualname__)
    if isinstance(value, Mapping):
        entries = [
            (
                _forward_annotation_signature(key),
                _forward_annotation_signature(nested),
            )
            for key, nested in value.items()
        ]
        return ("annotation-mapping", value_type, tuple(sorted(entries, key=repr)))
    if isinstance(value, Sequence):
        return (
            "annotation-sequence",
            value_type,
            tuple(_forward_annotation_signature(nested) for nested in value),
        )
    return ("annotation-leaf", id(value), value_type, repr(value))

def _bind_forward_global(
    value: object,
    path: str,
    module_names: Mapping[int, str],
    active: set[int],
) -> tuple[object, ...]:
    if isinstance(value, ModuleType):
        if _external_module_leaf(value.__name__) is not value:
            raise RuntimeContractError(
                f"frozen parent state {path} references an unauthenticated module"
            )
        return ("external-module-leaf", value.__name__, id(value))
    for name, expected in _PARENT_FORWARD_EXTERNAL_IDENTITY_LEAVES.items():
        if value is expected:
            return ("external-identity-leaf", name, id(value))
    optional = _optional_external_identity_leaf(value)
    if optional is not None:
        return ("external-identity-leaf", optional, id(value))
    return _bind_forward_state(value, path, module_names, active)

def _bind_forward_state(
    value: object,
    path: str,
    module_names: Mapping[int, str],
    active: set[int],
) -> tuple[object, ...]:
    value_type = type(value)
    if value is None or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        if not math.isfinite(value):
            raise RuntimeContractError(
                f"frozen parent state {path} contains a non-finite float"
            )
        return (float, value)
    if isinstance(value, Enum):
        return ("enum", type(value), value.name)
    if isinstance(value, torch.dtype):
        return ("torch.dtype", str(value))
    if isinstance(value, torch.device):
        return ("torch.device", str(value))
    if type(value) is Tensor:
        return (
            "tensor",
            id(value),
            int(value.untyped_storage().data_ptr()),
            int(value._version),
            _logical_tensor_signature(value),
        )

    identity = id(value)
    if identity in active:
        raise RuntimeContractError(f"frozen parent state {path} is cyclic")
    active.add(identity)
    try:
        if isinstance(value, partial):
            return (
                "partial",
                id(value),
                _bind_forward_state(value.func, f"{path}.func", module_names, active),
                _bind_forward_state(value.args, f"{path}.args", module_names, active),
                _bind_forward_state(
                    value.keywords, f"{path}.keywords", module_names, active
                ),
            )
        if isinstance(value, FunctionType):
            return _callable_code_signature(value, path, module_names, active)
        if isinstance(value, MethodType):
            owner = value.__self__
            if isinstance(owner, nn.Module):
                try:
                    owner_signature: object = (
                        "module-reference",
                        module_names[id(owner)],
                        id(owner),
                    )
                except KeyError as error:
                    raise RuntimeContractError(
                        f"frozen parent state {path} binds an external module"
                    ) from error
            else:
                owner_signature = _bind_forward_state(
                    owner, f"{path}.__self__", module_names, active
                )
            return (
                "method",
                id(value),
                _callable_code_signature(
                    value.__func__, f"{path}.__func__", module_names, active
                ),
                owner_signature,
            )
        if isinstance(value, (BuiltinFunctionType, BuiltinMethodType)):
            owner = getattr(value, "__self__", None)
            return (
                "builtin-callable",
                id(value),
                value.__module__,
                value.__qualname__,
                _bind_forward_state(owner, f"{path}.__self__", module_names, active),
            )
        if isinstance(value, type):
            return ("class", id(value), value.__module__, value.__qualname__)
        if is_dataclass(value) and not isinstance(value, type):
            return (
                "dataclass",
                type(value),
                tuple(
                    (
                        field.name,
                        _bind_forward_state(
                            getattr(value, field.name),
                            f"{path}.{field.name}",
                            module_names,
                            active,
                        ),
                    )
                    for field in fields(value)
                ),
            )
        if isinstance(value, Mapping):
            entries = [
                (
                    _bind_forward_state(key, f"{path}.key", module_names, active),
                    _bind_forward_state(
                        nested, f"{path}[{key!r}]", module_names, active
                    ),
                )
                for key, nested in value.items()
            ]
            return ("mapping", type(value), tuple(sorted(entries, key=repr)))
        if isinstance(value, Sequence):
            return (
                "sequence",
                type(value),
                tuple(
                    _bind_forward_state(
                        nested, f"{path}[{index}]", module_names, active
                    )
                    for index, nested in enumerate(value)
                ),
            )
        if isinstance(value, (set, frozenset)):
            values = [
                _bind_forward_state(nested, f"{path}.item", module_names, active)
                for nested in value
            ]
            return ("set", type(value), tuple(sorted(values, key=repr)))
        if callable(value):
            call = getattr(type(value), "__call__", None)
            if not isinstance(call, FunctionType) or not hasattr(value, "__dict__"):
                raise RuntimeContractError(
                    f"frozen parent state {path} has an unsupported callable"
                )
            return (
                "callable-object",
                id(value),
                type(value),
                _callable_code_signature(
                    call, f"{path}.__call__", module_names, active
                ),
                _bind_forward_state(
                    vars(value), f"{path}.__dict__", module_names, active
                ),
            )
    finally:
        active.remove(identity)
    raise RuntimeContractError(
        f"frozen parent state {path} has unsupported type {type(value).__name__}"
    )

def _module_execution_signature(module: nn.Module) -> tuple[object, ...]:
    named_modules = tuple(module.named_modules())
    module_names = {id(child): name for name, child in named_modules}
    modules: list[object] = []
    instance_state: list[object] = []
    for name, child in named_modules:
        forward = getattr(type(child), "forward", None)
        if not callable(forward) or "forward" in vars(child):
            raise RuntimeContractError(
                f"frozen parent module {name!r} has a forward instance override"
            )
        modules.append(
            (
                name,
                id(child),
                type(child),
                _bind_forward_state(
                    forward,
                    f"{name or '<root>'}.forward",
                    module_names,
                    set(),
                ),
            )
        )
        for registry in _MODULE_HOOK_REGISTRIES:
            hooks = vars(child).get(registry)
            if hooks:
                raise RuntimeContractError(
                    f"frozen parent module {name!r} has a nonempty hook registry"
                )
        child_state = tuple(
            (
                attribute,
                _bind_forward_state(
                    value,
                    f"{name or '<root>'}.{attribute}",
                    module_names,
                    set(),
                ),
            )
            for attribute, value in sorted(vars(child).items())
            if attribute not in _MODULE_BOUND_BOOKKEEPING
        )
        instance_state.append((name, child_state))
    state = []
    for kind, values in (
        ("parameter", module.named_parameters(remove_duplicate=False)),
        ("buffer", module.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in values:
            state.append(
                (
                    kind,
                    name,
                    id(tensor),
                    int(tensor.untyped_storage().data_ptr()),
                    int(tensor._version),
                    tuple(tensor.shape),
                    tensor.dtype,
                    tensor.device,
                    _logical_tensor_signature(tensor),
                )
            )
    return tuple(modules), tuple(instance_state), tuple(state)

_PARENT_FORWARD_EXTERNAL_CLASS_LEAVES = MappingProxyType(
    {
        module_class: module_class.forward
        for module_class in (
            nn.Dropout,
            nn.Embedding,
            nn.Identity,
            nn.LayerNorm,
            nn.Linear,
            nn.ModuleDict,
            nn.ModuleList,
            nn.ReLU,
            nn.Sequential,
            nn.Sigmoid,
        )
    }
)

_OPTIONAL_EXTERNAL_CLASS_LEAF_NAMES = (
    ("openfold.model.dropout", "DropoutColumnwise"),
    ("openfold.model.dropout", "DropoutRowwise"),
    ("openfold.model.pair_transition", "PairTransition"),
    ("openfold.model.primitives", "Attention"),
    ("openfold.model.primitives", "LayerNorm"),
    ("openfold.model.primitives", "Linear"),
    ("openfold.model.triangular_attention", "TriangleAttentionEndingNode"),
    ("openfold.model.triangular_attention", "TriangleAttentionStartingNode"),
    (
        "openfold.model.triangular_multiplicative_update",
        "TriangleMultiplicationIncoming",
    ),
    (
        "openfold.model.triangular_multiplicative_update",
        "TriangleMultiplicationOutgoing",
    ),
)

def _optional_external_class_leaves() -> tuple[tuple[type, FunctionType], ...]:

    resolved = []
    for module_name, attribute in _OPTIONAL_EXTERNAL_CLASS_LEAF_NAMES:
        module = sys.modules.get(module_name)
        candidate = None if module is None else getattr(module, attribute, None)
        if isinstance(candidate, type) and issubclass(candidate, nn.Module):
            resolved.append((candidate, candidate.forward))
    return tuple(resolved)

def _external_class_leaf(owner: type) -> FunctionType | None:

    leaf = _PARENT_FORWARD_EXTERNAL_CLASS_LEAVES.get(owner)
    if leaf is not None:
        return leaf
    for candidate, candidate_leaf in _optional_external_class_leaves():
        if owner is candidate:
            return candidate_leaf
    return None

def _external_class_leaves() -> Mapping[type, FunctionType]:

    return MappingProxyType(
        {
            **_PARENT_FORWARD_EXTERNAL_CLASS_LEAVES,
            **dict(_optional_external_class_leaves()),
        }
    )

@dataclass(frozen=True, slots=True)
class _CatalogClassReference:

    module_name: str
    qualname: str
    source_bytes: bytes
    source_sha256: str
    filename: str
    module: ModuleType

@dataclass(frozen=True, slots=True)
class _AuthenticatedClassExpectation:

    module_name: str
    qualname: str
    source_sha256: str
    base_paths: tuple[tuple[str, ...], ...]
    method_kinds: Mapping[str, str]
    method_fingerprints: Mapping[str, bytes]
    method_metadata: Mapping[str, tuple[object, object, object]]

@dataclass(frozen=True, slots=True)
class AuthenticatedParentGraph:

    catalog_identity: str
    traversal: tuple[str, ...]
    node_classes: Mapping[str, tuple[str, str]]
    external_leaves: Mapping[str, tuple[type, FunctionType]]
    classes: Mapping[tuple[str, str], _AuthenticatedClassExpectation]
    realm: object | None
    method_globals: Mapping[tuple[str, str], Mapping[str, tuple[object, ...]]]

_PARENT_METHOD_KINDS = frozenset(
    {"function", "staticmethod", "classmethod", "property"}
)
_PARENT_SOURCE_EXPRESSION_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.Slice,
    ast.BinOp,
    ast.BitOr,
    ast.UnaryOp,
    ast.USub,
    ast.UAdd,
    ast.Invert,
    ast.Load,
)

def _filename_normalized_code(code: CodeType) -> CodeType:

    return code.replace(
        co_filename="",
        co_consts=tuple(
            _filename_normalized_code(value) if isinstance(value, CodeType) else value
            for value in code.co_consts
        ),
    )

def _parent_code_fingerprint(code: CodeType) -> bytes:
    return _authenticated_code_fingerprint(_filename_normalized_code(code))

def _class_local_callables(owner: type) -> dict[str, tuple[str, FunctionType]]:

    callables: dict[str, tuple[str, FunctionType]] = {}
    for name, value in vars(owner).items():
        if isinstance(value, staticmethod):
            kind, function = "staticmethod", value.__func__
        elif isinstance(value, classmethod):
            kind, function = "classmethod", value.__func__
        elif isinstance(value, property):
            if value.fset is not None or value.fdel is not None:
                raise RuntimeContractError(
                    f"frozen parent class {owner.__qualname__!r} property {name!r} "
                    "is not an authenticated read-only accessor"
                )
            kind, function = "property", value.fget
        elif isinstance(value, FunctionType):
            kind, function = "function", value
        else:
            continue
        if not isinstance(function, FunctionType):
            raise RuntimeContractError(
                f"frozen parent class {owner.__qualname__!r} member {name!r} "
                "is not an authenticated Python function"
            )
        callables[name] = (kind, function)
    return callables

def _parent_method_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if not node.decorator_list:
        return "function"
    decorator = node.decorator_list[0]
    if (
        len(node.decorator_list) != 1
        or not isinstance(decorator, ast.Name)
        or decorator.id not in _PARENT_METHOD_KINDS
    ):
        raise RuntimeContractError(
            f"frozen parent authenticated method {node.name!r} has an unaudited decorator"
        )
    return decorator.id

def _parent_dotted_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise RuntimeContractError(
            "frozen parent authenticated class base is not an audited dotted name"
        )
    parts.append(node.id)
    return tuple(reversed(parts))

def _resolve_dotted_path(
    path: tuple[str, ...], namespace: Mapping[str, object], label: str
) -> object:
    if path[0] not in namespace:
        raise RuntimeContractError(
            f"frozen parent class {label} base {'.'.join(path)!r} is not bound"
        )
    value = namespace[path[0]]
    for part in path[1:]:
        value = getattr(value, part, None)
        if value is None:
            raise RuntimeContractError(
                f"frozen parent class {label} base {'.'.join(path)!r} is not bound"
            )
    return value

def _parent_source_value(
    node: ast.AST, namespace: dict[str, object], filename: str
) -> object:

    expression = ast.Expression(body=node)
    for child in ast.walk(expression):
        if not isinstance(child, _PARENT_SOURCE_EXPRESSION_NODES):
            raise RuntimeContractError(
                "frozen parent authenticated method metadata is not an audited expression"
            )
    expression = ast.fix_missing_locations(expression)
    try:
        compiled = compile(expression, filename, "eval", dont_inherit=True)
        return eval(compiled, namespace)
    except RuntimeContractError:
        raise
    except Exception as error:
        raise RuntimeContractError(
            "frozen parent authenticated method metadata cannot be evaluated"
        ) from error

def _parent_method_metadata(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    namespace: dict[str, object],
    filename: str,
    *,
    postponed_annotations: bool,
) -> tuple[object, object, object]:
    def annotation(value: ast.AST) -> object:
        if postponed_annotations:
            return ast.unparse(value)
        return _parent_source_value(value, namespace, filename)

    defaults = (
        tuple(
            _parent_source_value(default, namespace, filename)
            for default in node.args.defaults
        )
        if node.args.defaults
        else None
    )
    keyword_defaults = {
        argument.arg: _parent_source_value(default, namespace, filename)
        for argument, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=True
        )
        if default is not None
    }
    annotations = {
        argument.arg: annotation(argument.annotation)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *((node.args.vararg,) if node.args.vararg is not None else ()),
            *((node.args.kwarg,) if node.args.kwarg is not None else ()),
        )
        if argument.annotation is not None
    }
    if node.returns is not None:
        annotations["return"] = annotation(node.returns)
    return (defaults, keyword_defaults or None, annotations)

def _catalog_realm_and_source(catalog):

    from dive.evaluation.directional_catalog import CatalogFinder

    if catalog is None:
        return None, None
    if type(catalog) is CatalogFinder or _is_installed_realm_finder(catalog):
        return catalog, catalog.catalog
    return None, catalog

def _is_installed_realm_finder(candidate: object) -> bool:

    if not sys.meta_path or sys.meta_path[0] is not candidate:
        return False
    source = getattr(candidate, "catalog", None)
    return isinstance(getattr(source, "identity_sha256", None), str) and isinstance(
        getattr(source, "modules", None), Mapping
    )

def _assert_realm_owned_module(realm, module_name: str, module: ModuleType) -> None:

    if realm is None:
        return
    owned = realm.loaded_modules.get(module_name)
    if owned is None:
        raise RuntimeContractError(
            f"project module {module_name!r} was never loaded by the catalog realm"
        )
    if owned is not module:
        raise RuntimeContractError(
            f"project module {module_name!r} is not the catalog realm's own module object"
        )

def _parent_referenced_globals(function: FunctionType) -> tuple[object, ...]:

    module_globals = function.__globals__
    return tuple(
        (
            name,
            _bind_forward_global(
                module_globals[name],
                f"{function.__qualname__}.__globals__[{name!r}]",
                {},
                set(),
            ),
        )
        for name in sorted(set(function.__code__.co_names))
        if name in module_globals
    )

def _catalog_reference_class(catalog, live_class: type) -> _CatalogClassReference:

    realm, catalog = _catalog_realm_and_source(catalog)
    module_name = live_class.__module__
    qualname = live_class.__qualname__
    if type(module_name) is not str or type(qualname) is not str or "." in qualname:
        raise RuntimeContractError(
            f"frozen parent class {live_class!r} is not module-level"
        )
    if catalog is None:
        raise RuntimeContractError(
            f"frozen parent class {live_class!r} requires an authenticated source catalog"
        )
    if module_name.partition(".")[0] not in PROJECT_PREFIXES:
        raise RuntimeContractError(
            f"frozen parent class {live_class!r} is outside the project prefixes"
        )
    record = catalog.modules.get(module_name)
    if record is None:
        raise RuntimeContractError(
            f"frozen parent class module {module_name!r} is absent from the catalog"
        )
    if hashlib.sha256(record.source_bytes).hexdigest() != record.sha256:
        raise RuntimeContractError(
            f"catalog source bytes for {module_name!r} differ from their digest"
        )
    source_bytes = record.source_bytes
    source_sha256 = record.sha256
    filename = f"<catalog:{record.relative_path}>"
    module = importlib.import_module(module_name)
    if not isinstance(module, ModuleType) or vars(module).get(qualname) is not (
        live_class
    ):
        raise RuntimeContractError(
            f"reference module {module_name!r} does not define {qualname!r}"
        )
    _assert_realm_owned_module(realm, module_name, module)
    return _CatalogClassReference(
        module_name=module_name,
        qualname=qualname,
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        filename=filename,
        module=module,
    )

def _child_class_by_name(parent: nn.Module, name: str) -> type:

    child: object = parent
    for part in name.split("."):
        child = getattr(child, part, None)
        if child is None:
            raise RuntimeContractError(f"frozen parent child {name!r} is missing")
    if not isinstance(child, nn.Module):
        raise RuntimeContractError(f"frozen parent child {name!r} is not a module")
    return type(child)

def _audited_class_ancestry(owner: type) -> tuple[type, ...]:

    mro = owner.__mro__
    try:
        boundary = mro.index(nn.Module)
    except ValueError as error:
        raise RuntimeContractError(
            f"frozen parent class {owner.__qualname__!r} is not a Torch module"
        ) from error
    if mro[boundary:] != (nn.Module, object):
        raise RuntimeContractError(
            f"frozen parent class {owner.__qualname__!r} has an unsupported MRO"
        )
    return mro[:boundary]

def _compile_expected_class(
    catalog, live_class: type
) -> _AuthenticatedClassExpectation:
    reference = _catalog_reference_class(catalog, live_class)
    label = f"{reference.module_name}.{reference.qualname}"
    try:
        parsed = ast.parse(reference.source_bytes, filename=reference.filename)
    except (SyntaxError, ValueError) as error:
        raise RuntimeContractError(
            f"catalog source for {reference.module_name!r} cannot be parsed"
        ) from error
    class_nodes = [
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == reference.qualname
    ]
    if len(class_nodes) != 1:
        raise RuntimeContractError(
            f"frozen parent class {label} is not defined exactly once in catalog source"
        )
    class_node = class_nodes[0]
    if class_node.decorator_list or class_node.keywords:
        raise RuntimeContractError(
            f"frozen parent class {label} has an unaudited class decorator or keyword"
        )
    try:
        module_code = compile(
            reference.source_bytes, reference.filename, "exec", dont_inherit=True
        )
    except (SyntaxError, ValueError) as error:
        raise RuntimeContractError(
            f"catalog source for {reference.module_name!r} cannot be compiled"
        ) from error
    class_codes = [
        value
        for value in module_code.co_consts
        if isinstance(value, CodeType)
        and value.co_name == reference.qualname
        and value.co_qualname == reference.qualname
    ]
    if len(class_codes) != 1:
        raise RuntimeContractError(
            f"frozen parent class {label} authenticated class code is not exact"
        )
    postponed_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in parsed.body
    )
    namespace = dict(vars(reference.module))
    kinds: dict[str, str] = {}
    fingerprints: dict[str, bytes] = {}
    metadata: dict[str, tuple[object, object, object]] = {}
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in kinds:
            raise RuntimeContractError(
                f"frozen parent class {label} authenticated methods are not unique"
            )
        expected_qualname = f"{reference.qualname}.{node.name}"
        matches = [
            value
            for value in class_codes[0].co_consts
            if isinstance(value, CodeType)
            and value.co_name == node.name
            and value.co_qualname == expected_qualname
        ]
        if len(matches) != 1:
            raise RuntimeContractError(
                f"frozen parent class {label} method {node.name!r} "
                "authenticated code is not exact"
            )
        kinds[node.name] = _parent_method_kind(node)
        fingerprints[node.name] = _parent_code_fingerprint(matches[0])
        metadata[node.name] = _parent_method_metadata(
            node,
            namespace,
            reference.filename,
            postponed_annotations=postponed_annotations,
        )
    return _AuthenticatedClassExpectation(
        module_name=reference.module_name,
        qualname=reference.qualname,
        source_sha256=reference.source_sha256,
        base_paths=tuple(_parent_dotted_path(base) for base in class_node.bases),
        method_kinds=MappingProxyType(kinds),
        method_fingerprints=MappingProxyType(fingerprints),
        method_metadata=MappingProxyType(metadata),
    )

def compile_expected_parent_graph(
    catalog, parent: nn.Module, *, child_class_names: Sequence[str]
) -> AuthenticatedParentGraph:

    realm, source = _catalog_realm_and_source(catalog)
    identity = getattr(source, "identity_sha256", None)
    if (
        type(identity) is not str
        or len(identity) != 64
        or set(identity) - set("0123456789abcdef")
    ):
        raise RuntimeContractError("frozen parent catalog identity is not exact")
    traversal = ("", *tuple(child_class_names))
    node_classes: dict[str, tuple[str, str]] = {}
    external_leaves: dict[str, tuple[type, FunctionType]] = {}
    classes: dict[tuple[str, str], _AuthenticatedClassExpectation] = {}
    method_globals: dict[tuple[str, str], Mapping[str, tuple[object, ...]]] = {}
    for name in traversal:
        owner = type(parent) if not name else _child_class_by_name(parent, name)
        leaf = _external_class_leaf(owner)
        if leaf is not None:
            if getattr(owner, "forward", None) is not leaf:
                raise RuntimeContractError(
                    f"frozen parent module {name!r} external leaf changed"
                )
            external_leaves[name] = (owner, leaf)
            continue
        for ancestor in _audited_class_ancestry(owner):
            key = (ancestor.__module__, ancestor.__qualname__)
            if key in classes:
                continue
            classes[key] = _compile_expected_class(catalog, ancestor)
            method_globals[key] = MappingProxyType(
                {
                    method_name: _parent_referenced_globals(function)
                    for method_name, (_kind, function) in _class_local_callables(
                        ancestor
                    ).items()
                }
            )
        node_classes[name] = (owner.__module__, owner.__qualname__)
    return AuthenticatedParentGraph(
        catalog_identity=identity,
        traversal=traversal,
        node_classes=MappingProxyType(node_classes),
        external_leaves=MappingProxyType(external_leaves),
        classes=MappingProxyType(classes),
        realm=realm,
        method_globals=MappingProxyType(method_globals),
    )

def _authenticate_parent_closure(
    function: FunctionType, owner: type, label: str, name: str
) -> None:

    freevars = function.__code__.co_freevars
    closure = function.__closure__
    if not freevars:
        if closure is not None:
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} closure changed"
            )
        return
    if closure is None or len(closure) != len(freevars):
        raise RuntimeContractError(
            f"frozen parent class {label} method {name!r} closure changed"
        )
    for freevar, cell in zip(freevars, closure, strict=True):
        if freevar != "__class__":
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} has an "
                "unsupported closure"
            )
        try:
            contents = cell.cell_contents
        except ValueError as error:
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} closure changed"
            ) from error
        if contents is not owner:
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} closure changed"
            )

def _authenticate_project_class(
    live_class: type,
    expectation: _AuthenticatedClassExpectation,
    realm: object | None,
    referenced_globals: Mapping[str, tuple[object, ...]],
) -> None:
    label = f"{expectation.module_name}.{expectation.qualname}"
    module = sys.modules.get(expectation.module_name)
    if (
        not isinstance(module, ModuleType)
        or vars(module).get(expectation.qualname) is not live_class
    ):
        raise RuntimeContractError(
            f"frozen parent class {label} canonical module binding changed"
        )
    _assert_realm_owned_module(realm, expectation.module_name, module)
    if type(live_class) is not type:
        raise RuntimeContractError(
            f"frozen parent class {label} has a counterfeit metaclass"
        )
    canonical_globals = vars(module)
    expected_bases = tuple(
        _resolve_dotted_path(path, canonical_globals, label)
        for path in expectation.base_paths
    )
    if live_class.__bases__ != expected_bases:
        raise RuntimeContractError(
            f"frozen parent class {label} bases changed from authenticated source"
        )
    _audited_class_ancestry(live_class)
    live = _class_local_callables(live_class)
    if frozenset(live) != frozenset(expectation.method_kinds):
        missing = sorted(frozenset(expectation.method_kinds) - frozenset(live))
        extra = sorted(frozenset(live) - frozenset(expectation.method_kinds))
        raise RuntimeContractError(
            f"frozen parent class {label} method set changed from authenticated "
            f"source (missing={missing}, extra={extra})"
        )
    for name, (kind, function) in live.items():
        if kind != expectation.method_kinds[name]:
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} descriptor kind changed"
            )
        if function.__globals__ is not canonical_globals:
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} globals changed "
                "from the canonical module namespace"
            )
        if (
            _parent_code_fingerprint(function.__code__)
            != (expectation.method_fingerprints[name])
        ):
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} code changed from "
                "authenticated source"
            )
        live_metadata = (
            function.__defaults__,
            function.__kwdefaults__,
            function.__annotations__,
        )
        if not _exact_trusted_metadata_equal(
            live_metadata, expectation.method_metadata[name]
        ):
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} defaults or "
                "annotations changed from authenticated source"
            )
        _authenticate_parent_closure(function, live_class, label, name)
        expected_globals = referenced_globals.get(name)
        if expected_globals is None or (
            _parent_referenced_globals(function) != expected_globals
        ):
            raise RuntimeContractError(
                f"frozen parent class {label} method {name!r} referenced globals changed"
            )

def authenticate_parent_graph(
    parent: nn.Module, expected: AuthenticatedParentGraph
) -> None:

    observed = dict(parent.named_modules())
    if tuple(observed) != expected.traversal:
        raise RuntimeContractError("frozen parent module traversal changed")
    audited: dict[tuple[str, str], type] = {}
    for name, child in observed.items():
        owner = type(child)
        leaf = expected.external_leaves.get(name)
        if leaf is not None:
            leaf_class, leaf_forward = leaf
            if owner is not leaf_class or getattr(owner, "forward", None) is not (
                leaf_forward
            ):
                raise RuntimeContractError(
                    f"frozen parent module {name!r} external leaf changed"
                )
            continue
        key = expected.node_classes.get(name)
        if key is None or (owner.__module__, owner.__qualname__) != key:
            raise RuntimeContractError(f"frozen parent module {name!r} class changed")
        for ancestor in _audited_class_ancestry(owner):
            ancestor_key = (ancestor.__module__, ancestor.__qualname__)
            previous = audited.get(ancestor_key)
            if previous is not None:
                if previous is not ancestor:
                    raise RuntimeContractError(
                        f"frozen parent class {'.'.join(ancestor_key)} has two "
                        "counterfeit identities"
                    )
                continue
            expectation = expected.classes.get(ancestor_key)
            if expectation is None:
                raise RuntimeContractError(
                    f"frozen parent module {name!r} gained the unauthenticated "
                    f"ancestor class {'.'.join(ancestor_key)}"
                )
            _authenticate_project_class(
                ancestor,
                expectation,
                expected.realm,
                expected.method_globals.get(ancestor_key, {}),
            )
            audited[ancestor_key] = ancestor
    if frozenset(audited) != frozenset(expected.classes):
        raise RuntimeContractError("frozen parent authenticated class set changed")

def _installed_catalog_realm():

    finder = sys.meta_path[0] if sys.meta_path else None
    if not _is_installed_realm_finder(finder):
        raise RuntimeContractError(
            "frozen parent authentication requires the installed catalog realm"
        )
    return finder

class FrozenParentRuntime:

    __slots__ = (
        "_module",
        "_candidate",
        "_forward",
        "_parent_graph",
        "_execution_signature",
        "_diagnostic_signature",
        "_forward_calls",
    )

    def __init__(
        self,
        module: nn.Module,
        candidate: nn.Module,
        *,
        _token: object,
        _catalog: object | None = None,
    ) -> None:
        if _token not in (_PARENT_DESCRIPTOR_TOKEN, _PARENT_TEST_TOKEN):
            raise RuntimeContractError(
                "frozen parent construction requires the descriptor builder or private test helper"
            )
        if _catalog is None:
            raise RuntimeContractError(
                "parent construction requires the authenticated source catalog"
            )
        parent_class = _pinned_parent_class(_catalog_realm_and_source(_catalog)[0])
        if type(module) is not parent_class:
            raise RuntimeContractError(
                "frozen parent class is not the exact pinned denoiser"
            )
        _validate_parent_independence(candidate, module)
        self._module = module
        self._candidate = candidate
        self._forward = parent_class.forward
        self._parent_graph = compile_expected_parent_graph(
            _catalog,
            module,
            child_class_names=tuple(name for name, _ in module.named_modules())[1:],
        )
        authenticate_parent_graph(module, self._parent_graph)
        self._execution_signature = _module_execution_signature(module)
        self._diagnostic_signature = _diagnostic_execution_signature(module)
        self._forward_calls = 0

    @property
    def forward_calls(self) -> int:
        return self._forward_calls

    def _revalidate(self) -> None:
        parent_class = _pinned_parent_class(self._parent_graph.realm)
        authenticate_parent_graph(self._module, self._parent_graph)
        if (
            type(self._module) is not parent_class
            or parent_class.forward is not self._forward
            or _module_execution_signature(self._module) != self._execution_signature
            or _diagnostic_execution_signature(self._module)
            != self._diagnostic_signature
        ):
            raise RuntimeContractError(
                "frozen parent execution provenance changed or was overridden"
            )
        _validate_parent_independence(self._candidate, self._module)

    def execute(self, batch: Mapping[str, object]) -> object:
        self._revalidate()
        delegated: list[str] = []

        def observe_candidate(module: nn.Module, _args: tuple[object, ...]) -> None:
            delegated.append(type(module).__name__)

        handles = [
            module.register_forward_pre_hook(observe_candidate)
            for module in self._candidate.modules()
        ]
        try:
            self._forward_calls += 1
            result = self._forward(self._module, batch)
        finally:
            for handle in reversed(handles):
                handle.remove()
            self._revalidate()
        if delegated:
            raise RuntimeContractError(
                "frozen parent forward delegated into candidate execution"
            )
        return result

class DirectionalEvaluationRuntime:

    __slots__ = ("candidate_model", "_parent", "_bindings", "_execution_inputs")

    def __init__(
        self,
        candidate_model: nn.Module,
        parent_denoiser: FrozenParentRuntime,
        bindings: DirectionalProteinaBindings,
        *,
        _token: object,
        _execution_inputs: VerifiedBlindInputs | None = None,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise RuntimeContractError("runtime construction is not injectable")
        if type(getattr(candidate_model, "nn", None)) is not DirectionalV2Adapter:
            raise RuntimeContractError("candidate requires exact DirectionalV2Adapter")
        if type(bindings) is not DirectionalProteinaBindings:
            raise RuntimeContractError("candidate requires exact Proteina bindings")
        if type(parent_denoiser) is not FrozenParentRuntime:
            raise RuntimeContractError(
                "runtime requires evaluator-owned frozen parent provenance"
            )
        parent_denoiser._revalidate()
        self.candidate_model = candidate_model
        self._parent = parent_denoiser
        self._bindings = bindings
        self._execution_inputs = (
            None
            if _execution_inputs is None
            else VerifiedBlindInputs(
                _execution_inputs.seal_sha256,
                _execution_inputs.snapshots,
                MappingProxyType({}),
            )
        )

    def corrupt_once(self, raw_row: FlowRow) -> CorruptedFlowInput:
        row = validate_raw_row(raw_row)
        clean = row.request["payload"]["batch"]
        clean_clone = _clone_tree(clean)
        ledger = _ExecutionLedger()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(row.request["corruption_seed"]))
            ledger.corruptions += 1
            corrupted = self._bindings.corrupt(clean_clone)
        snapshot = VerifiedTensorTree.from_corruption(clean, corrupted)
        return CorruptedFlowInput(
            identity=row.identity,
            corruption_seed=42,
            corruption_digest=snapshot.digest(),
            input_digest=snapshot.field_digest("x_t"),
            time_digest=snapshot.field_digest("t"),
            target_digest=snapshot.field_digest("x_1"),
            mask_digest=snapshot.mask_digest(),
            _snapshot=snapshot,
            _ledger=ledger,
        )

    def evaluate_route(
        self, corrupted: CorruptedFlowInput, route: RouteState
    ) -> RouteFlowResult:
        if type(corrupted) is not CorruptedFlowInput:
            raise RuntimeContractError(
                "route evaluation requires exact corrupted input"
            )
        try:
            route = RouteState(route)
        except ValueError as error:
            raise RuntimeContractError(
                "route evaluation received unknown route"
            ) from error
        if route not in EXECUTABLE_ROUTE_STATES:
            raise RuntimeContractError("route evaluation received a derived control")
        batch = corrupted._snapshot.clone_for_forward()
        before = corrupted._snapshot.conditioning_digests()
        corrupted._ledger.routes += (route,)
        adapter = self.candidate_model.nn
        with torch.inference_mode(), adapter.route_context(route):
            observed = adapter._active_route.get()
            forward = adapter.forward_route(batch, route)
        if type(forward) is not DirectionalForward:
            raise RuntimeContractError("adapter returned wrong directional result type")
        return _route_result(
            corrupted,
            route,
            observed,
            forward,
            batch,
            self._bindings,
            before,
        )

    def evaluate_parent(self, corrupted: CorruptedFlowInput) -> FrozenParentResult:
        batch = corrupted._snapshot.clone_for_forward()
        before = corrupted._snapshot.conditioning_digests()
        corrupted._ledger.parent_forwards += 1
        with torch.inference_mode():
            result = self._parent.execute(batch)
        output = _prediction_mapping(result, self._bindings.output_key)
        values = _common_result_fields(corrupted, output, batch, self._bindings, before)
        return FrozenParentResult(
            *values[:11],
            values[12],
            values[13],
            corrupted._ledger.parent_forwards,
        )

def _module_storage_identities(
    module: nn.Module,
) -> tuple[set[int], set[int], set[int]]:
    modules = {id(item) for item in module.modules()}
    tensors = [*module.parameters(), *module.buffers()]
    identities = {id(item) for item in tensors}
    storages = {
        int(item.untyped_storage().data_ptr())
        for item in tensors
        if item.numel() and item.untyped_storage().data_ptr()
    }
    return modules, identities, storages

def _validate_parent_independence(candidate: nn.Module, parent: nn.Module) -> None:
    if not isinstance(parent, nn.Module) or isinstance(parent, DirectionalV2Adapter):
        raise RuntimeContractError("frozen parent must be an unwrapped module")
    forbidden = ("directionalv2adapter", "directionalbridge", "fourwayrouter", "lora")
    for name, module in parent.named_modules():
        identity = f"{name}.{type(module).__name__}".lower()
        if any(token in identity for token in forbidden):
            raise RuntimeContractError(
                "frozen parent contains adapter/bridge/router/LoRA"
            )
        if module.training:
            raise RuntimeContractError("frozen parent must remain in eval mode")
    if any(parameter.requires_grad for parameter in parent.parameters()):
        raise RuntimeContractError("frozen parent parameters must be frozen")
    if any("lora" in name.lower() for name, _ in parent.named_parameters()):
        raise RuntimeContractError("frozen parent contains LoRA parameters")
    candidate_sets = _module_storage_identities(candidate)
    parent_sets = _module_storage_identities(parent)
    if any(
        left & right for left, right in zip(candidate_sets, parent_sets, strict=True)
    ):
        raise RuntimeContractError("frozen parent is not storage-independent")

_PARENT_DIAGNOSTIC_AUDIT_COMMIT = "32b71ae1d9a8767414eae3921cf35969430db82f"
_PINNED_DIAGNOSTIC_SOURCE_SHA256 = MappingProxyType(
    {
        Path("src/proteinfoundation/nn/feature_factory/base_feature.py"): (
            "3aee22224963b917b5cd4647f11e112f8b2e79d859a3cc110f58c0d5826b0666"
        ),
        Path("src/proteinfoundation/nn/feature_factory/ligand_feats.py"): (
            "3412ece8c3e7659cee036ea897f331faf5d33196086c929bbbd78fda30bddb51"
        ),
        Path("src/proteinfoundation/nn/feature_factory/motif_feats.py"): (
            "2ed61d84fbc848ab19d7f8fd2c654992bd1ede2ca56e8c35d84e7101cd762f85"
        ),
        Path("src/proteinfoundation/nn/feature_factory/pair_feats.py"): (
            "af3f12c336ea6ac2cefadd5480dff26470e7a275935a9c2cbb8eb8f9fb996a21"
        ),
        Path("src/proteinfoundation/nn/feature_factory/seq_cond_feats.py"): (
            "6f65fce6dd1ae523a6859a455346c4638968e6c35b85d1f44bdae6a805f2e515"
        ),
        Path("src/proteinfoundation/nn/feature_factory/seq_feats.py"): (
            "a5b1b94a54a3b6f18117ab544af069b24c5da0f2d8c410d9475024e47c0de1ae"
        ),
        Path("src/proteinfoundation/nn/feature_factory/target_feats.py"): (
            "41ad9c0a8aa325ba2c872ba1f88f8be4aac7441f710ddc8cf4b752233879506a"
        ),
        Path("src/proteinfoundation/nn/old_feature_factory.py"): (
            "a4befe9a67d537891284cce2c8e9069f23a1b9846b9ea740b5f71843036ba632"
        ),
    }
)

def _self_has_logged(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == "_has_logged"
    )

def _diagnostic_assignment_value(node: ast.AST) -> bool | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _self_has_logged(node.targets[0])
        and isinstance(node.value, ast.Constant)
        and type(node.value.value) is bool
    ):
        return node.value.value
    if (
        isinstance(node, ast.AnnAssign)
        and _self_has_logged(node.target)
        and isinstance(node.value, ast.Constant)
        and type(node.value.value) is bool
    ):
        return node.value.value
    return None

def _logger_call_statement(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "logger"
    )

def _containing_function(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None

def _compiled_direct_method_code(
    source_bytes: bytes,
    expected: Path,
    module_class: type[nn.Module],
    class_node: ast.ClassDef,
) -> Mapping[str, CodeType]:
    try:
        compiled_module = compile(
            source_bytes,
            str(expected),
            "exec",
            dont_inherit=True,
        )
    except (SyntaxError, ValueError) as error:
        raise RuntimeContractError(
            "frozen parent diagnostic authenticated source cannot be compiled"
        ) from error
    class_codes = [
        value
        for value in compiled_module.co_consts
        if isinstance(value, CodeType)
        and value.co_name == module_class.__name__
        and value.co_qualname == module_class.__qualname__
    ]
    if len(class_codes) != 1:
        raise RuntimeContractError(
            "frozen parent diagnostic authenticated class code is not exact"
        )
    method_names = {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    methods: dict[str, CodeType] = {}
    for method_name in method_names:
        expected_qualname = f"{module_class.__qualname__}.{method_name}"
        matches = [
            value
            for value in class_codes[0].co_consts
            if isinstance(value, CodeType)
            and value.co_name == method_name
            and value.co_qualname == expected_qualname
        ]
        if len(matches) != 1:
            raise RuntimeContractError(
                "frozen parent diagnostic authenticated method code is not exact"
            )
        methods[method_name] = matches[0]
    return MappingProxyType(methods)

def _authenticated_code_fingerprint(code: CodeType) -> bytes:
    def frame(tag: bytes, *parts: bytes) -> bytes:
        return tag + b"".join(len(part).to_bytes(8, "big") + part for part in parts)

    def encode(value: object) -> bytes:
        if value is None:
            return b"N"
        if value is Ellipsis:
            return b"E"
        if type(value) is bool:
            return b"B1" if value else b"B0"
        if type(value) is int:
            return frame(b"I", str(value).encode("ascii"))
        if type(value) is float:
            return frame(b"F", struct.pack(">d", value))
        if type(value) is complex:
            return frame(b"J", struct.pack(">dd", value.real, value.imag))
        if type(value) is bytes:
            return frame(b"Y", value)
        if type(value) is str:
            return frame(b"S", value.encode("utf-8", "surrogatepass"))
        if type(value) is tuple:
            return frame(b"T", *(encode(item) for item in value))
        if type(value) is frozenset:
            return frame(b"R", *sorted(encode(item) for item in value))
        if isinstance(value, CodeType):
            return frame(
                b"C",
                encode(value.co_argcount),
                encode(value.co_posonlyargcount),
                encode(value.co_kwonlyargcount),
                encode(value.co_nlocals),
                encode(value.co_stacksize),
                encode(value.co_flags),
                encode(value.co_code),
                encode(value.co_consts),
                encode(value.co_names),
                encode(value.co_varnames),
                encode(value.co_filename),
                encode(value.co_name),
                encode(value.co_qualname),
                encode(value.co_firstlineno),
                encode(value.co_linetable),
                encode(value.co_exceptiontable),
                encode(value.co_freevars),
                encode(value.co_cellvars),
            )
        raise RuntimeContractError(
            "frozen parent diagnostic code contains an unsupported constant"
        )

    return hashlib.sha256(encode(code)).digest()

def _trusted_literal(node: ast.AST) -> object:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError) as error:
        raise RuntimeContractError(
            "frozen parent diagnostic method default is not an audited literal"
        ) from error

    def validate(nested: object) -> None:
        if (
            nested is None
            or nested is Ellipsis
            or type(nested)
            in (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
            )
        ):
            return
        if type(nested) in (tuple, list, set, frozenset):
            for item in nested:
                validate(item)
            return
        if type(nested) is dict:
            for key, item in nested.items():
                validate(key)
                validate(item)
            return
        raise RuntimeContractError(
            "frozen parent diagnostic method default has an unsupported value"
        )

    validate(value)
    return value

_TRUSTED_ANNOTATION_NAMES = MappingProxyType(
    {
        "bool": bool,
        "bytes": bytes,
        "complex": complex,
        "dict": dict,
        "float": float,
        "int": int,
        "list": list,
        "set": set,
        "str": str,
        "tuple": tuple,
    }
)

def _trusted_annotation(node: ast.AST, *, postponed: bool) -> object:
    if postponed:
        return ast.unparse(node)
    if isinstance(node, ast.Name) and node.id in _TRUSTED_ANNOTATION_NAMES:
        return _TRUSTED_ANNOTATION_NAMES[node.id]
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return node.value
    raise RuntimeContractError(
        "frozen parent diagnostic method annotation requires an explicit audit"
    )

def _trusted_method_metadata(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    postponed_annotations: bool,
) -> tuple[object, object, object]:
    if node.decorator_list:
        raise RuntimeContractError(
            "frozen parent diagnostic audited methods cannot be decorated"
        )
    defaults = (
        tuple(_trusted_literal(default) for default in node.args.defaults)
        if node.args.defaults
        else None
    )
    keyword_defaults = {
        argument.arg: _trusted_literal(default)
        for argument, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=True
        )
        if default is not None
    }
    annotations = {
        argument.arg: _trusted_annotation(
            argument.annotation, postponed=postponed_annotations
        )
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            *((node.args.vararg,) if node.args.vararg is not None else ()),
            *((node.args.kwarg,) if node.args.kwarg is not None else ()),
        )
        if argument.annotation is not None
    }
    if node.returns is not None:
        annotations["return"] = _trusted_annotation(
            node.returns, postponed=postponed_annotations
        )
    return (
        defaults,
        keyword_defaults or None,
        annotations,
    )

def _exact_trusted_metadata_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, type):
        return actual is expected
    if type(expected) is dict:
        unmatched = list(actual.items())
        for expected_key, expected_value in expected.items():
            matches = [
                index
                for index, (actual_key, actual_value) in enumerate(unmatched)
                if _exact_trusted_metadata_equal(actual_key, expected_key)
                and _exact_trusted_metadata_equal(actual_value, expected_value)
            ]
            if len(matches) != 1:
                return False
            unmatched.pop(matches[0])
        return not unmatched
    if type(expected) in (tuple, list):
        return len(actual) == len(expected) and all(
            _exact_trusted_metadata_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if type(expected) in (set, frozenset):
        unmatched = list(actual)
        for expected_item in expected:
            matches = [
                index
                for index, actual_item in enumerate(unmatched)
                if _exact_trusted_metadata_equal(actual_item, expected_item)
            ]
            if len(matches) != 1:
                return False
            unmatched.pop(matches[0])
        return not unmatched
    if type(expected) is float:
        return struct.pack(">d", actual) == struct.pack(">d", expected)
    if type(expected) is complex:
        return struct.pack(">dd", actual.real, actual.imag) == struct.pack(
            ">dd", expected.real, expected.imag
        )
    return actual == expected

def _trusted_metadata_signature(value: object) -> tuple[object, ...]:
    value_type = type(value)
    if value is None or value is Ellipsis or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (float, struct.pack(">d", value))
    if value_type is complex:
        return (complex, struct.pack(">dd", value.real, value.imag))
    if isinstance(value, type):
        return ("type", id(value), value.__module__, value.__qualname__)
    if value_type in (tuple, list):
        return (
            value_type,
            tuple(_trusted_metadata_signature(item) for item in value),
        )
    if value_type is dict:
        entries = [
            (
                _trusted_metadata_signature(key),
                _trusted_metadata_signature(item),
            )
            for key, item in value.items()
        ]
        return (dict, tuple(sorted(entries, key=repr)))
    if value_type in (set, frozenset):
        entries = [_trusted_metadata_signature(item) for item in value]
        return (value_type, tuple(sorted(entries, key=repr)))
    raise RuntimeContractError(
        "frozen parent diagnostic method metadata has an unsupported value"
    )

def _authenticated_closure_signature(
    method: FunctionType, owner: type[nn.Module]
) -> tuple[object, ...]:
    freevars = method.__code__.co_freevars
    closure = method.__closure__
    if not freevars:
        if closure is not None:
            raise RuntimeContractError(
                "frozen parent diagnostic method has a counterfeit closure"
            )
        return ()
    if closure is None or len(closure) != len(freevars):
        raise RuntimeContractError(
            "frozen parent diagnostic method closure does not match its code"
        )
    signature = []
    for name, cell in zip(freevars, closure, strict=True):
        if name != "__class__":
            raise RuntimeContractError(
                "frozen parent diagnostic method has an unsupported closure"
            )
        try:
            contents = cell.cell_contents
        except ValueError as error:
            raise RuntimeContractError(
                "frozen parent diagnostic method has an empty class closure"
            ) from error
        if contents is not owner:
            raise RuntimeContractError(
                "frozen parent diagnostic method class closure has a counterfeit owner"
            )
        signature.append((name, id(cell), id(contents)))
    return tuple(signature)

def _audit_authenticated_feature_class(
    module_class: type[nn.Module],
) -> tuple[bool, tuple[object, ...]]:
    if EMERGENT_UPSTREAM_COMMIT != _PARENT_DIAGNOSTIC_AUDIT_COMMIT:
        raise RuntimeContractError(
            "frozen parent diagnostic audit is not tied to the selected upstream commit"
        )
    expected = (
        EMERGENT_UPSTREAM_ROOT / "src" / Path(*module_class.__module__.split("."))
    ).with_suffix(".py")
    try:
        relative = expected.resolve().relative_to(EMERGENT_UPSTREAM_ROOT.resolve())
        source = inspect.getsourcefile(module_class)
        source_bytes = expected.read_bytes()
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeContractError(
            "frozen parent diagnostic class source cannot be audited"
        ) from error
    expected_sha256 = _PINNED_DIAGNOSTIC_SOURCE_SHA256.get(relative)
    if (
        source is None
        or Path(source).resolve() != expected.resolve()
        or expected_sha256 is None
    ):
        raise RuntimeContractError(
            "frozen parent diagnostic state has an unaudited class/source"
        )
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
        raise RuntimeContractError(
            "frozen parent diagnostic source digest differs from the pinned commit"
        )
    try:
        parsed = ast.parse(source_bytes, filename=str(expected))
    except SyntaxError as error:
        raise RuntimeContractError(
            "frozen parent diagnostic class source cannot be parsed"
        ) from error
    classes = [
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == module_class.__name__
    ]
    if len(classes) != 1 or classes[0].name != module_class.__name__:
        raise RuntimeContractError("frozen parent diagnostic class source is not exact")
    class_node = classes[0]
    function_nodes = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    method_names = [node.name for node in function_nodes]
    if len(method_names) != len(set(method_names)):
        raise RuntimeContractError(
            "frozen parent diagnostic authenticated methods are not unique"
        )
    compiled_methods = _compiled_direct_method_code(
        source_bytes, expected, module_class, class_node
    )
    live_methods = {
        name: member
        for name, member in vars(module_class).items()
        if isinstance(member, FunctionType)
    }
    if set(live_methods) != set(compiled_methods):
        raise RuntimeContractError(
            "frozen parent diagnostic class has extra or missing direct methods"
        )
    if any(
        isinstance(member, (staticmethod, classmethod, property))
        for member in vars(module_class).values()
    ):
        raise RuntimeContractError(
            "frozen parent diagnostic class has an unsupported executable descriptor"
        )
    postponed_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in parsed.body
    )
    function_nodes_by_name = {node.name: node for node in function_nodes}
    canonical_module = sys.modules.get(module_class.__module__)
    if (
        not isinstance(canonical_module, ModuleType)
        or vars(canonical_module).get(module_class.__name__) is not module_class
    ):
        raise RuntimeContractError(
            "frozen parent diagnostic class has no canonical module globals"
        )
    canonical_globals = vars(canonical_module)
    method_signatures = []
    for method_name, expected_code in compiled_methods.items():
        live_method = live_methods[method_name]
        if live_method.__globals__ is not canonical_globals:
            raise RuntimeContractError(
                "frozen parent diagnostic method uses counterfeit globals"
            )
        if _authenticated_code_fingerprint(
            live_method.__code__
        ) != _authenticated_code_fingerprint(expected_code):
            raise RuntimeContractError(
                "frozen parent diagnostic class executed code differs from "
                "authenticated source"
            )
        expected_metadata = _trusted_method_metadata(
            function_nodes_by_name[method_name],
            postponed_annotations=postponed_annotations,
        )
        live_metadata = (
            live_method.__defaults__,
            live_method.__kwdefaults__,
            live_method.__annotations__,
        )
        if not _exact_trusted_metadata_equal(live_metadata, expected_metadata):
            raise RuntimeContractError(
                "frozen parent diagnostic method defaults or annotations differ "
                "from authenticated source"
            )
        method_signatures.append(
            (
                method_name,
                id(live_method),
                id(live_method.__code__),
                _authenticated_code_fingerprint(live_method.__code__),
                id(live_method.__globals__),
                _trusted_metadata_signature(live_metadata),
                _authenticated_closure_signature(live_method, module_class),
            )
        )
    parents = {
        child: parent
        for parent in ast.walk(class_node)
        for child in ast.iter_child_nodes(parent)
    }
    attributes = [node for node in ast.walk(class_node) if _self_has_logged(node)]
    if not attributes:
        return (
            False,
            (
                "authenticated-diagnostic-class",
                id(module_class),
                id(canonical_globals),
                tuple(method_signatures),
            ),
        )
    gates: set[ast.If] = set()
    saw_false_initialization = False
    true_assignments: set[ast.AST] = set()
    for attribute in attributes:
        parent = parents.get(attribute)
        if isinstance(attribute.ctx, ast.Load):
            gate_test = parent
            gate = parents.get(gate_test) if gate_test is not None else None
            if (
                not isinstance(gate_test, ast.UnaryOp)
                or not isinstance(gate_test.op, ast.Not)
                or gate_test.operand is not attribute
                or not isinstance(gate, ast.If)
                or gate.test is not gate_test
            ):
                raise RuntimeContractError(
                    "frozen parent diagnostic state is used outside a logging gate"
                )
            gates.add(gate)
            continue
        assignment = parent
        assigned = (
            _diagnostic_assignment_value(assignment) if assignment is not None else None
        )
        if assigned is None:
            raise RuntimeContractError(
                "frozen parent diagnostic state has a non-boolean assignment"
            )
        if assigned:
            true_assignments.add(assignment)
        else:
            function = _containing_function(assignment, parents)
            if function is None or function.name != "__init__":
                raise RuntimeContractError(
                    "frozen parent diagnostic state resets outside construction"
                )
            saw_false_initialization = True
    gated_assignments: set[ast.AST] = set()
    for gate in gates:
        assignments = [
            statement
            for statement in gate.body
            if _diagnostic_assignment_value(statement) is True
        ]
        if (
            len(assignments) != 1
            or not any(_logger_call_statement(statement) for statement in gate.body)
            or any(
                not _logger_call_statement(statement)
                and _diagnostic_assignment_value(statement) is not True
                for statement in gate.body
            )
        ):
            raise RuntimeContractError(
                "frozen parent diagnostic gate does more than one-time logging"
            )
        gated_assignments.update(assignments)
    if (
        not gates
        or not saw_false_initialization
        or not true_assignments
        or true_assignments != gated_assignments
    ):
        raise RuntimeContractError(
            "frozen parent diagnostic state lacks exact logging lifecycle"
        )
    return (
        True,
        (
            "authenticated-diagnostic-class",
            id(module_class),
            id(canonical_globals),
            tuple(method_signatures),
        ),
    )

def _code_mentions_diagnostic_state(code: CodeType) -> bool:
    return "_has_logged" in code.co_names or any(
        (type(value) is str and value == "_has_logged")
        or (isinstance(value, CodeType) and _code_mentions_diagnostic_state(value))
        for value in code.co_consts
    )

def _direct_executable_codes(module_class: type[object]) -> tuple[CodeType, ...]:
    codes: list[CodeType] = []
    for member in vars(module_class).values():
        if isinstance(member, FunctionType):
            codes.append(member.__code__)
        elif isinstance(member, (staticmethod, classmethod)):
            codes.append(member.__func__.__code__)
        elif isinstance(member, property):
            codes.extend(
                function.__code__
                for function in (member.fget, member.fset, member.fdel)
                if isinstance(function, FunctionType)
            )
    return tuple(codes)

def _audit_pinned_diagnostic_class(module: nn.Module) -> tuple[object, ...]:
    module_class = type(module)
    mro = module_class.__mro__
    try:
        torch_module_index = mro.index(nn.Module)
    except ValueError as error:
        raise RuntimeContractError(
            "frozen parent diagnostic owner does not have the audited module MRO"
        ) from error
    if mro[torch_module_index:] != (nn.Module, object):
        raise RuntimeContractError(
            "frozen parent diagnostic owner has an unsupported module MRO"
        )
    authenticated_classes = tuple(
        (audited_class, *_audit_authenticated_feature_class(audited_class))
        for audited_class in mro[:torch_module_index]
    )
    logging_owners = tuple(
        audited_class
        for audited_class, owns_logging_state, _signature in authenticated_classes
        if owns_logging_state
    )
    if module_class not in logging_owners:
        raise RuntimeContractError(
            "frozen parent diagnostic class does not own audited logging state"
        )
    for inherited_class in mro[torch_module_index:]:
        if any(
            _code_mentions_diagnostic_state(code)
            for code in _direct_executable_codes(inherited_class)
        ):
            raise RuntimeContractError(
                "frozen parent diagnostic state is used by an unaudited inherited method"
            )
    return tuple(
        signature
        for _audited_class, _owns_logging_state, signature in authenticated_classes
    )

def _diagnostic_execution_signature(parent: nn.Module) -> tuple[object, ...]:
    signatures = []
    for name, module in parent.named_modules():
        if "_has_logged" not in vars(module):
            continue
        if type(module._has_logged) is not bool:
            raise RuntimeContractError(
                "frozen parent diagnostic state must be an exact bool"
            )
        signatures.append((name, id(module), _audit_pinned_diagnostic_class(module)))
    return tuple(signatures)

def _normalize_parent_diagnostic_state(parent: nn.Module) -> None:
    for module in parent.modules():
        if "_has_logged" not in vars(module):
            continue
        if type(module._has_logged) is not bool:
            raise RuntimeContractError(
                "frozen parent diagnostic state must be an exact bool"
            )
        _audit_pinned_diagnostic_class(module)
        module._has_logged = True

def _test_only_runtime(
    candidate: nn.Module,
    parent: nn.Module,
    bindings: DirectionalProteinaBindings,
    catalog: object,
    *,
    normalize_diagnostic_state: bool = False,
) -> DirectionalEvaluationRuntime:

    if normalize_diagnostic_state:
        _normalize_parent_diagnostic_state(parent)
    frozen_parent = FrozenParentRuntime(
        parent, candidate, _token=_PARENT_TEST_TOKEN, _catalog=catalog
    )
    return DirectionalEvaluationRuntime(
        candidate, frozen_parent, bindings, _token=_CONSTRUCTION_TOKEN
    )

def _prediction_mapping(value: object, output_key: str) -> tuple[Tensor, Tensor]:
    result = _exact_mapping("denoiser prediction", value)
    _exact_keys("denoiser prediction", result, set(_MODALITIES))
    tensors = []
    for mode in _MODALITIES:
        nested = _exact_mapping(f"denoiser prediction.{mode}", result[mode])
        _exact_keys(f"denoiser prediction.{mode}", nested, {output_key})
        tensors.append(_clone_tensor(nested[output_key], name=f"prediction.{mode}"))
    return tensors[0], tensors[1]

def _conditioning_mutations(
    before: Mapping[str, str], batch: Mapping[str, object]
) -> int:
    after = {key: _tree_digest(batch[key]) for key in before if key in batch}
    return len(set(before) ^ set(after)) + sum(
        before[key] != after[key] for key in set(before) & set(after)
    )

def _loss_fields(
    x: Tensor,
    z: Tensor,
    batch: Mapping[str, object],
    bindings: DirectionalProteinaBindings,
) -> tuple[float, int, tuple[float, ...], tuple[str, ...]]:
    per_residue = bindings.per_residue_flow_loss(x, z, batch)
    if type(per_residue) is not Tensor or per_residue.ndim != 2:
        raise RuntimeContractError(
            "product-space helper must return exact [B,N] Tensor"
        )
    valid = batch["mask"] & batch["generated_mask"]
    if tuple(per_residue.shape) != tuple(valid.shape):
        raise RuntimeContractError("product-space loss shape differs from exact mask")
    if not torch.is_floating_point(x) or not torch.is_floating_point(z):
        raise RuntimeContractError("product-space predictions must be floating point")
    events = [
        *_finite_events_residue_tensor("prediction.bb_ca", x, valid),
        *_finite_events_residue_tensor("prediction.local_latents", z, valid),
        *_finite_events_residue_tensor("per_residue_loss", per_residue, valid),
    ]
    selected = per_residue.masked_select(valid)
    finite_selected = selected[torch.isfinite(selected)]
    loss_sum = float(finite_selected.double().sum().item())
    if not math.isfinite(loss_sum) or loss_sum < 0:
        raise RuntimeContractError("finite product-space loss sum must be nonnegative")
    return (
        loss_sum,
        int(valid.sum().item()),
        tuple(
            float(value) if math.isfinite(float(value)) else 0.0
            for value in per_residue[0]
        ),
        tuple(events),
    )

def _common_result_fields(
    corrupted: CorruptedFlowInput,
    output: tuple[Tensor, Tensor],
    batch: Mapping[str, object],
    bindings: DirectionalProteinaBindings,
    before: Mapping[str, str],
) -> tuple[object, ...]:
    x, z = output
    targets = batch["x_1"]
    target_x = targets["bb_ca"]
    target_z = targets["local_latents"]
    loss_sum, denominator, per_residue, events = _loss_fields(x, z, batch, bindings)
    return (
        corrupted.corruption_digest,
        _tree_digest(batch["x_t"]),
        _tree_digest(batch["t"]),
        _tree_digest(batch["x_1"]),
        _tree_digest(
            {key: batch[key] for key in ("mask", "generated_mask", "fixed_mask")}
        ),
        (tuple(x.shape), tuple(z.shape)),
        (tuple(target_x.shape), tuple(target_z.shape)),
        (str(x.dtype), str(z.dtype)),
        (str(target_x.dtype), str(target_z.dtype)),
        loss_sum,
        denominator,
        per_residue,
        events,
        _conditioning_mutations(before, batch),
    )

def _route_result(
    corrupted: CorruptedFlowInput,
    requested: RouteState,
    observed: RouteState,
    forward: DirectionalForward,
    batch: Mapping[str, object],
    bindings: DirectionalProteinaBindings,
    before: Mapping[str, str],
) -> RouteFlowResult:
    output = _prediction_mapping(forward.nn_out, bindings.output_key)
    common = _common_result_fields(corrupted, output, batch, bindings, before)
    x, z = output
    loss_sum, denominator, per_residue, loss_events = common[9:13]
    valid = batch["mask"] & batch["generated_mask"]
    telemetry: list[TelemetryEvidence] = []
    telemetry_events: list[str] = []
    simplex = hard_one_hot = none_residual = 0
    expected_index = None
    if requested not in (RouteState.ADAPTIVE, RouteState.NONE):
        expected_index = RouteState.hard_states().index(requested)
    for index, item in enumerate(tuple(forward.telemetry)):
        probabilities = item.probabilities
        zx = item.applied_z_to_x
        xz = item.applied_x_to_z
        if tuple(probabilities.shape) != (*tuple(valid.shape), 4):
            raise RuntimeContractError("telemetry probabilities have wrong shape")
        if (
            zx.ndim != 3
            or xz.ndim != 3
            or tuple(zx.shape[:2]) != tuple(valid.shape)
            or tuple(xz.shape[:2]) != tuple(valid.shape)
        ):
            raise RuntimeContractError("telemetry residual has wrong residue shape")
        if not all(torch.is_floating_point(value) for value in (probabilities, zx, xz)):
            raise RuntimeContractError("telemetry tensors must be floating point")
        telemetry_events.extend(
            _finite_events_residue_tensor(
                f"telemetry[{index}].probabilities", probabilities, valid
            )
        )
        telemetry_events.extend(
            _finite_events_residue_tensor(f"telemetry[{index}].z_to_x", zx, valid)
        )
        telemetry_events.extend(
            _finite_events_residue_tensor(f"telemetry[{index}].x_to_z", xz, valid)
        )
        finite_probs = torch.nan_to_num(probabilities)
        selected_probs = finite_probs.masked_select(
            valid[..., None].expand_as(finite_probs)
        ).reshape(-1, 4)
        simplex += int(((selected_probs < 0) | (selected_probs > 1)).sum().item())
        simplex += int((torch.abs(selected_probs.sum(-1) - 1.0) > 1e-6).sum().item())
        if expected_index is not None:
            expected = torch.zeros_like(selected_probs)
            expected[..., expected_index] = 1.0
            hard_one_hot += int(torch.count_nonzero(selected_probs != expected).item())
        zx_selected = torch.nan_to_num(zx).masked_select(valid[..., None].expand_as(zx))
        xz_selected = torch.nan_to_num(xz).masked_select(valid[..., None].expand_as(xz))
        telemetry.append(
            TelemetryEvidence(
                tuple(float(value) for value in selected_probs.double().sum(0)),
                _masked_squared_sum(zx, valid),
                _masked_squared_sum(xz, valid),
            )
        )
        if requested is RouteState.NONE:
            none_residual += int(torch.count_nonzero(zx_selected).item())
            none_residual += int(torch.count_nonzero(xz_selected).item())
    if requested is RouteState.NONE:
        none_residual += int(bool(telemetry))
    if requested not in (RouteState.ADAPTIVE, RouteState.NONE) and len(telemetry) != 3:
        hard_one_hot += abs(len(telemetry) - 3) or 1
    return RouteFlowResult(
        requested,
        observed,
        *common[:9],
        loss_sum,
        denominator,
        tuple((*loss_events, *telemetry_events)),
        per_residue,
        _masked_squared_sum(x, valid),
        _masked_squared_sum(z, valid),
        tuple(telemetry),
        simplex,
        hard_one_hot,
        none_residual,
        int(common[-1]),
    )

def evaluate_one_flow_row(
    runtime: DirectionalEvaluationRuntime, row: FlowRow
) -> RowFlowEvidence:
    if type(runtime) is not DirectionalEvaluationRuntime:
        raise RuntimeContractError("flow evaluator requires exact runtime")
    corrupted = runtime.corrupt_once(row)
    routes = tuple(
        runtime.evaluate_route(corrupted, route) for route in EXECUTABLE_ROUTE_STATES
    )
    parent = runtime.evaluate_parent(corrupted)
    valid = corrupted._snapshot.valid_mask()[0]
    return RowFlowEvidence(
        identity=corrupted.identity,
        corruption_seed=corrupted.corruption_seed,
        corruption_digest=corrupted.corruption_digest,
        input_digest=corrupted.input_digest,
        time_digest=corrupted.time_digest,
        target_digest=corrupted.target_digest,
        mask_digest=corrupted.mask_digest,
        valid_mask=tuple(bool(value) for value in valid),
        times=corrupted._snapshot.times(),
        routes=routes,
        frozen_parent=parent,
        corruption_count=corrupted._ledger.corruptions,
        route_calls=corrupted._ledger.routes,
        conditioning_elements=corrupted._snapshot.conditioning_elements(),
    )

def _validity(evidence: RowFlowEvidence) -> dict[str, object]:
    routes = evidence.routes
    expected = EXECUTABLE_ROUTE_STATES
    denominator = sum(evidence.valid_mask)
    parent = evidence.frozen_parent
    results: tuple[RouteFlowResult | FrozenParentResult, ...] = (*routes, parent)
    digests = {
        "corruption_digest": evidence.corruption_digest,
        "input_digest": evidence.input_digest,
        "time_digest": evidence.time_digest,
        "target_digest": evidence.target_digest,
        "mask_digest": evidence.mask_digest,
    }
    adaptive = next(
        (item for item in routes if item.requested_route is RouteState.ADAPTIVE), None
    )
    return {
        "corruption_count": evidence.corruption_count,
        "route_order": [route.value for route in evidence.route_calls],
        "route_call_counts": {
            route.value: evidence.route_calls.count(route) for route in expected
        },
        **{
            f"{name}_mismatch_count": sum(
                getattr(result, name) != value for result in results
            )
            for name, value in digests.items()
        },
        "observed_route_mismatch_count": sum(
            item.observed_route is not item.requested_route for item in routes
        )
        + int(tuple(item.requested_route for item in routes) != expected),
        "shape_mismatch_count": sum(
            item.prediction_shapes != item.target_shapes for item in results
        ),
        "dtype_mismatch_count": sum(
            item.prediction_dtypes != item.target_dtypes for item in results
        ),
        "denominator_mismatch_count": sum(
            item.valid_generated_residues != denominator for item in results
        ),
        "valid_generated_residues": denominator,
        "non_finite_count": sum(item.non_finite_count for item in results),
        "adaptive_tap_count": 0 if adaptive is None else len(adaptive.telemetry),
        "adaptive_simplex_violation_count": 0
        if adaptive is None
        else adaptive.adaptive_simplex_violation_count,
        "hard_one_hot_violation_count": sum(
            item.hard_one_hot_violation_count for item in routes
        ),
        "none_bridge_residual_nonzero_count": sum(
            item.none_bridge_residual_nonzero_count for item in routes
        ),
    }

def _time_records(
    evidence: RowFlowEvidence,
) -> tuple[dict[str, float], dict[str, object]]:
    bb, zz = evidence.times

    def index(value: float) -> int:
        return sum(value > edge for edge in TIME_BIN_EDGES[1:-1])

    bb_index, zz_index = index(bb), index(zz)
    return (
        {"bb_ca": bb, "local_latents": zz},
        {
            "scheme": TIME_BIN_SCHEME,
            "edges": list(TIME_BIN_EDGES),
            "bb_ca_index": bb_index,
            "local_latents_index": zz_index,
            "index": bb_index * 5 + zz_index,
        },
    )

def _flow_example(evidence: RowFlowEvidence) -> dict[str, object]:
    identity = evidence.identity.to_mapping()
    by_route = {item.requested_route: item for item in evidence.routes}
    adaptive = by_route[RouteState.ADAPTIVE]
    denominator = sum(evidence.valid_mask)
    hard = [by_route[route] for route in RouteState.hard_states()]
    oracle = sum(
        min(item.per_residue_loss[index] for item in hard)
        for index, selected in enumerate(evidence.valid_mask)
        if selected
    )
    corruption_time, time_bin = _time_records(evidence)
    probabilities = {}
    residuals = {}
    for index, tap in enumerate(adaptive.telemetry):
        probabilities[f"tap_{index}"] = {
            route.value: {
                "numerator": tap.probability_numerators[route_index],
                "denominator": denominator,
            }
            for route_index, route in enumerate(RouteState.hard_states())
        }
        residuals[f"tap_{index}"] = {
            "z_to_x": {
                "applied_residual_squared_sum": tap.z_to_x_squared_sum,
                "joint_field_squared_norm_sum": max(
                    adaptive.prediction_x_squared_sum, 1e-30
                ),
                "valid_denominator": denominator,
            },
            "x_to_z": {
                "applied_residual_squared_sum": tap.x_to_z_squared_sum,
                "joint_field_squared_norm_sum": max(
                    adaptive.prediction_z_squared_sum, 1e-30
                ),
                "valid_denominator": denominator,
            },
        }
    return {
        **identity,
        "family_hash": hashlib.sha256(identity["family"].encode()).hexdigest(),
        "example_id_hash": hashlib.sha256(identity["example_id"].encode()).hexdigest(),
        "parent_id_hash": hashlib.sha256(identity["parent_id"].encode()).hexdigest(),
        "identity_hash": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        ).hexdigest(),
        "route_losses": {
            route.value: {
                "numerator": by_route[route].loss_sum,
                "denominator": denominator,
            }
            for route in EXECUTABLE_ROUTE_STATES
        },
        "per_residue_oracle": {"numerator": oracle, "denominator": denominator},
        "corruption_time": corruption_time,
        "time_bin": time_bin,
        "tap_probabilities": probabilities,
        "tap_residuals": residuals,
        "derivable_controls": [
            "per_residue_oracle",
            "best_fixed",
            "global",
            "time_only",
        ],
        "non_finite_count": adaptive.non_finite_count,
        "non_finite_events": [
            {"field": event.split(":", 1)[0], "value": event.split(":", 1)[-1]}
            for event in adaptive.non_finite_events
        ],
    }

def _blind_check(evidence: RowFlowEvidence) -> dict[str, object]:
    none = next(
        item for item in evidence.routes if item.requested_route is RouteState.NONE
    )
    parent = evidence.frozen_parent
    return {
        "schema_version": CHECK_SCHEMA_VERSION,
        "identity": evidence.identity.to_mapping(),
        "validity": _validity(evidence),
        "preservation": {
            "candidate_none_loss_sum": none.loss_sum,
            "frozen_parent_loss_sum": parent.loss_sum,
            "valid_generated_residues": none.valid_generated_residues,
            "conditioning_mutation_count": sum(
                item.conditioning_mutation_count for item in evidence.routes
            )
            + parent.conditioning_mutation_count,
            "conditioning_elements": evidence.conditioning_elements,
            "candidate_none_non_finite_count": none.non_finite_count,
            "frozen_parent_non_finite_count": parent.non_finite_count,
        },
    }

def flow_records(
    runtime: DirectionalEvaluationRuntime, rows: Sequence[FlowRow]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    examples, checks = [], []
    for row in rows:
        evidence = evaluate_one_flow_row(runtime, row)
        examples.append(_flow_example(evidence))
        checks.append(_blind_check(evidence))
    return examples, checks

_GENERATION_POLICY_KEYS = frozenset(
    {
        "schema",
        "version",
        "nsteps",
        "guidance_w",
        "ag_ratio",
        "n_recycle",
        "search_algorithm",
        "self_cond",
        "model",
    }
)

def _config_value(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        try:
            return value[key]
        except KeyError as error:
            raise RuntimeContractError(
                f"candidate generation config is missing {key!r}"
            ) from error
    try:
        return getattr(value, key)
    except AttributeError as error:
        raise RuntimeContractError(
            f"candidate generation config is missing {key!r}"
        ) from error

def _json_config_value(value: object, *, name: str) -> object:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise RuntimeContractError(f"{name} config keys must be exact strings")
        return {
            key: _json_config_value(nested, name=f"{name}.{key}")
            for key, nested in value.items()
        }
    if type(value) in (list, tuple):
        return [
            _json_config_value(nested, name=f"{name}[{index}]")
            for index, nested in enumerate(value)
        ]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise RuntimeContractError(f"{name} contains a non-JSON config value")

def _generation_policy_from_cfg(cfg: object) -> dict[str, object]:
    generation = _config_value(cfg, "generation")
    args = _config_value(generation, "args")
    self_cond = _config_value(args, "self_cond")
    if type(self_cond) is not bool:
        raise RuntimeContractError("candidate generation self_cond must be boolean")
    model = _json_config_value(
        _config_value(generation, "model"), name="candidate generation model"
    )
    if type(model) is not dict or not model:
        raise RuntimeContractError("candidate generation model config must be nonempty")
    return {
        "schema": "dive-directional-generation-policy-v1",
        "version": 1,
        "nsteps": 100,
        "guidance_w": 1.0,
        "ag_ratio": 0.0,
        "n_recycle": 0,
        "search_algorithm": "single-pass",
        "self_cond": self_cond,
        "model": model,
    }

def _validate_generation_policy(value: object) -> Mapping[str, object]:
    policy = _exact_mapping("generation policy", value)
    _exact_keys("generation policy", policy, _GENERATION_POLICY_KEYS)
    exact = {
        "schema": "dive-directional-generation-policy-v1",
        "version": 1,
        "nsteps": 100,
        "guidance_w": 1.0,
        "ag_ratio": 0.0,
        "n_recycle": 0,
        "search_algorithm": "single-pass",
    }
    if any(policy[key] != expected for key, expected in exact.items()):
        raise RuntimeContractError("generation policy differs from frozen settings")
    if type(policy["self_cond"]) is not bool:
        raise RuntimeContractError("generation policy self_cond must be boolean")
    model = _json_config_value(policy["model"], name="generation policy model")
    if type(model) is not dict or not model:
        raise RuntimeContractError("generation policy model must be nonempty")
    return policy

def _generation_policy_from_inputs(
    inputs: VerifiedBlindInputs,
) -> dict[str, object]:
    path = _descriptor_path(inputs, "common_checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["hyper_parameters"]["cfg_exp"]
    except Exception as error:
        raise RuntimeContractError(
            "cannot read candidate generation policy from retained common checkpoint"
        ) from error
    return _generation_policy_from_cfg(cfg)

def _representative_record(
    records: Sequence[Mapping[str, object]], family: str, parent_id: str
) -> Mapping[str, object]:
    candidates = [
        record
        for record in records
        if record.get("family") == family and record.get("parent_id") == parent_id
    ]
    if not candidates:
        raise RuntimeContractError(
            "generation parent has no strict-loader-manifest representative"
        )
    candidates.sort(
        key=lambda row: (str(row.get("cell_hash")), str(row.get("example_id")))
    )
    selected_key = (
        str(candidates[0].get("cell_hash")),
        str(candidates[0].get("example_id")),
    )
    if (
        sum(
            (str(row.get("cell_hash")), str(row.get("example_id"))) == selected_key
            for row in candidates
        )
        != 1
    ):
        raise RuntimeContractError(
            "generation representative minimum must select exactly one record"
        )
    return candidates[0]

def _set_config_value(value: object, key: str, replacement: object) -> None:
    if isinstance(value, Mapping):
        value[key] = replacement
    else:
        setattr(value, key, replacement)

def _generation_config(candidate: nn.Module, policy: Mapping[str, object]) -> object:
    generation = deepcopy(_config_value(candidate.cfg_exp, "generation"))
    try:
        from omegaconf import OmegaConf, open_dict

        context = open_dict(generation) if OmegaConf.is_config(generation) else None
    except ImportError:
        context = None

    def configure() -> None:
        args = _config_value(generation, "args")
        for key in ("nsteps", "guidance_w", "ag_ratio", "self_cond"):
            _set_config_value(args, key, policy[key])
        _set_config_value(generation, "n_recycle", 0)
        try:
            search = _config_value(generation, "search")
        except RuntimeContractError:
            search = {}
            _set_config_value(generation, "search", search)
        _set_config_value(search, "algorithm", "single-pass")

    if context is None:
        configure()
    else:
        with context:
            configure()
    if _generation_policy_from_cfg(
        type("Cfg", (), {"generation": generation})()
    ) != dict(policy):
        raise RuntimeContractError(
            "independent generation config differs from sealed candidate policy"
        )
    return generation

def _pinned_generate_method():
    return _pinned_proteina_class().generate

def _pinned_proteina_class():
    from proteinfoundation.proteina import Proteina

    expected = (
        EMERGENT_UPSTREAM_ROOT / "src" / "proteinfoundation" / "proteina.py"
    ).resolve()
    source = inspect.getsourcefile(Proteina)
    if (
        Proteina.__module__ != "proteinfoundation.proteina"
        or source is None
        or Path(source).resolve() != expected
        or Path(Proteina.generate.__code__.co_filename).resolve() != expected
        or Path(Proteina.configure_inference.__code__.co_filename).resolve() != expected
    ):
        raise RuntimeContractError(
            "candidate Proteina class source differs from the pinned checkout"
        )
    return Proteina

def _assert_pinned_generate_binding(candidate: nn.Module) -> None:
    Proteina = _pinned_proteina_class()
    if (
        type(candidate) is not Proteina
        or "generate" in vars(candidate)
        or "configure_inference" in vars(candidate)
        or Proteina.generate is not _pinned_generate_method()
    ):
        raise RuntimeContractError(
            "candidate generation type/class binding is not exact pinned Proteina"
        )

def _generation_class_functions(
    candidate: nn.Module, *, require_pinned: bool
) -> tuple[object, object]:
    if "generate" in vars(candidate) or "configure_inference" in vars(candidate):
        raise RuntimeContractError(
            "candidate generation methods must not have instance overrides"
        )
    if require_pinned:
        _assert_pinned_generate_binding(candidate)
    candidate_type = type(candidate)
    configure = getattr(candidate_type, "configure_inference", None)
    generate = getattr(candidate_type, "generate", None)
    if not callable(configure) or not callable(generate):
        raise RuntimeContractError("candidate class lacks exact generation methods")
    return configure, generate

def _generation_conditioning_digests(batch: Mapping[str, object]) -> dict[str, str]:
    return {key: _tree_digest(value) for key, value in batch.items()}

def _candidate_device(candidate: nn.Module) -> torch.device:
    devices = {
        value.device
        for value in (*candidate.parameters(), *candidate.buffers())
        if value.numel()
    }
    if len(devices) != 1:
        raise RuntimeContractError(
            "candidate generation requires one unambiguous parameter device"
        )
    return next(iter(devices))

def _generation_generators(
    device: torch.device, *, torch_api: object = torch
) -> tuple[object, ...]:
    cpu = torch_api.random.default_generator
    if device.type == "cpu":
        return (cpu,)
    if device.type != "cuda" or device.index is None:
        raise RuntimeContractError("candidate CUDA device ownership is ambiguous")
    cuda = torch_api.cuda
    if cuda.is_initialized() is not True:
        raise RuntimeContractError(
            "candidate CUDA runtime was not initialized by model construction"
        )
    defaults = cuda.default_generators
    if device.index < 0 or device.index >= len(defaults):
        raise RuntimeContractError("candidate CUDA generator does not exist")
    selected = defaults[device.index]
    if selected is cpu:
        raise RuntimeContractError("CPU and CUDA generators unexpectedly alias")
    return cpu, selected

@contextmanager
def _preserve_seeded_generators(generators: Sequence[object], seed: int):
    if type(seed) is not int or type(generators) not in (tuple, list):
        raise RuntimeContractError("generation RNG ownership is not exact")
    if not generators or len({id(generator) for generator in generators}) != len(
        generators
    ):
        raise RuntimeContractError("generation RNG generators are empty or aliased")
    states: list[object] = []
    for generator in generators:
        state = generator.get_state()
        states.append(state.clone() if type(state) is Tensor else deepcopy(state))
    touched = 0
    try:
        for generator in generators:
            touched += 1
            generator.manual_seed(seed)
        yield
    finally:
        for generator, state in zip(
            generators[:touched], states[:touched], strict=True
        ):
            generator.set_state(state)

def _tree_to_device(value: object, device: torch.device) -> object:
    if type(value) is Tensor:
        return value.detach().clone().to(device=device)
    if type(value) in (dict, MappingProxyType):
        return {key: _tree_to_device(nested, device) for key, nested in value.items()}
    raise RuntimeContractError("generation batch contains a non-tensor leaf")

def _check_generation_conditioning(
    batch: Mapping[str, object], expected: Mapping[str, str]
) -> None:
    forbidden = {"family", "parent_id", "example_id", "cell_hash"}
    if forbidden & set(batch):
        raise RuntimeContractError("sealed row identity reached candidate generation")
    for key, digest in expected.items():
        if key not in batch or _tree_digest(batch[key]) != digest:
            raise RuntimeContractError(f"generation changed conditioning field {key!r}")

def _generation_loss(
    generated: object, target: Mapping[str, object], valid: Tensor
) -> dict[str, object]:
    output = _exact_mapping("generation output", generated)
    _exact_keys("generation output", output, set(_MODALITIES))
    numerator = 0.0
    denominator = 0
    events: list[str] = []
    for mode in _MODALITIES:
        value = output[mode]
        clean = target[mode]
        if type(value) is not Tensor or type(clean) is not Tensor:
            raise RuntimeContractError(
                f"generation {mode} output and target must be exact Tensors"
            )
        if (
            tuple(value.shape) != tuple(clean.shape)
            or value.dtype != clean.dtype
            or value.ndim != 3
            or tuple(value.shape[:2]) != tuple(valid.shape)
            or not torch.is_floating_point(value)
        ):
            raise RuntimeContractError(
                f"generation {mode} output differs from exact clean target shape/dtype"
            )
        clean_events = _finite_events_residue_tensor(f"target.{mode}", clean, valid)
        if clean_events:
            raise RuntimeContractError(
                f"generation clean target is non-finite: {clean_events}"
            )
        events.extend(_finite_events_residue_tensor(f"generation.{mode}", value, valid))
        expanded = valid[..., None].expand_as(value)
        difference = (value - clean).masked_select(expanded)
        finite = difference[torch.isfinite(difference)]
        numerator += float(finite.double().square().sum().item())
        denominator += int(difference.numel())
    if not math.isfinite(numerator) or numerator < 0 or denominator <= 0:
        raise RuntimeContractError("generation diagnostic loss is invalid")
    return {
        "loss": {"numerator": numerator, "denominator": denominator},
        "non_finite_count": len(events),
        "non_finite_events": events,
    }

def _sample_generation(
    runtime: DirectionalEvaluationRuntime,
    cell: GenerationCell,
    *,
    roots: Mapping[str, Path],
    clean_batch: Mapping[str, object] | None,
    require_pinned_method: bool,
) -> dict[str, object]:
    if (
        type(runtime) is not DirectionalEvaluationRuntime
        or type(cell) is not GenerationCell
    ):
        raise RuntimeContractError("generation requires exact runtime and cell")
    request = _exact_mapping("generation request", cell.request)
    _exact_keys(
        "generation request",
        request,
        {"family", "parent_id", "seed", "arm", "cell_hash", "payload"},
    )
    identity = {
        "family": cell.family,
        "parent_id": cell.parent_id,
        "seed": cell.seed,
        "arm": cell.arm,
        "cell_hash": cell.cell_hash,
    }
    if {key: request[key] for key in identity} != identity:
        raise RuntimeContractError("generation request identity differs from cell")
    payload = _exact_mapping("generation payload", request["payload"])
    _exact_keys(
        "generation payload",
        payload,
        {"representative_record", "generation_policy"},
    )
    record = _exact_mapping(
        "generation representative record", payload["representative_record"]
    )
    if record.get("family") != cell.family or record.get("parent_id") != cell.parent_id:
        raise RuntimeContractError("generation representative identity differs")
    policy = _validate_generation_policy(payload["generation_policy"])
    candidate = runtime.candidate_model
    if _generation_policy_from_cfg(candidate.cfg_exp) != dict(policy):
        raise RuntimeContractError(
            "generation policy differs from verified candidate cfg"
        )
    generation_auth: (
        tuple[VerifiedBlindInputs, ModuleType, frozenset[str], tuple[object, ...]]
        | None
    ) = None
    if require_pinned_method:
        execution_inputs = runtime._execution_inputs
        proteina_module = sys.modules.get("proteinfoundation.proteina")
        if execution_inputs is None or not isinstance(proteina_module, ModuleType):
            raise RuntimeContractError(
                "production generation lacks authenticated execution inputs"
            )
        generation_roots = frozenset({"Proteina"})
        generation_auth = (
            execution_inputs,
            proteina_module,
            generation_roots,
            _authenticate_dynamic_project_module(
                execution_inputs,
                proteina_module,
                required_roots=generation_roots,
            ),
        )
        _assert_pinned_generate_binding(candidate)
    configure, generate = _generation_class_functions(
        candidate, require_pinned=require_pinned_method
    )
    clean_cpu = (
        _construct_clean_loader_batch(record, roots)
        if clean_batch is None
        else _clone_tree(clean_batch)
    )
    clean = _freeze_clean_batch(clean_cpu)
    device = _candidate_device(candidate)
    clean = _freeze_clean_batch(_tree_to_device(clean, device))
    simulation = _clone_tree(clean)
    target_input = _clone_tree(clean)
    from proteinfoundation.utils.sample_utils import add_clean_samples

    if require_pinned_method:
        assert runtime._execution_inputs is not None
        sample_module = sys.modules.get("proteinfoundation.utils.sample_utils")
        if not isinstance(sample_module, ModuleType):
            raise RuntimeContractError(
                "generation sample helper module is not authenticated"
            )
        sample_roots = frozenset({"add_clean_samples"})
        sample_signature = _authenticate_dynamic_project_module(
            runtime._execution_inputs,
            sample_module,
            required_roots=sample_roots,
        )
        try:
            target_batch = add_clean_samples(
                target_input,
                candidate.cfg_exp.product_flowmatcher,
                getattr(candidate, "autoencoder", None),
            )
        finally:
            _revalidate_dynamic_project_module(
                runtime._execution_inputs,
                sample_module,
                required_roots=sample_roots,
                expected_signature=sample_signature,
            )
    else:
        target_batch = add_clean_samples(
            target_input,
            candidate.cfg_exp.product_flowmatcher,
            getattr(candidate, "autoencoder", None),
        )
    target = _exact_mapping("generation clean target", target_batch["x_1"])
    _exact_keys("generation clean target", target, set(_MODALITIES))
    valid = simulation["mask"] & simulation["generated_mask"]
    expected_conditioning = _generation_conditioning_digests(simulation)
    adapter = candidate.nn
    if type(adapter) is not DirectionalV2Adapter:
        raise RuntimeContractError(
            "generation candidate lacks exact directional adapter"
        )
    route = adapter._active_route.get()
    if (cell.arm == "adaptive" and route is not RouteState.ADAPTIVE) or (
        cell.arm == "best_frozen_fixed_control"
        and route not in RouteState.hard_states()
    ):
        raise RuntimeContractError("generation arm route context differs")
    observed_routes: list[RouteState] = []

    def observe_call(_module: nn.Module, args: tuple[object, ...]) -> None:
        if len(args) != 1 or type(args[0]) is not dict:
            raise RuntimeContractError("generation adapter call has wrong batch schema")
        observed = adapter._active_route.get()
        if observed is not route:
            raise RuntimeContractError("generation adapter observed wrong route")
        _check_generation_conditioning(args[0], expected_conditioning)
        observed_routes.append(observed)

    handle = adapter.register_forward_pre_hook(observe_call)
    try:
        if generation_auth is not None:
            execution_inputs, proteina_module, generation_roots, signature = (
                generation_auth
            )
            _revalidate_dynamic_project_module(
                execution_inputs,
                proteina_module,
                required_roots=generation_roots,
                expected_signature=signature,
            )
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            inf_cfg = _generation_config(candidate, policy)
            configure(candidate, inf_cfg, None)
            generators = _generation_generators(device)
            with (
                _preserve_seeded_generators(generators, cell.seed),
                torch.inference_mode(),
            ):
                random.seed(cell.seed)
                np.random.seed(cell.seed)
                generated = generate(candidate, simulation)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
    finally:
        handle.remove()
        if generation_auth is not None:
            execution_inputs, proteina_module, generation_roots, signature = (
                generation_auth
            )
            _revalidate_dynamic_project_module(
                execution_inputs,
                proteina_module,
                required_roots=generation_roots,
                expected_signature=signature,
            )
    if len(observed_routes) != 100:
        raise RuntimeContractError(
            f"generation must make exactly 100 adapter calls, observed {len(observed_routes)}"
        )
    if any(observed is not route for observed in observed_routes):
        raise RuntimeContractError("generation adapter route drifted")
    _check_generation_conditioning(simulation, expected_conditioning)
    unknown = set(simulation) - set(expected_conditioning) - {"x_t", "t", "x_sc"}
    if unknown:
        raise RuntimeContractError(
            f"generation added unknown batch fields {sorted(unknown)}"
        )
    return _generation_loss(generated, target, valid)

def sample_reviewed_generation(
    runtime: DirectionalEvaluationRuntime, cell: GenerationCell
) -> dict[str, object]:

    return _sample_generation(
        runtime,
        cell,
        roots=_loader_roots(),
        clean_batch=None,
        require_pinned_method=True,
    )

def _test_only_sample_generation(
    runtime: DirectionalEvaluationRuntime,
    cell: GenerationCell,
    *,
    clean_batch: Mapping[str, object],
) -> dict[str, object]:

    return _sample_generation(
        runtime,
        cell,
        roots={},
        clean_batch=clean_batch,
        require_pinned_method=False,
    )

_DIVE_ROOT = Path(__file__).resolve().parents[3]
_BULK_ROOT = Path(joined('CODIRECT_DATA_ROOT'))
_LOADER_SOURCE_PATHS = (
    ("dive", "src/dive/data/dataset.py"),
    ("dive", "src/dive/data/pipeline.py"),
    ("dive", "src/dive/data/crop.py"),
    ("dive", "src/dive/data/motif.py"),
    ("dive", "src/dive/data/role_transform.py"),
    ("dive", "src/dive/data/roles.py"),
    ("dive", "src/dive/data/loaders.py"),
    ("upstream", "src/proteinfoundation/patches/atomworks_patches.py"),
    ("upstream", "src/proteinfoundation/datasets/transforms.py"),
)
_COLLATE_SOURCE = (
    "upstream",
    "src/proteinfoundation/datasets/structure_data.py",
)
_LOADER_MODULE_NAMES = MappingProxyType(
    {
        "src/dive/data/roles.py": "dive.data.roles",
        "src/dive/data/loaders.py": "dive.data.loaders",
        "src/dive/data/crop.py": "dive.data.crop",
        "src/dive/data/motif.py": "dive.data.motif",
        "src/dive/data/role_transform.py": "dive.data.role_transform",
        "src/dive/data/dataset.py": "dive.data.dataset",
        "src/dive/data/pipeline.py": "dive.data.pipeline",
        "src/proteinfoundation/patches/atomworks_patches.py": (
            "proteinfoundation.patches.atomworks_patches"
        ),
        "src/proteinfoundation/datasets/transforms.py": (
            "proteinfoundation.datasets.transforms"
        ),
        "src/proteinfoundation/datasets/structure_data.py": (
            "proteinfoundation.datasets.structure_data"
        ),
    }
)
_LOADER_MODULE_ORDER = (
    "src/proteinfoundation/patches/atomworks_patches.py",
    "src/proteinfoundation/datasets/transforms.py",
    "src/dive/data/roles.py",
    "src/dive/data/loaders.py",
    "src/dive/data/crop.py",
    "src/dive/data/motif.py",
    "src/dive/data/role_transform.py",
    "src/dive/data/dataset.py",
    "src/dive/data/pipeline.py",
    "src/proteinfoundation/datasets/structure_data.py",
)

@dataclass(slots=True)
class _RetainedSource:
    descriptor: int
    parent_descriptor: int
    leaf: str
    proc_path: str
    metadata: os.stat_result
    sha256: str
    snapshot: bytes

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.parent_descriptor)

def _open_retained_source(
    root: Path, identity: Mapping[str, object], *, name: str
) -> _RetainedSource:
    relative = Path(str(identity["relative_path"]))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current = os.open(root, directory_flags)
    descriptor: int | None = None
    retain_parent = False
    try:
        for part in relative.parent.parts:
            following = os.open(part, directory_flags, dir_fd=current)
            os.close(current)
            current = following
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current,
        )
        metadata = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        snapshot = b"".join(chunks)
        digest = hashlib.sha256(snapshot).hexdigest()
        os.lseek(descriptor, 0, os.SEEK_SET)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or len(snapshot) != identity["size_bytes"]
            or digest != identity["sha256"]
        ):
            raise RuntimeContractError(f"{name} differs from its retained identity")
        path_metadata = os.stat(relative.name, dir_fd=current, follow_symlinks=False)
        if (
            path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
        ):
            raise RuntimeContractError(f"{name} path differs from retained descriptor")
        retain_parent = True
        return _RetainedSource(
            descriptor,
            current,
            relative.name,
            f"/proc/self/fd/{descriptor}",
            metadata,
            digest,
            snapshot,
        )
    except OSError as error:
        raise RuntimeContractError(
            f"{name} cannot be opened component-safely"
        ) from error
    finally:
        if descriptor is not None and not retain_parent:
            os.close(descriptor)
        if not retain_parent:
            os.close(current)

def _revalidate_retained_source(source: _RetainedSource, *, name: str) -> None:
    current = os.stat(
        source.leaf, dir_fd=source.parent_descriptor, follow_symlinks=False
    )
    retained = os.fstat(source.descriptor)
    if (
        current.st_dev != source.metadata.st_dev
        or current.st_ino != source.metadata.st_ino
        or retained.st_dev != source.metadata.st_dev
        or retained.st_ino != source.metadata.st_ino
        or retained.st_size != source.metadata.st_size
    ):
        raise RuntimeContractError(f"{name} changed during row construction")

class _RetainedSourceLoader(importlib.abc.Loader):

    def __init__(self, source: _RetainedSource, module_name: str) -> None:
        self._source = source
        self._module_name = module_name

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        metadata = os.fstat(self._source.descriptor)
        raw = self._source.snapshot
        if (
            metadata.st_dev != self._source.metadata.st_dev
            or metadata.st_ino != self._source.metadata.st_ino
            or metadata.st_size != self._source.metadata.st_size
            or len(raw) != self._source.metadata.st_size
            or hashlib.sha256(raw).hexdigest() != self._source.sha256
        ):
            raise RuntimeContractError(
                f"retained module {self._module_name} changed before execution"
            )
        code = compile(raw, self._source.proc_path, "exec", dont_inherit=True)
        exec(code, module.__dict__)

def _retained_execution_value_signature(
    value: object, *, active: set[int], authenticated_modules: frozenset[str]
) -> tuple[object, ...]:

    value_type = type(value)
    if value is None or value is Ellipsis or value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (float, struct.pack(">d", value))
    if value_type is complex:
        return (complex, struct.pack(">dd", value.real, value.imag))
    if isinstance(value, CodeType):
        return ("code", _authenticated_code_fingerprint(value))
    if isinstance(value, ModuleType):
        return ("module-leaf", id(value), value.__name__)
    if isinstance(value, (BuiltinFunctionType, BuiltinMethodType)):
        return (
            "builtin-leaf",
            id(value),
            value.__module__,
            value.__qualname__,
        )

    identity = id(value)
    if identity in active:
        return (
            "recursive-reference",
            identity,
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", None),
        )
    active.add(identity)
    try:
        if isinstance(value, FunctionType):
            if value.__module__ not in authenticated_modules:
                return (
                    "external-function-leaf",
                    identity,
                    value.__module__,
                    value.__qualname__,
                    _authenticated_code_fingerprint(value.__code__),
                )
            referenced_globals = tuple(
                (
                    name,
                    _retained_execution_value_signature(
                        value.__globals__[name],
                        active=active,
                        authenticated_modules=authenticated_modules,
                    ),
                )
                for name in sorted(set(value.__code__.co_names))
                if name in value.__globals__
            )
            closure = ()
            if value.__closure__ is not None:
                closure = tuple(
                    (
                        id(cell),
                        _retained_execution_value_signature(
                            cell.cell_contents,
                            active=active,
                            authenticated_modules=authenticated_modules,
                        ),
                    )
                    for cell in value.__closure__
                )
            return (
                "function",
                identity,
                value.__module__,
                value.__qualname__,
                _authenticated_code_fingerprint(value.__code__),
                _retained_execution_value_signature(
                    value.__defaults__,
                    active=active,
                    authenticated_modules=authenticated_modules,
                ),
                _retained_execution_value_signature(
                    value.__kwdefaults__,
                    active=active,
                    authenticated_modules=authenticated_modules,
                ),
                _retained_execution_value_signature(
                    value.__annotations__,
                    active=active,
                    authenticated_modules=authenticated_modules,
                ),
                referenced_globals,
                closure,
            )
        if isinstance(value, type):
            if value.__module__ not in authenticated_modules:
                return (
                    "external-class-leaf",
                    identity,
                    value.__module__,
                    value.__qualname__,
                )
            methods = []
            class_state = []
            for name, nested in sorted(vars(value).items()):
                if isinstance(nested, (staticmethod, classmethod)):
                    nested = nested.__func__
                if isinstance(nested, FunctionType):
                    methods.append(
                        (
                            name,
                            _retained_execution_value_signature(
                                nested,
                                active=active,
                                authenticated_modules=authenticated_modules,
                            ),
                        )
                    )
                elif not name.startswith("__") and type(nested) in (
                    bool,
                    int,
                    float,
                    complex,
                    str,
                    bytes,
                    tuple,
                    list,
                    dict,
                    set,
                    frozenset,
                ):
                    class_state.append(
                        (
                            name,
                            _retained_execution_value_signature(
                                nested,
                                active=active,
                                authenticated_modules=authenticated_modules,
                            ),
                        )
                    )
            return (
                "class",
                identity,
                value.__module__,
                value.__qualname__,
                tuple(id(base) for base in value.__bases__),
                tuple(methods),
                tuple(class_state),
            )
        if isinstance(value, Mapping):
            entries = [
                (
                    _retained_execution_value_signature(
                        key,
                        active=active,
                        authenticated_modules=authenticated_modules,
                    ),
                    _retained_execution_value_signature(
                        nested,
                        active=active,
                        authenticated_modules=authenticated_modules,
                    ),
                )
                for key, nested in value.items()
            ]
            return ("mapping", value_type, tuple(sorted(entries, key=repr)))
        if isinstance(value, Sequence):
            return (
                "sequence",
                value_type,
                tuple(
                    _retained_execution_value_signature(
                        nested,
                        active=active,
                        authenticated_modules=authenticated_modules,
                    )
                    for nested in value
                ),
            )
        if isinstance(value, (set, frozenset)):
            entries = [
                _retained_execution_value_signature(
                    nested,
                    active=active,
                    authenticated_modules=authenticated_modules,
                )
                for nested in value
            ]
            return ("set", value_type, tuple(sorted(entries, key=repr)))
        return (
            "identity-leaf",
            identity,
            value_type,
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", None),
        )
    finally:
        active.remove(identity)

def _retained_module_execution_signature(module: ModuleType) -> tuple[object, ...]:
    authenticated_modules = frozenset(_LOADER_MODULE_NAMES.values()) | frozenset(
        {module.__name__}
    )
    roots = []
    for name, value in sorted(vars(module).items()):
        if (
            isinstance(value, (FunctionType, type))
            and value.__module__ == module.__name__
        ):
            roots.append(
                (
                    name,
                    _retained_execution_value_signature(
                        value,
                        active=set(),
                        authenticated_modules=authenticated_modules,
                    ),
                )
            )
    return tuple(roots)

def _construct_retained_module(
    source: _RetainedSource, module_name: str, sha256: str
) -> ModuleType:

    if type(module_name) is not str or not module_name:
        raise RuntimeContractError("retained module name is not exact")
    if type(sha256) is not str or sha256 != source.sha256:
        raise RuntimeContractError("retained module digest differs from its descriptor")
    if module_name in sys.modules:
        raise RuntimeContractError(
            f"preloaded module {module_name} lacks retained provenance"
        )
    parent_name, separator, _ = module_name.rpartition(".")
    if separator:
        importlib.import_module(parent_name)
        if module_name in sys.modules:
            raise RuntimeContractError(
                f"preloaded module {module_name} lacks retained provenance"
            )
    loader = _RetainedSourceLoader(source, module_name)
    spec = importlib.util.spec_from_loader(module_name, loader, origin=source.proc_path)
    if spec is None:
        raise RuntimeContractError(f"cannot define retained module {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module

@dataclass(frozen=True, slots=True)
class _RetainedCacheEntry:

    logical_module_name: str
    source_catalog_identity: str
    retained_row_source_sha256: str
    module: ModuleType
    loader: _RetainedSourceLoader
    execution_signature: tuple[object, ...]
    installation_generation: int

class RetainedModuleCache:

    __slots__ = ("_catalog_identity", "_entries", "_generation", "_pending")

    def __init__(self, *, catalog_identity: str) -> None:
        if type(catalog_identity) is not str or len(catalog_identity) != 64:
            raise RuntimeContractError("retained cache catalog identity is not exact")
        if set(catalog_identity) - set("0123456789abcdef"):
            raise RuntimeContractError("retained cache catalog identity is not exact")
        self._catalog_identity = catalog_identity
        self._entries: dict[str, _RetainedCacheEntry] = {}
        self._generation = 0
        self._pending: list[tuple[str, ModuleType]] | None = None

    @property
    def catalog_identity(self) -> str:
        return self._catalog_identity

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def installed_names(self) -> frozenset[str]:
        return frozenset(self._entries)

    @contextmanager
    def install_transaction(self):

        if self._pending is not None:
            raise RuntimeContractError("retained cache transaction is already open")
        self._pending = []
        try:
            yield self
        except BaseException:

            for name, module in reversed(self._pending):
                self._entries.pop(name, None)
                if sys.modules.get(name) is module:
                    del sys.modules[name]
            raise
        else:
            if self._pending:
                self._generation += 1
        finally:
            self._pending = None

    def get(
        self,
        module_name: str,
        *,
        row_source_sha256: str,
        source: _RetainedSource,
    ) -> ModuleType:

        entry = self._entries.get(module_name)
        if entry is not None:
            if (
                entry.retained_row_source_sha256 != row_source_sha256
                or entry.source_catalog_identity != self._catalog_identity
                or sys.modules.get(module_name) is not entry.module
                or entry.module.__loader__ is not entry.loader
            ):
                raise RuntimeContractError(
                    f"retained cache entry {module_name} lost its provenance"
                )
            if (
                _retained_module_execution_signature(entry.module)
                != entry.execution_signature
            ):
                raise RuntimeContractError(
                    f"retained cache entry {module_name} executable state changed"
                )
            return entry.module
        if self._pending is None:
            raise RuntimeContractError(
                "retained cache miss outside an installation transaction"
            )
        module = _construct_retained_module(source, module_name, row_source_sha256)

        self._pending.append((module_name, module))
        loader = module.__loader__
        if type(loader) is not _RetainedSourceLoader:
            raise RuntimeContractError(
                f"retained module {module_name} was not installed by its retained loader"
            )
        self._entries[module_name] = _RetainedCacheEntry(
            logical_module_name=module_name,
            source_catalog_identity=self._catalog_identity,
            retained_row_source_sha256=row_source_sha256,
            module=module,
            loader=loader,
            execution_signature=_retained_module_execution_signature(module),
            installation_generation=self._generation + 1,
        )
        return module

    def revalidate_row(self, module_names: Sequence[str]) -> None:

        for module_name in module_names:
            try:
                entry = self._entries[module_name]
            except KeyError as error:
                raise RuntimeContractError(
                    f"retained module {module_name} was never installed by this cache"
                ) from error
            if (
                sys.modules.get(module_name) is not entry.module
                or entry.module.__loader__ is not entry.loader
                or _retained_module_execution_signature(entry.module)
                != entry.execution_signature
            ):
                raise RuntimeContractError(
                    f"retained module {module_name} changed during the row"
                )

_UNBOUND_RETAINED_CATALOG_IDENTITY = hashlib.sha256(
    b"dive.directional.unbound-retained-module-cache"
).hexdigest()

_RETAINED_MODULE_CACHE_OWNER = RetainedModuleCache(
    catalog_identity=_UNBOUND_RETAINED_CATALOG_IDENTITY
)

MUTABLE_SOURCE_REALM_EXCLUSIONS = MappingProxyType(
    {
        "dive.evaluation.directional_runtime._RETAINED_MODULE_CACHE_OWNER": (
            RetainedModuleCache
        ),
    }
)

def _test_only_import_retained_module(
    source: _RetainedSource,
    module_name: str,
    sha256: str,
    cache: RetainedModuleCache,
) -> ModuleType:
    with cache.install_transaction():
        return cache.get(module_name, row_source_sha256=sha256, source=source)

def _load_retained_structure(
    source: _RetainedSource, identity: Mapping[str, object]
) -> object:

    relative = Path(str(identity["relative_path"]))
    if relative.name != source.leaf:
        raise RuntimeContractError("structure locator differs from retained leaf")
    suffix = relative.suffix.lower()
    file_type = {".pdb": "pdb", ".cif": "cif", ".mmcif": "cif"}.get(suffix)
    if file_type is None:
        raise RuntimeContractError(
            "retained structure suffix must be exact pdb, cif, or mmcif"
        )
    from atomworks.io.utils.io_utils import load_any

    snapshot = io.BytesIO(source.snapshot)
    text_snapshot = io.TextIOWrapper(snapshot, encoding="utf-8", newline=None)
    atom_array = load_any(text_snapshot, file_type=file_type, model=1)
    if not hasattr(atom_array, "occupancy") or atom_array.occupancy is None:
        atom_array.set_annotation(
            "occupancy", np.ones(len(atom_array), dtype=np.float32)
        )
    if not hasattr(atom_array, "b_factor") or atom_array.b_factor is None:
        atom_array.set_annotation(
            "b_factor", np.zeros(len(atom_array), dtype=np.float32)
        )
    return atom_array

def _validate_collated_batch(
    family: str,
    value: object,
    loader_row: Mapping[str, object],
) -> dict[str, Tensor]:
    collated = _exact_mapping("fixed collate output", value)
    try:
        expected = set(_COLLATE_KEYS_BY_FAMILY[family])
    except KeyError as error:
        raise RuntimeContractError("fixed collate family is unknown") from error
    _exact_keys("fixed collate output", collated, expected)
    for key in _COLLATE_TENSOR_KEYS:
        if type(collated[key]) is not Tensor:
            raise RuntimeContractError(f"fixed collate {key} must be an exact Tensor")
    coords = collated["coords"]
    coords_nm = collated["coords_nm"]
    assert isinstance(coords, Tensor) and isinstance(coords_nm, Tensor)
    if (
        coords.dtype is not torch.float32
        or coords_nm.dtype is not torch.float32
        or coords.ndim != 4
        or tuple(coords.shape) != tuple(coords_nm.shape)
        or coords.shape[0] != 1
        or coords.shape[2:] != (37, 3)
    ):
        raise RuntimeContractError(
            "fixed collate coordinates must be exact float32 [1,N,37,3]"
        )
    residues = int(coords.shape[1])
    target = collated["x_target"]
    assert isinstance(target, Tensor)
    if (
        target.dtype is not torch.float32
        or target.ndim != 4
        or target.shape[0] != 1
        or target.shape[2:] != (37, 3)
    ):
        raise RuntimeContractError(
            "fixed collate x_target must be exact float32 [1,M,37,3]"
        )
    targets = int(target.shape[1])
    if residues <= 0 or targets <= 0:
        raise RuntimeContractError("fixed collate residue groups must be nonempty")

    bool_shapes = {
        "chain_breaks_per_residue": (1, residues),
        "coord_mask": (1, residues, 37),
        "design_mask": (1, residues),
        "mask": (1, residues),
        "motif_mask": (1, residues, 37),
        "seq_target_mask": (1, targets),
        "target_hotspot_mask": (1, targets),
        "target_mask": (1, targets, 37),
        "target_padding_mask": (1, targets),
        "target_residue_mask": (1, residues),
    }
    int_shapes = {
        "chains": (1, residues),
        "residue_pdb_idx": (1, residues),
        "residue_type": (1, residues),
        "seq_pos": (1, residues, 1),
        "seq_target": (1, targets),
        "target_chains": (1, targets),
        "target_pdb_idx": (1, targets),
    }
    for key, shape in bool_shapes.items():
        tensor = collated[key]
        assert isinstance(tensor, Tensor)
        if tensor.dtype is not torch.bool or tuple(tensor.shape) != shape:
            raise RuntimeContractError(
                f"fixed collate {key} must be exact bool {shape}"
            )
    for key, shape in int_shapes.items():
        tensor = collated[key]
        assert isinstance(tensor, Tensor)
        if tensor.dtype is not torch.int64 or tuple(tensor.shape) != shape:
            raise RuntimeContractError(
                f"fixed collate {key} must be exact int64 {shape}"
            )
    if (
        type(collated["nres"]) is not int
        or collated["nres"] != residues
        or type(collated["n_target"]) is not int
        or collated["n_target"] != targets
        or collated["nsamples"] != 1
        or type(collated["nsamples"]) is not int
    ):
        raise RuntimeContractError("fixed collate scalar dimensions drifted")

    exact_metadata = {
        "id": loader_row["example_id"],
        "generated": loader_row["generated"],
        "context": loader_row["context"],
        "target": loader_row["target"],
        "database": "structure",
    }
    for key, expected_value in exact_metadata.items():
        if type(collated[key]) is not list or collated[key] != [expected_value]:
            raise RuntimeContractError(f"fixed collate metadata {key} drifted")
    for key in ("chain_id", "chain_names", "residues"):
        metadata = collated[key]
        if (
            type(metadata) is not list
            or len(metadata) != 1
            or type(metadata[0]) is not list
            or any(type(item) not in (str, np.str_) for item in metadata[0])
        ):
            raise RuntimeContractError(f"fixed collate metadata {key} is not exact")
    if (
        len(collated["chain_id"][0]) != residues
        or len(collated["residues"][0]) != residues
    ):
        raise RuntimeContractError("fixed collate residue metadata shape drifted")
    design = collated["design_mask"]
    mask = collated["mask"]
    assert isinstance(design, Tensor) and isinstance(mask, Tensor)
    if torch.any(design & ~mask) or int((design & mask).sum().item()) <= 0:
        raise RuntimeContractError("fixed collate design mask is invalid")
    _require_finite_tree(
        "fixed collate tensors", {key: collated[key] for key in _COLLATE_TENSOR_KEYS}
    )
    return {key: collated[key] for key in _COLLATE_TENSOR_KEYS}

def _test_only_validate_collated_batch(
    family: str, value: object, loader_row: Mapping[str, object]
) -> dict[str, Tensor]:
    return _validate_collated_batch(family, value, loader_row)

def _loader_roots() -> Mapping[str, Path]:
    return MappingProxyType(
        {"bulk": _BULK_ROOT, "dive": _DIVE_ROOT, "upstream": EMERGENT_UPSTREAM_ROOT}
    )

def _exact_loader_sources(
    family: str, loader: Mapping[str, object]
) -> tuple[Mapping[str, object], ...]:
    sources = loader["transform_chain_sources"]
    if type(sources) is not list:
        raise RuntimeContractError("loader transform source inventory is not exact")
    expected = _LOADER_SOURCE_PATHS
    observed = tuple(
        (str(source["root_class"]), str(source["relative_path"]))
        for source in sources
    )
    if observed != expected:
        raise RuntimeContractError("loader transform source inventory drifted")
    collate = _exact_mapping("loader collate identity", loader["collate_identity"])
    if (collate["root_class"], collate["relative_path"]) != _COLLATE_SOURCE:
        raise RuntimeContractError("loader collate source identity drifted")
    return tuple(sources) + (collate,)

def _construct_clean_loader_batch(
    record: Mapping[str, object],
    roots: Mapping[str, Path],
    *,
    cache: RetainedModuleCache | None = None,
) -> dict[str, object]:
    from dive.integrations.complexa_training import _import_upstream_v2

    if cache is None:
        cache = _RETAINED_MODULE_CACHE_OWNER
    if type(cache) is not RetainedModuleCache:
        raise RuntimeContractError("retained loader cache is not evaluator-owned")
    _import_upstream_v2(roots["upstream"])
    family = str(record["family"])
    loader = _exact_mapping("loader spec", record["loader_spec"])
    retained_code: list[_RetainedSource] = []
    retained_modules: dict[str, tuple[_RetainedSource, str]] = {}
    retained_structure: _RetainedSource | None = None
    try:
        with cache.install_transaction():
            for index, identity in enumerate(_exact_loader_sources(family, loader)):
                source = _exact_mapping(f"loader source[{index}]", identity)
                relative_path = str(source["relative_path"])
                if relative_path in retained_modules:
                    raise RuntimeContractError("loader source inventory is not unique")
                retained = _open_retained_source(
                    roots[str(source["root_class"])],
                    source,
                    name=f"loader source[{index}]",
                )
                retained_code.append(retained)
                retained_modules[relative_path] = (retained, str(source["sha256"]))
            expected_order = tuple(
                path for path in _LOADER_MODULE_ORDER if path in retained_modules
            )
            if set(expected_order) != set(retained_modules):
                raise RuntimeContractError("loader source has no fixed module identity")
            modules: dict[str, ModuleType] = {}
            for relative_path in expected_order:
                source, digest = retained_modules[relative_path]
                module_name = _LOADER_MODULE_NAMES[relative_path]
                modules[module_name] = cache.get(
                    module_name, row_source_sha256=digest, source=source
                )

            atomworks_patches = modules["proteinfoundation.patches.atomworks_patches"]

            from atomworks.ml.transforms import encoding

            if (
                encoding.atom_array_to_encoding
                is not atomworks_patches.patched_atom_array_to_encoding
            ):
                raise RuntimeContractError(
                    "pinned Atomworks patch did not bind exactly"
                )
            structure_collate_fn = modules[
                "proteinfoundation.datasets.structure_data"
            ].structure_collate_fn

            import pandas as pd

            DEFAULT_CROP_SIZE = modules["dive.data.crop"].DEFAULT_CROP_SIZE
            LOADER_COLUMNS = modules["dive.data.loaders"].LOADER_COLUMNS
            assert_loader_manifest_is_clean = modules[
                "dive.data.loaders"
            ].assert_loader_manifest_is_clean
            RoleAwareStructureDataset = modules[
                "dive.data.dataset"
            ].RoleAwareStructureDataset
            family_loader_kwargs = modules["dive.data.pipeline"].family_loader_kwargs
            family_transforms = modules["dive.data.pipeline"].family_transforms

            if loader["crop_size"] != DEFAULT_CROP_SIZE:
                raise RuntimeContractError(
                    "loader crop_size differs from the fixed pipeline"
                )
            if dict(loader["family_loader_kwargs"]) != family_loader_kwargs(family):
                raise RuntimeContractError(
                    "loader family kwargs differ from the fixed pipeline"
                )
            locator = _exact_mapping("source locator", record["source_locator"])
            retained_structure = _open_retained_source(
                roots["bulk"], locator, name="blind structure source"
            )
            loader_row = _exact_mapping("loader row", record["loader_row"])
            frame = pd.DataFrame(
                [
                    {
                        "example_id": loader_row["example_id"],
                        "path": retained_structure.proc_path,
                        "generated": loader_row["generated"],
                        "context": loader_row["context"],
                        "target": loader_row["target"],
                    }
                ],
                columns=list(LOADER_COLUMNS),
            )
            assert_loader_manifest_is_clean(frame)
            motif = _exact_mapping("loader motif seed", loader["motif_seed"])
            transforms = family_transforms(
                family,
                crop_size=int(loader["crop_size"]),
                seed=motif["value"],
            )

            def load_descriptor_structure(path: str) -> object:
                if type(path) is not str or path != retained_structure.proc_path:
                    raise RuntimeContractError(
                        "structure dataset did not preserve the retained leaf FD path"
                    )
                return _load_retained_structure(retained_structure, locator)

            dataset = RoleAwareStructureDataset(
                frame,
                transforms=transforms,
                structure_loader=load_descriptor_structure,
                **family_loader_kwargs(family),
            )
            sample = dataset[0]
            if sample is None:
                raise RuntimeContractError(
                    "blind structure source did not yield one clean row"
                )
            collated = structure_collate_fn([sample])
            batch = _validate_collated_batch(family, collated, loader_row)
            design = batch.pop("design_mask")
            mask = batch.get("mask")
            assert type(design) is Tensor and type(mask) is Tensor
            batch["generated_mask"] = design.bool()
            batch["fixed_mask"] = mask.bool() & ~design.bool()
            _revalidate_retained_source(
                retained_structure, name="blind structure source"
            )
            cache.revalidate_row(tuple(modules))
            return batch
    finally:
        if retained_structure is not None:
            retained_structure.close()
        for source in reversed(retained_code):
            source.close()

def _json_snapshot(inputs: VerifiedBlindInputs, name: str) -> object:
    try:
        raw = inputs.snapshots[name]
    except KeyError as error:
        raise RuntimeContractError(
            f"runtime is missing retained snapshot {name}"
        ) from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeContractError(f"retained snapshot {name} is not JSON") from error
    if raw != json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n":
        raise RuntimeContractError(f"retained snapshot {name} is not canonical")
    return value

def _construct_reviewed_data(
    inputs: VerifiedBlindInputs,
    roots: Mapping[str, Path],
    *,
    token: object,
) -> BlindEvaluationData:
    if token is not _DATA_CONSTRUCTION_TOKEN or type(inputs) is not VerifiedBlindInputs:
        raise RuntimeContractError("blind loader construction is not injectable")
    strict = _json_snapshot(inputs, "sealed_strict_flow_inventory")
    manifest_raw = inputs.snapshots.get("strict_blind_loader_manifest")
    if type(manifest_raw) is not bytes:
        raise RuntimeContractError("runtime is missing retained strict loader manifest")
    manifest = _validate_strict_blind_loader_manifest(
        manifest_raw, {"strict_flow_inventory": strict}
    )
    rows = []
    for record in manifest["records"]:
        clean = _construct_clean_loader_batch(record, roots)
        identity = {
            key: record[key]
            for key in ("family", "example_id", "parent_id", "cell_hash")
        }
        rows.append(
            FlowRow(
                **identity,
                request={
                    **identity,
                    "corruption_seed": 42,
                    "payload": {"batch": clean},
                },
            )
        )
    planned = _json_snapshot(inputs, "sealed_generation_cells")
    availability = _json_snapshot(inputs, "sealed_availability_manifest")
    if type(planned) is not list or type(availability) is not dict:
        raise RuntimeContractError("retained generation inventory is malformed")
    records = manifest["records"]
    if type(records) is not list:
        raise RuntimeContractError("retained strict loader records are malformed")
    generation_policy = _generation_policy_from_inputs(inputs)
    excluded = {item["cell_hash"] for item in availability["deterministic_exclusions"]}
    cells = tuple(
        GenerationCell(
            family=cell["family"],
            parent_id=cell["parent_id"],
            seed=cell["seed"],
            arm=cell["arm"],
            cell_hash=cell["cell_hash"],
            request={
                **cell,
                "payload": {
                    "representative_record": json.loads(
                        json.dumps(
                            _representative_record(
                                records, cell["family"], cell["parent_id"]
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                    "generation_policy": json.loads(
                        json.dumps(
                            generation_policy,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                },
            },
        )
        for cell in planned
        if cell["cell_hash"] not in excluded
    )
    return BlindEvaluationData(tuple(rows), cells)

def construct_reviewed_data(inputs: VerifiedBlindInputs) -> BlindEvaluationData:

    return _construct_reviewed_data(
        inputs, _loader_roots(), token=_DATA_CONSTRUCTION_TOKEN
    )

def _test_only_construct_clean_loader_batch(
    record: Mapping[str, object],
    roots: Mapping[str, Path],
    *,
    cache: RetainedModuleCache | None = None,
) -> dict[str, object]:

    return _construct_clean_loader_batch(record, roots, cache=cache)

def construct_reviewed_runtime(
    inputs: VerifiedBlindInputs, _data: BlindEvaluationData
) -> DirectionalEvaluationRuntime:

    if type(inputs) is not VerifiedBlindInputs:
        raise RuntimeContractError("runtime requires exact verified inputs")
    required = {
        "candidate_checkpoint",
        "parent_checkpoint",
        "common_checkpoint",
        "autoencoder_checkpoint",
    }
    missing = sorted(required - set(inputs.artifacts))
    if missing:
        raise RuntimeContractError(f"runtime is missing retained descriptors {missing}")
    candidate, parent = _models_from_retained_descriptors(inputs)
    return DirectionalEvaluationRuntime(
        candidate,
        parent,
        directional_proteina_bindings(candidate, n_recycle=0),
        _token=_CONSTRUCTION_TOKEN,
        _execution_inputs=inputs,
    )

def _descriptor_path(inputs: VerifiedBlindInputs, name: str) -> Path:
    artifact = inputs.artifacts[name]
    if artifact.descriptor < 0:
        raise RuntimeContractError(
            f"production artifact {name} has no retained descriptor"
        )
    return Path(f"/proc/self/fd/{artifact.descriptor}")

def _models_from_retained_descriptors(
    inputs: VerifiedBlindInputs,
) -> tuple[nn.Module, FrozenParentRuntime]:

    common = _descriptor_path(inputs, "common_checkpoint")
    autoencoder = _descriptor_path(inputs, "autoencoder_checkpoint")
    emergent_module = importlib.import_module("dive.emergent_contract")
    complexa_module = importlib.import_module("dive.integrations.complexa_training")
    training_module = importlib.import_module("dive.training.directional_module")
    dive_inventory = (
        (emergent_module, frozenset({"EmergentContract"})),
        (
            complexa_module,
            frozenset(
                {
                    "SpliceReport",
                    "_import_upstream_v2",
                    "assert_v2_selected",
                    "build_directional_v2_model",
                }
            ),
        ),
        (training_module, frozenset({"DirectionalTrainingModule"})),
    )
    with _authenticated_dynamic_project_execution(inputs, dive_inventory):
        complexa_module._import_upstream_v2(EMERGENT_UPSTREAM_ROOT)
        proteina_module = importlib.import_module("proteinfoundation.proteina")
        upstream_train_module = importlib.import_module("proteinfoundation.train")
        lora_module = importlib.import_module("proteinfoundation.utils.lora_utils")
        upstream_inventory = (
            (proteina_module, frozenset({"Proteina"})),
            (upstream_train_module, frozenset({"_splice_pretrained_weights"})),
            (lora_module, frozenset({"replace_lora_layers"})),
        )
        with _authenticated_dynamic_project_execution(inputs, upstream_inventory):
            contract = emergent_module.EmergentContract(
                upstream_root=EMERGENT_UPSTREAM_ROOT,
                upstream_commit=EMERGENT_UPSTREAM_COMMIT,
                common_checkpoint=common,
                autoencoder_checkpoint=autoencoder,
                architecture_v2=True,
                lora_rank=8,
                gpu_count=4,
                evidence_root=EMERGENT_EVIDENCE_ROOT,
                bulk_root=EMERGENT_BULK_ROOT,
                backup_deadline=date(2026, 8, 27),
            )
            candidate, _ = complexa_module.build_directional_v2_model(
                contract,
                store_dir=EMERGENT_BULK_ROOT / "model_store",
                upstream_root=EMERGENT_UPSTREAM_ROOT,
            )
            module = training_module.DirectionalTrainingModule(
                base=candidate,
                bindings=directional_proteina_bindings(candidate, n_recycle=0),
            )
            candidate_payload = _load_delta_payload(inputs, "candidate_checkpoint")
            _apply_delta(module, candidate_payload, expected_stage="joint_adaptation")
            parent_payload = _load_delta_payload(inputs, "parent_checkpoint")
            _validate_parent_delta(parent_payload)
            candidate.eval()
            parent = _build_plain_parent(common, autoencoder, candidate)
            return candidate, parent

def _authenticate_dynamic_project_module(
    inputs: VerifiedBlindInputs,
    module: ModuleType,
    *,
    required_roots: frozenset[str],
) -> tuple[object, ...]:
    from dive.evaluation.directional_seal import _authenticate_reviewed_module_code

    return _authenticate_reviewed_module_code(
        inputs, module, required_roots=required_roots
    )

def _revalidate_dynamic_project_module(
    inputs: VerifiedBlindInputs,
    module: ModuleType,
    *,
    required_roots: frozenset[str],
    expected_signature: tuple[object, ...],
) -> None:
    from dive.evaluation.directional_seal import _revalidate_reviewed_module

    _revalidate_reviewed_module(
        inputs,
        module,
        required_roots=required_roots,
        expected_signature=expected_signature,
    )

@contextmanager
def _authenticated_dynamic_project_execution(
    inputs: VerifiedBlindInputs,
    inventory: Sequence[tuple[ModuleType, frozenset[str]]],
):
    authenticated = tuple(
        (
            module,
            roots,
            _authenticate_dynamic_project_module(inputs, module, required_roots=roots),
        )
        for module, roots in inventory
    )
    try:
        yield
    finally:
        for module, roots, signature in authenticated:
            _revalidate_dynamic_project_module(
                inputs,
                module,
                required_roots=roots,
                expected_signature=signature,
            )

def _load_delta_payload(inputs: VerifiedBlindInputs, name: str) -> Mapping[str, object]:
    path = _descriptor_path(inputs, name)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeContractError(f"cannot parse retained {name}") from error
    payload = _exact_mapping(name, value)
    expected = {
        "schema_version",
        "mode",
        "trainable_state_by_name",
        "optimizer_state_by_name",
        "resume_state",
        "step_completed",
        "resumed_from",
        "rng_state",
        "execution_state",
        "size_policy",
        "ownership",
        "validation_record_path",
        "validation_record_hash",
        "completion_record_path",
    }
    _exact_keys(name, payload, expected)
    if payload["schema_version"] != 1:
        raise RuntimeContractError(f"{name} schema_version must be one")
    return payload

def _ownership(payload: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
    ownership = _exact_mapping("checkpoint ownership", payload["ownership"])
    _exact_keys(
        "checkpoint ownership",
        ownership,
        {
            "stage",
            "trainable_names",
            "trainable_name_count",
            "trainable_parameters",
            "groups",
        },
    )
    names = ownership["trainable_names"]
    if type(names) is not list or any(type(name) is not str for name in names):
        raise RuntimeContractError("checkpoint trainable_names are invalid")
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeContractError("checkpoint trainable_names are not canonical")
    if ownership["trainable_name_count"] != len(names):
        raise RuntimeContractError("checkpoint trainable_name_count differs")
    return str(ownership["stage"]), tuple(names)

def _apply_delta(
    module: nn.Module, payload: Mapping[str, object], *, expected_stage: str
) -> None:
    stage, names = _ownership(payload)
    if stage != expected_stage:
        raise RuntimeContractError("candidate checkpoint stage differs")
    state = _exact_mapping(
        "candidate trainable delta", payload["trainable_state_by_name"]
    )
    if set(state) != set(names):
        raise RuntimeContractError("candidate delta differs from ownership")
    live = dict(module.named_parameters())
    with torch.no_grad():
        for name in names:
            tensor = state[name]
            if (
                type(tensor) is not Tensor
                or name not in live
                or tensor.shape != live[name].shape
            ):
                raise RuntimeContractError(
                    f"candidate delta tensor {name!r} is invalid"
                )
            live[name].copy_(tensor.to(dtype=live[name].dtype))

def _validate_parent_delta(payload: Mapping[str, object]) -> None:
    stage, names = _ownership(payload)
    if stage != "bridge_warmup" or not names:
        raise RuntimeContractError("parent checkpoint is not exact warm-up lineage")
    if any(
        "lora_" in name.lower()
        or not any(token in name for token in ("bridge", "router"))
        for name in names
    ):
        raise RuntimeContractError(
            "parent warm-up delta contains non-directional base state"
        )
    state = _exact_mapping("parent trainable delta", payload["trainable_state_by_name"])
    if set(state) != set(names) or any(
        type(value) is not Tensor for value in state.values()
    ):
        raise RuntimeContractError("parent warm-up delta differs from ownership")

def _build_plain_parent(
    common: Path,
    autoencoder: Path,
    candidate: nn.Module,
) -> FrozenParentRuntime:
    from dive.integrations.complexa_training import (
        SpliceReport,
        _import_upstream_v2,
        assert_v2_selected,
    )

    _import_upstream_v2(EMERGENT_UPSTREAM_ROOT)
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.train import _splice_pretrained_weights

    assert_v2_selected()
    checkpoint = torch.load(common, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["cfg_exp"]
    model = Proteina(
        config,
        store_dir=str(EMERGENT_BULK_ROOT / "model_store"),
        autoencoder_ckpt_path=str(autoencoder),
    )
    model_state = model.state_dict()
    checkpoint_state = checkpoint["state_dict"]
    spliced = _splice_pretrained_weights(model_state, checkpoint_state)
    resized = tuple(
        key
        for key, value in spliced.items()
        if key in checkpoint_state and value.shape != checkpoint_state[key].shape
    )
    result = model.load_state_dict(spliced, strict=False)
    report = SpliceReport(
        model_nn_params=sum(1 for key in model_state if key.startswith("nn.")),
        checkpoint_nn_params=sum(
            1 for key in checkpoint_state if key.startswith("nn.")
        ),
        missing=tuple(key for key in result.missing_keys if key.startswith("nn.")),
        unexpected=tuple(
            key for key in result.unexpected_keys if key.startswith("nn.")
        ),
        resized=resized,
    )
    try:
        report.assert_sufficiently_loaded()
    except Exception as error:
        raise RuntimeContractError(
            "plain parent common-checkpoint splice is insufficient"
        ) from error
    parent = model.nn
    parent.eval()
    parent.requires_grad_(False)
    _normalize_parent_diagnostic_state(parent)
    return FrozenParentRuntime(
        parent,
        candidate,
        _token=_PARENT_DESCRIPTOR_TOKEN,
        _catalog=_installed_catalog_realm(),
    )

__all__ = [
    "AuthenticatedParentGraph",
    "CorruptedFlowInput",
    "DirectionalEvaluationRuntime",
    "EXECUTABLE_ROUTE_STATES",
    "FrozenParentResult",
    "RouteFlowResult",
    "RowFlowEvidence",
    "RuntimeContractError",
    "TelemetryEvidence",
    "authenticate_parent_graph",
    "compile_expected_parent_graph",
    "construct_reviewed_data",
    "construct_reviewed_runtime",
    "evaluate_one_flow_row",
    "flow_records",
    "seal_loaded_data",
    "validate_raw_row",
]
