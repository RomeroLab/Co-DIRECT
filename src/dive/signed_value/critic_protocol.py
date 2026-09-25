
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dive.signed_value.directional import LADDER, scale_token
from dive.signed_value.roots import EVIDENCE_ROOT
from typing import Any

import torch
from torch import Tensor

class CriticProtocolError(RuntimeError):
    pass

SIGNED_VALUE_EVIDENCE_ROOT = EVIDENCE_ROOT / "routing" / "signed-value"
PERTURBATION_NAMES = tuple(
    f"prior{prior}.{scale_token(scale)}.{side}"
    for prior in (0, 1)
    for scale in LADDER
    for side in ("minus", "plus")
)

CRITIC_CALLS = 2 + 2 + len(PERTURBATION_NAMES)

LEGACY_CRITIC_CALLS = 14
LEGACY_CRITIC_CALLS_V2 = 15
LEGACY_CRITIC_CALLS_V3 = 16
_REQUEST_STATE_KEYS = frozenset(
    {
        "python_state_before_sha256",
        "numpy_state_before_sha256",
        "model_key_after_prep_sha256",
        "model_key_after_run_sha256",
        "python_state_after_sha256",
        "numpy_state_after_sha256",
    }
)
_MEMORY_SOURCE_KEYS = (
    "peak_bytes_in_use",
    "peak_bytes_reserved",
    "peak_pool_bytes",
    "bytes_limit",
)
_MEMORY_EVIDENCE_KEYS = (
    "critic_peak_allocated_bytes",
    "critic_peak_reserved_bytes",
    "critic_peak_pool_bytes",
    "critic_gpu_total_memory_bytes",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_PAYLOAD_KEYS = frozenset(
    {
        "record_type",
        "target",
        "target_path",
        "target_path_evidence",
        "target_chains",
        "joint_logits",
        "perturbed_logits",
        "preparation_seed",
    }
)
_RESPONSE_PAYLOAD_KEYS = frozenset(
    {
        "record_type",
        "joint_reward",
        "repeat_reward",
        "repeat2_reward",
        "sequence_gradient2",
        "perturbed_rewards",
        "sequence_gradient",
        "loaded_model_names",
        "parameter_tree_sha256",
        "request_states",
        "memory_stats",
        "critic_calls",
    }
)

@dataclass(frozen=True, slots=True)
class CriticRequest:

    target: str
    target_path: Path
    target_path_evidence: Mapping[str, object]
    target_chains: str
    joint_logits: Tensor
    perturbed_logits: Mapping[str, Tensor]
    preparation_seed: int = 0

    def assert_exact(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise CriticProtocolError("critic target must be a non-empty string")
        if not isinstance(self.target_path, Path) or not self.target_path.is_file():
            raise CriticProtocolError("critic target_path must be an existing file")
        _assert_target_evidence(self.target_path, self.target_path_evidence)
        if not isinstance(self.target_chains, str) or not self.target_chains:
            raise CriticProtocolError("critic target_chains must be non-empty")
        if self.preparation_seed != 0 or isinstance(self.preparation_seed, bool):
            raise CriticProtocolError("critic requires literal preparation seed 0")
        _assert_logits(self.joint_logits, "joint logits")
        if set(self.perturbed_logits) != set(PERTURBATION_NAMES):
            raise CriticProtocolError(
                f"critic request requires exactly {len(PERTURBATION_NAMES)} named perturbations"
            )
        for name in PERTURBATION_NAMES:
            logits = self.perturbed_logits[name]
            _assert_logits(logits, name)
            if logits.shape != self.joint_logits.shape:
                raise CriticProtocolError(f"{name} shape must equal joint logits shape")
            if logits.dtype != self.joint_logits.dtype:
                raise CriticProtocolError(f"{name} dtype must equal joint logits dtype")

@dataclass(frozen=True, slots=True)
class CriticResponse:

    joint_reward: float
    repeat_reward: float
    perturbed_rewards: Mapping[str, float]

    repeat2_reward: float | None

    sequence_gradient2: Tensor | None
    sequence_gradient: Tensor
    loaded_model_names: tuple[str, ...]
    parameter_tree_sha256: str
    request_states: Sequence[Mapping[str, str]]
    memory_stats: Mapping[str, int]
    critic_calls: int

    _CALLS_WITH_REPEAT2 = (CRITIC_CALLS, LEGACY_CRITIC_CALLS_V3, LEGACY_CRITIC_CALLS_V2)

    _CALLS_WITH_GRADIENT2 = (CRITIC_CALLS, LEGACY_CRITIC_CALLS_V3)

    @staticmethod
    def accepts_calls(count: object) -> bool:

        if isinstance(count, bool) or type(count) is not int:
            return False
        return count in (
            CRITIC_CALLS,
            LEGACY_CRITIC_CALLS_V3,
            LEGACY_CRITIC_CALLS_V2,
            LEGACY_CRITIC_CALLS,
        )

    def assert_exact(self) -> None:
        if not self.accepts_calls(self.critic_calls):
            raise CriticProtocolError(
                f"critic response requires {CRITIC_CALLS} calls "
                f"({LEGACY_CRITIC_CALLS}/{LEGACY_CRITIC_CALLS_V2}/"
                f"{LEGACY_CRITIC_CALLS_V3} for recorded legacy responses)"
            )

        if (self.repeat2_reward is None) == (self.critic_calls in self._CALLS_WITH_REPEAT2):
            raise CriticProtocolError(
                f"repeat2 reward must be present exactly on "
                f"{self._CALLS_WITH_REPEAT2}-call responses"
            )
        if (self.sequence_gradient2 is None) == (
            self.critic_calls in self._CALLS_WITH_GRADIENT2
        ):
            raise CriticProtocolError(
                f"second gradient must be present exactly on "
                f"{self._CALLS_WITH_GRADIENT2}-call responses"
            )
        if self.sequence_gradient2 is not None:
            _assert_logits(self.sequence_gradient2, "second sequence gradient")
        if self.repeat2_reward is not None:
            _assert_finite_number(self.repeat2_reward, "repeat2 reward")
        _assert_finite_number(self.joint_reward, "joint reward")
        _assert_finite_number(self.repeat_reward, "repeat reward")
        if set(self.perturbed_rewards) != set(PERTURBATION_NAMES):
            raise CriticProtocolError(
                f"critic response requires exactly {len(PERTURBATION_NAMES)} named perturbed rewards"
            )
        for name in PERTURBATION_NAMES:
            _assert_finite_number(self.perturbed_rewards[name], f"{name} reward")
        _assert_logits(self.sequence_gradient, "sequence gradient")
        if float(torch.linalg.vector_norm(self.sequence_gradient.to(torch.float64))) == 0.0:
            raise CriticProtocolError("sequence gradient must have nonzero norm")
        if self.loaded_model_names != ("model_2_multimer_v3",):
            raise CriticProtocolError("loaded model names must contain only model_2_multimer_v3")
        _assert_sha256(self.parameter_tree_sha256, "parameter tree")
        if len(self.request_states) != self.critic_calls:
            raise CriticProtocolError(
                f"critic response requires exactly {self.critic_calls} request states"
            )
        for index, state in enumerate(self.request_states):
            if set(state) != _REQUEST_STATE_KEYS:
                raise CriticProtocolError(
                    f"request state {index} must carry the exact RNG/key hash fields"
                )
            for name, value in state.items():
                _assert_sha256(value, f"request state {index} {name}")
        for name in _REQUEST_STATE_KEYS:
            if len({state[name] for state in self.request_states}) != 1:
                raise CriticProtocolError(
                    f"request-order-dependent critic RNG/key state detected for {name}"
                )
        for name in (*_MEMORY_SOURCE_KEYS, *_MEMORY_EVIDENCE_KEYS):
            if name not in self.memory_stats:
                raise CriticProtocolError(f"critic memory stats missing {name}")
        for name, value in self.memory_stats.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CriticProtocolError(f"critic memory stat {name} must be a nonnegative integer")
        expected = {
            "critic_peak_allocated_bytes": self.memory_stats["peak_bytes_in_use"],
            "critic_peak_reserved_bytes": self.memory_stats["peak_bytes_reserved"],
            "critic_peak_pool_bytes": self.memory_stats["peak_pool_bytes"],

            "critic_gpu_total_memory_bytes": self.memory_stats["bytes_limit"],
        }
        for name, value in expected.items():
            if self.memory_stats[name] != value:
                raise CriticProtocolError(f"critic memory stat mapping drifted for {name}")

def validate_protocol_path(path: Path) -> Path:

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise CriticProtocolError("path must be absolute")
    return candidate.resolve(strict=False)

def create_attempt_directory(path: Path) -> Path:

    resolved = validate_protocol_path(path)
    resolved.mkdir(exist_ok=False)
    return resolved

def write_critic_request(path: Path, request: CriticRequest) -> None:
    request.assert_exact()
    _atomic_torch_save(validate_protocol_path(path), _request_payload(request))

def load_critic_request(path: Path) -> CriticRequest:
    payload = _load_tensor_payload(path)
    _assert_payload_keys(payload, _REQUEST_PAYLOAD_KEYS, "critic request")
    if payload.get("record_type") != "dive_signed_value_critic_request_v1":
        raise CriticProtocolError("critic request record type drifted")
    try:
        request = CriticRequest(
            target=payload["target"],
            target_path=Path(payload["target_path"]),
            target_path_evidence=payload["target_path_evidence"],
            target_chains=payload["target_chains"],
            joint_logits=payload["joint_logits"],
            perturbed_logits=payload["perturbed_logits"],
            preparation_seed=payload["preparation_seed"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CriticProtocolError(f"malformed critic request: {error}") from error
    request.assert_exact()
    return request

def write_critic_response(path: Path, response: CriticResponse) -> None:
    response.assert_exact()
    _atomic_torch_save(validate_protocol_path(path), _response_payload(response))

def load_critic_response(path: Path) -> CriticResponse:
    payload = _load_tensor_payload(path)
    _assert_payload_keys(payload, _RESPONSE_PAYLOAD_KEYS, "critic response")
    if payload.get("record_type") != "dive_signed_value_critic_response_v1":
        raise CriticProtocolError("critic response record type drifted")
    try:
        response = CriticResponse(
            joint_reward=payload["joint_reward"],
            repeat_reward=payload["repeat_reward"],
            repeat2_reward=payload.get("repeat2_reward"),
            sequence_gradient2=payload.get("sequence_gradient2"),
            perturbed_rewards=payload["perturbed_rewards"],
            sequence_gradient=payload["sequence_gradient"],
            loaded_model_names=tuple(payload["loaded_model_names"]),
            parameter_tree_sha256=payload["parameter_tree_sha256"],
            request_states=payload["request_states"],
            memory_stats=payload["memory_stats"],
            critic_calls=payload["critic_calls"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CriticProtocolError(f"malformed critic response: {error}") from error
    response.assert_exact()
    return response

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:

    resolved = validate_protocol_path(path)
    partial = _partial_path(resolved)
    if resolved.exists():
        raise CriticProtocolError(f"critic JSON record already exists: {resolved}")
    if partial.exists():
        raise CriticProtocolError(f"partial critic record already exists: {partial}")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, resolved)
        _fsync_directory(resolved.parent)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

def _request_payload(request: CriticRequest) -> dict[str, Any]:
    return {
        "record_type": "dive_signed_value_critic_request_v1",
        "target": request.target,
        "target_path": str(request.target_path),
        "target_path_evidence": dict(request.target_path_evidence),
        "target_chains": request.target_chains,
        "joint_logits": request.joint_logits,
        "perturbed_logits": dict(request.perturbed_logits),
        "preparation_seed": request.preparation_seed,
    }

def _response_payload(response: CriticResponse) -> dict[str, Any]:
    return {
        "record_type": "dive_signed_value_critic_response_v1",
        "joint_reward": response.joint_reward,
        "repeat_reward": response.repeat_reward,
        "repeat2_reward": response.repeat2_reward,
        "sequence_gradient2": response.sequence_gradient2,
        "perturbed_rewards": dict(response.perturbed_rewards),
        "sequence_gradient": response.sequence_gradient,
        "loaded_model_names": list(response.loaded_model_names),
        "parameter_tree_sha256": response.parameter_tree_sha256,
        "request_states": [dict(state) for state in response.request_states],
        "memory_stats": dict(response.memory_stats),
        "critic_calls": response.critic_calls,
    }

def _load_tensor_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as error:
        raise CriticProtocolError(f"cannot load critic tensor record {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise CriticProtocolError("critic tensor record must be a mapping")
    return payload

def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    partial = _partial_path(path)
    if path.exists():
        raise CriticProtocolError(f"critic record already exists: {path}")
    if partial.exists():
        raise CriticProtocolError(f"partial critic record already exists: {partial}")
    try:
        with partial.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

def _partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _assert_target_evidence(path: Path, evidence: Mapping[str, object]) -> None:
    if set(evidence) != {"path", "size_bytes", "sha256"}:
        raise CriticProtocolError("target_path_evidence must contain exact path/size/hash")
    resolved = path.resolve()
    if evidence["path"] != str(resolved):
        raise CriticProtocolError("target_path_evidence path mismatch")
    payload = resolved.read_bytes()
    if evidence["size_bytes"] != len(payload):
        raise CriticProtocolError("target_path_evidence size mismatch")
    observed = hashlib.sha256(payload).hexdigest()
    if evidence["sha256"] != observed:
        raise CriticProtocolError("target_path_evidence SHA-256 mismatch")

def _assert_logits(value: object, description: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise CriticProtocolError(f"{description} must be a floating tensor")
    if value.device.type != "cpu":
        raise CriticProtocolError(f"{description} must be detached on CPU")
    if value.requires_grad:
        raise CriticProtocolError(f"{description} must be detached from autograd")
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] != 20:
        raise CriticProtocolError(f"{description} shape must be [binder_length, 20]")
    if not bool(torch.isfinite(value).all()):
        raise CriticProtocolError(f"{description} must be finite")

def _assert_finite_number(value: object, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CriticProtocolError(f"{description} must be numeric")
    if not math.isfinite(float(value)):
        raise CriticProtocolError(f"{description} must be finite")

def _assert_sha256(value: object, description: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CriticProtocolError(f"{description} SHA-256 must be lowercase hexadecimal")

def _assert_payload_keys(
    payload: Mapping[str, Any], expected: frozenset[str], description: str
) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CriticProtocolError(
            f"{description} must carry exact payload keys; missing={missing}, extra={extra}"
        )
