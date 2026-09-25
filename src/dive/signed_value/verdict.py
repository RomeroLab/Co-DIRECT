
from __future__ import annotations

from dive.signed_value.contract import (
    CONTROL_DENOISER_CALLS,
    FROZEN_TARGETS,
    PROBE_DENOISER_CALLS,
    RUN_ID,
)

import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from dive.signed_value.basis import BasisResult
from dive.signed_value.budget import (
    HEADROOM_BYTES,
    MIN_DEVICE_BYTES,
    BudgetError,
    critic_memory_limit,
)
from dive.signed_value.contracts import FeasibilityTarget
from dive.signed_value.critic_protocol import (
    CRITIC_CALLS,
)
from dive.signed_value.critic_protocol import CriticRequest, CriticResponse
from dive.signed_value.directional import (
    DECODER_CALLS,
    LADDER,
    DirectionalBundle,
    DirectionalState,
    cosine_and_relative_norm,
    masked_values,
    scale_token,
    tangent,
)

_FROZEN_RUN_ID = RUN_ID
_FROZEN_TARGETS = FROZEN_TARGETS
PASS = "PASS"
BLOCKED_CRITIC = "BLOCKED_CRITIC"

_STRUCTURAL_CHECKS = (
    "provenance",
    "critic_input_binding",
    "basis",
    "calls",
    "no_op_identity",
    "torch_memory_before_worker",
    "process_group_empty",
    "memory",
    "critic_protocol",
    "directional_bundle",
)

ABSTAIN_UNRESOLVABLE = "ABSTAIN_UNRESOLVABLE"

SOFTMAX_RESOLUTION_FLOOR = 2.0**-8
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = (
    "initial_batch_sha256",
    "initial_torch_cpu_rng_sha256",
    "initial_torch_cuda_rng_sha256",
    "final_tensor_tree_sha256",
    "final_pdb_sha256",
    "final_torch_cpu_rng_sha256",
    "final_torch_cuda_rng_sha256",
)

@dataclass(slots=True)
class StateEvidence:

    run_id: str
    generation_seed: int
    checkpoint_index: int
    target: FeasibilityTarget
    frozen_target: FeasibilityTarget
    request: CriticRequest
    state: DirectionalState
    basis: BasisResult
    bundle: DirectionalBundle
    response: CriticResponse
    control: Mapping[str, object]
    probe: Mapping[str, object]
    torch_memory_before_worker: Mapping[str, object]
    generator_memory: Mapping[str, object]
    process_group_members: Sequence[int]

    @property
    def noise(self) -> float:

        return _noise_backprop(self)

    @property
    def noise_direct(self) -> float:

        return _noise_direct(self)

    @property
    def q(self) -> dict[tuple[int, float], float]:

        return {
            (prior, scale): _masked_dot(
                self.response.sequence_gradient,
                self.bundle.tangents[f"prior{prior}.{scale_token(scale)}.logits"],
                self.state.mask,
            )
            for prior in (0, 1)
            for scale in LADDER
        }

    @property
    def q2(self) -> dict[tuple[int, float], float] | None:

        second = getattr(self.response, "sequence_gradient2", None)
        if second is None:
            return None
        return {
            (prior, scale): _masked_dot(
                second,
                self.bundle.tangents[f"prior{prior}.{scale_token(scale)}.logits"],
                self.state.mask,
            )
            for prior in (0, 1)
            for scale in LADDER
        }

    @property
    def delta_reward(self) -> dict[tuple[int, float], float]:

        values: dict[tuple[int, float], float] = {}
        for prior in (0, 1):
            for scale in LADDER:
                token = scale_token(scale)
                plus = self.response.perturbed_rewards[f"prior{prior}.{token}.plus"]
                minus = self.response.perturbed_rewards[f"prior{prior}.{token}.minus"]
                values[(prior, scale)] = plus - minus
        return values

    @property
    def direct(self) -> dict[tuple[int, float], float]:

        delta = self.delta_reward
        return {
            (prior, scale): delta[(prior, scale)] / (2.0 * scale * self.bundle.h0[prior])
            for prior in (0, 1)
            for scale in LADDER
        }

    @property
    def error_models(self) -> dict[int, ErrorModel | None]:

        direct = self.direct
        return {
            prior: fit_error_model(
                {scale: direct[(prior, scale)] for scale in LADDER},
                self.bundle.h0[prior],
            )
            for prior in (0, 1)
        }

    @property
    def online_scales(self) -> dict[int, tuple[float, float] | None]:

        direct = self.direct
        delta = self.delta_reward
        return {
            prior: online_pair(
                {scale: direct[(prior, scale)] for scale in LADDER},
                {scale: delta[(prior, scale)] for scale in LADDER},
                self.bundle.h0[prior],
            )
            for prior in (0, 1)
        }

    @property
    def reproducibility(self) -> dict[str, float]:

        replicate = _noise_q(self)
        return {
            "backprop_gap": _noise_backprop(self),
            "repeat_gap": _noise_direct(self),
            "gradient_gap": 0.0 if replicate is None else replicate,
        }

@dataclass(frozen=True, slots=True)
class StateVerdict:
    target: str
    status: str
    informative: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class RunVerdict:
    run_id: str
    status: str
    targets: tuple[str, str, str]
    states: tuple[StateVerdict, StateVerdict, StateVerdict]

def same_nonzero_sign(left: float, right: float) -> bool:

    return (
        _finite_real(left)
        and _finite_real(right)
        and left != 0.0
        and right != 0.0
        and ((left > 0.0) == (right > 0.0))
    )

def informative(state: StateEvidence) -> bool:

    try:
        q = state.q
        direct = state.direct
        windows = state.online_scales
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        return False
    for prior in (0, 1):
        window = windows.get(prior)
        if window is None:
            return False
        for scale in window:
            left = q.get((prior, scale))
            right = direct.get((prior, scale))
            if not (_finite_real(left) and _finite_real(right)):
                return False
            if left == 0.0 or right == 0.0:
                return False
    return True

RESOLUTION_MARGIN = 10.0

@dataclass(frozen=True, slots=True)
class ErrorModel:

    derivative: float
    roughness: float
    curvature: float
    residual: float

    def total_error(self, h: float) -> float:

        return abs(self.roughness) / (2.0 * h) + abs(self.curvature) * h * h

def fit_error_model(
    direct: Mapping[float, float], h0: float
) -> ErrorModel | None:

    if not isinstance(direct, Mapping) or not _finite_real(h0) or h0 <= 0.0:
        return None
    scales = sorted(scale for scale in direct if _finite_real(scale) and scale > 0.0)
    if len(scales) < 4:
        return None
    values = [direct[scale] for scale in scales]
    if not all(_finite_real(value) for value in values):
        return None
    design = torch.tensor(
        [
            [1.0, 1.0 / (2.0 * scale * h0), (scale * h0) ** 2]
            for scale in scales
        ],
        dtype=torch.float64,
    )
    observed = torch.tensor(values, dtype=torch.float64).unsqueeze(1)

    try:
        column_scale = torch.linalg.vector_norm(design, dim=0)
        if not bool((column_scale > 0).all()):
            return None
        orthogonal, upper = torch.linalg.qr(design / column_scale, mode="reduced")
        scaled = torch.linalg.solve_triangular(
            upper, orthogonal.transpose(0, 1) @ observed, upper=True
        )
        solution = (scaled / column_scale.unsqueeze(1)).squeeze(1)
        residual = float(
            torch.linalg.vector_norm(design @ solution.unsqueeze(1) - observed).item()
        )
    except (RuntimeError, ValueError):
        return None
    derivative, roughness, curvature = (float(value) for value in solution)
    if not all(
        _finite_real(value)
        for value in (derivative, roughness, curvature, residual)
    ):
        return None
    return ErrorModel(derivative, roughness, curvature, residual)

def online_pair(
    direct: Mapping[float, float],
    delta_reward: Mapping[float, float],
    h0: float,
) -> tuple[float, float] | None:

    model = fit_error_model(direct, h0)
    if model is None:
        return None
    floor = RESOLUTION_MARGIN * abs(model.roughness)
    ordered = sorted(scale for scale in direct if _finite_real(scale) and scale > 0.0)
    best: tuple[float, tuple[float, float]] | None = None
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        magnitudes = [delta_reward.get(lower), delta_reward.get(upper)]
        if not all(_finite_real(value) and abs(value) > floor for value in magnitudes):
            continue
        worst = max(model.total_error(lower * h0), model.total_error(upper * h0))
        if not _finite_real(worst):
            continue
        if best is None or worst < best[0]:
            best = (worst, (lower, upper))
    return None if best is None else best[1]

def evaluate_state(state: StateEvidence) -> StateVerdict:

    target = _target_name(state)
    checks: dict[str, bool] = {
        "provenance": _provenance_is_exact(state),
        "critic_input_binding": _critic_inputs_are_bound(state),
        "basis": _basis_is_trustworthy(state),
        "calls": _call_ledger_is_exact(state),
        "no_op_identity": _noop_identity_is_exact(state),
        "torch_memory_before_worker": _torch_memory_is_empty(state),
        "process_group_empty": _process_group_is_empty(state),
        "memory": _peaks_fit_with_headroom(state),
        "critic_protocol": _critic_protocol_is_exact(state),
        "directional_bundle": _bundle_is_exact(state),
        "perturbation_resolvable": _resolvable(state),
    }
    try:
        q = state.q
        direct = state.direct
        windows = state.online_scales
        models = state.error_models
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        q = {}
        direct = {}
        windows = {0: None, 1: None}
        models = {0: None, 1: None}

    checks["informative"] = informative(state)
    for prior in (0, 1):
        window = windows.get(prior)
        checks[f"resolved_window.prior{prior}"] = window is not None
        checks[f"tangent_trust.prior{prior}"] = _tangent_trust(state, prior)
        checks[f"direct_stability.prior{prior}"] = _direct_stability(direct, window, prior)

        checks[f"extrapolated_agreement.prior{prior}"] = _extrapolated_agreement(
            models.get(prior), q, window, prior
        )
        for index in (0, 1):
            scale = None if window is None else window[index]
            checks[f"agreement.prior{prior}.online{index}"] = _agreement(
                q, direct, prior, scale
            )

        for scale in LADDER:
            checks[f"rung_agreement.prior{prior}.{scale_token(scale)}"] = _agreement(
                q, direct, prior, scale
            )

    required = (
        *_STRUCTURAL_CHECKS,
        "informative",
        "perturbation_resolvable",
        *(f"resolved_window.prior{prior}" for prior in (0, 1)),
        *(f"tangent_trust.prior{prior}" for prior in (0, 1)),
        *(f"direct_stability.prior{prior}" for prior in (0, 1)),
        *(
            f"agreement.prior{prior}.online{index}"
            for prior in (0, 1)
            for index in (0, 1)
        ),
    )
    passed = all(checks[name] for name in required)

    structurally_sound = all(checks[name] for name in _STRUCTURAL_CHECKS)
    unresolved = not checks["perturbation_resolvable"] or not all(
        checks[f"resolved_window.prior{prior}"] for prior in (0, 1)
    )
    if structurally_sound and unresolved:
        status = ABSTAIN_UNRESOLVABLE
    else:
        status = PASS if passed else BLOCKED_CRITIC
    return StateVerdict(
        target=target,
        status=status,
        informative=checks["informative"],
        checks=checks,
        reasons=tuple(name for name in required if not checks[name]),
    )

def evaluate_run(states: Sequence[StateEvidence]) -> RunVerdict:

    evaluated = tuple(evaluate_state(state) for state in states)
    targets = tuple(verdict.target for verdict in evaluated)
    run_ids = {state.run_id for state in states}
    exact_inventory = len(targets) == 3 and set(targets) == set(_FROZEN_TARGETS)
    all_pass = len(evaluated) == 3 and all(verdict.status == "PASS" for verdict in evaluated)
    run_is_exact = run_ids == {_FROZEN_RUN_ID}
    if len(targets) == 3:
        typed_targets = (targets[0], targets[1], targets[2])
        typed_states = (evaluated[0], evaluated[1], evaluated[2])
    else:
        typed_targets = ("", "", "")
        blocked = StateVerdict("", BLOCKED_CRITIC, False, {}, ("state_count",))
        typed_states = (blocked, blocked, blocked)
    return RunVerdict(
        run_id=_FROZEN_RUN_ID if run_is_exact else "",
        status=PASS
        if exact_inventory and all_pass and run_is_exact
        else BLOCKED_CRITIC,
        targets=typed_targets,
        states=typed_states,
    )

def _target_name(state: object) -> str:
    target = getattr(getattr(state, "target", None), "target", "")
    return target if isinstance(target, str) else ""

def _provenance_is_exact(state: StateEvidence) -> bool:
    try:
        target = state.target
        request = state.request
        if (
            state.run_id != _FROZEN_RUN_ID
            or state.generation_seed != 11
            or state.checkpoint_index != 200
            or target.target not in _FROZEN_TARGETS
            or target != state.frozen_target
            or request.target != target.target
            or request.target_chains != target.target_chains
            or request.target_path.resolve() != target.target_path.resolve()
            or not request.target_path.is_file()
        ):
            return False
        request.assert_exact()
    except (AttributeError, OSError, TypeError, ValueError, RuntimeError):
        return False
    return True

def _basis_is_trustworthy(state: StateEvidence) -> bool:
    try:
        basis = state.basis
        return (
            basis.active is True
            and type(basis.additional_denoiser_calls) is int
            and basis.additional_denoiser_calls == 3
            and basis.sampler_calls == ("bb_ca", "bb_ca")
            and _finite_real(basis.mc_ratio)
            and basis.mc_ratio <= 1.0
            and _finite_real(basis.transport_ratio)
        )
    except (AttributeError, TypeError):
        return False

def _call_ledger_is_exact(state: StateEvidence) -> bool:
    return _role_has_exact_calls(
        state.control, "control", CONTROL_DENOISER_CALLS, 0, 0
    ) and _role_has_exact_calls(
        state.probe, "probe", PROBE_DENOISER_CALLS, DECODER_CALLS, CRITIC_CALLS
    )

def _role_has_exact_calls(
    record: Mapping[str, object], role: str, denoiser: int, decoder: int, critic: int
) -> bool:
    try:
        return (
            record["role"] == role
            and type(record["denoiser_calls"]) is int
            and type(record["decoder_calls"]) is int
            and type(record["critic_calls"]) is int
            and record["denoiser_calls"] == denoiser
            and record["decoder_calls"] == decoder
            and record["critic_calls"] == critic
        )
    except (KeyError, TypeError):
        return False

def _noop_identity_is_exact(state: StateEvidence) -> bool:
    try:
        return all(
            isinstance(state.control[key], str)
            and _SHA256.fullmatch(state.control[key]) is not None
            and state.control[key] == state.probe[key]
            for key in _IDENTITY_KEYS
        )
    except (KeyError, TypeError):
        return False

def _torch_memory_is_empty(state: StateEvidence) -> bool:
    try:
        return (
            type(state.torch_memory_before_worker["allocated_bytes"]) is int
            and type(state.torch_memory_before_worker["reserved_bytes"]) is int
            and state.torch_memory_before_worker["allocated_bytes"] == 0
            and state.torch_memory_before_worker["reserved_bytes"] == 0
        )
    except (KeyError, TypeError):
        return False

def _process_group_is_empty(state: StateEvidence) -> bool:
    try:
        return tuple(state.process_group_members) == ()
    except TypeError:
        return False

def _memory_fits(
    generator: Mapping[str, object], critic: Mapping[str, object]
) -> bool:

    try:
        total = generator["total_memory_bytes"]
        critic_total = critic["critic_gpu_total_memory_bytes"]
        device_bounded = (
            generator["peak_allocated_bytes"],
            generator["peak_reserved_bytes"],
            critic["critic_peak_pool_bytes"],
        )
        allowance_bounded = (
            critic["critic_peak_allocated_bytes"],
            critic["critic_peak_reserved_bytes"],
        )
        values = (*device_bounded, *allowance_bounded, total, critic_total)
        if not all(type(value) is int and value >= 0 for value in values):
            return False
        return (
            critic_total == critic_memory_limit(total)
            and total >= MIN_DEVICE_BYTES
            and all(value <= total - HEADROOM_BYTES for value in device_bounded)
            and all(
                value <= critic_total - HEADROOM_BYTES for value in allowance_bounded
            )
        )
    except (BudgetError, KeyError, TypeError):
        return False

def _noise_backprop(state: object) -> float:

    response = state.response
    return abs(response.joint_reward - response.repeat_reward)

def _noise_q(state: object) -> float | None:

    try:
        first = state.q
        second = state.q2
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    if second is None:
        return None
    try:
        return max(abs(first[key] - second[key]) for key in first)
    except (KeyError, TypeError, ValueError):
        return None

def _noise_direct(state: object) -> float:

    response = state.response
    repeat2 = getattr(response, "repeat2_reward", None)
    if repeat2 is None:
        return _noise_backprop(state)
    return abs(response.repeat_reward - repeat2)

def _perturbation_is_resolvable(
    joint_logits: object, perturbed_logits: Mapping[str, object]
) -> bool:

    try:
        import torch

        joint = torch.softmax(joint_logits, dim=-1)
        for value in perturbed_logits.values():
            shifted = torch.softmax(value, dim=-1)
            if float((shifted - joint).abs().max()) >= SOFTMAX_RESOLUTION_FLOOR:
                return True
        return False
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        return False

def _peaks_fit_with_headroom(state: StateEvidence) -> bool:
    try:
        return _memory_fits(state.generator_memory, state.response.memory_stats)
    except (AttributeError, TypeError):
        return False

def _resolvable(state: StateEvidence) -> bool:
    try:
        request = state.request
        return _perturbation_is_resolvable(
            request.joint_logits, request.perturbed_logits
        )
    except (AttributeError, TypeError):
        return False

def _critic_protocol_is_exact(state: StateEvidence) -> bool:
    try:
        state.response.assert_exact()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False
    return True

def _bundle_is_exact(state: StateEvidence) -> bool:
    try:
        bundle = state.bundle
        expected_perturbations = {
            f"prior{prior}.{scale_token(scale)}.{side}"
            for prior in (0, 1)
            for scale in LADDER
            for side in ("minus", "plus")
        }
        expected_tangents = {
            f"prior{prior}.{scale_token(scale)}.{output}"
            for prior in (0, 1)
            for scale in LADDER
            for output in ("logits", "atom37")
        }
        expected_metrics = {
            f"prior{prior}.{output}.{metric}"
            for prior in (0, 1)
            for output in ("logits", "atom37")
            for metric in ("cosine", "relative_norm_change")
        }
        return (
            type(bundle.decoder_calls) is int
            and bundle.decoder_calls == DECODER_CALLS
            and set(bundle.perturbed_logits) == expected_perturbations
            and set(bundle.tangents) == expected_tangents
            and set(bundle.tangent_metrics) == expected_metrics
            and len(bundle.h0) == 2
            and all(_finite_real(h) and h > 0 for h in bundle.h0)
            and _valid_tangent(bundle.joint_logits, bundle.joint_logits)
            and _valid_tangent(bundle.joint_atom37, bundle.joint_atom37)
            and all(
                _valid_tangent(
                    bundle.tangents[f"prior{prior}.{scale_token(scale)}.logits"],
                    bundle.joint_logits,
                )
                and _valid_tangent(
                    bundle.tangents[f"prior{prior}.{scale_token(scale)}.atom37"],
                    bundle.joint_atom37,
                )
                and torch.equal(
                    tangent(
                        bundle.perturbed_logits[
                            f"prior{prior}.{scale_token(scale)}.plus"
                        ],
                        bundle.perturbed_logits[
                            f"prior{prior}.{scale_token(scale)}.minus"
                        ],
                        scale * bundle.h0[prior],
                    ),
                    bundle.tangents[f"prior{prior}.{scale_token(scale)}.logits"],
                )
                for prior in (0, 1)
                for scale in LADDER
            )
            and all(_finite_real(value) for value in bundle.tangent_metrics.values())
        )
    except (
        ArithmeticError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        return False

def _critic_inputs_are_bound(state: StateEvidence) -> bool:
    try:
        if not _same_logits(state.request.joint_logits, state.bundle.joint_logits):
            return False
        return all(
            _same_logits(
                state.request.perturbed_logits[name], state.bundle.perturbed_logits[name]
            )
            for name in state.request.perturbed_logits
        )
    except (AttributeError, KeyError, TypeError):
        return False

def _same_logits(request_logits: torch.Tensor, bundle_logits: torch.Tensor) -> bool:
    if bundle_logits.ndim == request_logits.ndim + 1 and bundle_logits.shape[0] == 1:
        bundle_logits = bundle_logits.squeeze(0)
    return (
        request_logits.shape == bundle_logits.shape
        and request_logits.dtype == bundle_logits.dtype
        and request_logits.device == bundle_logits.device
        and torch.equal(request_logits, bundle_logits)
    )

def _tangent_trust(state: StateEvidence, prior: int) -> bool:

    try:
        return _tangent_pair_is_trustworthy(
            state.bundle.tangents[f"prior{prior}.scale1.logits"],
            state.bundle.tangents[f"prior{prior}.scale2.logits"],
            state.state.mask,
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError):
        return False

def _direct_stability(
    values: Mapping[tuple[int, float], float],
    window: tuple[float, float] | None,
    prior: int,
) -> bool:

    if window is None:
        return False
    try:
        return (
            _symmetric_relative_error(
                values[(prior, window[0])], values[(prior, window[1])]
            )
            <= 0.10
        )
    except (KeyError, TypeError, ValueError):
        return False

def _extrapolated_agreement(
    model: "ErrorModel | None",
    q: Mapping[tuple[int, float], float],
    window: tuple[float, float] | None,
    prior: int,
) -> bool:

    if model is None or window is None:
        return False
    try:
        return same_nonzero_sign(q[(prior, window[0])], model.derivative) and (
            _symmetric_relative_error(q[(prior, window[0])], model.derivative) <= 0.10
        )
    except (KeyError, TypeError, ValueError):
        return False

def _agreement(
    q: Mapping[tuple[int, float], float],
    direct: Mapping[tuple[int, float], float],
    prior: int,
    scale: float | None,
) -> bool:
    if scale is None:
        return False
    try:
        return same_nonzero_sign(q[(prior, scale)], direct[(prior, scale)]) and _symmetric_relative_error(
            q[(prior, scale)], direct[(prior, scale)]
        ) <= 0.10
    except (KeyError, TypeError, ValueError):
        return False

def _masked_dot(gradient: torch.Tensor, tangent: torch.Tensor, mask: torch.Tensor) -> float:
    if gradient.shape != tangent.shape:
        if tangent.ndim == gradient.ndim + 1 and tangent.shape[0] == 1 and tangent.shape[1:] == gradient.shape:
            gradient = gradient.unsqueeze(0)
        else:
            raise ValueError("critic gradient and logit tangent shapes differ")
    left = masked_values(gradient, mask).to(torch.float64)
    right = masked_values(tangent, mask).to(torch.float64)
    value = torch.dot(left, right).item()
    if not math.isfinite(value):
        raise ValueError("masked critic/logit tangent dot product is non-finite")
    return value

def _symmetric_relative_error(left: float, right: float) -> float:
    if not _finite_real(left) or not _finite_real(right):
        raise ValueError("derivative is non-finite")
    return abs(left - right) / max(abs(left), abs(right), 1e-6)

def _finite_real(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, numbers.Real)
        and math.isfinite(float(value))
    )

def _valid_tangent(value: object, reference: torch.Tensor) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and value.shape == reference.shape
        and value.dtype == reference.dtype
        and value.device == reference.device
        and bool(torch.isfinite(value).all())
    )

def _tangent_pair_is_trustworthy(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor
) -> bool:
    cosine, norm_change = cosine_and_relative_norm(left, right, mask)
    return (
        _finite_real(cosine)
        and _finite_real(norm_change)
        and cosine >= 0.99
        and norm_change <= 0.10
    )
