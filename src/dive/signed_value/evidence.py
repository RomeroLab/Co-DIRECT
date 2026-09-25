
from __future__ import annotations

from dive.signed_value.contract import (
    CONTROL_DENOISER_CALLS,
    PROBE_DENOISER_CALLS,
    RUN_ID,
)
from dive.signed_value.contract import FROZEN_TARGETS as _FROZEN_TARGETS

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from dive.signed_value.critic_protocol import CRITIC_CALLS
from dive.signed_value.directional import DECODER_CALLS, LADDER, scale_token

FROZEN_TARGETS = _FROZEN_TARGETS

ATTEMPT_SCHEMA_VERSION = 2

_ATTEMPT_ALLOCATION_RETRIES = 32
_ATTEMPT_ID = re.compile(r"^(?:[0-9]+_[0-9]+|local_[0-9]{4})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPLETE_SUMMARY_KEYS = {
    "schema_version",
    "run_id",
    "target",
    "status",
    "attempt_id",
    "config_sha256",
}
_COMPLETE_ATTEMPT_KEYS = {
    "schema_version",
    "run_id",
    "target",
    "status",
    "attempt_id",
    "hostname",
    "execution_host",
    "elapsed_seconds",
    "config_sha256",
    "resolved_identities",
    "calls",
    "verdict",
}
_STATE_REQUIRED_KEYS = (
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

_STATE_CHECK_KEYS = (
    set(_STATE_REQUIRED_KEYS)
    | {
        f"rung_agreement.prior{prior}.{scale_token(scale)}"
        for prior in (0, 1)
        for scale in LADDER
    }
    | {f"extrapolated_agreement.prior{prior}" for prior in (0, 1)}
)

class EvidenceError(RuntimeError):
    pass

def target_root(evidence_root: Path, target: str) -> Path:

    _require_target(target)
    return Path(evidence_root).resolve() / "targets" / target

def target_state(root: Path) -> str:

    manifest = Path(root).resolve() / "manifest.json"
    if not manifest.exists():
        return "NEVER_RUN"
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "CORRUPT"
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        return "CORRUPT"
    status = payload.get("status")
    if status not in {"FAILED", "COMPLETE"}:
        return "CORRUPT"
    target = payload.get("target")
    attempt_id = payload.get("attempt_id")
    if (
        not isinstance(target, str)
        or target not in FROZEN_TARGETS
        or not isinstance(attempt_id, str)
        or _ATTEMPT_ID.fullmatch(attempt_id) is None
    ):
        return "CORRUPT"
    if Path(root).resolve().name != target:
        return "CORRUPT"
    attempt = Path(root).resolve() / "attempts" / attempt_id / "attempt.json"
    if not attempt.is_file():
        return "CORRUPT"
    try:
        attempt_payload = json.loads(attempt.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "CORRUPT"
    if not isinstance(attempt_payload, Mapping):
        return "CORRUPT"
    if status == "COMPLETE" and not _valid_complete_pair(payload, attempt_payload):
        return "CORRUPT"
    if status == "FAILED" and (
        type(attempt_payload.get("schema_version")) is not int
        or attempt_payload.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or attempt_payload.get("target") != target
        or attempt_payload.get("status") != status
    ):
        return "CORRUPT"
    return str(status)

def _valid_complete_pair(
    summary: Mapping[str, object], attempt: Mapping[str, object]
) -> bool:
    if set(summary) != _COMPLETE_SUMMARY_KEYS or set(attempt) != _COMPLETE_ATTEMPT_KEYS:
        return False
    target = summary["target"]
    attempt_id = summary["attempt_id"]
    config_sha256 = summary["config_sha256"]
    if (
        type(summary["schema_version"]) is not int
        or summary["schema_version"] != 1
        or summary["run_id"] != RUN_ID
        or summary["status"] != "COMPLETE"
        or not isinstance(target, str)
        or target not in FROZEN_TARGETS
        or not isinstance(attempt_id, str)
        or _ATTEMPT_ID.fullmatch(attempt_id) is None
        or not _valid_sha256(config_sha256)
    ):
        return False
    if (
        type(attempt["schema_version"]) is not int
        or attempt["schema_version"] != ATTEMPT_SCHEMA_VERSION
        or attempt["run_id"] != summary["run_id"]
        or attempt["target"] != target
        or attempt["status"] != "COMPLETE"
        or attempt["attempt_id"] != attempt_id
        or attempt["config_sha256"] != config_sha256
        or not isinstance(attempt["hostname"], str)
        or not attempt["hostname"]
        or not isinstance(attempt["execution_host"], str)
        or not attempt["execution_host"]
        or not _valid_elapsed_seconds(attempt["elapsed_seconds"])
        or not _valid_calls(attempt["calls"])
        or not _valid_identities(attempt["resolved_identities"], config_sha256)
        or not _valid_verdict(attempt["verdict"], target)
    ):
        return False
    return True

def _valid_elapsed_seconds(value: object) -> bool:

    if not isinstance(value, Mapping) or set(value) != {"control", "probe"}:
        return False
    for seconds in value.values():
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            return False
        if seconds < 0:
            return False
    return True

def _valid_calls(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"control", "probe"}:
        return False
    for role, expected in (
        ("control", (CONTROL_DENOISER_CALLS, 0, 0)),
        ("probe", (PROBE_DENOISER_CALLS, DECODER_CALLS, CRITIC_CALLS)),
    ):
        observed = value[role]
        if (
            not isinstance(observed, (list, tuple))
            or len(observed) != 3
            or not all(type(item) is int for item in observed)
            or tuple(observed) != expected
        ):
            return False
    return True

def _valid_identities(value: object, config_sha256: object) -> bool:
    expected = {
        "config_sha256",
        "cpu_preflight",
        "critic_identity",
        "fresh_evaluator_manifest",
        "runtime_inputs",
        "role_input_identities",
        "critic_input_identity",
        "critic_parameter_tree_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    runtime_inputs = value["runtime_inputs"]
    role_inputs = value["role_input_identities"]
    return (
        value["config_sha256"] == config_sha256
        and isinstance(runtime_inputs, Mapping)
        and set(runtime_inputs)
        == {
            "signed_value_config",
            "generator",
            "autoencoder",
            "critic_parameter",
        }
        and all(_valid_input_file_evidence(item) for item in runtime_inputs.values())
        and runtime_inputs["signed_value_config"]["sha256"] == config_sha256
        and isinstance(role_inputs, Mapping)
        and set(role_inputs) == {"control", "probe"}
        and all(_valid_file_evidence(item) for item in role_inputs.values())
        and _valid_file_evidence(value["critic_input_identity"])
        and _valid_sha256(value["critic_parameter_tree_sha256"])
        and all(
        _valid_file_evidence(value[name])
        for name in (
            "cpu_preflight",
            "critic_identity",
            "fresh_evaluator_manifest",
        )
        )
    )

def _valid_file_evidence(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"path", "size_bytes", "sha256"}:
        return False
    path = value["path"]
    size = value["size_bytes"]
    if not isinstance(path, str) or not Path(path).is_absolute():
        return False
    return (
        type(size) is int
        and size > 0
        and _valid_sha256(value["sha256"])
    )

def _valid_input_file_evidence(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"path", "size_bytes", "sha256"}:
        return False
    return (
        isinstance(value["path"], str)
        and Path(value["path"]).is_absolute()
        and type(value["size_bytes"]) is int
        and value["size_bytes"] > 0
        and _valid_sha256(value["sha256"])
    )

def _valid_verdict(value: object, target: object) -> bool:
    expected = {"target", "status", "informative", "checks", "reasons"}
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    checks = value["checks"]
    reasons = value["reasons"]
    if (
        value["target"] != target
        or value["status"] not in {"PASS", "BLOCKED_CRITIC", "ABSTAIN_UNRESOLVABLE"}
        or type(value["informative"]) is not bool
        or not isinstance(checks, Mapping)
        or set(checks) != _STATE_CHECK_KEYS
        or not all(
            isinstance(name, str) and type(result) is bool
            for name, result in checks.items()
        )
        or not isinstance(reasons, (list, tuple))
        or not all(isinstance(reason, str) and reason in checks for reason in reasons)
    ):
        return False
    expected_reasons = tuple(
        name for name in _STATE_REQUIRED_KEYS if checks[name] is False
    )
    required_pass = checks["informative"] is True and not expected_reasons

    if checks["perturbation_resolvable"] is False:
        expected_status = "ABSTAIN_UNRESOLVABLE"
    else:
        expected_status = "PASS" if required_pass else "BLOCKED_CRITIC"
    return (
        value["informative"] is checks["informative"]
        and tuple(reasons) == expected_reasons
        and value["status"] == expected_status
    )

def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None

def pending_targets(evidence_root: Path, targets: Sequence[str]) -> tuple[str, ...]:

    names = tuple(targets)
    if names != FROZEN_TARGETS:
        raise EvidenceError(f"frozen target inventory drifted: {names!r}")
    return tuple(
        target
        for target in names
        if target_state(target_root(evidence_root, target)) != "COMPLETE"
    )

def allocate_attempt_id(target_root: Path) -> str:

    attempts = Path(target_root) / "attempts"
    highest = 0
    if attempts.is_dir():
        for child in attempts.iterdir():
            match = re.fullmatch(r"local_([0-9]{4})", child.name)
            if match is not None:
                highest = max(highest, int(match.group(1)))
    if highest >= 9999:
        raise EvidenceError(f"attempt identity space exhausted under {attempts}")
    return f"local_{highest + 1:04d}"

def begin_local_attempt(root: Path, *, test_mode: bool = False) -> Path:

    for _ in range(_ATTEMPT_ALLOCATION_RETRIES):
        candidate = allocate_attempt_id(root)
        try:
            return begin_attempt(root, attempt_id=candidate, test_mode=test_mode)
        except EvidenceError as error:
            if "attempt already exists" not in str(error):
                raise
    raise EvidenceError(
        f"could not allocate an attempt identity under {root} after "
        f"{_ATTEMPT_ALLOCATION_RETRIES} attempts"
    )

def begin_attempt(root: Path, *, attempt_id: str, test_mode: bool = False) -> Path:

    destination = Path(root).resolve()
    _validate_terminal_location(destination, test_mode=test_mode)
    if target_state(destination) == "COMPLETE":
        raise EvidenceError(f"COMPLETE target evidence is immutable: {destination}")
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise EvidenceError(f"invalid attempt id: {attempt_id!r}")
    attempt = destination / "attempts" / attempt_id
    try:
        attempt.mkdir(parents=True, exist_ok=False)
        for role in ("control", "probe"):
            (attempt / role).mkdir(exist_ok=False)
    except FileExistsError as error:
        raise EvidenceError(
            f"attempt already exists and is immutable: {attempt}"
        ) from error
    return attempt

def write_attempt(
    attempt: Path, payload: Mapping[str, object], *, test_mode: bool = False
) -> Path:

    attempt_directory = Path(attempt).resolve()
    _validate_terminal_location(attempt_directory, test_mode=test_mode)
    destination = attempt_directory / "attempt.json"
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or payload.get("status")
        not in {
            "FAILED",
            "COMPLETE",
        }
    ):
        raise EvidenceError("attempt payload has invalid schema or status")
    if payload.get("status") == "COMPLETE":
        summary = {
            "schema_version": 1,
            "run_id": payload.get("run_id"),
            "target": payload.get("target"),
            "status": "COMPLETE",
            "attempt_id": payload.get("attempt_id"),
            "config_sha256": payload.get("config_sha256"),
        }
        if not _valid_complete_pair(summary, payload):
            raise EvidenceError("COMPLETE attempt payload does not match full schema")
    _write_json_new(destination, payload)
    return destination

def write_target_summary(
    root: Path,
    *,
    target: str,
    attempt_id: str,
    status: str,
    extra: Mapping[str, object] | None = None,
    test_mode: bool = False,
) -> Path:

    _require_target(target)
    if status not in {"FAILED", "COMPLETE"}:
        raise EvidenceError(f"invalid target status: {status!r}")
    target_directory = Path(root).resolve()
    _validate_terminal_location(target_directory, test_mode=test_mode)
    observed = target_state(target_directory)
    if observed == "COMPLETE":
        raise EvidenceError(
            f"COMPLETE target evidence is immutable: {target_directory}"
        )
    if observed == "CORRUPT":
        raise EvidenceError(
            f"corrupt target evidence cannot be rewritten: {target_directory}"
        )
    attempt_path = target_directory / "attempts" / attempt_id / "attempt.json"
    if not attempt_path.is_file():
        raise EvidenceError(
            f"target summary cannot point to missing attempt: {attempt_path}"
        )
    attempt_payload = read_json(attempt_path)
    if (
        attempt_payload.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or attempt_payload.get("target") != target
        or attempt_payload.get("status") != status
    ):
        raise EvidenceError("target summary does not match its immutable attempt")
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "target": target,
        "status": status,
        "attempt_id": attempt_id,
    }
    if extra:
        protected = set(payload).intersection(extra)
        if protected:
            raise EvidenceError(
                f"target summary extra fields replace protected keys: {sorted(protected)}"
            )
        payload.update(extra)
    if status == "COMPLETE" and not _valid_complete_pair(payload, attempt_payload):
        raise EvidenceError(
            "COMPLETE target summary does not match full attempt schema"
        )
    destination = target_directory / "manifest.json"
    _write_json_atomic(destination, payload)
    return destination

def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"invalid JSON evidence: {Path(path).resolve()}") from error
    if not isinstance(payload, dict):
        raise EvidenceError(f"JSON evidence must be an object: {Path(path).resolve()}")
    return payload

def _write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path).resolve()
    if destination.exists():
        raise EvidenceError(f"immutable evidence already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(payload)
    try:
        with destination.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise EvidenceError(
            f"immutable evidence already exists: {destination}"
        ) from error

def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.partial"
    if temporary.exists():
        raise EvidenceError(f"stale target-summary partial exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def write_json_new(
    path: Path, payload: Mapping[str, object], *, test_mode: bool = False
) -> None:

    _validate_terminal_location(Path(path).resolve(), test_mode=test_mode)
    _write_json_new(path, payload)

def _json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise EvidenceError("evidence is not finite canonical JSON") from error

def _require_target(target: str) -> None:
    if target not in FROZEN_TARGETS:
        raise EvidenceError(
            f"target is outside frozen feasibility inventory: {target!r}"
        )

def _validate_terminal_location(path: Path, *, test_mode: bool) -> None:
    del path, test_mode
