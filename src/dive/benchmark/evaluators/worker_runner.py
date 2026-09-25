
from __future__ import annotations

import json
import math
import subprocess
import time
from functools import cache
from pathlib import Path

from dive.benchmark.contracts import load_benchmark_contract
from dive.benchmark.evaluators.contracts import load_evaluator_registry
from dive.benchmark.evaluators.worker_protocol import (
    DisposableEvidencePolicy,
    WorkerFailureReason,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResult,
    _check_hash,
    _protocol_error,
    _validate_request_against_evaluator,
    _validate_result_identity,
)
from dive.benchmark.evaluators.worker_io import (
    close_descriptors,
    hash_open_descriptor,
    open_authenticated_file,
    open_cwd,
    open_read_fd,
    read_descriptor_bytes,
    reserve_outputs_atomically,
    resolve_evidence_roots,
    rewind_descriptor,
)
from dive.training.preflight import PreflightError, canonical_json_bytes

def _fixture_worker_path() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "tests/benchmark/evaluators/fixture_worker.py"
    )

def _fixture_result_path(family: str) -> Path:
    return _fixture_worker_path().with_name("fixtures") / f"{family}-result-v1.json"

def _read_request_fd(descriptor: int, request: WorkerRequest) -> None:
    try:
        payload = read_descriptor_bytes(descriptor)
    except OSError as error:
        raise _protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "cannot read retained request descriptor",
        ) from error
    expected = canonical_json_bytes(request.as_record())
    if payload != expected:
        raise _protocol_error(
            WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
            "request bytes differ from the authenticated in-memory request",
        )
    try:
        decoded = json.loads(payload.decode("utf-8"))
        if WorkerRequest.from_record(decoded).as_record() != request.as_record():
            raise _protocol_error(
                WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
                "request record differs from the authenticated request",
            )
    except WorkerProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, PreflightError) as error:
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID,
            "request is not canonical finite JSON",
        ) from error
    rewind_descriptor(descriptor)

def _parse_retained_result(descriptor: int) -> WorkerResult:
    try:
        payload = read_descriptor_bytes(descriptor)
        if not payload:
            raise _protocol_error(
                WorkerFailureReason.RESULT_MISSING, "fixture wrote no result"
            )
    except WorkerProtocolError:
        raise
    except OSError as error:
        raise _protocol_error(
            WorkerFailureReason.RESULT_MISSING, "cannot read retained result"
        ) from error
    try:
        record = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        if canonical_json_bytes(record) != payload:
            raise _protocol_error(
                WorkerFailureReason.RESULT_SCHEMA_INVALID,
                "result bytes are not canonical",
            )
        return WorkerResult.from_record(record)
    except WorkerProtocolError:
        raise
    except (UnicodeDecodeError, ValueError, PreflightError) as error:
        raise _protocol_error(
            WorkerFailureReason.RESULT_SCHEMA_INVALID, "result is not canonical JSON"
        ) from error

@cache
def _canonical_registry():

    repository = Path(__file__).resolve().parents[4]
    contract = load_benchmark_contract(repository / "configs/emergent/benchmark.yaml")
    return load_evaluator_registry(contract)

def run_worker(
    request: WorkerRequest,
    *,
    expected_dive_commit: str,
    policy: DisposableEvidencePolicy | None = None,
) -> Path:

    if policy is None:
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "production worker execution is not authorized",
        )
    _check_hash(
        expected_dive_commit,
        "expected_dive_commit",
        commit=True,
        reason=WorkerFailureReason.REQUEST_IDENTITY_DRIFT,
    )
    try:
        registry = _canonical_registry()
    except Exception as error:
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "canonical worktree registry could not authenticate",
        ) from error
    evaluator = _validate_request_against_evaluator(
        request, registry, expected_dive_commit
    )
    worker = evaluator.worker
    assert worker is not None
    source_path = Path(worker.resolved_working_directory) / worker.source.path
    fixture_path = Path(worker.resolved_working_directory) / worker.fixture_source.path
    if source_path != _fixture_worker_path() or fixture_path != _fixture_result_path(
        request.family
    ):
        raise _protocol_error(
            WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
            "registry does not select the checked-in fixture worker",
        )
    if request.timeout_seconds <= 0 or not math.isfinite(request.timeout_seconds):
        raise _protocol_error(
            WorkerFailureReason.REQUEST_SCHEMA_INVALID, "invalid request timeout"
        )
    terminal_root, _, expected_gid, group_policy = resolve_evidence_roots(
        evaluator, policy
    )
    retained: list[int] = []
    try:
        try:
            request_fd = open_read_fd(
                Path(request.request_path),
                terminal_root,
                expected_gid=expected_gid,
                group_policy=group_policy,
                policy=policy,
            )
        except WorkerProtocolError as error:
            if error.reason in {
                WorkerFailureReason.RESULT_MISSING,
                WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
            }:
                raise _protocol_error(
                    WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
                    "authenticated request path is unsafe or unavailable",
                ) from error
            raise
        retained.append(request_fd)
        _read_request_fd(request_fd, request)
        stdout_fd, stderr_fd, result_fd = reserve_outputs_atomically(
            (
                Path(request.stdout_path),
                Path(request.stderr_path),
                Path(request.result_path),
            ),
            terminal_root,
            request_path=Path(request.request_path),
            expected_gid=expected_gid,
            group_policy=group_policy,
            policy=policy,
        )
        retained.extend((stdout_fd, stderr_fd))
        retained.append(result_fd)
        source_fd = open_authenticated_file(source_path, worker.source, "worker source")
        retained.append(source_fd)
        interpreter_fd = open_authenticated_file(
            Path(worker.interpreter.path), worker.interpreter.identity, "interpreter"
        )
        retained.append(interpreter_fd)
        fixture_fd = open_authenticated_file(
            fixture_path, worker.fixture_source, "fixture result"
        )
        retained.append(fixture_fd)
        cwd_fd = open_cwd(request.working_directory)
        retained.append(cwd_fd)
        if policy.before_launch is not None:
            policy.before_launch()
        substitutions = {
            worker.interpreter.path: f"/proc/self/fd/{interpreter_fd}",
            worker.source.path: f"/proc/self/fd/{source_fd}",
            "{request_fd}": str(request_fd),
            "{result_fd}": str(result_fd),
            "{fixture_fd}": str(fixture_fd),
        }
        argv = tuple(substitutions.get(item, item) for item in request.argv)
        started = time.monotonic()
        try:
            with subprocess.Popen(
                argv,
                cwd=f"/proc/self/fd/{cwd_fd}",
                env=dict(request.environment),
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd,
                stderr=stderr_fd,
                pass_fds=(
                    request_fd,
                    result_fd,
                    source_fd,
                    interpreter_fd,
                    fixture_fd,
                    cwd_fd,
                ),
            ) as process:
                try:
                    remaining = request.timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, request.timeout_seconds)
                    return_code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as error:
                    process.kill()
                    process.wait()
                    raise _protocol_error(
                        WorkerFailureReason.WORKER_TIMEOUT,
                        "fixture worker exceeded authenticated request timeout",
                    ) from error
        except OSError as error:
            raise _protocol_error(
                WorkerFailureReason.REGISTRY_AUTHENTICATION_FAILED,
                "fixture worker could not start",
            ) from error
        if return_code != 0:
            raise _protocol_error(
                WorkerFailureReason.WORKER_NONZERO_EXIT,
                f"fixture worker exited {return_code}",
            )
        result = _parse_retained_result(result_fd)
        _validate_result_identity(result, request)
        for descriptor, identity in (
            (stdout_fd, result.stdout),
            (stderr_fd, result.stderr),
        ):
            digest, size = hash_open_descriptor(descriptor)
            if digest != identity.sha256 or size != identity.size_bytes:
                raise _protocol_error(
                    WorkerFailureReason.OUTPUT_IDENTITY_DRIFT,
                    "captured output identity drifted",
                )
        return Path(request.result_path)
    except FileExistsError as error:
        raise _protocol_error(
            WorkerFailureReason.EVIDENCE_PATH_VIOLATION,
            "create-new output already exists",
        ) from error
    finally:
        close_descriptors(retained)
