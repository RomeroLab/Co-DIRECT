
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import base64
import hashlib
import json
import os
import select
import signal
import sys
import threading
import time

WORKER_PROTOCOL_VERSION = 1
WORKER_ROLES = ("drive", "execute", "replay")

WORKER_READ_TIMEOUT_SECONDS = 21600.0
_MAX_FRAME_BYTES = 1 << 30
_BASELINE_KEYS = frozenset(
    {"meta_path", "path_hooks", "thread_ident", "signal_dispositions"}
)

class WorkerProtocolError(RuntimeError):
    pass

def send_message(descriptor: int, payload: Mapping[str, object]) -> None:
    body = json.dumps(
        {"protocol_version": WORKER_PROTOCOL_VERSION, "payload": dict(payload)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(body) > _MAX_FRAME_BYTES:
        raise WorkerProtocolError("worker frame exceeds the protocol limit")
    frame = len(body).to_bytes(8, "big") + body
    written = 0
    while written < len(frame):
        written += os.write(descriptor, frame[written:])

def _wait_readable(descriptor: int, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WorkerProtocolError("worker stream exceeded the read deadline")
    try:
        ready, _, _ = select.select([descriptor], [], [], remaining)
    except OSError as error:
        raise WorkerProtocolError("worker stream cannot be waited on") from error
    if not ready:
        raise WorkerProtocolError("worker stream exceeded the read deadline")

def _read_exactly(descriptor: int, count: int, deadline: float) -> bytes:
    buffer = b""
    while len(buffer) < count:
        _wait_readable(descriptor, deadline)
        chunk = os.read(descriptor, count - len(buffer))
        if not chunk:
            raise WorkerProtocolError("worker stream ended prematurely")
        buffer += chunk
    return buffer

def receive_message(
    descriptor: int, *, timeout_seconds: float = WORKER_READ_TIMEOUT_SECONDS
) -> Mapping[str, object]:

    deadline = time.monotonic() + timeout_seconds
    header = _read_exactly(descriptor, 8, deadline)
    length = int.from_bytes(header, "big")
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise WorkerProtocolError("worker frame length is not exact")
    body = _read_exactly(descriptor, length, deadline)

    try:
        ready, _, _ = select.select([descriptor], [], [], 0)
    except OSError as error:
        raise WorkerProtocolError("worker stream cannot be waited on") from error
    if ready and os.read(descriptor, 1):
        raise WorkerProtocolError("worker stream carries extra bytes after one frame")
    try:
        envelope = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerProtocolError("worker frame is not canonical JSON") from error
    if type(envelope) is not dict or set(envelope) != {"protocol_version", "payload"}:
        raise WorkerProtocolError("worker frame keys are not exact")
    if envelope["protocol_version"] != WORKER_PROTOCOL_VERSION:
        raise WorkerProtocolError("worker protocol version differs")
    payload = envelope["payload"]
    if type(payload) is not dict:
        raise WorkerProtocolError("worker payload is not an exact mapping")
    return payload

def signal_dispositions() -> tuple[tuple[int, object], ...]:

    observed: dict[int, object] = {}
    for name in dir(signal):
        if not name.startswith("SIG") or name.startswith("SIG_"):
            continue
        number = getattr(signal, name)
        if not isinstance(number, signal.Signals):
            continue
        try:
            observed[int(number)] = signal.getsignal(number)
        except (ValueError, OSError):
            continue
    return tuple(sorted(observed.items(), key=lambda row: row[0]))

@dataclass(frozen=True, slots=True)
class CapabilityBaseline:

    meta_path: tuple[object, ...]
    path_hooks: tuple[object, ...]
    thread_ident: int
    signal_dispositions: tuple[tuple[int, object], ...]

_BASELINE: CapabilityBaseline | None = None

def capture_capability_baseline() -> CapabilityBaseline:

    global _BASELINE
    if _BASELINE is not None:
        raise WorkerProtocolError("capability baseline is already captured")
    if threading.active_count() != 1:
        raise WorkerProtocolError("worker started with more than one thread")
    if sys.gettrace() is not None or sys.getprofile() is not None:
        raise WorkerProtocolError("worker started with a tracing or profiling hook")
    _BASELINE = CapabilityBaseline(
        meta_path=tuple(sys.meta_path),
        path_hooks=tuple(sys.path_hooks),
        thread_ident=threading.main_thread().ident or 0,
        signal_dispositions=signal_dispositions(),
    )
    return _BASELINE

def adopt_capability_baseline(snapshot: Mapping[str, object]) -> CapabilityBaseline:

    global _BASELINE
    if _BASELINE is not None:
        raise WorkerProtocolError("capability baseline is already captured")
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(_BASELINE_KEYS):
        raise WorkerProtocolError("capability baseline snapshot keys are not exact")
    registries: dict[str, tuple[object, ...]] = {}
    for key in ("meta_path", "path_hooks"):
        value = snapshot[key]
        if not isinstance(value, (list, tuple)):
            raise WorkerProtocolError(
                f"capability baseline {key} is not an exact sequence"
            )
        registries[key] = tuple(value)
    thread_ident = snapshot["thread_ident"]
    if type(thread_ident) is not int:
        raise WorkerProtocolError(
            "capability baseline thread_ident is not an exact int"
        )
    dispositions = snapshot["signal_dispositions"]
    if not isinstance(dispositions, (list, tuple)):
        raise WorkerProtocolError(
            "capability baseline signal_dispositions is not an exact sequence"
        )
    rows: list[tuple[int, object]] = []
    for row in dispositions:
        if not isinstance(row, tuple) or len(row) != 2 or type(row[0]) is not int:
            raise WorkerProtocolError(
                "capability baseline signal_dispositions row is not exact"
            )
        rows.append((row[0], row[1]))
    _BASELINE = CapabilityBaseline(
        meta_path=registries["meta_path"],
        path_hooks=registries["path_hooks"],
        thread_ident=thread_ident,
        signal_dispositions=tuple(rows),
    )
    return _BASELINE

def _assert_identical_registry(
    name: str, observed: tuple[object, ...], expected: tuple[object, ...]
) -> None:
    if len(observed) != len(expected):
        raise WorkerProtocolError(
            f"worker {name} changed length: {len(expected)} -> {len(observed)}"
        )
    for position, (live, original) in enumerate(zip(observed, expected)):
        if live is not original:
            raise WorkerProtocolError(
                f"worker {name} entry {position} was replaced or reordered"
            )

def _finder_label(finder: object) -> str:

    try:
        kind = type(finder)
        label = f"{kind.__module__}.{kind.__name__}"
    except Exception:
        return "<unnamable>"
    return label[:160] if type(label) is str else "<unnamable>"

def _assert_baseline_meta_path_intact(
    observed: tuple[tuple[int, object], ...], expected: tuple[object, ...]
) -> tuple[str, ...]:

    additions: list[str] = []
    cursor = 0
    for index, live in observed:
        if cursor < len(expected) and live is expected[cursor]:
            cursor += 1
            continue
        if index == 0:
            raise WorkerProtocolError(
                "worker sys.meta_path index 0 is not the authoritative finder"
            )
        additions.append(f"{index}:{_finder_label(live)}")
    if cursor != len(expected):
        raise WorkerProtocolError(
            "worker sys.meta_path baseline entry "
            f"{cursor} was removed, replaced or reordered"
        )
    return tuple(additions)

def _merge_recorded_additions(*groups: tuple[str, ...]) -> tuple[str, ...]:

    merged: list[str] = []
    for group in groups:
        for entry in group:
            if entry not in merged:
                merged.append(entry)
    return tuple(merged)

def assert_capability_boundary(realm_finder: object | None = None) -> tuple[str, ...]:

    baseline = _BASELINE
    if baseline is None:
        raise WorkerProtocolError("capability baseline was never captured")
    alive = [t for t in threading.enumerate() if t is not threading.main_thread()]
    if alive:
        raise WorkerProtocolError(f"worker has extra Python threads: {alive}")
    if threading.active_count() != 1:
        raise WorkerProtocolError("worker thread count is not exactly one")
    if threading.main_thread().ident != baseline.thread_ident:
        raise WorkerProtocolError("worker main thread identity changed")
    if sys.gettrace() is not None:
        raise WorkerProtocolError("worker has a tracing hook installed")
    if sys.getprofile() is not None:
        raise WorkerProtocolError("worker has a profiling hook installed")
    live_meta_path = list(sys.meta_path)
    if realm_finder is not None:
        if sum(1 for finder in live_meta_path if finder is realm_finder) != 1:
            raise WorkerProtocolError(
                "the catalog realm finder is not installed exactly once"
            )
        if live_meta_path[0] is not realm_finder:
            raise WorkerProtocolError(
                "the catalog realm finder is no longer first on sys.meta_path"
            )
    if realm_finder is None:
        guarded = tuple(enumerate(live_meta_path))
    else:
        guarded = tuple(
            (index, finder)
            for index, finder in enumerate(live_meta_path)
            if finder is not realm_finder
        )
    additions = _assert_baseline_meta_path_intact(guarded, baseline.meta_path)
    _assert_identical_registry(
        "sys.path_hooks", tuple(sys.path_hooks), baseline.path_hooks
    )
    observed_signals = signal_dispositions()
    if len(observed_signals) != len(baseline.signal_dispositions):
        raise WorkerProtocolError("worker signal disposition table changed size")
    for (number, handler), (expected_number, expected_handler) in zip(
        observed_signals, baseline.signal_dispositions
    ):
        if number != expected_number or handler is not expected_handler:
            raise WorkerProtocolError(
                f"worker signal disposition for signal {number} changed"
            )
    return additions

def _realm_catalog(finder: object) -> object:

    if not sys.meta_path or sys.meta_path[0] is not finder:
        raise WorkerProtocolError(
            "the catalog realm finder is not first on sys.meta_path"
        )
    catalog = getattr(finder, "catalog", None)
    if type(getattr(catalog, "identity_sha256", None)) is not str:
        raise WorkerProtocolError("catalog realm carries no authenticated identity")
    return catalog

def run_execution_worker(
    finder: object,
    *,
    seal_bytes: bytes,
    request_bytes: bytes,
    artifact_descriptors: Mapping[str, int],
    response_descriptor: int,
) -> int:

    catalog = _realm_catalog(finder)
    before = assert_capability_boundary(realm_finder=finder)
    from dive.evaluation.directional_seal import execute_confirmation_in_realm

    raw = execute_confirmation_in_realm(
        seal_bytes=seal_bytes,
        request_bytes=request_bytes,
        artifact_descriptors=artifact_descriptors,
    )

    after = assert_capability_boundary(realm_finder=finder)
    send_message(
        response_descriptor,
        {
            "kind": "raw_result",
            "catalog_identity": catalog.identity_sha256,
            "meta_path_additions": list(_merge_recorded_additions(before, after)),
            "schema": "dive-directional-raw-v2",
            "length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "raw_base64": base64.b64encode(raw).decode("ascii"),
        },
    )
    return 0

def run_replay_worker(
    finder: object,
    *,
    seal_bytes: bytes,
    request_bytes: bytes,
    raw_bytes: bytes,
    response_descriptor: int,
) -> int:

    catalog = _realm_catalog(finder)
    before = assert_capability_boundary(realm_finder=finder)
    from dive.evaluation.directional_seal import replay_confirmation_in_realm

    verdict = replay_confirmation_in_realm(
        seal_bytes=seal_bytes, request_bytes=request_bytes, raw_bytes=raw_bytes
    )
    after = assert_capability_boundary(realm_finder=finder)
    send_message(
        response_descriptor,
        {
            "kind": "replay_verdict",
            "catalog_identity": catalog.identity_sha256,
            "meta_path_additions": list(_merge_recorded_additions(before, after)),
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "verdict": verdict,
        },
    )
    return 0

def run_drive_worker(
    finder: object,
    *,
    seal_bytes: bytes,
    request_bytes: bytes,
    bootstrap_bytes: bytes,
    descriptors: Mapping[str, int],
    response_descriptor: int,
) -> int:

    catalog = _realm_catalog(finder)
    before = assert_capability_boundary(realm_finder=finder)
    from dive.evaluation.directional_seal import drive_confirmation_in_realm

    result = drive_confirmation_in_realm(
        seal_bytes=seal_bytes,
        request_bytes=request_bytes,
        bootstrap_bytes=bootstrap_bytes,
        descriptors=descriptors,
    )

    after = assert_capability_boundary(realm_finder=finder)
    send_message(
        response_descriptor,
        {
            "kind": "confirmation",
            "catalog_identity": catalog.identity_sha256,
            "meta_path_additions": list(_merge_recorded_additions(before, after)),
            "record_text": result["record_text"],
        },
    )
    return 0
