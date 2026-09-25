
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shlex
import socket
import sqlite3
import stat
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dive.benchmark.contracts import ArtifactIdentity, BenchmarkContract
from dive.benchmark.evidence import BenchmarkRun
from dive.benchmark.snapshot_authority import (
    PRODUCTION_SNAPSHOT_COUNT,
    MonitoredSnapshotAuthority,
    SnapshotAuthorityError,
    preflight_monitored_tree_authority,
)
from dive.signed_value.roots import EMERGENT_EVIDENCE_ROOT, RUNTIME_PYTHON
from dive.training.preflight import canonical_json_bytes

class IndependentFoldseekVerificationError(RuntimeError):
    pass

_ACCEPTED_STRUCTURE_RUN = "benchmark-structures-20260826d"
_REQUIRED_STRUCTURE_EVIDENCE = frozenset(
    {
        "attempt.json",
        "completion.json",
        "structure_inventory.jsonl",
        "structure_inventory.completion.json",
        "independent_verification.v5.json",
        "determinism_c_vs_d.json",
        "input_attempt.json",
        "input_completion.json",
        "input_inventory.json",
    }
)
_TASK4_ARTIFACT_ROLES = frozenset(
    {
        "query_database_manifest",
        "target_database_manifest",
        "raw_tsv",
        "mapped_parquet",
        "conflict_parquet",
        "structural_audit",
        "producer_database",
    }
)
_COUNT_ROLES = frozenset(
    {
        "raw_chain_hits",
        "conflicting_chain_hits",
        "parent_pair_conflicts",
        "affected_query_parent_groups",
        "affected_target_parent_groups",
        "affected_query_example_rows",
        "affected_target_example_rows",
    }
)
_PRODUCER_SCHEMA = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "contract_sha256",
        "execution_contract_sha256",
        "source_structure_run",
        "source_structure_evidence",
        "artifacts",
        "counts",
    }
)
_CANDIDATE_SCHEMA = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "contract_sha256",
        "execution_contract_sha256",
        "source_structure_run",
        "producer_commit",
        "producer_candidate",
        "artifacts",
        "source_structure_evidence",
        "counts",
        "verification",
    }
)
_COMPLETION_SCHEMA = frozenset(
    {"schema_version", "status", "run_id", "contract_sha256", "attempt", "payload"}
)
_COMPLETION_PAYLOAD_SCHEMA = frozenset(
    {
        "execution_contract_sha256",
        "source_structure_run",
        *_TASK4_ARTIFACT_ROLES,
        "producer_completion_candidate",
        "candidate_verification",
        "required_final_verifier",
    }
)
_FINAL_SCHEMA = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "contract_sha256",
        "execution_contract_sha256",
        "source_structure_run",
        "producer_commit",
        "completion",
        "candidate_verification",
        "producer_completion_candidate",
        "artifacts",
        "counts",
        "source_structure_evidence",
        "downstream_requirement",
    }
)
_AUDIT_SCHEMA = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "contract_sha256",
        "execution_contract_sha256",
        "source_structure_run",
        "source_structure_evidence",
        "thresholds",
        "execution",
        "query_database_manifest",
        "target_database_manifest",
        "raw_tsv",
        "mapped_parquet",
        "conflict_parquet",
        "counts",
        "aggregation",
        "view_effect",
        "units",
        "required_independent_verification",
    }
)
_EXECUTION_SCHEMA = frozenset(
    {
        "command",
        "environment",
        "binary",
        "binary_post_execution",
        "version_output",
        "version_output_post_execution",
        "repo_commit",
        "start_utc",
        "end_utc",
        "duration_seconds",
        "exit_code",
        "stdout",
        "stderr",
        "raw_tsv",
        "stage",
    }
)
_ATTEMPT_SCHEMA = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "argv",
        "argv_shell",
        "repo_commit",
        "contract_sha256",
        "upstream_commit",
        "runtime_python",
        "host",
        "evidence_path",
        "bulk_path",
    }
)
_VERIFICATION_SCHEMA = frozenset(
    {
        "independent_raw_parser",
        "independent_key_decoder_and_mapping",
        "independent_finite_coverage_and_inclusive_thresholds",
        "independent_mapped_parquet_reconciliation",
        "independent_conflict_parquet_reconciliation",
        "disk_bounded_duplicate_and_parent_pair_aggregation",
    }
)

@dataclass(frozen=True, slots=True)
class IndependentCounts:
    raw_hit_count: int
    conflicting_chain_hit_count: int
    parent_pair_conflict_count: int
    affected_query_parent_groups: int
    affected_target_parent_groups: int
    affected_query_example_rows: int
    affected_target_example_rows: int

@dataclass(frozen=True, slots=True)
class AuthenticatedFoldseekChain:

    final_verification: ArtifactIdentity
    completion: ArtifactIdentity
    candidate_verification: ArtifactIdentity
    producer_completion_candidate: ArtifactIdentity
    artifacts: Mapping[str, ArtifactIdentity]
    sources: Mapping[str, ArtifactIdentity]
    counts: Mapping[str, int]

@dataclass(frozen=True, slots=True)
class _RawHit:
    query: str
    target: str
    alignment_length: int
    query_length: int
    target_length: int
    tm_score: float
    evalue: float

@dataclass(frozen=True, slots=True)
class _RetainedArtifact:
    raw: bytes | None
    identity: ArtifactIdentity

class _RetainedEvidenceSet:

    def __init__(self) -> None:
        self._descriptors: dict[str, int] = {}
        self._paths: dict[str, Path] = {}
        self._before: dict[str, os.stat_result] = {}
        self._expected: dict[str, ArtifactIdentity | None] = {}
        self._reads: dict[str, tuple[bytes | None, str, int]] = {}

    def read(
        self,
        label: str,
        path: Path,
        *,
        expected: ArtifactIdentity | None,
        retain_bytes: bool,
    ) -> bytes | None:
        path = Path(path)
        if label in self._descriptors or path in self._paths.values():
            raise IndependentFoldseekVerificationError(
                f"retained evidence label or path is duplicated: {label}"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise IndependentFoldseekVerificationError(
                f"retained evidence {label} is missing or unreadable"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise IndependentFoldseekVerificationError(
                f"retained evidence is not a unique regular file: {label}"
            )
        self._descriptors[label] = descriptor
        self._paths[label] = path
        self._before[label] = metadata
        self._expected[label] = expected
        try:
            observed = _read_retained_descriptor(
                descriptor, label=label, retain_bytes=retain_bytes
            )
        except Exception:
            self.close()
            raise
        self._reads[label] = observed
        return observed[0]

    def finish(self) -> Mapping[str, _RetainedArtifact]:
        try:
            final_descriptors = {
                label: os.fstat(descriptor)
                for label, descriptor in self._descriptors.items()
            }
            final_paths = {
                label: os.stat(path, follow_symlinks=False)
                for label, path in self._paths.items()
            }
        except OSError as error:
            raise IndependentFoldseekVerificationError(
                "retained evidence changed during collective authentication"
            ) from error
        result: dict[str, _RetainedArtifact] = {}
        for label, path in self._paths.items():
            before = self._before[label]
            descriptor = final_descriptors[label]
            pathname = final_paths[label]
            raw, digest, size = self._reads[label]
            if not (
                _stat_key(before) == _stat_key(descriptor) == _stat_key(pathname)
                and stat.S_ISREG(pathname.st_mode)
                and pathname.st_nlink == 1
                and size == before.st_size
            ):
                raise IndependentFoldseekVerificationError(
                    f"retained evidence changed during collective authentication: {label}"
                )
            identity = ArtifactIdentity(str(path.absolute()), digest, size)
            expected = self._expected[label]
            if expected is not None and identity != expected:
                raise IndependentFoldseekVerificationError(
                    f"retained evidence identity drifted: {label}"
                )
            result[label] = _RetainedArtifact(raw, identity)
        return result

    def observed(self) -> Mapping[str, _RetainedArtifact]:

        return {
            label: _RetainedArtifact(
                raw,
                ArtifactIdentity(str(self._paths[label].absolute()), digest, size),
            )
            for label, (raw, digest, size) in self._reads.items()
        }

    def descriptor(self, label: str) -> int:

        try:
            descriptor = self._descriptors[label]
            os.lseek(descriptor, 0, os.SEEK_SET)
        except (KeyError, OSError) as error:
            raise IndependentFoldseekVerificationError(
                f"retained evidence descriptor is unavailable: {label}"
            ) from error
        return descriptor

    def close(self) -> None:
        for descriptor in self._descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._descriptors.clear()

def _read_retained_descriptor(
    descriptor: int, *, label: str, retain_bytes: bool
) -> tuple[bytes | None, str, int]:
    del label
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1 << 20):
        digest.update(chunk)
        size += len(chunk)
        if retain_bytes:
            chunks.append(chunk)
    return (b"".join(chunks) if retain_bytes else None, digest.hexdigest(), size)

def independently_verify_outputs(
    *,
    query_manifest: Path | bytes,
    target_manifest: Path | bytes,
    raw_tsv: Path | int,
    mapped_parquet: Path | int,
    conflict_parquet: Path | int,
    work_database: Path,
    min_tm_score: float,
    min_shorter_coverage: float,
    example_ids_by_parent: Mapping[str, tuple[str, ...]] | None = None,
    source_inventory: ArtifactIdentity | None = None,
    source_completion: ArtifactIdentity | None = None,
    source_inventory_bytes: bytes | None = None,
) -> IndependentCounts:

    query_payload = _manifest_payload(query_manifest, "query")
    target_payload = _manifest_payload(target_manifest, "target")
    query = _load_manifest_payload(query_payload, "query", None, None)
    target = _load_manifest_payload(target_payload, "target", None, None)
    overlap = set(query) & set(target)
    if overlap:
        raise IndependentFoldseekVerificationError(
            "independent key mapping spans query and target manifests"
        )
    required_parents = {row["parent_id"] for row in (*query.values(), *target.values())}
    if example_ids_by_parent is None:
        example_ids_by_parent = _manifest_examples_payload(
            query_payload, target_payload
        )
    if set(example_ids_by_parent) != required_parents:
        raise IndependentFoldseekVerificationError(
            "independent example-row mapping is incomplete"
        )
    if work_database.exists() or work_database.is_symlink():
        raise IndependentFoldseekVerificationError(
            "independent verification database already exists"
        )
    connection = sqlite3.connect(work_database)
    try:
        _initialize_database(connection, example_ids_by_parent)
        if (source_inventory is None) != (source_completion is None):
            raise IndependentFoldseekVerificationError(
                "independent source-inventory evidence is incomplete"
            )
        if source_inventory is not None and source_completion is not None:
            _populate_required_sources(
                connection,
                query_payload,
                "query",
                source_inventory,
                source_completion,
            )
            _populate_required_sources(
                connection,
                target_payload,
                "target",
                source_inventory,
                source_completion,
            )
            _verify_source_inventory(
                connection, source_inventory, raw=source_inventory_bytes
            )
        observed_rows = _parquet_rows(mapped_parquet)
        raw_count = 0
        conflict_count = 0
        for hit in _raw_hits(raw_tsv):
            raw_count += 1
            query_row = query.get(hit.query)
            target_row = target.get(hit.target)
            if query_row is None or target_row is None:
                raise IndependentFoldseekVerificationError(
                    "independent raw key is unknown"
                )
            try:
                connection.execute(
                    "INSERT INTO raw_pairs(query_key, target_key) VALUES (?, ?)",
                    (hit.query, hit.target),
                )
            except sqlite3.IntegrityError as error:
                raise IndependentFoldseekVerificationError(
                    "independent raw parser found a duplicate query/target pair"
                ) from error
            coverage = _coverage(hit)
            conflict = hit.tm_score >= min_tm_score and coverage >= min_shorter_coverage
            observed = next(observed_rows, None)
            expected = _mapped_row(hit, query_row, target_row, coverage, conflict)
            if observed != expected:
                raise IndependentFoldseekVerificationError(
                    "independent mapped Parquet reconciliation failed"
                )
            if conflict:
                conflict_count += 1
                connection.execute(
                    """
                    INSERT INTO conflicts(
                        query_family, query_parent_id,
                        target_family, target_parent_id, chain_hit_count
                    ) VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(
                        query_family, query_parent_id,
                        target_family, target_parent_id
                    ) DO UPDATE SET chain_hit_count = chain_hit_count + 1
                    """,
                    (
                        query_row["family"],
                        query_row["parent_id"],
                        target_row["family"],
                        target_row["parent_id"],
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO query_parents(parent_id) VALUES (?)",
                    (query_row["parent_id"],),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO target_parents(parent_id) VALUES (?)",
                    (target_row["parent_id"],),
                )
        if next(observed_rows, None) is not None:
            raise IndependentFoldseekVerificationError(
                "independent mapped Parquet has extra rows"
            )
        connection.commit()
        _verify_conflict_parquet(connection, conflict_parquet)
        return _counts(connection, raw_count, conflict_count)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

def _load_manifest(path: Path, scope: str) -> dict[str, dict[str, str]]:
    payload = _manifest_payload(path, scope)
    return _load_manifest_payload(payload, scope, None, None)

def _manifest_payload(source: Path | bytes, scope: str) -> Mapping[str, object]:
    try:
        if isinstance(source, bytes):
            payload = json.loads(source)
            if (
                not isinstance(payload, Mapping)
                or canonical_json_bytes(payload) != source
            ):
                raise IndependentFoldseekVerificationError(
                    f"independent {scope} manifest is not canonical JSON"
                )
        else:
            _raw, payload = _read_canonical_json(
                Path(source), f"independent {scope} manifest"
            )
    except (
        UnicodeError,
        json.JSONDecodeError,
        IndependentFoldseekVerificationError,
    ) as error:
        raise IndependentFoldseekVerificationError(
            f"independent {scope} manifest is unreadable"
        ) from error
    return payload

def _load_manifest_payload(
    payload: Mapping[str, object],
    scope: str,
    expected_directory: Path | None,
    sources: Mapping[str, ArtifactIdentity] | None,
    source_directory: Path | None = None,
) -> dict[str, dict[str, str]]:
    if expected_directory is not None:
        _validate_snapshot_directory(expected_directory)
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    expected_schema = {
        "schema_version",
        "scope",
        "source_structure_run",
        "source_structure_inventory",
        "source_structure_completion",
        "directory",
        "record_count",
        "parent_group_count",
        "example_rows",
        "entries",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_schema
        or payload.get("schema_version")
        != "dive-benchmark-foldseek-database-manifest-v1"
        or payload.get("scope") != scope
        or not isinstance(entries, list)
        or payload.get("record_count") != len(entries)
    ):
        raise IndependentFoldseekVerificationError(
            f"independent {scope} manifest omits or duplicates required keys"
        )
    directory_value = payload.get("directory")
    if type(directory_value) is not str or not directory_value:
        raise IndependentFoldseekVerificationError(
            f"independent {scope} manifest directory is invalid"
        )
    directory = Path(directory_value)
    if (
        not directory.is_absolute()
        or (expected_directory is not None and directory != expected_directory)
        or (
            sources is not None
            and (
                payload.get("source_structure_run") != _ACCEPTED_STRUCTURE_RUN
                or payload.get("source_structure_inventory")
                != _identity_mapping(sources["structure_inventory.jsonl"])
                or payload.get("source_structure_completion")
                != _identity_mapping(sources["structure_inventory.completion.json"])
            )
        )
    ):
        raise IndependentFoldseekVerificationError(
            f"independent {scope} snapshot directory or source is invalid"
        )
    result: dict[str, dict[str, str]] = {}
    parents: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping):
            raise IndependentFoldseekVerificationError(
                f"independent {scope} manifest entry is invalid"
            )
        if set(item) != {
            "foldseek_key",
            "relative_path",
            "family",
            "partition",
            "parent_id",
            "example_id",
            "chain_id",
            "component_id",
            "role",
            "materialized",
            "task3_materialized",
        }:
            raise IndependentFoldseekVerificationError(
                f"independent {scope} manifest entry schema is invalid"
            )
        key = item.get("foldseek_key")
        family = item.get("family")
        parent = item.get("parent_id")
        chain = item.get("chain_id")
        relative_path = item.get("relative_path")
        strings = {
            "family": family,
            "partition": item.get("partition"),
            "parent_id": parent,
            "example_id": item.get("example_id"),
            "chain_id": chain,
            "role": item.get("role"),
        }
        if (
            type(key) is not str
            or key in result
            or not all(type(value) is str and value for value in strings.values())
            or _decode_key(key) != (family, parent, chain)
            or relative_path != f"{key}.pdb"
        ):
            raise IndependentFoldseekVerificationError(
                f"independent {scope} key mapping is invalid or duplicated"
            )
        snapshot_identity = _authenticate_declared(
            item.get("materialized"), f"independent {scope} snapshot {key}"
        )
        task3_identity = _declared_identity(
            item.get("task3_materialized"),
            f"independent Task 3 source {key}",
        )
        expected_snapshot = directory / str(relative_path)
        if expected_directory is not None:
            try:
                snapshot_metadata = os.stat(expected_snapshot, follow_symlinks=False)
            except OSError as error:
                raise IndependentFoldseekVerificationError(
                    f"independent {scope} snapshot is unavailable: {key}"
                ) from error
            if (
                not stat.S_ISREG(snapshot_metadata.st_mode)
                or snapshot_metadata.st_nlink != 1
                or stat.S_IMODE(snapshot_metadata.st_mode) != 0o440
            ):
                raise IndependentFoldseekVerificationError(
                    f"independent {scope} snapshot permissions are mutable: {key}"
                )
        if (
            Path(snapshot_identity.path) != expected_snapshot
            or (
                source_directory is not None
                and Path(task3_identity.path) != source_directory / str(relative_path)
            )
            or snapshot_identity.sha256 != task3_identity.sha256
            or snapshot_identity.size_bytes != task3_identity.size_bytes
        ):
            raise IndependentFoldseekVerificationError(
                f"independent {scope} snapshot identity differs from Task 3: {key}"
            )
        result[key] = strings
        parents.add(str(parent))
    if payload.get("parent_group_count") != len(parents):
        raise IndependentFoldseekVerificationError(
            f"independent {scope} parent count is invalid"
        )
    if expected_directory is not None:
        _validate_snapshot_directory(expected_directory)
    example_rows = payload.get("example_rows")
    if not isinstance(example_rows, list) or any(
        not isinstance(row, Mapping) or set(row) != {"parent_id", "example_ids"}
        for row in example_rows
    ):
        raise IndependentFoldseekVerificationError(
            f"independent {scope} example-row schema is invalid"
        )
    return result

def _validate_snapshot_directory(path: Path) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise IndependentFoldseekVerificationError(
            f"immutable snapshot directory is unavailable: {path}"
        ) from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o2550
    ):
        raise IndependentFoldseekVerificationError(
            f"immutable snapshot directory permissions drifted: {path}"
        )

def _manifest_examples(*paths: Path) -> Mapping[str, tuple[str, ...]]:
    return _manifest_examples_payload(
        *(
            _read_canonical_json(path, "independent example-row manifest")[1]
            for path in paths
        )
    )

def _manifest_examples_payload(
    *payloads: Mapping[str, object],
) -> Mapping[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for payload in payloads:
        rows = payload.get("example_rows") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise IndependentFoldseekVerificationError(
                "independent example-row manifest is missing"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise IndependentFoldseekVerificationError(
                    "independent example-row record is invalid"
                )
            parent = row.get("parent_id")
            examples = row.get("example_ids")
            if (
                type(parent) is not str
                or not parent
                or parent in result
                or not isinstance(examples, list)
                or not examples
                or len(examples) != len(set(examples))
                or not all(type(item) is str and item for item in examples)
            ):
                raise IndependentFoldseekVerificationError(
                    "independent example-row record is duplicated or invalid"
                )
            result[parent] = tuple(examples)
    return result

def _populate_required_sources(
    connection: sqlite3.Connection,
    payload: Mapping[str, object],
    scope: str,
    source_inventory: ArtifactIdentity,
    source_completion: ArtifactIdentity,
) -> None:
    if payload.get("source_structure_inventory") != _identity_mapping(
        source_inventory
    ) or payload.get("source_structure_completion") != _identity_mapping(
        source_completion
    ):
        raise IndependentFoldseekVerificationError(
            f"independent {scope} manifest source cross-link drifted"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise IndependentFoldseekVerificationError(
            f"independent {scope} manifest entries are invalid"
        )
    for item in entries:
        if not isinstance(item, Mapping):
            raise IndependentFoldseekVerificationError(
                f"independent {scope} manifest entry is invalid"
            )
        source = _declared_identity(
            item.get("task3_materialized"),
            f"independent Task 3 source {item.get('foldseek_key')}",
        )
        connection.execute(
            """
            INSERT INTO required_sources(
                foldseek_key, scope, family, parent_id, chain_id,
                source_path, source_sha256, source_size_bytes, observed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                item.get("foldseek_key"),
                scope,
                item.get("family"),
                item.get("parent_id"),
                item.get("chain_id"),
                source.path,
                source.sha256,
                source.size_bytes,
            ),
        )

def _verify_source_inventory(
    connection: sqlite3.Connection,
    source_inventory: ArtifactIdentity,
    *,
    raw: bytes | None = None,
) -> None:
    if raw is None and _file_identity(Path(source_inventory.path)) != source_inventory:
        raise IndependentFoldseekVerificationError(
            "independent source structure inventory identity drifted"
        )
    try:
        handle = (
            io.BytesIO(raw)
            if raw is not None
            else Path(source_inventory.path).open("rb")
        )
    except OSError as error:
        raise IndependentFoldseekVerificationError(
            "independent source structure inventory is unreadable"
        ) from error
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                item = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise IndependentFoldseekVerificationError(
                    f"independent source inventory line {line_number} is malformed"
                ) from error
            if (
                not isinstance(item, Mapping)
                or canonical_json_bytes(item) != raw
                or item.get("schema_version") != "dive-benchmark-structure-record-v1"
                or item.get("failure_reason") is not None
            ):
                raise IndependentFoldseekVerificationError(
                    f"independent source inventory line {line_number} is invalid"
                )
            source = _declared_identity(
                item.get("materialized"),
                f"independent source inventory line {line_number}",
            )
            cursor = connection.execute(
                """
                UPDATE required_sources SET observed = 1
                WHERE foldseek_key = ? AND scope = ? AND family = ?
                  AND parent_id = ? AND chain_id = ? AND source_path = ?
                  AND source_sha256 = ? AND source_size_bytes = ?
                  AND observed = 0
                """,
                (
                    item.get("foldseek_key"),
                    item.get("scope"),
                    item.get("family"),
                    item.get("parent_id"),
                    item.get("chain_id"),
                    source.path,
                    source.sha256,
                    source.size_bytes,
                ),
            )
            if cursor.rowcount != 1:
                raise IndependentFoldseekVerificationError(
                    "independent source inventory has an unknown, duplicate, "
                    "or mismapped key"
                )
    if raw is None and _file_identity(Path(source_inventory.path)) != source_inventory:
        raise IndependentFoldseekVerificationError(
            "independent source structure inventory changed while reconciling"
        )
    missing = int(
        connection.execute(
            "SELECT COUNT(*) FROM required_sources WHERE observed = 0"
        ).fetchone()[0]
    )
    if missing:
        raise IndependentFoldseekVerificationError(
            "independent database manifests omit required source keys"
        )

def _decode_key(key: str) -> tuple[str, str, str]:
    if not key.startswith("dive1-"):
        raise IndependentFoldseekVerificationError("independent key prefix is invalid")
    encoded = key[6:]
    try:
        decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        value = json.loads(decoded)
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise IndependentFoldseekVerificationError(
            "independent key encoding is invalid"
        ) from error
    if (
        not isinstance(value, list)
        or len(value) != 3
        or value[0] not in {"binder", "ame", "antibody"}
        or not all(type(item) is str and item for item in value)
        or _encode_key(value[0], value[1], value[2]) != key
    ):
        raise IndependentFoldseekVerificationError("independent key payload is invalid")
    return value[0], value[1], value[2]

def _encode_key(family: str, parent: str, chain: str) -> str:
    raw = json.dumps(
        [family, parent, chain], ensure_ascii=False, separators=(",", ":")
    ).encode()
    return "dive1-" + base64.urlsafe_b64encode(raw).decode().rstrip("=")

def _raw_hits(source: Path | int) -> Iterator[_RawHit]:
    try:
        if type(source) is int:
            os.lseek(source, 0, os.SEEK_SET)
            handle = os.fdopen(os.dup(source), "r", encoding="utf-8", newline="")
        else:
            handle = Path(source).open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise IndependentFoldseekVerificationError(
            "independent raw TSV is unreadable"
        ) from error
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            values = raw.removesuffix("\n").removesuffix("\r").split("\t")
            if len(values) != 7:
                raise IndependentFoldseekVerificationError(
                    f"independent raw TSV line {line_number} is malformed"
                )
            try:
                hit = _RawHit(
                    values[0],
                    values[1],
                    int(values[2]),
                    int(values[3]),
                    int(values[4]),
                    float(values[5]),
                    float(values[6]),
                )
            except ValueError as error:
                raise IndependentFoldseekVerificationError(
                    f"independent raw TSV line {line_number} is malformed"
                ) from error
            if (
                not hit.query
                or not hit.target
                or hit.alignment_length < 0
                or hit.query_length <= 0
                or hit.target_length <= 0
                or not math.isfinite(hit.tm_score)
                or not 0.0 <= hit.tm_score <= 1.0
                or not math.isfinite(hit.evalue)
                or hit.evalue < 0.0
            ):
                raise IndependentFoldseekVerificationError(
                    f"independent raw TSV line {line_number} has invalid arithmetic"
                )
            yield hit

def _coverage(hit: _RawHit) -> float:
    value = hit.alignment_length / min(hit.query_length, hit.target_length)
    if not math.isfinite(value):
        raise IndependentFoldseekVerificationError(
            "independent shorter-chain coverage is not finite"
        )
    return min(1.0, max(0.0, value))

def _parquet_rows(source: Path | int) -> Iterator[dict[str, object]]:
    handle = None
    try:
        import pyarrow.parquet as pq

        if type(source) is int:
            os.lseek(source, 0, os.SEEK_SET)
            handle = os.fdopen(os.dup(source), "rb")
            parquet = pq.ParquetFile(handle)
        else:
            parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=50_000):
            yield from batch.to_pylist()
    except Exception as error:
        raise IndependentFoldseekVerificationError(
            "independent mapped Parquet is unreadable"
        ) from error
    finally:
        if handle is not None:
            handle.close()

def _mapped_row(
    hit: _RawHit,
    query: Mapping[str, str],
    target: Mapping[str, str],
    coverage: float,
    conflict: bool,
) -> dict[str, object]:
    return {
        "query_key": hit.query,
        "target_key": hit.target,
        "alignment_length": hit.alignment_length,
        "query_length": hit.query_length,
        "target_length": hit.target_length,
        "tm_score": hit.tm_score,
        "evalue": hit.evalue,
        "shorter_coverage": coverage,
        "structural_conflict": conflict,
        "query_family": query["family"],
        "query_partition": query["partition"],
        "query_parent_id": query["parent_id"],
        "query_example_id": query["example_id"],
        "query_chain_id": query["chain_id"],
        "query_role": query["role"],
        "target_family": target["family"],
        "target_partition": target["partition"],
        "target_parent_id": target["parent_id"],
        "target_example_id": target["example_id"],
        "target_chain_id": target["chain_id"],
        "target_role": target["role"],
    }

def _verify_conflict_parquet(
    connection: sqlite3.Connection, source: Path | int
) -> None:
    expected = iter(
        connection.execute(
            """
            SELECT query_family, query_parent_id,
                   target_family, target_parent_id, chain_hit_count
            FROM conflicts
            ORDER BY query_family, query_parent_id,
                     target_family, target_parent_id
            """
        )
    )
    observed = _parquet_rows(source)
    columns = (
        "query_family",
        "query_parent_id",
        "target_family",
        "target_parent_id",
        "chain_hit_count",
    )
    for row in observed:
        want = next(expected, None)
        if want is None or row != dict(zip(columns, want, strict=True)):
            raise IndependentFoldseekVerificationError(
                "independent conflict Parquet reconciliation failed"
            )
    if next(expected, None) is not None:
        raise IndependentFoldseekVerificationError(
            "independent conflict Parquet omits parent pairs"
        )

def _initialize_database(
    connection: sqlite3.Connection,
    examples: Mapping[str, tuple[str, ...]],
) -> None:
    connection.executescript(
        """
        CREATE TABLE raw_pairs(
            query_key TEXT NOT NULL,
            target_key TEXT NOT NULL,
            PRIMARY KEY(query_key, target_key)
        ) WITHOUT ROWID;
        CREATE TABLE conflicts(
            query_family TEXT NOT NULL,
            query_parent_id TEXT NOT NULL,
            target_family TEXT NOT NULL,
            target_parent_id TEXT NOT NULL,
            chain_hit_count INTEGER NOT NULL,
            PRIMARY KEY(
                query_family, query_parent_id,
                target_family, target_parent_id
            )
        ) WITHOUT ROWID;
        CREATE TABLE examples(
            parent_id TEXT NOT NULL,
            example_id TEXT NOT NULL UNIQUE,
            PRIMARY KEY(parent_id, example_id)
        ) WITHOUT ROWID;
        CREATE TABLE query_parents(
            parent_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE target_parents(
            parent_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE required_sources(
            foldseek_key TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            family TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            chain_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_size_bytes INTEGER NOT NULL,
            observed INTEGER NOT NULL CHECK(observed IN (0, 1))
        ) WITHOUT ROWID;
        """
    )
    connection.executemany(
        "INSERT INTO examples(parent_id, example_id) VALUES (?, ?)",
        (
            (parent, example)
            for parent, values in examples.items()
            for example in values
        ),
    )

def _counts(
    connection: sqlite3.Connection, raw_count: int, conflict_count: int
) -> IndependentCounts:
    pair_count = int(connection.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0])
    query_parents = int(
        connection.execute("SELECT COUNT(*) FROM query_parents").fetchone()[0]
    )
    target_parents = int(
        connection.execute("SELECT COUNT(*) FROM target_parents").fetchone()[0]
    )
    query_examples = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM examples e
            JOIN query_parents p ON p.parent_id = e.parent_id
            """
        ).fetchone()[0]
    )
    target_examples = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM examples e
            JOIN target_parents p ON p.parent_id = e.parent_id
            """
        ).fetchone()[0]
    )
    return IndependentCounts(
        raw_count,
        conflict_count,
        pair_count,
        query_parents,
        target_parents,
        query_examples,
        target_examples,
    )

@dataclass(frozen=True, slots=True)
class _ProducerChain:
    producer_identity: ArtifactIdentity
    producer: Mapping[str, object]
    audit: Mapping[str, object]
    artifacts: Mapping[str, ArtifactIdentity]
    sources: Mapping[str, ArtifactIdentity]
    attempt: Mapping[str, object]
    producer_commit: str
    manifest_raw: Mapping[str, bytes]

def write_candidate_verification(
    run: BenchmarkRun,
    contract: BenchmarkContract,
    *,
    producer_candidate: Path,
    work_database: Path,
    min_tm_score: float,
    min_shorter_coverage: float,
) -> ArtifactIdentity:

    _validate_run_context(run, contract)
    _preflight_verifier_snapshot_monitor()
    reader = _RetainedEvidenceSet()
    snapshot_authority: MonitoredSnapshotAuthority | None = None
    try:
        snapshot_authority = _begin_verifier_snapshot_authority(run, contract)
        chain = _open_producer_chain(
            reader,
            run,
            contract,
            producer_candidate=producer_candidate,
            expected_producer=None,
        )
        _initialize_verifier_snapshot_authority(
            snapshot_authority, chain, run, contract
        )
        retained = reader.observed()
        _validate_producer_chain(
            chain,
            retained,
            run,
            contract,
            min_tm_score=min_tm_score,
            min_shorter_coverage=min_shorter_coverage,
        )
        source_inventory_raw = retained["source:structure_inventory.jsonl"].raw
        if source_inventory_raw is None:
            raise IndependentFoldseekVerificationError(
                "retained source inventory bytes are unavailable"
            )
        counts = independently_verify_outputs(
            query_manifest=chain.manifest_raw["query_database_manifest"],
            target_manifest=chain.manifest_raw["target_database_manifest"],
            raw_tsv=reader.descriptor("artifact:raw_tsv"),
            mapped_parquet=reader.descriptor("artifact:mapped_parquet"),
            conflict_parquet=reader.descriptor("artifact:conflict_parquet"),
            work_database=work_database,
            min_tm_score=min_tm_score,
            min_shorter_coverage=min_shorter_coverage,
            source_inventory=chain.sources["structure_inventory.jsonl"],
            source_completion=chain.sources["structure_inventory.completion.json"],
            source_inventory_bytes=source_inventory_raw,
        )
        count_mapping = _count_mapping(counts)
        if (
            chain.producer.get("counts") != count_mapping
            or chain.audit.get("counts") != count_mapping
        ):
            raise IndependentFoldseekVerificationError(
                "independent counts disagree with producer evidence"
            )

        _validate_snapshot_manifests(chain, run, contract)
        reader.finish()
        _finish_snapshot_authority(snapshot_authority)
    finally:
        if snapshot_authority is not None:
            snapshot_authority.close()
        reader.close()
    payload = {
        "schema_version": "dive-benchmark-foldseek-independent-candidate-v1",
        "status": "PASS",
        "run_id": run.run_id,
        "contract_sha256": contract.semantic_sha256,
        "execution_contract_sha256": contract.execution_sha256,
        "source_structure_run": _ACCEPTED_STRUCTURE_RUN,
        "producer_commit": chain.producer_commit,
        "producer_candidate": _identity_mapping(chain.producer_identity),
        "artifacts": _identity_role_mapping(chain.artifacts),
        "source_structure_evidence": _identity_role_mapping(chain.sources),
        "counts": count_mapping,
        "verification": {name: True for name in sorted(_VERIFICATION_SCHEMA)},
    }
    destination = run.evidence_dir / "independent_verification.candidate.json"
    _write_terminal_payload(run, destination, payload)
    return _file_identity(destination)

def write_final_verification(
    run: BenchmarkRun,
    contract: BenchmarkContract,
    *,
    completion: Path,
) -> ArtifactIdentity:

    _validate_run_context(run, contract)
    _preflight_verifier_snapshot_monitor()
    reader = _RetainedEvidenceSet()
    snapshot_authority: MonitoredSnapshotAuthority | None = None
    try:
        snapshot_authority = _begin_verifier_snapshot_authority(run, contract)
        completed_raw = reader.read(
            "completion",
            _exact_path(completion, run.evidence_dir / "completion.json", "completion"),
            expected=None,
            retain_bytes=True,
        )
        completed = _canonical_mapping(completed_raw, "Foldseek completion")
        completion_payload = _validate_completion_header(completed, run, contract)
        candidate_identity = _declared_identity(
            completion_payload.get("candidate_verification"),
            "candidate verification",
        )
        _require_identity_path(
            candidate_identity,
            run.evidence_dir / "independent_verification.candidate.json",
            "candidate verification",
        )
        candidate_raw = reader.read(
            "candidate_verification",
            Path(candidate_identity.path),
            expected=candidate_identity,
            retain_bytes=True,
        )
        candidate = _canonical_mapping(candidate_raw, "candidate verification")
        _validate_candidate_header(candidate, run, contract)
        producer_identity = _declared_identity(
            candidate.get("producer_candidate"), "producer candidate"
        )
        chain = _open_producer_chain(
            reader,
            run,
            contract,
            producer_candidate=Path(producer_identity.path),
            expected_producer=producer_identity,
        )
        _initialize_verifier_snapshot_authority(
            snapshot_authority, chain, run, contract
        )
        retained = reader.observed()
        _validate_producer_chain(
            chain,
            retained,
            run,
            contract,
            min_tm_score=contract.foldseek.min_tm_score,
            min_shorter_coverage=contract.foldseek.min_shorter_coverage,
        )
        _validate_candidate_links(candidate, chain)
        _validate_completion_links(
            completed,
            completion_payload,
            candidate_identity,
            chain,
            run,
            contract,
        )
        retained = reader.finish()
        _finish_snapshot_authority(snapshot_authority)
        completion_identity = retained["completion"].identity
    finally:
        if snapshot_authority is not None:
            snapshot_authority.close()
        reader.close()
    result = {
        "schema_version": "dive-benchmark-foldseek-independent-final-v1",
        "status": "PASS",
        "run_id": run.run_id,
        "contract_sha256": contract.semantic_sha256,
        "execution_contract_sha256": contract.execution_sha256,
        "source_structure_run": _ACCEPTED_STRUCTURE_RUN,
        "producer_commit": chain.producer_commit,
        "completion": _identity_mapping(completion_identity),
        "candidate_verification": _identity_mapping(candidate_identity),
        "producer_completion_candidate": _identity_mapping(chain.producer_identity),
        "artifacts": _identity_role_mapping(chain.artifacts),
        "counts": candidate["counts"],
        "source_structure_evidence": _identity_role_mapping(chain.sources),
        "downstream_requirement": {
            "require_this_artifact": True,
            "generic_completion_alone_is_insufficient": True,
        },
    }
    destination = run.evidence_dir / "independent_verification.final.json"
    _write_terminal_payload(run, destination, result)
    return _file_identity(destination)

def authenticate_final_verification(
    run: BenchmarkRun, contract: BenchmarkContract
) -> ArtifactIdentity:

    return authenticate_final_chain(run, contract).final_verification

def authenticate_final_chain(
    run: BenchmarkRun, contract: BenchmarkContract
) -> AuthenticatedFoldseekChain:

    _validate_run_context(run, contract)
    _preflight_verifier_snapshot_monitor()
    reader = _RetainedEvidenceSet()
    snapshot_authority: MonitoredSnapshotAuthority | None = None
    try:
        snapshot_authority = _begin_verifier_snapshot_authority(run, contract)
        final_path = run.evidence_dir / "independent_verification.final.json"
        final_raw = reader.read(
            "final_verification",
            final_path,
            expected=None,
            retain_bytes=True,
        )
        final = _canonical_mapping(final_raw, "final independent verification")
        _validate_final_header(final, run, contract)
        completion_identity = _declared_identity(
            final.get("completion"), "final completion"
        )
        candidate_identity = _declared_identity(
            final.get("candidate_verification"), "final candidate verification"
        )
        _require_identity_path(
            completion_identity, run.evidence_dir / "completion.json", "completion"
        )
        _require_identity_path(
            candidate_identity,
            run.evidence_dir / "independent_verification.candidate.json",
            "candidate verification",
        )
        completed_raw = reader.read(
            "completion",
            Path(completion_identity.path),
            expected=completion_identity,
            retain_bytes=True,
        )
        candidate_raw = reader.read(
            "candidate_verification",
            Path(candidate_identity.path),
            expected=candidate_identity,
            retain_bytes=True,
        )
        completed = _canonical_mapping(completed_raw, "Foldseek completion")
        completion_payload = _validate_completion_header(completed, run, contract)
        candidate = _canonical_mapping(candidate_raw, "candidate verification")
        _validate_candidate_header(candidate, run, contract)
        producer_identity = _declared_identity(
            candidate.get("producer_candidate"), "producer candidate"
        )
        chain = _open_producer_chain(
            reader,
            run,
            contract,
            producer_candidate=Path(producer_identity.path),
            expected_producer=producer_identity,
        )
        _initialize_verifier_snapshot_authority(
            snapshot_authority, chain, run, contract
        )
        retained = reader.observed()
        _validate_producer_chain(
            chain,
            retained,
            run,
            contract,
            min_tm_score=contract.foldseek.min_tm_score,
            min_shorter_coverage=contract.foldseek.min_shorter_coverage,
        )
        _validate_candidate_links(candidate, chain)
        _validate_completion_links(
            completed,
            completion_payload,
            candidate_identity,
            chain,
            run,
            contract,
        )
        _validate_final_links(
            final,
            retained["completion"].identity,
            candidate_identity,
            chain,
        )
        retained = reader.finish()
        _finish_snapshot_authority(snapshot_authority)
        return AuthenticatedFoldseekChain(
            final_verification=retained["final_verification"].identity,
            completion=retained["completion"].identity,
            candidate_verification=retained["candidate_verification"].identity,
            producer_completion_candidate=chain.producer_identity,
            artifacts=dict(chain.artifacts),
            sources=dict(chain.sources),
            counts={
                role: int(value)
                for role, value in _mapping(final.get("counts"), "final counts").items()
            },
        )
    finally:
        if snapshot_authority is not None:
            snapshot_authority.close()
        reader.close()

def _open_producer_chain(
    reader: _RetainedEvidenceSet,
    run: BenchmarkRun,
    contract: BenchmarkContract,
    *,
    producer_candidate: Path,
    expected_producer: ArtifactIdentity | None,
) -> _ProducerChain:
    producer_path = _exact_path(
        producer_candidate,
        run.evidence_dir / "producer_completion_candidate.json",
        "producer candidate",
    )
    producer_raw = reader.read(
        "producer_candidate",
        producer_path,
        expected=expected_producer,
        retain_bytes=True,
    )
    producer = _canonical_mapping(producer_raw, "producer completion candidate")
    _validate_producer_header(producer, run, contract)
    artifact_values = _exact_role_mapping(
        producer.get("artifacts"), _TASK4_ARTIFACT_ROLES, "Task 4 artifacts"
    )
    artifacts = {
        role: _declared_identity(value, f"artifact {role}")
        for role, value in artifact_values.items()
    }
    source_values = _exact_role_mapping(
        producer.get("source_structure_evidence"),
        _REQUIRED_STRUCTURE_EVIDENCE,
        "Task 3 source evidence",
    )
    sources = {
        role: _declared_identity(value, f"source evidence {role}")
        for role, value in source_values.items()
    }
    _validate_declared_paths(run, contract, artifacts, sources)
    retained_json_roles = {
        "query_database_manifest",
        "target_database_manifest",
        "structural_audit",
    }
    artifact_raw: dict[str, bytes] = {}
    for role in sorted(artifacts):
        raw = reader.read(
            f"artifact:{role}",
            Path(artifacts[role].path),
            expected=artifacts[role],
            retain_bytes=role in retained_json_roles,
        )
        if raw is not None:
            artifact_raw[role] = raw
    for role in sorted(sources):
        reader.read(
            f"source:{role}",
            Path(sources[role].path),
            expected=sources[role],
            retain_bytes=role == "structure_inventory.jsonl",
        )
    attempt_raw = reader.read(
        "attempt",
        Path(run.attempt.path),
        expected=run.attempt,
        retain_bytes=True,
    )
    attempt = _canonical_mapping(attempt_raw, "benchmark attempt")
    audit = _canonical_mapping(artifact_raw["structural_audit"], "structural audit")
    execution = _mapping(audit.get("execution"), "structural audit execution")
    binary = _declared_identity(execution.get("binary"), "execution binary")
    stdout = _declared_identity(execution.get("stdout"), "execution stdout")
    stderr = _declared_identity(execution.get("stderr"), "execution stderr")
    _require_identity_path(binary, Path(contract.foldseek.identity.path), "binary")
    _require_identity_path(stdout, run.evidence_dir / "foldseek.stdout.log", "stdout")
    _require_identity_path(stderr, run.evidence_dir / "foldseek.stderr.log", "stderr")
    reader.read(
        "execution:binary",
        Path(binary.path),
        expected=binary,
        retain_bytes=False,
    )
    reader.read(
        "execution:stdout",
        Path(stdout.path),
        expected=stdout,
        retain_bytes=False,
    )
    reader.read(
        "execution:stderr",
        Path(stderr.path),
        expected=stderr,
        retain_bytes=False,
    )
    producer_identity = expected_producer or ArtifactIdentity(
        str(producer_path.absolute()),
        hashlib.sha256(producer_raw).hexdigest(),
        len(producer_raw),
    )
    commit = attempt.get("repo_commit")
    if type(commit) is not str:
        raise IndependentFoldseekVerificationError("producer commit is invalid")
    chain = _ProducerChain(
        producer_identity,
        producer,
        audit,
        artifacts,
        sources,
        attempt,
        commit,
        artifact_raw,
    )
    return chain

def _validate_run_context(run: BenchmarkRun, contract: BenchmarkContract) -> None:
    if not isinstance(run, BenchmarkRun) or not isinstance(contract, BenchmarkContract):
        raise IndependentFoldseekVerificationError(
            "final authentication requires a claimed run and contract"
        )
    expected_evidence = contract.evidence_root / "benchmark_ready" / run.run_id
    expected_bulk = contract.bulk_root / "benchmark_ready" / run.run_id
    if (
        run.evidence_dir != expected_evidence
        or run.bulk_dir != expected_bulk
        or Path(run.attempt.path) != expected_evidence / "attempt.json"
        or run.contract_sha256 != contract.semantic_sha256
        or not run.evidence_dir.is_dir()
        or run.evidence_dir.is_symlink()
        or not run.bulk_dir.is_dir()
        or run.bulk_dir.is_symlink()
    ):
        raise IndependentFoldseekVerificationError(
            "claimed run roots or semantic contract are invalid"
        )

def _preflight_verifier_snapshot_monitor() -> None:
    try:
        preflight_monitored_tree_authority()
    except SnapshotAuthorityError as error:
        raise IndependentFoldseekVerificationError(str(error)) from error

def _task4_paths(run: BenchmarkRun) -> Mapping[str, Path]:
    return {
        "query_database_manifest": run.evidence_dir / "query_database_manifest.json",
        "target_database_manifest": run.evidence_dir / "target_database_manifest.json",
        "raw_tsv": run.evidence_dir / "foldseek-output" / "raw_hits.tsv",
        "mapped_parquet": run.evidence_dir / "mapped_hits.parquet",
        "conflict_parquet": run.evidence_dir / "structural_conflicts.parquet",
        "structural_audit": run.evidence_dir / "structural_audit.json",
        "producer_database": run.bulk_dir
        / "foldseek-work"
        / "producer-aggregation.sqlite3",
    }

def _source_paths(contract: BenchmarkContract) -> Mapping[str, Path]:
    structure = contract.evidence_root / "benchmark_ready" / _ACCEPTED_STRUCTURE_RUN
    inputs = contract.evidence_root / "benchmark_ready" / "benchmark-inputs-20260826c"
    return {
        "attempt.json": structure / "attempt.json",
        "completion.json": structure / "completion.json",
        "structure_inventory.jsonl": structure / "structure_inventory.jsonl",
        "structure_inventory.completion.json": structure
        / "structure_inventory.completion.json",
        "independent_verification.v5.json": structure
        / "independent_verification.v5.json",
        "determinism_c_vs_d.json": structure / "determinism_c_vs_d.json",
        "input_attempt.json": inputs / "attempt.json",
        "input_completion.json": inputs / "completion.json",
        "input_inventory.json": inputs / "input_inventory.json",
    }

def _validate_declared_paths(
    run: BenchmarkRun,
    contract: BenchmarkContract,
    artifacts: Mapping[str, ArtifactIdentity],
    sources: Mapping[str, ArtifactIdentity],
) -> None:
    for role, path in _task4_paths(run).items():
        _require_identity_path(artifacts[role], path, f"artifact {role}")
    for role, path in _source_paths(contract).items():
        _require_identity_path(sources[role], path, f"source {role}")

def _exact_path(observed: Path, expected: Path, label: str) -> Path:
    observed = Path(observed)
    if observed != expected or not observed.is_absolute():
        raise IndependentFoldseekVerificationError(
            f"{label} path is outside the claimed run roots"
        )
    return observed

def _require_identity_path(
    identity: ArtifactIdentity, expected: Path, label: str
) -> None:
    if Path(identity.path) != expected or not Path(identity.path).is_absolute():
        raise IndependentFoldseekVerificationError(
            f"{label} path is outside its frozen root"
        )

def _canonical_mapping(raw: bytes | None, label: str) -> Mapping[str, object]:
    if raw is None:
        raise IndependentFoldseekVerificationError(f"{label} bytes were not retained")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IndependentFoldseekVerificationError(f"{label} is malformed") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise IndependentFoldseekVerificationError(f"{label} is not canonical JSON")
    return value

def _exact_role_mapping(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, object]:
    result = _mapping(value, label)
    if set(result) != expected:
        raise IndependentFoldseekVerificationError(
            f"{label} does not have the exact frozen roles"
        )
    return result

def _identity_role_mapping(
    values: Mapping[str, ArtifactIdentity],
) -> dict[str, object]:
    return {
        role: _identity_mapping(identity) for role, identity in sorted(values.items())
    }

def _require_schema(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    if set(value) != expected:
        raise IndependentFoldseekVerificationError(
            f"{label} has an open or incomplete schema"
        )

def _validate_producer_header(
    producer: Mapping[str, object],
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> None:
    _require_schema(producer, _PRODUCER_SCHEMA, "producer candidate")
    if (
        producer.get("schema_version")
        != "dive-benchmark-foldseek-producer-candidate-v1"
        or producer.get("status") != "PASS"
        or producer.get("run_id") != run.run_id
        or producer.get("contract_sha256") != contract.semantic_sha256
        or producer.get("execution_contract_sha256") != contract.execution_sha256
        or producer.get("source_structure_run") != _ACCEPTED_STRUCTURE_RUN
    ):
        raise IndependentFoldseekVerificationError(
            "producer candidate run, status, or contract is invalid"
        )
    _validate_counts(producer.get("counts"), "producer counts")

def _validate_candidate_header(
    candidate: Mapping[str, object],
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> None:
    _require_schema(candidate, _CANDIDATE_SCHEMA, "candidate verifier")
    verification = _exact_role_mapping(
        candidate.get("verification"), _VERIFICATION_SCHEMA, "verification claims"
    )
    if (
        candidate.get("schema_version")
        != "dive-benchmark-foldseek-independent-candidate-v1"
        or candidate.get("status") != "PASS"
        or candidate.get("run_id") != run.run_id
        or candidate.get("contract_sha256") != contract.semantic_sha256
        or candidate.get("execution_contract_sha256") != contract.execution_sha256
        or candidate.get("source_structure_run") != _ACCEPTED_STRUCTURE_RUN
        or any(value is not True for value in verification.values())
    ):
        raise IndependentFoldseekVerificationError(
            "candidate verifier run, status, or contract is invalid"
        )
    _validate_counts(candidate.get("counts"), "candidate counts")
    _exact_role_mapping(
        candidate.get("artifacts"), _TASK4_ARTIFACT_ROLES, "candidate artifacts"
    )
    _exact_role_mapping(
        candidate.get("source_structure_evidence"),
        _REQUIRED_STRUCTURE_EVIDENCE,
        "candidate sources",
    )

def _validate_completion_header(
    completed: Mapping[str, object],
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> Mapping[str, object]:
    _require_schema(completed, _COMPLETION_SCHEMA, "generic completion")
    payload = _mapping(completed.get("payload"), "generic completion payload")
    _require_schema(payload, _COMPLETION_PAYLOAD_SCHEMA, "generic completion payload")
    if (
        completed.get("schema_version") != "dive-benchmark-completion-v1"
        or completed.get("status") != "COMPLETE"
        or completed.get("run_id") != run.run_id
        or completed.get("contract_sha256") != contract.semantic_sha256
        or completed.get("attempt") != _identity_mapping(run.attempt)
        or payload.get("execution_contract_sha256") != contract.execution_sha256
        or payload.get("source_structure_run") != _ACCEPTED_STRUCTURE_RUN
        or payload.get("required_final_verifier")
        != "independent_verification.final.json"
    ):
        raise IndependentFoldseekVerificationError(
            "generic completion run, status, or contract is invalid"
        )
    return payload

def _validate_final_header(
    final: Mapping[str, object], run: BenchmarkRun, contract: BenchmarkContract
) -> None:
    _require_schema(final, _FINAL_SCHEMA, "final verifier")
    requirement = _mapping(
        final.get("downstream_requirement"), "downstream requirement"
    )
    if (
        final.get("schema_version") != "dive-benchmark-foldseek-independent-final-v1"
        or final.get("status") != "PASS"
        or final.get("run_id") != run.run_id
        or final.get("contract_sha256") != contract.semantic_sha256
        or final.get("execution_contract_sha256") != contract.execution_sha256
        or final.get("source_structure_run") != _ACCEPTED_STRUCTURE_RUN
        or requirement
        != {
            "require_this_artifact": True,
            "generic_completion_alone_is_insufficient": True,
        }
    ):
        raise IndependentFoldseekVerificationError(
            "final verifier run, status, schema, or contract is invalid"
        )
    _validate_counts(final.get("counts"), "final counts")
    _exact_role_mapping(
        final.get("artifacts"), _TASK4_ARTIFACT_ROLES, "final artifacts"
    )
    _exact_role_mapping(
        final.get("source_structure_evidence"),
        _REQUIRED_STRUCTURE_EVIDENCE,
        "final sources",
    )

def _validate_counts(value: object, label: str) -> None:
    counts = _exact_role_mapping(value, _COUNT_ROLES, label)
    if any(type(item) is not int or item < 0 for item in counts.values()):
        raise IndependentFoldseekVerificationError(f"{label} are invalid")

def _validate_producer_chain(
    chain: _ProducerChain,
    retained: Mapping[str, _RetainedArtifact],
    run: BenchmarkRun,
    contract: BenchmarkContract,
    *,
    min_tm_score: float,
    min_shorter_coverage: float,
) -> None:
    if retained["producer_candidate"].identity != chain.producer_identity:
        raise IndependentFoldseekVerificationError(
            "producer candidate identity changed during collective authentication"
        )
    _validate_attempt(chain.attempt, chain.producer_commit, run, contract)
    audit = chain.audit
    _require_schema(audit, _AUDIT_SCHEMA, "structural audit")
    if (
        audit.get("schema_version") != "dive-benchmark-structural-audit-v1"
        or audit.get("status") != "CANDIDATE"
        or audit.get("run_id") != run.run_id
        or audit.get("contract_sha256") != contract.semantic_sha256
        or audit.get("execution_contract_sha256") != contract.execution_sha256
        or audit.get("source_structure_run") != _ACCEPTED_STRUCTURE_RUN
        or audit.get("source_structure_evidence")
        != _identity_role_mapping(chain.sources)
        or chain.producer.get("artifacts") != _identity_role_mapping(chain.artifacts)
        or chain.producer.get("source_structure_evidence")
        != _identity_role_mapping(chain.sources)
    ):
        raise IndependentFoldseekVerificationError(
            "structural audit or producer candidate cross-links are invalid"
        )
    direct = {
        "query_database_manifest",
        "target_database_manifest",
        "raw_tsv",
        "mapped_parquet",
        "conflict_parquet",
    }
    if any(
        audit.get(role) != _identity_mapping(chain.artifacts[role]) for role in direct
    ):
        raise IndependentFoldseekVerificationError(
            "structural audit artifact cross-links are invalid"
        )
    _validate_audit_contract(
        audit,
        chain,
        min_tm_score=min_tm_score,
        min_shorter_coverage=min_shorter_coverage,
    )
    _validate_execution(
        _mapping(audit.get("execution"), "execution"),
        chain,
        retained,
        run,
        contract,
    )
    _validate_snapshot_manifests(chain, run, contract)

def _validate_snapshot_manifests(
    chain: _ProducerChain,
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> None:
    snapshot_root = run.bulk_dir / "structure-snapshot"
    _validate_snapshot_directory(snapshot_root)
    query_payload = _canonical_mapping(
        chain.manifest_raw["query_database_manifest"], "query manifest"
    )
    target_payload = _canonical_mapping(
        chain.manifest_raw["target_database_manifest"], "target manifest"
    )
    _load_manifest_payload(
        query_payload,
        "query",
        snapshot_root / "queries",
        chain.sources,
        contract.bulk_root / "benchmark_ready" / _ACCEPTED_STRUCTURE_RUN / "queries",
    )
    _load_manifest_payload(
        target_payload,
        "target",
        snapshot_root / "targets",
        chain.sources,
        contract.bulk_root / "benchmark_ready" / _ACCEPTED_STRUCTURE_RUN / "targets",
    )
    _validate_snapshot_directory(snapshot_root)

def _begin_verifier_snapshot_authority(
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> MonitoredSnapshotAuthority:
    snapshot_root = run.bulk_dir / "structure-snapshot"
    expected_count = (
        PRODUCTION_SNAPSHOT_COUNT
        if contract.evidence_root.resolve() == EMERGENT_EVIDENCE_ROOT.resolve()
        else None
    )
    try:
        return MonitoredSnapshotAuthority.begin(
            directories={
                "query": snapshot_root / "queries",
                "target": snapshot_root / "targets",
            },
            manifests={
                "query": run.evidence_dir / "query_database_manifest.json",
                "target": run.evidence_dir / "target_database_manifest.json",
            },
            expected_count=expected_count,
        )
    except SnapshotAuthorityError as error:
        raise IndependentFoldseekVerificationError(str(error)) from error

def _initialize_verifier_snapshot_authority(
    authority: MonitoredSnapshotAuthority,
    chain: _ProducerChain,
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> None:
    declarations: dict[Path, ArtifactIdentity] = {}
    snapshot_root = run.bulk_dir / "structure-snapshot"
    for scope, role, directory_name in (
        ("query", "query_database_manifest", "queries"),
        ("target", "target_database_manifest", "targets"),
    ):
        retained_raw = authority.manifest_bytes(scope, chain.artifacts[role])
        if retained_raw != chain.manifest_raw[role]:
            raise IndependentFoldseekVerificationError(
                f"retained {scope} manifest bytes disagree with evidence chain"
            )
        payload = _canonical_mapping(retained_raw, f"{scope} manifest")
        entries = payload.get("entries")
        if not isinstance(entries, list) or payload.get("record_count") != len(entries):
            raise IndependentFoldseekVerificationError(
                f"{scope} snapshot monitored-tree universe is invalid"
            )
        expected_directory = snapshot_root / directory_name
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise IndependentFoldseekVerificationError(
                    f"{scope} snapshot monitored-tree entry is invalid"
                )
            identity = _declared_identity(
                entry.get("materialized"), f"{scope} retained snapshot"
            )
            path = Path(identity.path)
            if path.parent != expected_directory or path in declarations:
                raise IndependentFoldseekVerificationError(
                    f"{scope} snapshot monitored-tree path is invalid"
                )
            declarations[path] = identity
    expected_count = len(declarations)
    if (
        contract.evidence_root.resolve() == EMERGENT_EVIDENCE_ROOT.resolve()
        and expected_count != PRODUCTION_SNAPSHOT_COUNT
    ):
        raise IndependentFoldseekVerificationError(
            "production snapshot monitored-tree universe is not exactly "
            f"{PRODUCTION_SNAPSHOT_COUNT}"
        )
    try:
        authority.authenticate(declarations)
    except SnapshotAuthorityError as error:
        raise IndependentFoldseekVerificationError(str(error)) from error

def _finish_snapshot_authority(
    authority: MonitoredSnapshotAuthority,
) -> None:
    try:
        authority.finish()
    except SnapshotAuthorityError as error:
        raise IndependentFoldseekVerificationError(str(error)) from error

def _validate_attempt(
    attempt: Mapping[str, object],
    producer_commit: str,
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> None:
    _require_schema(attempt, _ATTEMPT_SCHEMA, "benchmark attempt")
    expected_argv = [
        "scripts/emergent/audit_benchmark_structures.py",
        "--config",
        "configs/emergent/benchmark.yaml",
        "--run-id",
        run.run_id,
        "--structure-run",
        _ACCEPTED_STRUCTURE_RUN,
    ]
    if (
        attempt.get("schema_version") != "dive-benchmark-attempt-v1"
        or attempt.get("status") != "BUILDING"
        or attempt.get("run_id") != run.run_id
        or attempt.get("contract_sha256") != contract.semantic_sha256
        or attempt.get("evidence_path") != str(run.evidence_dir)
        or attempt.get("bulk_path") != str(run.bulk_dir)
        or attempt.get("argv") != expected_argv
        or attempt.get("argv_shell") != shlex.join(expected_argv)
        or attempt.get("upstream_commit") != contract.upstream_commit
        or attempt.get("runtime_python") != str(RUNTIME_PYTHON.resolve())
        or attempt.get("host") != socket.gethostname()
        or attempt.get("repo_commit") != producer_commit
        or len(producer_commit) != 40
        or any(character not in "0123456789abcdef" for character in producer_commit)
    ):
        raise IndependentFoldseekVerificationError(
            "benchmark attempt or producer commit is invalid"
        )

def _validate_audit_contract(
    audit: Mapping[str, object],
    chain: _ProducerChain,
    *,
    min_tm_score: float,
    min_shorter_coverage: float,
) -> None:
    thresholds = _mapping(audit.get("thresholds"), "thresholds")
    aggregation = _mapping(audit.get("aggregation"), "aggregation")
    view = _mapping(audit.get("view_effect"), "view effect")
    units = _mapping(audit.get("units"), "count units")
    requirement = _mapping(
        audit.get("required_independent_verification"), "verifier requirement"
    )
    if (
        thresholds
        != {
            "tm_score": {"operator": ">=", "value": min_tm_score},
            "shorter_chain_coverage": {
                "operator": ">=",
                "value": min_shorter_coverage,
                "clamped_to_unit_interval": True,
            },
        }
        or aggregation
        != {
            "backend": "sqlite",
            "producer_database": _identity_mapping(
                chain.artifacts["producer_database"]
            ),
            "raw_pair_uniqueness": "PRIMARY KEY(query_key, target_key)",
            "parent_pair_uniqueness": (
                "PRIMARY KEY(query_family, query_parent_id, "
                "target_family, target_parent_id)"
            ),
            "conflicts_streamed_to_parquet": True,
        }
        or view != {"strict_view_only": True, "partition_movements": 0}
        or units
        != {
            "raw_hits": "chain-pair alignments",
            "parent_pair_conflicts": "unique query-parent/target-parent pairs",
            "affected_parent_groups": "distinct parent groups",
            "affected_example_rows": "distinct frozen example rows",
        }
        or requirement
        != {
            "candidate_artifact": "independent_verification.candidate.json",
            "post_completion_artifact": "independent_verification.final.json",
            "generic_completion_alone_is_insufficient": True,
        }
    ):
        raise IndependentFoldseekVerificationError(
            "structural audit frozen contract is invalid"
        )

def _validate_execution(
    execution: Mapping[str, object],
    chain: _ProducerChain,
    retained: Mapping[str, _RetainedArtifact],
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> None:
    _require_schema(execution, _EXECUTION_SCHEMA, "execution")
    expected_command = [
        contract.foldseek.identity.path,
        "easy-search",
        str(run.bulk_dir / "structure-snapshot" / "queries"),
        str(run.bulk_dir / "structure-snapshot" / "targets"),
        str(run.evidence_dir / "foldseek-output" / "raw_hits.tsv"),
        str(run.bulk_dir / "foldseek-work"),
        "--format-output",
        "query,target,alnlen,qlen,tlen,alntmscore,evalue",
        "--threads",
        str(contract.foldseek.threads),
        "-v",
        "1",
    ]
    expected_environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": str(contract.foldseek.threads),
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    duration = execution.get("duration_seconds")
    version = execution.get("version_output")
    post_version = execution.get("version_output_post_execution")
    if (
        execution.get("command") != expected_command
        or execution.get("environment") != expected_environment
        or execution.get("binary") != _identity_mapping(contract.foldseek.identity)
        or execution.get("binary_post_execution")
        != _identity_mapping(contract.foldseek.identity)
        or execution.get("exit_code") != 0
        or execution.get("stage") != "subprocess"
        or execution.get("repo_commit") != chain.producer_commit
        or type(duration) not in (int, float)
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or float(duration) < 0.0
        or type(version) is not str
        or not version.startswith(contract.foldseek.version)
        or type(post_version) is not str
        or not post_version.startswith(contract.foldseek.version)
        or post_version != version
        or execution.get("raw_tsv") != _identity_mapping(chain.artifacts["raw_tsv"])
        or execution.get("stdout")
        != _identity_mapping(retained["execution:stdout"].identity)
        or execution.get("stderr")
        != _identity_mapping(retained["execution:stderr"].identity)
        or retained["execution:binary"].identity != contract.foldseek.identity
    ):
        raise IndependentFoldseekVerificationError(
            "Foldseek execution record is invalid"
        )
    start = _utc_timestamp(execution.get("start_utc"), "execution start")
    end = _utc_timestamp(execution.get("end_utc"), "execution end")
    if end < start:
        raise IndependentFoldseekVerificationError(
            "Foldseek execution timestamps are reversed"
        )

def _utc_timestamp(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise IndependentFoldseekVerificationError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise IndependentFoldseekVerificationError(f"{label} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise IndependentFoldseekVerificationError(f"{label} is not UTC")
    return parsed

def _validate_candidate_links(
    candidate: Mapping[str, object], chain: _ProducerChain
) -> None:
    if (
        candidate.get("producer_commit") != chain.producer_commit
        or candidate.get("producer_candidate")
        != _identity_mapping(chain.producer_identity)
        or candidate.get("artifacts") != _identity_role_mapping(chain.artifacts)
        or candidate.get("source_structure_evidence")
        != _identity_role_mapping(chain.sources)
        or candidate.get("counts") != chain.producer.get("counts")
    ):
        raise IndependentFoldseekVerificationError(
            "candidate verifier hashes or cross-links are invalid"
        )

def _validate_completion_links(
    completed: Mapping[str, object],
    payload: Mapping[str, object],
    candidate_identity: ArtifactIdentity,
    chain: _ProducerChain,
    run: BenchmarkRun,
    contract: BenchmarkContract,
) -> None:
    del completed, run, contract
    if (
        payload.get("candidate_verification") != _identity_mapping(candidate_identity)
        or payload.get("producer_completion_candidate")
        != _identity_mapping(chain.producer_identity)
        or any(
            payload.get(role) != _identity_mapping(chain.artifacts[role])
            for role in _TASK4_ARTIFACT_ROLES
        )
    ):
        raise IndependentFoldseekVerificationError(
            "generic completion hashes or cross-links are invalid"
        )

def _validate_final_links(
    final: Mapping[str, object],
    completion_identity: ArtifactIdentity,
    candidate_identity: ArtifactIdentity,
    chain: _ProducerChain,
) -> None:
    if (
        final.get("completion") != _identity_mapping(completion_identity)
        or final.get("candidate_verification") != _identity_mapping(candidate_identity)
        or final.get("producer_completion_candidate")
        != _identity_mapping(chain.producer_identity)
        or final.get("producer_commit") != chain.producer_commit
        or final.get("artifacts") != _identity_role_mapping(chain.artifacts)
        or final.get("source_structure_evidence")
        != _identity_role_mapping(chain.sources)
        or final.get("counts") != chain.producer.get("counts")
    ):
        raise IndependentFoldseekVerificationError(
            "final verifier hashes or cross-links are invalid"
        )

def _count_mapping(counts: IndependentCounts) -> dict[str, int]:
    return {
        "raw_chain_hits": counts.raw_hit_count,
        "conflicting_chain_hits": counts.conflicting_chain_hit_count,
        "parent_pair_conflicts": counts.parent_pair_conflict_count,
        "affected_query_parent_groups": counts.affected_query_parent_groups,
        "affected_target_parent_groups": counts.affected_target_parent_groups,
        "affected_query_example_rows": counts.affected_query_example_rows,
        "affected_target_example_rows": counts.affected_target_example_rows,
    }

def _read_canonical_json(path: Path, label: str) -> tuple[bytes, Mapping[str, object]]:
    try:
        raw = _stable_file_bytes(Path(path))
        payload = json.loads(raw)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        IndependentFoldseekVerificationError,
    ) as error:
        raise IndependentFoldseekVerificationError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise IndependentFoldseekVerificationError(f"{label} is not canonical JSON")
    return raw, payload

def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise IndependentFoldseekVerificationError(f"{label} must be a mapping")
    return value

def _authenticate_declared(value: object, label: str) -> ArtifactIdentity:
    identity = _declared_identity(value, label)
    if _file_identity(Path(identity.path)) != identity:
        raise IndependentFoldseekVerificationError(f"{label} identity drifted")
    return identity

def _declared_identity(value: object, label: str) -> ArtifactIdentity:
    item = _mapping(value, label)
    path = item.get("path")
    digest = item.get("sha256")
    size = item.get("size_bytes")
    if (
        set(item) != {"path", "sha256", "size_bytes"}
        or type(path) is not str
        or not path
        or type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(size) is not int
        or size < 0
    ):
        raise IndependentFoldseekVerificationError(f"{label} identity is malformed")
    return ArtifactIdentity(path, digest, size)

def _file_identity(path: Path) -> ArtifactIdentity:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IndependentFoldseekVerificationError(
            f"cannot independently hash {path}"
        ) from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OSError("not a unique regular file")
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
        pathname = os.stat(path, follow_symlinks=False)
        if not (
            _stat_key(before) == _stat_key(after) == _stat_key(pathname)
            and stat.S_ISREG(pathname.st_mode)
            and pathname.st_nlink == 1
        ):
            raise OSError("file changed while independently hashing")
    except OSError as error:
        raise IndependentFoldseekVerificationError(
            f"cannot independently hash {path}"
        ) from error
    finally:
        os.close(descriptor)
    return ArtifactIdentity(str(path.absolute()), digest.hexdigest(), after.st_size)

def _stable_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IndependentFoldseekVerificationError(
            f"cannot independently read {path}"
        ) from error
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise OSError("not a unique regular file")
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        pathname = os.stat(path, follow_symlinks=False)
        if not (
            _stat_key(before) == _stat_key(after) == _stat_key(pathname)
            and stat.S_ISREG(pathname.st_mode)
            and pathname.st_nlink == 1
        ):
            raise OSError("file changed while independently reading")
    except OSError as error:
        raise IndependentFoldseekVerificationError(
            f"cannot independently read {path}"
        ) from error
    finally:
        os.close(descriptor)
    return b"".join(chunks)

def _stat_key(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

def _write_terminal_payload(
    run: BenchmarkRun, destination: Path, payload: Mapping[str, object]
) -> None:
    from dive.benchmark.evidence import _write_terminal_json

    _write_terminal_json(
        destination,
        payload,
        run.evidence_dir.parents[1],
        run.bulk_dir.parents[1],
    )
