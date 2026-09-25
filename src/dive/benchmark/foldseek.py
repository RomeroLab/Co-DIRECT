
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import subprocess
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import (
    ArtifactIdentity,
    BenchmarkContract,
    file_identity,
)
from dive.benchmark.evidence import BenchmarkRun, complete_benchmark_run
from dive.benchmark.snapshot_authority import (
    PRODUCTION_SNAPSHOT_COUNT,
    MonitoredSnapshotAuthority,
    SnapshotAuthorityError,
    preflight_monitored_tree_authority,
)
from dive.benchmark.structures import (
    StructureInventory,
    StructureInventoryError,
    StructureRecord,
    decode_structure_key,
)
from dive.signed_value.roots import EMERGENT_EVIDENCE_ROOT
from dive.training.preflight import canonical_json_bytes

class FoldseekAuditError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class FoldseekHit:

    query_key: str
    target_key: str
    tm_score: float
    alignment_length: int
    query_length: int
    target_length: int
    evalue: float

    def __post_init__(self) -> None:
        if not self.query_key or not self.target_key:
            raise FoldseekAuditError("Foldseek hit keys must be non-empty")
        if (
            type(self.alignment_length) is not int
            or type(self.query_length) is not int
            or type(self.target_length) is not int
            or self.alignment_length < 0
            or self.query_length <= 0
            or self.target_length <= 0
        ):
            raise FoldseekAuditError("Foldseek hit lengths are invalid")
        if (
            type(self.tm_score) not in (int, float)
            or type(self.evalue) not in (int, float)
            or not math.isfinite(float(self.tm_score))
            or not math.isfinite(float(self.evalue))
            or not 0.0 <= float(self.tm_score) <= 1.0
            or float(self.evalue) < 0.0
        ):
            raise FoldseekAuditError("Foldseek scores must be finite and valid")

    @property
    def shorter_coverage(self) -> float:
        denominator = min(self.query_length, self.target_length)
        value = self.alignment_length / denominator
        if not math.isfinite(value):
            raise FoldseekAuditError("Foldseek shorter-chain coverage is not finite")
        return min(1.0, max(0.0, value))

@dataclass(frozen=True, slots=True)
class StructuralConflict:

    query_family: str
    query_parent_id: str
    target_family: str
    target_parent_id: str
    chain_hit_count: int

@dataclass(frozen=True, slots=True)
class FoldseekVerification:

    hits: tuple[FoldseekHit, ...]
    conflicts: tuple[StructuralConflict, ...]
    raw_hit_count: int
    conflicting_chain_hit_count: int
    parent_pair_conflict_count: int
    affected_query_parent_groups: int
    affected_target_parent_groups: int
    affected_query_example_rows: int
    affected_target_example_rows: int
    strict_view_only: bool = True
    partition_movements: int = 0

@dataclass(frozen=True, slots=True)
class FoldseekDatabaseManifest:

    scope: str
    identity: ArtifactIdentity
    record_count: int
    parent_group_count: int

@dataclass(frozen=True, slots=True)
class FoldseekDatabaseManifests:
    query: FoldseekDatabaseManifest
    target: FoldseekDatabaseManifest

@dataclass(frozen=True, slots=True)
class FoldseekBinary:
    identity: ArtifactIdentity
    version_output: str

@dataclass(frozen=True, slots=True)
class FoldseekAuditResult:
    raw_tsv: ArtifactIdentity
    mapped_parquet: ArtifactIdentity
    conflict_parquet: ArtifactIdentity
    producer_database: ArtifactIdentity
    structural_audit: ArtifactIdentity
    producer_completion_candidate: ArtifactIdentity
    candidate_verification: ArtifactIdentity
    completion: ArtifactIdentity
    final_verification: ArtifactIdentity
    verification: FoldseekVerification

@dataclass(frozen=True, slots=True)
class _ProducerArtifacts:
    mapped_parquet: ArtifactIdentity
    conflict_parquet: ArtifactIdentity
    database: ArtifactIdentity
    verification: FoldseekVerification

@dataclass(frozen=True, slots=True)
class _RetainedSnapshot:
    raw: bytes
    identity: ArtifactIdentity

@dataclass(frozen=True, slots=True)
class AuthenticatedStructureRun:

    run_id: str
    inventory: StructureInventory
    evidence: Mapping[str, ArtifactIdentity]
    example_ids_by_parent: Mapping[str, tuple[str, ...]]

@dataclass(frozen=True, slots=True)
class Task4StructureSnapshot:
    inventory: StructureInventory

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_ACCEPTED_STRUCTURE_RUN = "benchmark-structures-20260826d"
_ACCEPTED_STRUCTURE_COMMIT = "863b18d8b563fd60f78cb68705e19b6b938f7a79"
_ACCEPTED_STRUCTURE_IDENTITIES = MappingProxyType(
    {
        "attempt.json": (
            "3c7fef287e67505b6c4930f85a2a862d797879fa64c29550893afaa353054426",
            1_200,
        ),
        "completion.json": (
            "b08f67311b398d7aba778803086dde23ea0236271e0d0bccaf96cac1e356504c",
            1_407,
        ),
        "structure_inventory.jsonl": (
            "2221be69bb27dd6f6dd1e2704db2919c4fdf0716ba50385fa4197bd5e653e666",
            47_542_640,
        ),
        "structure_inventory.completion.json": (
            "19e599548629f4e5b59ab93eb669fc9c3ec154a5ff91ea6019608c0044bbf827",
            676,
        ),
        "independent_verification.v5.json": (
            "3ce9c1e0f956e699131ef9acf65522c9bcb1b039947f009c548c28d42bfe47aa",
            8_632,
        ),
        "determinism_c_vs_d.json": (
            "f56fc4cb9782954069a4173928f17d77ed7d2d21b35ffe1d07aae6fc3eb3daed",
            2_219,
        ),
    }
)
_ACCEPTED_INPUT_RUN = "benchmark-inputs-20260826c"
_ACCEPTED_INPUT_IDENTITIES = MappingProxyType(
    {
        "input_attempt.json": (
            "762b2e85071a3bc59ffd429fa7cd3283fe0f264d44c083bf0f8e5a89d70863bf",
            922,
        ),
        "input_completion.json": (
            "c23af46888908fbf81c60452c4b85016a27a6fa5f2563ad6b6926b0cf3ef34f2",
            799,
        ),
        "input_inventory.json": (
            "c612476557e3f26773e52020a24b0a103e997e46f520116f6c0c2667994d1903",
            11_898_559,
        ),
    }
)
_EXPECTED_STRUCTURE_COUNTS = MappingProxyType(
    {
        "query_records": 10_817,
        "target_records": 27_583,
        "query_parent_groups": 4_939,
        "target_parent_groups": 11_245,
    }
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

def structural_conflict(
    hit: FoldseekHit, *, min_tm: float, min_shorter_coverage: float
) -> bool:

    if not isinstance(hit, FoldseekHit):
        raise FoldseekAuditError("structural conflict requires a FoldseekHit")
    if (
        type(min_tm) not in (int, float)
        or type(min_shorter_coverage) not in (int, float)
        or not math.isfinite(float(min_tm))
        or not math.isfinite(float(min_shorter_coverage))
        or not 0.0 <= float(min_tm) <= 1.0
        or not 0.0 <= float(min_shorter_coverage) <= 1.0
    ):
        raise FoldseekAuditError("structural thresholds must be finite fractions")
    return hit.tm_score >= float(min_tm) and hit.shorter_coverage >= float(
        min_shorter_coverage
    )

def parse_foldseek_tsv(path: Path) -> tuple[FoldseekHit, ...]:

    return tuple(_iter_foldseek_tsv(path))

def _iter_foldseek_tsv(path: Path) -> Iterator[FoldseekHit]:

    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.removesuffix("\n").removesuffix("\r")
                fields = line.split("\t")
                if len(fields) != 7:
                    raise FoldseekAuditError(
                        f"malformed Foldseek TSV line {line_number}: expected seven columns"
                    )
                (
                    query,
                    target,
                    alignment,
                    query_length,
                    target_length,
                    tm_score,
                    evalue,
                ) = fields
                try:
                    hit = FoldseekHit(
                        query,
                        target,
                        float(tm_score),
                        int(alignment),
                        int(query_length),
                        int(target_length),
                        float(evalue),
                    )
                except (ValueError, FoldseekAuditError) as error:
                    raise FoldseekAuditError(
                        f"malformed Foldseek TSV line {line_number}: {error}"
                    ) from error
                yield hit
    except (OSError, UnicodeError) as error:
        raise FoldseekAuditError(f"cannot read raw Foldseek TSV: {error}") from error

def verify_foldseek_output(
    inventory: StructureInventory,
    raw_tsv: Path,
    *,
    example_ids_by_parent: Mapping[str, tuple[str, ...]] | None = None,
    min_tm: float = 0.50,
    min_shorter_coverage: float = 0.80,
    _collect_hits: bool = False,
    _work_database: Path | None = None,
) -> FoldseekVerification:

    if not isinstance(inventory, StructureInventory):
        raise FoldseekAuditError("verification requires a StructureInventory")
    query = _unique_records(inventory.queries, "query")
    target = _unique_records(inventory.targets, "target")
    if set(query) & set(target):
        raise FoldseekAuditError("a Foldseek key occurs in both database manifests")
    if set(inventory.expected_query_parents) != {
        item.parent_id for item in query.values()
    }:
        raise FoldseekAuditError("query database manifest is missing a required parent")
    if set(inventory.expected_target_parents) != {
        item.parent_id for item in target.values()
    }:
        raise FoldseekAuditError(
            "target database manifest is missing a required parent"
        )

    required_parents = set(inventory.expected_query_parents) | set(
        inventory.expected_target_parents
    )
    if example_ids_by_parent is None:
        rows: Mapping[str, tuple[str, ...]] = MappingProxyType(
            {
                record.parent_id: (record.example_id,)
                for record in (*inventory.queries, *inventory.targets)
            }
        )
    else:
        rows = example_ids_by_parent
        if set(rows) != required_parents or any(
            not isinstance(values, tuple)
            or not values
            or len(values) != len(set(values))
            or not all(type(item) is str and item for item in values)
            for values in rows.values()
        ):
            raise FoldseekAuditError(
                "example-row mapping must cover every required parent exactly"
            )
    database = _work_database or raw_tsv.with_name(
        f"{raw_tsv.name}.producer-verification.sqlite3"
    )
    if database.exists() or database.is_symlink():
        raise FoldseekAuditError("producer verification database already exists")
    connection = sqlite3.connect(database)
    _initialize_producer_database(connection, rows)
    hits: list[FoldseekHit] = []
    raw_hit_count = 0
    conflicting_hits = 0
    try:
        for hit in _iter_foldseek_tsv(raw_tsv):
            raw_hit_count += 1
            if _collect_hits:
                hits.append(hit)
            query_record = query.get(hit.query_key)
            if query_record is None:
                raise FoldseekAuditError(f"unknown query key {hit.query_key!r}")
            target_record = target.get(hit.target_key)
            if target_record is None:
                raise FoldseekAuditError(f"unknown target key {hit.target_key!r}")
            try:
                connection.execute(
                    "INSERT INTO raw_pairs(query_key, target_key) VALUES (?, ?)",
                    (hit.query_key, hit.target_key),
                )
            except sqlite3.IntegrityError as error:
                raise FoldseekAuditError(
                    "duplicate Foldseek query/target pair"
                ) from error
            if structural_conflict(
                hit,
                min_tm=min_tm,
                min_shorter_coverage=min_shorter_coverage,
            ):
                conflicting_hits += 1
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
                        query_record.family,
                        query_record.parent_id,
                        target_record.family,
                        target_record.parent_id,
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO affected_query_parents(parent_id) VALUES (?)",
                    (query_record.parent_id,),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO affected_target_parents(parent_id) VALUES (?)",
                    (target_record.parent_id,),
                )
        connection.commit()
        counted = _producer_counts(connection, raw_hit_count, conflicting_hits)
        return replace(counted, hits=tuple(hits))
    finally:
        connection.close()

def write_database_manifests(
    run: object,
    inventory: StructureInventory,
    *,
    source_structure_run: str,
    source_inventory: StructureInventory | None = None,
    example_ids_by_parent: Mapping[str, tuple[str, ...]] | None = None,
) -> FoldseekDatabaseManifests:

    evidence_dir = Path(getattr(run, "evidence_dir", ""))
    bulk_dir = Path(getattr(run, "bulk_dir", ""))
    if not evidence_dir.is_dir() or not bulk_dir.is_dir():
        raise FoldseekAuditError("database manifests require a claimed benchmark run")
    if type(source_structure_run) is not str or not source_structure_run:
        raise FoldseekAuditError("source structure run id must be non-empty")
    if not isinstance(inventory, StructureInventory):
        raise FoldseekAuditError("database manifests require a StructureInventory")

    query_payload = _database_manifest_payload(
        inventory,
        scope="query",
        records=inventory.queries,
        directory=inventory.query_dir,
        expected_parents=inventory.expected_query_parents,
        source_structure_run=source_structure_run,
        source_records=(source_inventory or inventory).queries,
        example_ids_by_parent=example_ids_by_parent,
    )
    target_payload = _database_manifest_payload(
        inventory,
        scope="target",
        records=inventory.targets,
        directory=inventory.target_dir,
        expected_parents=inventory.expected_target_parents,
        source_structure_run=source_structure_run,
        source_records=(source_inventory or inventory).targets,
        example_ids_by_parent=example_ids_by_parent,
    )
    query_path = evidence_dir / "query_database_manifest.json"
    target_path = evidence_dir / "target_database_manifest.json"
    from dive.benchmark.evidence import _write_terminal_json

    evidence_root = evidence_dir.parents[1]
    bulk_root = bulk_dir.parents[1]
    try:
        _write_terminal_json(query_path, query_payload, evidence_root, bulk_root)
        _write_terminal_json(target_path, target_payload, evidence_root, bulk_root)
    except Exception as error:
        raise FoldseekAuditError(
            f"cannot create Foldseek database manifests: {error}"
        ) from error
    return FoldseekDatabaseManifests(
        FoldseekDatabaseManifest(
            "query",
            file_identity(query_path),
            len(inventory.queries),
            len(inventory.expected_query_parents),
        ),
        FoldseekDatabaseManifest(
            "target",
            file_identity(target_path),
            len(inventory.targets),
            len(inventory.expected_target_parents),
        ),
    )

def authenticate_foldseek_binary(contract: BenchmarkContract) -> FoldseekBinary:

    if not isinstance(contract, BenchmarkContract):
        raise FoldseekAuditError("Foldseek authentication requires BenchmarkContract")
    expected = contract.foldseek.identity
    observed = file_identity(Path(expected.path))
    if observed != expected:
        raise FoldseekAuditError(
            f"Foldseek binary identity drifted: expected {expected}, observed {observed}"
        )
    try:
        result = subprocess.run(
            (expected.path, "version"),
            check=False,
            capture_output=True,
            env=_foldseek_environment(contract.foldseek.threads),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FoldseekAuditError(
            f"cannot execute pinned Foldseek version probe: {error}"
        ) from error
    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    if result.returncode != 0:
        raise FoldseekAuditError(
            f"Foldseek version probe exited {result.returncode}: {stderr}"
        )
    if not stdout.startswith(contract.foldseek.version):
        raise FoldseekAuditError(
            f"Foldseek version drifted: expected prefix {contract.foldseek.version}, observed {stdout!r}"
        )
    return FoldseekBinary(observed, stdout)

def load_authenticated_structure_run(
    contract: BenchmarkContract,
    structure_run_id: str,
) -> AuthenticatedStructureRun:

    if not isinstance(contract, BenchmarkContract):
        raise FoldseekAuditError("structure authentication requires BenchmarkContract")
    if structure_run_id != _ACCEPTED_STRUCTURE_RUN:
        raise FoldseekAuditError(
            f"Foldseek audit requires accepted structure run {_ACCEPTED_STRUCTURE_RUN}"
        )
    structure_root = contract.evidence_root / "benchmark_ready" / structure_run_id
    input_root = contract.evidence_root / "benchmark_ready" / _ACCEPTED_INPUT_RUN
    paths = {
        "attempt.json": structure_root / "attempt.json",
        "completion.json": structure_root / "completion.json",
        "structure_inventory.jsonl": structure_root / "structure_inventory.jsonl",
        "structure_inventory.completion.json": structure_root
        / "structure_inventory.completion.json",
        "independent_verification.v5.json": structure_root
        / "independent_verification.v5.json",
        "determinism_c_vs_d.json": structure_root / "determinism_c_vs_d.json",
        "input_attempt.json": input_root / "attempt.json",
        "input_completion.json": input_root / "completion.json",
        "input_inventory.json": input_root / "input_inventory.json",
    }
    snapshots = _read_retained_snapshots(paths)
    _require_accepted_snapshot_identities(contract, snapshots)

    attempt = _canonical_json_snapshot(snapshots["attempt.json"], "structure attempt")
    completion = _canonical_json_snapshot(
        snapshots["completion.json"], "structure completion"
    )
    structure_completion = _canonical_json_snapshot(
        snapshots["structure_inventory.completion.json"],
        "structure inventory completion",
    )
    verifier = _canonical_json_snapshot(
        snapshots["independent_verification.v5.json"],
        "final structure verifier",
    )
    determinism = _canonical_json_snapshot(
        snapshots["determinism_c_vs_d.json"], "structure determinism evidence"
    )
    input_attempt = _canonical_json_snapshot(
        snapshots["input_attempt.json"], "input attempt"
    )
    input_completion = _canonical_json_snapshot(
        snapshots["input_completion.json"], "input completion"
    )
    input_inventory = _canonical_json_snapshot(
        snapshots["input_inventory.json"], "input inventory"
    )
    _validate_structure_evidence_chain(
        contract=contract,
        structure_run_id=structure_run_id,
        snapshots=snapshots,
        attempt=attempt,
        completion=completion,
        structure_completion=structure_completion,
        verifier=verifier,
        determinism=determinism,
        input_attempt=input_attempt,
        input_completion=input_completion,
        input_inventory=input_inventory,
    )
    _authenticate_producer_git_evidence(contract, verifier)
    inventory = _structure_inventory_from_snapshot(
        contract,
        structure_run_id,
        snapshots,
        completion,
        structure_completion,
        verifier,
    )
    example_ids_by_parent = _example_ids_by_parent(
        input_inventory,
        inventory,
    )
    _assert_snapshot_paths_current(paths, snapshots)
    evidence = MappingProxyType(
        {label: snapshot.identity for label, snapshot in snapshots.items()}
    )
    return AuthenticatedStructureRun(
        structure_run_id,
        inventory,
        evidence,
        example_ids_by_parent,
    )

def _read_retained_snapshots(
    paths: Mapping[str, Path],
) -> dict[str, _RetainedSnapshot]:

    descriptors: dict[str, int] = {}
    before: dict[str, os.stat_result] = {}
    try:
        for label, path in paths.items():
            try:
                descriptor = os.open(
                    path,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
            except OSError as error:
                raise FoldseekAuditError(
                    f"cannot open retained structure evidence {label}: {error}"
                ) from error
            descriptors[label] = descriptor
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise FoldseekAuditError(
                    f"retained structure evidence {label} is not a regular file"
                )
            before[label] = metadata

        snapshots: dict[str, _RetainedSnapshot] = {}
        for label, path in paths.items():
            raw = _read_open_descriptor(descriptors[label], label=label)
            metadata = os.fstat(descriptors[label])
            try:
                pathname = os.stat(path, follow_symlinks=False)
            except OSError as error:
                raise FoldseekAuditError(
                    f"retained structure evidence {label} changed during authentication"
                ) from error
            if (
                not (
                    _stat_identity(before[label])
                    == _stat_identity(metadata)
                    == _stat_identity(pathname)
                )
                or len(raw) != before[label].st_size
            ):
                raise FoldseekAuditError(
                    f"retained structure evidence {label} changed during authentication"
                )
            snapshots[label] = _RetainedSnapshot(
                raw,
                ArtifactIdentity(
                    str(path.absolute()),
                    hashlib.sha256(raw).hexdigest(),
                    len(raw),
                ),
            )

        final_descriptors = {
            label: os.fstat(descriptor) for label, descriptor in descriptors.items()
        }
        try:
            final_paths = {
                label: os.stat(path, follow_symlinks=False)
                for label, path in paths.items()
            }
        except OSError as error:
            raise FoldseekAuditError(
                "retained structure evidence changed during final authentication"
            ) from error
        for label in paths:
            if not (
                _stat_identity(before[label])
                == _stat_identity(final_descriptors[label])
                == _stat_identity(final_paths[label])
            ):
                raise FoldseekAuditError(
                    f"retained structure evidence {label} changed during authentication"
                )
        return snapshots
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

def _read_open_descriptor(descriptor: int, *, label: str) -> bytes:
    del label
    chunks = []
    while chunk := os.read(descriptor, 1 << 20):
        chunks.append(chunk)
    return b"".join(chunks)

def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )

def _assert_snapshot_paths_current(
    paths: Mapping[str, Path], snapshots: Mapping[str, _RetainedSnapshot]
) -> None:
    for label, path in paths.items():
        observed = file_identity(path)
        if observed != snapshots[label].identity:
            raise FoldseekAuditError(
                f"retained structure evidence {label} changed after authentication"
            )

def _require_accepted_snapshot_identities(
    contract: BenchmarkContract,
    snapshots: Mapping[str, _RetainedSnapshot],
) -> None:
    production = contract.evidence_root.resolve() == EMERGENT_EVIDENCE_ROOT.resolve()
    if not production:
        return
    expected = {**_ACCEPTED_STRUCTURE_IDENTITIES, **_ACCEPTED_INPUT_IDENTITIES}
    for label, (digest, size) in expected.items():
        identity = snapshots[label].identity
        if identity.sha256 != digest or identity.size_bytes != size:
            raise FoldseekAuditError(f"accepted structure evidence drifted: {label}")

def _canonical_json_snapshot(
    snapshot: _RetainedSnapshot, label: str
) -> Mapping[str, object]:
    try:
        value = json.loads(snapshot.raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FoldseekAuditError(f"{label} is not JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != snapshot.raw:
        raise FoldseekAuditError(f"{label} is not canonical JSON")
    return value

def _validate_structure_evidence_chain(
    *,
    contract: BenchmarkContract,
    structure_run_id: str,
    snapshots: Mapping[str, _RetainedSnapshot],
    attempt: Mapping[str, object],
    completion: Mapping[str, object],
    structure_completion: Mapping[str, object],
    verifier: Mapping[str, object],
    determinism: Mapping[str, object],
    input_attempt: Mapping[str, object],
    input_completion: Mapping[str, object],
    input_inventory: Mapping[str, object],
) -> None:

    if (
        attempt.get("schema_version") != "dive-benchmark-attempt-v1"
        or attempt.get("status") != "BUILDING"
        or attempt.get("run_id") != structure_run_id
        or attempt.get("repo_commit") != _ACCEPTED_STRUCTURE_COMMIT
        or attempt.get("contract_sha256") != contract.semantic_sha256
        or attempt.get("upstream_commit") != contract.upstream_commit
    ):
        raise FoldseekAuditError(
            "accepted structure attempt does not match its contract"
        )
    if (
        completion.get("schema_version") != "dive-benchmark-completion-v1"
        or completion.get("status") != "COMPLETE"
        or completion.get("run_id") != structure_run_id
        or completion.get("contract_sha256") != contract.semantic_sha256
    ):
        raise FoldseekAuditError("accepted structure completion is invalid")
    _require_identity_link(
        completion.get("attempt"),
        snapshots["attempt.json"].identity,
        "structure attempt",
    )
    completion_payload = _required_mapping(
        completion.get("payload"), "structure completion payload"
    )
    _require_identity_link(
        completion_payload.get("structure_inventory"),
        snapshots["structure_inventory.jsonl"].identity,
        "structure inventory",
    )
    _require_identity_link(
        completion_payload.get("structure_inventory_completion"),
        snapshots["structure_inventory.completion.json"].identity,
        "structure inventory completion",
    )
    _require_identity_link(
        completion_payload.get("input_run_completion"),
        snapshots["input_completion.json"].identity,
        "input completion",
    )
    if completion_payload.get("input_run_id") != _ACCEPTED_INPUT_RUN:
        raise FoldseekAuditError("structure completion references the wrong input run")
    _require_count_mapping(completion_payload, "structure completion")

    if (
        structure_completion.get("schema_version")
        != "dive-benchmark-structure-inventory-completion-v1"
        or structure_completion.get("status") != "COMPLETE"
        or structure_completion.get("missing_parent_groups") != 0
        or structure_completion.get("extra_parent_groups") != 0
        or structure_completion.get("ambiguous_keys") != 0
    ):
        raise FoldseekAuditError("structure inventory completion is not complete")
    _require_identity_link(
        structure_completion.get("inventory"),
        snapshots["structure_inventory.jsonl"].identity,
        "structure inventory completion inventory",
    )
    _require_count_mapping(structure_completion, "structure inventory completion")

    expected_verification = {
        "all_inventory_rows_canonical",
        "all_materialized_sha256_sizes_and_source_derived_bytes_match",
        "all_roles_components_and_keys_reconstructed_one_to_one",
        "all_source_sha256_and_sizes_recomputed",
        "completion_chain_matches",
        "controlling_code_pin_and_provenance_authenticated",
        "exact_query_and_target_parent_sets_match",
        "final_collective_path_descriptor_state_matches",
        "input_run_and_transitive_frozen_inputs_authenticated",
        "no_missing_extra_duplicates_or_ambiguity",
        "producer_commit_environment_contract_authenticated",
        "secure_ownership_and_modes_match",
    }
    verification = _required_mapping(verifier.get("verification"), "verification")
    if (
        verifier.get("status") != "PASS"
        or verifier.get("verification_attempt") != "v5"
        or verifier.get("run_id") != structure_run_id
        or verifier.get("producer_commit") != _ACCEPTED_STRUCTURE_COMMIT
        or verifier.get("contract_sha256") != contract.semantic_sha256
        or verifier.get("upstream_commit") != contract.upstream_commit
        or not expected_verification.issubset(verification)
        or not all(verification.get(name) is True for name in expected_verification)
    ):
        raise FoldseekAuditError("final v5 structure verifier is not an accepted PASS")
    _require_identity_link(
        verifier.get("attempt"), snapshots["attempt.json"].identity, "verifier attempt"
    )
    _require_identity_link(
        verifier.get("completion"),
        snapshots["completion.json"].identity,
        "verifier completion",
    )
    _require_identity_link(
        verifier.get("structure_inventory"),
        snapshots["structure_inventory.jsonl"].identity,
        "verifier inventory",
    )
    _require_identity_link(
        verifier.get("structure_inventory_completion"),
        snapshots["structure_inventory.completion.json"].identity,
        "verifier structure completion",
    )
    input_run = _required_mapping(verifier.get("input_run"), "verifier input run")
    for snapshot_label, link_label in (
        ("input_attempt.json", "attempt.json"),
        ("input_completion.json", "completion.json"),
        ("input_inventory.json", "input_inventory.json"),
    ):
        _require_identity_link(
            input_run.get(link_label),
            snapshots[snapshot_label].identity,
            f"verifier {snapshot_label}",
        )
    counts = _required_mapping(verifier.get("counts"), "verifier counts")
    if (
        counts.get("total_records") != 38_400
        or counts.get("unique_foldseek_keys") != 38_400
    ):
        raise FoldseekAuditError("final v5 verifier has the wrong structure universe")
    _require_count_mapping(counts, "final v5 verifier")

    comparison = _required_mapping(
        determinism.get("comparison"), "determinism comparison"
    )
    raw_inventories = _required_mapping(
        determinism.get("raw_inventory_identities"), "determinism inventories"
    )
    if (
        determinism.get("status") != "PASS"
        or determinism.get("run_d") != structure_run_id
        or determinism.get("producer_commit") != _ACCEPTED_STRUCTURE_COMMIT
        or not comparison
        or not all(value is True for value in comparison.values())
    ):
        raise FoldseekAuditError(
            "structure determinism evidence is not an accepted PASS"
        )
    _require_identity_link(
        determinism.get("independent_verification_v5"),
        snapshots["independent_verification.v5.json"].identity,
        "determinism final verifier",
    )
    _require_identity_link(
        raw_inventories.get("d"),
        snapshots["structure_inventory.jsonl"].identity,
        "determinism run d inventory",
    )
    _require_count_mapping(
        _required_mapping(determinism.get("counts"), "determinism counts"),
        "determinism",
    )

    if (
        input_attempt.get("status") != "BUILDING"
        or input_attempt.get("run_id") != _ACCEPTED_INPUT_RUN
        or input_attempt.get("contract_sha256") != contract.semantic_sha256
        or input_attempt.get("upstream_commit") != contract.upstream_commit
        or input_completion.get("status") != "COMPLETE"
        or input_completion.get("run_id") != _ACCEPTED_INPUT_RUN
        or input_completion.get("contract_sha256") != contract.semantic_sha256
    ):
        raise FoldseekAuditError("authenticated input evidence is invalid")
    _require_identity_link(
        input_completion.get("attempt"),
        snapshots["input_attempt.json"].identity,
        "input attempt",
    )
    input_payload = _required_mapping(
        input_completion.get("payload"), "input completion payload"
    )
    _require_identity_link(
        input_payload.get("inventory"),
        snapshots["input_inventory.json"].identity,
        "input inventory",
    )
    projection = _required_mapping(
        input_inventory.get("projection"), "input projection"
    )
    if (
        input_inventory.get("schema_version") != "dive-benchmark-input-inventory-v1"
        or input_payload.get("projection_identity") != "v2"
        or input_payload.get("projection_row_count") != 26_309
        or projection.get("identity") != "v2"
        or projection.get("row_count") != 26_309
    ):
        raise FoldseekAuditError("input inventory does not bind parent projection v2")

def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FoldseekAuditError(f"{label} must be a mapping")
    return value

def _require_count_mapping(value: Mapping[str, object], label: str) -> None:
    for name, expected in _EXPECTED_STRUCTURE_COUNTS.items():
        if value.get(name) != expected:
            raise FoldseekAuditError(f"{label} has the wrong {name}")

def _require_identity_link(
    value: object, expected: ArtifactIdentity, label: str
) -> None:
    if _artifact_identity(value, label) != expected:
        raise FoldseekAuditError(f"{label} identity does not match retained bytes")

def _artifact_identity(value: object, label: str) -> ArtifactIdentity:
    item = _required_mapping(value, label)
    if set(item) != {"path", "sha256", "size_bytes"}:
        raise FoldseekAuditError(f"{label} has an invalid identity schema")
    path = item.get("path")
    digest = item.get("sha256")
    size = item.get("size_bytes")
    if (
        type(path) is not str
        or not path
        or type(digest) is not str
        or len(digest) != 64
        or not set(digest) <= _SHA256_CHARACTERS
        or type(size) is not int
        or size < 0
    ):
        raise FoldseekAuditError(f"{label} has an invalid identity")
    return ArtifactIdentity(path, digest, size)

def _authenticate_producer_git_evidence(
    contract: BenchmarkContract, verifier: Mapping[str, object]
) -> None:
    if not _git_is_ancestor(_REPOSITORY_ROOT, _ACCEPTED_STRUCTURE_COMMIT, "HEAD"):
        raise FoldseekAuditError(
            "structure producer commit is not an ancestor of current HEAD"
        )
    controlling = _required_mapping(
        verifier.get("controlling_sources"), "producer controlling sources"
    )
    _authenticate_git_identities(
        controlling,
        root=_REPOSITORY_ROOT,
        commit=_ACCEPTED_STRUCTURE_COMMIT,
        label="producer",
    )
    upstream = _required_mapping(
        verifier.get("upstream_controlling_sources"), "upstream controlling sources"
    )
    _authenticate_git_identities(
        upstream,
        root=contract.upstream_root,
        commit=contract.upstream_commit,
        label="upstream",
    )

def _authenticate_git_identities(
    values: Mapping[str, object], *, root: Path, commit: str, label: str
) -> None:
    for relative, value in values.items():
        if type(relative) is not str or not relative or Path(relative).is_absolute():
            raise FoldseekAuditError(f"{label} controlling source path is not relative")
        expected = _artifact_identity(value, f"{label} controlling source {relative}")
        if expected.path != relative:
            raise FoldseekAuditError(
                f"{label} controlling source path identity drifted"
            )
        raw = _git_blob(root, commit, relative)
        if (
            hashlib.sha256(raw).hexdigest() != expected.sha256
            or len(raw) != expected.size_bytes
        ):
            raise FoldseekAuditError(f"{label} controlling source drifted: {relative}")

def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            (
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                ancestor,
                descendant,
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise FoldseekAuditError(
            "cannot authenticate structure producer commit"
        ) from error
    return result.returncode == 0

def _git_blob(root: Path, commit: str, relative: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(root), "show", f"{commit}:{relative}"),
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise FoldseekAuditError(
            f"cannot authenticate Git blob {commit}:{relative}"
        ) from error

def _structure_inventory_from_snapshot(
    contract: BenchmarkContract,
    structure_run_id: str,
    snapshots: Mapping[str, _RetainedSnapshot],
    completion: Mapping[str, object],
    structure_completion: Mapping[str, object],
    verifier: Mapping[str, object],
) -> StructureInventory:
    raw = snapshots["structure_inventory.jsonl"].raw
    query_dir = contract.bulk_root / "benchmark_ready" / structure_run_id / "queries"
    target_dir = contract.bulk_root / "benchmark_ready" / structure_run_id / "targets"
    queries: list[StructureRecord] = []
    targets: list[StructureRecord] = []
    keys: set[str] = set()
    expected_fields = {
        "schema_version",
        "scope",
        "foldseek_key",
        "family",
        "partition",
        "source_id",
        "example_id",
        "parent_id",
        "component_id",
        "chain_id",
        "role",
        "source",
        "source_provenance",
        "materialized",
        "materialization_command",
        "code_identity",
        "failure_reason",
    }
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise FoldseekAuditError(
                f"structure inventory line {line_number} is not JSON"
            ) from error
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_fields
            or canonical_json_bytes(value) != line
            or value.get("schema_version") != "dive-benchmark-structure-record-v1"
            or value.get("failure_reason") is not None
        ):
            raise FoldseekAuditError(
                f"structure inventory line {line_number} is not canonical and complete"
            )
        record = _structure_record(value, line_number)
        if record.foldseek_key in keys:
            raise FoldseekAuditError(
                f"duplicate Foldseek key on inventory line {line_number}"
            )
        keys.add(record.foldseek_key)
        scope = value.get("scope")
        expected_dir = (
            query_dir if scope == "query" else target_dir if scope == "target" else None
        )
        if expected_dir is None:
            raise FoldseekAuditError(f"invalid inventory scope on line {line_number}")
        expected_path = expected_dir / f"{record.foldseek_key}.pdb"
        if Path(record.materialized.path) != expected_path:
            raise FoldseekAuditError(
                f"materialized path drift on inventory line {line_number}"
            )
        (queries if scope == "query" else targets).append(record)
    if (
        len(queries) != _EXPECTED_STRUCTURE_COUNTS["query_records"]
        or len(targets) != _EXPECTED_STRUCTURE_COUNTS["target_records"]
    ):
        raise FoldseekAuditError("structure inventory record counts drifted")
    query_parents = tuple(sorted({item.parent_id for item in queries}))
    target_parents = tuple(sorted({item.parent_id for item in targets}))
    if (
        len(query_parents) != _EXPECTED_STRUCTURE_COUNTS["query_parent_groups"]
        or len(target_parents) != _EXPECTED_STRUCTURE_COUNTS["target_parent_groups"]
        or set(query_parents) & set(target_parents)
    ):
        raise FoldseekAuditError("structure inventory parent universe drifted")
    del completion, structure_completion, verifier
    return StructureInventory(
        tuple(queries),
        tuple(targets),
        query_dir,
        target_dir,
        snapshots["structure_inventory.jsonl"].identity,
        snapshots["structure_inventory.completion.json"].identity,
        query_parents,
        target_parents,
    )

def _structure_record(value: Mapping[str, object], line_number: int) -> StructureRecord:
    string_fields = (
        "foldseek_key",
        "family",
        "partition",
        "source_id",
        "example_id",
        "parent_id",
        "component_id",
        "chain_id",
        "role",
    )
    if not all(
        type(value.get(name)) is str and value.get(name) for name in string_fields
    ):
        raise FoldseekAuditError(
            f"empty structure field on inventory line {line_number}"
        )
    key = str(value["foldseek_key"])
    try:
        decoded = decode_structure_key(key)
    except Exception as error:
        raise FoldseekAuditError(
            f"malformed Foldseek key on inventory line {line_number}"
        ) from error
    if decoded != (value["family"], value["parent_id"], value["chain_id"]):
        raise FoldseekAuditError(
            f"malformed Foldseek key on inventory line {line_number}"
        )
    command = value.get("materialization_command")
    provenance = value.get("source_provenance")
    if (
        not isinstance(command, list)
        or not command
        or not all(type(item) is str and item for item in command)
        or not isinstance(provenance, list)
    ):
        raise FoldseekAuditError(
            f"invalid structure provenance on inventory line {line_number}"
        )
    return StructureRecord(
        foldseek_key=key,
        family=str(value["family"]),
        partition=str(value["partition"]),
        source_id=str(value["source_id"]),
        example_id=str(value["example_id"]),
        parent_id=str(value["parent_id"]),
        component_id=str(value["component_id"]),
        chain_id=str(value["chain_id"]),
        role=str(value["role"]),
        source=_artifact_identity(value.get("source"), f"source line {line_number}"),
        source_provenance=tuple(
            _artifact_identity(item, f"source provenance line {line_number}")
            for item in provenance
        ),
        materialized=_artifact_identity(
            value.get("materialized"), f"materialized line {line_number}"
        ),
        materialization_command=tuple(command),
        code_identity=_artifact_identity(
            value.get("code_identity"), f"code identity line {line_number}"
        ),
    )

def _example_ids_by_parent(
    input_inventory: Mapping[str, object], inventory: StructureInventory
) -> Mapping[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {}
    examples = input_inventory.get("examples")
    if not isinstance(examples, list):
        raise FoldseekAuditError("input inventory examples are not a list")
    for index, value in enumerate(examples):
        row = _required_mapping(value, f"input example {index}")
        parent_id = row.get("parent_id")
        example_id = row.get("example_id")
        role_realized = row.get("role_realized")
        if (
            type(parent_id) is not str
            or not parent_id
            or type(example_id) is not str
            or not example_id
            or type(role_realized) is not bool
        ):
            raise FoldseekAuditError(f"input example {index} has invalid row identity")
        if not role_realized:
            continue
        result.setdefault(parent_id, set()).add(example_id)
    for record in (*inventory.queries, *inventory.targets):
        result.setdefault(record.parent_id, set()).add(record.example_id)
    required = set(inventory.expected_query_parents) | set(
        inventory.expected_target_parents
    )
    if not required <= result.keys():
        raise FoldseekAuditError(
            "example-row mapping omits a required structure parent"
        )
    return MappingProxyType(
        {parent: tuple(sorted(result[parent])) for parent in sorted(required)}
    )

def _require_production_structure_inputs(
    contract: BenchmarkContract,
    inventory: StructureInventory,
    *,
    source_structure_run: str,
    structure_evidence: Mapping[str, ArtifactIdentity] | None,
) -> None:

    if contract.evidence_root.resolve() != EMERGENT_EVIDENCE_ROOT.resolve():
        return
    if source_structure_run != _ACCEPTED_STRUCTURE_RUN or structure_evidence is None:
        raise FoldseekAuditError(
            "production Foldseek requires authenticated structure run d"
        )
    expected_values = {**_ACCEPTED_STRUCTURE_IDENTITIES, **_ACCEPTED_INPUT_IDENTITIES}
    if set(structure_evidence) != set(expected_values):
        raise FoldseekAuditError("production structure evidence chain is incomplete")
    structure_root = contract.evidence_root / "benchmark_ready" / source_structure_run
    input_root = contract.evidence_root / "benchmark_ready" / _ACCEPTED_INPUT_RUN
    for label, (digest, size) in expected_values.items():
        root = input_root if label.startswith("input_") else structure_root
        filename = (
            label.removeprefix("input_")
            if label in {"input_attempt.json", "input_completion.json"}
            else label
        )
        expected = ArtifactIdentity(str((root / filename).resolve()), digest, size)
        if structure_evidence[label] != expected:
            raise FoldseekAuditError(f"production structure evidence drifted: {label}")
    if (
        inventory.inventory != structure_evidence["structure_inventory.jsonl"]
        or inventory.completion
        != structure_evidence["structure_inventory.completion.json"]
        or inventory.query_dir
        != contract.bulk_root / "benchmark_ready" / source_structure_run / "queries"
        or inventory.target_dir
        != contract.bulk_root / "benchmark_ready" / source_structure_run / "targets"
    ):
        raise FoldseekAuditError("production structure inventory is not run d")

def snapshot_structure_inventory(
    run: BenchmarkRun, inventory: StructureInventory
) -> Task4StructureSnapshot:

    query_by_key = _unique_records(inventory.queries, "query")
    target_by_key = _unique_records(inventory.targets, "target")
    if set(query_by_key) & set(target_by_key):
        raise FoldseekAuditError("a structure snapshot key spans both scopes")
    if set(inventory.expected_query_parents) != {
        record.parent_id for record in query_by_key.values()
    } or set(inventory.expected_target_parents) != {
        record.parent_id for record in target_by_key.values()
    }:
        raise FoldseekAuditError("structure snapshot omits a required parent")
    root = run.bulk_dir / "structure-snapshot"
    query_dir = root / "queries"
    target_dir = root / "targets"
    _mkdir_create_new(root, 0o2700, "structure snapshot root")
    _mkdir_create_new(query_dir, 0o2700, "query snapshot directory")
    _mkdir_create_new(target_dir, 0o2700, "target snapshot directory")
    queries = tuple(
        _copy_snapshot_record(query_by_key[key], query_dir)
        for key in sorted(query_by_key)
    )
    targets = tuple(
        _copy_snapshot_record(target_by_key[key], target_dir)
        for key in sorted(target_by_key)
    )
    os.chmod(query_dir, 0o2550)
    os.chmod(target_dir, 0o2550)
    os.chmod(root, 0o2550)
    snapshot = StructureInventory(
        queries,
        targets,
        query_dir,
        target_dir,
        inventory.inventory,
        inventory.completion,
        inventory.expected_query_parents,
        inventory.expected_target_parents,
    )
    return Task4StructureSnapshot(snapshot)

def _copy_snapshot_record(
    record: StructureRecord,
    destination_dir: Path,
) -> StructureRecord:
    raw = _stable_artifact_bytes(Path(record.materialized.path), record.materialized)
    destination = destination_dir / f"{record.foldseek_key}.pdb"
    _write_create_new_bytes(destination, raw, 0o440)
    observed = file_identity(destination)
    if (
        observed.sha256 != record.materialized.sha256
        or observed.size_bytes != record.materialized.size_bytes
    ):
        raise FoldseekAuditError(
            f"Task 4 structure snapshot changed bytes for {record.foldseek_key}"
        )
    return replace(record, materialized=observed)

def _stable_artifact_bytes(path: Path, expected: ArtifactIdentity) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FoldseekAuditError(
            f"cannot open authenticated structure {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FoldseekAuditError(f"authenticated structure is not regular: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        pathname = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise FoldseekAuditError(f"authenticated structure changed: {path}") from error
    finally:
        os.close(descriptor)
    if not (
        _stat_identity(before) == _stat_identity(after) == _stat_identity(pathname)
    ):
        raise FoldseekAuditError(f"authenticated structure changed: {path}")
    raw = b"".join(chunks)
    observed = ArtifactIdentity(
        str(path.resolve()), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    if observed != expected:
        raise FoldseekAuditError(f"authenticated structure identity drifted: {path}")
    return raw

def _write_create_new_bytes(path: Path, raw: bytes, mode: int) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
    except OSError as error:
        raise FoldseekAuditError(f"cannot reserve immutable snapshot {path}") from error
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise FoldseekAuditError(f"short write for immutable snapshot {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _mkdir_create_new(path: Path, mode: int, label: str) -> None:
    try:
        os.mkdir(path, mode)
    except OSError as error:
        raise FoldseekAuditError(f"cannot reserve {label}: {path}") from error
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FoldseekAuditError(f"{label} is not a plain directory: {path}")
    os.chmod(path, mode)

def _reserve_foldseek_paths(run: BenchmarkRun) -> tuple[Path, Path, Path]:
    output_dir = run.evidence_dir / "foldseek-output"
    work_dir = run.bulk_dir / "foldseek-work"
    _mkdir_create_new(output_dir, 0o2750, "Foldseek output directory")
    _mkdir_create_new(work_dir, 0o2700, "Foldseek work directory")
    raw_tsv = output_dir / "raw_hits.tsv"
    _require_raw_absent(raw_tsv)
    return output_dir, work_dir, raw_tsv

def _require_raw_absent(raw_tsv: Path) -> None:
    try:
        os.lstat(raw_tsv)
    except FileNotFoundError:
        return
    except OSError as error:
        raise FoldseekAuditError("cannot preflight raw Foldseek output path") from error
    raise FoldseekAuditError("raw Foldseek output path already exists")

def _authenticate_created_raw(raw_tsv: Path) -> ArtifactIdentity:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(raw_tsv, flags)
    except OSError as error:
        raise FoldseekAuditError(
            "Foldseek did not create its reserved raw output"
        ) from error
    digest = hashlib.sha256()
    try:
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise FoldseekAuditError("Foldseek raw output is not a unique regular file")
        os.fchmod(descriptor, 0o440)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
        pathname = os.stat(raw_tsv, follow_symlinks=False)
    except OSError as error:
        raise FoldseekAuditError("Foldseek raw output changed while hashing") from error
    finally:
        os.close(descriptor)
    if not (
        _stat_identity(before) == _stat_identity(after) == _stat_identity(pathname)
        and stat.S_ISREG(pathname.st_mode)
        and pathname.st_nlink == 1
    ):
        raise FoldseekAuditError("Foldseek raw output is not a unique regular file")
    return ArtifactIdentity(str(raw_tsv.resolve()), digest.hexdigest(), after.st_size)

def _verify_structure_snapshot(
    snapshot: Task4StructureSnapshot, source: StructureInventory
) -> None:
    source_by_key = {
        record.foldseek_key: record for record in (*source.queries, *source.targets)
    }
    snapshot_records = (*snapshot.inventory.queries, *snapshot.inventory.targets)
    if set(source_by_key) != {record.foldseek_key for record in snapshot_records}:
        raise FoldseekAuditError("Task 4 structure snapshot key universe drifted")
    for record in snapshot_records:
        observed = file_identity(Path(record.materialized.path))
        original = source_by_key[record.foldseek_key].materialized
        if (
            observed != record.materialized
            or observed.sha256 != original.sha256
            or observed.size_bytes != original.size_bytes
        ):
            raise FoldseekAuditError(
                f"Task 4 structure snapshot drifted: {record.foldseek_key}"
            )

def _reauthenticate_structure_evidence(
    contract: BenchmarkContract,
    source_structure_run: str,
    expected: Mapping[str, ArtifactIdentity] | None,
) -> Mapping[str, ArtifactIdentity]:

    if contract.evidence_root.resolve() != EMERGENT_EVIDENCE_ROOT.resolve():
        if expected is None:
            return MappingProxyType({})
        for label, identity in expected.items():
            if file_identity(Path(identity.path)) != identity:
                raise FoldseekAuditError(
                    f"structure evidence changed during Foldseek: {label}"
                )
        return MappingProxyType(dict(expected))
    if source_structure_run != _ACCEPTED_STRUCTURE_RUN or expected is None:
        raise FoldseekAuditError(
            "production structure evidence cannot be reauthenticated"
        )
    structure_root = contract.evidence_root / "benchmark_ready" / source_structure_run
    input_root = contract.evidence_root / "benchmark_ready" / _ACCEPTED_INPUT_RUN
    paths = {
        "attempt.json": structure_root / "attempt.json",
        "completion.json": structure_root / "completion.json",
        "structure_inventory.jsonl": structure_root / "structure_inventory.jsonl",
        "structure_inventory.completion.json": structure_root
        / "structure_inventory.completion.json",
        "independent_verification.v5.json": structure_root
        / "independent_verification.v5.json",
        "determinism_c_vs_d.json": structure_root / "determinism_c_vs_d.json",
        "input_attempt.json": input_root / "attempt.json",
        "input_completion.json": input_root / "completion.json",
        "input_inventory.json": input_root / "input_inventory.json",
    }
    snapshots = _read_retained_snapshots(paths)
    _require_accepted_snapshot_identities(contract, snapshots)
    refreshed = MappingProxyType(
        {label: snapshot.identity for label, snapshot in snapshots.items()}
    )
    if dict(refreshed) != dict(expected):
        raise FoldseekAuditError("structure evidence mapping changed during Foldseek")
    _assert_snapshot_paths_current(paths, snapshots)
    return refreshed

def _write_producer_completion_candidate(
    run: BenchmarkRun,
    *,
    contract: BenchmarkContract,
    source_structure_run: str,
    structure_evidence: Mapping[str, ArtifactIdentity],
    manifests: FoldseekDatabaseManifests,
    raw_identity: ArtifactIdentity,
    artifacts: _ProducerArtifacts,
    structural_audit: ArtifactIdentity,
) -> ArtifactIdentity:
    from dive.benchmark.evidence import _write_terminal_json

    payload = {
        "schema_version": "dive-benchmark-foldseek-producer-candidate-v1",
        "status": "PASS",
        "run_id": run.run_id,
        "contract_sha256": contract.semantic_sha256,
        "execution_contract_sha256": contract.execution_sha256,
        "source_structure_run": source_structure_run,
        "source_structure_evidence": {
            name: _identity_mapping(identity)
            for name, identity in sorted(structure_evidence.items())
        },
        "artifacts": {
            "query_database_manifest": _identity_mapping(manifests.query.identity),
            "target_database_manifest": _identity_mapping(manifests.target.identity),
            "raw_tsv": _identity_mapping(raw_identity),
            "mapped_parquet": _identity_mapping(artifacts.mapped_parquet),
            "conflict_parquet": _identity_mapping(artifacts.conflict_parquet),
            "producer_database": _identity_mapping(artifacts.database),
            "structural_audit": _identity_mapping(structural_audit),
        },
        "counts": _audit_counts(artifacts.verification),
    }
    path = run.evidence_dir / "producer_completion_candidate.json"
    _write_terminal_json(
        path,
        payload,
        run.evidence_dir.parents[1],
        run.bulk_dir.parents[1],
    )
    return file_identity(path)

def _acquire_producer_snapshot_authority(
    run: BenchmarkRun,
    contract: BenchmarkContract,
    manifests: FoldseekDatabaseManifests,
) -> MonitoredSnapshotAuthority:
    expected_count = manifests.query.record_count + manifests.target.record_count
    if (
        contract.evidence_root.resolve() == EMERGENT_EVIDENCE_ROOT.resolve()
        and expected_count != PRODUCTION_SNAPSHOT_COUNT
    ):
        raise FoldseekAuditError(
            "production snapshot monitored-tree universe must contain exactly "
            f"{PRODUCTION_SNAPSHOT_COUNT} files"
        )
    snapshot_root = run.bulk_dir / "structure-snapshot"
    try:
        authority = MonitoredSnapshotAuthority.begin(
            directories={
                "query": snapshot_root / "queries",
                "target": snapshot_root / "targets",
            },
            manifests={
                "query": Path(manifests.query.identity.path),
                "target": Path(manifests.target.identity.path),
            },
            expected_count=expected_count,
        )
    except SnapshotAuthorityError as error:
        raise FoldseekAuditError(str(error)) from error
    declarations: dict[Path, ArtifactIdentity] = {}
    try:
        for manifest in (manifests.query, manifests.target):
            raw = authority.manifest_bytes(manifest.scope, manifest.identity)
            try:
                payload = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise FoldseekAuditError(
                    f"cannot parse {manifest.scope} snapshot manifest"
                ) from error
            entries = payload.get("entries") if isinstance(payload, Mapping) else None
            if (
                not isinstance(entries, list)
                or payload.get("record_count") != len(entries)
                or len(entries) != manifest.record_count
                or canonical_json_bytes(payload) != raw
            ):
                raise FoldseekAuditError(
                    f"{manifest.scope} snapshot manifest universe is invalid"
                )
            expected_directory = snapshot_root / (
                "queries" if manifest.scope == "query" else "targets"
            )
            for entry in entries:
                materialized = (
                    entry.get("materialized") if isinstance(entry, Mapping) else None
                )
                if not isinstance(materialized, Mapping) or set(materialized) != {
                    "path",
                    "sha256",
                    "size_bytes",
                }:
                    raise FoldseekAuditError("snapshot manifest identity is malformed")
                path_value = materialized.get("path")
                digest_value = materialized.get("sha256")
                size_value = materialized.get("size_bytes")
                if (
                    type(path_value) is not str
                    or not path_value
                    or type(digest_value) is not str
                    or len(digest_value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest_value
                    )
                    or type(size_value) is not int
                    or size_value < 0
                ):
                    raise FoldseekAuditError("snapshot manifest identity is invalid")
                path = Path(path_value)
                identity = ArtifactIdentity(path_value, digest_value, size_value)
                if path.parent != expected_directory or path in declarations:
                    raise FoldseekAuditError("snapshot manifest identity is invalid")
                declarations[path] = identity
        authority.authenticate(declarations)
        return authority
    except Exception:
        authority.close()
        raise

def _audit_counts(verification: FoldseekVerification) -> dict[str, int]:
    return {
        "raw_chain_hits": verification.raw_hit_count,
        "conflicting_chain_hits": verification.conflicting_chain_hit_count,
        "parent_pair_conflicts": verification.parent_pair_conflict_count,
        "affected_query_parent_groups": verification.affected_query_parent_groups,
        "affected_target_parent_groups": verification.affected_target_parent_groups,
        "affected_query_example_rows": verification.affected_query_example_rows,
        "affected_target_example_rows": verification.affected_target_example_rows,
    }

def run_foldseek_audit(
    run: BenchmarkRun,
    inventory: StructureInventory,
    contract: BenchmarkContract,
    *,
    source_structure_run: str,
    example_ids_by_parent: Mapping[str, tuple[str, ...]] | None = None,
    structure_evidence: Mapping[str, ArtifactIdentity] | None = None,
) -> FoldseekAuditResult:

    stage: dict[str, object] = {"stage": "post_claim_entry"}
    retained_resources: list[MonitoredSnapshotAuthority] = []
    try:
        return _run_foldseek_audit_claimed(
            run,
            inventory,
            contract,
            source_structure_run=source_structure_run,
            example_ids_by_parent=example_ids_by_parent,
            structure_evidence=structure_evidence,
            _stage=stage,
            _retained_resources=retained_resources,
        )
    except Exception as error:
        if (
            isinstance(run, BenchmarkRun)
            and not (run.evidence_dir / "foldseek.failure.json").exists()
        ):
            _write_foldseek_failure(
                run,
                "post_claim_stage_failure",
                f"{type(error).__name__}: {error}",
                stage,
            )
        if isinstance(error, FoldseekAuditError):
            raise
        raise FoldseekAuditError(
            f"Foldseek audit failed during {stage.get('stage')}: {error}"
        ) from error
    finally:
        for resource_authority in retained_resources:
            resource_authority.close()

def authenticate_foldseek_handoff(
    run: BenchmarkRun,
    contract: BenchmarkContract,
    result: FoldseekAuditResult,
) -> FoldseekAuditResult:

    expected_final = result.final_verification
    try:
        from dive.benchmark.foldseek_verify import authenticate_final_chain

        authenticated = authenticate_final_chain(run, contract)
        expected_final = authenticated.final_verification
        counts = authenticated.counts
        verified = FoldseekVerification(
            hits=(),
            conflicts=(),
            raw_hit_count=counts["raw_chain_hits"],
            conflicting_chain_hit_count=counts["conflicting_chain_hits"],
            parent_pair_conflict_count=counts["parent_pair_conflicts"],
            affected_query_parent_groups=counts["affected_query_parent_groups"],
            affected_target_parent_groups=counts["affected_target_parent_groups"],
            affected_query_example_rows=counts["affected_query_example_rows"],
            affected_target_example_rows=counts["affected_target_example_rows"],
        )
        reconstructed = FoldseekAuditResult(
            raw_tsv=authenticated.artifacts["raw_tsv"],
            mapped_parquet=authenticated.artifacts["mapped_parquet"],
            conflict_parquet=authenticated.artifacts["conflict_parquet"],
            producer_database=authenticated.artifacts["producer_database"],
            structural_audit=authenticated.artifacts["structural_audit"],
            producer_completion_candidate=(authenticated.producer_completion_candidate),
            candidate_verification=authenticated.candidate_verification,
            completion=authenticated.completion,
            final_verification=authenticated.final_verification,
            verification=verified,
        )
        if reconstructed != result:
            mismatches = [
                field
                for field in FoldseekAuditResult.__dataclass_fields__
                if getattr(reconstructed, field) != getattr(result, field)
            ]
            raise FoldseekAuditError(
                "returned Foldseek result disagrees with authenticated handoff: "
                + ", ".join(mismatches)
            )
        return reconstructed
    except Exception as error:
        _write_foldseek_handoff_failure(run, expected_final, error)
        if isinstance(error, FoldseekAuditError):
            raise
        raise FoldseekAuditError(
            f"final Foldseek CLI handoff authentication failed: {error}"
        ) from error

def _write_foldseek_handoff_failure(
    run: BenchmarkRun,
    expected_final: ArtifactIdentity,
    error: Exception,
) -> ArtifactIdentity:
    from dive.benchmark.evidence import _write_terminal_json

    destination = run.evidence_dir / "foldseek.handoff.failure.json"
    payload = {
        "schema_version": "dive-benchmark-foldseek-handoff-failure-v1",
        "status": "FAILED",
        "run_id": run.run_id,
        "stage": "cli_final_handoff",
        "reason_code": "final_handoff_authentication_failed",
        "detail": f"{type(error).__name__}: {error}",
        "expected_final_verification": _identity_mapping(expected_final),
    }
    try:
        _write_terminal_json(
            destination,
            payload,
            run.evidence_dir.parents[1],
            run.bulk_dir.parents[1],
        )
    except Exception as write_error:
        raise FoldseekAuditError(
            "final handoff failed and its create-new typed failure artifact "
            f"could not be written: {write_error}"
        ) from error
    return file_identity(destination)

def _run_foldseek_audit_claimed(
    run: BenchmarkRun,
    inventory: StructureInventory,
    contract: BenchmarkContract,
    *,
    source_structure_run: str,
    example_ids_by_parent: Mapping[str, tuple[str, ...]] | None = None,
    structure_evidence: Mapping[str, ArtifactIdentity] | None = None,
    _stage: dict[str, object],
    _retained_resources: list[MonitoredSnapshotAuthority],
) -> FoldseekAuditResult:

    if not isinstance(run, BenchmarkRun) or not isinstance(contract, BenchmarkContract):
        raise FoldseekAuditError("Foldseek audit requires a claimed run and contract")
    if run.contract_sha256 != contract.semantic_sha256:
        raise FoldseekAuditError("claimed run does not bind the benchmark contract")
    _stage["stage"] = "source_evidence_preflight"
    _require_production_structure_inputs(
        contract,
        inventory,
        source_structure_run=source_structure_run,
        structure_evidence=structure_evidence,
    )
    _stage["stage"] = "snapshot_monitor_preflight"
    try:
        preflight_monitored_tree_authority()
    except SnapshotAuthorityError as error:
        raise FoldseekAuditError(str(error)) from error
    _stage["stage"] = "attempt_authentication"
    repo_commit = _run_repository_commit(run)
    _stage["repo_commit"] = repo_commit
    _stage["stage"] = "binary_authentication"
    binary = authenticate_foldseek_binary(contract)
    _stage["stage"] = "structure_snapshot"
    snapshot = snapshot_structure_inventory(run, inventory)
    _stage["stage"] = "database_manifests"
    manifests = write_database_manifests(
        run,
        snapshot.inventory,
        source_structure_run=source_structure_run,
        source_inventory=inventory,
        example_ids_by_parent=example_ids_by_parent,
    )
    _stage["stage"] = "snapshot_monitor_authentication"
    snapshot_authority = _acquire_producer_snapshot_authority(run, contract, manifests)
    _retained_resources.append(snapshot_authority)
    _stage["stage"] = "path_reservation"
    try:
        _output_dir, tmp_dir, raw_tsv = _reserve_foldseek_paths(run)
    except Exception:
        snapshot_authority.close()
        raise
    stdout_path = run.evidence_dir / "foldseek.stdout.log"
    stderr_path = run.evidence_dir / "foldseek.stderr.log"
    command = (
        contract.foldseek.identity.path,
        "easy-search",
        str(snapshot.inventory.query_dir),
        str(snapshot.inventory.target_dir),
        str(raw_tsv),
        str(tmp_dir),
        "--format-output",
        "query,target,alnlen,qlen,tlen,alntmscore,evalue",
        "--threads",
        str(contract.foldseek.threads),
        "-v",
        "1",
    )
    environment = _foldseek_environment(contract.foldseek.threads)
    start_utc = datetime.now(UTC).isoformat()
    started = time.monotonic_ns()
    _stage["stage"] = "subprocess"
    try:
        with (
            _create_new_binary_log(stdout_path) as stdout,
            _create_new_binary_log(stderr_path) as stderr,
        ):
            _require_raw_absent(raw_tsv)
            result = subprocess.run(
                command,
                check=False,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
    except (OSError, subprocess.SubprocessError) as error:
        snapshot_authority.close()
        duration = (time.monotonic_ns() - started) / 1_000_000_000
        execution = _execution_mapping(
            command=command,
            environment=environment,
            binary=binary,
            start_utc=start_utc,
            end_utc=datetime.now(UTC).isoformat(),
            duration_seconds=duration,
            exit_code=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            raw_tsv=raw_tsv,
            repo_commit=repo_commit,
        )
        execution["stage"] = _stage["stage"]
        _write_foldseek_failure(run, "subprocess_error", str(error), execution)
        raise FoldseekAuditError(f"Foldseek subprocess failed: {error}") from error

    duration = (time.monotonic_ns() - started) / 1_000_000_000
    end_utc = datetime.now(UTC).isoformat()
    try:
        binary_post_execution = authenticate_foldseek_binary(contract)
    except FoldseekAuditError:
        binary_post_execution = None
    execution = _execution_mapping(
        command=command,
        environment=environment,
        binary=binary,
        binary_post_execution=binary_post_execution,
        start_utc=start_utc,
        end_utc=end_utc,
        duration_seconds=duration,
        exit_code=result.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        raw_tsv=raw_tsv,
        repo_commit=repo_commit,
    )
    execution["stage"] = _stage["stage"]
    if binary_post_execution is None or binary_post_execution != binary:
        snapshot_authority.close()
        _write_foldseek_failure(
            run,
            "binary_drift",
            "Foldseek binary identity changed during execution",
            execution,
        )
        raise FoldseekAuditError("Foldseek binary identity changed during execution")
    if result.returncode != 0:
        snapshot_authority.close()
        _write_foldseek_failure(
            run,
            "nonzero_exit",
            f"Foldseek exited with code {result.returncode}",
            execution,
        )
        raise FoldseekAuditError(f"Foldseek exited with exit code {result.returncode}")
    _stage["stage"] = "raw_output_authentication"
    try:
        created_raw = _authenticate_created_raw(raw_tsv)
    except FoldseekAuditError:
        snapshot_authority.close()
        execution["stage"] = _stage["stage"]
        _write_foldseek_failure(
            run,
            "raw_output_missing",
            "Foldseek exited zero without a regular raw TSV",
            execution,
        )
        raise
    if execution.get("raw_tsv") != _identity_mapping(created_raw):
        snapshot_authority.close()
        raise FoldseekAuditError("Foldseek raw output identity changed after execution")
    _stage["stage"] = "snapshot_post_execution_rehash"
    try:
        _verify_structure_snapshot(snapshot, inventory)
    except Exception:
        snapshot_authority.close()
        raise

    mapped_path = run.evidence_dir / "mapped_hits.parquet"
    conflict_path = run.evidence_dir / "structural_conflicts.parquet"
    _stage["stage"] = "producer_mapping_and_aggregation"
    try:
        rows = example_ids_by_parent or MappingProxyType(
            {
                record.parent_id: (record.example_id,)
                for record in (*snapshot.inventory.queries, *snapshot.inventory.targets)
            }
        )
        producer_artifacts = _produce_foldseek_artifacts(
            raw_tsv=raw_tsv,
            mapped_path=mapped_path,
            conflict_path=conflict_path,
            database_path=tmp_dir / "producer-aggregation.sqlite3",
            inventory=snapshot.inventory,
            example_ids_by_parent=rows,
            min_tm=contract.foldseek.min_tm_score,
            min_shorter_coverage=contract.foldseek.min_shorter_coverage,
        )
        producer = producer_artifacts.verification
        mapped_identity = producer_artifacts.mapped_parquet
        _assert_run_commit_current(repo_commit)
        raw_identity = file_identity(raw_tsv)
        _assert_output_artifacts_stable(
            execution=execution,
            raw_identity=raw_identity,
            mapped_identity=mapped_identity,
            conflict_identity=producer_artifacts.conflict_parquet,
            producer_database=producer_artifacts.database,
            manifests=manifests,
        )
        _stage["stage"] = "source_evidence_post_execution_reauthentication"
        refreshed_evidence = _reauthenticate_structure_evidence(
            contract,
            source_structure_run,
            structure_evidence,
        )
        _stage["stage"] = "structural_audit_candidate"
        audit_payload = _audit_mapping(
            run=run,
            contract=contract,
            source_structure_run=source_structure_run,
            structure_evidence=refreshed_evidence,
            manifests=manifests,
            execution=execution,
            raw_identity=raw_identity,
            mapped_identity=mapped_identity,
            conflict_identity=producer_artifacts.conflict_parquet,
            producer_database=producer_artifacts.database,
            verification=producer,
        )
        audit_path = run.evidence_dir / "structural_audit.json"
        from dive.benchmark.evidence import _write_terminal_json

        _write_terminal_json(
            audit_path,
            audit_payload,
            run.evidence_dir.parents[1],
            run.bulk_dir.parents[1],
        )
        audit_identity = file_identity(audit_path)
        _stage["stage"] = "producer_completion_candidate"
        producer_candidate = _write_producer_completion_candidate(
            run,
            contract=contract,
            source_structure_run=source_structure_run,
            structure_evidence=refreshed_evidence,
            manifests=manifests,
            raw_identity=raw_identity,
            artifacts=producer_artifacts,
            structural_audit=audit_identity,
        )
        _stage["stage"] = "producer_snapshot_monitor_finalization"
        try:
            snapshot_authority.finish()
        except SnapshotAuthorityError as error:
            raise FoldseekAuditError(str(error)) from error
        from dive.benchmark.foldseek_verify import (
            IndependentFoldseekVerificationError,
            write_candidate_verification,
            write_final_verification,
        )

        _stage["stage"] = "independent_candidate_verification"
        try:
            candidate_verification = write_candidate_verification(
                run,
                contract,
                producer_candidate=Path(producer_candidate.path),
                work_database=tmp_dir / "independent-verification.sqlite3",
                min_tm_score=contract.foldseek.min_tm_score,
                min_shorter_coverage=contract.foldseek.min_shorter_coverage,
            )
        except IndependentFoldseekVerificationError as error:
            raise FoldseekAuditError(
                f"independent Foldseek candidate verification failed: {error}"
            ) from error
        _assert_run_commit_current(repo_commit)
        _stage["stage"] = "generic_completion"
        completion = complete_benchmark_run(
            run,
            {
                "execution_contract_sha256": contract.execution_sha256,
                "source_structure_run": source_structure_run,
                "query_database_manifest": _identity_mapping(manifests.query.identity),
                "target_database_manifest": _identity_mapping(
                    manifests.target.identity
                ),
                "raw_tsv": _identity_mapping(raw_identity),
                "mapped_parquet": _identity_mapping(mapped_identity),
                "conflict_parquet": _identity_mapping(
                    producer_artifacts.conflict_parquet
                ),
                "producer_database": _identity_mapping(producer_artifacts.database),
                "structural_audit": _identity_mapping(audit_identity),
                "producer_completion_candidate": _identity_mapping(producer_candidate),
                "candidate_verification": _identity_mapping(candidate_verification),
                "required_final_verifier": "independent_verification.final.json",
            },
        )
        _stage["stage"] = "post_completion_independent_verification"
        try:
            final_verification = write_final_verification(
                run, contract, completion=Path(completion.path)
            )
        except IndependentFoldseekVerificationError as error:
            raise FoldseekAuditError(
                f"final independent Foldseek verification failed: {error}"
            ) from error
    except Exception as error:
        execution["stage"] = _stage["stage"]
        if not (run.evidence_dir / "foldseek.failure.json").exists():
            _write_foldseek_failure(
                run,
                "verification_or_publication_error",
                f"{type(error).__name__}: {error}",
                execution,
            )
        if isinstance(error, FoldseekAuditError):
            raise
        raise FoldseekAuditError(f"cannot verify Foldseek audit: {error}") from error
    finally:
        snapshot_authority.close()
    return FoldseekAuditResult(
        raw_identity,
        mapped_identity,
        producer_artifacts.conflict_parquet,
        producer_artifacts.database,
        audit_identity,
        producer_candidate,
        candidate_verification,
        completion,
        final_verification,
        producer,
    )

def _database_manifest_payload(
    inventory: StructureInventory,
    *,
    scope: str,
    records: tuple[StructureRecord, ...],
    directory: Path,
    expected_parents: tuple[str, ...],
    source_structure_run: str,
    source_records: tuple[StructureRecord, ...],
    example_ids_by_parent: Mapping[str, tuple[str, ...]] | None,
) -> dict[str, object]:
    by_key = _unique_records(records, scope)
    source_by_key = _unique_records(source_records, scope)
    if set(source_by_key) != set(by_key):
        raise FoldseekAuditError(f"{scope} source/snapshot key universe drifted")
    observed_parents = {item.parent_id for item in records}
    if observed_parents != set(expected_parents):
        raise FoldseekAuditError(
            f"{scope} database manifest is missing a required parent"
        )
    expected_files = {f"{key}.pdb" for key in by_key}
    try:
        entries = tuple(directory.iterdir())
    except OSError as error:
        raise FoldseekAuditError(
            f"cannot inspect {scope} structure directory: {error}"
        ) from error
    actual_files = {item.name for item in entries}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise FoldseekAuditError(
            f"{scope} database has missing {missing[:3]} or untracked extra {extra[:3]} files"
        )

    mapped = []
    for key, record in sorted(by_key.items()):
        try:
            decoded = decode_structure_key(key)
        except StructureInventoryError as error:
            raise FoldseekAuditError(
                f"malformed {scope} Foldseek key {key!r}"
            ) from error
        if decoded != (record.family, record.parent_id, record.chain_id):
            raise FoldseekAuditError(f"malformed {scope} Foldseek key {key!r}")
        path = directory / f"{key}.pdb"
        if path.is_symlink() or not path.is_file():
            raise FoldseekAuditError(
                f"{scope} database entry is not a regular non-symlink file: {key}"
            )
        observed = file_identity(path)
        if observed != record.materialized:
            raise FoldseekAuditError(f"{scope} database identity drifted for {key}")
        mapped.append(
            {
                "foldseek_key": key,
                "relative_path": path.name,
                "family": record.family,
                "partition": record.partition,
                "parent_id": record.parent_id,
                "example_id": record.example_id,
                "chain_id": record.chain_id,
                "component_id": record.component_id,
                "role": record.role,
                "materialized": _identity_mapping(record.materialized),
                "task3_materialized": _identity_mapping(
                    source_by_key[key].materialized
                ),
            }
        )
    row_mapping = example_ids_by_parent or MappingProxyType(
        {record.parent_id: (record.example_id,) for record in records}
    )
    if not observed_parents <= set(row_mapping):
        raise FoldseekAuditError(f"{scope} example-row manifest is incomplete")
    return {
        "schema_version": "dive-benchmark-foldseek-database-manifest-v1",
        "scope": scope,
        "source_structure_run": source_structure_run,
        "source_structure_inventory": _identity_mapping(inventory.inventory),
        "source_structure_completion": _identity_mapping(inventory.completion),
        "directory": str(directory),
        "record_count": len(records),
        "parent_group_count": len(observed_parents),
        "example_rows": [
            {"parent_id": parent, "example_ids": list(row_mapping[parent])}
            for parent in sorted(observed_parents)
        ],
        "entries": mapped,
    }

def _foldseek_environment(threads: int) -> dict[str, str]:
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": str(threads),
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }

def _create_new_binary_log(path: Path):
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o664,
        )
    except OSError as error:
        raise FoldseekAuditError(
            f"cannot reserve terminal log {path}: {error}"
        ) from error
    return os.fdopen(descriptor, "wb")

def _optional_identity(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return _identity_mapping(file_identity(path))
    except Exception as error:
        raise FoldseekAuditError(
            f"cannot authenticate execution artifact {path}: {error}"
        ) from error

def _run_repository_commit(run: BenchmarkRun) -> str:

    attempt_path = Path(run.attempt.path)
    if file_identity(attempt_path) != run.attempt:
        raise FoldseekAuditError("benchmark attempt drifted before Foldseek execution")
    try:
        raw = attempt_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FoldseekAuditError("cannot read Foldseek benchmark attempt") from error
    if (
        not isinstance(payload, Mapping)
        or canonical_json_bytes(payload) != raw
        or payload.get("status") != "BUILDING"
        or payload.get("run_id") != run.run_id
        or payload.get("contract_sha256") != run.contract_sha256
    ):
        raise FoldseekAuditError("benchmark attempt does not authenticate this run")
    commit = payload.get("repo_commit")
    if (
        type(commit) is not str
        or len(commit) != 40
        or not set(commit) <= _SHA256_CHARACTERS
    ):
        raise FoldseekAuditError("benchmark attempt has an invalid clean commit")
    return commit

def _assert_run_commit_current(expected: str) -> None:

    from dive.benchmark import evidence

    try:
        observed = evidence._current_clean_commit()
    except Exception as error:
        raise FoldseekAuditError(
            "cannot re-authenticate the clean Foldseek producer commit"
        ) from error
    if observed != expected:
        raise FoldseekAuditError(
            "repository commit or clean state changed during Foldseek execution"
        )

def _assert_output_artifacts_stable(
    *,
    execution: Mapping[str, object],
    raw_identity: ArtifactIdentity,
    mapped_identity: ArtifactIdentity,
    conflict_identity: ArtifactIdentity,
    producer_database: ArtifactIdentity,
    manifests: FoldseekDatabaseManifests,
) -> None:

    if execution.get("raw_tsv") != _identity_mapping(raw_identity):
        raise FoldseekAuditError("raw Foldseek TSV changed during verification")
    for label in ("stdout", "stderr"):
        declared = execution.get(label)
        if not isinstance(declared, Mapping):
            raise FoldseekAuditError(f"Foldseek {label} log was not authenticated")
        identity = _artifact_identity(declared, f"Foldseek {label}")
        if file_identity(Path(identity.path)) != identity:
            raise FoldseekAuditError(
                f"Foldseek {label} log changed during verification"
            )
    if file_identity(Path(mapped_identity.path)) != mapped_identity:
        raise FoldseekAuditError("mapped Foldseek Parquet changed during verification")
    if file_identity(Path(conflict_identity.path)) != conflict_identity:
        raise FoldseekAuditError("conflict Parquet changed during verification")
    if file_identity(Path(producer_database.path)) != producer_database:
        raise FoldseekAuditError(
            "producer aggregation database changed during verification"
        )
    for manifest in (manifests.query, manifests.target):
        if file_identity(Path(manifest.identity.path)) != manifest.identity:
            raise FoldseekAuditError(
                f"{manifest.scope} database manifest changed during verification"
            )

def _execution_mapping(
    *,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    binary: FoldseekBinary,
    binary_post_execution: FoldseekBinary | None = None,
    start_utc: str,
    end_utc: str,
    duration_seconds: float,
    exit_code: int | None,
    stdout_path: Path,
    stderr_path: Path,
    raw_tsv: Path,
    repo_commit: str,
) -> dict[str, object]:
    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        raise FoldseekAuditError("Foldseek duration is not finite")
    return {
        "command": list(command),
        "environment": dict(sorted(environment.items())),
        "binary": _identity_mapping(binary.identity),
        "binary_post_execution": (
            _identity_mapping(binary_post_execution.identity)
            if binary_post_execution is not None
            else _optional_identity(Path(binary.identity.path))
        ),
        "version_output": binary.version_output,
        "version_output_post_execution": (
            binary_post_execution.version_output
            if binary_post_execution is not None
            else None
        ),
        "repo_commit": repo_commit,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_seconds": duration_seconds,
        "exit_code": exit_code,
        "stdout": _optional_identity(stdout_path),
        "stderr": _optional_identity(stderr_path),
        "raw_tsv": _optional_identity(raw_tsv),
    }

def _write_foldseek_failure(
    run: BenchmarkRun,
    reason_code: str,
    detail: str,
    execution: Mapping[str, object],
) -> ArtifactIdentity:
    from dive.benchmark.evidence import _write_terminal_json

    destination = run.evidence_dir / "foldseek.failure.json"
    payload = {
        "schema_version": "dive-benchmark-foldseek-failure-v1",
        "status": "FAILED",
        "run_id": run.run_id,
        "stage": execution.get("stage", "unknown"),
        "reason_code": reason_code,
        "detail": detail,
        "execution": dict(execution),
    }
    try:
        _write_terminal_json(
            destination,
            payload,
            run.evidence_dir.parents[1],
            run.bulk_dir.parents[1],
        )
    except Exception as error:
        raise FoldseekAuditError(
            f"{detail}; additionally could not record Foldseek failure: {error}"
        ) from error
    return file_identity(destination)

_MAPPED_COLUMNS = (
    "query_key",
    "target_key",
    "alignment_length",
    "query_length",
    "target_length",
    "tm_score",
    "evalue",
    "shorter_coverage",
    "structural_conflict",
    "query_family",
    "query_partition",
    "query_parent_id",
    "query_example_id",
    "query_chain_id",
    "query_role",
    "target_family",
    "target_partition",
    "target_parent_id",
    "target_example_id",
    "target_chain_id",
    "target_role",
)

_CONFLICT_COLUMNS = (
    "query_family",
    "query_parent_id",
    "target_family",
    "target_parent_id",
    "chain_hit_count",
)

def _produce_foldseek_artifacts(
    *,
    raw_tsv: Path,
    mapped_path: Path,
    conflict_path: Path,
    database_path: Path,
    inventory: StructureInventory,
    example_ids_by_parent: Mapping[str, tuple[str, ...]],
    min_tm: float,
    min_shorter_coverage: float,
) -> _ProducerArtifacts:

    import pyarrow as pa
    import pyarrow.parquet as pq

    if database_path.exists() or database_path.is_symlink():
        raise FoldseekAuditError("producer aggregation database already exists")
    query = _unique_records(inventory.queries, "query")
    target = _unique_records(inventory.targets, "target")
    connection = sqlite3.connect(database_path)
    raw_count = 0
    conflicting_count = 0
    try:
        mapped_handle = _create_new_binary_log(mapped_path)
        try:
            mapped_writer = pq.ParquetWriter(mapped_handle, _mapped_arrow_schema(pa))
            try:
                _initialize_producer_database(connection, example_ids_by_parent)
                batch: list[dict[str, object]] = []
                for hit in _iter_foldseek_tsv(raw_tsv):
                    raw_count += 1
                    query_record = query.get(hit.query_key)
                    target_record = target.get(hit.target_key)
                    if query_record is None or target_record is None:
                        raise FoldseekAuditError("producer raw key is unknown")
                    try:
                        connection.execute(
                            "INSERT INTO raw_pairs(query_key, target_key) VALUES (?, ?)",
                            (hit.query_key, hit.target_key),
                        )
                    except sqlite3.IntegrityError as error:
                        raise FoldseekAuditError(
                            "duplicate Foldseek query/target pair"
                        ) from error
                    conflict = structural_conflict(
                        hit,
                        min_tm=min_tm,
                        min_shorter_coverage=min_shorter_coverage,
                    )
                    row = _mapped_row(
                        hit,
                        query_record,
                        target_record,
                        min_tm=min_tm,
                        min_shorter_coverage=min_shorter_coverage,
                    )
                    batch.append(row)
                    if len(batch) == 50_000:
                        mapped_writer.write_table(
                            pa.Table.from_pylist(batch, schema=_mapped_arrow_schema(pa))
                        )
                        batch.clear()
                    if conflict:
                        conflicting_count += 1
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
                                query_record.family,
                                query_record.parent_id,
                                target_record.family,
                                target_record.parent_id,
                            ),
                        )
                        connection.execute(
                            "INSERT OR IGNORE INTO affected_query_parents(parent_id) "
                            "VALUES (?)",
                            (query_record.parent_id,),
                        )
                        connection.execute(
                            "INSERT OR IGNORE INTO affected_target_parents(parent_id) "
                            "VALUES (?)",
                            (target_record.parent_id,),
                        )
                if batch:
                    mapped_writer.write_table(
                        pa.Table.from_pylist(batch, schema=_mapped_arrow_schema(pa))
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                mapped_writer.close()
            mapped_handle.flush()
            os.fsync(mapped_handle.fileno())
        finally:
            mapped_handle.close()
        verification = _producer_counts(connection, raw_count, conflicting_count)
        _write_conflict_parquet(connection, conflict_path)
    finally:
        connection.close()
    return _ProducerArtifacts(
        file_identity(mapped_path),
        file_identity(conflict_path),
        file_identity(database_path),
        verification,
    )

def _initialize_producer_database(
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
        CREATE TABLE affected_query_parents(
            parent_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TABLE affected_target_parents(
            parent_id TEXT PRIMARY KEY
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

def _producer_counts(
    connection: sqlite3.Connection, raw_count: int, conflict_count: int
) -> FoldseekVerification:
    def scalar(statement: str) -> int:
        return int(connection.execute(statement).fetchone()[0])

    return FoldseekVerification(
        hits=(),
        conflicts=(),
        raw_hit_count=raw_count,
        conflicting_chain_hit_count=conflict_count,
        parent_pair_conflict_count=scalar("SELECT COUNT(*) FROM conflicts"),
        affected_query_parent_groups=scalar(
            "SELECT COUNT(*) FROM affected_query_parents"
        ),
        affected_target_parent_groups=scalar(
            "SELECT COUNT(*) FROM affected_target_parents"
        ),
        affected_query_example_rows=scalar(
            """
            SELECT COUNT(*) FROM examples e
            JOIN affected_query_parents p ON p.parent_id = e.parent_id
            """
        ),
        affected_target_example_rows=scalar(
            """
            SELECT COUNT(*) FROM examples e
            JOIN affected_target_parents p ON p.parent_id = e.parent_id
            """
        ),
    )

def _write_conflict_parquet(connection: sqlite3.Connection, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            *(pa.field(name, pa.string()) for name in _CONFLICT_COLUMNS[:4]),
            pa.field("chain_hit_count", pa.int64()),
        ]
    )
    handle = _create_new_binary_log(path)
    writer = pq.ParquetWriter(handle, schema)
    try:
        cursor = connection.execute(
            """
            SELECT query_family, query_parent_id,
                   target_family, target_parent_id, chain_hit_count
            FROM conflicts
            ORDER BY query_family, query_parent_id,
                     target_family, target_parent_id
            """
        )
        while rows := cursor.fetchmany(50_000):
            writer.write_table(
                pa.Table.from_pylist(
                    [dict(zip(_CONFLICT_COLUMNS, row, strict=True)) for row in rows],
                    schema=schema,
                )
            )
    finally:
        writer.close()
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()

def _mapped_row(
    hit: FoldseekHit,
    query: StructureRecord,
    target: StructureRecord,
    *,
    min_tm: float,
    min_shorter_coverage: float,
) -> dict[str, object]:
    return {
        "query_key": hit.query_key,
        "target_key": hit.target_key,
        "alignment_length": hit.alignment_length,
        "query_length": hit.query_length,
        "target_length": hit.target_length,
        "tm_score": float(hit.tm_score),
        "evalue": float(hit.evalue),
        "shorter_coverage": float(hit.shorter_coverage),
        "structural_conflict": structural_conflict(
            hit,
            min_tm=min_tm,
            min_shorter_coverage=min_shorter_coverage,
        ),
        "query_family": query.family,
        "query_partition": query.partition,
        "query_parent_id": query.parent_id,
        "query_example_id": query.example_id,
        "query_chain_id": query.chain_id,
        "query_role": query.role,
        "target_family": target.family,
        "target_partition": target.partition,
        "target_parent_id": target.parent_id,
        "target_example_id": target.example_id,
        "target_chain_id": target.chain_id,
        "target_role": target.role,
    }

def _mapped_rows(
    hits: Iterable[FoldseekHit],
    inventory: StructureInventory,
    *,
    min_tm: float,
    min_shorter_coverage: float,
) -> Iterator[dict[str, object]]:
    query = _unique_records(inventory.queries, "query")
    target = _unique_records(inventory.targets, "target")
    for hit in hits:
        if not isinstance(hit, FoldseekHit):
            raise FoldseekAuditError("mapped Foldseek rows require typed hits")
        query_record = query.get(hit.query_key)
        target_record = target.get(hit.target_key)
        if query_record is None:
            raise FoldseekAuditError(f"unknown query key {hit.query_key!r}")
        if target_record is None:
            raise FoldseekAuditError(f"unknown target key {hit.target_key!r}")
        yield _mapped_row(
            hit,
            query_record,
            target_record,
            min_tm=min_tm,
            min_shorter_coverage=min_shorter_coverage,
        )

def _write_mapped_parquet(
    path: Path,
    raw_tsv: Path,
    inventory: StructureInventory,
    *,
    min_tm: float,
    min_shorter_coverage: float,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _mapped_arrow_schema(pa)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o664,
        )
        with os.fdopen(descriptor, "wb") as handle:
            writer = pq.ParquetWriter(handle, schema)
            try:
                batch: list[dict[str, object]] = []
                for row in _mapped_rows(
                    _iter_foldseek_tsv(raw_tsv),
                    inventory,
                    min_tm=min_tm,
                    min_shorter_coverage=min_shorter_coverage,
                ):
                    batch.append(row)
                    if len(batch) == 50_000:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            finally:
                writer.close()
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as error:
        raise FoldseekAuditError(
            f"cannot create mapped Foldseek Parquet: {error}"
        ) from error

def _independently_verify_mapped_output(
    inventory: StructureInventory,
    raw_tsv: Path,
    mapped_parquet: Path,
    conflict_parquet: Path,
    manifests: FoldseekDatabaseManifests,
    *,
    work_database: Path | None = None,
    example_ids_by_parent: Mapping[str, tuple[str, ...]] | None,
    min_tm: float,
    min_shorter_coverage: float,
) -> FoldseekVerification:

    from dive.benchmark.foldseek_verify import (
        IndependentFoldseekVerificationError,
        independently_verify_outputs,
    )

    rows = example_ids_by_parent
    if rows is None:
        rows = MappingProxyType(
            {
                record.parent_id: (record.example_id,)
                for record in (*inventory.queries, *inventory.targets)
            }
        )
    try:
        counts = independently_verify_outputs(
            query_manifest=Path(manifests.query.identity.path),
            target_manifest=Path(manifests.target.identity.path),
            raw_tsv=raw_tsv,
            mapped_parquet=mapped_parquet,
            conflict_parquet=conflict_parquet,
            work_database=work_database
            or mapped_parquet.with_suffix(".independent.sqlite3"),
            min_tm_score=min_tm,
            min_shorter_coverage=min_shorter_coverage,
            example_ids_by_parent=rows,
        )
    except IndependentFoldseekVerificationError as error:
        raise FoldseekAuditError(
            f"independent Foldseek verification failed: {error}"
        ) from error
    return FoldseekVerification(
        hits=(),
        conflicts=(),
        raw_hit_count=counts.raw_hit_count,
        conflicting_chain_hit_count=counts.conflicting_chain_hit_count,
        parent_pair_conflict_count=counts.parent_pair_conflict_count,
        affected_query_parent_groups=counts.affected_query_parent_groups,
        affected_target_parent_groups=counts.affected_target_parent_groups,
        affected_query_example_rows=counts.affected_query_example_rows,
        affected_target_example_rows=counts.affected_target_example_rows,
    )

def _mapped_arrow_schema(pa):
    return pa.schema(
        [
            *(pa.field(name, pa.string()) for name in _MAPPED_COLUMNS[:2]),
            *(pa.field(name, pa.int64()) for name in _MAPPED_COLUMNS[2:5]),
            *(pa.field(name, pa.float64()) for name in _MAPPED_COLUMNS[5:8]),
            pa.field(_MAPPED_COLUMNS[8], pa.bool_()),
            *(pa.field(name, pa.string()) for name in _MAPPED_COLUMNS[9:]),
        ]
    )

def _verification_counts(value: FoldseekVerification) -> tuple[int, ...]:
    return (
        value.raw_hit_count,
        value.conflicting_chain_hit_count,
        value.parent_pair_conflict_count,
        value.affected_query_parent_groups,
        value.affected_target_parent_groups,
        value.affected_query_example_rows,
        value.affected_target_example_rows,
        value.partition_movements,
    )

def _audit_mapping(
    *,
    run: BenchmarkRun,
    contract: BenchmarkContract,
    source_structure_run: str,
    structure_evidence: Mapping[str, ArtifactIdentity],
    manifests: FoldseekDatabaseManifests,
    execution: Mapping[str, object],
    raw_identity: ArtifactIdentity,
    mapped_identity: ArtifactIdentity,
    conflict_identity: ArtifactIdentity,
    producer_database: ArtifactIdentity,
    verification: FoldseekVerification,
) -> dict[str, object]:
    return {
        "schema_version": "dive-benchmark-structural-audit-v1",
        "status": "CANDIDATE",
        "run_id": run.run_id,
        "contract_sha256": contract.semantic_sha256,
        "execution_contract_sha256": contract.execution_sha256,
        "source_structure_run": source_structure_run,
        "source_structure_evidence": {
            name: _identity_mapping(identity)
            for name, identity in sorted(structure_evidence.items())
        },
        "thresholds": {
            "tm_score": {"operator": ">=", "value": contract.foldseek.min_tm_score},
            "shorter_chain_coverage": {
                "operator": ">=",
                "value": contract.foldseek.min_shorter_coverage,
                "clamped_to_unit_interval": True,
            },
        },
        "execution": dict(execution),
        "query_database_manifest": _identity_mapping(manifests.query.identity),
        "target_database_manifest": _identity_mapping(manifests.target.identity),
        "raw_tsv": _identity_mapping(raw_identity),
        "mapped_parquet": _identity_mapping(mapped_identity),
        "conflict_parquet": _identity_mapping(conflict_identity),
        "counts": _audit_counts(verification),
        "aggregation": {
            "backend": "sqlite",
            "producer_database": _identity_mapping(producer_database),
            "raw_pair_uniqueness": "PRIMARY KEY(query_key, target_key)",
            "parent_pair_uniqueness": (
                "PRIMARY KEY(query_family, query_parent_id, "
                "target_family, target_parent_id)"
            ),
            "conflicts_streamed_to_parquet": True,
        },
        "view_effect": {
            "strict_view_only": verification.strict_view_only,
            "partition_movements": verification.partition_movements,
        },
        "units": {
            "raw_hits": "chain-pair alignments",
            "parent_pair_conflicts": "unique query-parent/target-parent pairs",
            "affected_parent_groups": "distinct parent groups",
            "affected_example_rows": "distinct frozen example rows",
        },
        "required_independent_verification": {
            "candidate_artifact": "independent_verification.candidate.json",
            "post_completion_artifact": "independent_verification.final.json",
            "generic_completion_alone_is_insufficient": True,
        },
    }

def _unique_records(
    records: tuple[StructureRecord, ...], label: str
) -> dict[str, StructureRecord]:
    result: dict[str, StructureRecord] = {}
    allowed_partitions = (
        {"validation", "test-blind"}
        if label == "query"
        else {"train", "legacy-dev"}
        if label == "target"
        else None
    )
    if allowed_partitions is None:
        raise FoldseekAuditError(f"unknown Foldseek database scope {label!r}")
    for record in records:
        if record.foldseek_key in result:
            raise FoldseekAuditError(f"duplicate {label} key {record.foldseek_key}")
        try:
            decoded = decode_structure_key(record.foldseek_key)
        except StructureInventoryError as error:
            raise FoldseekAuditError(
                f"malformed {label} key {record.foldseek_key!r}"
            ) from error
        if decoded != (record.family, record.parent_id, record.chain_id):
            raise FoldseekAuditError(
                f"ambiguous {label} key mapping {record.foldseek_key!r}"
            )
        if record.partition not in allowed_partitions:
            raise FoldseekAuditError(
                f"{label} key has prohibited partition {record.partition!r}"
            )
        result[record.foldseek_key] = record
    return result

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }
