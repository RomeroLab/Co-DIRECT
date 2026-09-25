
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dive.benchmark.contracts import ArtifactIdentity, BenchmarkContract, file_identity
from dive.training.parent_projection import (
    PRODUCTION_CONTRACT_V2,
    attach_parent_projection,
    load_parent_projection,
    load_reviewed_projection_spec,
    verify_manifest_inventory,
)
from dive.training.preflight import write_create_new_json

class BenchmarkInputError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    example_id: str
    parent_id: str
    family: str
    partition: str
    path: str
    roles: Mapping[str, str]
    role_realized: bool
    quarantine_reason: str | None

@dataclass(frozen=True, slots=True)
class BenchmarkParent:
    parent_id: str
    family: str
    partition: str
    outer_partition: str
    view_standard: bool
    view_strict: bool
    deposition_date: str | None
    example_ids: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class BenchmarkCounts:
    example_rows: int
    parent_groups: int

@dataclass(frozen=True, slots=True)
class FrozenBenchmarkInputs:
    projection_identity: str
    projection_row_count: int
    loader_manifest_identities: tuple[ArtifactIdentity, ...]
    input_identities: Mapping[str, ArtifactIdentity]
    parents: tuple[BenchmarkParent, ...]
    examples: tuple[BenchmarkExample, ...]
    counts: Mapping[str, Mapping[str, BenchmarkCounts]]

_FAMILIES = ("binder", "ame", "antibody")
_PARTITIONS = ("train", "validation")
_QUARANTINE = {"no_compatible_chain", "role_contract_error", "no_complete_assignment"}
_REPO_ROOT = Path(__file__).resolve().parents[3]

def load_frozen_benchmark_inputs(contract: BenchmarkContract) -> FrozenBenchmarkInputs:

    if not isinstance(contract, BenchmarkContract):
        raise BenchmarkInputError(
            "load_frozen_benchmark_inputs requires BenchmarkContract"
        )
    inputs = contract.authenticated_inputs
    train_raw = _read_identity(inputs.train_data_ready, "train-data-ready")
    train_data = _canonical_object(train_raw, "train-data-ready")
    realized_identity, projection_identities = _verify_train_data_ready(
        train_data, inputs
    )
    _verify_descendant(str(train_data["code_commit"]))

    audit_raw = _read_identity(inputs.resolution_audit, "v5 resolution audit")
    audit_rows = _load_jsonl(audit_raw)
    if len(audit_rows) != 26_459:
        raise BenchmarkInputError(
            "v5 resolution audit does not contain 26,459 canonical rows"
        )

    canonical_rows, canonical_identities = _load_canonical_manifests(
        inputs.canonical_manifests
    )
    manifests = verify_manifest_inventory(
        inputs.loader_manifest_root, PRODUCTION_CONTRACT_V2.manifests
    )
    reviewed = load_reviewed_projection_spec(contract=PRODUCTION_CONTRACT_V2)
    projection = load_parent_projection(
        "v2",
        manifests=manifests,
        contract=PRODUCTION_CONTRACT_V2,
        reviewed_spec=reviewed,
    )
    if (
        projection.get("row_count") != 26_309
        or projection.get("projection_hash") != reviewed.spec.projection_hash
    ):
        raise BenchmarkInputError("v2 projection identity or row count drifted")
    try:
        attached = attach_parent_projection(manifests, projection)
    except Exception as error:
        raise BenchmarkInputError(f"missing parent projection: {error}") from error

    audit_by_example = _index_audit_rows(audit_rows)
    canonical_by_example = _index_canonical_rows(canonical_rows)
    examples = _join_admitted(attached, audit_by_example, canonical_by_example)
    examples += _rejected_examples(audit_by_example, canonical_by_example)
    _verify_quarantine(examples)
    _verify_reconciliation(examples)
    parents = _parents_for(examples, canonical_by_example)
    counts = _counts_for(examples, train_data)
    identities = {
        "train_data_ready": inputs.train_data_ready,
        "resolution_audit": inputs.resolution_audit,
        "realized_split_evidence": realized_identity,
        **projection_identities,
        **{
            f"canonical_manifest:{name}": identity
            for name, identity in canonical_identities.items()
        },
    }
    loader_identities = tuple(
        ArtifactIdentity(
            str(inputs.loader_manifest_root / item.spec.name),
            item.spec.sha256,
            item.spec.size_bytes,
        )
        for item in manifests
    )
    return FrozenBenchmarkInputs(
        projection_identity="v2",
        projection_row_count=26_309,
        loader_manifest_identities=loader_identities,
        input_identities=MappingProxyType(identities),
        parents=tuple(sorted(parents, key=lambda item: item.parent_id)),
        examples=tuple(sorted(examples, key=lambda item: item.example_id)),
        counts=MappingProxyType(
            {family: MappingProxyType(values) for family, values in counts.items()}
        ),
    )

def write_input_inventory(
    run: Any, inputs: FrozenBenchmarkInputs, *, supersedes_run_id: str | None = None
) -> ArtifactIdentity:

    if not hasattr(run, "evidence_dir") or not isinstance(
        inputs, FrozenBenchmarkInputs
    ):
        raise BenchmarkInputError(
            "write_input_inventory requires a claimed benchmark run and frozen inputs"
        )
    destination = Path(run.evidence_dir) / "input_inventory.json"
    payload = input_inventory_mapping(inputs, supersedes_run_id=supersedes_run_id)
    try:
        write_create_new_json(destination, payload)
    except Exception as error:
        raise BenchmarkInputError(f"cannot create input inventory: {error}") from error
    return file_identity(destination)

def input_inventory_mapping(
    inputs: FrozenBenchmarkInputs, *, supersedes_run_id: str | None = None
) -> dict[str, object]:

    if not isinstance(inputs, FrozenBenchmarkInputs):
        raise BenchmarkInputError("input inventory requires FrozenBenchmarkInputs")
    payload: dict[str, object] = {
        "schema_version": "dive-benchmark-input-inventory-v1",
        "projection": {
            "identity": inputs.projection_identity,
            "row_count": inputs.projection_row_count,
        },
        "loader_manifests": [
            _identity_mapping(item) for item in inputs.loader_manifest_identities
        ],
        "inputs": {
            name: _identity_mapping(identity)
            for name, identity in sorted(inputs.input_identities.items())
        },
        "counts": {
            family: {
                partition: {
                    "example_rows": value.example_rows,
                    "parent_groups": value.parent_groups,
                }
                for partition, value in values.items()
            }
            for family, values in inputs.counts.items()
        },
        "parents": [
            {
                "parent_id": item.parent_id,
                "family": item.family,
                "partition": item.partition,
                "outer_partition": item.outer_partition,
                "view_standard": item.view_standard,
                "view_strict": item.view_strict,
                "deposition_date": item.deposition_date,
                "example_ids": list(item.example_ids),
            }
            for item in inputs.parents
        ],
        "examples": [
            {
                "example_id": item.example_id,
                "parent_id": item.parent_id,
                "family": item.family,
                "partition": item.partition,
                "path": item.path,
                "roles": dict(item.roles),
                "role_realized": item.role_realized,
                "quarantine_reason": item.quarantine_reason,
            }
            for item in inputs.examples
        ],
    }
    if supersedes_run_id is not None:
        payload["supersedes_run_id"] = supersedes_run_id
    return payload

def _read_identity(expected: ArtifactIdentity, label: str) -> bytes:
    path = Path(expected.path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BenchmarkInputError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not os.path.stat.S_ISREG(before.st_mode):
            raise BenchmarkInputError(f"{label} must be a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    observed = ArtifactIdentity(
        str(path.resolve()), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or observed != expected
    ):
        raise BenchmarkInputError(f"{label} identity drifted")
    return raw

def _canonical_object(raw: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkInputError(f"{label} is not JSON") from error
    if not isinstance(value, Mapping):
        raise BenchmarkInputError(f"{label} is not an object")
    return value

def _verify_train_data_ready(
    record: Mapping[str, object], inputs: object
) -> tuple[ArtifactIdentity, Mapping[str, ArtifactIdentity]]:
    required = {
        "artifact",
        "verdict",
        "ready_for_confirmatory_training",
        "code_commit",
        "primary_corpus",
        "parent_projection",
        "realized_split_audit",
        "role_realization_yield",
    }
    if (
        not required.issubset(record)
        or record["artifact"] != "TRAIN_DATA_READY"
        or record["verdict"] != "PASS"
        or record["ready_for_confirmatory_training"] is not True
    ):
        raise BenchmarkInputError("TRAIN_DATA_READY is not an authenticated PASS")
    primary = record["primary_corpus"]
    projection = record["parent_projection"]
    realized = record["realized_split_audit"]
    role_yield = record["role_realization_yield"]
    if not all(
        isinstance(item, Mapping)
        for item in (primary, projection, realized, role_yield)
    ):
        raise BenchmarkInputError("TRAIN_DATA_READY input section is invalid")
    if (
        primary.get("admitted_example_rows") != 26_309
        or primary.get("canonical_example_rows") != 26_459
        or primary.get("rejected") != 150
    ):
        raise BenchmarkInputError("TRAIN_DATA_READY corpus reconciliation changed")
    if (
        primary.get("destination") != str(inputs.loader_manifest_root)
        or primary.get("audit_jsonl_sha256") != inputs.resolution_audit.sha256
    ):
        raise BenchmarkInputError("TRAIN_DATA_READY v5 input identity changed")
    published_manifests = primary.get("manifests")
    expected_manifests = {
        spec.name: {"sha256": spec.sha256, "size_bytes": spec.size_bytes}
        for spec in PRODUCTION_CONTRACT_V2.manifests
    }
    if published_manifests != expected_manifests:
        raise BenchmarkInputError("TRAIN_DATA_READY v5 manifest identities changed")
    if (
        projection.get("identity") != "v2"
        or projection.get("contract") != "PRODUCTION_CONTRACT_V2"
        or projection.get("row_count") != 26_309
    ):
        raise BenchmarkInputError(
            "TRAIN_DATA_READY does not require parent projection v2"
        )
    evidence = ArtifactIdentity(
        str(realized.get("evidence")),
        str(realized.get("evidence_sha256")),
        _file_size(Path(str(realized.get("evidence"))), "realized split evidence"),
    )
    _read_identity(evidence, "realized split evidence")
    if any(
        realized.get(key, {}).get("status") != "PASS"
        for key in (
            "train_validation_parent_overlap",
            "blind_parent_leakage",
            "parents_spanning_partitions",
        )
    ):
        raise BenchmarkInputError("realized split evidence is not PASS")
    antibody = role_yield.get("antibody")
    if (
        not isinstance(antibody, Mapping)
        or antibody.get("admitted") != 1_065
        or antibody.get("canonical") != 1_166
        or antibody.get("quarantined")
        != {
            "no_compatible_chain": 71,
            "role_contract_error": 28,
            "no_complete_assignment": 2,
        }
    ):
        raise BenchmarkInputError(
            "antibody role-realization yield or quarantine changed"
        )
    projection_identity = _identity_from_record(
        projection, "path", "file_sha256", "parent projection"
    )
    completion_identity = _identity_from_record(
        projection,
        "completion_path",
        "completion_sha256",
        "parent projection completion",
    )
    return evidence, MappingProxyType(
        {
            "parent_projection": projection_identity,
            "parent_projection_spec": ArtifactIdentity(
                str((_REPO_ROOT / Path(str(projection["spec_path"]))).resolve()),
                str(projection["spec_sha256"]),
                _file_size(
                    _REPO_ROOT / Path(str(projection["spec_path"])),
                    "parent projection spec",
                ),
            ),
            "parent_projection_completion": completion_identity,
        }
    )

def _identity_from_record(
    record: Mapping[str, object], path_key: str, digest_key: str, label: str
) -> ArtifactIdentity:
    path = Path(str(record.get(path_key)))
    identity = ArtifactIdentity(
        str(path.resolve()), str(record.get(digest_key)), _file_size(path, label)
    )
    _read_identity(identity, label)
    return identity

def _file_size(path: Path, label: str) -> int:
    try:
        return path.stat().st_size
    except OSError as error:
        raise BenchmarkInputError(f"cannot stat {label}: {error}") from error

def _verify_descendant(commit: str) -> None:
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise BenchmarkInputError(
            "TRAIN_DATA_READY commit is not an ancestor of current HEAD"
        )

def _load_jsonl(raw: bytes) -> tuple[Mapping[str, object], ...]:
    rows = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BenchmarkInputError("v5 resolution audit is invalid JSONL") from error
        if not isinstance(row, Mapping) or set(row) != {
            "canonical_chains",
            "detail",
            "disposition",
            "exact_edges",
            "example_id",
            "family",
            "homomer_loaded_identical",
            "loaded_chains",
            "mapping_sha256",
            "non_polymer_passthrough",
            "partition",
            "path",
            "wildcard_edges",
        }:
            raise BenchmarkInputError("v5 resolution audit row schema changed")
        rows.append(MappingProxyType(dict(row)))
    return tuple(rows)

def _load_canonical_manifests(
    specs: tuple[tuple[str, ArtifactIdentity], ...],
) -> tuple[tuple[Mapping[str, object], ...], dict[str, ArtifactIdentity]]:
    import pandas as pd

    rows = []
    identities = {}
    for family, identity in specs:
        raw = _read_identity(identity, f"canonical {family} manifest")
        try:
            frame = pd.read_parquet(io.BytesIO(raw))
        except Exception as error:
            raise BenchmarkInputError(
                f"cannot parse canonical {family} manifest"
            ) from error
        required = {
            "example_id",
            "parent_id",
            "family",
            "partition",
            "outer_partition",
            "view_standard",
            "view_strict",
            "deposition_date",
        }
        if not required.issubset(frame.columns):
            raise BenchmarkInputError(f"canonical {family} manifest schema changed")
        if set(str(value) for value in frame["family"].unique()) != {family}:
            raise BenchmarkInputError(f"canonical {family} manifest family drifted")
        rows.extend(MappingProxyType(dict(row)) for row in frame.to_dict("records"))
        identities[family] = identity
    return tuple(rows), identities

def _index_audit_rows(
    rows: tuple[Mapping[str, object], ...],
) -> Mapping[str, Mapping[str, object]]:
    indexed = {str(row["example_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise BenchmarkInputError("v5 resolution audit has duplicate example IDs")
    return MappingProxyType(indexed)

def _index_canonical_rows(
    rows: tuple[Mapping[str, object], ...],
) -> Mapping[str, Mapping[str, object]]:
    indexed = {str(row["example_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise BenchmarkInputError(
            "canonical family manifests have duplicate example IDs"
        )
    return MappingProxyType(indexed)

def _join_admitted(
    manifests: object,
    audit: Mapping[str, Mapping[str, object]],
    canonical: Mapping[str, Mapping[str, object]],
) -> tuple[BenchmarkExample, ...]:
    examples = []
    for manifest in manifests:
        for row in manifest.records:
            example_id = str(row["example_id"])
            audit_row = audit.get(example_id)
            canonical_row = canonical.get(example_id)
            if audit_row is None or canonical_row is None:
                raise BenchmarkInputError(
                    f"missing authenticated source row for {example_id}"
                )
            if audit_row["mapping_sha256"] is None:
                raise BenchmarkInputError(
                    f"loader row is not role realized: {example_id}"
                )
            if str(row.get("parent_id", "")) != str(canonical_row["parent_id"]):
                raise BenchmarkInputError(
                    f"missing parent projection or parent drift for {example_id}"
                )
            if (
                str(canonical_row["partition"]) != manifest.spec.partition
                or str(audit_row["partition"]) != manifest.spec.partition
            ):
                raise BenchmarkInputError(
                    f"parent or audit partition drift for {example_id}"
                )
            if str(row["path"]) != str(audit_row["path"]):
                raise BenchmarkInputError(f"loader path drift for {example_id}")
            examples.append(
                BenchmarkExample(
                    example_id,
                    str(row["parent_id"]),
                    manifest.spec.family,
                    manifest.spec.partition,
                    str(row["path"]),
                    MappingProxyType(
                        {
                            name: str(row[name])
                            for name in ("generated", "context", "target")
                        }
                    ),
                    True,
                    None,
                )
            )
    if len(examples) != 26_309:
        raise BenchmarkInputError("v5 admitted loader count changed")
    return tuple(examples)

def _rejected_examples(
    audit: Mapping[str, Mapping[str, object]],
    canonical: Mapping[str, Mapping[str, object]],
) -> tuple[BenchmarkExample, ...]:
    rows = []
    for example_id, audit_row in audit.items():
        if audit_row["mapping_sha256"] is not None:
            continue
        canonical_row = canonical.get(example_id)
        if canonical_row is None or str(canonical_row["partition"]) != str(
            audit_row["partition"]
        ):
            raise BenchmarkInputError(
                f"rejected parent or partition drift for {example_id}"
            )
        rows.append(
            BenchmarkExample(
                example_id,
                str(canonical_row["parent_id"]),
                str(audit_row["family"]),
                str(audit_row["partition"]),
                str(audit_row["path"]),
                MappingProxyType({}),
                False,
                str(audit_row["disposition"]),
            )
        )
    return tuple(rows)

def _verify_quarantine(examples: tuple[BenchmarkExample, ...]) -> None:
    quarantined = Counter(
        item.quarantine_reason
        for item in examples
        if item.family == "antibody" and item.quarantine_reason in _QUARANTINE
    )
    if quarantined != Counter(
        {
            "no_compatible_chain": 71,
            "role_contract_error": 28,
            "no_complete_assignment": 2,
        }
    ):
        raise BenchmarkInputError("the frozen 101 antibody quarantine rows changed")

def _verify_reconciliation(examples: tuple[BenchmarkExample, ...]) -> None:
    admitted = sum(item.role_realized for item in examples)
    rejected = len(examples) - admitted
    if (admitted, rejected, len(examples)) != (26_309, 150, 26_459):
        raise BenchmarkInputError("v5 admitted/rejected reconciliation changed")

def _parents_for(
    examples: tuple[BenchmarkExample, ...],
    canonical: Mapping[str, Mapping[str, object]],
) -> tuple[BenchmarkParent, ...]:
    grouped: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        grouped[example.parent_id].append(example)
    parents = []
    for parent_id, members in grouped.items():
        rows = [canonical[item.example_id] for item in members]
        first = rows[0]
        fields = (
            "family",
            "partition",
            "outer_partition",
            "view_standard",
            "view_strict",
        )
        if any(
            tuple(_clean(row.get(field)) for field in fields)
            != tuple(_clean(first.get(field)) for field in fields)
            for row in rows[1:]
        ):
            raise BenchmarkInputError(f"parent/partition drift for {parent_id}")
        dates = sorted(
            date
            for date in (_clean(row["deposition_date"]) for row in rows)
            if date is not None
        )
        parents.append(
            BenchmarkParent(
                parent_id,
                str(first["family"]),
                str(first["partition"]),
                str(_clean(first["outer_partition"]) or ""),
                bool(first["view_standard"]),
                bool(first["view_strict"]),
                dates[0] if dates else None,
                tuple(sorted(item.example_id for item in members)),
            )
        )
    return tuple(parents)

def _clean(value: object) -> str | None:
    return None if value is None or str(value) in {"", "nan", "NaT"} else str(value)

def _counts_for(
    examples: tuple[BenchmarkExample, ...], train: Mapping[str, object]
) -> dict[str, dict[str, BenchmarkCounts]]:
    admitted = [item for item in examples if item.role_realized]
    counts: dict[str, dict[str, BenchmarkCounts]] = {family: {} for family in _FAMILIES}
    audit_counts = train["realized_split_audit"]["counts"]
    for family in _FAMILIES:
        for partition in _PARTITIONS:
            subset = [
                item
                for item in admitted
                if item.family == family and item.partition == partition
            ]
            value = BenchmarkCounts(
                len(subset), len({item.parent_id for item in subset})
            )
            expected = audit_counts.get(f"{family}/{partition}")
            if (
                not isinstance(expected, Mapping)
                or value.example_rows != expected.get("example_rows")
                or value.parent_groups != expected.get("parent_groups")
            ):
                raise BenchmarkInputError(
                    f"realized split count drift for {family}/{partition}"
                )
            counts[family][partition] = value
    return counts

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }
