
from __future__ import annotations

from dive.codirect_paths import cache_dir

import hashlib
import io
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import (
    ArtifactIdentity,
    BenchmarkContract,
    ExposureClass,
    file_identity,
)
from dive.benchmark.evidence import (
    BenchmarkEvidenceError,
    BenchmarkRun,
    complete_benchmark_run,
    write_terminal_bytes,
)
from dive.benchmark.exposure import (
    ExposureRecord,
    aggregate_exposure,
    temporal_clean_eligible,
)
from dive.benchmark.inputs import (
    BenchmarkCounts,
    BenchmarkExample,
    BenchmarkParent,
    FrozenBenchmarkInputs,
)
from dive.training.preflight import canonical_json_bytes

class BenchmarkViewError(RuntimeError):
    pass

WAIVED_STRUCTURAL_AUDIT = "not_required_by_frozen_contract"
_SCHEMA_VERSION = "dive-benchmark-views-v1"
_FAMILIES = ("binder", "ame", "antibody")
_VIEW_PARTITIONS = ("validation", "test-blind")
_QUARANTINE_REASONS = {
    "no_compatible_chain": 71,
    "role_contract_error": 28,
    "no_complete_assignment": 2,
}
_ACCEPTED_INPUT_RUN = "benchmark-inputs-20260826c"
_ACCEPTED_EXPOSURE_RUN = "benchmark-exposure-20260827f"
_ACCEPTED_INPUT_IDENTITIES = MappingProxyType(
    {
        "attempt.json": (
            "762b2e85071a3bc59ffd429fa7cd3283fe0f264d44c083bf0f8e5a89d70863bf",
            922,
        ),
        "completion.json": (
            "c23af46888908fbf81c60452c4b85016a27a6fa5f2563ad6b6926b0cf3ef34f2",
            799,
        ),
        "input_inventory.json": (
            "c612476557e3f26773e52020a24b0a103e997e46f520116f6c0c2667994d1903",
            11_898_559,
        ),
    }
)
_ACCEPTED_EXPOSURE_IDENTITIES = MappingProxyType(
    {
        "attempt.json": (
            "06fc763d2610e7fb3ca997290466b9663bc0327e188b94a5bc0e6de008b44e3c",
            1_014,
        ),
        "completion.json": (
            "5bcf4bf488b97584606ea423a6d9348c908a5c9bd4cf3ec7b2ff019acbe9d32e",
            4_039,
        ),
    }
)
_ACCEPTED_EXPOSURE_AGGREGATE = ArtifactIdentity(
    cache_dir('emergent', 'benchmark_ready', 'benchmark-exposure-20260827f', 'exposure.aggregate.jsonl'),
    "0dcb9a3541b1377b72c735d0bcddb7b3c205a8be207c5333414af87c4a52e927",
    141_801_441,
)

@dataclass(frozen=True, slots=True)
class BenchmarkViewRow:
    parent_id: str
    family: str
    partition: str
    standard: bool
    strict: bool
    temporal_clean: bool
    structural_conflict: bool
    exposure: ExposureClass
    exclusion_reasons: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class BenchmarkViewExample:
    example_id: str
    parent_id: str
    family: str
    partition: str
    standard: bool
    strict: bool
    temporal_clean: bool

@dataclass(frozen=True, slots=True)
class AntibodyQuarantineRow:
    example_id: str
    parent_id: str
    family: str
    partition: str
    quarantine_reason: str

@dataclass(frozen=True, slots=True)
class BenchmarkViewBundle:
    parents: tuple[BenchmarkViewRow, ...]
    examples: tuple[BenchmarkViewExample, ...]
    quarantine: tuple[AntibodyQuarantineRow, ...]
    partition_hash: str
    semantic_hash: str
    structural_audit: str
    min_parent_groups: int

    def parent(self, parent_id: str) -> BenchmarkViewRow:
        matches = [row for row in self.parents if row.parent_id == parent_id]
        if len(matches) != 1:
            raise BenchmarkViewError(
                f"parent {parent_id!r} is missing or duplicated in the view bundle"
            )
        return matches[0]

@dataclass(frozen=True, slots=True)
class BenchmarkViewWriteResult:
    summary: ArtifactIdentity
    parents: tuple[tuple[str, ArtifactIdentity], ...]
    examples: tuple[tuple[str, ArtifactIdentity], ...]
    quarantine: ArtifactIdentity

@dataclass
class BenchmarkViewAudits:

    structural_conflicts: set[str] = field(default_factory=set)
    exposure_by_parent: dict[str, tuple[ExposureRecord, ...]] = field(
        default_factory=dict
    )
    required_subjects: tuple[str, ...] = (
        "common_training_manifest",
        "common_cutoff_evidence",
    )
    structural_audit: str = WAIVED_STRUCTURAL_AUDIT
    min_parent_groups: int = 20
    source_input: ArtifactIdentity | None = None
    source_exposure: ArtifactIdentity | None = None

    def add_structural_conflict(self, parent_id: str) -> None:
        if type(parent_id) is not str or not parent_id:
            raise BenchmarkViewError("structural conflict parent_id must be a string")
        self.structural_conflicts.add(parent_id)

    def set_exposure(self, parent_id: str, records: Sequence[ExposureRecord]) -> None:
        if type(parent_id) is not str or not parent_id:
            raise BenchmarkViewError("exposure parent_id must be a string")
        self.exposure_by_parent[parent_id] = tuple(records)

def partition_identity(parents: Sequence[BenchmarkParent]) -> str:

    payload = {
        "parents": [
            {
                "parent_id": item.parent_id,
                "partition": item.partition,
                "outer_partition": item.outer_partition,
            }
            for item in sorted(
                parents,
                key=lambda item: (item.family, item.partition, item.parent_id),
            )
        ]
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def build_benchmark_views(inputs, audits) -> BenchmarkViewBundle:

    parents, examples = _require_inputs(inputs)
    audit = _require_audits(audits)
    realized = {item.parent_id for item in examples if item.role_realized}
    by_parent = {item.parent_id: item for item in parents}
    if len(by_parent) != len(parents):
        raise BenchmarkViewError("frozen parents contain duplicate parent_id values")
    rows = []
    for parent in parents:
        row = _view_row(parent, realized, audit)
        if row.partition != parent.partition:
            raise BenchmarkViewError(
                f"view join mutated partition for {parent.parent_id}"
            )
        rows.append(row)
    rows.sort(key=lambda item: (item.family, item.partition, item.parent_id))
    example_rows = _example_rows(examples, rows)
    quarantine = _quarantine_rows(examples)
    _assert_group_floor(rows, audit.min_parent_groups)
    observed_hash = partition_identity(parents)
    recorded_hash = getattr(inputs, "partition_hash", None)
    if isinstance(recorded_hash, str) and recorded_hash:
        if recorded_hash != observed_hash:
            raise BenchmarkViewError("partition identity drifted during the view join")
        partition_hash = recorded_hash
    else:
        partition_hash = observed_hash
    bundle = BenchmarkViewBundle(
        tuple(rows),
        example_rows,
        quarantine,
        partition_hash,
        "0" * 64,
        audit.structural_audit,
        audit.min_parent_groups,
    )
    semantic = hashlib.sha256(
        canonical_json_bytes(_semantic_mapping(bundle))
    ).hexdigest()
    object.__setattr__(bundle, "semantic_hash", semantic)
    return bundle

def render_view_artifacts(bundle: BenchmarkViewBundle) -> dict[str, bytes]:

    if not isinstance(bundle, BenchmarkViewBundle):
        raise BenchmarkViewError("render_view_artifacts requires BenchmarkViewBundle")
    artifacts: dict[str, bytes] = {}
    parquet_hashes: dict[str, dict[str, object]] = {}
    for family in _FAMILIES:
        for partition in _VIEW_PARTITIONS:
            parent_rows = [
                _parent_parquet_row(row)
                for row in bundle.parents
                if row.family == family and row.partition == partition
            ]
            example_rows = [
                _example_parquet_row(row)
                for row in bundle.examples
                if row.family == family and row.partition == partition
            ]
            parent_name = f"{family}_{partition}_parents.parquet"
            example_name = f"{family}_{partition}_examples.parquet"
            parent_bytes = _parquet_bytes(parent_rows, _parent_schema())
            example_bytes = _parquet_bytes(example_rows, _example_schema())
            artifacts[parent_name] = parent_bytes
            artifacts[example_name] = example_bytes
            parquet_hashes[parent_name] = _bytes_identity(parent_bytes)
            parquet_hashes[example_name] = _bytes_identity(example_bytes)
    quarantine_bytes = _parquet_bytes(
        [_quarantine_parquet_row(row) for row in bundle.quarantine],
        _quarantine_schema(),
    )
    artifacts["antibody_quarantine.parquet"] = quarantine_bytes
    parquet_hashes["antibody_quarantine.parquet"] = _bytes_identity(quarantine_bytes)
    artifacts["view_summary.json"] = canonical_json_bytes(
        _summary_mapping(bundle, parquet_hashes)
    )
    return artifacts

def write_benchmark_views(
    run: BenchmarkRun, bundle: BenchmarkViewBundle
) -> BenchmarkViewWriteResult:

    if not isinstance(run, BenchmarkRun):
        raise BenchmarkViewError(
            "write_benchmark_views requires a claimed BenchmarkRun"
        )
    artifacts = render_view_artifacts(bundle)
    written: dict[str, ArtifactIdentity] = {}
    for relative, content in artifacts.items():
        destination = run.evidence_dir / relative
        _write_view_bytes(destination, content, run)
        written[relative] = file_identity(destination)
    parent_identities = tuple(
        (name, identity)
        for name, identity in written.items()
        if name.endswith("_parents.parquet")
    )
    example_identities = tuple(
        (name, identity)
        for name, identity in written.items()
        if name.endswith("_examples.parquet")
    )
    return BenchmarkViewWriteResult(
        written["view_summary.json"],
        parent_identities,
        example_identities,
        written["antibody_quarantine.parquet"],
    )

def load_view_sources(
    contract: BenchmarkContract,
    *,
    input_run_id: str,
    exposure_run_id: str,
    structural_audit: str,
) -> tuple[FrozenBenchmarkInputs, BenchmarkViewAudits, dict[str, ArtifactIdentity]]:

    if not isinstance(contract, BenchmarkContract):
        raise BenchmarkViewError("load_view_sources requires BenchmarkContract")
    if input_run_id != _ACCEPTED_INPUT_RUN:
        raise BenchmarkViewError(
            f"views require accepted input run {_ACCEPTED_INPUT_RUN}"
        )
    if exposure_run_id != _ACCEPTED_EXPOSURE_RUN:
        raise BenchmarkViewError(
            f"views require accepted exposure run {_ACCEPTED_EXPOSURE_RUN}"
        )
    if structural_audit != WAIVED_STRUCTURAL_AUDIT:
        raise BenchmarkViewError(
            "views require --structural-audit "
            f"{WAIVED_STRUCTURAL_AUDIT}; Foldseek was not executed"
        )
    evidence_root = Path(contract.evidence_root) / "benchmark_ready"
    input_completion, inventory = _authenticate_input_run(evidence_root / input_run_id)
    exposure_completion, exposure_by_parent = _authenticate_exposure_run(
        evidence_root / exposure_run_id,
        required_subjects=contract.cleanliness_required_subjects,
    )
    inventory_path = evidence_root / input_run_id / "input_inventory.json"
    inputs = _inputs_from_inventory(inventory)
    inputs = _extend_with_canonical_blind(inputs)
    audits = BenchmarkViewAudits(
        required_subjects=contract.cleanliness_required_subjects,
        structural_audit=WAIVED_STRUCTURAL_AUDIT,
        min_parent_groups=contract.views.min_parent_groups,
        source_input=input_completion,
        source_exposure=exposure_completion,
        exposure_by_parent=exposure_by_parent,
    )
    identities = {
        "input_completion": input_completion,
        "exposure_completion": exposure_completion,
        "input_inventory": file_identity(inventory_path),
    }
    return inputs, audits, identities

def complete_view_run(
    run: BenchmarkRun,
    bundle: BenchmarkViewBundle,
    written: BenchmarkViewWriteResult,
    identities: Mapping[str, ArtifactIdentity],
) -> ArtifactIdentity:

    counts = _count_mapping(bundle)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "structural_audit": bundle.structural_audit,
        "partition_hash": bundle.partition_hash,
        "semantic_hash": bundle.semantic_hash,
        "input_completion": _identity_mapping(identities["input_completion"]),
        "exposure_completion": _identity_mapping(identities["exposure_completion"]),
        "view_summary": _identity_mapping(written.summary),
        "parent_parquets": {
            name: _identity_mapping(identity) for name, identity in written.parents
        },
        "example_parquets": {
            name: _identity_mapping(identity) for name, identity in written.examples
        },
        "antibody_quarantine": _identity_mapping(written.quarantine),
        "counts": counts,
        "temporal_clean_parent_count": counts["temporal_clean_parent_groups"],
        "foldseek_executed": False,
    }
    try:
        return complete_benchmark_run(run, payload)
    except BenchmarkEvidenceError as error:
        raise BenchmarkViewError(str(error)) from error

def _require_inputs(
    inputs,
) -> tuple[tuple[BenchmarkParent, ...], tuple[BenchmarkExample, ...]]:
    parents = getattr(inputs, "parents", None)
    examples = getattr(inputs, "examples", None)
    if not isinstance(parents, Sequence) or not isinstance(examples, Sequence):
        raise BenchmarkViewError(
            "build_benchmark_views requires frozen parents and examples"
        )
    if not all(isinstance(item, BenchmarkParent) for item in parents):
        raise BenchmarkViewError("view inputs contain a non-parent row")
    if not all(isinstance(item, BenchmarkExample) for item in examples):
        raise BenchmarkViewError("view inputs contain a non-example row")
    return tuple(parents), tuple(examples)

def _require_audits(audits) -> BenchmarkViewAudits:
    if isinstance(audits, BenchmarkViewAudits):
        return audits
    raise BenchmarkViewError("build_benchmark_views requires BenchmarkViewAudits")

def _view_row(
    parent: BenchmarkParent,
    realized: set[str],
    audit: BenchmarkViewAudits,
) -> BenchmarkViewRow:
    has_realized = parent.parent_id in realized

    membership_strict = parent.view_strict or (
        parent.partition == "validation" and has_realized
    )
    membership_standard = parent.view_standard or membership_strict
    structural_conflict = parent.parent_id in audit.structural_conflicts
    standard = membership_standard
    strict = membership_strict and not structural_conflict
    records = audit.exposure_by_parent.get(parent.parent_id, ())
    exposure, temporal_ok = _exposure_state(records, audit.required_subjects)
    temporal_clean = strict and temporal_ok
    reasons = []
    if not standard:
        reasons.append("not_standard")
    if not membership_strict:
        reasons.append("not_strict")
    if membership_strict and structural_conflict:
        reasons.append("structural_conflict")
    if strict and not temporal_clean:
        if exposure is ExposureClass.KNOWN_EXPOSED:
            reasons.append("known_exposed")
        else:
            reasons.append("unknown_exposure")
    return BenchmarkViewRow(
        parent.parent_id,
        parent.family,
        parent.partition,
        standard,
        strict,
        temporal_clean,
        structural_conflict,
        exposure,
        tuple(reasons),
    )

def _exposure_state(
    records: Sequence[ExposureRecord], required_subjects: Sequence[str]
) -> tuple[ExposureClass, bool]:
    if not required_subjects:
        raise BenchmarkViewError("temporal-clean requires a non-empty subject set")
    by_subject = {record.subject: record for record in records}
    required = tuple(
        by_subject[name] for name in required_subjects if name in by_subject
    )
    if len(required) != len(required_subjects):
        return ExposureClass.UNKNOWN, False
    return aggregate_exposure(required), temporal_clean_eligible(required)

def _example_rows(
    examples: Sequence[BenchmarkExample], parents: Sequence[BenchmarkViewRow]
) -> tuple[BenchmarkViewExample, ...]:
    by_parent = {row.parent_id: row for row in parents}
    rows = []
    for example in examples:
        if not example.role_realized:
            continue
        parent = by_parent.get(example.parent_id)
        if parent is None:
            raise BenchmarkViewError(
                f"realized example {example.example_id} has no parent view row"
            )
        if example.partition != parent.partition:
            raise BenchmarkViewError(
                f"example {example.example_id} partition drifted from its parent"
            )
        rows.append(
            BenchmarkViewExample(
                example.example_id,
                example.parent_id,
                example.family,
                example.partition,
                parent.standard,
                parent.strict,
                parent.temporal_clean,
            )
        )
    rows.sort(
        key=lambda item: (
            item.family,
            item.partition,
            item.parent_id,
            item.example_id,
        )
    )
    return tuple(rows)

def _quarantine_rows(
    examples: Sequence[BenchmarkExample],
) -> tuple[AntibodyQuarantineRow, ...]:
    rows = [
        AntibodyQuarantineRow(
            item.example_id,
            item.parent_id,
            item.family,
            item.partition,
            item.quarantine_reason,
        )
        for item in examples
        if item.family == "antibody" and item.quarantine_reason
    ]
    rows.sort(key=lambda item: (item.parent_id, item.example_id))
    counted = Counter(item.quarantine_reason for item in rows)
    if sum(counted.values()) == 101 and counted != Counter(_QUARANTINE_REASONS):
        raise BenchmarkViewError("the frozen 101 antibody quarantine rows changed")
    return tuple(rows)

def _assert_group_floor(rows: Sequence[BenchmarkViewRow], minimum: int) -> None:
    if type(minimum) is not int or minimum < 1:
        raise BenchmarkViewError("min_parent_groups must be a positive integer")
    for family in _FAMILIES:
        for partition in _VIEW_PARTITIONS:
            count = len(
                {
                    row.parent_id
                    for row in rows
                    if row.family == family
                    and row.partition == partition
                    and row.strict
                }
            )
            if count < minimum:
                raise BenchmarkViewError(
                    f"{family} {partition} has {count} parent groups; "
                    f"floor is {minimum}"
                )

def _semantic_mapping(bundle: BenchmarkViewBundle) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "structural_audit": bundle.structural_audit,
        "partition_hash": bundle.partition_hash,
        "min_parent_groups": bundle.min_parent_groups,
        "parents": [_parent_mapping(row) for row in bundle.parents],
        "examples": [_example_mapping(row) for row in bundle.examples],
        "quarantine": [_quarantine_mapping(row) for row in bundle.quarantine],
    }

def _summary_mapping(
    bundle: BenchmarkViewBundle, parquet_hashes: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    counts = _count_mapping(bundle)
    reasons = Counter(
        reason for row in bundle.parents for reason in row.exclusion_reasons
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "structural_audit": bundle.structural_audit,
        "foldseek_executed": False,
        "partition_hash": bundle.partition_hash,
        "semantic_hash": bundle.semantic_hash,
        "min_parent_groups": bundle.min_parent_groups,
        "counts": counts,
        "exclusion_reasons": {
            name: int(value) for name, value in sorted(reasons.items())
        },
        "artifacts": {
            name: dict(identity) for name, identity in sorted(parquet_hashes.items())
        },
        "antibody_quarantine": {
            "rows": len(bundle.quarantine),
            "reasons": dict(
                Counter(row.quarantine_reason for row in bundle.quarantine)
            ),
        },
        "temporal_clean": {
            "parent_groups": counts["temporal_clean_parent_groups"],
            "empty_because": (
                "unknown_exposure"
                if counts["temporal_clean_parent_groups"] == 0
                else None
            ),
        },
    }

def _count_mapping(bundle: BenchmarkViewBundle) -> dict[str, object]:
    by_family: dict[str, dict[str, dict[str, int]]] = {}
    for family in _FAMILIES:
        by_family[family] = {}
        for partition in _VIEW_PARTITIONS:
            subset = [
                row
                for row in bundle.parents
                if row.family == family and row.partition == partition
            ]
            by_family[family][partition] = {
                "parent_groups": len(subset),
                "standard": sum(row.standard for row in subset),
                "strict": sum(row.strict for row in subset),
                "temporal_clean": sum(row.temporal_clean for row in subset),
            }
    return {
        "parent_rows": len(bundle.parents),
        "example_rows": len(bundle.examples),
        "quarantine_rows": len(bundle.quarantine),
        "standard_parent_groups": sum(row.standard for row in bundle.parents),
        "strict_parent_groups": sum(row.strict for row in bundle.parents),
        "temporal_clean_parent_groups": sum(
            row.temporal_clean for row in bundle.parents
        ),
        "families": by_family,
    }

def _parent_mapping(row: BenchmarkViewRow) -> dict[str, object]:
    return {
        "parent_id": row.parent_id,
        "family": row.family,
        "partition": row.partition,
        "standard": row.standard,
        "strict": row.strict,
        "temporal_clean": row.temporal_clean,
        "structural_conflict": row.structural_conflict,
        "exposure": row.exposure.value,
        "exclusion_reasons": list(row.exclusion_reasons),
    }

def _example_mapping(row: BenchmarkViewExample) -> dict[str, object]:
    return {
        "example_id": row.example_id,
        "parent_id": row.parent_id,
        "family": row.family,
        "partition": row.partition,
        "standard": row.standard,
        "strict": row.strict,
        "temporal_clean": row.temporal_clean,
    }

def _quarantine_mapping(row: AntibodyQuarantineRow) -> dict[str, object]:
    return {
        "example_id": row.example_id,
        "parent_id": row.parent_id,
        "family": row.family,
        "partition": row.partition,
        "quarantine_reason": row.quarantine_reason,
    }

def _parent_parquet_row(row: BenchmarkViewRow) -> dict[str, object]:
    return _parent_mapping(row)

def _example_parquet_row(row: BenchmarkViewExample) -> dict[str, object]:
    return _example_mapping(row)

def _quarantine_parquet_row(row: AntibodyQuarantineRow) -> dict[str, object]:
    return _quarantine_mapping(row)

def _parent_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("parent_id", pa.string()),
            pa.field("family", pa.string()),
            pa.field("partition", pa.string()),
            pa.field("standard", pa.bool_()),
            pa.field("strict", pa.bool_()),
            pa.field("temporal_clean", pa.bool_()),
            pa.field("structural_conflict", pa.bool_()),
            pa.field("exposure", pa.string()),
            pa.field("exclusion_reasons", pa.list_(pa.string())),
        ]
    )

def _example_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("example_id", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("family", pa.string()),
            pa.field("partition", pa.string()),
            pa.field("standard", pa.bool_()),
            pa.field("strict", pa.bool_()),
            pa.field("temporal_clean", pa.bool_()),
        ]
    )

def _quarantine_schema():
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("example_id", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("family", pa.string()),
            pa.field("partition", pa.string()),
            pa.field("quarantine_reason", pa.string()),
        ]
    )

def _parquet_bytes(rows: Sequence[Mapping[str, object]], schema) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(list(rows), schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="none",
        use_dictionary=False,
        write_statistics=False,
        store_schema=True,
        version="2.6",
        coerce_timestamps="us",
        allow_truncated_timestamps=False,
    )
    return sink.getvalue().to_pybytes()

def _bytes_identity(raw: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }

def _write_view_bytes(path: Path, content: bytes, run: BenchmarkRun) -> None:
    evidence_root = run.evidence_dir.parents[1]
    bulk_root = run.bulk_dir.parents[1]
    try:
        write_terminal_bytes(path, content, evidence_root, bulk_root)
    except (BenchmarkEvidenceError, OSError) as error:
        raise BenchmarkViewError(str(error)) from error

def _authenticate_input_run(
    source: Path,
) -> tuple[ArtifactIdentity, Mapping[str, object]]:
    snapshots = {
        name: _read_retained(ArtifactIdentity(str(source / name), sha256, size), name)
        for name, (sha256, size) in _ACCEPTED_INPUT_IDENTITIES.items()
    }
    attempt = _canonical_object(snapshots["attempt.json"], "input attempt")
    completion = _canonical_object(snapshots["completion.json"], "input completion")
    inventory = _canonical_object(snapshots["input_inventory.json"], "input inventory")
    if (
        attempt.get("run_id") != _ACCEPTED_INPUT_RUN
        or attempt.get("status") != "BUILDING"
        or completion.get("run_id") != _ACCEPTED_INPUT_RUN
        or completion.get("status") != "COMPLETE"
    ):
        raise BenchmarkViewError("accepted input run is not authentic")
    inventory_identity = ArtifactIdentity(
        str((source / "input_inventory.json").resolve()),
        _ACCEPTED_INPUT_IDENTITIES["input_inventory.json"][0],
        _ACCEPTED_INPUT_IDENTITIES["input_inventory.json"][1],
    )
    payload = completion.get("payload")
    if not isinstance(payload, Mapping) or payload.get(
        "inventory"
    ) != _identity_mapping(inventory_identity):
        raise BenchmarkViewError(
            "accepted input completion does not authenticate its inventory"
        )
    return (
        ArtifactIdentity(
            str((source / "completion.json").resolve()),
            _ACCEPTED_INPUT_IDENTITIES["completion.json"][0],
            _ACCEPTED_INPUT_IDENTITIES["completion.json"][1],
        ),
        inventory,
    )

def _authenticate_exposure_run(
    source: Path, *, required_subjects: Sequence[str]
) -> tuple[ArtifactIdentity, dict[str, tuple[ExposureRecord, ...]]]:
    snapshots = {
        name: _read_retained(ArtifactIdentity(str(source / name), sha256, size), name)
        for name, (sha256, size) in _ACCEPTED_EXPOSURE_IDENTITIES.items()
    }
    attempt = _canonical_object(snapshots["attempt.json"], "exposure attempt")
    completion = _canonical_object(snapshots["completion.json"], "exposure completion")
    if (
        attempt.get("run_id") != _ACCEPTED_EXPOSURE_RUN
        or completion.get("run_id") != _ACCEPTED_EXPOSURE_RUN
        or completion.get("status") != "COMPLETE"
    ):
        raise BenchmarkViewError("accepted exposure run is not authentic")
    payload = completion.get("payload")
    if not isinstance(payload, Mapping):
        raise BenchmarkViewError("accepted exposure completion payload is invalid")
    if payload.get("temporal_clean_parent_count") != 0:
        raise BenchmarkViewError(
            "accepted exposure run is not empty for temporal-clean"
        )
    aggregate = payload.get("aggregate_records")
    if aggregate != _identity_mapping(_ACCEPTED_EXPOSURE_AGGREGATE):
        raise BenchmarkViewError("accepted exposure aggregate identity drifted")
    records = _load_exposure_records(
        _ACCEPTED_EXPOSURE_AGGREGATE,
        required_subjects=required_subjects,
    )
    return (
        ArtifactIdentity(
            str((source / "completion.json").resolve()),
            _ACCEPTED_EXPOSURE_IDENTITIES["completion.json"][0],
            _ACCEPTED_EXPOSURE_IDENTITIES["completion.json"][1],
        ),
        records,
    )

def _load_exposure_records(
    identity: ArtifactIdentity, *, required_subjects: Sequence[str]
) -> dict[str, tuple[ExposureRecord, ...]]:
    raw = _read_retained(identity, "exposure aggregate")
    records: dict[str, tuple[ExposureRecord, ...]] = {}
    for line in raw.splitlines():
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise BenchmarkViewError("exposure aggregate row is not an object")
        parent_id = row.get("parent_id")
        subjects = row.get("subjects")
        if type(parent_id) is not str or not isinstance(subjects, Mapping):
            raise BenchmarkViewError("exposure aggregate row schema changed")
        parent_records = []
        for subject in required_subjects:
            item = subjects.get(subject)
            if not isinstance(item, Mapping):
                continue
            classification = item.get("classification")
            reason = item.get("reason_code")
            evidence = item.get("evidence")
            if classification not in {item.value for item in ExposureClass}:
                raise BenchmarkViewError("exposure classification drifted")
            identities = ()
            if isinstance(evidence, list):
                identities = tuple(
                    ArtifactIdentity(
                        str(entry["path"]),
                        str(entry["sha256"]),
                        int(entry["size_bytes"]),
                    )
                    for entry in evidence
                    if isinstance(entry, Mapping)
                )
            parent_records.append(
                ExposureRecord(
                    parent_id,
                    subject,
                    ExposureClass(classification),
                    str(reason or "unknown"),
                    identities,
                )
            )
        records[parent_id] = tuple(parent_records)
    if len(records) != 15_504:
        raise BenchmarkViewError("accepted exposure parent count drifted")
    return records

def _inputs_from_inventory(payload: Mapping[str, object]) -> FrozenBenchmarkInputs:
    parents_raw = payload.get("parents")
    examples_raw = payload.get("examples")
    counts_raw = payload.get("counts")
    identities_raw = payload.get("inputs")
    if not isinstance(parents_raw, list) or not isinstance(examples_raw, list):
        raise BenchmarkViewError("input inventory is missing parents or examples")
    if not isinstance(counts_raw, Mapping) or not isinstance(identities_raw, Mapping):
        raise BenchmarkViewError("input inventory is missing counts or identities")
    parents = tuple(_parent_from_mapping(item) for item in parents_raw)
    examples = tuple(_example_from_mapping(item) for item in examples_raw)
    counts: dict[str, dict[str, BenchmarkCounts]] = {}
    for family, partitions in counts_raw.items():
        if not isinstance(partitions, Mapping):
            raise BenchmarkViewError("input inventory counts schema changed")
        counts[str(family)] = {
            str(partition): BenchmarkCounts(
                int(value["example_rows"]), int(value["parent_groups"])
            )
            for partition, value in partitions.items()
            if isinstance(value, Mapping)
        }
    identities = {
        str(name): ArtifactIdentity(
            str(item["path"]), str(item["sha256"]), int(item["size_bytes"])
        )
        for name, item in identities_raw.items()
        if isinstance(item, Mapping)
    }
    projection = payload.get("projection")
    if not isinstance(projection, Mapping):
        raise BenchmarkViewError("input inventory projection is missing")
    return FrozenBenchmarkInputs(
        str(projection.get("identity", "")),
        int(projection.get("row_count", 0)),
        (),
        MappingProxyType(identities),
        parents,
        examples,
        MappingProxyType(
            {family: MappingProxyType(values) for family, values in counts.items()}
        ),
    )

def _extend_with_canonical_blind(
    inputs: FrozenBenchmarkInputs,
) -> FrozenBenchmarkInputs:

    import pandas as pd

    existing = {item.parent_id for item in inputs.parents}
    extra_parents: list[BenchmarkParent] = []
    extra_examples: list[BenchmarkExample] = []
    for family in _FAMILIES:
        identity = inputs.input_identities.get(f"canonical_manifest:{family}")
        if identity is None:
            raise BenchmarkViewError(f"canonical {family} manifest is missing")
        raw = _read_retained(identity, f"canonical {family} manifest")
        try:
            frame = pd.read_parquet(io.BytesIO(raw))
        except Exception as error:
            raise BenchmarkViewError(
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
            raise BenchmarkViewError(f"canonical {family} manifest schema changed")
        blind = frame[frame["partition"].astype(str) == "test-blind"]
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in blind.to_dict("records"):
            parent_id = str(row["parent_id"])
            grouped.setdefault(parent_id, []).append(row)
        for parent_id, rows in sorted(grouped.items()):
            if parent_id in existing:
                continue
            first = rows[0]
            example_ids = tuple(sorted(str(item["example_id"]) for item in rows))
            extra_parents.append(
                BenchmarkParent(
                    parent_id,
                    str(first["family"]),
                    "test-blind",
                    _clean(first.get("outer_partition")) or "",
                    bool(first["view_standard"]),
                    bool(first["view_strict"]),
                    _clean(first.get("deposition_date")),
                    example_ids,
                )
            )
            for item in rows:
                extra_examples.append(
                    BenchmarkExample(
                        str(item["example_id"]),
                        parent_id,
                        str(item["family"]),
                        "test-blind",
                        "",
                        MappingProxyType({}),
                        True,
                        None,
                    )
                )
    return FrozenBenchmarkInputs(
        inputs.projection_identity,
        inputs.projection_row_count,
        inputs.loader_manifest_identities,
        inputs.input_identities,
        tuple(
            sorted(
                inputs.parents + tuple(extra_parents), key=lambda item: item.parent_id
            )
        ),
        tuple(
            sorted(
                inputs.examples + tuple(extra_examples),
                key=lambda item: item.example_id,
            )
        ),
        inputs.counts,
    )

def _parent_from_mapping(value: object) -> BenchmarkParent:
    if not isinstance(value, Mapping):
        raise BenchmarkViewError("inventory parent is not an object")
    example_ids = value.get("example_ids")
    if not isinstance(example_ids, list) or not all(
        type(item) is str for item in example_ids
    ):
        raise BenchmarkViewError("inventory parent example ids changed")
    return BenchmarkParent(
        str(value["parent_id"]),
        str(value["family"]),
        str(value["partition"]),
        str(value.get("outer_partition") or ""),
        bool(value["view_standard"]),
        bool(value["view_strict"]),
        _clean(value.get("deposition_date")),
        tuple(example_ids),
    )

def _example_from_mapping(value: object) -> BenchmarkExample:
    if not isinstance(value, Mapping):
        raise BenchmarkViewError("inventory example is not an object")
    roles = value.get("roles") or {}
    if not isinstance(roles, Mapping):
        raise BenchmarkViewError("inventory example roles changed")
    return BenchmarkExample(
        str(value["example_id"]),
        str(value["parent_id"]),
        str(value["family"]),
        str(value["partition"]),
        str(value.get("path") or ""),
        MappingProxyType({str(key): str(item) for key, item in roles.items()}),
        bool(value["role_realized"]),
        _clean(value.get("quarantine_reason")),
    )

def _clean(value: object) -> str | None:
    if value is None or str(value) in {"", "nan", "NaT"}:
        return None
    return str(value)

def _read_retained(expected: ArtifactIdentity, label: str) -> bytes:

    path = Path(expected.path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BenchmarkViewError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not os.path.stat.S_ISREG(before.st_mode):
            raise BenchmarkViewError(f"{label} must be a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or hashlib.sha256(raw).hexdigest() != expected.sha256
        or len(raw) != expected.size_bytes
    ):
        raise BenchmarkViewError(f"{label} identity drifted")
    return raw

def _canonical_object(raw: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkViewError(f"{label} is not JSON") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise BenchmarkViewError(f"{label} is not canonical JSON")
    return value

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }
