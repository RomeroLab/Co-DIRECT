
from __future__ import annotations

from dive.codirect_paths import joined

import base64
import copy
import csv
import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import ArtifactIdentity, file_identity
from dive.benchmark.inputs import BenchmarkExample, FrozenBenchmarkInputs
from dive.provenance import ProvenanceError, verify_git_checkout
from dive.signed_value.roots import EMERGENT_UPSTREAM_COMMIT, EMERGENT_UPSTREAM_ROOT
from dive.training.parent_projection import PRODUCTION_CONTRACT_V2
from dive.training.preflight import canonical_json_bytes

class StructureInventoryError(RuntimeError):

    def __init__(
        self, message: str, *, reason_code: str = "component_mismatch"
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code

@dataclass(frozen=True, slots=True)
class StructureRecord:
    foldseek_key: str
    family: str
    partition: str
    source_id: str
    example_id: str
    parent_id: str
    component_id: str
    chain_id: str
    role: str
    source: ArtifactIdentity
    source_provenance: tuple[ArtifactIdentity, ...]
    materialized: ArtifactIdentity
    materialization_command: tuple[str, ...]
    code_identity: ArtifactIdentity
    failure_reason: str | None = None

@dataclass(frozen=True, slots=True)
class StructureInventory:
    queries: tuple[StructureRecord, ...]
    targets: tuple[StructureRecord, ...]
    query_dir: Path
    target_dir: Path
    inventory: ArtifactIdentity
    completion: ArtifactIdentity
    expected_query_parents: tuple[str, ...]
    expected_target_parents: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class _Candidate:
    family: str
    partition: str
    source_id: str
    example_id: str
    parent_id: str
    path: Path
    roles: Mapping[str, str]
    component_namespace: str
    provenance: tuple[ArtifactIdentity, ...] = ()
    component_roles: Mapping[str, str] | None = None
    excluded_components: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class _ParsedChain:
    author_chain_id: str
    payload: bytes

class _ParsedChains(dict[str, _ParsedChain]):

    def __init__(
        self,
        values: Mapping[str, _ParsedChain],
        *,
        nonprotein_authors: frozenset[str],
        component_namespace: str,
        model_serial: int,
    ) -> None:
        super().__init__(values)
        self.nonprotein_authors = nonprotein_authors
        self.component_namespace = component_namespace
        self.model_serial = model_serial

@dataclass(frozen=True, slots=True)
class _LegacySource:
    path: Path
    provenance: tuple[ArtifactIdentity, ...]
    component_roles: Mapping[str, str] | None = None

@dataclass(frozen=True, slots=True)
class _SabdabRoleTable:
    identity: ArtifactIdentity
    source_identity: str
    rows: Mapping[str, Mapping[str, str]]
    content: bytes

@dataclass(frozen=True, slots=True)
class _LegacyResolver:
    registry_identity: ArtifactIdentity
    dictionary_identities: Mapping[str, ArtifactIdentity]
    configs: Mapping[str, Mapping[str, object]]
    registered: tuple[Mapping[str, str], ...]

@dataclass(frozen=True, slots=True)
class _GitSourceSpec:
    relative_path: Path
    sha256: str
    size_bytes: int
    blob_oid: str

_FAMILIES = ("binder", "ame", "antibody")
_QUERY_PARTITIONS = frozenset({"validation", "test-blind"})
_TARGET_PARTITIONS = frozenset({"train", "legacy-dev"})
_BULK_ROOT = Path(joined('CODIRECT_STRUCTURE_ROOT'))
_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_ROOT = EMERGENT_UPSTREAM_ROOT
_UPSTREAM_COMMIT = EMERGENT_UPSTREAM_COMMIT
_LEGACY_REGISTRY_RELATIVE = Path("assets/emergent/legacy-dev-registry.csv")
_CODE_RELATIVE = Path("src/dive/benchmark/structures.py")
_SABDAB_TABLE = _BULK_ROOT / "sources/sabdab2-v0.1.0/splits_final/abag_split.csv"
_CODE_PATH = Path(__file__).resolve()
_LEGACY_REGISTRY_SPEC = _GitSourceSpec(
    _LEGACY_REGISTRY_RELATIVE,
    "3331732cad94d8bc2e55306e1d454bd759f63a1029a1940f33694df56d4c521a",
    23_049,
    "4c61406a04703607d19d3747ad5da7aa9c4973fa",
)
_LEGACY_DICTIONARY_SPECS = MappingProxyType(
    {
        "binder": (
            _GitSourceSpec(
                Path("configs/targets/targets_dict.yaml"),
                "386d7d3bb5409becc22344b094872ace64ed1e8185834f0b41ad7d224577c991",
                11_592,
                "77bfdbe42ad0fd9a47859392a99d79d6fd258658",
            ),
            "target_dict_cfg",
        ),
        "ame": (
            _GitSourceSpec(
                Path("configs/design_tasks/ame_dict_v2.yaml"),
                "e606bee37a53154acdaeef5b41fad662540b477db42654aa65887b3f1ae4aae3",
                16_323,
                "c0c623168f1131db84c3e56cad68e826c660120b",
            ),
            "motif_target_dict_cfg",
        ),
    }
)
_MATERIALIZATION_COMMAND = (
    "dive.benchmark.structures",
    "canonical-pdb-v1",
    "source-model-serial-1",
    "preserve-author-residue-identity",
    "stable-residue-and-atom-order",
)

_TEST_SOURCE_PATHS: Mapping[str, Path] | None = None
_TEST_CANONICAL_ROLES: Mapping[str, Mapping[str, str]] | None = None

def encode_structure_key(family: str, parent_id: str, chain_id: str) -> str:

    values = (family, parent_id, chain_id)
    if family not in _FAMILIES or not all(
        type(value) is str and value for value in values
    ):
        raise StructureInventoryError(
            "structure key fields must be non-empty frozen strings"
        )
    raw = json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return "dive1-" + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

def decode_structure_key(key: str) -> tuple[str, str, str]:

    if type(key) is not str or not key.startswith("dive1-"):
        raise StructureInventoryError("invalid Foldseek structure key")
    payload = key.removeprefix("dive1-")
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        value = json.loads(raw)
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise StructureInventoryError("invalid Foldseek structure key") from error
    if (
        not isinstance(value, list)
        or len(value) != 3
        or value[0] not in _FAMILIES
        or not all(type(item) is str and item for item in value)
        or encode_structure_key(*value) != key
    ):
        raise StructureInventoryError("invalid Foldseek structure key")
    return value[0], value[1], value[2]

def materialize_structure_inventory(
    inputs: FrozenBenchmarkInputs, run: object
) -> StructureInventory:

    if not isinstance(inputs, FrozenBenchmarkInputs):
        raise StructureInventoryError("materialization requires FrozenBenchmarkInputs")
    evidence_dir = Path(getattr(run, "evidence_dir", ""))
    bulk_dir = Path(getattr(run, "bulk_dir", ""))
    if not evidence_dir.is_dir() or not bulk_dir.is_dir():
        raise StructureInventoryError(
            "materialization requires a claimed benchmark run"
        )
    query_dir = bulk_dir / "queries"
    target_dir = bulk_dir / "targets"
    stage = "destination_claim"
    try:
        _create_new_directory(query_dir)
        _create_new_directory(target_dir)
        stage = "candidate_resolution"
        query_candidates, target_candidates = _candidate_universes(inputs)
        expected_queries = tuple(sorted(query_candidates))
        expected_targets = tuple(sorted(target_candidates))
        overlap = sorted(set(expected_queries) & set(expected_targets))
        if overlap:
            raise StructureInventoryError(
                f"parent groups span query and target universes: {overlap[:3]}"
            )
        stage = "source_preflight"
        _preflight_candidate_sources(query_candidates, target_candidates)
        stage = "materialization"
        code_identity = _canonical_code_identity()
        queries = _materialize_scope(
            query_candidates, query_dir, "query", code_identity
        )
        targets = _materialize_scope(
            target_candidates, target_dir, "target", code_identity
        )
        stage = "inventory_write"
        inventory_path = evidence_dir / "structure_inventory.jsonl"
        ordered = tuple(("query", record) for record in queries) + tuple(
            ("target", record) for record in targets
        )
        _write_jsonl_create_new(
            inventory_path,
            tuple(
                {"scope": scope, **_record_mapping(record)} for scope, record in ordered
            ),
        )
        inventory_identity = file_identity(inventory_path)
        provisional = StructureInventory(
            queries,
            targets,
            query_dir,
            target_dir,
            inventory_identity,
            ArtifactIdentity("", "0" * 64, 0),
            expected_queries,
            expected_targets,
        )
        stage = "independent_verification"
        summary = verify_structure_inventory(inputs, provisional)
        stage = "completion_publication"
        completion_path = evidence_dir / "structure_inventory.completion.json"
        completion_payload = {
            "schema_version": "dive-benchmark-structure-inventory-completion-v1",
            "status": "COMPLETE",
            "inventory": _identity_mapping(inventory_identity),
            "query_records": len(queries),
            "target_records": len(targets),
            "query_parent_groups": len(expected_queries),
            "target_parent_groups": len(expected_targets),
            "missing_parent_groups": summary["missing_parent_groups"],
            "extra_parent_groups": summary["extra_parent_groups"],
            "ambiguous_keys": summary["ambiguous_keys"],
            "code_identity": _identity_mapping(code_identity),
        }
        from dive.benchmark.evidence import _write_terminal_json

        _write_terminal_json(
            completion_path,
            completion_payload,
            evidence_dir.parents[1],
            bulk_dir.parents[1],
        )
    except Exception as error:
        if isinstance(error, StructureInventoryError):
            typed = error
            if stage in {
                "inventory_write",
                "independent_verification",
                "completion_publication",
            }:
                typed = StructureInventoryError(
                    f"cannot create or verify structure inventory: {error}",
                    reason_code=(
                        "source_drift"
                        if error.reason_code == "source_drift"
                        else "verification_or_write_error"
                    ),
                )
        else:
            typed = StructureInventoryError(
                f"{stage} failed: {type(error).__name__}: {error}",
                reason_code="internal_error",
            )
        _record_materialization_failure(evidence_dir, bulk_dir, run, typed, stage=stage)
        raise typed from error
    return StructureInventory(
        queries,
        targets,
        query_dir,
        target_dir,
        inventory_identity,
        file_identity(completion_path),
        expected_queries,
        expected_targets,
    )

def _record_materialization_failure(
    evidence_dir: Path,
    bulk_dir: Path,
    run: object,
    error: StructureInventoryError,
    *,
    stage: str,
) -> None:
    from dive.benchmark.evidence import _write_terminal_json

    payload = {
        "schema_version": "dive-benchmark-structure-inventory-failure-v1",
        "status": "FAILED",
        "run_id": str(getattr(run, "run_id", "")),
        "stage": stage,
        "reason_code": error.reason_code,
        "detail": str(error),
        "code_identity": _identity_mapping(_canonical_code_identity()),
    }
    try:
        _write_terminal_json(
            evidence_dir / "structure_inventory.failure.json",
            payload,
            evidence_dir.parents[1],
            bulk_dir.parents[1],
        )
    except Exception as write_error:
        raise StructureInventoryError(
            f"{error}; additionally could not record terminal failure: {write_error}"
        ) from write_error

def verify_structure_inventory(
    inputs: FrozenBenchmarkInputs, inventory: StructureInventory
) -> Mapping[str, int]:

    if not isinstance(inputs, FrozenBenchmarkInputs) or not isinstance(
        inventory, StructureInventory
    ):
        raise StructureInventoryError(
            "verification requires frozen inputs and inventory"
        )
    observed_inventory = file_identity(Path(inventory.inventory.path))
    if observed_inventory != inventory.inventory:
        raise StructureInventoryError("structure inventory identity drifted")
    rows = _read_jsonl(Path(inventory.inventory.path))
    expected_records = tuple(inventory.queries) + tuple(inventory.targets)
    if len(rows) != len(expected_records):
        raise StructureInventoryError("structure inventory record count drifted")
    independent_query_candidates, independent_target_candidates = _candidate_universes(
        inputs
    )
    independent_candidates = {
        (item.example_id, str(item.path.resolve())): item
        for groups in (independent_query_candidates, independent_target_candidates)
        for candidates in groups.values()
        for item in candidates
    }
    observed_code_identity = _canonical_code_identity()
    reconciled_sources: dict[
        tuple[str, str], tuple[tuple[str, str, str, bytes], ...]
    ] = {}
    observed_components: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(
        set
    )
    by_key: dict[str, tuple[str, StructureRecord]] = {}
    for row, expected in zip(rows, expected_records, strict=True):
        scope = "query" if expected.partition in _QUERY_PARTITIONS else "target"
        if row != {"scope": scope, **_record_mapping(expected)}:
            raise StructureInventoryError("structure inventory row drifted")
        if expected.foldseek_key in by_key:
            raise StructureInventoryError(
                f"ambiguous duplicate key {expected.foldseek_key}"
            )
        by_key[expected.foldseek_key] = (scope, expected)
        if decode_structure_key(expected.foldseek_key) != (
            expected.family,
            expected.parent_id,
            expected.chain_id,
        ):
            raise StructureInventoryError(
                "materialized key does not decode to its parent"
            )
        if expected.code_identity != observed_code_identity:
            raise StructureInventoryError(
                f"materialization code identity drifted for {expected.example_id}",
                reason_code="source_drift",
            )
        if file_identity(Path(expected.source.path)) != expected.source:
            raise StructureInventoryError(
                f"source identity drifted for {expected.example_id}",
                reason_code="source_drift",
            )
        if file_identity(Path(expected.materialized.path)) != expected.materialized:
            raise StructureInventoryError(
                f"materialized identity drifted for {expected.foldseek_key}"
            )
        _verify_one_chain_pdb(Path(expected.materialized.path))
        candidate_key = (expected.example_id, str(Path(expected.source.path).resolve()))
        candidate = independent_candidates.get(candidate_key)
        if candidate is None:
            raise StructureInventoryError(
                f"cannot independently recover candidate {candidate_key}"
            )
        if candidate.provenance != expected.source_provenance:
            raise StructureInventoryError(
                f"source provenance drifted for {expected.example_id}",
                reason_code="source_drift",
            )
        if candidate_key not in reconciled_sources:
            _source_identity, components = _resolve_candidate_components(candidate)
            reconciled_sources[candidate_key] = components
        expected_components = {
            (component, chain, role)
            for component, chain, role, _payload in reconciled_sources[candidate_key]
        }
        if (
            expected.component_id,
            expected.chain_id,
            expected.role,
        ) not in expected_components:
            raise StructureInventoryError(
                f"component mapping drifted for {expected.foldseek_key}"
            )
        observed_components[candidate_key].add(
            (expected.component_id, expected.chain_id, expected.role)
        )

    for candidate_key, expected_rows in reconciled_sources.items():
        expected_components = {
            (component, chain, role)
            for component, chain, role, _payload in expected_rows
        }
        if observed_components[candidate_key] != expected_components:
            raise StructureInventoryError(
                f"declared component record reconciliation failed for {candidate_key}"
            )

    query_parents = {record.parent_id for record in inventory.queries}
    target_parents = {record.parent_id for record in inventory.targets}
    independently_expected_query, independently_expected_target = (
        _independent_expected_parent_sets(inputs)
    )
    expected_query = set(independently_expected_query)
    expected_target = set(independently_expected_target)
    if expected_query != set(
        inventory.expected_query_parents
    ) or expected_target != set(inventory.expected_target_parents):
        raise StructureInventoryError(
            "stored expected parent sets do not match frozen inputs"
        )
    missing = len(expected_query - query_parents) + len(
        expected_target - target_parents
    )
    extra = len(query_parents - expected_query) + len(target_parents - expected_target)
    if missing or extra:
        raise StructureInventoryError(
            f"structure parent reconciliation failed: missing={missing}, extra={extra}"
        )
    if {record.partition for record in inventory.queries} - _QUERY_PARTITIONS:
        raise StructureInventoryError("query inventory contains a forbidden partition")
    if {record.partition for record in inventory.targets} - _TARGET_PARTITIONS:
        raise StructureInventoryError("target inventory contains a forbidden partition")
    return MappingProxyType(
        {
            "missing_parent_groups": missing,
            "extra_parent_groups": extra,
            "ambiguous_keys": 0,
        }
    )

def _independent_expected_parent_sets(
    inputs: FrozenBenchmarkInputs,
) -> tuple[frozenset[str], frozenset[str]]:

    canonical = _load_canonical_rows(inputs)
    queries = {
        item.parent_id
        for item in inputs.examples
        if item.role_realized and item.partition == "validation"
    }
    targets = {
        item.parent_id
        for item in inputs.examples
        if item.role_realized and item.partition == "train"
    }
    queries.update(
        str(row["parent_id"])
        for row in canonical.values()
        if str(row["partition"]) == "test-blind" and bool(row["view_strict"])
    )
    targets.update(
        str(row["parent_id"])
        for row in canonical.values()
        if str(row["partition"]) == "legacy-dev"
    )
    if queries & targets:
        raise StructureInventoryError(
            "frozen inputs place a parent in query and target"
        )
    return frozenset(queries), frozenset(targets)

def _preflight_candidate_sources(
    query: Mapping[str, tuple[_Candidate, ...]],
    target: Mapping[str, tuple[_Candidate, ...]],
) -> None:

    missing = []
    for scope, groups in (("query", query), ("target", target)):
        for parent_id, candidates in sorted(groups.items()):
            if not candidates:
                raise StructureInventoryError(
                    f"required {scope} parent {parent_id} has no source candidates",
                    reason_code="required_structure_unavailable",
                )
            if not any(
                item.path.is_file() and not item.path.is_symlink()
                for item in candidates
            ):
                missing.append(
                    {
                        "scope": scope,
                        "parent_id": parent_id,
                        "family": candidates[0].family,
                        "partition": candidates[0].partition,
                        "candidate_paths": [str(item.path) for item in candidates],
                    }
                )
    if missing:
        preview = json.dumps(missing[:10], sort_keys=True, separators=(",", ":"))
        raise StructureInventoryError(
            f"{len(missing)} required parent structure source(s) unavailable: {preview}",
            reason_code="required_structure_unavailable",
        )

def semantic_inventory_rows(
    inventory: StructureInventory,
) -> tuple[Mapping[str, object], ...]:

    rows = []
    bulk_run = inventory.query_dir.parent.resolve()
    evidence_run = Path(inventory.inventory.path).parent.resolve()

    def normalize(path: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        resolved = candidate.resolve()
        for root, prefix in ((bulk_run, "$BULK_RUN"), (evidence_run, "$EVIDENCE_RUN")):
            if resolved == root or root in resolved.parents:
                return f"{prefix}/{resolved.relative_to(root)}"
        return path

    for scope, records in (("query", inventory.queries), ("target", inventory.targets)):
        for record in records:
            value = _record_mapping(record)
            value["materialized"] = {
                **value["materialized"],
                "path": normalize(record.materialized.path),
            }
            value["source"] = {
                **value["source"],
                "path": normalize(record.source.path),
            }
            value["source_provenance"] = [
                {**item, "path": normalize(str(item["path"]))}
                for item in value["source_provenance"]
            ]
            rows.append(MappingProxyType({"scope": scope, **value}))
    return tuple(rows)

def _candidate_universes(
    inputs: FrozenBenchmarkInputs,
) -> tuple[dict[str, tuple[_Candidate, ...]], dict[str, tuple[_Candidate, ...]]]:
    canonical = _load_canonical_rows(inputs)
    sabdab_table = (
        _load_sabdab_role_table()
        if _TEST_CANONICAL_ROLES is None
        and any(str(row["family"]) == "antibody" for row in canonical.values())
        else None
    )
    legacy_resolver = (
        _load_legacy_resolver()
        if _TEST_SOURCE_PATHS is None
        and any(
            str(row["partition"]) == "legacy-dev"
            and str(row["family"]) in {"binder", "ame"}
            for row in canonical.values()
        )
        else None
    )
    admitted_by_id = {
        example.example_id: example
        for example in inputs.examples
        if example.role_realized
    }
    queries: dict[str, list[_Candidate]] = defaultdict(list)
    targets: dict[str, list[_Candidate]] = defaultdict(list)

    for example in admitted_by_id.values():
        if example.partition not in {"train", "validation"}:
            raise StructureInventoryError(
                "v5 admitted membership contains an unexpected partition"
            )
        row = canonical.get(example.example_id)
        if row is None or str(row["parent_id"]) != example.parent_id:
            raise StructureInventoryError(
                f"ambiguous parent mapping for {example.example_id}"
            )
        candidates = _candidates_from_row(
            row,
            example=example,
            sabdab_table=sabdab_table,
            legacy_resolver=legacy_resolver,
        )
        (queries if example.partition == "validation" else targets)[
            example.parent_id
        ].extend(candidates)

    for row in canonical.values():
        partition = str(row["partition"])
        if partition == "test-blind" and bool(row["view_strict"]):
            candidates = _candidates_from_row(
                row,
                example=None,
                sabdab_table=sabdab_table,
                legacy_resolver=legacy_resolver,
            )
            queries[str(row["parent_id"])].extend(candidates)
        elif partition == "legacy-dev":
            candidates = _candidates_from_row(
                row,
                example=None,
                sabdab_table=sabdab_table,
                legacy_resolver=legacy_resolver,
            )
            targets[str(row["parent_id"])].extend(candidates)

    for scope, groups in (("query", queries), ("target", targets)):
        for parent_id, candidates in groups.items():
            signatures = {
                (item.family, item.partition, item.parent_id) for item in candidates
            }
            if len(signatures) != 1:
                raise StructureInventoryError(
                    f"ambiguous parent mapping for {parent_id}"
                )
            if scope == "query" and candidates[0].partition not in _QUERY_PARTITIONS:
                raise StructureInventoryError(
                    f"invalid query partition for {parent_id}"
                )
            if scope == "target" and candidates[0].partition not in _TARGET_PARTITIONS:
                raise StructureInventoryError(
                    f"invalid target partition for {parent_id}"
                )
    return (
        {
            key: tuple(sorted(value, key=lambda item: item.example_id))
            for key, value in queries.items()
        },
        {
            key: tuple(sorted(value, key=lambda item: item.example_id))
            for key, value in targets.items()
        },
    )

def _load_canonical_rows(
    inputs: FrozenBenchmarkInputs,
) -> dict[str, Mapping[str, object]]:
    import pandas as pd

    rows: dict[str, Mapping[str, object]] = {}
    for family in _FAMILIES:
        identity = inputs.input_identities.get(f"canonical_manifest:{family}")
        if identity is None or file_identity(Path(identity.path)) != identity:
            raise StructureInventoryError(
                f"canonical {family} manifest identity drifted"
            )
        try:
            frame = pd.read_parquet(identity.path)
        except Exception as error:
            raise StructureInventoryError(
                f"cannot read canonical {family} manifest"
            ) from error
        required = {
            "example_id",
            "parent_id",
            "family",
            "partition",
            "view_strict",
            "pdb_id",
        }
        if not required.issubset(frame.columns):
            raise StructureInventoryError(
                f"canonical {family} structure fields are incomplete"
            )
        for raw in frame.to_dict("records"):
            example_id = str(raw["example_id"])
            if str(raw["family"]) != family or example_id in rows:
                raise StructureInventoryError(
                    f"ambiguous canonical example {example_id}"
                )
            rows[example_id] = MappingProxyType(dict(raw))
    return rows

def _candidates_from_row(
    row: Mapping[str, object],
    *,
    example: BenchmarkExample | None,
    sabdab_table: _SabdabRoleTable | None = None,
    legacy_resolver: _LegacyResolver | None = None,
) -> tuple[_Candidate, ...]:
    family = str(row["family"])
    example_id = str(row["example_id"])

    role_provenance: tuple[ArtifactIdentity, ...] = ()
    if _TEST_SOURCE_PATHS is not None and example is not None:
        roles = example.roles
    elif family == "antibody" and _TEST_CANONICAL_ROLES is None:
        table = sabdab_table or _load_sabdab_role_table()
        roles = _sabdab_roles(example_id, table)
        role_provenance = (table.identity,)
    else:
        roles = _roles_from_canonical_id(family, example_id)
    source_id = str(row.get("pdb_id") or example_id)
    excluded_components: tuple[str, ...] = ()
    if family == "ame" and str(row["partition"]) != "legacy-dev":
        from dive.data.plinder import ligand_chains_of

        try:
            excluded_components = ligand_chains_of(example_id.removeprefix("ame:"))
        except ValueError as error:
            raise StructureInventoryError(
                f"cannot decode AME ligand components for {example_id}"
            ) from error
    if _TEST_SOURCE_PATHS is not None:
        try:
            sources = (_LegacySource(Path(_TEST_SOURCE_PATHS[example_id]), ()),)
        except KeyError as error:
            raise StructureInventoryError(
                f"no test structure path for {example_id}",
                reason_code="required_structure_unavailable",
            ) from error
        namespace = "author"
    elif example is not None:
        sources = (_LegacySource(Path(example.path), ()),)
        namespace = "author"
    elif str(row["partition"]) == "legacy-dev":
        sources = _legacy_source_candidates(
            family,
            source_id,
            example_id,
            resolver=legacy_resolver,
        )
        namespace = "author"
    else:
        sources = (_LegacySource(_canonical_source_path(family, row), ()),)
        namespace = "author"
    candidates = tuple(
        _Candidate(
            family=family,
            partition=str(row["partition"]),
            source_id=source_id,
            example_id=example_id,
            parent_id=str(row["parent_id"]),
            path=source.path,
            roles=roles,
            component_namespace=namespace,
            provenance=source.provenance + role_provenance,
            component_roles=source.component_roles,
            excluded_components=excluded_components,
        )
        for source in sources
    )
    return candidates

def _canonical_source_path(family: str, row: Mapping[str, object]) -> Path:
    example_id = str(row["example_id"])
    if _TEST_SOURCE_PATHS is not None:
        try:
            return Path(_TEST_SOURCE_PATHS[example_id])
        except KeyError as error:
            raise StructureInventoryError(
                f"no test structure path for {example_id}"
            ) from error
    if family == "binder":
        return (
            _BULK_ROOT
            / "structures"
            / "binder"
            / f"{str(row['pdb_id']).lower()}.cif.gz"
        )
    if family == "ame":
        return (
            _BULK_ROOT / "structures" / "ame" / f"{example_id.removeprefix('ame:')}.cif"
        )
    if family == "antibody":
        return (
            _BULK_ROOT
            / "sources"
            / "sabdab2-v0.1.0"
            / "splits_final"
            / f"{example_id}.cif"
        )
    raise StructureInventoryError(f"unsupported structure family {family}")

def _roles_from_canonical_id(family: str, example_id: str) -> Mapping[str, str]:
    if _TEST_CANONICAL_ROLES is not None:
        try:
            return MappingProxyType(dict(_TEST_CANONICAL_ROLES[example_id]))
        except KeyError as error:
            raise StructureInventoryError(
                f"no test canonical roles for {example_id}"
            ) from error
    roles: dict[str, str] = {}
    if family == "binder":
        try:
            assignment = example_id.split(":", 2)[2]
            generated, targets = assignment.split("|", 1)
        except (IndexError, ValueError) as error:
            raise StructureInventoryError(
                f"cannot decode binder roles for {example_id}"
            ) from error
        roles = {
            "generated": generated,
            "context": generated,
            "target": targets.replace("+", ","),
        }
    elif family == "ame":
        parts = example_id.removeprefix("ame:").split("__")
        if len(parts) != 4 or not parts[2]:
            raise StructureInventoryError(f"cannot decode AME roles for {example_id}")
        generated = ",".join(
            component for component in parts[2].split("_") if component
        )
        roles = {"generated": generated, "context": generated, "target": ""}
    elif family == "antibody":
        roles = _sabdab_roles(example_id)
    return MappingProxyType(roles)

def _sabdab_roles(
    example_id: str, table: _SabdabRoleTable | None = None
) -> dict[str, str]:

    from dive.data.roles import antibody_roles

    retained = table or _load_sabdab_role_table()
    try:
        row = retained.rows.get(example_id)
        if row is not None:
            return antibody_roles(row).as_columns()
    except Exception as error:
        raise StructureInventoryError(
            f"cannot load frozen antibody roles for {example_id}: {error}"
        ) from error
    raise StructureInventoryError(
        f"frozen SAbDab2 table has no role row for {example_id}"
    )

def _load_sabdab_role_table() -> _SabdabRoleTable:

    matching = tuple(
        item
        for item in PRODUCTION_CONTRACT_V2.sources
        if item.name == "antibody_split_table"
    )
    if len(matching) != 1:
        raise StructureInventoryError(
            "projection-v2 has no unique antibody_split_table source identity"
        )
    source = matching[0]
    if source.kind != "public_raw" or source.path.resolve() != _SABDAB_TABLE.resolve():
        raise StructureInventoryError(
            "projection-v2 antibody_split_table path/source identity drifted",
            reason_code="source_drift",
        )
    raw, identity = _read_stable_source(_SABDAB_TABLE)
    expected = ArtifactIdentity(
        str(_SABDAB_TABLE.resolve()), source.sha256, source.size_bytes
    )
    if identity != expected:
        raise StructureInventoryError(
            "projection-v2 antibody_split_table source identity drifted",
            reason_code="source_drift",
        )
    required = {"INSTANCE", "Hchain", "Lchain", "Hseq", "CDRH3", "agchains"}
    indexed: dict[str, Mapping[str, str]] = {}
    try:
        handle = io.StringIO(raw.decode("utf-8"), newline="")
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise StructureInventoryError("pinned SAbDab2 role schema drifted")
        for row in reader:
            example_id = str(row["INSTANCE"])
            if example_id in indexed:
                raise StructureInventoryError(
                    f"duplicate SAbDab2 role row {example_id}",
                    reason_code="ambiguity_or_duplicate",
                )
            indexed[example_id] = MappingProxyType(
                {name: str(row.get(name, "")) for name in required}
            )
    except UnicodeDecodeError as error:
        raise StructureInventoryError(
            "pinned SAbDab2 role table is not UTF-8"
        ) from error
    return _SabdabRoleTable(
        identity,
        source.name,
        MappingProxyType(indexed),
        raw,
    )

def _sabdab_role_rows() -> Mapping[str, Mapping[str, str]]:

    return _load_sabdab_role_table().rows

def _load_legacy_resolver() -> _LegacyResolver:

    import yaml

    try:
        verify_git_checkout(_UPSTREAM_ROOT, _UPSTREAM_COMMIT)
    except ProvenanceError as error:
        raise StructureInventoryError(
            f"cannot authenticate pinned upstream checkout: {error}",
            reason_code="source_drift",
        ) from error
    try:
        repo_commit = _git_output(_REPO_ROOT, "rev-parse", "HEAD")
        registry_raw, registry_identity = _authenticated_git_snapshot(
            _REPO_ROOT,
            repo_commit,
            _LEGACY_REGISTRY_SPEC,
            label="legacy source registry",
        )
        configs: dict[str, Mapping[str, object]] = {}
        dictionary_identities: dict[str, ArtifactIdentity] = {}
        for family, (spec, key) in _LEGACY_DICTIONARY_SPECS.items():
            raw, identity = _authenticated_git_snapshot(
                _UPSTREAM_ROOT,
                _UPSTREAM_COMMIT,
                spec,
                label=f"{family} legacy dictionary",
            )
            loaded = yaml.safe_load(raw.decode("utf-8"))
            section = loaded.get(key) if isinstance(loaded, Mapping) else None
            if not isinstance(section, Mapping):
                raise StructureInventoryError(
                    f"pinned {family} dictionary is missing {key}"
                )
            configs[family] = MappingProxyType(dict(section))
            dictionary_identities[family] = identity
        handle = io.StringIO(registry_raw.decode("utf-8"), newline="")
        reader = csv.DictReader(handle)
        required = {"family", "target_id", "pdb_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise StructureInventoryError("legacy source registry schema drifted")
        registered = tuple(
            MappingProxyType({name: str(value or "") for name, value in row.items()})
            for row in reader
        )
    except StructureInventoryError:
        raise
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        raise StructureInventoryError(
            f"cannot authenticate legacy source adapter: {error}",
            reason_code="source_drift",
        ) from error
    if any(row["family"] not in {"binder", "ame"} for row in registered):
        raise StructureInventoryError(
            "legacy source registry has an unsupported family"
        )
    return _LegacyResolver(
        registry_identity,
        MappingProxyType(dictionary_identities),
        MappingProxyType(configs),
        registered,
    )

def _authenticated_git_snapshot(
    root: Path,
    commit: str,
    spec: _GitSourceSpec,
    *,
    label: str,
) -> tuple[bytes, ArtifactIdentity]:

    path = root / spec.relative_path
    raw, identity = _read_stable_source(
        path, recorded_path=spec.relative_path.as_posix()
    )
    if identity.sha256 != spec.sha256 or identity.size_bytes != spec.size_bytes:
        raise StructureInventoryError(
            f"authenticated Git blob bytes drifted for {label}",
            reason_code="source_drift",
        )
    object_name = f"{commit}:{spec.relative_path.as_posix()}"
    observed_oid = _git_output(root, "rev-parse", "--verify", object_name)
    if observed_oid != spec.blob_oid:
        raise StructureInventoryError(
            f"authenticated Git blob identity drifted for {label}",
            reason_code="source_drift",
        )
    if _git_output(root, "cat-file", "-t", observed_oid) != "blob":
        raise StructureInventoryError(
            f"authenticated Git object is not a blob: {label}"
        )
    live_oid = _git_output(root, "hash-object", "--stdin", input_bytes=raw)
    if live_oid != observed_oid:
        raise StructureInventoryError(
            f"authenticated Git blob content drifted for {label}",
            reason_code="source_drift",
        )
    return raw, identity

def _git_output(root: Path, *args: str, input_bytes: bytes | None = None) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=root,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise StructureInventoryError(
            f"Git authentication failed: {detail}", reason_code="source_drift"
        )
    return result.stdout.decode("ascii").strip()

def _legacy_source_candidates(
    family: str,
    pdb_id: str,
    example_id: str,
    *,
    resolver: _LegacyResolver | None = None,
) -> tuple[_LegacySource, ...]:

    if family not in {"binder", "ame"}:
        raise StructureInventoryError(f"unsupported legacy family {family}")
    authenticated = resolver or _load_legacy_resolver()
    configs = authenticated.configs
    registered = [
        row
        for row in authenticated.registered
        if str(row.get("pdb_id", "")).lower() == pdb_id.lower()
    ]
    resolved: dict[Path, _LegacySource] = {}
    for row in sorted(
        registered,
        key=lambda item: (str(item["family"]), str(item["target_id"])),
    ):
        registry_family = str(row["family"])
        entry = configs[registry_family].get(str(row["target_id"]))
        if not isinstance(entry, Mapping):
            continue
        raw_path = entry.get("target_path")
        if raw_path:
            path = (_UPSTREAM_ROOT / str(raw_path).removeprefix("./")).resolve()
        elif registry_family == "ame":
            path = (
                _UPSTREAM_ROOT
                / "assets/target_data/ame_input_structures"
                / f"{entry['target_filename']}.pdb"
            )
        else:
            path = (
                _UPSTREAM_ROOT
                / "assets/target_data"
                / str(entry["source"])
                / f"{entry['target_filename']}.pdb"
            )
        if registry_family == "binder":
            from dive.groundtruth.complex_spec import target_chain_ids

            try:
                source_components = target_chain_ids(str(entry["target_input"]))
            except Exception as error:
                raise StructureInventoryError(
                    f"cannot decode pinned binder source components for {row['target_id']}"
                ) from error
            role = "target"
        else:
            source_components = tuple(
                dict.fromkeys(
                    match.group(1)
                    for match in re.finditer(
                        r"([A-Za-z])(\d+): \[([^\]]+)\]",
                        str(entry.get("contig_atoms", "")),
                    )
                )
            )
            role = "generated"
        if not source_components:
            raise StructureInventoryError(
                f"pinned legacy source declares no components for {row['target_id']}"
            )
        component_roles = MappingProxyType(
            {component: role for component in source_components}
        )
        source = _LegacySource(
            path,
            (
                authenticated.registry_identity,
                authenticated.dictionary_identities[registry_family],
            ),
            component_roles,
        )
        prior = resolved.get(path)
        if prior is not None and prior.component_roles != source.component_roles:
            raise StructureInventoryError(
                f"conflicting pinned component declarations for {path}",
                reason_code="ambiguity_or_duplicate",
            )
        resolved[path] = source
    if not resolved:
        return ()
    return tuple(resolved[path] for path in sorted(resolved, key=str))

def _materialize_scope(
    groups: Mapping[str, tuple[_Candidate, ...]],
    destination: Path,
    scope: str,
    code_identity: ArtifactIdentity,
) -> tuple[StructureRecord, ...]:
    records: list[StructureRecord] = []
    keys: set[str] = set()
    for parent_id, candidates in sorted(groups.items()):
        failures = []
        selected: (
            tuple[
                _Candidate,
                ArtifactIdentity,
                tuple[tuple[str, str, str, bytes], ...],
            ]
            | None
        ) = None
        for candidate in candidates:
            try:
                source_identity, components = _resolve_candidate_components(candidate)
            except StructureInventoryError as error:
                failures.append(f"{candidate.example_id}:{error}")
                continue
            selected = candidate, source_identity, components
            break
        if selected is None:
            detail = "; ".join(failures) or "no source candidates"
            raise StructureInventoryError(
                f"required {scope} parent {parent_id} cannot be materialized: {detail}",
                reason_code=(
                    "required_structure_unavailable"
                    if failures
                    and all("source unavailable" in item for item in failures)
                    else "component_mismatch"
                ),
            )
        candidate, source_identity, components = selected
        for component_id, chain_id, role, payload in components:
            key = encode_structure_key(candidate.family, parent_id, chain_id)
            if key in keys:
                raise StructureInventoryError(
                    f"ambiguous duplicate key {key}",
                    reason_code="ambiguity_or_duplicate",
                )
            keys.add(key)
            output = destination / f"{key}.pdb"
            _write_materialized(output, payload)
            records.append(
                StructureRecord(
                    foldseek_key=key,
                    family=candidate.family,
                    partition=candidate.partition,
                    source_id=candidate.source_id,
                    example_id=candidate.example_id,
                    parent_id=parent_id,
                    component_id=component_id,
                    chain_id=chain_id,
                    role=role,
                    source=source_identity,
                    source_provenance=candidate.provenance,
                    materialized=file_identity(output),
                    materialization_command=_MATERIALIZATION_COMMAND,
                    code_identity=code_identity,
                )
            )
    return tuple(sorted(records, key=lambda item: item.foldseek_key))

def _component_roles(candidate: _Candidate) -> Mapping[str, str]:

    if candidate.component_roles is not None:
        if not candidate.component_roles or any(
            type(component) is not str
            or not component
            or role not in {"generated", "context", "target", "antigen"}
            for component, role in candidate.component_roles.items()
        ):
            raise StructureInventoryError(
                f"invalid pinned source components for {candidate.example_id}"
            )
        return MappingProxyType(dict(candidate.component_roles))

    from dive.data.roles import Selector

    named: dict[str, set[str]] = defaultdict(set)
    for role in ("generated", "context", "target"):
        descriptor = str(candidate.roles.get(role, ""))
        for raw in descriptor.replace("/", ",").split(","):
            if raw.strip():
                try:
                    named[Selector.parse(raw).chain].add(role)
                except Exception as error:
                    raise StructureInventoryError(
                        f"invalid {role} descriptor for {candidate.example_id}: {raw}"
                    ) from error
    if not named or not named.get(next(iter(named)), set()):
        raise StructureInventoryError(
            f"no declared structure components for {candidate.example_id}"
        )
    if not str(candidate.roles.get("generated", "")) or not str(
        candidate.roles.get("context", "")
    ):
        raise StructureInventoryError(
            f"incomplete generated/context contract for {candidate.example_id}"
        )
    labels: dict[str, str] = {}
    for component, roles in named.items():
        if "generated" in roles:
            labels[component] = "generated"
        elif "target" in roles:
            labels[component] = (
                "antigen" if candidate.family == "antibody" else "target"
            )
        else:
            labels[component] = "context"
    return MappingProxyType(labels)

def _resolve_candidate_components(
    candidate: _Candidate,
) -> tuple[ArtifactIdentity, tuple[tuple[str, str, str, bytes], ...]]:

    source_identity, parsed = _canonical_chains(
        candidate.path, component_namespace=candidate.component_namespace
    )
    return source_identity, _reconcile_components(candidate, parsed)

def _reconcile_components(
    candidate: _Candidate, parsed: Mapping[str, _ParsedChain]
) -> tuple[tuple[str, str, str, bytes], ...]:
    component_roles = _component_roles(candidate)
    nonprotein_authors = getattr(parsed, "nonprotein_authors", frozenset())
    if getattr(parsed, "component_namespace", candidate.component_namespace) != (
        candidate.component_namespace
    ):
        raise StructureInventoryError(
            f"parsed component namespace does not match {candidate.example_id}"
        )
    selected: list[tuple[str, str, str, bytes]] = []
    used_authors: set[str] = set()

    excluded_authors: set[str] = set()
    for component in candidate.excluded_components:
        matches = (
            [component]
            if component in parsed or component in nonprotein_authors
            else []
        )
        matches = sorted(set(matches))
        if len(matches) > 1:
            raise StructureInventoryError(
                f"excluded component {component!r} is ambiguous in {candidate.path}",
                reason_code="ambiguity_or_duplicate",
            )
        if matches and matches[0] in parsed:
            excluded_authors.add(matches[0])

    for component, role in sorted(component_roles.items()):
        matches = (
            [component]
            if component in parsed or component in nonprotein_authors
            else []
        )
        if not matches and candidate.family == "ame" and "." in component:
            suffix = component.split(".", 1)[1]
            matches = [name for name in parsed if name == suffix]
        matches = sorted(set(matches))
        if len(matches) != 1:
            raise StructureInventoryError(
                f"component {component!r} does not map one-to-one in {candidate.path}"
            )
        author = matches[0]
        if author in nonprotein_authors:
            if role == "antigen":
                continue
            raise StructureInventoryError(
                f"required protein component {component!r} is non-protein in "
                f"{candidate.path}"
            )
        if author not in parsed:
            raise StructureInventoryError(
                f"component {component!r} does not map to a protein chain in "
                f"{candidate.path}"
            )
        if author in excluded_authors:
            raise StructureInventoryError(
                f"component {component!r} is both a record and an authenticated "
                f"exclusion in {candidate.path}",
                reason_code="ambiguity_or_duplicate",
            )
        if author in used_authors:
            raise StructureInventoryError(
                f"duplicate component alias maps to author chain {author!r}",
                reason_code="ambiguity_or_duplicate",
            )
        used_authors.add(author)
        selected.append((component, author, role, parsed[author].payload))
    if not selected:
        raise StructureInventoryError(
            f"declared components emit no protein records for {candidate.example_id}"
        )
    return tuple(selected)

def _canonical_chains(
    path: Path, *, component_namespace: str = "author"
) -> tuple[ArtifactIdentity, _ParsedChains]:
    if component_namespace not in {"author", "label"}:
        raise StructureInventoryError(
            f"unsupported component namespace {component_namespace!r}"
        )
    raw, identity = _read_stable_source(path)
    try:
        decoded = gzip.decompress(raw) if path.suffix == ".gz" else raw
    except (OSError, EOFError) as error:
        raise StructureInventoryError(
            f"compressed source is invalid: {path}"
        ) from error
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StructureInventoryError(f"structure is not UTF-8 text: {path}") from error
    try:
        is_mmcif = path.name.endswith((".cif", ".cif.gz", ".mmcif", ".mmcif.gz"))
        if is_mmcif:
            from Bio.PDB import FastMMCIFParser

            structure = FastMMCIFParser(
                QUIET=True, auth_chains=component_namespace == "author"
            ).get_structure("source", io.StringIO(text))
        else:
            from Bio.PDB import PDBParser

            explicit_models = [
                int(line[10:14].strip())
                for line in text.splitlines()
                if line.startswith("MODEL") and line[10:14].strip()
            ]
            if explicit_models and 1 not in explicit_models:
                raise StructureInventoryError(
                    f"structure has no source model serial 1: {path}"
                )
            structure = PDBParser(QUIET=True).get_structure("source", io.StringIO(text))
    except Exception as error:
        raise StructureInventoryError(
            f"cannot parse structure {path}: {error}"
        ) from error
    models = list(structure.get_models())
    if not models:
        raise StructureInventoryError(f"structure has no model: {path}")
    if is_mmcif or explicit_models:
        matching_models = [
            item for item in models if int(getattr(item, "serial_num", -1)) == 1
        ]
    else:
        matching_models = [models[0]]
    if len(matching_models) != 1:
        raise StructureInventoryError(
            f"structure must contain exactly one source model serial 1: {path}"
        )
    model = matching_models[0]
    chains: dict[str, _ParsedChain] = {}
    nonprotein_authors: set[str] = set()
    for chain in sorted(model.get_chains(), key=lambda item: str(item.id)):
        payload = _canonical_chain_pdb(chain)
        if payload is not None:
            chain_id = str(chain.id)
            if not chain_id or chain_id in chains:
                raise StructureInventoryError(
                    f"duplicate or empty chain identity in {path}"
                )
            chains[chain_id] = _ParsedChain(chain_id, payload)
        else:
            nonprotein_authors.add(str(chain.id))
    if not chains:
        raise StructureInventoryError(
            f"structure has no non-empty polymer chains: {path}"
        )
    return identity, _ParsedChains(
        chains,
        nonprotein_authors=frozenset(nonprotein_authors),
        component_namespace=component_namespace,
        model_serial=1,
    )

def _canonical_chain_pdb(chain: object) -> bytes | None:
    from Bio.PDB import PDBIO
    from Bio.PDB.Atom import DisorderedAtom
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Model import Model
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Polypeptide import is_aa

    residues = [
        residue for residue in chain.get_residues() if is_aa(residue, standard=False)
    ]
    if not residues:
        return None
    output = Structure("canonical")
    model = Model(0)
    output.add(model)
    output_chain = Chain("A")
    model.add(output_chain)
    for residue in sorted(
        residues, key=lambda item: (item.id[1], str(item.id[2]), str(item.id[0]))
    ):
        new_residue = Residue(residue.id, residue.resname, residue.segid)
        atoms = []
        for atom in residue.child_list:
            choices = (
                atom.disordered_get_list()
                if isinstance(atom, DisorderedAtom)
                else [atom]
            )
            chosen = sorted(
                choices,
                key=lambda item: (
                    -(
                        item.get_occupancy()
                        if item.get_occupancy() is not None
                        else -1.0
                    ),
                    str(item.get_altloc()),
                ),
            )[0]
            atoms.append(chosen)
        for atom in sorted(
            atoms, key=lambda item: (str(item.get_name()), str(item.get_altloc()))
        ):
            cloned = copy.copy(atom)
            cloned.set_parent(None)
            cloned.disordered_flag = 0
            cloned.set_altloc(" ")
            new_residue.add(cloned)
        if new_residue.child_list:
            output_chain.add(new_residue)
    if not output_chain.child_list:
        return None
    handle = io.StringIO()
    writer = PDBIO()
    writer.set_structure(output)
    writer.save(handle, preserve_atom_numbering=False)
    return handle.getvalue().encode("ascii")

def _read_stable_source(
    path: Path, *, recorded_path: str | None = None
) -> tuple[bytes, ArtifactIdentity]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise StructureInventoryError(
            f"source unavailable {path}: {error}",
            reason_code="required_structure_unavailable",
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise StructureInventoryError(f"source is not a regular file: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise StructureInventoryError(
            f"source changed while being read: {path}", reason_code="source_drift"
        )
    identity = ArtifactIdentity(
        recorded_path or str(path.resolve()), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    return raw, identity

def _canonical_code_identity() -> ArtifactIdentity:

    return _read_stable_source(
        _CODE_PATH,
        recorded_path=_CODE_RELATIVE.as_posix(),
    )[1]

def _write_materialized(destination: Path, payload: bytes) -> None:
    if destination.exists():
        raise StructureInventoryError(
            f"materialized destination already exists: {destination}"
        )
    try:

        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o664,
        )
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise StructureInventoryError(
            f"cannot materialize {destination}: {error}",
            reason_code="verification_or_write_error",
        ) from error

def _verify_one_chain_pdb(path: Path) -> None:
    from Bio.PDB import PDBParser

    try:
        structure = PDBParser(QUIET=True).get_structure("verify", path)
        models = list(structure.get_models())
        chains = list(models[0].get_chains()) if models else []
    except Exception as error:
        raise StructureInventoryError(f"cannot independently parse {path}") from error
    if len(models) != 1 or len(chains) != 1 or not any(chains[0].get_residues()):
        raise StructureInventoryError(
            f"materialized PDB is not exactly one non-empty chain: {path}"
        )

def _create_new_directory(path: Path) -> None:
    try:
        os.mkdir(path, 0o2775)
    except FileExistsError as error:
        raise StructureInventoryError(
            f"structure destination already exists: {path}",
            reason_code="verification_or_write_error",
        ) from error
    except OSError as error:
        raise StructureInventoryError(
            f"cannot create structure destination {path}: {error}",
            reason_code="verification_or_write_error",
        ) from error

def _write_jsonl_create_new(path: Path, rows: tuple[Mapping[str, object], ...]) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o664,
        )
        try:
            for row in rows:
                _write_all(descriptor, canonical_json_bytes(row))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise StructureInventoryError(
            f"cannot create structure inventory: {error}",
            reason_code="verification_or_write_error",
        ) from error

def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while creating benchmark structure artifact")
        view = view[written:]

def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows = []
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise StructureInventoryError(
            f"cannot read structure inventory: {error}"
        ) from error
    for line in lines:
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise StructureInventoryError(
                "structure inventory is invalid JSONL"
            ) from error
        if not isinstance(row, Mapping) or canonical_json_bytes(row) != line:
            raise StructureInventoryError("structure inventory is not canonical JSONL")
        rows.append(MappingProxyType(dict(row)))
    return tuple(rows)

def _record_mapping(record: StructureRecord) -> dict[str, object]:
    return {
        "schema_version": "dive-benchmark-structure-record-v1",
        "foldseek_key": record.foldseek_key,
        "family": record.family,
        "partition": record.partition,
        "source_id": record.source_id,
        "example_id": record.example_id,
        "parent_id": record.parent_id,
        "component_id": record.component_id,
        "chain_id": record.chain_id,
        "role": record.role,
        "source": _identity_mapping(record.source),
        "source_provenance": [
            _identity_mapping(identity) for identity in record.source_provenance
        ],
        "materialized": _identity_mapping(record.materialized),
        "materialization_command": list(record.materialization_command),
        "code_identity": _identity_mapping(record.code_identity),
        "failure_reason": record.failure_reason,
    }

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }
