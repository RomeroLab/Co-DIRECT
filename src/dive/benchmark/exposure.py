
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from dive.benchmark.contracts import (
    ArtifactIdentity,
    BenchmarkContract,
    CutoffPredicateContract,
    ExposureClass,
    ExposureSubjectContract,
    file_identity,
)
from dive.benchmark.evidence import (
    BenchmarkRun,
    complete_benchmark_run,
    write_terminal_bytes,
)
from dive.training.preflight import canonical_json_bytes

class ExposureAuditError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ExposureEvidence:

    subject: str
    evidence: tuple[ArtifactIdentity, ...]
    provenance: tuple["_ParsedProvenance", ...] = ()
    cutoff_predicates: tuple["_ParsedProvenance", ...] = ()
    cutoff_predicate_rules: tuple[CutoffPredicateContract, ...] = ()
    _registry: "_AuthenticatedExposureRegistry | None" = field(
        default=None, init=False, repr=False, compare=False
    )

@dataclass(frozen=True, slots=True)
class ExposureRecord:
    parent_id: str
    subject: str
    classification: ExposureClass
    reason_code: str
    evidence: tuple[ArtifactIdentity, ...]

@dataclass(frozen=True, slots=True)
class ExposureAuditResult:
    source_input_completion: ArtifactIdentity
    search_log: ArtifactIdentity
    subject_records: tuple[tuple[str, ArtifactIdentity], ...]
    aggregate_records: ArtifactIdentity
    model_card_retrieval: ArtifactIdentity
    model_card: ArtifactIdentity | None
    completion: ArtifactIdentity
    parent_count: int
    temporal_clean_parent_count: int

_CURRENT_INPUT_RUN = "benchmark-inputs-20260826c"
_CURRENT_INPUT_IDENTITIES = {
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
_CURRENT_INPUT_CONTRACT_SHA256 = (
    "c98802efb47b3b4b18b338a9292eb70fcfa91b6e4875534d5a12f40dfb01141c"
)
_MODEL_CARD_URL = (
    "https://huggingface.co/nvidia/NV-Proteina-Complexa-Protein-Target-160M-v1/"
    "raw/main/README.md"
)
_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$", re.ASCII)
_MANIFEST_PROVENANCE_KINDS = frozenset(
    {
        "reference_database",
        "sequence_database",
        "template_database",
        "training_manifest",
    }
)
_DISCOVERY_PROVENANCE_KINDS = tuple(
    sorted(
        {
            *_MANIFEST_PROVENANCE_KINDS,
            "cutoff_evidence",
            "cutoff_predicate",
        }
    )
)

@dataclass(frozen=True, slots=True)
class _Snapshot:
    raw: bytes
    identity: ArtifactIdentity

@dataclass(frozen=True, slots=True)
class _ParsedProvenance:

    kind: str
    subject: str
    snapshot: _Snapshot
    manifest_ids: frozenset[str] | None = None
    manifest_complete: bool = False
    cutoff: date | None = None
    evaluator_family: str | None = None
    condition: str | None = None
    cutoff_identity: ArtifactIdentity | None = None

@dataclass(frozen=True, slots=True)
class _AuthenticatedExposureRegistry:

    subjects: tuple[ExposureSubjectContract, ...]
    cutoff_predicates: tuple[CutoffPredicateContract, ...]
    contract_sha256: str
    seal: object

_REGISTRY_SEAL = object()
_PROVENANCE_CACHE: dict[
    tuple[str, str, ArtifactIdentity], _ParsedProvenance | None
] = {}
_SOURCE_SNAPSHOTS: dict[Path, _Snapshot] = {}

def classify_exposure(
    candidate: Mapping[str, object] | object, subject: ExposureEvidence
) -> ExposureRecord:

    if not isinstance(subject, ExposureEvidence):
        raise ExposureAuditError("classify_exposure requires ExposureEvidence")
    parent_id = _candidate_parent_id(candidate)
    pdb_ids = _candidate_pdb_ids(candidate)
    if pdb_ids is None:
        return _record(
            parent_id,
            subject,
            ExposureClass.UNKNOWN,
            "unverified_candidate_neighbor_set",
        )
    manifests = _subject_provenance(
        subject,
        {
            "training_manifest",
            "template_database",
            "sequence_database",
            "reference_database",
        },
    )
    if any(
        pdb_ids & item.manifest_ids
        for item in manifests
        if item.manifest_ids is not None
    ):
        return _record(
            parent_id, subject, ExposureClass.KNOWN_EXPOSED, "exact_manifest_member"
        )
    deposited = _candidate_date(candidate)
    evaluator_family = _candidate_family(candidate)
    cutoffs = _subject_provenance(subject, {"cutoff_evidence"})
    if (
        deposited is not None
        and evaluator_family is not None
        and any(
            cutoff.cutoff is not None
            and deposited > cutoff.cutoff
            and _valid_cutoff_predicates(subject, cutoff, evaluator_family)
            for cutoff in cutoffs
        )
    ):
        return _record(parent_id, subject, ExposureClass.CLEAR, "verified_post_cutoff")
    if manifests and all(item.manifest_complete for item in manifests):
        return _record(
            parent_id, subject, ExposureClass.CLEAR, "complete_manifest_absent"
        )
    if manifests:
        return _record(
            parent_id, subject, ExposureClass.UNKNOWN, "incomplete_manifest_absence"
        )
    if cutoffs and deposited is None:
        return _record(
            parent_id, subject, ExposureClass.UNKNOWN, "missing_deposition_date"
        )
    return _record(
        parent_id,
        subject,
        ExposureClass.UNKNOWN,
        "no_complete_manifest_or_verified_cutoff",
    )

def aggregate_exposure(records: Sequence[ExposureRecord]) -> ExposureClass:

    if not records:
        raise ExposureAuditError("cannot aggregate an empty subject set")
    classifications = {record.classification for record in records}
    if ExposureClass.KNOWN_EXPOSED in classifications:
        return ExposureClass.KNOWN_EXPOSED
    if classifications == {ExposureClass.CLEAR}:
        return ExposureClass.CLEAR
    return ExposureClass.UNKNOWN

def temporal_clean_eligible(records: Sequence[ExposureRecord]) -> bool:

    return bool(records) and aggregate_exposure(records) is ExposureClass.CLEAR

def load_accepted_benchmark_parents(
    contract: BenchmarkContract, input_run_id: str
) -> tuple[tuple[Mapping[str, object], ...], ArtifactIdentity]:

    if input_run_id != _CURRENT_INPUT_RUN:
        raise ExposureAuditError(
            f"exposure audit requires accepted input run {_CURRENT_INPUT_RUN}"
        )
    source = Path(contract.evidence_root) / "benchmark_ready" / input_run_id
    snapshots = {
        name: _read_exact_snapshot(source / name, name)
        for name in ("attempt.json", "completion.json", "input_inventory.json")
    }
    try:
        attempt = _canonical_object(snapshots["attempt.json"].raw, "input attempt")
        completion = _canonical_object(
            snapshots["completion.json"].raw, "input completion"
        )
        inventory = _canonical_object(
            snapshots["input_inventory.json"].raw, "input inventory"
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ExposureAuditError(f"cannot parse accepted input run: {error}") from error
    if (
        attempt.get("schema_version") != "dive-benchmark-attempt-v1"
        or attempt.get("status") != "BUILDING"
        or attempt.get("run_id") != input_run_id
        or attempt.get("contract_sha256") != _CURRENT_INPUT_CONTRACT_SHA256
    ):
        raise ExposureAuditError("accepted input attempt does not bind this contract")
    if (
        completion.get("schema_version") != "dive-benchmark-completion-v1"
        or completion.get("status") != "COMPLETE"
        or completion.get("run_id") != input_run_id
        or completion.get("contract_sha256") != _CURRENT_INPUT_CONTRACT_SHA256
        or completion.get("attempt")
        != _identity_mapping(snapshots["attempt.json"].identity)
    ):
        raise ExposureAuditError(
            "accepted input completion does not authenticate its attempt"
        )
    payload = completion.get("payload")
    if not isinstance(payload, Mapping) or payload.get(
        "inventory"
    ) != _identity_mapping(snapshots["input_inventory.json"].identity):
        raise ExposureAuditError(
            "accepted input completion does not authenticate its inventory"
        )
    parents = inventory.get("parents")
    if not isinstance(parents, list) or not parents:
        raise ExposureAuditError("accepted input inventory has no parents")
    normalized = tuple(_parent_mapping(item) for item in parents)
    if tuple(sorted(item["parent_id"] for item in normalized)) != tuple(
        item["parent_id"] for item in normalized
    ):
        raise ExposureAuditError(
            "accepted input parents are not deterministically ordered"
        )
    return normalized, snapshots["completion.json"].identity

def run_exposure_audit(
    contract: BenchmarkContract,
    parents: Sequence[Mapping[str, object]],
    run: BenchmarkRun,
    *,
    source_input_completion: ArtifactIdentity,
) -> ExposureAuditResult:

    if not isinstance(contract, BenchmarkContract) or not isinstance(run, BenchmarkRun):
        raise ExposureAuditError(
            "run_exposure_audit requires a contract and claimed run"
        )
    if not parents:
        raise ExposureAuditError("exposure audit requires benchmark parents")
    if not isinstance(source_input_completion, ArtifactIdentity):
        raise ExposureAuditError(
            "exposure audit requires authenticated input completion"
        )
    _PROVENANCE_CACHE.clear()
    _SOURCE_SNAPSHOTS.clear()
    registry = _authenticate_exposure_registry(contract)
    model_card_retrieval, model_card = _fetch_model_card(run.evidence_dir, run)
    search_rows = _provenance_search(contract, model_card=model_card)
    search_log = _write_jsonl(
        run.evidence_dir / "provenance_search.jsonl", search_rows, run
    )
    evidence = _subjects(
        contract,
        search_log,
        model_card_retrieval,
        model_card,
        registry=registry,
    )
    subject_records: list[tuple[str, ArtifactIdentity]] = []
    aggregate_rows = []
    temporal_clean = 0
    for subject in evidence:
        rows = [classify_exposure(parent, subject) for parent in parents]
        identity = _write_jsonl(
            run.evidence_dir / f"exposure.{subject.subject}.jsonl",
            (_record_mapping(item) for item in rows),
            run,
        )
        subject_records.append((subject.subject, identity))
    by_subject = {name: identity for name, identity in subject_records}
    subject_evidence = {item.subject: item for item in evidence}
    for parent in parents:
        records = tuple(
            classify_exposure(parent, subject_evidence[name]) for name in by_subject
        )
        aggregate = aggregate_exposure(records)
        if temporal_clean_eligible(records):
            temporal_clean += 1
        aggregate_rows.append(
            {
                "parent_id": str(parent["parent_id"]),
                "classification": aggregate.value,
                "temporal_clean_eligible": aggregate is ExposureClass.CLEAR,
                "subjects": {
                    record.subject: _record_mapping(record) for record in records
                },
            }
        )
    aggregate_records = _write_jsonl(
        run.evidence_dir / "exposure.aggregate.jsonl", aggregate_rows, run
    )
    completion = complete_benchmark_run(
        run,
        {
            "source_input_completion": _identity_mapping(source_input_completion),
            "provenance_search": _identity_mapping(search_log),
            "official_model_card_retrieval": _identity_mapping(model_card_retrieval),
            "official_model_card": (
                None if model_card is None else _identity_mapping(model_card)
            ),
            "subject_records": {
                name: _identity_mapping(identity) for name, identity in subject_records
            },
            "aggregate_records": _identity_mapping(aggregate_records),
            "parent_count": len(parents),
            "subject_count": len(subject_records),
            "common_checkpoint_classification": ExposureClass.UNKNOWN.value,
            "temporal_clean_parent_count": temporal_clean,
        },
    )
    return ExposureAuditResult(
        source_input_completion,
        search_log,
        tuple(subject_records),
        aggregate_records,
        model_card_retrieval,
        model_card,
        completion,
        len(parents),
        temporal_clean,
    )

def _read_exact_snapshot(path: Path, label: str) -> _Snapshot:
    expected_sha256, expected_size = _CURRENT_INPUT_IDENTITIES[label]
    try:
        with path.open("rb") as handle:
            raw = handle.read()
    except OSError as error:
        raise ExposureAuditError(f"cannot read accepted {label}: {error}") from error
    identity = ArtifactIdentity(
        str(path.resolve()), hashlib.sha256(raw).hexdigest(), len(raw)
    )
    if identity.sha256 != expected_sha256 or identity.size_bytes != expected_size:
        raise ExposureAuditError(f"accepted input identity drifted: {label}")
    return _Snapshot(raw, identity)

def _canonical_object(raw: bytes, label: str) -> Mapping[str, object]:
    value = json.loads(raw)
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise ExposureAuditError(f"{label} is not canonical JSON")
    return value

def _parent_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExposureAuditError("accepted input parent is not an object")
    required = {"parent_id", "family", "partition", "deposition_date", "example_ids"}
    if not required.issubset(value) or type(value["parent_id"]) is not str:
        raise ExposureAuditError("accepted input parent schema changed")
    if not isinstance(value["example_ids"], list) or not all(
        type(item) is str for item in value["example_ids"]
    ):
        raise ExposureAuditError("accepted input parent example ids changed")
    return dict(value)

def _subjects(
    contract: BenchmarkContract,
    search_log: ArtifactIdentity,
    model_card_retrieval: ArtifactIdentity,
    model_card: ArtifactIdentity | None,
    *,
    registry: _AuthenticatedExposureRegistry | None = None,
) -> tuple[ExposureEvidence, ...]:
    if registry is None:
        registry = _authenticate_exposure_registry(contract)
    if not _registry_matches_contract(registry, contract):
        raise ExposureAuditError("exposure registry does not authenticate the contract")
    source_evidence = (
        search_log,
        model_card_retrieval,
        *(() if model_card is None else (model_card,)),
    )
    evidence = []
    for item in contract.exposure_subjects:
        rules = tuple(
            rule for rule in registry.cutoff_predicates if rule.subject == item.subject
        )
        parsed = _parse_subject_provenance(item.kind, item.source, subject=item.subject)
        predicates = tuple(
            predicate
            for rule in rules
            if (
                predicate := _parse_subject_provenance(
                    "cutoff_predicate", rule.source, subject=item.subject
                )
            )
            is not None
            and _predicate_matches_rule(predicate, rule, item.source)
        )
        admitted = ExposureEvidence(
            subject=item.subject,
            evidence=(
                item.source,
                *(predicate.snapshot.identity for predicate in predicates),
                *source_evidence,
            ),
            provenance=() if parsed is None else (parsed,),
            cutoff_predicates=predicates,
            cutoff_predicate_rules=rules,
        )
        object.__setattr__(admitted, "_registry", registry)
        evidence.append(admitted)
    return tuple(evidence)

def _authenticate_exposure_registry(
    contract: BenchmarkContract,
) -> _AuthenticatedExposureRegistry:

    contract.assert_authenticated()
    return _AuthenticatedExposureRegistry(
        contract.exposure_subjects,
        contract.cutoff_predicates,
        contract.semantic_sha256,
        _REGISTRY_SEAL,
    )

def _registry_matches_contract(
    registry: _AuthenticatedExposureRegistry, contract: BenchmarkContract
) -> bool:
    return (
        registry.seal is _REGISTRY_SEAL
        and registry.subjects == contract.exposure_subjects
        and registry.cutoff_predicates == contract.cutoff_predicates
        and registry.contract_sha256 == contract.semantic_sha256
    )

def _parse_subject_provenance(
    kind: str,
    identity: ArtifactIdentity,
    *,
    subject: str | None = None,
) -> _ParsedProvenance | None:

    if subject is not None:
        key = (subject, kind, identity)
        if key in _PROVENANCE_CACHE:
            return _PROVENANCE_CACHE[key]
    else:
        matches = [
            value
            for (
                cached_subject,
                cached_kind,
                cached_identity,
            ), value in _PROVENANCE_CACHE.items()
            if cached_kind == kind and cached_identity == identity
        ]
        if len(matches) == 1:
            return matches[0]
    if kind == "checkpoint":
        key = (subject or "untyped_discovery", kind, identity)
        _PROVENANCE_CACHE[key] = None
        return None
    snapshot = _authenticated_snapshot(identity)
    if snapshot is None:
        key = (subject or "untyped_discovery", kind, identity)
        _PROVENANCE_CACHE[key] = None
        return None
    try:
        document = json.loads(snapshot.raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        document = None
    parsed: _ParsedProvenance | None = None
    if (
        not isinstance(document, Mapping)
        or canonical_json_bytes(document) != snapshot.raw
    ):
        key = (subject or "untyped_discovery", kind, identity)
        _PROVENANCE_CACHE[key] = None
        return None
    document_subject = document.get("subject")
    key_subject = subject or document_subject
    key = (str(key_subject), kind, identity)
    if (
        type(document_subject) is not str
        or re.fullmatch(r"[a-z][a-z0-9_]{2,63}", document_subject) is None
        or (subject is not None and document_subject != subject)
    ):
        _PROVENANCE_CACHE[key] = None
        return None
    if kind in _MANIFEST_PROVENANCE_KINDS:
        ids = document.get("ids")
        if (
            set(document)
            != {"schema_version", "subject", "kind", "complete_manifest", "ids"}
            or document.get("schema_version") != "dive-exposure-provenance-v2"
            or document.get("kind") != kind
            or type(document.get("complete_manifest")) is not bool
        ):
            _PROVENANCE_CACHE[key] = None
            return None
        normalized = _canonical_ids(ids)
        if normalized is not None:
            parsed = _ParsedProvenance(
                kind,
                document_subject,
                snapshot,
                normalized,
                document["complete_manifest"],
            )
    if kind == "cutoff_evidence":
        cutoff_text = document.get("cutoff")
        if (
            set(document) == {"schema_version", "subject", "kind", "cutoff"}
            and document.get("schema_version") == "dive-exposure-provenance-v2"
            and document.get("kind") == kind
            and type(cutoff_text) is str
        ):
            cutoff = _strict_date(cutoff_text)
            if cutoff is not None:
                parsed = _ParsedProvenance(
                    kind, document_subject, snapshot, cutoff=cutoff
                )
    if kind == "cutoff_predicate":
        cutoff_mapping = document.get("cutoff_evidence")
        if (
            set(document)
            == {
                "schema_version",
                "subject",
                "evaluator_family",
                "condition",
                "cutoff_evidence",
            }
            and document.get("schema_version") == "dive-exposure-cutoff-predicate-v1"
            and type(document.get("evaluator_family")) is str
            and type(document.get("condition")) is str
            and isinstance(cutoff_mapping, Mapping)
        ):
            try:
                cutoff_identity = _identity_from_mapping(cutoff_mapping)
            except ExposureAuditError:
                cutoff_identity = None
            if cutoff_identity is not None:
                parsed = _ParsedProvenance(
                    kind,
                    document_subject,
                    snapshot,
                    evaluator_family=document["evaluator_family"],
                    condition=document["condition"],
                    cutoff_identity=cutoff_identity,
                )
    _PROVENANCE_CACHE[key] = parsed
    return parsed

def _authenticated_snapshot(identity: ArtifactIdentity) -> _Snapshot | None:

    path = Path(identity.path).resolve()
    cached = _SOURCE_SNAPSHOTS.get(path)
    if cached is not None:
        return cached if cached.identity == identity else None
    try:
        if not path.is_file() or path.is_symlink():
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    observed = ArtifactIdentity(str(path), hashlib.sha256(raw).hexdigest(), len(raw))
    if observed != identity:
        return None
    snapshot = _Snapshot(raw, observed)
    _SOURCE_SNAPSHOTS[path] = snapshot
    return snapshot

def _subject_provenance(
    subject: ExposureEvidence, kinds: set[str]
) -> tuple[_ParsedProvenance, ...]:
    registered = _registered_subject(subject)
    if registered is None:
        return ()
    return tuple(
        item
        for item in subject.provenance
        if item.kind == registered.kind
        and item.kind in kinds
        and item.subject == subject.subject
        and item.snapshot.identity == registered.source
        and item.snapshot.identity in subject.evidence
        and _parsed_is_authenticated(item)
    )

def _parsed_is_authenticated(item: _ParsedProvenance) -> bool:
    return (
        _PROVENANCE_CACHE.get((item.subject, item.kind, item.snapshot.identity)) is item
    )

def _registered_subject(subject: ExposureEvidence) -> ExposureSubjectContract | None:
    registry = subject._registry
    if registry is None or registry.seal is not _REGISTRY_SEAL:
        return None
    matches = tuple(
        item for item in registry.subjects if item.subject == subject.subject
    )
    return matches[0] if len(matches) == 1 else None

def _predicate_matches_rule(
    predicate: _ParsedProvenance,
    rule: CutoffPredicateContract,
    cutoff: ArtifactIdentity,
) -> bool:
    return (
        _parsed_is_authenticated(predicate)
        and predicate.kind == "cutoff_predicate"
        and predicate.subject == rule.subject
        and predicate.snapshot.identity == rule.source
        and predicate.evaluator_family == rule.evaluator_family
        and predicate.condition == rule.condition
        and predicate.cutoff_identity == cutoff
    )

def _valid_cutoff_predicates(
    subject: ExposureEvidence,
    cutoff: _ParsedProvenance,
    evaluator_family: str,
) -> bool:
    registry = subject._registry
    if registry is None or registry.seal is not _REGISTRY_SEAL:
        return False
    registered_rules = tuple(
        rule for rule in registry.cutoff_predicates if rule.subject == subject.subject
    )
    rules = tuple(
        rule for rule in registered_rules if rule.evaluator_family == evaluator_family
    )
    return (
        len(rules) == 1
        and registered_rules == subject.cutoff_predicate_rules
        and all(
            rule.source in subject.evidence
            and any(
                _predicate_matches_rule(predicate, rule, cutoff.snapshot.identity)
                for predicate in subject.cutoff_predicates
            )
            for rule in rules
        )
    )

def _provenance_search(
    contract: BenchmarkContract,
    *,
    model_card: ArtifactIdentity | None = None,
) -> tuple[dict[str, object], ...]:

    root = Path(contract.upstream_root)
    checkpoint_paths = tuple(
        Path(identity.path) for _, identity in contract.checkpoints
    )
    locations = (
        ("upstream_docs", root / "docs", ("**/*README*", "**/*TRAIN*", "**/*MODEL*")),
        ("upstream_model_card", root / "assets" / "model_card", ("**/*",)),
        (
            "upstream_configs",
            root / "configs",
            ("**/*train*.yaml", "**/*dataset*.yaml", "**/*evaluate*.yaml"),
        ),
        (
            "repository_config",
            Path(__file__).resolve().parents[3] / "configs" / "emergent",
            ("benchmark.yaml",),
        ),
        (
            "checkpoint_adjacent",
            None,
            ("*.json", "*.yaml", "*.yml", "*manifest*", "*README*"),
        ),
        (
            "cache_metadata",
            Path.home() / ".cache" / "torch" / "hub",
            ("*.json", "*.yaml", "*.yml", "*manifest*", "*README*"),
        ),
    )
    rows: list[dict[str, object]] = []
    declared: dict[str, list[tuple[str, str, ArtifactIdentity]]] = {}
    for item in contract.exposure_subjects:
        declared.setdefault(item.source.path, []).append(
            (item.subject, item.kind, item.source)
        )
    for item in contract.cutoff_predicates:
        declared.setdefault(item.source.path, []).append(
            (item.subject, "cutoff_predicate", item.source)
        )
    inspection_kinds = _DISCOVERY_PROVENANCE_KINDS
    for label, location, patterns in locations:
        roots = (
            tuple(path.parent for path in checkpoint_paths)
            if label == "checkpoint_adjacent"
            else (location,)
        )
        for search_root in roots:
            assert search_root is not None
            for pattern in patterns:
                rows.extend(
                    _search_pattern(
                        label,
                        search_root,
                        pattern,
                        declared,
                        inspection_kinds=inspection_kinds,
                        upstream_commit=contract.upstream_commit,
                    )
                )
    for name, identity in contract.inventories:
        rows.append(
            {
                "location": "recorded_external_inventory",
                "pattern": name,
                "path": identity.path,
                "result": "authenticated_inventory",
                "identity": _identity_mapping(identity),
                "conclusion": "records local metadata only; no complete manifest or certified cutoff recovered",
            }
        )
    if model_card is not None:
        rows.extend(
            _search_pattern(
                "official_model_card",
                Path(model_card.path).parent,
                Path(model_card.path).name,
                inspection_kinds=inspection_kinds,
                upstream_commit=contract.upstream_commit,
            )
        )
    for subject in contract.exposure_subjects:
        rows.append(
            _inspect_provenance_source(
                subject.subject, subject.kind, subject.source, contract
            )
        )
    return tuple(rows)

def _inspect_provenance_source(
    subject: str, kind: str, identity: ArtifactIdentity, contract: BenchmarkContract
) -> dict[str, object]:

    if kind == "checkpoint":
        return {
            "location": "registered_provenance_source",
            "subject": subject,
            "kind": kind,
            "command": ["contract_identity_only", identity.path],
            "parser_version": "dive-exposure-v5",
            "source_version": {
                "upstream_commit": contract.upstream_commit,
                "parser": "dive-exposure-v5",
            },
            "authority": "registered_contract_source",
            "identity": _identity_mapping(identity),
            "attempted_kind": None,
            "parsed_subject": None,
            "parsed_kind": None,
            "parsed_manifest_ids": [],
            "parsed_manifest_complete": False,
            "parsed_verified_cutoff": None,
            "conclusion": "unknown: checkpoint bytes are not a provenance document",
        }
    snapshot = _authenticated_snapshot(identity)
    if snapshot is None:
        raise ExposureAuditError(f"provenance source drifted for {subject}")
    parsed = _parse_subject_provenance(kind, identity, subject=subject)
    try:
        content = snapshot.raw.decode("utf-8")
    except UnicodeDecodeError:
        content = ""
    keywords = sorted(
        keyword
        for keyword in ("teddymer", "manifest", "cutoff", "pdb", "training")
        if keyword in content.lower()
    )
    return {
        "location": "registered_provenance_source",
        "subject": subject,
        "kind": kind,
        "command": [
            "in_process",
            "dive.benchmark.exposure._parse_subject_provenance",
            "--kind",
            kind,
            "--path",
            identity.path,
        ],
        "parser_version": "dive-exposure-v5",
        "source_version": {
            "upstream_commit": contract.upstream_commit,
            "parser": "dive-exposure-v5",
        },
        "authority": "registered_contract_source",
        "identity": _identity_mapping(identity),
        "content_bytes": len(snapshot.raw),
        "recognized_keywords": keywords,
        "attempted_kind": kind,
        "parsed_subject": None if parsed is None else parsed.subject,
        "parsed_kind": None if parsed is None else parsed.kind,
        "parsed_manifest_ids": []
        if parsed is None or parsed.manifest_ids is None
        else sorted(parsed.manifest_ids),
        "parsed_manifest_complete": False
        if parsed is None
        else parsed.manifest_complete,
        "parsed_verified_cutoff": None
        if parsed is None or parsed.cutoff is None
        else parsed.cutoff.isoformat(),
        "parsed_evaluator_family": (
            None if parsed is None else parsed.evaluator_family
        ),
        "parsed_condition": None if parsed is None else parsed.condition,
        "parsed_cutoff_identity": (
            None
            if parsed is None or parsed.cutoff_identity is None
            else _identity_mapping(parsed.cutoff_identity)
        ),
        "conclusion": "authenticated registered-source parser result",
    }

def _search_pattern(
    location: str,
    root: Path,
    pattern: str,
    declared: Mapping[str, Sequence[tuple[str, str, ArtifactIdentity]]] | None = None,
    *,
    inspection_kinds: Sequence[str] = (),
    upstream_commit: str | None = None,
) -> list[dict[str, object]]:
    base = {
        "location": location,
        "search_root": str(root),
        "pattern": pattern,
    }
    try:
        matches = sorted(root.glob(pattern))
    except PermissionError as error:
        return [
            {
                **base,
                "result": "permission_denied",
                "detail": str(error),
                "conclusion": "unknown",
            }
        ]
    except OSError as error:
        return [
            {
                **base,
                "result": "search_error",
                "detail": str(error),
                "conclusion": "unknown",
            }
        ]
    if not root.exists():
        return [{**base, "result": "missing", "conclusion": "unknown"}]
    rows = []
    for path in matches:
        if not path.is_file() or path.is_symlink():
            continue
        try:
            snapshot = _discovery_snapshot(path)
            if snapshot is None:
                raise OSError("cannot retain a regular-file snapshot")
            identity = snapshot.identity
        except (OSError, RuntimeError) as error:
            rows.append(
                {
                    **base,
                    "path": str(path),
                    "result": "unreadable",
                    "detail": str(error),
                    "conclusion": "unknown",
                }
            )
        else:
            declared_entries = tuple((declared or {}).get(identity.path, ()))
            if declared_entries:
                entries: tuple[tuple[str | None, str, ArtifactIdentity], ...] = tuple(
                    declared_entries
                )
                authority = "registered_contract_source"
            else:
                kinds = tuple(inspection_kinds)
                if not kinds and location in {
                    "training_manifest",
                    "cutoff_evidence",
                }:
                    kinds = (location,)
                entries = tuple((None, kind, identity) for kind in kinds)
                authority = "untyped_search_lead"
            if not entries:
                rows.append(
                    {
                        **base,
                        "path": identity.path,
                        "result": "read",
                        "identity": _identity_mapping(identity),
                        "command": ["retain_snapshot", identity.path],
                        "parser_version": "dive-exposure-v5",
                        "authority": authority,
                        "attempted_kind": None,
                        "attempted_kinds": [],
                        "parsed_kind": None,
                        "parsed_kinds": [],
                        "parsed_manifest_ids": [],
                        "parsed_manifest_complete": False,
                        "parsed_verified_cutoff": None,
                        "parsed_evaluator_family": None,
                        "parsed_condition": None,
                        "parsed_cutoff_identity": None,
                        "conclusion": "unknown",
                    }
                )
                continue
            parsed_entries = tuple(
                (
                    registered_subject,
                    registered_kind,
                    registered_identity,
                    _parse_subject_provenance(
                        registered_kind,
                        registered_identity,
                        subject=registered_subject,
                    ),
                )
                for registered_subject, registered_kind, registered_identity in entries
            )
            attempted_kinds = [
                kind for _subject, kind, _identity, _parsed in parsed_entries
            ]
            parsed_kinds = [
                parsed.kind
                for _subject, _kind, _identity, parsed in parsed_entries
                if parsed is not None
            ]
            for (
                registered_subject,
                registered_kind,
                registered_identity,
                parsed,
            ) in parsed_entries:
                rows.append(
                    {
                        **base,
                        "path": identity.path,
                        "result": "read",
                        "identity": _identity_mapping(identity),
                        "command": [
                            "in_process",
                            "dive.benchmark.exposure._parse_subject_provenance",
                            "--kind",
                            registered_kind,
                            "--path",
                            identity.path,
                        ],
                        "parser_version": "dive-exposure-v5",
                        "source_version": {
                            "upstream_commit": upstream_commit,
                            "parser": "dive-exposure-v5",
                        },
                        "authority": authority,
                        "declared_subject": registered_subject,
                        "attempted_kind": registered_kind,
                        "attempted_kinds": attempted_kinds,
                        "parsed_subject": None if parsed is None else parsed.subject,
                        "parsed_kind": None if parsed is None else parsed.kind,
                        "parsed_kinds": parsed_kinds,
                        "parsed_manifest_ids": (
                            []
                            if parsed is None or parsed.manifest_ids is None
                            else sorted(parsed.manifest_ids)
                        ),
                        "parsed_manifest_complete": (
                            False if parsed is None else parsed.manifest_complete
                        ),
                        "parsed_verified_cutoff": (
                            None
                            if parsed is None or parsed.cutoff is None
                            else parsed.cutoff.isoformat()
                        ),
                        "parsed_evaluator_family": (
                            None if parsed is None else parsed.evaluator_family
                        ),
                        "parsed_condition": None
                        if parsed is None
                        else parsed.condition,
                        "parsed_cutoff_identity": (
                            None
                            if parsed is None or parsed.cutoff_identity is None
                            else _identity_mapping(parsed.cutoff_identity)
                        ),
                        "conclusion": (
                            "authenticated registered-source parser result"
                            if authority == "registered_contract_source"
                            else "unknown"
                        ),
                    }
                )
    if not rows:
        rows.append({**base, "result": "no_matches", "conclusion": "unknown"})
    return rows

def _discovery_snapshot(path: Path) -> _Snapshot | None:

    if path.is_symlink():
        return None
    resolved = path.resolve()
    cached = _SOURCE_SNAPSHOTS.get(resolved)
    if cached is not None:
        return cached
    try:
        if not resolved.is_file():
            return None
        raw = resolved.read_bytes()
    except OSError:
        return None
    snapshot = _Snapshot(
        raw,
        ArtifactIdentity(str(resolved), hashlib.sha256(raw).hexdigest(), len(raw)),
    )
    _SOURCE_SNAPSHOTS[resolved] = snapshot
    return snapshot

def _fetch_model_card(
    evidence_dir: Path, run: BenchmarkRun
) -> tuple[ArtifactIdentity, ArtifactIdentity | None]:

    retrieved_utc = datetime.now(UTC).isoformat()
    card_path = evidence_dir / "official_model_card.md"
    record: dict[str, object] = {"url": _MODEL_CARD_URL, "retrieved_utc": retrieved_utc}
    card_identity: ArtifactIdentity | None = None
    try:
        request = urllib.request.Request(
            _MODEL_CARD_URL, headers={"User-Agent": "DIVE-provenance-audit/1"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            record["http_status"] = response.status
        _write_create_new_bytes(card_path, raw, run)
        card_identity = ArtifactIdentity(
            str(card_path.resolve()), hashlib.sha256(raw).hexdigest(), len(raw)
        )
        _SOURCE_SNAPSHOTS[card_path.resolve()] = _Snapshot(raw, card_identity)
        record.update(
            {
                "result": "retrieved",
                "response_size_bytes": len(raw),
                "response_sha256": card_identity.sha256,
                "artifact": _identity_mapping(card_identity),
            }
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        record.update(
            {
                "result": "network_failure",
                "http_status": getattr(error, "code", None),
                "detail": str(error),
                "response_size_bytes": None,
                "response_sha256": None,
            }
        )
    retrieval = _write_create_new_json(
        evidence_dir / "official_model_card.retrieval.json", record, run
    )
    return retrieval, card_identity

def _write_jsonl(
    path: Path, rows: Iterable[Mapping[str, object]], run: BenchmarkRun | None = None
) -> ArtifactIdentity:
    data = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    _write_create_new_bytes(path, data, run)
    return file_identity(path)

def _write_create_new_json(
    path: Path, payload: Mapping[str, object], run: BenchmarkRun | None = None
) -> ArtifactIdentity:
    _write_create_new_bytes(path, canonical_json_bytes(dict(payload)), run)
    return file_identity(path)

def _write_create_new_bytes(
    path: Path, data: bytes, run: BenchmarkRun | None = None
) -> None:
    if run is None:

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)
        return
    evidence_root = run.evidence_dir.parents[1]
    bulk_root = run.bulk_dir.parents[1]
    try:
        write_terminal_bytes(path, data, evidence_root, bulk_root)
    except Exception as error:
        raise ExposureAuditError(
            f"cannot create preflighted evidence {path}: {error}"
        ) from error

def _candidate_parent_id(candidate: Mapping[str, object] | object) -> str:
    if isinstance(candidate, Mapping):
        value = candidate.get("parent_id")
    else:
        value = getattr(candidate, "parent_id", None)
    if type(value) is not str or not value:
        raise ExposureAuditError("candidate has no parent_id")
    return value

def _candidate_pdb_ids(
    candidate: Mapping[str, object] | object,
) -> frozenset[str] | None:
    getter = (
        candidate.get
        if isinstance(candidate, Mapping)
        else lambda name: getattr(candidate, name, None)
    )
    if getter("candidate_set_authenticated") is not True:
        return None
    candidate_ids = getter("candidate_ids")
    neighbor_ids = getter("prohibited_neighbor_ids")
    if not isinstance(candidate_ids, list) or not isinstance(neighbor_ids, list):
        return None
    values = [*candidate_ids, *neighbor_ids]
    if (
        not values
        or not candidate_ids
        or not neighbor_ids
        or not all(type(item) is str for item in values)
    ):
        return None
    normalized = frozenset(item.lower() for item in values if _PDB_ID.fullmatch(item))
    return normalized if len(normalized) == len(values) else None

def _candidate_date(candidate: Mapping[str, object] | object) -> date | None:
    value = (
        candidate.get("deposition_date")
        if isinstance(candidate, Mapping)
        else getattr(candidate, "deposition_date", None)
    )
    if isinstance(value, date):
        return value
    return _strict_date(value)

def _candidate_family(candidate: Mapping[str, object] | object) -> str | None:
    value = (
        candidate.get("family")
        if isinstance(candidate, Mapping)
        else getattr(candidate, "family", None)
    )
    return (
        value if type(value) is str and value in {"binder", "ame", "antibody"} else None
    )

def _strict_date(value: object) -> date | None:
    if type(value) is not str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None

def _canonical_ids(values: object) -> frozenset[str] | None:
    if not isinstance(values, list) or not all(type(item) is str for item in values):
        return None
    normalized = frozenset(item.lower() for item in values if _PDB_ID.fullmatch(item))
    return normalized if len(normalized) == len(values) else None

def _record(
    parent_id: str,
    subject: ExposureEvidence,
    classification: ExposureClass,
    reason_code: str,
) -> ExposureRecord:
    return ExposureRecord(
        parent_id, subject.subject, classification, reason_code, subject.evidence
    )

def _record_mapping(record: ExposureRecord) -> dict[str, object]:
    return {
        "parent_id": record.parent_id,
        "subject": record.subject,
        "classification": record.classification.value,
        "reason_code": record.reason_code,
        "evidence": [_identity_mapping(item) for item in record.evidence],
    }

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

def _identity_from_mapping(value: Mapping[str, object]) -> ArtifactIdentity:
    if set(value) != {"path", "sha256", "size_bytes"}:
        raise ExposureAuditError("provenance identity has an invalid schema")
    path, digest, size = value["path"], value["sha256"], value["size_bytes"]
    if (
        type(path) is not str
        or not path
        or type(digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(size) is not int
        or size < 0
    ):
        raise ExposureAuditError("provenance identity has invalid values")
    return ArtifactIdentity(path, digest, size)
