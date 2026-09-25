
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from types import MappingProxyType
from urllib.parse import urlsplit

import yaml

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.external_catalog import (
    CandidateDeclaration,
    ExternalCandidateId,
    ExternalCatalog,
    ExternalCatalogError,
    load_external_catalog,
)
from dive.benchmark.external_eligibility import (
    ExternalEligibilityError,
    WeightReferenceDiscovery,
    decide_candidate,
    discover_weight_references,
    inspect_license,
    parse_reviewed_decisions_bytes,
    WeightLicenseJudgment,
)
from dive.benchmark.external_provenance import (
    AuditDisposition,
    CandidateObservation,
    ExternalAuditRun,
    ExternalProvenanceError,
    WeightBundleObservation,
    WeightComponentObservation,
    WeightContentClass,
    WeightEvidence,
    WeightLicenseStatus,
    WeightObservation,
    WeightReference,
    claim_external_audit,
    complete_external_audit,
    load_external_audit,
    resolve_official_source,
    write_candidate_observation,
)
import dive.benchmark.external_provenance as provenance
from dive.signed_value.roots import (
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_COMMIT,
    EMERGENT_UPSTREAM_ROOT,
    SIGNED_VALUE_UPSTREAM_COMMIT,
    UPSTREAM_ROOT,
)

class ExternalAuditCliError(RuntimeError):
    pass

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_PINNED_CATALOG_PATHS = MappingProxyType(
    {
        "dive-external-baseline-candidates-v1": (
            _REPOSITORY_ROOT
            / "configs"
            / "emergent"
            / "external_baseline_candidates.yaml"
        ),
        "dive-external-baseline-candidates-v2": (
            _REPOSITORY_ROOT
            / "configs"
            / "emergent"
            / "external_baseline_candidates_v2.yaml"
        ),
    }
)
_GIT_BINARY = Path("/usr/bin/git")
_REGISTRY_ROUTES_BY_CATALOG_SCHEMA = MappingProxyType(
    {
        "dive-external-baseline-candidates-v1": (
            Path("configs/emergent/external_baselines_resolved.yaml"),
            "dive-external-baselines-resolved-v1",
        ),
        "dive-external-baseline-candidates-v2": (
            Path("configs/emergent/external_baselines_resolved_v2.yaml"),
            "dive-external-baselines-resolved-v2",
        ),
    }
)
_DECISION_RECORD_SCHEMA = "dive-external-candidate-decision-v1"
_DECISION_COMPLETION_SCHEMA = "dive-external-decision-completion-v1"
_DECISION_REVIEW_SCHEMA = "dive-external-baseline-decisions-v1"
_DECISION_REVIEW_SCHEMA_V2 = "dive-external-baseline-decisions-v2"
_CATALOG_SCHEMA_V2 = "dive-external-baseline-candidates-v2"
_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_SENSITIVE_PROXY_FRAGMENTS = (
    "access_key",
    "api_key",
    "auth",
    "bearer",
    "credential",
    "password",
    "secret",
    "signature",
    "signed",
    "token",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

@dataclass(frozen=True, slots=True)
class ReviewedJudgment:

    candidate_id: ExternalCandidateId
    capability: str
    supported: bool
    relative_path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, ExternalCandidateId):
            raise ExternalAuditCliError("reviewed judgment candidate is invalid")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]{1,63}", self.capability):
            raise ExternalAuditCliError("reviewed judgment capability is invalid")
        if type(self.supported) is not bool:
            raise ExternalAuditCliError("reviewed judgment status is invalid")
        relative = Path(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or relative.as_posix() != self.relative_path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ExternalAuditCliError("reviewed judgment path is invalid")
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ExternalAuditCliError("reviewed judgment line span is invalid")

@dataclass(frozen=True, slots=True)
class ReviewedSourceSpan:

    relative_path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        relative = Path(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or relative.as_posix() != self.relative_path
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ExternalAuditCliError("reviewed source span path is invalid")
        if (
            type(self.start_line) is not int
            or type(self.end_line) is not int
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ExternalAuditCliError("reviewed source span line range is invalid")

@dataclass(frozen=True, slots=True)
class ReviewedWeightLicenseJudgment:

    candidate_id: ExternalCandidateId
    component_id: str
    status: WeightLicenseStatus
    evidence: tuple[ReviewedSourceSpan, ...]

    def __post_init__(self) -> None:
        if self.candidate_id is not ExternalCandidateId.RFANTIBODY_H3_V1:
            raise ExternalAuditCliError("weight license judgment candidate is invalid")
        expected = tuple(
            item[0]
            for item in provenance._WEIGHT_COMPONENTS_BY_CANDIDATE[self.candidate_id]
        )
        if self.component_id not in expected:
            raise ExternalAuditCliError("weight license judgment component is invalid")
        if self.status not in {
            WeightLicenseStatus.RESEARCH_EVALUATION_ALLOWED,
            WeightLicenseStatus.UNAVAILABLE,
        }:
            raise ExternalAuditCliError("weight license judgment status is invalid")
        if (
            not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(item, ReviewedSourceSpan) for item in self.evidence)
        ):
            raise ExternalAuditCliError("weight license judgment evidence is invalid")

@dataclass(frozen=True, slots=True)
class DecidedCandidate:

    observation: CandidateObservation
    final_observation: CandidateObservation
    disposition: AuditDisposition
    exposure: str

    def __post_init__(self) -> None:
        if self.final_observation.candidate_id is not self.observation.candidate_id:
            raise ExternalAuditCliError("decided candidate identity drifted")
        if self.final_observation.disposition is not self.disposition:
            raise ExternalAuditCliError("decided candidate disposition drifted")
        if self.exposure not in {"known_overlap", "known_disjoint", "unknown"}:
            raise ExternalAuditCliError("training exposure class is invalid")

def audit_external_baselines(argv: Sequence[str]) -> int:

    arguments = _parser().parse_args(argv)
    try:
        _require_cpu_only_environment()
        catalog = _load_catalog_for_schema(arguments.catalog_schema)
        if arguments.command == "observe":
            completion = _observe(arguments.run_id, catalog)
            _print_identity("completion", completion, run_id=arguments.run_id)
        elif arguments.command == "prepare-decisions":
            _production_preflight()
            observation = load_external_audit(
                arguments.observation_run,
                catalog,
                expected_completion=_identity_from_arguments(
                    arguments, "observation_completion"
                ),
            )
            judgments = tuple(_judgment_from_json(item) for item in arguments.judgment)
            weight_license_judgments = tuple(
                _weight_license_judgment_from_json(item)
                for item in arguments.weight_license_judgment
            )
            identity = prepare_reviewed_decisions(
                review_id=arguments.run_id,
                observation_run=observation,
                judgments=judgments,
                weight_license_judgments=weight_license_judgments,
            )
            _print_identity("decisions", identity, run_id=arguments.run_id)
        elif arguments.command == "decide":
            completion = _decide(
                run_id=arguments.run_id,
                observation_run_id=arguments.observation_run,
                observation_completion=_identity_from_arguments(
                    arguments, "observation_completion"
                ),
                decisions_path=arguments.decisions,
                catalog=catalog,
            )
            _print_identity("completion", completion, run_id=arguments.run_id)
        elif arguments.command == "build-registry":
            _production_preflight()
            run = load_completed_decision_audit(
                arguments.run_id,
                catalog,
                expected_completion=_identity_from_arguments(
                    arguments, "decision_completion"
                ),
            )
            identity = build_resolved_registry(run, arguments.output)
            _print_identity("registry", identity, run_id=arguments.run_id)
        else:
            raise ExternalAuditCliError("unknown external audit command")
        return 0
    except (
        ExternalAuditCliError,
        ExternalCatalogError,
        ExternalEligibilityError,
        ExternalProvenanceError,
        OSError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"external audit refused: {error}", file=sys.stderr)
        return 2

def prepare_reviewed_decisions(
    *,
    review_id: str,
    observation_run: ExternalAuditRun,
    judgments: Sequence[ReviewedJudgment],
    weight_license_judgments: Sequence[ReviewedWeightLicenseJudgment] = (),
) -> ArtifactIdentity:

    provenance._validate_run_id(review_id)
    if observation_run.completion is None:
        raise ExternalAuditCliError("observation run lacks its completion identity")
    catalog = _catalog_for_run(observation_run)
    authenticated = load_external_audit(
        observation_run.run_id,
        catalog,
        expected_completion=observation_run.completion,
    )
    _require_production_repo_commit(authenticated.repo_commit)
    if authenticated.argv != ("audit", "observe"):
        raise ExternalAuditCliError("review input is not an observation run")
    supplied = tuple(judgments)
    if any(not isinstance(item, ReviewedJudgment) for item in supplied):
        raise ExternalAuditCliError("review judgments contain an invalid record")
    by_candidate: dict[ExternalCandidateId, list[ReviewedJudgment]] = {}
    for judgment in supplied:
        by_candidate.setdefault(judgment.candidate_id, []).append(judgment)
    supplied_weight_licenses = tuple(weight_license_judgments)
    if any(
        not isinstance(item, ReviewedWeightLicenseJudgment)
        for item in supplied_weight_licenses
    ):
        raise ExternalAuditCliError(
            "weight license judgments contain an invalid record"
        )
    licenses_by_candidate: dict[
        ExternalCandidateId, list[ReviewedWeightLicenseJudgment]
    ] = {}
    for judgment in supplied_weight_licenses:
        licenses_by_candidate.setdefault(judgment.candidate_id, []).append(judgment)

    observations = {item.candidate_id: item for item in authenticated.observations}
    decisions: list[dict[str, object]] = []
    expected_candidates: set[ExternalCandidateId] = set()
    for declaration in catalog.candidates:
        observation = observations[declaration.candidate_id]
        reviewable = not (
            observation.source is None
            or observation.license is None
            or observation.license.artifact is None
        )
        if (
            reviewable
            and declaration.candidate_id is ExternalCandidateId.RFANTIBODY_H3_V1
            and catalog.schema_version == _CATALOG_SCHEMA_V2
        ):
            reviewable = _rfantibody_weight_license_reviewable(observation.weight)
        if reviewable:
            expected_candidates.add(declaration.candidate_id)
        else:
            if declaration.candidate_id in licenses_by_candidate:
                raise ExternalAuditCliError(
                    "weight license judgments require a complete RFantibody bundle"
                )
            continue
        candidate_judgments = by_candidate.get(declaration.candidate_id, [])
        if tuple(item.capability for item in candidate_judgments) != (
            declaration.required_capabilities
        ):
            raise ExternalAuditCliError(
                f"review judgments are incomplete or unordered: "
                f"{declaration.candidate_id.value}"
            )
        decision: dict[str, object] = {
            "candidate_id": declaration.candidate_id.value,
            "source_commit": observation.source.commit,
            "source_tree_sha256": observation.source.tracked_tree_sha256,
            "capabilities": [
                {
                    "capability": item.capability,
                    "status": "supported" if item.supported else "unsupported",
                    "evidence": {
                        "path": item.relative_path,
                        "start_line": item.start_line,
                        "end_line": item.end_line,
                    },
                }
                for item in candidate_judgments
            ],
        }
        if catalog.schema_version == _CATALOG_SCHEMA_V2:
            candidate_licenses = licenses_by_candidate.pop(declaration.candidate_id, [])
            if declaration.candidate_id is ExternalCandidateId.RFANTIBODY_H3_V1:
                expected_components = tuple(
                    item[0]
                    for item in provenance._WEIGHT_COMPONENTS_BY_CANDIDATE[
                        declaration.candidate_id
                    ]
                )
                if (
                    tuple(item.component_id for item in candidate_licenses)
                    != expected_components
                ):
                    raise ExternalAuditCliError(
                        "weight license judgments are incomplete or unordered"
                    )
            elif candidate_licenses:
                raise ExternalAuditCliError(
                    "non-RFantibody weight licenses are invalid"
                )
            decision["weight_licenses"] = [
                {
                    "candidate_id": item.candidate_id.value,
                    "component_id": item.component_id,
                    "status": item.status.value,
                    "evidence": [
                        {
                            "relative_path": span.relative_path,
                            "start_line": span.start_line,
                            "end_line": span.end_line,
                        }
                        for span in item.evidence
                    ],
                }
                for item in candidate_licenses
            ]
        decisions.append(decision)
    if set(by_candidate) != expected_candidates:
        raise ExternalAuditCliError("review judgment candidate membership is invalid")
    if licenses_by_candidate:
        raise ExternalAuditCliError(
            "weight license judgment candidate membership is invalid"
        )

    evidence_root, _ = provenance._validate_fixed_roots()
    external_root = provenance._ensure_named_directory(
        evidence_root, "external_baselines"
    )
    reviews_root = provenance._ensure_named_directory(external_root, "reviews")
    review_dir = reviews_root / review_id
    try:
        provenance._mkdir_create_new(
            review_dir, "external review evidence", root=evidence_root
        )
    except FileExistsError as error:
        raise ExternalAuditCliError("review evidence run already exists") from error
    payload = {
        "schema_version": (
            _DECISION_REVIEW_SCHEMA_V2
            if catalog.schema_version == _CATALOG_SCHEMA_V2
            else _DECISION_REVIEW_SCHEMA
        ),
        "reviewed_at": _timestamp_now(),
        "decisions": decisions,
    }
    raw = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    destination = review_dir / "reviewed-decisions.yaml"
    identity = _write_new_bytes(destination, raw, root=evidence_root)

    sources = {
        declaration.candidate_id: observations[declaration.candidate_id].source
        for declaration in catalog.candidates
        if declaration.candidate_id in expected_candidates
    }
    parse_reviewed_decisions_bytes(
        raw,
        sources,
        catalog_schema_version=catalog.schema_version,
    )
    return identity

def load_completed_decision_audit(
    run_id: str,
    catalog: ExternalCatalog,
    *,
    expected_completion: ArtifactIdentity,
) -> ExternalAuditRun:

    provenance._validate_run_id(run_id)
    evidence_root, bulk_root = provenance._validate_fixed_roots()
    evidence_dir = evidence_root / "external_baselines" / run_id
    bulk_dir = bulk_root / "external_baselines" / run_id
    completion = provenance._read_authenticated_canonical_mapping_at(
        evidence_dir,
        Path("completion.json"),
        "decision completion",
        expected_completion,
    )
    completion = _exact_mapping(
        completion,
        {
            "schema_version",
            "status",
            "run_id",
            "catalog_sha256",
            "attempt",
            "observation_run",
            "observation_completion",
            "decisions_input",
            "decisions",
        },
        "decision completion",
    )
    completed_attempt = _artifact_from_mapping(
        completion["attempt"], "decision attempt"
    )
    attempt_payload = provenance._read_authenticated_canonical_mapping_at(
        evidence_root,
        Path("external_baselines") / run_id / "attempt.json",
        "decision attempt",
        completed_attempt,
    )
    run = ExternalAuditRun.from_mapping(
        attempt_payload,
        catalog=catalog,
        attempt=completed_attempt,
        evidence_dir=evidence_dir,
        bulk_dir=bulk_dir,
    )
    provenance._authenticate_run(run, require_building=False, recheck_execution=False)
    if run.argv != ("audit", "decide"):
        raise ExternalAuditCliError("completed run is not a decision audit")
    if os.listdir(run.sources_dir):
        raise ExternalAuditCliError("decision audit sources directory is not empty")
    if (
        completion["schema_version"] != _DECISION_COMPLETION_SCHEMA
        or completion["status"] != "COMPLETE"
        or completion["run_id"] != run_id
        or completion["catalog_sha256"] != catalog.semantic_sha256
        or _artifact_from_mapping(completion["attempt"], "decision attempt")
        != run.attempt
    ):
        raise ExternalAuditCliError("decision completion does not bind its run")
    observation_completion = _artifact_from_mapping(
        completion["observation_completion"], "observation completion"
    )
    observation_run_id = _required_string(
        completion, "observation_run", "decision completion"
    )
    observation_run = load_external_audit(
        observation_run_id,
        catalog,
        expected_completion=observation_completion,
    )
    decisions_input = _artifact_from_mapping(
        completion["decisions_input"], "reviewed decisions"
    )
    actual_decisions, _ = _read_reviewed_decisions_input(Path(decisions_input.path))
    if actual_decisions != decisions_input:
        raise ExternalAuditCliError("reviewed decision input identity drifted")
    declared = _exact_mapping(
        completion["decisions"],
        {item.candidate_id.value for item in catalog.candidates},
        "completed candidate decisions",
    )
    observed_by_id = {item.candidate_id: item for item in observation_run.observations}
    finals: list[CandidateObservation] = []
    for declaration in catalog.candidates:
        candidate_id = declaration.candidate_id
        identity = _artifact_from_mapping(
            declared[candidate_id.value],
            f"candidate decision {candidate_id.value}",
        )
        expected_path = (
            evidence_dir / "candidates" / candidate_id.value / "decision.json"
        )
        if identity.path != str(expected_path):
            raise ExternalAuditCliError("candidate decision is outside its fixed slot")
        raw = provenance._read_authenticated_canonical_mapping_at(
            evidence_dir,
            Path("candidates") / candidate_id.value / "decision.json",
            f"candidate decision {candidate_id.value}",
            identity,
        )
        record = _exact_mapping(
            raw,
            {
                "schema_version",
                "candidate_id",
                "observation",
                "observation_identity",
                "final_observation",
                "exposure",
            },
            "candidate decision",
        )
        original = observed_by_id[candidate_id]
        original_identity = _artifact_from_mapping(
            record["observation_identity"], "source observation identity"
        )
        if (
            record["schema_version"] != _DECISION_RECORD_SCHEMA
            or record["candidate_id"] != candidate_id.value
            or record["exposure"] != "unknown"
            or CandidateObservation.from_mapping(record["observation"]) != original
            or original_identity
            != _read_identity(
                observation_run.evidence_dir
                / "candidates"
                / candidate_id.value
                / "observation.json",
                "source observation",
            )
        ):
            raise ExternalAuditCliError(
                "candidate decision observation binding drifted"
            )
        final = CandidateObservation.from_mapping(record["final_observation"])
        provenance._validate_observation_binding(observation_run, declaration, final)
        finals.append(final)
    object.__setattr__(run, "observations", tuple(finals))
    object.__setattr__(run, "completion", expected_completion)
    return run

def build_resolved_registry(run: ExternalAuditRun, output: Path) -> ArtifactIdentity:

    if not isinstance(run, ExternalAuditRun) or run.completion is None:
        raise ExternalAuditCliError("registry requires a completed decision audit")
    catalog = _catalog_for_run(run)
    authenticated = load_completed_decision_audit(
        run.run_id, catalog, expected_completion=run.completion
    )
    _require_production_repo_commit(authenticated.repo_commit)
    blocked = [
        item.candidate_id.value
        for item in authenticated.observations
        if item.disposition is AuditDisposition.REVIEW_REQUIRED
    ]
    if blocked:
        raise ExternalAuditCliError(
            f"review_required blocks resolved registry publication: {blocked}"
        )
    output = Path(output)
    registry_relative, registry_schema = _registry_route_for_catalog(catalog)
    expected_output = _REPOSITORY_ROOT / registry_relative
    if output != expected_output:
        raise ExternalAuditCliError(
            "resolved registry output path is not the closed path"
        )

    decision_completion = provenance._read_authenticated_canonical_mapping_at(
        authenticated.evidence_dir,
        Path("completion.json"),
        "decision completion",
        authenticated.completion,
    )
    observation_completion = _artifact_from_mapping(
        decision_completion["observation_completion"], "observation completion"
    )
    decisions_input = _artifact_from_mapping(
        decision_completion["decisions_input"], "reviewed decisions"
    )
    catalog_identity = _read_identity(
        _catalog_path_for_schema(catalog.schema_version), "pinned catalog"
    )
    declarations = {item.candidate_id: item for item in catalog.candidates}
    candidates: list[dict[str, object]] = []
    for observation in authenticated.observations:
        declaration = declarations[observation.candidate_id]
        source = observation.source
        candidates.append(
            {
                "candidate_id": observation.candidate_id.value,
                "family": declaration.family.value,
                "source": (
                    None
                    if source is None
                    else {
                        "url": source.url,
                        "commit": source.commit,
                        "tracked_tree_sha256": source.tracked_tree_sha256,
                        "retrieval_log": _identity_mapping(source.retrieval_log),
                        "git_binary": _identity_mapping(source.git_binary),
                        "git_lfs_pointer_paths": list(source.git_lfs_pointer_paths),
                    }
                ),
                "license": (
                    None
                    if observation.license is None
                    else observation.license.as_mapping()
                ),
                "weight": observation.weight.as_mapping(),
                "exposure": "unknown",
                "capability": (
                    None
                    if observation.capability is None
                    else observation.capability.as_mapping()
                ),
                "final_disposition": observation.disposition.value,
            }
        )
    payload = {
        "schema_version": registry_schema,
        "catalog": {
            "schema_version": catalog.schema_version,
            "semantic_sha256": catalog.semantic_sha256,
            "artifact": _identity_mapping(catalog_identity),
        },
        "evidence": {
            "observation_run": decision_completion["observation_run"],
            "observation_completion": _identity_mapping(observation_completion),
            "decision_run": authenticated.run_id,
            "decision_completion": _identity_mapping(authenticated.completion),
            "reviewed_decisions": _identity_mapping(decisions_input),
        },
        "candidates": candidates,
    }
    _reject_secret_like_content(payload)
    raw = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    return _write_new_bytes(output, raw, root=_REPOSITORY_ROOT)

def _observe(run_id: str, catalog: ExternalCatalog) -> ArtifactIdentity:
    _production_preflight()
    run = claim_external_audit(run_id, ["audit", "observe"], catalog)
    for declaration in catalog.candidates:
        try:
            source = resolve_official_source(declaration, run, git_binary=_GIT_BINARY)
            license_observation = inspect_license(source, declaration)
            try:
                references = discover_weight_references(source)
            except ExternalEligibilityError as error:
                if not _is_weight_reference_review_error(error):
                    raise
                weight = _weight_review_required_observation(declaration)
            else:
                weight = _weight_observation(declaration, references)
            disposition = (
                AuditDisposition.LICENSE_UNAVAILABLE
                if license_observation.review_class == "unavailable"
                else AuditDisposition.REVIEW_REQUIRED
            )
            observation = CandidateObservation(
                candidate_id=declaration.candidate_id,
                source=source,
                license=license_observation,
                weight=weight,
                capability=None,
                disposition=disposition,
                observed_at=_timestamp_now(),
            )
        except (
            ExternalEligibilityError,
            ExternalProvenanceError,
            OSError,
        ) as error:
            observation = _source_failure_observation(declaration, error)
        write_candidate_observation(run, observation)
    return complete_external_audit(run)

def _read_reviewed_decisions_input(path: Path) -> tuple[ArtifactIdentity, bytes]:
    path = Path(path)
    evidence_root, _ = provenance._validate_fixed_roots()
    if not path.is_absolute():
        raise ExternalAuditCliError("reviewed decisions path must be absolute")
    try:
        relative = path.relative_to(evidence_root)
    except ValueError as error:
        raise ExternalAuditCliError(
            "reviewed decisions path is outside the fixed review namespace"
        ) from error
    if (
        len(relative.parts) != 4
        or relative.parts[:2] != ("external_baselines", "reviews")
        or relative.parts[3] != "reviewed-decisions.yaml"
    ):
        raise ExternalAuditCliError(
            "reviewed decisions path is outside the fixed review namespace"
        )
    review_id = relative.parts[2]
    provenance._validate_run_id(review_id)
    raw = provenance._read_regular_bytes_at(
        evidence_root, relative, "reviewed decisions"
    )
    return (
        ArtifactIdentity(
            str(evidence_root / relative),
            hashlib.sha256(raw).hexdigest(),
            len(raw),
        ),
        raw,
    )

def _decide(
    *,
    run_id: str,
    observation_run_id: str,
    observation_completion: ArtifactIdentity,
    decisions_path: Path,
    catalog: ExternalCatalog,
) -> ArtifactIdentity:
    _production_preflight()
    observation_run = load_external_audit(
        observation_run_id,
        catalog,
        expected_completion=observation_completion,
    )
    if observation_run.argv != ("audit", "observe"):
        raise ExternalAuditCliError("decision input is not an observation audit")
    _require_production_repo_commit(observation_run.repo_commit)
    decisions_identity, decisions_raw = _read_reviewed_decisions_input(decisions_path)
    decided = _load_decisions_for_observation(observation_run, decisions_raw)
    run = claim_external_audit(run_id, ["audit", "decide"], catalog)
    identities: dict[str, dict[str, object]] = {}
    for row in decided:
        candidate_id = row.observation.candidate_id
        observation_path = (
            observation_run.evidence_dir
            / "candidates"
            / candidate_id.value
            / "observation.json"
        )
        payload = {
            "schema_version": _DECISION_RECORD_SCHEMA,
            "candidate_id": candidate_id.value,
            "observation": row.observation.as_mapping(),
            "observation_identity": _identity_mapping(
                _read_identity(observation_path, "source observation")
            ),
            "final_observation": row.final_observation.as_mapping(),
            "exposure": row.exposure,
        }
        destination = (
            run.evidence_dir / "candidates" / candidate_id.value / "decision.json"
        )
        provenance._recheck_execution_binding(run.repo_commit)
        identity = provenance._write_new_json(
            destination, payload, root=run.evidence_dir
        )
        identities[candidate_id.value] = _identity_mapping(identity)
    completion_payload = {
        "schema_version": _DECISION_COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "run_id": run.run_id,
        "catalog_sha256": catalog.semantic_sha256,
        "attempt": _identity_mapping(run.attempt),
        "observation_run": observation_run.run_id,
        "observation_completion": _identity_mapping(observation_completion),
        "decisions_input": _identity_mapping(decisions_identity),
        "decisions": identities,
    }
    provenance._recheck_execution_binding(run.repo_commit)
    completion = provenance._write_new_json(
        run.evidence_dir / "completion.json",
        completion_payload,
        root=run.evidence_dir,
    )
    object.__setattr__(
        run, "observations", tuple(row.final_observation for row in decided)
    )
    object.__setattr__(run, "completion", completion)
    return completion

def _load_decisions_for_observation(
    observation_run: ExternalAuditRun,
    decisions_raw: bytes,
) -> tuple[DecidedCandidate, ...]:
    catalog = _catalog_for_run(observation_run)
    eligible_sources = {
        item.candidate_id: item.source
        for item in observation_run.observations
        if item.source is not None
        and item.license is not None
        and item.license.artifact is not None
        and (
            catalog.schema_version != _CATALOG_SCHEMA_V2
            or item.candidate_id is not ExternalCandidateId.RFANTIBODY_H3_V1
            or _rfantibody_weight_license_reviewable(item.weight)
        )
    }
    decisions = parse_reviewed_decisions_bytes(
        decisions_raw,
        eligible_sources,
        catalog_schema_version=catalog.schema_version,
    )
    by_id = dict(zip(eligible_sources, decisions, strict=True))
    declarations = {item.candidate_id: item for item in catalog.candidates}
    rows: list[DecidedCandidate] = []
    for observation in observation_run.observations:
        declaration = declarations[observation.candidate_id]
        reviewed_decision = by_id.get(observation.candidate_id)
        if reviewed_decision is None:
            final = observation
        else:
            if observation.source is None or observation.license is None:
                raise ExternalAuditCliError("capability decision lacks source evidence")
            weight = _apply_weight_license_judgments(
                observation.weight, reviewed_decision.weight_licenses
            )
            disposition = decide_candidate(
                declaration,
                observation.source,
                observation.license,
                weight,
                reviewed_decision.capability,
            )
            final = CandidateObservation(
                candidate_id=observation.candidate_id,
                source=observation.source,
                license=observation.license,
                weight=weight,
                capability=reviewed_decision.capability,
                disposition=disposition,
                observed_at=_timestamp_now(),
            )
        provenance._validate_observation_binding(observation_run, declaration, final)
        rows.append(
            DecidedCandidate(
                observation=observation,
                final_observation=final,
                disposition=final.disposition,
                exposure="unknown",
            )
        )
    return tuple(rows)

def _apply_weight_license_judgments(
    weight: WeightEvidence,
    judgments: tuple[WeightLicenseJudgment, ...],
) -> WeightEvidence:

    if not isinstance(weight, WeightBundleObservation):
        if judgments:
            raise ExternalAuditCliError(
                "singleton weights cannot receive license judgments"
            )
        return weight
    expected = tuple(item.component_id for item in weight.components)
    if tuple(item.component_id for item in judgments) != expected:
        raise ExternalAuditCliError("weight license judgment coverage is invalid")
    by_component = {item.component_id: item for item in judgments}
    return WeightBundleObservation(
        components=tuple(
            WeightComponentObservation(
                component_id=component.component_id,
                official_url=component.official_url,
                artifact=component.artifact,
                unavailable_reason=component.unavailable_reason,
                observed_at=component.observed_at,
                citation=component.citation,
                content_class=component.content_class,
                license_status=by_component[component.component_id].status,
                license_evidence=by_component[component.component_id].evidence,
            )
            for component in weight.components
        ),
        non_task_references=weight.non_task_references,
    )

def _rfantibody_weight_license_reviewable(weight: WeightEvidence) -> bool:

    return isinstance(weight, WeightBundleObservation) and all(
        component.citation is not None for component in weight.components
    )

def _weight_observation(
    declaration: CandidateDeclaration, references: Sequence[WeightReference]
) -> WeightEvidence:
    if not isinstance(declaration, CandidateDeclaration):
        raise ExternalAuditCliError("weight observation requires a declaration")
    if declaration.candidate_id is ExternalCandidateId.RFANTIBODY_H3_V1:
        return _rfantibody_weight_observation(references)
    unique_urls = tuple(dict.fromkeys(item.url for item in references))
    rejected_reference_count = (
        references.rejected_reference_count
        if isinstance(references, WeightReferenceDiscovery)
        else 0
    )
    if len(unique_urls) == 1:
        citation = next(item for item in references if item.url == unique_urls[0])
        return WeightObservation(
            official_url=unique_urls[0],
            artifact=None,
            unavailable_reason=_weight_reason(
                "not_retrieved", rejected_reference_count
            ),
            observed_at=_timestamp_now(),
            citation=citation,
            content_class=WeightContentClass.NOT_RETAINED,
        )
    reason = (
        "no_official_weight_reference"
        if not unique_urls
        else "ambiguous_weight_references"
    )
    return WeightObservation(
        official_url=None,
        artifact=None,
        unavailable_reason=_weight_reason(reason, rejected_reference_count),
        observed_at=_timestamp_now(),
        citation=None,
        content_class=WeightContentClass.NOT_RETAINED,
    )

def _rfantibody_weight_observation(
    references: Sequence[WeightReference],
) -> WeightBundleObservation:
    required = provenance._WEIGHT_COMPONENTS_BY_CANDIDATE[
        ExternalCandidateId.RFANTIBODY_H3_V1
    ]
    components: list[WeightComponentObservation] = []
    required_urls = {url for _, url in required}
    for component_id, url in required:
        citations = tuple(item for item in references if item.url == url)
        reason = (
            "not_retrieved"
            if len(citations) == 1
            else "missing_required_reference"
            if not citations
            else "ambiguous_required_references"
        )
        components.append(
            WeightComponentObservation(
                component_id=component_id,
                official_url=url,
                artifact=None,
                unavailable_reason=reason,
                observed_at=_timestamp_now(),
                citation=citations[0] if len(citations) == 1 else None,
                content_class=WeightContentClass.NOT_RETAINED,
                license_status=WeightLicenseStatus.UNREVIEWED,
                license_evidence=(),
            )
        )
    return WeightBundleObservation(
        components=tuple(components),
        non_task_references=tuple(
            item for item in references if item.url not in required_urls
        ),
    )

def _weight_reason(base: str, rejected_reference_count: int) -> str:
    if rejected_reference_count:
        return f"{base}_unsafe_references_{rejected_reference_count}"
    return base

def _is_weight_reference_review_error(error: ExternalEligibilityError) -> bool:
    return str(error).lower().startswith("weight reference") or str(
        error
    ).lower().startswith("unsafe official-source weight reference")

def _weight_review_required_observation(
    declaration: CandidateDeclaration,
) -> WeightEvidence:
    if declaration.candidate_id is ExternalCandidateId.RFANTIBODY_H3_V1:
        return _rfantibody_weight_observation(())
    return WeightObservation(
        official_url=None,
        artifact=None,
        unavailable_reason="weight_reference_review_required",
        observed_at=_timestamp_now(),
        citation=None,
        content_class=WeightContentClass.NOT_RETAINED,
    )

def _source_failure_observation(
    declaration: CandidateDeclaration,
    error: Exception,
) -> CandidateObservation:
    lowered = str(error).lower()
    source_unavailable = any(
        fragment in lowered
        for fragment in (
            "clone source",
            "official source remote",
            "resolve official source",
            "source remote head",
            "source unavailable",
        )
    )
    disposition = (
        AuditDisposition.SOURCE_UNAVAILABLE
        if source_unavailable
        else AuditDisposition.PROVENANCE_FAILED
    )
    return CandidateObservation(
        candidate_id=declaration.candidate_id,
        source=None,
        license=None,
        weight=_weight_review_required_observation(declaration)
        if declaration.candidate_id is ExternalCandidateId.RFANTIBODY_H3_V1
        else WeightObservation(
            official_url=None,
            artifact=None,
            unavailable_reason=disposition.value,
            observed_at=_timestamp_now(),
            citation=None,
            content_class=WeightContentClass.NOT_RETAINED,
        ),
        capability=None,
        disposition=disposition,
        observed_at=_timestamp_now(),
    )

def _production_preflight() -> None:
    if provenance._TEST_ROOTS is not None:
        return
    provenance._current_clean_commit()
    for root, expected in (
        (UPSTREAM_ROOT, SIGNED_VALUE_UPSTREAM_COMMIT),
        (EMERGENT_UPSTREAM_ROOT, EMERGENT_UPSTREAM_COMMIT),
    ):
        actual = subprocess.run(
            (str(_GIT_BINARY), "-C", str(root), "rev-parse", "HEAD^{commit}"),
            check=True,
            capture_output=True,
            text=True,
            env=provenance._GIT_ENVIRONMENT,
        ).stdout.strip()
        if actual != expected:
            raise ExternalAuditCliError(f"pinned upstream drifted: {root.name}")
    for root in (EMERGENT_EVIDENCE_ROOT, EMERGENT_BULK_ROOT):
        if shutil.disk_usage(root).free < 100 * 1024**3:
            raise ExternalAuditCliError(f"storage reserve is below 100 GiB: {root}")
    own_pid = os.getpid()
    runtime = Path(sys.executable).resolve()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            executable = (entry / "exe").resolve(strict=True)
            if executable != runtime:
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if b"audit_external_baselines.py" in command:
            raise ExternalAuditCliError("another external audit worker is active")

def _require_production_repo_commit(expected: str) -> None:
    if (
        provenance._TEST_ROOTS is None
        and provenance._current_clean_commit() != expected
    ):
        raise ExternalAuditCliError("repository commit differs from the anchored run")

def _require_cpu_only_environment() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise ExternalAuditCliError("CUDA_VISIBLE_DEVICES must be absent")
    for name in _PROXY_VARIABLES:
        value = os.environ.get(name)
        if value and _proxy_contains_credentials(value):
            raise ExternalAuditCliError(
                f"credential-bearing proxy variable is forbidden: {name}"
            )

def _proxy_contains_credentials(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    if parsed.username is not None or parsed.password is not None:
        return True
    lowered = value.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_PROXY_FRAGMENTS)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    observe = subparsers.add_parser("observe")
    observe.add_argument(
        "--catalog-schema", choices=tuple(_PINNED_CATALOG_PATHS), required=True
    )
    observe.add_argument("--run-id", required=True)

    prepare = subparsers.add_parser("prepare-decisions")
    prepare.add_argument(
        "--catalog-schema", choices=tuple(_PINNED_CATALOG_PATHS), required=True
    )
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--observation-run", required=True)
    _add_identity_arguments(prepare, "observation-completion")
    prepare.add_argument("--judgment", action="append", required=True)
    prepare.add_argument("--weight-license-judgment", action="append", default=[])

    decide = subparsers.add_parser("decide")
    decide.add_argument(
        "--catalog-schema", choices=tuple(_PINNED_CATALOG_PATHS), required=True
    )
    decide.add_argument("--run-id", required=True)
    decide.add_argument("--observation-run", required=True)
    _add_identity_arguments(decide, "observation-completion")
    decide.add_argument("--decisions", type=Path, required=True)

    build = subparsers.add_parser("build-registry")
    build.add_argument(
        "--catalog-schema", choices=tuple(_PINNED_CATALOG_PATHS), required=True
    )
    build.add_argument("--run-id", required=True)
    _add_identity_arguments(build, "decision-completion")
    build.add_argument("--output", type=Path, required=True)
    return parser

def _add_identity_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-path", type=Path, required=True)
    parser.add_argument(f"--{prefix}-sha256", required=True)
    parser.add_argument(f"--{prefix}-size", type=int, required=True)

def _identity_from_arguments(arguments, prefix: str) -> ArtifactIdentity:
    path = getattr(arguments, f"{prefix}_path")
    sha256 = getattr(arguments, f"{prefix}_sha256")
    size = getattr(arguments, f"{prefix}_size")
    if not path.is_absolute() or _SHA256.fullmatch(sha256) is None or size < 0:
        raise ExternalAuditCliError(f"{prefix} identity is invalid")
    return ArtifactIdentity(str(path), sha256, size)

def _judgment_from_json(value: str) -> ReviewedJudgment:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise ExternalAuditCliError("judgment JSON is malformed") from error
    mapping = _exact_mapping(
        raw,
        {
            "candidate_id",
            "capability",
            "supported",
            "relative_path",
            "start_line",
            "end_line",
        },
        "judgment",
    )
    try:
        candidate_id = ExternalCandidateId(mapping["candidate_id"])
    except (TypeError, ValueError) as error:
        raise ExternalAuditCliError("judgment candidate is invalid") from error
    return ReviewedJudgment(
        candidate_id=candidate_id,
        capability=mapping["capability"],
        supported=mapping["supported"],
        relative_path=mapping["relative_path"],
        start_line=mapping["start_line"],
        end_line=mapping["end_line"],
    )

def _weight_license_judgment_from_json(value: str) -> ReviewedWeightLicenseJudgment:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise ExternalAuditCliError(
            "weight license judgment JSON is malformed"
        ) from error
    mapping = _exact_mapping(
        raw,
        {"candidate_id", "component_id", "status", "evidence"},
        "weight license judgment",
    )
    try:
        candidate_id = ExternalCandidateId(mapping["candidate_id"])
        status = WeightLicenseStatus(mapping["status"])
    except (TypeError, ValueError) as error:
        raise ExternalAuditCliError(
            "weight license judgment enum is invalid"
        ) from error
    evidence = mapping["evidence"]
    if not isinstance(evidence, list):
        raise ExternalAuditCliError("weight license judgment evidence is invalid")
    spans: list[ReviewedSourceSpan] = []
    for item in evidence:
        span = _exact_mapping(
            item,
            {"relative_path", "start_line", "end_line"},
            "weight license evidence",
        )
        spans.append(
            ReviewedSourceSpan(
                relative_path=span["relative_path"],
                start_line=span["start_line"],
                end_line=span["end_line"],
            )
        )
    return ReviewedWeightLicenseJudgment(
        candidate_id=candidate_id,
        component_id=mapping["component_id"],
        status=status,
        evidence=tuple(spans),
    )

def _catalog_for_run(run: ExternalAuditRun) -> ExternalCatalog:
    if not isinstance(run, ExternalAuditRun):
        raise ExternalAuditCliError("operation requires an ExternalAuditRun")
    catalog = ExternalCatalog(run.catalog_schema_version, run.declarations)
    provenance._validate_closed_catalog(catalog)
    return catalog

def _catalog_path_for_schema(schema_version: str) -> Path:
    try:
        return _PINNED_CATALOG_PATHS[schema_version]
    except (KeyError, TypeError) as error:
        raise ExternalAuditCliError("catalog schema version is unsupported") from error

def _load_catalog_for_schema(schema_version: str) -> ExternalCatalog:
    catalog = load_external_catalog(_catalog_path_for_schema(schema_version))
    if catalog.schema_version != schema_version:
        raise ExternalAuditCliError("catalog schema does not match its closed path")
    return catalog

def _registry_route_for_catalog(catalog: ExternalCatalog) -> tuple[Path, str]:
    try:
        return _REGISTRY_ROUTES_BY_CATALOG_SCHEMA[catalog.schema_version]
    except (KeyError, TypeError) as error:
        raise ExternalAuditCliError("catalog schema version is unsupported") from error

def _write_new_bytes(path: Path, raw: bytes, *, root: Path) -> ArtifactIdentity:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ExternalAuditCliError(
            "create-new output is outside its fixed root"
        ) from error
    provenance._validate_relative_path(relative, "create-new audit output")
    try:
        parent_fd = provenance._open_directory_at_path(
            root, relative.parent, "create-new audit output parent"
        )
    except OSError as error:
        raise ExternalAuditCliError("cannot open create-new output parent") from error
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(relative.name, flags, 0o664, dir_fd=parent_fd)
        os.fsync(parent_fd)
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("create-new write made no progress")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o664)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
    except FileExistsError as error:
        raise ExternalAuditCliError("create-new output already exists") from error
    except OSError as error:
        raise ExternalAuditCliError(
            "create-new output publication failed; occupied path is preserved"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    if not stat.S_ISREG(info.st_mode):
        raise ExternalAuditCliError("create-new output is not a regular file")
    identity = ArtifactIdentity(str(path), hashlib.sha256(raw).hexdigest(), len(raw))
    if _read_identity(path, "create-new output") != identity:
        raise ExternalAuditCliError("create-new output identity drifted")
    return identity

def _read_identity(path: Path, label: str) -> ArtifactIdentity:
    raw = _read_regular_bytes(path, label)
    return ArtifactIdentity(str(path), hashlib.sha256(raw).hexdigest(), len(raw))

def _read_regular_bytes(path: Path, label: str) -> bytes:
    path = Path(path)
    if not path.is_absolute():
        raise ExternalAuditCliError(f"{label} path must be absolute")
    try:
        before_path = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ExternalAuditCliError(f"cannot stat {label}") from error
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ExternalAuditCliError(f"{label} must be a plain regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.stat(path, follow_symlinks=False)
    if not (
        provenance._stat_identity(before_path)
        == provenance._stat_identity(before)
        == provenance._stat_identity(after)
        == provenance._stat_identity(after_path)
    ):
        raise ExternalAuditCliError(f"{label} identity drifted")
    return b"".join(chunks)

def _artifact_from_mapping(value: object, label: str) -> ArtifactIdentity:
    mapping = _exact_mapping(value, {"path", "sha256", "size_bytes"}, label)
    path = mapping["path"]
    sha256 = mapping["sha256"]
    size = mapping["size_bytes"]
    if (
        type(path) is not str
        or not Path(path).is_absolute()
        or type(sha256) is not str
        or _SHA256.fullmatch(sha256) is None
        or type(size) is not int
        or size < 0
    ):
        raise ExternalAuditCliError(f"{label} identity is invalid")
    return ArtifactIdentity(path, sha256, size)

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

def _exact_mapping(
    value: object, expected: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ExternalAuditCliError(f"{label} keys are invalid")
    if any(type(key) is not str for key in value):
        raise ExternalAuditCliError(f"{label} keys are invalid")
    return value

def _required_string(mapping: Mapping[str, object], key: str, label: str) -> str:
    value = mapping.get(key)
    if type(value) is not str or not value:
        raise ExternalAuditCliError(f"{label} {key} is invalid")
    return value

def _reject_secret_like_content(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if type(key) is not str:
                raise ExternalAuditCliError("registry keys must be strings")
            normalized = key.lower().replace("-", "_")
            if any(fragment in normalized for fragment in _SENSITIVE_PROXY_FRAGMENTS):
                raise ExternalAuditCliError("registry contains a secret-like key")
            _reject_secret_like_content(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_secret_like_content(nested)
    elif isinstance(value, str):
        parsed = urlsplit(value)
        if parsed.scheme and (
            parsed.username is not None or parsed.password is not None
        ):
            raise ExternalAuditCliError("registry contains a credential-bearing URL")

def _timestamp_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def _print_identity(prefix: str, identity: ArtifactIdentity, *, run_id: str) -> None:
    print(
        json.dumps(
            {
                "run_id": run_id,
                f"{prefix}_path": identity.path,
                f"{prefix}_sha256": identity.sha256,
                f"{prefix}_size_bytes": identity.size_bytes,
            },
            sort_keys=True,
        )
    )
