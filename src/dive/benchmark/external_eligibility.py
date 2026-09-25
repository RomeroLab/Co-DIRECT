
from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from urllib.parse import parse_qsl, urlsplit

import yaml
from yaml.events import AliasEvent

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.external_catalog import (
    CandidateDeclaration,
    ExternalCandidateId,
)
from dive.benchmark.external_provenance import (
    AuditDisposition,
    CapabilityDecision,
    CapabilityJudgment,
    ExternalProvenanceError,
    LicenseObservation,
    SourceObservation,
    WeightBundleObservation,
    WeightComponentObservation,
    WeightContentClass,
    WeightEvidence,
    WeightLicenseStatus,
    WeightObservation,
    WeightReference,
    SourceSpan,
)
import dive.benchmark.external_provenance as provenance

class ExternalEligibilityError(RuntimeError):
    pass

_DECISION_SCHEMA = "dive-external-baseline-decisions-v1"
_DECISION_SCHEMA_V2 = "dive-external-baseline-decisions-v2"
_CATALOG_SCHEMA_V1 = "dive-external-baseline-candidates-v1"
_CATALOG_SCHEMA_V2 = "dive-external-baseline-candidates-v2"
_ROOT_KEYS = frozenset({"schema_version", "reviewed_at", "decisions"})
_DECISION_KEYS = frozenset(
    {"candidate_id", "source_commit", "source_tree_sha256", "capabilities"}
)
_DECISION_V2_KEYS = _DECISION_KEYS | frozenset({"weight_licenses"})
_CAPABILITY_KEYS = frozenset({"capability", "status", "evidence"})
_EVIDENCE_KEYS = frozenset({"path", "start_line", "end_line"})
_WEIGHT_LICENSE_KEYS = frozenset({"candidate_id", "component_id", "status", "evidence"})
_WEIGHT_LICENSE_EVIDENCE_KEYS = frozenset({"relative_path", "start_line", "end_line"})
_CAPABILITY_STATUSES = frozenset({"supported", "unsupported"})
_WEIGHT_LICENSE_STATUSES = frozenset(
    {
        WeightLicenseStatus.RESEARCH_EVALUATION_ALLOWED.value,
        WeightLicenseStatus.UNAVAILABLE.value,
    }
)
_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_URL = re.compile(r"https?://[^\s<>\"'`]+")
_TEXT_SUFFIXES = frozenset(
    {".cfg", ".conf", ".ini", ".json", ".py", ".sh", ".toml", ".yaml", ".yml"}
)
_WEIGHT_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".h5",
    ".model",
    ".params",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".tar.gz",
    ".weights",
)
_WEIGHT_WORDS = re.compile(r"\b(checkpoint|pretrained|weight|weights)\b", re.I)
_SHORTENER_HOSTS = frozenset(
    {"bit.ly", "goo.gl", "is.gd", "ow.ly", "t.co", "tinyurl.com", "rb.gy"}
)
_SENSITIVE_QUERY_FRAGMENTS = (
    "access_key",
    "api_key",
    "auth",
    "bearer",
    "cookie",
    "credential",
    "password",
    "secret",
    "signature",
    "signed",
    "token",
)
_BENIGN_QUERY_KEYS = frozenset(
    {"confirm", "download", "export", "filename", "id", "raw"}
)
_OFFICIAL_WEIGHT_HOSTS = frozenset(
    {
        "api.figshare.com",
        "chroma-weights.generatebiomedicines.com",
        "drive.google.com",
        "figshare.com",
        "files.ipd.uw.edu",
        "github.com",
        "huggingface.co",
        "storage.googleapis.com",
        "zenodo.org",
    }
)
_PLAUSIBLE_WEIGHT_CLASSES = frozenset(
    {
        WeightContentClass.PYTORCH_ZIP,
        WeightContentClass.PICKLE_STREAM,
        WeightContentClass.SAFETENSORS,
        WeightContentClass.HDF5,
    }
)
_CANONICAL_MIT = (
    "MIT License\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy "
    'of this software and associated documentation files (the "Software"), to deal '
    "in the Software without restriction, including without limitation the rights "
    "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies "
    "of the Software, and to permit persons to whom the Software is furnished to do "
    "so, subject to the following conditions:\n\n"
    "The above copyright notice and this permission notice shall be included in all "
    "copies or substantial portions of the Software.\n\n"
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR '
    "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS "
    "FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR "
    "COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER "
    "IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN "
    "CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
).encode("utf-8")
_RESEARCH_LICENSE_CLASS = "research_evaluation_allowed"
_LICENSE_IDENTITY_POLICY = MappingProxyType(
    {
        hashlib.sha256(_CANONICAL_MIT).hexdigest(): (
            "MIT",
            _RESEARCH_LICENSE_CLASS,
        ),
        hashlib.sha256(b"SPDX-License-Identifier: MIT\n").hexdigest(): (
            "MIT",
            _RESEARCH_LICENSE_CLASS,
        ),
        hashlib.sha256(b"SPDX-License-Identifier: Apache-2.0\n").hexdigest(): (
            "Apache-2.0",
            _RESEARCH_LICENSE_CLASS,
        ),
        hashlib.sha256(b"SPDX-License-Identifier: GPL-3.0-only\n").hexdigest(): (
            "GPL-3.0-only",
            _RESEARCH_LICENSE_CLASS,
        ),
        hashlib.sha256(b"SPDX-License-Identifier: GPL-3.0-or-later\n").hexdigest(): (
            "GPL-3.0-or-later",
            _RESEARCH_LICENSE_CLASS,
        ),
        "eefc2ae77cb92b1414a6ac76b246642fae2189747b1b07747e92e5ddc696ec24": (
            "BSD-3-Clause",
            _RESEARCH_LICENSE_CLASS,
        ),
        "d9a1b1e30d633d5732ea18e3cba9538d293ebc53e1a9e4e96ab739e0c5c4f1cb": (
            "MIT",
            _RESEARCH_LICENSE_CLASS,
        ),
        "82058ed9e887497cee68ef2e064926a6ad248b24adc71c58c75e732aeb6b37cc": (
            "MIT",
            "research_evaluation_allowed",
        ),
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30": (
            "Apache-2.0",
            _RESEARCH_LICENSE_CLASS,
        ),
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4": (
            "Apache-2.0",
            _RESEARCH_LICENSE_CLASS,
        ),
    }
)
_REQUIRED_CAPABILITIES = {
    ExternalCandidateId.RFDIFFUSION_BINDER_V1: (
        "target_structure",
        "target_hotspots",
        "backbone_output",
    ),
    ExternalCandidateId.FRAMEFLOW_MOTIF_V1: (
        "target_structure",
        "target_hotspots",
        "backbone_output",
    ),
    ExternalCandidateId.CHROMA_CONDITIONED_V1: (
        "target_structure",
        "target_hotspots",
        "backbone_output",
        "joint_sequence_structure_output",
    ),
    ExternalCandidateId.RFDIFFUSIONAA_LIGAND_V1: (
        "ligand_identity",
        "motif_atoms",
        "backbone_output",
    ),
    ExternalCandidateId.DIFFAB_CDR_V1: (
        "antibody_antigen_complex",
        "insertion_aware_cdr_h3",
        "joint_sequence_structure_output",
    ),
    ExternalCandidateId.RFANTIBODY_H3_V1: (
        "antibody_antigen_complex",
        "fixed_framework",
        "h3_only_design",
        "fixed_h3_length",
        "joint_sequence_structure_output",
        "insertion_aware_cdr_h3",
    ),
    ExternalCandidateId.ABX_CDR_V1: (
        "antibody_antigen_complex",
        "insertion_aware_cdr_h3",
        "joint_sequence_structure_output",
    ),
}

@dataclass(frozen=True, slots=True)
class WeightReferenceDiscovery(Sequence[WeightReference]):

    references: tuple[WeightReference, ...]
    rejected_reference_reason: str | None = None
    rejected_reference_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.references, tuple) or any(
            not isinstance(item, WeightReference) for item in self.references
        ):
            raise ExternalEligibilityError("weight reference discovery is invalid")
        if (
            type(self.rejected_reference_count) is not int
            or self.rejected_reference_count < 0
        ):
            raise ExternalEligibilityError("rejected weight reference count is invalid")
        if self.rejected_reference_count == 0:
            if self.rejected_reference_reason is not None:
                raise ExternalEligibilityError(
                    "weight reference rejection reason lacks a rejection"
                )
        elif self.rejected_reference_reason != "unsafe_reference":
            raise ExternalEligibilityError(
                "weight reference rejection reason is not a safe typed code"
            )

    def __getitem__(self, index):
        return self.references[index]

    def __len__(self) -> int:
        return len(self.references)

@dataclass(frozen=True, slots=True)
class WeightLicenseJudgment:

    candidate_id: ExternalCandidateId
    component_id: str
    status: WeightLicenseStatus
    evidence: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.candidate_id is not ExternalCandidateId.RFANTIBODY_H3_V1:
            raise ExternalEligibilityError("weight license candidate is invalid")
        expected_components = tuple(
            item[0]
            for item in provenance._WEIGHT_COMPONENTS_BY_CANDIDATE[self.candidate_id]
        )
        if self.component_id not in expected_components:
            raise ExternalEligibilityError("weight license component is invalid")
        if self.status not in {
            WeightLicenseStatus.RESEARCH_EVALUATION_ALLOWED,
            WeightLicenseStatus.UNAVAILABLE,
        }:
            raise ExternalEligibilityError("weight license status is invalid")
        if (
            not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(item, SourceSpan) for item in self.evidence)
        ):
            raise ExternalEligibilityError("weight license evidence is invalid")

@dataclass(frozen=True, slots=True)
class ReviewedCandidateDecision:

    capability: CapabilityDecision
    weight_licenses: tuple[WeightLicenseJudgment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capability, CapabilityDecision):
            raise ExternalEligibilityError("reviewed capability decision is invalid")
        if not isinstance(self.weight_licenses, tuple) or any(
            not isinstance(item, WeightLicenseJudgment) for item in self.weight_licenses
        ):
            raise ExternalEligibilityError(
                "reviewed weight license decisions are invalid"
            )

class _ClosedDecisionLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep: bool = False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ExternalEligibilityError(
                    "decision YAML key is unhashable"
                ) from error
            if duplicate:
                raise ExternalEligibilityError("decision YAML has duplicate keys")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping

@dataclass(frozen=True, slots=True)
class _AuthenticatedSource:
    candidate_id: ExternalCandidateId
    checkout: Path
    tracked_paths: tuple[str, ...]
    tracked_blobs: Mapping[str, str]

def inspect_license(
    source: SourceObservation, declaration: CandidateDeclaration
) -> LicenseObservation:

    if not isinstance(declaration, CandidateDeclaration):
        raise ExternalEligibilityError("license inspection requires a declaration")
    authenticated = _authenticate_source(source, expected_id=declaration.candidate_id)
    if source.url != declaration.url:
        raise ExternalEligibilityError("source URL differs from its declaration")
    tracked = set(authenticated.tracked_paths)
    observations: list[tuple[ArtifactIdentity, str]] = []
    for relative_path in declaration.license_file_candidates:
        if relative_path not in tracked:
            continue
        if relative_path in source.git_lfs_pointer_paths:
            raise ExternalEligibilityError("declared license is a Git-LFS pointer")
        raw = _read_tracked_bytes(authenticated, relative_path, "license")
        observations.append(
            (
                _artifact(authenticated.checkout / relative_path, raw),
                _classify_license(raw),
            )
        )
    if not observations:
        return LicenseObservation(
            artifact=None,
            review_class="unavailable",
            observed_at=_timestamp_now(),
        )
    _authenticate_source(source, expected_id=declaration.candidate_id)
    artifact, review_class = observations[0]
    if any(item[1] != review_class for item in observations[1:]):
        review_class = "restricted_review_required"
    return LicenseObservation(
        artifact=artifact,
        review_class=review_class,
        observed_at=_timestamp_now(),
    )

def discover_weight_references(
    source: SourceObservation,
) -> WeightReferenceDiscovery:

    authenticated = _authenticate_source(source)
    references: list[WeightReference] = []
    rejected_reference_count = 0
    lfs_paths = set(source.git_lfs_pointer_paths)
    for relative_path in authenticated.tracked_paths:
        if relative_path in lfs_paths or not _is_searchable_source_text(relative_path):
            continue
        raw = _read_tracked_bytes(
            authenticated, relative_path, "weight-reference source"
        )
        if raw.startswith(_GIT_LFS_POINTER_PREFIX):
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ExternalEligibilityError(
                f"weight-reference source is not UTF-8: {relative_path}"
            ) from error
        identity = _artifact(authenticated.checkout / relative_path, raw)
        for start_line, end_line, logical_text in _logical_source_spans(text):
            for match in _URL.finditer(logical_text):
                url = _trim_url_punctuation(match.group(0))
                if not _looks_like_weight_reference(url, logical_text):
                    continue
                try:
                    _validate_reference_url(url)
                except ExternalEligibilityError:
                    rejected_reference_count += 1
                    continue
                references.append(
                    WeightReference(
                        url=url,
                        source_artifact=identity,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )
    _authenticate_source(source, expected_id=authenticated.candidate_id)
    return WeightReferenceDiscovery(
        references=tuple(references),
        rejected_reference_reason=(
            "unsafe_reference" if rejected_reference_count else None
        ),
        rejected_reference_count=rejected_reference_count,
    )

def load_reviewed_decisions(
    path: Path,
    sources: Mapping[ExternalCandidateId, SourceObservation],
) -> tuple[CapabilityDecision, ...]:

    raw = _read_retained_regular_file(path, "reviewed decision file")
    return tuple(
        item.capability
        for item in parse_reviewed_decisions_bytes(
            raw, sources, catalog_schema_version=_CATALOG_SCHEMA_V1
        )
    )

def parse_reviewed_decisions_bytes(
    raw: bytes,
    sources: Mapping[ExternalCandidateId, SourceObservation],
    *,
    catalog_schema_version: str,
) -> tuple[ReviewedCandidateDecision, ...]:

    if type(raw) is not bytes:
        raise ExternalEligibilityError("reviewed decision content must be bytes")
    expected_decision_schema = {
        _CATALOG_SCHEMA_V1: _DECISION_SCHEMA,
        _CATALOG_SCHEMA_V2: _DECISION_SCHEMA_V2,
    }.get(catalog_schema_version)
    if expected_decision_schema is None:
        raise ExternalEligibilityError("catalog schema is unsupported")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExternalEligibilityError(
            "reviewed decision file must be UTF-8"
        ) from error
    try:
        if any(isinstance(event, AliasEvent) for event in yaml.parse(text)):
            raise ExternalEligibilityError("decision YAML aliases are forbidden")
        document = yaml.load(text, Loader=_ClosedDecisionLoader)
    except yaml.YAMLError as error:
        raise ExternalEligibilityError("reviewed decision YAML is malformed") from error
    root = _exact_mapping(document, _ROOT_KEYS, "reviewed decisions")
    if root["schema_version"] != expected_decision_schema:
        raise ExternalEligibilityError("reviewed decision schema is unsupported")
    reviewed_at = _required_string(root, "reviewed_at", "reviewed decisions")
    raw_decisions = root["decisions"]
    if not isinstance(raw_decisions, list):
        raise ExternalEligibilityError("reviewed decisions must be an ordered list")
    source_items = _validate_sources(sources)
    expected_ids = tuple(candidate_id for candidate_id, _ in source_items)
    if len(raw_decisions) != len(expected_ids):
        raise ExternalEligibilityError("reviewed decision membership is incomplete")

    decisions: list[ReviewedCandidateDecision] = []
    observed_ids: list[ExternalCandidateId] = []
    for raw_decision, (expected_id, source) in zip(
        raw_decisions, source_items, strict=True
    ):
        decision = _exact_mapping(
            raw_decision,
            _DECISION_V2_KEYS
            if catalog_schema_version == _CATALOG_SCHEMA_V2
            else _DECISION_KEYS,
            "capability decision",
        )
        candidate_id = _candidate_id(decision["candidate_id"])
        observed_ids.append(candidate_id)
        if candidate_id is not expected_id:
            raise ExternalEligibilityError(
                "reviewed decision order or membership is invalid"
            )
        authenticated = _authenticate_source(source, expected_id=candidate_id)
        if (
            decision["source_commit"] != source.commit
            or decision["source_tree_sha256"] != source.tracked_tree_sha256
        ):
            raise ExternalEligibilityError(
                "capability decision source identity does not match observation"
            )
        required = _REQUIRED_CAPABILITIES[candidate_id]
        raw_capabilities = decision["capabilities"]
        if not isinstance(raw_capabilities, list):
            raise ExternalEligibilityError("capabilities must be an ordered list")
        names: list[str] = []
        judgments: list[CapabilityJudgment] = []
        for raw_capability in raw_capabilities:
            item = _exact_mapping(raw_capability, _CAPABILITY_KEYS, "capability")
            capability = _required_string(item, "capability", "capability")
            names.append(capability)
            status = item["status"]
            if status not in _CAPABILITY_STATUSES:
                raise ExternalEligibilityError("capability status is invalid")
            artifact, start_line, end_line = _validate_line_evidence(
                item["evidence"], authenticated, source, label="capability evidence"
            )
            judgments.append(
                CapabilityJudgment(
                    capability=capability,
                    supported=status == "supported",
                    source_artifact=artifact,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        if tuple(names) != required:
            raise ExternalEligibilityError(
                "capability coverage must exactly match catalog requirements"
            )
        _authenticate_source(source, expected_id=candidate_id)
        capability_decision = CapabilityDecision(
            source_commit=source.commit,
            source_tree_sha256=source.tracked_tree_sha256,
            judgments=tuple(judgments),
            reviewed_at=reviewed_at,
        )
        weight_licenses = _parse_weight_license_judgments(
            decision.get("weight_licenses", []),
            candidate_id=candidate_id,
            authenticated=authenticated,
            source=source,
            catalog_schema_version=catalog_schema_version,
        )
        decisions.append(
            ReviewedCandidateDecision(
                capability=capability_decision,
                weight_licenses=weight_licenses,
            )
        )
    if tuple(observed_ids) != expected_ids:
        raise ExternalEligibilityError("reviewed decision membership is invalid")
    return tuple(decisions)

def _parse_weight_license_judgments(
    raw_judgments: object,
    *,
    candidate_id: ExternalCandidateId,
    authenticated: _AuthenticatedSource,
    source: SourceObservation,
    catalog_schema_version: str,
) -> tuple[WeightLicenseJudgment, ...]:
    if catalog_schema_version == _CATALOG_SCHEMA_V1:
        return ()
    if not isinstance(raw_judgments, list):
        raise ExternalEligibilityError("weight licenses must be an ordered list")
    if candidate_id is not ExternalCandidateId.RFANTIBODY_H3_V1:
        if raw_judgments:
            raise ExternalEligibilityError("non-RFantibody weight licenses are invalid")
        return ()
    judgments: list[WeightLicenseJudgment] = []
    component_ids: list[str] = []
    for raw_judgment in raw_judgments:
        item = _exact_mapping(
            raw_judgment, _WEIGHT_LICENSE_KEYS, "weight license judgment"
        )
        item_candidate_id = _candidate_id(item["candidate_id"])
        if item_candidate_id is not candidate_id:
            raise ExternalEligibilityError("weight license candidate is invalid")
        component_id = _required_string(item, "component_id", "weight license judgment")
        component_ids.append(component_id)
        status = item["status"]
        if status not in _WEIGHT_LICENSE_STATUSES:
            raise ExternalEligibilityError("weight license status is invalid")
        raw_evidence = item["evidence"]
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ExternalEligibilityError("weight license evidence is invalid")
        evidence = tuple(
            SourceSpan(
                source_artifact=artifact,
                start_line=start_line,
                end_line=end_line,
            )
            for artifact, start_line, end_line in (
                _validate_line_evidence(
                    entry,
                    authenticated,
                    source,
                    label="weight license evidence",
                    path_key="relative_path",
                )
                for entry in raw_evidence
            )
        )
        judgments.append(
            WeightLicenseJudgment(
                candidate_id=candidate_id,
                component_id=component_id,
                status=WeightLicenseStatus(status),
                evidence=evidence,
            )
        )
    expected_components = tuple(
        item[0] for item in provenance._WEIGHT_COMPONENTS_BY_CANDIDATE[candidate_id]
    )
    if tuple(component_ids) != expected_components:
        raise ExternalEligibilityError(
            "weight license coverage must exactly match the RFantibody bundle"
        )
    return tuple(judgments)

def decide_candidate(
    declaration: CandidateDeclaration,
    source: SourceObservation,
    license: LicenseObservation,
    weight: WeightEvidence,
    capability: CapabilityDecision,
) -> AuditDisposition:

    if not isinstance(declaration, CandidateDeclaration):
        return AuditDisposition.PROVENANCE_FAILED
    if not isinstance(source, SourceObservation) or source.url != declaration.url:
        return AuditDisposition.PROVENANCE_FAILED
    if not _pure_source_candidate_matches(source, declaration.candidate_id):
        return AuditDisposition.PROVENANCE_FAILED
    if not isinstance(license, LicenseObservation):
        return AuditDisposition.LICENSE_UNAVAILABLE
    if license.artifact is None:
        return AuditDisposition.LICENSE_UNAVAILABLE
    expected_license_paths = {
        str(Path(source.checkout_path) / relative)
        for relative in declaration.license_file_candidates
    }
    if license.artifact.path not in expected_license_paths:
        return AuditDisposition.PROVENANCE_FAILED
    license_lfs_paths = {
        str(Path(source.checkout_path) / relative)
        for relative in source.git_lfs_pointer_paths
    }
    if license.artifact.path in license_lfs_paths:
        return AuditDisposition.PROVENANCE_FAILED
    if license.review_class == "restricted_review_required":
        return AuditDisposition.REVIEW_REQUIRED
    if license.review_class != "research_evaluation_allowed":
        return AuditDisposition.LICENSE_UNAVAILABLE
    if not isinstance(capability, CapabilityDecision):
        return AuditDisposition.REVIEW_REQUIRED
    if (
        capability.source_commit != source.commit
        or capability.source_tree_sha256 != source.tracked_tree_sha256
    ):
        return AuditDisposition.PROVENANCE_FAILED
    decided = {item.capability for item in capability.judgments}
    if decided != set(declaration.required_capabilities):
        return AuditDisposition.PROVENANCE_FAILED
    checkout = Path(source.checkout_path)
    lfs_paths = {str(checkout / relative) for relative in source.git_lfs_pointer_paths}
    for judgment in capability.judgments:
        if (
            not _path_is_below(Path(judgment.source_artifact.path), checkout)
            or judgment.source_artifact.path in lfs_paths
            or judgment.start_line < 1
            or judgment.end_line < judgment.start_line
        ):
            return AuditDisposition.PROVENANCE_FAILED
    if capability.unsupported_capabilities:
        return AuditDisposition.TASK_CONTRACT_UNREPRESENTABLE
    if tuple(capability.supported_capabilities) != declaration.required_capabilities:
        return AuditDisposition.PROVENANCE_FAILED
    if isinstance(weight, WeightBundleObservation):
        return _decide_weight_bundle(weight, checkout, lfs_paths, source)
    if not isinstance(weight, WeightObservation):
        return AuditDisposition.WEIGHTS_UNAVAILABLE
    return _decide_singleton_weight(weight, checkout, lfs_paths, source)

def _decide_singleton_weight(
    weight: WeightObservation,
    checkout: Path,
    lfs_paths: set[str],
    source: SourceObservation,
) -> AuditDisposition:
    if weight.official_url is None or not _is_official_weight_url(weight.official_url):
        return AuditDisposition.WEIGHTS_UNAVAILABLE
    if (
        weight.citation is None
        or weight.citation.url != weight.official_url
        or not _path_is_below(Path(weight.citation.source_artifact.path), checkout)
        or weight.citation.source_artifact.path in lfs_paths
    ):
        return AuditDisposition.PROVENANCE_FAILED
    if weight.artifact is None:
        return AuditDisposition.ELIGIBLE_FOR_WEIGHT_ACQUISITION
    return _retained_weight_disposition(weight, lfs_paths, source)

def _decide_weight_bundle(
    bundle: WeightBundleObservation,
    checkout: Path,
    lfs_paths: set[str],
    source: SourceObservation,
) -> AuditDisposition:
    components = bundle.components
    if any(
        not _is_official_weight_url(component.official_url)
        or component.citation is None
        for component in components
    ):
        return AuditDisposition.WEIGHTS_UNAVAILABLE
    if any(
        component.citation.url != component.official_url
        or not _path_is_below(Path(component.citation.source_artifact.path), checkout)
        or component.citation.source_artifact.path in lfs_paths
        for component in components
    ):
        return AuditDisposition.PROVENANCE_FAILED
    if any(
        not _path_is_below(Path(span.source_artifact.path), checkout)
        or span.source_artifact.path in lfs_paths
        for component in components
        for span in component.license_evidence
    ):
        return AuditDisposition.PROVENANCE_FAILED
    if any(
        component.license_status is WeightLicenseStatus.UNAVAILABLE
        for component in components
    ):
        return AuditDisposition.LICENSE_UNAVAILABLE
    if any(
        component.license_status is WeightLicenseStatus.UNREVIEWED
        for component in components
    ):
        return AuditDisposition.REVIEW_REQUIRED
    if any(
        component.license_status is not WeightLicenseStatus.RESEARCH_EVALUATION_ALLOWED
        for component in components
    ):
        return AuditDisposition.LICENSE_UNAVAILABLE
    retained = tuple(
        component for component in components if component.artifact is not None
    )
    if not retained:
        return AuditDisposition.ELIGIBLE_FOR_WEIGHT_ACQUISITION
    if len(retained) != len(components):
        return AuditDisposition.WEIGHTS_UNAVAILABLE
    dispositions = {
        _retained_weight_disposition(component, lfs_paths, source)
        for component in retained
    }
    if AuditDisposition.PROVENANCE_FAILED in dispositions:
        return AuditDisposition.PROVENANCE_FAILED
    if dispositions == {AuditDisposition.ELIGIBLE_FOR_ADAPTER}:
        return AuditDisposition.ELIGIBLE_FOR_ADAPTER
    return AuditDisposition.WEIGHTS_UNAVAILABLE

def _retained_weight_disposition(
    weight: WeightObservation | WeightComponentObservation,
    lfs_paths: set[str],
    source: SourceObservation,
) -> AuditDisposition:
    if weight.content_class not in _PLAUSIBLE_WEIGHT_CLASSES:
        return AuditDisposition.WEIGHTS_UNAVAILABLE
    if weight.artifact is None or weight.artifact.size_bytes < 1:
        return AuditDisposition.WEIGHTS_UNAVAILABLE
    if weight.artifact.path in lfs_paths:
        return AuditDisposition.PROVENANCE_FAILED
    expected_weight_root = _candidate_evidence_dir(source) / "weights"
    try:
        Path(weight.artifact.path).relative_to(expected_weight_root)
    except ValueError:
        return AuditDisposition.PROVENANCE_FAILED
    return AuditDisposition.ELIGIBLE_FOR_ADAPTER

def _authenticate_source(
    source: SourceObservation,
    *,
    expected_id: ExternalCandidateId | None = None,
) -> _AuthenticatedSource:
    if not isinstance(source, SourceObservation):
        raise ExternalEligibilityError("source observation is invalid")
    try:
        candidate_id = _candidate_id_from_source_paths(source)
        if expected_id is not None and candidate_id is not expected_id:
            raise ExternalEligibilityError("source candidate identity drifted")
        checkout = Path(source.checkout_path)
        git_binary = Path(source.git_binary.path)
        binding = SimpleNamespace(candidate_id=candidate_id, url=source.url)
        run = SimpleNamespace(evidence_dir=_run_evidence_dir(source))
        provenance._reauthenticate_source_observation(run, binding, source)
        stages = provenance._tracked_source_stages(
            checkout, git_binary, source.git_binary_stat
        )
        tracked_paths = tuple(stages)
        tracked_blobs = {path: blob for path, (_, blob) in stages.items()}
        return _AuthenticatedSource(
            candidate_id, checkout, tracked_paths, tracked_blobs
        )
    except ExternalEligibilityError:
        raise
    except (ExternalProvenanceError, OSError, ValueError) as error:
        raise ExternalEligibilityError(
            f"source authentication failed: {error}"
        ) from error

def _candidate_id_from_source_paths(source: SourceObservation) -> ExternalCandidateId:
    retrieval = Path(source.retrieval_log.path)
    checkout = Path(source.checkout_path)
    if (
        retrieval.name != "retrieval.log"
        or retrieval.parent.parent.name != "candidates"
    ):
        raise ExternalEligibilityError("source retrieval log path is outside its slot")
    candidate_id = _candidate_id(retrieval.parent.name)
    run_dir = retrieval.parents[2]
    if checkout != run_dir / "sources" / candidate_id.value:
        raise ExternalEligibilityError("source checkout path is outside its slot")
    return candidate_id

def _run_evidence_dir(source: SourceObservation) -> Path:
    return Path(source.retrieval_log.path).parents[2]

def _candidate_evidence_dir(source: SourceObservation) -> Path:
    return Path(source.retrieval_log.path).parent

def _pure_source_candidate_matches(
    source: SourceObservation, candidate_id: ExternalCandidateId
) -> bool:
    try:
        return _candidate_id_from_source_paths(source) is candidate_id
    except ExternalEligibilityError:
        return False

def _classify_license(raw: bytes) -> str:
    if not raw or b"\x00" in raw:
        return "unavailable"
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "unavailable"
    if not decoded.strip():
        return "unavailable"
    identity = _classify_license_identity(hashlib.sha256(raw).hexdigest())
    if identity is not None:
        return identity[1]
    return "restricted_review_required"

def _classify_license_identity(digest: str) -> tuple[str, str] | None:

    return _LICENSE_IDENTITY_POLICY.get(digest)

def _is_searchable_source_text(relative_path: str) -> bool:
    path = Path(relative_path)
    return (
        path.name.lower().startswith("readme") or path.suffix.lower() in _TEXT_SUFFIXES
    )

def _looks_like_weight_reference(url: str, line: str) -> bool:
    parsed = urlsplit(url)
    path = parsed.path.lower()
    return (
        path.endswith(_WEIGHT_SUFFIXES)
        or _WEIGHT_WORDS.search(line) is not None
        or (
            (parsed.hostname or "").lower() == "drive.google.com" and bool(parsed.query)
        )
    )

def _validate_reference_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ExternalEligibilityError("weight reference must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ExternalEligibilityError("weight reference contains credentials")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ExternalEligibilityError("weight reference host is missing")
    if hostname in _SHORTENER_HOSTS or any(
        hostname.endswith(f".{item}") for item in _SHORTENER_HOSTS
    ):
        raise ExternalEligibilityError("weight reference URL shortener is forbidden")
    if hostname not in _OFFICIAL_WEIGHT_HOSTS:
        raise ExternalEligibilityError("weight reference host is not official")
    if parsed.fragment:
        raise ExternalEligibilityError("weight reference fragment is forbidden")
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered_key = key.lower()
        lowered_value = value.lower()
        if lowered_key not in _BENIGN_QUERY_KEYS or any(
            fragment in lowered_key or fragment in lowered_value
            for fragment in _SENSITIVE_QUERY_FRAGMENTS
        ):
            raise ExternalEligibilityError(
                "weight reference query contains secret or unsupported parameters"
            )

def _is_official_weight_url(url: str) -> bool:
    try:
        _validate_reference_url(url)
    except ExternalEligibilityError:
        return False
    return True

def _logical_source_spans(text: str) -> tuple[tuple[int, int, str], ...]:

    lines = text.splitlines()
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        start = index
        end = index
        logical = lines[index]
        while end + 1 < len(lines) and (
            logical.rstrip().endswith("\\")
            or _adjacent_quoted_string(logical, lines[end + 1])
        ):
            end += 1
            logical = f"{logical}\n{lines[end]}"
        logical = re.sub(r"\\\s*\n\s*", "", logical)
        logical = re.sub(r"([\"'])\s*\1", "", logical)
        spans.append((start + 1, end + 1, logical))
        index = end + 1
    return tuple(spans)

def _adjacent_quoted_string(current: str, following: str) -> bool:
    left = re.search(r"([\"'])\s*$", current)
    right = re.match(r"\s*([\"'])", following)
    return left is not None and right is not None and left.group(1) == right.group(1)

def _trim_url_punctuation(url: str) -> str:
    url = url.rstrip(".,;:!?")
    pairs = ((")", "("), ("]", "["), ("}", "{"))
    changed = True
    while changed and url:
        changed = False
        for closing, opening in pairs:
            if url.endswith(closing) and url.count(closing) > url.count(opening):
                url = url[:-1]
                changed = True
    return url

def _validate_sources(
    sources: Mapping[ExternalCandidateId, SourceObservation],
) -> tuple[tuple[ExternalCandidateId, SourceObservation], ...]:
    if not isinstance(sources, Mapping):
        raise ExternalEligibilityError("sources mapping is invalid")
    items = tuple(sources.items())
    if any(
        not isinstance(candidate_id, ExternalCandidateId)
        or not isinstance(source, SourceObservation)
        for candidate_id, source in items
    ):
        raise ExternalEligibilityError("sources mapping contains invalid records")
    if len({candidate_id for candidate_id, _ in items}) != len(items):
        raise ExternalEligibilityError("sources mapping contains duplicate candidates")
    return items

def _validate_line_evidence(
    raw: object,
    source: _AuthenticatedSource,
    observation: SourceObservation,
    *,
    label: str,
    path_key: str = "path",
) -> tuple[ArtifactIdentity, int, int]:
    evidence = _exact_mapping(
        raw,
        _WEIGHT_LICENSE_EVIDENCE_KEYS
        if path_key == "relative_path"
        else _EVIDENCE_KEYS,
        label,
    )
    relative_path = _required_string(evidence, path_key, label)
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative_path not in source.tracked_paths
    ):
        raise ExternalEligibilityError(f"{label} must be a tracked path")
    if relative_path in observation.git_lfs_pointer_paths:
        raise ExternalEligibilityError(f"{label} cannot be a Git-LFS pointer")
    start = evidence["start_line"]
    end = evidence["end_line"]
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        raise ExternalEligibilityError(f"{label} line span is invalid")
    file_raw = _read_tracked_bytes(source, relative_path, label)
    try:
        lines = file_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ExternalEligibilityError(f"{label} must be UTF-8 text") from error
    if end > len(lines) or not any(line.strip() for line in lines[start - 1 : end]):
        raise ExternalEligibilityError(f"{label} line span is invalid")
    return _artifact(source.checkout / relative_path, file_raw), start, end

def _read_tracked_bytes(
    source: _AuthenticatedSource, relative_path: str, label: str
) -> bytes:
    try:
        raw = provenance._read_regular_bytes_at(
            source.checkout, Path(relative_path), label
        )
    except ExternalProvenanceError as error:
        raise ExternalEligibilityError(f"cannot read {label}: {error}") from error
    expected_blob = source.tracked_blobs.get(relative_path)
    actual_blob = hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw,
        usedforsecurity=False,
    ).hexdigest()
    if expected_blob is None or actual_blob != expected_blob:
        raise ExternalEligibilityError(f"{label} differs from retained source HEAD")
    return raw

def _read_retained_regular_file(path: Path, label: str) -> bytes:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ExternalEligibilityError(f"{label} path must be absolute")
    try:
        before_path = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ExternalEligibilityError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(before_path.st_mode):
        raise ExternalEligibilityError(f"{label} symlink is forbidden")
    if not stat.S_ISREG(before_path.st_mode):
        raise ExternalEligibilityError(f"{label} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ExternalEligibilityError(f"{label} symlink is forbidden") from error
        raise ExternalEligibilityError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExternalEligibilityError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ExternalEligibilityError(f"{label} identity drifted") from error
    if not (
        _stat_identity(before_path)
        == _stat_identity(before)
        == _stat_identity(after)
        == _stat_identity(after_path)
    ):
        raise ExternalEligibilityError(f"{label} identity drifted while reading")
    return b"".join(chunks)

def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )

def _artifact(path: Path, raw: bytes) -> ArtifactIdentity:
    return ArtifactIdentity(
        path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)
    )

def _path_is_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents

def _exact_mapping(
    value: object, keys: frozenset[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ExternalEligibilityError(f"{label} has invalid keys")
    if any(type(key) is not str for key in value):
        raise ExternalEligibilityError(f"{label} keys must be strings")
    return value

def _required_string(mapping: Mapping[str, object], key: str, label: str) -> str:
    value = mapping[key]
    if type(value) is not str or not value or value != value.strip():
        raise ExternalEligibilityError(f"{label} {key} must be a nonempty string")
    return value

def _candidate_id(value: object) -> ExternalCandidateId:
    if type(value) is not str:
        raise ExternalEligibilityError("candidate id is invalid")
    try:
        return ExternalCandidateId(value)
    except ValueError as error:
        raise ExternalEligibilityError("candidate id is invalid") from error

def _timestamp_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
