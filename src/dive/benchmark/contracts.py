
from __future__ import annotations

from dive.codirect_paths import cache_dir

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from dive.signed_value.roots import (
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
    EMERGENT_UPSTREAM_COMMIT,
    EMERGENT_UPSTREAM_ROOT,
)
from dive.training.preflight import canonical_json_bytes

class BenchmarkContractError(RuntimeError):
    pass

class ExposureClass(StrEnum):
    CLEAR = "clear"
    KNOWN_EXPOSED = "known_exposed"
    UNKNOWN = "unknown"

class MetricStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"

@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    sha256: str
    size_bytes: int

@dataclass(frozen=True, slots=True)
class FoldseekContract:
    identity: ArtifactIdentity
    version: str
    threads: int
    min_tm_score: float
    min_shorter_coverage: float

@dataclass(frozen=True, slots=True)
class ViewContract:
    names: tuple[str, ...]
    min_parent_groups: int

@dataclass(frozen=True, slots=True)
class EvaluatorContract:
    family: str
    primary: str

@dataclass(frozen=True, slots=True)
class ExposureSubjectContract:

    subject: str
    kind: str
    source: ArtifactIdentity

@dataclass(frozen=True, slots=True)
class CutoffPredicateContract:

    subject: str
    evaluator_family: str
    condition: str
    source: ArtifactIdentity

@dataclass(frozen=True, slots=True)
class BenchmarkInputContract:

    train_data_ready: ArtifactIdentity
    train_data_ready_relative_path: Path
    loader_manifest_root: Path
    resolution_audit: ArtifactIdentity
    canonical_manifests: tuple[tuple[str, ArtifactIdentity], ...]

@dataclass(frozen=True, slots=True)
class BenchmarkContract:

    schema_version: str
    upstream_root: Path
    upstream_commit: str
    foldseek: FoldseekContract
    checkpoints: tuple[tuple[str, ArtifactIdentity], ...]
    inventories: tuple[tuple[str, ArtifactIdentity], ...]
    views: ViewContract
    exposure_classes: tuple[ExposureClass, ...]
    base_checkpoint_exposure: ExposureClass
    exposure_subjects: tuple[ExposureSubjectContract, ...]
    cutoff_predicates: tuple[CutoffPredicateContract, ...]
    evaluator_required_subjects: tuple[tuple[str, tuple[str, ...]], ...]
    cleanliness_required_subjects: tuple[str, ...]
    evaluators: tuple[EvaluatorContract, ...]
    arms: tuple[str, ...]
    smoke_scope: str
    authenticated_inputs: BenchmarkInputContract
    evidence_root: Path = EMERGENT_EVIDENCE_ROOT
    bulk_root: Path = EMERGENT_BULK_ROOT
    source_path: Path | None = None

    @property
    def semantic_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_mapping())).hexdigest()

    @property
    def execution_sha256(self) -> str:

        payload = {
            "semantic_contract_sha256": self.semantic_sha256,
            "foldseek": {"threads": self.foldseek.threads},
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def as_mapping(self) -> dict[str, object]:

        return {
            "schema_version": self.schema_version,
            "upstream": {
                "root": str(self.upstream_root),
                "commit": self.upstream_commit,
            },
            "foldseek": {
                **_identity_mapping(self.foldseek.identity),
                "version": self.foldseek.version,

                "min_tm_score": self.foldseek.min_tm_score,
                "min_shorter_coverage": self.foldseek.min_shorter_coverage,
            },
            "checkpoints": {
                name: _identity_mapping(identity) for name, identity in self.checkpoints
            },
            "inventories": {
                name: _identity_mapping(identity) for name, identity in self.inventories
            },
            "views": {
                "names": list(self.views.names),
                "min_parent_groups": self.views.min_parent_groups,
            },
            "exposure": {
                "classes": [item.value for item in self.exposure_classes],
                "base_checkpoint": self.base_checkpoint_exposure.value,
                "subjects": {
                    item.subject: {
                        "kind": item.kind,
                        "source": _identity_mapping(item.source),
                    }
                    for item in self.exposure_subjects
                },
                "cutoff_predicates": [
                    {
                        "subject": item.subject,
                        "evaluator_family": item.evaluator_family,
                        "condition": item.condition,
                        "source": _identity_mapping(item.source),
                    }
                    for item in self.cutoff_predicates
                ],
                "evaluator_required_subjects": {
                    family: list(subjects)
                    for family, subjects in self.evaluator_required_subjects
                },
                "cleanliness_required_subjects": list(
                    self.cleanliness_required_subjects
                ),
            },
            "evaluators": {
                item.family: {"primary": item.primary} for item in self.evaluators
            },
            "arms": list(self.arms),
            "smoke": {"scope": self.smoke_scope},
            "authenticated_inputs": {
                "train_data_ready": {
                    "path": self.authenticated_inputs.train_data_ready_relative_path.as_posix(),
                    "sha256": self.authenticated_inputs.train_data_ready.sha256,
                    "size_bytes": self.authenticated_inputs.train_data_ready.size_bytes,
                },
                "loader_manifest_root": str(
                    self.authenticated_inputs.loader_manifest_root
                ),
                "resolution_audit": _identity_mapping(
                    self.authenticated_inputs.resolution_audit
                ),
                "canonical_manifests": {
                    name: _identity_mapping(identity)
                    for name, identity in self.authenticated_inputs.canonical_manifests
                },
            },
        }

    def assert_authenticated(self) -> None:

        identities = (
            ("foldseek", self.foldseek.identity),
            *self.checkpoints,
            *self.inventories,
            *(
                (f"exposure:{item.subject}", item.source)
                for item in self.exposure_subjects
            ),
            *(
                (f"cutoff-predicate:{item.subject}", item.source)
                for item in self.cutoff_predicates
            ),
        )
        seen: set[ArtifactIdentity] = set()
        for role, identity in identities:
            if identity in seen:
                continue
            seen.add(identity)
            observed = file_identity(Path(identity.path))
            if observed != identity:
                raise BenchmarkContractError(
                    f"{role} identity drifted: expected {identity}, observed {observed}"
                )

_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "upstream",
        "foldseek",
        "checkpoints",
        "inventories",
        "views",
        "exposure",
        "evaluators",
        "arms",
        "smoke",
        "authenticated_inputs",
    }
)
_OPTIONAL_TOP_LEVEL = frozenset({"evaluator_registry"})
_CHECKPOINT_NAMES = (
    "common",
    "autoencoder",
    "rf3",
    "af2_multimer_v3",
    "proteinmpnn_ca",
    "soluble_mpnn",
    "ligandmpnn",
    "esmfold",
)
_INVENTORY_NAMES = ("local_tools", "checkpoint_training_metadata")
_EVALUATORS = (
    ("binder", "target_conditioned_success"),
    ("ame", "ame_motif_ligand_success"),
    ("antibody", "cdr_h3_ca_rmsd"),
)
_ARMS = (
    "base",
    "lora_joint",
    "fixed_self",
    "fixed_backbone_led",
    "fixed_local_led",
    "fixed_joint",
    "continuous_fixed",
    "time_router",
    "sample_router",
    "residue_router",
    "task_id_upper_bound",
    "generic_adapter",
    "pass_matched",
    "shuffled_gates",
    "adaptive",
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_DATA_READY_RELATIVE_PATH = Path("docs/results/emergent/train-data-ready.json")

def load_benchmark_contract(path: Path) -> BenchmarkContract:

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BenchmarkContractError(
            f"cannot read benchmark contract {path}: {error}"
        ) from error
    root = _mapping(raw, "benchmark contract")
    _exact_keys(
        root,
        _TOP_LEVEL | (set(root) & _OPTIONAL_TOP_LEVEL),
        "benchmark contract",
    )
    if root["schema_version"] != "dive-benchmark-contract-v1":
        raise BenchmarkContractError(
            "benchmark contract schema_version is not supported"
        )

    upstream = _mapping(root["upstream"], "upstream")
    _exact_keys(upstream, {"root", "commit"}, "upstream")
    upstream_root = _path(upstream["root"], "upstream.root")
    upstream_commit = _sha(upstream["commit"], "upstream.commit", length=40)
    if (
        upstream_root != EMERGENT_UPSTREAM_ROOT
        or upstream_commit != EMERGENT_UPSTREAM_COMMIT
    ):
        raise BenchmarkContractError(
            "benchmark contract must use the emergent upstream pin"
        )

    foldseek = _foldseek(root["foldseek"])
    checkpoints = _identities(root["checkpoints"], _CHECKPOINT_NAMES, "checkpoints")
    inventories = _identities(root["inventories"], _INVENTORY_NAMES, "inventories")
    evaluators = _evaluators(root["evaluators"])
    views = _views(root["views"])
    (
        classes,
        base_exposure,
        subjects,
        cutoff_predicates,
        evaluator_required,
        cleanliness_required,
    ) = _exposure(root["exposure"], checkpoints, inventories, evaluators)
    arms = _arms(root["arms"])
    smoke_scope = _smoke(root["smoke"])
    authenticated_inputs = _authenticated_inputs(root["authenticated_inputs"])
    return BenchmarkContract(
        schema_version="dive-benchmark-contract-v1",
        upstream_root=upstream_root,
        upstream_commit=upstream_commit,
        foldseek=foldseek,
        checkpoints=checkpoints,
        inventories=inventories,
        views=views,
        exposure_classes=classes,
        base_checkpoint_exposure=base_exposure,
        exposure_subjects=subjects,
        cutoff_predicates=cutoff_predicates,
        evaluator_required_subjects=evaluator_required,
        cleanliness_required_subjects=cleanliness_required,
        evaluators=evaluators,
        arms=arms,
        smoke_scope=smoke_scope,
        authenticated_inputs=authenticated_inputs,
        source_path=Path(path).resolve(),
    )

def file_identity(path: Path) -> ArtifactIdentity:

    candidate = Path(path)
    try:
        stat = candidate.stat()
    except OSError as error:
        raise BenchmarkContractError(
            f"cannot stat artifact {candidate}: {error}"
        ) from error
    if not candidate.is_file() or candidate.is_symlink():
        raise BenchmarkContractError(
            f"artifact must be a regular non-symlink file: {candidate}"
        )
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError as error:
        raise BenchmarkContractError(
            f"cannot hash artifact {candidate}: {error}"
        ) from error
    return ArtifactIdentity(str(candidate.resolve()), digest.hexdigest(), stat.st_size)

def _foldseek(value: object) -> FoldseekContract:
    section = _mapping(value, "foldseek")
    _exact_keys(
        section,
        {
            "path",
            "sha256",
            "size_bytes",
            "version",
            "threads",
            "min_tm_score",
            "min_shorter_coverage",
        },
        "foldseek",
    )
    version = section["version"]
    if type(version) is not str or not version:
        raise BenchmarkContractError("foldseek.version must be a non-empty string")
    threads = section["threads"]
    if type(threads) is not int or threads < 1 or threads > 256:
        raise BenchmarkContractError("foldseek.threads must be an integer in [1, 256]")
    tm_score = _float(section["min_tm_score"], "foldseek.min_tm_score")
    coverage = _float(section["min_shorter_coverage"], "foldseek.min_shorter_coverage")
    if tm_score != 0.50 or coverage != 0.80:
        raise BenchmarkContractError(
            "Foldseek scientific thresholds are frozen at 0.50 and 0.80"
        )
    identity = ArtifactIdentity(
        str(_path(section["path"], "foldseek.path")),
        _sha(section["sha256"], "foldseek.sha256", length=64),
        _size(section["size_bytes"], "foldseek.size_bytes"),
    )
    return FoldseekContract(identity, version, threads, tm_score, coverage)

def _identities(
    value: object, names: tuple[str, ...], label: str
) -> tuple[tuple[str, ArtifactIdentity], ...]:
    section = _mapping(value, label)
    _exact_keys(section, set(names), label)
    return tuple((name, _identity(section[name], f"{label}.{name}")) for name in names)

def _authenticated_inputs(value: object) -> BenchmarkInputContract:
    section = _mapping(value, "authenticated_inputs")
    _exact_keys(
        section,
        {
            "train_data_ready",
            "loader_manifest_root",
            "resolution_audit",
            "canonical_manifests",
        },
        "authenticated_inputs",
    )
    root = _path(
        section["loader_manifest_root"], "authenticated_inputs.loader_manifest_root"
    )
    expected = Path(
        cache_dir('emergent', 'loader_manifests_v5')
    )
    if root != expected:
        raise BenchmarkContractError(
            "authenticated inputs must use the frozen v5 loader root"
        )
    train_data_ready = _repository_relative_identity(
        section["train_data_ready"], "authenticated_inputs.train_data_ready"
    )
    return BenchmarkInputContract(
        train_data_ready=train_data_ready,
        train_data_ready_relative_path=_TRAIN_DATA_READY_RELATIVE_PATH,
        loader_manifest_root=root,
        resolution_audit=_identity(
            section["resolution_audit"], "authenticated_inputs.resolution_audit"
        ),
        canonical_manifests=_identities(
            section["canonical_manifests"],
            ("binder", "ame", "antibody"),
            "authenticated_inputs.canonical_manifests",
        ),
    )

def _repository_relative_identity(value: object, label: str) -> ArtifactIdentity:
    section = _mapping(value, label)
    _exact_keys(section, {"path", "sha256", "size_bytes"}, label)
    relative = Path(str(section["path"]))
    if (
        type(section["path"]) is not str
        or relative.is_absolute()
        or relative != _TRAIN_DATA_READY_RELATIVE_PATH
    ):
        raise BenchmarkContractError(
            f"{label}.path must be the tracked repository-relative train-data-ready path"
        )
    return ArtifactIdentity(
        str(_REPOSITORY_ROOT / relative),
        _sha(section["sha256"], f"{label}.sha256", length=64),
        _size(section["size_bytes"], f"{label}.size_bytes"),
    )

def _views(value: object) -> ViewContract:
    section = _mapping(value, "views")
    _exact_keys(section, {"names", "min_parent_groups"}, "views")
    names = _string_tuple(section["names"], "views.names")
    if names != ("standard", "strict", "temporal-clean"):
        raise BenchmarkContractError(
            "views.names must be standard, strict, temporal-clean"
        )
    floor = section["min_parent_groups"]
    if type(floor) is not int or floor != 20:
        raise BenchmarkContractError(
            "views.min_parent_groups must be the frozen parent-group floor 20"
        )
    return ViewContract(names, floor)

def _exposure(
    value: object,
    checkpoints: tuple[tuple[str, ArtifactIdentity], ...],
    inventories: tuple[tuple[str, ArtifactIdentity], ...],
    evaluators: tuple[EvaluatorContract, ...],
) -> tuple[
    tuple[ExposureClass, ...],
    ExposureClass,
    tuple[ExposureSubjectContract, ...],
    tuple[CutoffPredicateContract, ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[str, ...],
]:
    section = _mapping(value, "exposure")
    _exact_keys(
        section,
        {
            "classes",
            "base_checkpoint",
            "subjects",
            "cutoff_predicates",
            "evaluator_required_subjects",
            "cleanliness_required_subjects",
        },
        "exposure",
    )
    vocabulary = _string_tuple(section["classes"], "exposure.classes")
    expected = tuple(item.value for item in ExposureClass)
    if vocabulary != expected:
        raise BenchmarkContractError(
            "exposure.classes must use the closed exposure vocabulary"
        )
    try:
        base = ExposureClass(section["base_checkpoint"])
    except (TypeError, ValueError) as error:
        raise BenchmarkContractError(
            "exposure.base_checkpoint is not a valid exposure class"
        ) from error
    if base is not ExposureClass.UNKNOWN:
        raise BenchmarkContractError(
            "base-checkpoint exposure remains unknown without authenticated provenance"
        )
    sources = {
        **{f"checkpoint:{name}": identity for name, identity in checkpoints},
        **{f"inventory:{name}": identity for name, identity in inventories},
    }
    raw_subjects = _mapping(section["subjects"], "exposure.subjects")
    expected_subjects = {
        *(name for name, _ in checkpoints),
        "common_training_manifest",
        "common_cutoff_evidence",
    }
    missing_subjects = sorted(expected_subjects - set(raw_subjects))
    if missing_subjects:
        raise BenchmarkContractError(
            f"exposure.subjects is missing required subject(s) {missing_subjects}"
        )
    allowed_kinds = {
        "checkpoint",
        "template_database",
        "sequence_database",
        "reference_database",
        "training_manifest",
        "cutoff_evidence",
    }
    subjects = []
    for subject in sorted(raw_subjects):
        if re.fullmatch(r"[a-z][a-z0-9_]{2,63}", subject) is None:
            raise BenchmarkContractError(
                f"exposure subject has invalid name {subject!r}"
            )
        entry = _mapping(raw_subjects[subject], f"exposure.subjects.{subject}")
        _exact_keys(entry, {"kind", "source"}, f"exposure.subjects.{subject}")
        kind, source = entry["kind"], entry["source"]
        if type(kind) is not str or kind not in allowed_kinds:
            raise BenchmarkContractError(f"exposure subject {subject} has invalid kind")
        if type(source) is not str or source not in sources:
            raise BenchmarkContractError(
                f"exposure subject {subject} has unknown source"
            )
        if subject in dict(checkpoints) and (
            kind != "checkpoint" or source != f"checkpoint:{subject}"
        ):
            raise BenchmarkContractError(
                f"checkpoint exposure subject {subject} must bind itself"
            )
        subjects.append(ExposureSubjectContract(subject, kind, sources[source]))
    subject_kinds = {item.subject: item.kind for item in subjects}
    raw_predicates = section["cutoff_predicates"]
    if not isinstance(raw_predicates, list):
        raise BenchmarkContractError("exposure.cutoff_predicates must be a list")
    subject_sources = {item.subject: item.source for item in subjects}
    predicate_keys: set[tuple[str, str]] = set()
    predicates = []
    allowed_families = {item.family for item in evaluators}
    for index, raw_predicate in enumerate(raw_predicates):
        entry = _mapping(raw_predicate, f"exposure.cutoff_predicates[{index}]")
        _exact_keys(
            entry,
            {"subject", "evaluator_family", "condition", "source"},
            f"exposure.cutoff_predicates[{index}]",
        )
        subject = entry["subject"]
        family = entry["evaluator_family"]
        condition = entry["condition"]
        source = entry["source"]
        if (
            type(subject) is not str
            or subject_kinds.get(subject) != "cutoff_evidence"
            or type(family) is not str
            or family not in allowed_families
            or condition != "evaluator_reference_corpus_precedes_cutoff"
            or type(source) is not str
            or source not in sources
        ):
            raise BenchmarkContractError(
                "cutoff predicate is not a registered evaluator-specific condition"
            )
        if sources[source] == subject_sources[subject]:
            raise BenchmarkContractError(
                "cutoff predicate must be distinct from its cutoff evidence"
            )
        key = (subject, family)
        if key in predicate_keys:
            raise BenchmarkContractError(
                "cutoff predicate repeats a subject/evaluator family"
            )
        predicate_keys.add(key)
        predicates.append(
            CutoffPredicateContract(subject, family, condition, sources[source])
        )
    raw_required = _mapping(
        section["evaluator_required_subjects"], "exposure.evaluator_required_subjects"
    )
    evaluator_names = tuple(item.family for item in evaluators)
    _exact_keys(
        raw_required, set(evaluator_names), "exposure.evaluator_required_subjects"
    )
    evaluator_required = []
    database_kinds = {"template_database", "sequence_database", "reference_database"}
    for family in evaluator_names:
        required = _string_tuple(
            raw_required[family], f"exposure.evaluator_required_subjects.{family}"
        )
        if len(set(required)) != len(required):
            raise BenchmarkContractError(
                f"evaluator {family} repeats an exposure subject"
            )
        for subject in required:
            if (
                subject not in subject_kinds
                or subject_kinds[subject] not in database_kinds
            ):
                raise BenchmarkContractError(
                    f"evaluator {family} requires an unregistered database subject"
                )
        evaluator_required.append((family, required))
    cleanliness = _string_tuple(
        section["cleanliness_required_subjects"],
        "exposure.cleanliness_required_subjects",
    )
    if len(set(cleanliness)) != len(cleanliness) or not cleanliness:
        raise BenchmarkContractError(
            "cleanliness_required_subjects must be a non-empty unique list"
        )
    for subject in cleanliness:
        if subject not in subject_kinds or subject_kinds[subject] not in {
            "training_manifest",
            "cutoff_evidence",
        }:
            raise BenchmarkContractError(
                "cleanliness evidence lacks its own registered subject"
            )
    return (
        tuple(ExposureClass(item) for item in vocabulary),
        base,
        tuple(subjects),
        tuple(predicates),
        tuple(evaluator_required),
        cleanliness,
    )

def _evaluators(value: object) -> tuple[EvaluatorContract, ...]:
    section = _mapping(value, "evaluators")
    _exact_keys(section, {name for name, _primary in _EVALUATORS}, "evaluators")
    result = []
    for family, primary in _EVALUATORS:
        entry = _mapping(section[family], f"evaluators.{family}")
        _exact_keys(entry, {"primary"}, f"evaluators.{family}")
        if entry["primary"] != primary:
            raise BenchmarkContractError(
                f"evaluators.{family}.primary is frozen as {primary}"
            )
        result.append(EvaluatorContract(family, primary))
    return tuple(result)

def _arms(value: object) -> tuple[str, ...]:
    arms = _string_tuple(value, "arms")
    if arms != _ARMS:
        raise BenchmarkContractError(
            "arms must match the frozen evaluation-arm vocabulary"
        )
    return arms

def _smoke(value: object) -> str:
    section = _mapping(value, "smoke")
    _exact_keys(section, {"scope"}, "smoke")
    if section["scope"] != "train_only":
        raise BenchmarkContractError(
            "smoke.scope must be train_only; blind access remains sealed"
        )
    return "train_only"

def _identity(value: object, label: str) -> ArtifactIdentity:
    section = _mapping(value, label)
    _exact_keys(section, {"path", "sha256", "size_bytes"}, label)
    path = _path(section["path"], f"{label}.path")
    return ArtifactIdentity(
        str(path),
        _sha(section["sha256"], f"{label}.sha256", length=64),
        _size(section["size_bytes"], f"{label}.size_bytes"),
    )

def _identity_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise BenchmarkContractError(f"{label} must be a string-keyed mapping")
    return value

def _exact_keys(
    value: Mapping[str, object], expected: set[str] | frozenset[str], label: str
) -> None:
    unknown = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unknown or missing:
        raise BenchmarkContractError(
            f"{label} has unknown {unknown} or missing {missing} field(s)"
        )

def _path(value: object, label: str) -> Path:
    if type(value) is not str or not value:
        raise BenchmarkContractError(f"{label} must be a non-empty path string")
    return Path(os.path.abspath(value))

def _sha(value: object, label: str, *, length: int) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or not set(value) <= _SHA256_CHARACTERS
    ):
        raise BenchmarkContractError(
            f"{label} must be a lowercase {length}-character SHA-256 identity"
        )
    return value

def _size(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise BenchmarkContractError(f"{label} must be a non-negative built-in integer")
    return value

def _float(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise BenchmarkContractError(f"{label} must be a built-in number")
    return float(value)

def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(type(item) is str for item in value):
        raise BenchmarkContractError(f"{label} must be a list of strings")
    return tuple(value)
