
from __future__ import annotations

import errno
import ctypes
import hashlib
import json
import math
import os
import platform
import re
import secrets
import socket
import stat
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.paper_inputs import PaperInputBundle
from dive.benchmark.paper_registry import (
    PaperEvaluatorRegistry,
    PaperProtocolIdentity,
    PaperRegistryError,
    compose_paper_registry_semantic_identity,
    paper_protocol_identity_token,
    verify_paper_evaluator_registry,
)
from dive.signed_value import roots as signed_value_roots
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_EVIDENCE_ROOT,
    RUNTIME_PYTHON,
)
from dive.training.preflight import canonical_json_bytes

class AuditError(RuntimeError):
    pass

class AuditUnit(StrEnum):
    EXAMPLE_ROW = "example_row"
    PARENT_GROUP = "parent_group"

_SCHEMA_ROW = "dive-paper-protocol-audit-row-v1"
_SCHEMA_PARENT = "dive-paper-protocol-audit-parent-v1"
_SCHEMA_SUMMARY = "dive-paper-protocol-audit-summary-v1"
_SCHEMA_ATTEMPT = "dive-paper-protocol-audit-attempt-v1"
_SCHEMA_ENVIRONMENT = "dive-paper-protocol-audit-environment-v1"
_SCHEMA_MANIFEST = "dive-paper-protocol-audit-manifest-v1"
_SCHEMA_COMPLETION = "dive-paper-protocol-audit-completion-v1"
_FAMILIES = ("binder", "ame", "antibody")
_BACKBONE = ("N", "CA", "C", "O")
TYPED_REQUESTED_CELL_FAILURES = frozenset({"loader_exclusion", "incomplete_backbone"})
_SHA256 = frozenset("0123456789abcdef")
_GIT_SHA = frozenset("0123456789abcdef")
_RUN_ID = re.compile(r"paper-baseline-protocol-audit-[a-z0-9][a-z0-9-]{2,63}")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALLOWED_ENVIRONMENT_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_OFFLINE",
        "PATH",
        "PYTHONPATH",
        "USE_V2_COMPLEXA_ARCH",
    }
)

_TEST_EVIDENCE_ROOT: Path | None = None

def _require_sha256(label: str, value: object) -> str:
    if type(value) is not str or len(value) != 64 or set(value) - _SHA256:
        raise AuditError(f"{label} must be a lowercase sha256")
    return value

def _require_nonempty(label: str, value: object) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise AuditError(f"{label} must be a nonempty string")
    return value

def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))

@dataclass(frozen=True, slots=True)
class GitCheckoutIdentity:
    root: str
    commit: str
    clean: bool

@dataclass(frozen=True, slots=True)
class AuditAuthentication:
    repository: GitCheckoutIdentity
    interpreter: ArtifactIdentity
    interpreter_environment: ArtifactIdentity
    interpreter_version: str
    view_semantic_hash: str
    view_partition_hash: str
    input_identities: tuple[ArtifactIdentity, ...]
    upstreams: tuple[GitCheckoutIdentity, ...]
    checkpoints: Mapping[str, ArtifactIdentity]
    evaluator_registry_base_sha256: str
    protocol_identities: tuple[PaperProtocolIdentity, ...]
    evaluator_registry_semantic_sha256: str

@dataclass(frozen=True, slots=True)
class BinderResidueObservation:
    residue_key: str
    coordinate_source: str
    before_atoms: Mapping[str, tuple[float, float, float]]
    normalized_atoms: Mapping[str, tuple[float, float, float]]

@dataclass(frozen=True, slots=True)
class BinderAudit:
    total_residues: int
    missing_backbone_by_coordinate_source: Mapping[str, Mapping[str, int]]
    repairable_residues: int
    unrepairable_residues: int
    four_atom_preflight_result: bool
    unchanged_coordinate_proof: bool

@dataclass(frozen=True, slots=True)
class AmeComponentObservation:
    component_token: str
    roles: tuple[str, ...]
    representation: str
    atom_membership_sha256: str
    coordinate_sha256: str
    normalized_atom_membership_sha256: str
    normalized_coordinate_sha256: str

@dataclass(frozen=True, slots=True)
class AmeObservation:
    canonical_component_token: str | None
    canonical_identity: str | None
    canonical_smiles: str | None
    components: tuple[AmeComponentObservation, ...]
    parser_normalization_result: bool
    normalization_identity: str

@dataclass(frozen=True, slots=True)
class AmeAudit:
    ligand_class: str
    canonical_identity_available: bool
    parser_normalization_result: bool
    unchanged_membership_proof: bool
    unchanged_coordinate_proof: bool

@dataclass(frozen=True, slots=True, order=True)
class ResidueKey:
    chain_id: str
    auth_seq_id: int
    insertion_code: str

@dataclass(frozen=True, slots=True, order=True)
class AtomKey:
    residue_key: ResidueKey
    atom_name: str

@dataclass(frozen=True, slots=True)
class AntibodyResidueObservation:
    source_key: ResidueKey
    frozen_h3: bool
    predicted_keys: tuple[ResidueKey, ...]
    source_atoms: tuple[AtomKey, ...]
    predicted_atoms: tuple[AtomKey, ...]

@dataclass(frozen=True, slots=True)
class AntibodyAudit:
    insertion_coded_source_count: int
    insertion_coded_h3_count: int
    correspondences: tuple[AntibodyResidueObservation, ...]
    source_atom_keys: tuple[AtomKey, ...]
    predicted_atom_keys: tuple[AtomKey, ...]
    duplicate_keys_before: int
    duplicate_keys_after: int
    mapping_ambiguity: bool
    common_h3_ca_count: int
    evaluable: bool

@dataclass(frozen=True, slots=True)
class ExampleAuditRecord:
    unit: AuditUnit
    family: str
    parent_id: str
    example_id: str
    admitted: bool
    representable: bool
    blocker_codes: tuple[str, ...]
    normalization_identity: str
    binder: BinderAudit | None = None
    ame: AmeAudit | None = None
    antibody: AntibodyAudit | None = None
    requested_cell_failure: str | None = None

@dataclass(frozen=True, slots=True)
class ParentAuditRecord:
    unit: AuditUnit
    family: str
    parent_id: str
    example_rows: int
    representable_rows: int
    unrepresentable_rows: int

@dataclass(frozen=True, slots=True)
class AuditExpectedCounts:
    example_rows: int
    parent_groups: int

@dataclass(frozen=True, slots=True)
class AuditSummary:
    example_counts: Mapping[str, int]
    parent_counts: Mapping[str, int]
    parents: tuple[ParentAuditRecord, ...]
    family_outputs: Mapping[str, Mapping[str, object]]
    unrepresentable_admitted_rows: int
    replacement_smoke_blocked: bool

def _coordinate_mapping(
    value: Mapping[str, tuple[float, float, float]], *, label: str
) -> dict[str, tuple[float, float, float]]:
    if not isinstance(value, Mapping):
        raise AuditError(f"{label} must be an atom mapping")
    observed: dict[str, tuple[float, float, float]] = {}
    for atom, coordinate in value.items():
        name = _require_nonempty(f"{label} atom", atom)
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 3
            or any(type(item) not in {int, float} for item in coordinate)
            or any(not math.isfinite(float(item)) for item in coordinate)
        ):
            raise AuditError(f"{label} coordinate for {name} must be finite xyz")
        observed[name] = tuple(float(item) for item in coordinate)
    return observed

def scan_binder_example(
    *,
    parent_id: str,
    example_id: str,
    residues: tuple[BinderResidueObservation, ...],
    normalization_identity: str,
    requested_cell_failure: str | None = None,
) -> ExampleAuditRecord:

    _require_nonempty("parent_id", parent_id)
    _require_nonempty("example_id", example_id)
    identity = _require_nonempty("normalization_identity", normalization_identity)
    if not isinstance(residues, tuple) or not residues:
        raise AuditError("binder residues must be a nonempty tuple")
    seen: set[str] = set()
    missing: dict[str, dict[str, int]] = {}
    repairable = 0
    unrepairable = 0
    unchanged = True
    preflight = True
    blocker_codes: set[str] = set()
    for residue in residues:
        if not isinstance(residue, BinderResidueObservation):
            raise AuditError("malformed binder residue")
        key = _require_nonempty("binder residue key", residue.residue_key)
        if key in seen:
            raise AuditError(f"duplicate binder residue key {key!r}")
        seen.add(key)
        source = _require_nonempty(
            "binder coordinate source", residue.coordinate_source
        )
        before = _coordinate_mapping(residue.before_atoms, label=f"{key} before")
        normalized = _coordinate_mapping(
            residue.normalized_atoms, label=f"{key} normalized"
        )
        counts = missing.setdefault(source, {atom: 0 for atom in _BACKBONE})
        for atom in _BACKBONE:
            if atom not in before:
                counts[atom] += 1
        changed = any(
            atom not in normalized or normalized[atom] != coordinate
            for atom, coordinate in before.items()
        )
        if changed:
            unchanged = False
            blocker_codes.add("existing_coordinate_changed")
        extra = set(normalized) - set(before)
        if extra - {"O"}:
            unchanged = False
            blocker_codes.add("unauthorized_atom_added")
        residue_ready = all(atom in normalized for atom in _BACKBONE)
        preflight &= residue_ready
        if residue_ready and not changed and not (extra - {"O"}):
            repairable += 1
        else:
            unrepairable += 1
    if not preflight:
        blocker_codes.add("four_atom_preflight_failed")
    if requested_cell_failure is not None:
        if requested_cell_failure not in TYPED_REQUESTED_CELL_FAILURES:
            raise AuditError("unknown requested-cell failure")
        blocker_codes.add(requested_cell_failure)
    audit = BinderAudit(
        total_residues=len(residues),
        missing_backbone_by_coordinate_source=MappingProxyType(
            {
                source: MappingProxyType(dict(counts))
                for source, counts in sorted(missing.items())
            }
        ),
        repairable_residues=repairable,
        unrepairable_residues=unrepairable,
        four_atom_preflight_result=preflight,
        unchanged_coordinate_proof=unchanged,
    )
    return ExampleAuditRecord(
        AuditUnit.EXAMPLE_ROW,
        "binder",
        parent_id,
        example_id,
        True,
        not blocker_codes and unrepairable == 0,
        tuple(sorted(blocker_codes)),
        identity,
        binder=audit,
        requested_cell_failure=requested_cell_failure,
    )

def scan_ame_example(
    *, parent_id: str, example_id: str, observation: AmeObservation
) -> ExampleAuditRecord:

    _require_nonempty("parent_id", parent_id)
    _require_nonempty("example_id", example_id)
    if not isinstance(observation, AmeObservation):
        raise AuditError("malformed AME observation")
    identity = _require_nonempty(
        "normalization_identity", observation.normalization_identity
    )
    if not isinstance(observation.components, tuple):
        raise AuditError("AME components must be a tuple")
    if type(observation.parser_normalization_result) is not bool:
        raise AuditError("AME parser-normalization result must be boolean")
    canonical_available = all(
        type(value) is str and bool(value.strip())
        for value in (
            observation.canonical_component_token,
            observation.canonical_identity,
            observation.canonical_smiles,
        )
    )
    blockers: set[str] = set()
    if not canonical_available:
        blockers.add("missing_canonical_identity")
    token = observation.canonical_component_token or ""
    selected: list[AmeComponentObservation] = []
    for component in observation.components:
        if not isinstance(component, AmeComponentObservation):
            raise AuditError("malformed AME component")
        _require_nonempty("AME component token", component.component_token)
        if not isinstance(component.roles, tuple) or any(
            type(role) is not str or not role for role in component.roles
        ):
            raise AuditError("AME component roles must be a nonempty string tuple")
        if not component.roles:
            raise AuditError("AME component roles must not be empty")
        if component.representation not in {"non_polymer", "polymer_peptide", "water"}:
            raise AuditError("unknown AME component representation")
        for label, digest in (
            ("atom membership", component.atom_membership_sha256),
            ("coordinates", component.coordinate_sha256),
            ("normalized atom membership", component.normalized_atom_membership_sha256),
            ("normalized coordinates", component.normalized_coordinate_sha256),
        ):
            _require_sha256(f"AME {label}", digest)
        if component.component_token == token:
            selected.append(component)
    unchanged_membership = all(
        item.atom_membership_sha256 == item.normalized_atom_membership_sha256
        for item in selected
    )
    unchanged_coordinates = all(
        item.coordinate_sha256 == item.normalized_coordinate_sha256 for item in selected
    )
    if not selected:
        ligand_class = "missing"
        blockers.add("missing_canonical_component")
    elif len(selected) != 1:
        ligand_class = "ambiguous"
        blockers.add("ambiguous_canonical_component")
    else:
        component = selected[0]
        if component.roles != ("ligand",):
            ligand_class = "mixed"
            blockers.add("mixed_role_membership")
        elif component.representation == "water":
            ligand_class = "water_only"
            blockers.add("water_only_ligand")
        else:
            ligand_class = component.representation
    if not unchanged_membership:
        blockers.add("component_membership_changed")
    if not unchanged_coordinates:
        blockers.add("ligand_coordinates_changed")
    if not observation.parser_normalization_result:
        blockers.add("parser_normalization_failed")
    audit = AmeAudit(
        ligand_class,
        canonical_available,
        observation.parser_normalization_result,
        unchanged_membership,
        unchanged_coordinates,
    )
    return ExampleAuditRecord(
        AuditUnit.EXAMPLE_ROW,
        "ame",
        parent_id,
        example_id,
        True,
        not blockers,
        tuple(sorted(blockers)),
        identity,
        ame=audit,
    )

def _validate_residue_key(key: ResidueKey, *, label: str) -> ResidueKey:
    if not isinstance(key, ResidueKey):
        raise AuditError(f"{label} must be a ResidueKey")
    _require_nonempty(f"{label} chain", key.chain_id)
    if type(key.auth_seq_id) is not int:
        raise AuditError(f"{label} auth_seq_id must be an integer")
    if type(key.insertion_code) is not str:
        raise AuditError(f"{label} insertion_code must be an explicit string")
    return key

def _validate_atom_key(key: AtomKey, *, label: str) -> AtomKey:
    if not isinstance(key, AtomKey):
        raise AuditError(f"{label} must be an AtomKey")
    _validate_residue_key(key.residue_key, label=f"{label} residue")
    atom_name = _require_nonempty(f"{label} atom name", key.atom_name)
    if any(character.isspace() for character in atom_name):
        raise AuditError(f"{label} atom name must not contain whitespace")
    return key

def _excess_duplicates(values: Sequence[object]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)

def scan_antibody_example(
    *,
    parent_id: str,
    example_id: str,
    residues: tuple[AntibodyResidueObservation, ...],
    normalization_identity: str,
) -> ExampleAuditRecord:

    _require_nonempty("parent_id", parent_id)
    _require_nonempty("example_id", example_id)
    identity = _require_nonempty("normalization_identity", normalization_identity)
    if not isinstance(residues, tuple) or not residues:
        raise AuditError("antibody residues must be a nonempty tuple")
    source_keys: list[ResidueKey] = []
    source_atoms: list[AtomKey] = []
    numeric_atom_keys: list[tuple[str, int, str]] = []
    predicted_atoms: list[AtomKey] = []
    mapping_ambiguity = False
    identity_mismatch = False
    common_h3 = 0
    for residue in residues:
        if not isinstance(residue, AntibodyResidueObservation):
            raise AuditError("malformed antibody residue")
        source = _validate_residue_key(residue.source_key, label="source residue")
        if type(residue.frozen_h3) is not bool:
            raise AuditError("frozen_h3 must be boolean")
        if (
            not isinstance(residue.predicted_keys, tuple)
            or not isinstance(residue.source_atoms, tuple)
            or not isinstance(residue.predicted_atoms, tuple)
        ):
            raise AuditError("antibody residue and atom keys must be tuples")
        source_keys.append(source)
        validated = tuple(
            _validate_residue_key(item, label="predicted residue")
            for item in residue.predicted_keys
        )
        validated_source_atoms = tuple(
            _validate_atom_key(item, label="source atom")
            for item in residue.source_atoms
        )
        validated_predicted_atoms = tuple(
            _validate_atom_key(item, label="predicted atom")
            for item in residue.predicted_atoms
        )
        if any(item.residue_key != source for item in validated_source_atoms):
            identity_mismatch = True
        if any(item.residue_key not in validated for item in validated_predicted_atoms):
            identity_mismatch = True
        source_atoms.extend(validated_source_atoms)
        predicted_atoms.extend(validated_predicted_atoms)
        numeric_atom_keys.extend(
            (source.chain_id, source.auth_seq_id, item.atom_name)
            for item in validated_source_atoms
        )
        if len(validated) != 1:
            mapping_ambiguity = True
        else:
            if validated[0] != source:
                identity_mismatch = True
            if (
                residue.frozen_h3
                and validated[0] == source
                and validated_source_atoms.count(AtomKey(source, "CA")) == 1
                and validated_predicted_atoms.count(AtomKey(source, "CA")) == 1
            ):
                common_h3 += 1
    duplicate_before = _excess_duplicates(numeric_atom_keys)
    duplicate_after = _excess_duplicates(source_atoms) + _excess_duplicates(
        predicted_atoms
    )
    blockers: set[str] = set()
    if mapping_ambiguity:
        blockers.add("ambiguous_role_mapping")
    if identity_mismatch:
        blockers.add("atom_identity_mismatch")
    if duplicate_after:
        blockers.add("duplicate_insertion_aware_atom_key")
    if common_h3 < 3:
        blockers.add("fewer_than_three_common_h3_ca")
    evaluable = not blockers
    audit = AntibodyAudit(
        insertion_coded_source_count=sum(
            bool(key.insertion_code) for key in source_keys
        ),
        insertion_coded_h3_count=sum(
            bool(item.source_key.insertion_code) and item.frozen_h3 for item in residues
        ),
        correspondences=residues,
        source_atom_keys=tuple(sorted(source_atoms)),
        predicted_atom_keys=tuple(sorted(predicted_atoms)),
        duplicate_keys_before=duplicate_before,
        duplicate_keys_after=duplicate_after,
        mapping_ambiguity=mapping_ambiguity,
        common_h3_ca_count=common_h3,
        evaluable=evaluable,
    )
    return ExampleAuditRecord(
        AuditUnit.EXAMPLE_ROW,
        "antibody",
        parent_id,
        example_id,
        True,
        evaluable,
        tuple(sorted(blockers)),
        identity,
        antibody=audit,
    )

def typed_requested_cell_failure_record(
    *,
    family: str,
    parent_id: str,
    example_id: str,
    reason_code: str,
    normalization_identity: str,
) -> ExampleAuditRecord:

    if family not in _FAMILIES:
        raise AuditError("row family is unknown")
    if reason_code not in TYPED_REQUESTED_CELL_FAILURES:
        raise AuditError("unknown requested-cell failure")
    identity = _require_nonempty("normalization_identity", normalization_identity)
    _require_nonempty("parent_id", parent_id)
    _require_nonempty("example_id", example_id)
    binder = None
    ame = None
    antibody = None
    if family == "binder":
        binder = BinderAudit(0, MappingProxyType({}), 0, 0, False, True)
    elif family == "ame":
        ame = AmeAudit("missing", False, False, True, True)
    else:
        antibody = AntibodyAudit(0, 0, (), (), (), 0, 0, False, 0, False)
    return ExampleAuditRecord(
        AuditUnit.EXAMPLE_ROW,
        family,
        parent_id,
        example_id,
        True,
        False,
        (reason_code,),
        identity,
        binder=binder,
        ame=ame,
        antibody=antibody,
        requested_cell_failure=reason_code,
    )

def _binder_mapping(value: BinderAudit) -> dict[str, object]:
    return {
        "total_residues": value.total_residues,
        "missing_backbone_by_coordinate_source": {
            source: dict(counts)
            for source, counts in value.missing_backbone_by_coordinate_source.items()
        },
        "repairable_residues": value.repairable_residues,
        "unrepairable_residues": value.unrepairable_residues,
        "four_atom_preflight_result": value.four_atom_preflight_result,
        "unchanged_coordinate_proof": value.unchanged_coordinate_proof,
    }

def _ame_mapping(value: AmeAudit) -> dict[str, object]:
    return {
        "ligand_class": value.ligand_class,
        "canonical_identity_available": value.canonical_identity_available,
        "parser_normalization_result": value.parser_normalization_result,
        "unchanged_membership_proof": value.unchanged_membership_proof,
        "unchanged_coordinate_proof": value.unchanged_coordinate_proof,
    }

def _residue_key_mapping(value: ResidueKey) -> dict[str, object]:
    return {
        "chain_id": value.chain_id,
        "auth_seq_id": value.auth_seq_id,
        "insertion_code": value.insertion_code,
    }

def _atom_key_mapping(value: AtomKey) -> dict[str, object]:
    return {
        **_residue_key_mapping(value.residue_key),
        "atom_name": value.atom_name,
    }

def _antibody_observation_mapping(
    value: AntibodyResidueObservation,
) -> dict[str, object]:
    return {
        "source_key": _residue_key_mapping(value.source_key),
        "frozen_h3": value.frozen_h3,
        "predicted_keys": [_residue_key_mapping(item) for item in value.predicted_keys],
        "source_atoms": [_atom_key_mapping(item) for item in value.source_atoms],
        "predicted_atoms": [_atom_key_mapping(item) for item in value.predicted_atoms],
    }

def _antibody_mapping(value: AntibodyAudit) -> dict[str, object]:
    return {
        "insertion_coded_source_count": value.insertion_coded_source_count,
        "insertion_coded_h3_count": value.insertion_coded_h3_count,
        "correspondences": [
            _antibody_observation_mapping(item) for item in value.correspondences
        ],
        "source_atom_keys": [
            _atom_key_mapping(item) for item in value.source_atom_keys
        ],
        "predicted_atom_keys": [
            _atom_key_mapping(item) for item in value.predicted_atom_keys
        ],
        "duplicate_keys_before": value.duplicate_keys_before,
        "duplicate_keys_after": value.duplicate_keys_after,
        "mapping_ambiguity": value.mapping_ambiguity,
        "common_h3_ca_count": value.common_h3_ca_count,
        "evaluable": value.evaluable,
    }

def example_audit_record_mapping(record: ExampleAuditRecord) -> dict[str, object]:
    if not isinstance(record, ExampleAuditRecord):
        raise AuditError("expected an ExampleAuditRecord")
    return {
        "schema_version": _SCHEMA_ROW,
        "unit": record.unit.value,
        "family": record.family,
        "parent_id": record.parent_id,
        "example_id": record.example_id,
        "admitted": record.admitted,
        "representable": record.representable,
        "blocker_codes": list(record.blocker_codes),
        "normalization_identity": record.normalization_identity,
        "requested_cell_failure": record.requested_cell_failure,
        "binder": None if record.binder is None else _binder_mapping(record.binder),
        "ame": None if record.ame is None else _ame_mapping(record.ame),
        "antibody": (
            None if record.antibody is None else _antibody_mapping(record.antibody)
        ),
    }

_ROW_KEYS = {
    "schema_version",
    "unit",
    "family",
    "parent_id",
    "example_id",
    "admitted",
    "representable",
    "blocker_codes",
    "normalization_identity",
    "requested_cell_failure",
    "binder",
    "ame",
    "antibody",
}

def _exact_mapping(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise AuditError(f"{label} keys are malformed")
    return value

def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise AuditError(f"{label} must be boolean")
    return value

def _exact_nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AuditError(f"{label} must be a nonnegative integer")
    return value

def _parse_residue_key_mapping(value: object, *, label: str) -> ResidueKey:
    mapping = _exact_mapping(
        value,
        {"chain_id", "auth_seq_id", "insertion_code"},
        label,
    )
    key = ResidueKey(
        _require_nonempty(f"{label} chain", mapping["chain_id"]),
        mapping["auth_seq_id"],
        mapping["insertion_code"],
    )
    return _validate_residue_key(key, label=label)

def _parse_atom_key_mapping(value: object, *, label: str) -> AtomKey:
    mapping = _exact_mapping(
        value,
        {"chain_id", "auth_seq_id", "insertion_code", "atom_name"},
        label,
    )
    residue_key = _parse_residue_key_mapping(
        {
            "chain_id": mapping["chain_id"],
            "auth_seq_id": mapping["auth_seq_id"],
            "insertion_code": mapping["insertion_code"],
        },
        label=f"{label} residue",
    )
    return _validate_atom_key(
        AtomKey(residue_key, mapping["atom_name"]),
        label=label,
    )

def parse_example_audit_record(value: object) -> ExampleAuditRecord:

    raw = _exact_mapping(value, _ROW_KEYS, "row")
    if raw["schema_version"] != _SCHEMA_ROW or raw["unit"] != AuditUnit.EXAMPLE_ROW:
        raise AuditError("row schema/unit is malformed")
    family = _require_nonempty("row family", raw["family"])
    if family not in _FAMILIES:
        raise AuditError("row family is unknown")
    parent_id = _require_nonempty("row parent_id", raw["parent_id"])
    example_id = _require_nonempty("row example_id", raw["example_id"])
    normalization_identity = _require_nonempty(
        "row normalization identity", raw["normalization_identity"]
    )
    admitted = _exact_bool(raw["admitted"], "row admitted")
    if not admitted:
        raise AuditError("audit rows cannot change frozen membership")
    representable = _exact_bool(raw["representable"], "row representable")
    requested_raw = raw["requested_cell_failure"]
    if requested_raw is None:
        requested_cell_failure = None
    else:
        requested_cell_failure = _require_nonempty(
            "requested_cell_failure", requested_raw
        )
        if requested_cell_failure not in TYPED_REQUESTED_CELL_FAILURES:
            raise AuditError("unknown requested-cell failure")
        if representable:
            raise AuditError("typed requested-cell failure cannot be representable")
    blockers_raw = raw["blocker_codes"]
    if not isinstance(blockers_raw, list) or any(
        type(item) is not str or not item for item in blockers_raw
    ):
        raise AuditError("row blocker_codes are malformed")
    blockers = tuple(blockers_raw)
    if representable == bool(blockers):
        raise AuditError("row representability/blockers disagree")
    if requested_cell_failure is not None and requested_cell_failure not in blockers:
        raise AuditError("typed requested-cell failure missing from blockers")
    details = {name: raw[name] for name in _FAMILIES}
    if details[family] is None or any(
        details[name] is not None for name in _FAMILIES if name != family
    ):
        raise AuditError("row must carry exactly one family detail")
    binder = None
    ame = None
    antibody = None
    if family == "binder":
        item = _exact_mapping(
            details[family],
            {
                "total_residues",
                "missing_backbone_by_coordinate_source",
                "repairable_residues",
                "unrepairable_residues",
                "four_atom_preflight_result",
                "unchanged_coordinate_proof",
            },
            "binder detail",
        )
        missing_raw = item["missing_backbone_by_coordinate_source"]
        if not isinstance(missing_raw, Mapping):
            raise AuditError("binder missing-backbone counts are malformed")
        missing: dict[str, Mapping[str, int]] = {}
        for source, counts in missing_raw.items():
            source_name = _require_nonempty("binder coordinate source", source)
            count_map = _exact_mapping(counts, set(_BACKBONE), "binder atom counts")
            missing[source_name] = MappingProxyType(
                {
                    atom: _exact_nonnegative(count_map[atom], f"binder missing {atom}")
                    for atom in _BACKBONE
                }
            )
        binder = BinderAudit(
            _exact_nonnegative(item["total_residues"], "binder total residues"),
            MappingProxyType(missing),
            _exact_nonnegative(item["repairable_residues"], "binder repairable"),
            _exact_nonnegative(item["unrepairable_residues"], "binder unrepairable"),
            _exact_bool(item["four_atom_preflight_result"], "binder preflight"),
            _exact_bool(item["unchanged_coordinate_proof"], "binder unchanged"),
        )
        if (
            binder.repairable_residues + binder.unrepairable_residues
            != binder.total_residues
        ):
            raise AuditError("binder residue counts do not reconcile")
        binder_representable = (
            binder.unrepairable_residues == 0
            and binder.four_atom_preflight_result
            and binder.unchanged_coordinate_proof
        )
        if representable != binder_representable:
            raise AuditError("binder detail contradicts row representability")
    elif family == "ame":
        item = _exact_mapping(
            details[family],
            {
                "ligand_class",
                "canonical_identity_available",
                "parser_normalization_result",
                "unchanged_membership_proof",
                "unchanged_coordinate_proof",
            },
            "AME detail",
        )
        ligand_class = _require_nonempty("AME ligand class", item["ligand_class"])
        if ligand_class not in {
            "non_polymer",
            "polymer_peptide",
            "water_only",
            "missing",
            "ambiguous",
            "mixed",
        }:
            raise AuditError("unknown AME ligand class")
        ame = AmeAudit(
            ligand_class,
            _exact_bool(item["canonical_identity_available"], "AME identity"),
            _exact_bool(item["parser_normalization_result"], "AME parser result"),
            _exact_bool(item["unchanged_membership_proof"], "AME membership proof"),
            _exact_bool(item["unchanged_coordinate_proof"], "AME coordinate proof"),
        )
        ame_representable = (
            ame.ligand_class in {"non_polymer", "polymer_peptide"}
            and ame.canonical_identity_available
            and ame.parser_normalization_result
            and ame.unchanged_membership_proof
            and ame.unchanged_coordinate_proof
        )
        if representable != ame_representable:
            raise AuditError("AME detail contradicts row representability")
    else:
        item = _exact_mapping(
            details[family],
            {
                "insertion_coded_source_count",
                "insertion_coded_h3_count",
                "correspondences",
                "source_atom_keys",
                "predicted_atom_keys",
                "duplicate_keys_before",
                "duplicate_keys_after",
                "mapping_ambiguity",
                "common_h3_ca_count",
                "evaluable",
            },
            "antibody detail",
        )
        parsed_atom_keys: dict[str, tuple[AtomKey, ...]] = {}
        for key_name in ("source_atom_keys", "predicted_atom_keys"):
            raw_keys = item[key_name]
            if not isinstance(raw_keys, list):
                raise AuditError(f"antibody {key_name} must be a list")
            parsed_atom_keys[key_name] = tuple(
                _parse_atom_key_mapping(raw_key, label=f"antibody {key_name} item")
                for raw_key in raw_keys
            )
        raw_correspondences = item["correspondences"]
        if requested_cell_failure == "loader_exclusion":
            if raw_correspondences != []:
                raise AuditError(
                    "typed antibody loader exclusion cannot carry dummy correspondences"
                )
            antibody = AntibodyAudit(0, 0, (), (), (), 0, 0, False, 0, False)
            if representable or antibody.evaluable:
                raise AuditError("typed requested-cell failure cannot be representable")
            return ExampleAuditRecord(
                AuditUnit.EXAMPLE_ROW,
                family,
                parent_id,
                example_id,
                True,
                representable,
                blockers,
                normalization_identity,
                binder,
                ame,
                antibody,
                requested_cell_failure=requested_cell_failure,
            )
        if not isinstance(raw_correspondences, list) or not raw_correspondences:
            raise AuditError("antibody correspondences must be a nonempty list")
        correspondences: list[AntibodyResidueObservation] = []
        for raw_correspondence in raw_correspondences:
            correspondence = _exact_mapping(
                raw_correspondence,
                {
                    "source_key",
                    "frozen_h3",
                    "predicted_keys",
                    "source_atoms",
                    "predicted_atoms",
                },
                "antibody correspondence",
            )
            predicted_keys_raw = correspondence["predicted_keys"]
            source_atoms_raw = correspondence["source_atoms"]
            predicted_atoms_raw = correspondence["predicted_atoms"]
            if (
                not isinstance(predicted_keys_raw, list)
                or not isinstance(source_atoms_raw, list)
                or not isinstance(predicted_atoms_raw, list)
            ):
                raise AuditError("antibody correspondence keys/atoms must be lists")
            correspondences.append(
                AntibodyResidueObservation(
                    source_key=_parse_residue_key_mapping(
                        correspondence["source_key"], label="antibody source key"
                    ),
                    frozen_h3=_exact_bool(
                        correspondence["frozen_h3"], "antibody frozen H3"
                    ),
                    predicted_keys=tuple(
                        _parse_residue_key_mapping(key, label="antibody predicted key")
                        for key in predicted_keys_raw
                    ),
                    source_atoms=tuple(
                        _parse_atom_key_mapping(
                            key, label="antibody correspondence source atom"
                        )
                        for key in source_atoms_raw
                    ),
                    predicted_atoms=tuple(
                        _parse_atom_key_mapping(
                            key, label="antibody correspondence predicted atom"
                        )
                        for key in predicted_atoms_raw
                    ),
                )
            )
        antibody = AntibodyAudit(
            _exact_nonnegative(item["insertion_coded_source_count"], "insertions"),
            _exact_nonnegative(item["insertion_coded_h3_count"], "H3 insertions"),
            tuple(correspondences),
            parsed_atom_keys["source_atom_keys"],
            parsed_atom_keys["predicted_atom_keys"],
            _exact_nonnegative(item["duplicate_keys_before"], "duplicates before"),
            _exact_nonnegative(item["duplicate_keys_after"], "duplicates after"),
            _exact_bool(item["mapping_ambiguity"], "mapping ambiguity"),
            _exact_nonnegative(item["common_h3_ca_count"], "common H3 CA"),
            _exact_bool(item["evaluable"], "antibody evaluable"),
        )
        derived = scan_antibody_example(
            parent_id=parent_id,
            example_id=example_id,
            residues=tuple(correspondences),
            normalization_identity=normalization_identity,
        )
        if (
            antibody != derived.antibody
            or representable != derived.representable
            or blockers != derived.blocker_codes
        ):
            raise AuditError(
                "antibody detail contradicts correspondence evidence; "
                "duplicate counts do not reconcile"
            )
    return ExampleAuditRecord(
        AuditUnit.EXAMPLE_ROW,
        family,
        parent_id,
        example_id,
        True,
        representable,
        blockers,
        normalization_identity,
        binder,
        ame,
        antibody,
        requested_cell_failure=requested_cell_failure,
    )

def _parent_mapping(parent: ParentAuditRecord) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_PARENT,
        "unit": parent.unit.value,
        "family": parent.family,
        "parent_id": parent.parent_id,
        "example_rows": parent.example_rows,
        "representable_rows": parent.representable_rows,
        "unrepresentable_rows": parent.unrepresentable_rows,
    }

def _binder_family_output(records: Sequence[ExampleAuditRecord]) -> dict[str, object]:
    missing: dict[str, dict[str, int]] = defaultdict(
        lambda: {atom: 0 for atom in _BACKBONE}
    )
    total_residues = 0
    for record in records:
        assert record.binder is not None
        total_residues += record.binder.total_residues
        for (
            source,
            counts,
        ) in record.binder.missing_backbone_by_coordinate_source.items():
            for atom in _BACKBONE:
                missing[source][atom] += int(counts[atom])
    return {
        "example_rows": len(records),
        "total_residues": total_residues,
        "missing_backbone_by_coordinate_source": {
            source: counts for source, counts in sorted(missing.items())
        },
        "repairable_rows": sum(record.representable for record in records),
        "unrepairable_rows": sum(not record.representable for record in records),
        "four_atom_preflight_pass_rows": sum(
            record.binder.four_atom_preflight_result for record in records
        ),
        "unchanged_coordinate_proof_rows": sum(
            record.binder.unchanged_coordinate_proof for record in records
        ),
    }

def _ame_family_output(records: Sequence[ExampleAuditRecord]) -> dict[str, object]:
    classes = {
        name: 0
        for name in (
            "non_polymer",
            "polymer_peptide",
            "water_only",
            "missing",
            "ambiguous",
            "mixed",
        )
    }
    for record in records:
        assert record.ame is not None
        classes[record.ame.ligand_class] += 1
    return {
        "example_rows": len(records),
        **classes,
        "canonical_identity_available_rows": sum(
            record.ame.canonical_identity_available for record in records
        ),
        "canonical_identity_unavailable_rows": sum(
            not record.ame.canonical_identity_available for record in records
        ),
        "parser_normalization_pass_rows": sum(
            record.ame.parser_normalization_result for record in records
        ),
        "parser_normalization_fail_rows": sum(
            not record.ame.parser_normalization_result for record in records
        ),
    }

def _antibody_family_output(
    records: Sequence[ExampleAuditRecord],
) -> dict[str, object]:
    return {
        "example_rows": len(records),
        "insertion_coded_source_count": sum(
            record.antibody.insertion_coded_source_count
            for record in records
        ),
        "insertion_coded_h3_count": sum(
            record.antibody.insertion_coded_h3_count
            for record in records
        ),
        "duplicate_keys_before": sum(
            record.antibody.duplicate_keys_before
            for record in records
        ),
        "duplicate_keys_after": sum(
            record.antibody.duplicate_keys_after
            for record in records
        ),
        "mapping_ambiguity_rows": sum(
            record.antibody.mapping_ambiguity
            for record in records
        ),
        "common_h3_ca_count": sum(
            record.antibody.common_h3_ca_count
            for record in records
        ),
        "evaluable_rows": sum(
            record.antibody.evaluable
            for record in records
        ),
        "unevaluable_rows": sum(
            not record.antibody.evaluable
            for record in records
        ),
    }

def reconcile_audit_records(
    records: Sequence[ExampleAuditRecord],
    expected: Mapping[str, AuditExpectedCounts],
    *,
    protocol_identities: tuple[PaperProtocolIdentity, ...],
    expected_membership: frozenset[tuple[str, str, str]] | None = None,
) -> AuditSummary:

    if not isinstance(expected, Mapping) or not expected:
        raise AuditError("expected counts must be a nonempty mapping")
    expected_families = set(expected)
    if expected_families - set(_FAMILIES):
        raise AuditError("expected counts contain an unknown family")
    if not isinstance(protocol_identities, tuple):
        raise AuditError("protocol identities must be a tuple")
    protocol_bindings: dict[str, str] = {}
    try:
        for identity in protocol_identities:
            token = paper_protocol_identity_token(identity)
            family, separator, _ = identity.name.partition(".")
            if not separator or family not in _FAMILIES:
                raise AuditError(f"unexpected protocol family {family}")
            if family not in expected_families:
                raise AuditError(f"unexpected protocol family {family}")
            if family in protocol_bindings:
                raise AuditError(
                    "protocol identities do not match: "
                    f"duplicate protocol binding for family {family}"
                )
            protocol_bindings[family] = token
    except PaperRegistryError as error:
        raise AuditError(str(error)) from error
    missing_bindings = sorted(expected_families - set(protocol_bindings))
    if missing_bindings:
        raise AuditError(
            "protocol identities do not match: missing protocol binding for family "
            f"{missing_bindings[0]}"
        )
    by_family: dict[str, list[ExampleAuditRecord]] = {
        family: [] for family in expected_families
    }
    seen: set[tuple[str, str]] = set()
    observed_membership: set[tuple[str, str, str]] = set()
    observed_protocol_tokens: set[str] = set()
    for record in records:
        if not isinstance(record, ExampleAuditRecord):
            raise AuditError("malformed audit row")

        parsed = parse_example_audit_record(example_audit_record_mapping(record))
        if parsed.family not in expected_families:
            raise AuditError(f"unexpected audit family {parsed.family}")
        if parsed.normalization_identity != protocol_bindings[parsed.family]:
            raise AuditError(
                "protocol identities do not match: "
                f"family protocol binding mismatch for {parsed.family}"
            )
        key = (parsed.family, parsed.example_id)
        if key in seen:
            raise AuditError(f"duplicate audit example {parsed.example_id!r}")
        seen.add(key)
        membership = (parsed.family, parsed.parent_id, parsed.example_id)
        observed_membership.add(membership)
        observed_protocol_tokens.add(parsed.normalization_identity)
        by_family[parsed.family].append(parsed)
    expected_protocol_tokens = set(protocol_bindings.values())
    if len(expected_protocol_tokens) != len(protocol_identities):
        raise AuditError("protocol identities contain a duplicate")
    if observed_protocol_tokens != expected_protocol_tokens:
        raise AuditError("row protocol identities do not match authentication")
    if expected_membership is not None and observed_membership != set(
        expected_membership
    ):
        raise AuditError("audit membership does not match frozen strict validation")
    example_counts = {family: len(rows) for family, rows in by_family.items()}
    for family, counts in expected.items():
        if not isinstance(counts, AuditExpectedCounts):
            raise AuditError("expected count values must be AuditExpectedCounts")
        if counts.example_rows < 0 or counts.parent_groups < 0:
            raise AuditError("expected counts must be nonnegative")
        if example_counts[family] != counts.example_rows:
            raise AuditError(
                f"example-row counts do not reconcile for {family}: "
                f"{example_counts[family]} != {counts.example_rows}"
            )
    grouped: dict[tuple[str, str], list[ExampleAuditRecord]] = defaultdict(list)
    for family_rows in by_family.values():
        for row in family_rows:
            grouped[(row.family, row.parent_id)].append(row)
    parent_counts = {
        family: sum(key[0] == family for key in grouped) for family in expected_families
    }
    for family, counts in expected.items():
        if parent_counts[family] != counts.parent_groups:
            raise AuditError(
                f"parent-group counts do not reconcile for {family}: "
                f"{parent_counts[family]} != {counts.parent_groups}"
            )
    parents = tuple(
        ParentAuditRecord(
            AuditUnit.PARENT_GROUP,
            family,
            parent_id,
            len(rows),
            sum(row.representable for row in rows),
            sum(not row.representable for row in rows),
        )
        for (family, parent_id), rows in sorted(grouped.items())
    )
    family_outputs: dict[str, Mapping[str, object]] = {}
    for family, family_rows in sorted(by_family.items()):
        if family == "binder":
            output = _binder_family_output(family_rows)
        elif family == "ame":
            output = _ame_family_output(family_rows)
        else:
            output = _antibody_family_output(family_rows)
        family_outputs[family] = MappingProxyType(output)
    unrepresentable = sum(
        not row.representable and row.requested_cell_failure is None
        for row in records
        if row.admitted
    )
    return AuditSummary(
        MappingProxyType(dict(sorted(example_counts.items()))),
        MappingProxyType(dict(sorted(parent_counts.items()))),
        parents,
        MappingProxyType(family_outputs),
        unrepresentable,
        bool(unrepresentable),
    )

def _summary_mapping(summary: AuditSummary) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_SUMMARY,
        "non_performance_audit": True,
        "membership_changed": False,
        "example_row_counts": dict(summary.example_counts),
        "parent_group_counts": dict(summary.parent_counts),
        "parent_groups": [_parent_mapping(parent) for parent in summary.parents],
        "family_outputs": {
            family: dict(output) for family, output in summary.family_outputs.items()
        },
        "unrepresentable_admitted_rows": summary.unrepresentable_admitted_rows,
        "replacement_smoke_blocked": summary.replacement_smoke_blocked,
    }

def _open_parent_directory(path: Path, *, label: str) -> tuple[int, str]:
    if not path.is_absolute() or not path.name or ".." in path.parts:
        raise AuditError(f"{label} path must be absolute without parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise AuditError(f"{label} parent symlink is forbidden") from error
                raise AuditError(f"cannot open {label} parent: {error}") from error
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, path.name

def _open_regular_file(
    path: Path, *, label: str, allow_final_symlink: bool = False
) -> int:
    parent_fd, name = _open_parent_directory(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            return os.open(name, flags, dir_fd=parent_fd)
        except OSError as error:
            if error.errno != errno.ELOOP or not allow_final_symlink:
                if error.errno == errno.ELOOP:
                    raise AuditError(f"{label} symlink is forbidden") from error
                raise AuditError(f"cannot open {label}: {error}") from error
            target = os.readlink(name, dir_fd=parent_fd)
            target_path = Path(target)
            if target_path.is_absolute() or len(target_path.parts) != 1:
                raise AuditError(f"{label} symlink target is unsafe")
            return os.open(target, flags, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)

def verify_artifact_identity(
    identity: ArtifactIdentity, *, label: str, allow_final_symlink: bool = False
) -> bytes:

    if not isinstance(identity, ArtifactIdentity):
        raise AuditError(f"{label} identity is malformed")
    _require_sha256(f"{label} sha256", identity.sha256)
    if type(identity.size_bytes) is not int or identity.size_bytes < 0:
        raise AuditError(f"{label} size is malformed")
    path = Path(identity.path)
    descriptor = _open_regular_file(
        path, label=label, allow_final_symlink=allow_final_symlink
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuditError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise AuditError(f"{label} identity drifted while reading")
    raw = b"".join(chunks)
    if len(raw) != identity.size_bytes:
        raise AuditError(f"{label} size mismatch")
    if hashlib.sha256(raw).hexdigest() != identity.sha256:
        raise AuditError(f"{label} hash mismatch")
    return raw

def _git(repo: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError(f"cannot authenticate git checkout {repo}") from error
    return result.stdout

def _authenticate_git(
    repo: Path, expected_commit: str | None = None
) -> GitCheckoutIdentity:
    commit = _git(repo, "rev-parse", "HEAD").strip()
    if len(commit) != 40 or set(commit) - _GIT_SHA:
        raise AuditError(f"invalid git HEAD for {repo}")
    if expected_commit is not None and commit != expected_commit:
        raise AuditError(f"git commit mismatch for {repo}")
    clean = not _git(repo, "status", "--porcelain").strip()
    if not clean:
        raise AuditError(f"git checkout is dirty: {repo}")
    return GitCheckoutIdentity(str(repo), commit, True)

def authenticate_fixed_upstreams() -> tuple[GitCheckoutIdentity, GitCheckoutIdentity]:

    emergent = _authenticate_git(
        signed_value_roots.EMERGENT_UPSTREAM_ROOT,
        signed_value_roots.EMERGENT_UPSTREAM_COMMIT,
    )
    signed = _authenticate_git(
        signed_value_roots.UPSTREAM_ROOT,
        signed_value_roots.SIGNED_VALUE_UPSTREAM_COMMIT,
    )
    return emergent, signed

def build_live_audit_authentication(
    *,
    inputs: PaperInputBundle,
    registry: PaperEvaluatorRegistry,
    protocol_identities: tuple[PaperProtocolIdentity, ...] = (),
) -> AuditAuthentication:

    if not isinstance(inputs, PaperInputBundle):
        raise AuditError("paper inputs must be a PaperInputBundle")
    if not isinstance(registry, PaperEvaluatorRegistry):
        raise AuditError("registry must be a PaperEvaluatorRegistry")
    try:
        verify_paper_evaluator_registry(registry)
    except PaperRegistryError as error:
        raise AuditError(str(error)) from error
    repository = _authenticate_git(_REPO_ROOT)
    if Path(registry.interpreter.path) != RUNTIME_PYTHON:
        raise AuditError("audit interpreter differs from the fixed runtime")
    interpreter = registry.interpreter.identity
    verify_artifact_identity(interpreter, label="interpreter", allow_final_symlink=True)
    verify_artifact_identity(
        registry.interpreter.environment, label="interpreter environment"
    )
    input_identities = tuple(sorted(inputs.sources, key=lambda item: item.path))
    if not input_identities:
        raise AuditError("audit input identity set is empty")
    for index, identity in enumerate(input_identities):
        verify_artifact_identity(identity, label=f"input[{index}]")
    checkpoints = {
        name: registry.checkpoints[name] for name in ("common", "autoencoder")
    }
    for name, identity in checkpoints.items():
        verify_artifact_identity(identity, label=f"checkpoint {name}")
    emergent, signed = authenticate_fixed_upstreams()
    semantic = compose_paper_registry_semantic_identity(
        registry.semantic_sha256, protocol_identities
    )
    return AuditAuthentication(
        repository=repository,
        interpreter=interpreter,
        interpreter_environment=registry.interpreter.environment,
        interpreter_version=registry.interpreter.version,
        view_semantic_hash=_require_sha256("view semantic hash", inputs.semantic_hash),
        view_partition_hash=_require_sha256(
            "view partition hash", inputs.partition_hash
        ),
        input_identities=input_identities,
        upstreams=(emergent, signed),
        checkpoints=MappingProxyType(checkpoints),
        evaluator_registry_base_sha256=registry.semantic_sha256,
        protocol_identities=protocol_identities,
        evaluator_registry_semantic_sha256=semantic,
    )

def _artifact_mapping(identity: ArtifactIdentity) -> dict[str, object]:
    return {
        "path": identity.path,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
    }

def _git_mapping(identity: GitCheckoutIdentity) -> dict[str, object]:
    return {"root": identity.root, "commit": identity.commit, "clean": identity.clean}

def is_fixed_interpreter(path: str | Path) -> bool:

    try:
        return Path(path).resolve() == RUNTIME_PYTHON.resolve()
    except OSError:
        return False

def revalidate_live_audit_authentication(
    authentication: AuditAuthentication, *, require_fixed_roots: bool
) -> None:
    if not isinstance(authentication, AuditAuthentication):
        raise AuditError("audit authentication is required")
    if not authentication.repository.clean:
        raise AuditError("repository authentication is not clean")
    if (
        len(authentication.repository.commit) != 40
        or set(authentication.repository.commit) - _GIT_SHA
    ):
        raise AuditError("repository commit is malformed")
    if len(authentication.upstreams) != 2 or any(
        not item.clean for item in authentication.upstreams
    ):
        raise AuditError("both clean upstream identities are required")
    if set(authentication.checkpoints) != {"common", "autoencoder"}:
        raise AuditError("checkpoint and autoencoder identities are required")
    if require_fixed_roots:
        if Path(authentication.repository.root) != _REPO_ROOT:
            raise AuditError("authentication does not name the fixed DIVE repository")
        if not is_fixed_interpreter(authentication.interpreter.path):
            raise AuditError("authentication does not name the fixed interpreter")
        fixed_upstreams = (
            (
                signed_value_roots.EMERGENT_UPSTREAM_ROOT,
                signed_value_roots.EMERGENT_UPSTREAM_COMMIT,
            ),
            (
                signed_value_roots.UPSTREAM_ROOT,
                signed_value_roots.SIGNED_VALUE_UPSTREAM_COMMIT,
            ),
        )
        if (
            tuple(
                (Path(identity.root), identity.commit)
                for identity in authentication.upstreams
            )
            != fixed_upstreams
        ):
            raise AuditError("authentication does not name both fixed upstream pins")
    if (
        Path(sys.executable).resolve()
        != Path(authentication.interpreter.path).resolve()
    ):
        raise AuditError("authenticated interpreter is not the executing interpreter")
    if authentication.interpreter_version != platform.python_version():
        raise AuditError(
            "authenticated interpreter version differs from the live runtime"
        )
    _require_sha256("view semantic hash", authentication.view_semantic_hash)
    _require_sha256("view partition hash", authentication.view_partition_hash)
    _require_sha256("registry base hash", authentication.evaluator_registry_base_sha256)
    observed = compose_paper_registry_semantic_identity(
        authentication.evaluator_registry_base_sha256,
        authentication.protocol_identities,
    )
    if observed != authentication.evaluator_registry_semantic_sha256:
        raise AuditError("registry semantic identity does not authenticate protocols")
    _authenticate_git(
        Path(authentication.repository.root), authentication.repository.commit
    )
    for upstream in authentication.upstreams:
        _authenticate_git(Path(upstream.root), upstream.commit)
    verify_artifact_identity(
        authentication.interpreter,
        label="interpreter",
        allow_final_symlink=True,
    )
    for label, identity in (
        ("interpreter environment", authentication.interpreter_environment),
        *(
            (f"input[{index}]", identity)
            for index, identity in enumerate(authentication.input_identities)
        ),
        *(
            (f"checkpoint {name}", identity)
            for name, identity in authentication.checkpoints.items()
        ),
    ):
        verify_artifact_identity(identity, label=label)

def _authentication_mapping(authentication: AuditAuthentication) -> dict[str, object]:
    return {
        "repository": _git_mapping(authentication.repository),
        "interpreter": {
            **_artifact_mapping(authentication.interpreter),
            "version": authentication.interpreter_version,
            "environment": _artifact_mapping(authentication.interpreter_environment),
        },
        "view_semantic_hash": authentication.view_semantic_hash,
        "view_partition_hash": authentication.view_partition_hash,
        "inputs": [_artifact_mapping(item) for item in authentication.input_identities],
        "upstreams": [_git_mapping(item) for item in authentication.upstreams],
        "checkpoints": {
            name: _artifact_mapping(item)
            for name, item in sorted(authentication.checkpoints.items())
        },
        "evaluator_registry_base_sha256": authentication.evaluator_registry_base_sha256,
        "protocol_identities": [
            {"name": item.name, "version": item.version, "sha256": item.sha256}
            for item in sorted(
                authentication.protocol_identities, key=lambda item: item.name
            )
        ],
        "evaluator_registry_semantic_sha256": (
            authentication.evaluator_registry_semantic_sha256
        ),
    }

def validate_audit_destination(destination: Path) -> Path:

    candidate = Path(destination)
    absolute = candidate if candidate.is_absolute() else candidate.absolute()
    for denied in DENIED_PREFIXES:
        denied_absolute = denied.absolute()
        if absolute == denied_absolute or denied_absolute in absolute.parents:
            raise AuditError(f"denied root for terminal evidence: {absolute}")
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as error:
            raise AuditError(f"cannot validate audit destination: {error}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise AuditError(f"symlink traversal is forbidden: {current}")
    return absolute

def _open_directory_at(parent_fd: int, name: str, *, label: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise AuditError(f"{label} symlink traversal is forbidden") from error
        raise AuditError(f"cannot open {label}: {error}") from error

def _open_absolute_directory(path: Path, *, label: str) -> int:
    parent_fd, name = _open_parent_directory(path, label=label)
    try:
        return _open_directory_at(parent_fd, name, label=label)
    finally:
        os.close(parent_fd)

def _claim_run_directory(run_id: str) -> tuple[Path, Path, int, int, int, str]:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise AuditError(f"invalid audit run id {run_id!r}")
    root = _TEST_EVIDENCE_ROOT or EMERGENT_EVIDENCE_ROOT
    if _TEST_EVIDENCE_ROOT is None:
        if root != EMERGENT_EVIDENCE_ROOT:
            raise AuditError(
                "audit evidence root differs from the configured group root"
            )
    validate_audit_destination(root / "paper_quality_baseline" / run_id)
    root_fd = _open_absolute_directory(root, label="configured evidence root")
    parent_fd = -1
    transaction_parent_fd = -1
    transaction_fd = -1
    try:
        created_parent = False
        try:
            os.mkdir("paper_quality_baseline", 0o2775, dir_fd=root_fd)
            created_parent = True
        except FileExistsError:
            pass
        parent_fd = _open_directory_at(
            root_fd, "paper_quality_baseline", label="paper audit parent"
        )
        if created_parent:
            os.fchmod(parent_fd, 0o2775)
        if _TEST_EVIDENCE_ROOT is None:
            root_stat = os.fstat(root_fd)
            parent_stat = os.fstat(parent_fd)
            if parent_stat.st_gid != root_stat.st_gid or not (
                parent_stat.st_mode & stat.S_ISGID
            ):
                raise AuditError("paper audit parent is not group-owned/setgid")
            if not (parent_stat.st_mode & stat.S_IWGRP):
                raise AuditError("paper audit parent is not group-writable")
        try:
            os.stat(run_id, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(errno.EEXIST, "audit run already exists", run_id)
        created_transactions = False
        try:
            os.mkdir(".audit-transactions", 0o700, dir_fd=parent_fd)
            created_transactions = True
        except FileExistsError:
            pass
        transaction_parent_fd = _open_directory_at(
            parent_fd, ".audit-transactions", label="paper audit transaction root"
        )
        if created_transactions:
            os.fchmod(transaction_parent_fd, 0o700)
        transaction_parent_stat = os.fstat(transaction_parent_fd)
        if (
            transaction_parent_stat.st_uid != os.geteuid()
            or stat.S_IMODE(transaction_parent_stat.st_mode) != 0o700
        ):
            raise AuditError("paper audit transaction root must be private mode 0700")
        transaction_name = f"{run_id}.{os.getpid()}.{secrets.token_hex(12)}"
        os.mkdir(transaction_name, 0o700, dir_fd=transaction_parent_fd)
        transaction_fd = _open_directory_at(
            transaction_parent_fd, transaction_name, label="paper audit transaction"
        )
        os.fchmod(transaction_fd, 0o700)
        os.fsync(parent_fd)
        os.fsync(transaction_parent_fd)
        final_path = root / "paper_quality_baseline" / run_id
        transaction_path = (
            root / "paper_quality_baseline" / ".audit-transactions" / transaction_name
        )
        return (
            final_path,
            transaction_path,
            transaction_fd,
            parent_fd,
            transaction_parent_fd,
            transaction_name,
        )
    finally:
        if transaction_fd < 0 and transaction_parent_fd >= 0:
            os.close(transaction_parent_fd)
        if transaction_fd < 0 and parent_fd >= 0:
            os.close(parent_fd)
        os.close(root_fd)

def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise AuditError("audit write made no forward progress")
        view = view[written:]

def _create_private_file_at(run_fd: int, name: str, content: bytes) -> ArtifactIdentity:
    if "/" in name or name in {"", ".", ".."}:
        raise AuditError("audit artifact name is invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o664, dir_fd=run_fd)
    try:
        os.fchmod(descriptor, 0o664)
        _write_all(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(run_fd)
    return ArtifactIdentity(name, hashlib.sha256(content).hexdigest(), len(content))

def _rename_directory_noreplace(
    source_parent_fd: int,
    source_name: str,
    destination_parent_fd: int,
    destination_name: str,
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise AuditError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_fd,
        os.fsencode(source_name),
        destination_parent_fd,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number, "audit final run already exists", destination_name
        )
    raise AuditError(
        f"atomic audit directory publication failed: {os.strerror(error_number)}"
    )

def _open_raw_log(run_fd: int) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("raw.log", flags, 0o664, dir_fd=run_fd)
    os.fchmod(descriptor, 0o664)
    return descriptor

def _identity_at(run_fd: int, name: str) -> ArtifactIdentity:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=run_fd)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AuditError(f"published audit artifact {name} is not regular")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    return ArtifactIdentity(name, digest.hexdigest(), size)

class AuditEvidencePublication:

    def __init__(
        self,
        *,
        run_id: str,
        argv: tuple[str, ...],
        authentication: AuditAuthentication,
        environment: Mapping[str, str],
    ) -> None:
        revalidate_live_audit_authentication(
            authentication, require_fixed_roots=_TEST_EVIDENCE_ROOT is None
        )
        if (
            not isinstance(argv, tuple)
            or not argv
            or any(type(item) is not str for item in argv)
        ):
            raise AuditError("audit argv must be a nonempty string tuple")
        if not isinstance(environment, Mapping) or any(
            type(key) is not str
            or type(value) is not str
            or key not in _ALLOWED_ENVIRONMENT_KEYS
            for key, value in environment.items()
        ):
            raise AuditError("audit environment contains an unapproved key/value")
        self.run_id = run_id
        self._authentication = authentication
        (
            self.path,
            self.transaction_path,
            self._run_fd,
            self._parent_fd,
            self._transaction_parent_fd,
            self._transaction_name,
        ) = _claim_run_directory(run_id)
        self._raw_fd = -1
        self._completed = False
        self._closed = False
        self._started_monotonic = time.monotonic()
        self._started_at = datetime.now(UTC).isoformat()
        attempt = {
            "schema_version": _SCHEMA_ATTEMPT,
            "status": "BUILDING",
            "run_id": run_id,
            "non_performance_audit": True,
            "membership_mutation_allowed": False,
            "argv": list(argv),
            "started_at_utc": self._started_at,
            "authentication": _authentication_mapping(authentication),
        }
        environment_payload = {
            "schema_version": _SCHEMA_ENVIRONMENT,
            "run_id": run_id,
            "cwd": str(Path.cwd()),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "environment": dict(sorted(environment.items())),
        }
        try:
            _create_private_file_at(
                self._run_fd, "attempt.json", canonical_json_bytes(attempt)
            )
            _create_private_file_at(
                self._run_fd,
                "environment.json",
                canonical_json_bytes(environment_payload),
            )
            self._raw_fd = _open_raw_log(self._run_fd)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> AuditEvidencePublication:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def log(self, message: str) -> None:
        if self._raw_fd < 0 or self._closed:
            raise AuditError("audit raw log is closed")
        if type(message) is not str:
            raise AuditError("audit log message must be text")
        _write_all(self._raw_fd, message.encode("utf-8"))
        os.fsync(self._raw_fd)

    def complete(
        self,
        records: Sequence[ExampleAuditRecord],
        expected: Mapping[str, AuditExpectedCounts],
        *,
        expected_membership: frozenset[tuple[str, str, str]] | None = None,
    ) -> ArtifactIdentity:
        if self._completed or self._closed:
            raise AuditError("audit publication is already terminal")
        revalidate_live_audit_authentication(
            self._authentication, require_fixed_roots=_TEST_EVIDENCE_ROOT is None
        )
        summary = reconcile_audit_records(
            records,
            expected,
            protocol_identities=self._authentication.protocol_identities,
            expected_membership=expected_membership,
        )
        if self._raw_fd >= 0:
            os.fsync(self._raw_fd)
            os.close(self._raw_fd)
            self._raw_fd = -1
        ordered = sorted(
            records, key=lambda row: (row.family, row.parent_id, row.example_id)
        )
        rows_content = b"".join(
            canonical_json_bytes(example_audit_record_mapping(row)) for row in ordered
        )
        _create_private_file_at(self._run_fd, "rows.jsonl", rows_content)
        _create_private_file_at(
            self._run_fd,
            "summary.json",
            canonical_json_bytes(_summary_mapping(summary)),
        )
        manifest_files = tuple(
            _identity_at(self._run_fd, name)
            for name in (
                "attempt.json",
                "environment.json",
                "raw.log",
                "rows.jsonl",
                "summary.json",
            )
        )
        manifest_payload = {
            "schema_version": _SCHEMA_MANIFEST,
            "run_id": self.run_id,
            "hash_algorithm": "sha256",
            "files": [_artifact_mapping(item) for item in manifest_files],
        }
        manifest = _create_private_file_at(
            self._run_fd,
            "manifest.json",
            canonical_json_bytes(manifest_payload),
        )
        ended_at = datetime.now(UTC).isoformat()
        duration = time.monotonic() - self._started_monotonic
        completion_payload = {
            "schema_version": _SCHEMA_COMPLETION,
            "status": "COMPLETE",
            "run_id": self.run_id,
            "non_performance_audit": True,
            "membership_changed": False,
            "started_at_utc": self._started_at,
            "ended_at_utc": ended_at,
            "duration_seconds": duration,
            "example_row_counts": dict(summary.example_counts),
            "parent_group_counts": dict(summary.parent_counts),
            "unrepresentable_admitted_rows": summary.unrepresentable_admitted_rows,
            "replacement_smoke_blocked": summary.replacement_smoke_blocked,
            "manifest": _artifact_mapping(manifest),
        }
        completion = _create_private_file_at(
            self._run_fd,
            "completion.json",
            canonical_json_bytes(completion_payload),
        )
        revalidate_live_audit_authentication(
            self._authentication, require_fixed_roots=_TEST_EVIDENCE_ROOT is None
        )
        os.fchmod(self._run_fd, 0o2775)
        os.fsync(self._run_fd)
        _rename_directory_noreplace(
            self._transaction_parent_fd,
            self._transaction_name,
            self._parent_fd,
            self.run_id,
        )
        os.fsync(self._parent_fd)
        os.fsync(self._transaction_parent_fd)
        self._completed = True
        return ArtifactIdentity(
            str(self.path / "completion.json"),
            completion.sha256,
            completion.size_bytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._raw_fd >= 0:
            os.fsync(self._raw_fd)
            os.close(self._raw_fd)
            self._raw_fd = -1
        os.fsync(self._run_fd)
        os.close(self._run_fd)
        os.close(self._transaction_parent_fd)
        os.close(self._parent_fd)
        self._closed = True

def claim_audit_evidence(
    run_id: str,
    *,
    argv: tuple[str, ...],
    authentication: AuditAuthentication,
    environment: Mapping[str, str],
) -> AuditEvidencePublication:

    return AuditEvidencePublication(
        run_id=run_id,
        argv=argv,
        authentication=authentication,
        environment=environment,
    )

def load_example_records_jsonl(
    path: Path, *, expected_identity: ArtifactIdentity | None = None
) -> tuple[ExampleAuditRecord, ...]:

    descriptor = _open_regular_file(Path(path), label="audit row input")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuditError("audit row input must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise AuditError("audit row input identity drifted while reading")
    raw = b"".join(chunks)
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise AuditError("audit row input must use LF framing with a final LF")
    if expected_identity is not None:
        if not isinstance(expected_identity, ArtifactIdentity):
            raise AuditError("audit row input identity is malformed")
        if str(path) != expected_identity.path:
            raise AuditError("audit row input path mismatch")
        if len(raw) != expected_identity.size_bytes:
            raise AuditError("audit row input size mismatch")
        if hashlib.sha256(raw).hexdigest() != expected_identity.sha256:
            raise AuditError("audit row input hash mismatch")
    records: list[ExampleAuditRecord] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise AuditError(f"empty JSONL row at line {line_number}")
        try:
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AuditError(f"malformed JSONL row at line {line_number}") from error
        if canonical_json_bytes(payload).rstrip(b"\n") != line:
            raise AuditError(f"noncanonical JSONL row at line {line_number}")
        records.append(parse_example_audit_record(payload))
    if not records:
        raise AuditError("audit row input is empty")
    return tuple(records)
