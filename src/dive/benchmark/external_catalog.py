
from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

import yaml
from yaml.events import AliasEvent

from dive.training.preflight import canonical_json_bytes

class ExternalCatalogError(RuntimeError):
    pass

class ExternalCandidateId(StrEnum):
    RFDIFFUSION_BINDER_V1 = "rfdiffusion_binder_v1"
    FRAMEFLOW_MOTIF_V1 = "frameflow_motif_v1"
    CHROMA_CONDITIONED_V1 = "chroma_conditioned_v1"
    RFDIFFUSIONAA_LIGAND_V1 = "rfdiffusionaa_ligand_v1"
    DIFFAB_CDR_V1 = "diffab_cdr_v1"
    RFANTIBODY_H3_V1 = "rfantibody_h3_v1"
    ABX_CDR_V1 = "abx_cdr_v1"

class ExternalFamily(StrEnum):
    BINDER = "binder"
    AME = "ame"
    ANTIBODY = "antibody"

class AccessMode(StrEnum):
    PUBLIC = "public"
    CREDENTIAL_GATED = "credential_gated"

_SCHEMA_V1 = "dive-external-baseline-candidates-v1"
_SCHEMA_V2 = "dive-external-baseline-candidates-v2"
_ROOT_KEYS = frozenset({"schema_version", "candidates"})
_CANDIDATE_KEYS = frozenset(
    {
        "candidate_id",
        "family",
        "url",
        "observed_ref",
        "access_mode",
        "license_file_candidates",
        "required_capabilities",
    }
)
_V1_EXPECTED_DECLARATIONS = MappingProxyType(
    {
        ExternalCandidateId.RFDIFFUSION_BINDER_V1: (
            ExternalFamily.BINDER,
            "https://github.com/RosettaCommons/RFdiffusion.git",
            AccessMode.PUBLIC,
            ("LICENSE",),
            ("target_structure", "target_hotspots", "backbone_output"),
        ),
        ExternalCandidateId.FRAMEFLOW_MOTIF_V1: (
            ExternalFamily.BINDER,
            "https://github.com/microsoft/protein-frame-flow.git",
            AccessMode.PUBLIC,
            ("LICENSE",),
            ("target_structure", "target_hotspots", "backbone_output"),
        ),
        ExternalCandidateId.CHROMA_CONDITIONED_V1: (
            ExternalFamily.BINDER,
            "https://github.com/generatebio/chroma.git",
            AccessMode.CREDENTIAL_GATED,
            ("LICENSE.txt",),
            (
                "target_structure",
                "target_hotspots",
                "backbone_output",
                "joint_sequence_structure_output",
            ),
        ),
        ExternalCandidateId.RFDIFFUSIONAA_LIGAND_V1: (
            ExternalFamily.AME,
            "https://github.com/baker-laboratory/rf_diffusion_all_atom.git",
            AccessMode.PUBLIC,
            ("LICENSE",),
            ("ligand_identity", "motif_atoms", "backbone_output"),
        ),
        ExternalCandidateId.DIFFAB_CDR_V1: (
            ExternalFamily.ANTIBODY,
            "https://github.com/luost26/diffab.git",
            AccessMode.PUBLIC,
            ("LICENSE",),
            (
                "antibody_antigen_complex",
                "insertion_aware_cdr_h3",
                "joint_sequence_structure_output",
            ),
        ),
        ExternalCandidateId.ABX_CDR_V1: (
            ExternalFamily.ANTIBODY,
            "https://github.com/CarbonMatrixLab/AbX.git",
            AccessMode.PUBLIC,
            ("LICENCE",),
            (
                "antibody_antigen_complex",
                "insertion_aware_cdr_h3",
                "joint_sequence_structure_output",
            ),
        ),
    }
)
_V2_EXPECTED_DECLARATIONS = MappingProxyType(
    {
        ExternalCandidateId.RFDIFFUSION_BINDER_V1: _V1_EXPECTED_DECLARATIONS[
            ExternalCandidateId.RFDIFFUSION_BINDER_V1
        ],
        ExternalCandidateId.FRAMEFLOW_MOTIF_V1: _V1_EXPECTED_DECLARATIONS[
            ExternalCandidateId.FRAMEFLOW_MOTIF_V1
        ],
        ExternalCandidateId.CHROMA_CONDITIONED_V1: _V1_EXPECTED_DECLARATIONS[
            ExternalCandidateId.CHROMA_CONDITIONED_V1
        ],
        ExternalCandidateId.RFDIFFUSIONAA_LIGAND_V1: _V1_EXPECTED_DECLARATIONS[
            ExternalCandidateId.RFDIFFUSIONAA_LIGAND_V1
        ],
        ExternalCandidateId.RFANTIBODY_H3_V1: (
            ExternalFamily.ANTIBODY,
            "https://github.com/RosettaCommons/RFantibody.git",
            AccessMode.PUBLIC,
            ("LICENSE",),
            (
                "antibody_antigen_complex",
                "fixed_framework",
                "h3_only_design",
                "fixed_h3_length",
                "joint_sequence_structure_output",
                "insertion_aware_cdr_h3",
            ),
        ),
        ExternalCandidateId.ABX_CDR_V1: _V1_EXPECTED_DECLARATIONS[
            ExternalCandidateId.ABX_CDR_V1
        ],
    }
)
_EXPECTED_DECLARATIONS_BY_SCHEMA = MappingProxyType(
    {
        _SCHEMA_V1: _V1_EXPECTED_DECLARATIONS,
        _SCHEMA_V2: _V2_EXPECTED_DECLARATIONS,
    }
)
_CATALOG_RELATIVE_PATHS_BY_SCHEMA = MappingProxyType(
    {
        _SCHEMA_V1: Path("configs/emergent/external_baseline_candidates.yaml"),
        _SCHEMA_V2: Path("configs/emergent/external_baseline_candidates_v2.yaml"),
    }
)

class _ClosedCatalogLoader(yaml.SafeLoader):

    def construct_mapping(self, node, deep: bool = False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ExternalCatalogError("catalog YAML key is unhashable") from error
            if duplicate:
                raise ExternalCatalogError("catalog YAML has duplicate keys")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping

@dataclass(frozen=True, slots=True)
class CandidateDeclaration:
    candidate_id: ExternalCandidateId
    family: ExternalFamily
    url: str
    observed_ref: str
    access_mode: AccessMode
    license_file_candidates: tuple[str, ...]
    required_capabilities: tuple[str, ...]

    def as_mapping(self) -> dict[str, object]:

        return {
            "candidate_id": self.candidate_id.value,
            "family": self.family.value,
            "url": self.url,
            "access_mode": self.access_mode.value,
            "license_file_candidates": list(self.license_file_candidates),
            "required_capabilities": list(self.required_capabilities),
        }

@dataclass(frozen=True, slots=True)
class ExternalCatalog:
    schema_version: str
    candidates: tuple[CandidateDeclaration, ...]

    @property
    def semantic_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_mapping())).hexdigest()

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidates": [item.as_mapping() for item in self.candidates],
        }

def catalog_path_for_schema(schema_version: str, repository_root: Path) -> Path:

    if type(schema_version) is not str:
        raise ExternalCatalogError("catalog schema_version is unsupported")
    if not isinstance(repository_root, Path):
        raise ExternalCatalogError("catalog repository root is invalid")
    try:
        relative_path = _CATALOG_RELATIVE_PATHS_BY_SCHEMA[schema_version]
    except KeyError as error:
        raise ExternalCatalogError("catalog schema_version is unsupported") from error
    return repository_root / relative_path

def load_external_catalog(path: Path) -> ExternalCatalog:

    raw = _read_retained_regular_file(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExternalCatalogError("catalog must be UTF-8") from error
    try:
        if any(isinstance(event, AliasEvent) for event in yaml.parse(text)):
            raise ExternalCatalogError("catalog YAML aliases are forbidden")
        document = yaml.load(text, Loader=_ClosedCatalogLoader)
    except yaml.YAMLError as error:
        raise ExternalCatalogError("catalog YAML is malformed") from error
    return _parse_closed_catalog(document)

def _read_retained_regular_file(path: Path) -> bytes:
    if not isinstance(path, Path):
        raise ExternalCatalogError("catalog path is invalid")
    try:
        before_path = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ExternalCatalogError(f"cannot stat catalog: {error}") from error
    if stat.S_ISLNK(before_path.st_mode):
        raise ExternalCatalogError("catalog symlink is forbidden")
    if not stat.S_ISREG(before_path.st_mode):
        raise ExternalCatalogError("catalog must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ExternalCatalogError("catalog symlink is forbidden") from error
        raise ExternalCatalogError(f"cannot open catalog: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExternalCatalogError("catalog must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ExternalCatalogError("catalog identity drifted while reading") from error
    if _file_identity(before_path) != _file_identity(before):
        raise ExternalCatalogError("catalog identity drifted before reading")
    if _file_identity(before) != _file_identity(after) or _file_identity(
        before_path
    ) != _file_identity(after_path):
        raise ExternalCatalogError("catalog identity drifted while reading")
    return b"".join(chunks)

def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )

def _parse_closed_catalog(document: object) -> ExternalCatalog:
    if not isinstance(document, Mapping):
        raise ExternalCatalogError("catalog must be a mapping")
    _reject_unknown_keys(document, _ROOT_KEYS, label="catalog")
    schema_version = _required_string(document, "schema_version", label="catalog")
    declarations = _EXPECTED_DECLARATIONS_BY_SCHEMA.get(schema_version)
    if declarations is None:
        raise ExternalCatalogError("catalog schema_version is unsupported")
    raw_candidates = document.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ExternalCatalogError("catalog candidates must be an ordered list")
    candidates = tuple(_parse_candidate(item, declarations) for item in raw_candidates)
    expected_order = tuple(declarations)
    if tuple(candidate.candidate_id for candidate in candidates) != expected_order:
        raise ExternalCatalogError("catalog candidate order or membership is invalid")
    return ExternalCatalog(schema_version=schema_version, candidates=candidates)

def _parse_candidate(
    raw: object,
    declarations: Mapping[
        ExternalCandidateId,
        tuple[ExternalFamily, str, AccessMode, tuple[str, ...], tuple[str, ...]],
    ],
) -> CandidateDeclaration:
    if not isinstance(raw, Mapping):
        raise ExternalCatalogError("candidate must be a mapping")
    _reject_unknown_keys(raw, _CANDIDATE_KEYS, label="candidate")
    candidate_id = _parse_enum(
        ExternalCandidateId,
        _required_string(raw, "candidate_id", label="candidate"),
        label="candidate id",
    )
    family = _parse_enum(
        ExternalFamily,
        _required_string(raw, "family", label="candidate"),
        label="candidate family",
    )
    url = _required_string(raw, "url", label="candidate")
    _validate_repository_url(url)
    observed_ref = _required_string(raw, "observed_ref", label="candidate")
    if observed_ref != "HEAD":
        raise ExternalCatalogError("candidate observed_ref must be literal HEAD")
    access_mode = _parse_enum(
        AccessMode,
        _required_string(raw, "access_mode", label="candidate"),
        label="candidate access_mode",
    )
    license_file_candidates = _required_string_tuple(
        raw, "license_file_candidates", label="candidate license paths"
    )
    required_capabilities = _required_string_tuple(
        raw, "required_capabilities", label="candidate capabilities"
    )
    try:
        expected = declarations[candidate_id]
    except KeyError as error:
        raise ExternalCatalogError(
            "candidate id is not reviewed for this schema"
        ) from error
    if family is not expected[0]:
        raise ExternalCatalogError("candidate family does not match its reviewed role")
    if url != expected[1]:
        raise ExternalCatalogError("candidate URL is not its official reviewed URL")
    if access_mode is not expected[2]:
        raise ExternalCatalogError("candidate access_mode does not match review")
    if license_file_candidates != expected[3]:
        raise ExternalCatalogError("candidate license paths do not match review")
    if required_capabilities != expected[4]:
        raise ExternalCatalogError("candidate capabilities do not match review")
    return CandidateDeclaration(
        candidate_id=candidate_id,
        family=family,
        url=url,
        observed_ref=observed_ref,
        access_mode=access_mode,
        license_file_candidates=license_file_candidates,
        required_capabilities=required_capabilities,
    )

def _reject_unknown_keys(
    document: Mapping[object, object], allowed: frozenset[str], *, label: str
) -> None:
    if any(type(key) is not str for key in document):
        raise ExternalCatalogError(f"{label} keys must be strings")
    unknown = set(document) - allowed
    if unknown:
        raise ExternalCatalogError(f"{label} has unknown keys: {sorted(unknown)}")
    missing = allowed - set(document)
    if missing:
        raise ExternalCatalogError(f"{label} is missing keys: {sorted(missing)}")

def _required_string(document: Mapping[object, object], key: str, *, label: str) -> str:
    value = document.get(key)
    if type(value) is not str or not value:
        raise ExternalCatalogError(f"{label} {key} must be a non-empty string")
    return value

def _required_string_tuple(
    document: Mapping[object, object], key: str, *, label: str
) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or not value:
        raise ExternalCatalogError(f"{label} must be a non-empty list")
    if any(type(item) is not str or not item for item in value):
        raise ExternalCatalogError(f"{label} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ExternalCatalogError(f"{label} must not contain duplicates")
    return tuple(value)

def _parse_enum(enum_type: type[StrEnum], value: str, *, label: str) -> StrEnum:
    try:
        return enum_type(value)
    except ValueError as error:
        raise ExternalCatalogError(f"{label} is unknown") from error

def _validate_repository_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ExternalCatalogError("candidate URL is not an allowed GitHub HTTPS URL")
