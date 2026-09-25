
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.evaluators.contracts import (
    EvaluatorContractError,
    InterpreterContract,
    ModuleTreeIdentity,
    ReducerContract,
    _assert_identity,
    _authenticate_interpreter,
)
from dive.benchmark.paper_baseline import FIXTURE_EVALUATOR_REGISTRY_SHA256
from dive.signed_value.roots import (
    EMERGENT_UPSTREAM_COMMIT,
    EMERGENT_UPSTREAM_ROOT,
)
from dive.training.preflight import canonical_json_bytes

class PaperRegistryError(RuntimeError):
    pass

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAPER_EVALUATOR_REGISTRY_PATH = (
    _REPO_ROOT / "configs" / "emergent" / "paper_baseline_evaluators.yaml"
)
_SCHEMA = "dive-paper-evaluator-registry-v1"
_FAMILIES = ("binder", "ame", "antibody")
_SHA256 = frozenset("0123456789abcdef")
_FIXTURE_WORKER_SHA256 = (
    "e19d578cab977060f8469bcc43b96fc218f7c53727e8b45893de90b7ae11ce95"
)
_FORBIDDEN_GENERATION_NAMES = frozenset({"complexa_ame.ckpt", "complexa_ligand.ckpt"})
_FORBIDDEN_GENERATION_HASHES = frozenset(
    {
        "d11319693d024d0427abc356a86350a694a1cb7dceb8db642c8041e5a20a9f7b",
        "8175213eac5ec6433fed1756d055ce3f867129bfdb033040e1a0560ca558bfe5",
    }
)
_ALLOWED_OPTIONAL_GAPS = frozenset(
    {
        "freesasa_dependency_unavailable",
        "novelty_database_provenance_unavailable",
    }
)
_FREESASA_METRICS = (
    "binder_interface_dSASA",
    "binder_interface_fraction",
    "binder_interface_hydrophobicity",
    "binder_interface_nres",
    "binder_interface_sc",
    "binder_surface_hydrophobicity",
)
_BINDER_BACKBONE_NORMALIZATION_ROLE = "binder_atom37_backbone_normalization_v1"
_BINDER_BACKBONE_REFERENCE_ROLE = "binder_frameflow_oxygen_reference_v1"
_BINDER_PDB_AUTHENTICATION_ROLE = "binder_pdb_authentication_v1"
_BINDER_PRIVATE_SNAPSHOT_ROLE = "binder_private_snapshot_protocol_v3"
_METHODS = {
    "binder": "soluble_mpnn_2x_af2_multimer",
    "ame": "ligandmpnn_rf3_motif_ligand_joint",
    "antibody": "frozen_native_geometry",
}
_PRIMARIES = {
    "binder": ("target_conditioned_success", "higher"),
    "ame": ("ame_motif_ligand_success", "higher"),
    "antibody": ("cdr_h3_ca_rmsd", "lower"),
}
_SCHEMAS = {
    "binder": (
        "binder-evaluator-retained-private-snapshot-input-v3",
        "binder-evaluator-output-v1",
    ),
    "ame": ("ame-evaluator-input-v1", "ame-evaluator-output-v1"),
    "antibody": (
        "antibody-native-geometry-input-v1",
        "antibody-native-geometry-output-v1",
    ),
}
_SELECTION_RULES = {
    "binder": "or_over_two_sequence_redesigns",
    "ame": "same_index_joint_or",
    "antibody": "in_process_native_geometry",
}
_COUNTS = {
    "binder": (2, 2),
    "ame": (2, 2),
    "antibody": (0, 1),
}
_AME_LIGAND_TRANSPORT_ROLE = "ame_canonical_ligand_transport_v1"
_ANTIBODY_CORRESPONDENCE_ROLE = "antibody_insertion_aware_correspondence_v1"
_AME_REQUIRED_ROLES = (
    "ligandmpnn_source",
    "ligandmpnn_checkpoint",
    "rf3_source",
    "rf3_checkpoint",
    "rf3_executable",
    "rf3_compatibility_wrapper",
    "motif_evaluation",
    "ligand_clash_source",
    _AME_LIGAND_TRANSPORT_ROLE,
)
_FAMILY_KEYS = {
    "method",
    "primary",
    "direction",
    "input_schema",
    "output_schema",
    "redesign_count",
    "evaluator_count",
    "selection_rule",
    "required_raw_secondaries",
    "thresholds",
    "reducer",
    "assets",
    "availability",
}

@dataclass(frozen=True, slots=True)
class PaperThreshold:
    name: str
    op: str
    value: float | bool | int

@dataclass(frozen=True, slots=True)
class PaperAvailabilityGap:
    reason_code: str
    required: bool
    metrics: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class PaperEvaluatorAsset:
    role: str
    kind: str
    required: bool
    identity: ArtifactIdentity | None
    tree: ModuleTreeIdentity | None
    runtime: Mapping[str, str] | None = None

@dataclass(frozen=True, slots=True)
class PaperFamilyEvaluator:
    method: str
    primary: str
    direction: str
    input_schema: str
    output_schema: str
    redesign_count: int
    evaluator_count: int
    selection_rule: str
    required_raw_secondaries: tuple[str, ...]
    thresholds: tuple[PaperThreshold, ...]
    reducer: ReducerContract
    reducer_source: ArtifactIdentity
    assets: tuple[PaperEvaluatorAsset, ...]
    availability: tuple[PaperAvailabilityGap, ...]

@dataclass(frozen=True, slots=True)
class PaperEvaluatorRegistry:
    families: Mapping[str, PaperFamilyEvaluator]
    semantic_sha256: str
    interpreter: InterpreterContract
    worker: ArtifactIdentity
    upstream_root: str
    upstream_commit: str
    checkpoints: Mapping[str, ArtifactIdentity]

@dataclass(frozen=True, slots=True)
class PaperProtocolIdentity:

    name: str
    version: str
    sha256: str

def paper_protocol_identity_token(identity: PaperProtocolIdentity) -> str:

    if not isinstance(identity, PaperProtocolIdentity):
        raise PaperRegistryError("invalid protocol identity")
    for label, value in (("name", identity.name), ("version", identity.version)):
        if (
            type(value) is not str
            or not value
            or value.strip() != value
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
        ):
            raise PaperRegistryError(
                f"protocol identity {label} must use [A-Za-z0-9_.-]+"
            )
    if len(identity.sha256) != 64 or set(identity.sha256) - _SHA256:
        raise PaperRegistryError("protocol identity must carry a lowercase sha256")
    return f"{identity.name}@{identity.version}:{identity.sha256}"

def compose_paper_registry_semantic_identity(
    base_semantic_sha256: str,
    protocol_identities: tuple[PaperProtocolIdentity, ...],
) -> str:

    if (
        type(base_semantic_sha256) is not str
        or len(base_semantic_sha256) != 64
        or set(base_semantic_sha256) - _SHA256
    ):
        raise PaperRegistryError("base registry identity must be a lowercase sha256")
    if not isinstance(protocol_identities, tuple):
        raise PaperRegistryError("protocol identities must be a tuple")
    if not protocol_identities:
        return base_semantic_sha256
    records: list[dict[str, str]] = []
    names: set[str] = set()
    for identity in protocol_identities:
        paper_protocol_identity_token(identity)
        if identity.name in names:
            raise PaperRegistryError(f"duplicate protocol identity {identity.name!r}")
        names.add(identity.name)
        if len(identity.sha256) != 64 or set(identity.sha256) - _SHA256:
            raise PaperRegistryError(
                f"protocol identity {identity.name!r} must carry a lowercase sha256"
            )
        records.append(
            {
                "name": identity.name,
                "version": identity.version,
                "sha256": identity.sha256,
            }
        )
    payload = {
        "schema_version": "dive-paper-registry-protocol-composition-v1",
        "base_semantic_sha256": base_semantic_sha256,
        "protocol_identities": sorted(records, key=lambda item: item["name"]),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def load_paper_evaluator_registry(path: Path) -> PaperEvaluatorRegistry:

    config_path = _resolve_path(path)
    raw = _read_yaml(config_path)
    _exact_keys(
        raw,
        {
            "schema_version",
            "upstream",
            "interpreter",
            "worker",
            "checkpoints",
            "families",
        },
        "paper evaluator registry",
    )
    if raw["schema_version"] != _SCHEMA:
        raise PaperRegistryError("unsupported paper evaluator registry schema_version")
    upstream = _upstream(raw["upstream"])
    interpreter = _interpreter(raw["interpreter"])
    worker = _file_asset(raw["worker"], "worker")
    if worker.sha256 == _FIXTURE_WORKER_SHA256:
        raise PaperRegistryError("fixture hash cannot authenticate as production")
    checkpoints = _checkpoints(raw["checkpoints"])
    families_raw = _mapping(raw["families"], "families")
    _exact_keys(families_raw, set(_FAMILIES), "families")
    families = {family: _family(family, families_raw[family]) for family in _FAMILIES}
    registry = PaperEvaluatorRegistry(
        families=MappingProxyType(families),
        semantic_sha256="",
        interpreter=interpreter,
        worker=worker,
        upstream_root=upstream[0],
        upstream_commit=upstream[1],
        checkpoints=MappingProxyType(checkpoints),
    )
    verify_paper_evaluator_registry(registry)
    _live_verify_registry(registry)
    semantic = hashlib.sha256(
        canonical_json_bytes(_registry_mapping(registry))
    ).hexdigest()
    if semantic == FIXTURE_EVALUATOR_REGISTRY_SHA256:
        raise PaperRegistryError("fixture hash cannot authenticate as production")
    return PaperEvaluatorRegistry(
        families=registry.families,
        semantic_sha256=semantic,
        interpreter=registry.interpreter,
        worker=registry.worker,
        upstream_root=registry.upstream_root,
        upstream_commit=registry.upstream_commit,
        checkpoints=registry.checkpoints,
    )

def verify_paper_evaluator_registry(registry: PaperEvaluatorRegistry) -> None:

    if set(registry.families) != set(_FAMILIES):
        raise PaperRegistryError("paper evaluator families drifted")
    if registry.semantic_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256:
        raise PaperRegistryError("fixture hash cannot authenticate as production")
    if registry.worker.sha256 == _FIXTURE_WORKER_SHA256:
        raise PaperRegistryError("fixture hash cannot authenticate as production")
    _refuse_unknown_generation_identities(registry)
    for family in _FAMILIES:
        entry = registry.families[family]
        if entry.method != _METHODS[family]:
            if family == "ame":
                raise PaperRegistryError("incomplete_ame_production_method")
            raise PaperRegistryError(f"{family} method is not the production method")
        primary, direction = _PRIMARIES[family]
        if entry.primary != primary or entry.direction != direction:
            if family == "ame":
                raise PaperRegistryError("incomplete_ame_production_method")
            raise PaperRegistryError(f"{family} primary or direction drifted")
        input_schema, output_schema = _SCHEMAS[family]
        if entry.input_schema != input_schema or entry.output_schema != output_schema:
            raise PaperRegistryError(f"{family} schema drifted")
        redesigns, evaluators = _COUNTS[family]
        if entry.redesign_count != redesigns or entry.evaluator_count != evaluators:
            raise PaperRegistryError(f"{family} redesign/evaluator counts drifted")
        if entry.selection_rule != _SELECTION_RULES[family]:
            raise PaperRegistryError(f"{family} selection rule drifted")
        _verify_availability(family, entry)
    _verify_binder_thresholds(registry.families["binder"])
    _verify_ame_method(registry.families["ame"])
    _verify_antibody_method(registry.families["antibody"])

def _verify_binder_thresholds(entry: PaperFamilyEvaluator) -> None:
    expected = {
        ("i_pAE_times_31", "<=", 7),
        ("pLDDT", ">=", 0.9),
        ("binder_scRMSD_ca", "<", 1.5),
    }
    observed = {(item.name, item.op, item.value) for item in entry.thresholds}
    if observed != expected:
        raise PaperRegistryError("binder production thresholds drifted")
    normalizers = [
        asset
        for asset in entry.assets
        if asset.role == _BINDER_BACKBONE_NORMALIZATION_ROLE
    ]
    if (
        len(normalizers) != 1
        or not normalizers[0].required
        or normalizers[0].kind != "source_file"
        or normalizers[0].identity is None
    ):
        raise PaperRegistryError("binder backbone normalization identity drifted")
    references = [
        asset for asset in entry.assets if asset.role == _BINDER_BACKBONE_REFERENCE_ROLE
    ]
    if (
        len(references) != 1
        or not references[0].required
        or references[0].kind != "source_file"
        or references[0].identity is None
    ):
        raise PaperRegistryError("binder backbone geometry reference drifted")
    for role, label in (
        (_BINDER_PDB_AUTHENTICATION_ROLE, "PDB authentication"),
        (_BINDER_PRIVATE_SNAPSHOT_ROLE, "private snapshot protocol"),
    ):
        assets = [asset for asset in entry.assets if asset.role == role]
        if (
            len(assets) != 1
            or not assets[0].required
            or assets[0].kind != "source_file"
            or assets[0].identity is None
        ):
            raise PaperRegistryError(f"binder {label} identity drifted")

def _verify_ame_method(entry: PaperFamilyEvaluator) -> None:
    roles = tuple(asset.role for asset in entry.assets)
    if (
        entry.method != _METHODS["ame"]
        or entry.primary != "ame_motif_ligand_success"
        or entry.primary == "min_ipAE"
        or "min_ipAE" not in entry.required_raw_secondaries
        or any(role not in roles for role in _AME_REQUIRED_ROLES)
        or set(roles) <= {"rf3_checkpoint", "rf3_source", "rf3_executable"}
    ):
        raise PaperRegistryError("incomplete_ame_production_method")
    expected = {
        ("binder_scRMSD_bb3", "<=", 2.0),
        ("motif_all_atom_rmsd", "<=", 1.5),
        ("exact_motif_sequence_recovery", "==", True),
        ("ligand_clash", "==", False),
    }
    observed = {(item.name, item.op, item.value) for item in entry.thresholds}
    if observed != expected:
        raise PaperRegistryError("incomplete_ame_production_method")
    transports = [
        asset for asset in entry.assets if asset.role == _AME_LIGAND_TRANSPORT_ROLE
    ]
    if (
        len(transports) != 1
        or not transports[0].required
        or transports[0].kind != "source_file"
        or transports[0].identity is None
    ):
        raise PaperRegistryError("incomplete_ame_production_method")

def _verify_antibody_method(entry: PaperFamilyEvaluator) -> None:
    if (
        entry.redesign_count != 0
        or entry.selection_rule != "in_process_native_geometry"
    ):
        raise PaperRegistryError(
            "antibody production method forbids redesign/refolding"
        )
    roles = [asset.role for asset in entry.assets]
    if "evaluate_antibody" not in roles:
        raise PaperRegistryError("antibody must bind in-process evaluate_antibody")
    correspondences = [
        asset for asset in entry.assets if asset.role == _ANTIBODY_CORRESPONDENCE_ROLE
    ]
    if (
        len(correspondences) != 1
        or not correspondences[0].required
        or correspondences[0].kind != "source_file"
        or correspondences[0].identity is None
    ):
        raise PaperRegistryError(
            "antibody insertion-aware correspondence identity drifted"
        )

def _verify_availability(family: str, entry: PaperFamilyEvaluator) -> None:
    codes = tuple(item.reason_code for item in entry.availability)
    unknown = [code for code in codes if code not in _ALLOWED_OPTIONAL_GAPS]
    if unknown:
        raise PaperRegistryError(f"{family} has undeclared availability gap {unknown}")
    if family != "binder" and entry.availability:
        raise PaperRegistryError(f"{family} may not declare optional evaluator gaps")
    if family == "binder":
        expected = (
            "freesasa_dependency_unavailable",
            "novelty_database_provenance_unavailable",
        )
        if codes != expected:
            raise PaperRegistryError("binder optional availability drifted")
        freesasa, novelty = entry.availability
        if freesasa.required or novelty.required:
            raise PaperRegistryError("binder optional gaps must not be required")
        if tuple(freesasa.metrics) != _FREESASA_METRICS:
            raise PaperRegistryError("binder FreeSASA availability metrics drifted")
        if novelty.metrics != ("novelty_from_list",):
            raise PaperRegistryError("binder novelty availability metrics drifted")

def _refuse_unknown_generation_identities(registry: PaperEvaluatorRegistry) -> None:
    identities: list[tuple[str, str]] = [
        (Path(item.path).name, item.sha256) for item in registry.checkpoints.values()
    ]
    identities.append((Path(registry.worker.path).name, registry.worker.sha256))
    for family in registry.families.values():
        identities.append(
            (Path(family.reducer_source.path).name, family.reducer_source.sha256)
        )
        for asset in family.assets:
            if asset.identity is not None:
                identities.append(
                    (Path(asset.identity.path).name, asset.identity.sha256)
                )
            if asset.tree is not None:
                identities.append((Path(asset.tree.root).name, asset.tree.sha256))
    for name, digest in identities:
        if (
            name in _FORBIDDEN_GENERATION_NAMES
            or digest in _FORBIDDEN_GENERATION_HASHES
        ):
            raise PaperRegistryError(
                "unknown-provenance generation checkpoint is forbidden"
            )

def _upstream(value: object) -> tuple[str, str]:
    entry = _mapping(value, "upstream")
    _exact_keys(entry, {"root", "commit"}, "upstream")
    root, commit = entry["root"], entry["commit"]
    if type(root) is not str or not root or type(commit) is not str or not commit:
        raise PaperRegistryError("upstream root and commit must be non-empty strings")
    if Path(root) != EMERGENT_UPSTREAM_ROOT or commit != EMERGENT_UPSTREAM_COMMIT:
        raise PaperRegistryError("unexpected upstream pin")
    return str(EMERGENT_UPSTREAM_ROOT), EMERGENT_UPSTREAM_COMMIT

def _verify_upstream_checkout(root: str, commit: str) -> None:
    head = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", root, "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        head.returncode != 0
        or head.stdout.strip() != commit
        or dirty.returncode != 0
        or dirty.stdout
    ):
        raise PaperRegistryError("pinned upstream checkout HEAD or cleanliness drifted")

def _live_verify_registry(registry: PaperEvaluatorRegistry) -> None:
    try:
        _authenticate_interpreter(registry.interpreter, "paper evaluator interpreter")
        _assert_identity(registry.interpreter.environment, "interpreter.environment")
        _assert_identity(registry.worker, "worker")
        for name, identity in registry.checkpoints.items():
            _assert_identity(identity, f"checkpoints.{name}")
        for family, entry in registry.families.items():
            _assert_identity(entry.reducer_source, f"{family}.reducer")
            for asset in entry.assets:
                if asset.identity is not None:
                    _assert_identity(asset.identity, f"{family}.assets.{asset.role}")
                if asset.tree is not None:
                    observed = _observe_tree(Path(asset.tree.root))
                    if observed != asset.tree:
                        raise PaperRegistryError(
                            f"{family}.assets.{asset.role} tree identity drifted"
                        )
                if asset.kind == "executable" and asset.runtime is not None:
                    _live_executable_runtime(
                        asset.identity, asset.runtime, f"{family}.assets.{asset.role}"
                    )
    except EvaluatorContractError as error:
        raise PaperRegistryError(str(error)) from error
    _verify_upstream_checkout(registry.upstream_root, registry.upstream_commit)

def _interpreter(value: object) -> InterpreterContract:
    entry = _mapping(value, "interpreter")
    _exact_keys(
        entry,
        {"path", "sha256", "size_bytes", "version", "environment"},
        "interpreter",
    )
    version = entry["version"]
    declared = entry["path"]
    if type(version) is not str or not version:
        raise PaperRegistryError("interpreter.version must be a non-empty string")
    if type(declared) is not str or not declared:
        raise PaperRegistryError("interpreter.path must be a non-empty string")
    digest, size = _digest_size(entry, "interpreter")
    resolved = Path(declared).resolve()
    identity = ArtifactIdentity(str(resolved), digest, size)
    environment = _file_asset(
        _mapping(entry["environment"], "interpreter.environment"),
        "interpreter.environment",
    )
    return InterpreterContract(declared, identity, version, environment)

def _checkpoints(value: object) -> dict[str, ArtifactIdentity]:
    entry = _mapping(value, "checkpoints")
    _exact_keys(entry, {"common", "autoencoder"}, "checkpoints")
    checkpoints = {
        name: _file_asset(
            _mapping(entry[name], f"checkpoints.{name}"), f"checkpoints.{name}"
        )
        for name in ("common", "autoencoder")
    }
    for identity in checkpoints.values():
        if Path(identity.path).name in _FORBIDDEN_GENERATION_NAMES:
            raise PaperRegistryError(
                "unknown-provenance generation checkpoint is forbidden"
            )
    return checkpoints

def _family(family: str, value: object) -> PaperFamilyEvaluator:
    entry = _mapping(value, f"families.{family}")
    _exact_keys(entry, _FAMILY_KEYS, f"families.{family}")
    method = _nonempty(entry["method"], f"{family}.method")
    primary = _nonempty(entry["primary"], f"{family}.primary")
    direction = _nonempty(entry["direction"], f"{family}.direction")
    input_schema = _nonempty(entry["input_schema"], f"{family}.input_schema")
    output_schema = _nonempty(entry["output_schema"], f"{family}.output_schema")
    selection_rule = _nonempty(entry["selection_rule"], f"{family}.selection_rule")
    redesign_count = _count(entry["redesign_count"], f"{family}.redesign_count")
    evaluator_count = _count(entry["evaluator_count"], f"{family}.evaluator_count")
    secondaries = _string_tuple(
        entry["required_raw_secondaries"], f"{family}.required_raw_secondaries"
    )
    if family == "ame" and (
        primary in {"min_ipAE", "ame_success_min_ipae"}
        or method != _METHODS["ame"]
        or "min_ipAE" not in secondaries
    ):
        raise PaperRegistryError("incomplete_ame_production_method")
    thresholds = _thresholds(entry["thresholds"], family)
    reducer_source, reducer = _reducer(entry["reducer"], family)
    assets = _assets(entry["assets"], family)
    availability = _availability(entry["availability"], family)
    return PaperFamilyEvaluator(
        method=method,
        primary=primary,
        direction=direction,
        input_schema=input_schema,
        output_schema=output_schema,
        redesign_count=redesign_count,
        evaluator_count=evaluator_count,
        selection_rule=selection_rule,
        required_raw_secondaries=secondaries,
        thresholds=thresholds,
        reducer=reducer,
        reducer_source=reducer_source,
        assets=assets,
        availability=availability,
    )

def _thresholds(value: object, family: str) -> tuple[PaperThreshold, ...]:
    label = f"{family}.thresholds"
    entry = _mapping(value, label)
    if family == "antibody" and entry:
        raise PaperRegistryError("antibody has no production success thresholds")
    rows = []
    for name in sorted(entry):
        item = _mapping(entry[name], f"{label}.{name}")
        _exact_keys(item, {"op", "value"}, f"{label}.{name}")
        op, raw_value = item["op"], item["value"]
        if type(op) is not str or not op:
            raise PaperRegistryError(f"{label}.{name} op is invalid")
        if type(raw_value) not in (int, float, bool):
            raise PaperRegistryError(f"{label}.{name} value is invalid")
        rows.append(PaperThreshold(name, op, raw_value))
    return tuple(rows)

def _reducer(value: object, family: str) -> tuple[ArtifactIdentity, ReducerContract]:
    label = f"{family}.reducer"
    entry = _mapping(value, label)
    _exact_keys(entry, {"identifier", "path", "sha256", "size_bytes"}, label)
    identifier = _nonempty(entry["identifier"], f"{label}.identifier")
    identity = _file_asset(entry, label)
    return identity, ReducerContract(identifier, identity.sha256)

def _assets(value: object, family: str) -> tuple[PaperEvaluatorAsset, ...]:
    label = f"{family}.assets"
    if not isinstance(value, list) or not value:
        raise PaperRegistryError(f"{label} must be a non-empty list")
    assets: list[PaperEvaluatorAsset] = []
    roles: set[str] = set()
    for index, raw in enumerate(value):
        entry = _mapping(raw, f"{label}[{index}]")
        role = _nonempty(entry.get("role"), f"{label}[{index}].role")
        kind = _nonempty(entry.get("kind"), f"{label}[{index}].kind")
        required = entry.get("required")
        if role in roles or type(required) is not bool:
            raise PaperRegistryError(f"{label}[{index}] has invalid or repeated role")
        roles.add(role)
        identity = tree = runtime = None
        if kind == "module_tree":
            _exact_keys(
                entry,
                {
                    "role",
                    "kind",
                    "required",
                    "root",
                    "sha256",
                    "size_bytes",
                    "file_count",
                },
                f"{label}[{index}]",
            )
            tree = _tree_asset(entry, f"{label}[{index}]")
        elif kind in {"checkpoint", "source_file", "executable"}:
            expected = {"role", "kind", "required", "path", "sha256", "size_bytes"}
            if kind == "executable":
                expected.add("runtime")
            _exact_keys(entry, expected, f"{label}[{index}]")
            identity = _file_asset(entry, f"{label}[{index}]")
            if Path(identity.path).name in _FORBIDDEN_GENERATION_NAMES:
                raise PaperRegistryError(
                    "unknown-provenance generation checkpoint is forbidden"
                )
            if identity.sha256 in _FORBIDDEN_GENERATION_HASHES:
                raise PaperRegistryError(
                    "unknown-provenance generation checkpoint is forbidden"
                )
            if kind == "executable":
                runtime = _declared_runtime(entry["runtime"], f"{label}[{index}]")
        else:
            raise PaperRegistryError(f"{label}[{index}] has unknown asset kind")
        assets.append(
            PaperEvaluatorAsset(role, kind, required, identity, tree, runtime)
        )
    if family == "ame":
        missing = [role for role in _AME_REQUIRED_ROLES if role not in roles]
        if missing:
            raise PaperRegistryError("incomplete_ame_production_method")
    return tuple(assets)

def _availability(value: object, family: str) -> tuple[PaperAvailabilityGap, ...]:
    label = f"{family}.availability"
    if not isinstance(value, list):
        raise PaperRegistryError(f"{label} must be a list")
    gaps = []
    for index, raw in enumerate(value):
        entry = _mapping(raw, f"{label}[{index}]")
        _exact_keys(entry, {"reason_code", "required", "metrics"}, f"{label}[{index}]")
        reason = _nonempty(entry["reason_code"], f"{label}[{index}].reason_code")
        required = entry["required"]
        if type(required) is not bool:
            raise PaperRegistryError(f"{label}[{index}].required must be a boolean")
        metrics = _string_tuple(entry["metrics"], f"{label}[{index}].metrics")
        if reason not in _ALLOWED_OPTIONAL_GAPS:
            raise PaperRegistryError(
                f"{family} has undeclared availability gap {reason}"
            )
        gaps.append(PaperAvailabilityGap(reason, required, metrics))
    return tuple(gaps)

def _declared_runtime(value: object, label: str) -> Mapping[str, str]:
    entry = _mapping(value, f"{label}.runtime")
    _exact_keys(entry, {"shebang", "entry_module"}, f"{label}.runtime")
    return MappingProxyType(
        {
            "shebang": _nonempty(entry["shebang"], f"{label}.runtime.shebang"),
            "entry_module": _nonempty(
                entry["entry_module"], f"{label}.runtime.entry_module"
            ),
        }
    )

def _live_executable_runtime(
    identity: ArtifactIdentity | None, runtime: Mapping[str, str], label: str
) -> None:
    if identity is None:
        raise PaperRegistryError(f"{label} executable is missing an identity")
    path = Path(identity.path)
    if not os.access(path, os.X_OK):
        raise PaperRegistryError(f"{label} is not executable")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PaperRegistryError(f"{label} cannot be read: {error}") from error
    first = text.splitlines()[0] if text.splitlines() else ""
    if first != f"#!{runtime['shebang']}":
        raise PaperRegistryError(f"{label} shebang drifted")
    if runtime["entry_module"] not in text:
        raise PaperRegistryError(f"{label} does not import {runtime['entry_module']}")

def _file_asset(value: Mapping[str, object], label: str) -> ArtifactIdentity:
    path_value = value["path"]
    digest, size = _digest_size(value, label)
    if type(path_value) is not str or not path_value:
        raise PaperRegistryError(f"{label}.path must be a non-empty string")
    resolved = _resolve_path(Path(path_value))
    return ArtifactIdentity(str(resolved), digest, size)

def _tree_asset(value: Mapping[str, object], label: str) -> ModuleTreeIdentity:
    root_value = value["root"]
    digest, size = _digest_size(value, label)
    count = value["file_count"]
    if type(root_value) is not str or not root_value:
        raise PaperRegistryError(f"{label}.root must be a non-empty string")
    if type(count) is not int or count < 1:
        raise PaperRegistryError(f"{label}.file_count must be a positive integer")
    root = Path(root_value).resolve()
    return ModuleTreeIdentity(str(root), digest, size, count)

def _observe_tree(root: Path) -> ModuleTreeIdentity:
    if not root.is_dir():
        raise PaperRegistryError(f"directory asset is not a directory: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    records = []
    for path in files:
        data = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return ModuleTreeIdentity(
        str(root),
        hashlib.sha256(payload).hexdigest(),
        sum(record["size_bytes"] for record in records),
        len(records),
    )

def _registry_mapping(registry: PaperEvaluatorRegistry) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA,
        "upstream": {
            "root": registry.upstream_root,
            "commit": registry.upstream_commit,
        },
        "interpreter": {
            "path": registry.interpreter.path,
            "sha256": registry.interpreter.identity.sha256,
            "size_bytes": registry.interpreter.identity.size_bytes,
            "version": registry.interpreter.version,
            "environment": {
                "path": registry.interpreter.environment.path,
                "sha256": registry.interpreter.environment.sha256,
                "size_bytes": registry.interpreter.environment.size_bytes,
            },
        },
        "worker": {
            "path": registry.worker.path,
            "sha256": registry.worker.sha256,
            "size_bytes": registry.worker.size_bytes,
        },
        "checkpoints": {
            name: {
                "path": identity.path,
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            }
            for name, identity in registry.checkpoints.items()
        },
        "families": {
            family: {
                "method": entry.method,
                "primary": entry.primary,
                "direction": entry.direction,
                "input_schema": entry.input_schema,
                "output_schema": entry.output_schema,
                "redesign_count": entry.redesign_count,
                "evaluator_count": entry.evaluator_count,
                "selection_rule": entry.selection_rule,
                "required_raw_secondaries": list(entry.required_raw_secondaries),
                "thresholds": {
                    item.name: {"op": item.op, "value": item.value}
                    for item in entry.thresholds
                },
                "reducer": {
                    "identifier": entry.reducer.identifier,
                    "semantic_sha256": entry.reducer.semantic_sha256,
                    "path": entry.reducer_source.path,
                    "sha256": entry.reducer_source.sha256,
                    "size_bytes": entry.reducer_source.size_bytes,
                },
                "assets": [_asset_mapping(asset) for asset in entry.assets],
                "availability": [
                    {
                        "reason_code": gap.reason_code,
                        "required": gap.required,
                        "metrics": list(gap.metrics),
                    }
                    for gap in entry.availability
                ],
            }
            for family, entry in registry.families.items()
        },
    }

def _asset_mapping(asset: PaperEvaluatorAsset) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": asset.role,
        "kind": asset.kind,
        "required": asset.required,
    }
    if asset.identity is not None:
        payload.update(
            {
                "path": asset.identity.path,
                "sha256": asset.identity.sha256,
                "size_bytes": asset.identity.size_bytes,
            }
        )
    if asset.tree is not None:
        payload.update(
            {
                "root": asset.tree.root,
                "sha256": asset.tree.sha256,
                "size_bytes": asset.tree.size_bytes,
                "file_count": asset.tree.file_count,
            }
        )
    if asset.runtime is not None:
        payload["runtime"] = dict(asset.runtime)
    return payload

def _read_yaml(path: Path) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PaperRegistryError(
            f"cannot read paper evaluator registry {path}: {error}"
        ) from error
    return _mapping(raw, "paper evaluator registry")

def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise PaperRegistryError(f"{label} must be a string-keyed mapping")
    return value

def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    unknown, missing = sorted(set(value) - expected), sorted(expected - set(value))
    if unknown or missing:
        raise PaperRegistryError(
            f"{label} has unknown {unknown} or missing {missing} field(s)"
        )

def _digest_size(value: Mapping[str, object], label: str) -> tuple[str, int]:
    digest, size = value["sha256"], value["size_bytes"]
    if type(digest) is not str or len(digest) != 64 or set(digest) - _SHA256:
        raise PaperRegistryError(f"{label}.sha256 must be a lowercase SHA-256 identity")
    if type(size) is not int or size < 0:
        raise PaperRegistryError(f"{label}.size_bytes must be a non-negative integer")
    return digest, size

def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise PaperRegistryError(f"{label} must be a non-empty string")
    return value

def _count(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PaperRegistryError(f"{label} must be a non-negative integer")
    return value

def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        type(item) is not str or not item for item in value
    ):
        raise PaperRegistryError(f"{label} must be a list of non-empty strings")
    return tuple(value)

def _resolve_path(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return candidate.resolve()
