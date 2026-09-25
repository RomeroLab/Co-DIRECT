
from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.evaluators.antibody import ResidueKey as EvaluatorResidueKey
from dive.benchmark.paper_batches import (
    PaperBatch,
    PaperBatchError,
    PaperRequestedCellFailure,
    load_paper_batch,
)
from dive.benchmark.paper_export import (
    AME_LIGAND_TRANSPORT_VERSION,
    BINDER_BACKBONE_NORMALIZATION_VERSION,
    PaperExportError,
    _atom_is_present,
    _classify_ame_ligand,
    _auth_chain_for_label,
    _label_to_auth_mapping,
    _ligand_coordinate_payload,
    _ligand_membership_payload,
    _load_ame_ligand_with_author_identity,
    _native_atom_records,
    _normalize_binder_backbone,
    _predicted_atom_records,
    _residue_rows,
    _unique_residue_projection,
    _sha256_payload,
    _structure_from_batch,
    ame_ligand_chain_id,
)
from dive.benchmark.paper_inputs import PaperTargetRecord
from dive.benchmark.paper_protocol_audit import (
    AmeComponentObservation,
    AmeObservation,
    AntibodyResidueObservation,
    AtomKey,
    AuditError,
    BinderResidueObservation,
    ExampleAuditRecord,
    ResidueKey,
    example_audit_record_mapping,
    scan_ame_example,
    scan_antibody_example,
    scan_binder_example,
    typed_requested_cell_failure_record,
    validate_audit_destination,
)
from dive.benchmark.paper_registry import (
    PaperEvaluatorRegistry,
    PaperProtocolIdentity,
    paper_protocol_identity_token,
)
from dive.signed_value.roots import EMERGENT_EVIDENCE_ROOT
from dive.training.preflight import canonical_json_bytes

ANTIBODY_CORRESPONDENCE_VERSION = "antibody-insertion-aware-correspondence-v1"
_SCAN_ID = re.compile(r"paper-baseline-protocol-scan-[a-z0-9][a-z0-9-]{2,63}")
_BACKBONE_INDEX = (("N", 0), ("CA", 1), ("C", 2), ("O", 4))
_ZERO_DIGEST = "0" * 64
_TEST_EVIDENCE_ROOT: Path | None = None

def paper_protocol_identities(
    registry: PaperEvaluatorRegistry,
) -> tuple[PaperProtocolIdentity, ...]:

    return (
        PaperProtocolIdentity(
            "binder.atom37",
            BINDER_BACKBONE_NORMALIZATION_VERSION,
            _required_asset_hash(
                registry, "binder", "binder_atom37_backbone_normalization_v1"
            ),
        ),
        PaperProtocolIdentity(
            "ame.canonical_ligand",
            AME_LIGAND_TRANSPORT_VERSION,
            _required_asset_hash(registry, "ame", "ame_canonical_ligand_transport_v1"),
        ),
        PaperProtocolIdentity(
            "antibody.atom_key",
            ANTIBODY_CORRESPONDENCE_VERSION,
            _required_asset_hash(
                registry, "antibody", "antibody_insertion_aware_correspondence_v1"
            ),
        ),
    )

def protocol_identities_by_family(
    identities: Sequence[PaperProtocolIdentity],
) -> dict[str, PaperProtocolIdentity]:
    bound: dict[str, PaperProtocolIdentity] = {}
    for identity in identities:
        family, separator, _rest = identity.name.partition(".")
        if not separator:
            raise AuditError(f"unexpected protocol family {identity.name}")
        if family in bound:
            raise AuditError(f"duplicate protocol binding for family {family}")
        bound[family] = identity
    return bound

def scan_paper_target(
    target: PaperTargetRecord,
    *,
    protocol_identity: PaperProtocolIdentity,
    batch: PaperBatch | None = None,
) -> ExampleAuditRecord:

    if not isinstance(target, PaperTargetRecord):
        raise AuditError("target must be a PaperTargetRecord")
    token = paper_protocol_identity_token(protocol_identity)
    family, separator, _rest = protocol_identity.name.partition(".")
    if not separator or family != target.family:
        raise AuditError("protocol identity family does not match target")
    if target.family == "binder":
        return _scan_binder(target, batch, token)
    if target.family == "ame":
        return _scan_ame(target, token)
    if target.family == "antibody":
        return _scan_antibody(target, batch, token)
    raise AuditError(f"unknown audit family {target.family}")

def scan_all_targets(
    targets: Sequence[PaperTargetRecord],
    identities: Mapping[str, PaperProtocolIdentity],
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[ExampleAuditRecord, ...]:
    records: list[ExampleAuditRecord] = []
    total = len(targets)
    for index, target in enumerate(targets, start=1):
        if log is not None:
            log(f"{index}/{total} {target.family} {target.example_id}\n")
        identity = identities.get(target.family)
        if identity is None:
            raise AuditError(f"missing protocol identity for {target.family}")
        records.append(scan_paper_target(target, protocol_identity=identity))
    return tuple(records)

def claim_scanner_destination(scan_id: str) -> Path:

    if type(scan_id) is not str or _SCAN_ID.fullmatch(scan_id) is None:
        raise AuditError(f"invalid scan id {scan_id!r}")
    root = _TEST_EVIDENCE_ROOT or EMERGENT_EVIDENCE_ROOT
    destination = root / "paper_quality_baseline" / scan_id
    validate_audit_destination(destination)
    try:
        metadata = os.lstat(destination)
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        raise AuditError(f"scan destination already exists: {destination}")
    destination.mkdir(mode=0o2775, parents=True)
    return destination

def write_scanner_jsonl(
    records: Sequence[ExampleAuditRecord], destination: Path
) -> ArtifactIdentity:

    path = validate_audit_destination(Path(destination))
    if not records:
        raise AuditError("scanner jsonl is empty")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        raise AuditError("scanner jsonl already exists")
    raw = b"".join(
        canonical_json_bytes(example_audit_record_mapping(record)) for record in records
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o664)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AuditError("scanner jsonl write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o664)
    finally:
        os.close(descriptor)
    return ArtifactIdentity(str(path), hashlib.sha256(raw).hexdigest(), len(raw))

def _required_asset_hash(
    registry: PaperEvaluatorRegistry, family: str, role: str
) -> str:
    assets = [asset for asset in registry.families[family].assets if asset.role == role]
    if (
        len(assets) != 1
        or assets[0].identity is None
        or type(assets[0].identity.sha256) is not str
    ):
        raise AuditError(f"missing {family} protocol asset {role}")
    return assets[0].identity.sha256

def _load_batch(target: PaperTargetRecord, batch: PaperBatch | None) -> PaperBatch:
    if batch is not None:
        return batch
    try:
        return load_paper_batch(target)
    except PaperRequestedCellFailure:
        raise
    except (PaperBatchError, PaperExportError) as error:
        raise PaperRequestedCellFailure(
            f"loader exclusion for example {target.example_id!r}",
            reason_code="loader_exclusion",
        ) from error

def _atoms_from_row(row: Mapping[str, object]) -> dict[str, tuple[float, float, float]]:
    coords = row["coords"]
    atoms: dict[str, tuple[float, float, float]] = {}
    for name, index in _BACKBONE_INDEX:
        if index >= len(coords) and not hasattr(coords, "shape"):
            continue
        try:
            xyz = coords[index]
        except (IndexError, TypeError, KeyError):
            continue
        if _atom_is_present(xyz):
            atoms[name] = tuple(float(value) for value in xyz)
    return atoms

def _binder_observations(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
) -> tuple[BinderResidueObservation, ...]:
    if len(before_rows) != len(after_rows):
        raise PaperExportError("binder normalization changed residue count")
    observations = []
    seen: set[str] = set()
    for before, after in zip(before_rows, after_rows, strict=True):
        key = f"{before['chain']}:{int(before['residue'])}:"
        if key in seen:
            raise PaperExportError(f"duplicate binder residue key {key}")
        seen.add(key)
        source = "generated" if bool(before["generated"]) else "conditioning"
        observations.append(
            BinderResidueObservation(
                key,
                source,
                _atoms_from_row(before),
                _atoms_from_row(after),
            )
        )
    if not observations:
        raise PaperExportError("binder residues are empty")
    return tuple(observations)

def _typed_failure(
    target: PaperTargetRecord, token: str, reason_code: str
) -> ExampleAuditRecord:
    return typed_requested_cell_failure_record(
        family=target.family,
        parent_id=target.parent_id,
        example_id=target.example_id,
        reason_code=reason_code,
        normalization_identity=token,
    )

def _scan_binder(
    target: PaperTargetRecord, batch: PaperBatch | None, token: str
) -> ExampleAuditRecord:
    try:
        loaded = _load_batch(target, batch)
    except PaperRequestedCellFailure:
        return _typed_failure(target, token, "loader_exclusion")
    try:
        before_rows = _residue_rows(_structure_from_batch(loaded), loaded)
        try:
            after_rows, _normalization = _normalize_binder_backbone(before_rows, loaded)
        except PaperExportError:
            after_rows = before_rows
        observations = _binder_observations(before_rows, after_rows)
    except (PaperExportError, PaperBatchError, TypeError, ValueError, KeyError):
        return _typed_failure(target, token, "incomplete_backbone")
    record = scan_binder_example(
        parent_id=target.parent_id,
        example_id=target.example_id,
        residues=observations,
        normalization_identity=token,
    )
    if (
        not record.representable
        and record.binder is not None
        and not record.binder.four_atom_preflight_result
    ):
        return scan_binder_example(
            parent_id=target.parent_id,
            example_id=target.example_id,
            residues=observations,
            normalization_identity=token,
            requested_cell_failure="incomplete_backbone",
        )
    return record

def _ame_dummy_component(
    token: str, *, representation: str, roles: tuple[str, ...]
) -> AmeComponentObservation:
    return AmeComponentObservation(
        token,
        roles,
        representation,
        _ZERO_DIGEST,
        _ZERO_DIGEST,
        _ZERO_DIGEST,
        _ZERO_DIGEST,
    )

def _ame_component(
    token: str,
    ligand,
    *,
    representation: str,
    roles: tuple[str, ...],
) -> AmeComponentObservation:
    membership = _sha256_payload(_ligand_membership_payload(ligand))
    coordinates = _sha256_payload(_ligand_coordinate_payload(ligand))
    return AmeComponentObservation(
        token,
        roles,
        representation,
        membership,
        coordinates,
        membership,
        coordinates,
    )

def _scan_ame(target: PaperTargetRecord, token: str) -> ExampleAuditRecord:
    smiles = target.ligand_smiles
    identity = target.ligand_identity
    try:
        component_token = ame_ligand_chain_id(target.example_id)
    except PaperExportError:
        component_token = target.canonical_component_token
    recorded = target.canonical_component_token
    if (
        recorded is not None
        and component_token is not None
        and recorded != (component_token)
    ):
        component_token = None
    if identity is None:
        identity = component_token
    parser_ok = True
    components: tuple[AmeComponentObservation, ...] = ()
    if (
        type(component_token) is str
        and component_token
        and "_" in component_token
        and len([part for part in component_token.split("_") if part]) != 1
    ):
        dummy = _ame_dummy_component(
            component_token, representation="non_polymer", roles=("ligand",)
        )
        components = (dummy, dummy)
        parser_ok = False
    elif type(component_token) is str and component_token:
        try:
            ligand = _load_ame_ligand_with_author_identity(
                Path(target.source_path), component_token
            )
            try:
                representation, _hetero = _classify_ame_ligand(ligand)
                roles: tuple[str, ...] = ("ligand",)
            except PaperExportError as error:
                parser_ok = False
                message = str(error)
                if "water-only" in message:
                    representation = "water"
                    roles = ("ligand",)
                elif "mixed" in message:
                    representation = "non_polymer"
                    roles = ("ligand", "other")
                elif "ambiguous" in message:
                    dummy = _ame_dummy_component(
                        component_token,
                        representation="non_polymer",
                        roles=("ligand",),
                    )
                    components = (dummy, dummy)
                    representation = None
                    roles = ("ligand",)
                else:
                    representation = "non_polymer"
                    roles = ("ligand",)
            if representation is not None:
                components = (
                    _ame_component(
                        component_token,
                        ligand,
                        representation=representation,
                        roles=roles,
                    ),
                )
        except PaperExportError as error:
            parser_ok = False
            message = str(error)
            if "ambiguous" in message and type(component_token) is str:
                dummy = _ame_dummy_component(
                    component_token, representation="non_polymer", roles=("ligand",)
                )
                components = (dummy, dummy)
            else:
                components = ()
    else:
        parser_ok = False
        components = ()
    if identity != component_token:
        parser_ok = False
    observation = AmeObservation(
        canonical_component_token=(
            component_token if type(component_token) is str else None
        ),
        canonical_identity=identity if type(identity) is str else None,
        canonical_smiles=smiles if type(smiles) is str else None,
        components=components,
        parser_normalization_result=parser_ok,
        normalization_identity=token,
    )
    return scan_ame_example(
        parent_id=target.parent_id,
        example_id=target.example_id,
        observation=observation,
    )

def _audit_residue_key(key: EvaluatorResidueKey) -> ResidueKey:
    return ResidueKey(key.chain_id, key.auth_seq_id, key.insertion_code)

def _group_atom_keys(records: Sequence[object]) -> dict[ResidueKey, list[AtomKey]]:
    grouped: dict[ResidueKey, list[AtomKey]] = defaultdict(list)
    for atom in records:
        residue = _audit_residue_key(atom.residue_key)
        grouped[residue].append(AtomKey(residue, atom.atom))
    return grouped

def _scan_antibody(
    target: PaperTargetRecord, batch: PaperBatch | None, token: str
) -> ExampleAuditRecord:
    try:
        loaded = _load_batch(target, batch)
    except PaperRequestedCellFailure:
        return _typed_failure(target, token, "loader_exclusion")
    source = Path(target.source_path)
    try:
        mapping = _label_to_auth_mapping(source)
        rows = _residue_rows(_structure_from_batch(loaded), loaded)
        generated_rows = tuple(row for row in rows if bool(row["generated"]))
        if len(generated_rows) < 3:
            raise PaperExportError(
                "antibody train parent is missing frozen H3 or antigen roles"
            )
        projected = _unique_residue_projection(generated_rows, mapping)
        heavy_chains = {key.chain_id for key in projected}
        if len(heavy_chains) != 1:
            raise PaperExportError("antibody H3 maps to multiple heavy chains")
        antigen = tuple(
            _auth_chain_for_label(mapping, part.strip())
            for part in str(loaded.target.role_payload.get("target") or "").split(",")
            if part.strip()
        )
        if not antigen:
            raise PaperExportError(
                "antibody train parent is missing frozen H3 or antigen roles"
            )
        native_atoms = _native_atom_records(source, mapping)
        predicted_atoms = _predicted_atom_records(rows, mapping)
    except (PaperExportError, PaperBatchError, TypeError, ValueError, KeyError):
        return _typed_failure(target, token, "loader_exclusion")
    h3 = {_audit_residue_key(key) for key in projected}
    source_atoms = _group_atom_keys(native_atoms)
    predicted = _group_atom_keys(predicted_atoms)
    keys = set(source_atoms) | set(predicted) | h3
    if not keys:
        return _typed_failure(target, token, "loader_exclusion")
    observations = tuple(
        AntibodyResidueObservation(
            source_key=key,
            frozen_h3=key in h3,
            predicted_keys=(key,),
            source_atoms=tuple(source_atoms.get(key, ())),
            predicted_atoms=tuple(predicted.get(key, ())),
        )
        for key in sorted(keys)
    )
    return scan_antibody_example(
        parent_id=target.parent_id,
        example_id=target.example_id,
        residues=observations,
        normalization_identity=token,
    )
