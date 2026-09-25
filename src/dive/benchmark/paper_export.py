
from __future__ import annotations

import hashlib
import io
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import (
    ArtifactIdentity,
    BenchmarkContractError,
    file_identity,
)
from dive.benchmark.evaluators.antibody import (
    AtomRecord,
    FrozenAntibodyRoles,
    ResidueKey,
    normalize_insertion_code,
)
from dive.benchmark.paper_baseline import PaperBaselineError
from dive.benchmark.paper_batches import PaperBatch
from dive.benchmark.paper_pdb_auth import AuthenticatedPdb
from dive.signed_value.roots import DENIED_PREFIXES, EMERGENT_UPSTREAM_ROOT
from dive.training.preflight import canonical_json_bytes

class PaperExportError(PaperBaselineError):
    pass

BINDER_BACKBONE_NORMALIZATION_VERSION = "binder-atom37-backbone-v1"
FRAMEFLOW_OXYGEN_ALGORITHM_VERSION = "frameflow-adjust-oxygen-pos-v1"
FRAMEFLOW_REFERENCE_IDENTITY = (
    "proteina-complexa@32b71ae1d9a8767414eae3921cf35969430db82f:"
    "coors_utils.adjust_oxygen_pos@"
    "6ce932fddfe9daa58428f7aeeab9a31d8e6814da861bbda6619504d901422f94"
)
FRAMEFLOW_NEXT_SOURCE_IDENTITY = (
    f"{FRAMEFLOW_REFERENCE_IDENTITY}:authenticated-next-frame"
)
FRAMEFLOW_TERMINAL_SOURCE_IDENTITY = (
    f"{FRAMEFLOW_REFERENCE_IDENTITY}:terminal-or-unavailable-frame"
)
NATIVE_OXYGEN_ALGORITHM_VERSION = "native-exact-oxygen-v1"
NATIVE_OXYGEN_SOURCE_IDENTITY = "authenticated-native-structure-residue-v1"
AME_LIGAND_TRANSPORT_VERSION = "ame-canonical-ligand-transport-v1"
AME_EVALUATOR_LIGAND_CHAIN = "A"
AME_EVALUATOR_PROTEIN_CHAIN = "B"
AME_CLASH_LIGAND_RESNAME = "LIG"
_WATER_RESNAMES = frozenset({"HOH", "WAT"})
_STANDARD_AA = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

@dataclass(frozen=True, slots=True)
class BinderFrameEvidence:
    chain: str
    residue: int
    tensor_index: int
    chain_index: int
    coordinates: tuple[tuple[float, float, float], ...]
    coordinate_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "chain": self.chain,
            "chain_index": self.chain_index,
            "coordinate_sha256": self.coordinate_sha256,
            "coordinates": [list(xyz) for xyz in self.coordinates],
            "residue": self.residue,
            "tensor_index": self.tensor_index,
        }

@dataclass(frozen=True, slots=True)
class BinderAtomRepair:
    chain: str
    residue: int
    atom: str
    coordinate_source: str
    source_identity: str
    algorithm_version: str
    source_artifact: ArtifactIdentity | None
    source_residue_identity: str | None
    source_coordinate: tuple[float, float, float] | None
    source_coordinate_sha256: str | None
    current_frame: BinderFrameEvidence | None
    neighbor_frame: BinderFrameEvidence | None
    adjacency_decision: str
    before_coordinate: tuple[float, float, float]
    before_sha256: str
    after_sha256: str
    source_binding_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "adjacency_decision": self.adjacency_decision,
            "after_sha256": self.after_sha256,
            "atom": self.atom,
            "before_coordinate": list(self.before_coordinate),
            "before_sha256": self.before_sha256,
            "chain": self.chain,
            "coordinate_source": self.coordinate_source,
            "residue": self.residue,
            "source_identity": self.source_identity,
            "source_binding_sha256": self.source_binding_sha256,
            "source_artifact": (
                None
                if self.source_artifact is None
                else {
                    "path": self.source_artifact.path,
                    "sha256": self.source_artifact.sha256,
                    "size_bytes": self.source_artifact.size_bytes,
                }
            ),
            "source_coordinate": (
                None if self.source_coordinate is None else list(self.source_coordinate)
            ),
            "source_coordinate_sha256": self.source_coordinate_sha256,
            "source_residue_identity": self.source_residue_identity,
            "current_frame": (
                None if self.current_frame is None else self.current_frame.to_mapping()
            ),
            "neighbor_frame": (
                None
                if self.neighbor_frame is None
                else self.neighbor_frame.to_mapping()
            ),
        }

@dataclass(frozen=True, slots=True)
class BinderBackboneNormalization:
    version: str
    before_coordinate_sha256: str
    after_coordinate_sha256: str
    before_topology_sha256: str
    after_topology_sha256: str
    roles_before_sha256: str
    roles_after_sha256: str
    sequences_before_sha256: str
    sequences_after_sha256: str
    finite_atoms_before_sha256: str
    finite_atoms_after_sha256: str
    repaired_atoms: tuple[BinderAtomRepair, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "after_coordinate_sha256": self.after_coordinate_sha256,
            "after_topology_sha256": self.after_topology_sha256,
            "before_coordinate_sha256": self.before_coordinate_sha256,
            "before_topology_sha256": self.before_topology_sha256,
            "finite_atoms_after_sha256": self.finite_atoms_after_sha256,
            "finite_atoms_before_sha256": self.finite_atoms_before_sha256,
            "repaired_atoms": [item.to_mapping() for item in self.repaired_atoms],
            "roles_after_sha256": self.roles_after_sha256,
            "roles_before_sha256": self.roles_before_sha256,
            "sequences_after_sha256": self.sequences_after_sha256,
            "sequences_before_sha256": self.sequences_before_sha256,
            "version": self.version,
        }

def binder_backbone_normalization_from_mapping(
    value: object,
) -> BinderBackboneNormalization:
    if not isinstance(value, Mapping):
        raise PaperExportError("binder backbone normalization must be a mapping")
    expected = {
        "after_coordinate_sha256",
        "after_topology_sha256",
        "before_coordinate_sha256",
        "before_topology_sha256",
        "finite_atoms_after_sha256",
        "finite_atoms_before_sha256",
        "repaired_atoms",
        "roles_after_sha256",
        "roles_before_sha256",
        "sequences_after_sha256",
        "sequences_before_sha256",
        "version",
    }
    if set(value) != expected:
        raise PaperExportError("binder backbone normalization schema drifted")

    def digest(key: str, mapping: Mapping[str, object] = value) -> str:
        item = mapping[key]
        if (
            type(item) is not str
            or len(item) != 64
            or set(item) - set("0123456789abcdef")
        ):
            raise PaperExportError(f"binder backbone normalization {key} is invalid")
        return item

    if value["version"] != BINDER_BACKBONE_NORMALIZATION_VERSION:
        raise PaperExportError("binder backbone normalization version drifted")
    raw_repairs = value["repaired_atoms"]
    if not isinstance(raw_repairs, list):
        raise PaperExportError("binder backbone repaired_atoms must be a list")
    repairs = []
    repair_identities: set[tuple[str, int, str]] = set()
    repair_keys = {
        "algorithm_version",
        "adjacency_decision",
        "after_sha256",
        "atom",
        "before_coordinate",
        "before_sha256",
        "chain",
        "coordinate_source",
        "current_frame",
        "neighbor_frame",
        "residue",
        "source_identity",
        "source_artifact",
        "source_binding_sha256",
        "source_coordinate",
        "source_coordinate_sha256",
        "source_residue_identity",
    }
    for raw in raw_repairs:
        if not isinstance(raw, Mapping) or set(raw) != repair_keys:
            raise PaperExportError("binder repaired atom schema drifted")
        if (
            raw["atom"] != "O"
            or type(raw["chain"]) is not str
            or not raw["chain"]
            or type(raw["residue"]) is not int
            or type(raw["source_identity"]) is not str
            or not raw["source_identity"]
        ):
            raise PaperExportError("binder repaired atom identity drifted")
        source_artifact = _binder_source_artifact_from_mapping(raw["source_artifact"])
        source_residue_identity = raw["source_residue_identity"]
        source_coordinate = _binder_coordinate_from_mapping(
            raw["source_coordinate"], label="source coordinate", allow_none=True
        )
        source_coordinate_sha256 = raw["source_coordinate_sha256"]
        current_frame = _binder_frame_evidence_from_mapping(raw["current_frame"])
        neighbor_frame = _binder_frame_evidence_from_mapping(raw["neighbor_frame"])
        before_coordinate = _binder_coordinate_from_mapping(
            raw["before_coordinate"], label="before coordinate", allow_none=False
        )
        assert before_coordinate is not None
        before_sha256 = digest("before_sha256", raw)
        after_sha256 = digest("after_sha256", raw)
        source_binding_sha256 = digest("source_binding_sha256", raw)
        if before_sha256 != _sha256_payload(list(before_coordinate)):
            raise PaperExportError("binder repair before coordinate hash drifted")
        if raw["coordinate_source"] == "authenticated_native_exact_identity":
            expected_residue = f"{raw['chain']}:{raw['residue']}"
            valid_provenance = (
                raw["source_identity"] == NATIVE_OXYGEN_SOURCE_IDENTITY
                and raw["algorithm_version"] == NATIVE_OXYGEN_ALGORITHM_VERSION
                and source_artifact is not None
                and source_residue_identity == expected_residue
                and source_coordinate is not None
                and source_coordinate_sha256 == _sha256_payload(list(source_coordinate))
                and current_frame is None
                and neighbor_frame is None
                and raw["adjacency_decision"] == "not_applicable_native"
            )
            expected_oxygen = source_coordinate
        elif raw["coordinate_source"] == "frameflow_backbone_geometry":
            expected_decision, expected_identity = _derive_geometry_branch(
                raw["chain"], raw["residue"], current_frame, neighbor_frame
            )
            valid_provenance = (
                raw["source_identity"] == expected_identity
                and raw["algorithm_version"] == FRAMEFLOW_OXYGEN_ALGORITHM_VERSION
                and source_artifact is None
                and source_residue_identity is None
                and source_coordinate is None
                and source_coordinate_sha256 is None
                and raw["adjacency_decision"] == expected_decision
            )
            expected_oxygen = _oxygen_from_frame_evidence(current_frame, neighbor_frame)
        else:
            valid_provenance = False
            expected_oxygen = None
        if not valid_provenance:
            raise PaperExportError("binder repair provenance source identity drifted")
        assert expected_oxygen is not None
        if after_sha256 != _sha256_payload(list(_pdb_coordinate(expected_oxygen))):
            raise PaperExportError("binder repair output coordinate hash drifted")
        if source_artifact is not None:
            try:
                live_source = file_identity(Path(source_artifact.path))
            except (OSError, RuntimeError) as error:
                raise PaperExportError(
                    "binder repair source artifact cannot be authenticated"
                ) from error
            if (
                live_source.sha256 != source_artifact.sha256
                or live_source.size_bytes != source_artifact.size_bytes
            ):
                raise PaperExportError("binder repair source artifact identity drifted")
        expected_binding = _binder_repair_source_binding_sha256(
            chain=raw["chain"],
            residue=raw["residue"],
            coordinate_source=raw["coordinate_source"],
            source_identity=raw["source_identity"],
            algorithm_version=raw["algorithm_version"],
            source_artifact=source_artifact,
            source_residue_identity=source_residue_identity,
            source_coordinate=source_coordinate,
            source_coordinate_sha256=source_coordinate_sha256,
            current_frame=current_frame,
            neighbor_frame=neighbor_frame,
            adjacency_decision=raw["adjacency_decision"],
            before_coordinate=before_coordinate,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
        )
        if source_binding_sha256 != expected_binding:
            raise PaperExportError("binder repair source binding drifted")
        repair_identity = (raw["chain"], raw["residue"], "O")
        if repair_identity in repair_identities:
            raise PaperExportError("duplicate binder repaired atom identity")
        repair_identities.add(repair_identity)
        repair = BinderAtomRepair(
            chain=raw["chain"],
            residue=raw["residue"],
            atom="O",
            coordinate_source=raw["coordinate_source"],
            source_identity=raw["source_identity"],
            algorithm_version=raw["algorithm_version"],
            source_artifact=source_artifact,
            source_residue_identity=source_residue_identity,
            source_coordinate=source_coordinate,
            source_coordinate_sha256=source_coordinate_sha256,
            current_frame=current_frame,
            neighbor_frame=neighbor_frame,
            adjacency_decision=raw["adjacency_decision"],
            before_coordinate=before_coordinate,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
            source_binding_sha256=source_binding_sha256,
        )
        if repair.before_sha256 == repair.after_sha256:
            raise PaperExportError("binder repaired atom did not change coordinates")
        repairs.append(repair)
    normalization = BinderBackboneNormalization(
        version=BINDER_BACKBONE_NORMALIZATION_VERSION,
        before_coordinate_sha256=digest("before_coordinate_sha256"),
        after_coordinate_sha256=digest("after_coordinate_sha256"),
        before_topology_sha256=digest("before_topology_sha256"),
        after_topology_sha256=digest("after_topology_sha256"),
        roles_before_sha256=digest("roles_before_sha256"),
        roles_after_sha256=digest("roles_after_sha256"),
        sequences_before_sha256=digest("sequences_before_sha256"),
        sequences_after_sha256=digest("sequences_after_sha256"),
        finite_atoms_before_sha256=digest("finite_atoms_before_sha256"),
        finite_atoms_after_sha256=digest("finite_atoms_after_sha256"),
        repaired_atoms=tuple(repairs),
    )
    if (
        normalization.roles_before_sha256 != normalization.roles_after_sha256
        or normalization.sequences_before_sha256 != normalization.sequences_after_sha256
        or normalization.finite_atoms_before_sha256
        != normalization.finite_atoms_after_sha256
    ):
        raise PaperExportError("binder normalization changed roles or sequences")
    return normalization

def authenticate_binder_repaired_atoms_against_pdb(
    normalization: BinderBackboneNormalization, pdb: AuthenticatedPdb
) -> None:
    expected = {
        (repair.chain, repair.residue, repair.atom): repair.after_sha256
        for repair in normalization.repaired_atoms
    }
    if not expected:
        return
    if not isinstance(pdb, AuthenticatedPdb):
        raise PaperExportError("binder repaired-atom PDB is not authenticated")
    observed: dict[tuple[str, int, str], list[tuple[float, float, float]]] = {}
    residue_order: list[tuple[str, int]] = []
    for atom in pdb.atoms:
        identity = (atom.chain, atom.residue, atom.atom)
        coordinate = atom.coordinate
        residue_identity = identity[:2]
        if residue_identity not in residue_order:
            residue_order.append(residue_identity)
        observed.setdefault(identity, []).append(coordinate)
    for identity, after_sha256 in expected.items():
        coordinates = observed.get(identity, [])
        if len(coordinates) != 1:
            raise PaperExportError(
                "binder repaired-atom PDB identity is missing or duplicated"
            )
        if _sha256_payload(list(coordinates[0])) != after_sha256:
            raise PaperExportError("binder repair output coordinate drifted in PDB")
    for repair in normalization.repaired_atoms:
        if repair.coordinate_source != "frameflow_backbone_geometry":
            continue
        _authenticate_geometry_repair_against_exported_frames(
            repair, residue_order, observed
        )

def _authenticate_geometry_repair_against_exported_frames(
    repair: BinderAtomRepair,
    residue_order: Sequence[tuple[str, int]],
    observed: Mapping[tuple[str, int, str], Sequence[tuple[float, float, float]]],
) -> None:
    current_identity = (repair.chain, repair.residue)
    if current_identity not in residue_order or repair.current_frame is None:
        raise PaperExportError("binder exported geometry current frame is unavailable")
    current_position = residue_order.index(current_identity)

    def frame_matches(frame: BinderFrameEvidence, identity: tuple[str, int]) -> bool:
        if (frame.chain, frame.residue) != identity:
            return False
        for atom, coordinate in zip(("N", "CA", "C"), frame.coordinates, strict=True):
            exported = observed.get((*identity, atom), ())
            if len(exported) != 1 or exported[0] != _pdb_coordinate(coordinate):
                return False
        return True

    if not frame_matches(repair.current_frame, current_identity):
        raise PaperExportError("binder exported geometry current frame drifted")
    exported_neighbor = None
    if current_position + 1 < len(residue_order):
        candidate = residue_order[current_position + 1]
        if (
            candidate[0] == current_identity[0]
            and candidate[1] == current_identity[1] + 1
        ):
            exported_neighbor = candidate
    if exported_neighbor is None:
        if repair.neighbor_frame is not None:
            raise PaperExportError("binder exported geometry branch drifted")
    elif repair.neighbor_frame is None or not frame_matches(
        repair.neighbor_frame, exported_neighbor
    ):
        raise PaperExportError("binder exported geometry neighbor frame drifted")

def authenticate_binder_native_repairs_against_batch(
    normalization: BinderBackboneNormalization, batch: object
) -> None:
    from dive.benchmark.paper_batches import PaperBatch

    native_repairs = tuple(
        repair
        for repair in normalization.repaired_atoms
        if repair.coordinate_source == "authenticated_native_exact_identity"
    )
    if not native_repairs:
        return
    if type(batch) is not PaperBatch:
        raise PaperExportError("binder expected native batch is unavailable")
    try:
        live_structure = file_identity(Path(batch.structure.path))
    except BenchmarkContractError as error:
        raise PaperExportError(
            "binder expected native source artifact drifted"
        ) from error
    if live_structure != batch.structure or Path(
        batch.target.source_path
    ).resolve() != Path(batch.structure.path):
        raise PaperExportError("binder expected native source artifact drifted")
    expected = _native_rows_by_identity(batch)
    for repair in native_repairs:
        key = (repair.chain, repair.residue)
        native = expected.get(key)
        if repair.source_artifact != batch.structure or native is None:
            raise PaperExportError("binder native repair artifact or residue drifted")
        coordinates = _as_numpy(native["coords"], dtype=float)
        if coordinates.shape[0] <= 4 or not _atom_is_present(coordinates[4]):
            raise PaperExportError("binder expected native O is unavailable")
        oxygen = tuple(float(value) for value in coordinates[4])
        if (
            repair.source_residue_identity != f"{key[0]}:{key[1]}"
            or repair.source_coordinate != oxygen
            or repair.source_coordinate_sha256 != _sha256_payload(list(oxygen))
        ):
            raise PaperExportError("binder native repair source coordinate drifted")

def authenticate_binder_native_inputs(
    *,
    complex_pdb: AuthenticatedPdb,
    native_target: AuthenticatedPdb,
    expected_batch: PaperBatch,
) -> None:

    if not isinstance(complex_pdb, AuthenticatedPdb) or not isinstance(
        native_target, AuthenticatedPdb
    ):
        raise PaperExportError("binder native PDB inputs are not authenticated")
    if type(expected_batch) is not PaperBatch:
        raise PaperExportError("binder expected native batch is unavailable")
    try:
        live_structure = file_identity(Path(expected_batch.structure.path))
    except BenchmarkContractError as error:
        raise PaperExportError(
            "binder expected native source artifact drifted"
        ) from error
    if live_structure != expected_batch.structure or Path(
        expected_batch.target.source_path
    ).resolve() != Path(expected_batch.structure.path):
        raise PaperExportError("binder expected native source artifact drifted")

    batch_rows = _generated_chains_last(
        _residue_rows(_structure_from_batch(expected_batch), expected_batch)
    )
    fixed_rows = tuple(row for row in batch_rows if bool(row["fixed"]))
    _authenticate_exact_pdb_projection(
        expected_rows=fixed_rows,
        observed=complex_pdb,
        label="binder fixed",
        allow_unexpected_residues=True,
    )
    _authenticate_exact_pdb_projection(
        expected_rows=_native_target_rows(expected_batch),
        observed=native_target,
        label="binder native-target",
        allow_unexpected_residues=False,
    )

def _authenticate_exact_pdb_projection(
    *,
    expected_rows: Sequence[Mapping[str, object]],
    observed: AuthenticatedPdb,
    label: str,
    allow_unexpected_residues: bool,
) -> None:
    expected: dict[
        tuple[str, int],
        tuple[str, dict[str, tuple[float, float, float]]],
    ] = {}
    expected_order: list[tuple[str, int]] = []
    for row in expected_rows:
        identity = (str(row["chain"]), int(row["residue"]))
        if identity in expected:
            raise PaperExportError(f"{label} expected residue identity is ambiguous")
        coordinates = _as_numpy(row["coords"], dtype=float)
        atoms = {
            atom: _pdb_coordinate(coordinates[index])
            for index, atom in enumerate(_ATOM37[: coordinates.shape[0]])
            if _atom_is_present(coordinates[index])
        }
        expected[identity] = (str(row["resname"]), atoms)
        expected_order.append(identity)

    observed_records: dict[tuple[str, int], dict[str, object]] = {}
    observed_order: list[tuple[str, int]] = []
    for atom in observed.atoms:
        identity = (atom.chain, atom.residue)
        if identity not in observed_records:
            observed_order.append(identity)
            observed_records[identity] = {"resname": atom.resname, "atoms": {}}
        record = observed_records[identity]
        if record["resname"] != atom.resname:
            raise PaperExportError(f"{label} residue-name identity is ambiguous")
        atoms = record["atoms"]
        if not isinstance(atoms, dict) or atom.atom in atoms:
            raise PaperExportError(f"{label} atom identity is duplicated")
        atoms[atom.atom] = atom.coordinate

    expected_identities = set(expected)
    if allow_unexpected_residues:
        selected_order = [
            identity for identity in observed_order if identity in expected_identities
        ]
        if selected_order != expected_order:
            raise PaperExportError(f"{label} residue membership drifted")
    elif observed_order != expected_order:
        raise PaperExportError(f"{label} residue membership drifted")

    for identity in expected_order:
        record = observed_records.get(identity)
        if record is None:
            raise PaperExportError(f"{label} residue membership drifted")
        expected_resname, expected_atoms = expected[identity]
        if record["resname"] != expected_resname:
            raise PaperExportError(f"{label} residue name drifted")
        observed_atoms = record["atoms"]
        if not isinstance(observed_atoms, dict) or set(observed_atoms) != set(
            expected_atoms
        ):
            raise PaperExportError(f"{label} atom membership drifted")
        for atom_name, coordinate in expected_atoms.items():
            if observed_atoms[atom_name] != coordinate:
                raise PaperExportError(f"{label} atom coordinate drifted")

def authenticate_binder_geometry_repairs_against_generated_sample(
    normalization: BinderBackboneNormalization,
    generated_sample: object,
    cell: object,
    expected_batch: object,
    exported_pdb: AuthenticatedPdb,
) -> None:

    geometry_repairs = tuple(
        repair
        for repair in normalization.repaired_atoms
        if repair.coordinate_source == "frameflow_backbone_geometry"
    )
    if type(generated_sample) is not ArtifactIdentity:
        raise PaperExportError("binder generated sample artifact is unavailable")
    raw = _read_authenticated_generated_sample(generated_sample)
    try:
        import torch

        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except Exception as error:
        raise PaperExportError(
            "binder generated sample artifact cannot be loaded"
        ) from error
    required = {
        "arm",
        "cell_id",
        "chain_names",
        "family",
        "panel",
        "residue_pdb_idx",
        "sample",
        "schema",
        "seed",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PaperExportError("binder generated sample artifact schema drifted")
    if (
        payload.get("schema") != "dive.paper.generated_sample.v1"
        or payload.get("cell_id") != getattr(cell, "cell_id", None)
        or payload.get("family") != "binder"
        or payload.get("seed") != getattr(cell, "seed", None)
        or payload.get("arm") != getattr(cell, "arm", None)
        or payload.get("panel") != getattr(cell, "panel", None)
    ):
        raise PaperExportError("binder generated sample artifact identity drifted")
    sample = payload.get("sample")
    if not isinstance(sample, Mapping):
        raise PaperExportError("binder generated sample artifact schema drifted")
    rows = _generated_authority_rows(
        sample,
        chain_names=payload.get("chain_names"),
        residue_pdb_idx=payload.get("residue_pdb_idx"),
    )
    _authenticate_generated_roles_against_batch(
        rows,
        sample=sample,
        chain_names=payload.get("chain_names"),
        residue_pdb_idx=payload.get("residue_pdb_idx"),
        expected_batch=expected_batch,
    )
    _authenticate_generated_rows_against_pdb(rows, exported_pdb)
    by_tensor_index = {int(row["index"]): row for row in rows}
    for repair in geometry_repairs:
        current = by_tensor_index.get(
            -1 if repair.current_frame is None else repair.current_frame.tensor_index
        )
        if current is None or not bool(current["generated"]):
            raise PaperExportError(
                "binder generated artifact current frame is unavailable"
            )
        expected_current = _frame_evidence(current)
        if expected_current != repair.current_frame:
            raise PaperExportError("binder generated artifact current frame drifted")
        candidate = by_tensor_index.get(expected_current.tensor_index + 1)
        if (
            candidate is not None
            and _authenticated_next_frame(current, candidate) is None
        ):
            candidate = None
        expected_neighbor = None if candidate is None else _frame_evidence(candidate)
        if expected_neighbor != repair.neighbor_frame:
            raise PaperExportError("binder generated artifact neighbor frame drifted")
        decision, source_identity = _derive_geometry_branch(
            repair.chain,
            repair.residue,
            expected_current,
            expected_neighbor,
        )
        if (
            decision != repair.adjacency_decision
            or source_identity != repair.source_identity
        ):
            raise PaperExportError("binder generated artifact geometry branch drifted")

def _read_authenticated_generated_sample(identity: ArtifactIdentity) -> bytes:
    path = Path(identity.path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PaperExportError(
            "binder generated sample artifact cannot be opened"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PaperExportError("binder generated sample artifact is not regular")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            size += len(chunk)
        if digest.hexdigest() != identity.sha256 or size != identity.size_bytes:
            raise PaperExportError("binder generated sample artifact identity drifted")
        try:
            path_metadata = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise PaperExportError(
                "binder generated sample artifact path drifted"
            ) from error
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
        ):
            raise PaperExportError("binder generated sample artifact path drifted")
        return b"".join(chunks)
    finally:
        os.close(descriptor)

def _generated_authority_rows(
    sample: Mapping[str, object], *, chain_names: object, residue_pdb_idx: object
) -> tuple[dict[str, object], ...]:
    import numpy as np

    required = {
        "chain_index",
        "coors",
        "fixed_mask",
        "generated_mask",
        "mask",
        "residue_type",
    }
    if not required.issubset(sample):
        raise PaperExportError(
            "binder generated sample artifact lacks identity tensors"
        )
    mask = _drop_batch(_as_numpy(sample["mask"], dtype=bool))
    generated = _drop_batch(_as_numpy(sample["generated_mask"], dtype=bool))
    fixed = _drop_batch(_as_numpy(sample["fixed_mask"], dtype=bool))
    coordinates = _drop_batch(_as_numpy(sample["coors"]))
    chain_indices = _drop_batch(_as_numpy(sample["chain_index"], dtype=int))
    residue_ids = _drop_batch(_as_numpy(residue_pdb_idx, dtype=int))
    residue_types = _drop_batch(_as_numpy(sample["residue_type"], dtype=int))
    names = _chain_names(chain_names)
    lengths = {
        len(mask),
        len(generated),
        len(fixed),
        len(coordinates),
        len(chain_indices),
        len(residue_ids),
        len(residue_types),
    }
    if (
        len(lengths) != 1
        or bool((generated & fixed).any())
        or not np.array_equal(generated | fixed, mask)
    ):
        raise PaperExportError("binder generated sample artifact identity drifted")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for index, valid in enumerate(mask.tolist()):
        if not valid:
            continue
        chain_index = int(chain_indices[index])
        if chain_index < 0 or chain_index >= len(names):
            raise PaperExportError("binder generated sample artifact chain drifted")
        identity = (names[chain_index], int(residue_ids[index]))
        if identity in seen:
            raise PaperExportError(
                "binder generated sample artifact residue is ambiguous"
            )
        seen.add(identity)
        frame = np.asarray(coordinates[index], dtype=float)
        if frame.shape[0] < 3 or not np.isfinite(frame[:3]).all():
            raise PaperExportError("binder generated sample artifact frame is invalid")
        rows.append(
            {
                "chain": identity[0],
                "chain_index": chain_index,
                "coords": frame,
                "fixed": bool(fixed[index]),
                "generated": bool(generated[index]),
                "index": index,
                "residue": identity[1],
                "residue_type": int(residue_types[index]),
            }
        )
    return tuple(rows)

def _authenticate_generated_roles_against_batch(
    rows: Sequence[Mapping[str, object]],
    *,
    sample: Mapping[str, object],
    chain_names: object,
    residue_pdb_idx: object,
    expected_batch: object,
) -> None:
    import numpy as np

    from dive.benchmark.paper_batches import PaperBatch

    if type(expected_batch) is not PaperBatch:
        raise PaperExportError("binder expected generated batch is unavailable")
    tensors = expected_batch.tensors
    batch_chain_index = tensors.get("chain_index", tensors.get("chains"))
    if batch_chain_index is None:
        raise PaperExportError("binder expected generated batch lacks chain identity")
    comparisons = (
        (sample["mask"], tensors.get("mask")),
        (sample["generated_mask"], tensors.get("generated_mask")),
        (sample["fixed_mask"], tensors.get("fixed_mask")),
        (sample["chain_index"], batch_chain_index),
        (residue_pdb_idx, tensors.get("residue_pdb_idx")),
    )
    for observed, expected in comparisons:
        if expected is None or not np.array_equal(
            _as_numpy(observed), _as_numpy(expected)
        ):
            raise PaperExportError("binder generated artifact frozen role drifted")
    if _chain_names(chain_names) != _chain_names(tensors.get("chain_names")):
        raise PaperExportError("binder generated artifact frozen role drifted")
    batch_residue_types = _drop_batch(_as_numpy(tensors.get("residue_type"), dtype=int))
    for row in rows:
        index = int(row["index"])
        if bool(row["fixed"]) and int(row["residue_type"]) != int(
            batch_residue_types[index]
        ):
            raise PaperExportError("binder generated artifact frozen sequence drifted")

def _authenticate_generated_rows_against_pdb(
    rows: Sequence[Mapping[str, object]], pdb: AuthenticatedPdb
) -> None:
    if not isinstance(pdb, AuthenticatedPdb):
        raise PaperExportError("binder generated export PDB is not authenticated")
    observed: dict[tuple[str, int], dict[str, object]] = {}
    for pdb_atom in pdb.atoms:
        identity = (pdb_atom.chain, pdb_atom.residue)
        record = observed.setdefault(identity, {"atoms": {}, "resnames": set()})
        record["resnames"].add(pdb_atom.resname)
        record["atoms"].setdefault(pdb_atom.atom, []).append(pdb_atom.coordinate)
    expected_identities = {(str(row["chain"]), int(row["residue"])) for row in rows}
    if set(observed) != expected_identities:
        raise PaperExportError("binder generated export residue membership drifted")
    for row in rows:
        identity = (str(row["chain"]), int(row["residue"]))
        record = observed[identity]
        if not bool(row["generated"]):
            continue
        expected_resname = _resname(int(row["residue_type"]))
        if record["resnames"] != {expected_resname}:
            raise PaperExportError("binder generated export residue name drifted")
        coordinates = _as_numpy(row["coords"], dtype=float)
        required_atoms = [0, 1, 2]
        if coordinates.shape[0] > 4 and _atom_is_present(coordinates[4]):
            required_atoms.append(4)
        for atom_index in required_atoms:
            atom = _ATOM37[atom_index]
            exported = record["atoms"].get(atom, [])
            if len(exported) != 1 or exported[0] != _pdb_coordinate(
                coordinates[atom_index]
            ):
                raise PaperExportError("binder generated export frame drifted")

def _binder_source_artifact_from_mapping(
    value: object,
) -> ArtifactIdentity | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise PaperExportError("binder repair source artifact schema drifted")
    path, sha256, size_bytes = value["path"], value["sha256"], value["size_bytes"]
    if (
        type(path) is not str
        or not path
        or type(sha256) is not str
        or len(sha256) != 64
        or set(sha256) - set("0123456789abcdef")
        or type(size_bytes) is not int
        or size_bytes < 0
    ):
        raise PaperExportError("binder repair source artifact identity drifted")
    return ArtifactIdentity(path, sha256, size_bytes)

def _binder_coordinate_from_mapping(
    value: object, *, label: str, allow_none: bool
) -> tuple[float, float, float] | None:
    import math

    if value is None and allow_none:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(item) not in {int, float} for item in value)
    ):
        raise PaperExportError(f"binder repair {label} drifted")
    coordinate = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in coordinate):
        raise PaperExportError(f"binder repair {label} is non-finite")
    return coordinate

def _binder_frame_evidence_from_mapping(value: object) -> BinderFrameEvidence | None:
    if value is None:
        return None
    expected = {
        "chain",
        "chain_index",
        "coordinate_sha256",
        "coordinates",
        "residue",
        "tensor_index",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PaperExportError("binder repair frame evidence schema drifted")
    if (
        type(value["chain"]) is not str
        or not value["chain"]
        or type(value["residue"]) is not int
        or type(value["tensor_index"]) is not int
        or type(value["chain_index"]) is not int
        or not isinstance(value["coordinates"], (list, tuple))
        or len(value["coordinates"]) != 3
    ):
        raise PaperExportError("binder repair frame identity drifted")
    coordinates = tuple(
        _binder_coordinate_from_mapping(
            item, label="frame coordinate", allow_none=False
        )
        for item in value["coordinates"]
    )
    assert all(item is not None for item in coordinates)
    typed_coordinates = tuple(item for item in coordinates if item is not None)
    coordinate_sha256 = value["coordinate_sha256"]
    if (
        type(coordinate_sha256) is not str
        or len(coordinate_sha256) != 64
        or set(coordinate_sha256) - set("0123456789abcdef")
        or coordinate_sha256
        != _sha256_payload([list(item) for item in typed_coordinates])
    ):
        raise PaperExportError("binder repair frame coordinate hash drifted")
    return BinderFrameEvidence(
        chain=value["chain"],
        residue=value["residue"],
        tensor_index=value["tensor_index"],
        chain_index=value["chain_index"],
        coordinates=typed_coordinates,
        coordinate_sha256=coordinate_sha256,
    )

def _frame_evidence(row: Mapping[str, object]) -> BinderFrameEvidence:
    import numpy as np

    coords = np.asarray(row["coords"], dtype=float)
    backbone = tuple(tuple(float(value) for value in xyz) for xyz in coords[:3])
    if len(backbone) != 3:
        raise PaperExportError("binder repair frame lacks N/CA/C inputs")
    return BinderFrameEvidence(
        chain=str(row["chain"]),
        residue=int(row["residue"]),
        tensor_index=int(row["index"]),
        chain_index=int(row["chain_index"]),
        coordinates=backbone,
        coordinate_sha256=_sha256_payload([list(item) for item in backbone]),
    )

def _derive_geometry_branch(
    chain: object,
    residue: object,
    current: BinderFrameEvidence | None,
    neighbor: BinderFrameEvidence | None,
) -> tuple[str, str]:
    if current is None or current.chain != chain or current.residue != residue:
        raise PaperExportError("binder repair current frame identity drifted")
    if neighbor is None:
        return "terminal_or_unavailable_frame", FRAMEFLOW_TERMINAL_SOURCE_IDENTITY
    if (
        neighbor.chain != current.chain
        or neighbor.chain_index != current.chain_index
        or neighbor.tensor_index != current.tensor_index + 1
        or neighbor.residue != current.residue + 1
    ):
        raise PaperExportError("binder repair neighbor frame is not adjacent")
    return "authenticated_next_frame", FRAMEFLOW_NEXT_SOURCE_IDENTITY

def _oxygen_from_frame_evidence(
    current: BinderFrameEvidence | None,
    neighbor: BinderFrameEvidence | None,
):
    if current is None:
        raise PaperExportError("binder repair current frame is unavailable")
    return _reconstruct_generated_oxygen(
        {"coords": current.coordinates},
        None if neighbor is None else {"coords": neighbor.coordinates},
    )

def _pdb_coordinate(value: object) -> tuple[float, float, float]:
    import numpy as np

    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise PaperExportError("binder repair output coordinate drifted")
    coordinate = tuple(float(item) for item in array)
    return tuple(float(f"{item:.3f}") for item in coordinate)

def _binder_repair_source_binding_sha256(
    *,
    chain: str,
    residue: int,
    coordinate_source: str,
    source_identity: str,
    algorithm_version: str,
    source_artifact: ArtifactIdentity | None,
    source_residue_identity: str | None,
    source_coordinate: tuple[float, float, float] | None,
    source_coordinate_sha256: str | None,
    current_frame: BinderFrameEvidence | None,
    neighbor_frame: BinderFrameEvidence | None,
    adjacency_decision: str,
    before_coordinate: tuple[float, float, float],
    before_sha256: str,
    after_sha256: str,
) -> str:
    return _sha256_payload(
        {
            "adjacency_decision": adjacency_decision,
            "after_sha256": after_sha256,
            "algorithm_version": algorithm_version,
            "before_coordinate": list(before_coordinate),
            "before_sha256": before_sha256,
            "chain": chain,
            "coordinate_source": coordinate_source,
            "current_frame": None
            if current_frame is None
            else current_frame.to_mapping(),
            "neighbor_frame": (
                None if neighbor_frame is None else neighbor_frame.to_mapping()
            ),
            "residue": residue,
            "source_artifact": (
                None
                if source_artifact is None
                else {
                    "path": source_artifact.path,
                    "sha256": source_artifact.sha256,
                    "size_bytes": source_artifact.size_bytes,
                }
            ),
            "source_coordinate": (
                None if source_coordinate is None else list(source_coordinate)
            ),
            "source_coordinate_sha256": source_coordinate_sha256,
            "source_identity": source_identity,
            "source_residue_identity": source_residue_identity,
        }
    )

@dataclass(frozen=True, slots=True)
class AmeLigandTransport:
    version: str
    canonical_component_token: str
    ligand_identity: str
    ligand_smiles: str
    original_representation: str
    original_hetero: bool
    native_hetatm_claimed: bool
    evaluator_ligand_chain: str
    sequence: str
    atom_count: int
    atom_membership_sha256: str
    coordinate_sha256: str
    normalized_atom_membership_sha256: str
    normalized_coordinate_sha256: str
    clash_membership_before_sha256: str
    clash_membership_after_sha256: str
    clash_coordinate_before_sha256: str
    clash_coordinate_after_sha256: str
    clash_selector: Mapping[str, str]

    def to_mapping(self) -> dict[str, object]:
        return {
            "atom_count": self.atom_count,
            "atom_membership_sha256": self.atom_membership_sha256,
            "canonical_component_token": self.canonical_component_token,
            "clash_coordinate_after_sha256": self.clash_coordinate_after_sha256,
            "clash_coordinate_before_sha256": self.clash_coordinate_before_sha256,
            "clash_membership_after_sha256": self.clash_membership_after_sha256,
            "clash_membership_before_sha256": self.clash_membership_before_sha256,
            "clash_selector": dict(self.clash_selector),
            "coordinate_sha256": self.coordinate_sha256,
            "evaluator_ligand_chain": self.evaluator_ligand_chain,
            "ligand_identity": self.ligand_identity,
            "ligand_smiles": self.ligand_smiles,
            "native_hetatm_claimed": self.native_hetatm_claimed,
            "normalized_atom_membership_sha256": (
                self.normalized_atom_membership_sha256
            ),
            "normalized_coordinate_sha256": self.normalized_coordinate_sha256,
            "original_hetero": self.original_hetero,
            "original_representation": self.original_representation,
            "sequence": self.sequence,
            "version": self.version,
        }

def ame_ligand_transport_from_mapping(value: object) -> AmeLigandTransport:
    if not isinstance(value, Mapping):
        raise PaperExportError("AME ligand transport must be a mapping")
    expected = {
        "atom_count",
        "atom_membership_sha256",
        "canonical_component_token",
        "clash_coordinate_after_sha256",
        "clash_coordinate_before_sha256",
        "clash_membership_after_sha256",
        "clash_membership_before_sha256",
        "clash_selector",
        "coordinate_sha256",
        "evaluator_ligand_chain",
        "ligand_identity",
        "ligand_smiles",
        "native_hetatm_claimed",
        "normalized_atom_membership_sha256",
        "normalized_coordinate_sha256",
        "original_hetero",
        "original_representation",
        "sequence",
        "version",
    }
    if set(value) != expected:
        raise PaperExportError("AME ligand transport schema drifted")
    if value["version"] != AME_LIGAND_TRANSPORT_VERSION:
        raise PaperExportError("AME ligand transport version drifted")

    def digest(key: str) -> str:
        item = value[key]
        if (
            type(item) is not str
            or len(item) != 64
            or set(item) - set("0123456789abcdef")
        ):
            raise PaperExportError(f"AME ligand transport {key} is invalid")
        return item

    def nonempty(key: str) -> str:
        item = value[key]
        if type(item) is not str or not item or item.strip() != item:
            raise PaperExportError(f"AME ligand transport {key} is invalid")
        return item

    selector = value["clash_selector"]
    if not isinstance(selector, Mapping) or set(selector) != {
        "chain_id",
        "component_token",
        "kind",
    }:
        raise PaperExportError("AME ligand clash selector drifted")
    kind = selector["kind"]
    chain_id = selector["chain_id"]
    component_token = selector["component_token"]
    if (
        kind != "canonical_chain"
        or type(chain_id) is not str
        or chain_id != AME_EVALUATOR_LIGAND_CHAIN
        or type(component_token) is not str
        or not component_token
    ):
        raise PaperExportError("AME ligand clash selector drifted")
    representation = value["original_representation"]
    if representation not in {"polymer_peptide", "non_polymer"}:
        raise PaperExportError("AME ligand representation drifted")
    if type(value["original_hetero"]) is not bool:
        raise PaperExportError("AME original hetero flag is invalid")
    if type(value["native_hetatm_claimed"]) is not bool:
        raise PaperExportError("AME native HETATM claim is invalid")
    if representation == "polymer_peptide" and value["native_hetatm_claimed"]:
        raise PaperExportError("polymer peptide cannot be claimed as native HETATM")
    if type(value["atom_count"]) is not int or value["atom_count"] <= 0:
        raise PaperExportError("AME ligand atom count is invalid")
    if type(value["sequence"]) is not str:
        raise PaperExportError("AME ligand sequence is invalid")
    membership = digest("atom_membership_sha256")
    coordinates = digest("coordinate_sha256")
    normalized_membership = digest("normalized_atom_membership_sha256")
    normalized_coordinates = digest("normalized_coordinate_sha256")
    clash_membership_before = digest("clash_membership_before_sha256")
    clash_membership_after = digest("clash_membership_after_sha256")
    clash_coordinate_before = digest("clash_coordinate_before_sha256")
    clash_coordinate_after = digest("clash_coordinate_after_sha256")
    if membership != normalized_membership or coordinates != normalized_coordinates:
        raise PaperExportError("AME ligand membership or coordinates changed")
    if (
        clash_membership_before != membership
        or clash_coordinate_before != coordinates
        or clash_coordinate_after != coordinates
    ):
        raise PaperExportError("AME ligand membership or coordinates changed")
    if nonempty("evaluator_ligand_chain") != AME_EVALUATOR_LIGAND_CHAIN:
        raise PaperExportError("AME ligand clash selector drifted")
    return AmeLigandTransport(
        version=AME_LIGAND_TRANSPORT_VERSION,
        canonical_component_token=nonempty("canonical_component_token"),
        ligand_identity=nonempty("ligand_identity"),
        ligand_smiles=nonempty("ligand_smiles"),
        original_representation=representation,
        original_hetero=value["original_hetero"],
        native_hetatm_claimed=value["native_hetatm_claimed"],
        evaluator_ligand_chain=AME_EVALUATOR_LIGAND_CHAIN,
        sequence=value["sequence"],
        atom_count=value["atom_count"],
        atom_membership_sha256=membership,
        coordinate_sha256=coordinates,
        normalized_atom_membership_sha256=normalized_membership,
        normalized_coordinate_sha256=normalized_coordinates,
        clash_membership_before_sha256=clash_membership_before,
        clash_membership_after_sha256=clash_membership_after,
        clash_coordinate_before_sha256=clash_coordinate_before,
        clash_coordinate_after_sha256=clash_coordinate_after,
        clash_selector=MappingProxyType(
            {
                "kind": kind,
                "chain_id": chain_id,
                "component_token": component_token,
            }
        ),
    )

@dataclass(frozen=True, slots=True)
class PaperExportResult:
    family: str
    complex_pdb: ArtifactIdentity
    manifest: ArtifactIdentity
    native_target: ArtifactIdentity | None = None
    motif_positions: ArtifactIdentity | None = None
    native_motif: ArtifactIdentity | None = None
    predicted_atoms: tuple[AtomRecord, ...] | None = None
    native_atoms: tuple[AtomRecord, ...] | None = None
    antibody_roles: FrozenAntibodyRoles | None = None
    binder_backbone_normalization: BinderBackboneNormalization | None = None
    ame_ligand_transport: AmeLigandTransport | None = None

_ATOM37 = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)
_RESNAMES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)

def export_generated_sample(
    *,
    sample: Mapping[str, object],
    batch: PaperBatch,
    output_dir: Path,
) -> PaperExportResult:
    destination = _safe_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    residues = _residue_rows(sample, batch)
    family = batch.target.family
    binder_backbone_normalization = None
    if family == "binder":
        residues, binder_backbone_normalization = _normalize_binder_backbone(
            residues, batch
        )
    _assert_finite_and_backbone(residues)
    stem = _stem(batch)
    artifacts: list[tuple[str, ArtifactIdentity]] = []
    native_target = None
    motif_positions = None
    native_motif = None
    predicted_atoms = None
    native_atoms = None
    antibody_roles = None
    ame_ligand_transport = None
    if family == "ame":
        complex_pdb, protein_ids, ame_ligand_transport = _export_ame_complex(
            residues, batch, destination / f"{stem}.pdb"
        )
        motif_positions = _write_motif_positions(
            residues,
            batch,
            destination / f"{stem}.motif.json",
            protein_ids=protein_ids,
        )
        native_motif = _write_native_motif_pdb(
            residues,
            batch,
            destination / f"{stem}.native_motif.pdb",
            protein_ids=protein_ids,
        )
        artifacts.append(("complex_pdb", complex_pdb))
        artifacts.append(("motif_positions", motif_positions))
        artifacts.append(("native_motif", native_motif))
    elif family == "binder":
        ordered = _generated_chains_last(residues)
        complex_pdb = _write_pdb(ordered, destination / f"{stem}.pdb")
        native_rows = _native_target_rows(batch)
        native_target = _write_pdb(
            native_rows, destination / f"{stem}.native_target.pdb"
        )
        artifacts.append(("complex_pdb", complex_pdb))
        artifacts.append(("native_target", native_target))
    elif family == "antibody":
        complex_pdb = _write_pdb(residues, destination / f"{stem}.pdb")
        artifacts.append(("complex_pdb", complex_pdb))
        source = Path(batch.target.source_path)
        mapping = _label_to_auth_mapping(source)
        antibody_roles = _frozen_antibody_roles(
            batch.target.role_payload, residues, mapping
        )
        native_atoms = _native_atom_records(source, mapping)
        predicted_atoms = _predicted_atom_records(residues, mapping)
    else:
        raise PaperExportError(f"unsupported family {family}")
    manifest = _write_manifest(
        destination / f"{stem}.manifest.json",
        family=family,
        example_id=batch.target.example_id,
        artifacts=artifacts,
        binder_backbone_normalization=binder_backbone_normalization,
        ame_ligand_transport=ame_ligand_transport,
    )
    return PaperExportResult(
        family=family,
        complex_pdb=complex_pdb,
        manifest=manifest,
        native_target=native_target,
        motif_positions=motif_positions,
        native_motif=native_motif,
        predicted_atoms=predicted_atoms,
        native_atoms=native_atoms,
        antibody_roles=antibody_roles,
        binder_backbone_normalization=binder_backbone_normalization,
        ame_ligand_transport=ame_ligand_transport,
    )

def _safe_output_dir(path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or any(
        parent.is_symlink() for parent in candidate.parents if parent.exists()
    ):
        raise PaperExportError("symlink traversal is forbidden")
    return _refuse_denied(candidate)

def _refuse_denied(path: Path) -> Path:
    resolved = Path(path).resolve(strict=False)
    for denied in DENIED_PREFIXES:
        prefix = Path(denied).resolve(strict=False)
        if resolved == prefix or prefix in resolved.parents:
            raise PaperBaselineError(f"root is denied: {resolved}")
    return resolved

def _stem(batch: PaperBatch) -> str:
    raw = f"{batch.target.family}-{batch.target.example_id}"
    return "".join(char if char.isalnum() or char in "-._" else "_" for char in raw)

def _as_numpy(value, *, dtype=None):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype)
    return array

def _drop_batch(array):
    if getattr(array, "ndim", 0) >= 1 and array.shape[0] == 1:
        return array[0]
    return array

def _chain_names(value: object) -> tuple[str, ...]:
    if (
        isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)) or not value:
        raise PaperExportError("generated artifact lacks chain-name identity")
    return tuple(map(str, value))

def _residue_rows(sample: Mapping[str, object], batch: PaperBatch) -> tuple[dict, ...]:
    import numpy as np

    mask = _drop_batch(_as_numpy(sample["mask"], dtype=bool))
    generated = _drop_batch(_as_numpy(sample["generated_mask"], dtype=bool))
    fixed = _drop_batch(_as_numpy(sample["fixed_mask"], dtype=bool))
    if bool((generated & fixed).any()) or not np.array_equal(generated | fixed, mask):
        raise PaperExportError("sample role masks do not partition valid residues")
    coors = _drop_batch(_as_numpy(sample["coors"]))
    residue_type = _drop_batch(_as_numpy(sample["residue_type"], dtype=int))
    chain_index = _drop_batch(
        _as_numpy(sample.get("chain_index", batch.tensors["chains"]), dtype=int)
    )
    residue_index = _drop_batch(_as_numpy(batch.tensors["residue_pdb_idx"], dtype=int))
    names = _chain_names(batch.tensors.get("chain_names"))
    rows = []
    generated_chains = set()
    for index, valid in enumerate(mask.tolist()):
        if not valid:
            continue
        chain_i = int(chain_index[index])
        if chain_i < 0 or chain_i >= len(names):
            raise PaperExportError("chain index is outside chain-name identity")
        chain = names[chain_i]
        is_generated = bool(generated[index])
        if is_generated:
            generated_chains.add(chain)
        rows.append(
            {
                "index": index,
                "chain": chain,
                "chain_index": chain_i,
                "residue": int(residue_index[index]),
                "resname": _resname(int(residue_type[index])),
                "coords": coors[index],
                "generated": is_generated,
                "fixed": bool(fixed[index]),
            }
        )
    for row in rows:
        row["generated_chain"] = row["chain"] in generated_chains
    if not rows:
        raise PaperExportError("sample contains no valid residues")
    return tuple(rows)

def _resname(index: int) -> str:
    if 0 <= index < len(_RESNAMES):
        return _RESNAMES[index]
    return "UNK"

def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes({"value": payload})).hexdigest()

def _coordinate_payload(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    import numpy as np

    return [
        {
            "chain": str(row["chain"]),
            "coordinates": np.asarray(row["coords"], dtype=float).tolist(),
            "index": int(row["index"]),
            "residue": int(row["residue"]),
        }
        for row in rows
    ]

def _atom_is_present(xyz) -> bool:
    import numpy as np

    value = np.asarray(xyz, dtype=float)
    return bool(np.isfinite(value).all() and float(np.abs(value).sum()) > 1e-7)

def _topology_payload(rows: Sequence[Mapping[str, object]]) -> list[list[object]]:
    import numpy as np

    present: list[list[object]] = []
    for row in rows:
        coords = np.asarray(row["coords"], dtype=float)
        for atom_index, atom_name in enumerate(_ATOM37[: coords.shape[0]]):
            if _atom_is_present(coords[atom_index]):
                present.append([str(row["chain"]), int(row["residue"]), atom_name])
    return present

def _preexisting_finite_atom_payload(
    reference: Sequence[Mapping[str, object]],
    observed: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    import numpy as np

    if len(reference) != len(observed):
        raise PaperExportError("binder normalization changed residue count")
    payload = []
    for before, after in zip(reference, observed, strict=True):
        before_identity = (before["chain"], before["residue"], before["resname"])
        after_identity = (after["chain"], after["residue"], after["resname"])
        if before_identity != after_identity:
            raise PaperExportError("binder normalization changed residue identity")
        before_coords = np.asarray(before["coords"])
        after_coords = np.asarray(after["coords"])
        if before_coords.shape != after_coords.shape:
            raise PaperExportError("binder normalization changed coordinate shape")
        for atom_index, before_xyz in enumerate(before_coords):
            if not _atom_is_present(before_xyz):
                continue
            after_xyz = after_coords[atom_index]
            if not np.array_equal(before_xyz, after_xyz):
                raise PaperExportError("binder normalization changed a finite atom")
            payload.append(
                {
                    "atom_index": atom_index,
                    "chain": str(before["chain"]),
                    "coordinates": np.asarray(after_xyz, dtype=float).tolist(),
                    "residue": int(before["residue"]),
                }
            )
    return payload

def _roles_payload(
    rows: Sequence[Mapping[str, object]], batch: PaperBatch
) -> dict[str, object]:
    return {
        "residue_roles": [
            {
                "chain": str(row["chain"]),
                "fixed": bool(row["fixed"]),
                "generated": bool(row["generated"]),
                "generated_chain": bool(row["generated_chain"]),
                "residue": int(row["residue"]),
            }
            for row in rows
        ],
        "role_payload": {
            key: str(batch.target.role_payload[key])
            for key in ("context", "generated", "target")
        },
    }

def _sequences_payload(rows: Sequence[Mapping[str, object]]) -> list[list[object]]:
    return [
        [str(row["chain"]), int(row["residue"]), str(row["resname"])] for row in rows
    ]

def _copy_rows(rows: Sequence[Mapping[str, object]]) -> tuple[dict, ...]:
    import numpy as np

    return tuple({**row, "coords": np.asarray(row["coords"]).copy()} for row in rows)

def _unit_vector(vector, *, label: str):
    import numpy as np

    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(norm) or norm <= 1e-7:
        raise PaperExportError(f"binder oxygen geometry is degenerate: {label}")
    return value / norm

def _reconstruct_generated_oxygen(
    row: Mapping[str, object], next_row: Mapping[str, object] | None
):
    import numpy as np

    coords = np.asarray(row["coords"], dtype=float)
    n, ca, carbonyl = coords[0], coords[1], coords[2]
    ca_to_carbonyl = _unit_vector(carbonyl - ca, label="current CA-to-C vector")
    if next_row is None:
        second = _unit_vector(n - ca, label="terminal CA-to-N vector")
    else:
        next_coords = np.asarray(next_row["coords"], dtype=float)
        second = _unit_vector(
            carbonyl - next_coords[0], label="next-N-to-current-C vector"
        )
    direction = _unit_vector(
        ca_to_carbonyl + second, label="carbonyl-to-oxygen bisector"
    )
    oxygen = carbonyl + direction * 1.23
    if not np.isfinite(oxygen).all():
        raise PaperExportError("binder oxygen geometry produced non-finite coordinates")
    return oxygen

def _authenticated_next_frame(
    row: Mapping[str, object], next_row: Mapping[str, object] | None
) -> Mapping[str, object] | None:
    if next_row is None:
        return None
    if (
        str(next_row["chain"]) != str(row["chain"])
        or int(next_row["chain_index"]) != int(row["chain_index"])
        or int(next_row["index"]) != int(row["index"]) + 1
        or int(next_row["residue"]) != int(row["residue"]) + 1
    ):
        return None
    return next_row

def _native_rows_by_identity(batch: PaperBatch) -> dict[tuple[str, int], dict]:
    rows = _residue_rows(_structure_from_batch(batch), batch)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["chain"]), int(row["residue"])), []).append(row)
    ambiguous = {key for key, values in grouped.items() if len(values) != 1}
    if ambiguous:
        rendered = ", ".join(
            f"{chain}:{residue}" for chain, residue in sorted(ambiguous)
        )
        raise PaperExportError(f"ambiguous native residue identity: {rendered}")
    return {key: values[0] for key, values in grouped.items()}

def _normalize_binder_backbone(
    rows: Sequence[Mapping[str, object]], batch: PaperBatch
) -> tuple[tuple[dict, ...], BinderBackboneNormalization]:
    import numpy as np

    before = _copy_rows(rows)
    after = _copy_rows(rows)
    _assert_finite_and_backbone(before)
    before_coordinates = _sha256_payload(_coordinate_payload(before))
    before_topology = _sha256_payload(_topology_payload(before))
    roles_before = _sha256_payload(_roles_payload(before, batch))
    sequences_before = _sha256_payload(_sequences_payload(before))
    finite_atoms_before = _sha256_payload(
        _preexisting_finite_atom_payload(before, before)
    )
    missing_fixed = []
    for row in after:
        coords = np.asarray(row["coords"])
        if coords.shape[0] <= 4:
            raise PaperExportError("binder atom37 coordinates lack the O index")
        if not row["generated"] and not _atom_is_present(coords[4]):
            missing_fixed.append(row)
    native_by_identity = _native_rows_by_identity(batch) if missing_fixed else {}
    repairs: list[BinderAtomRepair] = []
    chain_rows: dict[str, list[dict]] = {}
    for row in after:
        chain_rows.setdefault(str(row["chain"]), []).append(row)
    next_by_index: dict[int, dict | None] = {}
    for chain in chain_rows.values():
        for position, row in enumerate(chain):
            next_by_index[int(row["index"])] = (
                chain[position + 1] if position + 1 < len(chain) else None
            )
    for row in after:
        coords = np.asarray(row["coords"])
        if coords.shape[0] <= 4:
            raise PaperExportError("binder atom37 coordinates lack the O index")
        if _atom_is_present(coords[4]):
            continue
        before_atom = coords[4].copy()
        before_coordinate = tuple(float(value) for value in before_atom)
        before_sha256 = _sha256_payload(list(before_coordinate))
        key = (str(row["chain"]), int(row["residue"]))
        if row["generated"]:
            next_frame = _authenticated_next_frame(
                row, next_by_index[int(row["index"])]
            )
            oxygen = _reconstruct_generated_oxygen(row, next_frame)
            coordinate_source = "frameflow_backbone_geometry"
            source_identity = (
                FRAMEFLOW_NEXT_SOURCE_IDENTITY
                if next_frame is not None
                else FRAMEFLOW_TERMINAL_SOURCE_IDENTITY
            )
            algorithm_version = FRAMEFLOW_OXYGEN_ALGORITHM_VERSION
            source_artifact = None
            source_residue_identity = None
            source_coordinate = None
            source_coordinate_sha256 = None
            current_frame = _frame_evidence(row)
            neighbor_frame = None if next_frame is None else _frame_evidence(next_frame)
            adjacency_decision = (
                "authenticated_next_frame"
                if next_frame is not None
                else "terminal_or_unavailable_frame"
            )
        else:
            native = native_by_identity.get(key)
            if native is None:
                raise PaperExportError(
                    f"missing native residue identity for binder O: {key[0]}:{key[1]}"
                )
            native_coords = np.asarray(native["coords"], dtype=float)
            if native_coords.shape[0] <= 4 or not _atom_is_present(native_coords[4]):
                raise PaperExportError(
                    f"authenticated native O is unavailable for {key[0]}:{key[1]}"
                )
            oxygen = native_coords[4].copy()
            coordinate_source = "authenticated_native_exact_identity"
            source_identity = NATIVE_OXYGEN_SOURCE_IDENTITY
            algorithm_version = NATIVE_OXYGEN_ALGORITHM_VERSION
            source_artifact = batch.structure
            source_residue_identity = f"{key[0]}:{key[1]}"
            source_coordinate = tuple(float(value) for value in oxygen)
            source_coordinate_sha256 = _sha256_payload(list(source_coordinate))
            current_frame = None
            neighbor_frame = None
            adjacency_decision = "not_applicable_native"
        coords[4] = oxygen
        after_sha256 = _sha256_payload(list(_pdb_coordinate(coords[4])))
        source_binding_sha256 = _binder_repair_source_binding_sha256(
            chain=key[0],
            residue=key[1],
            coordinate_source=coordinate_source,
            source_identity=source_identity,
            algorithm_version=algorithm_version,
            source_artifact=source_artifact,
            source_residue_identity=source_residue_identity,
            source_coordinate=source_coordinate,
            source_coordinate_sha256=source_coordinate_sha256,
            current_frame=current_frame,
            neighbor_frame=neighbor_frame,
            adjacency_decision=adjacency_decision,
            before_coordinate=before_coordinate,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
        )
        repairs.append(
            BinderAtomRepair(
                chain=key[0],
                residue=key[1],
                atom="O",
                coordinate_source=coordinate_source,
                source_identity=source_identity,
                algorithm_version=algorithm_version,
                source_artifact=source_artifact,
                source_residue_identity=source_residue_identity,
                source_coordinate=source_coordinate,
                source_coordinate_sha256=source_coordinate_sha256,
                current_frame=current_frame,
                neighbor_frame=neighbor_frame,
                adjacency_decision=adjacency_decision,
                before_coordinate=before_coordinate,
                before_sha256=before_sha256,
                after_sha256=after_sha256,
                source_binding_sha256=source_binding_sha256,
            )
        )
    roles_after = _sha256_payload(_roles_payload(after, batch))
    sequences_after = _sha256_payload(_sequences_payload(after))
    if roles_after != roles_before or sequences_after != sequences_before:
        raise PaperExportError("binder normalization changed roles or sequences")
    finite_atoms_after = _sha256_payload(
        _preexisting_finite_atom_payload(before, after)
    )
    if finite_atoms_after != finite_atoms_before:
        raise PaperExportError("binder normalization changed a finite atom")
    normalization = BinderBackboneNormalization(
        version=BINDER_BACKBONE_NORMALIZATION_VERSION,
        before_coordinate_sha256=before_coordinates,
        after_coordinate_sha256=_sha256_payload(_coordinate_payload(after)),
        before_topology_sha256=before_topology,
        after_topology_sha256=_sha256_payload(_topology_payload(after)),
        roles_before_sha256=roles_before,
        roles_after_sha256=roles_after,
        sequences_before_sha256=sequences_before,
        sequences_after_sha256=sequences_after,
        finite_atoms_before_sha256=finite_atoms_before,
        finite_atoms_after_sha256=finite_atoms_after,
        repaired_atoms=tuple(repairs),
    )
    return after, normalization

def _assert_finite_and_backbone(rows: Sequence[Mapping[str, object]]) -> None:
    import numpy as np

    for row in rows:
        coords = np.asarray(row["coords"], dtype=float)
        if not np.isfinite(coords).all():
            raise PaperExportError("coordinates must be finite")
        if row["generated"]:

            backbone = coords[:3]
            if backbone.shape[0] < 3 or not bool(
                (np.abs(backbone).sum(axis=-1) > 1e-7).all()
            ):
                raise PaperExportError("generated backbone is incomplete")

def _structure_from_batch(batch: PaperBatch) -> dict[str, object]:
    tensors = batch.tensors
    coords = tensors.get("coords", tensors.get("coors"))
    if coords is None:
        raise PaperExportError("batch lacks native coordinates")
    chain_index = tensors.get("chain_index", tensors.get("chains"))
    if chain_index is None:
        raise PaperExportError("batch lacks chain identity")
    return {
        "coors": coords,
        "residue_type": tensors["residue_type"],
        "mask": tensors["mask"],
        "generated_mask": tensors["generated_mask"],
        "fixed_mask": tensors["fixed_mask"],
        "chain_index": chain_index,
    }

def _native_target_rows(batch: PaperBatch) -> tuple[dict, ...]:
    rows = _generated_chains_last(_residue_rows(_structure_from_batch(batch), batch))
    native = tuple(row for row in rows if not row["generated_chain"])
    if not native:
        raise PaperExportError("binder native target is empty")
    _assert_finite_and_backbone(native)
    return native

def _generated_chains_last(rows: Sequence[Mapping[str, object]]) -> tuple[dict, ...]:
    generated_chains = []
    conditioning_chains = []
    seen = []
    for row in rows:
        chain = row["chain"]
        if chain not in seen:
            seen.append(chain)
            if row["generated_chain"]:
                generated_chains.append(chain)
            else:
                conditioning_chains.append(chain)
    if not generated_chains or not conditioning_chains:
        raise PaperExportError(
            "complex evaluation requires generated and conditioning chains"
        )
    order = {
        chain: index
        for index, chain in enumerate(conditioning_chains + generated_chains)
    }
    return tuple(sorted(rows, key=lambda row: (order[row["chain"]], row["index"])))

def _atom_line(
    serial: int, name: str, resname: str, chain: str, resid: int, xyz
) -> str:
    atom_name = name if len(name) == 4 else f" {name}"
    return (
        f"{'ATOM':<6}{serial:>5} {atom_name:<4}{'':>1}"
        f"{resname:>3} {chain:>1}{int(resid):>4}{'':>1}   "
        f"{float(xyz[0]):>8.3f}{float(xyz[1]):>8.3f}{float(xyz[2]):>8.3f}"
        f"{1.00:>6.2f}{0.00:>6.2f}          "
        f"{name[0]:>2}{'':>2}"
    )

def _pdb_text(rows: Sequence[Mapping[str, object]]) -> str:
    import numpy as np

    lines = ["MODEL        1"]
    serial = 1
    last_chain = None
    last_resname = None
    for row in rows:
        if last_chain is not None and row["chain"] != last_chain:
            lines.append(f"TER   {serial:5d}      {last_resname:>3} {last_chain:>1}")
            serial += 1
        coords = np.asarray(row["coords"], dtype=float)
        for atom_index, name in enumerate(_ATOM37):
            if atom_index >= coords.shape[0]:
                break
            xyz = coords[atom_index]
            if float(abs(xyz).sum()) <= 1e-7:
                continue
            lines.append(
                _atom_line(
                    serial,
                    name,
                    str(row["resname"]),
                    str(row["chain"]),
                    int(row["residue"]),
                    xyz,
                )
            )
            serial += 1
        last_chain = row["chain"]
        last_resname = row["resname"]
    if last_chain is not None:
        lines.append(f"TER   {serial:5d}      {last_resname:>3} {last_chain:>1}")
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines) + "\n"

def _write_bytes(path: Path, content: bytes) -> ArtifactIdentity:
    path = _refuse_denied(path)
    if path.exists() or path.is_symlink():
        raise PaperExportError("refusing to overwrite export artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise PaperExportError("partial export artifact exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(partial, flags, 0o664)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PaperExportError("create-new write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, path)
    return file_identity(path)

def _write_pdb(rows: Sequence[Mapping[str, object]], path: Path) -> ArtifactIdentity:
    return _write_bytes(path, _pdb_text(rows).encode("utf-8"))

def ame_ligand_chain_id(example_id: str) -> str:
    parts = str(example_id).split("__")
    if not str(example_id).startswith("ame:") or len(parts) != 4 or not parts[-1]:
        raise PaperExportError("AME example identity lacks its target ligand component")
    return parts[-1]

def _authenticated_ame_component_token(batch: PaperBatch) -> str:
    token = ame_ligand_chain_id(batch.target.example_id)
    recorded = batch.target.canonical_component_token
    if recorded is not None and recorded != token:
        raise PaperExportError("AME canonical component token drifted")
    if "_" in token and len([part for part in token.split("_") if part]) != 1:
        raise PaperExportError("AME target component is ambiguous")
    return token

def _authenticated_ame_ligand_identity(
    batch: PaperBatch, token: str
) -> tuple[str, str]:
    smiles = batch.target.ligand_smiles
    if type(smiles) is not str or not smiles.strip():
        raise PaperExportError("AME ligand identity/SMILES is missing")
    identity = batch.target.ligand_identity
    if identity is None:
        identity = token
    if type(identity) is not str or not identity.strip():
        raise PaperExportError("AME ligand identity/SMILES is missing")
    if identity != token:
        raise PaperExportError("AME ligand identity does not match canonical token")
    return identity, smiles

def _ligand_atom_key(atom_index: int, ligand) -> tuple[object, ...]:
    return (
        int(ligand.res_id[atom_index]),
        str(ligand.res_name[atom_index]),
        str(ligand.atom_name[atom_index]),
        str(ligand.element[atom_index]),
    )

def _load_ame_ligand_with_author_identity(source: Path, ligand_chain: str):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    cif = CIFFile.read(source)
    source_atoms = get_structure(
        cif,
        model=1,
        use_author_fields=True,
        include_bonds=True,
        extra_fields=["auth_seq_id", "label_asym_id"],
    )
    by_chain = source_atoms[source_atoms.chain_id == ligand_chain]
    by_label = None
    if "label_asym_id" in source_atoms.get_annotation_categories():
        by_label = source_atoms[source_atoms.label_asym_id == ligand_chain]
    if len(by_chain) and by_label is not None and len(by_label):
        chain_keys = {
            _ligand_atom_key(index, by_chain) for index in range(len(by_chain))
        }
        label_keys = {
            _ligand_atom_key(index, by_label) for index in range(len(by_label))
        }
        if chain_keys != label_keys:
            raise PaperExportError("AME target component is ambiguous")
        return by_chain.copy()
    if len(by_chain):
        return by_chain.copy()
    if by_label is not None and len(by_label):
        return by_label.copy()
    raise PaperExportError("AME target component does not match the canonical token")

def _ligand_membership_payload(ligand) -> list[list[object]]:
    rows = []
    for index in range(len(ligand)):
        rows.append(
            [
                int(ligand.res_id[index]),
                str(ligand.res_name[index]),
                str(ligand.atom_name[index]),
                str(ligand.element[index]),
            ]
        )
    return rows

def _ligand_coordinate_payload(ligand) -> list[list[object]]:
    rows = []
    for index in range(len(ligand)):
        coord = ligand.coord[index]
        rows.append(
            [
                int(ligand.res_id[index]),
                str(ligand.atom_name[index]),
                [float(coord[0]), float(coord[1]), float(coord[2])],
            ]
        )
    return rows

def _reclassify_ame_clash_ligand(ligand):
    rewritten = ligand.copy()
    names = [str(name) for name in ligand.atom_name.tolist()]
    resids = [int(resid) for resid in ligand.res_id.tolist()]
    rewritten.res_name[:] = AME_CLASH_LIGAND_RESNAME
    if [str(name) for name in rewritten.atom_name.tolist()] != names:
        raise PaperExportError("AME ligand membership or coordinates changed")
    if [int(resid) for resid in rewritten.res_id.tolist()] != resids:
        raise PaperExportError("AME ligand membership or coordinates changed")
    before_membership = _sha256_payload(_ligand_membership_payload(ligand))
    after_membership = _sha256_payload(_ligand_membership_payload(rewritten))
    before_coords = _sha256_payload(_ligand_coordinate_payload(ligand))
    after_coords = _sha256_payload(_ligand_coordinate_payload(rewritten))
    if after_coords != before_coords:
        raise PaperExportError("AME ligand membership or coordinates changed")
    return rewritten, {
        "membership_before": before_membership,
        "membership_after": after_membership,
        "coordinate_before": before_coords,
        "coordinate_after": after_coords,
        "sequence": _ligand_sequence(ligand),
    }

def _ligand_sequence(ligand) -> str:
    letters: list[str] = []
    seen: set[int] = set()
    for index in range(len(ligand)):
        resid = int(ligand.res_id[index])
        if resid in seen:
            continue
        seen.add(resid)
        resname = str(ligand.res_name[index])
        letters.append(_AA3_TO_1.get(resname, "X"))
    return "".join(letters)

def _classify_ame_ligand(ligand) -> tuple[str, bool]:
    import numpy as np

    if len(ligand) == 0:
        raise PaperExportError(
            "AME target component does not match the canonical token"
        )
    resnames = {str(name) for name in ligand.res_name.tolist()}
    hetero = np.asarray(ligand.hetero)
    water = bool(resnames & _WATER_RESNAMES)
    if water and not resnames <= _WATER_RESNAMES:
        raise PaperExportError("AME target component has mixed membership")
    if resnames <= _WATER_RESNAMES:
        raise PaperExportError("AME target component is water-only")
    if bool(hetero.all()):
        return "non_polymer", True
    return "polymer_peptide", False

def _export_ame_complex(
    rows: Sequence[Mapping[str, object]], batch: PaperBatch, output: Path
) -> tuple[ArtifactIdentity, tuple[tuple[str, int], ...], AmeLigandTransport]:
    import biotite.structure
    import numpy as np

    import torch

    _ensure_upstream()
    from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_from_encoding
    from biotite.structure.io import save_structure

    token = _authenticated_ame_component_token(batch)
    identity, smiles = _authenticated_ame_ligand_identity(batch, token)
    ligand = _load_ame_ligand_with_author_identity(
        Path(batch.target.source_path), token
    )
    representation, original_hetero = _classify_ame_ligand(ligand)
    membership_sha = _sha256_payload(_ligand_membership_payload(ligand))
    coordinate_sha = _sha256_payload(_ligand_coordinate_payload(ligand))
    sequence = _ligand_sequence(ligand) if representation == "polymer_peptide" else ""
    atom_count = int(len(ligand))
    coors = torch.tensor(
        np.stack([row["coords"] for row in rows], axis=0), dtype=torch.float32
    )
    residue_type = torch.tensor(
        [
            _RESNAMES.index(row["resname"]) if row["resname"] in _RESNAMES else 0
            for row in rows
        ],
        dtype=torch.int64,
    )
    atom_mask = torch.sum(torch.abs(coors), dim=-1) > 1e-7
    protein = atom_array_from_encoding(
        encoded_coord=coors,
        encoded_mask=atom_mask,
        encoded_seq=residue_type,
        encoding=AF2_ATOM37_ENCODING,
    ).copy()
    residue_names = np.asarray(protein.res_name, dtype="U5")
    protein.del_annotation("res_name")
    protein.set_annotation("res_name", residue_names)
    protein.chain_id[:] = AME_EVALUATOR_PROTEIN_CHAIN
    ligand.chain_id[:] = AME_EVALUATOR_LIGAND_CHAIN
    normalized_membership = _sha256_payload(_ligand_membership_payload(ligand))
    normalized_coordinates = _sha256_payload(_ligand_coordinate_payload(ligand))
    if (
        normalized_membership != membership_sha
        or normalized_coordinates != coordinate_sha
    ):
        raise PaperExportError("AME ligand membership or coordinates changed")
    _clash_ligand, clash_hashes = _reclassify_ame_clash_ligand(ligand)
    if (
        clash_hashes["membership_before"] != membership_sha
        or clash_hashes["coordinate_before"] != coordinate_sha
        or clash_hashes["coordinate_after"] != coordinate_sha
    ):
        raise PaperExportError("AME ligand membership or coordinates changed")
    protein_ids = _protein_residue_identities(protein)
    if len(protein_ids) != len(rows):
        raise PaperExportError("AME protein residue identity drifted")
    complex_atoms = biotite.structure.concatenate([ligand, protein])
    output = _refuse_denied(output)
    if output.exists() or output.is_symlink():
        raise PaperExportError("refusing to overwrite export artifact")
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}.pdb")
    try:
        save_structure(str(partial), complex_atoms)
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    transport = AmeLigandTransport(
        version=AME_LIGAND_TRANSPORT_VERSION,
        canonical_component_token=token,
        ligand_identity=identity,
        ligand_smiles=smiles,
        original_representation=representation,
        original_hetero=original_hetero,
        native_hetatm_claimed=False,
        evaluator_ligand_chain=AME_EVALUATOR_LIGAND_CHAIN,
        sequence=sequence,
        atom_count=atom_count,
        atom_membership_sha256=membership_sha,
        coordinate_sha256=coordinate_sha,
        normalized_atom_membership_sha256=normalized_membership,
        normalized_coordinate_sha256=normalized_coordinates,
        clash_membership_before_sha256=clash_hashes["membership_before"],
        clash_membership_after_sha256=clash_hashes["membership_after"],
        clash_coordinate_before_sha256=clash_hashes["coordinate_before"],
        clash_coordinate_after_sha256=clash_hashes["coordinate_after"],
        clash_selector=MappingProxyType(
            {
                "kind": "canonical_chain",
                "chain_id": AME_EVALUATOR_LIGAND_CHAIN,
                "component_token": token,
            }
        ),
    )
    return file_identity(output), protein_ids, transport

def _protein_residue_identities(protein) -> tuple[tuple[str, int], ...]:
    identities: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for chain, resid in zip(protein.chain_id, protein.res_id, strict=True):
        key = (str(chain), int(resid))
        if key in seen:
            continue
        seen.add(key)
        identities.append(key)
    return tuple(identities)

def _write_motif_positions(
    rows: Sequence[Mapping[str, object]],
    batch: PaperBatch,
    path: Path,
    *,
    protein_ids: Sequence[tuple[str, int]],
) -> ArtifactIdentity:
    if len(protein_ids) != len(rows):
        raise PaperExportError("AME motif identity does not match exported protein")
    motif = batch.tensors.get("motif_mask")
    if motif is None:
        raise PaperExportError("missing motif")
    motif_mask = _drop_batch(_as_numpy(motif, dtype=bool))
    if motif_mask.ndim > 1:
        motif_mask = motif_mask.reshape(motif_mask.shape[0], -1).any(axis=-1)
    native_coords = batch.tensors.get("coords", batch.tensors.get("coors"))
    if native_coords is None:
        raise PaperExportError("batch lacks native coordinates")
    native_coords = _drop_batch(_as_numpy(native_coords, dtype=float))
    selected = []
    for row, identity in zip(rows, protein_ids, strict=True):
        if not bool(motif_mask[row["index"]]):
            continue
        ca = native_coords[row["index"], 1]
        selected.append(
            {
                "residue_index": int(identity[1]),
                "chain_id": str(identity[0]),
                "ca": [float(value) for value in ca.tolist()],
            }
        )
    if not selected:
        raise PaperExportError("missing motif")
    payload = {
        "chain_id": [item["chain_id"] for item in selected],
        "residue_index": [item["residue_index"] for item in selected],
        "ca": [item["ca"] for item in selected],
    }
    return _write_bytes(path, canonical_json_bytes(payload))

def _write_native_motif_pdb(
    rows: Sequence[Mapping[str, object]],
    batch: PaperBatch,
    path: Path,
    *,
    protein_ids: Sequence[tuple[str, int]],
) -> ArtifactIdentity:
    if len(protein_ids) != len(rows):
        raise PaperExportError("AME motif identity does not match exported protein")
    motif = batch.tensors.get("motif_mask")
    if motif is None:
        raise PaperExportError("missing motif")
    motif_mask = _drop_batch(_as_numpy(motif, dtype=bool))
    if motif_mask.ndim > 1:
        motif_mask = motif_mask.reshape(motif_mask.shape[0], -1).any(axis=-1)
    native_coords = batch.tensors.get("coords", batch.tensors.get("coors"))
    if native_coords is None:
        raise PaperExportError("batch lacks native coordinates")
    native_coords = _drop_batch(_as_numpy(native_coords, dtype=float))
    selected = []
    for row, identity in zip(rows, protein_ids, strict=True):
        if not bool(motif_mask[row["index"]]):
            continue
        selected.append(
            {
                "index": row["index"],
                "chain": str(identity[0]),
                "chain_index": row["chain_index"],
                "residue": int(identity[1]),
                "resname": row["resname"],
                "coords": native_coords[row["index"]],
                "generated": False,
                "fixed": True,
                "generated_chain": False,
            }
        )
    if not selected:
        raise PaperExportError("missing motif")
    return _write_pdb(selected, path)

def _frozen_antibody_roles(
    payload: Mapping[str, str],
    rows: Sequence[Mapping[str, object]],
    mapping: Mapping[tuple[str, int], ResidueKey],
) -> FrozenAntibodyRoles:
    generated_rows = tuple(row for row in rows if bool(row["generated"]))
    if len(generated_rows) < 3:
        raise PaperExportError(
            "antibody train parent is missing frozen H3 or antigen roles"
        )
    keys = _unique_residue_projection(generated_rows, mapping)
    heavy_chains = {key.chain_id for key in keys}
    if len(heavy_chains) != 1:
        raise PaperExportError("antibody H3 maps to multiple heavy chains")
    antigen = tuple(
        _auth_chain_for_label(mapping, part.strip())
        for part in str(payload.get("target") or "").split(",")
        if part.strip()
    )
    if len(keys) < 3 or not antigen:
        raise PaperExportError(
            "antibody train parent is missing frozen H3 or antigen roles"
        )
    return FrozenAntibodyRoles(next(iter(heavy_chains)), keys, antigen)

def _positional_h3_rows(
    rows: Sequence[Mapping[str, object]], generated: str
) -> tuple[Mapping[str, object], ...]:
    from dive.data.roles import RoleError, Selector

    try:
        selector = Selector.parse(generated)
    except RoleError as error:
        raise PaperExportError("antibody roles require a CDR range") from error
    if selector.is_whole_chain or selector.start is None or selector.end is None:
        raise PaperExportError("antibody roles require a CDR range")
    positions = tuple(row for row in rows if str(row["chain"]) == selector.chain)
    if not positions:
        raise PaperExportError("antibody H3 selector names a missing batch chain")
    if selector.end >= len(positions):
        raise PaperExportError("antibody H3 selector runs past the batch chain")
    selected = positions[selector.start : selector.end + 1]
    if len(selected) < 3:
        raise PaperExportError(
            "antibody train parent is missing frozen H3 or antigen roles"
        )
    return selected

def _label_to_auth_mapping(source: Path) -> dict[tuple[str, int], ResidueKey]:
    if ".cif" in source.name.lower():
        return _atom_site_mapping(source)
    return _pdb_residue_mapping(source)

def _atom_site_mapping(source: Path) -> dict[tuple[str, int], ResidueKey]:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    raw = MMCIF2Dict(str(source))
    required = (
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
    )
    if any(column not in raw for column in required):
        raise PaperExportError("antibody source lacks _atom_site mapping")
    mapping: dict[tuple[str, int], ResidueKey] = {}
    reverse: dict[ResidueKey, tuple[str, int]] = {}
    for label, lseq, auth, aseq, ins in zip(*(raw[c] for c in required), strict=True):
        if lseq in {".", "?"}:
            continue
        key = (str(label), int(lseq))
        value = ResidueKey(str(auth), int(aseq), _antibody_insertion_code(ins))
        if key in mapping and mapping[key] != value:
            raise PaperExportError("ambiguous antibody source mapping")
        if value in reverse and reverse[value] != key:
            raise PaperExportError("multiply mapped antibody role")
        mapping[key] = value
        reverse[value] = key
    if not mapping:
        raise PaperExportError("antibody source mapping is empty")
    return mapping

def _pdb_residue_mapping(source: Path) -> dict[tuple[str, int], ResidueKey]:
    atoms = _native_atom_records(source)
    mapping: dict[tuple[str, int], ResidueKey] = {}
    for atom in atoms:
        key = (atom.chain, int(atom.residue))
        value = atom.residue_key
        if key in mapping and mapping[key] != value:
            raise PaperExportError("ambiguous antibody source mapping")
        mapping[key] = value
    if not mapping:
        raise PaperExportError("antibody source mapping is empty")
    return mapping

def _unique_residue_projection(
    rows: Sequence[Mapping[str, object]],
    mapping: Mapping[tuple[str, int], ResidueKey],
) -> tuple[ResidueKey, ...]:
    keys: list[ResidueKey] = []
    seen: dict[ResidueKey, tuple[str, int]] = {}
    for row in rows:
        label = (str(row["chain"]), int(row["residue"]))
        if label not in mapping:
            raise PaperExportError("unmapped antibody residue identity")
        key = mapping[label]
        if key in seen and seen[key] != label:
            raise PaperExportError("multiply mapped antibody role")
        seen[key] = label
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise PaperExportError("duplicate antibody role mapping")
    return tuple(keys)

def _auth_chain_for_label(
    mapping: Mapping[tuple[str, int], ResidueKey], label_chain: str
) -> str:
    chains = {
        value.chain_id for (lab, _), value in mapping.items() if lab == label_chain
    }
    if len(chains) != 1:
        raise PaperExportError("ambiguous antibody antigen mapping")
    return next(iter(chains))

def _antibody_insertion_code(value: object) -> str:
    try:
        return normalize_insertion_code(value)
    except ValueError as error:
        raise PaperExportError("antibody insertion code is invalid") from error

def _native_atom_records(
    source: Path, mapping: Mapping[tuple[str, int], ResidueKey] | None = None
) -> tuple[AtomRecord, ...]:
    del mapping
    from Bio.PDB import MMCIFParser, PDBParser

    handle = source.open("rt")
    with handle as stream:
        if ".cif" in source.name:
            structure = MMCIFParser(QUIET=True).get_structure("native", stream)
        else:
            structure = PDBParser(QUIET=True).get_structure("native", stream)
    atoms = []
    seen: set[tuple[str, str, str, str]] = set()
    for atom in structure.get_atoms():
        residue = atom.get_parent()
        chain = residue.get_parent()
        _hetflag, seq, icode = residue.get_id()
        insertion = _antibody_insertion_code(icode)
        identity = (chain.id, str(seq), insertion, atom.get_name())
        if identity in seen:
            raise PaperExportError("duplicate antibody native atom identity")
        seen.add(identity)
        coord = tuple(float(value) for value in atom.get_coord())
        atoms.append(
            AtomRecord(
                chain.id,
                str(seq),
                atom.get_name(),
                coord,
                element=atom.element,
                residue_name=residue.get_resname(),
                insertion_code=insertion,
            )
        )
    if not atoms:
        raise PaperExportError("native antibody structure has no atoms")
    return tuple(atoms)

def _predicted_atom_records(
    rows: Sequence[Mapping[str, object]],
    mapping: Mapping[tuple[str, int], ResidueKey],
) -> tuple[AtomRecord, ...]:
    import numpy as np

    projected = _unique_residue_projection(rows, mapping)
    atoms = []
    for row, key in zip(rows, projected, strict=True):
        coords = np.asarray(row["coords"], dtype=float)
        for atom_index, name in enumerate(_ATOM37):
            if atom_index >= coords.shape[0]:
                break
            xyz = coords[atom_index]
            if float(abs(xyz).sum()) <= 1e-7:
                continue
            atoms.append(
                AtomRecord(
                    key.chain_id,
                    str(key.auth_seq_id),
                    name,
                    (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                    element=name[0],
                    residue_name=str(row["resname"]),
                    insertion_code=key.insertion_code,
                )
            )
    if not atoms:
        raise PaperExportError("generated sample is missing predicted atoms")
    return tuple(atoms)

def _write_manifest(
    path: Path,
    *,
    family: str,
    example_id: str,
    artifacts: Sequence[tuple[str, ArtifactIdentity]],
    binder_backbone_normalization: BinderBackboneNormalization | None = None,
    ame_ligand_transport: AmeLigandTransport | None = None,
) -> ArtifactIdentity:
    payload = {
        "artifacts": [
            {
                "kind": kind,
                "path": identity.path,
                "sha256": identity.sha256,
                "size_bytes": identity.size_bytes,
            }
            for kind, identity in artifacts
        ],
        "example_id": example_id,
        "family": family,
        "schema": "dive.paper.export_manifest.v1",
    }
    if binder_backbone_normalization is not None:
        payload["binder_backbone_normalization"] = (
            binder_backbone_normalization.to_mapping()
        )
    if ame_ligand_transport is not None:
        payload["ame_ligand_transport"] = ame_ligand_transport.to_mapping()
    return _write_bytes(path, canonical_json_bytes(payload))

def _ensure_upstream() -> None:
    import sys

    source = str(EMERGENT_UPSTREAM_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    os.environ.setdefault("USE_V2_COMPLEXA_ARCH", "True")
    import proteinfoundation.patches.atomworks_patches
