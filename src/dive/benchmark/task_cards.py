
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from dive.data.contracts import CanonicalExample, Family
from dive.training.preflight import canonical_json_bytes

class TaskCardError(RuntimeError):
    pass

class SequencePolicy(StrEnum):
    FIXED = "fixed"
    SAMPLED = "sampled"
    SELF_GENERATED = "self_generated"
    REDESIGNED = "redesigned"

class ManifestPartition(StrEnum):
    PUBLIC = "public"
    LEGACY_DEV = "legacy-dev"
    CALIBRATION = "calibration"
    SEALED = "sealed"
    REFERENCE_RECOVERY = "reference-recovery"
    STRESS = "stress"
    PAIRED = "paired"

class ContaminationStatus(StrEnum):
    KNOWN_OVERLAP = "known_overlap"
    NEAR_OVERLAP = "near_overlap"
    CLEAN_BY_DECLARED_DATA = "clean_by_declared_data"
    UNKNOWN = "unknown"

class OutputContract(StrEnum):
    MMCIF_FASTA = "mmcif_fasta"

@dataclass(frozen=True, slots=True)
class ResidueAtomMap:
    chain_id: str
    residue_index: int
    residue_name: str
    atom_names: tuple[str, ...]
    atom_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.chain_id:
            raise TaskCardError("residue maps require a chain_id")
        if type(self.residue_index) is not int:
            raise TaskCardError("residue_index must be an int")
        if not self.residue_name:
            raise TaskCardError("residue maps require a residue_name")
        if not self.atom_names:
            raise TaskCardError("residue maps require atom names")
        if len(self.atom_names) != len(self.atom_mask):
            raise TaskCardError("atom_names and atom_mask must have equal length")

@dataclass(frozen=True, slots=True)
class LigandGraph:
    ligand_id: str
    atom_names: tuple[str, ...]
    elements: tuple[str, ...]
    bonds: tuple[tuple[int, int, int], ...]
    formal_charges: tuple[int, ...]
    stereochemistry: str

    def __post_init__(self) -> None:
        if not self.ligand_id:
            raise TaskCardError("ligand graphs require a ligand_id")
        if not self.atom_names:
            raise TaskCardError("ligand graphs require atoms")
        if len(self.atom_names) != len(self.elements):
            raise TaskCardError("ligand atom names and elements must match")
        if len(self.formal_charges) != len(self.atom_names):
            raise TaskCardError("formal charges must match ligand atoms")
        n_atoms = len(self.atom_names)
        for left, right, order in self.bonds:
            if min(left, right) < 0 or max(left, right) >= n_atoms:
                raise TaskCardError("ligand bond index is out of range")
            if order <= 0:
                raise TaskCardError("bond order must be a positive integer")
        if not self.stereochemistry:
            raise TaskCardError("ligand stereochemistry must be declared")

@dataclass(frozen=True, slots=True)
class TaskCard:
    family: Family
    task_type: str
    source_id: str
    pdb_id: str | None
    mmcif_path: str | None
    source_release: str
    source_hash: str
    conditioning_chains: tuple[str, ...]
    generated_chains: tuple[str, ...]
    fixed_regions: tuple[str, ...]
    design_regions: tuple[str, ...]
    allowed_inputs: tuple[str, ...]
    hidden_reference: tuple[str, ...]
    sequence_policy: SequencePolicy
    residue_atom_maps: tuple[ResidueAtomMap, ...]
    ligand_graph: LigandGraph | None
    motif_or_epitope: tuple[str, ...]
    native_index_visible: bool
    native_rotamer_visible: bool
    native_dock_visible: bool
    length: int
    topology: str
    output_contract: str
    invalid_definition: str
    unsupported_definition: str
    failed_definition: str
    partition: ManifestPartition
    contamination: ContaminationStatus
    example: CanonicalExample
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.family is not self.example.family:
            raise TaskCardError("task card family must match the canonical example")
        if not self.task_type:
            raise TaskCardError("task_type is required")
        if not self.source_id:
            raise TaskCardError("source_id is required")
        if not self.generated_chains:
            raise TaskCardError("generated_chains is required")
        if not self.design_regions:
            raise TaskCardError("design_regions is required")
        if self.length <= 0:
            raise TaskCardError("length must be a positive integer")
        if type(self.native_index_visible) is not bool:
            raise TaskCardError("native_index_visible must be bool")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def semantic_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_mapping())).hexdigest()

    def as_mapping(self) -> dict[str, object]:
        return {
            "allowed_inputs": list(self.allowed_inputs),
            "conditioning_chains": list(self.conditioning_chains),
            "contamination": self.contamination.value,
            "design_regions": list(self.design_regions),
            "example_id": self.example.example_id,
            "failed_definition": self.failed_definition,
            "family": str(self.family),
            "fixed_regions": list(self.fixed_regions),
            "generated_chains": list(self.generated_chains),
            "hidden_reference": list(self.hidden_reference),
            "invalid_definition": self.invalid_definition,
            "length": self.length,
            "ligand_graph": None
            if self.ligand_graph is None
            else {
                "atom_names": list(self.ligand_graph.atom_names),
                "bonds": [list(bond) for bond in self.ligand_graph.bonds],
                "elements": list(self.ligand_graph.elements),
                "formal_charges": list(self.ligand_graph.formal_charges),
                "ligand_id": self.ligand_graph.ligand_id,
                "stereochemistry": self.ligand_graph.stereochemistry,
            },
            "metadata": dict(self.metadata),
            "mmcif_path": self.mmcif_path,
            "motif_or_epitope": list(self.motif_or_epitope),
            "native_dock_visible": self.native_dock_visible,
            "native_index_visible": self.native_index_visible,
            "native_rotamer_visible": self.native_rotamer_visible,
            "output_contract": self.output_contract,
            "parent_id": self.example.parent_id,
            "partition": self.partition.value,
            "pdb_id": self.pdb_id,
            "residue_atom_maps": [
                {
                    "atom_mask": list(item.atom_mask),
                    "atom_names": list(item.atom_names),
                    "chain_id": item.chain_id,
                    "residue_index": item.residue_index,
                    "residue_name": item.residue_name,
                }
                for item in self.residue_atom_maps
            ],
            "sequence_policy": self.sequence_policy.value,
            "source_hash": self.source_hash,
            "source_id": self.source_id,
            "source_release": self.source_release,
            "task_type": self.task_type,
            "topology": self.topology,
            "unsupported_definition": self.unsupported_definition,
        }

_BACKBONE_ATOMS = ("N", "CA", "C", "O")

def _maps_for_example(example: CanonicalExample) -> tuple[ResidueAtomMap, ...]:
    maps: list[ResidueAtomMap] = []
    for chain in example.chains:
        for index, residue in enumerate(chain.sequence):
            maps.append(
                ResidueAtomMap(
                    chain.chain_id,
                    index,
                    residue,
                    _BACKBONE_ATOMS,
                    (True, True, True, True),
                )
            )
    return tuple(maps)

def _source_hash(example: CanonicalExample) -> str:
    return example.identity_hash()

def _length(example: CanonicalExample, generated: Sequence[str]) -> int:
    generated_set = set(generated)
    return sum(
        len(chain.sequence)
        for chain in example.chains
        if chain.chain_id in generated_set or chain.role == "designed"
    )

def task_card_from_canonical(
    example: CanonicalExample,
    *,
    task_type: str,
    design_regions: Sequence[str],
    hidden_reference: Sequence[str],
    conditioning_chains: Sequence[str] | None = None,
    generated_chains: Sequence[str] | None = None,
    fixed_regions: Sequence[str] | None = None,
    allowed_inputs: Sequence[str] | None = None,
    sequence_policy: SequencePolicy = SequencePolicy.SAMPLED,
    ligand_graph: LigandGraph | None = None,
    motif_or_epitope: Sequence[str] = (),
    native_index_visible: bool = False,
    native_rotamer_visible: bool = False,
    native_dock_visible: bool = False,
    topology: str = "protein",
    partition: ManifestPartition = ManifestPartition.CALIBRATION,
    contamination: ContaminationStatus = ContaminationStatus.UNKNOWN,
    design_length: int | None = None,
    metadata: Mapping[str, str] | None = None,
) -> TaskCard:

    if example.family is Family.ANTIBODY and task_type.startswith("binder_"):
        raise TaskCardError("antibody tasks require an explicit CDR contract")
    if not tuple(design_regions):
        raise TaskCardError("design_regions is required")
    generated = tuple(
        generated_chains
        if generated_chains is not None
        else tuple(
            chain.chain_id
            for chain in example.chains
            if chain.role in {"designed", "antibody"}
        )
    )
    if not generated:
        generated = tuple(chain.chain_id for chain in example.chains[-1:])
    conditioning = tuple(
        conditioning_chains
        if conditioning_chains is not None
        else tuple(
            chain.chain_id
            for chain in example.chains
            if chain.chain_id not in generated
        )
    )
    length = design_length if design_length is not None else _length(example, generated)
    if length == 0:
        raise TaskCardError("design_regions produced a zero-length design")
    return TaskCard(
        family=example.family,
        task_type=task_type,
        source_id=example.example_id,
        pdb_id=example.pdb_id,
        mmcif_path=None,
        source_release=example.source_release,
        source_hash=_source_hash(example),
        conditioning_chains=conditioning,
        generated_chains=generated,
        fixed_regions=tuple(fixed_regions or conditioning),
        design_regions=tuple(design_regions),
        allowed_inputs=tuple(allowed_inputs or ()),
        hidden_reference=tuple(hidden_reference),
        sequence_policy=sequence_policy,
        residue_atom_maps=_maps_for_example(example),
        ligand_graph=ligand_graph,
        motif_or_epitope=tuple(motif_or_epitope),
        native_index_visible=native_index_visible,
        native_rotamer_visible=native_rotamer_visible,
        native_dock_visible=native_dock_visible,
        length=length,
        topology=topology,
        output_contract=OutputContract.MMCIF_FASTA.value,
        invalid_definition="missing_atom_or_nonphysical_geometry",
        unsupported_definition="method_cannot_express_required_conditions",
        failed_definition="timeout_oom_or_raised_exception",
        partition=partition,
        contamination=contamination,
        example=example,
        metadata=dict(metadata or {}),
    )

def binder_task_card(
    example: CanonicalExample,
    *,
    hotspot_residues: Sequence[str],
    design_length: int | None = None,
    partition: ManifestPartition = ManifestPartition.CALIBRATION,
) -> TaskCard:
    if example.family is not Family.BINDER:
        raise TaskCardError("binder_task_card requires a binder example")
    task_type = "binder_hotspot" if tuple(hotspot_residues) else "binder_site_free"
    return task_card_from_canonical(
        example,
        task_type=task_type,
        design_regions=tuple(
            chain.chain_id for chain in example.chains if chain.role == "designed"
        )
        or ("B",),
        hidden_reference=(
            "native_binder_sequence",
            "native_binder_structure",
            "native_dock",
            "residue_contact_pairing",
        ),
        allowed_inputs=(
            "target_coordinates",
            "target_sequence",
            "hotspots",
            "binder_length",
            "residue_alphabet",
        ),
        motif_or_epitope=tuple(hotspot_residues),
        native_dock_visible=False,
        design_length=design_length,
        partition=partition,
        metadata={"hotspot_task": task_type},
    )

def enzyme_task_card(
    example: CanonicalExample,
    *,
    ligand_graph: LigandGraph | None,
    catalytic_atoms: Sequence[str],
    partition: ManifestPartition = ManifestPartition.CALIBRATION,
) -> TaskCard:
    if example.family is not Family.AME:
        raise TaskCardError("enzyme_task_card requires an AME example")
    if ligand_graph is None:
        raise TaskCardError("AME task cards require a ligand_graph")
    if not tuple(catalytic_atoms):
        raise TaskCardError("AME task cards require catalytic named atoms")
    return task_card_from_canonical(
        example,
        task_type="ame_unindexed_motif_ligand",
        design_regions=tuple(chain.chain_id for chain in example.chains),
        hidden_reference=(
            "native_catalytic_index",
            "native_rotamer",
            "native_dock",
        ),
        allowed_inputs=(
            "ligand_coordinates",
            "ligand_graph",
            "catalytic_named_atoms",
            "motif_ligand_frame",
            "scaffold_length",
        ),
        ligand_graph=ligand_graph,
        motif_or_epitope=tuple(catalytic_atoms),
        native_index_visible=False,
        native_rotamer_visible=False,
        native_dock_visible=False,
        partition=partition,
    )

def vhh_cdr_h3_task_card(
    example: CanonicalExample,
    *,
    epitope_hotspots: Sequence[str],
    cdr_h3_length: int,
    framework_id: str,
    partition: ManifestPartition = ManifestPartition.CALIBRATION,
) -> TaskCard:
    if example.family is not Family.ANTIBODY:
        raise TaskCardError("vhh_cdr_h3_task_card requires an antibody example")
    if cdr_h3_length <= 0:
        raise TaskCardError("CDR-H3 length must be positive")
    if not framework_id:
        raise TaskCardError("a fixed framework_id is required")
    return task_card_from_canonical(
        example,
        task_type="antibody_vhh_cdr_h3",
        design_regions=("CDR-H3",),
        generated_chains=tuple(
            chain.chain_id for chain in example.chains if chain.role == "antibody"
        ),
        conditioning_chains=tuple(
            chain.chain_id for chain in example.chains if chain.role == "antigen"
        ),
        fixed_regions=("framework",),
        hidden_reference=(
            "native_cdr_sequence",
            "native_cdr_conformation",
            "native_antibody_antigen_dock",
        ),
        allowed_inputs=(
            "antigen_coordinates",
            "epitope_hotspots",
            "framework_structure",
            "framework_sequence",
            "cdr_mask",
            "cdr_h3_length",
        ),
        motif_or_epitope=tuple(epitope_hotspots),
        native_dock_visible=False,
        design_length=cdr_h3_length,
        partition=partition,
        metadata={"framework_id": framework_id, "capability_tier": "vhh"},
    )
