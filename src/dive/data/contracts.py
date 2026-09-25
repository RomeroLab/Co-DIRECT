
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

class ContractError(ValueError):
    pass

class Family(StrEnum):

    BINDER = "binder"
    AME = "ame"
    ANTIBODY = "antibody"

class Partition(StrEnum):

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_OUTER = "test-outer"
    TEST_BLIND = "test-blind"
    LEGACY_DEV = "legacy-dev"
    EXCLUDED_TEST_NEIGHBOR = "excluded-test-neighbor"
    EXCLUDED_VAL_NEIGHBOR = "excluded-validation-neighbor"

@dataclass(frozen=True, slots=True)
class EligibilityFailure:

    code: str
    detail: str
    source_id: str

@dataclass(frozen=True, slots=True)
class ChainRecord:

    chain_id: str
    role: str
    sequence: str
    molecule_type: str = "protein"
    entity_id: str | None = None

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ContractError("every sequence-bearing chain needs an explicit role")
        if not self.sequence.strip():
            raise ContractError(f"chain {self.chain_id!r} has an empty sequence")

@dataclass(frozen=True, slots=True)
class LigandRecord:

    ligand_id: str
    smiles: str
    inchikey: str | None
    murcko_scaffold: str

    def __post_init__(self) -> None:
        if not self.smiles.strip():
            raise ContractError("ligand records require canonical SMILES")

@dataclass(frozen=True, slots=True)
class CanonicalExample:

    example_id: str
    parent_id: str
    family: Family
    source_release: str
    pdb_id: str | None
    assembly_id: str | None
    deposition_date: date | None
    chains: tuple[ChainRecord, ...]
    ligands: tuple[LigandRecord, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chains:
            raise ContractError("a canonical example needs at least one chain")

    def identity_hash(self) -> str:

        digest = hashlib.sha256()
        digest.update(self.example_id.encode())
        digest.update(self.parent_id.encode())
        digest.update(str(self.family).encode())
        digest.update(self.source_release.encode())
        for key in sorted(self.metadata):
            digest.update(f"{key}={self.metadata[key]}".encode())
        return digest.hexdigest()

def sequence_hash(sequence: str) -> str:

    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ContractError("cannot hash an empty sequence")
    return hashlib.sha256(normalized.encode()).hexdigest()

def parent_hash(
    chains: Iterable[ChainRecord], ligands: Sequence[LigandRecord] | None = None
) -> str:

    pairs = sorted((chain.role, sequence_hash(chain.sequence)) for chain in chains)
    if not pairs:
        raise ContractError("a parent group needs at least one sequence-bearing chain")
    digest = hashlib.sha256()
    for role, seq_hash in pairs:
        digest.update(f"{role}:{seq_hash}".encode())
    return digest.hexdigest()
