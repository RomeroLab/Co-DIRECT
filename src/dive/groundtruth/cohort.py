
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dive.groundtruth.complex_spec import RoleAssignment
from dive.groundtruth.contamination import Assessment
from dive.groundtruth.rcsb import EntryMetadata

GATE_B_MINIMUM = 20

class CohortError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class CohortEntry:

    pdb_id: str
    binder_chain: str
    target_chains: tuple[str, ...]
    n_binder_residues: int
    revision_date: str
    contamination: Assessment

    def to_dict(self) -> dict[str, object]:

        return {
            "pdb_id": self.pdb_id,
            "binder_chain": self.binder_chain,
            "target_chains": list(self.target_chains),
            "n_binder_residues": self.n_binder_residues,
            "revision_date": self.revision_date,
            "contamination": self.contamination.to_dict(),
        }

@dataclass(frozen=True, slots=True)
class CohortVerdict:

    size: int
    minimum: int
    passed: bool
    statement: str

    def to_dict(self) -> dict[str, object]:

        return {
            "cohort_size": self.size,
            "minimum": self.minimum,
            "passed": self.passed,
            "statement": self.statement,
        }

def apply_gate_b(entries: Sequence[CohortEntry]) -> CohortVerdict:

    size = len(entries)
    passed = size >= GATE_B_MINIMUM
    if passed:
        statement = (
            f"{size} complexes meets the pre-registered minimum of {GATE_B_MINIMUM}. "
            f"The calibration proceeds, and every report states that 90% identity "
            f"excludes near-duplicates but not homologs."
        )
    else:
        statement = (
            f"{size} complexes is below the pre-registered minimum of "
            f"{GATE_B_MINIMUM}. The study is reported as DESCRIPTIVE ONLY and is "
            f"not called a validation."
        )
    return CohortVerdict(
        size=size, minimum=GATE_B_MINIMUM, passed=passed, statement=statement
    )

def build_entry(
    metadata: EntryMetadata, roles: RoleAssignment, assessment: Assessment
) -> CohortEntry | None:

    if not roles.eligible or not assessment.clean:
        return None
    if roles.binder_chain is None:
        raise CohortError(f"{metadata.pdb_id}: eligible assignment carries no binder")

    sizes = {chain.chain_id: chain.n_residues for chain in metadata.chains}
    if roles.binder_chain not in sizes:
        raise CohortError(
            f"{metadata.pdb_id}: binder chain {roles.binder_chain!r} is absent from "
            f"the entry's chain inventory"
        )

    return CohortEntry(
        pdb_id=metadata.pdb_id,
        binder_chain=roles.binder_chain,
        target_chains=roles.target_chains,
        n_binder_residues=int(sizes[roles.binder_chain]),
        revision_date=metadata.revision_date,
        contamination=assessment,
    )
