
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_SEGMENT = re.compile(r"^(?P<chain>[A-Za-z]+)(?P<start>-?\d+)-(?P<end>-?\d+)$")

class ComplexSpecError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class ChainSpec:

    chain_id: str
    n_residues: int
    is_protein: bool

    def to_dict(self) -> dict[str, object]:

        return {
            "chain_id": self.chain_id,
            "n_residues": self.n_residues,
            "is_protein": self.is_protein,
        }

@dataclass(frozen=True, slots=True)
class Eligibility:

    pdb_id: str
    target: str
    eligible: bool
    binder_chain: str | None
    reason: str
    candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:

        return {
            "pdb_id": self.pdb_id,
            "target": self.target,
            "eligible": self.eligible,
            "binder_chain": self.binder_chain,
            "reason": self.reason,
            "candidates": list(self.candidates),
        }

def target_chain_ids(target_input: str) -> tuple[str, ...]:

    if not target_input or not target_input.strip():
        raise ComplexSpecError("target_input is empty")

    seen: list[str] = []
    for raw in target_input.split(","):
        segment = raw.strip()
        match = _SEGMENT.match(segment)
        if match is None:
            raise ComplexSpecError(f"malformed target_input segment: {segment!r}")
        chain = match.group("chain")
        if chain not in seen:
            seen.append(chain)
    return tuple(seen)

def evaluate_eligibility(
    pdb_id: str,
    target: str,
    chains: Sequence[ChainSpec],
    target_input: str,
    binder_length: Sequence[int],
) -> Eligibility:

    if len(binder_length) != 2:
        raise ComplexSpecError(f"binder_length must have two entries: {binder_length!r}")
    low, high = int(binder_length[0]), int(binder_length[1])
    if low > high:
        raise ComplexSpecError(f"binder_length is inverted: {binder_length!r}")

    target_chains = set(target_chain_ids(target_input))
    candidates = tuple(
        sorted(
            chain.chain_id
            for chain in chains
            if chain.is_protein
            and chain.chain_id not in target_chains
            and low <= chain.n_residues <= high
        )
    )

    if len(candidates) == 1:
        return Eligibility(
            pdb_id=pdb_id,
            target=target,
            eligible=True,
            binder_chain=candidates[0],
            reason=f"exactly one candidate chain in [{low}, {high}]",
            candidates=candidates,
        )
    if not candidates:
        return Eligibility(
            pdb_id=pdb_id,
            target=target,
            eligible=False,
            binder_chain=None,
            reason=f"no candidate protein chain in [{low}, {high}] outside the target",
            candidates=candidates,
        )
    return Eligibility(
        pdb_id=pdb_id,
        target=target,
        eligible=False,
        binder_chain=None,
        reason=(
            f"ambiguous: {len(candidates)} candidate chains in [{low}, {high}] "
            f"({', '.join(candidates)}); no principled choice between them"
        ),
        candidates=candidates,
    )

BINDER_WINDOW: tuple[int, int] = (40, 250)

@dataclass(frozen=True, slots=True)
class RoleAssignment:

    binder_chain: str | None
    target_chains: tuple[str, ...]
    eligible: bool
    reason: str

    def to_dict(self) -> dict[str, object]:

        return {
            "binder_chain": self.binder_chain,
            "target_chains": list(self.target_chains),
            "eligible": self.eligible,
            "reason": self.reason,
        }

def assign_roles(
    chains: Sequence[ChainSpec], *, window: Sequence[int] = BINDER_WINDOW
) -> RoleAssignment:

    low, high = int(window[0]), int(window[1])
    if low > high:
        raise ComplexSpecError(f"binder window is inverted: {window!r}")

    proteins = [chain for chain in chains if chain.is_protein]
    in_window = sorted(
        chain.chain_id for chain in proteins if low <= chain.n_residues <= high
    )
    outside = tuple(
        sorted(
            chain.chain_id
            for chain in proteins
            if not (low <= chain.n_residues <= high)
        )
    )

    if len(in_window) > 1:
        return RoleAssignment(
            binder_chain=None,
            target_chains=outside,
            eligible=False,
            reason=(
                f"ambiguous: {len(in_window)} chains in [{low}, {high}] "
                f"({', '.join(in_window)}); no principled choice between them"
            ),
        )
    if not in_window:
        return RoleAssignment(
            binder_chain=None,
            target_chains=outside,
            eligible=False,
            reason=f"no chain in the binder window [{low}, {high}]",
        )
    if not outside:
        return RoleAssignment(
            binder_chain=None,
            target_chains=(),
            eligible=False,
            reason="no target chain outside the binder window",
        )
    return RoleAssignment(
        binder_chain=in_window[0],
        target_chains=outside,
        eligible=True,
        reason=f"exactly one chain in [{low}, {high}]",
    )
