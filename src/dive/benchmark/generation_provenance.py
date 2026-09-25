
from __future__ import annotations

import math
from typing import Sequence

__all__ = [
    "ProvenanceError",
    "classify_exported_residues",
    "design_region_summary",
]

MATCH_TOLERANCE = 0.05

class ProvenanceError(RuntimeError):
    pass

def _ca_by_resnum(pdb_text: str, chain: str) -> "dict[int, tuple[float, float, float]]":
    out: dict[int, tuple[float, float, float]] = {}
    seen_chain = False
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != chain:
            continue
        seen_chain = True
        if line[12:16].strip() != "CA":
            continue
        out.setdefault(
            int(line[22:26]),
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        )
    if not seen_chain:
        raise ProvenanceError(f"chain {chain!r} absent from the exported structure")
    return out

def classify_exported_residues(
    pdb_text: str,
    chain: str,
    raw_ca: Sequence[Sequence[float]],
    generated_mask: Sequence[bool],
    tolerance: float = MATCH_TOLERANCE,
) -> dict:

    if len(raw_ca) != len(generated_mask):
        raise ProvenanceError(
            f"sampler tensor has {len(raw_ca)} positions but the mask has "
            f"{len(generated_mask)}"
        )
    exported = _ca_by_resnum(pdb_text, chain)
    if not exported:
        raise ProvenanceError(f"chain {chain!r} has no CA atoms")

    generated: list[int] = []
    carried: list[int] = []
    unmatched: list[int] = []
    for resnum in sorted(exported):
        point = exported[resnum]
        best_index, best_distance = None, math.inf
        for index, candidate in enumerate(raw_ca):
            distance = math.dist(point, candidate)
            if distance < best_distance:
                best_distance, best_index = distance, index
        if best_distance > tolerance:
            unmatched.append(resnum)
        elif generated_mask[best_index]:
            generated.append(resnum)
        else:
            carried.append(resnum)

    if unmatched:
        raise ProvenanceError(
            f"{len(unmatched)} exported residues of chain {chain!r} match no "
            f"sampler position within {tolerance} A (first: {unmatched[:5]})"
        )

    total = len(exported)
    return {
        "chain": chain,
        "n_exported": total,
        "n_generated": len(generated),
        "n_carried": len(carried),
        "generated_fraction": len(generated) / total,
        "generated_residues": generated,
        "carried_residues": carried,
        "fully_generated": len(carried) == 0,
        "match_tolerance": tolerance,
    }

def design_region_summary(*, exported: int, generated: int) -> dict:

    if exported <= 0:
        raise ProvenanceError(f"exported length must be positive, got {exported}")
    if generated > exported:
        raise ProvenanceError(
            f"generated ({generated}) exceeds exported ({exported}); the sample "
            f"and the export do not correspond"
        )
    return {
        "n_exported": exported,
        "n_generated": generated,
        "n_carried": exported - generated,
        "generated_fraction": generated / exported,
        "truncated": generated < exported,
    }
