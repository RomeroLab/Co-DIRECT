
from __future__ import annotations

import math

DEFAULT_BREAK_THRESHOLD = 4.5

def ca_trace(pdb_text: str) -> dict[str, list[tuple[int, tuple[float, float, float]]]]:

    per: dict[str, list[tuple[int, tuple[float, float, float]]]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain = line[21]
        per.setdefault(chain, []).append((
            int(line[22:26]),
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        ))
    return per

def chain_breaks(pdb_text: str, threshold: float = DEFAULT_BREAK_THRESHOLD) -> int:

    total = 0
    for residues in ca_trace(pdb_text).values():
        for (_n1, a), (_n2, b) in zip(residues, residues[1:]):
            if math.dist(a, b) > threshold:
                total += 1
    return total
