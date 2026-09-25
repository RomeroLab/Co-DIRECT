
from __future__ import annotations

from collections.abc import Mapping, Sequence

def interface_iptm(pair_chains_iptm: Mapping, *, framework: Sequence[int],
                   antigen: Sequence[int]) -> float:

    if not antigen:
        raise ValueError("no antigen chain; this score names an interface that "
                         "would not exist and must not default to the global iPTM")
    if not framework:
        raise ValueError("no framework chain")
    values = []
    for f in framework:
        for a in antigen:
            forward = pair_chains_iptm[str(f)][str(a)]
            reverse = pair_chains_iptm[str(a)][str(f)]
            values.append((float(forward) + float(reverse)) / 2.0)
    return float(sum(values) / len(values))
