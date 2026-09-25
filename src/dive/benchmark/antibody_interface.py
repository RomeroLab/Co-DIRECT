
from __future__ import annotations

from collections.abc import Mapping, Sequence

class InterfaceError(RuntimeError):
    pass

def chain_role_indices(
    chains: Sequence[str], *, context: str, target: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:

    order = {name: index for index, name in enumerate(chains)}
    context_names = [c for c in context.split(",") if c]
    target_names = [c for c in target.split(",") if c]
    if not target_names:
        raise InterfaceError(
            "the task declares no antigen; a binding metric is undefined rather "
            "than zero"
        )
    missing = [c for c in (*context_names, *target_names) if c not in order]
    if missing:
        raise InterfaceError(
            f"chains {missing} are named by the task but absent from the run's "
            f"chain order {list(chains)}"
        )
    framework = tuple(order[c] for c in context_names)
    antigen = tuple(order[c] for c in target_names)
    overlap = set(framework) & set(antigen)
    if overlap:
        raise InterfaceError(
            f"chain index {sorted(overlap)} is both framework and antigen; the "
            f"interface would include a chain with itself"
        )
    return framework, antigen

def antibody_antigen_iptm(
    pair_chains_iptm: Mapping, framework: Sequence[int], antigen: Sequence[int]
) -> float:

    if not antigen:
        raise InterfaceError("no antigen chain; the binding metric is undefined")
    if not framework:
        raise InterfaceError("no framework chain; the binding metric is undefined")

    def lookup(row: int, column: int) -> float:
        try:
            return float(pair_chains_iptm[str(row)][str(column)])
        except (KeyError, TypeError, ValueError) as error:
            raise InterfaceError(
                f"pair_chains_iptm has no entry [{row}][{column}]; the matrix "
                f"covers {sorted(pair_chains_iptm)}"
            ) from error

    values = []
    for f in framework:
        for a in antigen:
            values.append(lookup(f, a))
            values.append(lookup(a, f))
    return sum(values) / len(values)

def harvest(confidence: Mapping, chains: Sequence[str], *, context: str, target: str) -> dict:

    framework, antigen = chain_role_indices(chains, context=context, target=target)
    pairs = confidence.get("pair_chains_iptm")
    if not isinstance(pairs, Mapping):
        raise InterfaceError("confidence record carries no pair_chains_iptm")
    within_framework = None
    if len(framework) > 1:
        inner = []
        for i, f in enumerate(framework):
            for g in framework[i + 1:]:
                inner.append(float(pairs[str(f)][str(g)]))
                inner.append(float(pairs[str(g)][str(f)]))
        within_framework = sum(inner) / len(inner)
    return {
        "antibody_antigen_iptm": antibody_antigen_iptm(pairs, framework, antigen),
        "framework_internal_iptm": within_framework,
        "global_iptm": float(confidence.get("iptm")) if "iptm" in confidence else None,
        "ptm": float(confidence.get("ptm")) if "ptm" in confidence else None,
        "complex_plddt": (float(confidence["complex_plddt"])
                          if "complex_plddt" in confidence else None),
        "complex_iplddt": (float(confidence["complex_iplddt"])
                           if "complex_iplddt" in confidence else None),
        "complex_ipde": (float(confidence["complex_ipde"])
                         if "complex_ipde" in confidence else None),
        "chains": list(chains),
        "framework_indices": list(framework),
        "antigen_indices": list(antigen),
    }
