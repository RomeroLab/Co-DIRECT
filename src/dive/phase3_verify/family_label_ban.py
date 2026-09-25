
from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence

from dive.phase3_verify.family_verdict import FAMILIES

class FamilyLabelLeak(ValueError):
    pass

_BANNED_TOKENS = frozenset({
    "family", "families", "familyid", "router", "routing", "route",
    "expert", "experts", "moe", "gate",
})

def _tokens(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t}

def assert_no_family_label(
    features: Sequence[str], *, declared: Collection[str]
) -> None:

    if not features:
        raise FamilyLabelLeak(
            "no features given; an empty set must not read as a clean one"
        )
    declared_set = set(declared)
    undeclared = sorted(set(features) - declared_set)
    if undeclared:
        raise FamilyLabelLeak(
            "features were not declared in advance, and the check cannot vouch "
            f"for what it was not shown: {', '.join(undeclared)}"
        )
    banned = sorted(
        name for name in features if _tokens(name) & _BANNED_TOKENS
    )
    if banned:
        raise FamilyLabelLeak(
            "these inputs name a family assignment or a routing decision, which "
            f"the goal forbids and a declaration cannot permit: {', '.join(banned)}"
        )
    return None

MIN_OBSERVATIONS = 30

def refuse_family_separable(
    name: str,
    values_by_family: Mapping[str, Collection],
    *,
    observations: Mapping[str, int] | None = None,
) -> None:

    seen = set(values_by_family)
    if not seen <= set(FAMILIES):
        raise FamilyLabelLeak(f"unknown family in {sorted(seen)}")
    if len(seen) < 2:
        raise FamilyLabelLeak(
            "separability needs at least two families to compare; "
            f"only {sorted(seen)} was given"
        )
    for family, values in values_by_family.items():
        if not values:
            raise FamilyLabelLeak(
                f"{name}: no values observed for {family}, so the feature cannot "
                "be cleared"
            )
    families = sorted(values_by_family)
    for i, left in enumerate(families):
        for right in families[i + 1:]:
            if set(values_by_family[left]) & set(values_by_family[right]):
                return None
    if observations is not None:
        missing = sorted(set(values_by_family) - set(observations))
        if missing:
            raise FamilyLabelLeak(
                f"{name}: no observation count for {', '.join(missing)}, and a "
                "missing count must not be read as a large one"
            )
        thin = sorted(f for f in values_by_family
                      if observations[f] < MIN_OBSERVATIONS)
        if thin:
            raise FamilyLabelLeak(
                f"{name} separates the families, but on too few observations to "
                f"judge ({', '.join(f'{f}={observations[f]}' for f in thin)} "
                f"below {MIN_OBSERVATIONS}); a spread-out integer feature "
                "separates a small sample by default"
            )
    raise FamilyLabelLeak(
        f"{name} separates {', '.join(families)} perfectly: no value is shared "
        "by two families, so it carries the family identity whatever it is called"
    )
