
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from dive.necessity_v3.dataflow import assert_verdict_allowed

REGIME_SHARE = 0.10

MIN_FAMILIES = 2

MIN_TARGETS = 10

MATCHED_CONTRAST_MIN = 1e-3

@dataclass(frozen=True)
class Evidence:

    category: str
    support_valid: bool
    integrity_valid: bool
    instrumentation_parity_valid: bool
    families_covered: int
    n_targets: int
    step_size_sign_stable: bool
    sign_survives_leave_one_target_out: bool
    regime_fractions: Mapping[str, float]
    matched_region_contrast_nonzero: bool
    any_primary_interval_excludes_zero: bool
    path_effects_selective: bool

@dataclass(frozen=True)
class Decision:

    verdict: str
    reason: str
    criteria: dict = field(default_factory=dict)

def _substantial_regimes(fractions: Mapping[str, float]) -> list[str]:
    return sorted(k for k, v in fractions.items() if v >= REGIME_SHARE)

def decide(evidence: Evidence) -> Decision:

    reproducible = (
        evidence.step_size_sign_stable
        and evidence.sign_survives_leave_one_target_out
        and evidence.any_primary_interval_excludes_zero
    )
    regimes = _substantial_regimes(evidence.regime_fractions)
    heterogeneous = len(regimes) >= 2 and evidence.matched_region_contrast_nonzero
    criteria = {
        "reproducible": reproducible,
        "heterogeneous": heterogeneous,
        "substantial_regimes": regimes,
        "step_size_sign_stable": evidence.step_size_sign_stable,
        "sign_survives_leave_one_target_out":
            evidence.sign_survives_leave_one_target_out,
        "matched_region_contrast_nonzero":
            evidence.matched_region_contrast_nonzero,
        "any_primary_interval_excludes_zero":
            evidence.any_primary_interval_excludes_zero,
    }

    if not evidence.support_valid:
        return Decision("INCONCLUSIVE",
                        "the probe grid is not inside the training corruption "
                        "support", criteria)
    if not evidence.integrity_valid:
        return Decision("INCONCLUSIVE",
                        "frozen-model integrity failed; a parameter changed "
                        "during the run", criteria)
    if not evidence.instrumentation_parity_valid:
        return Decision("INCONCLUSIVE",
                        "instrumentation parity failed at the default scalars",
                        criteria)
    if evidence.families_covered < MIN_FAMILIES:
        return Decision("INCONCLUSIVE",
                        f"only {evidence.families_covered} benchmark family on "
                        "one common parameter set; the scope is insufficient",
                        criteria)
    if evidence.n_targets < MIN_TARGETS:
        return Decision("INCONCLUSIVE",
                        f"{evidence.n_targets} independent targets is too few "
                        "for target-clustered inference", criteria)

    if evidence.category == "P1":
        if reproducible and heterogeneous and evidence.path_effects_selective:
            return Decision("PROCEED_TO_PROBE",
                            "P1 architecture with reproducible, heterogeneous "
                            "value at both the data and path level, and "
                            "selective realized path effects", criteria)
        return Decision("STOP_ROUTER",
                        "the architecture is P1, but the path-level evidence "
                        "does not meet the reproducibility, heterogeneity and "
                        "selectivity bar", criteria)

    if reproducible and heterogeneous:
        decision = Decision(
            "VALUE_ONLY",
            "on-manifold directional value is reproducible across step sizes "
            "and survives target-level controls, and it is heterogeneous "
            f"across regimes {regimes}; but the architecture is "
            f"{evidence.category}, so internal causal leadership is not "
            "identifiable", criteria)
    else:
        missing = [k for k in ("reproducible", "heterogeneous") if not criteria[k]]
        decision = Decision(
            "STOP_ROUTER",
            "directional value fails the "
            f"{' and '.join(missing)} requirement(s), so no routing conclusion "
            "is supported", criteria)
    assert_verdict_allowed(evidence.category, decision.verdict)
    return decision
