
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MIN_RELATIVE_IMPROVEMENT = 0.02

MIN_ORACLE_RECOVERY = 0.30

MIN_ORACLE_GAP = 0.002

_ORACLE_CAVEAT = (
    "The oracle selects the best regime per example using the CLEAN TARGET. It "
    "is an optimistic reference that no deployable model can reach, not a "
    "baseline. Recovering a fraction of the gap to it describes available "
    "headroom and says nothing about absolute design quality."
)

class GateError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class GateVerdict:

    routed: float
    best_fixed: float
    oracle: float
    panel_hash: str
    relative_improvement: float
    oracle_gap: float
    oracle_recovery: float | None
    passed: bool
    undetermined: bool
    reasons: list[str] = field(default_factory=list)

    def as_provenance(self) -> dict[str, Any]:
        return {
            "routed": self.routed,
            "best_fixed": self.best_fixed,
            "oracle": self.oracle,
            "panel_hash": self.panel_hash,
            "relative_improvement": self.relative_improvement,
            "oracle_gap": self.oracle_gap,
            "oracle_recovery": self.oracle_recovery,
            "passed": self.passed,
            "undetermined": self.undetermined,
            "reasons": list(self.reasons),
            "thresholds": {
                "min_relative_improvement": MIN_RELATIVE_IMPROVEMENT,
                "min_oracle_recovery": MIN_ORACLE_RECOVERY,
                "min_oracle_gap": MIN_ORACLE_GAP,
            },
            "oracle_caveat": _ORACLE_CAVEAT,
            "rule": (
                "docs/superpowers/plans/2026-08-19-dive-emergent-leadership-"
                "implementation.md line 1374"
            ),
        }

def routing_gate_verdict(
    *,
    routed: float,
    best_fixed: float,
    oracle: float,
    routed_panel_hash: str,
    best_fixed_panel_hash: str,
    oracle_panel_hash: str,
) -> GateVerdict:

    hashes = {routed_panel_hash, best_fixed_panel_hash, oracle_panel_hash}
    if len(hashes) != 1:
        raise GateError(
            f"the three losses come from different panels {sorted(hashes)}. The "
            f"metric was measured drifting 3.2-7.1% between panels, which is "
            f"larger than the 2% this gate decides on, so they are not comparable."
        )

    if best_fixed <= 0:
        raise GateError(
            f"best_fixed must be positive to form a relative improvement, got "
            f"{best_fixed}"
        )
    if oracle > best_fixed:
        raise GateError(
            f"oracle {oracle} is worse than the best fixed regime {best_fixed}. "
            f"The oracle is a per-example minimum over the same regimes, so it "
            f"cannot lose to any single global one -- this means the two were "
            f"computed differently."
        )

    improvement = (best_fixed - routed) / best_fixed
    gap = best_fixed - oracle

    reasons: list[str] = []
    undetermined = gap < MIN_ORACLE_GAP
    recovery = None if undetermined else (best_fixed - routed) / gap

    if improvement < MIN_RELATIVE_IMPROVEMENT:
        reasons.append(
            f"relative improvement over the best fixed regime is "
            f"{improvement:.4%}, below the approved {MIN_RELATIVE_IMPROVEMENT:.0%}"
        )
    if undetermined:
        reasons.append(
            f"the oracle gap is {gap:.6f}, below {MIN_ORACLE_GAP}: the best "
            f"fixed regime is already at the oracle on this panel, so the "
            f"recovery fraction is not measurable and the gate is UNDETERMINED "
            f"rather than failed"
        )
    elif recovery < MIN_ORACLE_RECOVERY:
        reasons.append(
            f"oracle gap recovery is {recovery:.4%}, below the approved "
            f"{MIN_ORACLE_RECOVERY:.0%}"
        )

    return GateVerdict(
        routed=routed,
        best_fixed=best_fixed,
        oracle=oracle,
        panel_hash=routed_panel_hash,
        relative_improvement=improvement,
        oracle_gap=gap,
        oracle_recovery=recovery,
        passed=not reasons,
        undetermined=undetermined,
        reasons=reasons,
    )
