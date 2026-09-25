
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from dive.phase3_verify.family_verdict import FAMILIES, Answer

class Q2Error(ValueError):
    pass

class InterventionKind(Enum):

    MAGNITUDE_PRESERVING_SCRAMBLE = "magnitude_preserving_scramble"

    ABLATE_TO_ZERO = "ablate_to_zero"

    FRESH_VALUES = "fresh_values"

_ADMISSIBLE = {InterventionKind.MAGNITUDE_PRESERVING_SCRAMBLE}

EARLY_FRACTION = 0.25

@dataclass(frozen=True)
class Q2Reading:

    family: str
    kind: InterventionKind
    read_steps: Sequence[int]
    total_steps: int
    effect_ci_low: float
    effect_ci_high: float
    null_ci_low: float | None = None
    null_ci_high: float | None = None

    equivalence_margin: float | None = None

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise Q2Error(f"unknown family {self.family!r}")
        if self.kind not in _ADMISSIBLE:
            raise Q2Error(
                f"{self.kind.value} is not an admissible intervention: it changes "
                "the channel's magnitude as well as its correspondence, so a "
                "change in the output cannot be attributed to the correspondence"
            )
        if self.total_steps <= 0:
            raise Q2Error("total_steps must be positive")
        if not self.read_steps:
            raise Q2Error("no read step declared; Q2 reads a trajectory")
        for step in self.read_steps:
            if not 0 <= step < self.total_steps:
                raise Q2Error(
                    f"read step {step} is outside the trajectory "
                    f"[0, {self.total_steps})"
                )
        cutoff = self.total_steps * EARLY_FRACTION
        if not any(step < cutoff for step in self.read_steps):
            raise Q2Error(
                f"no read step falls in the first quarter (< {cutoff:.0f} of "
                f"{self.total_steps}); the design is set there, so a schedule "
                "that starts later measures after the decision"
            )
        if self.effect_ci_low > self.effect_ci_high:
            raise Q2Error("effect interval is inverted")
        if (self.null_ci_low is None) != (self.null_ci_high is None):
            raise Q2Error("the null interval is half specified")
        if self.null_ci_low is not None and self.null_ci_low > self.null_ci_high:
            raise Q2Error("null interval is inverted")
        if self.equivalence_margin is not None and self.equivalence_margin <= 0:
            raise Q2Error("the equivalence margin must be positive")

@dataclass(frozen=True)
class Q2Decision:
    answer: Answer
    reason: str
    detail: Mapping[str, object] = field(default_factory=dict)

def _excludes_zero(low: float, high: float) -> bool:
    return low > 0.0 or high < 0.0

def decide_q2(reading: Q2Reading) -> Q2Decision:

    reading.validate()
    detail = {
        "family": reading.family,
        "intervention": reading.kind.value,
        "read_steps": tuple(reading.read_steps),
        "total_steps": reading.total_steps,
        "effect": (reading.effect_ci_low, reading.effect_ci_high),
        "null": (reading.null_ci_low, reading.null_ci_high),
        "equivalence_margin": reading.equivalence_margin,
    }

    if reading.null_ci_low is None:
        return Q2Decision(
            Answer.INCONCLUSIVE,
            "no null intervention was run, so a pipeline that reacts to any "
            "perturbation cannot be distinguished from one that reads this "
            "channel",
            detail,
        )
    if _excludes_zero(reading.null_ci_low, reading.null_ci_high):
        return Q2Decision(
            Answer.INCONCLUSIVE,
            "the null intervention also moves the output "
            f"({reading.null_ci_low:+.3f}, {reading.null_ci_high:+.3f}); the "
            "effect cannot be attributed to this channel",
            detail,
        )
    if not _excludes_zero(reading.effect_ci_low, reading.effect_ci_high):
        margin = reading.equivalence_margin
        if margin is not None and -margin <= reading.effect_ci_low and                reading.effect_ci_high <= margin:
            return Q2Decision(
                Answer.CONTRADICTED,
                "the effect is bounded inside the declared margin of "
                f"{margin:+.3f} ({reading.effect_ci_low:+.3f}, "
                f"{reading.effect_ci_high:+.3f}), so the change is negligible "
                "rather than merely unmeasured",
                detail,
            )
        detail_reason = (
            "the intervention's effect interval contains zero "
            f"({reading.effect_ci_low:+.3f}, {reading.effect_ci_high:+.3f})"
        )
        if margin is None:
            detail_reason += (
                "; with no declared equivalence margin this cannot be read as "
                "the path ignoring the channel"
            )
        else:
            detail_reason += (
                f"and is wider than the declared margin {margin:+.3f}, so the "
                "test lacked the power to bound the effect"
            )
        return Q2Decision(Answer.INCONCLUSIVE, detail_reason, detail)
    return Q2Decision(
        Answer.SUPPORTED,
        "a magnitude-preserving scramble reliably changes the generated output "
        "while the null intervention does not, so the generation path reads this "
        "channel",
        detail,
    )
