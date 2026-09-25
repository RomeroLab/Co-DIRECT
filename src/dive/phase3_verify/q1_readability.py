
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from dive.phase3_verify.family_verdict import FAMILIES, Answer

class Q1Error(ValueError):
    pass

NULL_TOLERANCE = 0.05

@dataclass(frozen=True)
class ProbeReading:

    family: str
    n: int
    r2_representation: float
    r2_shuffled_null: float
    r2_own_coords: float
    effect_ci_low: float
    effect_ci_high: float
    half_effects: tuple[float, ...] | None = None

    def validate(self) -> None:
        if self.family not in FAMILIES:
            raise Q1Error(f"unknown family {self.family!r}")
        if self.n <= 0:
            raise Q1Error("a reading needs a positive number of residues")
        if self.effect_ci_low > self.effect_ci_high:
            raise Q1Error("effect interval is inverted")

@dataclass(frozen=True)
class Q1Decision:
    answer: Answer
    reason: str
    detail: Mapping[str, object] = field(default_factory=dict)

def decide_q1(reading: ProbeReading) -> Q1Decision:

    reading.validate()
    detail = {
        "family": reading.family,
        "n": reading.n,
        "r2_representation": reading.r2_representation,
        "r2_shuffled_null": reading.r2_shuffled_null,
        "r2_own_coords": reading.r2_own_coords,
        "effect_vs_own_coordinates": (reading.effect_ci_low, reading.effect_ci_high),
        "half_effects": reading.half_effects,
    }

    if abs(reading.r2_shuffled_null) > NULL_TOLERANCE:
        return Q1Decision(
            Answer.INCONCLUSIVE,
            f"the shuffled-label null is {reading.r2_shuffled_null:+.3f}, outside "
            f"+/-{NULL_TOLERANCE}; a null that far from zero means the decoder is "
            "fitting noise, so nothing beside it can be read",
            detail,
        )

    excludes_zero = reading.effect_ci_low > 0.0 or reading.effect_ci_high < 0.0
    if not excludes_zero:
        return Q1Decision(
            Answer.INCONCLUSIVE,
            "the advantage over own coordinates has an interval containing zero "
            f"({reading.effect_ci_low:+.3f}, {reading.effect_ci_high:+.3f})",
            detail,
        )

    improving = reading.effect_ci_low > 0.0

    if reading.half_effects is None:
        return Q1Decision(
            Answer.INCONCLUSIVE,
            "the split-half check was not run; an effect that was not shown to "
            "survive halving the sample is not taken as stable",
            detail,
        )
    if not reading.half_effects:
        raise Q1Error("half_effects is empty; pass None if the check was not run")
    if any((half > 0.0) != improving for half in reading.half_effects):
        return Q1Decision(
            Answer.INCONCLUSIVE,
            f"the effect does not hold on each half {reading.half_effects}; "
            "an effect that changes sign when the sample is halved is overfitting",
            detail,
        )

    if improving:
        return Q1Decision(
            Answer.SUPPORTED,
            "the representation beats its own-coordinate control with a null at "
            "zero and the effect holds on both halves",
            detail,
        )
    return Q1Decision(
        Answer.CONTRADICTED,
        "the representation is significantly WORSE than its own-coordinate "
        "control, so the read point carries less than the input it was given",
        detail,
    )

def refuse_cross_family_ranking(scores: Mapping[str, float]) -> None:

    raise Q1Error(
        "families cannot be ranked by R2: it is normalised by each family's own "
        "label variance and they do not share one, so the ordering is an artifact "
        f"of the label range (asked to rank {sorted(scores)})"
    )
