
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from dive.phase3_verify.family_verdict import FAMILIES

class HeadroomError(ValueError):
    pass

class Headroom(Enum):
    PRESENT = "present"
    ABSENT = "absent"
    NOT_MEASURABLE = "not_measurable"

MIN_NATIVE_GATE_FRACTION = 0.5

@dataclass(frozen=True)
class HeadroomReading:

    family: str
    lower_is_better: bool
    native_values: Mapping[str, float]
    arm_values: Mapping[str, float]
    native_gate_pass: int
    native_gate_total: int

    arm_replicate_spread: float | None = None

@dataclass(frozen=True)
class HeadroomDecision:
    verdict: Headroom
    reason: str
    gap: float
    detail: Mapping[str, object] = field(default_factory=dict)

def assess_headroom(reading: HeadroomReading) -> HeadroomDecision:

    if reading.family not in FAMILIES:
        raise HeadroomError(f"unknown family {reading.family!r}")
    if set(reading.native_values) != set(reading.arm_values):
        raise HeadroomError(
            "native and arm must be compared on the same tasks; "
            f"native has {sorted(reading.native_values)}, "
            f"arm has {sorted(reading.arm_values)}"
        )
    tasks = tuple(sorted(reading.native_values))
    if not tasks:
        raise HeadroomError("no shared task to compare on")
    if reading.arm_replicate_spread is None:
        raise HeadroomError(
            "the arm's replicate spread is required: a gap has nothing to be "
            "larger than without it, and an unmeasured spread is not zero"
        )
    if reading.arm_replicate_spread < 0:
        raise HeadroomError("replicate spread cannot be negative")
    if reading.native_gate_total <= 0:
        raise HeadroomError("the native gate needs a positive denominator")

    native = sum(reading.native_values[t] for t in tasks) / len(tasks)
    arm = sum(reading.arm_values[t] for t in tasks) / len(tasks)
    gap = (arm - native) if reading.lower_is_better else (native - arm)
    detail = {
        "tasks": tasks,
        "native_mean": native,
        "arm_mean": arm,
        "native_gate": (reading.native_gate_pass, reading.native_gate_total),
        "replicate_spread": reading.arm_replicate_spread,
    }

    fraction = reading.native_gate_pass / reading.native_gate_total
    if fraction < MIN_NATIVE_GATE_FRACTION:
        return HeadroomDecision(
            Headroom.NOT_MEASURABLE,
            "the native control passes its own gate only "
            f"{reading.native_gate_pass}/{reading.native_gate_total}, so the "
            "ceiling is not located and the distance to it is not a number; a "
            "gap measured here would be measuring the evaluator",
            gap,
            detail,
        )
    if gap > reading.arm_replicate_spread:
        return HeadroomDecision(
            Headroom.PRESENT,
            f"the arm sits {gap:.2f} from the native reference, beyond its own "
            f"replicate spread of {reading.arm_replicate_spread:.2f}",
            gap,
            detail,
        )
    return HeadroomDecision(
        Headroom.ABSENT,
        f"the gap of {gap:.2f} is within the arm's own replicate spread of "
        f"{reading.arm_replicate_spread:.2f}, so there is no room to act that "
        "could be distinguished from the arm's own noise",
        gap,
        detail,
    )
