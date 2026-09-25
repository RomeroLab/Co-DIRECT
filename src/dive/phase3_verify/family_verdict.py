
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

FAMILIES = ("binder", "ame", "antibody")

NOT_IDENTIFIABLE = {("antibody", "binding")}

class Stage(Enum):
    Q1 = "q1_readable_by_probe"
    Q2 = "q2_used_by_generation_path"
    Q3 = "q3_improves_joint_outcome"

class Answer(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"

class FamilyVerdict(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    NOT_IDENTIFIABLE = "not_identifiable"

@dataclass(frozen=True, slots=True)
class FamilyDecision:
    verdict: FamilyVerdict
    reason: str
    stages_consulted: tuple[Stage, ...]

_ORDER = (Stage.Q1, Stage.Q2, Stage.Q3)

def decide_family(
    family: str,
    evidence: Mapping,
    *,
    endpoint: str | None = None,
) -> FamilyDecision:

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")

    if (family, endpoint) in NOT_IDENTIFIABLE:
        return FamilyDecision(
            FamilyVerdict.NOT_IDENTIFIABLE,
            f"{family}/{endpoint} cannot express this claim; no amount of "
            f"evidence makes it reachable",
            (),
        )

    consulted: list[Stage] = []
    for stage in _ORDER:
        consulted.append(stage)
        answer = evidence.get(stage)
        if answer is None:
            return FamilyDecision(
                FamilyVerdict.INCONCLUSIVE,
                f"{stage.value} was not measured",
                tuple(consulted),
            )
        if answer is Answer.CONTRADICTED:
            return FamilyDecision(
                FamilyVerdict.CONTRADICTED,
                f"{stage.value} is contradicted; the later questions are not "
                f"consulted because their answers would not be interpretable",
                tuple(consulted),
            )
        if answer is Answer.INCONCLUSIVE:
            return FamilyDecision(
                FamilyVerdict.INCONCLUSIVE,
                f"{stage.value} is inconclusive; the later questions are not "
                f"consulted",
                tuple(consulted),
            )

    preservation = evidence.get("preservation")
    if preservation is None:
        return FamilyDecision(
            FamilyVerdict.INCONCLUSIVE,
            "all three questions are supported but preservation was not "
            "measured; an unmeasured preservation cannot be read as held",
            tuple(consulted),
        )
    if preservation is False:
        return FamilyDecision(
            FamilyVerdict.CONTRADICTED,
            "all three questions are supported but preservation failed; the "
            "endpoint moved at the cost of what had to be kept",
            tuple(consulted),
        )
    return FamilyDecision(
        FamilyVerdict.SUPPORTED,
        "readable by a probe, used by the generation path, improves the joint "
        "outcome, and preservation held",
        tuple(consulted),
    )

__all__ = (
    "Answer",
    "FAMILIES",
    "FamilyDecision",
    "FamilyVerdict",
    "NOT_IDENTIFIABLE",
    "Stage",
    "decide_family",
)
