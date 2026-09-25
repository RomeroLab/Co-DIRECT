
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from dive.phase3_verify.family_verdict import Answer

class Q3Error(ValueError):
    pass

class Direction(Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

class Role(Enum):

    IMPROVEMENT = "improvement"

    PRESERVATION = "preservation"

@dataclass(frozen=True)
class MetricEffect:

    name: str
    role: Role
    direction: Direction
    estimate: float
    ci_low: float
    ci_high: float

    def validate(self) -> None:
        if self.ci_low > self.ci_high:
            raise Q3Error(f"{self.name}: interval is inverted")
        if not (self.ci_low <= self.estimate <= self.ci_high):
            raise Q3Error(f"{self.name}: estimate lies outside its own interval")

    @property
    def verdict(self) -> str:

        if self.ci_low <= 0.0 <= self.ci_high:
            return "contains zero"
        positive = self.ci_low > 0.0
        improving = positive == (self.direction is Direction.HIGHER_IS_BETTER)
        return "better" if improving else "worse"

@dataclass(frozen=True)
class Q3Decision:
    answer: Answer
    reason: str
    per_metric: Mapping[str, str]
    joint_rate: float | None

def decide_q3(
    declared: Sequence[str],
    observed: Mapping[str, MetricEffect],
    *,
    joint_rate: float | None = None,
) -> Q3Decision:

    if not declared:
        raise Q3Error("no declared metrics; Q3 cannot default to an answer")
    declared = tuple(declared)
    if len(set(declared)) != len(declared):
        raise Q3Error("the declared metric set repeats a name")

    undeclared = sorted(set(observed) - set(declared))
    if undeclared:
        raise Q3Error(
            "metrics observed but not declared in advance: "
            f"{', '.join(undeclared)}"
        )

    missing = [name for name in declared if name not in observed]
    per_metric: dict[str, str] = {}
    for name in declared:
        effect = observed.get(name)
        if effect is None:
            per_metric[name] = "not measured"
            continue
        effect.validate()
        per_metric[name] = effect.verdict

    if missing:
        return Q3Decision(
            Answer.INCONCLUSIVE,
            "declared metrics were not measured, and the set must not shrink to "
            f"what was: {', '.join(missing)}",
            per_metric,
            joint_rate,
        )

    broken = sorted(
        name for name in declared
        if observed[name].role is Role.PRESERVATION
        and observed[name].verdict == "worse"
    )
    if broken:
        return Q3Decision(
            Answer.CONTRADICTED,
            f"preservation is significantly worse on {', '.join(broken)}; "
            "Q3 requires improvement AND preservation",
            per_metric,
            joint_rate,
        )

    improvement = [n for n in declared if observed[n].role is Role.IMPROVEMENT]
    better = sorted(n for n in improvement if observed[n].verdict == "better")
    worse = sorted(n for n in improvement if observed[n].verdict == "worse")

    if better and worse:
        return Q3Decision(
            Answer.INCONCLUSIVE,
            f"metrics move in both directions: better on {', '.join(better)}, "
            f"worse on {', '.join(worse)}; a trade is not an improvement",
            per_metric,
            joint_rate,
        )
    if better:
        return Q3Decision(
            Answer.SUPPORTED,
            f"improved on {', '.join(better)} with no metric significantly worse",
            per_metric,
            joint_rate,
        )
    if worse:
        return Q3Decision(
            Answer.CONTRADICTED,
            f"worse on {', '.join(worse)} with no metric significantly better",
            per_metric,
            joint_rate,
        )
    return Q3Decision(
        Answer.INCONCLUSIVE,
        "every interval contains zero",
        per_metric,
        joint_rate,
    )
