
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from dive.phase3_verify.coverage import MIN_SHORTER_COVERAGE, MIN_TM_SCORE

class MultimerCoverageRule(Enum):

    MAX = "max"

    MIN = "min"

GOVERNING_MULTIMER_RULE = MultimerCoverageRule.MAX

def _number(value: object) -> float | None:

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed

def multimer_overlaps(row: Mapping[str, object], *, rule: MultimerCoverageRule) -> bool:

    if not isinstance(rule, MultimerCoverageRule):
        raise TypeError("the multimer coverage rule must be named explicitly")

    qtm = _number(row.get("complexqtmscore"))
    ttm = _number(row.get("complexttmscore"))
    qcov = _number(row.get("qcomplexcoverage"))
    tcov = _number(row.get("tcomplexcoverage"))
    if None in (qtm, ttm, qcov, tcov):
        return False

    pick = max if rule is MultimerCoverageRule.MAX else min
    return pick(qtm, ttm) >= MIN_TM_SCORE and pick(qcov, tcov) >= MIN_SHORTER_COVERAGE

__all__ = (
    "GOVERNING_MULTIMER_RULE",
    "MultimerCoverageRule",
    "multimer_overlaps",
)
