
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MIN_TM = 0.50
DEFAULT_MIN_COVERAGE = 0.80

_MID = 0.70
_HIGH = 0.90
_NEAR_IDENTICAL = 0.95

_QUERY, _QLEN, _TLEN, _QCOV, _TCOV, _QTM, _TTM = 0, 2, 3, 5, 6, 7, 8
_EXPECTED_FIELDS = 14

@dataclass(frozen=True, slots=True)
class TierCounts:

    entities_seen: int
    entities_with_edge: int
    tier_fold: int
    tier_mid: int
    tier_high: int
    tier_near_identical: int
    entities_without_edge: int
    skipped_rows: int = 0

    def rate(self, *, total_entities: int) -> float:

        if total_entities <= 0:
            return 0.0
        return self.entities_with_edge / total_entities

    def near_identical_rate(self, *, total_entities: int) -> float:

        if total_entities <= 0:
            return 0.0
        return self.tier_near_identical / total_entities

def tier_monomer_overlap(
    tsv_path: Path | str,
    *,
    min_tm: float = DEFAULT_MIN_TM,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> TierCounts:

    best: dict[str, float] = {}
    seen: set[str] = set()
    skipped = 0

    with open(tsv_path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != _EXPECTED_FIELDS:
                skipped += 1
                continue
            entity = fields[_QUERY].split("_", 1)[0]
            seen.add(entity)
            try:
                qlen = int(fields[_QLEN])
                tlen = int(fields[_TLEN])
                if qlen <= tlen:
                    tm = float(fields[_QTM])
                    coverage = float(fields[_QCOV])
                else:
                    tm = float(fields[_TTM])
                    coverage = float(fields[_TCOV])
            except ValueError:
                skipped += 1
                continue
            if tm != tm or coverage != coverage:
                continue
            if tm < min_tm or coverage < min_coverage:
                continue
            if tm > best.get(entity, -1.0):
                best[entity] = tm

    fold = mid = high = near = 0
    for value in best.values():
        if value >= _NEAR_IDENTICAL:
            near += 1
        elif value >= _HIGH:
            high += 1
        elif value >= _MID:
            mid += 1
        else:
            fold += 1

    return TierCounts(
        entities_seen=len(seen),
        entities_with_edge=len(best),
        tier_fold=fold,
        tier_mid=mid,
        tier_high=high,
        tier_near_identical=near,
        entities_without_edge=len(seen) - len(best),
        skipped_rows=skipped,
    )

__all__ = ("DEFAULT_MIN_COVERAGE", "DEFAULT_MIN_TM", "TierCounts", "tier_monomer_overlap")
