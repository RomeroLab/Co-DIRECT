
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

MIN_TM_SCORE = 0.50
MIN_SHORTER_COVERAGE = 0.80

@dataclass(frozen=True, slots=True)
class CoverageComparison:
    rows: int
    spec_edges: int
    impl_edges: int
    spec_only: int
    impl_only: int
    rows_spec_coverage_above_one: int

    @property
    def nested(self) -> bool:

        return self.impl_only == 0

def _shorter_is_query(row: Mapping[str, object]) -> bool:
    return int(row["qlen"]) <= int(row["tlen"])

def spec_coverage(row: Mapping[str, object]) -> float:

    shorter = min(int(row["qlen"]), int(row["tlen"]))
    return int(row["alnlen"]) / shorter

def impl_coverage(row: Mapping[str, object]) -> float:

    return float(row["qcov"] if _shorter_is_query(row) else row["tcov"])

def shorter_tm(row: Mapping[str, object]) -> float:
    return float(row["qtmscore"] if _shorter_is_query(row) else row["ttmscore"])

def compare_coverage_definitions(
    rows: Iterable[Mapping[str, object]],
    *,
    min_tm: float = MIN_TM_SCORE,
    min_coverage: float = MIN_SHORTER_COVERAGE,
) -> CoverageComparison:

    n = spec = impl = spec_only = impl_only = above_one = 0
    for row in rows:
        n += 1
        s_cov = spec_coverage(row)
        i_cov = impl_coverage(row)
        if s_cov > 1.0:
            above_one += 1
        tm_ok = shorter_tm(row) >= min_tm
        s_edge = tm_ok and s_cov >= min_coverage
        i_edge = tm_ok and i_cov >= min_coverage
        spec += s_edge
        impl += i_edge
        if s_edge and not i_edge:
            spec_only += 1
        elif i_edge and not s_edge:
            impl_only += 1
    return CoverageComparison(
        rows=n,
        spec_edges=spec,
        impl_edges=impl,
        spec_only=spec_only,
        impl_only=impl_only,
        rows_spec_coverage_above_one=above_one,
    )

__all__ = (
    "CoverageComparison",
    "MIN_SHORTER_COVERAGE",
    "MIN_TM_SCORE",
    "compare_coverage_definitions",
    "impl_coverage",
    "shorter_tm",
    "spec_coverage",
)
