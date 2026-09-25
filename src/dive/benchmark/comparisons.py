
from __future__ import annotations

from dataclasses import dataclass

from dive.benchmark.baselines import (
    BaselineReadiness,
    BaselineRecord,
    BaselineRegistry,
    ComparisonKind,
)

class ComparisonError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ComparisonRow:
    row_id: str
    baseline_id: str
    family: str
    kind: ComparisonKind
    sequence_budget_label: str
    readiness: BaselineReadiness
    score: float | None
    mixes_generator_and_filter: bool

def _row(
    baseline: BaselineRecord,
    family: str,
    kind: ComparisonKind,
    label: str,
) -> ComparisonRow:
    score = None
    return ComparisonRow(
        row_id=f"{kind.value}:{baseline.baseline_id}:{family}:{label}",
        baseline_id=baseline.baseline_id,
        family=family,
        kind=kind,
        sequence_budget_label=label,
        readiness=baseline.readiness,
        score=score,
        mixes_generator_and_filter=kind is ComparisonKind.NATIVE_PIPELINE,
    )

def native_pipeline_rows(registry: BaselineRegistry) -> tuple[ComparisonRow, ...]:
    rows: list[ComparisonRow] = []
    for baseline in registry.baselines:
        if not baseline.native_pipeline:
            continue
        for family in baseline.families:
            rows.append(
                _row(baseline, family, ComparisonKind.NATIVE_PIPELINE, "native")
            )
    return tuple(rows)

def common_redesign_rows(registry: BaselineRegistry) -> tuple[ComparisonRow, ...]:
    rows: list[ComparisonRow] = []
    for baseline in registry.baselines:
        if baseline.role != "generator" or not baseline.common_redesign:
            continue
        for family in baseline.families:
            if baseline.readiness is BaselineReadiness.READY_EXACT:
                for label in ("self", "common_IF_1", "common_IF_K"):
                    rows.append(
                        _row(baseline, family, ComparisonKind.COMMON_REDESIGN, label)
                    )
            else:
                rows.append(
                    _row(baseline, family, ComparisonKind.COMMON_REDESIGN, "self")
                )
    return tuple(rows)

def separate_comparison_tables(
    registry: BaselineRegistry,
) -> tuple[tuple[ComparisonRow, ...], tuple[ComparisonRow, ...]]:
    native = native_pipeline_rows(registry)
    common = common_redesign_rows(registry)
    if {row.row_id for row in native} & {row.row_id for row in common}:
        raise ComparisonError("native and common-redesign row ids collided")
    return native, common
