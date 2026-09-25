
from __future__ import annotations

import math
from collections.abc import Sequence

class TimeGridError(ValueError):
    pass

Cell = tuple[int, int]

def _checked(values: Sequence[float], cells: Sequence[Cell]) -> None:
    if len(values) != len(cells):
        raise TimeGridError(
            f"length mismatch: {len(values)} values, {len(cells)} cells"
        )
    if not values:
        raise TimeGridError("cannot aggregate an empty set of observations")
    for value in values:
        if value is None or not math.isfinite(float(value)):
            raise TimeGridError(f"observation must be finite, got {value!r}")

def cell_means(values: Sequence[float], cells: Sequence[Cell]) -> dict[Cell, float]:

    _checked(values, cells)
    totals: dict[Cell, list[float]] = {}
    for value, cell in zip(values, cells):
        totals.setdefault(tuple(cell), []).append(float(value))
    return {cell: sum(v) / len(v) for cell, v in totals.items()}

def equal_cell_mean(values: Sequence[float], cells: Sequence[Cell]) -> float:

    per_cell = cell_means(values, cells)
    return sum(per_cell.values()) / len(per_cell)

def sampler_weighted_mean(values: Sequence[float]) -> float:

    if not values:
        raise TimeGridError("cannot aggregate an empty set of observations")
    return sum(float(v) for v in values) / len(values)

def weighting_disagreement(
    values: Sequence[float],
    cells: Sequence[Cell],
    *,
    expected_cells: int | None = None,
) -> dict:

    equal = equal_cell_mean(values, cells)
    weighted = sampler_weighted_mean(values)
    observed = len(cell_means(values, cells))
    return {
        "equal_cell": equal,
        "sampler_weighted": weighted,
        "ratio_sampler_over_equal": (weighted / equal) if equal != 0 else None,
        "cells": observed,
        "expected_cells": expected_cells,
        "complete_grid": (None if expected_cells is None
                          else observed == expected_cells),
        "observations": len(values),
    }

__all__ = [
    "Cell",
    "TimeGridError",
    "cell_means",
    "equal_cell_mean",
    "sampler_weighted_mean",
    "weighting_disagreement",
]
