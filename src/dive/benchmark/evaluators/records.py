
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

from dive.benchmark.evaluators.contracts import MetricUnavailable, MetricValue

MetricResult: TypeAlias = MetricValue | MetricUnavailable
RawValue: TypeAlias = (
    str | bool | float | tuple[str, ...] | tuple[bool, ...] | tuple[float, ...]
)
Runner: TypeAlias = Callable[..., object]

def validate_raw_value(value: object) -> RawValue:

    if type(value) is str or type(value) is bool:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite_metric")
        return value
    if isinstance(value, tuple):
        if not value:
            raise ValueError("raw vectors must not be empty")
        kinds = {type(item) for item in value}
        if len(kinds) != 1 or kinds.pop() not in {str, bool, float}:
            raise TypeError("raw vectors must have one exact scalar type")
        if type(value[0]) is float and not all(math.isfinite(item) for item in value):
            raise ValueError("nonfinite_metric")
        return value
    raise TypeError("raw values must be strings, booleans, floats, or uniform tuples")

@dataclass(frozen=True, slots=True)
class FamilyMetricRecord:

    family: str
    primary: MetricResult
    raw: Mapping[str, RawValue] = field(default_factory=dict)
    optional: Mapping[str, MetricResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in {"binder", "ame"}:
            raise ValueError(f"unsupported evaluator family: {self.family}")
        if not isinstance(self.primary, (MetricValue, MetricUnavailable)):
            raise TypeError("primary must be a MetricValue or MetricUnavailable")
        if isinstance(self.primary, MetricValue):
            if type(self.primary.value) is not float or not math.isfinite(self.primary.value):
                raise ValueError("primary metric must be a finite float")
        if not isinstance(self.raw, Mapping) or not isinstance(self.optional, Mapping):
            raise TypeError("raw and optional records must be mappings")
        checked_raw: dict[str, RawValue] = {}
        for name, value in self.raw.items():
            if type(name) is not str or not name:
                raise ValueError("raw metric names must be non-empty strings")
            checked_raw[name] = validate_raw_value(value)
        checked_optional: dict[str, MetricResult] = {}
        for name, metric in self.optional.items():
            if type(name) is not str or not name:
                raise ValueError("optional metric names must be non-empty strings")
            if not isinstance(metric, (MetricValue, MetricUnavailable)):
                raise TypeError("optional records must be metric records")
            if isinstance(metric, MetricValue) and (
                type(metric.value) is not float or not math.isfinite(metric.value)
            ):
                raise ValueError("optional metric must be a finite float")
            checked_optional[name] = metric
        object.__setattr__(self, "raw", MappingProxyType(checked_raw))
        object.__setattr__(self, "optional", MappingProxyType(checked_optional))
