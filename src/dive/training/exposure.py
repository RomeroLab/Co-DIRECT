
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

class ExposureError(ValueError):
    pass

INELIGIBLE_PREFIXES = ("excluded-",)

_UNIT_NOTE = (
    "Counts are given as parents and examples. The parent cluster is the "
    "independent unit: repeated crops or corruptions of one design target are "
    "one observation, so example counts overstate the power available for any "
    "interval computed on them."
)

@dataclass
class ExposureRegistry:

    partitions: dict[str, dict[str, int]]
    _exposure: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(self, partition: str, *, activity: str, detail: str | None = None) -> None:

        if partition not in self.partitions:
            raise ExposureError(
                f"unknown partition {partition!r}; the frozen contract declares "
                f"{sorted(self.partitions)} and inventing one would put a claim "
                f"on data whose provenance is not recorded"
            )
        self._exposure.setdefault(partition, []).append(
            {"activity": activity, "detail": detail}
        )

    def activities(self, partition: str) -> list[str]:
        return [entry["activity"] for entry in self._exposure.get(partition, [])]

    def is_exposed(self, partition: str) -> bool:
        return bool(self._exposure.get(partition))

    def ineligible_reason(self, partition: str) -> str | None:

        if partition.startswith(INELIGIBLE_PREFIXES):
            return (
                "holds homology neighbours of the split it protects; proximity "
                "to the confirmation set is why it exists, so a result confirmed "
                "here is leakage under a clean partition name"
            )
        if self.is_exposed(partition):
            return (
                "already exposed to "
                + ", ".join(sorted(set(self.activities(partition))))
                + "; a result confirmed on it restates a choice rather than "
                "testing it"
            )
        return None

    def as_provenance(self) -> dict[str, Any]:
        return {
            "unit_note": _UNIT_NOTE,
            "partitions": {
                name: {
                    **counts,
                    "exposure": list(self._exposure.get(name, [])),
                    "exposed": self.is_exposed(name),
                    "eligible_for_confirmation": self.ineligible_reason(name) is None,
                    "ineligible_reason": self.ineligible_reason(name),
                }
                for name, counts in sorted(self.partitions.items())
            },
            "confirmation_candidates": confirmation_candidates(self),
        }

def confirmation_candidates(registry: ExposureRegistry) -> list[str]:

    return sorted(
        name
        for name in registry.partitions
        if registry.ineligible_reason(name) is None
    )
