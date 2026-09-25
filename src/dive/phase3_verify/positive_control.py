
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

_PDB_STYLE = re.compile(r"^pdb_0000([0-9a-z]{4})")
_AME_STYLE = re.compile(r"^ame:([0-9a-z]{4})__")
_BINDER_STYLE = re.compile(r"^binder:([0-9a-z]{4}):")

class ControlOutcome(Enum):

    CONFIRMED = "confirmed"
    MISSED = "missed"
    NOT_TESTED = "not_tested"

@dataclass(frozen=True, slots=True)
class DetectionResult:
    status: ControlOutcome
    n_expected: int
    n_found: int
    missed: frozenset[str]

def entry_of(example_id: str) -> str | None:

    text = str(example_id)
    for pattern in (_PDB_STYLE, _AME_STYLE, _BINDER_STYLE):
        found = pattern.match(text)
        if found:
            return found.group(1)
    return None

def straddling_entries(rows: Iterable[Mapping[str, object]]) -> set[str]:

    sides: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        entry = entry_of(row.get("example_id", ""))
        if entry is None:
            continue
        partition = str(row.get("partition", ""))
        if partition:
            sides[entry].add(partition)
    return {
        entry
        for entry, partitions in sides.items()
        if {"train", "validation"} <= partitions
    }

def verify_detection(*, expected: Iterable[str], detected: Iterable[str]) -> DetectionResult:

    want = frozenset(expected)
    have = frozenset(detected)
    missed = want - have
    if not want:
        status = ControlOutcome.NOT_TESTED
    elif missed:
        status = ControlOutcome.MISSED
    else:
        status = ControlOutcome.CONFIRMED
    return DetectionResult(
        status=status,
        n_expected=len(want),
        n_found=len(want & have),
        missed=missed,
    )

__all__ = (
    "ControlOutcome",
    "DetectionResult",
    "entry_of",
    "straddling_entries",
    "verify_detection",
)
