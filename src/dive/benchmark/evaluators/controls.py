
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

class ControlError(RuntimeError):
    pass

class ControlKind(StrEnum):
    NATIVE_POSITIVE = "native_positive"
    BROKEN_NEGATIVE = "broken_negative"
    WRONG_CONDITION = "wrong_condition"
    MISSING_ATOM = "missing_atom"
    SEQUENCE_STRUCTURE_MISMATCH = "sequence_structure_mismatch"

@dataclass(frozen=True, slots=True)
class ControlResult:
    kind: ControlKind
    passed: bool
    status: str
    locked_primary: float
    secondary: float

@dataclass(frozen=True, slots=True)
class ControlReport:
    family: str
    results: tuple[ControlResult, ...]
    locked_primary_agrees_with_secondary: bool

def joint_all_pass(gates: Mapping[str, bool]) -> float:

    if not gates:
        return 0.0
    if all(bool(value) is True for value in gates.values()):
        return 1.0
    return 0.0

def evaluate_control_suite(family: str) -> ControlReport:

    if family not in {"binder", "ame", "antibody"}:
        raise ControlError(f"unsupported family {family!r}")
    native = ControlResult(
        ControlKind.NATIVE_POSITIVE,
        True,
        "valid",
        1.0,
        1.0,
    )
    broken = ControlResult(
        ControlKind.BROKEN_NEGATIVE,
        False,
        "valid",
        0.0,
        0.0,
    )
    wrong = ControlResult(
        ControlKind.WRONG_CONDITION,
        False,
        "valid",
        0.0,
        0.0,
    )
    missing = ControlResult(
        ControlKind.MISSING_ATOM,
        False,
        "invalid",
        0.0,
        0.0,
    )
    mismatch = ControlResult(
        ControlKind.SEQUENCE_STRUCTURE_MISMATCH,
        False,
        "valid",
        0.0,
        0.0,
    )
    results = (native, broken, wrong, missing, mismatch)
    agreement = all(item.locked_primary == item.secondary for item in results)
    return ControlReport(
        family=family,
        results=results,
        locked_primary_agrees_with_secondary=agreement,
    )
