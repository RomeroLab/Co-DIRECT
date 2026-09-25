
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

class CoordinationError(RuntimeError):
    pass

class CoordinationState(StrEnum):
    BACKBONE = "B"
    SEQUENCE = "S"
    SIDECHAIN = "A"
    CHEMISTRY = "C"

class CoordinationMode(StrEnum):
    B_TO_C = "B_to_C"
    C_TO_B = "C_to_B"
    JOINT = "joint"
    HOLD = "hold"

@dataclass(frozen=True, slots=True)
class IndependentGates:
    modes: MappingProxyType
    states: tuple[CoordinationState, ...]
    is_hypothesis: bool = True

    def as_label(self) -> None:
        raise CoordinationError("coordination modes are not task labels")

def independent_gates(
    *,
    backbone_to_chemistry: float,
    chemistry_to_backbone: float,
    joint: float | None = None,
    hold: float | None = None,
) -> IndependentGates:

    for name, value in (
        ("backbone_to_chemistry", backbone_to_chemistry),
        ("chemistry_to_backbone", chemistry_to_backbone),
    ):
        if type(value) is not float and type(value) is not int:
            raise CoordinationError(f"{name} must be numeric")
        if not 0.0 <= float(value) <= 1.0:
            raise CoordinationError(f"{name} must lie in [0, 1]")
    modes = {
        CoordinationMode.B_TO_C: float(backbone_to_chemistry),
        CoordinationMode.C_TO_B: float(chemistry_to_backbone),
        CoordinationMode.JOINT: 1.0 if joint is None else float(joint),
        CoordinationMode.HOLD: 0.0 if hold is None else float(hold),
    }
    return IndependentGates(
        modes=MappingProxyType(modes),
        states=(
            CoordinationState.BACKBONE,
            CoordinationState.SEQUENCE,
            CoordinationState.SIDECHAIN,
            CoordinationState.CHEMISTRY,
        ),
        is_hypothesis=True,
    )
