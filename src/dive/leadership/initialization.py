
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from dive.leadership.fields import FIXED_CORNERS

PRACTICAL_EQUIVALENCE_BAND = 0.01

GATE_ENDPOINT_CLIP = 0.05

JOINT_GATES = (1.0, 1.0)

_SCHEDULE = re.compile(r"^z_to_x=(\d+\.\d+),x_to_z=(\d+\.\d+)$")

class InitializationError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class InitialRegime:

    name: str
    gates: tuple[float, float]
    loss: float
    argmin_name: str
    argmin_loss: float
    selected_by: str
    relative_gap_to_argmin: float
    band: float

    def as_provenance(self) -> dict:
        return {
            "name": self.name,
            "gates": {"z_to_x": self.gates[0], "x_to_z": self.gates[1]},
            "loss": self.loss,
            "argmin_name": self.argmin_name,
            "argmin_loss": self.argmin_loss,
            "selected_by": self.selected_by,
            "relative_gap_to_argmin": self.relative_gap_to_argmin,
            "practical_equivalence_band": self.band,
            "endpoint_clip": GATE_ENDPOINT_CLIP,
            "initial_logits": {
                "z_to_x": gate_logit(self.gates[0]),
                "x_to_z": gate_logit(self.gates[1]),
            },
        }

def parse_regime(name: str) -> tuple[float, float]:

    if name in FIXED_CORNERS:
        return FIXED_CORNERS[name]
    match = _SCHEDULE.match(name)
    if match is None:
        raise InitializationError(
            f"unknown regime {name!r}; expected a named corner "
            f"({sorted(FIXED_CORNERS)}) or 'z_to_x=A.AA,x_to_z=B.BB'"
        )
    return float(match.group(1)), float(match.group(2))

def select_initial_regime(
    losses: dict[str, float], *, band: float = PRACTICAL_EQUIVALENCE_BAND
) -> InitialRegime:

    if not losses:
        raise InitializationError(
            "the regime scan is empty; there is nothing to select from and a "
            "default would be an unrecorded choice"
        )

    ordered = sorted(losses.items(), key=lambda kv: (kv[1], parse_regime(kv[0])))
    argmin_name, argmin_loss = ordered[0]
    if argmin_loss <= 0:
        raise InitializationError(
            f"regime {argmin_name!r} has non-positive loss {argmin_loss}; a "
            f"relative band is undefined against it"
        )

    within = [
        (name, loss)
        for name, loss in ordered
        if (loss - argmin_loss) / argmin_loss <= band
    ]
    chosen_name, chosen_loss = min(
        within,
        key=lambda item: (
            _distance_to_joint(parse_regime(item[0])),
            parse_regime(item[0]),
        ),
    )

    return InitialRegime(
        name=chosen_name,
        gates=parse_regime(chosen_name),
        loss=chosen_loss,
        argmin_name=argmin_name,
        argmin_loss=argmin_loss,
        selected_by=(
            "argmin"
            if chosen_name == argmin_name
            else "closest_to_joint_within_band"
        ),
        relative_gap_to_argmin=(chosen_loss - argmin_loss) / argmin_loss,
        band=band,
    )

def _distance_to_joint(gates: tuple[float, float]) -> float:
    return math.dist(gates, JOINT_GATES)

def gate_logit(value: float, *, clip: float = GATE_ENDPOINT_CLIP) -> float:

    if not 0.0 <= value <= 1.0:
        raise InitializationError(
            f"gate {value} is outside the unit interval; gates are sigmoid "
            f"outputs and cannot be initialized beyond it"
        )
    bounded = min(max(value, clip), 1.0 - clip)
    return math.log(bounded / (1.0 - bounded))

def regime_from_scan(path, *, band: float = PRACTICAL_EQUIVALENCE_BAND):

    import json
    from pathlib import Path

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise InitializationError(
            f"no fixed-gate scan at {path}; the router's initial regime is "
            f"selected from a recorded measurement, never from a default"
        ) from error

    partition = payload.get("flow_partition")
    if partition != "validation":
        raise InitializationError(
            f"scan {path} reports flow_partition {partition!r}; the initial "
            f"regime is a validation-only decision"
        )

    losses = payload.get("equal_family_normalized_flow_loss_by_regime")
    if not losses:
        raise InitializationError(
            f"scan {path} carries no equal_family_normalized_flow_loss_by_regime"
        )

    chosen = select_initial_regime(losses, band=band)
    provenance = {
        "scan_path": str(path),
        "scan_checkpoint_sha256": payload.get("checkpoint_sha256"),
        "flow_partition": partition,
        "regimes_scanned": len(losses),
        **chosen.as_provenance(),
    }
    return chosen, provenance
