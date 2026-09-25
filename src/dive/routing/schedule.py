
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DIRECTIONS: tuple[str, ...] = ("z_to_x", "x_to_z")

AMPLITUDES: tuple[float, ...] = (-0.5, 0.0, 0.5)

class ScheduleError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class GateSchedule:

    direction: str
    amplitude: float
    profile_t: tuple[float, ...]
    shape: tuple[float, ...]

    def gate_at(self, t: float) -> float:

        if self.direction not in DIRECTIONS:
            raise ScheduleError(f"unknown direction {self.direction!r}")
        if not any(abs(self.amplitude - a) < 1e-12 for a in AMPLITUDES):
            raise ScheduleError(
                f"amplitude {self.amplitude} is not in the pre-registered grid {AMPLITUDES}"
            )
        if len(self.profile_t) != len(self.shape):
            raise ScheduleError(
                f"profile_t and shape length differ: "
                f"{len(self.profile_t)} != {len(self.shape)}"
            )
        if len(self.shape) < 2:
            raise ScheduleError("a schedule needs at least two points")
        return 1.0 + self.amplitude * _interpolate(self.profile_t, self.shape, float(t))

    def to_dict(self) -> dict[str, object]:

        return {
            "direction": self.direction,
            "amplitude": self.amplitude,
            "profile_t": list(self.profile_t),
            "shape": list(self.shape),
        }

def build_shape(profile: Sequence[float]) -> tuple[float, ...]:

    if len(profile) < 2:
        raise ScheduleError("a profile needs at least two points")
    uniform = 1.0 / len(profile)
    centred = [float(p) - uniform for p in profile]
    largest_positive = max(centred)
    if largest_positive <= 0.0:
        return tuple(0.0 for _ in centred)
    return tuple(max(-1.0, min(1.0, c / largest_positive)) for c in centred)

def amplitude_grid() -> tuple[tuple[float, float], ...]:

    return tuple((ax, az) for ax in AMPLITUDES for az in AMPLITUDES)

def load_schedules(
    path: Path | str, *, amplitude_x: float, amplitude_z: float
) -> dict[str, GateSchedule]:

    payload = json.loads(Path(path).read_text())
    grid = tuple(float(t) for t in payload["t_bb_ca_per_step"])
    amplitudes = {"z_to_x": float(amplitude_x), "x_to_z": float(amplitude_z)}

    schedules: dict[str, GateSchedule] = {}
    for direction in DIRECTIONS:
        entry = payload["profiles"][direction]
        shape = build_shape(entry["mean_normalized_profile"])
        if len(shape) != len(grid):
            raise ScheduleError(
                f"{direction}: profile has {len(shape)} points but the t grid has {len(grid)}"
            )
        schedules[direction] = GateSchedule(
            direction=direction,
            amplitude=amplitudes[direction],
            profile_t=grid,
            shape=shape,
        )
    return schedules

def _interpolate(grid: Sequence[float], values: Sequence[float], t: float) -> float:

    if t <= grid[0]:
        return float(values[0])
    if t >= grid[-1]:
        return float(values[-1])
    for index in range(1, len(grid)):
        if t <= grid[index]:
            left, right = grid[index - 1], grid[index]
            span = right - left
            if span <= 0:
                return float(values[index])
            weight = (t - left) / span
            return float(values[index - 1] * (1.0 - weight) + values[index] * weight)
    return float(values[-1])
