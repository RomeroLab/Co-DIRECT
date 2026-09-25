
from __future__ import annotations

import math
from collections.abc import Sequence

EPS = 1e-12

class ReducerError(ValueError):
    pass

def _finite_nonnegative(value: object, name: str) -> float:

    if value is None or isinstance(value, bool):
        raise ReducerError(f"{name} must be a real number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReducerError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ReducerError(f"{name} must be finite, got {number!r}")
    if number < 0.0:
        raise ReducerError(f"{name} must not be negative, got {number!r}")
    return number

def influence_energy(
    numerator_sq: Sequence[float], denominator_sq: Sequence[float]
) -> float:

    if len(numerator_sq) != len(denominator_sq):
        raise ReducerError(
            f"length mismatch: {len(numerator_sq)} numerators, "
            f"{len(denominator_sq)} denominators"
        )
    if not numerator_sq:
        raise ReducerError("cannot pool an empty set of residual statistics")
    total_num = sum(_finite_nonnegative(v, "numerator_sq") for v in numerator_sq)
    total_den = sum(_finite_nonnegative(v, "denominator_sq") for v in denominator_sq)
    if total_den <= 0.0:
        raise ReducerError("pooled denominator is zero; influence is undefined")
    return total_num / total_den

def directional_index(i_z_to_x: float, i_x_to_z: float) -> float:

    a = _finite_nonnegative(i_z_to_x, "i_z_to_x")
    b = _finite_nonnegative(i_x_to_z, "i_x_to_z")
    total = a + b
    if total <= EPS:

        return 0.0
    return (a - b) / total

def coupling(i_z_to_x: float, i_x_to_z: float) -> float:

    return _finite_nonnegative(i_z_to_x, "i_z_to_x") + _finite_nonnegative(
        i_x_to_z, "i_x_to_z"
    )

def normalized_utility(e_self: float, e_joint: float) -> float:

    self_error = _finite_nonnegative(e_self, "e_self")
    joint_error = _finite_nonnegative(e_joint, "e_joint")
    if self_error <= EPS:

        return 0.0
    return (self_error - joint_error) / self_error

def classify_observation(
    *,
    influence: float,
    utility: float,
    headroom: float,
    influence_hi: float = 0.10,
    utility_tol: float = 0.01,
    headroom_tol: float = 0.02,
) -> str:

    influence_value = _finite_nonnegative(influence, "influence")
    headroom_value = _finite_nonnegative(headroom, "headroom")
    if utility is None or not math.isfinite(float(utility)):
        raise ReducerError(f"utility must be a finite number, got {utility!r}")
    utility_value = float(utility)

    if influence_value >= influence_hi:
        if utility_value < -utility_tol:
            return "harmful_over_coupling"
        if utility_value > utility_tol:
            return "useful_implicit_routing"
        return "high_influence_neutral"
    if headroom_value > headroom_tol:
        return "under_routing"
    return "correctly_ignored"

__all__ = [
    "EPS",
    "ReducerError",
    "classify_observation",
    "coupling",
    "directional_index",
    "influence_energy",
    "normalized_utility",
]
