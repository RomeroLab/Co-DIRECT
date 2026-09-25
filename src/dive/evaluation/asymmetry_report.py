
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PRIMARY_CONTRASTS: tuple[str, ...] = ("equal_gate", "magnitude_matched")

SPECIFICITY_CONTRASTS: tuple[str, ...] = ("reversed", "shuffled")

MAX_CLAMPED_FRACTION = 0.25

class AsymmetryReportError(RuntimeError):
    pass

def receiver_scales(
    *, x_norms: Sequence[float], z_norms: Sequence[float]
) -> dict[str, float]:

    for name, values in (("x_norms", x_norms), ("z_norms", z_norms)):
        if len(values) == 0:
            raise AsymmetryReportError(
                f"{name} is empty; the reference set must be non-empty"
            )
    scales = {}
    for key, values in (("sigma_x", x_norms), ("sigma_z", z_norms)):
        mean_square = sum(float(v) ** 2 for v in values) / len(values)
        scale = mean_square**0.5
        if not scale > 0:
            raise AsymmetryReportError(
                f"{key} is not positive; that modality's response is identically "
                f"zero over the reference set and cannot define a scale"
            )
        scales[key] = scale
    return scales

def receiver_scales_from_sums(
    *, x_sum_squares: float, z_sum_squares: float, count: int
) -> dict[str, float]:

    if count <= 0:
        raise AsymmetryReportError(
            "count is zero; the reference set must be non-empty"
        )
    scales = {}
    for key, total in (("sigma_x", x_sum_squares), ("sigma_z", z_sum_squares)):
        if float(total) < 0:
            raise AsymmetryReportError(
                f"{key} received a negative sum of squares ({total}), which cannot "
                f"come from norms"
            )
        scale = (float(total) / int(count)) ** 0.5
        if not scale > 0:
            raise AsymmetryReportError(
                f"{key} is not positive; that modality's response is identically "
                f"zero over the reference set and cannot define a scale"
            )
        scales[key] = scale
    return scales

def asymmetry_contrasts(
    per_arm_losses: Mapping[str, Sequence[float]],
    *,
    parents: Sequence[str],
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:

    from dive.training.perexample import per_family_bootstrap

    required = ("full", *PRIMARY_CONTRASTS, *SPECIFICITY_CONTRASTS)
    missing = [name for name in required if name not in per_arm_losses]
    if missing:
        raise AsymmetryReportError(
            f"the intervention needs every arm from the same forward; missing "
            f"{', '.join(missing)}"
        )

    full = per_arm_losses["full"]
    expected = len(full)
    for name in required:
        if len(per_arm_losses[name]) != expected:
            raise AsymmetryReportError(
                f"arm {name!r} has length {len(per_arm_losses[name])} against "
                f"full's {expected}; the arms are paired per example"
            )
    if len(parents) != expected:
        raise AsymmetryReportError(
            f"parents has length {len(parents)} against full's {expected}"
        )

    roles = dict.fromkeys(PRIMARY_CONTRASTS, "primary")
    roles.update(dict.fromkeys(SPECIFICITY_CONTRASTS, "specificity"))

    contrasts: dict[str, Any] = {}
    for name in (*PRIMARY_CONTRASTS, *SPECIFICITY_CONTRASTS):
        interval = per_family_bootstrap(
            full,
            per_arm_losses[name],
            parents=parents,
            resamples=resamples,
            seed=seed,
        )
        contrasts[name] = {**interval, "role": roles[name]}

    return {
        "contrasts": contrasts,
        "examples": expected,
        "parents": len(set(parents)),
        "sign_convention": (
            "positive means the full adaptive policy achieved the LOWER loss"
        ),
        "scope": "flow-stage receiver loss; not a terminal design endpoint",
    }

def shuffle_is_usable(report: Mapping[str, Any]) -> bool:

    permuted = int(report.get("permuted_residues", 0))
    if permuted <= 0:
        return False
    clamped = int(report.get("clamped_residues", 0))
    return (clamped / permuted) <= MAX_CLAMPED_FRACTION
