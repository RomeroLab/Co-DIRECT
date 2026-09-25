
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Generator, Tensor

from dive.analysis.heterogeneity import (
    NullInterval,
    association_test,
    covariate_association,
    median_across_targets,
)

class CalibrationError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class CalibrationResult:

    direction: str
    null: str
    interval: NullInterval
    sign_agreement: float
    n_complexes: int

    def to_dict(self) -> dict[str, object]:

        return {
            "direction": self.direction,
            "null": self.null,
            "observed": self.interval.observed,
            "null_low": self.interval.low,
            "null_high": self.interval.high,
            "inside_null": self.interval.inside,
            "sign_agreement_across_timesteps": self.sign_agreement,
            "n_complexes": self.n_complexes,
        }

def calibrate(
    magnitudes: Tensor,
    values: Tensor,
    valid: Tensor,
    *,
    generator: Generator,
    draws: int = 200,
) -> NullInterval:

    _check(magnitudes, values, valid)
    return association_test(
        magnitudes,
        values,
        valid,
        generator=generator,
        draws=draws,
        observed_fn=median_across_targets,
    )

def sign_agreement_across_timesteps(
    magnitudes: Tensor, values: Tensor, valid: Tensor
) -> float:

    _check(magnitudes, values, valid)
    n_complexes, n_steps = valid.shape[0], valid.shape[1]
    if n_steps < 2:
        raise CalibrationError("sign agreement needs at least two timesteps")

    agreed = 0
    for index in range(n_complexes):
        signs = set()
        for step in range(n_steps):
            rho = covariate_association(
                magnitudes[index, step], values[index, step], valid[index, step]
            )
            if rho == 0.0:
                signs.add(0.0)
            else:
                signs.add(1.0 if rho > 0 else -1.0)
        if len(signs) == 1 and 0.0 not in signs:
            agreed += 1
    return agreed / n_complexes

def _check(magnitudes: Tensor, values: Tensor, valid: Tensor) -> None:

    if valid.dim() != 3:
        raise CalibrationError(
            f"grids need three axes [complex, timestep, residue], observed "
            f"{tuple(valid.shape)}"
        )
    if magnitudes.shape != valid.shape or values.shape != valid.shape:
        raise CalibrationError(
            f"shape mismatch: magnitudes {tuple(magnitudes.shape)}, "
            f"values {tuple(values.shape)}, valid {tuple(valid.shape)}"
        )
    if valid.dtype is not torch.bool:
        raise CalibrationError(f"valid must be bool, observed {valid.dtype}")
