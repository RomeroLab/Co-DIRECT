
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Generator, Tensor

DENSITY_RADIUS_NM: float = 1.0

_COVARIATE_NAMES: tuple[str, ...] = (
    "dist_to_target",
    "local_density",
    "relative_position",
)

class StatisticsError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class Concentration:

    top_mass: float
    uniform_baseline: float
    n_valid: int

@dataclass(frozen=True, slots=True)
class NullInterval:

    low: float
    high: float
    observed: float
    inside: bool

@dataclass(frozen=True, slots=True)
class Asymmetry:

    correlation: float
    argmax_z_to_x: int
    argmax_x_to_z: int
    argmax_agrees: bool

def concentration(norms: Tensor, valid: Tensor, *, top_fraction: float = 0.2) -> Concentration:

    if not 0.0 < top_fraction <= 1.0:
        raise StatisticsError(f"top_fraction must be in (0, 1], observed {top_fraction}")
    selected = _valid_values(norms, valid)
    n_valid = selected.numel()

    k = max(1, int(round(top_fraction * n_valid)))
    baseline = k / n_valid

    total = float(selected.sum())
    if total <= 0.0:

        return Concentration(top_mass=baseline, uniform_baseline=baseline, n_valid=n_valid)

    top = torch.topk(selected / total, k).values
    return Concentration(top_mass=float(top.sum()), uniform_baseline=baseline, n_valid=n_valid)

def spearman(x: Tensor, y: Tensor) -> float:

    if x.shape != y.shape:
        raise StatisticsError(f"shape mismatch: {tuple(x.shape)} != {tuple(y.shape)}")
    if x.numel() < 2:
        raise StatisticsError("spearman needs at least two observations")
    return _pearson(_average_ranks(x.flatten()), _average_ranks(y.flatten()))

def rank_stability(per_residue_by_seed: Tensor, valid: Tensor) -> float:

    if per_residue_by_seed.dim() != 2:
        raise StatisticsError("per_residue_by_seed must have shape (seeds, residues)")
    n_seeds = per_residue_by_seed.shape[0]
    if n_seeds < 2:
        raise StatisticsError("rank stability needs at least two seeds")

    columns = per_residue_by_seed[:, valid]
    if columns.shape[1] < 2:
        raise StatisticsError("rank stability needs at least two valid residues")

    correlations = [
        spearman(columns[i], columns[j])
        for i in range(n_seeds)
        for j in range(i + 1, n_seeds)
    ]
    return float(sum(correlations) / len(correlations))

def step_profile(norms: Tensor, valid: Tensor) -> Tensor:

    if norms.shape != valid.shape:
        raise StatisticsError(f"shape mismatch: {tuple(norms.shape)} != {tuple(valid.shape)}")
    counts = valid.sum(dim=-1)
    if not bool((counts > 0).all()):
        raise StatisticsError("every step must have at least one valid residue")

    cleaned = torch.where(valid, norms, torch.zeros_like(norms))
    if not torch.isfinite(cleaned).all():
        raise StatisticsError("norms contain non-finite values at valid cells")

    means = cleaned.sum(dim=-1) / counts
    total = means.sum()
    if float(total) <= 0.0:
        return torch.full_like(means, 1.0 / means.numel())
    return means / total

def direction_asymmetry(profile_z_to_x: Tensor, profile_x_to_z: Tensor) -> Asymmetry:

    if profile_z_to_x.shape != profile_x_to_z.shape:
        raise StatisticsError("directional profiles must have the same shape")
    argmax_one = int(torch.argmax(profile_z_to_x))
    argmax_two = int(torch.argmax(profile_x_to_z))
    return Asymmetry(
        correlation=_pearson(profile_z_to_x.flatten(), profile_x_to_z.flatten()),
        argmax_z_to_x=argmax_one,
        argmax_x_to_z=argmax_two,
        argmax_agrees=argmax_one == argmax_two,
    )

def covariates(
    x_t_bb_ca: Tensor,
    x_target: Tensor,
    valid: Tensor,
    *,
    density_radius: float,
) -> dict[str, Tensor]:

    if valid.dtype is not torch.bool:
        raise StatisticsError(f"valid must be bool, observed {valid.dtype}")
    if not float(density_radius) > 0.0:
        raise StatisticsError(f"density_radius must be positive, observed {density_radius}")
    if x_t_bb_ca.shape[:-1] != valid.shape or x_t_bb_ca.shape[-1] != 3:
        raise StatisticsError(
            f"x_t_bb_ca {tuple(x_t_bb_ca.shape)} does not match valid {tuple(valid.shape)}"
        )
    if x_target.dim() < 2 or x_target.shape[-1] != 3:
        raise StatisticsError(f"x_target must end in a 3 axis, observed {tuple(x_target.shape)}")

    cells, n_res = valid.shape[:-1], valid.shape[-1]
    binder = x_t_bb_ca.to(torch.float64).reshape(-1, n_res, 3)
    target = _broadcast_target(x_target, cells)
    flat_valid = valid.reshape(-1, n_res)

    if not bool(flat_valid.any(dim=-1).all()):
        raise StatisticsError("no valid cells to summarize")
    if not torch.isfinite(binder[flat_valid]).all():
        raise StatisticsError("non-finite coordinate at a residue marked valid")
    if not torch.isfinite(target).all():
        raise StatisticsError("non-finite target coordinate")

    to_target = torch.cdist(binder, target).amin(dim=-1)
    within = torch.cdist(binder, binder) <= float(density_radius)
    neighbours = within & flat_valid.unsqueeze(-2)
    self_pair = torch.eye(n_res, dtype=torch.bool, device=within.device)
    density = (neighbours & ~self_pair).sum(dim=-1).to(torch.float64)

    rank = torch.cumsum(flat_valid.to(torch.int64), dim=-1) - 1
    spread = (flat_valid.sum(dim=-1, keepdim=True) - 1).clamp(min=1).to(torch.float64)
    relative = rank.to(torch.float64) / spread

    values = (to_target, density, relative)
    return {
        name: _pad_with_nan(value, flat_valid).reshape(*cells, n_res)
        for name, value in zip(_COVARIATE_NAMES, values, strict=True)
    }

def covariate_association(magnitudes: Tensor, covariate: Tensor, valid: Tensor) -> float:

    ranked, against = _valid_pair(magnitudes, covariate, valid)
    return spearman(ranked, against)

def association_grid(magnitudes: Tensor, covariate: Tensor, valid: Tensor) -> Tensor:

    cells, n_res = valid.shape[:-1], valid.shape[-1]
    flat_magnitudes = magnitudes.reshape(-1, n_res)
    flat_covariate = covariate.reshape(-1, n_res)
    flat_valid = valid.reshape(-1, n_res)
    rows = [
        covariate_association(flat_magnitudes[i], flat_covariate[i], flat_valid[i])
        for i in range(flat_valid.shape[0])
    ]
    return torch.tensor(rows, dtype=torch.float64).reshape(cells)

def permutation_null(
    observed_fn: Callable[[Tensor], float],
    magnitudes: Tensor,
    covariate: Tensor,
    valid: Tensor,
    *,
    generator: Generator,
    draws: int = 200,
) -> Tensor:

    if draws < 1:
        raise StatisticsError(f"draws must be at least one, observed {draws}")
    cells, n_res = valid.shape[:-1], valid.shape[-1]
    flat_magnitudes = magnitudes.reshape(-1, n_res)
    flat_covariate = covariate.reshape(-1, n_res)
    flat_valid = valid.reshape(-1, n_res)

    grid = torch.empty(draws, flat_valid.shape[0], dtype=torch.float64)
    for i in range(flat_valid.shape[0]):
        left, right = _valid_pair(flat_magnitudes[i], flat_covariate[i], flat_valid[i])
        ranks_left = _average_ranks(left)
        ranks_right = _average_ranks(right)
        orders = _cell_permutations(left.numel(), draws, generator)
        grid[:, i] = _pearson_rows(ranks_left[orders], ranks_right)

    reshaped = grid.reshape(draws, *cells)
    return torch.tensor(
        [float(observed_fn(reshaped[draw])) for draw in range(draws)], dtype=torch.float64
    )

def association_test(
    magnitudes: Tensor,
    covariate: Tensor,
    valid: Tensor,
    *,
    generator: Generator,
    draws: int = 200,
    observed_fn: Callable[[Tensor], float] = None,
    coverage: float = 0.95,
) -> NullInterval:

    reduce = median_across_targets if observed_fn is None else observed_fn
    observed = float(reduce(association_grid(magnitudes, covariate, valid)))
    null = permutation_null(
        reduce, magnitudes, covariate, valid, generator=generator, draws=draws
    )
    return central_interval(null, observed, coverage=coverage)

def profile_deviation_from_uniform(profile: Tensor) -> float:

    if profile.dim() != 1:
        raise StatisticsError(f"profile must be one-dimensional, observed {tuple(profile.shape)}")
    n_steps = profile.numel()
    if n_steps < 2:
        raise StatisticsError("a profile needs at least two steps")
    values = profile.to(torch.float64)
    if not torch.isfinite(values).all():
        raise StatisticsError("non-finite value in a step profile")
    if bool((values < 0).any()):
        raise StatisticsError("a step profile must be non-negative")
    if abs(float(values.sum()) - 1.0) > 1e-6:
        raise StatisticsError(f"a step profile must sum to one, observed {float(values.sum())}")
    return float((values - 1.0 / n_steps).abs().sum())

def profile_deviation_grid(norms: Tensor, valid: Tensor) -> Tensor:

    if norms.shape != valid.shape:
        raise StatisticsError(f"shape mismatch: {tuple(norms.shape)} != {tuple(valid.shape)}")
    cells = valid.shape[:-2]
    n_steps, n_res = valid.shape[-2], valid.shape[-1]
    flat_norms = norms.reshape(-1, n_steps, n_res)
    flat_valid = valid.reshape(-1, n_steps, n_res)
    rows = [
        profile_deviation_from_uniform(step_profile(flat_norms[i], flat_valid[i]))
        for i in range(flat_valid.shape[0])
    ]
    return torch.tensor(rows, dtype=torch.float64).reshape(cells)

def profile_permutation_null(
    observed_fn: Callable[[Tensor], float],
    norms: Tensor,
    valid: Tensor,
    *,
    generator: Generator,
    draws: int = 200,
) -> Tensor:

    if draws < 1:
        raise StatisticsError(f"draws must be at least one, observed {draws}")
    cells = valid.shape[:-2]
    n_steps, n_res = valid.shape[-2], valid.shape[-1]
    flat_norms = norms.reshape(-1, n_steps, n_res)
    flat_valid = valid.reshape(-1, n_steps, n_res)

    values = torch.empty(draws, dtype=torch.float64)
    for draw in range(draws):
        rows = []
        for i in range(flat_valid.shape[0]):
            order = torch.argsort(
                torch.rand(n_steps, n_res, generator=generator), dim=0
            )
            shuffled_norms = torch.gather(flat_norms[i], 0, order)
            shuffled_valid = torch.gather(flat_valid[i], 0, order)
            rows.append(
                profile_deviation_from_uniform(step_profile(shuffled_norms, shuffled_valid))
            )
        grid = torch.tensor(rows, dtype=torch.float64).reshape(cells)
        values[draw] = float(observed_fn(grid))
    return values

def profile_test(
    norms: Tensor,
    valid: Tensor,
    *,
    generator: Generator,
    draws: int = 200,
    observed_fn: Callable[[Tensor], float] = None,
    coverage: float = 0.95,
) -> NullInterval:

    reduce = median_across_targets if observed_fn is None else observed_fn
    observed = float(reduce(profile_deviation_grid(norms, valid)))
    null = profile_permutation_null(reduce, norms, valid, generator=generator, draws=draws)
    return central_interval(null, observed, coverage=coverage)

def median_across_targets(grid: Tensor) -> float:

    if grid.dim() < 1:
        raise StatisticsError("the grid needs a target axis")
    if grid.shape[0] == 0:
        raise StatisticsError("the grid needs at least one target")
    values = grid.to(torch.float64).reshape(grid.shape[0], -1)
    if not torch.isfinite(values).all():
        raise StatisticsError("non-finite value in the grid")
    return float(values.mean(dim=1).median())

def central_interval(
    null_sample: Tensor, observed: float, *, coverage: float = 0.95
) -> NullInterval:

    if not 0.0 < coverage < 1.0:
        raise StatisticsError(f"coverage must be in (0, 1), observed {coverage}")
    if null_sample.dim() != 1 or null_sample.numel() < 2:
        raise StatisticsError("a null sample must be one-dimensional with at least two draws")
    values = null_sample.to(torch.float64)
    if not torch.isfinite(values).all():
        raise StatisticsError("non-finite value in the null sample")
    if not float(observed) == float(observed):
        raise StatisticsError("the observed value is not a number")

    tail = (1.0 - coverage) / 2.0
    bounds = torch.quantile(values, torch.tensor([tail, 1.0 - tail], dtype=torch.float64))
    low, high = float(bounds[0]), float(bounds[1])
    return NullInterval(
        low=low, high=high, observed=float(observed), inside=low <= float(observed) <= high
    )

def _broadcast_target(x_target: Tensor, cells: torch.Size) -> Tensor:

    target = x_target.to(torch.float64)
    while target.dim() < len(cells) + 2:
        target = target.unsqueeze(0)
    if target.dim() != len(cells) + 2:
        raise StatisticsError(
            f"x_target {tuple(x_target.shape)} has more axes than the cells {tuple(cells)}"
        )
    try:
        target = target.expand(*cells, target.shape[-2], 3)
    except RuntimeError as error:
        raise StatisticsError(
            f"x_target {tuple(x_target.shape)} does not broadcast over cells {tuple(cells)}"
        ) from error
    return target.reshape(-1, target.shape[-2], 3)

def _pad_with_nan(values: Tensor, flat_valid: Tensor) -> Tensor:

    return torch.where(flat_valid, values, torch.full_like(values, float("nan")))

def _valid_pair(magnitudes: Tensor, covariate: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:

    if magnitudes.shape != valid.shape or covariate.shape != valid.shape:
        raise StatisticsError(
            f"shape mismatch: magnitudes {tuple(magnitudes.shape)}, "
            f"covariate {tuple(covariate.shape)}, valid {tuple(valid.shape)}"
        )
    if valid.dtype is not torch.bool:
        raise StatisticsError(f"valid must be bool, observed {valid.dtype}")
    left = magnitudes[valid].to(torch.float64)
    right = covariate[valid].to(torch.float64)
    if left.numel() < 2:
        raise StatisticsError("a covariate association needs at least two valid residues")
    if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise StatisticsError("non-finite value at a cell marked valid")
    return left, right

def _cell_permutations(n_valid: int, draws: int, generator: Generator) -> Tensor:

    return torch.argsort(torch.rand(draws, n_valid, generator=generator), dim=-1)

def _pearson_rows(rows: Tensor, other: Tensor) -> Tensor:

    a = rows.to(torch.float64)
    a = a - a.mean(dim=-1, keepdim=True)
    b = other.to(torch.float64)
    b = b - b.mean()
    denominator = torch.sqrt((a * a).sum(dim=-1) * (b * b).sum())
    numerator = (a * b).sum(dim=-1)
    return torch.where(denominator == 0, torch.zeros_like(numerator), numerator / denominator)

def _valid_values(norms: Tensor, valid: Tensor) -> Tensor:

    if norms.shape != valid.shape:
        raise StatisticsError(f"shape mismatch: {tuple(norms.shape)} != {tuple(valid.shape)}")
    if valid.dtype is not torch.bool:
        raise StatisticsError(f"valid must be bool, observed {valid.dtype}")
    selected = norms[valid].to(torch.float64)
    if selected.numel() == 0:
        raise StatisticsError("no valid cells to summarize")
    if not torch.isfinite(selected).all():
        raise StatisticsError("non-finite value at a cell marked valid")
    if bool((selected < 0).any()):
        raise StatisticsError("magnitudes must be non-negative")
    return selected

def _average_ranks(values: Tensor) -> Tensor:

    x = values.to(torch.float64)
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(n, dtype=torch.float64)

    unique, inverse = torch.unique(x, return_inverse=True)
    sums = torch.zeros(unique.numel(), dtype=torch.float64).index_add_(0, inverse, ranks)
    counts = torch.zeros(unique.numel(), dtype=torch.float64).index_add_(
        0, inverse, torch.ones(n, dtype=torch.float64)
    )
    return (sums / counts)[inverse]

def _pearson(x: Tensor, y: Tensor) -> float:

    a = x.to(torch.float64) - x.to(torch.float64).mean()
    b = y.to(torch.float64) - y.to(torch.float64).mean()
    denominator = torch.sqrt((a * a).sum() * (b * b).sum())
    if float(denominator) == 0.0:
        return 0.0
    return float((a * b).sum() / denominator)
