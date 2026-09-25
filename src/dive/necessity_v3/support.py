
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CLEAN_ENDPOINT = 1.0

BACKBONE_MODE = "bb_ca"
LATENT_MODE = "local_latents"

_SHARED_GROUPS_KEY = "shared_groups"

class UnsupportedSampler(RuntimeError):
    pass

class UnsupportedAnchor(RuntimeError):
    pass

def _get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:

    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def _keys(cfg: Mapping[str, Any] | Any) -> list[str]:
    if isinstance(cfg, Mapping):
        return list(cfg.keys())
    return [k for k in dir(cfg) if not k.startswith("_")]

@dataclass(frozen=True)
class TimeDistributionSpec:

    backbone: Mapping[str, Any]
    latent: Mapping[str, Any]
    shared_groups: tuple[tuple[str, ...], ...]
    independent: bool

    @property
    def off_diagonal_supported(self) -> bool:

        return self.independent

def parse_t_distribution(cfg: Mapping[str, Any] | Any) -> TimeDistributionSpec:

    raw_groups = _get(cfg, _SHARED_GROUPS_KEY, None)
    groups: tuple[tuple[str, ...], ...] = ()
    if raw_groups:
        groups = tuple(tuple(str(m) for m in group) for group in raw_groups)

    backbone = _get(cfg, BACKBONE_MODE)
    latent = _get(cfg, LATENT_MODE)
    if backbone is None or latent is None:
        present = [k for k in _keys(cfg) if k != _SHARED_GROUPS_KEY]
        raise UnsupportedSampler(
            f"t_distribution must cover {BACKBONE_MODE!r} and {LATENT_MODE!r}; "
            f"found {present}"
        )

    tied = any(
        BACKBONE_MODE in group and LATENT_MODE in group for group in groups
    )
    return TimeDistributionSpec(
        backbone=backbone, latent=latent, shared_groups=groups, independent=not tied
    )

def _beta_pdf(t: float, p1: float, p2: float) -> float:
    if not 0.0 < t < 1.0:

        return 0.0
    log_pdf = (
        (p1 - 1.0) * math.log(t)
        + (p2 - 1.0) * math.log1p(-t)
        + math.lgamma(p1 + p2)
        - math.lgamma(p1)
        - math.lgamma(p2)
    )
    return math.exp(log_pdf)

def marginal_density(dist: Mapping[str, Any] | Any, t: float) -> float:

    name = str(_get(dist, "name"))
    p1 = float(_get(dist, "p1", 1.0) or 1.0)
    p2 = float(_get(dist, "p2", 1.0) or 1.0)

    if name == "uniform":

        if not 0.0 <= t <= p2:
            return 0.0
        return 1.0 / p2
    if name == "beta":
        return _beta_pdf(t, p1, p2)
    if name == "logit-normal":
        if not 0.0 < t < 1.0:
            return 0.0
        mean = p1
        std = p2

        logit = math.log(t) - math.log1p(-t)
        z = (logit - mean) / std
        return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi) * t * (1.0 - t))
    if name == "mix_unif_beta":
        p3 = float(_get(dist, "p3"))
        if not 0.0 < p3 < 1.0:
            raise UnsupportedSampler(f"mix_unif_beta p3 {p3} not in (0, 1)")
        if not 0.0 <= t <= 1.0:
            return 0.0
        return p3 * 1.0 + (1.0 - p3) * _beta_pdf(t, p1, p2)

    raise UnsupportedSampler(
        f"sampler {name!r} is not evaluable here; refusing to approximate a "
        "marginal, because an approximation would mislabel cells as supported"
    )

def validate_anchors(
    spec: TimeDistributionSpec,
    anchors: Sequence[float],
    steps: Iterable[float],
) -> None:

    steps = tuple(steps)
    if not spec.off_diagonal_supported and len(set(anchors)) > 1:
        raise UnsupportedAnchor(
            "the training sampler tied t_bb_ca to t_local_latents via "
            f"shared_groups={spec.shared_groups}; only the diagonal t_x == t_z "
            "was ever seen, so an off-diagonal grid is out of distribution"
        )
    largest = max(steps) if steps else 0.0
    for anchor in anchors:
        for probe in (anchor - largest, anchor + largest):
            if not 0.0 < probe < 1.0:
                raise UnsupportedAnchor(
                    f"anchor {anchor} with step {largest} reaches t={probe}, "
                    "which is outside the open interval (0, 1); t=0 and t=1 are "
                    "the base and clean endpoints, not corruption levels"
                )
        for dist in (spec.backbone, spec.latent):
            for probe in (anchor - largest, anchor, anchor + largest):
                if marginal_density(dist, probe) <= 0.0:
                    raise UnsupportedAnchor(
                        f"t={probe} has zero training density under "
                        f"{_get(dist, 'name')!r}"
                    )

def cell_occupancy(
    spec: TimeDistributionSpec,
    backbone_anchors: Sequence[float],
    latent_anchors: Sequence[float],
) -> np.ndarray:

    rows = np.array(
        [marginal_density(spec.backbone, t) for t in backbone_anchors], dtype=float
    )
    cols = np.array(
        [marginal_density(spec.latent, t) for t in latent_anchors], dtype=float
    )
    joint = rows[:, None] * cols[None, :]
    total = joint.sum()
    if total <= 0.0:
        raise UnsupportedAnchor("every requested anchor cell has zero density")
    return joint / total

def describe_convention(t_a: float, t_b: float) -> str:

    if t_a == t_b:
        return f"{t_a} and {t_b} are equally corrupted"
    cleaner, dirtier = (t_a, t_b) if t_a > t_b else (t_b, t_a)
    return f"{cleaner} is cleaner than {dirtier}"
