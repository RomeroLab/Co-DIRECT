
from __future__ import annotations

from dive.data.crop import (
    DEFAULT_CROP_SIZE,
    DEFAULT_TARGET_BUDGET,
    MAX_DESIGN_FRACTION,
    RoleAwareCrop,
)
from dive.data.motif import SampleMotifWithinDesign, UseMotifAsTarget
from dive.data.role_transform import RequireSupportedResidueTypes, RoleMaskTransform

FAMILIES = ("binder", "ame", "antibody")

def family_transforms(
    family: str,
    *,
    crop_size: int = DEFAULT_CROP_SIZE,
    seed: int | None = None,
    max_design_fraction: float = MAX_DESIGN_FRACTION,
    target_budget: int = DEFAULT_TARGET_BUDGET,
):

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")

    from proteinfoundation.datasets import transforms as upstream

    chain = [

        RequireSupportedResidueTypes(),
        upstream.CoordsToNanometers(),
        upstream.ChainBreakPerResidueTransform(),

        RoleMaskTransform(),

        RoleAwareCrop(
            crop_size=crop_size,
            max_design_fraction=max_design_fraction,
            target_budget=target_budget,
        ),
    ]

    if family == "ame":
        chain += [SampleMotifWithinDesign(seed=seed), UseMotifAsTarget()]

    chain.append(upstream.ExtractTargetCoordinatesTransform(compact_mode=True))
    return chain

def family_loader_kwargs(family: str) -> dict:

    return {}
