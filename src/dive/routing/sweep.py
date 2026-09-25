
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dive.routing.cell import ArmSpec
from dive.routing.schedule import amplitude_grid

SWEEP_RUN_ID = "sweep-dev-001"
SWEEP_SEEDS: tuple[int, ...] = (5, 6, 7)
ROUTED_STEPS = 400
MATCHED_BASELINE_STEPS = 1200
N_DEV_GROUPS = 16

class SweepPlanError(RuntimeError):
    pass

def sweep_arms() -> tuple[ArmSpec, ...]:

    routed = tuple(
        ArmSpec(kind="routed", amplitude_x=ax, amplitude_z=az, steps=ROUTED_STEPS)
        for ax, az in amplitude_grid()
    )
    baseline = ArmSpec(
        kind="baseline", amplitude_x=0.0, amplitude_z=0.0, steps=MATCHED_BASELINE_STEPS
    )
    return (*routed, baseline)

@dataclass(frozen=True, slots=True)
class Block:

    index: int
    target: str
    seed: int
    arms: tuple[ArmSpec, ...]

    def to_dict(self) -> dict[str, object]:

        return {
            "index": self.index,
            "target": self.target,
            "seed": self.seed,
            "arms": [arm.to_dict() for arm in self.arms],
        }

def blocks(
    targets: Sequence[str], seeds: Sequence[int] = SWEEP_SEEDS
) -> tuple[Block, ...]:

    if not targets:
        raise SweepPlanError("a sweep needs at least one target")
    if not seeds:
        raise SweepPlanError("a sweep needs at least one seed")
    if len(set(targets)) != len(targets):
        raise SweepPlanError(f"duplicate target in the sweep plan: {list(targets)}")
    if len(set(seeds)) != len(seeds):
        raise SweepPlanError(f"duplicate seed in the sweep plan: {list(seeds)}")

    arms = sweep_arms()
    return tuple(
        Block(
            index=index,
            target=str(targets[index // len(seeds)]),
            seed=int(seeds[index % len(seeds)]),
            arms=arms,
        )
        for index in range(len(targets) * len(seeds))
    )

def block_for(
    index: int, targets: Sequence[str], seeds: Sequence[int] = SWEEP_SEEDS
) -> Block:

    if isinstance(index, bool) or not isinstance(index, int):
        raise SweepPlanError(f"array index must be an integer, observed {index!r}")
    plan = blocks(targets, seeds)
    if not 0 <= index < len(plan):
        raise SweepPlanError(
            f"array index {index} is outside 0..{len(plan) - 1}; submit with "
            f"--array=0-{len(plan) - 1}"
        )
    return plan[index]
