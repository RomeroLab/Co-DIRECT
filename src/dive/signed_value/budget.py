
from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

_PER_TARGET_WALL_SECONDS = 7_200
_FEASIBILITY_CEILING_SECONDS = 10_092

MIN_DEVICE_BYTES = 47_695_921_152
HEADROOM_BYTES = 2_147_483_648

CRITIC_MEMORY_FRACTION_NUMERATOR = 23
CRITIC_MEMORY_FRACTION_DENOMINATOR = 25
CRITIC_MEMORY_FRACTION_LITERAL = "0.92"

class BudgetError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class CallLedger:

    denoiser: int = 803
    decoder: int = 13
    critic: int = 14

    def assert_exact(self) -> None:

        if (self.denoiser, self.decoder, self.critic) != (803, 13, 14):
            raise BudgetError(f"per-target call ledger drifted: {self}")

@dataclass(frozen=True, slots=True)
class BudgetState:

    previous_debit_gpu_seconds: int
    requested_targets: int
    projected_total_gpu_seconds: int

def assert_reconciled(*, attempt_ids: Set[str], debit_ids: Set[str]) -> None:

    if attempt_ids != debit_ids:
        missing = sorted(attempt_ids - debit_ids)
        unexpected = sorted(debit_ids - attempt_ids)
        raise BudgetError(
            "unreconciled signed-value attempts: "
            f"missing_debits={missing}, unexpected_debits={unexpected}"
        )

def critic_memory_limit(total_memory_bytes: int) -> int:

    if (
        isinstance(total_memory_bytes, bool)
        or type(total_memory_bytes) is not int
        or total_memory_bytes < 0
    ):
        raise BudgetError(
            "device total memory must be a literal nonnegative integer, "
            f"observed {total_memory_bytes!r}"
        )
    return (
        total_memory_bytes
        * CRITIC_MEMORY_FRACTION_NUMERATOR
        // CRITIC_MEMORY_FRACTION_DENOMINATOR
    )

def project_submission(*, previous_debit: int, requested_targets: int) -> BudgetState:

    if (
        isinstance(previous_debit, bool)
        or isinstance(requested_targets, bool)
        or not isinstance(previous_debit, int)
        or not isinstance(requested_targets, int)
        or previous_debit < 0
        or requested_targets not in (0, 1, 2, 3)
    ):
        raise BudgetError("invalid signed-value budget input")
    projected = previous_debit + requested_targets * _PER_TARGET_WALL_SECONDS
    if projected > _FEASIBILITY_CEILING_SECONDS:
        raise BudgetError(
            f"projection {projected:,} exceeds {_FEASIBILITY_CEILING_SECONDS:,} GPU-seconds"
        )
    return BudgetState(previous_debit, requested_targets, projected)
