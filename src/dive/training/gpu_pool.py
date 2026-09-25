
from __future__ import annotations

from collections.abc import Iterable, Sequence

ALLOWED_TRAIN_GPUS: tuple[int, ...] = (1, 2, 3, 4, 5)
FORBIDDEN_GPUS: tuple[int, ...] = (0, 6, 7)

def validate_gpu_pool(candidates: Sequence[int] | Iterable[int]) -> list[int]:

    pool = list(candidates)
    if not pool:
        raise ValueError("GPU pool is empty")
    forbidden = sorted({int(i) for i in pool if int(i) in FORBIDDEN_GPUS})
    outside = sorted({int(i) for i in pool if int(i) not in ALLOWED_TRAIN_GPUS})
    if forbidden or outside:
        raise ValueError(
            "GPU pool must be a subset of {1,2,3,4,5}; "
            f"refusing {sorted(set(forbidden) | set(outside))}"
        )
    return [int(i) for i in pool]
