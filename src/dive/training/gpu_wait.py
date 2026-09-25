
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from dive.training.gpu_pool import validate_gpu_pool

def next_free_gpu(
    candidates: Sequence[int],
    *,
    probe: Callable[[Sequence[int]], Sequence[int]],
    claimed: Sequence[int] = (),
) -> int | None:

    pool = validate_gpu_pool(candidates)
    taken = set(claimed)
    free = set(probe(pool))
    for index in pool:
        if index in free and index not in taken:
            return index
    return None

def wait_for_free_gpus(
    indices: Sequence[int],
    *,
    probe: Callable[[Sequence[int]], Sequence[int]],
    timeout_s: float,
    poll_s: float = 300.0,
    now: Callable[[], float] = time.monotonic,
    report: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:

    wanted = validate_gpu_pool(indices)

    started = now()
    while True:
        free = set(probe(wanted))
        missing = [i for i in wanted if i not in free]
        if not missing:
            return True
        if report is not None:
            report(
                "waiting for GPUs "
                + ", ".join(str(i) for i in missing)
                + " to free up"
            )
        if now() - started >= timeout_s:
            return False
        sleep(poll_s)

def select_free_gpus(
    candidates: Sequence[int],
    *,
    need: int,
    probe: Callable[[Sequence[int]], Sequence[int]],
    timeout_s: float,
    poll_s: float = 300.0,
    now: Callable[[], float] = time.monotonic,
    report: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int] | None:

    pool = validate_gpu_pool(candidates)
    if need <= 0:
        raise ValueError(f"need must be positive, got {need}")
    if need > len(pool):
        raise ValueError(f"need {need} exceeds the {len(pool)}-GPU candidate pool")

    started = now()
    while True:
        free = set(probe(pool))
        chosen = [i for i in pool if i in free][:need]
        if len(chosen) == need:
            return chosen
        if report is not None:
            report(
                f"only {len(chosen)} of {need} GPUs free in pool "
                + ", ".join(str(i) for i in pool)
                + "; short by "
                + ", ".join(str(i) for i in pool if i not in free)
            )
        if now() - started >= timeout_s:
            return None
        sleep(poll_s)
