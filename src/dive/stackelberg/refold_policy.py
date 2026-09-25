
from __future__ import annotations

BIG_CHUNK = 64
SMALL_CHUNK = 32
TINY_CHUNK = 16
CHUNK_GUARD_TOKENS = 520
SERIAL_GUARD_TOKENS = 520

def serial_sample_seeds(seed: int, samples: int) -> list[int]:
    if int(samples) < 1:
        raise ValueError("samples must be >= 1; dropping samples is not allowed")
    return [int(seed) + i for i in range(int(samples))]

def fold_plan(n_tokens: int, *, serial_flag: bool = False, oom_retry: bool = False) -> dict:

    n = int(n_tokens)
    serial = bool(serial_flag) or bool(oom_retry) or n > SERIAL_GUARD_TOKENS
    if oom_retry or serial:
        chunk = TINY_CHUNK
    elif n > CHUNK_GUARD_TOKENS:
        chunk = SMALL_CHUNK
    else:
        chunk = BIG_CHUNK
    return {"chunk_size": chunk, "serial": serial, "n_tokens": n}
