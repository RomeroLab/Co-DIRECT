
from __future__ import annotations

from dive.codirect_paths import joined

import functools
import hashlib
from pathlib import Path

import torch

UPSTREAM = Path(joined('PROTEINA_COMPLEXA_ROOT'))

SAMPLING_CONFIG = UPSTREAM / "configs/pipeline/model_sampling.yaml"

NSTEPS = 400

MODALITIES = ("bb_ca", "local_latents")

def config_sha256() -> str:

    return hashlib.sha256(SAMPLING_CONFIG.read_bytes()).hexdigest()

@functools.lru_cache(maxsize=4)
def sampling_spec(path: str | None = None) -> dict:

    import yaml

    source = Path(path) if path else SAMPLING_CONFIG
    cfg = yaml.safe_load(source.read_text())
    nsteps = int(cfg["args"]["nsteps"])
    spec: dict = {"nsteps": nsteps, "modes": {}}
    for mode in MODALITIES:
        block = cfg["model"][mode]["schedule"]
        spec["modes"][mode] = (str(block["mode"]), float(block["p"]))
    return spec

def _get_schedule():

    import sys

    root = str(UPSTREAM / "src")
    if root not in sys.path:
        sys.path.insert(0, root)
    from proteinfoundation.flow_matching.product_space_flow_matcher import (
        get_schedule,
    )

    return get_schedule

@functools.lru_cache(maxsize=4)
def schedule_times(nsteps: int | None = None) -> dict[str, torch.Tensor]:

    spec = sampling_spec()
    steps = int(nsteps if nsteps is not None else spec["nsteps"])
    if nsteps is None and steps != NSTEPS:
        raise ValueError(
            f"{SAMPLING_CONFIG} now carries nsteps={steps}, not {NSTEPS}; every "
            "recorded t in this campaign describes the 400-step schedule, so "
            "the change must be made deliberately rather than inherited"
        )
    get_schedule = _get_schedule()
    out = {}
    for mode, (kind, p) in spec["modes"].items():
        ts = get_schedule(mode=kind, nsteps=steps, p1=p).double()
        out[mode] = ts
    return out

def call_times(nsteps: int | None = None) -> dict[str, torch.Tensor]:

    times = schedule_times(nsteps)
    steps = next(iter(times.values())).shape[0] - 1
    return {mode: ts[:steps].clone() for mode, ts in times.items()}

def probe_enabled_indices(nsteps: int | None = None) -> list[int]:

    times = call_times(nsteps)
    steps = next(iter(times.values())).shape[0]
    enabled = []
    for k in range(1, steps):
        deltas = [float(times[m][k] - times[m][k - 1]) for m in times]
        if deltas and sum(deltas) / len(deltas) > 0.0:
            enabled.append(k)
    return enabled

def select_indices(enabled: list[int], count: int) -> list[int]:

    enabled = list(enabled)
    if not enabled:
        raise ValueError(
            "no probe-enabled call index exists; with an empty enabled set "
            "there is no sampler state to measure at and a fallback would "
            "invent one"
        )
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if len(enabled) <= count:
        return enabled
    positions = torch.linspace(0, len(enabled) - 1, count).round().long().tolist()
    return sorted({enabled[p] for p in positions})

def h_k(index: int, nsteps: int | None = None) -> float:

    times = call_times(nsteps)
    steps = next(iter(times.values())).shape[0]
    if index <= 0:
        raise ValueError(
            f"call index {index} has no predecessor, so it carries no h_k; the "
            "real hook bypasses it and this campaign must too"
        )
    if index >= steps:
        raise ValueError(f"call index {index} is beyond the {steps} evaluated calls")
    deltas = [float(times[m][index] - times[m][index - 1]) for m in times]
    return sum(deltas) / len(deltas)

def state_times(index: int, nsteps: int | None = None) -> dict[str, float]:

    times = call_times(nsteps)
    return {mode: float(ts[index]) for mode, ts in times.items()}
