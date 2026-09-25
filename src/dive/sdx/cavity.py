
from __future__ import annotations

import torch

SDX_CAVITY = "sdx_is_cavity"

SDX_IDENTITY = "sdx_identity"

SDX_QUERY = "sdx_query"

STATE_VIEWS = ("x_sc", "x_recycle")

def build_cavity_batch(
    batch: dict,
    flow_matcher,
    *,
    identity: torch.Tensor,
    t_value: float = 0.0,
) -> dict:

    mask = batch["mask"]
    b, n = mask.shape
    device = mask.device
    noise = flow_matcher.sample_noise(n=n, shape=(b,), mask=mask, device=device)

    cavity = dict(batch)
    for key in STATE_VIEWS:
        cavity.pop(key, None)
    cavity["x_t"] = {mode: value for mode, value in noise.items()}
    cavity["t"] = {
        mode: torch.full((b,), float(t_value), device=device)
        for mode in noise
    }
    cavity[SDX_CAVITY] = True
    cavity[SDX_IDENTITY] = identity
    return cavity

def query_span(
    mask: torch.Tensor, *, fraction: float, generator: torch.Generator | None = None
) -> torch.Tensor:

    b, n = mask.shape
    lengths = mask.sum(-1)
    out = torch.zeros_like(mask)
    positions = torch.arange(n, device=mask.device)
    for row in range(b):
        live = int(lengths[row])
        if live <= 0:
            continue
        width = max(1, min(live, int(round(live * fraction))))
        high = live - width
        if generator is not None:
            start = int(torch.randint(0, high + 1, (1,), generator=generator).item())
        else:
            start = int(torch.randint(0, high + 1, (1,)).item())

        live_index = positions[mask[row]]
        out[row, live_index[start:start + width]] = True
    return out

def build_local_cavity_batch(
    batch: dict,
    flow_matcher,
    *,
    identity: torch.Tensor,
    query: torch.Tensor,
) -> dict:

    mask = batch["mask"]
    b, n = mask.shape
    device = mask.device
    noise = flow_matcher.sample_noise(n=n, shape=(b,), mask=mask, device=device)
    hole = (query & mask)[..., None]

    cavity = dict(batch)
    cavity["x_t"] = {
        mode: torch.where(hole, noise[mode], value)
        for mode, value in batch["x_t"].items()
    }
    if "x_sc" in batch and batch["x_sc"] is not None:
        cavity["x_sc"] = {
            mode: torch.where(hole, noise[mode], value)
            for mode, value in batch["x_sc"].items()
        }
    cavity.pop("x_recycle", None)
    cavity[SDX_CAVITY] = True
    cavity[SDX_IDENTITY] = identity
    cavity[SDX_QUERY] = query
    return cavity
