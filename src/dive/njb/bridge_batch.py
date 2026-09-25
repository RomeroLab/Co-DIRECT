
from __future__ import annotations

import torch

class BridgeBatchError(RuntimeError):
    pass

def substitute_source(flow_matcher, batch: dict, source: dict,
                      *, align_rng: bool = True) -> dict:

    x_1, mask, batch_shape, n, _dtype, device = flow_matcher.process_batch(batch)

    missing = [mode for mode in flow_matcher.data_modes if mode not in source]
    if missing:
        raise BridgeBatchError(
            f"the source carries no {', '.join(missing)}; filling a missing mode "
            f"with a noise draw would make this arm half ordinary and its loss "
            f"curve unreadable"
        )

    x_0 = {}
    for mode in flow_matcher.data_modes:
        value = source[mode]
        if value.shape != x_1[mode].shape:
            raise BridgeBatchError(
                f"shape disagreement on {mode!r}: source {tuple(value.shape)} "
                f"against native {tuple(x_1[mode].shape)}. The source is "
                f"generated at the native's own length precisely so that this "
                f"cannot happen; a mismatch here means the bank and the batch "
                f"came from different examples"
            )

        x_0[mode] = value.detach().to(x_1[mode].dtype).to(x_1[mode].device)

    t = flow_matcher.sample_t(shape=batch_shape, device=device)
    if align_rng:
        flow_matcher.sample_noise(n=n, shape=batch_shape, mask=mask, device=device)
    x_t = flow_matcher.interpolate(x_0=x_0, x_1=x_1, t=t, mask=mask)

    batch["x_0"] = x_0
    batch["x_1"] = x_1
    batch["x_t"] = x_t
    batch["t"] = t
    batch["mask"] = mask
    return batch
