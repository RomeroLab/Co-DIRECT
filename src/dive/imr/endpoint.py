
from __future__ import annotations

import torch

DERIVED = ("v_guided", "score", "score_guided", "guided_v", "guided_score")

def apply_endpoint_delta(output, clean, deltas, times):

    result = {}
    for mode, original in output.items():
        delta = deltas[mode]
        if not torch.isfinite(delta).all():
            raise ValueError("nonfinite endpoint increment")
        if not delta.any():
            result[mode] = original
            continue
        revised = {k: v for k, v in original.items() if k not in DERIVED}
        revised["x_1"] = torch.where(delta != 0, clean[mode] + delta, clean[mode])
        if "v" in original:
            active = delta.ne(0).any(dim=(-1, -2))
            if ((times[mode] >= 1) & active).any():
                raise ValueError("nonzero endpoint increment cannot update velocity at t=1")
            denominator = torch.where(active, 1 - times[mode], torch.ones_like(times[mode]))
            denominator = denominator[:, None, None]
            revised["v"] = torch.where(delta != 0, original["v"] + delta / denominator,
                                       original["v"])
        elif "x_1" not in original:
            raise ValueError("unsupported native prediction parameterization")
        result[mode] = revised
    return result
