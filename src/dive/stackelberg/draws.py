
from __future__ import annotations

import random

def seed_all_streams(seed: int) -> None:

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plan_draws(*, frozen, extra: int = 0, extra_start: int = 200):

    frozen = [int(s) for s in frozen]
    if extra < 0:
        raise ValueError("extra draw count cannot be negative")
    extras = [int(extra_start) + i for i in range(int(extra))]
    clash = sorted(set(extras) & set(frozen))
    if clash:
        raise ValueError(
            f"extra draws {clash} collide with frozen seeds; an extra draw "
            "reusing a frozen seed would be indistinguishable from the paired "
            "one in the record")
    return ([{"seed": s, "paired": True} for s in frozen]
            + [{"seed": s, "paired": False} for s in extras])
