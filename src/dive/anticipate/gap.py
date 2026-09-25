
from __future__ import annotations

from torch import Tensor

from dive.imr.revision import implicit_vote

def follower_gap(decoded: dict, proposed_ca: Tensor, designed: Tensor) -> Tensor | None:

    _, gap, have = implicit_vote(decoded, proposed_ca, designed)
    if not bool(have.any()):
        return None
    weight = have.to(gap.dtype)
    return (gap * weight).sum() / weight.sum()
