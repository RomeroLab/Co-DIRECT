
from __future__ import annotations

import torch

from dive.njb.framework_condition import FrameworkConditionWrapper, echo_free_rows

class ScheduledFrameworkCondition(FrameworkConditionWrapper):

    def __init__(self, inner, supplied_ca, given, *, release_below_t: float,
                 supply_identity: bool = True, identities=None):
        super().__init__(inner, supplied_ca, given,
                         supply_identity=supply_identity, identities=identities)
        if not 0.0 <= float(release_below_t) <= 1.0:
            raise ValueError("release_below_t must lie in [0, 1]")
        self.release_below_t = float(release_below_t)
        self.released = 0

    @staticmethod
    def _time(batch: dict) -> float:
        t = batch.get("t")
        if t is None:
            raise ValueError(
                "the batch carries no t, so the schedule cannot be evaluated; "
                "guessing a step index here is how a schedule silently becomes "
                "a no-op")

        if isinstance(t, dict):
            if "bb_ca" not in t:
                raise ValueError(
                    f"t is a dict without a bb_ca entry (keys {sorted(t)}); "
                    "refusing to pick a modality clock by position")
            t = t["bb_ca"]
        if torch.is_tensor(t):
            return float(t.reshape(-1)[0])
        return float(t)

    def forward(self, batch: dict):
        state = batch.get("x_t")
        if not (isinstance(state, dict) and state.get("bb_ca") is not None):
            self.calls += 1
            return self.inner(batch)
        if self._time(batch) < self.release_below_t:
            self.calls += 1
            self.released += 1
            return self.inner(batch)
        return super().forward(batch)

    def telemetry(self) -> dict:
        return {"release_below_t": self.release_below_t, "calls": self.calls,
                "rewrites": self.rewrites, "released": self.released}
