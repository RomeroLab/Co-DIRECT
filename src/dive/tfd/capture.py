
from __future__ import annotations

from typing import Any, Iterable

import torch

class StopSampling(Exception):
    pass

def _snapshot(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    return value

class StateCapture:

    def __init__(
        self,
        model: Any,
        *,
        steps: Iterable[int],
        stop_after_last: bool = True,
        keys: tuple[str, ...] = ("x_t", "t", "x_sc", "mask"),
    ) -> None:
        wanted = tuple(sorted({int(s) for s in steps}))
        if not wanted:
            raise ValueError("no steps requested; a capture with nothing to capture has no use")
        self.model = model
        self.wanted = wanted
        self.stop_after_last = stop_after_last
        self.keys = keys
        self._inner = model.predict_for_sampling
        self.states: dict[int, dict] = {}
        self.step = 0

    def install(self) -> "StateCapture":
        self.model.predict_for_sampling = self
        return self

    def remove(self) -> None:
        self.model.predict_for_sampling = self._inner

    def __enter__(self) -> "StateCapture":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, batch: dict, mode: str = "full", n_recycle: int = 0):
        if mode != "full":
            return self._inner(batch, mode=mode, n_recycle=n_recycle)
        step = self.step
        self.step += 1
        if step in self.wanted:
            self.states[step] = {k: _snapshot(batch.get(k)) for k in self.keys}
            if self.stop_after_last and step == self.wanted[-1]:
                raise StopSampling(f"captured every requested step up to {step}")
        return self._inner(batch, mode=mode, n_recycle=n_recycle)
