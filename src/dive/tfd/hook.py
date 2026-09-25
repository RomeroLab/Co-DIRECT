
from __future__ import annotations

from typing import Any

import torch

from dive.tfd.arms import (
    ARMS,
    MODE_X,
    MODE_Z,
    adopt_from,
    merge_nn_out,
    second_call_self_conditioning,
)

class _RngSnapshot:

    def __init__(self) -> None:
        self.cpu = torch.get_rng_state()
        self.cuda: list[torch.Tensor] = []
        if torch.cuda.is_available():
            self.cuda = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]

    def restore(self) -> None:
        torch.set_rng_state(self.cpu)
        for i, state in enumerate(self.cuda):
            torch.cuda.set_rng_state(state, i)

class InterventionHook:

    def __init__(
        self,
        model: Any,
        *,
        arm: str,
        window: tuple[int, int],
        region_mask: torch.Tensor | None = None,
        record: bool = True,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        first, last = window
        if first > last:
            raise ValueError(f"empty window {window!r}")
        if first < 0:
            raise ValueError(f"window starts before step 0: {window!r}")
        self.model = model
        self.arm = arm
        self.window = (int(first), int(last))
        self.region_mask = region_mask
        self.record = record
        self._inner = model.predict_for_sampling
        self.stats = {
            "steps": 0,
            "forward_calls": 0,
            "intervened_steps": 0,
            "skipped_no_self_conditioning": 0,
        }
        self.trace: list[dict[str, float]] = []

    def install(self) -> "InterventionHook":
        self.model.predict_for_sampling = self
        return self

    def remove(self) -> None:
        self.model.predict_for_sampling = self._inner

    def __enter__(self) -> "InterventionHook":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, batch: dict, mode: str = "full", n_recycle: int = 0):
        step = self.stats["steps"]
        first_pass = self._inner(batch, mode=mode, n_recycle=n_recycle)
        self.stats["forward_calls"] += 1

        if mode != "full":
            return first_pass
        self.stats["steps"] = step + 1

        in_window = self.window[0] <= step <= self.window[1]
        if self.arm == "released" or not in_window:
            return first_pass

        sc_old = batch.get("x_sc")
        if sc_old is None:
            self.stats["skipped_no_self_conditioning"] += 1
            return first_pass

        snapshot = _RngSnapshot()
        try:
            x1_pred = self.model.fm.nn_out_to_clean_sample_prediction(batch, first_pass)
            sc_new = second_call_self_conditioning(
                self.arm, sc_old, x1_pred, region_mask=self.region_mask
            )
            saved = batch.get("x_sc")
            batch["x_sc"] = sc_new
            try:
                second_pass = self._inner(batch, mode=mode, n_recycle=n_recycle)
            finally:
                batch["x_sc"] = saved
            self.stats["forward_calls"] += 1
            self.stats["intervened_steps"] += 1
            merged = merge_nn_out(self.arm, first_pass, second_pass)
            if self.record:
                self.trace.append(self._movement(step, first_pass, merged))
            return merged
        finally:
            snapshot.restore()

    @staticmethod
    def _movement(step: int, first, merged) -> dict[str, float]:
        row: dict[str, float] = {"step": float(step)}
        for mode in (MODE_X, MODE_Z):
            if mode in first and mode in merged and "x_1" in first[mode]:
                d = (merged[mode]["x_1"].float() - first[mode]["x_1"].float()).abs()
                row[f"{mode}_max_abs"] = float(d.max()) if d.numel() else 0.0
                row[f"{mode}_mean_abs"] = float(d.mean()) if d.numel() else 0.0
        return row

    def adoption(self) -> dict[str, int]:
        return adopt_from(self.arm)
