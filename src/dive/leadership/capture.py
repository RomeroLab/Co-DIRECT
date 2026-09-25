
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from torch import Tensor, nn

class CaptureError(RuntimeError):
    pass

@dataclass(slots=True)
class Capture:

    _tensor: Tensor | None = field(default=None)
    calls: int = 0

    @property
    def tensor(self) -> Tensor:
        if self._tensor is None:
            raise CaptureError(
                "captured tensor is not available yet; it exists only after the "
                "`with` block has run the module"
            )
        return self._tensor

@contextmanager
def capture_module_output(module: nn.Module) -> Iterator[Capture]:

    capture = Capture()

    def hook(_module, _inputs, output):
        capture.calls += 1

        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, Tensor):
            raise CaptureError(f"expected a tensor output, got {type(output).__name__}")
        capture._tensor = output

    handle = module.register_forward_hook(hook)
    try:
        yield capture
    finally:
        handle.remove()

    if capture.calls == 0:
        raise CaptureError(
            "the module was never called inside the capture block; the router "
            "would receive nothing, far from the cause"
        )
    if capture.calls > 1:
        raise CaptureError(
            f"the module was called {capture.calls} times inside the capture block; "
            f"the capture would silently be the last call's output. Recycling calls "
            f"the trunk repeatedly, so scope the capture to a single forward."
        )
