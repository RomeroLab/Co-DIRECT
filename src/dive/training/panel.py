
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable

class PanelError(RuntimeError):
    pass

def tensor_hash(value: Any) -> str:

    digest = hashlib.sha256()
    _feed(digest, value)
    return digest.hexdigest()

def _feed(digest, value) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())

        flat = tensor.contiguous().flatten()
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode())
            _feed(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _feed(digest, item)
    else:
        digest.update(repr(value).encode())

@dataclass(slots=True)
class ValidationPanel:

    seed: int
    _hashes: dict[str, str] = field(default_factory=dict)
    _references: dict[str, float] = field(default_factory=dict)

    def _generator_seed(self, key: str) -> int:

        material = f"{self.seed}:{key}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)

    def corrupt(self, corrupt_fn: Callable, batch: dict, *, key: str) -> dict:

        import torch

        incoming = tensor_hash(batch)
        seen = self._hashes.get(key)
        if seen is not None and seen != incoming:
            raise PanelError(
                f"validation panel entry {key!r} changed: input hash {seen[:16]} "
                f"-> {incoming[:16]}. The panel must be the same examples in the "
                f"same order on every evaluation, or consecutive checkpoints are "
                f"not comparable."
            )
        self._hashes[key] = incoming

        state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            torch.manual_seed(self._generator_seed(key))
            return corrupt_fn(batch)
        finally:

            torch.random.set_rng_state(state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def reference(self, key: str, compute: Callable[[], float]) -> float:

        if key in self._references:
            return self._references[key]
        value = float(compute())
        if not math.isfinite(value) or value == 0.0:
            raise PanelError(
                f"reference denominator for {key!r} is {value}; every ratio "
                f"built on it would be meaningless, so it is refused rather "
                f"than propagated"
            )
        self._references[key] = value
        return value

    def as_provenance(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "entries": len(self._hashes),
            "cached_references": len(self._references),
            "input_hashes": dict(sorted(self._hashes.items())),
            "references": dict(sorted(self._references.items())),
        }

def materialize_panel(loaders, *, max_batches: int) -> dict[str, list]:

    panel: dict[str, list] = {}
    for family, loader in sorted(loaders.items()):
        batches = []
        for batch in loader:
            if batch is None:
                continue
            if len(batches) >= max_batches:
                break
            batches.append(batch)
        if not batches:
            raise PanelError(
                f"{family} produced no validation batch, so the panel cannot be "
                f"built; an equal-family score with a family missing is not the "
                f"declared metric"
            )
        panel[family] = batches
    return panel

def save_panel(panel: dict[str, list], path) -> str:

    import torch
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "families": {family: list(batches) for family, batches in sorted(panel.items())},
        "hashes": {
            family: [tensor_hash(batch) for batch in batches]
            for family, batches in sorted(panel.items())
        },
    }
    torch.save(payload, path)
    return tensor_hash(payload["hashes"])

def load_panel(path) -> dict[str, list]:

    import torch
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise PanelError(
            f"no saved panel at {path}; a comparison across checkpoints needs "
            f"the examples that were used, not a rebuild from the same seed"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    panel = {}
    for family, batches in payload["families"].items():
        expected = payload["hashes"][family]
        for index, (batch, digest) in enumerate(zip(batches, expected)):
            observed = tensor_hash(batch)
            if observed != digest:
                raise PanelError(
                    f"saved panel entry {family}:{index} does not match its "
                    f"recorded hash ({digest[:16]} -> {observed[:16]}); the file "
                    f"is not the panel it claims to be"
                )
        panel[family] = list(batches)
    return panel
