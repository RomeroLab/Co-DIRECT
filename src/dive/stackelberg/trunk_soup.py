
from __future__ import annotations

def soup_nn(states: list[dict], weights: list[float] | None = None) -> dict:

    if not states:
        raise ValueError("soup_nn needs at least one state dict")
    n = len(states)
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(
            f"weights has length {len(weights)} for {n} state dicts")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("soup weights must sum to a positive number")
    weights = [float(w) / total for w in weights]
    keys = set(states[0])
    for i, state in enumerate(states[1:], start=1):
        other = set(state)
        if other != keys:
            raise ValueError(
                f"state {i} keys differ from state 0; "
                f"only-left={sorted(keys - other)[:8]} "
                f"only-right={sorted(other - keys)[:8]}")
    out = {}
    for key in states[0]:
        tensors = [s[key] for s in states]
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) != 1:
            raise ValueError(
                f"soup key {key!r} has shapes {sorted(shapes)}")
        acc = None
        for w, t in zip(weights, tensors):
            piece = t.float() * w
            acc = piece if acc is None else acc + piece
        out[key] = acc.to(dtype=tensors[0].dtype)
    return out

def soup_paper_files(paths: list, dest, *, weights=None) -> dict:

    from pathlib import Path
    import torch

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payloads = []
    nns = []
    for path in paths:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "nn" not in payload:
            raise ValueError(
                f"{path} is not a paper trunk { nn, step, args} ; "
                "soup would average the wrong object")
        payloads.append(payload)
        nns.append(payload["nn"])
    mixed = soup_nn(nns, weights=weights)
    torch.save({
        "nn": mixed,
        "step": [int(p.get("step") or 0) for p in payloads],
        "args": {
            "soup_of": [str(p) for p in paths],
            "weights": list(weights) if weights is not None else None,
            "sources": [p.get("args") for p in payloads],
        },
    }, dest)
    return {"n": len(mixed), "dest": str(dest)}
