
from __future__ import annotations

from pathlib import Path

LORA_SCALE = 64.0 / 32.0

def merge_lora(state: dict, *, scale: float = LORA_SCALE) -> dict:

    out = {k: v for k, v in state.items() if "lora_" not in k}
    for key, A in state.items():
        if not key.endswith(".lora_A"):
            continue
        prefix = key[: -len("lora_A")]
        B = state[prefix + "lora_B"]
        wkey = prefix + "weight"
        if wkey not in out:
            raise KeyError(f"no base weight for LoRA pair {key}")
        delta = scale * (B @ A)
        if tuple(delta.shape) != tuple(out[wkey].shape):
            raise ValueError(
                f"LoRA delta {tuple(delta.shape)} vs weight {tuple(out[wkey].shape)} for {wkey}")
        out[wkey] = out[wkey] + delta.to(dtype=out[wkey].dtype)
    return out

def nn_state_for_train_continue(state: dict) -> dict:

    if any("lora_" in key for key in state):
        return state
    return merge_lora(state)

def best_val_checkpoint(train_dir: Path) -> dict:

    import json
    train_dir = Path(train_dir)
    log = train_dir / "train.log"
    if not log.is_file():
        raise FileNotFoundError(f"no train.log under {train_dir}")
    cands = []
    for line in log.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in row:
            continue
        if "mean_loss" not in row or "step" not in row:
            continue
        step = int(row["step"])
        if step <= 0:
            continue
        loss = row["mean_loss"]
        if loss is None:
            continue
        trunk = train_dir / f"trunk_step{step}.pt"
        router = train_dir / f"router_step{step}.pt"
        if not trunk.is_file():
            continue
        cands.append({"step": step, "mean_loss": float(loss),
                      "trunk": str(trunk), "router": str(router),
                      "router_exists": router.is_file()})
    if not cands:
        raise ValueError(
            f"no val checkpoints with trunk_stepN.pt under {train_dir}")
    best = min(cands, key=lambda c: (c["mean_loss"], c["step"]))
    return best

def wrap_for_paper_generate(src: Path, dest: Path) -> dict:

    import torch
    payload = torch.load(src, map_location="cpu", weights_only=False)
    nn = payload["nn"] if isinstance(payload, dict) and "nn" in payload else payload
    merged = merge_lora(nn)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "nn": {k: v.detach().cpu() for k, v in merged.items()},
        "step": int(payload.get("step") or 0),
        "args": {"family": payload.get("family"),
                 "lambda_iface": payload.get("lambda_iface"),
                 "init_checkpoint": payload.get("init_checkpoint"),
                 "source": str(src)},
    }, dest)
    return {"n_in": len(nn), "n_out": len(merged),
            "dropped_lora": len(nn) - len(merged), "dest": str(dest)}
