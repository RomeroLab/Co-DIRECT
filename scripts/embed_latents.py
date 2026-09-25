
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def _to_device(value, device):
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value

def main() -> int:
    import torch

    from dive.cre.model import build_ame_model
    from dive.cre.train import ame_dataloader

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "latent_one.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, report = build_ame_model(device=device, verify_sha=False, trainable=False)
    from proteinfoundation.utils.sample_utils import add_clean_samples
    loader = ame_dataloader("train", batch_size=1, seed=1, v6=True, shuffle=False)
    batch = None
    for candidate in loader:
        if candidate is not None:
            batch = candidate
            break
    if batch is None:
        raise SystemExit("v6 loader produced no usable batch")
    batch = _to_device(batch, device)
    clean = add_clean_samples(batch, model.cfg_exp.product_flowmatcher, model.autoencoder)
    latent = clean["x_1"]["local_latents"]
    ca = clean["x_1"]["bb_ca"]
    if latent.shape[-1] != 8 or not torch.isfinite(latent).all():
        raise SystemExit(f"bad latent {tuple(latent.shape)} finite={torch.isfinite(latent).all()}")
    example = batch.get("example_id", batch.get("id"))
    torch.save({
        "local_latents": latent.detach().cpu(),
        "bb_ca": ca.detach().cpu(),
        "example_id": example if not torch.is_tensor(example) else None,
        "checkpoint": report.checkpoint,
    }, out)
    print(f"latent {tuple(latent.shape)} mean_abs {float(latent.detach().float().abs().mean()):.4f}")
    print(f"wrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
