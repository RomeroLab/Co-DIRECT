
from __future__ import annotations

from dive.codirect_paths import joined

import os
import sys
from dataclasses import dataclass
from pathlib import Path

UPSTREAM = Path(joined('PROTEINA_COMPLEXA_ROOT'))
CKPT = UPSTREAM / "ckpts/complexa.ckpt"
AE = UPSTREAM / "ckpts/complexa_ae.ckpt"

@dataclass(frozen=True)
class BuildReport:
    n_parameters: int
    n_trainable: int
    trainable_prefixes: tuple[str, ...]
    n_checkpoint_tensors: int
    n_loaded: int
    n_missing: int
    n_unexpected: int
    architecture_v2: bool
    n_repaired: int = 0
    n_unrestored: int = 0
    repaired_keys: tuple[str, ...] = ()

def bootstrap_v1() -> None:

    os.environ["USE_V2_COMPLEXA_ARCH"] = "False"
    os.environ.setdefault("DATA_PATH", str(UPSTREAM / "assets"))
    for path in (str(UPSTREAM / "src"),):
        if path not in sys.path:
            sys.path.insert(0, path)
    import proteinfoundation.patches.atomworks_patches

def _restore_unspliced(model, checkpoint) -> tuple[str, ...]:

    import torch

    live = model.state_dict()
    repaired = []
    for key, value in checkpoint.items():
        current = live.get(key)
        if current is None or tuple(current.shape) != tuple(value.shape):
            continue

        reference = value.to(device=current.device, dtype=current.dtype)
        if not torch.equal(current.float(), reference.float()):
            current.copy_(reference)
            repaired.append(key)
    return tuple(repaired)

def _assert_restored(model, checkpoint) -> tuple[str, ...]:

    import torch

    live = model.state_dict()
    unrestored = []
    for key, value in checkpoint.items():
        current = live.get(key)
        if current is None:
            unrestored.append(f"{key}: absent")
        elif tuple(current.shape) != tuple(value.shape):
            unrestored.append(f"{key}: shape {tuple(current.shape)} != {tuple(value.shape)}")
        else:
            reference = value.to(device=current.device, dtype=current.dtype)
            if not torch.equal(current.float(), reference.float()):
                deviation = float((current.float() - reference.float()).abs().max())
                unrestored.append(f"{key}: max deviation {deviation:.4g}")
    return tuple(unrestored)

def build_operator_model(
    *,
    store_dir: Path,
    lora_rank: int = 8,
    lora_dropout: float = 0.0,
    with_message: bool = True,
    checkpoint: Path = CKPT,
    autoencoder: Path = AE,
):

    import torch
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.train import _splice_pretrained_weights

    import proteinfoundation.proteina as proteina_module

    from dive.ccr.trunk import install_message_adapter
    from dive.integrations.complexa_training import configure_lora

    if bool(getattr(proteina_module, "_USE_V2", False)):
        raise RuntimeError("v2 architecture imported; this campaign's path is v1")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["hyper_parameters"]["cfg_exp"]

    model = Proteina(
        config, store_dir=str(store_dir), autoencoder_ckpt_path=str(autoencoder)
    )
    configure_lora(model, rank=lora_rank, dropout=lora_dropout, upstream_root=UPSTREAM)

    state = model.state_dict()
    spliced = _splice_pretrained_weights(state, payload["state_dict"])
    result = model.load_state_dict(spliced, strict=False)

    with torch.no_grad():
        repaired = _restore_unspliced(model, payload["state_dict"])
    unrestored = _assert_restored(model, payload["state_dict"])
    if unrestored:
        raise RuntimeError(
            "pretrained tensors not restored after splice and repair: "
            + "; ".join(unrestored[:5])
        )

    if getattr(model, "autoencoder", None) is not None and model.autoencoder.decoder is None:
        full = Proteina.load_from_checkpoint(
            str(autoencoder), strict=False, map_location="cpu"
        )
        if full.autoencoder is not None and full.autoencoder.decoder is not None:
            model.autoencoder = full.autoencoder
        del full
    if getattr(model, "autoencoder", None) is not None:
        model.autoencoder.eval()
        for parameter in model.autoencoder.parameters():
            parameter.requires_grad_(False)

    adapter = install_message_adapter(model) if with_message else None

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    report = BuildReport(
        n_parameters=sum(p.numel() for p in model.parameters()),
        n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_prefixes=tuple(sorted({n.split(".")[0] for n in trainable})),
        n_checkpoint_tensors=len(payload["state_dict"]),
        n_loaded=len(spliced),
        n_missing=len(result.missing_keys),
        n_unexpected=len(result.unexpected_keys),
        architecture_v2=False,
        n_repaired=len(repaired),
        n_unrestored=len(unrestored),
        repaired_keys=tuple(repaired),
    )
    return model, adapter, report
