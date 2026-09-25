
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from dive.ccr.model import (
    AE,
    CKPT,
    UPSTREAM,
    _assert_restored,
    _restore_unspliced,
    bootstrap_v1,
)

N_CANDIDATE_SLOTS = 6

@dataclass(frozen=True)
class CCRBuildReport:
    scope: str
    arm: str
    n_parameters: int
    n_trainable: int
    trainable_prefixes: tuple[str, ...]
    n_checkpoint_tensors: int
    n_repaired: int
    n_unrestored: int
    lora_rank: int | None
    n_slots: int
    n_residue_features: int
    n_candidate_features: int

def build_ccr_model(
    *,
    store_dir: Path,
    scope: str = "lora",
    arm: str = "ccr",
    lora_rank: int = 8,
    lora_dropout: float = 0.0,
    n_slots: int = N_CANDIDATE_SLOTS,
    checkpoint: Path = CKPT,
    autoencoder: Path = AE,
):

    if scope not in ("lora", "full"):
        raise ValueError(f"scope must be 'lora' or 'full', got {scope!r}")
    if arm not in ("ccr", "control"):
        raise ValueError(f"arm must be 'ccr' or 'control', got {arm!r}")

    import torch
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.train import _splice_pretrained_weights
    import proteinfoundation.proteina as proteina_module

    from dive.ccr.exchange import CANDIDATE_FEATURES
    from dive.ccr.heads import install_candidate_adapter
    from dive.ccr.message import FEATURE_NAMES

    if bool(getattr(proteina_module, "_USE_V2", False)):
        raise RuntimeError("v2 architecture imported; this campaign's path is v1")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["hyper_parameters"]["cfg_exp"]
    model = Proteina(config, store_dir=str(store_dir),
                     autoencoder_ckpt_path=str(autoencoder))

    if scope == "lora":
        from dive.integrations.complexa_training import configure_lora

        configure_lora(model, rank=lora_rank, dropout=lora_dropout,
                       upstream_root=UPSTREAM)

    state = model.state_dict()
    spliced = _splice_pretrained_weights(state, payload["state_dict"])
    model.load_state_dict(spliced, strict=False)
    with torch.no_grad():
        repaired = _restore_unspliced(model, payload["state_dict"])
    unrestored = _assert_restored(model, payload["state_dict"])
    if unrestored:
        raise RuntimeError(
            "pretrained tensors not restored after splice and repair: "
            + "; ".join(unrestored[:5])
        )

    if getattr(model, "autoencoder", None) is not None and model.autoencoder.decoder is None:
        full = Proteina.load_from_checkpoint(str(autoencoder), strict=False,
                                             map_location="cpu")
        if full.autoencoder is not None and full.autoencoder.decoder is not None:
            model.autoencoder = full.autoencoder
        del full

    if scope == "full":

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.nn.parameters():
            parameter.requires_grad_(True)

    if getattr(model, "autoencoder", None) is not None:
        model.autoencoder.eval()
        for parameter in model.autoencoder.parameters():
            parameter.requires_grad_(False)

    adapter = install_candidate_adapter(
        model, n_slots=n_slots, n_residue_features=len(FEATURE_NAMES),
        n_candidate_features=len(CANDIDATE_FEATURES),
    )
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)

    for name, parameter in adapter.inner.named_parameters():
        parameter.requires_grad_(scope == "full" or "lora_" in name)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    report = CCRBuildReport(
        scope=scope, arm=arm,
        n_parameters=sum(p.numel() for p in model.parameters()),
        n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_prefixes=tuple(sorted({n.split(".")[0] for n in trainable})),
        n_checkpoint_tensors=len(payload["state_dict"]),
        n_repaired=len(repaired), n_unrestored=len(unrestored),
        lora_rank=lora_rank if scope == "lora" else None,
        n_slots=n_slots, n_residue_features=len(FEATURE_NAMES),
        n_candidate_features=len(CANDIDATE_FEATURES),
    )
    return model, adapter, report

def freeze_reference_trunk(model):

    reference = copy.deepcopy(model.nn)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference

__all__ = [
    "N_CANDIDATE_SLOTS",
    "CCRBuildReport",
    "bootstrap_v1",
    "build_ccr_model",
    "freeze_reference_trunk",
]
