
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from dive.codirect_paths import proteina_root

UPSTREAM = proteina_root()
CKPT = UPSTREAM / "ckpts/complexa_ame.ckpt"
AE_CKPT = UPSTREAM / "ckpts/complexa_ame_ae.ckpt"

LORA_RANK = 32
LORA_ALPHA = 64.0
LORA_DROPOUT = 0.0

@dataclass
class AmeLoadReport:

    checkpoint: str
    checkpoint_sha256: str
    autoencoder: str
    lora_rank: int
    lora_alpha: float
    missing: int
    unexpected: int
    missing_names: list[str] = field(default_factory=list)
    unexpected_names: list[str] = field(default_factory=list)
    lora_tensors: int = 0
    lora_landed_bitwise: int = 0
    lora_B_nonzero: int = 0
    released_tensors: int = 0
    released_landed_bitwise: int = 0
    motif_features_enabled: bool = False
    ligand_features_enabled: bool = False
    autoencoder_trainable_parameters: int = -1
    trunk_trainable_parameters: int = 0
    architecture_module: str = ""
    architecture_v2: bool = False

    @property
    def exact_load(self) -> bool:
        return (
            self.missing == 0
            and self.unexpected == 0
            and self.released_landed_bitwise == self.released_tensors
            and self.lora_landed_bitwise == self.lora_tensors
        )

RUNTIME = Path(__file__).resolve().parents[3] / "scripts" / "necessity_v4"

def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _runtime():

    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    import _ame_v2_rt as runtime

    runtime.bootstrap()
    return runtime

def _find_feature_flags(model) -> tuple[bool, bool]:
    motif = ligand = False
    for module in model.modules():
        if getattr(module, "enable_motif", False):
            motif = True
        if getattr(module, "enable_ligand", False):
            ligand = True
    return motif, ligand

def build_ame_model(*, device: str = "cuda", verify_sha: bool = False, trainable: bool = False):

    import torch

    runtime = _runtime()
    model, info = runtime.load(device)

    if trainable:
        model.train()
        for parameter in model.nn.parameters():
            parameter.requires_grad_(True)

        model.autoencoder.eval()
        for parameter in model.autoencoder.parameters():
            parameter.requires_grad_(False)

    motif_on, ligand_on = _find_feature_flags(model)
    ae_trainable = sum(
        p.numel() for p in model.autoencoder.parameters() if p.requires_grad
    )
    trunk_trainable = sum(p.numel() for p in model.nn.parameters() if p.requires_grad)
    report = AmeLoadReport(
        checkpoint=str(runtime.CKPT),
        checkpoint_sha256=_sha256(runtime.CKPT) if verify_sha else "",
        autoencoder=str(runtime.AE),
        lora_rank=int(info["lora_cfg"]["r"]),
        lora_alpha=float(info["lora_cfg"]["lora_alpha"]),
        missing=int(info["missing"]),
        unexpected=int(info["unexpected"]),
        lora_tensors=int(info["lora_tensors"]),
        lora_landed_bitwise=int(info["lora_landed_bitwise"]),
        lora_B_nonzero=int(info["lora_B_nonzero"]),
        released_tensors=int(info["lora_tensors"]),
        released_landed_bitwise=int(info["lora_landed_bitwise"]),
        motif_features_enabled=motif_on,
        ligand_features_enabled=ligand_on,
        autoencoder_trainable_parameters=ae_trainable,
        trunk_trainable_parameters=trunk_trainable,
        architecture_module=str(info["module"]),
        architecture_v2=True,
    )
    return model, report
