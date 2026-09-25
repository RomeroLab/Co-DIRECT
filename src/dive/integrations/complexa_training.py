
from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from torch import nn

from dive.integrations.splice_migration import (
    CRITICAL_TARGET_PARAMETERS,
    apply_semantic_migrations,
    assert_target_conditioning_restored,
)
from dive.emergent_contract import (
    CONTEMPLATED_LORA_RANKS,
    TASK_SPECIFIC_CHECKPOINT_NAMES,
    EmergentContract,
)

FAMILIES = ("binder", "ame", "antibody")

TRAINABLE_NAME_MARKERS = ("lora_", "presence", "router")

DIRECTIONAL_TRAINABLE_NAME_MARKERS = (*TRAINABLE_NAME_MARKERS, "bridge")

MIN_LOADED_FRACTION = 0.95

LORA_ALPHA = 16
LORA_DROPOUT = 0.05

class ModelSetupError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class SpliceReport:

    model_nn_params: int
    checkpoint_nn_params: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    resized: tuple[str, ...]

    target_migration_audit: Mapping[str, object] | None = None

    @property
    def loaded_nn_params(self) -> int:
        return self.model_nn_params - len(self.missing)

    @property
    def loaded_fraction(self) -> float:
        if self.model_nn_params == 0:
            return 0.0
        return self.loaded_nn_params / self.model_nn_params

    def assert_sufficiently_loaded(
        self, minimum_fraction: float = MIN_LOADED_FRACTION
    ) -> None:

        if self.loaded_fraction < minimum_fraction:
            raise ModelSetupError(
                f"only {self.loaded_nn_params}/{self.model_nn_params} denoiser "
                f"parameters ({self.loaded_fraction:.1%}) were loaded from the common "
                f"checkpoint, below the {minimum_fraction:.0%} floor; the shared "
                f"initialization premise does not hold. Missing prefixes: "
                f"{sorted({'.'.join(k.split('.')[:3]) for k in self.missing})[:6]}"
            )

    def as_provenance(self) -> dict[str, object]:
        return {
            "model_nn_params": self.model_nn_params,
            "checkpoint_nn_params": self.checkpoint_nn_params,
            "loaded_nn_params": self.loaded_nn_params,
            "loaded_fraction": self.loaded_fraction,
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "resized": list(self.resized),
            "target_migration_audit": dict(self.target_migration_audit or {}),
        }

class SharedModelRegistry:

    __slots__ = ("_model",)

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def for_family(self, family: str) -> nn.Module:
        if family not in FAMILIES:
            raise ModelSetupError(
                f"unknown family {family!r}; expected one of {FAMILIES}"
            )

        return self._model

def assert_common_checkpoint(path: Path) -> None:

    if Path(path).name in TASK_SPECIFIC_CHECKPOINT_NAMES:
        raise ModelSetupError(
            f"task-specific checkpoint {Path(path).name!r}; all three families "
            f"initialize from the common checkpoint"
        )

def assert_v2_selected() -> None:

    module = sys.modules.get("proteinfoundation.proteina")
    if module is None:
        raise ModelSetupError(
            "proteinfoundation.proteina is not imported yet; call this after the "
            "import so the flag's effect can be verified"
        )
    if not getattr(module, "_USE_V2", False):
        raise ModelSetupError(
            "the v1 architecture is in force. USE_V2_COMPLEXA_ARCH must be set "
            "BEFORE proteinfoundation.proteina is imported; setting it afterwards "
            "does nothing and silently yields a v1 graph"
        )

def configure_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    upstream_root: Path | None = None,
) -> None:

    if rank not in CONTEMPLATED_LORA_RANKS:
        raise ModelSetupError(
            f"lora rank {rank} is not contemplated; expected one of "
            f"{list(CONTEMPLATED_LORA_RANKS)}"
        )
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise ModelSetupError(
            "LoRA layers are already present; configure_lora replaces modules, so "
            "a second call would discard whatever weights they now hold"
        )
    if not hasattr(model, "nn"):
        raise ModelSetupError("model has no `.nn` denoiser to adapt")

    import loralib as lora

    replace_lora_layers = _load_replace_lora_layers(upstream_root)

    replace_lora_layers(model.nn, rank, alpha, dropout)

    lora.mark_only_lora_as_trainable(model, bias="none")

    for name, parameter in model.named_parameters():
        if any(marker in name for marker in TRAINABLE_NAME_MARKERS[1:]):
            parameter.requires_grad_(True)

    autoencoder = getattr(model, "autoencoder", None)
    if autoencoder is not None:
        leaked = [n for n, p in autoencoder.named_parameters() if p.requires_grad]
        if leaked:
            raise ModelSetupError(
                f"the autoencoder must stay frozen, but {len(leaked)} parameter(s) "
                f"are trainable, e.g. {leaked[:3]}"
            )

def trainable_parameter_manifest(
    model: nn.Module, *, directional: bool = False
) -> dict[str, object]:

    markers = (
        DIRECTIONAL_TRAINABLE_NAME_MARKERS if directional else TRAINABLE_NAME_MARKERS
    )
    trainable_names, trainable, total = [], 0, 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable += parameter.numel()

    autoencoder = getattr(model, "autoencoder", None)
    frozen_ae = (
        sum(p.numel() for p in autoencoder.parameters())
        if autoencoder is not None
        else 0
    )
    unexpected = [
        n for n in trainable_names if not any(marker in n for marker in markers)
    ]
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_names": trainable_names,
        "trainable_name_count": len(trainable_names),
        "frozen_autoencoder_parameters": frozen_ae,
        "unexpected_trainable_names": unexpected,
    }

def disable_campaign_dropout(model: nn.Module) -> tuple[str, ...]:

    affected = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
            affected.append(name)
    return tuple(affected)

def assert_campaign_dropout_zero(model: nn.Module) -> None:

    nonzero = [
        f"{name}={module.p}"
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout) and module.p != 0.0
    ]
    if nonzero:
        raise ModelSetupError(
            "directional campaign dropout must be exactly zero; nonzero modules: "
            f"{nonzero}"
        )

def install_presence_conditioning(model, *, token_dim: int):

    from dive.leadership.presence import PresenceConditionedFactory

    denoiser = getattr(model, "nn", None)
    factory = getattr(denoiser, "init_repr_factory", None)
    if factory is None:
        raise ModelSetupError(
            "model.nn has no init_repr_factory to wrap; the presence code has "
            "nowhere to enter and every condition would differ only by its "
            "field of zeros"
        )
    if isinstance(factory, PresenceConditionedFactory):
        raise ModelSetupError(
            "presence conditioning is already installed; wrapping twice would "
            "nest the offset and rename every checkpoint key again"
        )
    denoiser.init_repr_factory = PresenceConditionedFactory(
        factory, token_dim=token_dim
    )
    return model

def build_common_v2_model(
    contract: EmergentContract,
    *,
    store_dir: Path,
    upstream_root: Path | None = None,
    install_presence: bool = True,
    lora_dropout: float = LORA_DROPOUT,
    repair_target_conditioning: bool = True,
):

    import torch

    assert_common_checkpoint(contract.common_checkpoint)
    root = Path(upstream_root or contract.upstream_root)

    _import_upstream_v2(root)
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.train import _splice_pretrained_weights

    assert_v2_selected()

    checkpoint = torch.load(
        contract.common_checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint["hyper_parameters"]["cfg_exp"]

    model = Proteina(
        config,
        store_dir=str(store_dir),
        autoencoder_ckpt_path=str(contract.autoencoder_checkpoint),
    )

    configure_lora(
        model,
        rank=contract.lora_rank,
        dropout=lora_dropout,
        upstream_root=root,
    )

    model_state = model.state_dict()
    spliced = _splice_pretrained_weights(model_state, checkpoint["state_dict"])
    resized = tuple(
        key
        for key, value in spliced.items()
        if key in checkpoint["state_dict"]
        and value.shape != checkpoint["state_dict"][key].shape
    )

    migration_audit = None
    if repair_target_conditioning:
        spliced, migration_audit = apply_semantic_migrations(
            model_state, checkpoint["state_dict"], into=spliced
        )

    result = model.load_state_dict(spliced, strict=False)

    if repair_target_conditioning:

        assert_target_conditioning_restored(model.state_dict(), checkpoint["state_dict"])

    if install_presence:
        install_presence_conditioning(model, token_dim=int(config.nn.token_dim))

    report = SpliceReport(
        model_nn_params=sum(
            1 for k in model_state if k.startswith("nn.") and "lora_" not in k
        ),
        checkpoint_nn_params=sum(
            1 for k in checkpoint["state_dict"] if k.startswith("nn.")
        ),
        missing=tuple(
            k
            for k in result.missing_keys
            if k.startswith("nn.")
            and "lora_" not in k
            and not (repair_target_conditioning and k in CRITICAL_TARGET_PARAMETERS)
        ),
        unexpected=tuple(k for k in result.unexpected_keys if k.startswith("nn.")),
        resized=resized,
        target_migration_audit=migration_audit,
    )
    report.assert_sufficiently_loaded()
    return model, report

def build_directional_v2_model(
    contract: EmergentContract,
    *,
    store_dir: Path,
    upstream_root: Path | None = None,
) -> tuple[nn.Module, SpliceReport]:

    from dive.integrations.directional_v2 import DirectionalV2Adapter
    from dive.leadership.directional_bridge import FourWayRouter

    model, report = build_common_v2_model(
        contract,
        store_dir=store_dir,
        upstream_root=upstream_root,
        install_presence=False,
        lora_dropout=0.0,
    )
    router = FourWayRouter(hidden_dim=int(model.cfg_exp.nn.token_dim))
    model.nn = DirectionalV2Adapter.from_loaded(model.nn, router=router, rank=8)
    disable_campaign_dropout(model)
    assert_campaign_dropout_zero(model)
    return model, report

def _import_upstream_v2(root: Path) -> None:

    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    if "proteinfoundation.proteina" in sys.modules:

        return
    os.environ["USE_V2_COMPLEXA_ARCH"] = "True"

def _load_replace_lora_layers(root: Path | None):

    if root is None:
        from dive.signed_value.roots import EMERGENT_UPSTREAM_ROOT

        root = EMERGENT_UPSTREAM_ROOT
    source = str(Path(root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from proteinfoundation.utils.lora_utils import replace_lora_layers
    except ImportError as error:
        raise ModelSetupError(
            f"cannot import upstream replace_lora_layers: {error}"
        ) from error
    return replace_lora_layers

MUTABLE_SOURCE_REALM_EXCLUSIONS: Mapping[str, type] = MappingProxyType({})
