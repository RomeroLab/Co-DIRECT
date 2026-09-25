
from __future__ import annotations

from dive.codirect_paths import cache_dir

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from dive.benchmark.paper_baseline import (
    panel_arm,
    PAPER_SEEDS,
    PaperArm,
    PaperBaselineError,
    family_steps,
)
from types import MappingProxyType

from dive.training.preflight import canonical_json_bytes

NEW_PARAMETER_SEED = 20260828

@dataclass(frozen=True, slots=True)
class SpliceReport:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    resized_keys: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    new_parameter_names: tuple[str, ...]
    new_parameter_seed: int | None
    live_parameter_sha256: str

def hash_live_parameters(model) -> str:
    live = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        live.update(name.encode())
        array = tensor.detach().cpu().contiguous()
        live.update(repr(tuple(array.shape)).encode())
        live.update(bytes(array.numpy()))
    return live.hexdigest()

def seed_generation(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def configure_family_generation(
    model, family: str, *, steps: int | None = None, shipped_protocol: bool = False
) -> int:

    frozen = family_steps(
        family, panel="A", arm=panel_arm("A"), steps=steps,
        shipped_protocol=shipped_protocol,
    )
    cfg_exp = getattr(model, "cfg_exp", None)
    if cfg_exp is not None:
        from copy import deepcopy

        from omegaconf import open_dict

        generation = deepcopy(cfg_exp.generation)
        with open_dict(generation):
            generation.args.nsteps = frozen
            generation.args.guidance_w = 1.0
            generation.args.ag_ratio = 0.0

            generation.args.self_cond = True
            generation.n_recycle = 0
            generation.model = deepcopy(generation.model.ode)
        model.configure_inference(generation, nn_ag=None)
    else:
        from types import SimpleNamespace

        model.configure_inference(
            SimpleNamespace(
                args=SimpleNamespace(
                    nsteps=frozen,
                    guidance_w=1.0,
                    ag_ratio=0.0,
                    self_cond=True,
                ),
                n_recycle=0,
                model={},
            ),
            nn_ag=None,
        )
    return frozen

def attach_chain_identity(formatted, batch):
    import torch

    required_sample = {"coors", "residue_type", "mask"}
    required_batch = {"mask", "generated_mask", "fixed_mask", "residue_type"}
    if not required_sample.issubset(formatted) or not required_batch.issubset(batch):
        raise PaperBaselineError(
            "decoded sample or source batch lacks identity tensors"
        )
    sample_mask = formatted["mask"].bool()
    source_mask = batch["mask"].bool()
    generated = batch["generated_mask"].bool()
    fixed = batch["fixed_mask"].bool()
    if sample_mask.shape != source_mask.shape or not torch.equal(
        sample_mask, source_mask
    ):
        raise PaperBaselineError("decoded mask differs from source batch")
    if generated.shape != source_mask.shape or fixed.shape != source_mask.shape:
        raise PaperBaselineError("generated/fixed mask shape differs")
    if bool((generated & fixed).any()) or not torch.equal(
        generated | fixed, source_mask
    ):
        raise PaperBaselineError(
            "generated/fixed masks do not partition valid residues"
        )
    chains = batch.get("chains", batch.get("chain_index"))
    if chains is None or chains.shape != source_mask.shape:
        raise PaperBaselineError("chain identity shape differs from residue mask")
    decoded_restype = formatted["residue_type"]
    batch_restype = batch["residue_type"]
    if decoded_restype.shape != batch_restype.shape:
        raise PaperBaselineError("decoded residue_type shape differs from source batch")
    batch_coords = batch.get("coords", batch.get("coors"))
    decoded_coors = formatted["coors"]
    if batch_coords is None:
        raise PaperBaselineError("source batch lacks coordinates")
    if decoded_coors.shape != batch_coords.shape:
        raise PaperBaselineError("decoded coordinates shape differs from source batch")
    if not bool(torch.isfinite(decoded_coors).all()):
        raise PaperBaselineError("decoded coordinates are non-finite")
    if not bool(torch.isfinite(batch_coords[fixed]).all()):
        raise PaperBaselineError("source-batch frozen coordinates are non-finite")
    restored_restype = decoded_restype.clone()
    restored_restype[fixed] = batch_restype.to(dtype=restored_restype.dtype)[fixed]
    restored_coors = decoded_coors.clone()
    restored_coors[fixed] = batch_coords.to(dtype=restored_coors.dtype)[fixed]
    return {
        **dict(formatted),
        "coors": restored_coors,
        "residue_type": restored_restype,
        "chain_index": chains.clone(),
        "generated_mask": generated.clone(),
        "fixed_mask": fixed.clone(),
    }

def generate_paper_sample(
    model, batch, *, family: str, seed: int, shipped_protocol: bool = False,
    nsteps_override: int | None = None,
):
    if seed not in PAPER_SEEDS:
        raise PaperBaselineError("generation seed must come from the plan cell")
    configure_family_generation(model, family, shipped_protocol=shipped_protocol)
    if nsteps_override is not None:

        model.inf_cfg.args.nsteps = int(nsteps_override)
    seed_generation(seed)
    import torch

    tensors = dict(batch)
    with torch.inference_mode():
        generated = model.generate(tensors)
    if {"coors", "residue_type", "mask"} <= set(generated):
        formatted = generated
    else:
        from proteinfoundation.utils.sample_utils import sample_formatting

        formatted = sample_formatting(
            x=generated,
            extra_info={"mask": tensors["mask"]},
            ret_mode="coors37_n_aatype",
            data_modes=list(model.cfg_exp.product_flowmatcher),
            autoencoder=getattr(model, "autoencoder", None),
        )
    return attach_chain_identity(formatted, tensors)

def hash_splice_report(report: SpliceReport) -> str:
    payload = {
        "live_parameter_sha256": report.live_parameter_sha256,
        "missing_keys": list(report.missing_keys),
        "new_parameter_names": list(report.new_parameter_names),
        "new_parameter_seed": report.new_parameter_seed,
        "resized_keys": [list(item) for item in report.resized_keys],
        "unexpected_keys": list(report.unexpected_keys),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def assert_lora_free(parameter_names: Sequence[str]) -> None:
    lora = tuple(name for name in parameter_names if "lora_" in name)
    if lora:
        raise PaperBaselineError(f"lora parameters are forbidden: {lora}")

_CONDITIONING_PREFIXES = ("nn.concat_factory", "nn.concat_pair_factory")

def is_conditioning_projection(key: str) -> bool:

    return (
        any(key.startswith(prefix) for prefix in _CONDITIONING_PREFIXES)
        and "linear_out" in key
        and key.endswith(".weight")
    )

def assert_conditioning_pathway_loaded(missing_keys: Sequence[str]) -> None:

    offenders = sorted(key for key in missing_keys if is_conditioning_projection(key))
    if offenders:
        raise PaperBaselineError(
            "conditioning projections would be randomly initialized: "
            + ", ".join(offenders)
        )

PERMITTED_ZERO_FALLBACKS = frozenset({"hotspot_mask_seq", "hotspot_mask_pair"})

def assert_expected_zero_fallbacks(warned_features: Sequence[str]) -> None:

    offenders = sorted(set(warned_features) - PERMITTED_ZERO_FALLBACKS)
    if offenders:
        raise PaperBaselineError(
            "unexpected zero-filled features: " + ", ".join(offenders)
        )

def bind_splice_report(report: SpliceReport) -> str:
    if not report.live_parameter_sha256:
        raise PaperBaselineError("live parameter hash is required")
    if len(report.live_parameter_sha256) != 64:
        raise PaperBaselineError("live parameter hash must be sha256")
    if report.new_parameter_names and report.new_parameter_seed is None:
        raise PaperBaselineError(
            "new v2 parameters require a frozen initialization seed"
        )
    return hash_splice_report(report)

def reinitialize_missing_parameters(
    model, names, *, seed: int = NEW_PARAMETER_SEED
) -> None:
    import torch

    wanted = set(names)
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in wanted:
                continue
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
    leftover = wanted - {name for name, _ in model.named_parameters()}
    if leftover:
        raise PaperBaselineError(
            f"cannot reinitialize missing parameters: {sorted(leftover)}"
        )

LEGACY_ARMS = frozenset({PaperArm.BASE_COMMON_V2_SPLICE})

ARM_LOAD_EXPECTATION = MappingProxyType(
    {
        PaperArm.BASE_COMMON_V2_SPLICE: "reinitializes_target_conditioning",
        PaperArm.BASE_COMMON_EXACT_V1: "exact",

        PaperArm.BASE_COMMON_FINETUNE_V1: "exact",
    }
)

def build_model_for_arm(arm, contract, *, store_dir, allow_legacy_arm: bool = False):

    try:
        selected = PaperArm(arm)
    except ValueError:
        raise PaperBaselineError(f"unknown arm {arm!r}") from None
    if selected in LEGACY_ARMS and not allow_legacy_arm:
        raise PaperBaselineError(
            f"arm {str(selected)!r} is legacy and requires an explicit legacy opt-in"
        )
    try:
        builder = ARM_BUILDERS[selected]
    except KeyError:
        raise PaperBaselineError(f"no builder is bound to arm {str(selected)!r}") from None
    return builder(contract, store_dir=store_dir)

def assert_load_matches_arm(arm, report: SpliceReport) -> None:

    expectation = ARM_LOAD_EXPECTATION.get(PaperArm(arm))
    if expectation != "exact":
        return
    offenders = {
        "missing_keys": report.missing_keys,
        "unexpected_keys": report.unexpected_keys,
        "new_parameter_names": report.new_parameter_names,
        "resized_keys": report.resized_keys,
    }
    dirty = {name: value for name, value in offenders.items() if value}
    if dirty:
        raise PaperBaselineError(
            f"arm {str(PaperArm(arm))!r} promises an exact load but the report is not "
            f"exact: {sorted(dirty)}"
        )

def build_native_v1_model(contract, *, store_dir, upstream_root=None):

    from pathlib import Path

    from dive.integrations.complexa_training import (
        _import_upstream_v1,
        assert_common_checkpoint,
        assert_v1_selected,
    )

    assert_common_checkpoint(contract.common_checkpoint)
    root = Path(upstream_root or contract.upstream_root)
    _import_upstream_v1(root)
    from proteinfoundation.proteina import Proteina

    assert_v1_selected()
    import torch

    checkpoint = torch.load(
        contract.common_checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint["hyper_parameters"]["cfg_exp"]
    model = Proteina(
        config,
        store_dir=str(store_dir),
        autoencoder_ckpt_path=str(contract.autoencoder_checkpoint),
    )
    model_state = model.state_dict()
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)

    missing = tuple(key for key in result.missing_keys if key.startswith("nn."))
    unexpected = tuple(key for key in result.unexpected_keys if key.startswith("nn."))
    assert_conditioning_pathway_loaded(missing)
    if missing or unexpected:
        raise PaperBaselineError(
            f"native v1 load is not exact: missing={missing} unexpected={unexpected}"
        )
    assert_lora_free(tuple(model_state))
    report = SpliceReport(
        missing_keys=(),
        unexpected_keys=(),
        resized_keys=(),
        new_parameter_names=(),
        new_parameter_seed=None,
        live_parameter_sha256=hash_live_parameters(model),
    )
    bind_splice_report(report)
    return model, report

def build_lora_free_v2_model(contract, *, store_dir, upstream_root=None):

    from pathlib import Path

    from dive.integrations.complexa_training import (
        SpliceReport as TrainingSplice,
    )
    from dive.integrations.complexa_training import (
        _import_upstream_v2,
        assert_common_checkpoint,
        assert_v2_selected,
    )

    assert_common_checkpoint(contract.common_checkpoint)
    root = Path(upstream_root or contract.upstream_root)
    _import_upstream_v2(root)
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.train import _splice_pretrained_weights

    assert_v2_selected()
    import torch

    checkpoint = torch.load(
        contract.common_checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint["hyper_parameters"]["cfg_exp"]
    model = Proteina(
        config,
        store_dir=str(store_dir),
        autoencoder_ckpt_path=str(contract.autoencoder_checkpoint),
    )
    model_state = model.state_dict()
    spliced = _splice_pretrained_weights(model_state, checkpoint["state_dict"])
    result = model.load_state_dict(spliced, strict=False)
    training_report = TrainingSplice(
        model_nn_params=sum(1 for key in model_state if key.startswith("nn.")),
        checkpoint_nn_params=sum(
            1 for key in checkpoint["state_dict"] if key.startswith("nn.")
        ),
        missing=tuple(key for key in result.missing_keys if key.startswith("nn.")),
        unexpected=tuple(
            key for key in result.unexpected_keys if key.startswith("nn.")
        ),
        resized=tuple(
            key
            for key, value in spliced.items()
            if key in checkpoint["state_dict"]
            and value.shape != checkpoint["state_dict"][key].shape
        ),
    )
    training_report.assert_sufficiently_loaded()
    assert_lora_free(tuple(model_state))
    new_seed = None
    if training_report.missing:
        reinitialize_missing_parameters(
            model, training_report.missing, seed=NEW_PARAMETER_SEED
        )
        new_seed = NEW_PARAMETER_SEED
    report = SpliceReport(
        missing_keys=training_report.missing,
        unexpected_keys=training_report.unexpected,
        resized_keys=tuple(
            (key, tuple(checkpoint["state_dict"][key].shape), tuple(spliced[key].shape))
            for key in training_report.resized
        ),
        new_parameter_names=training_report.missing,
        new_parameter_seed=new_seed,
        live_parameter_sha256=hash_live_parameters(model),
    )
    bind_splice_report(report)
    return model, report

def build_finetune_v1_model(contract, *, store_dir, upstream_root=None):

    model, report = build_native_v1_model(
        contract, store_dir=store_dir, upstream_root=upstream_root
    )

    import dataclasses

    from dive.benchmark.paper_finetune import TrunkFormatError, load_trunk_into

    try:
        load_trunk_into(model.nn, FINETUNE_TRUNK)
    except TrunkFormatError as error:
        raise PaperBaselineError(f"fine-tune trunk is not loadable: {error}") from error
    model.nn.eval()

    report = dataclasses.replace(
        report, live_parameter_sha256=hash_live_parameters(model)
    )
    bind_splice_report(report)
    return model, report

FINETUNE_TRUNK = (
    cache_dir('emergent', 'benchmark_finetune', 'common-v1-lr1e-6-20260915a', 'trunk_step1500.pt')
)

ARM_BUILDERS = {
    PaperArm.BASE_COMMON_V2_SPLICE: build_lora_free_v2_model,
    PaperArm.BASE_COMMON_EXACT_V1: build_native_v1_model,
    PaperArm.BASE_COMMON_FINETUNE_V1: build_finetune_v1_model,
}
