
from __future__ import annotations

from dive.codirect_paths import ROOT, joined

import os
import sys
from dataclasses import dataclass
from pathlib import Path

UPSTREAM = Path(joined('PROTEINA_COMPLEXA_ROOT'))
REPO = Path(ROOT)

CKPTS = {
    "binder": ("complexa.ckpt", "complexa_ae.ckpt"),
    "ame": ("complexa_ame.ckpt", "complexa_ame_ae.ckpt"),
    "antibody": ("complexa_ame.ckpt", "complexa_ame_ae.ckpt"),
}
PIPELINES = {
    "binder": "search_binder_local_pipeline",
    "ame": "search_ame_local_pipeline",
    "antibody": "search_antibody_local_pipeline",
}
ARCH_V2 = {"binder": False, "ame": True, "antibody": True}

CONDITION_KEYS = {
    "binder": [("x_target", "seq_target_mask")],
    "ame": [("x_motif", "seq_motif_mask"), ("x_target", "target_mask")],
    "antibody": [("x_motif", "seq_motif_mask"), ("x_target", "target_mask")],
}

FLAG_CONDITION_KEYS = {
    "binder": ["target_hotspot_mask"],
    "ame": [],
    "antibody": [],
}

@dataclass(frozen=True)
class FamilySpec:
    family: str
    checkpoint: Path
    autoencoder: Path
    pipeline: str
    architecture_v2: bool

def spec(family: str) -> FamilySpec:
    if family not in CKPTS:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(CKPTS)}")
    ckpt, ae = CKPTS[family]
    return FamilySpec(
        family=family,
        checkpoint=UPSTREAM / "ckpts" / ckpt,
        autoencoder=UPSTREAM / "ckpts" / ae,
        pipeline=PIPELINES[family],
        architecture_v2=ARCH_V2[family],
    )

def bootstrap(family: str) -> FamilySpec:

    s = spec(family)
    already = os.environ.get("USE_V2_COMPLEXA_ARCH")
    want = "True" if s.architecture_v2 else "False"
    if "proteinfoundation.proteina" in sys.modules and already != want:
        raise RuntimeError(
            "proteinfoundation.proteina is already imported under "
            f"USE_V2_COMPLEXA_ARCH={already!r}; this family needs {want!r}. "
            "Run one family per process."
        )
    os.environ["USE_V2_COMPLEXA_ARCH"] = want
    os.environ["DATA_PATH"] = str(UPSTREAM / "assets")
    os.environ.setdefault("AF2_DIR", str(joined("CODIRECT_AF2_ROOT")))
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for p in (UPSTREAM / "src", REPO / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    os.chdir(UPSTREAM)
    if s.architecture_v2:

        import proteinfoundation.patches.atomworks_patches
    return s

def _dive_config_dir() -> str:
    return str(REPO / "configs" / "emergent" / "upstream_ext")

def compose(
    family: str,
    *,
    task: str,
    seed: int,
    nsteps: int,
    nsamples: int,
    extra_overrides: list[str] | None = None,
    drop_target_feature: bool = False,
):

    from hydra import compose as hydra_compose, initialize_config_dir
    from omegaconf import open_dict

    s = spec(family)
    if family == "antibody":
        primary, other = _dive_config_dir(), str(UPSTREAM / "configs")
    else:
        primary, other = str(UPSTREAM / "configs"), _dive_config_dir()
    overrides = [
        f"hydra.searchpath=[file://{other}]",
        f"++seed={seed}",
        f"++ckpt_path={s.checkpoint.parent}",
        f"++ckpt_name={s.checkpoint.name}",
        f"++autoencoder_ckpt_path={s.autoencoder}",
        f"++generation.task_name={task}",
        f"++generation.args.nsteps={nsteps}",
        "++generation.args.guidance_w=1.0",
        "++generation.args.ag_ratio=0.0",
        "++generation.args.self_cond=True",
        "++generation.n_recycle=0",
        f"++generation.dataloader.batch_size={nsamples}",
        f"++generation.dataloader.dataset.nres.nsamples={nsamples}",
        "++generation.search.algorithm=single-pass",
    ]
    overrides += list(extra_overrides or [])
    with initialize_config_dir(version_base=None, config_dir=primary, job_name=f"tfd-{family}"):
        cfg = hydra_compose(config_name=s.pipeline, overrides=overrides)
    if drop_target_feature:
        with open_dict(cfg):
            ds = cfg.generation.dataloader.dataset
            ds.conditional_features = [ds.conditional_features[0]]
            cfg.generation.dataloader.dataset.transforms[0].additional_tensors = []
    return cfg

def load_model(cfg, device: str = "cuda:0"):

    import torch
    from proteinfoundation.generate import load_ckpt_n_configure_inference

    model = load_ckpt_n_configure_inference(cfg)
    model = model.to(torch.device(device)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    return model

def build_batch(cfg, device: str = "cuda:0") -> dict:

    import hydra
    import torch

    loader = hydra.utils.instantiate(cfg.generation.dataloader, _convert_="all")
    raw = next(iter(loader))
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw.items()}
