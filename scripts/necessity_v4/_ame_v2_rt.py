
from __future__ import annotations
import os, sys
from pathlib import Path

_CODIRECT = Path(__file__).resolve().parents[2]
import sys as _sys
if str(_CODIRECT / "src") not in _sys.path:
    _sys.path.insert(0, str(_CODIRECT / "src"))
from dive.codirect_paths import proteina_root, proteina_src

UPSTREAM = proteina_root()
CKPT = UPSTREAM / "ckpts/complexa_ame.ckpt"
AE = UPSTREAM / "ckpts/complexa_ame_ae.ckpt"

def bootstrap() -> None:
    os.environ["USE_V2_COMPLEXA_ARCH"] = "True"
    os.environ.setdefault("DATA_PATH", str(UPSTREAM / "assets"))
    for p in (proteina_src(), _CODIRECT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import proteinfoundation.patches.atomworks_patches

def load(device: str = "cuda:0"):

    import torch
    from proteinfoundation.proteina import Proteina
    from proteinfoundation.utils.lora_utils import replace_lora_layers

    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    lc = ck["hyper_parameters"]["cfg_exp"]["lora"]
    model = Proteina.load_from_checkpoint(
        str(CKPT), strict=False, autoencoder_ckpt_path=str(AE), map_location="cpu")
    if type(model.nn).__module__ != "proteinfoundation.nn.local_latents_transformer_v2":
        raise RuntimeError(f"wrong architecture module: {type(model.nn).__module__}")
    replace_lora_layers(model.nn, lc["r"], lc["lora_alpha"], lc["lora_dropout"])
    res = model.load_state_dict(ck["state_dict"], strict=False)
    missing = [k for k in res.missing_keys if k.startswith("nn.")]
    unexpected = [k for k in res.unexpected_keys if k.startswith("nn.")]
    if missing or unexpected:
        raise RuntimeError(f"AME load not exact: missing={missing[:5]} unexpected={unexpected[:5]}")

    sd_ck = ck["state_dict"]
    live = dict(model.state_dict())
    lora_keys = [k for k in sd_ck if k.endswith(("lora_A", "lora_B"))]
    landed = sum(1 for k in lora_keys
                 if k in live and torch.equal(live[k].cpu(), sd_ck[k].cpu()))
    nonzero_b = sum(1 for k in lora_keys if k.endswith("lora_B") and float(sd_ck[k].abs().sum()) > 0)
    info = {"module": type(model.nn).__module__,
            "lora_cfg": {kk: (float(vv) if isinstance(vv, (int, float)) else vv)
                         for kk, vv in dict(lc).items()},
            "lora_tensors": len(lora_keys), "lora_landed_bitwise": landed,
            "lora_B_nonzero": nonzero_b,
            "exact_load": True, "missing": 0, "unexpected": 0,
            "concat_pair_module": type(model.nn.concat_pair_factory).__module__,
            "concat_seq_module": type(model.nn.concat_factory).__module__}
    del ck, sd_ck, live
    if landed != len(lora_keys):
        raise RuntimeError(f"only {landed}/{len(lora_keys)} LoRA tensors landed bitwise")
    if getattr(model, "autoencoder", None) is None or model.autoencoder.decoder is None:
        raise RuntimeError("AME autoencoder decoder missing")
    model = model.to(torch.device(device)).eval()
    for p in model.parameters():
        p.requires_grad_(False); p.grad = None
    model.autoencoder.eval()
    for p in model.autoencoder.parameters():
        p.requires_grad_(False)
    return model, info

def ligand_resname(pdb: Path, declared: str | None = None) -> str:

    names = {l[17:20].strip() for l in pdb.read_text().splitlines()
             if l.startswith("HETATM")}
    if not names:
        raise RuntimeError(f"no HETATM records in {pdb}")
    if isinstance(declared, (list, tuple)):

        want = [str(x).strip() for x in declared]
        if all(w in names for w in want):
            return declared
        raise RuntimeError(f"declared components {want} not all present in {pdb.name}: {sorted(names)}")
    if declared and str(declared).strip() in names:
        return str(declared).strip()
    if len(names) == 1:
        return names.pop()
    raise RuntimeError(
        f"declared ligand {declared!r} is not in {pdb.name}, and the file has "
        f"several HETATM residues {sorted(names)} so the selector is ambiguous")

def config(task: str, *, nsteps: int, nsamples: int):
    import yaml
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    td = yaml.safe_load((UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    entry = td["motif_target_dict_cfg"][task]
    pdb = UPSTREAM / str(entry["target_path"]).lstrip("./")
    lig = ligand_resname(pdb, entry.get("ligand"))
    ov = [f"++pipeline.ame.task_name={task}"]
    if not isinstance(lig, (list, tuple)) and lig != str(entry.get("ligand")):
        ov.append(f"++pipeline.ame.motif_target_dict_cfg.{task}.ligand={lig}")
    with initialize_config_dir(config_dir=str(UPSTREAM / "configs"), version_base="1.3"):
        cfg = compose(config_name="pipeline/ame/ame_generate", overrides=ov)
    gen = cfg.pipeline.ame
    with open_dict(gen):
        gen.dataloader.batch_size = nsamples
        gen.dataloader.dataset.nres.nsamples = nsamples
        gen.dataloader.dataset.nrepeat_per_sample = 1
        gen.args.nsteps = nsteps
        gen.search.algorithm = "single-pass"
        gen.refinement.algorithm = None
        gen.reward_model = None
    return gen, {"task": task, "pdb": str(pdb), "ligand_in_dict": entry.get("ligand"),
                 "ligand_in_pdb": lig,
                 "override_applied": (not isinstance(lig, (list, tuple)))
                                     and lig != str(entry.get("ligand"))}
