
from __future__ import annotations

REQUIRED_PATHWAYS = ("enable_ligand", "ligand_pair_features",
                     "enable_motif", "motif_pair_features")

class AmeCheckpointError(RuntimeError):
    pass

def assert_ame_pathways(flags) -> None:

    off = [k for k in REQUIRED_PATHWAYS if not flags.get(k)]
    if off:
        raise AmeCheckpointError(
            f"AME conditioning is not enabled in this checkpoint: {off} are "
            f"absent or false. The common checkpoint has no ligand pathway and "
            f"builds no motif features; using it here is the defect this loader "
            f"exists to prevent."
        )

def count_lora_tensors(keys) -> int:

    return sum(1 for k in keys if k.endswith((".lora_A", ".lora_B")))

def lora_scaling(cfg) -> float:

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass
    node = (cfg or {}).get("lora") if isinstance(cfg, dict) else None
    r = (node or {}).get("r")
    alpha = (node or {}).get("lora_alpha")
    if not r or alpha is None:
        raise AmeCheckpointError(
            f"checkpoint does not declare lora.r and lora.lora_alpha (got "
            f"r={r!r}, alpha={alpha!r}); refusing to guess a scaling")
    return float(alpha) / float(r)

def merge_lora(state_dict, scaling):

    import torch

    out = {k: v for k, v in state_dict.items()}
    prefixes = {k[: -len(".lora_A")] for k in state_dict if k.endswith(".lora_A")}
    for pre in sorted(prefixes):
        a, b, w = f"{pre}.lora_A", f"{pre}.lora_B", f"{pre}.weight"
        if w not in out:
            raise AmeCheckpointError(
                f"adapter {pre} has no base weight {w}; merging would drop it")
        delta = (out[b].to(torch.float32) @ out[a].to(torch.float32)) * float(scaling)
        out[w] = (out[w].to(torch.float32) + delta).to(out[w].dtype)
        out.pop(a, None)
        out.pop(b, None)
    return out

def concat_feature_flags(cfg) -> dict:

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass
    node = cfg
    for part in ("nn", "concat_features"):
        node = (node or {}).get(part, {}) if isinstance(node, dict) else getattr(node, part, {})
    return dict(node or {})

def build_ame_specialist(checkpoint_path, autoencoder_path, store_dir):

    import torch
    from proteinfoundation.proteina import Proteina

    import os
    if os.getenv("USE_V2_COMPLEXA_ARCH") != "True":

        raise AmeCheckpointError(
            "USE_V2_COMPLEXA_ARCH must be the string 'True' before importing "
            "proteinfoundation: the v1 feature factory builds motif only and "
            "would leave the ligand pathway unbuilt")

    ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ck["hyper_parameters"]["cfg_exp"]
    flags = concat_feature_flags(cfg)
    assert_ame_pathways(flags)

    scaling = lora_scaling(cfg)
    n_lora = count_lora_tensors(ck["state_dict"].keys())
    merged = merge_lora(ck["state_dict"], scaling)

    model = Proteina(cfg, store_dir=str(store_dir),
                     autoencoder_ckpt_path=str(autoencoder_path))
    result = model.load_state_dict(merged, strict=False)
    missing = tuple(k for k in result.missing_keys if k.startswith("nn."))
    unexpected = tuple(k for k in result.unexpected_keys if k.startswith("nn."))
    if missing or unexpected:
        raise AmeCheckpointError(
            f"native AME load is not exact after merging {n_lora} LoRA factors at "
            f"scaling {scaling}: missing={missing[:6]} unexpected={unexpected[:6]}")
    return model, {
        "checkpoint": str(checkpoint_path),
        "autoencoder": str(autoencoder_path),
        "concat_feature_flags": flags,
        "lora_tensors_in_checkpoint": n_lora,
        "lora_scaling_applied": scaling,
        "provenance_note": (
            "complexa_ame.ckpt is marked unknown_provenance upstream and carries "
            "LoRA factors of its own; it is used here because it is the only "
            "checkpoint with the ligand and motif pathways this task needs"),
    }
