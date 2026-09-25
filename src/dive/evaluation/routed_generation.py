
from __future__ import annotations

from typing import Any, Mapping

BASE_PREFIX = "base."
ROUTER_PREFIX = "router."

def split_module_state(
    state_dict: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:

    if not state_dict:
        raise ValueError("empty state dict; nothing to load")

    base: dict[str, Any] = {}
    router: dict[str, Any] = {}
    unknown: set[str] = set()
    for key, value in state_dict.items():
        if key.startswith(BASE_PREFIX):
            base[key[len(BASE_PREFIX):]] = value
        elif key.startswith(ROUTER_PREFIX):
            router[key[len(ROUTER_PREFIX):]] = value
        else:
            unknown.add(key.split(".")[0])

    if unknown:
        raise ValueError(
            f"unrecognised top-level prefixes {sorted(unknown)}; refusing to "
            f"drop them, because a partial load reports no error"
        )
    if not base:
        raise ValueError("no base parameters; this is not a Co-DIRECT checkpoint")
    if not router:
        raise ValueError(
            "no router parameters; a routed arm cannot default to an untrained "
            "router without silently becoming a fixed-regime arm"
        )
    return base, router

def load_routed_generator(
    checkpoint,
    *,
    contract,
    store_dir,
    generation_cfg=None,
    device: str = "cuda",
):

    import torch

    from dive.evaluation.fixed_regimes import routed_predictor
    from dive.integrations.complexa_training import build_common_v2_model
    from dive.integrations.flow import proteina_bindings
    from dive.leadership.router import LeadershipRouter
    from dive.training.checkpoint_stage import stage_of_checkpoint
    from dive.training.module import DiveLeadershipModule

    stage = stage_of_checkpoint(checkpoint)

    model, _report = build_common_v2_model(contract, store_dir=store_dir)
    model = model.to(device)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base_state, router_state = split_module_state(payload["state_dict"])

    model.load_state_dict(base_state, strict=True)
    router = LeadershipRouter(hidden_dim=int(model.cfg_exp.nn.token_dim)).to(device)
    router.load_state_dict(router_state, strict=True)

    bindings = proteina_bindings(model)
    module = DiveLeadershipModule(
        base=model,
        router=router,
        stage=stage,
        call_denoiser=bindings.call_denoiser,
        flow_loss=bindings.flow_loss,
        per_residue_flow_loss=bindings.per_residue_flow_loss,
        allow_surrogate_loss=False,
    ).to(device)
    module.eval()

    model.predict_for_sampling = routed_predictor(module)

    if generation_cfg is not None:
        from proteinfoundation.generate import load_ag_ckpt

        model.inf_cfg = generation_cfg
        model.nn_ag = load_ag_ckpt(generation_cfg.args)

    return model, module, stage
