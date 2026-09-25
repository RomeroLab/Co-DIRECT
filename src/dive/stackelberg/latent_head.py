
from __future__ import annotations

import torch

def freeze_all_but_latent_head(module: torch.nn.Module) -> list[torch.nn.Parameter]:

    trainable: list[torch.nn.Parameter] = []
    for name, param in module.named_parameters():
        if "local_latents_linear" in name:
            param.requires_grad_(True)
            trainable.append(param)
        else:
            param.requires_grad_(False)
    if not trainable:
        raise ValueError(
            "no local_latents_linear parameters; refusing to train nothing")
    return trainable
