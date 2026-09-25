
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

TIME_EMBED_DIM = 32

MASK_DIM = 4

HIDDEN_WIDTH = 128
DROPOUT = 0.1

OUTPUT_WEIGHT_SCALE = 0.01

class RouterError(RuntimeError):
    pass

@dataclass(slots=True)
class DirectionalGates:

    z_to_x: Tensor
    x_to_z: Tensor

def sinusoidal_time_embedding(t: Tensor, dim: int = TIME_EMBED_DIM) -> Tensor:

    if dim % 2 != 0:
        raise RouterError(f"time embedding dim must be even, got {dim}")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

class LeadershipRouter(nn.Module):

    def __init__(
        self,
        *,
        hidden_dim: int,
        width: int = HIDDEN_WIDTH,
        initial_logits: tuple[float, float] | None = None,
        output_scale: float = OUTPUT_WEIGHT_SCALE,
    ) -> None:

        super().__init__()
        self.hidden_dim = hidden_dim
        in_features = 3 * hidden_dim + 2 * TIME_EMBED_DIM + MASK_DIM
        self.net = nn.Sequential(
            nn.Linear(in_features, width),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(width, 2),
        )
        self.initial_logits = initial_logits
        if initial_logits is not None:
            z_to_x, x_to_z = initial_logits
            with torch.no_grad():
                self.net[-1].weight.mul_(output_scale)
                self.net[-1].bias.copy_(
                    torch.tensor([z_to_x, x_to_z], dtype=self.net[-1].bias.dtype)
                )

        self.norm = nn.LayerNorm(3 * hidden_dim)

    def forward(
        self,
        *,
        h_joint: Tensor,
        h_without_latent: Tensor,
        h_without_backbone: Tensor,
        t_backbone: Tensor,
        t_latent: Tensor,
        generated_mask: Tensor,
        fixed_mask: Tensor,
        presence: Tensor,
    ) -> DirectionalGates:

        return DirectionalGates(*self._gates_from(self._assemble(
            h_joint=h_joint,
            h_without_latent=h_without_latent,
            h_without_backbone=h_without_backbone,
            t_backbone=t_backbone,
            t_latent=t_latent,
            generated_mask=generated_mask,
            fixed_mask=fixed_mask,
            presence=presence,
        )))

    def _gates_from(self, features: Tensor) -> tuple[Tensor, Tensor]:
        gates = torch.sigmoid(self.net(features))
        return gates[..., 0], gates[..., 1]

    def _assemble(
        self,
        *,
        h_joint: Tensor,
        h_without_latent: Tensor,
        h_without_backbone: Tensor,
        t_backbone: Tensor,
        t_latent: Tensor,
        generated_mask: Tensor,
        fixed_mask: Tensor,
        presence: Tensor,
    ) -> Tensor:
        expected = h_joint.shape
        for name, tensor in (
            ("h_without_latent", h_without_latent),
            ("h_without_backbone", h_without_backbone),
        ):
            if tensor.shape != expected:
                raise RouterError(
                    f"{name} shape {tuple(tensor.shape)} does not match h_joint "
                    f"{tuple(expected)}"
                )
        if expected[-1] != self.hidden_dim:
            raise RouterError(
                f"hidden width {expected[-1]} does not match the configured "
                f"{self.hidden_dim}"
            )

        batch, residues, _ = expected

        features = self.norm(
            torch.cat(
                [h_joint, h_joint - h_without_backbone, h_joint - h_without_latent], dim=-1
            )
        )

        times = torch.cat(
            [
                sinusoidal_time_embedding(t_backbone),
                sinusoidal_time_embedding(t_latent),
            ],
            dim=-1,
        ).unsqueeze(1).expand(batch, residues, 2 * TIME_EMBED_DIM)

        masks = torch.stack(
            [
                generated_mask.to(features.dtype),
                fixed_mask.to(features.dtype),
                presence[:, 0].to(features.dtype).unsqueeze(1).expand(batch, residues),
                presence[:, 1].to(features.dtype).unsqueeze(1).expand(batch, residues),
            ],
            dim=-1,
        )

        return torch.cat([features, times, masks], dim=-1)

    def input_features(self, **inputs) -> Tensor:

        return self._assemble(**inputs)

class TiedLeadershipRouter(LeadershipRouter):

    def __init__(
        self,
        *,
        hidden_dim: int,
        width: int = HIDDEN_WIDTH,
        initial_logit: float | tuple[float, float] | None = None,
        output_scale: float = OUTPUT_WEIGHT_SCALE,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            width=width,
            initial_logits=None,
            output_scale=output_scale,
        )
        if isinstance(initial_logit, tuple):
            z_to_x, x_to_z = initial_logit
            if float(z_to_x) != float(x_to_z):
                raise RouterError(
                    f"asymmetric initial regime ({z_to_x}, {x_to_z}) given to the "
                    f"tied-gate control, which has one gate. Averaging it would "
                    f"discard the asymmetry this arm exists NOT to have and record "
                    f"a starting regime the model never occupied. Choose a tied "
                    f"regime, or use the free router."
                )
            initial_logit = float(z_to_x)

        self.net[-1] = nn.Linear(width, 1)
        self.initial_logit = initial_logit
        self.initial_logits = (
            None if initial_logit is None else (float(initial_logit), float(initial_logit))
        )
        if initial_logit is not None:
            with torch.no_grad():
                self.net[-1].weight.mul_(output_scale)
                self.net[-1].bias.copy_(
                    torch.tensor([float(initial_logit)], dtype=self.net[-1].bias.dtype)
                )

    def _gates_from(self, features: Tensor) -> tuple[Tensor, Tensor]:

        gate = torch.sigmoid(self.net(features))[..., 0]
        return gate, gate

ROUTER_CONFIG_KEYS = frozenset({"tied", "initial_regime_scan_run"})

def build_router(
    router_config: "dict | None",
    *,
    hidden_dim: int,
    initial_logits: tuple[float, float] | None,
) -> LeadershipRouter:

    config = dict(router_config or {})
    unknown = sorted(set(config) - ROUTER_CONFIG_KEYS)
    if unknown:
        raise RouterError(
            f"unknown router config key(s) {unknown}; expected a subset of "
            f"{sorted(ROUTER_CONFIG_KEYS)}"
        )
    if config.get("tied", False):
        return TiedLeadershipRouter(hidden_dim=hidden_dim, initial_logit=initial_logits)
    return LeadershipRouter(hidden_dim=hidden_dim, initial_logits=initial_logits)
