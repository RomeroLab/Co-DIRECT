
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import torch
from torch import Tensor, nn

ROUTE_ORDER = ("none", "z_to_x", "x_to_z", "bidirectional")
TIME_EMBED_DIM = 32
PRESENCE_EMBED_DIM = 8
ROUTER_WIDTH = 128

class DirectionalBridgeError(RuntimeError):
    pass

class RouteState(StrEnum):
    NONE = "none"
    Z_TO_X = "z_to_x"
    X_TO_Z = "x_to_z"
    BIDIRECTIONAL = "bidirectional"
    ADAPTIVE = "adaptive"

    @classmethod
    def hard_states(cls) -> tuple["RouteState", ...]:
        return (cls.NONE, cls.Z_TO_X, cls.X_TO_Z, cls.BIDIRECTIONAL)

@dataclass(frozen=True, slots=True)
class RouteProbabilities:

    values: Tensor

@dataclass(frozen=True, slots=True)
class BridgeTelemetry:

    probabilities: Tensor
    applied_z_to_x: Tensor
    applied_x_to_z: Tensor

@dataclass(frozen=True, slots=True)
class BridgeOutput:

    h_x: Tensor
    h_z: Tensor
    telemetry: BridgeTelemetry

def _time_embedding(times: Tensor) -> Tensor:
    half = TIME_EMBED_DIM // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=times.device)
        / half
    )
    angles = times.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise DirectionalBridgeError(
            f"{name} shape {tuple(value.shape)} does not match expected {shape}"
        )

def _require_finite(name: str, value: Tensor) -> None:
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise DirectionalBridgeError(
            f"{name} must contain only finite floating-point values"
        )

class FourWayRouter(nn.Module):

    def __init__(self, *, hidden_dim: int, width: int = ROUTER_WIDTH) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise DirectionalBridgeError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(3 * hidden_dim)
        self.presence_embedding = nn.Embedding(4, PRESENCE_EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(
                3 * hidden_dim + 2 * TIME_EMBED_DIM + 2 + PRESENCE_EMBED_DIM, width
            ),
            nn.GELU(),
            nn.Linear(width, 4),
        )

    def forward(
        self,
        h_x: Tensor,
        h_z: Tensor,
        t_backbone: Tensor,
        t_latent: Tensor,
        generated_mask: Tensor,
        fixed_mask: Tensor,
        presence: Tensor,
    ) -> RouteProbabilities:

        if h_x.ndim != 3:
            raise DirectionalBridgeError(
                f"h_x shape {tuple(h_x.shape)} must be [B, N, D]"
            )
        batch, residues, hidden = h_x.shape
        if hidden != self.hidden_dim:
            raise DirectionalBridgeError(
                f"h_x hidden width {hidden} does not match configured {self.hidden_dim}"
            )
        _require_shape("h_z", h_z, (batch, residues, hidden))
        _require_shape("t_backbone", t_backbone, (batch,))
        _require_shape("t_latent", t_latent, (batch,))
        _require_shape("generated_mask", generated_mask, (batch, residues))
        _require_shape("fixed_mask", fixed_mask, (batch, residues))
        _require_shape("presence", presence, (batch, 2))
        for name, value in (
            ("h_x", h_x),
            ("h_z", h_z),
            ("t_backbone", t_backbone),
            ("t_latent", t_latent),
        ):
            _require_finite(name, value)
        for name, mask in (
            ("generated_mask", generated_mask),
            ("fixed_mask", fixed_mask),
        ):
            if mask.dtype != torch.bool:
                raise DirectionalBridgeError(f"{name} must have boolean dtype")
        if not torch.all((presence == 0) | (presence == 1)):
            raise DirectionalBridgeError("presence values must be binary")

        hidden_features = self.norm(torch.cat((h_x, h_z, h_x - h_z), dim=-1))
        times = torch.cat(
            (_time_embedding(t_backbone), _time_embedding(t_latent)), dim=-1
        )
        times = times.unsqueeze(1).expand(batch, residues, -1)
        masks = torch.stack((generated_mask, fixed_mask), dim=-1).to(h_x.dtype)
        presence_code = presence[:, 0].long() * 2 + presence[:, 1].long()
        presence_features = (
            self.presence_embedding(presence_code)
            .unsqueeze(1)
            .expand(batch, residues, -1)
        )
        logits = self.net(
            torch.cat((hidden_features, times, masks, presence_features), dim=-1)
        )
        if not torch.isfinite(logits).all():
            raise DirectionalBridgeError("router logits must be finite")
        return RouteProbabilities(torch.softmax(logits, dim=-1))

def hard_route_probabilities(
    route: RouteState, *, batch: int, residues: int, device: torch.device | str
) -> Tensor:

    try:
        index = RouteState.hard_states().index(RouteState(route))
    except ValueError as exc:
        raise DirectionalBridgeError(f"{route!r} is not a hard route") from exc
    if batch < 0 or residues < 0:
        raise DirectionalBridgeError("batch and residues must be non-negative")
    values = torch.zeros((batch, residues, len(ROUTE_ORDER)), device=device)
    values[..., index] = 1.0
    return values

class _LowRankExpert(nn.Module):

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.activation = nn.SiLU()
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.up(self.activation(self.down(self.norm(hidden))))

class DirectionalBridge(nn.Module):

    def __init__(self, *, hidden_dim: int, rank: int = 8) -> None:
        super().__init__()
        if hidden_dim <= 0 or rank <= 0:
            raise DirectionalBridgeError("hidden_dim and rank must be positive")
        self.hidden_dim = hidden_dim
        self.z_to_x = _LowRankExpert(hidden_dim, rank)
        self.x_to_z = _LowRankExpert(hidden_dim, rank)
        initial_scale = math.atanh(0.10)
        self.raw_scale_x = nn.Parameter(torch.tensor(initial_scale))
        self.raw_scale_z = nn.Parameter(torch.tensor(initial_scale))

    def forward(
        self,
        h_x: Tensor,
        h_z: Tensor,
        probabilities: RouteProbabilities,
        mask: Tensor,
        presence: Tensor,
    ) -> BridgeOutput:

        if h_x.ndim != 3:
            raise DirectionalBridgeError(
                f"h_x shape {tuple(h_x.shape)} must be [B, N, D]"
            )
        batch, residues, hidden = h_x.shape
        if hidden != self.hidden_dim:
            raise DirectionalBridgeError(
                f"h_x hidden width {hidden} does not match configured {self.hidden_dim}"
            )
        _require_shape("h_z", h_z, (batch, residues, hidden))
        _require_shape("mask", mask, (batch, residues))
        _require_shape("presence", presence, (batch, 2))
        _require_shape(
            "probabilities", probabilities.values, (batch, residues, len(ROUTE_ORDER))
        )
        for name, value in (
            ("h_x", h_x),
            ("h_z", h_z),
            ("probabilities", probabilities.values),
        ):
            _require_finite(name, value)
        if mask.dtype != torch.bool:
            raise DirectionalBridgeError("mask must have boolean dtype")
        if not torch.all((presence == 0) | (presence == 1)):
            raise DirectionalBridgeError("presence values must be binary")

        m_zx = self.z_to_x(h_z)
        m_xz = self.x_to_z(h_x)
        _, p_zx, p_xz, p_bi = probabilities.values.unbind(-1)
        gate_zx = (p_zx + p_bi).to(dtype=m_zx.dtype).unsqueeze(-1)
        gate_xz = (p_xz + p_bi).to(dtype=m_xz.dtype).unsqueeze(-1)
        scale_x = torch.tanh(self.raw_scale_x).to(dtype=m_zx.dtype)
        scale_z = torch.tanh(self.raw_scale_z).to(dtype=m_xz.dtype)
        applied_zx = scale_x * gate_zx * m_zx
        applied_xz = scale_z * gate_xz * m_xz
        valid_z_source = mask & presence[:, 1].bool().unsqueeze(-1)
        valid_x_source = mask & presence[:, 0].bool().unsqueeze(-1)
        applied_zx = applied_zx * valid_z_source.unsqueeze(-1)
        applied_xz = applied_xz * valid_x_source.unsqueeze(-1)
        return BridgeOutput(
            h_x=h_x + applied_zx,
            h_z=h_z + applied_xz,
            telemetry=BridgeTelemetry(
                probabilities=probabilities.values,
                applied_z_to_x=applied_zx,
                applied_x_to_z=applied_xz,
            ),
        )

MUTABLE_SOURCE_REALM_EXCLUSIONS: Mapping[str, type] = MappingProxyType(
    {
        "dive.leadership.directional_bridge.RouteState._member_names_": list,
        "dive.leadership.directional_bridge.RouteState._member_map_": dict,
        "dive.leadership.directional_bridge.RouteState._value2member_map_": dict,
        "dive.leadership.directional_bridge.RouteState._unhashable_values_": list,
    }
)
