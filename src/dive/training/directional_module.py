
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor, nn

from dive.data.mixture import FamilyBatch, unwrap_family_batch
from dive.integrations.directional_v2 import DirectionalForward
from dive.integrations.flow import DirectionalProteinaBindings
from dive.leadership.directional_bridge import BridgeTelemetry, RouteState
from dive.training.directional_objective import (
    DirectionalLossBreakdown,
    DirectionalStage,
    RoutePrediction,
    directional_objective,
)

class DirectionalModuleError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class RoutedBatch:

    prediction: RoutePrediction
    per_residue_loss: Tensor
    telemetry: tuple[BridgeTelemetry, ...]

class DirectionalTrainingModule(nn.Module):

    def __init__(
        self,
        *,
        base: nn.Module,
        bindings: DirectionalProteinaBindings,
        stage: DirectionalStage = DirectionalStage.BRIDGE_WARMUP,
        rank_override: float | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.bindings = bindings
        if rank_override is not None and (
            isinstance(rank_override, bool)
            or not isinstance(rank_override, Real)
            or not math.isfinite(rank_override)
            or rank_override != 0.0
        ):
            raise DirectionalModuleError(
                "rank_override may only be None or exactly 0.0"
            )
        self.rank_override = rank_override

        self.adapter
        self.stage = DirectionalStage.BRIDGE_WARMUP
        self.set_stage(stage)

    @property
    def adapter(self) -> nn.Module:
        adapter = getattr(self.base, "nn", None)
        if not callable(getattr(adapter, "forward_route", None)):
            raise DirectionalModuleError("base.nn must be a directional adapter")
        return adapter

    def set_stage(self, stage: DirectionalStage | str) -> None:

        try:
            selected = DirectionalStage(stage)
        except ValueError as error:
            raise DirectionalModuleError(
                f"unknown directional stage {stage!r}"
            ) from error
        self.stage = selected
        allowed = {"bridge", "router"}
        if selected is DirectionalStage.JOINT_ADAPTATION:
            allowed.add("lora")

        for name, parameter in self.named_parameters():
            parameter.requires_grad_(_parameter_group(name) in allowed)

        expected = self._expected_owned_names(selected)
        actual = self._owned_names()
        if actual != expected:
            raise DirectionalModuleError(
                "stage ownership did not apply exactly: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
        expected_groups = {"bridge", "router"}
        if selected is DirectionalStage.JOINT_ADAPTATION:
            expected_groups.add("lora")
        present_groups = {_parameter_group(name) for name in expected}
        missing = sorted(expected_groups - present_groups)
        if missing:
            raise DirectionalModuleError(
                f"stage {selected.value} owns missing parameter group(s) {missing}"
            )

        autoencoder = getattr(self.base, "autoencoder", None)
        if autoencoder is not None:
            leaked = [
                name
                for name, parameter in autoencoder.named_parameters()
                if parameter.requires_grad
            ]
            if leaked:
                raise DirectionalModuleError(
                    f"autoencoder parameters must remain frozen: {leaked}"
                )

    def trainable_parameter_groups(self) -> list[dict[str, Any]]:

        expected = self._expected_owned_names(self.stage)
        actual = self._owned_names()
        if actual != expected:
            unexpected = sorted(actual - expected)
            missing = sorted(expected - actual)
            if unexpected:
                raise DirectionalModuleError(
                    f"parameter {unexpected[0]!r} is trainable but belongs to no "
                    "directional optimizer group for this stage"
                )
            raise DirectionalModuleError(
                f"stage {self.stage.value} is missing trainable parameter(s) {missing}"
            )

        grouped: dict[str, list[nn.Parameter]] = {
            "bridge": [],
            "router": [],
            "lora": [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            group = _parameter_group(name)
            if group is None:
                raise DirectionalModuleError(
                    f"parameter {name!r} is trainable but belongs to no "
                    "directional optimizer group"
                )
            grouped[group].append(parameter)
        return [
            {"name": name, "params": parameters}
            for name, parameters in grouped.items()
            if parameters
        ]

    def ownership_manifest(self) -> dict[str, Any]:

        groups = self.trainable_parameter_groups()
        names_by_id = {
            id(parameter): name for name, parameter in self.named_parameters()
        }
        grouped_names = {
            group["name"]: sorted(
                names_by_id[id(parameter)] for parameter in group["params"]
            )
            for group in groups
        }
        trainable_names = sorted(self._owned_names())
        return {
            "stage": self.stage.value,
            "trainable_names": trainable_names,
            "trainable_name_count": len(trainable_names),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "groups": grouped_names,
        }

    def _owned_names(self) -> set[str]:
        return {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def _expected_owned_names(self, stage: DirectionalStage) -> set[str]:
        allowed = {"bridge", "router"}
        if stage is DirectionalStage.JOINT_ADAPTATION:
            allowed.add("lora")
        return {
            name
            for name, _ in self.named_parameters()
            if _parameter_group(name) in allowed
        }

    def route_batch(self, corrupted: dict[str, Any], route: RouteState) -> RoutedBatch:

        try:
            route = RouteState(route)
        except ValueError as error:
            raise DirectionalModuleError(
                f"unknown directional route {route!r}"
            ) from error
        if route is not RouteState.ADAPTIVE:
            with torch.no_grad():
                routed = self._route_batch(corrupted, route)
            return _detach_routed_batch(routed)
        return self._route_batch(corrupted, route)

    def _route_batch(self, corrupted: dict[str, Any], route: RouteState) -> RoutedBatch:
        forwarded = self.bindings.call_route(self.base, corrupted, route)
        prediction = _route_prediction(forwarded, self.bindings.output_key)
        per_residue = self.bindings.per_residue_flow_loss(
            prediction.bb_ca,
            prediction.local_latents,
            corrupted,
        )
        telemetry = tuple(forwarded.telemetry)
        if route is RouteState.ADAPTIVE and len(telemetry) != 3:
            raise DirectionalModuleError(
                "directional objective requires exactly three adaptive telemetry taps, "
                f"got {len(telemetry)}"
            )
        return RoutedBatch(
            prediction=prediction,
            per_residue_loss=per_residue,
            telemetry=telemetry,
        )

    def forward(
        self, batch: FamilyBatch | Mapping[str, Any], step: int
    ) -> DirectionalLossBreakdown:

        return self.training_step(batch, step=step)

    def training_step(
        self, batch: FamilyBatch | Mapping[str, Any], *, step: int
    ) -> DirectionalLossBreakdown:

        model_batch = _unwrap_model_batch(batch)
        corrupted = self.bindings.corrupt(model_batch)
        routed = {
            route: self.route_batch(corrupted, route)
            for route in (RouteState.ADAPTIVE, *RouteState.hard_states())
        }
        adaptive = routed[RouteState.ADAPTIVE]
        none = routed[RouteState.NONE]
        probabilities = torch.stack(
            [record.probabilities for record in adaptive.telemetry], dim=0
        )
        valid_mask = _valid_generated_mask(corrupted)
        return directional_objective(
            adaptive_losses=adaptive.per_residue_loss,
            hard_losses=torch.stack(
                [routed[route].per_residue_loss for route in RouteState.hard_states()],
                dim=-1,
            ),
            adaptive_prediction=adaptive.prediction,
            none_prediction=none.prediction,
            probabilities=probabilities,
            valid_mask=valid_mask,
            stage=self.stage,
            step=step,
            rank_override=self.rank_override,
        )

def _route_prediction(
    forwarded: DirectionalForward, output_key: str
) -> RoutePrediction:
    try:
        return RoutePrediction(
            bb_ca=forwarded.nn_out["bb_ca"][output_key],
            local_latents=forwarded.nn_out["local_latents"][output_key],
        )
    except KeyError as error:
        raise DirectionalModuleError(
            f"route output is missing {error.args[0]!r}"
        ) from error

def _detach_routed_batch(routed: RoutedBatch) -> RoutedBatch:
    return RoutedBatch(
        prediction=RoutePrediction(
            bb_ca=routed.prediction.bb_ca.detach(),
            local_latents=routed.prediction.local_latents.detach(),
        ),
        per_residue_loss=routed.per_residue_loss.detach(),
        telemetry=tuple(
            BridgeTelemetry(
                probabilities=record.probabilities.detach(),
                applied_z_to_x=record.applied_z_to_x.detach(),
                applied_x_to_z=record.applied_x_to_z.detach(),
            )
            for record in routed.telemetry
        ),
    )

def _valid_generated_mask(batch: Mapping[str, Any]) -> Tensor:
    try:
        generated_mask = batch["generated_mask"]
        mask = batch["mask"]
    except KeyError as error:
        raise DirectionalModuleError(
            f"corrupted batch is missing {error.args[0]!r}"
        ) from error
    if not isinstance(generated_mask, Tensor) or not isinstance(mask, Tensor):
        raise DirectionalModuleError("generated_mask and mask must be tensors")
    if generated_mask.dtype != torch.bool or mask.dtype != torch.bool:
        raise DirectionalModuleError("generated_mask and mask must be boolean")
    if generated_mask.shape != mask.shape or generated_mask.ndim != 2:
        raise DirectionalModuleError("generated_mask and mask must share shape [B, N]")
    return generated_mask & mask

def _unwrap_model_batch(batch: FamilyBatch | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(batch, FamilyBatch):
        _, model_batch = unwrap_family_batch(batch)
        return model_batch
    if not isinstance(batch, Mapping) or set(batch) != {
        "family",
        "example_ids",
        "batch",
    }:
        raise DirectionalModuleError(
            "batch must be FamilyBatch or an exact family/example_ids/batch wrapper"
        )
    model_payload = batch["batch"]
    if not isinstance(model_payload, Mapping):
        raise DirectionalModuleError("family batch payload must be a mapping")
    family = batch["family"]
    example_ids = batch["example_ids"]
    if not isinstance(family, str) or isinstance(example_ids, str):
        raise DirectionalModuleError("family and example_ids metadata are malformed")
    wrapped = FamilyBatch(
        family=family,
        example_ids=tuple(example_ids),
        model_batch=dict(model_payload),
    )
    _, model_batch = unwrap_family_batch(wrapped)
    return model_batch

def _parameter_group(name: str) -> str | None:

    parts = tuple(name.split("."))
    if parts[:3] == ("base", "nn", "bridges") and len(parts) > 3:
        return "bridge"
    if parts[:3] == ("base", "nn", "router") and len(parts) > 3:
        return "router"
    if parts[:3] == ("base", "nn", "base") and parts[-1] in {
        "lora_A",
        "lora_B",
    }:
        return "lora"
    return None
