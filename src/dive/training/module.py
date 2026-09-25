
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from dive.data.mixture import unwrap_family_batch
from dive.leadership.conditions import (
    DIRECTIONAL_CONDITION_NAMES,
    PRESENCE_BY_CONDITION,
    PRESENCE_KEY,
)
from dive.leadership.fields import fixed_gates, route_fields
from dive.leadership.router import DirectionalGates, LeadershipRouter
from dive.training.stages import TrainingStage, contract_for

MAX_JOINT_RELATIVE_LOSS = 1.02

MIN_CONSECUTIVE_FINITE_STEPS = 1000

MAX_LEAKAGE_PROBE_GAIN = 0.05

BACKBONE_PATHS: tuple[str, ...] = ("x_t.bb_ca", "x_sc.bb_ca", "x_recycle.bb_ca")
LATENT_PATHS: tuple[str, ...] = (
    "x_t.local_latents",
    "x_sc.local_latents",
    "x_recycle.local_latents",
)

_LORA_MARKER = "lora_"
_ROUTER_MARKER = "router"
_PRESENCE_MARKER = "presence"

class ModuleError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class HealthVerdict:

    passed: bool
    reasons: tuple[str, ...]
    record: Mapping[str, Any]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "HealthVerdict":
        reasons = []

        joint = float(record["joint_relative_val_loss"])
        if joint > MAX_JOINT_RELATIVE_LOSS:
            reasons.append(
                f"joint relative validation loss {joint:.4f} exceeds "
                f"{MAX_JOINT_RELATIVE_LOSS}; the adapted model no longer reproduces "
                f"the base checkpoint on the joint condition"
            )

        steps = int(record["consecutive_finite_steps"])
        if steps < MIN_CONSECUTIVE_FINITE_STEPS:
            reasons.append(
                f"only {steps} consecutive finite steps, below "
                f"{MIN_CONSECUTIVE_FINITE_STEPS}"
            )

        gains = dict(record.get("leakage_probe_gains", {}))
        leaking = {k: v for k, v in gains.items() if float(v) >= MAX_LEAKAGE_PROBE_GAIN}
        if leaking:
            reasons.append(
                f"leakage probe(s) recovered an absent modality beyond the "
                f"{MAX_LEAKAGE_PROBE_GAIN} allowance: {leaking}; the null is not a "
                f"null and every directional reading would be measuring a modality "
                f"that was supposed to be gone"
            )

        return cls(passed=not reasons, reasons=tuple(reasons), record=dict(record))

    def as_provenance(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "thresholds": {
                "max_joint_relative_val_loss": MAX_JOINT_RELATIVE_LOSS,
                "min_consecutive_finite_steps": MIN_CONSECUTIVE_FINITE_STEPS,
                "max_leakage_probe_gain": MAX_LEAKAGE_PROBE_GAIN,
            },
            "record": dict(self.record),
        }

def equal_family_relative_loss(
    current: Mapping[str, float], base: Mapping[str, float]
) -> float:

    missing = sorted(set(current) - set(base))
    if missing:
        raise ModuleError(
            f"no frozen-base baseline for {missing}; scoring without it would "
            f"silently change the selection rule"
        )
    ratios = []
    for family, value in sorted(current.items()):
        denominator = float(base[family])
        if denominator == 0.0:
            raise ModuleError(f"frozen-base loss for {family!r} is zero")
        ratios.append(float(value) / denominator)
    if not ratios:
        raise ModuleError("no family losses to score")
    return sum(ratios) / len(ratios)

class DiveLeadershipModule(nn.Module):

    def __init__(
        self,
        *,
        base: nn.Module,
        router: LeadershipRouter,
        stage: TrainingStage = TrainingStage.NULL_WARMUP,
        call_denoiser: Callable[[nn.Module, dict], Mapping] | None = None,
        flow_loss: Callable[[Tensor, Tensor, Mapping], Tensor] | None = None,
        per_residue_flow_loss: Callable[[Tensor, Tensor, Mapping], Tensor] | None = None,
        allow_surrogate_loss: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.router = router
        self._call_denoiser = call_denoiser or (lambda model, batch: model.nn(batch))
        if flow_loss is None and not allow_surrogate_loss:
            raise ModuleError(
                "no flow loss supplied. The real objective is upstream's "
                "ProductSpaceFlowMatcher, which the trainer owns; the surrogate "
                "here exists only so this module's ownership, boundary, and "
                "routing contracts are testable without one, and it must never "
                "train a reported model"
            )
        self._flow_loss = flow_loss
        self._per_residue_flow_loss = per_residue_flow_loss
        self.stage = stage
        self.set_stage(stage)

    @property
    def uses_surrogate_loss(self) -> bool:

        return self._flow_loss is None

    def set_stage(self, stage: TrainingStage) -> None:

        contract = contract_for(stage)
        self.stage = TrainingStage(stage)

        for name, parameter in self.named_parameters():
            if _LORA_MARKER in name:
                parameter.requires_grad_(contract.train_lora)
            elif _ROUTER_MARKER in name:
                parameter.requires_grad_(contract.train_router)
            elif _PRESENCE_MARKER in name:
                parameter.requires_grad_(contract.train_presence)
            else:
                parameter.requires_grad_(False)

        autoencoder = getattr(self.base, "autoencoder", None)
        if autoencoder is not None:
            for parameter in autoencoder.parameters():
                parameter.requires_grad_(contract.train_autoencoder)

    def trainable_parameter_groups(self) -> list[dict[str, Any]]:

        contract = contract_for(self.stage)
        groups: dict[str, list[nn.Parameter]] = {"lora": [], "router": [], "presence": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if _LORA_MARKER in name:
                groups["lora"].append(parameter)
            elif _ROUTER_MARKER in name:
                groups["router"].append(parameter)
            elif _PRESENCE_MARKER in name:
                groups["presence"].append(parameter)
            else:
                raise ModuleError(
                    f"parameter {name!r} is trainable but belongs to no group; "
                    f"only lora, router, and presence may train"
                )

        declared = {
            "lora": contract.train_lora,
            "router": contract.train_router,
            "presence": contract.train_presence,
        }
        missing = sorted(
            name for name, wanted in declared.items() if wanted and not groups[name]
        )
        if missing:
            raise ModuleError(
                f"stage {self.stage} declares {missing} trainable, but the model "
                f"carries no such parameter; the stage would train something "
                f"narrower than its contract without saying so"
            )
        return [
            {"name": name, "params": params} for name, params in groups.items() if params
        ]

    def forward(self, batch, batch_index: int = 0) -> Tensor:

        return self.training_step(batch, batch_index)

    def training_step(self, batch, batch_index: int) -> Tensor:

        _, model_batch = unwrap_family_batch(batch)
        packed = self._pack(model_batch)

        outputs = self._call_denoiser(self.base, packed.batch)
        gates = self._gates(outputs, packed, model_batch)

        v_x, v_z = self._route(outputs, packed, gates)
        if self._flow_loss is not None:
            return self._flow_loss(v_x, v_z, model_batch)
        return _surrogate_loss(v_x, v_z)

    def loss_at_corner(self, model_batch: dict, corner: str) -> Tensor:

        packed = self._pack(model_batch)
        outputs = self._call_denoiser(self.base, packed.batch)
        gates = fixed_gates(
            corner,
            packed.batch_size,
            int(model_batch["mask"].shape[1]),
            model_batch["mask"].device,
        )
        v_x, v_z = self._route(outputs, packed, gates)
        if self._flow_loss is not None:
            return self._flow_loss(v_x, v_z, model_batch)
        return _surrogate_loss(v_x, v_z)

    def loss_at_gates(
        self,
        model_batch: dict,
        *,
        z_to_x: float,
        x_to_z: float,
    ) -> Tensor:

        packed = self._pack(model_batch)
        outputs = self._call_denoiser(self.base, packed.batch)
        shape = (packed.batch_size, int(model_batch["mask"].shape[1]))
        gates = DirectionalGates(
            z_to_x=torch.full(shape, float(z_to_x), device=model_batch["mask"].device),
            x_to_z=torch.full(shape, float(x_to_z), device=model_batch["mask"].device),
        )
        v_x, v_z = self._route(outputs, packed, gates)
        if self._flow_loss is not None:
            return self._flow_loss(v_x, v_z, model_batch)
        return _surrogate_loss(v_x, v_z)

    def velocities_at_gates(
        self,
        model_batch: dict,
        *,
        z_to_x: float,
        x_to_z: float,
    ) -> dict[str, dict[str, Tensor]]:

        packed = self._pack(model_batch)
        outputs = self._call_denoiser(self.base, packed.batch)
        shape = (packed.batch_size, int(model_batch["mask"].shape[1]))
        gates = DirectionalGates(
            z_to_x=torch.full(
                shape, float(z_to_x), device=model_batch["mask"].device
            ),
            x_to_z=torch.full(
                shape, float(x_to_z), device=model_batch["mask"].device
            ),
        )
        v_x, v_z = self._route(outputs, packed, gates)
        return {"bb_ca": {"v": v_x}, "local_latents": {"v": v_z}}

    def per_residue_losses_at_gate_grid(
        self,
        model_batch: dict,
        schedules: Mapping[str, tuple[float, float]],
    ) -> dict[str, Tensor]:

        return self.fixed_scan_batch(model_batch, schedules)["per_residue_losses"]

    def fixed_scan_batch(
        self,
        model_batch: dict,
        schedules: Mapping[str, tuple[float, float]],
    ) -> dict[str, object]:

        if self._per_residue_flow_loss is None:
            raise ModuleError(
                "per-residue flow loss was not supplied; oracle headroom cannot "
                "be computed from the scalar training objective"
            )
        packed = self._pack(model_batch)
        outputs = self._call_denoiser(self.base, packed.batch)
        shape = (packed.batch_size, int(model_batch["mask"].shape[1]))
        losses = {}
        for name, (z_to_x, x_to_z) in schedules.items():
            gates = DirectionalGates(
                z_to_x=torch.full(
                    shape, float(z_to_x), device=model_batch["mask"].device
                ),
                x_to_z=torch.full(
                    shape, float(x_to_z), device=model_batch["mask"].device
                ),
            )
            v_x, v_z = self._route(outputs, packed, gates)
            losses[name] = self._per_residue_flow_loss(v_x, v_z, model_batch)
        take = lambda modality, name: outputs[modality]["v"][packed.slices[name]]
        conditions = {
            name: {
                modality: take(modality, name)
                for modality in ("bb_ca", "local_latents")
            }
            for name in DIRECTIONAL_CONDITION_NAMES
        }
        return {"per_residue_losses": losses, "condition_velocities": conditions}

    def condition_velocities(self, model_batch: dict) -> dict[str, dict[str, Tensor]]:

        packed = self._pack(model_batch)
        outputs = self._call_denoiser(self.base, packed.batch)
        return {
            name: {
                modality: outputs[modality]["v"][packed.slices[name]]
                for modality in ("bb_ca", "local_latents")
            }
            for name in DIRECTIONAL_CONDITION_NAMES
        }

    def _pack(self, model_batch: dict):

        from dive.leadership.conditions import PackedConditions

        batch_size = int(model_batch["mask"].shape[0])
        parts: dict[str, dict] = {}
        for name in DIRECTIONAL_CONDITION_NAMES:
            backbone_present, latent_present = PRESENCE_BY_CONDITION[name]
            sibling = {k: _clone(v) for k, v in model_batch.items()}
            if not backbone_present:
                for path in BACKBONE_PATHS:
                    _zero_path(sibling, path)
            if not latent_present:
                for path in LATENT_PATHS:
                    _zero_path(sibling, path)
            parts[name] = sibling

        packed: dict[str, Any] = {}
        for key in model_batch:
            values = [parts[n][key] for n in DIRECTIONAL_CONDITION_NAMES]
            packed[key] = _pack_value(values, batch_size)

        packed[PRESENCE_KEY] = torch.tensor(
            [
                list(PRESENCE_BY_CONDITION[name])
                for name in DIRECTIONAL_CONDITION_NAMES
                for _ in range(batch_size)
            ],
            dtype=torch.long,
        )
        slices = {
            name: slice(i * batch_size, (i + 1) * batch_size)
            for i, name in enumerate(DIRECTIONAL_CONDITION_NAMES)
        }
        return PackedConditions(batch=packed, slices=slices, batch_size=batch_size)

    def _gates(self, outputs: Mapping, packed, model_batch: dict) -> DirectionalGates:
        contract = contract_for(self.stage)
        batch_size = packed.batch_size
        residues = int(model_batch["mask"].shape[1])

        if contract.fixed_corner is not None:

            return fixed_gates(
                contract.fixed_corner, batch_size, residues, model_batch["mask"].device
            )

        hidden = outputs["hidden"]
        unpack = lambda name: hidden[packed.slices[name]]
        return self.router(
            h_joint=unpack("joint"),
            h_without_latent=unpack("without_latent"),
            h_without_backbone=unpack("without_backbone"),
            t_backbone=model_batch["t"]["bb_ca"],
            t_latent=model_batch["t"]["local_latents"],
            generated_mask=model_batch["generated_mask"],
            fixed_mask=model_batch["fixed_mask"],
            presence=torch.ones(batch_size, 2, dtype=torch.long),
        )

    def _route(self, outputs: Mapping, packed, gates: DirectionalGates):
        take = lambda modality, name: outputs[modality]["v"][packed.slices[name]]
        return route_fields(
            gates=gates,
            v_x_joint=take("bb_ca", "joint"),
            v_x_self=take("bb_ca", "without_latent"),
            v_z_joint=take("local_latents", "joint"),
            v_z_self=take("local_latents", "without_backbone"),
        )

def _zero_path(batch: dict, path: str) -> None:

    parts = path.split(".")
    node = batch
    for key in parts[:-1]:
        if not isinstance(node, Mapping) or key not in node:
            return
        node = node[key]
    leaf = parts[-1]
    if isinstance(node, dict) and leaf in node and isinstance(node[leaf], Tensor):
        node[leaf] = torch.zeros_like(node[leaf])

def _clone(value):
    return value.clone() if isinstance(value, Tensor) else (
        {k: _clone(v) for k, v in value.items()} if isinstance(value, Mapping) else value
    )

def _pack_value(values: list, batch_size: int):
    first = values[0]
    if isinstance(first, Mapping):
        return {k: _pack_value([v[k] for v in values], batch_size) for k in first}
    if isinstance(first, Tensor):
        if first.shape and first.shape[0] == batch_size:
            return torch.cat(values, dim=0)
        return first.clone()
    return first

def _surrogate_loss(v_x: Tensor, v_z: Tensor) -> Tensor:

    return v_x.square().mean() + v_z.square().mean()
