
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

class ResumeError(RuntimeError):
    pass

STAGE_EDGES: frozenset[tuple[str, str]] = frozenset(
    {

        ("null_warmup", "router_warmup"),
        ("router_warmup", "joint_lora"),

        ("null_warmup", "joint_fixed"),
    }
)

SAME_STAGE = "same_stage"
TRANSITION = "transition"
EVALUATION = "evaluation"

@dataclass(frozen=True, slots=True)
class ResumePlan:

    kind: str
    previous_stage: str
    current_stage: str
    restore_optimizer: bool
    restore_progress: bool
    preserve_groups: tuple[str, ...] = ()
    reinitialize_groups: tuple[str, ...] = ()

    def as_provenance(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "previous_stage": self.previous_stage,
            "current_stage": self.current_stage,
            "restore_optimizer": self.restore_optimizer,
            "restore_progress": self.restore_progress,
            "preserve_groups": list(self.preserve_groups),
            "reinitialize_groups": list(self.reinitialize_groups),
        }

@dataclass(frozen=True, slots=True)
class ResumeReport:

    plan: ResumePlan
    source: str
    restored_optimizer: bool
    preserved_groups: tuple[str, ...]
    reinitialized_groups: tuple[str, ...]
    global_step: int
    stage_step: int
    restored_rng: bool
    data_progress: dict[str, Any] = field(default_factory=dict)
    parameter_count: int = 0

    def as_provenance(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_provenance(),
            "source": self.source,
            "restored_optimizer": self.restored_optimizer,
            "preserved_groups": list(self.preserved_groups),
            "reinitialized_groups": list(self.reinitialized_groups),
            "global_step": self.global_step,
            "stage_step": self.stage_step,
            "restored_rng": self.restored_rng,
            "data_progress": dict(self.data_progress),
            "parameter_count": self.parameter_count,
        }

def plan_resume(
    *,
    previous_stage: str,
    current_stage: str,
    preserve_router_optimizer_state: bool = False,
) -> ResumePlan:

    previous_stage = str(previous_stage)
    current_stage = str(current_stage)

    if previous_stage == current_stage:
        return ResumePlan(
            kind=SAME_STAGE,
            previous_stage=previous_stage,
            current_stage=current_stage,
            restore_optimizer=True,
            restore_progress=True,
        )

    if (previous_stage, current_stage) not in STAGE_EDGES:
        raise ResumeError(
            f"no declared resume edge {previous_stage!r} -> {current_stage!r}; "
            f"the campaign declares {sorted(STAGE_EDGES)} and same-stage resumes"
        )

    preserve: tuple[str, ...] = ()
    if preserve_router_optimizer_state:
        if (previous_stage, current_stage) != ("router_warmup", "joint_lora"):
            raise ResumeError(
                "preserve_router_optimizer_state applies only to the "
                "router_warmup -> joint_lora edge, where the router group "
                f"exists on both sides; got {previous_stage!r} -> {current_stage!r}"
            )
        preserve = ("router",)

    return ResumePlan(
        kind=TRANSITION,
        previous_stage=previous_stage,
        current_stage=current_stage,

        restore_optimizer=False,
        restore_progress=False,
        preserve_groups=preserve,

        reinitialize_groups=(
            ("router",)
            if (previous_stage, current_stage) == ("null_warmup", "router_warmup")
            else ()
        ),
    )

def load_resume_checkpoint(
    path,
    *,
    module,
    optimizer,
    plan: ResumePlan,
    map_location: str = "cpu",
) -> ResumeReport:

    import torch

    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)

    state_dict = payload.get("state_dict")
    if state_dict is None:
        raise ResumeError(
            f"{path} carries no state_dict; a resume that cannot restore weights "
            f"would train a fresh splice and report success"
        )

    reinitialized: tuple[str, ...] = ()
    if plan.reinitialize_groups:

        current = module.state_dict()
        held = {
            name
            for name in state_dict
            if any(marker in name for marker in plan.reinitialize_groups)
        }
        if not held:
            raise ResumeError(
                f"the plan reinitializes {list(plan.reinitialize_groups)}, but "
                f"{path} carries no parameter matching any of them; the policy "
                f"would silently do nothing"
            )
        state_dict = {
            name: (current[name] if name in held else tensor)
            for name, tensor in state_dict.items()
        }
        reinitialized = tuple(plan.reinitialize_groups)

    module.load_state_dict(state_dict, strict=True)
    restored_parameters = sum(
        v.numel() for v in state_dict.values() if hasattr(v, "numel")
    )

    global_step = int(payload.get("global_step", payload.get("step", 0)) or 0)
    stage_step = int(payload.get("stage_step", global_step) or 0)
    restored_optimizer = False
    preserved: tuple[str, ...] = ()
    restored_rng = False
    data_progress: dict[str, Any] = {}

    if plan.restore_optimizer:
        saved = payload.get("optimizer")
        if saved is None:
            raise ResumeError(
                f"{path} carries no optimizer state, but a same-stage resume of "
                f"{plan.current_stage} must restore its moments; continuing with "
                f"a cold optimizer is a different experiment"
            )
        if optimizer is None:
            raise ResumeError("same-stage resume needs an optimizer to restore into")
        optimizer.load_state_dict(saved)
        restored_optimizer = True
    elif plan.preserve_groups and optimizer is not None:
        preserved = _preserve_named_groups(
            optimizer, payload.get("optimizer"), plan.preserve_groups, source=path
        )

    if plan.restore_progress:
        rng = payload.get("rng")
        if rng is not None:
            torch.set_rng_state(_as_byte_tensor(rng))
            restored_rng = True
        data_progress = dict(payload.get("data_progress") or {})
    else:

        stage_step = 0

    return ResumeReport(
        plan=plan,
        source=str(path),
        restored_optimizer=restored_optimizer,
        preserved_groups=preserved,
        reinitialized_groups=reinitialized,
        global_step=global_step,
        stage_step=stage_step,
        restored_rng=restored_rng,
        data_progress=data_progress,
        parameter_count=restored_parameters,
    )

def _as_byte_tensor(rng):
    import torch

    if isinstance(rng, torch.Tensor):
        return rng.cpu().to(torch.uint8)
    return torch.tensor(rng, dtype=torch.uint8)

def _group_index_map(param_groups) -> dict[str, list[int]]:

    mapping: dict[str, list[int]] = {}
    index = 0
    for group in param_groups:
        name = group.get("name")
        count = len(group["params"])
        if name is not None:
            mapping.setdefault(str(name), []).extend(range(index, index + count))
        index += count
    return mapping

def _preserve_named_groups(optimizer, saved, names, *, source) -> tuple[str, ...]:

    if saved is None:
        raise ResumeError(
            f"{source} carries no optimizer state, so the reviewed "
            f"{list(names)} preservation cannot be performed as declared"
        )

    saved_state = saved.get("state") or {}
    saved_map = _group_index_map(saved.get("param_groups") or [])
    target_map = _group_index_map(optimizer.param_groups)
    target_params = [p for group in optimizer.param_groups for p in group["params"]]

    mapped: dict[int, Any] = {}
    preserved: list[str] = []
    for name in names:
        source_indices = saved_map.get(name)
        target_indices = target_map.get(name)
        if not source_indices or not target_indices:
            raise ResumeError(
                f"group {name!r} is not present on both sides of the resume "
                f"({'absent' if not source_indices else 'present'} in {source}, "
                f"{'absent' if not target_indices else 'present'} in the new "
                f"optimizer); refusing a partial transfer"
            )
        if len(source_indices) != len(target_indices):
            raise ResumeError(
                f"group {name!r} has {len(source_indices)} parameter(s) in "
                f"{source} and {len(target_indices)} in the new optimizer"
            )
        for source_index, target_index in zip(source_indices, target_indices):
            entry = saved_state.get(source_index)
            if entry is None:
                entry = saved_state.get(str(source_index))
            if entry is None:
                continue
            expected = target_params[target_index].shape
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = entry.get(key)
                if tensor is None:
                    continue
                if tuple(tensor.shape) != tuple(expected):
                    raise ResumeError(
                        f"group {name!r} parameter {target_index}: saved {key} has "
                        f"shape {tuple(tensor.shape)} but the new optimizer's "
                        f"parameter has shape {tuple(expected)}; refusing to "
                        f"preserve mismatched optimizer state"
                    )
            mapped[target_index] = entry
        preserved.append(name)

    target = optimizer.state_dict()
    target["state"] = mapped
    optimizer.load_state_dict(target)
    return tuple(preserved)

def evaluation_plan(stage: str) -> ResumePlan:

    stage = str(stage)
    return ResumePlan(
        kind=EVALUATION,
        previous_stage=stage,
        current_stage=stage,
        restore_optimizer=False,
        restore_progress=False,
        preserve_groups=(),
        reinitialize_groups=(),
    )
