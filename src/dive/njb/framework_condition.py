
from __future__ import annotations

import torch

class FrameworkConditionError(RuntimeError):
    pass

def echo_free_rows(
    batch: dict,
    supplied_ca: torch.Tensor,
    given: torch.Tensor,
) -> torch.Tensor:

    if supplied_ca.dim() != 3 or supplied_ca.shape[-1] != 3:
        raise FrameworkConditionError(
            f"supplied_ca has shape {tuple(supplied_ca.shape)}; expected [b, n, 3]")
    if given.shape != supplied_ca.shape[:2]:
        raise FrameworkConditionError(
            f"shape disagreement: given {tuple(given.shape)} against supplied "
            f"{tuple(supplied_ca.shape[:2])}")
    if not bool(given.any()):
        raise FrameworkConditionError(
            "no fixed residue; this condition would supply nothing and the "
            "arm would be mislabelled as conditioned")

    state = batch.get("x_t")
    current = state.get("bb_ca") if isinstance(state, dict) else None
    if current is None:
        raise FrameworkConditionError(
            "the batch carries no x_t bb_ca, so there is no current estimate to "
            "echo; filling the free rows with zeros here is the exact failure "
            "this function exists to prevent")
    if current.shape != supplied_ca.shape:
        raise FrameworkConditionError(
            f"shape disagreement: x_t bb_ca {tuple(current.shape)} against "
            f"supplied {tuple(supplied_ca.shape)}")

    keep = given[..., None].to(torch.bool)
    return torch.where(keep, supplied_ca.to(current.dtype), current)

class FrameworkConditionWrapper(torch.nn.Module):

    def __init__(self, inner: torch.nn.Module, supplied_ca: torch.Tensor,
                 given: torch.Tensor, *, supply_identity: bool = True,
                 identities: torch.Tensor | None = None,
                 retime_free_rows: bool = False,
                 mix_free_rows: bool = False, mix_seed: int = 20260917,
                 anneal_free_rows: bool = False,
                 anneal_free_mask: torch.Tensor | None = None):
        super().__init__()
        self.inner = inner
        self.register_buffer("supplied_ca", supplied_ca, persistent=False)
        self.register_buffer("given", given, persistent=False)
        self.supply_identity = bool(supply_identity)

        self.condition_state_override: torch.Tensor | None = None
        if identities is not None:
            self.register_buffer("identities", identities, persistent=False)
        else:
            self.identities = None
        self.calls = 0
        self.rewrites = 0

        self.retime_free_rows = bool(retime_free_rows) or bool(mix_free_rows)
        self._last_v: torch.Tensor | None = None
        self.retimed = 0

        self.anneal_free_rows = bool(anneal_free_rows)
        self.register_buffer("anneal_free_mask", anneal_free_mask, persistent=False)
        self.annealed = 0
        self.mix_free_rows = bool(mix_free_rows)
        self.mix_seed = int(mix_seed)
        self._mix_gen: torch.Generator | None = None
        self.mixed = 0

    def _mix_noise(self, like: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        if self._mix_gen is None or self._mix_gen.device != like.device:
            self._mix_gen = torch.Generator(device=like.device)
            self._mix_gen.manual_seed(self.mix_seed)
        eps = torch.randn(like.shape, generator=self._mix_gen,
                          device=like.device, dtype=like.dtype)
        scale = (1.0 - t.to(like.dtype)).clamp_min(0.0)
        while scale.dim() < like.dim():
            scale = scale[..., None]
        return scale * eps

    def forward(self, batch: dict):
        self.calls += 1
        state = batch.get("x_t")
        if isinstance(state, dict) and state.get("bb_ca") is not None:
            batch = dict(batch)
            source = batch
            if self.condition_state_override is not None:
                override = self.condition_state_override
                if override.shape != self.supplied_ca.shape:
                    raise FrameworkConditionError(
                        f"condition_state_override has shape {tuple(override.shape)}; "
                        f"expected {tuple(self.supplied_ca.shape)}. Broadcasting it "
                        "would condition the trunk on coordinates for other residues")
                source = {**batch, "x_t": {**state, "bb_ca": override}}
            elif self.retime_free_rows and self._last_v is not None:
                current = state["bb_ca"]
                if self._last_v.shape == current.shape:
                    from dive.stackelberg.joint_update import anticipated_clean_state

                    t = batch.get("t", {}).get("bb_ca")
                    if t is not None:
                        from dive.stackelberg.joint_update import PRIOR_SCALE_NM

                        hat1 = anticipated_clean_state(
                            current, self._last_v.to(current.dtype),
                            t.to(current.dtype), max_step_nm=PRIOR_SCALE_NM)
                        self.retimed += 1
                        if self.mix_free_rows:
                            hat1 = hat1 + self._mix_noise(hat1, t)
                            self.mixed += 1
                        source = {**batch, "x_t": {**state, "bb_ca": hat1}}
            condition = echo_free_rows(source, self.supplied_ca, self.given)
            if self.anneal_free_rows:
                free = self.anneal_free_mask
                current = state["bb_ca"]
                if free is None or free.shape != current.shape[:2]:
                    raise FrameworkConditionError(
                        f"anneal_free_mask {None if free is None else tuple(free.shape)} "
                        f"does not match the residue axis {tuple(current.shape[:2])}; "
                        "annealing the wrong rows would release the framework")
                t = batch.get("t", {}).get("bb_ca")
                if t is not None:
                    w = t.to(current.dtype).clamp(0.0, 1.0)
                    while w.dim() < current.dim():
                        w = w[..., None]
                    blended = w * current + (1.0 - w) * condition
                    condition = torch.where(free[..., None].to(torch.bool),
                                            blended, condition)
                    self.annealed += 1
            batch["ca_coors_nm"] = condition
            batch["use_ca_coors_nm_feature"] = True
            if self.supply_identity and self.identities is not None:
                batch["residue_type"] = self.identities
                batch["use_residue_type_feature"] = True

                batch["mask_dict"] = dict(batch.get("mask_dict") or {})
                batch["mask_dict"]["residue_type"] = self.given.to(
                    self.identities.dtype)
            else:

                batch["use_residue_type_feature"] = False
            self.rewrites += 1
        out = self.inner(batch)
        if self.retime_free_rows:
            try:
                v = out["bb_ca"]["v"]
            except (TypeError, KeyError, IndexError):
                v = None
            if v is not None and torch.is_tensor(v):
                self._last_v = v.detach()
        return out
