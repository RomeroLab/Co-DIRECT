
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

CATEGORIES = ("P1", "P2", "P3")

PERMITTED_UNDER_ENTANGLEMENT = (
    "directional information value",
    "directional leverage",
    "cross-modal sensitivity",
)

VERDICTS = ("PROCEED_TO_PROBE", "VALUE_ONLY", "STOP_ROUTER", "INCONCLUSIVE")
P1_ONLY_VERDICTS = ("PROCEED_TO_PROBE",)

_PERTURBATION = 1e-2

class ClaimNotIdentifiable(RuntimeError):
    pass

@dataclass(frozen=True)
class InterventionSite:

    module: str
    direction: str
    affects_recipient: str
    spares_recipient: str
    depth: int

@dataclass
class DataflowTrace:

    order: list[str] = field(default_factory=list)
    responds_to_sender: dict[str, set[str]] = field(default_factory=dict)
    reaches_output: dict[str, set[str]] = field(default_factory=dict)
    senders: tuple[str, ...] = ()
    recipients: tuple[str, ...] = ()

    def _depth(self, module: str) -> int:
        return self.order.index(module)

    @property
    def first_module_for(self) -> dict[str, str | None]:

        out: dict[str, str | None] = {}
        for sender in self.senders:
            hits = [m for m in self.order if sender in self.responds_to_sender.get(m, ())]
            out[sender] = hits[0] if hits else None
        return out

    @property
    def mixing_module(self) -> str | None:

        for module in self.order:
            if set(self.senders) <= self.responds_to_sender.get(module, set()):
                return module
        return None

    @property
    def mixing_depth(self) -> int:
        module = self.mixing_module
        return self._depth(module) if module is not None else -1

    @property
    def divergence_depth(self) -> int:

        shared = [
            m for m in self.order
            if set(self.recipients) <= self.reaches_output.get(m, set())
        ]
        return self._depth(shared[-1]) if shared else -1

    @property
    def recipients_share_a_stream(self) -> bool:
        return self.divergence_depth >= 0

    @property
    def output_heads_are_distinct(self) -> bool:

        return any(
            len(self.reaches_output.get(m, set())) == 1 for m in self.order
        )

def _leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules()
            if n and not list(m.children())]

def _selected_modules(
    model: nn.Module, module_names: Sequence[str] | None
) -> list[tuple[str, nn.Module]]:

    if module_names is None:
        return _leaf_modules(model)
    lookup = dict(model.named_modules())
    missing = [n for n in module_names if n not in lookup]
    if missing:
        raise KeyError(f"no such module(s) on the model: {missing}")
    return [(n, lookup[n]) for n in module_names]

def _flatten(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, Mapping):
        return [t for v in value.values() for t in _flatten(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _flatten(v)]
    return []

def _same(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> bool:
    if len(a) != len(b):
        return False
    return all(
        x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)
    )

def _capture(
    model: nn.Module,
    run: Callable[[], Any],
    selected: Sequence[tuple[str, nn.Module]],
) -> tuple[list[str], dict, Any]:

    order: list[str] = []
    captured: dict[str, list[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            if name not in captured:
                order.append(name)
            captured[name] = [t.detach().clone() for t in _flatten(out)]
        return hook

    for name, module in selected:
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        result = run()
    finally:
        for handle in handles:
            handle.remove()
    return order, captured, result

def perturb_value(value: Any, seed: int) -> Any:

    if torch.is_tensor(value):
        if not value.is_floating_point():
            return value
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(value.shape, generator=gen).to(value.device, value.dtype)
        return value + _PERTURBATION * noise
    if isinstance(value, Mapping):
        return type(value)(
            (k, perturb_value(v, seed + i)) for i, (k, v) in enumerate(value.items())
        )
    if isinstance(value, tuple):
        return tuple(perturb_value(v, seed + i) for i, v in enumerate(value))
    if isinstance(value, list):
        return [perturb_value(v, seed + i) for i, v in enumerate(value)]
    return value

def _perturb_output(model: nn.Module, target: str, seed: int):

    module = dict(model.named_modules())[target]

    def hook(_mod, _inp, out):
        return perturb_value(out, seed)

    return module.register_forward_hook(hook)

@torch.no_grad()
def trace_modality_dataflow(
    model: nn.Module,
    *,
    inputs: Mapping[str, torch.Tensor],
    sender_keys: Mapping[str, str],
    output_keys: Mapping[str, str],
    module_names: Sequence[str] | None = None,
) -> DataflowTrace:

    model.eval()

    def run(overrides: Mapping[str, torch.Tensor] | None = None):
        payload = dict(inputs)
        if overrides:
            payload.update(overrides)
        return model(**payload)

    selected = _selected_modules(model, module_names)
    order, baseline, base_out = _capture(model, run, selected)
    base_flat = {k: _flatten(base_out[k]) for k in output_keys.values()}

    senders = tuple(sender_keys)
    recipients = tuple(output_keys)

    responds: dict[str, set[str]] = {name: set() for name in order}
    for sender, key in sender_keys.items():
        gen = torch.Generator(device="cpu").manual_seed(hash(sender) % (2**31))
        nudged = inputs[key] + _PERTURBATION * torch.randn(
            inputs[key].shape, generator=gen
        ).to(inputs[key].device, inputs[key].dtype)
        _, captured, _ = _capture(model, lambda: run({key: nudged}), selected)
        for name in order:
            if not _same(captured.get(name, []), baseline.get(name, [])):
                responds[name].add(sender)

    reaches: dict[str, set[str]] = {name: set() for name in order}
    for depth, name in enumerate(order):
        handle = _perturb_output(model, name, seed=1000 + depth)
        try:
            out = run()
        finally:
            handle.remove()
        for recipient, key in output_keys.items():
            if not _same(_flatten(out[key]), base_flat[key]):
                reaches[name].add(recipient)

    return DataflowTrace(
        order=order, responds_to_sender=responds, reaches_output=reaches,
        senders=senders, recipients=recipients,
    )

def direction_specific_sites(trace: DataflowTrace) -> list[InterventionSite]:

    sites: list[InterventionSite] = []
    for depth, module in enumerate(trace.order):
        senders = trace.responds_to_sender.get(module, set())
        recipients = trace.reaches_output.get(module, set())
        if len(senders) != 1 or len(recipients) != 1:
            continue
        sender = next(iter(senders))
        recipient = next(iter(recipients))
        if sender == recipient:
            continue
        independent_source = any(
            other != module
            and recipient in trace.reaches_output.get(other, set())
            and sender not in trace.responds_to_sender.get(other, set())
            for other in trace.order
        )
        if not independent_source:
            continue
        spared = [r for r in trace.recipients if r != recipient]
        sites.append(
            InterventionSite(
                module=module, direction=f"{sender}->{recipient}",
                affects_recipient=recipient,
                spares_recipient=spared[0] if spared else "",
                depth=depth,
            )
        )
    return sites

def classify(trace: DataflowTrace) -> str:

    if direction_specific_sites(trace):
        return "P1"
    if trace.mixing_module is not None and trace.divergence_depth > trace.mixing_depth:
        return "P3"
    return "P2"

def _check_category(category: str) -> None:
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}, got {category!r}")

def path_intervention_valid(category: str) -> bool:

    _check_category(category)
    return category == "P1"

def assert_claim_allowed(category: str, claim: str) -> None:

    _check_category(category)
    if category == "P1":
        return
    lowered = claim.lower()
    if any(permitted in lowered for permitted in PERMITTED_UNDER_ENTANGLEMENT):
        return
    if any(word in lowered for word in ("lead", "causal", "route", "routing")):
        raise ClaimNotIdentifiable(
            f"the architecture is {category}, so {claim!r} is not identifiable. "
            f"Permitted instead: {', '.join(PERMITTED_UNDER_ENTANGLEMENT)}."
        )

def assert_verdict_allowed(category: str, verdict: str) -> None:

    _check_category(category)
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    if verdict in P1_ONLY_VERDICTS and category != "P1":
        raise ClaimNotIdentifiable(
            f"{verdict} requires P1 path-level evidence; the architecture "
            f"measured as {category}. Permitted instead: "
            f"{', '.join(v for v in VERDICTS if v not in P1_ONLY_VERDICTS)}."
        )

@torch.no_grad()
def forward_is_deterministic(
    model: nn.Module,
    *,
    inputs: Mapping[str, torch.Tensor],
    output_keys: Mapping[str, str],
) -> bool:

    model.eval()
    first = model(**inputs)
    second = model(**inputs)
    return all(
        _same(_flatten(first[key]), _flatten(second[key]))
        for key in output_keys.values()
    )
