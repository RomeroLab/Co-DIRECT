
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dive.routing.splits import SplitError, representative_targets, target_groups
from dive.signed_value.contract import (
    FROZEN_TARGETS,
    MEDIAN_SLOT_CANDIDATES,
    MEDIAN_SLOT_TARGET,
)

_TARGET_SEGMENT = re.compile(
    r"^(?P<chain>[A-Za-z]+)(?P<start>-?\d+)-(?P<end>-?\d+)$"
)

class ContractError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class FeasibilityTarget:

    target: str
    target_residues: int
    binder_upper: int
    complex_length_ceiling: int
    target_path: Path
    target_chains: str

def _parse_target_input(target_input: str) -> tuple[tuple[str, int, int], ...]:
    if not isinstance(target_input, str) or not target_input.strip():
        raise ContractError("target_input is empty")

    parsed: list[tuple[str, int, int]] = []
    for raw_segment in target_input.split(","):
        segment = raw_segment.strip()
        match = _TARGET_SEGMENT.fullmatch(segment)
        if match is None:
            raise ContractError(f"malformed target_input segment: {segment!r}")
        start = int(match.group("start"))
        end = int(match.group("end"))
        if start > end:
            raise ContractError(f"reversed target_input range: {segment!r}")
        parsed.append((match.group("chain"), start, end))
    return tuple(parsed)

def count_selected_target_residues(target_input: str) -> int:

    return sum(end - start + 1 for _, start, end in _parse_target_input(target_input))

def _target_chains(target_input: str) -> str:
    chains: list[str] = []
    for chain, _, _ in _parse_target_input(target_input):
        if chain not in chains:
            chains.append(chain)
    return ",".join(chains)

def _binder_upper(entry: Mapping[str, Any], *, target: str) -> int:
    binder_length = entry.get("binder_length")
    if (
        not isinstance(binder_length, Sequence)
        or isinstance(binder_length, (str, bytes))
        or len(binder_length) != 2
    ):
        raise ContractError(f"{target} has invalid binder_length: {binder_length!r}")
    lower, upper = binder_length
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, int)
        or not isinstance(upper, int)
        or lower < 0
        or lower > upper
    ):
        raise ContractError(f"{target} has invalid binder_length: {binder_length!r}")
    return upper

def _resolve_target_path(entry: Mapping[str, Any], *, target: str, upstream_root: Path) -> Path:
    raw_path = entry.get("target_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ContractError(f"{target} has no target_path")
    configured_path = Path(raw_path)
    if configured_path.is_absolute():
        raise ContractError(f"{target} target_path must be relative to the upstream root")

    root = upstream_root.expanduser().resolve()
    resolved = (root / configured_path).resolve()
    if not resolved.is_relative_to(root):
        raise ContractError(f"{target} target_path escapes upstream root: {raw_path!r}")
    if not resolved.is_file():
        raise ContractError(f"{target} target_path is missing: {resolved}")
    return resolved

def _feasibility_target(
    target: str, entry: Mapping[str, Any], *, upstream_root: Path
) -> FeasibilityTarget:
    target_input = entry.get("target_input")
    if not isinstance(target_input, str):
        raise ContractError(f"{target} has invalid target_input: {target_input!r}")
    target_residues = count_selected_target_residues(target_input)
    binder_upper = _binder_upper(entry, target=target)
    return FeasibilityTarget(
        target=target,
        target_residues=target_residues,
        binder_upper=binder_upper,
        complex_length_ceiling=target_residues + binder_upper,
        target_path=_resolve_target_path(entry, target=target, upstream_root=upstream_root),
        target_chains=_target_chains(target_input),
    )

MEDIAN_SLOT_INDEX = 7

def choose_extremes(
    items: Sequence[FeasibilityTarget],
) -> tuple[FeasibilityTarget, FeasibilityTarget, FeasibilityTarget]:

    ordered = sorted(items, key=lambda item: (item.complex_length_ceiling, item.target))
    if len(ordered) != 16:
        raise ContractError(f"expected 16 development representatives, observed {len(ordered)}")
    by_name = {item.target: item for item in ordered}
    if MEDIAN_SLOT_TARGET not in MEDIAN_SLOT_CANDIDATES:
        raise ContractError(
            f"median slot {MEDIAN_SLOT_TARGET!r} is outside the declared scan"
        )
    missing = [name for name in FROZEN_TARGETS if name not in by_name]
    if missing:
        raise ContractError(f"frozen inventory names non-representatives: {missing}")
    small, median, large = (by_name[name] for name in FROZEN_TARGETS)
    if (small.target, large.target) != (ordered[0].target, ordered[-1].target):
        raise ContractError(
            "frozen length extremes drifted: "
            f"{[small.target, large.target]} against {[ordered[0].target, ordered[-1].target]}"
        )
    if not (
        small.complex_length_ceiling
        < median.complex_length_ceiling
        < large.complex_length_ceiling
    ):
        raise ContractError(
            f"median slot {median.target!r} does not sit between the extremes"
        )
    return (small, median, large)

def _load_mapping(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"cannot read {description}: {path}") from error
    if not isinstance(loaded, Mapping):
        raise ContractError(f"{description} is not a mapping: {path}")
    return loaded

def select_feasibility_targets(
    split_path: Path, target_dictionary: Path, upstream_root: Path
) -> tuple[FeasibilityTarget, FeasibilityTarget, FeasibilityTarget]:

    split = _load_mapping(split_path, description="routing split")
    dev_targets = split.get("dev_targets")
    if not isinstance(dev_targets, list) or not dev_targets or not all(
        isinstance(target, str) for target in dev_targets
    ):
        raise ContractError(f"routing split has no valid dev_targets list: {split_path}")

    dictionary = _load_mapping(target_dictionary, description="target dictionary")
    targets = dictionary.get("target_dict_cfg")
    if not isinstance(targets, Mapping):
        raise ContractError(f"target dictionary has no target_dict_cfg map: {target_dictionary}")
    if not all(isinstance(name, str) and isinstance(entry, Mapping) for name, entry in targets.items()):
        raise ContractError(f"target dictionary has invalid target entries: {target_dictionary}")

    try:
        representatives = representative_targets(
            dev_targets, target_groups(targets), 16
        )
    except SplitError as error:
        raise ContractError(f"cannot select development representatives: {error}") from error

    inventory = tuple(
        _feasibility_target(target, targets[target], upstream_root=upstream_root)
        for target in representatives
    )
    return choose_extremes(inventory)

def select_from_files(
    split_path: Path, target_dictionary: Path, upstream_root: Path | None = None
) -> tuple[FeasibilityTarget, FeasibilityTarget, FeasibilityTarget]:

    root = target_dictionary.parents[2] if upstream_root is None else upstream_root
    return select_feasibility_targets(split_path, target_dictionary, root)
