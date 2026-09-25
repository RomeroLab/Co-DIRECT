
from __future__ import annotations

from enum import Enum

import numpy as np

from dive.ccr.constraints import ConstraintSet
from dive.ccr.operator import build_constraints, residue_obstructions

FEATURE_NAMES: tuple[str, ...] = (
    "D_rotamer",
    "D_backbone",
    "D_joint",
    "needs_joint_action",
    "rotamer_minus_joint",
    "backbone_minus_joint",
    "witness_bond_fraction",
    "witness_clash_fraction",
    "violated_fraction",
    "n_constraints_scaled",
    "time_fraction",
)

OBSTRUCTION_SCALE = 0.5

class MessageMode(Enum):

    FULL = "full"

    SCALAR_ONLY = "scalar_only"

    NO_WITNESS = "no_witness"

    NO_ACTION_SPLIT = "no_action_split"

def _squash(value: float) -> float:
    return float(np.tanh(value / OBSTRUCTION_SCALE))

def obstruction_features(
    coors: np.ndarray,
    *,
    residue: int,
    radius: float,
    time_fraction: float,
    mask: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
    constraints: ConstraintSet | None = None,
    constraint_index: dict[int, list[int]] | None = None,
    solver_iterations: int = 4000,
    max_local_constraints: int | None = None,
) -> np.ndarray:

    if constraints is None:
        constraints = build_constraints(coors, mask=mask, chain_index=chain_index)
    out = residue_obstructions(
        coors, residue=residue, radius=radius, mask=mask, chain_index=chain_index,
        constraints=constraints, constraint_index=constraint_index, check_primal=False,
        solver_iterations=solver_iterations,
        max_local_constraints=max_local_constraints,
    )
    rotamer, backbone, joint = out["rotamer"], out["backbone"], out["joint"]

    families = [name.split("[")[0] for name in joint.active_constraint_names]
    total = len(families) or 1
    bond_fraction = sum(1 for f in families if f != "clash") / total
    clash_fraction = sum(1 for f in families if f == "clash") / total

    local_values = np.array([
        constraints.values[i]
        for i, (a, b) in enumerate(constraints.atom_pairs)
        if a[0] == residue or b[0] == residue
    ]) if constraints.n_constraints else np.zeros(0)
    violated_fraction = float((local_values < 0).mean()) if local_values.size else 0.0

    needs_joint = float(
        joint.value <= 1e-6 and rotamer.value > 1e-6 and backbone.value > 1e-6
    )
    return np.array([
        _squash(rotamer.value),
        _squash(backbone.value),
        _squash(joint.value),
        needs_joint,
        _squash(rotamer.value - joint.value),
        _squash(backbone.value - joint.value),
        bond_fraction if joint.value > 1e-6 else 0.0,
        clash_fraction if joint.value > 1e-6 else 0.0,
        violated_fraction,
        float(np.tanh(local_values.size / 200.0)),
        float(time_fraction),
    ], dtype=np.float64)

def residue_message(
    coors: np.ndarray,
    *,
    residue: int,
    radius: float,
    time_fraction: float,
    mode: MessageMode = MessageMode.FULL,
    **kwargs,
) -> np.ndarray:

    features = obstruction_features(
        coors, residue=residue, radius=radius, time_fraction=time_fraction, **kwargs
    )
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out = features.copy()

    if mode in (MessageMode.SCALAR_ONLY, MessageMode.NO_ACTION_SPLIT):

        joint = out[index["D_joint"]]
        out[index["D_rotamer"]] = joint
        out[index["D_backbone"]] = joint
        out[index["rotamer_minus_joint"]] = 0.0
        out[index["backbone_minus_joint"]] = 0.0
        out[index["needs_joint_action"]] = 0.0

    if mode in (MessageMode.SCALAR_ONLY, MessageMode.NO_WITNESS):
        out[index["witness_bond_fraction"]] = 0.0
        out[index["witness_clash_fraction"]] = 0.0

    return out
