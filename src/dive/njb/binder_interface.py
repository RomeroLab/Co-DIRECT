
from __future__ import annotations

import re

import numpy as np

_SPAN = re.compile(r"^([A-Za-z])(\d+)-(\d+)$")

_HOTSPOT = re.compile(r"^([A-Za-z])(\d+)$")

class InterfaceError(RuntimeError):
    pass

def parse_target_input(target_input: str) -> tuple[str, int, int]:

    match = _SPAN.match((target_input or "").strip())
    if not match:
        raise InterfaceError(
            f"target span {target_input!r} is not of the form <chain><start>-<end>; "
            f"guessing its bounds would misplace every hotspot")
    chain, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if end < start:
        raise InterfaceError(f"target span {target_input!r} ends before it starts")
    return chain, start, end

def hotspot_positions(hotspots, target_input: str) -> list[int]:

    chain, start, end = parse_target_input(target_input)
    positions = []
    for hotspot in hotspots:
        match = _HOTSPOT.match(str(hotspot).strip())
        if not match:
            raise InterfaceError(f"hotspot {hotspot!r} is not of the form <chain><residue>")
        if match.group(1) != chain:
            raise InterfaceError(
                f"hotspot {hotspot!r} names chain {match.group(1)!r} but the target "
                f"span is on chain {chain!r}")
        residue = int(match.group(2))
        if not (start <= residue <= end):
            raise InterfaceError(
                f"hotspot {hotspot!r} lies outside the target span {target_input!r}")
        positions.append(residue - start + 1)
    return positions

def contacts_and_clashes(target_atoms, binder_atoms, *, hotspots,
                         contact_max: float, clash_min: float) -> dict:

    hotspots = list(hotspots)
    if not hotspots:
        raise InterfaceError(
            "no hotspot was declared, so the epitope condition is vacuous; a task "
            "without hotspots must be reported as having no epitope condition "
            "rather than as satisfying one")
    missing = [h for h in hotspots if h not in target_atoms]
    if missing:
        raise InterfaceError(
            f"hotspot positions {missing} are not in the target chain, which has "
            f"{len(target_atoms)} residues; the span and the structure disagree")

    binder = np.concatenate([np.asarray(v, dtype=float) for v in binder_atoms.values()])
    if binder.size == 0:
        raise InterfaceError("the binder chain carries no heavy atoms")

    in_contact, per_hotspot = 0, {}
    for position in hotspots:
        coords = np.asarray(target_atoms[position], dtype=float)
        distance = float(np.linalg.norm(
            coords[:, None, :] - binder[None, :, :], axis=-1).min())
        per_hotspot[position] = round(distance, 3)
        if distance <= contact_max:
            in_contact += 1

    target = np.concatenate([np.asarray(v, dtype=float) for v in target_atoms.values()])
    minimum = float(np.linalg.norm(
        target[:, None, :] - binder[None, :, :], axis=-1).min())

    return {
        "hotspots_declared": len(hotspots),
        "hotspots_in_contact": in_contact,
        "closest_approach_per_hotspot": per_hotspot,
        "min_interchain_distance": round(minimum, 3),

        "C_epitope": in_contact == len(hotspots),
        "C_clash": minimum >= clash_min,
    }
