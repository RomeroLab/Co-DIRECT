
from __future__ import annotations

import ast
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = [
    "MpnnFasta",
    "backbone_bond_integrity",
    "binder_chain",
    "ca_rmsd",
    "chain_residues",
    "hotspot_target_positions",
    "interface_metrics",
    "parse_ligandmpnn_fasta",
    "parse_mpnn_fasta",
    "renumber_target_template",
    "resolve_af2_target_residues",
]

CLASH_DISTANCE = 2.0

CONTACT_CUTOFF = 8.0

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def binder_chain(complex_chains: Iterable[str], native_chains: Iterable[str]) -> str:

    extra = sorted(set(complex_chains) - set(native_chains))
    if len(extra) != 1:
        raise ValueError(
            f"expected exactly one generated chain, got {extra!r} "
            f"(complex={sorted(set(complex_chains))!r}, "
            f"native={sorted(set(native_chains))!r})"
        )
    return extra[0]

@dataclass(frozen=True)
class MpnnFasta:

    fixed_chains: list[str]
    designed_chains: list[str]
    chain_order: list[str]
    layout: str
    sequence_by_chain: "OrderedDict[str, str]"
    input_by_chain: "OrderedDict[str, str]"
    header: str = ""
    sample_header: str = ""

    @property
    def _designed(self) -> str:
        if len(self.designed_chains) != 1:
            raise ValueError(
                f"expected one designed chain, got {self.designed_chains!r}"
            )
        return self.designed_chains[0]

    @property
    def raw_designed_sequence(self) -> str:

        return self.sequence_by_chain[self._designed]

    @property
    def designed_sequence(self) -> str:

        return self.raw_designed_sequence.replace("X", "")

    @property
    def input_sequence(self) -> str:

        return self.input_by_chain[self._designed].replace("X", "")

    @property
    def span_length(self) -> int:

        return len(self.raw_designed_sequence)

    @property
    def resolved_length(self) -> int:

        return len(self.designed_sequence)

    @property
    def unresolved_positions(self) -> list[int]:

        return [i for i, c in enumerate(self.raw_designed_sequence) if c == "X"]

def _chain_list(header: str, key: str) -> list[str]:
    match = re.search(rf"{key}=(\[[^\]]*\])", header)
    if match is None:
        raise ValueError(f"header has no {key}: {header!r}")
    value = ast.literal_eval(match.group(1))
    return [str(item) for item in value]

def parse_mpnn_fasta(text: str) -> MpnnFasta:

    blocks = [b for b in text.split(">")[1:] if b.strip()]
    if len(blocks) < 2:
        raise ValueError("FASTA has no sampled record (need input + sample)")

    input_header, _, input_body = blocks[0].partition("\n")
    sample_header, _, sample_body = blocks[1].partition("\n")

    fixed = _chain_list(input_header, "fixed_chains")
    designed = _chain_list(input_header, "designed_chains")
    if not designed:
        raise ValueError(f"header declares no designed chain: {input_header!r}")
    order = list(fixed) + list(designed)

    def parts_of(body: str) -> list[str]:
        return [
            "".join(p.split()).upper()
            for p in "".join(body.split()).split("/")
        ]

    sample_parts = parts_of(sample_body)
    input_parts = parts_of(input_body)
    if len(sample_parts) != len(input_parts):
        raise ValueError(
            f"input record has {len(input_parts)} chains but the sample record "
            f"has {len(sample_parts)}"
        )

    if len(sample_parts) == len(order):
        layout = "all_chains"
        keys = order
    elif len(sample_parts) == len(designed):
        layout = "designed_only"
        keys = list(designed)
    else:
        raise ValueError(
            f"record has {len(sample_parts)} chains, which matches neither the "
            f"{len(order)} declared chains {order!r} nor the "
            f"{len(designed)} designed chains {designed!r}"
        )

    parsed = MpnnFasta(
        fixed_chains=fixed,
        designed_chains=designed,
        chain_order=list(order),
        layout=layout,
        sequence_by_chain=OrderedDict(zip(keys, sample_parts)),
        input_by_chain=OrderedDict(zip(keys, input_parts)),
        header=input_header.strip(),
        sample_header=sample_header.strip(),
    )
    for chain in designed:
        if not parsed.sequence_by_chain[chain].replace("X", ""):
            raise ValueError(
                f"designed chain {chain!r} has no resolved residues "
                "(all positions are X)"
            )
    return parsed

def chain_residues(pdb_text: str) -> "OrderedDict[str, list[tuple[int, str]]]":

    out: "OrderedDict[str, OrderedDict[str, tuple[int, str]]]" = OrderedDict()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        key = line[22:27]
        out.setdefault(chain, OrderedDict())[key] = (
            int(line[22:26]),
            line[17:20].strip(),
        )
    return OrderedDict((c, list(d.values())) for c, d in out.items())

def chain_sequence(pdb_text: str, chain: str) -> str:
    residues = chain_residues(pdb_text).get(chain)
    if residues is None:
        raise ValueError(f"chain {chain!r} absent from structure")
    return "".join(_THREE_TO_ONE.get(name, "X") for _, name in residues)

def renumber_target_template(pdb_text: str, chains: Sequence[str]) -> str:

    wanted = list(chains)
    present = {line[21] for line in pdb_text.splitlines() if line.startswith("ATOM")}
    missing = [c for c in wanted if c not in present]
    if missing:
        raise ValueError(f"chains {missing!r} absent from structure")

    counters: dict[str, int] = {c: 0 for c in wanted}
    last_key: dict[str, str] = {}
    serial = 0
    out: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        if chain not in counters:
            continue
        key = line[22:27]
        if last_key.get(chain) != key:
            counters[chain] += 1
            last_key[chain] = key
        serial += 1
        out.append(
            f"{line[:6]}{serial:>5d}{line[11:22]}{counters[chain]:>4d} {line[27:]}"
        )
    out.append("END")
    return "\n".join(out) + "\n"

def _atoms(pdb_text: str) -> list[tuple[str, int, str, str, tuple[float, float, float]]]:
    rows = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        rows.append(
            (
                line[21],
                int(line[22:26]),
                line[12:16].strip(),
                line[17:20].strip(),
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            )
        )
    return rows

def _representative_atoms(
    rows: Sequence[tuple[str, int, str, str, tuple[float, float, float]]],
    chain: str,
) -> "OrderedDict[int, tuple[float, float, float]]":

    ca: "OrderedDict[int, tuple[float, float, float]]" = OrderedDict()
    cb: dict[int, tuple[float, float, float]] = {}
    for ch, resseq, name, _resname, xyz in rows:
        if ch != chain:
            continue
        if name == "CA":
            ca.setdefault(resseq, xyz)
        elif name == "CB":
            cb.setdefault(resseq, xyz)
    return OrderedDict((r, cb.get(r, xyz)) for r, xyz in ca.items())

def ca_alpha_coords(pdb_text: str, chain: str) -> list[tuple[float, float, float]]:
    rows = _atoms(pdb_text)
    coords = [xyz for ch, _r, name, _rn, xyz in rows if ch == chain and name == "CA"]
    if not coords:
        raise ValueError(f"chain {chain!r} has no CA atoms")
    return coords

def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )

def interface_metrics(
    pdb_text: str,
    binder: str,
    targets: Sequence[str],
    contact_cutoff: float = CONTACT_CUTOFF,
    hotspots: Sequence[str] | None = None,
    clash_distance: float = CLASH_DISTANCE,
) -> dict:

    rows = _atoms(pdb_text)
    chains_present = {r[0] for r in rows}
    if binder not in chains_present:
        raise ValueError(f"binder chain {binder!r} absent from structure")

    binder_reps = _representative_atoms(rows, binder)
    target_reps: dict[str, "OrderedDict[int, tuple[float, float, float]]"] = {
        c: _representative_atoms(rows, c) for c in targets
    }

    contact_pairs = 0
    binder_hits: set[int] = set()
    target_hits: set[tuple[str, int]] = set()
    for chain, reps in target_reps.items():
        for tres, txyz in reps.items():
            for bres, bxyz in binder_reps.items():
                if _dist(txyz, bxyz) <= contact_cutoff:
                    contact_pairs += 1
                    binder_hits.add(bres)
                    target_hits.add((chain, tres))

    binder_heavy = [xyz for ch, _r, _n, _rn, xyz in rows if ch == binder]
    target_heavy = [
        (ch, xyz) for ch, _r, _n, _rn, xyz in rows if ch in set(targets)
    ]
    min_distance = math.inf
    clashes = 0
    for _ch, txyz in target_heavy:
        for bxyz in binder_heavy:
            d = _dist(txyz, bxyz)
            if d < min_distance:
                min_distance = d
            if d < clash_distance:
                clashes += 1

    declared = list(hotspots or [])
    contacted = 0
    for token in declared:
        match = re.fullmatch(r"([A-Za-z])(-?\d+)", token.strip())
        if match and (match.group(1), int(match.group(2))) in target_hits:
            contacted += 1

    return {
        "binder_chain": binder,
        "target_chains": list(targets),
        "contact_cutoff": contact_cutoff,
        "clash_distance": clash_distance,
        "n_contact_pairs": contact_pairs,
        "n_interface_binder_residues": len(binder_hits),
        "n_interface_target_residues": len(target_hits),
        "min_interchain_distance": None if min_distance is math.inf else min_distance,
        "n_interchain_clashes": clashes,
        "n_hotspots_declared": len(declared),
        "n_hotspots_contacted": contacted,
        "hotspot_recovery": (contacted / len(declared)) if declared else None,
    }

def ca_rmsd(
    p: Sequence[Sequence[float]], q: Sequence[Sequence[float]]
) -> float:

    if len(p) != len(q):
        raise ValueError(f"length mismatch: {len(p)} != {len(q)}")
    if not p:
        raise ValueError("cannot superpose empty coordinate sets")

    n = len(p)
    pc = [sum(row[i] for row in p) / n for i in range(3)]
    qc = [sum(row[i] for row in q) / n for i in range(3)]
    x = [[row[i] - pc[i] for i in range(3)] for row in p]
    y = [[row[i] - qc[i] for i in range(3)] for row in q]

    try:
        import numpy as np
    except ImportError:
        raise

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    cov = xa.T @ ya
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rot = vt.T @ correction @ u.T
    diff = (rot @ xa.T).T - ya
    return float(np.sqrt((diff**2).sum() / n))

CA_BOND_MIN = 3.5
CA_BOND_MAX = 4.2

BACKBONE_VALID_FRACTION = 0.95

def backbone_bond_integrity(
    pdb_text: str,
    chain: str,
    bond_min: float = CA_BOND_MIN,
    bond_max: float = CA_BOND_MAX,
    valid_fraction: float = BACKBONE_VALID_FRACTION,
) -> dict:

    residues: dict[int, tuple[float, float, float]] = {}
    seen = False
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        seen = True
        if line[12:16].strip() != "CA":
            continue
        residues.setdefault(
            int(line[22:26]),
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        )
    if not seen:
        raise ValueError(f"chain {chain!r} absent from structure")

    numbers = sorted(residues)
    distances: list[float] = []
    gaps = 0
    for a, b in zip(numbers, numbers[1:]):
        if b - a == 1:
            distances.append(_dist(residues[a], residues[b]))
        else:
            gaps += 1

    if not distances:
        return {
            "chain": chain,
            "n_residues": len(numbers),
            "n_adjacent_pairs": 0,
            "n_numbering_gaps": gaps,
            "frac_in_range": None,
            "min_adjacent_distance": None,
            "max_adjacent_distance": None,
            "valid": None,
        }

    in_range = sum(1 for d in distances if bond_min < d < bond_max)
    fraction = in_range / len(distances)
    return {
        "chain": chain,
        "n_residues": len(numbers),
        "n_adjacent_pairs": len(distances),
        "n_numbering_gaps": gaps,
        "frac_in_range": fraction,
        "min_adjacent_distance": min(distances),
        "max_adjacent_distance": max(distances),
        "valid": bool(fraction >= valid_fraction),
    }

def parse_ligandmpnn_fasta(text: str) -> dict:

    blocks = [b for b in text.split(">")[1:] if b.strip()]
    if len(blocks) < 2:
        raise ValueError("LigandMPNN FASTA has no sampled record")

    input_header, _, input_body = blocks[0].partition("\n")
    sample_header, _, sample_body = blocks[1].partition("\n")

    def clean(body: str) -> str:
        return re.sub(r"[^A-Z]", "", "".join(body.split()).upper())

    raw = clean(sample_body)
    designed = raw.replace("X", "")
    if not designed:
        raise ValueError("design has no resolved residues (all positions are X)")

    def field(name: str):
        match = re.search(rf"{name}=([^,\s]+)", input_header)
        return None if match is None else match.group(1)

    def as_int(name: str):
        value = field(name)
        return None if value is None else int(value)

    ligand_context = field("use_ligand_context")
    return {
        "input_sequence": clean(input_body).replace("X", ""),
        "raw_designed_sequence": raw,
        "designed_sequence": designed,
        "span_length": len(raw),
        "resolved_length": len(designed),
        "unresolved_positions": [i for i, c in enumerate(raw) if c == "X"],
        "num_res": as_int("num_res"),
        "num_ligand_res": as_int("num_ligand_res"),
        "use_ligand_context": None if ligand_context is None else ligand_context == "True",
        "ligand_cutoff_distance": (
            None if field("ligand_cutoff_distance") is None
            else float(field("ligand_cutoff_distance"))
        ),
        "header": input_header.strip(),
        "sample_header": sample_header.strip(),
    }

def hotspot_target_positions(
    native_text: str, target_chains: Sequence[str], tokens: Sequence[str]
) -> tuple[list[int], list[str]]:

    residues = chain_residues(native_text)
    missing_chains = [c for c in target_chains if c not in residues]
    if missing_chains:
        raise ValueError(f"chains {missing_chains!r} absent from structure")

    index: dict[tuple[str, int], int] = {}
    position = 0
    for chain in target_chains:
        for resnum, _name in residues[chain]:
            index.setdefault((chain, resnum), position)
            position += 1

    positions: list[int] = []
    unresolved: list[str] = []
    for token in tokens:
        match = re.fullmatch(r"([A-Za-z])(-?\d+)", token.strip())
        key = (match.group(1), int(match.group(2))) if match else None
        if key in index:
            positions.append(index[key])
        else:
            unresolved.append(token)
    return positions, unresolved

def resolve_af2_target_residues(
    af2_text: str, target_chains: Sequence[str], positions: Sequence[int]
) -> list[tuple[str, int]]:

    residues = chain_residues(af2_text)
    ordered: list[tuple[str, int]] = []
    for chain in target_chains:
        for resnum, _name in residues.get(chain, []):
            ordered.append((chain, resnum))
    return [ordered[p] for p in positions if 0 <= p < len(ordered)]
