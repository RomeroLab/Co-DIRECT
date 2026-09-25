
from __future__ import annotations

import re

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
MOTIF_RE = re.compile(r"^([A-Za-z])(\d+)-(\d+)$")
CONTIG_RE = re.compile(r"contigs=\[\\?'?([^\]'\\]+)")

def contig_pieces(raw: str) -> list:

    if not raw:
        raise ValueError("task has no raw_contig")
    m = CONTIG_RE.search(raw)
    if not m:
        raise ValueError("raw_contig has no contigmap.contigs=[...] body")
    out = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        if MOTIF_RE.match(part):
            out.append(part)
        else:
            out.append(int(part))
    if not out:
        raise ValueError("contig body is empty")
    return out

def contig_length(raw: str) -> int:

    n = 0
    for part in contig_pieces(raw):
        if isinstance(part, int):
            n += part
            continue
        m = MOTIF_RE.match(part)
        n += int(m.group(3)) - int(m.group(2)) + 1
    if n <= 0:
        raise ValueError("contig length is not positive")
    return n

def placements_in_length(raw: str, motif: dict, length: int = 180) -> list[dict]:

    pieces = contig_pieces(raw)
    scaffolds = [p for p in pieces if isinstance(p, int)]
    n_motif = sum(1 for p in pieces if not isinstance(p, int))
    if n_motif > length:
        raise ValueError(f"{n_motif} motif residues cannot fit in length {length}")
    total = contig_length(raw)
    if total == length:
        return placements(raw, motif, design_length=length)
    budget = length - n_motif
    ssum = sum(scaffolds)
    if ssum <= 0:
        raise ValueError("contig has no scaffold to shrink")
    scaled = [max(0, int(round(s * budget / ssum))) for s in scaffolds]
    delta = budget - sum(scaled)
    i = 0
    while delta != 0 and scaled:
        j = i % len(scaled)
        if delta > 0:
            scaled[j] += 1
            delta -= 1
        elif scaled[j] > 0:
            scaled[j] -= 1
            delta += 1
        i += 1
        if i > length * 4:
            raise ValueError("could not scale scaffolds to the design length")
    rebuilt = []
    si = 0
    for part in pieces:
        if isinstance(part, int):
            rebuilt.append(scaled[si])
            si += 1
        else:
            rebuilt.append(part)
    idx = 0
    out = []
    for part in rebuilt:
        if isinstance(part, int):
            idx += part
            continue
        m = MOTIF_RE.match(part)
        key = f"{m.group(1)}{int(m.group(2))}"
        spec = motif.get(key)
        if spec is None:
            raise ValueError(f"contig names {key} which is not in the task motif")
        atoms = list(spec.get("atoms") or [])
        if not atoms:
            raise ValueError(f"motif {key} declares no atoms")
        out.append({"task_motif_key": key, "design_index": idx, "atoms": atoms})
        idx += 1
    if idx != length:
        raise ValueError(f"fitted contig walks to {idx}, want {length}")
    if len(out) != len(motif):
        raise ValueError(
            f"fitted contig mapped {len(out)} motif residues, task declares {len(motif)}"
        )
    return out

def pin_motif_in_design(batch: dict, placements: list[dict], types: dict[str, str], aa1: str):

    import torch

    gen = batch["generated_mask"].bool()
    if gen.dim() == 2:
        gen0 = gen[0]
    else:
        gen0 = gen
    gen_idx = gen0.nonzero(as_tuple=False).view(-1)
    need = max(int(p["design_index"]) for p in placements) + 1
    if int(gen_idx.numel()) < need:
        raise ValueError(
            f"designed chain has {int(gen_idx.numel())} residues, "
            f"motif design_index needs {need}"
        )
    restype = batch["residue_type"]
    coords = batch.get("coords", batch.get("coors"))
    x_motif = batch.get("x_motif")
    for j, p in enumerate(placements):
        i = int(gen_idx[int(p["design_index"])])
        batch["generated_mask"][..., i] = False
        batch["fixed_mask"][..., i] = True
        aa = types[p["task_motif_key"]]
        code = aa1.index(aa)
        restype[..., i] = code
        if x_motif is not None and coords is not None and x_motif.shape[-3] == len(placements):
            coords[..., i, :, :] = x_motif[..., j, :, :]
    return batch

def placements(raw: str, motif: dict, design_length: int | None = None) -> list[dict]:

    pieces = contig_pieces(raw)
    idx = 0
    out = []
    for part in pieces:
        if isinstance(part, int):
            idx += part
            continue
        m = MOTIF_RE.match(part)
        chain, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        if start != end:
            raise ValueError(
                f"contig segment {part} spans {end - start + 1} residues; "
                "AME theozyme residues are single positions"
            )
        key = f"{chain}{start}"
        spec = motif.get(key)
        if spec is None:
            raise ValueError(f"contig names {key} which is not in the task motif")
        atoms = list(spec.get("atoms") or [])
        if not atoms:
            raise ValueError(f"motif {key} declares no atoms")
        out.append({"task_motif_key": key, "design_index": idx, "atoms": atoms})
        idx += 1
    if design_length is None:
        design_length = idx
    if idx != design_length:
        raise ValueError(
            f"contig walks to length {idx}, design_length is {design_length}"
        )
    if len(out) != len(motif):
        raise ValueError(
            f"contig mapped {len(out)} motif residues, task declares {len(motif)}"
        )
    return out

def native_types(pdb_path: str, motif: dict) -> dict[str, str]:

    got = {}
    for ln in open(pdb_path):
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if ln[12:16].strip() != "CA":
            continue
        key = f"{ln[21]}{int(ln[22:26])}"
        if key in motif and key not in got:
            aa = AA3.get(ln[17:20].strip())
            if not aa:
                raise ValueError(f"motif {key} has non-standard resname {ln[17:20]!r}")
            got[key] = aa
    missing = sorted(set(motif) - set(got))
    if missing:
        raise ValueError(f"no CA in {pdb_path} for motif residues {missing}")
    return got

def graft_motif_types(sequence: str, places: list[dict], types: dict[str, str]) -> str:

    if not sequence:
        raise ValueError("empty sequence")
    chars = list(sequence)
    for p in places:
        i = int(p["design_index"])
        key = p["task_motif_key"]
        if i < 0 or i >= len(chars):
            raise ValueError(
                f"design_index {i} for {key} is outside sequence length {len(chars)}"
            )
        aa = types.get(key)
        if not aa:
            raise ValueError(f"no native type for {key}")
        chars[i] = aa
    return "".join(chars)
