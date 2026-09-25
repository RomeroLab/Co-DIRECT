
from __future__ import annotations

import json
from pathlib import Path

from dive.codirect_paths import benchv2_root

E3_ROOT = benchv2_root() / "generation" / "E3" / "ame"
DESIGN_LENGTH = 180

def e3_meta_path(task_id: str, candidate_index: int) -> Path:
    return E3_ROOT / task_id / f"candidate_{int(candidate_index)}" / "meta.json"

def e3_motif_positions(task_id: str, candidate_index: int) -> list[dict]:
    path = e3_meta_path(task_id, candidate_index)
    if not path.is_file():
        raise FileNotFoundError(f"no E3 motif record at {path}")
    meta = json.loads(path.read_text())
    places = meta.get("motif_positions")
    if not places:
        raise ValueError(f"E3 {task_id} cand{candidate_index} has empty motif_positions")
    for p in places:
        i = int(p["design_index"])
        if i < 0 or i >= DESIGN_LENGTH:
            raise ValueError(
                f"E3 design_index {i} is outside the 180-mer for {task_id}")
    seq = meta.get("sequence") or ""
    if seq and len(seq) != DESIGN_LENGTH:
        raise ValueError(
            f"E3 sequence length {len(seq)} != {DESIGN_LENGTH} for {task_id}")
    return places

AA1 = "ARNDCQEGHILKMFPSTWYV"

def planted_sequence(places: list[dict], types: dict[str, str],
                     length: int = DESIGN_LENGTH, fill: str = "G") -> str:

    if length != DESIGN_LENGTH:
        raise ValueError(f"planted chain must be {DESIGN_LENGTH}, got {length}")
    chars = [fill] * length
    for p in places:
        i = int(p["design_index"])
        key = p["task_motif_key"]
        if i < 0 or i >= length:
            raise ValueError(f"design_index {i} outside {length}")
        aa = types.get(key)
        if not aa:
            raise ValueError(f"no native type for {key}")
        chars[i] = aa
    return "".join(chars)

def freeze_catalyst_rows(tensors: dict, places: list[dict]) -> dict:

    idx = [int(p["design_index"]) for p in places]

    def _set(t, i, val):
        if t.dim() >= 2 and i < t.shape[-1 if t.dim() == 1 else 1]:
            if t.dim() == 2:
                t[..., i] = val
            elif t.dim() == 1:
                t[i] = val
        elif t.dim() == 1 and i < t.shape[0]:
            t[i] = val

    gen = tensors.get("generated_mask")
    if gen is None:
        gen = tensors.get("design_mask")
    fix = tensors.get("fixed_mask")
    for i in idx:
        if gen is not None:
            _set(gen, i, False)
        if fix is not None:
            _set(fix, i, True)
    return tensors

def plant_residue_types(tensors: dict, places: list[dict], types: dict[str, str]) -> dict:
    if "residue_type" not in tensors:
        return tensors
    rt = tensors["residue_type"]
    for p in places:
        i = int(p["design_index"])
        aa = types[p["task_motif_key"]]
        code = AA1.index(aa)
        if rt.dim() == 2:
            rt[..., i] = code
        elif rt.dim() == 1:
            rt[i] = code
    return tensors

def graft_at_e3_indices(sequence: str, places: list[dict], types: dict[str, str]) -> str:

    if len(sequence) != DESIGN_LENGTH:
        raise ValueError(
            f"refusing to graft onto length {len(sequence)}; E3 environment is "
            f"a {DESIGN_LENGTH}-mer (contigmap.length=180-180). Longer contig "
            "walks are not E3's design."
        )
    chars = list(sequence)
    for p in places:
        i = int(p["design_index"])
        key = p["task_motif_key"]
        aa = types.get(key)
        if not aa:
            raise ValueError(f"no native type for {key}")
        if i < 0 or i >= len(chars):
            raise ValueError(f"design_index {i} outside {len(chars)}")
        chars[i] = aa
    return "".join(chars)
