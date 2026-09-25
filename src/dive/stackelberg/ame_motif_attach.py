
from __future__ import annotations

import re
from pathlib import Path

_CAND = re.compile(r"^(?:.+__)?.+_cand(\d+)_seed\d+\.meta\.json$")
_CID = re.compile(r"^([^_]+(?:_[^_]+)*)__([^_].*)__candidate_(\d+)$")

def parse_candidate_id(candidate_id: str) -> tuple[str, str, int] | None:

    parts = candidate_id.split("__")
    if len(parts) != 3 or not parts[2].startswith("candidate_"):
        return None
    try:
        idx = int(parts[2].split("_", 1)[1])
    except (IndexError, ValueError):
        return None
    return parts[0], parts[1], idx

def generation_meta_path(generation_root: Path, candidate_id: str) -> Path | None:

    parsed = parse_candidate_id(candidate_id)
    if parsed is None:
        return None
    method, task_id, idx = parsed
    root = Path(generation_root)
    matches = sorted(root.glob(f"{task_id}/{task_id}_cand{idx}_*.meta.json"))
    if matches:
        return matches[0]
    canonical = root / method / "ame" / task_id / f"candidate_{idx}" / "meta.json"
    if canonical.exists():
        return canonical
    return None

def link_canonical_tree(generation_root: Path, dest_root: Path, method_id: str = "ANTIC") -> int:

    n = 0
    dest_root = Path(dest_root)
    for meta in sorted(Path(generation_root).glob("*/*_cand*_*.meta.json")):
        m = _CAND.match(meta.name)
        if m is None:
            continue
        idx = int(m.group(1))
        task_id = meta.parent.name
        target = dest_root / method_id / "ame" / task_id / f"candidate_{idx}" / "meta.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(meta.resolve())
        n += 1
    return n

def copy_ame_motif_onto_motif(metrics: dict) -> dict:

    for sample in metrics.get("samples") or []:
        block = sample.get("ame_motif")
        if isinstance(block, dict) and "motif" not in sample:
            sample["motif"] = {
                "motif_rmsd_self": block.get("motif_rmsd_self"),
                "motif_rmsd_ligand_frame": block.get("motif_rmsd_ligand_frame"),
            }
    return metrics
