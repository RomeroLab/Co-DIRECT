
from __future__ import annotations

from pathlib import Path

def existing_design_pdb(task_dir: Path, stem: str) -> Path | None:

    path = Path(task_dir) / f"{stem}.pdb"
    if path.is_file() and path.stat().st_size > 0:
        return path
    return None
