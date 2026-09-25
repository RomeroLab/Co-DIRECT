
from __future__ import annotations

import json
from pathlib import Path

def output_dir_for(out: Path, example_id: str) -> Path:

    return Path(out) / example_id.replace(":", "_")

def existing_record(out: Path, example_id: str) -> dict | None:

    stats = output_dir_for(out, example_id) / "stats.json"
    if not stats.is_file():
        return None
    return json.loads(stats.read_text(encoding="utf-8"))
