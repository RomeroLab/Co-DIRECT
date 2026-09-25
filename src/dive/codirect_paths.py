
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _required(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise SystemExit(
            f"set {name}. This repository does not vendor weights or structures.")
    return Path(raw)

def proteina_root() -> Path:

    return _required("PROTEINA_COMPLEXA_ROOT")

def proteina_src() -> Path:
    vendored = ROOT / "vendor" / "proteina-complexa" / "src"
    if not vendored.is_dir():
        raise SystemExit(f"vendored Proteina sources missing at {vendored}")
    return vendored

def v6_data() -> Path:
    return Path(os.environ.get("CODIRECT_V6_DATA", str(ROOT / "data" / "v6")))

def benchv2_root() -> Path:

    return _required("BENCHV2_ROOT")

def cache_dir(*parts: str) -> Path:

    base = ROOT / "runs" / "cache"
    return base.joinpath(*parts) if parts else base

def structure_root() -> Path:

    return _required("CODIRECT_STRUCTURE_ROOT")

def joined(env_name: str, *parts: str, default: Path | None = None) -> Path:

    raw = os.environ.get(env_name, "").strip()
    if raw:
        base = Path(raw)
    elif default is not None:
        base = default
    else:
        base = Path()
    return base.joinpath(*parts) if parts else base
