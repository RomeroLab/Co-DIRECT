
from __future__ import annotations

import yaml

def yaml_scalar(value: str) -> str:

    if not value:
        raise ValueError("refusing to emit an empty YAML scalar")
    dumped = yaml.safe_dump(
        value, default_flow_style=True, width=10**9, allow_unicode=True
    ).strip()

    if dumped.endswith("\n..."):
        dumped = dumped[: -len("\n...")]
    dumped = dumped.rstrip()
    if dumped.endswith("..."):
        dumped = dumped[:-3].rstrip()
    if yaml.safe_load(f"v: {dumped}\n")["v"] != value:
        raise ValueError(f"could not emit a round-tripping scalar for {value!r}")
    return dumped
