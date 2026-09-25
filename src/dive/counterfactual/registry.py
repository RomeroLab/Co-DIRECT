
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

class UnknownFeaturePathError(RuntimeError):
    pass

class CleanFeatureLeakageError(RuntimeError):
    pass

class FeatureClass(StrEnum):

    PRESERVED = "preserved"
    BACKBONE_DERIVED = "backbone_derived"
    LATENT_DERIVED = "latent_derived"
    CONDITIONING = "conditioning"
    FORBIDDEN_CLEAN = "forbidden_clean"

def flatten_paths(batch: Mapping) -> dict[str, object]:

    flat: dict[str, object] = {}
    _flatten_into(batch, prefix="", flat=flat)
    return flat

def _flatten_into(node: object, prefix: str, flat: dict[str, object]) -> None:

    if isinstance(node, Mapping):
        for key, child in node.items():
            _flatten_into(child, f"{prefix}.{key}" if prefix else str(key), flat)
        return
    if isinstance(node, (list, tuple)) and not isinstance(node, (str, bytes)):
        for index, child in enumerate(node):
            _flatten_into(child, f"{prefix}.{index}", flat)
        return
    flat[prefix] = node

@dataclass(frozen=True, slots=True)
class FeatureRegistry:

    classes: Mapping[str, FeatureClass]

    @classmethod
    def from_yaml(cls, path: Path | str) -> FeatureRegistry:

        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls({key: FeatureClass(value) for key, value in raw.items()})

    def assert_complete(self, batch: Mapping) -> None:

        unknown = sorted(set(flatten_paths(batch)) - set(self.classes))
        if unknown:
            raise UnknownFeaturePathError(
                "unclassified upstream feature paths: " + ", ".join(unknown)
            )

    def assert_denoiser_safe(self, batch: Mapping) -> None:

        forbidden = sorted(
            path
            for path in flatten_paths(batch)
            if self.classes.get(path) is FeatureClass.FORBIDDEN_CLEAN
        )
        if forbidden:
            raise CleanFeatureLeakageError(
                "clean-target paths must never reach the denoiser: " + ", ".join(forbidden)
            )

    def paths_of(self, feature_class: FeatureClass) -> tuple[str, ...]:

        return tuple(sorted(p for p, c in self.classes.items() if c is feature_class))

    def sha256(self) -> str:

        canonical = json.dumps(
            {key: str(value) for key, value in sorted(self.classes.items())},
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, str]:

        return {key: str(value) for key, value in sorted(self.classes.items())}
