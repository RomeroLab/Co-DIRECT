
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from dive.provenance import FileEvidence

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INPUT_SPECS = {
    "generator": ("generator", "checkpoint", "size_bytes", "sha256"),
    "autoencoder": ("autoencoder", "checkpoint", "size_bytes", "sha256"),
    "critic_parameter": (
        "critic",
        "parameter_path",
        "parameter_size_bytes",
        "parameter_sha256",
    ),
}

class InputIdentityError(RuntimeError):
    pass

def validate_runtime_inputs(
    config_path: Path,
    config: Mapping[str, object],
    *,
    required: Sequence[str],
) -> dict[str, dict[str, object]]:

    names = tuple(required)
    if len(names) != len(set(names)) or any(name not in _INPUT_SPECS for name in names):
        raise InputIdentityError(f"unknown or duplicate runtime input inventory: {names!r}")
    preflight = _load_preflight(config)
    files = preflight.get("files")
    if not isinstance(files, Mapping):
        raise InputIdentityError("CPU preflight runtime input files are missing or malformed")

    observed: dict[str, dict[str, object]] = {}
    config_evidence = FileEvidence.from_path(Path(config_path).resolve()).to_dict()
    _require_preflight_match(files, "signed_value_config", config_evidence)
    observed["signed_value_config"] = config_evidence

    for name in names:
        mapping_name, path_name, size_name, hash_name = _INPUT_SPECS[name]
        section = config.get(mapping_name)
        if not isinstance(section, Mapping):
            raise InputIdentityError(f"runtime input config mapping is missing: {mapping_name}")
        raw_path = section.get(path_name)
        expected_size = section.get(size_name)
        expected_sha256 = section.get(hash_name)
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise InputIdentityError(f"runtime input path is invalid: {name}")
        if type(expected_size) is not int or expected_size <= 0:
            raise InputIdentityError(f"runtime input size is not a literal positive int: {name}")
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
            raise InputIdentityError(f"runtime input SHA-256 is malformed: {name}")
        evidence = FileEvidence.from_path(Path(raw_path).resolve()).to_dict()
        if evidence["size_bytes"] != expected_size or evidence["sha256"] != expected_sha256:
            raise InputIdentityError(f"runtime input differs from typed frozen config: {name}")
        _require_preflight_match(files, name, evidence)
        observed[name] = evidence
    return observed

def _load_preflight(config: Mapping[str, object]) -> Mapping[str, object]:
    raw_path = config.get("cpu_preflight")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise InputIdentityError("CPU preflight path is missing or malformed")
    try:
        payload = json.loads(Path(raw_path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InputIdentityError("CPU preflight evidence is missing or malformed") from error
    if not isinstance(payload, Mapping):
        raise InputIdentityError("CPU preflight evidence must be an object")
    return payload

def _require_preflight_match(
    files: Mapping[str, object], name: str, observed: Mapping[str, object]
) -> None:
    expected = files.get(name)
    if not _valid_file_evidence(expected) or dict(expected) != dict(observed):
        raise InputIdentityError(f"runtime input differs from CPU preflight: {name}")

def _valid_file_evidence(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"path", "size_bytes", "sha256"}:
        return False
    return (
        isinstance(value["path"], str)
        and Path(value["path"]).is_absolute()
        and type(value["size_bytes"]) is int
        and value["size_bytes"] > 0
        and isinstance(value["sha256"], str)
        and _SHA256.fullmatch(value["sha256"]) is not None
    )
