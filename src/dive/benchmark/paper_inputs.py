
from __future__ import annotations

from dive.codirect_paths import joined

import hashlib
import io
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.views import (
    BenchmarkViewError,
    _ACCEPTED_INPUT_IDENTITIES,
    _authenticate_input_run,
    _canonical_object,
    _inputs_from_inventory,
    _read_retained,
)

class PaperInputError(RuntimeError):
    pass

_FAMILIES = ("binder", "ame", "antibody")
_RESOLVED_LIGAND_PATH = Path(
    joined('CODIRECT_STRUCTURE_ROOT', 'sources', 'plinder-2024-06', 'resolved_systems.parquet')
)
_RESOLVED_LIGAND_SHA256 = (
    "085928368204c61b7d405ecf3e2d2be724927ece5dff1c1b77a943534ac5e63c"
)
_RESOLVED_LIGAND_SIZE_BYTES = 24476351
_VIEW_RUN_ID = "benchmark-views-20260828a"
_VIEW_SEMANTIC_HASH = "f5e54f8e0d91fe857f3d36135fb603f921523fa16563eb83d2b099e34ac8d253"
_VIEW_PARTITION_HASH = (
    "527c359aefa234ea3015f645a8d128b587ad22c7e0b1aa263491c6a54a11fc03"
)
_FROZEN_COUNTS = {
    "binder": {"parent_groups": 2200, "example_rows": 3226},
    "ame": {"parent_groups": 1893, "example_rows": 4435},
    "antibody": {"parent_groups": 141, "example_rows": 281},
}
_GATE0_ANTIBODY_CANONICAL = 1166
_QUARANTINE_ROWS = 101
_QUARANTINE_REASONS = {
    "no_compatible_chain": 71,
    "role_contract_error": 28,
    "no_complete_assignment": 2,
}
_EXAMPLE_COLUMNS = {
    "example_id",
    "parent_id",
    "family",
    "partition",
    "strict",
}
_QUARANTINE_COLUMNS = {
    "example_id",
    "parent_id",
    "family",
    "partition",
    "quarantine_reason",
}

@dataclass(frozen=True, slots=True)
class PaperTargetRecord:
    parent_id: str
    example_id: str
    family: str
    partition: str
    strict: bool
    source_path: str
    source_sha256: str
    role_payload: Mapping[str, str]
    loader_manifest_sha256: str
    canonical_component_token: str | None = None
    ligand_identity: str | None = None
    ligand_smiles: str | None = None

@dataclass(frozen=True, slots=True)
class PaperInputBundle:
    targets: tuple[PaperTargetRecord, ...]
    counts: Mapping[str, Mapping[str, int]]
    semantic_hash: str
    partition_hash: str
    sources: tuple[ArtifactIdentity, ...]

def load_paper_inputs(*, view_run: Path, input_run: Path) -> PaperInputBundle:

    view_dir = _require_directory(view_run, "view run")
    input_dir = _require_directory(input_run, "input run")
    _reject_blind(view_dir, input_dir)

    input_completion, inventory = _call_view(_authenticate_input_run, input_dir)
    frozen_inputs = _call_view(_inputs_from_inventory, inventory)
    _assert_gate0_antibody_denominator(frozen_inputs.examples)

    loader_manifests = _loader_manifests(inventory)
    sources = [input_completion, _inventory_identity(input_dir)]
    for family in _FAMILIES:
        identity = loader_manifests.get((family, "validation"))
        if identity is None:
            raise PaperInputError(f"missing {family} validation loader manifest")
        _reject_blind(identity.path)
        _call_view(_read_retained, identity, f"{family} validation loader manifest")
        sources.append(identity)

    completion_path = view_dir / "completion.json"
    completion_raw = _read_regular_bytes(completion_path, "view completion")
    completion = _call_view(_canonical_object, completion_raw, "view completion")
    if (
        completion.get("run_id") != _VIEW_RUN_ID
        or completion.get("status") != "COMPLETE"
    ):
        raise PaperInputError("published view run is not authentic")
    payload = completion.get("payload")
    if not isinstance(payload, Mapping):
        raise PaperInputError("view completion payload is invalid")
    semantic_hash = payload.get("semantic_hash")
    partition_hash = payload.get("partition_hash")
    if semantic_hash != _VIEW_SEMANTIC_HASH or partition_hash != _VIEW_PARTITION_HASH:
        raise PaperInputError("view hash drifted")
    completion_identity = ArtifactIdentity(
        str(completion_path.resolve()),
        hashlib.sha256(completion_raw).hexdigest(),
        len(completion_raw),
    )
    sources.append(completion_identity)

    summary_identity = _declared_file(
        payload.get("view_summary"),
        view_dir,
        "view_summary.json",
        "view summary",
    )
    summary_raw = _call_view(_read_retained, summary_identity, "view summary")
    summary = _call_view(_canonical_object, summary_raw, "view summary")
    if (
        summary.get("semantic_hash") != _VIEW_SEMANTIC_HASH
        or summary.get("partition_hash") != _VIEW_PARTITION_HASH
    ):
        raise PaperInputError("view hash drifted")
    sources.append(
        ArtifactIdentity(
            str((view_dir / "view_summary.json").resolve()),
            summary_identity.sha256,
            summary_identity.size_bytes,
        )
    )

    example_parquets = payload.get("example_parquets")
    if not isinstance(example_parquets, Mapping):
        raise PaperInputError("view completion is missing example parquets")

    examples_by_id = _unique_examples(frozen_inputs.examples)
    parents_by_key = _unique_parents(frozen_inputs.parents)
    quarantine_identity = _declared_file(
        payload.get("antibody_quarantine"),
        view_dir,
        "antibody_quarantine.parquet",
        "antibody quarantine",
    )
    quarantine_raw = _call_view(
        _read_retained, quarantine_identity, "antibody quarantine"
    )
    quarantine_ids = _quarantine_example_ids(quarantine_raw)
    sources.append(
        ArtifactIdentity(
            str((view_dir / "antibody_quarantine.parquet").resolve()),
            quarantine_identity.sha256,
            quarantine_identity.size_bytes,
        )
    )

    targets: list[PaperTargetRecord] = []
    seen_example_ids: set[str] = set()
    resolved_smiles: Mapping[str, str] | None = None
    for family in _FAMILIES:
        name = f"{family}_validation_examples.parquet"
        _reject_blind(name)
        identity = _declared_file(
            example_parquets.get(name),
            view_dir,
            name,
            f"{family} validation examples",
        )
        raw = _call_view(_read_retained, identity, f"{family} validation examples")
        sources.append(
            ArtifactIdentity(
                str((view_dir / name).resolve()),
                identity.sha256,
                identity.size_bytes,
            )
        )
        manifest = loader_manifests[(family, "validation")]
        for row in _example_rows(raw, family):
            _reject_blind(row["partition"])
            if row["partition"] != "validation":
                raise PaperInputError("paper inputs retain validation only")
            if not row["strict"]:
                continue
            example_id = row["example_id"]
            if example_id in seen_example_ids:
                raise PaperInputError(
                    f"duplicate example {example_id!r} in strict validation"
                )
            seen_example_ids.add(example_id)
            inventory_example = examples_by_id.get(example_id)
            if inventory_example is None:
                raise PaperInputError(
                    f"example {example_id!r} is missing from the inventory"
                )
            if (
                inventory_example.parent_id != row["parent_id"]
                or inventory_example.family != row["family"]
                or inventory_example.partition != row["partition"]
            ):
                raise PaperInputError(f"parent mismatch for example {example_id!r}")
            parent_key = (
                inventory_example.parent_id,
                inventory_example.family,
                inventory_example.partition,
            )
            if parent_key not in parents_by_key:
                raise PaperInputError(
                    f"parent/family/partition match is missing for {example_id!r}"
                )
            _reject_blind(inventory_example.path)
            if not inventory_example.role_realized:
                raise PaperInputError(
                    f"role-unrealized example {example_id!r} is forbidden"
                )
            if example_id in quarantine_ids or inventory_example.quarantine_reason:
                raise PaperInputError(
                    f"quarantined example {example_id!r} is forbidden"
                )
            if inventory_example.family == "ame" and resolved_smiles is None:
                resolved_smiles = _load_resolved_ligand_smiles()
            token, ligand_identity, smiles = _ame_ligand_identity(
                inventory_example.family, example_id, resolved_smiles
            )
            targets.append(
                PaperTargetRecord(
                    inventory_example.parent_id,
                    example_id,
                    inventory_example.family,
                    inventory_example.partition,
                    True,
                    inventory_example.path,
                    identity.sha256,
                    MappingProxyType(dict(inventory_example.roles)),
                    manifest.sha256,
                    token,
                    ligand_identity,
                    smiles,
                )
            )

    targets.sort(key=lambda item: (item.family, item.parent_id, item.example_id))
    leaked = quarantine_ids.intersection(item.example_id for item in targets)
    if leaked:
        raise PaperInputError("antibody quarantine rows leaked into paper targets")
    counts = _counts_for(targets)
    if counts != _FROZEN_COUNTS:
        raise PaperInputError("strict-validation membership drifted")
    unique_sources = tuple(
        sorted({item for item in sources}, key=lambda item: item.path)
    )
    return PaperInputBundle(
        tuple(targets),
        MappingProxyType(
            {
                family: MappingProxyType(dict(values))
                for family, values in counts.items()
            }
        ),
        _VIEW_SEMANTIC_HASH,
        _VIEW_PARTITION_HASH,
        unique_sources,
    )

def _require_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path):
        raise PaperInputError(f"{label} must be a Path")
    _reject_blind(path)
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise PaperInputError(f"cannot stat {label}: {error}") from error
    if not os.path.stat.S_ISDIR(metadata.st_mode):
        raise PaperInputError(f"{label} must be a directory")
    return path

def _inventory_identity(input_dir: Path) -> ArtifactIdentity:
    sha256, size_bytes = _ACCEPTED_INPUT_IDENTITIES["input_inventory.json"]
    return ArtifactIdentity(
        str((input_dir / "input_inventory.json").resolve()),
        sha256,
        size_bytes,
    )

def _assert_gate0_antibody_denominator(examples) -> None:
    antibody = sum(1 for item in examples if item.family == "antibody")
    if antibody != _GATE0_ANTIBODY_CANONICAL:
        raise PaperInputError("Gate 0 antibody role-yield denominator drifted")

def _loader_manifests(
    inventory: Mapping[str, object],
) -> dict[tuple[str, str], ArtifactIdentity]:
    raw = inventory.get("loader_manifests")
    if not isinstance(raw, list):
        raise PaperInputError("input inventory is missing loader manifests")
    manifests: dict[tuple[str, str], ArtifactIdentity] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise PaperInputError("loader manifest identity is invalid")
        try:
            path = Path(str(item["path"]))
            sha256 = str(item["sha256"])
            size_bytes = int(item["size_bytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise PaperInputError("loader manifest identity is invalid") from error
        _reject_blind(path)
        name = path.name
        if not name.endswith(".parquet") or "_" not in name:
            raise PaperInputError("loader manifest name is invalid")
        family, partition = name[: -len(".parquet")].split("_", 1)
        key = (family, partition)
        if key in manifests:
            raise PaperInputError("loader manifests contain a duplicated family")
        manifests[key] = ArtifactIdentity(str(path), sha256, size_bytes)
    return manifests

def _declared_file(
    entry: object, source: Path, name: str, label: str
) -> ArtifactIdentity:
    if not isinstance(entry, Mapping):
        raise PaperInputError(f"{label} identity is missing")
    sha256 = entry.get("sha256")
    size_bytes = entry.get("size_bytes")
    if type(sha256) is not str or not sha256:
        raise PaperInputError(f"missing source hash for {label}")
    if type(size_bytes) is not int or size_bytes < 0:
        raise PaperInputError(f"missing source size for {label}")
    declared_path = entry.get("path")
    if declared_path is not None and Path(str(declared_path)).name != name:
        raise PaperInputError(f"{label} path does not match {name}")
    path = source / name
    _reject_blind(path, name)
    return ArtifactIdentity(str(path), sha256, size_bytes)

def _unique_examples(examples) -> dict[str, object]:
    by_id: dict[str, object] = {}
    for item in examples:
        if item.example_id in by_id:
            raise PaperInputError(f"duplicate inventory example {item.example_id!r}")
        by_id[item.example_id] = item
    return by_id

def _unique_parents(parents) -> dict[tuple[str, str, str], object]:
    by_key: dict[tuple[str, str, str], object] = {}
    for item in parents:
        key = (item.parent_id, item.family, item.partition)
        if key in by_key:
            raise PaperInputError(f"duplicate inventory parent {item.parent_id!r}")
        by_key[key] = item
    return by_key

def _example_rows(raw: bytes, family: str) -> tuple[dict[str, object], ...]:
    import pandas as pd

    try:
        frame = pd.read_parquet(io.BytesIO(raw))
    except Exception as error:
        raise PaperInputError(f"cannot parse {family} example parquet") from error
    if not _EXAMPLE_COLUMNS.issubset(frame.columns):
        raise PaperInputError(f"{family} example parquet schema changed")
    rows = []
    for item in frame.to_dict("records"):
        row_family = str(item["family"])
        if row_family != family:
            raise PaperInputError(
                f"{family} example parquet contains {row_family} rows"
            )
        rows.append(
            {
                "example_id": str(item["example_id"]),
                "parent_id": str(item["parent_id"]),
                "family": row_family,
                "partition": str(item["partition"]),
                "strict": bool(item["strict"]),
            }
        )
    return tuple(rows)

def _quarantine_example_ids(raw: bytes) -> frozenset[str]:
    import pandas as pd

    try:
        frame = pd.read_parquet(io.BytesIO(raw))
    except Exception as error:
        raise PaperInputError("cannot parse antibody quarantine parquet") from error
    if not _QUARANTINE_COLUMNS.issubset(frame.columns):
        raise PaperInputError("antibody quarantine parquet schema changed")
    rows = frame.to_dict("records")
    if len(rows) != _QUARANTINE_ROWS:
        raise PaperInputError("the frozen 101 antibody quarantine rows changed")
    counted = Counter(str(item["quarantine_reason"]) for item in rows)
    if counted != Counter(_QUARANTINE_REASONS):
        raise PaperInputError("the frozen 101 antibody quarantine rows changed")
    return frozenset(str(item["example_id"]) for item in rows)

def _ame_ligand_identity(
    family: str, example_id: str, resolved_smiles: Mapping[str, str] | None
) -> tuple[str | None, str | None, str | None]:
    if family != "ame":
        return None, None, None
    if not example_id.startswith("ame:"):
        raise PaperInputError(f"AME example {example_id!r} lacks ame: prefix")
    system_id = example_id[4:]
    parts = system_id.split("__")
    if len(parts) != 4 or not parts[-1]:
        raise PaperInputError(
            f"AME example {example_id!r} lacks its target ligand component"
        )
    token = parts[-1]
    if resolved_smiles is None:
        raise PaperInputError("AME resolved ligand identity is missing")
    smiles = resolved_smiles.get(system_id)
    if type(smiles) is not str or not smiles.strip():
        raise PaperInputError(
            f"AME canonical ligand identity/SMILES is missing for {example_id!r}"
        )
    return token, token, smiles

def _load_resolved_ligand_smiles() -> Mapping[str, str]:
    import pandas as pd

    path = _RESOLVED_LIGAND_PATH
    _reject_blind(path)
    raw = _read_regular_bytes(path, "resolved AME ligand identity")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _RESOLVED_LIGAND_SHA256 or len(raw) != _RESOLVED_LIGAND_SIZE_BYTES:
        raise PaperInputError("resolved AME ligand identity drifted")
    try:
        frame = pd.read_parquet(io.BytesIO(raw), columns=["system_id", "ligand_smiles"])
    except Exception as error:
        raise PaperInputError("cannot parse resolved AME ligand identity") from error
    index: dict[str, str] = {}
    for item in frame.to_dict("records"):
        system_id = str(item["system_id"])
        smiles = item["ligand_smiles"]
        if type(smiles) is not str or not smiles.strip():
            raise PaperInputError(
                f"resolved AME ligand identity/SMILES is missing for {system_id!r}"
            )
        if system_id in index:
            raise PaperInputError(f"duplicate resolved AME system {system_id!r}")
        index[system_id] = smiles
    return MappingProxyType(index)

def _counts_for(targets: list[PaperTargetRecord]) -> dict[str, dict[str, int]]:
    parents: dict[str, set[str]] = {family: set() for family in _FAMILIES}
    rows = {family: 0 for family in _FAMILIES}
    for item in targets:
        parents[item.family].add(item.parent_id)
        rows[item.family] += 1
    return {
        family: {
            "parent_groups": len(parents[family]),
            "example_rows": rows[family],
        }
        for family in _FAMILIES
    }

def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PaperInputError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not os.path.stat.S_ISREG(before.st_mode):
            raise PaperInputError(f"{label} must be a regular file")
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if before.st_dev != after.st_dev or before.st_ino != after.st_ino:
        raise PaperInputError(f"{label} identity drifted")
    return raw

def _call_view(fn, *args):
    try:
        return fn(*args)
    except BenchmarkViewError as error:
        raise PaperInputError(str(error)) from error

def _reject_blind(*values: object) -> None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        lowered = text.lower()
        if lowered in {"blind", "test-blind", "test_blind"}:
            raise PaperInputError("blind aliases are forbidden")
        if "test-blind" in lowered or "test_blind" in lowered:
            raise PaperInputError("blind aliases are forbidden")
        if (
            isinstance(value, Path)
            or "/" in text
            or text.endswith((".parquet", ".cif", ".pdb", ".gz", ".json"))
        ):
            if any("blind" in part.lower() for part in Path(text).parts):
                raise PaperInputError("blind aliases are forbidden")
