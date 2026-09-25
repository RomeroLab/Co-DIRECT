
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.paper_inputs import PaperInputBundle, PaperTargetRecord
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
)
from dive.training import preflight
from dive.training.preflight import canonical_json_bytes, write_create_new_json

if TYPE_CHECKING:
    from dive.benchmark.paper_registry import PaperEvaluatorRegistry

class PaperBaselineError(RuntimeError):
    pass

class PaperArm(StrEnum):
    BASE_COMMON_V2_SPLICE = "base_common_v2_splice"
    BASE_COMMON_NATIVE_V1 = "base_common_native_v1"
    BASE_COMMON_EXACT_V1 = "base_common_exact_v1"

    BASE_COMMON_FINETUNE_V1 = "base_common_finetune_v1"

PAPER_SEEDS: tuple[int, ...] = (42001, 42002, 42003, 42004)
AUTHENTICATED_BENCHMARK_READY_HEAD = "f6e63b2bd07f5441e076054172d26949dc484f15"
FIXTURE_EVALUATOR_REGISTRY_SHA256 = (
    "2706b77b41829f28f57bd0607c56dc7b900df7c25282d9da1d3f069dd0868df3"
)
PRODUCTION_EVALUATOR_REGISTRY_SHA256 = (
    "a53c687e16d85903ef8f9d20c7a80386d988278482004092dddebab8f1b45ca8"
)
PANEL_A_CHECKPOINT_SHA256 = (
    "589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9"
)
PANEL_A_AUTOENCODER_SHA256 = (
    "35f8865efd269995eeaf1670e1c1085acfe2988c40abdeda8e09a0e15eb40816"
)
PANEL_A_VIEW_SEMANTIC_HASH = (
    "f5e54f8e0d91fe857f3d36135fb603f921523fa16563eb83d2b099e34ac8d253"
)
PANEL_A_VIEW_PARTITION_HASH = (
    "527c359aefa234ea3015f645a8d128b587ad22c7e0b1aa263491c6a54a11fc03"
)
_PANEL_A_STEPS = {"binder": 400, "ame": 100, "antibody": 100}

SHIPPED_PROTOCOL_STEPS = 400
_FROZEN_MEMBERSHIP = {
    "binder": {"parent_groups": 2200, "example_rows": 3226},
    "ame": {"parent_groups": 1893, "example_rows": 4435},
    "antibody": {"parent_groups": 141, "example_rows": 281},
}
_FROZEN_FAMILY_CELLS = {"binder": 12904, "ame": 17740, "antibody": 1124}
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_SHA = frozenset("0123456789abcdef")
DIVE_ARM_NAMES = frozenset(
    {
        "lora_joint",
        "fixed_self",
        "fixed_backbone_led",
        "fixed_local_led",
        "fixed_joint",
        "continuous_fixed",
        "time_router",
        "sample_router",
        "residue_router",
        "task_id_upper_bound",
        "generic_adapter",
        "pass_matched",
        "shuffled_gates",
        "adaptive",
    }
)
_INVENTORY = {
    "base_common_exact_v1": "include_panel_a",
    "base_common_finetune_v1": "include_panel_a",

    "base_common_v2_splice": "legacy_panel_a_v2_splice",
    "base_common_native_v1": "include_panel_b",
    "complexa_ame.ckpt": "unknown_provenance",
    "complexa_ligand.ckpt": "unknown_provenance",
    "null_warmup": "unauthorized_checkpoint",
    "base": "pairing_fixture_not_executed",
    **{name: "dive_or_adaptive_arm" for name in DIVE_ARM_NAMES},
}

_PANEL_ARMS = MappingProxyType(
    {
        "A": PaperArm.BASE_COMMON_EXACT_V1,
        "B": PaperArm.BASE_COMMON_NATIVE_V1,
    }
)

def panel_arm(panel: str) -> PaperArm:

    try:
        return _PANEL_ARMS[panel]
    except KeyError:
        raise PaperBaselineError(f"unknown panel {panel!r}") from None

def family_steps(
    family: str,
    *,
    panel: str,
    arm: str,
    steps: int | None = None,
    shipped_protocol: bool = False,
) -> int:

    if panel != "A":
        raise PaperBaselineError(f"unsupported panel/arm {panel}/{arm}")

    if arm not in (
        panel_arm("A"),
        PaperArm.BASE_COMMON_FINETUNE_V1,
        PaperArm.BASE_COMMON_V2_SPLICE,
    ):
        raise PaperBaselineError(f"unsupported panel/arm {panel}/{arm}")
    if family not in _PANEL_A_STEPS:
        raise PaperBaselineError(f"unknown family {family}")
    frozen = _PANEL_A_STEPS[family]

    selected = max(frozen, SHIPPED_PROTOCOL_STEPS) if shipped_protocol else frozen
    if steps is not None and steps != selected:
        if family == "binder":
            raise PaperBaselineError("classified_protocol_mismatch")
        raise PaperBaselineError(f"steps {steps} != frozen {selected} for {family}")
    return selected

def inventory_disposition(candidate: str) -> str:
    try:
        return _INVENTORY[candidate]
    except KeyError as error:
        raise PaperBaselineError(f"unknown candidate {candidate}") from error

_SHA256 = frozenset("0123456789abcdef")
_BLIND_PARTITIONS = frozenset({"test-blind", "test_blind", "blind"})

@dataclass(frozen=True, slots=True)
class PaperTarget:
    parent_id: str
    family: str
    example_id: str
    partition: str = "validation"

@dataclass(frozen=True, slots=True)
class PaperCell:
    cell_id: str
    panel: str
    arm: str
    family: str
    partition: str
    parent_id: str
    example_id: str
    seed: int
    sample_index: int
    generation_replicas: int
    steps: int
    condition_count: int
    denoiser_equivalent_passes: int
    checkpoint: str
    splice_report_sha256: str
    evaluator_registry_sha256: str
    view_semantic_hash: str
    code_commit: str

def _require_sha256(label: str, value: str) -> str:
    if type(value) is not str or len(value) != 64 or set(value) - _SHA256:
        raise PaperBaselineError(f"{label} must be a lowercase sha256")
    return value

def paper_cell_identity(
    *,
    panel: str,
    arm: str,
    family: str,
    partition: str,
    parent_id: str,
    example_id: str,
    seed: int,
    sample_index: int,
    generation_replicas: int,
    steps: int,
    condition_count: int,
    denoiser_equivalent_passes: int,
    checkpoint: str,
    splice_report_sha256: str,
    evaluator_registry_sha256: str,
    view_semantic_hash: str,
    code_commit: str,
) -> str:
    payload: Mapping[str, object] = {
        "arm": arm,
        "checkpoint": checkpoint,
        "code_commit": code_commit,
        "condition_count": condition_count,
        "denoiser_equivalent_passes": denoiser_equivalent_passes,
        "evaluator_registry_sha256": evaluator_registry_sha256,
        "example_id": example_id,
        "family": family,
        "generation_replicas": generation_replicas,
        "panel": panel,
        "parent_id": parent_id,
        "partition": partition,
        "sample_index": sample_index,
        "seed": seed,
        "splice_report_sha256": splice_report_sha256,
        "steps": steps,
        "view_semantic_hash": view_semantic_hash,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def build_paper_baseline_plan(
    *,
    panel: str,
    targets: Sequence[PaperTarget],
    view_semantic_hash: str,
    checkpoint: str,
    splice_report_sha256: str,
    evaluator_registry_sha256: str,
    code_commit: str,
) -> tuple[PaperCell, ...]:
    if panel != "A":
        raise PaperBaselineError(f"unsupported panel {panel}")
    _require_sha256("view_semantic_hash", view_semantic_hash)
    _require_sha256("checkpoint", checkpoint)
    if not splice_report_sha256:
        raise PaperBaselineError("splice report hash is required")
    _require_sha256("splice_report_sha256", splice_report_sha256)
    _require_sha256("evaluator_registry_sha256", evaluator_registry_sha256)
    if evaluator_registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256:
        raise PaperBaselineError(
            "production evaluator hash must not reuse fixture registry"
        )
    if type(code_commit) is not str or len(code_commit) != 40:
        raise PaperBaselineError("code_commit must be a 40-character git SHA")
    cells: list[PaperCell] = []
    for target in targets:
        if target.partition != "validation" or target.partition in _BLIND_PARTITIONS:
            raise PaperBaselineError("blind partitions are forbidden")
        if any(
            marker in target.partition
            for marker in ("blind", "test-blind", "test_blind")
        ):
            raise PaperBaselineError("blind partitions are forbidden")
        steps = family_steps(
            target.family, panel=panel, arm=panel_arm(panel)
        )
        for seed in PAPER_SEEDS:
            identity = dict(
                panel=panel,
                arm=panel_arm(panel).value,
                family=target.family,
                partition=target.partition,
                parent_id=target.parent_id,
                example_id=target.example_id,
                seed=seed,
                sample_index=0,
                generation_replicas=1,
                steps=steps,
                condition_count=1,
                denoiser_equivalent_passes=1,
                checkpoint=checkpoint,
                splice_report_sha256=splice_report_sha256,
                evaluator_registry_sha256=evaluator_registry_sha256,
                view_semantic_hash=view_semantic_hash,
                code_commit=code_commit,
            )
            cells.append(PaperCell(cell_id=paper_cell_identity(**identity), **identity))
    return tuple(cells)

@dataclass(frozen=True, slots=True)
class PaperPlanManifest:
    panel: str
    arm: str
    code_commit: str
    view_semantic_hash: str
    view_partition_hash: str
    evaluator_registry_sha256: str
    checkpoint_sha256: str
    autoencoder_sha256: str
    splice_report_sha256: str
    sampling_contract: Mapping[str, object]
    parent_counts: Mapping[str, int]
    example_counts: Mapping[str, int]
    family_cells: Mapping[str, int]
    cells: tuple[PaperCell, ...]
    input_sources: tuple[ArtifactIdentity, ...]
    blind_opened: bool
    foldseek_executed: bool
    executed_dive_arms: tuple[str, ...]
    plan_artifact: ArtifactIdentity | None = None
    cells_artifact: ArtifactIdentity | None = None

def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout

def _clean_repo_commit(repo: Path | None = None) -> str:
    root = Path(repo) if repo is not None else _REPO_ROOT
    try:
        dirty = _git(root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as error:
        raise PaperBaselineError("cannot determine git state") from error
    if dirty.strip():
        raise PaperBaselineError("git tree is dirty")
    try:
        commit = _git(root, "rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PaperBaselineError("cannot determine git state") from error
    if type(commit) is not str or len(commit) != 40 or set(commit) - _GIT_SHA:
        raise PaperBaselineError("bad code commit")
    return commit

def _declared_counts(
    value: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    try:
        return {
            family: {
                "parent_groups": int(entry["parent_groups"]),
                "example_rows": int(entry["example_rows"]),
            }
            for family, entry in value.items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise PaperBaselineError("altered counts") from error

def _membership_counts(
    records: Sequence[PaperTargetRecord],
) -> dict[str, dict[str, int]]:
    parents: dict[str, set[str]] = {family: set() for family in _PANEL_A_STEPS}
    examples = {family: 0 for family in _PANEL_A_STEPS}
    for row in records:
        if row.family not in _PANEL_A_STEPS:
            raise PaperBaselineError(f"unknown family {row.family}")
        parents[row.family].add(row.parent_id)
        examples[row.family] += 1
    return {
        family: {
            "parent_groups": len(parents[family]),
            "example_rows": examples[family],
        }
        for family in _PANEL_A_STEPS
    }

def _cell_mapping(cell: PaperCell) -> dict[str, object]:
    return {
        "arm": cell.arm,
        "cell_id": cell.cell_id,
        "checkpoint": cell.checkpoint,
        "code_commit": cell.code_commit,
        "condition_count": cell.condition_count,
        "denoiser_equivalent_passes": cell.denoiser_equivalent_passes,
        "evaluator_registry_sha256": cell.evaluator_registry_sha256,
        "example_id": cell.example_id,
        "family": cell.family,
        "generation_replicas": cell.generation_replicas,
        "panel": cell.panel,
        "parent_id": cell.parent_id,
        "partition": cell.partition,
        "sample_index": cell.sample_index,
        "seed": cell.seed,
        "splice_report_sha256": cell.splice_report_sha256,
        "steps": cell.steps,
        "view_semantic_hash": cell.view_semantic_hash,
    }

def _write_create_new(path: Path, content: bytes) -> ArtifactIdentity:
    destination = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o664)
    except FileExistsError as error:
        raise PaperBaselineError(
            f"create-new plan artifact already exists: {destination}"
        ) from error
    except OSError as error:
        raise PaperBaselineError(
            f"cannot write create-new plan artifact: {error}"
        ) from error
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PaperBaselineError("create-new write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(content).hexdigest()
    return ArtifactIdentity(str(destination), digest, len(content))

def materialize_panel_a_plan(
    *,
    inputs: PaperInputBundle,
    registry: PaperEvaluatorRegistry,
    checkpoint_sha256: str,
    autoencoder_sha256: str,
    splice_report_sha256: str,
    code_commit: str,
    output_dir: Path | None = None,
) -> PaperPlanManifest:
    from dive.benchmark.paper_registry import (
        PaperEvaluatorRegistry as RegistryType,
        PaperRegistryError,
        verify_paper_evaluator_registry,
    )

    if not isinstance(inputs, PaperInputBundle):
        raise PaperBaselineError("materialize_panel_a_plan requires a PaperInputBundle")
    if not isinstance(registry, RegistryType):
        raise PaperBaselineError(
            "materialize_panel_a_plan requires a verified production registry"
        )
    try:
        verify_paper_evaluator_registry(registry)
    except PaperRegistryError as error:
        raise PaperBaselineError(str(error)) from error
    if registry.semantic_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256:
        raise PaperBaselineError(
            "production evaluator hash must not reuse fixture registry"
        )
    if registry.semantic_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256:
        raise PaperBaselineError("production evaluator registry identity drifted")

    if (
        type(code_commit) is not str
        or len(code_commit) != 40
        or set(code_commit) - _GIT_SHA
    ):
        raise PaperBaselineError("bad code commit")
    observed_commit = _clean_repo_commit()
    if code_commit != observed_commit:
        raise PaperBaselineError("bad code commit")

    if not splice_report_sha256:
        raise PaperBaselineError("splice report hash is required")
    _require_sha256("splice_report_sha256", splice_report_sha256)
    _require_sha256("checkpoint_sha256", checkpoint_sha256)
    _require_sha256("autoencoder_sha256", autoencoder_sha256)
    common = registry.checkpoints.get("common")
    autoencoder = registry.checkpoints.get("autoencoder")
    if (
        common is None
        or checkpoint_sha256 != PANEL_A_CHECKPOINT_SHA256
        or checkpoint_sha256 != common.sha256
    ):
        raise PaperBaselineError("checkpoint identity drifted")
    if (
        autoencoder is None
        or autoencoder_sha256 != PANEL_A_AUTOENCODER_SHA256
        or autoencoder_sha256 != autoencoder.sha256
    ):
        raise PaperBaselineError("autoencoder identity drifted")

    if (
        inputs.semantic_hash != PANEL_A_VIEW_SEMANTIC_HASH
        or inputs.partition_hash != PANEL_A_VIEW_PARTITION_HASH
    ):
        raise PaperBaselineError("view hash drifted")

    declared = _declared_counts(inputs.counts)
    observed = _membership_counts(inputs.targets)
    if declared != _FROZEN_MEMBERSHIP or observed != _FROZEN_MEMBERSHIP:
        raise PaperBaselineError("altered counts")

    example_ids = [row.example_id for row in inputs.targets]
    if len(example_ids) != len(set(example_ids)):
        raise PaperBaselineError("duplicate cells")

    for row in inputs.targets:
        if not row.strict:
            raise PaperBaselineError("paper inputs retain strict validation only")
        if row.partition != "validation" or row.partition in _BLIND_PARTITIONS:
            raise PaperBaselineError("blind partitions are forbidden")
        if any(
            marker in row.partition for marker in ("blind", "test-blind", "test_blind")
        ):
            raise PaperBaselineError("blind partitions are forbidden")

    if tuple(PAPER_SEEDS) != (42001, 42002, 42003, 42004):
        raise PaperBaselineError("each target must expand to exactly four seeds")

    cells = build_paper_baseline_plan(
        panel="A",
        targets=tuple(
            PaperTarget(row.parent_id, row.family, row.example_id, row.partition)
            for row in inputs.targets
        ),
        view_semantic_hash=inputs.semantic_hash,
        checkpoint=checkpoint_sha256,
        splice_report_sha256=splice_report_sha256,
        evaluator_registry_sha256=registry.semantic_sha256,
        code_commit=code_commit,
    )
    cells = tuple(
        sorted(
            cells,
            key=lambda cell: (cell.family, cell.parent_id, cell.example_id, cell.seed),
        )
    )
    cell_ids = [cell.cell_id for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise PaperBaselineError("duplicate cells")

    seeds_by_target: dict[tuple[str, str, str], set[int]] = {}
    family_cells = {family: 0 for family in _PANEL_A_STEPS}
    for cell in cells:
        if cell.arm != panel_arm("A"):
            raise PaperBaselineError(f"panel A requires {panel_arm('A').value}")
        if cell.arm in DIVE_ARM_NAMES or cell.arm == "base":
            raise PaperBaselineError(f"arm {cell.arm} is excluded")
        if cell.panel != "A" or cell.partition != "validation":
            raise PaperBaselineError("blind partitions are forbidden")
        if cell.sample_index != 0 or cell.generation_replicas != 1:
            raise PaperBaselineError("sampling contract drifted")
        if cell.condition_count != 1 or cell.denoiser_equivalent_passes != 1:
            raise PaperBaselineError("sampling contract drifted")
        expected_steps = _PANEL_A_STEPS[cell.family]
        if cell.steps != expected_steps:
            if cell.family == "binder":
                raise PaperBaselineError("classified_protocol_mismatch")
            raise PaperBaselineError(
                f"steps {cell.steps} != frozen {expected_steps} for {cell.family}"
            )
        key = (cell.family, cell.parent_id, cell.example_id)
        seeds_by_target.setdefault(key, set()).add(cell.seed)
        family_cells[cell.family] += 1

    expected_targets = {
        (row.family, row.parent_id, row.example_id) for row in inputs.targets
    }
    if set(seeds_by_target) != expected_targets:
        raise PaperBaselineError("each target must expand to exactly four seeds")
    frozen_seeds = {42001, 42002, 42003, 42004}
    if any(seeds != frozen_seeds for seeds in seeds_by_target.values()):
        raise PaperBaselineError("each target must expand to exactly four seeds")
    if family_cells != _FROZEN_FAMILY_CELLS or len(cells) != 31768:
        raise PaperBaselineError("altered counts")

    sampling = MappingProxyType(
        {
            "seeds": PAPER_SEEDS,
            "sample_index": 0,
            "generation_replicas": 1,
            "guidance": 1.0,
            "model_recycles": 0,
            "family_steps": MappingProxyType(dict(_PANEL_A_STEPS)),
        }
    )
    parent_counts = MappingProxyType(
        {family: values["parent_groups"] for family, values in observed.items()}
    )
    example_counts = MappingProxyType(
        {family: values["example_rows"] for family, values in observed.items()}
    )
    family_cell_counts = MappingProxyType(dict(family_cells))

    plan_artifact = None
    cells_artifact = None
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        cells_lines = []
        for cell in cells:
            cells_lines.append(
                json.dumps(
                    _cell_mapping(cell),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        cells_payload = ("\n".join(cells_lines) + "\n").encode("utf-8")
        cells_artifact = _write_create_new(destination / "cells.jsonl", cells_payload)
        plan_payload = {
            "arm": panel_arm("A").value,
            "autoencoder_sha256": autoencoder_sha256,
            "blind_opened": False,
            "cells_sha256": cells_artifact.sha256,
            "cells_size_bytes": cells_artifact.size_bytes,
            "checkpoint_sha256": checkpoint_sha256,
            "code_commit": code_commit,
            "evaluator_registry_sha256": registry.semantic_sha256,
            "example_counts": dict(example_counts),
            "executed_dive_arms": [],
            "family_cells": dict(family_cell_counts),
            "foldseek_executed": False,
            "input_sources": [
                {
                    "path": item.path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in inputs.sources
            ],
            "n_cells": len(cells),
            "panel": "A",
            "parent_counts": dict(parent_counts),
            "sampling_contract": {
                "family_steps": dict(_PANEL_A_STEPS),
                "generation_replicas": 1,
                "guidance": 1.0,
                "model_recycles": 0,
                "sample_index": 0,
                "seeds": [42001, 42002, 42003, 42004],
            },
            "schema_version": "dive-paper-plan-manifest-v1",
            "splice_report_sha256": splice_report_sha256,
            "view_partition_hash": inputs.partition_hash,
            "view_semantic_hash": inputs.semantic_hash,
        }
        plan_artifact = _write_create_new(
            destination / "plan.json", canonical_json_bytes(plan_payload)
        )

    return PaperPlanManifest(
        panel="A",
        arm=panel_arm("A").value,
        code_commit=code_commit,
        view_semantic_hash=inputs.semantic_hash,
        view_partition_hash=inputs.partition_hash,
        evaluator_registry_sha256=registry.semantic_sha256,
        checkpoint_sha256=checkpoint_sha256,
        autoencoder_sha256=autoencoder_sha256,
        splice_report_sha256=splice_report_sha256,
        sampling_contract=sampling,
        parent_counts=parent_counts,
        example_counts=example_counts,
        family_cells=family_cell_counts,
        cells=cells,
        input_sources=inputs.sources,
        blind_opened=False,
        foldseek_executed=False,
        executed_dive_arms=(),
        plan_artifact=plan_artifact,
        cells_artifact=cells_artifact,
    )

_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_TEST_ROOTS: tuple[Path, Path] | None = None

@dataclass(frozen=True, slots=True)
class PaperRun:
    run_id: str
    evidence_dir: Path
    bulk_dir: Path
    attempt: ArtifactIdentity

def claim_paper_baseline_run(
    run_id: str,
    argv: Sequence[str],
    *,
    evidence_root: Path,
    bulk_root: Path,
    repo_commit: str,
) -> PaperRun:
    if (
        type(run_id) is not str
        or _RUN_ID.fullmatch(run_id) is None
        or not (
            run_id.startswith("paper-baseline-") or run_id.startswith("paper-smoke-")
        )
    ):
        raise PaperBaselineError(f"invalid paper baseline run id {run_id}")
    evidence_root = Path(evidence_root)
    bulk_root = Path(bulk_root)
    expected = _TEST_ROOTS or (EMERGENT_EVIDENCE_ROOT, EMERGENT_BULK_ROOT)
    if evidence_root != expected[0] or bulk_root != expected[1]:
        raise PaperBaselineError(
            "evidence or bulk root is denied by the fixed production contract"
        )
    if _TEST_ROOTS is None:
        for root in (evidence_root, bulk_root):
            if any(
                root == denied or denied in root.parents for denied in DENIED_PREFIXES
            ):
                raise PaperBaselineError(f"root is denied: {root}")
    evidence_dir = evidence_root / "paper_quality_baseline" / run_id
    bulk_dir = bulk_root / "paper_quality_baseline" / run_id
    evidence_dir.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
    try:
        os.mkdir(evidence_dir, 0o2775)
    except FileExistsError:
        raise
    except OSError as error:
        raise PaperBaselineError(
            f"cannot claim paper evidence directory: {error}"
        ) from error
    try:
        bulk_dir.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
        os.mkdir(bulk_dir, 0o2775)
    except FileExistsError as error:
        raise PaperBaselineError("paper bulk run directory already exists") from error
    payload = {
        "schema_version": "dive-paper-baseline-attempt-v1",
        "status": "BUILDING",
        "run_id": run_id,
        "argv": list(argv),
        "argv_shell": shlex.join(tuple(argv)),
        "repo_commit": repo_commit,
        "evidence_path": str(evidence_dir),
        "bulk_path": str(bulk_dir),
        "evidence_prefix": "paper_quality_baseline",
    }
    attempt_path = evidence_dir / "attempt.json"
    try:
        with _preflight_roots(evidence_root, bulk_root):
            write_create_new_json(attempt_path, payload)
    except preflight.PreflightError as error:
        raise PaperBaselineError(str(error)) from error
    raw = attempt_path.read_bytes()
    return PaperRun(
        run_id=run_id,
        evidence_dir=evidence_dir,
        bulk_dir=bulk_dir,
        attempt=ArtifactIdentity(
            str(attempt_path), hashlib.sha256(raw).hexdigest(), len(raw)
        ),
    )

def complete_paper_baseline_run(
    run: PaperRun, payload: Mapping[str, object]
) -> ArtifactIdentity:
    if not isinstance(run, PaperRun):
        raise PaperBaselineError("complete_paper_baseline_run requires a PaperRun")
    if not isinstance(payload, Mapping):
        raise PaperBaselineError("completion payload must be a mapping")
    evidence_root = run.evidence_dir.parent.parent
    bulk_root = run.bulk_dir.parent.parent
    completion = {
        "schema_version": "dive-paper-baseline-completion-v1",
        "status": "COMPLETE",
        "run_id": run.run_id,
        "attempt": {
            "path": run.attempt.path,
            "sha256": run.attempt.sha256,
            "size_bytes": run.attempt.size_bytes,
        },
        "payload": dict(payload),
    }
    destination = run.evidence_dir / "completion.json"
    try:
        with _preflight_roots(evidence_root, bulk_root):
            write_create_new_json(destination, completion)
    except preflight.PreflightError as error:
        raise PaperBaselineError(str(error)) from error
    raw = destination.read_bytes()
    return ArtifactIdentity(str(destination), hashlib.sha256(raw).hexdigest(), len(raw))

@contextmanager
def _preflight_roots(evidence_root: Path, bulk_root: Path):
    original = (
        preflight._TERMINAL_EVIDENCE_ROOT,
        preflight._TERMINAL_BULK_ROOT,
        preflight._TERMINAL_DENIED_PREFIXES,
    )
    preflight._TERMINAL_EVIDENCE_ROOT = evidence_root
    preflight._TERMINAL_BULK_ROOT = bulk_root
    if _TEST_ROOTS is not None:
        preflight._TERMINAL_DENIED_PREFIXES = ()
    try:
        yield
    finally:
        (
            preflight._TERMINAL_EVIDENCE_ROOT,
            preflight._TERMINAL_BULK_ROOT,
            preflight._TERMINAL_DENIED_PREFIXES,
        ) = original

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260828

@dataclass(frozen=True, slots=True)
class PaperObservation:
    parent_id: str
    example_id: str
    seed: int
    family: str
    primary: float
    completed: bool
    failure_reason: str | None

@dataclass(frozen=True, slots=True)
class FamilyHeadline:
    family: str
    n_parents: int
    n_cells: int
    n_successes: int
    cell_rate: float
    parent_balanced: float
    ci_low: float
    ci_high: float
    n_failed: int

def parent_bootstrap_ci(
    parent_values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if not parent_values:
        raise PaperBaselineError("cannot bootstrap an empty parent sample")
    rng = random.Random(seed)
    n = len(parent_values)
    means = sorted(
        math.fsum(parent_values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    return means[int(0.025 * resamples)], means[
        min(int(0.975 * resamples), resamples - 1)
    ]

def nested_parent_means(
    seed_values: Sequence[tuple[str, str, float]],
) -> tuple[float, ...]:

    if not seed_values:
        raise PaperBaselineError("cannot nest an empty parent sample")
    example_groups: dict[tuple[str, str], list[float]] = {}
    example_order: list[tuple[str, str]] = []
    parent_order: list[str] = []
    seen_parents: set[str] = set()
    for parent_id, example_id, value in seed_values:
        key = (parent_id, example_id)
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_order.append(parent_id)
        if key not in example_groups:
            example_groups[key] = []
            example_order.append(key)
        example_groups[key].append(value)
    example_means: dict[str, list[float]] = {
        parent_id: [] for parent_id in parent_order
    }
    for parent_id, example_id in example_order:
        group = example_groups[(parent_id, example_id)]
        example_means[parent_id].append(math.fsum(group) / len(group))
    return tuple(math.fsum(group) / len(group) for group in example_means.values())

def summarize_family(observations: Sequence[PaperObservation]) -> FamilyHeadline:
    if not observations:
        raise PaperBaselineError("no observations")
    families = {item.family for item in observations}
    if len(families) != 1:
        raise PaperBaselineError("summarize_family requires one family")
    family = next(iter(families))
    n_cells = len(observations)
    n_failed = sum(1 for item in observations if not item.completed)
    seed_values: list[tuple[str, str, float]] = []
    n_successes = 0
    for item in observations:
        value = item.primary if item.completed else 0.0
        if value != 0.0:
            n_successes += 1
        seed_values.append((item.parent_id, item.example_id, value))
    cell_rate = n_successes / n_cells
    parent_means = nested_parent_means(tuple(seed_values))
    parent_balanced = math.fsum(parent_means) / len(parent_means)
    ci_low, ci_high = parent_bootstrap_ci(parent_means)
    return FamilyHeadline(
        family=family,
        n_parents=len(parent_means),
        n_cells=n_cells,
        n_successes=n_successes,
        cell_rate=cell_rate,
        parent_balanced=parent_balanced,
        ci_low=ci_low,
        ci_high=ci_high,
        n_failed=n_failed,
    )

def production_registry_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

def assert_production_registry(hash_: str) -> None:
    _require_sha256("evaluator_registry_sha256", hash_)
    if hash_ == FIXTURE_EVALUATOR_REGISTRY_SHA256:
        raise PaperBaselineError(
            "production evaluator hash must not reuse fixture registry"
        )

def _cli_materialize(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="evaluate_paper_baseline plan")
    parser.add_argument("--view-run", type=Path, required=True)
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/emergent/paper_baseline_evaluators.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splice-report-sha256", required=True)
    parser.add_argument("--checkpoint-sha256", default=PANEL_A_CHECKPOINT_SHA256)
    parser.add_argument("--autoencoder-sha256", default=PANEL_A_AUTOENCODER_SHA256)
    parser.add_argument("--code-commit")
    parser.add_argument("--foldseek", action="store_true")
    args = parser.parse_args(list(argv))
    if args.foldseek:
        raise PaperBaselineError("Foldseek is forbidden")
    output_dir = Path(args.output_dir).resolve(strict=False)
    for denied in DENIED_PREFIXES:
        prefix = Path(denied).resolve(strict=False)
        if output_dir == prefix or prefix in output_dir.parents:
            raise PaperBaselineError(f"root is denied: {output_dir}")
    from dive.benchmark.paper_inputs import load_paper_inputs
    from dive.benchmark.paper_registry import load_paper_evaluator_registry

    commit = args.code_commit or _clean_repo_commit()
    bundle = load_paper_inputs(view_run=args.view_run, input_run=args.input_run)
    loaded = load_paper_evaluator_registry(args.registry)
    materialize_panel_a_plan(
        inputs=bundle,
        registry=loaded,
        checkpoint_sha256=args.checkpoint_sha256,
        autoencoder_sha256=args.autoencoder_sha256,
        splice_report_sha256=args.splice_report_sha256,
        code_commit=commit,
        output_dir=output_dir,
    )
    return 0

def cli_main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    command = None
    if argv_list:
        command = argv_list[0] if argv_list else None
    elif len(sys.argv) > 1:
        command = sys.argv[1]
    if command in {"plan", "materialize"}:
        rest = argv_list[1:] if argv_list is not None else sys.argv[2:]
        from dive.benchmark.paper_inputs import PaperInputError
        from dive.benchmark.paper_registry import PaperRegistryError

        try:
            return _cli_materialize(rest)
        except (PaperBaselineError, PaperInputError, PaperRegistryError):
            return 2
    parser = argparse.ArgumentParser(prog="evaluate_paper_baseline")
    parser.add_argument("--panel", default="A")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--family")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--evaluator-registry-sha256")
    parser.add_argument("--foldseek", action="store_true")
    args = parser.parse_args(argv_list)
    try:
        if args.foldseek:
            raise PaperBaselineError("Foldseek is forbidden")
        if args.partition != "validation" or "blind" in args.partition:
            raise PaperBaselineError("blind partitions are forbidden")
        if args.arm in DIVE_ARM_NAMES or args.arm == "base":
            raise PaperBaselineError(f"arm {args.arm} is excluded")
        if args.panel == "A" and args.arm != panel_arm("A"):
            raise PaperBaselineError(f"panel A requires {panel_arm('A').value}")
        if args.panel == "B" and args.arm != PaperArm.BASE_COMMON_NATIVE_V1:
            raise PaperBaselineError("panel B requires base_common_native_v1")
        if args.family:
            family_steps(
                args.family,
                panel=args.panel,
                arm=args.arm,
                steps=args.steps,
            )
        elif args.panel == "A" and args.steps == 100 and args.family is None:
            raise PaperBaselineError("classified_protocol_mismatch")
        if args.evaluator_registry_sha256:
            assert_production_registry(args.evaluator_registry_sha256)
    except PaperBaselineError:
        return 2
    return 0

def rebuild_paper_report(payload: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(payload)

def render_paper_report(headlines: Sequence[FamilyHeadline], *, panel: str) -> str:
    arm = panel_arm(panel).value
    lines = [
        f"# Paper baseline panel {panel}",
        f"arm: {arm}",
        "headline: parent-balanced estimate with parent-bootstrap CI",
        "auxiliary: cell rate",
    ]
    for item in headlines:
        lines.extend(
            [
                f"## {item.family}",
                f"parent-balanced: {item.parent_balanced}",
                f"cell rate: {item.cell_rate}",
                f"n_parents: {item.n_parents}",
                f"n_cells: {item.n_cells}",
                f"n_successes: {item.n_successes}",
                f"n_failed: {item.n_failed}",
                f"ci: [{item.ci_low}, {item.ci_high}]",
            ]
        )
    return "\n".join(lines) + "\n"

def panel_b_citation_allowed(reconstruction: Mapping[str, object]) -> bool:
    required = {
        "plan_cells",
        "raw_pdbs",
        "constituents",
        "replica_accounting",
        "rebuilt_summary_sha256",
        "n_cells",
    }
    if required - reconstruction.keys():
        return False
    return (
        reconstruction.get("replica_accounting") == "64_generated_replica_pdbs"
        and reconstruction.get("n_cells") == 64
        and reconstruction.get("plan_cells") == 64
        and reconstruction.get("raw_pdbs") == 64
        and reconstruction.get("constituents") is True
    )
