
from __future__ import annotations

from dive.codirect_paths import ROOT

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dive.benchmark.contracts import load_benchmark_contract
from dive.benchmark.evidence import (
    claim_benchmark_run,
    complete_benchmark_run,
    write_terminal_bytes,
)
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_EVIDENCE_ROOT,
    RUNTIME_PYTHON,
)
from dive.training.preflight import canonical_json_bytes

class VerdictError(RuntimeError):
    pass

WAIVED_STRUCTURAL_AUDIT = "not_required_by_frozen_contract"
REQUIRED_AND_COMPLETE = "required_and_complete"
BINDER_PANEL_NOT_REQUIRED = "not_required_by_frozen_contract"
SMOKE_COMPLETE = "complete"
SMOKE_NOT_RUN = "not_run"
_SCHEMA_VERSION = "dive-benchmark-readiness-verdict-v1"
_FAMILIES = ("binder", "ame", "antibody")
_MIN_PARENT_GROUPS = 20
_REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_WORKTREE = Path(ROOT)
BASELINE_COMMIT = "4f6af1f72d53d6d8e9f8128f83c98213bb5b4f28"
PUBLICATION_ALLOWLIST = (
    "BENCHMARK_READY",
    "docs/results/emergent/BENCHMARK_READY.json",
    "docs/results/emergent/BENCHMARK_READY.md",
    "HANDOFF.md",
)
_FORBIDDEN_BLIND_NAMES = frozenset(
    {
        "build_blind_loader",
        "build_blind_inventory",
        "unseal",
        "unseal_blind",
        "drive_confirmation_in_realm",
        "execute_confirmation_in_realm",
        "execute_generation_arm",
        "replay_confirmation_in_realm",
    }
)
_READINESS_SOURCES = (
    Path("src/dive/benchmark/verdict.py"),
    Path("scripts/emergent/publish_benchmark_ready.py"),
    Path("scripts/emergent/run_benchmark_test_delta.py"),
)
_VIEW_SEMANTIC_HASH = "f5e54f8e0d91fe857f3d36135fb603f921523fa16563eb83d2b099e34ac8d253"
_VIEW_PARTITION_HASH = (
    "527c359aefa234ea3015f645a8d128b587ad22c7e0b1aa263491c6a54a11fc03"
)
_TEST_DELTA_COMPLETE = "complete"
_TEST_DELTA_MISSING = "not_run"
_SCAN_CLEAN = "clean"
_SCAN_NOT_RUN = "not_run"
_FORBIDDEN_TREE_TOKENS = (
    "build_blind_loader",
    "unseal_blind",
    "BlindEvaluationData",
)
_FORBIDDEN_JSON_KEYS = frozenset({"blind_output", "blind_outputs", "unsealed_blind"})
_VERDICT_ARTIFACT_NAMES = frozenset(
    {
        "blocked-verdict.json",
        "post-publication-authentication.json",
    }
)
_VERDICT_SCHEMAS = frozenset(
    {
        "dive-benchmark-readiness-verdict-v1",
        "dive-benchmark-post-publication-v1",
    }
)
_ACCEPTED_VIEW_RUN = "benchmark-views-20260828a"
_TEST_DELTA_SCHEMA = "dive-benchmark-test-delta-v1"
_GENERIC_COMPLETION_SCHEMA = "dive-benchmark-completion-v1"
_SMOKE_COMPLETION_SCHEMA = "dive-benchmark-smoke-completion-v1"
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_PYTEST_SUMMARY = re.compile(
    r"(?:(\d+) failed,?\s*)?(?:(\d+) errors?,?\s*)?(?:(\d+) passed,?\s*)?"
    r"(?:(\d+) skipped,?\s*)?(?:(\d+) xfailed,?\s*)?(?:(\d+) xpassed,?\s*)?"
    r"(?:(\d+) warnings?,?\s*)?in\s+[0-9.]+s"
)
_PYTEST_COMMAND = (
    str(RUNTIME_PYTHON),
    "-m",
    "pytest",
    "tests/",
    "-q",
    "--tb=no",
    "-ra",
)
_SMOKE_REQUIRED_FAMILIES = frozenset(_FAMILIES)
_SMOKE_REQUIRED_PRIMARIES = {
    "binder": "target_conditioned_success",
    "ame": "ame_motif_ligand_success",
    "antibody": "cdr_h3_ca_rmsd",
}
FOCUSED_PYTEST_PATHS = (
    "tests/benchmark",
    "tests/data/test_chain_assignment_contract.py",
    "tests/data/test_versioned_run_isolation.py",
    "tests/training/test_parent_projection.py",
    "tests/evaluation/test_directional_seal.py",
)

def _default_views() -> dict[str, dict[str, dict[str, int]]]:
    return {
        "binder": {
            "strict_blind": {"example_rows": 187, "parent_groups": 131},
            "strict_validation": {"example_rows": 2501, "parent_groups": 2200},
        },
        "ame": {
            "strict_blind": {"example_rows": 175, "parent_groups": 120},
            "strict_validation": {"example_rows": 2100, "parent_groups": 1893},
        },
        "antibody": {
            "strict_blind": {"example_rows": 510, "parent_groups": 454},
            "strict_validation": {"example_rows": 188, "parent_groups": 141},
        },
    }

@dataclass
class ReadyEvidence:

    train_data_authenticated: bool = False
    projection_v2_authenticated: bool = False
    v1_unchanged: bool = False
    track_b_unchanged: bool = False
    structural_audit: str = WAIVED_STRUCTURAL_AUDIT
    foldseek_complete: bool = False
    mapping_complete: bool = True
    unmapped_queries: tuple[str, ...] = ()
    thresholds_unchanged: bool = True
    unknown_reported_clear: bool = False
    exposure_all_unknown: bool = True
    view_hash_matches: bool = False
    view_semantic_hash: str = _VIEW_SEMANTIC_HASH
    view_partition_hash: str = _VIEW_PARTITION_HASH
    views: dict[str, dict[str, dict[str, int]]] = field(default_factory=_default_views)
    min_parent_groups: int = _MIN_PARENT_GROUPS
    evaluators_available: bool = False
    evaluator_registry_authenticated: bool = False
    required_metric_unavailable: bool = False
    smoke_status: str = SMOKE_NOT_RUN
    arm_pairing_ok: bool = False
    blind_output_seen: bool = False
    cli_imports_blind_loader: bool = False
    binder_panel: str = BINDER_PANEL_NOT_REQUIRED
    new_failures: tuple[str, ...] = ()
    focused_tests_pass: bool = False
    worktree_clean: bool = False
    test_delta_status: str = _TEST_DELTA_MISSING
    run_tree_scan_status: str = _SCAN_NOT_RUN
    run_tree_root: Path | None = None
    run_tree_findings: tuple[str, ...] = ()
    view_run_id: str = _ACCEPTED_VIEW_RUN

    def apply(self, mutation: str) -> None:

        if mutation == "train_data_auth_failed":
            self.train_data_authenticated = False
        elif mutation == "foldseek_incomplete":
            self.structural_audit = "required"
            self.foldseek_complete = False
        elif mutation == "unmapped_structure":
            self.structural_audit = REQUIRED_AND_COMPLETE
            self.foldseek_complete = True
            self.mapping_complete = False
            self.unmapped_queries = ("parent-unmapped",)
        elif mutation == "unknown_reported_clear":
            self.unknown_reported_clear = True
        elif mutation == "view_hash_drift":
            self.view_hash_matches = False
        elif mutation == "underpowered_family":
            self.views["binder"]["strict_blind"]["parent_groups"] = 19
        elif mutation == "required_metric_unavailable":
            self.evaluators_available = False
            self.required_metric_unavailable = True
        elif mutation == "arm_pairing_failed":
            self.arm_pairing_ok = False
        elif mutation == "blind_output_seen":
            self.blind_output_seen = True
        elif mutation == "new_failure":
            self.new_failures = ("tests/x.py::test_new",)
        elif mutation == "v1_changed":
            self.v1_unchanged = False
        else:
            raise VerdictError(f"unknown readiness mutation: {mutation}")

    def set_all_base_exposure_unknown(self) -> None:

        self.exposure_all_unknown = True
        self.unknown_reported_clear = False
        for family_views in self.views.values():
            family_views["temporal_clean"] = {"example_rows": 0, "parent_groups": 0}

    @classmethod
    def synthetic_pass(cls) -> ReadyEvidence:

        evidence = cls()
        evidence.train_data_authenticated = True
        evidence.projection_v2_authenticated = True
        evidence.v1_unchanged = True
        evidence.track_b_unchanged = True
        evidence.structural_audit = WAIVED_STRUCTURAL_AUDIT
        evidence.foldseek_complete = False
        evidence.mapping_complete = True
        evidence.thresholds_unchanged = True
        evidence.view_hash_matches = True
        evidence.evaluators_available = True
        evidence.evaluator_registry_authenticated = True
        evidence.required_metric_unavailable = False
        evidence.smoke_status = SMOKE_COMPLETE
        evidence.arm_pairing_ok = True
        evidence.blind_output_seen = False
        evidence.cli_imports_blind_loader = False
        evidence.binder_panel = BINDER_PANEL_NOT_REQUIRED
        evidence.new_failures = ()
        evidence.focused_tests_pass = True
        evidence.worktree_clean = True
        evidence.test_delta_status = _TEST_DELTA_COMPLETE
        evidence.run_tree_scan_status = _SCAN_CLEAN
        evidence.run_tree_root = None
        evidence.set_all_base_exposure_unknown()
        return evidence

    @classmethod
    def production_snapshot(cls) -> ReadyEvidence:

        evidence = cls()
        evidence.structural_audit = WAIVED_STRUCTURAL_AUDIT
        evidence.foldseek_complete = False
        evidence.mapping_complete = False
        evidence.thresholds_unchanged = False
        evidence.binder_panel = BINDER_PANEL_NOT_REQUIRED
        evidence.view_run_id = _ACCEPTED_VIEW_RUN
        evidence.smoke_status = SMOKE_NOT_RUN
        evidence.test_delta_status = _TEST_DELTA_MISSING
        evidence.run_tree_scan_status = _SCAN_NOT_RUN
        evidence.views = {
            family: {
                "strict_blind": {"example_rows": 0, "parent_groups": 0},
                "strict_validation": {"example_rows": 0, "parent_groups": 0},
                "temporal_clean": {"example_rows": 0, "parent_groups": 0},
            }
            for family in _FAMILIES
        }
        evidence.set_all_base_exposure_unknown()
        return evidence

    def __post_init__(self) -> None:
        self.views = copy.deepcopy(self.views)

@dataclass(frozen=True, slots=True)
class CriterionResult:
    index: int
    name: str
    status: str
    blocker_code: str | None
    detail: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "status": self.status,
            "blocker_code": self.blocker_code,
            "detail": self.detail,
        }

@dataclass(frozen=True, slots=True)
class AuthenticatedTestDelta:
    new_failures: tuple[str, ...]
    focused_tests_pass: bool

@dataclass(frozen=True, slots=True)
class AuthenticatedSmoke:
    evaluators_available: bool

@dataclass(frozen=True, slots=True)
class TestDelta:
    new_failures: tuple[str, ...]
    fixed_failures: tuple[str, ...]
    pre_existing_failures: tuple[str, ...]
    changed_skips: tuple[str, ...] = ()
    changed_xfails: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "new_failures": list(self.new_failures),
            "fixed_failures": list(self.fixed_failures),
            "pre_existing_failures": list(self.pre_existing_failures),
            "changed_skips": list(self.changed_skips),
            "changed_xfails": list(self.changed_xfails),
        }

@dataclass(frozen=True, slots=True)
class BenchmarkReadinessVerdict:
    status: str
    criteria: tuple[CriterionResult, ...]
    blocker_codes: tuple[str, ...]
    claims: Mapping[str, str]
    views: Mapping[str, Mapping[str, Mapping[str, int]]]
    hashes: Mapping[str, str]
    limitations: tuple[str, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "status": self.status,
            "verdict": self.status,
            "blocker_codes": list(self.blocker_codes),
            "claims": dict(self.claims),
            "views": {
                family: {
                    name: {
                        "example_rows": cell["example_rows"],
                        "parent_groups": cell["parent_groups"],
                    }
                    for name, cell in family_views.items()
                }
                for family, family_views in self.views.items()
            },
            "hashes": dict(self.hashes),
            "limitations": list(self.limitations),
            "criteria": [item.to_mapping() for item in self.criteria],
        }

def normalize_pytest_node(line: str) -> str:

    if type(line) is not str:
        raise VerdictError("pytest node line must be a string")
    text = line.strip()
    for prefix in ("FAILED ", "SKIPPED ", "XFAIL ", "XPASS "):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    node, separator, _rest = text.partition(" - ")
    return node.strip() if separator else text.strip()

def classify_test_delta(
    base: Sequence[str],
    branch: Sequence[str],
    *,
    base_skips: Sequence[str] = (),
    branch_skips: Sequence[str] = (),
    base_xfails: Sequence[str] = (),
    branch_xfails: Sequence[str] = (),
) -> TestDelta:

    base_nodes = {normalize_pytest_node(line) for line in base if str(line).strip()}
    branch_nodes = {normalize_pytest_node(line) for line in branch if str(line).strip()}
    base_skip_nodes = {
        normalize_pytest_node(line) for line in base_skips if str(line).strip()
    }
    branch_skip_nodes = {
        normalize_pytest_node(line) for line in branch_skips if str(line).strip()
    }
    base_xfail_nodes = {
        normalize_pytest_node(line) for line in base_xfails if str(line).strip()
    }
    branch_xfail_nodes = {
        normalize_pytest_node(line) for line in branch_xfails if str(line).strip()
    }
    return TestDelta(
        new_failures=tuple(sorted(branch_nodes - base_nodes)),
        fixed_failures=tuple(sorted(base_nodes - branch_nodes)),
        pre_existing_failures=tuple(sorted(base_nodes & branch_nodes)),
        changed_skips=tuple(sorted(base_skip_nodes ^ branch_skip_nodes)),
        changed_xfails=tuple(sorted(base_xfail_nodes ^ branch_xfail_nodes)),
    )

def evaluate_readiness(evidence: ReadyEvidence) -> BenchmarkReadinessVerdict:

    if not isinstance(evidence, ReadyEvidence):
        raise VerdictError("evaluate_readiness requires ReadyEvidence")
    results: list[CriterionResult] = []
    blockers: list[str] = []

    def record(
        index: int,
        name: str,
        *,
        blocked: bool,
        code: str | None,
        detail: str,
    ) -> None:
        status = "BLOCKED" if blocked else "PASS"
        if blocked and code:
            blockers.append(code)
        results.append(
            CriterionResult(index, name, status, code if blocked else None, detail)
        )

    record(
        1,
        "authenticated_inputs",
        blocked=not (
            evidence.train_data_authenticated and evidence.projection_v2_authenticated
        ),
        code="train_data_auth_failed",
        detail="TRAIN_DATA_READY and parent-projection v2 must authenticate v5 inputs",
    )
    record(
        1,
        "v1_track_b_preservation",
        blocked=not (
            evidence.v1_unchanged
            and evidence.track_b_unchanged
            and _v1_and_track_b_preserved()
        ),
        code="v1_changed",
        detail="parent-projection v1 and Track B pin must remain byte-identical",
    )

    waived = evidence.structural_audit == WAIVED_STRUCTURAL_AUDIT
    foldseek_required_missing = (not waived) and (not evidence.foldseek_complete)
    record(
        2,
        "structural_audit",
        blocked=foldseek_required_missing,
        code="foldseek_incomplete",
        detail=(
            "structural audit is not_required_by_frozen_contract"
            if waived
            else "pinned Foldseek audit must complete with recorded mapping"
        ),
    )
    unmapped = (not waived) and (
        (not evidence.mapping_complete) or bool(evidence.unmapped_queries)
    )
    record(
        3,
        "structural_mapping",
        blocked=unmapped or not evidence.thresholds_unchanged,
        code="unmapped_structure",
        detail=(
            "waived audit has an empty disclosed conflict set"
            if waived
            else "every required query must map; thresholds stay inclusive 0.50/0.80"
        ),
    )

    unknown_as_clean = evidence.unknown_reported_clear or (
        evidence.exposure_all_unknown and _temporal_clean_parent_groups(evidence) > 0
    )
    record(
        4,
        "exposure_taxonomy",
        blocked=unknown_as_clean,
        code="unknown_reported_clear",
        detail="unknown exposure is never reported clean or temporal-clean",
    )
    record(
        5,
        "view_authentication",
        blocked=not evidence.view_hash_matches,
        code="view_hash_drift",
        detail="standard/strict/temporal-clean manifests must authenticate their inputs",
    )
    underpowered = _underpowered_families(evidence)
    record(
        6,
        "strict_group_floor",
        blocked=bool(underpowered),
        code="underpowered_family",
        detail=(
            f"strict validation and blind need >= {evidence.min_parent_groups} parent groups"
        ),
    )

    smoke_missing = evidence.smoke_status != SMOKE_COMPLETE
    metric_unavailable = (
        evidence.required_metric_unavailable
        or not evidence.evaluators_available
        or not evidence.evaluator_registry_authenticated
    )
    if metric_unavailable and not smoke_missing:
        record(
            7,
            "evaluators_and_smoke",
            blocked=True,
            code="required_metric_unavailable",
            detail="binder/AME/antibody primaries must be available",
        )
    elif smoke_missing:
        record(
            7,
            "evaluators_and_smoke",
            blocked=True,
            code="smoke_not_run",
            detail="permitted train-only smoke has not been launched",
        )
        blockers.append("required_metric_unavailable")
    else:
        record(
            7,
            "evaluators_and_smoke",
            blocked=False,
            code="required_metric_unavailable",
            detail="primaries available and permitted non-blind smoke completed",
        )

    record(
        8,
        "evaluation_arms",
        blocked=not evidence.arm_pairing_ok,
        code="arm_pairing_failed",
        detail="arm vocabulary, pairing, cell identity, and compute accounting are frozen",
    )
    cli_findings = scan_readiness_cli_for_blind_access()
    tree_findings = list(evidence.run_tree_findings)
    tree_scanned = evidence.run_tree_scan_status == _SCAN_CLEAN
    if evidence.run_tree_root is not None:
        root = evidence.run_tree_root
        if not isinstance(root, Path) or not root.is_dir():
            tree_scanned = False
        else:
            try:
                tree_findings.extend(scan_readiness_run_trees(root))
                tree_scanned = True
            except VerdictError:
                tree_scanned = False
    blind_violation = bool(
        cli_findings
        or tree_findings
        or evidence.blind_output_seen
        or evidence.cli_imports_blind_loader
    )
    if blind_violation:
        record(
            9,
            "blind_sentinel",
            blocked=True,
            code="blind_output_seen",
            detail="readiness workflow must not generate, inspect, or unseal blind outputs",
        )
    elif not tree_scanned:
        record(
            9,
            "blind_sentinel",
            blocked=True,
            code="blind_scan_not_run",
            detail="blind sentinel has not searched readiness CLIs and run trees",
        )
    else:
        record(
            9,
            "blind_sentinel",
            blocked=False,
            code="blind_output_seen",
            detail="readiness workflow must not generate, inspect, or unseal blind outputs",
        )
    panel_ok = evidence.binder_panel in {
        BINDER_PANEL_NOT_REQUIRED,
        REQUIRED_AND_COMPLETE,
    }
    record(
        10,
        "binder_panel",
        blocked=not panel_ok,
        code="binder_panel_unresolved",
        detail="binder-panel search is not_required_by_frozen_contract",
    )
    delta_missing = evidence.test_delta_status != _TEST_DELTA_COMPLETE
    if delta_missing:
        criterion_11_code = "test_delta_missing"
        criterion_11_blocked = True
    elif evidence.new_failures:
        criterion_11_code = "new_failure"
        criterion_11_blocked = True
    elif not evidence.focused_tests_pass:
        criterion_11_code = "focused_tests_missing"
        criterion_11_blocked = True
    else:
        criterion_11_code = "new_failure"
        criterion_11_blocked = False
    record(
        11,
        "test_delta",
        blocked=criterion_11_blocked,
        code=criterion_11_code,
        detail="focused tests pass and the normalized full-suite delta has no new nodes",
    )
    record(
        12,
        "worktree_clean",
        blocked=not evidence.worktree_clean,
        code="worktree_dirty",
        detail="evidence-commit worktree must be clean before publication",
    )

    temporal_clean = (
        "UNAVAILABLE"
        if evidence.exposure_all_unknown or _temporal_clean_parent_groups(evidence) == 0
        else "AVAILABLE"
    )
    claims = {
        "temporal_clean": temporal_clean,
        "structural_audit": (
            WAIVED_STRUCTURAL_AUDIT if waived else evidence.structural_audit
        ),
        "binder_panel": evidence.binder_panel,
        "smoke": evidence.smoke_status,
        "exposure": "unknown" if evidence.exposure_all_unknown else "classified",
    }
    limitations = []
    if evidence.exposure_all_unknown:
        limitations.append(
            "temporal-clean is empty because every base-checkpoint exposure subject is unknown"
        )
    if waived:
        limitations.append(
            "structural audit is not_required_by_frozen_contract; Foldseek was not executed"
        )
    if smoke_missing:
        limitations.append("train-only base-checkpoint smoke has not been launched")
    unique_blockers = tuple(dict.fromkeys(blockers))
    status = "BLOCKED" if unique_blockers else "PASS"
    spec_path = _v1_spec_path()
    completion_path = _v1_completion_path()
    hashes = {
        "view_semantic_hash": evidence.view_semantic_hash,
        "view_partition_hash": evidence.view_partition_hash,
        "v1_spec_sha256": (
            hashlib.sha256(spec_path.read_bytes()).hexdigest()
            if spec_path.is_file()
            else ""
        ),
        "v1_completion_sha256": (
            hashlib.sha256(completion_path.read_bytes()).hexdigest()
            if completion_path.is_file()
            else ""
        ),
    }
    if not evidence.view_hash_matches:
        hashes["observed_view_semantic_hash"] = "drifted"
    return BenchmarkReadinessVerdict(
        status=status,
        criteria=tuple(results),
        blocker_codes=unique_blockers,
        claims=claims,
        views=copy.deepcopy(evidence.views),
        hashes=hashes,
        limitations=tuple(limitations),
    )

def render_readiness_markdown(verdict: BenchmarkReadinessVerdict) -> str:

    if not isinstance(verdict, BenchmarkReadinessVerdict):
        raise VerdictError(
            "render_readiness_markdown requires BenchmarkReadinessVerdict"
        )
    record = verdict.to_mapping()
    lines = [
        "# Benchmark readiness",
        "",
        f"Status: {record['status']}",
        "",
        "## Claims",
    ]
    for key, value in record["claims"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Views"])
    for family, family_views in record["views"].items():
        for name, cell in family_views.items():
            lines.append(
                f"- {family} {name}: example_rows={cell['example_rows']} "
                f"parent_groups={cell['parent_groups']}"
            )
    lines.extend(["", "## Hashes"])
    for key, value in record["hashes"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Criteria"])
    for item in record["criteria"]:
        code = item["blocker_code"] or "none"
        lines.append(
            f"- {item['index']}. {item['name']}: {item['status']} ({code}) {item['detail']}"
        )
    lines.extend(["", "## Blockers"])
    if record["blocker_codes"]:
        for code in record["blocker_codes"]:
            lines.append(f"- {code}")
    else:
        lines.append("- none")
    lines.extend(["", "## Limitations"])
    if record["limitations"]:
        for item in record["limitations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    claims = record["claims"]
    if claims.get("temporal_clean") == "UNAVAILABLE":
        lines.append("")
        if claims.get("exposure") == "unknown":
            lines.append(
                "Temporal-clean is empty and base exposure is unknown. This is a "
                "disclosed limitation, not a contamination-clean claim."
            )
        else:
            lines.append(
                "Temporal-clean is empty. This is a disclosed no-clean-claim limitation."
            )
    return "\n".join(lines) + "\n"

def publish_readiness(
    verdict: BenchmarkReadinessVerdict,
    *,
    output_dir: Path,
    authorize_publish: bool,
    marker_path: Path,
    evidence_commit: str = "",
) -> None:

    if not isinstance(verdict, BenchmarkReadinessVerdict):
        raise VerdictError("publish_readiness requires BenchmarkReadinessVerdict")
    if verdict.status != "PASS":
        raise VerdictError("refusing publication of a BLOCKED verdict")
    if not authorize_publish:
        raise VerdictError("explicit publish authorization is required")
    assert_terminal_path_allowed(output_dir)
    assert_terminal_path_allowed(marker_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = verdict.to_mapping()
    payload["evidence_commit"] = evidence_commit
    json_path = output_dir / "BENCHMARK_READY.json"
    md_path = output_dir / "BENCHMARK_READY.md"
    json_path.write_bytes(canonical_json_bytes(payload))
    md_path.write_text(render_readiness_markdown(verdict), encoding="utf-8")
    marker_path.write_text("PASS\n", encoding="utf-8")
    _update_handoff(_REPO_ROOT / "HANDOFF.md", evidence_commit)

def scan_readiness_cli_for_blind_access(
    sources: Sequence[Path] | None = None,
) -> tuple[str, ...]:

    findings: list[str] = []
    relatives = sources if sources is not None else _READINESS_SOURCES
    for relative in relatives:
        path = relative if relative.is_absolute() else _REPO_ROOT / relative
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        label = str(relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_BLIND_NAMES:
                        findings.append(f"{label}:{alias.name}")
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_BLIND_NAMES:
                findings.append(f"{label}:{node.id}")
            elif (
                isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_BLIND_NAMES
            ):
                findings.append(f"{label}:{node.attr}")
    return tuple(sorted(set(findings)))

def _entry_kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "dir"
    return "other"

def _snapshot_run_tree(root: Path) -> dict[str, tuple[object, ...]]:
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise VerdictError(
            "readiness run tree root is missing or not a directory"
        ) from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise VerdictError("readiness run tree root is missing or not a directory")
    snapshot: dict[str, tuple[object, ...]] = {}
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                children = list(iterator)
        except OSError as error:
            raise VerdictError(
                f"read failed in readiness run tree: {current}"
            ) from error
        for entry in children:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise VerdictError(
                    f"read failed in readiness run tree: {path}"
                ) from error
            relative = path.relative_to(root).as_posix()
            kind = _entry_kind(info.st_mode)
            snapshot[relative] = (
                kind,
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            )
            if kind == "dir":
                stack.append(path)
    return snapshot

def _read_regular_file_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerdictError(f"read failed in readiness run tree: {path}") from error
    try:
        info = os.fstat(descriptor)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise VerdictError(f"non-regular entry in readiness run tree: {path}")
        chunks: list[bytes] = []
        while True:
            data = os.read(descriptor, 1024 * 1024)
            if not data:
                break
            chunks.append(data)
        return b"".join(chunks)
    finally:
        os.close(descriptor)

def scan_readiness_run_trees(root: Path) -> tuple[str, ...]:

    if not isinstance(root, Path):
        raise VerdictError("readiness run tree root is missing or not a directory")
    before = _snapshot_run_tree(root)
    findings: list[str] = []
    for relative, metadata in sorted(before.items()):
        kind = str(metadata[0])
        path = root / relative
        if kind == "symlink":
            raise VerdictError(f"symlink in readiness run tree: {path}")
        if kind == "dir":
            continue
        if kind != "file":
            raise VerdictError(f"non-regular entry in readiness run tree: {path}")
        if path.name in _VERDICT_ARTIFACT_NAMES:
            continue
        raw = _read_regular_file_nofollow(path)
        text = raw.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if (
            isinstance(payload, Mapping)
            and payload.get("schema_version") in _VERDICT_SCHEMAS
        ):
            continue
        for token in _FORBIDDEN_TREE_TOKENS:
            if token in text:
                findings.append(f"{path}:{token}")
        if payload is not None:
            findings.extend(f"{path}:{name}" for name in _forbidden_json_keys(payload))
    after = _snapshot_run_tree(root)
    if after != before:
        raise VerdictError("readiness run tree mutated during scan")
    return tuple(sorted(set(findings)))

def verify_baseline_worktree_absent(
    path: Path,
    *,
    registered_paths: Sequence[str] | None = None,
) -> None:

    if not isinstance(path, Path):
        raise VerdictError("baseline worktree path must be a Path")
    if path.exists():
        raise VerdictError(f"baseline worktree already exists: {path}")
    registered = (
        tuple(registered_paths)
        if registered_paths is not None
        else _git_worktree_paths()
    )
    resolved = str(path.resolve())
    for item in registered:
        if item == str(path) or str(Path(item).resolve()) == resolved:
            raise VerdictError(f"baseline worktree is already registered: {path}")

def assert_terminal_path_allowed(path: Path) -> None:

    resolved = path.resolve()
    for denied in DENIED_PREFIXES:
        if resolved == denied or denied in resolved.parents:
            raise VerdictError(f"terminal evidence path is denied: {path}")

def extract_failed_nodes(log_text: str) -> tuple[str, ...]:

    return _extract_prefixed_nodes(log_text, "FAILED ")

def extract_skipped_nodes(log_text: str) -> tuple[str, ...]:

    return _extract_prefixed_nodes(log_text, "SKIPPED ")

def extract_xfailed_nodes(log_text: str) -> tuple[str, ...]:

    return _extract_prefixed_nodes(log_text, "XFAIL ")

def _extract_prefixed_nodes(log_text: str, prefix: str) -> tuple[str, ...]:
    if type(log_text) is not str:
        raise VerdictError("pytest log text must be a string")
    nodes = [
        normalize_pytest_node(line)
        for line in log_text.splitlines()
        if line.startswith(prefix)
    ]
    return tuple(sorted(set(nodes)))

def parse_pytest_summary(log_text: str) -> dict[str, int] | None:

    if type(log_text) is not str or not log_text.strip():
        return None
    if "Interrupted:" in log_text or "errors during collection" in log_text:
        return None
    matched = None
    for line in log_text.splitlines():
        stripped = line.strip().strip("=").strip()
        lowered = stripped.lower()
        if any(
            token in lowered
            for token in ("deselected", "selected in", "collected ", "cached in")
        ):
            continue
        found = _PYTEST_SUMMARY.search(stripped)
        if found is not None:
            matched = found
    if matched is None:
        return None
    failed, errors, passed, skipped, xfailed, xpassed, _warnings = (
        int(group or 0) for group in matched.groups()
    )
    if failed + errors + passed + skipped + xfailed + xpassed == 0:
        return None
    return {
        "failed": failed,
        "errors": errors,
        "passed": passed,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "collected": failed + errors + passed + skipped + xfailed + xpassed,
    }

def focused_tests_passed(log_text: str, exit_code: int) -> bool:

    if type(log_text) is not str or type(exit_code) is not int:
        raise VerdictError("focused test log and exit code are required")
    if exit_code != 0:
        return False
    return not extract_failed_nodes(log_text)

def test_delta_cli_main(argv: Sequence[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Normalized pytest failure delta against the frozen baseline"
    )
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--branch-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "emergent" / "benchmark.yaml",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if Path(sys.executable).resolve() != RUNTIME_PYTHON.resolve():
        raise RuntimeError("run_benchmark_test_delta requires the pinned interpreter")
    if args.base_commit != BASELINE_COMMIT:
        raise VerdictError(f"base commit must be {BASELINE_COMMIT}")
    _ensure_baseline_worktree(args.base_commit)
    contract = load_benchmark_contract(args.config)
    assert_terminal_path_allowed(Path(contract.evidence_root))
    command = tuple(
        sys.argv if argv is None else ["run_benchmark_test_delta.py", *argv]
    )
    run = claim_benchmark_run(args.run_id, command, contract)
    started = datetime.now(UTC)
    branch_root = _REPO_ROOT
    branch_commit = _current_commit(branch_root)
    if args.branch_commit not in {branch_commit, "HEAD"}:
        raise VerdictError("branch commit must be the current clean HEAD")
    base_log, base_exit, base_started, base_ended = _run_full_suite(BASELINE_WORKTREE)
    branch_log, branch_exit, branch_started, branch_ended = _run_full_suite(branch_root)
    focused_log, focused_exit, focused_started, focused_ended = _run_focused_suite(
        branch_root
    )
    delta = classify_test_delta(
        extract_failed_nodes(base_log),
        extract_failed_nodes(branch_log),
        base_skips=extract_skipped_nodes(base_log),
        branch_skips=extract_skipped_nodes(branch_log),
        base_xfails=extract_xfailed_nodes(base_log),
        branch_xfails=extract_xfailed_nodes(branch_log),
    )
    ended = datetime.now(UTC)
    _write_run_bytes(run, "environment.txt", _environment_bytes())
    _write_run_bytes(run, "base-pytest.log", base_log.encode())
    _write_run_bytes(run, "branch-pytest.log", branch_log.encode())
    _write_run_bytes(run, "focused-pytest.log", focused_log.encode())
    suites = {
        "base": (base_log, base_exit, str(BASELINE_WORKTREE), base_started, base_ended),
        "branch": (
            branch_log,
            branch_exit,
            str(branch_root),
            branch_started,
            branch_ended,
        ),
        "focused": (
            focused_log,
            focused_exit,
            str(branch_root),
            focused_started,
            focused_ended,
        ),
    }
    suite_records = {}
    complete_run = True
    for name, (log_text, exit_code, cwd, suite_started, suite_ended) in suites.items():
        summary = parse_pytest_summary(log_text)
        identity = _log_identity(run.evidence_dir / f"{name}-pytest.log")
        if identity is None or summary is None or _abnormal_pytest_exit(exit_code):
            complete_run = False
        record = {
            "cwd": cwd,
            "exit_code": exit_code,
            "started_utc": suite_started,
            "ended_utc": suite_ended,
            "log": identity,
        }
        if summary is not None:
            record["summary"] = _summary_line(log_text)
            record["collected"] = summary["collected"]
        if name == "focused":
            record["paths"] = list(FOCUSED_PYTEST_PATHS)
        suite_records[name] = record
        if name == "focused" and not focused_tests_passed(log_text, exit_code):
            complete_run = False
    terminal_status = "PASS" if complete_run else "FAIL"
    payload = {
        "schema_version": "dive-benchmark-test-delta-v1",
        "base_commit": args.base_commit,
        "branch_commit": branch_commit,
        "host": socket.gethostname(),
        "runtime_python": str(RUNTIME_PYTHON),
        "command": list(_PYTEST_COMMAND),
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "terminal_status": terminal_status,
        "base": suite_records["base"],
        "branch": suite_records["branch"],
        "focused": suite_records["focused"],
        "delta": delta.to_mapping(),
        "unexplained_new_failures": list(delta.new_failures),
        "focused_tests_pass": focused_tests_passed(focused_log, focused_exit),
    }
    _write_run_bytes(run, "delta.json", canonical_json_bytes(payload))
    complete_benchmark_run(run, payload)
    print(json.dumps(payload, sort_keys=True))
    if terminal_status != "PASS" or delta.new_failures:
        return 1
    return 0

def publish_cli_main(argv: Sequence[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Fail-closed benchmark readiness verdict"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--authorize-publish", action="store_true")
    parser.add_argument("--authenticate-publication", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if Path(sys.executable).resolve() != RUNTIME_PYTHON.resolve():
        raise RuntimeError("publish_benchmark_ready requires the pinned interpreter")
    if args.authorize_publish and args.authenticate_publication:
        raise VerdictError(
            "exactly one of --authorize-publish or --authenticate-publication is required"
        )
    contract = load_benchmark_contract(args.config)
    evidence = load_readiness_evidence(contract)
    verdict = evaluate_readiness(evidence)
    record = verdict.to_mapping()
    record["evidence_commit"] = args.evidence_commit
    print(json.dumps(record, sort_keys=True))
    command = tuple(sys.argv if argv is None else ["publish_benchmark_ready.py", *argv])
    if args.authenticate_publication:
        if not args.run_id:
            raise VerdictError("post-publication authentication requires --run-id")
        post_run = claim_benchmark_run(args.run_id, command, contract)
        authenticate_publication(
            evidence_commit=args.evidence_commit,
            contract=contract,
            run=post_run,
        )
        return 0
    if verdict.status != "PASS":
        run_id = args.run_id or (
            "benchmark-verdict-blocked-" + args.evidence_commit[:12].lower()
        )
        write_blocked_verdict(
            verdict,
            contract=contract,
            run_id=run_id,
            argv=command,
            evidence_commit=args.evidence_commit,
        )
        return 1
    if not args.authorize_publish:
        raise VerdictError("explicit publish authorization is required")
    publish_readiness(
        verdict,
        output_dir=args.output_dir,
        authorize_publish=True,
        marker_path=_REPO_ROOT / "BENCHMARK_READY",
        evidence_commit=args.evidence_commit,
    )
    return 0

def load_readiness_evidence(contract) -> ReadyEvidence:

    if contract is None:
        raise VerdictError("load_readiness_evidence requires a benchmark contract")
    evidence = ReadyEvidence.production_snapshot()
    evidence.train_data_authenticated, evidence.projection_v2_authenticated = (
        _authenticate_train_inputs(contract)
    )
    evidence.thresholds_unchanged = _thresholds_match_frozen_contract(contract)
    if evidence.structural_audit == WAIVED_STRUCTURAL_AUDIT:
        evidence.mapping_complete = True
    preserved = _v1_and_track_b_preserved()
    evidence.v1_unchanged = preserved
    evidence.track_b_unchanged = preserved
    evidence_root = Path(contract.evidence_root)
    view_dir = evidence_root / "benchmark_ready" / evidence.view_run_id
    evidence.view_hash_matches = _authenticate_view_hashes(view_dir, evidence)
    evidence.arm_pairing_ok = _authenticate_arm_pairing(contract)
    evidence.evaluator_registry_authenticated, registry_ready = (
        _authenticate_evaluator_registry(contract)
    )
    if evidence.evaluator_registry_authenticated:
        evidence.evaluators_available = registry_ready
        evidence.required_metric_unavailable = not registry_ready
    ready_root = evidence_root / "benchmark_ready"
    if ready_root.is_dir():
        evidence.run_tree_root = ready_root
        evidence.run_tree_scan_status = _SCAN_NOT_RUN
        smoke = _authenticate_smoke(ready_root, contract)
        if smoke is not None:
            evidence.smoke_status = SMOKE_COMPLETE
            if evidence.evaluator_registry_authenticated:
                evidence.evaluators_available = (
                    registry_ready and smoke.evaluators_available
                )
                evidence.required_metric_unavailable = not evidence.evaluators_available
            else:
                evidence.evaluators_available = smoke.evaluators_available
                evidence.required_metric_unavailable = not smoke.evaluators_available
        delta = _authenticate_test_delta(ready_root, contract)
        if delta is not None:
            evidence.test_delta_status = _TEST_DELTA_COMPLETE
            evidence.new_failures = delta.new_failures
            evidence.focused_tests_pass = delta.focused_tests_pass
    evidence.worktree_clean = _worktree_is_clean(_REPO_ROOT)
    return evidence

def _thresholds_match_frozen_contract(contract) -> bool:
    foldseek = getattr(contract, "foldseek", None)
    return (
        getattr(foldseek, "min_tm_score", None) == 0.50
        and getattr(foldseek, "min_shorter_coverage", None) == 0.80
    )

def _authenticate_train_inputs(contract) -> tuple[bool, bool]:
    from dive.benchmark.inputs import BenchmarkInputError, load_frozen_benchmark_inputs

    try:
        load_frozen_benchmark_inputs(contract)
    except (BenchmarkInputError, OSError, TypeError, AttributeError, ValueError):
        return False, False
    return True, True

def _authenticate_evaluator_registry(contract) -> tuple[bool, bool]:
    from dive.benchmark.evaluators.contracts import (
        EvaluatorContractError,
        load_evaluator_registry,
        verify_evaluator_availability,
    )

    try:
        registry = load_evaluator_registry(contract)
        availability = verify_evaluator_availability(registry)
    except (EvaluatorContractError, OSError, TypeError, AttributeError, ValueError):
        return False, False
    return True, bool(availability.ready)

def write_blocked_verdict(
    verdict: BenchmarkReadinessVerdict,
    *,
    contract,
    run_id: str,
    argv: Sequence[str],
    evidence_commit: str = "",
):

    if not isinstance(verdict, BenchmarkReadinessVerdict):
        raise VerdictError("write_blocked_verdict requires BenchmarkReadinessVerdict")
    if verdict.status == "PASS":
        raise VerdictError("refusing to write a blocked verdict for a PASS")
    assert_terminal_path_allowed(Path(contract.evidence_root))
    run = claim_benchmark_run(run_id, argv, contract)
    payload = verdict.to_mapping()
    payload["evidence_commit"] = evidence_commit
    payload["publication"] = "refused"
    write_terminal_bytes(
        run.evidence_dir / "blocked-verdict.json",
        canonical_json_bytes(payload),
        run.evidence_dir.parents[1],
        run.bulk_dir.parents[1],
    )
    complete_benchmark_run(run, payload)
    return run

def authenticate_publication(
    *,
    evidence_commit: str,
    contract,
    run,
    repo: Path | None = None,
) -> dict[str, object]:

    if type(evidence_commit) is not str or not evidence_commit:
        raise VerdictError("publication commits must be strings")
    if contract is None:
        raise VerdictError("contract hash is required")
    if run is None:
        raise VerdictError(
            "post-publication authentication requires a group evidence run"
        )
    repo = repo if repo is not None else _REPO_ROOT
    head_commit = _current_commit(repo)
    if evidence_commit == head_commit:
        raise VerdictError("self-referential final-commit hash is forbidden")
    if not _worktree_is_clean(repo):
        raise VerdictError("worktree is not clean")
    if not _is_git_ancestor(evidence_commit, head_commit, repo):
        raise VerdictError("evidence commit is not an ancestor of HEAD")
    observed = _git_changed_files(evidence_commit, head_commit, repo)
    observed_set = frozenset(observed)
    allow = frozenset(PUBLICATION_ALLOWLIST)
    required = frozenset(
        {
            "docs/results/emergent/BENCHMARK_READY.json",
            "docs/results/emergent/BENCHMARK_READY.md",
        }
    )
    if not observed_set or not observed_set <= allow or not required <= observed_set:
        raise VerdictError(
            "intervening changed-file set is not the publication allowlist"
        )
    contract_sha256 = getattr(contract, "semantic_sha256", "")
    if type(contract_sha256) is not str or not contract_sha256:
        raise VerdictError("contract hash is required")
    if not _v1_and_track_b_preserved():
        raise VerdictError("v1 or Track B hash drifted after publication")
    if not _views_match_pins(contract):
        raise VerdictError("view hashes drifted after publication")
    evidence = load_readiness_evidence(contract)
    reconstructed = evaluate_readiness(evidence)
    if reconstructed.status != "PASS":
        raise VerdictError("reconstructed readiness is not PASS")
    marker = repo / "BENCHMARK_READY"
    if not marker.is_file() or marker.is_symlink() or marker.read_bytes() != b"PASS\n":
        raise VerdictError("BENCHMARK_READY marker is not exactly PASS\\n")
    json_path = repo / "docs" / "results" / "emergent" / "BENCHMARK_READY.json"
    md_path = repo / "docs" / "results" / "emergent" / "BENCHMARK_READY.md"
    payload = reconstructed.to_mapping()
    payload["evidence_commit"] = evidence_commit
    expected_json = canonical_json_bytes(payload)
    expected_md = render_readiness_markdown(reconstructed).encode("utf-8")
    try:
        observed_json = json_path.read_bytes()
        observed_md = md_path.read_bytes()
    except OSError as error:
        raise VerdictError("publication artifacts are missing") from error
    if observed_json != expected_json:
        raise VerdictError("publication JSON drifted from reconstructed verdict")
    if observed_md != expected_md:
        raise VerdictError("publication Markdown drifted from reconstructed verdict")
    record = {
        "schema_version": "dive-benchmark-post-publication-v1",
        "status": "PASS",
        "terminal_status": "PASS",
        "evidence_commit": evidence_commit,
        "head_commit": head_commit,
        "changed_files": sorted(observed),
        "worktree_clean": True,
        "contract_sha256": contract_sha256,
        "view_run_id": evidence.view_run_id,
        "smoke_status": evidence.smoke_status,
        "test_delta_status": evidence.test_delta_status,
        "marker_sha256": hashlib.sha256(b"PASS\n").hexdigest(),
        "json_sha256": hashlib.sha256(observed_json).hexdigest(),
        "markdown_sha256": hashlib.sha256(observed_md).hexdigest(),
    }
    write_terminal_bytes(
        run.evidence_dir / "post-publication-authentication.json",
        canonical_json_bytes(record),
        run.evidence_dir.parents[1],
        run.bulk_dir.parents[1],
    )
    complete_benchmark_run(run, record)
    return record

def _v1_spec_path() -> Path:
    return _REPO_ROOT / "configs" / "emergent" / "directional_parent_projection_v1.json"

def _v1_completion_path() -> Path:
    return (
        EMERGENT_EVIDENCE_ROOT / "directional_parent_projections" / "v1.completion.json"
    )

def _v1_and_track_b_preserved() -> bool:

    from dive.training.parent_projection import (
        PRODUCTION_CONTRACT,
        _PROJECTION_APPROVAL_POLICY,
    )

    policy = _PROJECTION_APPROVAL_POLICY
    spec_path = _v1_spec_path()
    if not spec_path.is_file() or spec_path.is_symlink():
        return False
    spec_raw = spec_path.read_bytes()
    if hashlib.sha256(spec_raw).hexdigest() != policy.spec_sha256:
        return False
    try:
        payload = json.loads(spec_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    if (
        payload.get("identity") != "v1"
        or payload.get("contract_identity") != PRODUCTION_CONTRACT.identity
    ):
        return False
    completion_path = _v1_completion_path()
    if not completion_path.is_file() or completion_path.is_symlink():
        return False
    completion_raw = completion_path.read_bytes()
    return hashlib.sha256(completion_raw).hexdigest() == policy.completion_sha256

def _temporal_clean_parent_groups(evidence: ReadyEvidence) -> int:
    total = 0
    for family_views in evidence.views.values():
        cell = family_views.get("temporal_clean")
        if isinstance(cell, Mapping):
            total += int(cell.get("parent_groups", 0))
    return total

def _underpowered_families(evidence: ReadyEvidence) -> tuple[str, ...]:
    weak = []
    floor = evidence.min_parent_groups
    for family in _FAMILIES:
        family_views = evidence.views.get(family, {})
        for name in ("strict_blind", "strict_validation"):
            cell = family_views.get(name)
            if not isinstance(cell, Mapping):
                weak.append(f"{family}:{name}")
                continue
            if int(cell.get("parent_groups", 0)) < floor:
                weak.append(f"{family}:{name}")
    return tuple(weak)

def _git_worktree_paths() -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "-C", str(_REPO_ROOT), "worktree", "list", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    )
    paths = []
    for line in completed.stdout.splitlines():
        if line.startswith("worktree "):
            paths.append(line.split(" ", 1)[1])
    return tuple(paths)

def _current_commit(repo: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

def _ensure_baseline_worktree(commit: str) -> None:
    path = BASELINE_WORKTREE
    if path.exists():
        observed = _current_commit(path)
        if observed != commit:
            raise VerdictError(f"baseline worktree is not {commit}")
        if not _worktree_is_clean(path):
            raise VerdictError("baseline worktree is dirty")
        return
    verify_baseline_worktree_absent(path)
    _create_baseline_worktree(commit)

def _create_baseline_worktree(commit: str) -> None:
    verify_baseline_worktree_absent(BASELINE_WORKTREE)
    subprocess.run(
        (
            "git",
            "-C",
            str(_REPO_ROOT),
            "worktree",
            "add",
            "--detach",
            str(BASELINE_WORKTREE),
            commit,
        ),
        check=True,
        capture_output=True,
        text=True,
    )

def _run_full_suite(cwd: Path) -> tuple[str, int, str, str]:
    started = datetime.now(UTC).isoformat()
    completed = subprocess.run(
        _PYTEST_COMMAND,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(cwd / "src")},
    )
    ended = datetime.now(UTC).isoformat()
    log = completed.stdout + completed.stderr
    return log, int(completed.returncode), started, ended

def _run_focused_suite(cwd: Path) -> tuple[str, int, str, str]:
    started = datetime.now(UTC).isoformat()
    completed = subprocess.run(
        (
            str(RUNTIME_PYTHON),
            "-m",
            "pytest",
            *FOCUSED_PYTEST_PATHS,
            "-q",
            "--tb=no",
            "-ra",
        ),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(cwd / "src")},
    )
    ended = datetime.now(UTC).isoformat()
    log = completed.stdout + completed.stderr
    return log, int(completed.returncode), started, ended

def _environment_bytes() -> bytes:
    payload = {
        "host": socket.gethostname(),
        "runtime_python": str(RUNTIME_PYTHON),
        "cwd": str(_REPO_ROOT),
    }
    return canonical_json_bytes(payload)

def _write_run_bytes(run, relative: str, content: bytes) -> None:
    destination = run.evidence_dir / relative
    write_terminal_bytes(
        destination,
        content,
        run.evidence_dir.parents[1],
        run.bulk_dir.parents[1],
    )

def _authenticate_view_hashes(view_dir: Path, evidence: ReadyEvidence) -> bool:
    completion_path = view_dir / "completion.json"
    summary_path = view_dir / "view_summary.json"
    if not completion_path.is_file() or not summary_path.is_file():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(completion, Mapping) or not isinstance(summary, Mapping):
        return False
    payload = completion.get("payload", completion)
    if not isinstance(payload, Mapping):
        return False
    semantic = payload.get("semantic_hash")
    partition = payload.get("partition_hash")
    if (
        type(semantic) is not str
        or type(partition) is not str
        or semantic != summary.get("semantic_hash")
        or partition != summary.get("partition_hash")
        or semantic != _VIEW_SEMANTIC_HASH
        or partition != _VIEW_PARTITION_HASH
    ):
        return False
    evidence.view_semantic_hash = _VIEW_SEMANTIC_HASH
    evidence.view_partition_hash = _VIEW_PARTITION_HASH
    families = summary.get("counts")
    if not isinstance(families, Mapping):
        return False
    family_counts = families.get("families")
    if not isinstance(family_counts, Mapping):
        return False
    loaded: dict[str, dict[str, dict[str, int]]] = {}
    partition_map = {
        "test-blind": "strict_blind",
        "validation": "strict_validation",
    }
    for family in _FAMILIES:
        raw_family = family_counts.get(family)
        if not isinstance(raw_family, Mapping):
            return False
        loaded[family] = {}
        for raw_name, view_name in partition_map.items():
            cell = raw_family.get(raw_name)
            if not isinstance(cell, Mapping):
                return False
            parent_groups = cell.get("strict")
            if type(parent_groups) is not int or parent_groups < 0:
                return False
            parquet_name = f"{family}_{raw_name}_examples.parquet"
            example_rows = _count_view_parquet(
                view_dir / parquet_name,
                summary.get("artifacts", {}),
            )
            if example_rows is None:
                return False
            loaded[family][view_name] = {
                "example_rows": example_rows,
                "parent_groups": parent_groups,
            }
        loaded[family]["temporal_clean"] = {"example_rows": 0, "parent_groups": 0}
    evidence.views = loaded
    return True

def _count_view_parquet(path: Path, artifacts: object) -> int | None:
    if not path.is_file() or path.is_symlink():
        return None
    if not isinstance(artifacts, Mapping):
        return None
    declared = artifacts.get(path.name)
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    digest = hashlib.sha256(raw).hexdigest()
    if (
        not isinstance(declared, Mapping)
        or declared.get("sha256") != digest
        or declared.get("size_bytes") != len(raw)
    ):
        return None
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        if "strict" in table.column_names:
            return int(sum(bool(value) for value in table.column("strict").to_pylist()))
        return int(table.num_rows)
    except (OSError, ValueError, TypeError, ImportError):
        return None

def _views_match_pins(contract) -> bool:
    evidence_root = Path(getattr(contract, "evidence_root", EMERGENT_EVIDENCE_ROOT))
    view_dir = evidence_root / "benchmark_ready" / _ACCEPTED_VIEW_RUN
    probe = ReadyEvidence()
    return _authenticate_view_hashes(view_dir, probe)

def _update_handoff(path: Path, evidence_commit: str) -> None:

    stanza = (
        "# DIVE handoff\n\n"
        "## Current checkpoint — BENCHMARK_READY published\n\n"
        f"Evidence commit `{evidence_commit}` is ready for a focused publication "
        "commit of exactly BENCHMARK_READY, "
        "docs/results/emergent/BENCHMARK_READY.json, "
        "docs/results/emergent/BENCHMARK_READY.md, and HANDOFF.md. "
        "Run publish_benchmark_ready.py --authenticate-publication after that "
        "commit; do not train Track A or push.\n\n"
    )
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing.startswith("# DIVE handoff"):
        rest = existing.split("\n", 1)[1].lstrip("\n")
        path.write_text(stanza + rest, encoding="utf-8")
        return
    path.write_text(stanza + existing, encoding="utf-8")

def _authenticate_arm_pairing(contract) -> bool:

    from dive.benchmark.arms import (
        BenchmarkArmError,
        EvaluationArm,
        _require_contract_arms,
    )

    try:
        _require_contract_arms(contract)
    except (BenchmarkArmError, TypeError, AttributeError):
        return False
    expected = tuple(arm.value for arm in EvaluationArm)
    return tuple(getattr(contract, "arms", ())) == expected

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SMOKE_PURPOSE = "engineering_validation_not_performance_baseline"

def _authenticate_smoke(ready_root: Path, contract=None) -> AuthenticatedSmoke | None:

    from dive.benchmark.smoke import SMOKE_STEPS, SMOKE_VERIFICATIONS

    authenticated: AuthenticatedSmoke | None = None
    expected_contract = getattr(contract, "semantic_sha256", None)
    expected_checkpoint = _contract_checkpoint_sha(contract, "common")
    expected_autoencoder = _contract_checkpoint_sha(contract, "autoencoder")
    for path in sorted(ready_root.glob("*/smoke.completion.json")):
        if not path.is_file() or path.is_symlink():
            continue
        completion = path.with_name("completion.json")
        if not completion.is_file() or completion.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            wrapper = json.loads(completion.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping) or not isinstance(wrapper, Mapping):
            continue
        if not wrapper:
            continue
        if wrapper.get("schema_version") != _GENERIC_COMPLETION_SCHEMA:
            continue
        if wrapper.get("status") != "COMPLETE":
            continue
        if wrapper.get("run_id") != path.parent.name:
            continue
        inner = wrapper.get("payload")
        if not isinstance(inner, Mapping) or inner != payload:
            continue
        if (
            type(expected_contract) is str
            and expected_contract
            and wrapper.get("contract_sha256") != expected_contract
        ):
            continue
        if payload.get("schema_version") != _SMOKE_COMPLETION_SCHEMA:
            continue
        if payload.get("purpose") != _SMOKE_PURPOSE:
            continue
        if payload.get("blind") is not False or payload.get("partition") != "train":
            continue
        verifications = payload.get("verifications")
        if not isinstance(verifications, Sequence) or isinstance(
            verifications, (str, bytes)
        ):
            continue
        if any(item not in verifications for item in SMOKE_VERIFICATIONS):
            continue
        cells = payload.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            continue
        families: set[str] = set()
        primaries_available = True
        valid = True
        for cell in cells:
            if not isinstance(cell, Mapping):
                valid = False
                break
            family = cell.get("family")
            if type(family) is not str or family in families:
                valid = False
                break
            if cell.get("partition") != "train" or cell.get("arm") != "base":
                valid = False
                break
            if cell.get("steps") != SMOKE_STEPS:
                valid = False
                break
            if (
                type(cell.get("seed")) is not int
                or type(cell.get("sample_index")) is not int
            ):
                valid = False
                break
            cell_id = cell.get("cell_id")
            if type(cell_id) is not str or _HEX64.fullmatch(cell_id) is None:
                valid = False
                break
            if type(cell.get("parent_id")) is not str or not cell["parent_id"]:
                valid = False
                break
            checkpoint = cell.get("checkpoint")
            autoencoder = cell.get("autoencoder")
            if type(checkpoint) is not str or _HEX64.fullmatch(checkpoint) is None:
                valid = False
                break
            if type(autoencoder) is not str or _HEX64.fullmatch(autoencoder) is None:
                valid = False
                break
            if expected_checkpoint and checkpoint != expected_checkpoint:
                valid = False
                break
            if expected_autoencoder and autoencoder != expected_autoencoder:
                valid = False
                break
            families.add(family)
            metrics = cell.get("metrics")
            if not isinstance(metrics, Mapping):
                valid = False
                break
            primary = metrics.get("primary")
            if not isinstance(primary, Mapping):
                valid = False
                break
            expected_name = _SMOKE_REQUIRED_PRIMARIES.get(family)
            if primary.get("name") != expected_name:
                valid = False
                break
            manifest = primary.get("evaluator_manifest_hash")
            if type(manifest) is not str or _HEX64.fullmatch(manifest) is None:
                valid = False
                break
            status = primary.get("status")
            if status == "available":
                value = primary.get("value")
                if type(value) not in (int, float) or isinstance(value, bool):
                    valid = False
                    break
                if not math.isfinite(float(value)):
                    valid = False
                    break
            elif status == "unavailable":
                primaries_available = False
            else:
                valid = False
                break
        if not valid or families != _SMOKE_REQUIRED_FAMILIES:
            continue
        authenticated = AuthenticatedSmoke(evaluators_available=primaries_available)
    return authenticated

def _contract_checkpoint_sha(contract, name: str) -> str | None:
    checkpoints = dict(getattr(contract, "checkpoints", ()) or ())
    identity = checkpoints.get(name)
    digest = getattr(identity, "sha256", None)
    if type(digest) is str and _HEX64.fullmatch(digest) is not None:
        return digest
    return None

def _authenticate_test_delta(
    ready_root: Path, contract=None
) -> AuthenticatedTestDelta | None:
    authenticated: AuthenticatedTestDelta | None = None
    for path in sorted(ready_root.glob("*/delta.json")):
        result = _authenticate_one_test_delta(path, contract)
        if result is not None:
            authenticated = result
    return authenticated

def _authenticate_one_test_delta(path: Path, contract) -> AuthenticatedTestDelta | None:
    if not path.is_file() or path.is_symlink():
        return None
    completion_path = path.with_name("completion.json")
    if not completion_path.is_file() or completion_path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or not isinstance(completion, Mapping):
        return None
    if not completion:
        return None
    if completion.get("schema_version") != _GENERIC_COMPLETION_SCHEMA:
        return None
    if completion.get("status") != "COMPLETE":
        return None
    if completion.get("run_id") != path.parent.name:
        return None
    inner = completion.get("payload")
    if not isinstance(inner, Mapping) or inner != payload:
        return None
    expected_hash = getattr(contract, "semantic_sha256", None)
    if type(expected_hash) is str and expected_hash:
        observed_hash = completion.get("contract_sha256")
        if observed_hash != expected_hash:
            return None
    if payload.get("schema_version") != _TEST_DELTA_SCHEMA:
        return None
    if payload.get("base_commit") != BASELINE_COMMIT:
        return None
    branch_commit = payload.get("branch_commit")
    if type(branch_commit) is not str or _COMMIT_SHA.fullmatch(branch_commit) is None:
        return None
    if payload.get("runtime_python") != str(RUNTIME_PYTHON):
        return None
    command = payload.get("command")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return None
    if str(RUNTIME_PYTHON) not in command or "pytest" not in command:
        return None
    if payload.get("terminal_status") != "PASS":
        return None
    if "unexplained_new_failures" not in payload:
        return None
    unexplained = payload["unexplained_new_failures"]
    if not isinstance(unexplained, Sequence) or isinstance(unexplained, (str, bytes)):
        return None
    if type(payload.get("focused_tests_pass")) is not bool:
        return None
    delta = payload.get("delta")
    if not isinstance(delta, Mapping):
        return None
    for key in (
        "new_failures",
        "fixed_failures",
        "pre_existing_failures",
        "changed_skips",
        "changed_xfails",
    ):
        value = delta.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
    for name, required_exit_zero in (
        ("base", False),
        ("branch", False),
        ("focused", True),
    ):
        record = payload.get(name)
        if not isinstance(record, Mapping):
            return None
        if not _authenticate_suite_log(path.parent, name, record, required_exit_zero):
            return None
    focused_pass = payload["focused_tests_pass"]
    if not focused_pass:
        return None
    try:
        base_log = (path.parent / "base-pytest.log").read_text(encoding="utf-8")
        branch_log = (path.parent / "branch-pytest.log").read_text(encoding="utf-8")
        focused_log = (path.parent / "focused-pytest.log").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    computed = classify_test_delta(
        extract_failed_nodes(base_log),
        extract_failed_nodes(branch_log),
        base_skips=extract_skipped_nodes(base_log),
        branch_skips=extract_skipped_nodes(branch_log),
        base_xfails=extract_xfailed_nodes(base_log),
        branch_xfails=extract_xfailed_nodes(branch_log),
    )
    if tuple(str(item) for item in unexplained) != computed.new_failures:
        return None
    declared_new = delta.get("new_failures")
    if tuple(str(item) for item in declared_new) != computed.new_failures:
        return None
    focused_exit = payload["focused"].get("exit_code")
    if focused_tests_passed(focused_log, int(focused_exit)) is not focused_pass:
        return None
    return AuthenticatedTestDelta(
        computed.new_failures,
        focused_pass,
    )

def _authenticate_suite_log(
    run_dir: Path,
    name: str,
    record: Mapping[str, object],
    required_exit_zero: bool,
) -> bool:
    exit_code = record.get("exit_code")
    if type(exit_code) is not int:
        return False
    if _abnormal_pytest_exit(exit_code):
        return False
    if required_exit_zero and exit_code != 0:
        return False
    log_record = record.get("log")
    if not isinstance(log_record, Mapping):
        return False
    log_path = run_dir / f"{name}-pytest.log"
    identity = _log_identity(log_path)
    if identity is None:
        return False
    if (
        log_record.get("sha256") != identity["sha256"]
        or log_record.get("size_bytes") != identity["size_bytes"]
    ):
        return False
    try:
        log_text = log_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    summary = parse_pytest_summary(log_text)
    if summary is None:
        return False
    declared_summary = record.get("summary")
    observed_summary = _summary_line(log_text)
    if type(declared_summary) is not str or declared_summary != observed_summary:
        return False
    collected = record.get("collected")
    if type(collected) is not int or collected != summary["collected"]:
        return False
    if summary["failed"] > 0 and exit_code == 0:
        return False
    if summary["failed"] == 0 and summary["errors"] == 0 and exit_code != 0:
        return False
    if required_exit_zero and (summary["failed"] > 0 or summary["errors"] > 0):
        return False
    return True

def _log_identity(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }

def _abnormal_pytest_exit(exit_code: int) -> bool:
    return exit_code < 0 or exit_code >= 128

def _summary_line(log_text: str) -> str | None:
    for line in reversed(log_text.splitlines()):
        stripped = line.strip().strip("=").strip()
        lowered = stripped.lower()
        if any(
            token in lowered
            for token in ("deselected", "selected in", "collected ", "cached in")
        ):
            continue
        if _PYTEST_SUMMARY.search(stripped) and parse_pytest_summary(stripped):
            return stripped
    return None

def _forbidden_json_keys(value: object) -> tuple[str, ...]:
    names: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_JSON_KEYS:
                names.append(str(key))
            names.extend(_forbidden_json_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            names.extend(_forbidden_json_keys(item))
    return tuple(names)

def _worktree_is_clean(repo: Path) -> bool:
    completed = subprocess.run(
        ("git", "-C", str(repo), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()

def _is_git_ancestor(ancestor: str, descendant: str, repo: Path | None = None) -> bool:
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(repo if repo is not None else _REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0

def _git_changed_files(
    base: str, head: str, repo: Path | None = None
) -> tuple[str, ...]:
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(repo if repo is not None else _REPO_ROOT),
            "diff",
            "--name-only",
            f"{base}..{head}",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in completed.stdout.splitlines() if line)
