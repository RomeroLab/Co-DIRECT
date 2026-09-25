
from __future__ import annotations

import json
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from dive.provenance import (
    FileEvidence,
    GitEvidence,
    ProvenanceError,
    verify_git_checkout,
)

_REQUIRED_CRITERIA = (
    "model_loaded_strictly",
    "autoencoder_loaded_strictly",
    "architecture_compatible",
    "generation_finite",
    "decoded_output_valid",
    "fits_24gb",
    "evaluator_executable",
    "provenance_complete",
)

_EXPECTED_TRACKS = {
    "protein_target": {"example_id": "02_PDL1", "architecture_v2": False},
    "ame": {"example_id": "M0024_1nzy_v3", "architecture_v2": True},
}
_VRAM_LIMIT_BYTES = 24 * 1024**3
_PINNED_UPSTREAM_COMMIT = "916eaaedce5b07c205efb6ef32370c01d366591e"
_VALIDATOR_SOURCE = "proteinfoundation.cli.validate.validate_evaluate"
_ALLOWED_VALIDATOR_WARNINGS = frozenset(
    {"Foldseek", "Shape complementarity (sc)"}
)

@dataclass(frozen=True, slots=True)
class StrictLoadEvidence:

    attempted: bool
    strict: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    error: str | None

    @property
    def passed(self) -> bool:
        return (
            self.attempted
            and self.strict
            and not self.missing_keys
            and not self.unexpected_keys
            and self.error is None
        )

@dataclass(frozen=True, slots=True)
class UpstreamDependencyEvidence:

    role: str
    path: Path
    size_bytes: int
    sha256: str

    @classmethod
    def from_path(cls, role: str, path: Path) -> UpstreamDependencyEvidence:
        file_evidence = FileEvidence.from_path(path)
        return cls(
            role,
            file_evidence.path,
            file_evidence.size_bytes,
            file_evidence.sha256,
        )

    def as_file_evidence(self) -> FileEvidence:
        return FileEvidence(self.path, self.size_bytes, self.sha256)

    def to_dict(self) -> dict[str, int | str]:
        return {"role": self.role, **self.as_file_evidence().to_dict()}

@dataclass(frozen=True, slots=True)
class TrackEvidence:

    track: str
    example_id: str
    seed: int
    resolved_config_path: Path
    resolved_config_sha256: str
    model_evidence: FileEvidence
    autoencoder_evidence: FileEvidence
    evaluator_evidence: FileEvidence
    model_loaded_strictly: bool
    autoencoder_loaded_strictly: bool
    architecture_compatible: bool
    generation_finite: bool
    decoded_output_valid: bool
    fits_24gb: bool
    evaluator_executable: bool
    provenance_complete: bool
    denoiser_calls: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    wall_seconds: float
    sampling_steps: int | None
    seconds_per_step: float | None
    stdout_path: Path
    stderr_path: Path
    output_paths: tuple[Path, ...]
    architecture_v2: bool | None = None
    architecture_observations: tuple[bool, ...] = ()
    model_load_evidence: StrictLoadEvidence | None = None
    autoencoder_load_evidence: StrictLoadEvidence | None = None
    generation_exit_code: int = 1
    evaluation_exit_code: int = 1
    official_validator_passed: bool = False
    evaluation_config_evidence: FileEvidence | None = None
    upstream_module_evidence: tuple[UpstreamDependencyEvidence, ...] = ()
    evaluator_asset_evidence: tuple[FileEvidence, ...] = ()
    generation_output_evidence: tuple[FileEvidence, ...] = ()
    evaluator_output_evidence: tuple[FileEvidence, ...] = ()
    required_evaluator_metrics: tuple[str, ...] = ()
    evaluator_metrics: tuple[tuple[str, float], ...] = ()
    evaluation_stdout_path: Path | None = None
    evaluation_stderr_path: Path | None = None
    upstream_checkout: GitEvidence | None = None
    official_validator_observed: bool = False
    official_validator_source: str | None = None
    official_validator_results: tuple[tuple[str, bool, str, bool], ...] = ()
    official_validator_all_passed: bool = False
    official_validator_has_errors: bool = True
    official_motif_ligand_contract: bool | None = None
    evaluator_tree_roots: tuple[tuple[str, Path], ...] = ()

    def __post_init__(self) -> None:

        object.__setattr__(
            self, "resolved_config_path", Path(self.resolved_config_path).resolve()
        )
        object.__setattr__(self, "output_paths", tuple(self.output_paths))
        for name in (
            "architecture_observations",
            "upstream_module_evidence",
            "evaluator_asset_evidence",
            "generation_output_evidence",
            "evaluator_output_evidence",
            "required_evaluator_metrics",
            "evaluator_metrics",
            "official_validator_results",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "evaluator_tree_roots",
            tuple(
                (str(role), Path(root).resolve())
                for role, root in self.evaluator_tree_roots
            ),
        )

    def to_dict(self) -> dict[str, object]:

        return {
            "track": self.track,
            "example_id": self.example_id,
            "seed": self.seed,
            "resolved_config_path": str(self.resolved_config_path),
            "resolved_config_sha256": self.resolved_config_sha256,
            "model_evidence": self.model_evidence.to_dict(),
            "autoencoder_evidence": self.autoencoder_evidence.to_dict(),
            "evaluator_evidence": self.evaluator_evidence.to_dict(),
            "model_loaded_strictly": self.model_loaded_strictly,
            "autoencoder_loaded_strictly": self.autoencoder_loaded_strictly,
            "architecture_compatible": self.architecture_compatible,
            "generation_finite": self.generation_finite,
            "decoded_output_valid": self.decoded_output_valid,
            "fits_24gb": self.fits_24gb,
            "evaluator_executable": self.evaluator_executable,
            "provenance_complete": self.provenance_complete,
            "denoiser_calls": self.denoiser_calls,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "wall_seconds": self.wall_seconds,
            "sampling_steps": self.sampling_steps,
            "seconds_per_step": self.seconds_per_step,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "output_paths": [str(path) for path in self.output_paths],
            "architecture_v2": self.architecture_v2,
            "architecture_observations": list(self.architecture_observations),
            "model_load_evidence": _strict_load_dict(self.model_load_evidence),
            "autoencoder_load_evidence": _strict_load_dict(
                self.autoencoder_load_evidence
            ),
            "generation_exit_code": self.generation_exit_code,
            "evaluation_exit_code": self.evaluation_exit_code,
            "official_validator_passed": self.official_validator_passed,
            "official_validator_observed": self.official_validator_observed,
            "official_validator_source": self.official_validator_source,
            "official_validator_results": [
                {
                    "name": name,
                    "passed": passed,
                    "message": message,
                    "is_warning": is_warning,
                }
                for name, passed, message, is_warning in self.official_validator_results
            ],
            "official_validator_all_passed": self.official_validator_all_passed,
            "official_validator_has_errors": self.official_validator_has_errors,
            "official_motif_ligand_contract": self.official_motif_ligand_contract,
            "upstream_checkout": (
                self.upstream_checkout.to_dict() if self.upstream_checkout else None
            ),
            "evaluator_tree_roots": {
                role: str(root) for role, root in self.evaluator_tree_roots
            },
            "evaluation_config_evidence": (
                self.evaluation_config_evidence.to_dict()
                if self.evaluation_config_evidence
                else None
            ),
            "upstream_module_evidence": [
                item.to_dict() for item in self.upstream_module_evidence
            ],
            "evaluator_asset_evidence": [
                item.to_dict() for item in self.evaluator_asset_evidence
            ],
            "generation_output_evidence": [
                item.to_dict() for item in self.generation_output_evidence
            ],
            "evaluator_output_evidence": [
                item.to_dict() for item in self.evaluator_output_evidence
            ],
            "required_evaluator_metrics": list(self.required_evaluator_metrics),
            "evaluator_metrics": dict(self.evaluator_metrics),
            "evaluation_stdout_path": (
                str(self.evaluation_stdout_path)
                if self.evaluation_stdout_path is not None
                else None
            ),
            "evaluation_stderr_path": (
                str(self.evaluation_stderr_path)
                if self.evaluation_stderr_path is not None
                else None
            ),
        }

    def to_json(self) -> str:

        return json.dumps(self.to_dict(), sort_keys=True)

@dataclass(frozen=True, slots=True)
class TrackVerdict:

    status: Literal["PASS", "BLOCKED"]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:

        return {"status": self.status, "blockers": list(self.blockers)}

    def to_json(self) -> str:

        return json.dumps(self.to_dict(), sort_keys=True)

def compute_track_verdict(evidence: TrackEvidence) -> TrackVerdict:

    expected = _EXPECTED_TRACKS.get(evidence.track)
    generation_valid = _decoded_outputs_are_valid(
        evidence.generation_output_evidence, track=evidence.track
    )
    metric_values = _read_required_metrics(
        evidence.evaluator_output_evidence, evidence.required_evaluator_metrics
    )
    derived = {
        "model_loaded_strictly": bool(
            evidence.model_load_evidence and evidence.model_load_evidence.passed
        ),
        "autoencoder_loaded_strictly": bool(
            evidence.autoencoder_load_evidence
            and evidence.autoencoder_load_evidence.passed
        ),
        "architecture_compatible": bool(
            expected
            and evidence.architecture_v2 == expected["architecture_v2"]
            and evidence.architecture_observations
            and all(evidence.architecture_observations)
        ),
        "generation_finite": generation_valid,
        "decoded_output_valid": generation_valid,
        "fits_24gb": max(
            evidence.peak_allocated_bytes, evidence.peak_reserved_bytes
        ) < _VRAM_LIMIT_BYTES,
        "evaluator_executable": bool(
            evidence.evaluation_exit_code == 0
            and evidence.official_validator_passed
            and _validator_observation_is_current(evidence)
            and evidence.required_evaluator_metrics
            and metric_values == evidence.evaluator_metrics
        ),
        "provenance_complete": _provenance_is_current(evidence),
    }
    blockers = [
        criterion
        for criterion in _REQUIRED_CRITERIA
        if not derived[criterion] or getattr(evidence, criterion) != derived[criterion]
    ]
    if expected is None or evidence.example_id != expected["example_id"]:
        blockers.append("example_id")
    if evidence.seed != 5:
        blockers.append("seed")
    if evidence.sampling_steps != 20:
        blockers.append("sampling_steps")
    if evidence.denoiser_calls <= 0:
        blockers.append("denoiser_calls")
    if evidence.generation_exit_code != 0:
        blockers.append("generation_exit_code")
    if evidence.seconds_per_step is None or not math.isclose(
        evidence.seconds_per_step, evidence.wall_seconds / 20, rel_tol=1e-9
    ):
        blockers.append("seconds_per_step")
    blockers_tuple = tuple(dict.fromkeys(blockers))
    status: Literal["PASS", "BLOCKED"] = "BLOCKED" if blockers_tuple else "PASS"
    return TrackVerdict(status=status, blockers=blockers_tuple)

def _provenance_is_current(evidence: TrackEvidence) -> bool:

    try:
        if not _checkout_is_current(evidence):
            return False
        if not _dependencies_are_current(evidence):
            return False
        for recorded in (
            evidence.model_evidence,
            evidence.autoencoder_evidence,
            evidence.evaluator_evidence,
            *evidence.evaluator_asset_evidence,
            *evidence.generation_output_evidence,
            *evidence.evaluator_output_evidence,
        ):
            if FileEvidence.from_path(recorded.path) != recorded:
                return False
        if evidence.track == "protein_target" and not evidence.evaluator_tree_roots:
            return False
        if evidence.evaluator_tree_roots and not _evaluator_trees_are_current(evidence):
            return False
        current_config = FileEvidence.from_path(evidence.resolved_config_path)
    except (OSError, ProvenanceError):
        return False
    if current_config.sha256 != evidence.resolved_config_sha256:
        return False
    if evidence.evaluation_config_evidence is None:
        return False
    try:
        return (
            FileEvidence.from_path(evidence.evaluation_config_evidence.path)
            == evidence.evaluation_config_evidence
        )
    except (OSError, ProvenanceError):
        return False

def aggregate_track_runtime(
    generation: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> TrackEvidence:

    track = str(contract["track"])
    for field in ("track", "example_id", "seed"):
        if generation.get(field) != evaluation.get(field):
            raise ValueError(f"generation/evaluation {field} mismatch")
    generation_outputs = tuple(
        FileEvidence.from_path(Path(path)) for path in generation["output_paths"]
    )
    evaluator_outputs = tuple(
        FileEvidence.from_path(Path(path)) for path in evaluation["output_paths"]
    )
    required_metrics = tuple(contract.get("required_evaluator_metrics", ()))
    metrics = _read_required_metrics(evaluator_outputs, required_metrics)
    decoded_valid = _decoded_outputs_are_valid(generation_outputs, track=track)
    model_load = _strict_load_from_record(generation["strict_load_audit"]["model"])
    autoencoder_load = _strict_load_from_record(
        generation["strict_load_audit"]["autoencoder"]
    )
    peak_allocated = max(
        int(generation["peak_allocated_bytes"]),
        int(evaluation["peak_allocated_bytes"]),
    )
    peak_reserved = max(
        int(generation["peak_reserved_bytes"]),
        int(evaluation["peak_reserved_bytes"]),
    )
    validator = contract.get("official_validator")
    if not isinstance(validator, Mapping):
        raise ProvenanceError("official validator observation is unavailable")
    validator_results = _validated_validator_results(validator, track=track)
    validator_observed = True
    validator_passed = True
    evaluator_ok = bool(
        int(evaluation["exit_code"]) == 0
        and evaluation.get("official_validator_passed") is True
        and validator_passed
        and required_metrics
        and len(metrics) == len(required_metrics)
    )
    generation_config = FileEvidence.from_path(Path(generation["resolved_config_path"]))
    evaluation_config = FileEvidence.from_path(Path(evaluation["resolved_config_path"]))
    if generation_config.sha256 != generation.get("resolved_config_sha256"):
        raise ProvenanceError("generation resolved config changed before aggregation")
    if evaluation_config.sha256 != evaluation.get("resolved_config_sha256"):
        raise ProvenanceError("evaluation resolved config changed before aggregation")
    upstream_record = contract.get("upstream")
    if not isinstance(upstream_record, Mapping):
        raise ProvenanceError("upstream checkout evidence is required")
    if upstream_record.get("commit") != _PINNED_UPSTREAM_COMMIT:
        raise ProvenanceError(
            "upstream commit mismatch: expected "
            f"{_PINNED_UPSTREAM_COMMIT}, recorded {upstream_record.get('commit')}"
        )
    if upstream_record.get("clean") is not True:
        raise ProvenanceError("upstream checkout was not recorded clean")
    upstream_checkout = verify_git_checkout(
        Path(str(upstream_record["path"])), _PINNED_UPSTREAM_COMMIT
    )
    upstream_modules = _dependencies_from_contract(
        contract.get("official_modules"), track=track, checkout=upstream_checkout.path
    )
    evaluator_assets = tuple(
        _file_from_record(record)
        for _, record in sorted(contract.get("evaluator_assets", {}).items())
    )
    evaluator_tree_roots = tuple(
        (str(role), Path(str(root)).resolve())
        for role, root in sorted(contract.get("evaluator_tree_roots", {}).items())
    )
    evidence = TrackEvidence(
        track=track,
        example_id=str(generation["example_id"]),
        seed=int(generation["seed"]),
        resolved_config_path=generation_config.path,
        resolved_config_sha256=generation_config.sha256,
        model_evidence=_file_from_record(generation["model_evidence"]),
        autoencoder_evidence=_file_from_record(generation["autoencoder_evidence"]),
        evaluator_evidence=_file_from_record(generation["evaluator_evidence"]),
        model_loaded_strictly=model_load.passed,
        autoencoder_loaded_strictly=autoencoder_load.passed,
        architecture_compatible=bool(
            generation.get("architecture_compatible")
            and evaluation.get("architecture_compatible")
        ),
        generation_finite=decoded_valid,
        decoded_output_valid=decoded_valid,
        fits_24gb=max(peak_allocated, peak_reserved) < _VRAM_LIMIT_BYTES,
        evaluator_executable=evaluator_ok,
        provenance_complete=True,
        denoiser_calls=int(generation["denoiser_calls"]),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        wall_seconds=float(generation["wall_seconds"]),
        sampling_steps=int(generation["sampling_steps"]),
        seconds_per_step=float(generation["seconds_per_step"]),
        stdout_path=Path(generation["stdout_path"]),
        stderr_path=Path(generation["stderr_path"]),
        output_paths=tuple(item.path for item in (*generation_outputs, *evaluator_outputs)),
        architecture_v2=bool(contract["architecture_v2"]),
        architecture_observations=(
            bool(generation.get("architecture_compatible")),
            bool(evaluation.get("architecture_compatible")),
        ),
        model_load_evidence=model_load,
        autoencoder_load_evidence=autoencoder_load,
        generation_exit_code=int(generation["exit_code"]),
        evaluation_exit_code=int(evaluation["exit_code"]),
        official_validator_passed=bool(
            evaluation.get("official_validator_passed") and validator_passed
        ),
        official_validator_observed=validator_observed,
        official_validator_source=(
            str(validator["source"]) if validator_observed else None
        ),
        official_validator_results=validator_results,
        official_validator_all_passed=bool(validator["all_passed"]),
        official_validator_has_errors=bool(validator["has_errors"]),
        official_motif_ligand_contract=(
            validator.get("motif_ligand_contract") is True
            if track == "ame" and isinstance(validator, Mapping)
            else None
        ),
        upstream_checkout=upstream_checkout,
        evaluation_config_evidence=evaluation_config,
        upstream_module_evidence=upstream_modules,
        evaluator_asset_evidence=evaluator_assets,
        evaluator_tree_roots=evaluator_tree_roots,
        generation_output_evidence=generation_outputs,
        evaluator_output_evidence=evaluator_outputs,
        required_evaluator_metrics=required_metrics,
        evaluator_metrics=metrics,
        evaluation_stdout_path=Path(evaluation["stdout_path"]),
        evaluation_stderr_path=Path(evaluation["stderr_path"]),
    )
    return dataclass_replace_provenance(evidence)

def dataclass_replace_provenance(evidence: TrackEvidence) -> TrackEvidence:

    from dataclasses import replace

    return replace(evidence, provenance_complete=_provenance_is_current(evidence))

def _expected_dependency_paths(track: str, checkout: Path) -> dict[str, Path]:

    if track not in _EXPECTED_TRACKS:
        return {}
    root = Path(checkout).resolve()
    pipeline = (
        "search_ame_local_pipeline.yaml"
        if track == "ame"
        else "search_binder_local_pipeline.yaml"
    )
    evaluator_pipeline = (
        "pipeline/ame/ame_evaluate.yaml"
        if track == "ame"
        else "pipeline/binder/binder_evaluate.yaml"
    )
    return {
        "generate": root / "src/proteinfoundation/generate.py",
        "evaluate": root / "src/proteinfoundation/evaluate.py",
        "validate": root / "src/proteinfoundation/cli/validate.py",
        "pipeline": root / "configs" / pipeline,
        "evaluator_pipeline": root / "configs" / evaluator_pipeline,
    }

def _dependencies_from_contract(
    raw_dependencies: object, *, track: str, checkout: Path
) -> tuple[UpstreamDependencyEvidence, ...]:
    if not isinstance(raw_dependencies, Mapping):
        raise ProvenanceError("official dependency role mapping is required")
    expected = _expected_dependency_paths(track, checkout)
    if set(raw_dependencies) != set(expected):
        raise ProvenanceError(
            "official dependency roles mismatch: expected "
            f"{sorted(expected)}, observed {sorted(map(str, raw_dependencies))}"
        )
    dependencies: list[UpstreamDependencyEvidence] = []
    for role, canonical_path in expected.items():
        raw = raw_dependencies[role]
        if not isinstance(raw, Mapping):
            raise ProvenanceError(f"official dependency record is malformed: {role}")
        evidence = _file_from_record(raw)
        if evidence.path != canonical_path.resolve():
            raise ProvenanceError(
                f"official dependency canonical path mismatch for {role}: "
                f"expected {canonical_path.resolve()}, observed {evidence.path}"
            )
        dependencies.append(
            UpstreamDependencyEvidence(
                role, evidence.path, evidence.size_bytes, evidence.sha256
            )
        )
    return tuple(dependencies)

def _dependencies_are_current(evidence: TrackEvidence) -> bool:
    if evidence.upstream_checkout is None:
        return False
    expected = _expected_dependency_paths(
        evidence.track, evidence.upstream_checkout.path
    )
    if not all(
        isinstance(dependency, UpstreamDependencyEvidence)
        for dependency in evidence.upstream_module_evidence
    ):
        return False
    observed = {
        dependency.role: dependency
        for dependency in evidence.upstream_module_evidence
    }
    if len(observed) != len(evidence.upstream_module_evidence):
        return False
    if set(observed) != set(expected):
        return False
    try:
        for role, canonical_path in expected.items():
            dependency = observed[role]
            if dependency.path != canonical_path.resolve():
                return False
            if FileEvidence.from_path(dependency.path) != dependency.as_file_evidence():
                return False
    except (OSError, ProvenanceError):
        return False
    return True

def _checkout_is_current(evidence: TrackEvidence) -> bool:
    checkout = evidence.upstream_checkout
    if (
        not isinstance(checkout, GitEvidence)
        or checkout.commit != _PINNED_UPSTREAM_COMMIT
    ):
        return False
    try:
        return (
            verify_git_checkout(checkout.path, _PINNED_UPSTREAM_COMMIT) == checkout
        )
    except ProvenanceError:
        return False

def _derive_validator_summary(
    results: Sequence[tuple[object, object, object, object]],
) -> tuple[bool, bool]:
    if not results:
        raise ProvenanceError("official validator results are empty")
    names: set[str] = set()
    normalized: list[tuple[str, bool, str, bool]] = []
    for result in results:
        if not isinstance(result, tuple) or len(result) != 4:
            raise ProvenanceError("official validator result is malformed")
        name, passed, message, is_warning = result
        if (
            not isinstance(name, str)
            or not name.strip()
            or type(passed) is not bool
            or not isinstance(message, str)
            or not message.strip()
            or type(is_warning) is not bool
        ):
            raise ProvenanceError("official validator result is malformed")
        if name in names:
            raise ProvenanceError(f"duplicate official validator result: {name}")
        names.add(name)
        if is_warning and name not in _ALLOWED_VALIDATOR_WARNINGS:
            raise ProvenanceError(f"unapproved official validator warning: {name}")
        normalized.append((name, passed, message, is_warning))
    all_passed = all(passed or is_warning for _, passed, _, is_warning in normalized)
    has_errors = any(
        not passed and not is_warning for _, passed, _, is_warning in normalized
    )
    return all_passed, has_errors

def _validated_validator_results(
    observation: object, *, track: str
) -> tuple[tuple[str, bool, str, bool], ...]:
    if not isinstance(observation, Mapping):
        raise ProvenanceError("official validator observation is unavailable")
    if observation.get("observed") is not True:
        raise ProvenanceError("official validator was not observed")
    if observation.get("source") != _VALIDATOR_SOURCE:
        raise ProvenanceError("official validator source is not canonical")
    if type(observation.get("all_passed")) is not bool or type(
        observation.get("has_errors")
    ) is not bool:
        raise ProvenanceError("official validator summary is malformed")
    raw_results = observation.get("results")
    if not isinstance(raw_results, list):
        raise ProvenanceError("official validator results are malformed")
    results: list[tuple[str, bool, str, bool]] = []
    for raw in raw_results:
        if not isinstance(raw, Mapping):
            raise ProvenanceError("official validator result is malformed")
        values = (
            raw.get("name"),
            raw.get("passed"),
            raw.get("message"),
            raw.get("is_warning"),
        )
        _derive_validator_summary((values,))
        results.append(values)
    normalized = tuple(results)
    all_passed, has_errors = _derive_validator_summary(normalized)
    if (
        observation["all_passed"] is not all_passed
        or observation["has_errors"] is not has_errors
    ):
        raise ProvenanceError("official validator summary contradicts persisted results")
    if not all_passed or has_errors:
        raise ProvenanceError("official validator results did not pass")
    if track == "ame" and observation.get("motif_ligand_contract") is not True:
        raise ProvenanceError("official AME motif/ligand contract was not observed")
    return normalized

def _validator_observation_is_current(evidence: TrackEvidence) -> bool:
    if (
        evidence.official_validator_observed is not True
        or evidence.official_validator_source != _VALIDATOR_SOURCE
        or (
            evidence.track == "ame"
            and evidence.official_motif_ligand_contract is not True
        )
        or not _checkout_is_current(evidence)
        or not _dependencies_are_current(evidence)
    ):
        return False
    try:
        all_passed, has_errors = _derive_validator_summary(
            evidence.official_validator_results
        )
    except ProvenanceError:
        return False
    return bool(
        all_passed
        and not has_errors
        and evidence.official_validator_all_passed is all_passed
        and evidence.official_validator_has_errors is has_errors
    )

def _file_from_record(record: Mapping[str, Any]) -> FileEvidence:
    evidence = FileEvidence(
        path=Path(str(record["path"])).resolve(),
        size_bytes=int(record["size_bytes"]),
        sha256=str(record["sha256"]),
    )
    if FileEvidence.from_path(evidence.path) != evidence:
        raise ProvenanceError(f"recorded artifact changed: {evidence.path}")
    return evidence

def _strict_load_from_record(record: Mapping[str, Any]) -> StrictLoadEvidence:
    return StrictLoadEvidence(
        attempted=bool(record.get("attempted")),
        strict=record.get("strict") is True,
        missing_keys=tuple(record.get("missing_keys") or ()),
        unexpected_keys=tuple(record.get("unexpected_keys") or ()),
        error=record.get("error"),
    )

def _strict_load_dict(evidence: StrictLoadEvidence | None) -> dict[str, object] | None:
    if evidence is None:
        return None
    return {
        "attempted": evidence.attempted,
        "strict": evidence.strict,
        "missing_keys": list(evidence.missing_keys),
        "unexpected_keys": list(evidence.unexpected_keys),
        "error": evidence.error,
    }

def _evaluator_trees_are_current(evidence: TrackEvidence) -> bool:

    roots = tuple((role, Path(root)) for role, root in evidence.evaluator_tree_roots)
    resolved_roots = tuple((role, root.resolve()) for role, root in roots)
    for index, (_left_role, left) in enumerate(resolved_roots):
        if roots[index][1].is_symlink() or not left.is_dir():
            return False
        for _right_role, right in resolved_roots[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                return False
    expected = {
        asset.path: asset
        for asset in evidence.evaluator_asset_evidence
        if any(asset.path.is_relative_to(root) for _, root in resolved_roots)
    }
    observed: dict[Path, FileEvidence] = {}
    physical: set[tuple[int, int]] = set()
    try:
        for _role, root in resolved_roots:
            for directory, directory_names, file_names in os.walk(
                root, followlinks=False
            ):
                directory_path = Path(directory)
                for name in tuple(directory_names):
                    child = directory_path / name
                    if child.is_symlink():
                        return False
                for name in file_names:
                    child = directory_path / name
                    if child.is_symlink() or not child.is_file():
                        return False
                    stat = child.stat()
                    identity = (stat.st_dev, stat.st_ino)
                    if identity in physical:
                        return False
                    physical.add(identity)
                    observed[child.resolve()] = FileEvidence.from_path(child)
    except (OSError, ProvenanceError):
        return False
    return observed == expected

def _decoded_outputs_are_valid(
    outputs: Sequence[FileEvidence], *, track: str
) -> bool:

    del track
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBExceptions import PDBConstructionException

    structures = [item.path for item in outputs if item.path.suffix.lower() == ".pdb"]
    if not structures:
        return False
    for path in structures:
        try:
            structure = PDBParser(QUIET=True, PERMISSIVE=False).get_structure(
                "decoded", path
            )
            atoms = list(structure.get_atoms())
        except (OSError, UnicodeError, ValueError, TypeError, PDBConstructionException):
            return False
        if not atoms or not all(
            all(math.isfinite(float(value)) for value in atom.coord) for atom in atoms
        ):
            return False
        complete_chain = False
        for chain in structure.get_chains():
            protein_residues = [
                residue
                for residue in chain.get_residues()
                if residue.id[0] == " "
            ]
            if len(protein_residues) < 2:
                continue
            if all(
                {"N", "CA", "C", "O"} <= {atom.name for atom in residue}
                for residue in protein_residues
            ):
                complete_chain = True
                break
        if not complete_chain:
            return False
    return True

def _read_required_metrics(
    outputs: Sequence[FileEvidence], required: Sequence[str]
) -> tuple[tuple[str, float], ...]:
    if not required:
        return ()
    found: dict[str, float] = {}
    for output in outputs:
        if output.path.suffix.lower() != ".csv":
            continue
        try:
            with output.path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    for metric in required:
                        if metric in found or row.get(metric) in (None, ""):
                            continue
                        value = float(row[metric])
                        if math.isfinite(value):
                            found[metric] = value
        except (OSError, UnicodeError, ValueError):
            return ()
    if any(metric not in found for metric in required):
        return ()
    return tuple((metric, found[metric]) for metric in required)
