
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BLIND_PARTITIONS = frozenset({"test", "test-blind", "blind", "holdout"})

BAND_TAIL = 5

BAND_SIGMA = 2.0

class SelectionError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class CandidateRecord:

    run_id: str
    evidence_dir: str
    seed: int
    stage: str
    metric: str
    score: float
    selected_step: int
    selected_checkpoint: str
    selected_checkpoint_sha256: str | None
    stage_start_checkpoint: str | None
    split_hash: str | None
    config_hash: str | None
    upstream_commit: str | None
    repo_commit: str | None
    world_size: int
    validation_tail: tuple[float, ...] = ()

    def as_provenance(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evidence_dir": self.evidence_dir,
            "seed": self.seed,
            "stage": self.stage,
            "metric": self.metric,
            "score": self.score,
            "selected_step": self.selected_step,
            "selected_checkpoint": self.selected_checkpoint,
            "selected_checkpoint_sha256": self.selected_checkpoint_sha256,
            "stage_start_checkpoint": self.stage_start_checkpoint,
            "split_hash": self.split_hash,
            "config_hash": self.config_hash,
            "upstream_commit": self.upstream_commit,
            "repo_commit": self.repo_commit,
            "world_size": self.world_size,
        }

@dataclass(frozen=True, slots=True)
class SelectionResult:

    selected: CandidateRecord
    candidates: tuple[CandidateRecord, ...]
    metric: str
    caveat: str
    reproduction_band: dict[str, Any]
    single_seed_warning: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(c.seed for c in self.candidates))

    @property
    def per_seed_scores(self) -> dict[int, float]:
        return {c.seed: c.score for c in self.candidates}

    def as_provenance(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "selected": self.selected.as_provenance(),
            "candidates": [c.as_provenance() for c in self.candidates],
            "seeds": list(self.seeds),
            "per_seed_scores": {str(k): v for k, v in self.per_seed_scores.items()},
            "caveat": self.caveat,
            "single_seed_warning": self.single_seed_warning,
            "reproduction_band": self.reproduction_band,
            "notes": list(self.notes),

            "reproduction": {
                "stage": self.selected.stage,
                "seed": self.selected.seed,
                "stage_start_checkpoint": self.selected.stage_start_checkpoint,
                "world_size": self.selected.world_size,
                "expected_metric": self.metric,
                "expected_score": self.selected.score,
                "band": self.reproduction_band,
            },
        }

def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise SelectionError(f"{path} is missing; the run's evidence is incomplete") from error

def load_candidate(evidence_dir, *, metric: str) -> CandidateRecord:

    evidence_dir = Path(evidence_dir)
    provenance = _read(evidence_dir / "provenance.json")
    verdict = _read(evidence_dir / "health_verdict.json")

    resolved = provenance.get("resolved_config") or {}
    partition = (resolved.get("validation") or {}).get("partition", "validation")
    if partition in BLIND_PARTITIONS:
        raise SelectionError(
            f"{evidence_dir} was scored on partition {partition!r}; selection "
            f"reads validation records only, and choosing on a blind split "
            f"makes the later confirmation a restatement of this choice"
        )
    if partition != "validation":
        raise SelectionError(
            f"{evidence_dir} reports partition {partition!r}; expected 'validation'"
        )

    checkpoint = verdict.get("selected_checkpoint")
    step = verdict.get("selected_step")
    score = verdict.get("selected_score")
    if checkpoint is None or step is None or score is None:
        raise SelectionError(
            f"{evidence_dir} has no selected checkpoint; a run that never "
            f"improved on its first evaluation cannot be a candidate"
        )

    resume_state = provenance.get("resume_state") or {}
    best = (verdict.get("checkpoints") or {}).get("best") or {}

    start = None
    resume_path = evidence_dir / "resume.json"
    if resume_path.exists():
        start = _read(resume_path).get("source")

    tail: tuple[float, ...] = ()
    validation_path = evidence_dir / "validation.json"
    if validation_path.exists():
        records = _read(validation_path)
        if isinstance(records, list):
            tail = tuple(
                float(r[metric]) for r in records[-BAND_TAIL:] if metric in r
            )

    return CandidateRecord(
        run_id=provenance.get("run_id", evidence_dir.name),
        evidence_dir=str(evidence_dir),
        seed=int(provenance["seed"]),
        stage=str(provenance["stage"]),
        metric=metric,
        score=float(score),
        selected_step=int(step),
        selected_checkpoint=str(checkpoint),
        selected_checkpoint_sha256=best.get("sha256"),
        stage_start_checkpoint=start,
        split_hash=resume_state.get("split_hash"),
        config_hash=resume_state.get("config_hash"),
        upstream_commit=resume_state.get("upstream_commit"),
        repo_commit=provenance.get("repo_commit"),
        world_size=int(provenance.get("world_size", 1)),
        validation_tail=tail,
    )

def select_candidate(candidates, *, metric: str) -> SelectionResult:

    candidates = list(candidates)
    if not candidates:
        raise SelectionError("no candidates were given; there is nothing to select")

    seeds = [c.seed for c in candidates]
    duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
    if duplicates:
        raise SelectionError(
            f"candidates share the same seed {duplicates}; a replay at an "
            f"identical seed confirms determinism but is not an independent "
            f"draw, and averaging it in would understate seed variance"
        )

    stages = {c.stage for c in candidates}
    if len(stages) > 1:
        raise SelectionError(
            f"candidates span more than one stage {sorted(stages)}; their scores "
            f"are not comparable"
        )

    splits = {c.split_hash for c in candidates}
    if len(splits) > 1:
        raise SelectionError(
            f"candidates were validated against different split hashes "
            f"{sorted(map(str, splits))}; the scores measure different data"
        )

    ordered = tuple(sorted(candidates, key=lambda c: (c.score, c.seed)))
    selected = ordered[0]

    warning = None
    if len(ordered) == 1:
        warning = (
            "one training seed only. This selects a checkpoint; it says nothing "
            "about seed-to-seed variability, and the score must not be reported "
            "as the method's expected performance."
        )

    return SelectionResult(
        selected=selected,
        candidates=ordered,
        metric=metric,
        caveat=(
            "The selected score is the MINIMUM over training seeds. It is a "
            "checkpoint choice and is not an estimate of the method's average "
            "performance -- a minimum over few draws is biased downward as a "
            "loss. Report every seed in `per_seed_scores` alongside it."
        ),
        reproduction_band=_band(selected),
        single_seed_warning=warning,
    )

def _band(candidate: CandidateRecord) -> dict[str, Any]:

    tail = [v for v in candidate.validation_tail if v == v]
    if len(tail) >= 2:
        spread = statistics.stdev(tail)
        method = f"within-run stability: {BAND_SIGMA} x stdev of the last {len(tail)} validation points"
    else:
        spread = abs(candidate.score) * 0.02
        method = (
            "within-run stability unavailable (fewer than two validation "
            "points); fell back to 2% of the selected score"
        )
    half = BAND_SIGMA * spread
    return {
        "low": candidate.score - half,
        "high": candidate.score + half,
        "centre": candidate.score,
        "half_width": half,
        "method": method,
        "is_example_level_bootstrap": False,
        "points_used": len(tail),
    }

@dataclass(frozen=True, slots=True)
class ReproductionPlan:

    resume: str
    seed: int
    stage: str
    world_size: int
    expected_metric: str
    expected_score: float
    band: dict[str, Any]
    selected_run_id: str
    world_size_changed: bool = False
    world_size_note: str | None = None

    def as_provenance(self) -> dict[str, Any]:
        return {
            "resume": self.resume,
            "seed": self.seed,
            "stage": self.stage,
            "recorded_world_size": self.world_size,
            "expected_metric": self.expected_metric,
            "expected_score": self.expected_score,
            "band": self.band,
            "selected_run_id": self.selected_run_id,
            "world_size_changed": self.world_size_changed,
            "world_size_note": self.world_size_note,
        }

def reproduction_plan(record: dict, *, stage: str, world_size: int | None = None):

    reproduction = record.get("reproduction") or {}
    selected = record.get("selected") or {}

    if str(reproduction.get("stage")) != str(stage):
        raise SelectionError(
            f"the record selects stage {reproduction.get('stage')!r} but this "
            f"run is configured for {stage!r}; a reproduction must rerun the "
            f"same stage"
        )

    start = reproduction.get("stage_start_checkpoint")
    if not start:
        raise SelectionError(
            "the record names no stage start point, so there is nothing to "
            "reproduce from. The selected run's resume.json is what records it; "
            "a run launched without --resume has no start point to return to."
        )

    produced = selected.get("selected_checkpoint")
    if produced and Path(start) == Path(produced):
        raise SelectionError(
            "the recorded start point is the selected run's own output. "
            "Training from there would append another full stage to a finished "
            "one and report it as a reproduction; the start point must be the "
            "checkpoint that stage resumed FROM."
        )

    recorded_world = int(reproduction.get("world_size", 0) or 0)
    changed = world_size is not None and world_size != recorded_world
    note = None
    if changed:
        note = (
            f"world size {recorded_world} -> {world_size}. Verify the effective "
            f"batch, data exposure and optimizer update count match before "
            f"comparing; otherwise record this as a separate run rather than a "
            f"reproduction."
        )

    return ReproductionPlan(
        resume=str(start),
        seed=int(reproduction["seed"]),
        stage=str(stage),
        world_size=recorded_world,
        expected_metric=str(reproduction.get("expected_metric", "")),
        expected_score=float(reproduction.get("expected_score", float("nan"))),
        band=dict(reproduction.get("band") or {}),
        selected_run_id=str(selected.get("run_id", "")),
        world_size_changed=changed,
        world_size_note=note,
    )
