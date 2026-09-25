
from __future__ import annotations

import hashlib
from pathlib import Path

from dive.benchmark.evaluators.binder import BinderRedesign
from dive.benchmark.paper_baseline import (
    FIXTURE_EVALUATOR_REGISTRY_SHA256,
    PaperBaselineError,
)
from dive.benchmark.paper_registry import (
    DEFAULT_PAPER_EVALUATOR_REGISTRY_PATH,
    PaperEvaluatorRegistry,
    load_paper_evaluator_registry,
    verify_paper_evaluator_registry,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_WORKER_PATH = (
    _REPO_ROOT / "scripts" / "emergent" / "paper_baseline_production_worker.py"
)
_AME_REQUIRED = (
    "binder_scRMSD_bb3",
    "binder_scRMSD_bb3_source_present",
    "exact_motif_sequence_recovery",
    "exact_motif_sequence_recovery_source_present",
    "ligand_clash",
    "ligand_clash_source_present",
    "min_ipAE",
    "min_ipAE_source_present",
    "motif_all_atom_rmsd",
    "motif_all_atom_rmsd_source_present",
    "redesign_id",
    "sequence",
)

def production_worker_sha256() -> str:
    return hashlib.sha256(PRODUCTION_WORKER_PATH.read_bytes()).hexdigest()

def authenticate_production_registry(
    path: Path | None = None,
) -> PaperEvaluatorRegistry:

    registry = load_paper_evaluator_registry(
        path if path is not None else DEFAULT_PAPER_EVALUATOR_REGISTRY_PATH
    )
    verify_paper_evaluator_registry(registry)
    if registry.semantic_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256:
        raise PaperBaselineError("fixture hash cannot authenticate as production")
    if registry.worker.sha256 == (
        "e19d578cab977060f8469bcc43b96fc218f7c53727e8b45893de90b7ae11ce95"
    ):
        raise PaperBaselineError("fixture hash cannot authenticate as production")
    if Path(registry.worker.path).resolve() != PRODUCTION_WORKER_PATH.resolve():
        raise PaperBaselineError("fixture hash cannot authenticate as production")
    if registry.worker.sha256 != production_worker_sha256():
        raise PaperBaselineError("production worker identity drifted")
    return registry

def binder_redesigns_from_track_e_eval(
    stats: dict[str, object],
    sequences: dict[str, object],
    *,
    sequence_type: str,
) -> tuple[BinderRedesign, ...]:
    family = stats.get(sequence_type)
    seq_rows = sequences.get(sequence_type)
    if not isinstance(family, dict) or not isinstance(seq_rows, list):
        raise PaperBaselineError("binder eval payload missing sequence type rows")
    complex_rows = family.get("complex_stats")
    rmsd_rows = family.get("rmsd_stats")
    if not isinstance(complex_rows, list) or not isinstance(rmsd_rows, list):
        raise PaperBaselineError("binder eval payload missing complex/rmsd stats")
    if len(complex_rows) != len(rmsd_rows) or len(complex_rows) != len(seq_rows):
        raise PaperBaselineError("binder redesign vectors have unequal length")
    if len(complex_rows) != 2:
        raise PaperBaselineError("binder production worker requires two redesigns")
    rows: list[BinderRedesign] = []
    for index, (complex_row, rmsd_row, sequence_row) in enumerate(
        zip(complex_rows, rmsd_rows, seq_rows, strict=True)
    ):
        if not isinstance(sequence_row, dict) or not sequence_row.get("seq"):
            raise PaperBaselineError("binder sequence row is incomplete")
        rows.append(
            BinderRedesign(
                f"{sequence_type}-{index}",
                str(sequence_row["seq"]),
                float(complex_row["i_pAE"]),
                float(complex_row["pLDDT"]),
                float(rmsd_row["binder_scRMSD_ca"]),
                {},
            )
        )
    return tuple(rows)

def map_ame_eval_to_raw(payload: dict[str, object]) -> dict[str, object]:
    missing = [name for name in _AME_REQUIRED if name not in payload]
    if missing:
        raise PaperBaselineError(
            f"incomplete AME production payload: missing {missing}"
        )
    return dict(payload)
