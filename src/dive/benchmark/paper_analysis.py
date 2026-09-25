
from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260828
_PANEL_A_ARM = "base_common_exact_v1"

_LEGACY_PANEL_A_ARM = "base_common_v2_splice"
_PANEL_B_ARM = "base_common_native_v1"
_DENIED_PREFIXES: tuple[Path, ...] = ()
_BINARY_FAMILIES = frozenset({"binder", "ame"})
_CONTINUOUS_FAMILIES = frozenset({"antibody"})
_PRIMARY_NAMES = {
    "binder": "target_conditioned_success",
    "ame": "ame_motif_ligand_success",
    "antibody": "cdr_h3_ca_rmsd",
}
_STEPS = {"binder": 400, "ame": 100, "antibody": 100}
_FAN_OUT = {"binder": 2, "ame": 2, "antibody": 1}
_FAMILY_ORDER = ("binder", "ame", "antibody")
_ARTIFACTS = (
    "sufficient_statistics.json",
    "panel_a.json",
    "panel_a.md",
    "panel_b.json",
    "panel_b.md",
    "tidy.csv",
    "appendix.json",
)
_CROSS_FAMILY_NOTE = (
    "Families are not pooled; unequal step counts "
    "(binder 400, ame 100, antibody 100)."
)
_LIMITATIONS = MappingProxyType(
    {
        "structural_novelty": "not_measured",
        "exposure": "unknown",
        "temporal_clean": (),
    }
)
_REQUIRED_STATISTICS = frozenset(
    {
        "arm",
        "bootstrap",
        "compute",
        "cross_family",
        "cross_family_note",
        "diversity",
        "failures",
        "families",
        "limitations",
        "panel",
        "unavailable_metrics",
    }
)

@dataclass(frozen=True, slots=True)
class PaperCellRecord:
    parent_id: str
    example_id: str
    seed: int
    family: str
    completed: bool = True
    primary: object = None
    failure_reason: str | None = None
    generated_sequence: str | None = None
    generated_mask: tuple[object, ...] | None = None
    generated_ca: tuple[tuple[float, float, float], ...] | None = None
    atom_map: tuple[str, ...] | None = None
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    peak_memory_bytes: int = 0
    retries: int = 0
    evaluator_calls: int = 0
    redesigns: int = 0
    secondaries: Mapping[str, object] = MappingProxyType({})
    raw: Mapping[str, object] = MappingProxyType({})
    cell_id: str | None = None

@dataclass(frozen=True, slots=True)
class PaperSufficientStatistics:
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_mapping(self) -> dict[str, object]:
        return _canonical_mapping(self.payload)

@dataclass(frozen=True, slots=True)
class PaperArtifactSet:
    directory: Path
    contents: Mapping[str, bytes]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contents", MappingProxyType(dict(self.contents)))

def build_paper_sufficient_statistics(
    plan: object,
    records: Sequence[PaperCellRecord],
    *,
    stage_records: Mapping[str, object] | None = None,
    reconstruction: object | None = None,
) -> PaperSufficientStatistics:
    from dive.benchmark.paper_baseline import PaperPlanManifest

    if not isinstance(plan, PaperPlanManifest):
        _fail("build_paper_sufficient_statistics requires a PaperPlanManifest")
    _require_evaluated(stage_records)
    planned = tuple(plan.cells)
    if not planned:
        _fail("no observations")
    by_key: dict[tuple[str, str, str, int], PaperCellRecord] = {}
    planned_keys = {
        (cell.family, cell.parent_id, cell.example_id, cell.seed) for cell in planned
    }
    for record in records:
        if type(record) is not PaperCellRecord:
            _fail("records must be PaperCellRecord")
        key = (record.family, record.parent_id, record.example_id, record.seed)
        if key not in planned_keys:
            _fail(f"unplanned cell {key}")
        if key in by_key:
            _fail("duplicate cells")
        by_key[key] = record

    families = {
        family: tuple(cell for cell in planned if cell.family == family)
        for family in _family_names(planned)
    }
    family_rows: dict[str, dict[str, object]] = {}
    diversity: dict[str, dict[str, object]] = {}
    compute: dict[str, dict[str, object]] = {}
    failures: dict[str, dict[str, object]] = {}
    unavailable: list[dict[str, object]] = []
    for family, cells in families.items():
        resolved = [
            by_key.get((cell.family, cell.parent_id, cell.example_id, cell.seed))
            for cell in cells
        ]
        family_rows[family] = _family_statistics(family, cells, resolved)
        diversity[family] = _family_diversity(cells, resolved)
        compute[family] = _family_compute(family, cells, resolved, family_rows[family])
        failures[family] = {
            "n_failed": family_rows[family]["n_failed"],
            "failure_rate": family_rows[family]["failure_rate"],
            "failure_reasons": dict(family_rows[family]["failure_reasons"]),
        }
        unavailable.extend(_unavailable_rows(resolved))
    payload = {
        "schema_version": "dive-paper-sufficient-statistics-v1",
        "panel": plan.panel,
        "arm": plan.arm,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci": 0.95,
            "unit": "parent",
        },
        "families": family_rows,
        "diversity": diversity,
        "compute": compute,
        "failures": failures,
        "unavailable_metrics": unavailable,
        "limitations": {
            "structural_novelty": "not_measured",
            "exposure": "unknown",
            "temporal_clean": [],
        },
        "cross_family": "not_applicable",
        "cross_family_note": _CROSS_FAMILY_NOTE,
        "panel_b_citation_allowed": _citation_allowed_from(reconstruction),
    }
    return PaperSufficientStatistics(payload)

def render_paper_artifacts(
    statistics: PaperSufficientStatistics | Mapping[str, object],
    output_dir: Path,
    *,
    reconstruction: object | None = None,
    citation_allowed: bool | None = None,
) -> PaperArtifactSet:
    return _write_artifacts(
        _as_mapping(statistics),
        output_dir,
        reconstruction=reconstruction,
        citation_allowed=citation_allowed,
    )

def rebuild_paper_artifacts(
    statistics: Mapping[str, object],
    output_dir: Path,
) -> PaperArtifactSet:
    if not isinstance(statistics, Mapping):
        _fail("rebuild requires sufficient statistics")
    mapping = _as_mapping(statistics)
    missing = _REQUIRED_STATISTICS - mapping.keys()
    if missing:
        _fail(f"insufficient statistics {sorted(missing)}")
    return _write_artifacts(mapping, output_dir)

def _require_evaluated(stage_records: Mapping[str, object] | None) -> None:
    if stage_records is None:
        return
    value = None
    for key in ("EVALUATED", "evaluated"):
        if key in stage_records:
            value = stage_records[key]
            break
    if value is None:
        return
    status = getattr(value, "status", None)
    if status is None and isinstance(value, Mapping):
        status = value.get("status")
    if str(status) != "COMPLETE":
        _fail("EVALUATED stage is not COMPLETE")

def _family_names(cells: Sequence[object]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for family in _FAMILY_ORDER:
        if any(cell.family == family for cell in cells):
            names.append(family)
            seen.add(family)
    for cell in cells:
        if cell.family not in seen:
            names.append(cell.family)
            seen.add(cell.family)
    return tuple(names)

def _family_statistics(
    family: str,
    cells: Sequence[object],
    records: Sequence[PaperCellRecord | None],
) -> dict[str, object]:
    kind = _family_kind(family)
    primary = _PRIMARY_NAMES.get(family, family)
    steps = _STEPS.get(family)
    if steps is None:
        steps = int(cells[0].steps)
    n_cells = len(cells)
    failure_reasons: dict[str, int] = {}
    n_failed = 0
    n_successes = 0
    seed_values: list[tuple[str, str, float]] = []
    for cell, record in zip(cells, records, strict=True):
        failed, reason = _cell_failure(kind, record)
        if failed:
            n_failed += 1
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        if kind == "binary":
            value = _binary_value(record)
            if value != 0.0:
                n_successes += 1
            seed_values.append((cell.parent_id, cell.example_id, value))
        else:
            value = _continuous_value(record)
            if value is not None:
                seed_values.append((cell.parent_id, cell.example_id, value))
    from dive.benchmark.paper_baseline import nested_parent_means, parent_bootstrap_ci

    parent_means = nested_parent_means(tuple(seed_values)) if seed_values else ()
    parent_balanced = (
        math.fsum(parent_means) / len(parent_means) if parent_means else None
    )
    ci: tuple[float | None, float | None] = (None, None)
    if parent_means:
        ci = parent_bootstrap_ci(parent_means)
    row: dict[str, object] = {
        "family": family,
        "kind": kind,
        "primary": primary,
        "n_parents": len(parent_means),
        "n_cells": n_cells,
        "n_successes": n_successes if kind == "binary" else None,
        "n_failed": n_failed,
        "cell_rate": (n_successes / n_cells) if kind == "binary" else None,
        "parent_balanced": parent_balanced,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "mean": None,
        "median": None,
        "sd": None,
        "q25": None,
        "q75": None,
        "failure_rate": n_failed / n_cells,
        "failure_reasons": failure_reasons,
        "steps": steps,
    }
    if kind == "continuous" and parent_means:
        mean = parent_balanced
        row["mean"] = mean
        row["median"] = _median(parent_means)
        row["sd"] = _population_sd(parent_means, float(mean))
        row["q25"] = _quantile(parent_means, 0.25)
        row["q75"] = _quantile(parent_means, 0.75)
    return row

def _family_kind(family: str) -> str:
    if family in _CONTINUOUS_FAMILIES:
        return "continuous"
    if family in _BINARY_FAMILIES:
        return "binary"
    raise _error(f"unknown family {family}")

def _cell_failure(kind: str, record: PaperCellRecord | None) -> tuple[bool, str]:
    if record is None:
        return True, "missing_record"
    if _is_unavailable(record.primary):
        return True, str(getattr(record.primary, "reason_code", "unavailable"))
    if not record.completed:
        return True, record.failure_reason or "failed"
    if record.primary is None:
        return True, record.failure_reason or "missing_primary"
    if kind == "continuous" and _continuous_value(record) is None:
        return True, record.failure_reason or "non_finite"
    return False, ""

def _binary_value(record: PaperCellRecord | None) -> float:
    if record is None or not record.completed or record.primary is None:
        return 0.0
    if _is_unavailable(record.primary):
        return 0.0
    number = _as_float(record.primary)
    if number is None:
        return 0.0
    return number

def _continuous_value(record: PaperCellRecord | None) -> float | None:
    if record is None or not record.completed or record.primary is None:
        return None
    if _is_unavailable(record.primary):
        return None
    return _as_float(record.primary)

def _as_float(primary: object) -> float | None:
    if _is_metric_value(primary):
        number = primary.value
    else:
        number = primary
    try:
        value = float(number)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value

def _is_unavailable(value: object) -> bool:
    return (
        hasattr(value, "reason_code")
        and hasattr(value, "detail")
        and hasattr(value, "name")
    )

def _is_metric_value(value: object) -> bool:
    return (
        hasattr(value, "value")
        and hasattr(value, "direction")
        and hasattr(value, "name")
    )

def _family_diversity(
    cells: Sequence[object],
    records: Sequence[PaperCellRecord | None],
) -> dict[str, object]:
    by_parent: dict[str, list[PaperCellRecord]] = {}
    parent_order: list[str] = []
    for cell, record in zip(cells, records, strict=True):
        if cell.parent_id not in by_parent:
            by_parent[cell.parent_id] = []
            parent_order.append(cell.parent_id)
        if record is not None:
            by_parent[cell.parent_id].append(record)
    within: dict[str, dict[str, object]] = {}
    for parent_id in parent_order:
        members = sorted(
            by_parent[parent_id], key=lambda item: (item.example_id, item.seed)
        )
        identities: list[float] = []
        rmsds: list[float] = []
        sequence_pairs = 0
        for left, right in combinations(members, 2):
            if left.generated_sequence is None or right.generated_sequence is None:
                continue
            sequence_pairs += 1
            identity = _sequence_identity(left, right)
            if identity is not None:
                identities.append(identity)
            rmsd = _pair_ca_rmsd(left, right)
            if rmsd is not None:
                rmsds.append(rmsd)
        if identities:
            mean_identity: float | str = math.fsum(identities) / len(identities)
            collapse: float | str = sum(1 for item in identities if item == 1.0) / len(
                identities
            )
        else:
            mean_identity = "not_applicable"
            collapse = "not_applicable"
        within[parent_id] = {
            "n_pairs": sequence_pairs,
            "mean_sequence_identity": mean_identity,
            "collapse_rate": collapse,
            "ca_rmsd": (math.fsum(rmsds) / len(rmsds)) if rmsds else "not_applicable",
        }
    return {"within_parent": within}

def _sequence_identity(left: PaperCellRecord, right: PaperCellRecord) -> float | None:
    left_seq = left.generated_sequence
    right_seq = right.generated_sequence
    if not left_seq or not right_seq:
        return None
    left_mask = left.generated_mask
    right_mask = right.generated_mask
    if left_mask is None or right_mask is None:
        return None
    if len(left_mask) != len(left_seq) or len(right_mask) != len(right_seq):
        return None
    if len(left_mask) != len(right_mask):
        return None
    generated = [
        (a, b)
        for a, b, keep_left, keep_right in zip(
            left_seq, right_seq, left_mask, right_mask, strict=True
        )
        if bool(keep_left) and bool(keep_right)
    ]
    if not generated:
        return None
    return sum(a == b for a, b in generated) / len(generated)

def _pair_ca_rmsd(left: PaperCellRecord, right: PaperCellRecord) -> float | None:
    if (
        left.generated_mask is None
        or right.generated_mask is None
        or left.atom_map is None
        or right.atom_map is None
        or left.generated_ca is None
        or right.generated_ca is None
    ):
        return None
    if tuple(left.generated_mask) != tuple(right.generated_mask):
        return None
    if tuple(left.atom_map) != tuple(right.atom_map):
        return None
    if len(left.generated_ca) != len(right.generated_ca):
        return None
    if len(left.generated_ca) < 3:
        return None
    if not _finite_coords(left.generated_ca) or not _finite_coords(right.generated_ca):
        return None
    return _kabsch_rmsd(left.generated_ca, right.generated_ca)

def _finite_coords(coords: Sequence[Sequence[float]]) -> bool:
    return all(
        len(point) == 3 and all(math.isfinite(float(axis)) for axis in point)
        for point in coords
    )

def _kabsch_rmsd(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    import numpy as np

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    covariance = a.T @ b
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt = vt.copy()
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    diff = (a @ rotation) - b
    return float(np.sqrt(np.sum(diff * diff) / len(a)))

def _family_compute(
    family: str,
    cells: Sequence[object],
    records: Sequence[PaperCellRecord | None],
    statistics: Mapping[str, object],
) -> dict[str, object]:
    present = [record for record in records if record is not None]
    peak = max((record.peak_memory_bytes for record in present), default=0)
    return {
        "steps": statistics["steps"],
        "generation_cells": len(cells),
        "evaluator_fan_out": _FAN_OUT.get(family, 1),
        "wall_seconds": math.fsum(record.wall_seconds for record in present),
        "gpu_seconds": math.fsum(record.gpu_seconds for record in present),
        "peak_memory_bytes": peak,
        "retries": sum(record.retries for record in present),
        "failures": statistics["n_failed"],
        "evaluator_calls": sum(record.evaluator_calls for record in present),
        "redesigns": sum(record.redesigns for record in present),
    }

def _unavailable_rows(
    records: Sequence[PaperCellRecord | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for record in records:
        if record is None:
            continue
        fields: list[tuple[str, object]] = []
        if _is_unavailable(record.primary):
            fields.append((str(record.primary.name), record.primary))
        for collection in (record.secondaries, record.raw):
            if not isinstance(collection, Mapping):
                continue
            for name, value in collection.items():
                if _is_unavailable(value):
                    fields.append((str(name), value))
        for name, value in fields:
            key = (
                record.family,
                name,
                record.parent_id,
                record.example_id,
                record.seed,
                getattr(value, "reason_code", "unavailable"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "family": record.family,
                    "name": name,
                    "reason_code": str(key[-1]),
                    "parent_id": record.parent_id,
                    "example_id": record.example_id,
                    "seed": record.seed,
                }
            )
    rows.sort(
        key=lambda row: (row["family"], row["name"], row["parent_id"], row["seed"])
    )
    return rows

def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0

def _population_sd(values: Sequence[float], mean: float) -> float:
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))

def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

def _as_mapping(
    statistics: PaperSufficientStatistics | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(statistics, PaperSufficientStatistics):
        payload = statistics.to_mapping()
    elif isinstance(statistics, Mapping):
        payload = dict(statistics)
    else:
        raise _error("statistics must be a mapping")
    return _canonical_mapping(payload)

def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    if not isinstance(payload, Mapping):
        _fail("canonical JSON payload must be a mapping")
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _error(f"statistics are not canonicalizable: {error}") from error

def _canonical_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    return json.loads(_canonical_json_bytes(dict(payload)).decode("utf-8"))

def _write_artifacts(
    mapping: Mapping[str, object],
    output_dir: Path,
    *,
    reconstruction: object | None = None,
    citation_allowed: bool | None = None,
) -> PaperArtifactSet:
    destination = _refuse_denied(Path(output_dir))
    destination.mkdir(parents=True, exist_ok=True)
    allowed = _resolve_panel_b_citation(mapping, reconstruction, citation_allowed)
    files = {
        "sufficient_statistics.json": _canonical_json_bytes(dict(mapping)),
        "panel_a.json": _canonical_json_bytes(_panel_a_json(mapping)),
        "panel_a.md": _panel_a_markdown(mapping).encode("utf-8"),
        "panel_b.json": _canonical_json_bytes(
            _panel_b_json(
                mapping,
                citation_allowed=allowed,
                reconstruction=reconstruction,
            )
        ),
        "panel_b.md": _panel_b_markdown(
            citation_allowed=allowed, reconstruction=reconstruction
        ).encode("utf-8"),
        "tidy.csv": _tidy_csv(mapping),
        "appendix.json": _canonical_json_bytes(_appendix(mapping)),
    }
    contents: dict[str, bytes] = {}
    for name in _ARTIFACTS:
        path = destination / name
        payload = files[name]
        _write_create_new(path, payload)
        contents[name] = payload
    return PaperArtifactSet(destination, contents)

def _panel_a_json(mapping: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "dive-paper-baseline-panel-v1",
        "panel": mapping.get("panel", "A"),
        "arm": mapping.get("arm", _PANEL_A_ARM),
        "headline": "parent-balanced estimate with parent-bootstrap CI",
        "auxiliary": "cell rate",
        "bootstrap": mapping.get("bootstrap", {}),
        "families": mapping.get("families", {}),
        "limitations": mapping.get("limitations", dict(_LIMITATIONS)),
        "cross_family": mapping.get("cross_family", "not_applicable"),
        "cross_family_note": mapping.get("cross_family_note", _CROSS_FAMILY_NOTE),
    }

def _panel_b_json(
    mapping: Mapping[str, object],
    *,
    citation_allowed: bool,
    reconstruction: object | None = None,
) -> dict[str, object]:
    payload = {
        "schema_version": "dive-paper-baseline-panel-v1",
        "panel": "B",
        "arm": _PANEL_B_ARM,
        "citation_allowed": citation_allowed,
        "headline": "parent-balanced estimate with parent-bootstrap CI",
        "auxiliary": "cell rate",
        "limitations": mapping.get("limitations", dict(_LIMITATIONS)),
    }
    if not citation_allowed:
        payload["reason"] = "panel_b_requires_raw_reconstruction"
        return payload
    if reconstruction is not None:
        n_successes = getattr(reconstruction, "n_successes", None)
        n_cells = getattr(reconstruction, "n_cells", None)
        if n_successes is not None:
            payload["n_successes"] = n_successes
        if n_cells is not None:
            payload["n_cells"] = n_cells
    return payload

def _panel_a_markdown(mapping: Mapping[str, object]) -> str:
    lines = [
        f"# Paper baseline panel {mapping.get('panel', 'A')}",
        f"arm: {mapping.get('arm', _PANEL_A_ARM)}",
        "headline: parent-balanced estimate with parent-bootstrap CI",
        "auxiliary: cell rate",
        str(mapping.get("cross_family_note", _CROSS_FAMILY_NOTE)),
        "structural novelty: not_measured",
        "exposure: unknown",
        "temporal-clean: empty",
    ]
    families = mapping.get("families", {})
    if isinstance(families, Mapping):
        for family in _family_names_from_mapping(families):
            row = families[family]
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    f"## {family}",
                    f"parent-balanced: {row.get('parent_balanced')}",
                    f"cell rate: {row.get('cell_rate')}",
                    f"n_parents: {row.get('n_parents')}",
                    f"n_cells: {row.get('n_cells')}",
                    f"n_failed: {row.get('n_failed')}",
                    f"ci: [{row.get('ci_low')}, {row.get('ci_high')}]",
                    f"steps: {row.get('steps')}",
                ]
            )
    return "\n".join(lines) + "\n"

def _panel_b_markdown(
    *, citation_allowed: bool, reconstruction: object | None = None
) -> str:
    lines = [
        "# Paper baseline panel B",
        f"arm: {_PANEL_B_ARM}",
        f"citation_allowed: {'true' if citation_allowed else 'false'}",
    ]
    if not citation_allowed:
        lines.append("reason: panel_b_requires_raw_reconstruction")
    lines.append("structural novelty: not_measured")
    return "\n".join(lines) + "\n"

def _citation_allowed_from(reconstruction: object | None) -> bool:
    return getattr(reconstruction, "citation_allowed", False) is True

def _resolve_panel_b_citation(
    mapping: Mapping[str, object],
    reconstruction: object | None,
    citation_allowed: bool | None,
) -> bool:
    if reconstruction is not None:
        return _citation_allowed_from(reconstruction)
    if citation_allowed is True:
        return False
    return mapping.get("panel_b_citation_allowed") is True

def _tidy_csv(mapping: Mapping[str, object]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "panel",
            "arm",
            "family",
            "primary",
            "statistic",
            "value",
            "ci_low",
            "ci_high",
            "n_parents",
            "n_cells",
            "n_failed",
            "steps",
        )
    )
    families = mapping.get("families", {})
    if isinstance(families, Mapping):
        panel = mapping.get("panel", "A")
        arm = mapping.get("arm", _PANEL_A_ARM)
        for family in _family_names_from_mapping(families):
            row = families[family]
            if not isinstance(row, Mapping):
                continue
            statistics = ["parent_balanced"]
            if row.get("kind") == "binary":
                statistics.append("cell_rate")
            else:
                statistics.extend(("mean", "median", "sd", "q25", "q75"))
            statistics.append("failure_rate")
            for name in statistics:
                writer.writerow(
                    (
                        panel,
                        arm,
                        family,
                        row.get("primary"),
                        name,
                        row.get(name),
                        row.get("ci_low") if name == "parent_balanced" else "",
                        row.get("ci_high") if name == "parent_balanced" else "",
                        row.get("n_parents"),
                        row.get("n_cells"),
                        row.get("n_failed"),
                        row.get("steps"),
                    )
                )
    return buffer.getvalue().encode("utf-8")

def _appendix(mapping: Mapping[str, object]) -> dict[str, object]:
    return {
        "compute": mapping.get("compute", {}),
        "failures": mapping.get("failures", {}),
        "unavailable_metrics": mapping.get("unavailable_metrics", []),
    }

def _family_names_from_mapping(families: Mapping[str, object]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for family in _FAMILY_ORDER:
        if family in families:
            names.append(family)
            seen.add(family)
    for family in families:
        if family not in seen:
            names.append(family)
    return tuple(names)

def _refuse_denied(path: Path) -> Path:
    destination = Path(path)
    resolved = destination.resolve(strict=False)
    for denied in _DENIED_PREFIXES:
        prefix = Path(denied).resolve(strict=False)
        if resolved == prefix or prefix in resolved.parents:
            _fail(f"root is denied: {resolved}")
    return destination

def _write_create_new(path: Path, content: bytes) -> None:
    destination = _refuse_denied(Path(path))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o664)
    except FileExistsError as error:
        raise _error(f"create-new artifact already exists: {destination}") from error
    except OSError as error:
        raise _error(f"cannot write create-new artifact: {error}") from error
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("create-new write made no forward progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def _fail(message: str) -> None:
    raise _error(message)

def _error(message: str) -> Exception:
    from dive.benchmark.paper_baseline import PaperBaselineError

    return PaperBaselineError(message)
