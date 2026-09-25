
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from dive.benchmark.task_cards import ContaminationStatus, ManifestPartition
from dive.data.contracts import CanonicalExample, Family
from dive.training.preflight import canonical_json_bytes

class ManifestError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class FamilyManifestBundle:
    family: Family
    public: tuple[CanonicalExample, ...]
    legacy_dev: tuple[CanonicalExample, ...]
    calibration: tuple[CanonicalExample, ...]
    sealed_parent_count: int
    sealed_bound: bool
    exclusions: tuple[tuple[str, str], ...]
    parse_success_rate: float
    counts: Mapping[ManifestPartition, int]
    manifest_sha256: str

def contamination_label(
    *,
    exact_pdb_overlap: bool,
    sequence_neighbor: bool,
    structure_neighbor: bool,
    same_epitope: bool,
    training_data_disclosed: bool,
) -> ContaminationStatus:
    if exact_pdb_overlap or same_epitope:
        return ContaminationStatus.KNOWN_OVERLAP
    if sequence_neighbor or structure_neighbor:
        return ContaminationStatus.NEAR_OVERLAP
    if not training_data_disclosed:
        return ContaminationStatus.UNKNOWN
    return ContaminationStatus.CLEAN_BY_DECLARED_DATA

def build_family_manifests(
    *,
    family: Family,
    public: Sequence[CanonicalExample],
    legacy_dev: Sequence[CanonicalExample],
    calibration: Sequence[CanonicalExample],
    sealed_parent_ids: Sequence[str],
    inspected_parent_ids: Iterable[str],
    exclusions: Sequence[tuple[str, str]] = (),
    eligible: int | None = None,
    parsed: int | None = None,
) -> FamilyManifestBundle:

    if not public:
        raise ManifestError("public partition must be non-empty")
    inspected = frozenset(inspected_parent_ids)
    sealed = tuple(sealed_parent_ids)
    leaked = inspected.intersection(sealed)
    if leaked:
        raise ManifestError(
            f"legacy inspected parents cannot enter sealed: {sorted(leaked)}"
        )
    _assert_family(family, public, "public")
    _assert_family(family, legacy_dev, "legacy-dev")
    _assert_family(family, calibration, "calibration")
    parents: dict[str, str] = {}
    for label, rows in (
        ("public", public),
        ("legacy-dev", legacy_dev),
        ("calibration", calibration),
    ):
        for example in rows:
            previous = parents.get(example.parent_id)
            if previous is not None and previous != label:
                raise ManifestError(
                    f"duplicate parent {example.parent_id!r} across {previous} and {label}"
                )
            parents[example.parent_id] = label
    for source_id, reason in exclusions:
        if not source_id or not reason:
            raise ManifestError("exclusions require a stable reason code")
    parsed_count = (
        parsed
        if parsed is not None
        else (len(public) + len(legacy_dev) + len(calibration))
    )

    eligible_count = eligible if eligible is not None else parsed_count
    if eligible_count <= 0:
        raise ManifestError("eligible row count must be positive")
    parse_success_rate = parsed_count / eligible_count
    if parse_success_rate < 0.95:
        raise ManifestError("parse success rate is below 95% after eligibility")
    payload = {
        "calibration": [example.example_id for example in calibration],
        "exclusions": [list(item) for item in exclusions],
        "family": str(family),
        "legacy_dev": [example.example_id for example in legacy_dev],
        "public": [example.example_id for example in public],
        "sealed_parent_count": len(sealed),
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return FamilyManifestBundle(
        family=family,
        public=tuple(public),
        legacy_dev=tuple(legacy_dev),
        calibration=tuple(calibration),
        sealed_parent_count=len(sealed),
        sealed_bound=True,
        exclusions=tuple(exclusions),
        parse_success_rate=parse_success_rate,
        counts={
            ManifestPartition.PUBLIC: len(public),
            ManifestPartition.LEGACY_DEV: len(legacy_dev),
            ManifestPartition.CALIBRATION: len(calibration),
            ManifestPartition.SEALED: len(sealed),
        },
        manifest_sha256=digest,
    )

def _assert_family(
    family: Family, rows: Sequence[CanonicalExample], label: str
) -> None:
    for example in rows:
        if example.family is not family:
            raise ManifestError(f"{label} row {example.example_id} is not {family}")
