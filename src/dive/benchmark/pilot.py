
from __future__ import annotations

from dataclasses import dataclass

from dive.benchmark.adapters import (
    IdentityAdapter,
    UnsupportedContract,
    ame_ligand_motif_adapter,
    antibody_cdr_adapter,
    binder_hotspot_adapter,
)
from dive.benchmark.baselines import (
    BaselineReadiness,
    BaselineRegistry,
    load_codirect_baseline_registry,
)
from dive.benchmark.comparisons import (
    common_redesign_rows,
    native_pipeline_rows,
)
from dive.benchmark.compute_ledger import LedgerRow
from dive.benchmark.task_cards import (
    LigandGraph,
    SequencePolicy,
    TaskCard,
    binder_task_card,
    enzyme_task_card,
    vhh_cdr_h3_task_card,
)
from dive.data.contracts import CanonicalExample, ChainRecord, Family, LigandRecord

class PilotError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class SmokeResult:
    baseline_id: str
    family: str
    readiness: BaselineReadiness
    passed: bool
    score: float | None
    roundtrip_ok: bool
    ledger: LedgerRow
    conditions_consumed: tuple[str, ...]
    exception: str | None = None

@dataclass(frozen=True, slots=True)
class PilotReport:
    native_pipeline: tuple[object, ...]
    common_redesign: tuple[object, ...]
    failures_unsupported_unavailable: tuple[SmokeResult, ...]
    compute_ledger: tuple[LedgerRow, ...]
    family_smoke: dict[str, tuple[SmokeResult, ...]]
    sealed_opened: bool

def _binder_card() -> TaskCard:
    return binder_task_card(
        CanonicalExample(
            example_id="cal-binder-1",
            parent_id="cal-binder-parent",
            family=Family.BINDER,
            source_release="2024-06",
            pdb_id="1abc",
            assembly_id="1",
            deposition_date=None,
            chains=(
                ChainRecord("A", "target", "ACDE"),
                ChainRecord("B", "designed", "FGHI"),
            ),
            ligands=(),
        ),
        hotspot_residues=("A-1",),
    )

def _ame_card() -> TaskCard:
    return enzyme_task_card(
        CanonicalExample(
            example_id="cal-ame-1",
            parent_id="cal-ame-parent",
            family=Family.AME,
            source_release="2024-06",
            pdb_id="2enz",
            assembly_id="1",
            deposition_date=None,
            chains=(ChainRecord("A", "designed", "ACDEFGHIK"),),
            ligands=(LigandRecord("LIG", "CCO", None, ""),),
        ),
        ligand_graph=LigandGraph(
            ligand_id="LIG",
            atom_names=("C1", "O1"),
            elements=("C", "O"),
            bonds=((0, 1, 1),),
            formal_charges=(0, 0),
            stereochemistry="none",
        ),
        catalytic_atoms=("A-57-NE2",),
    )

def _antibody_card() -> TaskCard:
    return vhh_cdr_h3_task_card(
        CanonicalExample(
            example_id="cal-ab-1",
            parent_id="cal-ab-parent",
            family=Family.ANTIBODY,
            source_release="2024-06",
            pdb_id="3ab1",
            assembly_id="1",
            deposition_date=None,
            chains=(
                ChainRecord("H", "antibody", "EVQLVESGG"),
                ChainRecord("A", "antigen", "ACDE"),
            ),
            ligands=(),
        ),
        epitope_hotspots=("A-4",),
        cdr_h3_length=7,
        framework_id="vhh-fixed-1",
    )

def _card_for(family: str) -> TaskCard:
    if family == "binder":
        return _binder_card()
    if family == "ame":
        return _ame_card()
    if family == "antibody":
        return _antibody_card()
    raise PilotError(f"unsupported family {family!r}")

def _adapter_for(family: str):
    if family == "binder":
        return binder_hotspot_adapter()
    if family == "ame":
        return ame_ligand_motif_adapter(supports_unindexed_catalytic_atoms=True)
    return antibody_cdr_adapter()

def _ledger(
    *,
    baseline_id: str,
    family: str,
    passed: bool,
    reason_code: str,
    model_calls: int,
    exception: str | None = None,
) -> LedgerRow:
    return LedgerRow(
        run_id="codirect-public-dev-pilot-v1",
        method_id=baseline_id,
        family=family,
        target_cluster_id=f"cal-{family}",
        comparison="native_pipeline",
        generated_trajectories=int(passed),
        final_candidates=int(passed),
        sequences_per_structure=1,
        refold_samples_per_sequence=1,
        reward_evaluations=0,
        backward_calls=0,
        filter_calls=1,
        gpu_hours=0.0,
        wall_seconds=0.0,
        timeouts=0,
        ooms=0,
        invalids=0,
        model_calls=model_calls,
        predictor_draws=0,
        passed=passed,
        reason_code=reason_code,
        exception=exception,
    )

def smoke_baseline(baseline_id: str, card: TaskCard) -> SmokeResult:

    registry = {
        row.baseline_id: row for row in load_codirect_baseline_registry().baselines
    }
    try:
        record = registry[baseline_id]
    except KeyError as error:
        raise PilotError(f"unknown baseline {baseline_id!r}") from error
    family = str(card.family)
    if record.readiness is BaselineReadiness.UNSUPPORTED_CONTRACT:
        return SmokeResult(
            baseline_id=baseline_id,
            family=family,
            readiness=record.readiness,
            passed=False,
            score=None,
            roundtrip_ok=False,
            ledger=_ledger(
                baseline_id=baseline_id,
                family=family,
                passed=False,
                reason_code="UNSUPPORTED_CONTRACT",
                model_calls=0,
            ),
            conditions_consumed=(),
        )
    if record.readiness not in {
        BaselineReadiness.READY_EXACT,
        BaselineReadiness.READY_PATCHED,
    }:
        return SmokeResult(
            baseline_id=baseline_id,
            family=family,
            readiness=record.readiness,
            passed=False,
            score=None,
            roundtrip_ok=False,
            ledger=_ledger(
                baseline_id=baseline_id,
                family=family,
                passed=False,
                reason_code=record.readiness.value,
                model_calls=0,
            ),
            conditions_consumed=(),
        )
    adapter = _adapter_for(family)
    try:
        encoded = adapter.encode(card)
        IdentityAdapter().roundtrip(card)
        adapter.decode(
            encoded,
            mmcif="data_cal",
            fasta=">B\nAAAA",
            sequence_origin=SequencePolicy.SELF_GENERATED,
        )
    except UnsupportedContract as error:
        return SmokeResult(
            baseline_id=baseline_id,
            family=family,
            readiness=BaselineReadiness.UNSUPPORTED_CONTRACT,
            passed=False,
            score=None,
            roundtrip_ok=False,
            ledger=_ledger(
                baseline_id=baseline_id,
                family=family,
                passed=False,
                reason_code="UNSUPPORTED_CONTRACT",
                model_calls=0,
                exception=str(error),
            ),
            conditions_consumed=(),
            exception=str(error),
        )
    consumed = encoded.provided_conditions
    return SmokeResult(
        baseline_id=baseline_id,
        family=family,
        readiness=record.readiness,
        passed=True,
        score=None,
        roundtrip_ok=True,
        ledger=_ledger(
            baseline_id=baseline_id,
            family=family,
            passed=True,
            reason_code="ok",
            model_calls=1,
        ),
        conditions_consumed=consumed,
    )

def run_public_dev_pilot(
    registry: BaselineRegistry | None = None,
    *,
    include_sealed: bool = False,
) -> PilotReport:
    if include_sealed:
        raise PilotError(
            "sealed partition must stay closed during the public/dev pilot"
        )
    registry = registry or load_codirect_baseline_registry()
    family_smoke: dict[str, tuple[SmokeResult, ...]] = {}
    failures: list[SmokeResult] = []
    ledger: list[LedgerRow] = []
    for family in ("binder", "ame", "antibody"):
        card = _card_for(family)
        results: list[SmokeResult] = []
        for baseline in registry.baselines:
            if family not in baseline.families:
                continue
            result = smoke_baseline(baseline.baseline_id, card)
            results.append(result)
            ledger.append(result.ledger)
            if (
                result.readiness is not BaselineReadiness.READY_EXACT
                or not result.passed
            ):
                failures.append(result)
        family_smoke[family] = tuple(results)
    return PilotReport(
        native_pipeline=native_pipeline_rows(registry),
        common_redesign=common_redesign_rows(registry),
        failures_unsupported_unavailable=tuple(failures),
        compute_ledger=tuple(ledger),
        family_smoke=family_smoke,
        sealed_opened=False,
    )
