
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

class BaselineRegistryError(RuntimeError):
    pass

class BaselineReadiness(StrEnum):
    REUSE_EXACT = "REUSE_EXACT"
    EXTEND = "EXTEND"
    REPLACE_WITH_REASON = "REPLACE_WITH_REASON"
    DO_NOT_TOUCH = "DO_NOT_TOUCH"
    READY_EXACT = "READY_EXACT"
    READY_PATCHED = "READY_PATCHED"
    MISSING_CODE = "MISSING_CODE"
    MISSING_WEIGHTS = "MISSING_WEIGHTS"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    LICENSE_BLOCKED = "LICENSE_BLOCKED"
    UNSUPPORTED_CONTRACT = "UNSUPPORTED_CONTRACT"
    SMOKE_FAILED = "SMOKE_FAILED"
    PLANNED_ONLY = "PLANNED_ONLY"

class ComparisonKind(StrEnum):
    NATIVE_PIPELINE = "native_pipeline"
    COMMON_REDESIGN = "common_redesign"

@dataclass(frozen=True, slots=True)
class BaselineRecord:
    baseline_id: str
    families: tuple[str, ...]
    role: str
    readiness: BaselineReadiness
    official_url: str
    commit: str | None
    weight_sha256: str | None
    reason: str
    native_pipeline: bool
    common_redesign: bool
    conditions: MappingProxyType

@dataclass(frozen=True, slots=True)
class ConditionMatrixRow:
    baseline_id: str
    family: str
    condition: str
    consumption: str

@dataclass(frozen=True, slots=True)
class BaselineRegistry:
    schema_version: str
    baselines: tuple[BaselineRecord, ...]

_CONDITIONS = (
    "target_coordinates",
    "target_sequence",
    "hotspots_epitope",
    "motif_atoms",
    "residue_identity",
    "residue_index",
    "rotamer",
    "ligand_coordinates",
    "ligand_bonds_charges",
    "framework",
    "cdr_mask",
    "native_dock",
    "auxiliary_hbond",
)

def _conditions(**values: str) -> MappingProxyType:
    payload = {name: "unsupported" for name in _CONDITIONS}
    payload.update(values)
    unknown = set(payload) - set(_CONDITIONS)
    if unknown:
        raise BaselineRegistryError(f"unknown conditions: {sorted(unknown)}")
    if set(payload.values()) - {"provided", "hidden", "derived", "unsupported"}:
        raise BaselineRegistryError("condition consumption vocabulary is closed")
    return MappingProxyType(payload)

def _record(**kwargs: object) -> BaselineRecord:
    conditions = kwargs.pop("conditions")
    return BaselineRecord(conditions=conditions, **kwargs)

def load_codirect_baseline_registry() -> BaselineRegistry:

    hidden_native = _conditions(
        target_coordinates="provided",
        target_sequence="provided",
        hotspots_epitope="provided",
        residue_identity="hidden",
        residue_index="hidden",
        rotamer="hidden",
        native_dock="hidden",
        auxiliary_hbond="unsupported",
    )
    ame_conditions = _conditions(
        ligand_coordinates="provided",
        ligand_bonds_charges="provided",
        motif_atoms="provided",
        residue_identity="provided",
        residue_index="hidden",
        rotamer="hidden",
        native_dock="hidden",
        target_coordinates="derived",
    )
    antibody_conditions = _conditions(
        target_coordinates="provided",
        hotspots_epitope="provided",
        framework="provided",
        cdr_mask="provided",
        residue_identity="hidden",
        native_dock="hidden",
        rotamer="hidden",
    )
    baselines = (
        _record(
            baseline_id="proteina_complexa_frozen_joint_v1",
            families=("binder", "ame", "antibody"),
            role="generator",
            readiness=BaselineReadiness.READY_EXACT,
            official_url="https://github.com/NVIDIA-Digital-Bio/proteina-complexa.git",
            commit="32b71ae1d9a8767414eae3921cf35969430db82f",
            weight_sha256="589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9",
            reason="authenticated common checkpoint on its v1 graph; paper-smoke-real-20260829q",
            native_pipeline=True,
            common_redesign=True,
            conditions=hidden_native,
        ),
        _record(
            baseline_id="proteina_complexa_common_if_1",
            families=("binder", "ame", "antibody"),
            role="inverse_folding",
            readiness=BaselineReadiness.READY_EXACT,
            official_url="https://github.com/dauparas/LigandMPNN.git",
            commit=None,
            weight_sha256="161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970",
            reason="common 1x LigandMPNN/ProteinMPNN redesign on frozen generator backbones",
            native_pipeline=False,
            common_redesign=True,
            conditions=hidden_native,
        ),
        _record(
            baseline_id="proteina_complexa_common_if_k",
            families=("binder", "ame", "antibody"),
            role="inverse_folding",
            readiness=BaselineReadiness.READY_EXACT,
            official_url="https://github.com/dauparas/LigandMPNN.git",
            commit=None,
            weight_sha256="161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970",
            reason="common Kx LigandMPNN/ProteinMPNN redesign on frozen generator backbones",
            native_pipeline=False,
            common_redesign=True,
            conditions=hidden_native,
        ),
        _record(
            baseline_id="proteinmpnn_common_v1",
            families=("binder", "antibody"),
            role="inverse_folding",
            readiness=BaselineReadiness.READY_EXACT,
            official_url="https://github.com/dauparas/ProteinMPNN.git",
            commit="8907e6671bfbfc92303b5f79c4b5e6ce47cdef57",
            weight_sha256="f28f40170e21858c5ff31ef50b6e63414ff76dc331b19f85aa8586a12031744a",
            reason="official ProteinMPNN ca_model v_48_020 authenticated in benchmark.yaml",
            native_pipeline=True,
            common_redesign=True,
            conditions=_conditions(
                target_coordinates="provided",
                residue_identity="derived",
                native_dock="hidden",
            ),
        ),
        _record(
            baseline_id="ligandmpnn_common_v1",
            families=("ame", "binder"),
            role="inverse_folding",
            readiness=BaselineReadiness.READY_EXACT,
            official_url="https://github.com/dauparas/LigandMPNN.git",
            commit=None,
            weight_sha256="161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970",
            reason="official LigandMPNN v_32_010_25 authenticated in benchmark.yaml",
            native_pipeline=True,
            common_redesign=True,
            conditions=ame_conditions,
        ),
        _record(
            baseline_id="rfdiffusion_binder_v1",
            families=("binder",),
            role="generator",
            readiness=BaselineReadiness.MISSING_WEIGHTS,
            official_url="https://github.com/RosettaCommons/RFdiffusion.git",
            commit=None,
            weight_sha256=None,
            reason="official weights unavailable; retained as B->S canonical baseline",
            native_pipeline=True,
            common_redesign=True,
            conditions=hidden_native,
        ),
        _record(
            baseline_id="rfdiffusion3_self_v1",
            families=("binder", "ame"),
            role="generator",
            readiness=BaselineReadiness.MISSING_CODE,
            official_url="https://github.com/RosettaCommons/RFdiffusion.git",
            commit=None,
            weight_sha256=None,
            reason="RF3 checkpoint bytes exist locally but no official runnable adapter",
            native_pipeline=True,
            common_redesign=True,
            conditions=hidden_native,
        ),
        _record(
            baseline_id="bindcraft_native_v1",
            families=("binder",),
            role="generator",
            readiness=BaselineReadiness.MISSING_CODE,
            official_url="https://github.com/nrbennet/dl_binder_design.git",
            commit=None,
            weight_sha256=None,
            reason="BindCraft source/weights not retained in this environment",
            native_pipeline=True,
            common_redesign=False,
            conditions=hidden_native,
        ),
        _record(
            baseline_id="boltzgen_self_v1",
            families=("binder", "antibody"),
            role="generator",
            readiness=BaselineReadiness.MISSING_CODE,
            official_url="https://github.com/jwohlwend/boltz.git",
            commit=None,
            weight_sha256=None,
            reason="BoltzGen not installed; AME inclusion requires unindexed catalytic proof",
            native_pipeline=True,
            common_redesign=True,
            conditions=hidden_native,
        ),
        _record(
            baseline_id="frameflow_motif_v1",
            families=("binder",),
            role="generator",
            readiness=BaselineReadiness.UNSUPPORTED_CONTRACT,
            official_url="https://github.com/microsoft/protein-frame-flow.git",
            commit="f50d8dbbdae827be291e9f73d732b61b195f8816",
            weight_sha256=None,
            reason="motif scaffolding cannot express the frozen target/hotspot task",
            native_pipeline=True,
            common_redesign=False,
            conditions=_conditions(hotspots_epitope="unsupported"),
        ),
        _record(
            baseline_id="chroma_conditioned_v1",
            families=("binder",),
            role="generator",
            readiness=BaselineReadiness.UNSUPPORTED_CONTRACT,
            official_url="https://github.com/generatebio/chroma.git",
            commit=None,
            weight_sha256=None,
            reason="public interface lacks required target-hotspot input",
            native_pipeline=True,
            common_redesign=False,
            conditions=_conditions(hotspots_epitope="unsupported"),
        ),
        _record(
            baseline_id="rfdiffusionaa_ligand_v1",
            families=("ame",),
            role="generator",
            readiness=BaselineReadiness.DO_NOT_TOUCH,
            official_url="https://github.com/baker-laboratory/rf_diffusion_all_atom.git",
            commit=None,
            weight_sha256=None,
            reason="provenance_failed; reviewed source-policy change only",
            native_pipeline=True,
            common_redesign=True,
            conditions=ame_conditions,
        ),
        _record(
            baseline_id="rfantibody_h3_v1",
            families=("antibody",),
            role="generator",
            readiness=BaselineReadiness.LICENSE_BLOCKED,
            official_url="https://github.com/RosettaCommons/RFantibody.git",
            commit="8fe311415754e0276d1a39c87c57e69c88927a2d",
            weight_sha256=None,
            reason="source MIT but weight bytes lack an explicit research-evaluation grant",
            native_pipeline=True,
            common_redesign=True,
            conditions=antibody_conditions,
        ),
        _record(
            baseline_id="diffab_cdr_v1",
            families=("antibody",),
            role="generator",
            readiness=BaselineReadiness.AUTH_REQUIRED,
            official_url="https://github.com/luost26/diffab.git",
            commit=None,
            weight_sha256=None,
            reason="eligible_for_weight_acquisition; acquisition plan not reviewed",
            native_pipeline=True,
            common_redesign=True,
            conditions=antibody_conditions,
        ),
        _record(
            baseline_id="abx_cdr_v1",
            families=("antibody",),
            role="generator",
            readiness=BaselineReadiness.DO_NOT_TOUCH,
            official_url="",
            commit=None,
            weight_sha256=None,
            reason="provenance_failed; no unofficial mirror substitution",
            native_pipeline=True,
            common_redesign=False,
            conditions=antibody_conditions,
        ),
    )
    ids = [item.baseline_id for item in baselines]
    if len(ids) != len(set(ids)):
        raise BaselineRegistryError("baseline ids must be unique")
    return BaselineRegistry(
        schema_version="dive-codirect-baselines-v1",
        baselines=baselines,
    )

def condition_matrix_rows(
    registry: BaselineRegistry,
) -> tuple[ConditionMatrixRow, ...]:
    rows: list[ConditionMatrixRow] = []
    for baseline in registry.baselines:
        for family in baseline.families:
            for condition, consumption in baseline.conditions.items():
                rows.append(
                    ConditionMatrixRow(
                        baseline_id=baseline.baseline_id,
                        family=family,
                        condition=condition,
                        consumption=str(consumption),
                    )
                )
    return tuple(rows)
