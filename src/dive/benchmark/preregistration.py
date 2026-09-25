
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dive.benchmark.sealed_guard import bind_sealed_envelope
from dive.training.preflight import canonical_json_bytes

class PreregistrationError(RuntimeError):
    pass

_PRIMARY_ENDPOINT = "target_level_joint_all_pass_rate"
_STATISTICAL_UNIT = "target_cluster"
_INVALID_POLICY = (
    "invalid_timeout_unsupported_remain_in_denominator_and_are_not_score_zero"
)

@dataclass(frozen=True, slots=True)
class Preregistration:
    payload: dict[str, object]
    sha256: str

    def as_mapping(self) -> dict[str, object]:
        return dict(self.payload)

def freeze_preregistration(
    *,
    primary_population: str,
    statistical_unit: str,
) -> Preregistration:
    if (
        "codirect" in primary_population.lower()
        and "hard" in primary_population.lower()
    ):
        raise PreregistrationError(
            "hard/coupled subsets must not be selected from Co-DIRECT results"
        )
    if primary_population != "full_sealed_population":
        raise PreregistrationError("primary population must be the full sealed set")
    if statistical_unit != _STATISTICAL_UNIT:
        raise PreregistrationError("residue/seed/sample are not statistical units")
    envelope = bind_sealed_envelope(
        view_semantic_hash="f5e54f8e0d91fe857f3d36135fb603f921523fa16563eb83d2b099e34ac8d253",
        view_partition_hash="527c359aefa234ea3015f645a8d128b587ad22c7e0b1aa263491c6a54a11fc03",
        family="all",
        metric_hash="aa" * 32,
        threshold_hash="bb" * 32,
        seed=20260826,
        budget_gpu_seconds=0,
    )
    payload = {
        "baseline_version_hash": "cc" * 32,
        "compute_budget_gpu_seconds": 0,
        "evaluator_version_hash": "dd" * 32,
        "invalid_timeout_unsupported_policy": _INVALID_POLICY,
        "minimum_practical_effect": {
            "binder": 0.05,
            "ame": 0.05,
            "antibody_cdr_h3_ca_rmsd_angstrom": -0.25,
        },
        "noninferiority_margins": {
            "binder": 0.02,
            "ame": 0.02,
            "antibody_cdr_h3_ca_rmsd_angstrom": 0.25,
        },
        "primary_endpoint": _PRIMARY_ENDPOINT,
        "primary_population": primary_population,
        "region_mask_hash": "ee" * 32,
        "representative_example_rule": (
            "pre-registered strata; never selected from Co-DIRECT outcomes"
        ),
        "sealed_envelope_sha256": envelope.envelope_sha256,
        "sealed_opened": False,
        "seeds": [20260826, 42001, 42002, 42003, 42004],
        "statistical_model": "target_cluster_bootstrap_parent_balanced",
        "statistical_unit": statistical_unit,
        "stopping_rule": "pre-registered compute budget; no peeking at sealed metrics",
        "target_cluster_manifest_hash": envelope.view_partition_hash,
        "task_card_hash": "ff" * 32,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    payload["sha256"] = digest
    return Preregistration(payload=payload, sha256=digest)
