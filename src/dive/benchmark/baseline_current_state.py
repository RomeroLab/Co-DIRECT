
from __future__ import annotations

from dive.codirect_paths import cache_dir

from dataclasses import dataclass, field
from types import MappingProxyType

from dive.benchmark.baselines import load_codirect_baseline_registry

ARM_EVIDENCE = (
    cache_dir('emergent', 'benchmarks', 'external-arms-20260906a')
)
V2_EVIDENCE = (
    cache_dir('emergent', 'benchmarks', 'codirect-public-dev-v2')
)

class BaselineStateError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class CurrentState:
    baseline_id: str
    readiness_now: str
    executed: bool
    blocked: bool
    families_executed: tuple[str, ...]
    evidence_root: str
    note: str
    blocked_reason: str = ""
    acquisition: str = ""

    def __post_init__(self) -> None:
        if self.executed and not self.evidence_root:
            raise BaselineStateError(
                f"{self.baseline_id}: executed rows require an evidence root"
            )
        if self.executed and not self.families_executed:
            raise BaselineStateError(
                f"{self.baseline_id}: executed rows require at least one family"
            )
        if self.executed and self.blocked:
            raise BaselineStateError(
                f"{self.baseline_id}: a baseline cannot be both executed and blocked"
            )
        if self.blocked and not self.blocked_reason:
            raise BaselineStateError(
                f"{self.baseline_id}: blocked rows require a reason"
            )

def corrected_upstreams() -> MappingProxyType:

    return MappingProxyType({
        "boltzgen_self_v1": {
            "registered_url": "https://github.com/jwohlwend/boltz.git",
            "corrected_url": "https://github.com/HannesStark/boltzgen",
            "why": (
                "the registered URL is Boltz, a structure predictor. BoltzGen "
                "is a separate generative model with its own repository and "
                "PyPI distribution"
            ),
            "license": "MIT",
            "evidence": "pip install boltzgen; boltzgen --help exits 0",
        },
        "bindcraft_native_v1": {
            "registered_url": "https://github.com/nrbennet/dl_binder_design.git",
            "corrected_url": "https://github.com/martinpacesa/BindCraft",
            "why": (
                "the registered URL is the Baker-lab dl_binder_design pipeline, "
                "not BindCraft"
            ),
            "license": "MIT",
            "evidence": "GitHub API license field for martinpacesa/BindCraft",
        },
        "rfdiffusion3_self_v1": {
            "registered_url": "https://github.com/RosettaCommons/RFdiffusion.git",
            "corrected_url": "https://pypi.org/project/rc-foundry/",
            "why": (
                "the registered URL is RFdiffusion v1. RFdiffusion3 ships inside "
                "the IPD `rc-foundry` distribution as the `rfd3` package, which "
                "was already installed on this machine"
            ),
            "license": "BSD-3-Clause (Institute for Protein Design)",
            "evidence": (
                "rc_foundry-0.2.0.dist-info/METADATA; foundry checkpoint "
                "registry entry 'rfd3'"
            ),
        },
    })

def load_current_states() -> dict[str, CurrentState]:

    states = [

        CurrentState(
            "proteina_complexa_frozen_joint_v1", "READY_EXACT", True, False,
            ("binder", "ame", "antibody"), V2_EVIDENCE,
            "the system under test; self and common-IF arms both executed",
        ),
        CurrentState(
            "proteina_complexa_common_if_1", "READY_EXACT", True, False,
            ("binder", "ame", "antibody"), V2_EVIDENCE,
            "the common inverse folder actually used, at K=1",
        ),
        CurrentState(
            "proteina_complexa_common_if_k", "READY_EXACT", True, False,
            ("binder", "ame", "antibody"), V2_EVIDENCE,
            "K was never pinned in any frozen artifact, so amendment A4 pins it "
            "PROSPECTIVELY: K=8, declared 2026-09-07, before any K-arm number "
            "existed, and never presented as a past pre-registered value. The "
            "sequence is chosen by the inverse folder's own score, never by the "
            "evaluator's metric, so the arm is not an internal search judged by "
            "its own criterion. Control: the K run's first sample reproduces "
            "the K=1 arm's sequence identically on every backbone, which is "
            "what shows the two arms differ by K and nothing else. Executed on "
            "ALL THREE families the row declares: binder 22 cells, AME 20, "
            "antibody 26. K=8 helps on none of them; every interval contains "
            "zero.",
        ),
        CurrentState(
            "proteinmpnn_common_v1", "READY_EXACT", True, False,
            ("binder", "antibody"), V2_EVIDENCE,
            "common redesign for binder and antibody",
        ),
        CurrentState(
            "ligandmpnn_common_v1", "READY_EXACT", True, False,
            ("ame",), V2_EVIDENCE,
            "common redesign for AME, with ligand atom context",
        ),
        CurrentState(
            "rfdiffusion_binder_v1", "READY_PATCHED", True, False,
            ("binder",), V2_EVIDENCE,
            "executed for binder from locally retained weights; the registered "
            "MISSING_WEIGHTS note refers to fresh official retrieval",
        ),
        CurrentState(
            "rfdiffusion3_self_v1", "READY_EXACT", True, False,
            ("binder", "ame"), ARM_EVIDENCE,
            "code was already installed as the `rfd3` package of rc-foundry "
            "0.2.0 (BSD-3-Clause, IPD). Official weights retrieved with the "
            "distribution's own `foundry install rfd3`. Binder ran the same "
            "target crop and design length as the other binder arms; AME ran "
            "the declared UNINDEXED motif contract using rfd3's `unindex` "
            "input, which no earlier AME arm expressed.",
            acquisition="foundry install rfd3 -> "
                        "files.ipd.uw.edu/pub/rfd3/rfd3_foundry_2025_12_01_"
                        "remapped.ckpt, sha256 9b3f85923e0d51e9453e15cdd2f8c666"
                        "e7ce096a60577f57d11bbc54ae6d67c1",
        ),
        CurrentState(
            "boltzgen_self_v1", "READY_EXACT", True, False,
            ("binder", "antibody"), ARM_EVIDENCE,
            "registered MISSING_CODE against the wrong upstream. BoltzGen is "
            "MIT, installs from PyPI, and its weights download from its own "
            "CLI. Executed on all 11 binder tasks at both replicates: 22 cells "
            "generated, common-inverse-folded and AF2-evaluated. Its own "
            "pipeline runs an internal search, so the declared budget "
            "(--num_designs 4 --budget 2, against a recommended 10,000-60,000) "
            "and the fact that its filtering selects on the same iPTM the "
            "evaluator scores must accompany any result. The antibody "
            "fixed-framework and H3-only contract is not part of this record.",
            acquisition="pip install boltzgen; boltzgen download all",
        ),
        CurrentState(
            "diffab_cdr_v1", "READY_EXACT", True, False,
            ("antibody",), ARM_EVIDENCE,
            "registered AUTH_REQUIRED with an unreviewed acquisition plan. The "
            "weights are Apache-2.0, ungated and author-hosted at "
            "huggingface.co/luost26/DiffAb; `codesign_single.pt` is the "
            "single-CDR model that matches the H3-only contract. Executed on 12 "
            "of 13 antibody tasks at both replicates in single_cdr mode "
            "restricted to H_CDR3, then scored by the common antibody "
            "evaluator. Framework and antigen are byte-identical on every task. "
            "One task is input_ineligible: abnumber/ANARCI Chothia renumbering "
            "raises IndexError on its LIGHT chain. DiffAb designs the canonical "
            "Chothia H3, which is narrower than the benchmark's declared H3 "
            "window, and that mismatch is published per task.",
            acquisition="git clone luost26/diffab @ "
                        "c3e2966601bf8025025ab87717b31b08fdd4834e; "
                        "hf luost26/DiffAb codesign_single.pt",
        ),
        CurrentState(
            "bindcraft_native_v1", "READY_EQUIVALENT", True, False,
            ("binder",), ARM_EVIDENCE,
            "registered against the wrong upstream; BindCraft is MIT at "
            "martinpacesa/BindCraft. Executed on one "
            "binder task under a declared budget (5 trajectories, 20 sequences "
            "per trajectory, 2 MPNN designs kept). It is an internal-search "
            "pipeline that selects on AF2 metrics, the same quantity the binder "
            "evaluator scores, so its selection cost is reported beside it. "
            "Completed on 1 of 11 binder tasks: 6 trajectories, 66 candidate "
            "sequences, 64 rejected by its own filters, 2 final designs. Its "
            "rank-1 design is the ONLY design in the binder family to pass the "
            "joint gate under the common evaluator. One task supports no macro "
            "mean and no arm-level ranking, and the design cost roughly two "
            "orders of magnitude more compute than any other arm's.",
        ),

        CurrentState(
            "frameflow_motif_v1", "UNSUPPORTED_CONTRACT", False, False,
            (), "",
            "verified in this campaign against the checkout at "
            "FrameFlow, not inherited as a label. Its model forward takes input_feats with only res_mask, diffuse_mask, res_idx, so3_t, r3_t, trans_t, rotmats_t and optional trans_sc: there is no chain index and no receptor coordinate tensor, and nothing under models/ mentions a receptor, binder or hotspot. The binder task conditions on a separate target chain, and no input exists through which that target could reach the model. Trap worth recording: its contig parser is inherited from the RFdiffusion codebase and DOES accept a receptor chain token, but the sampler counts only the inpainted chains toward the sample length and silently discards the receptor, so a naive integration would run without error and quietly produce a monomer that never saw the target.",
        ),
        CurrentState(
            "chroma_conditioned_v1", "READY_EXACT", True, False,
            ("binder",), ARM_EVIDENCE,
            "Registered UNSUPPORTED_CONTRACT for lacking a target-hotspot "
            "input, which is wrong: the executed binder tier passes NO "
            "hotspots and Chroma's SubstructureConditioner over multi-chain "
            "complexes expresses it. It was re-typed access_restricted (weights "
            "need a token under the Chroma Parameters License). A token was "
            "supplied, so the restriction was lifted and never circumvented. "
            "Executed on binder: 11 tasks x 2 replicates, "
            "22 cells generated and evaluated. "
            "The first run redesigned the "
            "TARGET instead of building a binder, because Chroma RELABELS "
            "chains in the combined protein (the target becomes A, the appended "
            "binder B) and the selections had been named after the target file. "
            "Selections are now resolved on the combined protein and checked by "
            "mask size, and every design is refused unless the target sequence "
            "is preserved and its aligned CA RMSD is under 1 A. Final: target "
            "preserved on 22/22 cells at 0.00 A.",
        ),
        CurrentState(
            "rfdiffusionaa_ligand_v1", "DO_NOT_TOUCH", False, True,
            (), "",
            "not touched; no mirror or successor is recorded as its execution",
            blocked_reason="provenance_failed; a reviewed source-policy change "
                           "is required and none was made",
        ),
        CurrentState(
            "rfantibody_h3_v1", "READY_EXACT", True, False,
            ("antibody",), ARM_EVIDENCE,
            "the registration-time `license_unavailable` finding was FACTUALLY "
            "WRONG and was corrected on user authorisation. Its README states, "
            "in the same paragraph that discusses the weights it distributes: "
            "\"RFantibody is released under an MIT License (see LICENSE file). "
            "It is free for both non-profit and for-profit use.\" That is an "
            "explicit grant, and broader than research-evaluation. The earlier "
            "audit reported no such grant existed. The verdict was also "
            "inconsistent with the RFdiffusion3 run on weights "
            "from the same files.ipd.uw.edu host under a weaker grant "
            "(rc-foundry is BSD-3 but says nothing about weight licensing). "
            "Executed H3-only at the declared length: all CDR loops are "
            "labelled because RFantibody refuses a subset from one chain, but "
            "only H3 is passed to --design-loops, so every other loop and the "
            "framework stay fixed. Verified on output: all identity changes lie "
            "inside the declared H3 window.",
            acquisition="git clone RosettaCommons/RFantibody @ "
                        "8fe311415754e0276d1a39c87c57e69c88927a2d; "
                        "bash include/download_weights.sh -> "
                        "files.ipd.uw.edu/pub/RFantibody/{RFdiffusion_Ab.pt, "
                        "ProteinMPNN_v48_noise_0.2.pt, RF2_ab.pt}",
        ),
        CurrentState(
            "abx_cdr_v1", "DO_NOT_TOUCH", False, True,
            (), "",
            "not touched; no unofficial mirror substitution",
            blocked_reason="provenance_failed; no official source resolved",
        ),
    ]
    out: dict[str, CurrentState] = {}
    for state in states:
        if state.baseline_id in out:
            raise BaselineStateError(f"duplicate state row {state.baseline_id}")
        out[state.baseline_id] = state
    registered = {b.baseline_id for b in load_codirect_baseline_registry().baselines}
    missing = registered - set(out)
    extra = set(out) - registered
    if missing:
        raise BaselineStateError(f"registered baselines with no state: {sorted(missing)}")
    if extra:
        raise BaselineStateError(f"state rows for unregistered baselines: {sorted(extra)}")
    return out

def reconcile_with_registry() -> list[dict]:

    registry = load_codirect_baseline_registry()
    states = load_current_states()
    corrections = corrected_upstreams()
    rows = []
    for baseline in registry.baselines:
        state = states[baseline.baseline_id]
        rows.append({
            "baseline_id": baseline.baseline_id,
            "role": baseline.role,
            "families_registered": list(baseline.families),
            "readiness_at_registration": str(baseline.readiness),
            "official_url_at_registration": baseline.official_url,
            "readiness_now": state.readiness_now,
            "upstream_corrected": baseline.baseline_id in corrections,
            "corrected_url": corrections.get(baseline.baseline_id, {}).get(
                "corrected_url", ""),
            "executed": state.executed,
            "blocked": state.blocked,
            "blocked_reason": state.blocked_reason,
            "families_executed": list(state.families_executed),
            "acquisition": state.acquisition,
            "evidence_root": state.evidence_root,
            "note": state.note,
        })
    return rows
