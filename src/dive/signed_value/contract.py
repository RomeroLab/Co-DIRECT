
from __future__ import annotations

RUN_ID = "signed-critic-feasibility-003"

CLOSED_RUN_IDS = (
    "signed-critic-feasibility-001",
    "signed-critic-feasibility-002",
)

SMALL_SLOT_TARGET = "33_TrkA"
LARGE_SLOT_TARGET = "35_H1"

MEDIAN_SLOT_CANDIDATES = (
    "31_IL7RA",
    "05_CD45",
    "30_SC2RBD",
    "04_IFNAR2",
    "36_VEGFA",
    "29_BHRF1",
)

MEDIAN_SLOT_TARGET = MEDIAN_SLOT_CANDIDATES[0]

FROZEN_TARGETS = (SMALL_SLOT_TARGET, MEDIAN_SLOT_TARGET, LARGE_SLOT_TARGET)

CONTROL_DENOISER_CALLS = 400
PROBE_DENOISER_CALLS = 403

BASIS_DENOISER_CALLS = PROBE_DENOISER_CALLS - CONTROL_DENOISER_CALLS
DENOISER_CALLS_PER_TARGET = CONTROL_DENOISER_CALLS + PROBE_DENOISER_CALLS

def frozen_feasibility_block() -> dict[str, object]:

    from dive.signed_value.critic_protocol import CRITIC_CALLS
    from dive.signed_value.directional import DECODER_CALLS, LADDER

    return {
        "generation_seed": 11,
        "steps": CONTROL_DENOISER_CALLS,
        "checkpoint_call": 200,
        "prior_count": 2,
        "scale_multipliers": [
            int(scale) if float(scale).is_integer() else scale for scale in LADDER
        ],
        "initial_targets": list(FROZEN_TARGETS),
        "denoiser_calls_per_target": DENOISER_CALLS_PER_TARGET,
        "decoder_calls_per_target": DECODER_CALLS,
        "critic_calls_per_target": CRITIC_CALLS,
    }
