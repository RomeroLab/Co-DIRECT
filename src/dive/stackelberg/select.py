
from __future__ import annotations

import random

IPTM_WEIGHT = 0.8
PTM_WEIGHT = 0.2

def rank_confidence(sample: dict) -> float:

    ptm = sample.get("ptm")
    if not isinstance(ptm, (int, float)):
        raise ValueError(
            "sample carries no pTM, so it cannot be ranked; selecting it anyway "
            "would silently fall back to listing order")
    iface = sample.get("interface") or {}
    iptm = iface.get("iptm")
    if isinstance(iptm, (int, float)):
        return IPTM_WEIGHT * float(iptm) + PTM_WEIGHT * float(ptm)
    return float(ptm)

def best_sample(samples, *, seed: int = 0) -> dict:

    if not samples:
        raise ValueError("no refold samples to select from")
    scored = [(rank_confidence(s), i) for i, s in enumerate(samples)]
    top = max(c for c, _ in scored)

    winners = [i for c, i in scored if c == top]
    return samples[random.Random(seed).choice(winners)]

def best_design(designs: dict, *, seed: int = 0):

    if not designs:
        raise ValueError("no designs to select from")
    picks = {k: best_sample(v, seed=seed) for k, v in designs.items()}
    scored = [(rank_confidence(s), k) for k, s in picks.items()]
    top = max(c for c, _ in scored)
    winners = sorted(k for c, k in scored if c == top)
    key = random.Random(seed).choice(winners)
    return key, picks[key]
