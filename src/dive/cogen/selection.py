
from __future__ import annotations

from math import comb
from typing import Sequence

def selection_probabilities(scores: Sequence[float], k: int, *,
                            prefer_low: bool = True) -> list[float]:

    n = len(scores)
    if k < 1 or k > n:
        raise ValueError(f"K={k} outside 1..{n}")
    order = sorted(range(n), key=lambda i: scores[i], reverse=not prefer_low)
    probs = [0.0] * n
    total = comb(n, k)
    position = 0
    while position < n:
        end = position
        while end + 1 < n and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        group = order[position:end + 1]
        m = len(group)
        after = n - (end + 1)
        share = 0.0
        for j in range(0, min(m - 1, k - 1) + 1):
            rest = k - 1 - j
            if rest < 0 or rest > after:
                continue
            share += comb(m - 1, j) * comb(after, rest) / (j + 1)
        share /= total
        for index in group:
            probs[index] = share
        position = end + 1
    return probs

def best_of_k_expectation(values: Sequence[float | None],
                          scores: Sequence[float | None],
                          k: int, prefer_low: bool = True) -> float | None:

    pool = [(s, v) for s, v in zip(scores, values) if s is not None and v is not None]
    if len(pool) < k or k < 1:
        return None
    probs = selection_probabilities([s for s, _ in pool], k, prefer_low=prefer_low)
    return float(sum(p * v for p, (_, v) in zip(probs, pool)))

def oracle_expectation(values: Sequence[float | None], k: int,
                       direction: int) -> float | None:

    vals = [v for v in values if v is not None]
    if len(vals) < k or k < 1:
        return None
    return best_of_k_expectation(vals, [v * -direction for v in vals], k, prefer_low=True)

def expectation_with_missing(values: Sequence[float | None],
                             scores: Sequence[float | None],
                             k: int, prefer_low: bool = True):

    pool = [(s, v) for s, v in zip(scores, values) if s is not None]
    if len(pool) < k or k < 1:
        return None, None
    probs = selection_probabilities([s for s, _ in pool], k, prefer_low=prefer_low)
    mass = sum(p for p, (_, v) in zip(probs, pool) if v is not None)
    if mass <= 0:
        return None, 1.0
    total = sum(p * v for p, (_, v) in zip(probs, pool) if v is not None)
    return float(total / mass), float(1.0 - mass)

def choose(scores: Sequence[float | None], *, prefer_low: bool = True,
           seed: int) -> tuple[int, dict]:

    import random

    pool = [(index, value) for index, value in enumerate(scores) if value is not None]
    if not pool:
        raise ValueError("no draw in the pool carries a score")
    best = min(v for _, v in pool) if prefer_low else max(v for _, v in pool)
    tied = [index for index, value in pool if value == best]
    chosen = random.Random(seed).choice(tied)
    return chosen, {
        "pool": len(pool),
        "unscored": len(scores) - len(pool),
        "best_score": best,
        "tied": len(tied),
        "tie_break_seed": seed,
        "margin": (None if len(pool) < 2 else
                   (sorted(v for _, v in pool)[1] - best if prefer_low
                    else best - sorted((v for _, v in pool), reverse=True)[1])),
    }
