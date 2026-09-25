
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dive.phase3_verify.q1_readability import ProbeReading

ALPHAS = np.logspace(-1, 6, 29)
BOOTSTRAP = 400

def _fit_predict(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xtr, ytr)
    return yte, model.predict(Xte)

def _r2(truth, pred) -> float:
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

def _advantage(H, C, y, seed) -> tuple[float, float, float]:

    truth, pred_h = _fit_predict(H, y, seed)
    truth_c, pred_c = _fit_predict(C, y, seed)

    assert np.array_equal(truth, truth_c), "arms scored on different rows"
    point = _r2(truth, pred_h) - _r2(truth, pred_c)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP)
    n = len(truth)
    for i in range(BOOTSTRAP):
        take = rng.integers(0, n, n)
        draws[i] = _r2(truth[take], pred_h[take]) - _r2(truth[take], pred_c[take])
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, float(low), float(high)

def probe_reading(
    family: str, H, Y, C, *, seed: int = 1401
) -> ProbeReading:

    H, Y, C = np.asarray(H), np.asarray(Y), np.asarray(C)
    if not (len(H) == len(Y) == len(C)):
        raise ValueError("H, Y and C must describe the same residues")
    truth, pred_h = _fit_predict(H, Y, seed)
    r2_rep = _r2(truth, pred_h)
    rng = np.random.default_rng(seed)
    truth_n, pred_n = _fit_predict(H, rng.permutation(Y), seed)
    r2_null = _r2(truth_n, pred_n)
    truth_c, pred_c = _fit_predict(C, Y, seed)
    r2_coords = _r2(truth_c, pred_c)

    _, low, high = _advantage(H, C, Y, seed)

    order = np.random.default_rng(seed + 1).permutation(len(Y))
    halves = []
    for part in np.array_split(order, 2):
        if len(part) < 120:
            halves = None
            break
        halves.append(_advantage(H[part], C[part], Y[part], seed)[0])
    return ProbeReading(
        family=family,
        n=int(len(Y)),
        r2_representation=float(r2_rep),
        r2_shuffled_null=float(r2_null),
        r2_own_coords=float(r2_coords),
        effect_ci_low=float(low),
        effect_ci_high=float(high),
        half_effects=tuple(halves) if halves else None,
    )
