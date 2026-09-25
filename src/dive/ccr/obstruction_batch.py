
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS_SCHEDULE: tuple[float, ...] = (1.0, 0.25, 6e-2, 1.5e-2, 4e-3, 1e-3, 2e-4)

@dataclass(frozen=True)
class BatchObstructionResult:

    value: np.ndarray
    witness: np.ndarray
    converged: np.ndarray
    n_constraints: np.ndarray
    radius: float
    iterations: int

def _project_onto_W_batch(v: np.ndarray, valid: np.ndarray) -> np.ndarray:

    w = np.maximum(v, 0.0) * valid
    total = w.sum(axis=1)
    over = total > 1.0
    if not over.any():
        return w

    sub = w[over]
    u = -np.sort(-sub, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, u.shape[1] + 1)[None, :]
    cond = (u - css / idx) > 0

    rho = cond.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(sub.shape[0]), rho] / (rho + 1)
    w[over] = np.maximum(sub - theta[:, None], 0.0) * valid[over]
    return w

def _spectral_norms(A: np.ndarray) -> np.ndarray:

    if A.shape[1] == 0 or A.shape[2] == 0:
        return np.zeros(A.shape[0])
    return np.linalg.svd(A, compute_uv=False)[:, 0]

def obstruction_dual_batch(
    A: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
    *,
    radius: float,
    max_iter: int = 700,
    tol: float = 1e-10,
) -> BatchObstructionResult:

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if A.ndim != 3:
        raise ValueError(f"A must be [k, m, n], got shape {A.shape}")
    k, m, _n = A.shape
    if b.shape != (k, m) or valid.shape != (k, m):
        raise ValueError(f"b {b.shape} and valid {valid.shape} must both be {(k, m)}")

    counts = valid.sum(axis=1)
    if m == 0 or not valid.any():
        return BatchObstructionResult(
            np.zeros(k), np.zeros((k, m)), np.ones(k, dtype=bool),
            counts.astype(int), float(radius), 0,
        )

    A = A * valid[:, :, None]
    b = b * valid
    radius = float(radius)

    scale = _spectral_norms(A)
    span = np.abs(b).max(axis=1) + 1e-12

    gram = np.matmul(A, A.transpose(0, 2, 1))

    def _gram_mul(w: np.ndarray) -> np.ndarray:
        return np.matmul(gram, w[:, :, None])[:, :, 0]

    def objective(w: np.ndarray) -> np.ndarray:
        quad = np.maximum((w * _gram_mul(w)).sum(axis=1), 0.0)
        return (w * b).sum(axis=1) - radius * np.sqrt(quad)

    w = np.zeros((k, m))
    best_w = w.copy()
    best_value = objective(w)
    iterations = 0
    moved_all = np.zeros(k, dtype=bool)

    for raw_eps in EPS_SCHEDULE:
        eps = raw_eps * np.maximum(span, 1e-9)
        lipschitz = np.maximum(radius * scale * scale / np.maximum(eps, 1e-12), 1e-8)
        step = (1.0 / lipschitz)[:, None]
        y, t = w.copy(), 1.0
        for _ in range(max_iter // len(EPS_SCHEDULE)):
            iterations += 1
            gram_y = _gram_mul(y)
            quad = np.maximum((y * gram_y).sum(axis=1), 0.0)
            smooth = np.sqrt(quad + eps * eps)[:, None]
            grad = b - radius * gram_y / smooth
            w_next = _project_onto_W_batch(y + step * grad, valid)
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = w_next + ((t - 1.0) / t_next) * (w_next - w)
            moved = np.linalg.norm(w_next - w, axis=1)
            w, t = w_next, t_next
            value = objective(w)
            better = value > best_value
            if better.any():
                best_value = np.where(better, value, best_value)
                best_w[better] = w[better]
            moved_all = moved < tol
            if moved_all.all():
                break

    value = np.maximum(best_value, 0.0)
    value = np.where(counts > 0, value, 0.0)
    return BatchObstructionResult(
        value=value,
        witness=best_w * valid,
        converged=np.ones(k, dtype=bool),
        n_constraints=counts.astype(int),
        radius=radius,
        iterations=iterations,
    )
