
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ObstructionResult:

    value: float
    witness: np.ndarray
    primal_move: np.ndarray | None
    feasible_linearized: bool
    duality_gap: float | None
    converged: bool
    n_constraints: int
    radius: float
    iterations: int = 0

    @property
    def active_constraints(self) -> np.ndarray:

        if self.witness.size == 0:
            return np.zeros(0, dtype=int)
        return np.flatnonzero(self.witness > 1e-6)

def ball_support(s: np.ndarray, radius: float) -> float:

    return float(radius) * float(np.linalg.norm(s))

def _project_onto_W(v: np.ndarray) -> np.ndarray:

    w = np.maximum(v, 0.0)
    if w.sum() <= 1.0:
        return w
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, u.size + 1)
    cond = u - css / idx > 0
    rho = int(idx[cond][-1])
    theta = css[rho - 1] / rho
    return np.maximum(w - theta, 0.0)

def obstruction_dual(
    A: np.ndarray,
    b: np.ndarray,
    *,
    radius: float,
    max_iter: int = 4000,
    tol: float = 1e-10,
    check_primal: bool = True,
) -> ObstructionResult:

    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    b = np.asarray(b, dtype=np.float64).ravel()
    m = int(b.size)
    if m == 0:
        return ObstructionResult(0.0, np.zeros(0), np.zeros(A.shape[1] if A.ndim == 2 else 0),
                                 True, 0.0, True, 0, float(radius))
    if A.shape[0] != m:
        raise ValueError(f"A has {A.shape[0]} rows but b has {m} entries")
    radius = float(radius)

    def objective(w: np.ndarray) -> float:
        return float(w @ b) - radius * float(np.linalg.norm(A.T @ w))

    scale = float(np.linalg.norm(A, 2)) if A.size else 1.0
    span = float(np.abs(b).max()) + 1e-12

    w = np.zeros(m)
    best_w, best_value = w.copy(), objective(w)
    iterations = 0
    for eps in (1.0, 0.25, 6e-2, 1.5e-2, 4e-3, 1e-3, 2e-4):
        eps = eps * max(span, 1e-9)
        lipschitz = max(radius * scale * scale / max(eps, 1e-12), 1e-8)
        step = 1.0 / lipschitz
        y, t = w.copy(), 1.0
        for _ in range(max_iter // 7):
            iterations += 1
            s_vec = A.T @ y
            smooth = float(np.sqrt(s_vec @ s_vec + eps * eps))
            grad = b - radius * (A @ s_vec) / smooth
            w_next = _project_onto_W(y + step * grad)
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = w_next + ((t - 1.0) / t_next) * (w_next - w)
            moved = float(np.linalg.norm(w_next - w))
            w, t = w_next, t_next
            value = objective(w)
            if value > best_value:
                best_value, best_w = value, w.copy()
            if moved < tol:
                break

    value = max(best_value, 0.0)

    primal = obstruction_primal(A, b, radius=radius) if check_primal else None
    return ObstructionResult(
        value=value,
        witness=best_w,
        primal_move=primal.primal_move if primal is not None else None,
        feasible_linearized=bool(value <= 1e-7),
        duality_gap=float(abs(value - primal.value)) if primal is not None else None,
        converged=bool(iterations < max_iter),
        n_constraints=m,
        radius=radius,
        iterations=iterations,
    )

def obstruction_primal(
    A: np.ndarray,
    b: np.ndarray,
    *,
    radius: float,
    max_iter: int = 4000,
    tol: float = 1e-12,
) -> ObstructionResult:

    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    b = np.asarray(b, dtype=np.float64).ravel()
    m = int(b.size)
    n = int(A.shape[1]) if A.ndim == 2 else 0
    if m == 0:
        return ObstructionResult(0.0, np.zeros(0), np.zeros(n), True, 0.0, True, 0, float(radius))
    radius = float(radius)
    if radius <= 0.0:
        value = max(0.0, float(b.max()))
        return ObstructionResult(value, np.zeros(m), np.zeros(n),
                                 bool(value <= 1e-7), None, True, m, radius)

    delta = np.zeros(n)
    scale = float(np.linalg.norm(A, 2)) or 1.0
    best_delta = delta.copy()
    best_value = max(0.0, float(b.max()))
    iterations = 0
    spread = float(np.abs(b).max()) + scale * radius + 1.0

    for beta in (2.0, 8.0, 32.0, 128.0, 512.0):
        beta = beta / spread
        step = 1.0 / max(beta * scale * scale, 1e-8)
        for _ in range(max_iter // 5):
            iterations += 1
            residuals = b - A @ delta
            shifted = beta * (residuals - residuals.max())
            weights = np.exp(shifted)
            weights /= weights.sum()
            grad = -(A.T @ weights)
            candidate = delta - step * grad
            norm = float(np.linalg.norm(candidate))
            if norm > radius:
                candidate *= radius / norm
            moved = float(np.linalg.norm(candidate - delta))
            delta = candidate
            value = max(0.0, float((b - A @ delta).max()))
            if value < best_value:
                best_value, best_delta = value, delta.copy()
            if best_value <= 1e-12 or moved < tol:
                break
        if best_value <= 1e-12:
            break

    return ObstructionResult(
        value=best_value,
        witness=np.zeros(m),
        primal_move=best_delta,
        feasible_linearized=bool(best_value <= 1e-7),
        duality_gap=None,
        converged=bool(iterations < max_iter),
        n_constraints=m,
        radius=radius,
        iterations=iterations,
    )
