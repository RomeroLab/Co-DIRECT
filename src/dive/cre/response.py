
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

@dataclass
class QuadraticCandidate:

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    a: np.ndarray
    d: np.ndarray
    k: float = 0.0

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, float)
        self.B = np.asarray(self.B, float)
        self.C = np.asarray(self.C, float)
        self.a = np.asarray(self.a, float).reshape(-1)
        self.d = np.asarray(self.d, float).reshape(-1)
        nb, nu = self.B.shape
        if self.A.shape != (nb, nb) or self.C.shape != (nu, nu):
            raise ValueError("block shapes disagree")
        if self.a.shape != (nb,) or self.d.shape != (nu,):
            raise ValueError("linear terms disagree with block shapes")

        self.C = 0.5 * (self.C + self.C.T)
        self.A = 0.5 * (self.A + self.A.T)
        try:
            self._chol = cho_factor(self.C, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError("C must be positive definite") from error

    @property
    def n_shared(self) -> int:
        return self.B.shape[0]

    @property
    def n_own(self) -> int:
        return self.B.shape[1]

    def _solve_c(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self._chol, rhs)

    def cost(self, b: np.ndarray, u: np.ndarray) -> float:
        b = np.asarray(b, float).reshape(-1)
        u = np.asarray(u, float).reshape(-1)
        return float(
            0.5 * b @ self.A @ b + b @ self.B @ u + 0.5 * u @ self.C @ u
            + self.a @ b + self.d @ u + self.k
        )

    def response(self, b: np.ndarray) -> np.ndarray:

        b = np.asarray(b, float).reshape(-1)
        return -self._solve_c(self.B.T @ b + self.d)

    def value(self, b: np.ndarray) -> float:

        return self.cost(b, self.response(b))

    def value_quadratic(self) -> tuple[np.ndarray, np.ndarray, float]:

        solved_bt = self._solve_c(self.B.T)
        solved_d = self._solve_c(self.d)
        S = self.A - self.B @ solved_bt
        q = self.a - self.B @ solved_d
        kappa = self.k - 0.5 * float(self.d @ solved_d)
        return 0.5 * (S + S.T), q, kappa

def primal_minimise(candidate: QuadraticCandidate, b: np.ndarray) -> tuple[float, np.ndarray]:

    b = np.asarray(b, float).reshape(-1)
    gradient_constant = candidate.B.T @ b + candidate.d
    u = np.linalg.solve(candidate.C, -gradient_constant)
    return candidate.cost(b, u), u

@dataclass
class JointUpdate:

    candidate: str
    backbone_update: np.ndarray
    response: np.ndarray
    objective: float
    per_candidate_objective: dict[str, float]

def select_joint_update(
    candidates: dict[str, QuadraticCandidate],
    log_prior: dict[str, float],
    *,
    backbone_quadratic: tuple[np.ndarray, np.ndarray],
    tau: float = 1.0,
    trust_radius: float | None = None,
) -> JointUpdate:

    backbone_S, backbone_q = backbone_quadratic
    backbone_S = np.asarray(backbone_S, float)
    backbone_S = 0.5 * (backbone_S + backbone_S.T)
    backbone_q = np.asarray(backbone_q, float).reshape(-1)

    objectives: dict[str, float] = {}
    solutions: dict[str, np.ndarray] = {}
    for name, candidate in candidates.items():
        S, q, kappa = candidate.value_quadratic()
        total_S = backbone_S + S
        total_q = backbone_q + q
        try:
            b = np.linalg.solve(total_S, -total_q)
        except np.linalg.LinAlgError:

            objectives[name] = float("inf")
            solutions[name] = np.zeros(candidate.n_shared)
            continue
        if trust_radius is not None:
            norm = float(np.linalg.norm(b))
            if norm > trust_radius:
                b = b * (trust_radius / norm)
        objectives[name] = (
            0.5 * b @ backbone_S @ b + backbone_q @ b
            + candidate.value(b)
            - tau * float(log_prior.get(name, 0.0))
        )
        solutions[name] = b

    best = min(objectives, key=lambda name: objectives[name])
    b_star = solutions[best]
    return JointUpdate(
        candidate=best,
        backbone_update=b_star,
        response=candidates[best].response(b_star),
        objective=objectives[best],
        per_candidate_objective=objectives,
    )

def shared_feasible_backbone(requirements, *, grid=None):

    if not requirements:
        return 0.0
    lower = max(r["target"] - r["tolerance"] for r in requirements)
    upper = min(r["target"] + r["tolerance"] for r in requirements)
    if lower > upper:
        return None
    return 0.5 * (lower + upper)
