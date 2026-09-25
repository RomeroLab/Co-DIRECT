
from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import numpy as np

def scan_one(job: dict) -> dict:

    from dive.ccr.live import residue_exchange_inputs
    from dive.ccr.operator import build_constraints, index_by_residue

    row = int(job.get("row", 0))
    try:
        coors = np.asarray(job["coors"], dtype=np.float64)
        chain_index = np.asarray(job["chain_index"], dtype=int)
        n_design = int(job["n_design"])
        raw = job.get("residues")
        residues = None if raw is None else np.asarray(raw, dtype=int)
        constraints = build_constraints(
            coors, chain_index=chain_index,
            max_clash_constraints=int(job.get("outer_clash_constraints", 20000)),
        )
        index = index_by_residue(constraints)
        if residues is None or residues.size == 0 and job.get("residues") is None:

            residues = _select_residues(
                constraints, index, n_design=n_design,
                top_k=int(job.get("top_k", 8)), seed=int(job.get("select_seed", row)),
            )
        out = residue_exchange_inputs(
            coors, seq_logits=np.asarray(job["seq_logits"], dtype=np.float64),
            residues=residues, arm=job["arm"], radius=float(job["radius"]),
            time_fraction=float(job["time_fraction"]),
            n_slots=int(job["n_slots"]), chain_index=chain_index,
            constraints=constraints, constraint_index=index, n_design=n_design,
            n_rotamers=int(job.get("n_rotamers", 2)),
            solver_iterations=int(job.get("solver_iterations", 700)),
            max_local_constraints=int(job.get("max_local_constraints", 64)),
            patch_margin=float(job.get("patch_margin", 8.0)),
            patch_max_residues=int(job.get("patch_max_residues", 48)),
            max_clash_constraints=int(job.get("max_clash_constraints", 4000)),
            time_budget=job.get("time_budget", 60.0),
            with_choice_labels=bool(job.get("with_choice_labels", True)),
        )
        return {
            "row": row, "message": out.message, "cand_index": out.cand_index,
            "cand_feat": out.cand_feat, "choice_labels": out.choice_labels,
            "stats": out.stats,
        }
    except Exception as error:
        return {"row": row, "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=4)}

def _select_residues(constraints, index, *, n_design: int, top_k: int,
                     seed: int = 0) -> np.ndarray:

    if n_design <= 0:
        return np.zeros(0, dtype=int)
    top_k = min(int(top_k), int(n_design))
    violated = np.zeros(0, dtype=int)
    if constraints.n_constraints:
        deficit = np.zeros(n_design)
        for residue, rows in index.items():
            if residue < n_design:
                deficit[residue] = float(
                    np.maximum(0.0, -constraints.values[np.array(rows, dtype=int)]).sum()
                )

        order = np.argsort(deficit)[::-1][: max(1, (2 * top_k) // 3)]
        violated = order[deficit[order] > 0.0].astype(int)

    chosen = list(dict.fromkeys(int(r) for r in violated))
    rest = [r for r in range(n_design) if r not in set(chosen)]
    if rest and len(chosen) < top_k:
        rng = np.random.default_rng(seed)
        extra = rng.choice(len(rest), size=min(top_k - len(chosen), len(rest)),
                           replace=False)
        chosen.extend(int(rest[i]) for i in extra)
    return np.array(sorted(chosen[:top_k]), dtype=int)

def _initialise() -> None:

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"

class ScanPool:

    def __init__(self, n_workers: int = 0):
        self.n_workers = max(int(n_workers), 0)
        self._pool = None
        if self.n_workers:
            context = mp.get_context("spawn")
            self._pool = context.Pool(self.n_workers, initializer=_initialise)

    def map_examples(self, jobs: list[dict]) -> list[dict]:
        if not jobs:
            return []
        for i, job in enumerate(jobs):
            job.setdefault("row", i)
        if self._pool is None:
            return [scan_one(job) for job in jobs]
        return self._pool.map(scan_one, jobs)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
