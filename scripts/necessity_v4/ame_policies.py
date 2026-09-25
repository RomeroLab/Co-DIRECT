
from __future__ import annotations
import numpy as np
import torch

import ame_moves as MV
import ame_search as S

class Budget:

    def __init__(self, scorer, limit: int):
        self.scorer, self.limit, self.used = scorer, limit, 0

    def __call__(self, coors) -> float:
        self.used += 1
        return self.scorer(coors)

    @property
    def left(self) -> int:
        return max(self.limit - self.used, 0)

def _anneal(coors, am, rt, rm, budget, dofs, n_eval, rng, native, t0=0.05, t1=1e-4):

    if not dofs or n_eval <= 0:
        return coors, budget(coors) if budget.left else float("inf"), 0
    cur = coors
    cur_e = budget(cur)
    best, best_e, rej = cur, cur_e, 0

    target = min(n_eval, budget.left)
    spent, step, guard = 0, 0, 0
    while spent < target and budget.left > 0 and guard < 50 * target + 1000:
        guard += 1
        T = t0 * (t1 / t0) ** (min(spent, target - 1) / max(target - 1, 1))
        kind, r, c = dofs[int(rng.integers(len(dofs)))]
        if kind == "chi":
            cand = MV.apply_chi(cur, am, rt, r, c, float(rng.normal()) * S.SIGMA_CHI)
        else:
            cand = MV.apply_backrub(cur, am, r, float(rng.normal()) * S.SIGMA_BACKRUB)
            if MV.ca_angle_drift_sigma(cand, native, am, rt, (r - 1, r, r + 1)) > S.ANGLE_SIGMA_MAX:
                rej += 1
                continue
        e = budget(cand)
        spent += 1
        step += 1
        if e <= cur_e or rng.random() < np.exp(-(e - cur_e) / T):
            cur, cur_e = cand, e
            if e < best_e:
                best, best_e = cand, e
    return best, best_e, rej

def joint(coors, am, rt, rm, budget, chi_dof, br_dof, rng, native, **_ignored):
    dofs = [("chi", r, c) for r, c in chi_dof] + [("br", r, None) for r in br_dof]
    best, e, rej = _anneal(coors, am, rt, rm, budget, dofs, budget.limit, rng, native)
    return best, e, {"rejected": rej, "outer": 1}

def alternating(coors, am, rt, rm, budget, chi_dof, br_dof, rng, native, block=800,
                **_ignored):

    cdofs = [("chi", r, c) for r, c in chi_dof]
    bdofs = [("br", r, None) for r in br_dof]
    cur, best, best_e, rej, k = coors, coors, float("inf"), 0, 0
    while budget.left > 0:
        dofs = cdofs if k % 2 == 0 else bdofs
        cur, e, rj = _anneal(cur, am, rt, rm, budget, dofs, min(block, budget.left),
                             rng, native)
        rej += rj
        if e < best_e:
            best, best_e = cur, e
        k += 1
    return best, best_e, {"rejected": rej, "outer": k}

def _led(coors, am, rt, rm, budget, leader_dofs, follower_dofs, rng, native,
         n_cand=4, lead_eval=200, follow_eval=200, **_ignored):

    cur = coors
    cur_e = budget(cur)
    best, best_e, rej, outer = cur, cur_e, 0, 0
    lead = [("chi", r, c) if isinstance(c, (int, np.integer)) else ("br", r, None)
            for r, c in leader_dofs]
    foll = [("chi", r, c) if isinstance(c, (int, np.integer)) else ("br", r, None)
            for r, c in follower_dofs]
    if not lead:
        return best, best_e, {"rejected": 0, "outer": 0}
    while budget.left > (lead_eval + follow_eval):
        outer += 1
        cands = []
        for _ in range(n_cand):
            if budget.left <= (lead_eval + follow_eval):
                break
            prop, _pe, rj = _anneal(cur, am, rt, rm, budget, lead,
                                    min(lead_eval, budget.left), rng, native)
            rej += rj
            resp, re_, rj2 = _anneal(prop, am, rt, rm, budget, foll,
                                     min(follow_eval, budget.left), rng, native)
            rej += rj2
            cands.append((re_, prop, resp))
        if not cands:
            break
        re_, prop, resp = min(cands, key=lambda x: x[0])
        if re_ < cur_e:
            cur, cur_e = resp, re_
        if re_ < best_e:
            best, best_e = resp, re_
    return best, best_e, {"rejected": rej, "outer": outer}

def local_led(coors, am, rt, rm, budget, chi_dof, br_dof, rng, native, **kw):
    return _led(coors, am, rt, rm, budget,
                [(r, c) for r, c in chi_dof], [(r, None) for r in br_dof],
                rng, native, **kw)

def backbone_led(coors, am, rt, rm, budget, chi_dof, br_dof, rng, native, **kw):
    return _led(coors, am, rt, rm, budget,
                [(r, None) for r in br_dof], [(r, c) for r, c in chi_dof],
                rng, native, **kw)

def fixed_only(coors, am, rt, rm, budget, chi_dof, br_dof, rng, native, **_ignored):

    return joint(coors, am, rt, rm, budget, chi_dof, [], rng, native)

POLICIES = {"joint": joint, "alternating": alternating, "local_led": local_led,
            "backbone_led": backbone_led, "fixed_only": fixed_only}
