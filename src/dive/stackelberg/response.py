
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

def leader_gradient(a: Tensor, response: Callable[[Tensor], Tensor],
                    leader_loss: Callable[[Tensor, Tensor], Tensor],
                    *, anticipate: bool) -> Tensor:

    a = a.detach().requires_grad_(True)
    b = response(a)
    if not anticipate:
        b = b.detach()
    leader_loss(a, b).backward()
    return a.grad.detach().clone()

def solve(a0: float, response, leader_loss, *, anticipate: bool,
          steps: int = 2000, lr: float = 0.05) -> dict:

    a = torch.tensor([float(a0)])
    for _ in range(steps):
        a = (a - lr * leader_gradient(a, response, leader_loss,
                                      anticipate=anticipate)).detach()
    b = response(a).detach()
    return {"a": float(a), "b": float(b),
            "leader_loss_after_the_real_response": float(leader_loss(a, b))}
