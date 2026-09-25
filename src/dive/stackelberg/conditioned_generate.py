
from __future__ import annotations

from contextlib import nullcontext

from dive.stackelberg.router_condition import trunk_reads_router_condition
from dive.stackelberg.sampling import anticipating_sampler

def generate_with_anticipation(model, batch, *, mode="anticipate", alpha=1.28,
                               clip=0.5, router=None, partner_only=False,
                               alpha_latent=None, condition_router=False,
                               self_cond_hat=False):

    if alpha_latent is not None:
        alpha = {"bb_ca": float(alpha), "local_latents": float(alpha_latent)}
    if condition_router and router is None:
        raise ValueError(
            "condition_router=True needs a router; otherwise the trunk would "
            "see an empty x_sc while the arm still claims the wrap")
    cond = (trunk_reads_router_condition(
                model, router, missing=("zeros" if self_cond_hat else "x_t"))
            if condition_router else nullcontext())
    ctx = anticipating_sampler(
        model, mode=mode, alpha=alpha, clip=clip, router=router,
        partner_only=partner_only,
    )
    with cond, ctx:
        return model.generate(batch)
