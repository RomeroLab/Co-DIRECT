
from __future__ import annotations

from contextlib import contextmanager

import torch

from dive.imr.endpoint import apply_endpoint_delta
from dive.imr.revision import (
    blind_displacement, cap_per_residue, follower_is_coherent, implicit_vote,
    in_window, scramble_ca,
)

MODES = ("off", "revise", "blind", "blocked")

DEFAULT_WINDOW = (0.20, 0.75)

DEFAULT_CAP_NM = 0.05

DEFAULT_STRIDE = 1

class Telemetry(list):

    def summary(self):
        fired = [r for r in self if r["fired"]]
        if not fired:
            return {"steps": len(self), "fired": 0,
                "decoder_calls": sum(r["decoder_calls"] for r in self),
                "in_window": sum(r["follower_coherent"] is not None for r in self),
                "follower_incoherent": sum(r["follower_coherent"] is False for r in self)}
        import statistics as st
        return {
            "steps": len(self),
            "fired": len(fired),
            "t_ca_min": min(r["t_ca"] for r in fired),
            "t_ca_max": max(r["t_ca"] for r in fired),
            "decoder_calls": sum(r["decoder_calls"] for r in self),
            "in_window": sum(r["follower_coherent"] is not None for r in self),
            "follower_incoherent": sum(r["follower_coherent"] is False for r in self),
            "mean_applied_A": st.fmean(r["applied_mean_A"] for r in fired),
            "max_applied_A": max(r["applied_max_A"] for r in fired),
            "mean_gap_before_A": st.fmean(r["gap_before_A"] for r in fired),

            "mean_gap_after_A": (st.fmean(after) if (after := [
                r["gap_after_A"] for r in fired if r["gap_after_A"] is not None]) else None),
        }

@contextmanager
def imr_sampling(model, mask, *, mode="revise", alpha=1.0, cap_nm=DEFAULT_CAP_NM,
                 window=DEFAULT_WINDOW, stride=DEFAULT_STRIDE, seed=20260916,
                 telemetry=None, verify_gap=False, health_gate=True):

    if mode not in MODES:
        raise ValueError(f"unsupported IMR mode: {mode!r}")
    if not 0.0 <= window[0] < window[1] <= 1.0:
        raise ValueError("window must satisfy 0 <= low < high <= 1")
    if alpha != 0.0 and mode == "off":
        raise ValueError("mode 'off' must carry no revision weight")

    original = model.call_nn
    had_instance_binding = "call_nn" in model.__dict__

    gen = torch.Generator(device="cpu").manual_seed(seed)
    counter = {"k": 0}

    def call(batch, *args, **kwargs):

        safe = batch
        output = original(safe, *args, **kwargs)
        step = counter["k"]
        counter["k"] = step + 1
        t_ca = float(safe["t"]["bb_ca"].mean())
        row = {"step": step, "t_ca": t_ca, "fired": False, "decoder_calls": 0,
               "applied_mean_A": None, "applied_max_A": None,
               "gap_before_A": None, "gap_after_A": None,
               "follower_coherent": None, "follower_bonds_A": None}

        eligible = (mode != "off"
                    and alpha != 0.0
                    and in_window(safe["t"]["bb_ca"], window)
                    and step % stride == 0)
        if not eligible:
            if telemetry is not None:
                telemetry.append(row)
            return output

        copied = {m: dict(v) for m, v in output.items()}
        clean = model.fm.nn_out_to_clean_sample_prediction(safe, copied)
        leader_ca, latents = clean["bb_ca"], clean["local_latents"]

        decoded = model.autoencoder.decode(latents, leader_ca, mask)
        row["decoder_calls"] = 1
        vote, gap, have = implicit_vote(decoded, leader_ca, mask)
        if not bool(have.any()):
            if telemetry is not None:
                telemetry.append(row)
            return output
        row["gap_before_A"] = float(gap[have].mean()) * 10.0

        coherent, bonds = follower_is_coherent(decoded, leader_ca, mask)
        row["follower_coherent"] = coherent
        row["follower_bonds_A"] = [b * 10.0 if b is not None else None for b in bonds]
        if health_gate and not coherent:
            if telemetry is not None:
                telemetry.append(row)
            return output

        if mode == "blind":
            vote = blind_displacement(vote, mask, generator=gen)
        elif mode == "blocked":

            other = scramble_ca(leader_ca, mask, generator=gen)
            off_vote, _, off_have = implicit_vote(
                model.autoencoder.decode(latents, other, mask), other, mask)
            row["decoder_calls"] = 2
            direction = off_vote / off_vote.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vote = direction * vote.norm(dim=-1, keepdim=True)
            vote = torch.where((have & off_have)[..., None], vote, torch.zeros_like(vote))
        delta = alpha * cap_per_residue(vote, cap_nm)
        delta = torch.where(mask.bool()[..., None], delta, torch.zeros_like(delta))
        if not bool(delta.any()):
            if telemetry is not None:
                telemetry.append(row)
            return output

        row["fired"] = True
        moved = delta[have].norm(dim=-1) * 10.0
        row["applied_mean_A"] = float(moved.mean())
        row["applied_max_A"] = float(moved.max())

        if verify_gap:
            after = model.autoencoder.decode(latents, leader_ca + delta, mask)
            _, gap2, have2 = implicit_vote(after, leader_ca + delta, mask)
            row["decoder_calls"] += 1
            row["gap_after_A"] = float(gap2[have2].mean()) * 10.0 if bool(have2.any()) else None

        if telemetry is not None:
            telemetry.append(row)
        deltas = {"bb_ca": delta, "local_latents": torch.zeros_like(clean["local_latents"])}
        return apply_endpoint_delta(output, clean, deltas, safe["t"])

    model.call_nn = call
    try:
        yield telemetry
    finally:
        if had_instance_binding:
            model.call_nn = original
        else:
            delattr(model, "call_nn")
