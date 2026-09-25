
from __future__ import annotations

import torch
import torch.nn.functional as F

from dive.sdx.cavity import (
    SDX_CAVITY, SDX_IDENTITY, SDX_QUERY, build_cavity_batch,
    build_local_cavity_batch, query_span,
)
from dive.sdx.config import SdxConfig
from dive.sdx.exchange import apply_variant, message_features
from dive.sdx.relations import anchor_positions, relation_distances, relation_targets
from dive.sdx.wiring import SDX_MESSAGE

MAX_ANCHORS = 8

def rows_to_score(valid: torch.Tensor) -> torch.Tensor:

    return valid

def relation_loss(
    logits: torch.Tensor, target: torch.Tensor, keep: torch.Tensor
) -> tuple[torch.Tensor, int]:

    scored = int(keep.sum())
    if scored == 0:
        return logits.sum() * 0.0, 0
    flat_logits = logits[keep]
    flat_target = target[keep]
    return F.cross_entropy(flat_logits, flat_target), scored

def _tokens(adapter, n: int) -> torch.Tensor | None:
    captured = adapter.capture.captured
    if captured is None:
        return None
    return captured[:, :n]

def sdx_training_step(
    model,
    batch: dict,
    adapter,
    config: SdxConfig,
    *,
    batch_idx: int = 0,
    device=None,
) -> tuple[torch.Tensor, dict]:

    from proteinfoundation.utils.sample_utils import add_clean_samples
    from proteinfoundation.utils.training_handlers import handle_batch_conditioning

    if device is None:
        device = next(model.parameters()).device
    batch = {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    batch = add_clean_samples(
        batch, model.cfg_exp.product_flowmatcher, getattr(model, "autoencoder", None)
    )
    batch = model.fm.corrupt_batch(batch)
    batch_size = batch["mask"].shape[0]

    adapter.capture.clear()
    batch, n_recycle = handle_batch_conditioning(
        batch, batch_size, model.cfg_exp.training, model.call_nn, model.fm
    )
    geometry_tokens = _tokens(adapter, batch["mask"].shape[1])

    mask = batch["mask"]
    n = mask.shape[1]
    target_bins, target_valid, _, anchors = relation_targets(
        batch["coords_nm"], batch["coord_mask"], batch["x_motif"],
        batch["motif_mask"], batch["seq_motif"], mask, max_anchors=MAX_ANCHORS,
    )
    anchor_valid = target_valid.any(dim=1)
    keep = rows_to_score(target_valid)

    parts: dict[str, float] = {}
    losses: list[torch.Tensor] = []

    cavity_logits = None
    if config.trains_relation_head:
        adapter.capture.clear()
        if config.cavity == "local":
            query = query_span(mask.to(torch.bool), fraction=config.query_fraction)
            cavity = build_local_cavity_batch(
                batch, model.fm, identity=batch["residue_type"], query=query)

            keep = keep & query[:, :, None]
            parts["query_residues"] = float(query.sum())
        else:
            cavity = build_cavity_batch(batch, model.fm, identity=batch["residue_type"])
        cavity["use_residue_type_feature"] = True
        cavity["use_ca_coors_nm_feature"] = False
        if not config.extra_identity_embedding:
            cavity.pop(SDX_IDENTITY, None)
        model.nn(cavity)
        cavity_tokens = _tokens(adapter, n)
        if cavity_tokens is None:
            raise RuntimeError(
                "the cavity pass produced no token capture; the operator is not "
                "installed on this model"
            )
        cavity_logits = adapter.relation(cavity_tokens, anchors, anchor_valid)
        loss_cavity, scored = relation_loss(cavity_logits, target_bins, keep)
        losses.append(config.relation_weight * loss_cavity)
        parts["relation_cavity"] = float(loss_cavity.detach())
        parts["relation_scored_pairs"] = float(scored)

    message = None
    if config.deliver_message and cavity_logits is not None:
        clean = batch.get("x_sc")
        if clean is not None and geometry_tokens is not None:
            with torch.no_grad():
                geometry_logits = adapter.relation(geometry_tokens, anchors, anchor_valid)
                anchor_nm, _ = anchor_positions(
                    batch["x_motif"], batch["motif_mask"], max_anchors=MAX_ANCHORS
                )
                usable_distance, usable = relation_distances(
                    clean["bb_ca"], mask.to(torch.bool), anchor_nm, anchor_valid
                )
                distance = usable_distance
                identity_logits = (
                    cavity_logits.detach() if config.source == "cavity" else geometry_logits
                )
                message = message_features(
                    identity_logits, geometry_logits, distance, usable
                )
                message = apply_variant(message, config.variant)
            parts["message_absolute_mean"] = float(message.abs().mean())
            parts["message_rows_nonzero"] = float((message.abs().sum(-1) > 0).float().mean())

    adapter.capture.clear()
    if message is not None:
        batch[SDX_MESSAGE] = message
    else:
        batch.pop(SDX_MESSAGE, None)
    batch.pop(SDX_CAVITY, None)
    batch.pop(SDX_IDENTITY, None)

    nn_out = model.call_nn(batch, n_recycle=n_recycle)
    fm_losses = model.fm.compute_loss(batch=batch, nn_out=nn_out)
    fm_loss = sum(torch.mean(fm_losses[k]) for k in fm_losses if "_justlog" not in k)
    losses.append(fm_loss)
    for key, value in fm_losses.items():
        parts[key] = float(torch.mean(value).detach())
    parts["flow_matching"] = float(fm_loss.detach())

    if config.trains_relation_head and not batch.get("use_ca_coors_nm_feature", False):
        main_tokens = _tokens(adapter, n)
        if main_tokens is not None:
            full_logits = adapter.relation(main_tokens, anchors, anchor_valid)
            loss_full, scored_full = relation_loss(full_logits, target_bins, keep)
            losses.append(config.relation_weight * loss_full)
            parts["relation_full"] = float(loss_full.detach())
            parts["relation_full_scored_pairs"] = float(scored_full)

    total = sum(losses)
    parts["total"] = float(total.detach())
    return total, parts

def diagnose_identity_use(model, batch, adapter, config, *, device=None) -> dict:

    from proteinfoundation.utils.sample_utils import add_clean_samples

    if device is None:
        device = next(model.parameters()).device
    batch = {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
    batch = add_clean_samples(
        batch, model.cfg_exp.product_flowmatcher, getattr(model, "autoencoder", None)
    )
    batch = model.fm.corrupt_batch(batch)
    mask = batch["mask"]
    n = mask.shape[1]
    target_bins, target_valid, _, anchors = relation_targets(
        batch["coords_nm"], batch["coord_mask"], batch["x_motif"],
        batch["motif_mask"], batch["seq_motif"], mask, max_anchors=MAX_ANCHORS,
    )
    anchor_valid = target_valid.any(dim=1)
    keep = rows_to_score(target_valid)
    query = None
    if config.cavity == "local":
        query = query_span(mask.to(torch.bool), fraction=config.query_fraction)
        keep = keep & query[:, :, None]

    def score(identity: torch.Tensor) -> float:
        adapter.capture.clear()
        if query is not None:
            cavity = build_local_cavity_batch(
                batch, model.fm, identity=identity, query=query)
        else:
            cavity = build_cavity_batch(batch, model.fm, identity=identity)
        cavity["residue_type"] = identity
        cavity["use_residue_type_feature"] = True
        cavity["use_ca_coors_nm_feature"] = False
        cavity.pop(SDX_IDENTITY, None)
        model.nn(cavity)
        tokens = _tokens(adapter, n)
        logits = adapter.relation(tokens, anchors, anchor_valid)
        loss, _ = relation_loss(logits, target_bins, keep)
        return float(loss.detach())

    native = batch["residue_type"]
    permuted = native.clone()
    for row in range(native.shape[0]):
        permuted[row] = native[row][torch.randperm(n, device=native.device)]
    with torch.no_grad():
        native_loss = score(native)
        shuffled_loss = score(permuted)
    adapter.capture.clear()
    return {
        "relation_cavity_native_identity": native_loss,
        "relation_cavity_shuffled_identity": shuffled_loss,
        "identity_gain_nats": shuffled_loss - native_loss,
        "scored_pairs": float(keep.sum()),
    }
