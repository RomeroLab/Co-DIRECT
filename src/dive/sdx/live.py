
from __future__ import annotations

import torch

from dive.sdx.cavity import SDX_CAVITY, SDX_IDENTITY, build_cavity_batch
from dive.sdx.config import SdxConfig
from dive.sdx.exchange import apply_variant, message_features
from dive.sdx.relations import anchor_features, anchor_positions, relation_distances
from dive.sdx.training import MAX_ANCHORS

class LiveExchange:

    def __init__(self, model, adapter, config: SdxConfig):

        self._model = [model]
        self._adapter = [adapter]
        self.config = config
        self.stats = {
            "calls": 0,
            "messages_sent": 0,
            "no_clean_prediction": 0,
            "no_geometry_view": 0,
            "withheld": 0,
            "decodes": 0,
            "errors": 0,

            "sum_abs_mean": 0.0,
            "sum_nll_identity": 0.0,
            "sum_entropy_identity": 0.0,
            "sum_rows_nonzero": 0.0,

            "sum_abs_by_column": [0.0] * 8,
        }
        self._cached: torch.Tensor | None = None
        self._cached_shape: tuple | None = None
        self._identity: torch.Tensor | None = None
        self._busy = False

    @property
    def model(self):
        return self._model[0]

    @property
    def adapter(self):
        return self._adapter[0]

    def message_for(self, batch: dict) -> torch.Tensor | None:
        if self._busy:
            return None
        self.stats["calls"] += 1
        clean = batch.get("x_sc")
        if clean is None:
            self.stats["no_clean_prediction"] += 1
            return None
        mask = batch["mask"]
        shape = tuple(mask.shape)
        n = shape[1]

        adapter = self.adapter
        geometry_tokens = adapter.capture.captured
        if geometry_tokens is None:
            self.stats["no_geometry_view"] += 1
            return None
        geometry_tokens = geometry_tokens[:, :n]

        due = (self.stats["calls"] - 1) % self.config.stride == 0
        if not due:

            self.stats["withheld"] += 1
            return None

        for key in ("x_motif", "motif_mask", "seq_motif"):
            if key not in batch:
                raise KeyError(
                    f"the generation batch carries no `{key}`, so the supplied "
                    "anchors cannot be located; refusing to send a message "
                    "computed against something else"
                )
        anchor_nm, anchor_valid = anchor_positions(
            batch["x_motif"], batch["motif_mask"], max_anchors=MAX_ANCHORS
        )
        anchors = anchor_features(
            batch["x_motif"], batch["motif_mask"], batch["seq_motif"],
            max_anchors=MAX_ANCHORS,
        )

        self._busy = True
        try:
            with torch.no_grad():
                decoded = self.model.autoencoder.decode(
                    clean["local_latents"], clean["bb_ca"], mask
                )
                identity = decoded["residue_type"].long()
                self.stats["decodes"] += 1

                cavity = build_cavity_batch(batch, self.model.fm, identity=identity)
                cavity["residue_type"] = identity
                cavity["use_residue_type_feature"] = True
                cavity["use_ca_coors_nm_feature"] = False
                if not self.config.extra_identity_embedding:
                    cavity.pop(SDX_IDENTITY, None)
                cavity[SDX_CAVITY] = True
                adapter.capture.clear()
                self.model.nn(cavity)
                cavity_tokens = adapter.capture.captured
                if cavity_tokens is None:
                    raise RuntimeError("the cavity pass produced no token capture")
                cavity_tokens = cavity_tokens[:, :n]

                cavity_logits = adapter.relation(cavity_tokens, anchors, anchor_valid)
                geometry_logits = adapter.relation(geometry_tokens, anchors, anchor_valid)
                distance, usable = relation_distances(
                    clean["bb_ca"], mask.to(torch.bool), anchor_nm, anchor_valid
                )
                identity_logits = (
                    cavity_logits if self.config.source == "cavity" else geometry_logits
                )
                message = message_features(
                    identity_logits, geometry_logits, distance, usable
                )
                message = apply_variant(message, self.config.variant)
        finally:
            self._busy = False

            adapter.capture.clear()

        self._cached, self._cached_shape = message, shape
        self.stats["messages_sent"] += 1
        self.stats["sum_abs_mean"] += float(message.abs().mean())
        self.stats["sum_nll_identity"] += float(message[..., 0].mean())
        self.stats["sum_entropy_identity"] += float(message[..., 2].mean())
        self.stats["sum_rows_nonzero"] += float((message.abs().sum(-1) > 0).float().mean())
        per_column = message.abs().mean(dim=(0, 1))
        for index in range(min(8, per_column.shape[0])):
            self.stats["sum_abs_by_column"][index] += float(per_column[index])
        return message

    def summary(self) -> dict:

        sent = max(self.stats["messages_sent"], 1)
        out = {k: v for k, v in self.stats.items() if not k.startswith("sum_")}
        for key in ("abs_mean", "nll_identity", "entropy_identity", "rows_nonzero"):
            out[f"mean_{key}"] = round(self.stats[f"sum_{key}"] / sent, 5)

        measured = max(self.stats.get("injections_measured", 0), 1)
        for key in ("delta_rms", "repr_rms", "delta_over_repr"):
            if f"sum_{key}" in self.stats:
                out[f"mean_{key}"] = round(self.stats[f"sum_{key}"] / measured, 6)
        from dive.sdx.exchange import FEATURE_NAMES

        out["mean_abs_by_column"] = {
            name: round(value / sent, 5)
            for name, value in zip(FEATURE_NAMES, self.stats["sum_abs_by_column"])
        }
        return out

def install_live_exchange(model, adapter, config: SdxConfig) -> LiveExchange:

    exchange = LiveExchange(model, adapter, config)
    adapter.factory.attach_live(exchange)
    return exchange
