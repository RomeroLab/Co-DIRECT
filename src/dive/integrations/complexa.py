
from __future__ import annotations

from dive.codirect_paths import joined

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from dive.counterfactual.registry import flatten_paths
from dive.provenance import GitEvidence, ProvenanceError, verify_git_checkout

UPSTREAM_PIN = "916eaaedce5b07c205efb6ef32370c01d366591e"
UPSTREAM_ROOT = Path(
    joined('PROTEINA_COMPLEXA_ROOT')
)

_MODALITIES: tuple[str, ...] = ("bb_ca", "local_latents")

_PAIR_FEATURE_SUFFIX = "_pair_dists"

class UpstreamContractError(RuntimeError):
    pass

def assert_pinned(root: Path | None = None) -> GitEvidence:

    checkout = Path(root) if root is not None else UPSTREAM_ROOT
    try:
        return verify_git_checkout(checkout, UPSTREAM_PIN)
    except ProvenanceError as error:
        raise UpstreamContractError(
            f"upstream must be pinned to {UPSTREAM_PIN}: {error}"
        ) from error

@dataclass(frozen=True, slots=True)
class ComplexaAdapter:

    output_parameterization: Mapping[str, str]

    @classmethod
    def from_config(cls, cfg: Mapping) -> ComplexaAdapter:

        try:
            parameterization = cfg["nn"]["output_parameterization"]
        except (KeyError, TypeError) as error:
            raise UpstreamContractError(
                "resolved config does not carry nn.output_parameterization"
            ) from error
        missing = [m for m in _MODALITIES if m not in parameterization]
        if missing:
            raise UpstreamContractError(
                "output_parameterization is missing " + ", ".join(missing)
            )
        return cls(
            output_parameterization={m: str(parameterization[m]) for m in _MODALITIES}
        )

    def extract_velocities(self, nn_out: Mapping) -> dict[str, Tensor]:

        extracted: dict[str, Tensor] = {}
        for modality in _MODALITIES:
            if modality not in nn_out:
                raise UpstreamContractError(
                    f"nn output is missing modality '{modality}'"
                )
            entry = nn_out[modality]
            if not isinstance(entry, Mapping):
                raise UpstreamContractError(
                    f"nn output for '{modality}' is not a mapping"
                )
            if len(entry) != 1:
                raise UpstreamContractError(
                    f"nn output for '{modality}' must carry exactly one key, "
                    f"observed {sorted(entry)}"
                )
            key = self.output_parameterization[modality]
            if key not in entry:
                raise UpstreamContractError(
                    f"nn output for '{modality}' lacks configured key '{key}', "
                    f"observed {sorted(entry)}"
                )
            extracted[modality] = entry[key]
        return extracted

    def sample_backbone_prior(self, model: object, batch: Mapping) -> Tensor:

        current = batch["x_t"]["bb_ca"]
        mask = batch["mask"]
        matcher = model.fm.base_flow_matchers["bb_ca"]
        sampled = matcher.sample_noise(
            n=current.shape[-2],
            shape=tuple(current.shape[:-2]),
            device=current.device,
            mask=mask,
            training=False,
        )
        if sampled.shape != current.shape or sampled.dtype != current.dtype:
            raise UpstreamContractError("bb_ca prior shape/dtype drift")
        return sampled

    def decode_endpoint(
        self,
        model: object,
        *,
        z_latent: Tensor,
        ca_coors_nm: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:

        _assert_decoder_inputs(z_latent, ca_coors_nm, mask)
        try:
            decoded = model.autoencoder.decode(
                z_latent=z_latent, ca_coors_nm=ca_coors_nm, mask=mask
            )
        except (AttributeError, TypeError, KeyError) as error:
            raise UpstreamContractError(
                f"autoencoder decoder contract failed: {error}"
            ) from error
        if not isinstance(decoded, Mapping):
            raise UpstreamContractError("autoencoder decoder did not return a mapping")
        try:
            logits = decoded["seq_logits"]
            atom37 = decoded["coors_nm"]
            residue_mask = decoded["residue_mask"]
            atom_mask = decoded["atom_mask"]
        except KeyError as error:
            raise UpstreamContractError(
                f"autoencoder decoder missing output {error}"
            ) from error
        length = z_latent.shape[1]
        if (
            not isinstance(logits, Tensor)
            or not logits.is_floating_point()
            or logits.shape != (1, length, 20)
            or logits.device != z_latent.device
            or not bool(torch.isfinite(logits).all())
        ):
            raise UpstreamContractError("autoencoder decoder seq_logits contract drift")
        if (
            not isinstance(atom37, Tensor)
            or not atom37.is_floating_point()
            or atom37.shape != (1, length, 37, 3)
            or atom37.device != z_latent.device
            or not bool(torch.isfinite(atom37).all())
        ):
            raise UpstreamContractError("autoencoder decoder atom37 contract drift")
        if (
            not isinstance(residue_mask, Tensor)
            or residue_mask.dtype is not torch.bool
            or residue_mask.shape != mask.shape
            or residue_mask.device != mask.device
            or not bool(torch.equal(residue_mask, mask))
        ):
            raise UpstreamContractError(
                "autoencoder decoder residue mask contract drift"
            )
        if (
            not isinstance(atom_mask, Tensor)
            or atom_mask.dtype is not torch.bool
            or atom_mask.shape != (1, length, 37)
            or atom_mask.device != mask.device
            or bool(torch.any(atom_mask & ~mask[..., None]))
        ):
            raise UpstreamContractError("autoencoder decoder atom mask contract drift")
        return logits, atom37

    def recompute_pair_features(self, batch: dict) -> dict:

        carried = sorted(
            path
            for path in flatten_paths(batch)
            if path.rsplit(".", 1)[-1].endswith(_PAIR_FEATURE_SUFFIX)
        )
        if carried:
            raise UpstreamContractError(
                "batch carries precomputed pair feature(s) "
                + ", ".join(carried)
                + "; they must be rebuilt from this batch's own coordinates by the "
                "pinned upstream feature factory before any counterfactual is "
                "evaluated, and this adapter has no reviewed binding for them"
            )
        return batch

    def router_feature_paths(self) -> tuple[str, ...]:

        return ("x_t.bb_ca", "x_t.local_latents", "t", "mask")

    def to_dict(self) -> dict[str, object]:

        return {
            "upstream_pin": UPSTREAM_PIN,
            "upstream_root": str(UPSTREAM_ROOT),
            "output_parameterization": dict(self.output_parameterization),
        }

def _assert_decoder_inputs(z_latent: Tensor, ca_coors_nm: Tensor, mask: Tensor) -> None:
    if (
        not isinstance(z_latent, Tensor)
        or not z_latent.is_floating_point()
        or z_latent.ndim != 3
        or z_latent.shape[0] != 1
        or not bool(torch.isfinite(z_latent).all())
    ):
        raise UpstreamContractError("decoder z_latent contract drift")
    if (
        not isinstance(ca_coors_nm, Tensor)
        or not ca_coors_nm.is_floating_point()
        or ca_coors_nm.shape != (*z_latent.shape[:2], 3)
        or ca_coors_nm.dtype != z_latent.dtype
        or ca_coors_nm.device != z_latent.device
        or not bool(torch.isfinite(ca_coors_nm).all())
    ):
        raise UpstreamContractError("decoder ca_coors_nm contract drift")
    if (
        not isinstance(mask, Tensor)
        or mask.dtype is not torch.bool
        or mask.shape != z_latent.shape[:2]
        or mask.device != z_latent.device
        or mask.numel() == 0
        or not bool(mask.any())
    ):
        raise UpstreamContractError("decoder mask contract drift")
