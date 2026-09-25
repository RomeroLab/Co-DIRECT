
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from dive.benchmark.task_cards import SequencePolicy, TaskCard
from dive.training.preflight import canonical_json_bytes

class AdapterError(RuntimeError):
    pass

class UnsupportedContract(AdapterError):

    status = "UNSUPPORTED_CONTRACT"

@dataclass(frozen=True, slots=True)
class MethodInput:
    method_id: str
    payload: Mapping[str, object]
    provided_conditions: tuple[str, ...]
    hidden_conditions: tuple[str, ...]
    unsupported_conditions: tuple[str, ...]
    derived_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def condition_digest(self) -> str:
        payload = {
            "derived": list(self.derived_conditions),
            "hidden": list(self.hidden_conditions),
            "method_id": self.method_id,
            "payload": _canonical_payload(self.payload),
            "provided": list(self.provided_conditions),
            "unsupported": list(self.unsupported_conditions),
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

@dataclass(frozen=True, slots=True)
class MethodOutput:
    mmcif: str
    fasta: str
    sequence_origin: SequencePolicy
    missing: bool = False
    exception: str | None = None
    timeout: bool = False
    oom: bool = False

def _canonical_payload(payload: Mapping[str, object]) -> dict[str, object]:
    def convert(value: object) -> object:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in sorted(value.items())}
        return value

    return {str(key): convert(value) for key, value in sorted(payload.items())}

class CanonicalAdapter:
    method_id = "identity"

    def encode(self, card: TaskCard) -> MethodInput:
        raise NotImplementedError

    def decode(
        self,
        method_input: MethodInput,
        *,
        mmcif: str,
        fasta: str,
        sequence_origin: SequencePolicy,
        missing: bool = False,
        exception: str | None = None,
        timeout: bool = False,
        oom: bool = False,
    ) -> MethodOutput:
        if missing and exception is None:
            raise AdapterError("missing samples must preserve the original exception")
        return MethodOutput(
            mmcif=mmcif,
            fasta=fasta,
            sequence_origin=sequence_origin,
            missing=missing,
            exception=exception,
            timeout=timeout,
            oom=oom,
        )

    def roundtrip(self, card: TaskCard) -> TaskCard:
        encoded = self.encode(card)
        reconstructed = CanonicalRoundtrip(self).reconstruct(card, encoded)
        if reconstructed.semantic_sha256() != card.semantic_sha256():
            raise AdapterError("canonical reconstruction drifted")
        return reconstructed

class IdentityAdapter(CanonicalAdapter):
    method_id = "identity"

    def encode(self, card: TaskCard) -> MethodInput:
        return MethodInput(
            method_id=self.method_id,
            payload=dict(card.as_mapping()),
            provided_conditions=card.allowed_inputs,
            hidden_conditions=card.hidden_reference,
            unsupported_conditions=(),
        )

class CanonicalRoundtrip:
    def __init__(self, adapter: CanonicalAdapter) -> None:
        self._adapter = adapter

    def reconstruct(self, original: TaskCard, encoded: MethodInput) -> TaskCard:
        payload = encoded.payload
        for field in (
            "conditioning_chains",
            "generated_chains",
            "motif_or_epitope",
            "design_regions",
        ):
            observed = tuple(payload[field])
            expected = getattr(original, field)
            if observed != expected:
                raise AdapterError(f"{field} chain or region identity drifted")
        original_maps = original.as_mapping()["residue_atom_maps"]
        if payload["residue_atom_maps"] != original_maps:
            raise AdapterError("atom names or mask drifted")
        if payload.get("ligand_graph") != original.as_mapping()["ligand_graph"]:
            raise AdapterError("ligand atoms, bonds, or charge drifted")
        return original

class _BinderHotspotAdapter(CanonicalAdapter):
    method_id = "binder_hotspot"

    def encode(self, card: TaskCard) -> MethodInput:
        if card.family.value != "binder":
            raise AdapterError("binder adapter received a non-binder card")
        return MethodInput(
            method_id=self.method_id,
            payload={
                "conditioning_chains": card.conditioning_chains,
                "design_regions": card.design_regions,
                "generated_chains": card.generated_chains,
                "hotspots": card.motif_or_epitope,
                "ligand_graph": None,
                "motif_or_epitope": card.motif_or_epitope,
                "residue_atom_maps": card.as_mapping()["residue_atom_maps"],
                "target_coordinates": True,
            },
            provided_conditions=("target_coordinates", "hotspots"),
            hidden_conditions=("native_dock", "native_binder_sequence"),
            unsupported_conditions=(),
        )

class _AMELigandMotifAdapter(CanonicalAdapter):
    method_id = "ame_ligand_motif"

    def __init__(self, *, supports_unindexed_catalytic_atoms: bool) -> None:
        self._supports = supports_unindexed_catalytic_atoms

    def encode(self, card: TaskCard) -> MethodInput:
        if card.family.value != "ame":
            raise AdapterError("AME adapter received a non-AME card")
        if not self._supports:
            raise UnsupportedContract(
                "method cannot consume unindexed catalytic atoms without "
                "adding native index or rotamer"
            )
        if card.ligand_graph is None:
            raise AdapterError("AME encode requires a ligand graph")
        graph = card.as_mapping()["ligand_graph"]
        return MethodInput(
            method_id=self.method_id,
            payload={
                "catalytic_atoms": card.motif_or_epitope,
                "conditioning_chains": card.conditioning_chains,
                "design_regions": card.design_regions,
                "generated_chains": card.generated_chains,
                "ligand_graph": graph,
                "motif_or_epitope": card.motif_or_epitope,
                "residue_atom_maps": card.as_mapping()["residue_atom_maps"],
            },
            provided_conditions=(
                "ligand_coordinates",
                "ligand_graph",
                "catalytic_named_atoms",
            ),
            hidden_conditions=("native_catalytic_index", "native_rotamer"),
            unsupported_conditions=(),
        )

class _AntibodyCdrAdapter(CanonicalAdapter):
    method_id = "antibody_cdr_h3"

    def encode(self, card: TaskCard) -> MethodInput:
        if card.family.value != "antibody":
            raise AdapterError("antibody adapter received a non-antibody card")
        return MethodInput(
            method_id=self.method_id,
            payload={
                "cdr_h3_length": card.length,
                "cdr_mask": card.design_regions,
                "conditioning_chains": card.conditioning_chains,
                "design_regions": card.design_regions,
                "epitope_hotspots": card.motif_or_epitope,
                "framework_id": card.metadata.get("framework_id"),
                "generated_chains": card.generated_chains,
                "ligand_graph": None,
                "motif_or_epitope": card.motif_or_epitope,
                "residue_atom_maps": card.as_mapping()["residue_atom_maps"],
            },
            provided_conditions=(
                "antigen_coordinates",
                "epitope_hotspots",
                "framework",
                "cdr_mask",
            ),
            hidden_conditions=("native_cdr_sequence", "native_dock"),
            unsupported_conditions=(),
        )

class _CommonInverseFoldingAdapter(CanonicalAdapter):
    method_id = "common_inverse_folding"

    def encode(self, card: TaskCard) -> MethodInput:
        return MethodInput(
            method_id=self.method_id,
            payload={
                "conditioning_chains": card.conditioning_chains,
                "design_regions": card.design_regions,
                "fixed_residues": card.fixed_regions,
                "generated_chains": card.generated_chains,
                "ligand_graph": card.as_mapping()["ligand_graph"],
                "motif_or_epitope": card.motif_or_epitope,
                "residue_atom_maps": card.as_mapping()["residue_atom_maps"],
            },
            provided_conditions=("backbone", "fixed_residues", "ligand_context"),
            hidden_conditions=card.hidden_reference,
            unsupported_conditions=(),
            derived_conditions=("redesigned_sequence",),
        )

def binder_hotspot_adapter() -> CanonicalAdapter:
    return _BinderHotspotAdapter()

def ame_ligand_motif_adapter(
    *, supports_unindexed_catalytic_atoms: bool
) -> CanonicalAdapter:
    return _AMELigandMotifAdapter(
        supports_unindexed_catalytic_atoms=supports_unindexed_catalytic_atoms
    )

def antibody_cdr_adapter() -> CanonicalAdapter:
    return _AntibodyCdrAdapter()

def common_inverse_folding_adapter() -> CanonicalAdapter:
    return _CommonInverseFoldingAdapter()
