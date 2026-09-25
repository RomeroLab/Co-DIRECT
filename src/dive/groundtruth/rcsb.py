
from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from dive.groundtruth.complex_spec import ChainSpec

_TIMEOUT_SECONDS = 30
_UNOBSERVED = "UNOBSERVED_RESIDUE_XYZ"

class RcsbError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class EntryMetadata:

    pdb_id: str
    chains: tuple[ChainSpec, ...]
    revision_date: str
    title: str
    source_urls: dict[str, str]

    def to_dict(self) -> dict[str, object]:

        return {
            "pdb_id": self.pdb_id,
            "revision_date": self.revision_date,
            "title": self.title,
            "chains": [chain.to_dict() for chain in self.chains],
            "source_urls": dict(self.source_urls),
        }

def entry_urls(pdb_id: str) -> dict[str, str]:

    upper = pdb_id.upper()
    return {
        "entry": f"https://data.rcsb.org/rest/v1/core/entry/{upper}",
        "polymer_entity": f"https://data.rcsb.org/rest/v1/core/polymer_entity/{upper}",
        "polymer_entity_instance": (
            f"https://data.rcsb.org/rest/v1/core/polymer_entity_instance/{upper}"
        ),
        "download": f"https://files.rcsb.org/download/{upper}.cif",
    }

def _urlopen(url: str) -> bytes:

    with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
        return response.read()

def fetch_json(url: str, *, opener: Callable[[str], bytes] = _urlopen) -> dict:

    try:
        body = opener(url)
    except OSError as error:
        raise RcsbError(f"could not fetch {url}: {error}") from error
    try:
        return json.loads(body)
    except ValueError as error:
        raise RcsbError(f"response from {url} is not JSON: {error}") from error

def unobserved_residue_count(instance_json: Mapping) -> int:

    total = 0
    for feature in instance_json.get("rcsb_polymer_instance_feature") or []:
        if feature.get("type") != _UNOBSERVED:
            continue
        for position in feature.get("feature_positions") or []:
            begin = position.get("beg_seq_id")
            if begin is None:
                continue
            end = position.get("end_seq_id", begin)
            total += int(end) - int(begin) + 1
    return total

def parse_entry(
    entry_json: Mapping,
    polymer_json: Sequence[Mapping],
    instance_json: Mapping[str, Mapping],
) -> EntryMetadata:

    identifiers = entry_json.get("rcsb_entry_container_identifiers") or {}
    pdb_id = identifiers.get("entry_id") or entry_json.get("rcsb_id")
    if not pdb_id:
        raise RcsbError("entry response carries no entry id")

    accession = entry_json.get("rcsb_accession_info") or {}
    revision_date = str(accession.get("revision_date") or "")
    struct = entry_json.get("struct") or {}
    title = str(struct.get("title") or "")

    chains: list[ChainSpec] = []
    for entity in polymer_json:
        entity_ids = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        label_ids = entity_ids.get("asym_ids") or []
        auth_ids = entity_ids.get("auth_asym_ids") or []
        if len(label_ids) != len(auth_ids):
            raise RcsbError(
                f"{pdb_id}: asym_ids and auth_asym_ids differ in length "
                f"({len(label_ids)} vs {len(auth_ids)})"
            )

        poly = entity.get("entity_poly") or {}
        sample_length = poly.get("rcsb_sample_sequence_length")
        polymer_type = str(poly.get("rcsb_entity_polymer_type") or "")
        if sample_length is None:
            raise RcsbError(
                f"polymer entity of {pdb_id} carries no rcsb_sample_sequence_length"
            )

        for label_id, auth_id in zip(label_ids, auth_ids, strict=True):
            instance = instance_json.get(str(label_id))
            if instance is None:
                raise RcsbError(
                    f"{pdb_id}: no instance record for label chain {label_id!r} "
                    f"(author {auth_id!r}); modeled residue count is unknown"
                )
            modeled = int(sample_length) - unobserved_residue_count(instance)
            chains.append(
                ChainSpec(
                    chain_id=str(auth_id),
                    n_residues=max(modeled, 0),
                    is_protein=polymer_type.lower().startswith("protein"),
                )
            )

    if not chains:
        raise RcsbError(f"no polymer chain parsed for {pdb_id}")

    return EntryMetadata(
        pdb_id=str(pdb_id),
        chains=tuple(sorted(chains, key=lambda c: c.chain_id)),
        revision_date=revision_date,
        title=title,
        source_urls=entry_urls(str(pdb_id)),
    )
