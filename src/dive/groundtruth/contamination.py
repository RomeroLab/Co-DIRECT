
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

CUTOFF_DATE = "2024-01-01"

IDENTITY_THRESHOLD = 90

class ContaminationError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class Assessment:

    pdb_id: str
    clean: bool
    relatives: int
    group_ids: tuple[str, ...]
    threshold: int

    def to_dict(self) -> dict[str, object]:

        return {
            "pdb_id": self.pdb_id,
            "clean": self.clean,
            "pre_cutoff_relatives": self.relatives,
            "group_ids": list(self.group_ids),
            "threshold": self.threshold,
            "cutoff_date": CUTOFF_DATE,
        }

def group_ids_at(entity_json: Mapping, threshold: int) -> tuple[str, ...]:

    ids: list[str] = []
    for group in entity_json.get("rcsb_polymer_entity_group_membership") or []:
        if group.get("aggregation_method") != "sequence_identity":
            continue
        cutoff = group.get("similarity_cutoff")
        if cutoff is None or int(cutoff) != int(threshold):
            continue
        group_id = group.get("group_id")
        if group_id:
            ids.append(str(group_id))
    return tuple(ids)

def assess(
    pdb_id: str,
    entity_docs: Sequence[Mapping],
    *,
    counter: Callable[[str], int],
    threshold: int = IDENTITY_THRESHOLD,
) -> Assessment:

    if not entity_docs:
        raise ContaminationError(f"{pdb_id}: no polymer entity to assess")

    all_ids: list[str] = []
    relatives = 0
    for index, doc in enumerate(entity_docs):
        ids = group_ids_at(doc, threshold)
        if not ids:
            raise ContaminationError(
                f"{pdb_id}: entity {index} has no sequence cluster at {threshold}% "
                f"identity, so contamination cannot be assessed"
            )
        all_ids.extend(ids)
        for group_id in ids:
            relatives += int(counter(group_id))

    return Assessment(
        pdb_id=pdb_id,
        clean=relatives == 0,
        relatives=relatives,
        group_ids=tuple(all_ids),
        threshold=int(threshold),
    )
