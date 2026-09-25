
from __future__ import annotations

from dive.data.roles import RoleSpec, Selector

SIGNATURE_LENGTH = 12

MIN_SIGNATURE_LENGTH = 4

class StructureRoleError(RuntimeError):
    pass

def match_chains_by_sequence(
    canonical: dict[str, str], loaded: dict[str, str]
) -> dict[str, str]:

    taken: set[str] = set()
    matched: dict[str, str] = {}

    for canonical_id, sequence in canonical.items():
        candidates = []
        for loaded_id, loaded_sequence in loaded.items():
            if loaded_id in taken:
                continue
            if len(loaded_sequence) < MIN_SIGNATURE_LENGTH:
                continue
            signature = loaded_sequence[:SIGNATURE_LENGTH]
            if sequence.find(signature) >= 0:
                candidates.append((len(loaded_sequence), loaded_id))
        if not candidates:
            raise StructureRoleError(
                f"no chain in the structure matches canonical chain {canonical_id!r}; "
                f"loaded chains are {sorted(loaded)} with lengths "
                f"{ {k: len(v) for k, v in loaded.items()} }"
            )

        candidates.sort(reverse=True)
        chosen = candidates[0][1]
        matched[canonical_id] = chosen
        taken.add(chosen)

    return matched

def resolve_against_structure(
    spec: RoleSpec,
    canonical: dict[str, str],
    loaded: dict[str, str],
    non_polymer: frozenset[str] = frozenset(),
) -> RoleSpec:

    mapping = match_chains_by_sequence(canonical, loaded)

    def rename(selector_text: str) -> str:
        selector = Selector.parse(selector_text)
        if selector.chain in non_polymer:

            if not selector.is_whole_chain:
                raise StructureRoleError(
                    f"a residue range was requested on non-polymer chain "
                    f"{selector.chain!r}"
                )
            return selector.chain
        if selector.chain not in mapping:
            raise StructureRoleError(
                f"role names chain {selector.chain!r}, which is neither a canonical "
                f"chain {sorted(canonical)} nor a non-polymer chain in the structure "
                f"{sorted(non_polymer)}"
            )
        loaded_id = mapping[selector.chain]
        if selector.is_whole_chain:
            return loaded_id
        return _shift_range(selector, canonical[selector.chain], loaded[loaded_id], loaded_id)

    generated = ",".join(rename(s) for s in spec.generated.split(",") if s)
    context = ",".join(rename(s) for s in spec.context.split(",") if s)
    target = ",".join(rename(s) for s in spec.target.split(",") if s) if spec.target else ""
    return RoleSpec(generated=generated, context=context, target=target)

def _shift_range(
    selector: Selector, canonical_sequence: str, loaded_sequence: str, loaded_id: str
) -> str:

    region = canonical_sequence[selector.start : selector.end + 1]
    if not region:
        raise StructureRoleError("the selected region is empty in the canonical sequence")

    occurrences = loaded_sequence.count(region)
    if occurrences == 0:
        raise StructureRoleError(
            f"the selected region is not resolved in loaded chain {loaded_id!r}; "
            f"generating the whole chain instead would be a different task"
        )
    if occurrences > 1:
        raise StructureRoleError(
            f"the selected region appears {occurrences} times in loaded chain "
            f"{loaded_id!r}; the range is ambiguous"
        )

    start = loaded_sequence.index(region)
    return Selector(chain=loaded_id, start=start, end=start + len(region) - 1).as_text()
