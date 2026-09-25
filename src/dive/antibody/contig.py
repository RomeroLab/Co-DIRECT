
from __future__ import annotations

class ContigError(RuntimeError):
    pass

def contiguous_ranges(indices) -> list[tuple[int, int]]:

    ordered = sorted(set(int(i) for i in indices))
    runs: list[tuple[int, int]] = []
    for value in ordered:
        if runs and value == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], value)
        else:
            runs.append((value, value))
    return runs

def contig_from_masks(chain_ids, res_ids, design_mask) -> dict:

    chain_ids = list(chain_ids)
    res_ids = list(res_ids)
    design_mask = [bool(v) for v in design_mask]
    if not (len(chain_ids) == len(res_ids) == len(design_mask)):
        raise ContigError(
            f"length mismatch: {len(chain_ids)} chain ids, {len(res_ids)} residue "
            f"ids, {len(design_mask)} mask entries; they must describe one order"
        )
    if not any(design_mask):
        raise ContigError(
            "no designed residue; an all-false design mask generates nothing"
        )
    if all(design_mask):
        raise ContigError(
            "every residue is designed; that is de novo generation, not CDR "
            "redesign, and nothing would be given as framework"
        )

    designed = [i for i, flag in enumerate(design_mask) if flag]
    if contiguous_ranges(designed) != [(designed[0], designed[-1])]:
        raise ContigError(
            f"the designed residues {designed} are not contiguous; CDR-H3 is one "
            f"loop, so a split mask means the roles were applied wrongly"
        )

    given: list[str] = []
    for chain in _chain_order(chain_ids):
        kept = [
            res_ids[i]
            for i, (c, flag) in enumerate(zip(chain_ids, design_mask))
            if c == chain and not flag
        ]
        given.extend(f"{chain}{lo}-{hi}" for lo, hi in contiguous_ranges(kept))

    return {
        "given": given,
        "position": "/".join(given),
        "design_length": len(designed),
        "design_span": (designed[0], designed[-1]),
    }

def _chain_order(chain_ids) -> list[str]:

    seen: list[str] = []
    for chain in chain_ids:
        if chain not in seen:
            seen.append(chain)
    return seen

def align_author_ids(dive_chain_ids, author_residues) -> list:

    dive_chain_ids = list(dive_chain_ids)
    author_residues = [(str(c), r) for c, r in author_residues]
    if len(dive_chain_ids) != len(author_residues):
        raise ContigError(
            f"residue count differs: {len(dive_chain_ids)} in DIVE's order, "
            f"{len(author_residues)} in the file's; positional alignment would "
            f"shift the design mask"
        )

    dive_order = _chain_order(dive_chain_ids)
    author_order = _chain_order([c for c, _ in author_residues])
    if len(dive_order) != len(author_order):
        raise ContigError(
            f"chain count differs: DIVE has {dive_order}, the file has "
            f"{author_order}"
        )
    mapping = dict(zip(dive_order, author_order))
    if len(set(mapping.values())) != len(mapping):
        raise ContigError(f"chain map is not one-to-one: {mapping}")

    for dive_chain, author_chain in mapping.items():
        left = sum(1 for c in dive_chain_ids if c == dive_chain)
        right = sum(1 for c, _ in author_residues if c == author_chain)
        if left != right:
            raise ContigError(
                f"chain {dive_chain!r}->{author_chain!r} residue count differs: "
                f"{left} vs {right}"
            )

    per_chain: dict[str, list] = {}
    for chain, res in author_residues:
        per_chain.setdefault(chain, []).append((chain, res))
    cursor = dict.fromkeys(per_chain, 0)
    aligned = []
    for dive_chain in dive_chain_ids:
        author_chain = mapping[dive_chain]
        index = cursor[author_chain]
        aligned.append(per_chain[author_chain][index])
        cursor[author_chain] = index + 1
    return aligned

def split_roles(
    chain_ids, res_ids, design_mask, *, context_chains, target_chains
) -> dict:

    chain_ids = list(chain_ids)
    res_ids = list(res_ids)
    design_mask = [bool(v) for v in design_mask]
    if not (len(chain_ids) == len(res_ids) == len(design_mask)):
        raise ContigError(
            f"length mismatch: {len(chain_ids)}, {len(res_ids)}, {len(design_mask)}"
        )
    present = set(chain_ids)
    for chain in context_chains:
        if chain not in present:
            raise ContigError(
                f"context chain {chain!r} is not in the structure; present: "
                f"{sorted(present)}"
            )
    for chain in target_chains:
        if chain not in present:
            raise ContigError(
                f"target chain {chain!r} is not in the structure; present: "
                f"{sorted(present)}"
            )

    context = set(context_chains)
    for chain, flag in zip(chain_ids, design_mask):
        if flag and chain not in context:
            raise ContigError(
                f"a designed residue sits on chain {chain!r}, outside the context "
                f"chains {sorted(context)}; the loop must be carved out of the "
                f"chain that is being designed"
            )

    motif_segments: list[str] = []
    for chain in context_chains:
        kept = [
            res_ids[i]
            for i, (c, flag) in enumerate(zip(chain_ids, design_mask))
            if c == chain and not flag
        ]
        motif_segments.extend(f"{chain}{lo}-{hi}" for lo, hi in contiguous_ranges(kept))

    target_segments: list[str] = []
    for chain in target_chains:
        kept = [r for c, r in zip(chain_ids, res_ids) if c == chain]
        target_segments.extend(f"{chain}{lo}-{hi}" for lo, hi in contiguous_ranges(kept))

    designed = [i for i, flag in enumerate(design_mask) if flag]
    if not designed:
        raise ContigError("no designed residue")

    return {
        "motif_position": "/".join(motif_segments),
        "motif_segments": motif_segments,
        "target_spec": "/".join(target_segments),
        "target_segments": target_segments,
        "design_length": len(designed),
    }
