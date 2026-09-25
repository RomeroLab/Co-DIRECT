
from __future__ import annotations

from dive.data.roles import Selector

ATOM37 = 37

class RoleTransformError(RuntimeError):
    pass

class RoleMaskTransform:

    def __call__(self, graph):
        import torch

        chain_ids = _chain_ids(graph)
        residues = len(chain_ids)

        design = _select(chain_ids, _text(graph, "generated"), residues)
        context = _select(chain_ids, _text(graph, "context"), residues)
        target = _select(chain_ids, _text(graph, "target"), residues, allow_empty=True)

        if not bool(design.any()):
            raise RoleTransformError(
                f"the design selection {_text(graph, 'generated')!r} matches no "
                f"residue; an all-false design mask trains to predict nothing"
            )
        if bool((design & target).any()):
            raise RoleTransformError(
                "the design region overlaps the conditioning target, which would "
                "condition the model on its own answer"
            )

        coord_mask = getattr(graph, "coord_mask", None)

        graph.design_mask = design

        graph.target_mask = _per_atom(target, residues, coord_mask)

        graph.motif_mask = _per_atom(context & ~design, residues, coord_mask)
        return graph

def _per_atom(residue_mask, residues: int, coord_mask):

    expanded = residue_mask.unsqueeze(-1).expand(residues, ATOM37).clone()
    if coord_mask is not None:

        expanded = expanded & coord_mask.bool()
    return expanded

def _select(chain_ids, spec: str, residues: int, allow_empty: bool = False):

    import torch

    mask = torch.zeros(residues, dtype=torch.bool)
    entries = [part for part in (spec or "").split(",") if part]
    if not entries:
        if allow_empty:
            return mask
        raise RoleTransformError("empty selector where one was required")

    for entry in entries:
        selector = Selector.parse(entry)
        in_chain = torch.tensor(
            [chain == selector.chain for chain in chain_ids], dtype=torch.bool
        )
        if not bool(in_chain.any()):
            raise RoleTransformError(
                f"selector {entry!r} names chain {selector.chain!r}, which is not in "
                f"the graph (chains present: {sorted(set(chain_ids))})"
            )
        if selector.is_whole_chain:
            mask |= in_chain
            continue

        positions = in_chain.nonzero(as_tuple=True)[0]
        if selector.end >= len(positions):
            raise RoleTransformError(
                f"selector {entry!r} runs past chain {selector.chain!r}, which has "
                f"{len(positions)} residues in the graph"
            )
        mask[positions[selector.start : selector.end + 1]] = True
    return mask

def _chain_ids(graph) -> list[str]:
    chain_ids = getattr(graph, "chain_id", None)
    if chain_ids is None:
        raise RoleTransformError("graph carries no per-residue chain_id")
    return [str(c) for c in chain_ids]

def _text(graph, name: str) -> str:
    value = getattr(graph, name, None)
    if value is None:
        raise RoleTransformError(
            f"graph carries no {name!r}; RoleAwareStructureDataset attaches the "
            f"roles and must run first"
        )
    return str(value)

MAX_SUPPORTED_RESIDUE_TYPE = 19

class RequireSupportedResidueTypes:

    def __init__(self, max_index: int = MAX_SUPPORTED_RESIDUE_TYPE) -> None:
        self.max_index = max_index

    def __call__(self, graph):
        residue_type = getattr(graph, "residue_type", None)
        if residue_type is None:
            raise RoleTransformError("graph carries no residue_type to check")
        if residue_type.numel() == 0:
            raise RoleTransformError("graph has no residues")

        highest = int(residue_type.max())
        if highest > self.max_index:
            raise RoleTransformError(
                f"residue_type {highest} exceeds the model's vocabulary "
                f"(0-{self.max_index}); this structure carries a modified or "
                f"non-standard residue the model has no token for"
            )
        if int(residue_type.min()) < 0:
            raise RoleTransformError(f"negative residue_type {int(residue_type.min())}")
        return graph
