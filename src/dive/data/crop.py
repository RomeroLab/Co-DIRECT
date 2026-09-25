
from __future__ import annotations

DEFAULT_CROP_SIZE = 256

DEFAULT_TARGET_BUDGET = 96

MAX_EXTENDED_TOKENS = DEFAULT_CROP_SIZE + DEFAULT_TARGET_BUDGET

MAX_DESIGN_FRACTION = 0.5

class CropError(RuntimeError):
    pass

class RoleAwareCrop:

    def __init__(
        self,
        crop_size: int = DEFAULT_CROP_SIZE,
        max_design_fraction: float = MAX_DESIGN_FRACTION,
        target_budget: int = DEFAULT_TARGET_BUDGET,
    ) -> None:
        if crop_size < 1:
            raise CropError(f"crop size must be positive, got {crop_size}")
        if target_budget < 1:
            raise CropError(f"target budget must be positive, got {target_budget}")
        if not 0.0 < max_design_fraction <= 1.0:
            raise CropError(
                f"max_design_fraction must be in (0, 1], got {max_design_fraction}"
            )
        self.crop_size = crop_size
        self.max_design_fraction = max_design_fraction
        self.target_budget = target_budget

    def __call__(self, graph):
        import torch

        design = getattr(graph, "design_mask", None)
        if design is None:
            raise CropError(
                "graph has no design_mask; RoleMaskTransform must run before cropping"
            )

        graph = _cap_target(graph, self.target_budget)

        residues = int(design.shape[0])
        if residues <= self.crop_size:
            _assert_target_survived(graph)
            return graph

        design_count = int(design.sum())
        if design_count == 0:
            raise CropError("design region is empty before cropping")

        design_budget = max(1, int(self.crop_size * self.max_design_fraction))

        if design_count > design_budget:

            if not _generated_is_whole_chain(graph):
                if design_count > self.crop_size:
                    raise CropError(
                        f"the design region is a specific {design_count}-residue "
                        f"span, larger than the crop budget of {self.crop_size}; "
                        f"cropping it would change which region is being generated"
                    )

                keep = _fill(
                    design.clone(),
                    order=_distance_order(graph, design),
                    budget=self.crop_size - design_count,
                )
                cropped = _subset(graph, keep.nonzero(as_tuple=True)[0])
                _assert_target_survived(cropped)
                return cropped
            design = _contiguous_head(design, design_budget)
            design_count = int(design.sum())
            graph.design_mask = design

        keep = _fill(
            design.clone(),
            order=_distance_order(graph, design),
            budget=self.crop_size - design_count,
        )

        cropped = _subset(graph, keep.nonzero(as_tuple=True)[0])
        _assert_target_survived(cropped)
        return cropped

def _cap_target(graph, budget: int):

    import torch

    target_mask = getattr(graph, "target_mask", None)
    if target_mask is None or target_mask.numel() == 0:
        return graph
    per_residue = target_mask.sum(dim=-1).bool() if target_mask.dim() == 2 else target_mask.bool()
    count = int(per_residue.sum())
    if count <= budget:
        return graph

    design = graph.design_mask
    order = _distance_order(graph, design)
    keep = torch.zeros_like(per_residue)
    remaining = budget
    for index in order:
        if remaining == 0:
            break
        if per_residue[index]:
            keep[index] = True
            remaining -= 1
    graph.target_mask = target_mask & keep[:, None] if target_mask.dim() == 2 else keep
    return graph

def _fill(keep, *, order, budget: int):

    for index in order:
        if budget <= 0:
            break
        if keep[index]:
            continue
        keep[index] = True
        budget -= 1
    return keep

def _assert_target_survived(graph) -> None:

    declared = str(getattr(graph, "target", "") or "")
    if not declared:
        return
    target_mask = getattr(graph, "target_mask", None)
    if target_mask is None or not bool(target_mask.any()):
        raise CropError(
            f"the crop removed every residue of the declared target {declared!r}; "
            f"target-conditioned design without its target is a different task"
        )

def _distance_order(graph, design):

    import torch

    coords = getattr(graph, "coords", None)
    if coords is None:

        positions = torch.arange(design.shape[0])
        design_positions = positions[design]
        distance = (positions[:, None] - design_positions[None, :]).abs().min(dim=1).values
        return torch.argsort(distance).tolist()

    alpha = coords[:, 1, :]
    finite = torch.isfinite(alpha).all(dim=-1)
    design_alpha = alpha[design & finite]
    if design_alpha.numel() == 0:
        positions = torch.arange(design.shape[0])
        return torch.argsort(positions).tolist()

    distance = torch.cdist(alpha, design_alpha).min(dim=1).values
    distance = torch.where(finite, distance, torch.full_like(distance, float("inf")))
    return torch.argsort(distance).tolist()

def _subset(graph, indices):

    import torch

    residues = int(graph.design_mask.shape[0])
    for key, value in list(vars(graph).items()):
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == residues:
            setattr(graph, key, value[indices])
        elif isinstance(value, list) and len(value) == residues:
            setattr(graph, key, [value[i] for i in indices.tolist()])
    if hasattr(graph, "num_nodes"):
        graph.num_nodes = int(len(indices))
    return graph

def _generated_is_whole_chain(graph) -> bool:

    from dive.data.roles import Selector

    spec = getattr(graph, "generated", None)
    if spec is None:
        raise CropError(
            "graph carries no `generated` role, so whether its design region may "
            "be cropped cannot be decided"
        )
    entries = [part for part in str(spec).split(",") if part]
    return bool(entries) and all(Selector.parse(e).is_whole_chain for e in entries)

def _contiguous_head(design, budget: int):

    import torch

    kept = torch.zeros_like(design)
    positions = design.nonzero(as_tuple=True)[0][:budget]
    kept[positions] = True
    return kept
