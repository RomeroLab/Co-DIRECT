
from __future__ import annotations

import hashlib

MIN_FRACTION = 0.05
MAX_FRACTION = 0.30
MIN_SEGMENTS = 1
MAX_SEGMENTS = 4
MIN_SEGMENT_RESIDUES = 1
MAX_SEGMENT_RESIDUES = 8

class MotifError(RuntimeError):
    pass

class SampleMotifWithinDesign:

    def __init__(
        self,
        min_fraction: float = MIN_FRACTION,
        max_fraction: float = MAX_FRACTION,
        min_segments: int = MIN_SEGMENTS,
        max_segments: int = MAX_SEGMENTS,
        min_segment_residues: int = MIN_SEGMENT_RESIDUES,
        max_segment_residues: int = MAX_SEGMENT_RESIDUES,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < min_fraction <= max_fraction < 1.0:
            raise MotifError(
                f"fractions must satisfy 0 < min <= max < 1, got "
                f"{min_fraction} and {max_fraction}"
            )
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.min_segment_residues = min_segment_residues
        self.max_segment_residues = max_segment_residues
        self.seed = seed

    def __call__(self, graph):
        import random

        import torch

        design = getattr(graph, "design_mask", None)
        if design is None:
            raise MotifError(
                "graph has no design_mask; RoleMaskTransform must run before "
                "motif sampling"
            )

        positions = design.nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise MotifError("the design region is empty, so no motif can be carved")

        if self.seed is None:
            rng = random.Random()
        else:
            example_id = getattr(graph, "id", None)
            if not isinstance(example_id, str) or not example_id:
                raise MotifError(
                    "deterministic motif sampling requires a nonempty graph.id"
                )
            material = f"dive-ame-motif-v1\0{self.seed}\0{example_id}".encode()
            derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            rng = random.Random(derived_seed)
        count = positions.numel()
        wanted = int(round(count * rng.uniform(self.min_fraction, self.max_fraction)))

        wanted = max(1, min(wanted, count - 1))

        segments = rng.randint(self.min_segments, self.max_segments)
        chosen: set[int] = set()
        for _ in range(segments * 8):
            if len(chosen) >= wanted:
                break
            length = rng.randint(self.min_segment_residues, self.max_segment_residues)
            length = min(length, wanted - len(chosen))
            start = rng.randrange(0, max(1, count - length + 1))
            chosen.update(int(positions[start + offset]) for offset in range(length))
        if not chosen:
            raise MotifError("sampling produced no motif residue")

        motif = torch.zeros_like(design)
        motif[sorted(chosen)] = True

        graph.design_mask = design & ~motif
        if not bool(graph.design_mask.any()):
            raise MotifError("the sampled motif consumed the whole design region")

        _add_to_motif(graph, motif)
        return graph

def _add_to_motif(graph, motif) -> None:

    existing = getattr(graph, "motif_mask", None)
    if existing is None:
        raise MotifError("graph has no motif_mask to extend")

    per_atom = motif.unsqueeze(-1).expand_as(existing)
    coord_mask = getattr(graph, "coord_mask", None)
    if coord_mask is not None:

        per_atom = per_atom & coord_mask.bool()
    graph.motif_mask = existing | per_atom

class UseMotifAsTarget:

    def __call__(self, graph):
        motif = getattr(graph, "motif_mask", None)
        if motif is None:
            raise MotifError(
                "graph has no motif_mask; motif sampling must run before this"
            )
        if not bool(motif.any()):
            raise MotifError(
                "the motif is empty, so there is nothing to condition on and the "
                "task would be unconditioned generation"
            )

        design = getattr(graph, "design_mask", None)
        if design is not None and bool((motif[:, 0] & design).any()):
            raise MotifError(
                "the motif overlaps the design region, which would condition the "
                "model on its own answer"
            )

        graph.target_mask = motif.clone()
        return graph
