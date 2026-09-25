
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from dive.data.contracts import CanonicalExample, Family, LigandRecord

class SimilarityError(ValueError):
    pass

@dataclass(frozen=True, order=True, slots=True)
class SimilarityEdge:

    left: str
    right: str
    kind: str
    value: float
    threshold: float

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise SimilarityError(f"{self.left!r} cannot conflict with itself")
        if self.left > self.right:

            low, high = self.right, self.left
            object.__setattr__(self, "left", low)
            object.__setattr__(self, "right", high)

@dataclass(frozen=True, slots=True)
class MmseqsHit:

    identity: float
    aligned: int
    query_length: int
    target_length: int

@dataclass(frozen=True, slots=True)
class ExactGroup:

    parent_id: str
    family: Family
    example_ids: tuple[str, ...]

def normalize_identity(value: float) -> float:

    if value < 0 or value > 100:
        raise SimilarityError(f"identity {value} is outside [0, 100]")
    fraction = value / 100.0 if value > 1.0 else float(value)
    if not 0.0 <= fraction <= 1.0:
        raise SimilarityError(f"identity {value} normalized outside [0, 1]")
    return fraction

def normalize_identity_column(values: Sequence[float]) -> list[float]:

    if not values:
        return []
    if min(values) < 0 or max(values) > 100:
        raise SimilarityError("identity column contains values outside [0, 100]")
    scale = 100.0 if max(values) > 1.0 else 1.0
    return [float(value) / scale for value in values]

def shorter_coverage(hit: MmseqsHit) -> float:

    shorter = min(hit.query_length, hit.target_length)
    if shorter <= 0 or hit.aligned < 0:
        raise SimilarityError(f"non-positive length in {hit}")
    if hit.aligned > hit.query_length + hit.target_length:

        raise SimilarityError(
            f"{hit.aligned} alignment columns exceed the combined length "
            f"{hit.query_length + hit.target_length}: {hit}"
        )
    return min(1.0, hit.aligned / shorter)

def protein_edge(
    hit: MmseqsHit, *, min_identity: float = 0.30, min_shorter_coverage: float = 0.80
) -> bool:

    return hit.identity >= min_identity and shorter_coverage(hit) >= min_shorter_coverage

def paired_antibody_conflict(
    vh: MmseqsHit,
    vl: MmseqsHit,
    *,
    min_identity: float = 0.70,
    min_shorter_coverage: float = 0.80,
) -> bool:

    return all(
        hit.identity >= min_identity and shorter_coverage(hit) >= min_shorter_coverage
        for hit in (vh, vl)
    )

def cdr_h3_conflict(
    left: str,
    right: str,
    *,
    min_identity: float = 0.80,
    min_length_ratio: float = 0.80,
) -> bool:

    left, right = left.strip().upper(), right.strip().upper()
    if not left or not right:
        raise SimilarityError("cannot compare an empty CDR-H3")

    shorter, longer = sorted((len(left), len(right)))
    if shorter / longer < min_length_ratio:
        return False
    return _aligned_identity(left, right) >= min_identity

@lru_cache(maxsize=1)
def _h3_aligner():

    from Bio import Align

    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"

    aligner.end_insertion_score = 0.0
    aligner.end_deletion_score = 0.0
    return aligner

def _aligned_identity(left: str, right: str) -> float:

    alignment = _h3_aligner().align(left, right)[0]
    top, bottom = alignment[0], alignment[1]
    identities = sum(1 for a, b in zip(top, bottom, strict=True) if a == b and a != "-")
    return identities / min(len(left), len(right))

def ligand_conflict(
    left: LigandRecord, right: LigandRecord, *, min_tanimoto: float = 0.70
) -> bool:

    if left.murcko_scaffold and left.murcko_scaffold == right.murcko_scaffold:
        return True
    return ecfp4_tanimoto(left.smiles, right.smiles) >= min_tanimoto

def ecfp4_tanimoto(left_smiles: str, right_smiles: str) -> float:

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    left, right = Chem.MolFromSmiles(left_smiles), Chem.MolFromSmiles(right_smiles)
    if left is None or right is None:
        raise SimilarityError(f"unparseable SMILES: {left_smiles!r} / {right_smiles!r}")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return DataStructs.TanimotoSimilarity(
        generator.GetFingerprint(left), generator.GetFingerprint(right)
    )

def direct_neighbors(node: str, edges: Iterable[SimilarityEdge]) -> frozenset[str]:

    neighbors = set()
    for edge in edges:
        if edge.left == node:
            neighbors.add(edge.right)
        elif edge.right == node:
            neighbors.add(edge.left)
    return frozenset(neighbors)

class NeighborIndex:

    __slots__ = ("_adjacency",)

    def __init__(self, edges: Iterable[SimilarityEdge]) -> None:
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.left, set()).add(edge.right)
            adjacency.setdefault(edge.right, set()).add(edge.left)
        self._adjacency = adjacency

    def neighbors(self, node: str) -> frozenset[str]:
        return frozenset(self._adjacency.get(node, ()))

    def neighbors_of_any(self, nodes: Iterable[str]) -> frozenset[str]:

        out: set[str] = set()
        for node in nodes:
            out |= self._adjacency.get(node, set())
        return frozenset(out)

    def node_count(self) -> int:
        return len(self._adjacency)

    def max_degree(self) -> int:
        return max((len(v) for v in self._adjacency.values()), default=0)

    def degree(self, node: str) -> int:
        return len(self._adjacency.get(node, ()))

def build_exact_groups(examples: Iterable[CanonicalExample]) -> tuple[ExactGroup, ...]:

    grouped: dict[str, list[CanonicalExample]] = {}
    for example in examples:
        grouped.setdefault(example.parent_id, []).append(example)

    groups = []
    for parent_id in sorted(grouped):
        members = grouped[parent_id]
        families = {member.family for member in members}
        if len(families) != 1:
            raise SimilarityError(
                f"parent {parent_id!r} spans families {sorted(families)}; parents are "
                f"family-homogeneous by construction"
            )
        groups.append(
            ExactGroup(
                parent_id=parent_id,
                family=members[0].family,
                example_ids=tuple(sorted(member.example_id for member in members)),
            )
        )
    return tuple(groups)

def morgan_fingerprints(smiles: Sequence[str]) -> list:

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for item in smiles:
        mol = Chem.MolFromSmiles(item)
        if mol is None:
            raise SimilarityError(f"unparseable SMILES: {item!r}")
        fingerprints.append(generator.GetFingerprint(mol))
    return fingerprints

def bulk_tanimoto(query, others) -> list[float]:

    from rdkit import DataStructs

    return list(DataStructs.BulkTanimotoSimilarity(query, list(others)))
