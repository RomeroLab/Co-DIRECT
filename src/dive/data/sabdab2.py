
from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path

from dive.data.contracts import (
    CanonicalExample,
    ChainRecord,
    EligibilityFailure,
    Family,
    Partition,
    sequence_hash,
)

RELEASE_VERSION = "0.1.0"
RELEASE_DOI = "10.5281/zenodo.20083995"
RELEASE_MD5 = "0dbb4cc499e9eb77f14008b232f2c38c"
RELEASE_SIZE_BYTES = 876_381_859
RELEASE_URL = "https://zenodo.org/records/20083995/files/splits.tar.gz"

_READ_CHUNK = 1 << 20

class SabdabError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ReleaseFingerprint:

    md5: str
    sha256: str
    size_bytes: int

    def as_provenance(self) -> dict[str, object]:
        return {
            "version": RELEASE_VERSION,
            "doi": RELEASE_DOI,
            "url": RELEASE_URL,
            "md5": self.md5,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

@dataclass(frozen=True, slots=True)
class SabdabOuterSplit:

    official_train: frozenset[str]
    official_test: frozenset[str]

    def assert_disjoint(self) -> None:
        overlap = self.official_train & self.official_test
        if overlap:
            raise SabdabError(
                f"official ab-ag split overlaps on {len(overlap)} id(s): "
                f"{sorted(overlap)[:5]}"
            )

    def partition_of(self, example_id: str) -> Partition:

        if example_id in self.official_test:
            return Partition.TEST_OUTER
        if example_id in self.official_train:
            return Partition.TRAIN
        raise SabdabError(f"{example_id!r} is not in the official split")

def verify_release_archive(
    path: Path, expected_md5: str = RELEASE_MD5
) -> ReleaseFingerprint:

    path = Path(path)
    if not path.is_file():
        raise SabdabError(f"release archive not found: {path}")

    md5, sha256, size = _digests(path)
    if md5 != expected_md5:
        raise SabdabError(
            f"release archive md5 is {md5}, expected {expected_md5}; this is not "
            f"SAbDab2 v{RELEASE_VERSION} ({path})"
        )
    return ReleaseFingerprint(md5=md5, sha256=sha256, size_bytes=size)

def safe_extract(archive_path: Path, destination: Path) -> Path:

    archive_path, destination = Path(archive_path), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()

    with tarfile.open(archive_path, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise SabdabError(
                    f"refusing link member {member.name!r} -> {member.linkname!r}"
                )
            target = (resolved_destination / member.name).resolve()
            if (
                target != resolved_destination
                and resolved_destination not in target.parents
            ):
                raise SabdabError(
                    f"member {member.name!r} escapes the destination directory"
                )

        tar.extractall(destination, members=members, filter="data")
    return destination

def _digests(path: Path) -> tuple[str, str, int]:

    md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            md5.update(chunk)
            sha256.update(chunk)
            size += len(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size

PAIRED_FORMATS = frozenset({"FAB", "FV", "FAB+FC"})

PROTEIN_ANTIGEN_TYPES = frozenset({"PROTEIN", "PEPTIDE"})

NUMBERING_SCHEME = "IMGT"

_REQUIRED_COLUMNS = (
    "INSTANCE",
    "SABDAB_ID",
    "Hseq",
    "Lseq",
    "CDRH3",
    "type",
    "agtypes",
    "agresolvedseqs",
    "ab_ag_split",
)

_METADATA_COLUMNS = (
    "SABDAB_ID",
    "type",
    "method",
    "resolution",
    "holo",
    "agtypes",
    "agchains",
    "cdrh3_cluster",
    "ab_cluster",
    "agclusters",
    "ab_ag_cluster",
    "ab_ag_split",
)

@dataclass(frozen=True, slots=True)
class SabdabRelease:

    examples: tuple[CanonicalExample, ...]
    rejections: tuple[EligibilityFailure, ...]
    outer_split: SabdabOuterSplit

    @property
    def eligibility_counts(self) -> dict[str, int]:
        counts = {"eligible": len(self.examples)}
        for rejection in self.rejections:
            counts[rejection.code] = counts.get(rejection.code, 0) + 1
        return counts

def load_sabdab2_release(
    root: Path,
    *,
    split_file: str = "abag_split.csv",
    max_total_tokens: int | None = None,
) -> SabdabRelease:

    import pandas as pd

    table = pd.read_csv(Path(root) / split_file, low_memory=False)
    return canonicalize_sabdab2_table(
        table, split_file=split_file, max_total_tokens=max_total_tokens
    )

def canonicalize_sabdab2_table(
    table,
    *,
    split_file: str = "abag_split.csv",
    max_total_tokens: int | None = None,
) -> SabdabRelease:

    missing = [column for column in _REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise SabdabError(f"{split_file} is missing column(s) {missing}")

    rows = table.to_dict("records")
    parents = _parent_groups(rows)

    examples: list[CanonicalExample] = []
    rejections: list[EligibilityFailure] = []
    for record in rows:
        outcome = _canonicalize(record, parents, max_total_tokens)
        (examples if isinstance(outcome, CanonicalExample) else rejections).append(
            outcome
        )

    return SabdabRelease(
        examples=tuple(examples),
        rejections=tuple(rejections),
        outer_split=_outer_split(rows),
    )

def _outer_split(rows: list[dict]) -> SabdabOuterSplit:

    train, test = set(), set()
    for record in rows:
        instance = str(record["INSTANCE"])
        assignment = str(record["ab_ag_split"]).strip().lower()
        if assignment == "test":
            test.add(instance)
        elif assignment == "train":
            train.add(instance)
        else:
            raise SabdabError(f"unknown ab_ag_split value {record['ab_ag_split']!r}")
    split = SabdabOuterSplit(
        official_train=frozenset(train), official_test=frozenset(test)
    )
    split.assert_disjoint()
    return split

def _parent_groups(rows: list[dict]) -> dict[str, str]:

    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    keys: dict[str, tuple[tuple[str, str], ...]] = {}
    for record in rows:
        instance = str(record["INSTANCE"])
        node_keys = [("sabdab_id", str(record["SABDAB_ID"]))]
        pair = _sequence_pair_key(record)
        if pair is not None:
            node_keys.append(("vh_vl", pair))
        for other in node_keys[1:]:
            union(node_keys[0], other)
        keys[instance] = tuple(node_keys)

    members: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for node_keys in keys.values():
        members.setdefault(find(node_keys[0]), set()).update(node_keys)

    identifiers = {
        root: "ab:"
        + hashlib.sha256(
            "|".join(f"{kind}={value}" for kind, value in sorted(component)).encode()
        ).hexdigest()[:32]
        for root, component in members.items()
    }
    return {
        instance: identifiers[find(node_keys[0])]
        for instance, node_keys in keys.items()
    }

def _sequence_pair_key(record: dict) -> str | None:

    heavy, light = _text(record.get("Hseq")), _text(record.get("Lseq"))
    if not heavy or not light:
        return None
    return f"{sequence_hash(heavy)}:{sequence_hash(light)}"

def _canonicalize(
    record: dict, parents: dict[str, str], max_total_tokens: int | None
) -> CanonicalExample | EligibilityFailure:

    instance = str(record["INSTANCE"])

    antibody_format = _text(record.get("type"))
    if antibody_format not in PAIRED_FORMATS:
        return EligibilityFailure(
            "unsupported_antibody_format",
            f"type {antibody_format!r} is not paired-chain Fv-like",
            instance,
        )

    heavy, light = _text(record.get("Hseq")), _text(record.get("Lseq"))
    if not heavy or not light:
        return EligibilityFailure(
            "missing_sequence", "heavy or light chain sequence absent", instance
        )

    antigen_types = _text(record.get("agtypes"))
    antigen_sequences = [s for s in _split_multi(record.get("agresolvedseqs")) if s]
    accepted = [
        t for t in _split_multi(antigen_types) if t.upper() in PROTEIN_ANTIGEN_TYPES
    ]
    if not accepted or not antigen_sequences:
        return EligibilityFailure(
            "missing_antigen",
            f"agtypes={antigen_types!r} with {len(antigen_sequences)} resolved sequence(s)",
            instance,
        )

    cdr_h3 = _text(record.get("CDRH3"))
    if not cdr_h3:
        return EligibilityFailure("unresolved_h3", "CDRH3 is absent", instance)

    total_tokens = len(heavy) + len(light) + sum(len(s) for s in antigen_sequences)
    if max_total_tokens is not None and total_tokens > max_total_tokens:
        return EligibilityFailure(
            "token_overflow",
            f"{total_tokens} tokens exceeds {max_total_tokens}",
            instance,
        )

    chains = [
        ChainRecord(
            chain_id=_text(record.get("Hchain")) or "H", role="heavy", sequence=heavy
        ),
        ChainRecord(
            chain_id=_text(record.get("Lchain")) or "L", role="light", sequence=light
        ),
    ]
    antigen_chain_ids = _split_multi(record.get("agchains")) or []
    for index, sequence in enumerate(antigen_sequences):
        chain_id = (
            antigen_chain_ids[index] if index < len(antigen_chain_ids) else f"ag{index}"
        )
        chains.append(ChainRecord(chain_id=chain_id, role="antigen", sequence=sequence))

    metadata = {
        column: str(record[column])
        for column in _METADATA_COLUMNS
        if column in record and record[column] is not None and _text(record[column])
    }
    metadata["cdr_h3_sequence"] = cdr_h3
    metadata["cdr_h3_numbering_scheme"] = NUMBERING_SCHEME
    metadata["cdr_h3_length"] = str(len(cdr_h3))
    metadata["total_tokens"] = str(total_tokens)

    return CanonicalExample(
        example_id=instance,
        parent_id=parents[instance],
        family=Family.ANTIBODY,
        source_release=f"sabdab2-v{RELEASE_VERSION}",
        pdb_id=_text(record.get("PDB_ID")) or None,
        assembly_id=None,
        deposition_date=_as_date(record.get("PDBdepo")),
        chains=tuple(chains),
        ligands=(),
        metadata=metadata,
    )

def _text(value: object) -> str:

    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text

def _split_multi(value: object) -> list[str]:

    text = _text(value)
    return [part.strip() for part in text.split("/") if part.strip()] if text else []

def _as_date(value: object):
    from datetime import date, datetime

    text = _text(value)
    if not text:
        return None
    if isinstance(value, (datetime,)):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None
