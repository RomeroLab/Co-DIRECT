
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from dive.data.contracts import CanonicalExample, EligibilityFailure, Family

@dataclass(frozen=True, slots=True)
class FamilyCorpus:

    family: Family
    examples: tuple[CanonicalExample, ...]
    rejections: tuple[EligibilityFailure, ...] = field(default=())
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.examples)

    def eligibility_counts(self) -> dict[str, int]:
        counts = {"eligible": len(self.examples)}
        for rejection in self.rejections:
            counts[rejection.code] = counts.get(rejection.code, 0) + 1
        return counts

def resolve_path(template: str, bulk: Path) -> Path:

    return Path(str(template).replace("${roots.bulk}", str(bulk)))

def load_families(config: dict) -> dict[Family, FamilyCorpus]:
    return {
        Family.BINDER: load_binder(config),
        Family.AME: load_ame(config),
        Family.ANTIBODY: load_antibody(config),
    }

def load_binder(config: dict) -> FamilyCorpus:
    import pandas as pd

    path = Path(config["sources"]["binder"]["path"])
    if not path.exists():
        return FamilyCorpus(Family.BINDER, (), (), f"metadata table absent: {path}")

    table = pd.read_csv(path, low_memory=False)
    return canonicalize_binder_table(table)

def canonicalize_binder_table(table) -> FamilyCorpus:

    from dive.data.sources import canonicalize_binder

    examples, rejections = [], []
    for record in table.to_dict("records"):
        chains = ast.literal_eval(str(record["chain"]))
        if len(chains) < 2:
            rejections.append(
                EligibilityFailure(
                    "monomer", "fewer than two chains", str(record["pdb"])
                )
            )
            continue

        outcome = canonicalize_binder(
            dict(
                record,
                binder_chain=str(chains[0]),
                target_chains=",".join(str(c) for c in chains[1:]),
            )
        )
        (examples if isinstance(outcome, CanonicalExample) else rejections).append(
            outcome
        )
    return FamilyCorpus(Family.BINDER, tuple(examples), tuple(rejections))

def load_ame(config: dict) -> FamilyCorpus:
    import pandas as pd

    bulk = Path(config["roots"]["bulk"])
    source = config["sources"]["ame"]
    table_path = Path(source["path"])
    resolved_path = bulk / "sources" / "plinder-2024-06" / "resolved_systems.parquet"

    if not table_path.exists():
        return FamilyCorpus(Family.AME, (), (), f"metadata table absent: {table_path}")
    if not resolved_path.exists():
        return FamilyCorpus(
            Family.AME,
            (),
            (),
            "PLINDER systems are not resolved; run scripts/data/fetch_plinder_systems.py",
        )

    resolved_table = pd.read_parquet(resolved_path)
    table = pd.read_csv(table_path, low_memory=False)
    return canonicalize_ame_tables(table, resolved_table)

def canonicalize_ame_tables(table, resolved_table) -> FamilyCorpus:

    from dive.data.sources import canonicalize_ame

    resolved = {str(row["system_id"]): row for row in resolved_table.to_dict("records")}

    examples, rejections = [], []
    for record in table.to_dict("records"):
        system_id = str(record["example_id"])
        extra = resolved.get(system_id)
        if extra is None:
            rejections.append(
                EligibilityFailure(
                    "unresolved_system", "no sequence/SMILES resolved", system_id
                )
            )
            continue
        outcome = canonicalize_ame(
            dict(
                record,
                scaffold_chains=[tuple(pair) for pair in extra["scaffold_chains"]],
                ligand_smiles=extra["ligand_smiles"],
            )
        )
        (examples if isinstance(outcome, CanonicalExample) else rejections).append(
            outcome
        )
    return FamilyCorpus(Family.AME, tuple(examples), tuple(rejections))

def load_antibody(config: dict) -> FamilyCorpus:
    from dive.data.sabdab2 import load_sabdab2_release

    bulk = Path(config["roots"]["bulk"])
    source = config["sources"]["antibody"]
    extracted = resolve_path(source["extracted"], bulk)
    if not extracted.exists():
        return FamilyCorpus(
            Family.ANTIBODY, (), (), f"release not extracted: {extracted}"
        )

    release = load_sabdab2_release(
        extracted,
        split_file=source["split_file"],
        max_total_tokens=config["eligibility"]["antibody"]["max_total_tokens"],
    )
    return FamilyCorpus(Family.ANTIBODY, release.examples, release.rejections)

def antibody_outer_split(config: dict):

    from dive.data.sabdab2 import load_sabdab2_release

    bulk = Path(config["roots"]["bulk"])
    source = config["sources"]["antibody"]
    extracted = resolve_path(source["extracted"], bulk)
    if not extracted.exists():
        return None
    return load_sabdab2_release(
        extracted,
        split_file=source["split_file"],
        max_total_tokens=config["eligibility"]["antibody"]["max_total_tokens"],
    ).outer_split
