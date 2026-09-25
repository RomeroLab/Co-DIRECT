
from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from datetime import date, datetime

from dive.data.contracts import (
    CanonicalExample,
    ChainRecord,
    ContractError,
    EligibilityFailure,
    Family,
    LigandRecord,
    parent_hash,
)

BINDER_SOURCE_RELEASE = "proteina-complexa/assets/data/pdb_multimer.csv"
AME_SOURCE_RELEASE = "proteina-complexa/assets/data/plinder_valid_dataset.csv"

_TIMESTAMP = re.compile(r"Timestamp\('([^']+)'\)")

_BINDER_LIST_COLUMNS = ("chain", "sequence", "molecule_type")

def canonicalize_binder(row: Mapping[str, str]) -> CanonicalExample | EligibilityFailure:

    source_id = str(row.get("pdb", "<unknown>"))

    try:
        columns = {name: _literal_list(row[name]) for name in _BINDER_LIST_COLUMNS}
    except (KeyError, ValueError, SyntaxError) as error:
        return EligibilityFailure("unparseable_row", str(error), source_id)

    widths = {name: len(value) for name, value in columns.items()}
    if len(set(widths.values())) != 1:
        return EligibilityFailure(
            "ragged_row", f"parallel columns disagree in length: {widths}", source_id
        )

    by_chain = {
        str(chain_id): {"sequence": str(seq), "molecule_type": str(kind)}
        for chain_id, seq, kind in zip(
            columns["chain"], columns["sequence"], columns["molecule_type"], strict=True
        )
    }

    binder_chain = str(row.get("binder_chain", "")).strip()
    target_chains = [c.strip() for c in str(row.get("target_chains", "")).split(",") if c.strip()]

    if binder_chain not in by_chain:
        return EligibilityFailure(
            "unknown_binder_chain", f"{binder_chain!r} not in {sorted(by_chain)}", source_id
        )
    for target in target_chains:
        if target not in by_chain:
            return EligibilityFailure(
                "unknown_target_chain", f"{target!r} not in {sorted(by_chain)}", source_id
            )
    if not target_chains:
        return EligibilityFailure("unknown_target_chain", "no target chain given", source_id)
    if binder_chain in target_chains:
        return EligibilityFailure(
            "role_overlap", f"chain {binder_chain!r} is both binder and target", source_id
        )

    roles = [(binder_chain, "binder"), *((t, "target") for t in target_chains)]
    for chain_id, role in roles:
        if by_chain[chain_id]["molecule_type"] != "protein":
            return EligibilityFailure(
                "non_protein_chain",
                f"{role} chain {chain_id!r} is {by_chain[chain_id]['molecule_type']!r}",
                source_id,
            )
    for chain_id, role in roles:
        if not by_chain[chain_id]["sequence"].strip():
            return EligibilityFailure(
                f"missing_{role}_sequence", f"chain {chain_id!r} has no sequence", source_id
            )

    chains = tuple(
        ChainRecord(
            chain_id=chain_id,
            role=role,
            sequence=by_chain[chain_id]["sequence"],
            molecule_type="protein",
        )
        for chain_id, role in roles
    )

    metadata = _string_metadata(
        row,

        keys=("n_chains", "total_length", "experiment_type", "resolution", "source", "name"),
    )
    metadata["binder_chain"] = binder_chain
    metadata["target_chains"] = ",".join(target_chains)

    return CanonicalExample(
        example_id=f"binder:{source_id}:{binder_chain}|{'+'.join(target_chains)}",
        parent_id=f"binder:{parent_hash(chains)[:32]}",
        family=Family.BINDER,
        source_release=BINDER_SOURCE_RELEASE,
        pdb_id=source_id,
        assembly_id=None,
        deposition_date=_first_timestamp(row.get("deposition_date")),
        chains=chains,
        ligands=(),
        metadata=metadata,
    )

def canonicalize_ame(row: Mapping[str, object]) -> CanonicalExample | EligibilityFailure:

    complex_name = str(row.get("complex_name", "") or "")
    source_id = complex_name or str(row.get("example_id", "<unknown>"))

    parts = complex_name.split("__")
    if len(parts) != 4:
        return EligibilityFailure(
            "malformed_complex_name",
            f"expected pdb__assembly__receptor__ligand, got {complex_name!r}",
            source_id,
        )
    pdb_id, assembly_id, receptor_chain, ligand_chain = parts

    scaffold_chains = _scaffold_chains(row, receptor_chain)
    if not scaffold_chains:
        return EligibilityFailure(
            "missing_scaffold_sequence",
            f"no sequence resolved from {row.get('path')!r}",
            source_id,
        )

    smiles = row.get("ligand_smiles")
    if not smiles or not str(smiles).strip():
        return EligibilityFailure(
            "missing_ligand_identity",
            f"no SMILES resolved from {row.get('ligand_paths')!r}",
            source_id,
        )

    try:
        ligand = _ligand_record(ligand_chain, str(smiles))
    except ContractError as error:
        return EligibilityFailure("unparseable_ligand", str(error), source_id)

    chains = tuple(
        ChainRecord(chain_id=chain_id, role="scaffold", sequence=sequence)
        for chain_id, sequence in scaffold_chains
    )

    return CanonicalExample(
        example_id=f"ame:{complex_name}",
        parent_id=f"ame:{parent_hash(chains)[:32]}",
        family=Family.AME,
        source_release=AME_SOURCE_RELEASE,
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        deposition_date=None,
        chains=chains,
        ligands=(ligand,),
        metadata=_string_metadata(
            row, keys=("num_residues_protein", "num_heavy_atoms_ligand", "path", "protein_path")
        ),
    )

def _literal_list(value: object) -> list:

    if isinstance(value, list):
        return value
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected a list, got {type(parsed).__name__}")
    return parsed

def _first_timestamp(value: object) -> date | None:

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    match = _TIMESTAMP.search(str(value))
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group(1)).date()
    except ValueError:
        return None

def _ligand_record(ligand_id: str, smiles: str) -> LigandRecord:

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ContractError(f"RDKit could not parse SMILES {smiles!r}")

    return LigandRecord(
        ligand_id=ligand_id,
        smiles=Chem.MolToSmiles(mol),
        inchikey=Chem.MolToInchiKey(mol) or None,
        murcko_scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
    )

def _scaffold_chains(row: Mapping[str, object], receptor_field: str) -> list[tuple[str, str]]:

    chains = row.get("scaffold_chains")
    if chains is not None and len(chains) > 0:
        resolved = []
        for entry in chains:
            chain_id, sequence = entry[0], entry[1]
            if str(sequence).strip():
                resolved.append((str(chain_id), str(sequence).strip()))
        return resolved

    sequence = row.get("scaffold_sequence")
    if sequence and str(sequence).strip():
        return [(receptor_field, str(sequence).strip())]
    return []

def _string_metadata(row: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, str]:

    return {key: str(row[key]) for key in keys if key in row and row[key] is not None}
