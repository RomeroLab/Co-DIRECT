
from __future__ import annotations

import gzip
from pathlib import Path

_ATOM_PREFIX = ("ATOM", "HETATM")

class Rfd3OutputError(RuntimeError):
    pass

def _atom_site_rows(cif_text: str) -> tuple[dict[str, int], list[list[str]]]:
    columns: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    for line in cif_text.splitlines():
        if line.startswith("_atom_site."):
            columns.append(line.strip().split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if line.startswith(_ATOM_PREFIX):
                fields = line.split()
                if len(fields) == len(columns):
                    rows.append(fields)
            elif rows and (line.startswith("#") or not line.strip()):
                break
    if not columns:
        raise Rfd3OutputError("no _atom_site loop in RFdiffusion3 output")
    if not rows:
        raise Rfd3OutputError("no atom rows in RFdiffusion3 output")
    return {name: i for i, name in enumerate(columns)}, rows

def designed_chain_and_residues(
    cif_text: str, diffused_index_map: dict[str, str]
) -> tuple[str, tuple[int, ...]]:

    if not diffused_index_map:
        raise Rfd3OutputError(
            "diffused_index_map is empty; the designed region cannot be "
            "identified without the model's own input-to-output mapping"
        )
    index, rows = _atom_site_rows(cif_text)
    for key in ("label_asym_id", "label_seq_id", "group_PDB"):
        if key not in index:
            raise Rfd3OutputError(f"_atom_site lacks {key!r}")

    fixed = set(diffused_index_map.values())
    per_chain: dict[str, set[int]] = {}
    for row in rows:
        if row[index["group_PDB"]] != "ATOM":
            continue
        chain = row[index["label_asym_id"]]
        raw = row[index["label_seq_id"]]
        if raw in (".", "?"):
            continue
        if f"{chain}{raw}" in fixed:
            continue
        per_chain.setdefault(chain, set()).add(int(raw))

    if not per_chain:
        raise Rfd3OutputError(
            "every polymer residue is present in diffused_index_map; the "
            "output carries no designed region"
        )
    if len(per_chain) > 1:
        raise Rfd3OutputError(
            f"designed residues span several chains: {sorted(per_chain)}"
        )
    chain, residues = next(iter(per_chain.items()))
    return chain, tuple(sorted(residues))

def rfd3_cif_to_pdb(cif_text: str) -> str:

    index, rows = _atom_site_rows(cif_text)
    needed = (
        "group_PDB", "type_symbol", "label_atom_id", "label_comp_id",
        "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
    )
    missing = [name for name in needed if name not in index]
    if missing:
        raise Rfd3OutputError(f"_atom_site lacks {missing!r}")

    out: list[str] = []
    hetatm_seq: dict[tuple[str, str], int] = {}
    for serial, row in enumerate(rows, start=1):
        group = row[index["group_PDB"]]
        chain = row[index["label_asym_id"]]
        raw_seq = row[index["label_seq_id"]]
        if raw_seq in (".", "?"):

            key = (chain, row[index["label_comp_id"]])
            raw_seq = str(hetatm_seq.setdefault(key, len(hetatm_seq) + 1))
        name = row[index["label_atom_id"]]
        atom_name = f" {name:<3s}" if len(name) < 4 else name
        out.append(
            f"{group:<6s}{serial:>5d} {atom_name}"
            f"{row[index['label_comp_id']]:>4s} {chain[:1]}"
            f"{int(raw_seq):>4d}    "
            f"{float(row[index['Cartn_x']]):>8.3f}"
            f"{float(row[index['Cartn_y']]):>8.3f}"
            f"{float(row[index['Cartn_z']]):>8.3f}"
            f"  1.00  0.00          {row[index['type_symbol']]:>2s}"
        )
    return "\n".join(out) + "\nEND\n"

def read_rfd3_cif(path: str | Path) -> str:

    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8").read()
    return path.read_text(encoding="utf-8")
