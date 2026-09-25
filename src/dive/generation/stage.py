
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BACKBONE = ("N", "CA", "C", "O")

@dataclass(frozen=True)
class Staged:
    path: Path
    generated: str
    context: str
    target: str
    design_length: int
    design_chain: str
    hotspot_labels: tuple[str, ...]
    notes: dict

def _read_pdb(path: Path):
    from biotite.structure.io.pdb import PDBFile

    return PDBFile.read(str(path)).get_structure(model=1)

def _read_any(path: Path):
    if str(path).endswith((".cif", ".cif.gz", ".bcif")):
        from biotite.structure.io.pdbx import CIFFile, get_structure

        return get_structure(CIFFile.read(str(path)), model=1, use_author_fields=True)
    return _read_pdb(path)

def _protein_only(atoms):
    import biotite.structure as struc

    return atoms[struc.filter_amino_acids(atoms) & (atoms.element != "H")]

def _serpentine(length: int, centre: np.ndarray, spacing: float = 3.8) -> np.ndarray:

    side = max(2, int(math.ceil(length ** (1.0 / 3.0))))
    points = []
    for z in range(side):
        for y in range(side):
            ys = y if z % 2 == 0 else side - 1 - y
            for x in range(side):
                xs = x if (ys + z) % 2 == 0 else side - 1 - x
                points.append((xs, ys, z))
                if len(points) >= length:
                    break
            if len(points) >= length:
                break
        if len(points) >= length:
            break
    grid = np.asarray(points[:length], dtype=float) * spacing
    grid -= grid.mean(axis=0)
    return grid + centre

def _placeholder_chain(length: int, centre: np.ndarray, chain_id: str, resname: str = "GLY"):

    import biotite.structure as struc

    ca = _serpentine(length, centre)
    n_atoms = length * len(BACKBONE)
    array = struc.AtomArray(n_atoms)
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    chain_ids, res_ids, res_names, atom_names, elements = [], [], [], [], []
    for i in range(length):
        if i + 1 < length:
            direction = ca[i + 1] - ca[i]
        else:
            direction = ca[i] - ca[i - 1]
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        ortho = np.cross(direction, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        ortho = ortho / np.linalg.norm(ortho)
        base = i * len(BACKBONE)
        coords[base + 0] = ca[i] - 1.20 * direction
        coords[base + 1] = ca[i]
        coords[base + 2] = ca[i] + 1.20 * direction
        coords[base + 3] = ca[i] + 1.20 * direction + 1.10 * ortho
        for name, element in zip(BACKBONE, ("N", "C", "C", "O")):
            chain_ids.append(chain_id)
            res_ids.append(i + 1)
            res_names.append(resname)
            atom_names.append(name)
            elements.append(element)
    array.coord = coords
    array.chain_id = np.asarray(chain_ids)
    array.res_id = np.asarray(res_ids, dtype=int)
    array.res_name = np.asarray(res_names)
    array.atom_name = np.asarray(atom_names)
    array.element = np.asarray(elements)
    array.hetero = np.zeros(n_atoms, dtype=bool)
    return array

def _free_chain_id(used) -> str:
    for letter in "ZYXWVUTSRQPONMLKJIHGFEDCBA":
        if letter not in used:
            return letter
    raise RuntimeError("no free chain id")

def _write_pdb(atoms, path: Path) -> Path:
    from biotite.structure.io.pdb import PDBFile

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = PDBFile()
    handle.set_structure(atoms)
    handle.write(str(path))
    return path

def stage_binder(task: dict, out_dir: Path, *, design_length: int) -> Staged:

    import biotite.structure as struc

    source = Path(task["path"])
    atoms = _protein_only(_read_any(source))
    wanted = set(task["target_chains"])
    atoms = atoms[np.isin(atoms.chain_id, list(wanted))]
    if len(atoms) == 0:
        raise RuntimeError(f"no target atoms for {task['name']}")

    ca = atoms[atoms.atom_name == "CA"]
    centre = ca.coord.mean(axis=0)

    chain_id = _free_chain_id(set(np.unique(atoms.chain_id)))
    design = _placeholder_chain(int(design_length), centre, chain_id)
    combined = struc.concatenate([atoms, design])
    path = _write_pdb(combined, out_dir / f"{task['name']}.staged.pdb")
    return Staged(
        path=path,
        generated=chain_id,
        context=chain_id,
        target=",".join(task["target_chains"]),
        design_length=int(design_length),
        design_chain=chain_id,
        hotspot_labels=(),
        notes={
            "binder_length_range": task["binder_length"],
            "design_length_source": "benchv2/generation/binder_lengths.json",
            "hotspots_used": False,
            "placeholder_centre": "target CA centroid",
            "target_chains": task["target_chains"],
            "target_residues": int(len(ca)),
            "source_pdb": str(source),
        },
    )

def stage_ame(task: dict, out_dir: Path, design_length: int | None = None) -> Staged:

    import biotite.structure as struc

    source = Path(task["input_pdb"])
    atoms = _protein_only(_read_any(source))
    keys = {(m["chain"], int(m["resnum"])) for m in task["motif"].values()}
    sel = np.asarray(
        [(c, int(r)) in keys for c, r in zip(atoms.chain_id, atoms.res_id)], dtype=bool
    )
    motif_atoms = atoms[sel]
    if len(motif_atoms) == 0:
        raise RuntimeError(f"no motif atoms for {task['task_id']}")
    resolved = len(np.unique(motif_atoms.res_id))

    design_length = int(design_length) if design_length is not None else int(task["design_length"])
    centroid = motif_atoms.coord.mean(axis=0)
    motif_chain = str(np.unique(motif_atoms.chain_id)[0])
    chain_id = _free_chain_id({motif_chain})
    design = _placeholder_chain(design_length, centroid, chain_id)
    combined = struc.concatenate([motif_atoms, design])
    path = _write_pdb(combined, out_dir / f"{task['task_id']}.staged.pdb")
    return Staged(
        path=path,
        generated=chain_id,
        context=chain_id,
        target=motif_chain,
        design_length=design_length,
        design_chain=chain_id,
        hotspot_labels=(),
        notes={
            "motif_residues_declared": len(task["motif"]),
            "motif_residues_resolved": resolved,
            "motif_chain": motif_chain,
            "ligand_dropped": task.get("ligands", []),
            "ligand_note": "complexa.ckpt v1 has no ligand pathway (enable_ligand off)",
            "source_pdb": str(source),
        },
    )

def _loaded_chain_sequences(path: Path):

    import biotite.structure as struc
    from biotite.structure.info import one_letter_code
    from proteinfoundation.datasets.structure_data import load_structure

    atoms = load_structure(str(path))
    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.element != "H")]
    out = {}
    for chain in np.unique(atoms.chain_id):
        sub = atoms[atoms.chain_id == chain]
        _, names = struc.get_residues(sub)
        out[str(chain)] = "".join((one_letter_code(n) or "X") for n in names)
    return out

def _containment(loaded: str, declared: str) -> float:

    from difflib import SequenceMatcher

    if not loaded or not declared:
        return 0.0
    matcher = SequenceMatcher(None, loaded, declared, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / min(len(loaded), len(declared))

def _best_chain(sequences, wanted, taken):
    best, score = None, 0.0
    for chain, sequence in sequences.items():
        if chain in taken:
            continue
        ratio = _containment(sequence, wanted)
        if ratio > score:
            best, score = chain, ratio
    return best, score

def stage_antibody(task: dict, structure_root: Path, *, min_ratio: float = 0.90) -> Staged:

    source = structure_root / f"{task['INSTANCE']}.cif"
    if not source.exists():
        raise RuntimeError(f"missing structure {source}")
    sequences = _loaded_chain_sequences(source)

    heavy, h_score = _best_chain(sequences, str(task["Hseq"]), set())

    single_chain_fv = _containment(sequences[heavy], str(task["Lseq"])) >= min_ratio
    if single_chain_fv:
        light, l_score = heavy, _containment(sequences[heavy], str(task["Lseq"]))
    else:
        light, l_score = _best_chain(sequences, str(task["Lseq"]), {heavy})
    taken = {heavy, light}
    antigen_chains, ag_scores = [], []
    for wanted in [s for s in str(task["ag_seq"]).split(",") if s]:
        chain, score = _best_chain(sequences, wanted, taken)
        if chain is None or score < min_ratio:
            raise RuntimeError(
                f"{task['INSTANCE']}: antigen sequence matches no remaining chain "
                f"(best {chain!r} ratio {score:.2f}); chains {sorted(sequences)}"
            )
        antigen_chains.append(chain)
        ag_scores.append(round(score, 3))
        taken.add(chain)
    if h_score < min_ratio or l_score < min_ratio:
        raise RuntimeError(
            f"{task['INSTANCE']}: heavy/light match too weak ({h_score:.2f}/{l_score:.2f})"
        )

    sequence = sequences[heavy]
    cdr = str(task["CDRH3"])
    start = sequence.find(cdr)
    if start < 0:
        raise RuntimeError(f"{task['INSTANCE']}: CDR-H3 {cdr!r} not in loaded heavy chain")
    if sequence.find(cdr, start + 1) >= 0:
        raise RuntimeError(f"{task['INSTANCE']}: CDR-H3 {cdr!r} occurs more than once")
    end = start + len(cdr) - 1
    return Staged(
        path=source,
        generated=f"{heavy}:{start}-{end}",
        context=heavy if light == heavy else f"{heavy},{light}",
        target=",".join(antigen_chains),
        design_length=len(cdr),
        design_chain=heavy,
        hotspot_labels=(),
        notes={
            "cdr_h3": cdr,
            "cdr_h3_graph_span": [start, end],
            "author_chains": {"H": task["Hchain"], "L": task["Lchain"],
                              "antigen": task["agchains"]},
            "loaded_chains": {"heavy": heavy, "light": light, "antigen": antigen_chains},
            "single_chain_fv": bool(single_chain_fv),
            "match_ratios": {"heavy": round(h_score, 3), "light": round(l_score, 3),
                             "antigen": ag_scores},
            "heavy_chain_residues": len(sequence),
            "staged": False,
        },
    )
