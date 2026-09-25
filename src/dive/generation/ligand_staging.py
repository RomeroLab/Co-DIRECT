
from __future__ import annotations

REQUIRED_TENSORS = (
    "x_motif", "motif_mask", "seq_motif",
    "x_target", "seq_target", "target_mask",
    "target_bond_mask", "target_bond_order",
    "target_charge", "target_atom_name", "target_laplacian_pe",
)

class AmeStagingError(RuntimeError):
    pass

def contig_atoms(motif) -> str:

    if not motif:
        raise AmeStagingError("AME conditioning cannot be an empty motif")
    parts = []
    for key, rec in motif.items():
        atoms = (rec or {}).get("atoms") or []
        if not atoms:
            raise AmeStagingError(
                f"motif residue {key} declares no atoms; it cannot specify a "
                "geometry and the theozyme would be silently incomplete")
        parts.append(f"{key}: [{', '.join(atoms)}]")
    return "; ".join(parts)

def ligand_arg(task):

    ligands = (task or {}).get("ligands") or []
    if not ligands:
        raise AmeStagingError(
            "AME task declares no ligand; staging it would repeat the "
            "ligand-blind run this module exists to end")
    return list(ligands)

def assert_ligand_present(batch) -> None:

    missing = [k for k in REQUIRED_TENSORS if k not in batch]
    if missing:
        raise AmeStagingError(
            f"staged AME batch is missing {missing}; the specialist checkpoint "
            f"would run without the conditioning it was trained on")

def build_features(task, upstream_import=None):

    if upstream_import is None:
        from proteinfoundation.datasets.gen_dataset import (
            LigandFeatures, MotifFeatures,
        )
    else:
        MotifFeatures, LigandFeatures = upstream_import
    spec = contig_atoms(task["motif"])
    nres = [int(task["design_length"])]
    motif = MotifFeatures(task_name=task["task_id"], pdb_path=task["input_pdb"],
                          motif_atom_spec=spec)
    ligand = LigandFeatures(task_name=task["task_id"], pdb_path=task["input_pdb"],
                            ligand=ligand_arg(task), use_bonds_from_file=True)
    return motif, ligand, nres

MOTIF_TENSORS = ("x_motif", "motif_mask", "seq_motif", "seq_motif_mask")
LIGAND_TENSORS = ("x_target", "seq_target", "seq_target_mask", "target_mask",
                  "target_bond_mask", "target_bond_order", "target_charge",
                  "target_atom_name", "target_laplacian_pe")

SCALAR_FLAGS = ("atomistic_target", "prepend_target")

def inject_conditioning(batch, features):

    missing = [k for k in LIGAND_TENSORS + MOTIF_TENSORS if k not in features]
    if missing:
        raise AmeStagingError(
            f"feature set is missing {missing}; refusing to build a batch that "
            f"would silently condition on part of the theozyme")

    out = dict(batch)
    for key in MOTIF_TENSORS + LIGAND_TENSORS:
        value = features[key]

        out[key] = value.unsqueeze(0) if hasattr(value, "unsqueeze") else value
    for flag in SCALAR_FLAGS:
        if flag in features:
            out[flag] = features[flag]
    out["atomistic_target"] = True

    if out["x_target"].ndim != 3:
        raise AmeStagingError(
            f"x_target is {out['x_target'].ndim}-D after injection; the motif is "
            "still occupying the target slot and the ligand consumer would read "
            "protein residues")
    return out
