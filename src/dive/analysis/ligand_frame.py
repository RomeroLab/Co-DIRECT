
import numpy as np
from rdkit import Chem

class DegenerateLigandFrame(ValueError):
    pass

def atom_maps(reference, prediction, limit=4096):

    ref, pred = [Chem.RemoveHs(m) for m in (reference, prediction)]
    if Chem.MolToSmiles(ref) != Chem.MolToSmiles(pred):
        raise ValueError('chemical identity differs')
    names = []
    for mol in (ref, pred):
        ns = [a.GetProp('name') if a.HasProp('name') else '' for a in mol.GetAtoms()]
        if not all(ns) or len(set(ns)) != len(ns):
            raise ValueError('atom names must be nonempty and unique')
        names.append(ns)
    matches = pred.GetSubstructMatches(ref, uniquify=False, useChirality=True,
                                      maxMatches=limit + 1)
    if len(matches) > limit:
        raise ValueError('chemical symmetry exceeds enumeration limit')
    if not matches or any(len(m) != pred.GetNumAtoms() for m in matches):
        raise ValueError('no complete chemical correspondence')
    return [dict(zip(names[0], [names[1][i] for i in m])) for m in matches]

def placement(pred_ligand, ref_ligand, pred_motif, ref_motif):

    pl, rl, pm, rm = [np.asarray(x, dtype=float) for x in
                      (pred_ligand, ref_ligand, pred_motif, ref_motif)]
    if (any(x.ndim != 2 or x.shape[1] != 3 for x in (pl, rl, pm, rm))
            or pl.shape != rl.shape or pm.shape != rm.shape or len(pm) == 0):
        raise ValueError('coordinate shape or coverage mismatch')
    if not all(np.isfinite(x).all() for x in (pl, rl, pm, rm)):
        raise ValueError('coordinates must be finite')
    pc, rc = pl.mean(axis=0), rl.mean(axis=0)
    if min(np.linalg.matrix_rank(pl-pc), np.linalg.matrix_rank(rl-rc)) < 2:
        raise DegenerateLigandFrame('degenerate ligand frame')
    u, _, vt = np.linalg.svd((pl-pc).T @ (rl-rc))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    ligand_d = np.linalg.norm((pl-pc) @ rotation + rc - rl, axis=1)
    motif_d = np.linalg.norm((pm-pc) @ rotation + rc - rm, axis=1)
    return {'ligand_fit_rmsd': float(np.sqrt(np.mean(ligand_d**2))),
            'motif_rmsd': float(np.sqrt(np.mean(motif_d**2))),
            'motif_max_displacement': float(motif_d.max()),
            'motif_atom_displacements': motif_d.tolist()}
