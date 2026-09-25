
from __future__ import annotations

import hashlib

CONTACT_CUTOFF_NM = 0.45

MIN_LIGAND_HEAVY = 6

MCSA_RES_HIST = ((3, 9), (4, 11), (5, 4), (6, 7), (7, 4))

MAX_ATOMS_PER_RES = 4

MIN_MOTIF_RES = 2
MIN_MOTIF_ATOMS = 4

def motif_size_for(example_id: str) -> int:

    total = sum(w for _, w in MCSA_RES_HIST)
    digest = hashlib.sha256(str(example_id).encode()).digest()
    pick = int.from_bytes(digest[:8], "big") % total
    for n, w in MCSA_RES_HIST:
        if pick < w:
            return n
        pick -= w
    return MCSA_RES_HIST[-1][0]

class LigandContactMotifTransform:

    def __init__(self, *, cutoff_nm: float = CONTACT_CUTOFF_NM,
                 max_atoms_per_res: int = MAX_ATOMS_PER_RES,
                 min_ligand_heavy: int = MIN_LIGAND_HEAVY,
                 min_motif_res: int = MIN_MOTIF_RES, stats: list | None = None):
        self.cutoff_nm = cutoff_nm
        self.max_atoms_per_res = max_atoms_per_res
        self.min_ligand_heavy = min_ligand_heavy
        self.min_motif_res = min_motif_res

        self.stats = stats

    def __call__(self, data):
        import torch

        coords = data.coords_nm
        valid = data.coord_mask.bool()
        n = coords.shape[0]
        motif = torch.zeros(n, 37, dtype=torch.bool)

        protein = getattr(data, "binder_residue_mask", None)
        ligand = getattr(data, "target_residue_mask", None)
        if protein is None or ligand is None:
            data.motif_mask = motif
            return data
        protein, ligand = protein.bool(), ligand.bool()

        lig_xyz = coords[ligand][valid[ligand]]
        if lig_xyz.shape[0] < self.min_ligand_heavy:
            data.motif_mask = motif
            self._record(data, 0, 0, "ligand too small")
            return data

        rows = torch.nonzero(protein, as_tuple=True)[0]
        candidates = []
        for r in rows.tolist():
            slots = torch.nonzero(valid[r], as_tuple=True)[0]
            if slots.numel() == 0:
                continue
            per_atom = torch.cdist(coords[r, slots], lig_xyz).min(dim=1).values

            touching = per_atom <= self.cutoff_nm
            if not bool(touching.any()):
                continue
            candidates.append({"row": r,
                               "slots": slots[touching],
                               "dists": per_atom[touching],
                               "n": int(touching.sum()),
                               "min_d": float(per_atom[touching].min())})
        if len(candidates) < self.min_motif_res:
            data.motif_mask = motif
            self._record(data, 0, 0, "too few contacts")
            return data

        candidates.sort(key=lambda c: (-c["n"], c["min_d"], c["row"]))
        want = motif_size_for(getattr(data, "id", "") or "")
        chosen = candidates[:want]
        n_atoms = sum(min(c["n"], self.max_atoms_per_res) for c in chosen)
        if len(chosen) < self.min_motif_res or n_atoms < MIN_MOTIF_ATOMS:
            data.motif_mask = motif
            self._record(data, len(chosen), n_atoms, "motif too small")
            return data

        for c in chosen:
            keep = torch.argsort(c["dists"])[: self.max_atoms_per_res]
            motif[c["row"], c["slots"][keep]] = True
        data.motif_mask = motif
        self._record(data, len(chosen), n_atoms, "ok")
        return data

    def _record(self, data, n_res, n_atoms, why):
        if self.stats is not None:
            self.stats.append({"id": getattr(data, "id", None),
                               "n_motif_residues": n_res,
                               "n_motif_atoms": n_atoms, "status": why})
