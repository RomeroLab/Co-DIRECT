
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ame_v2_rt as R
R.bootstrap()
import numpy as np, torch, yaml
import biotite.structure.io.pdb as pdbio
from biotite.structure import filter_amino_acids
from openfold.np.residue_constants import atom_order, atom_types, restype_3to1, restypes
import ame_requirements as Q

RESTYPE_1TO3 = {v: k for k, v in restype_3to1.items()}

def check(task, sample_path):
    td = yaml.safe_load((R.UPSTREAM / "configs/design_tasks/ame_dict_v2.yaml").read_text())
    entry = td["motif_target_dict_cfg"][task]
    pdb = R.UPSTREAM / str(entry["target_path"]).lstrip("./")
    arr = pdbio.PDBFile.read(str(pdb)).get_structure(model=1)
    prot = arr[filter_amino_acids(arr)]
    het = arr[~filter_amino_acids(arr)]

    s = torch.load(sample_path, map_location="cpu", weights_only=False)
    xm, mm, sm = s["x_motif"][0], s["motif_mask"][0].bool(), s["seq_motif"][0]
    parsed = Q.parse_contig_atoms(entry["contig_atoms"])

    out = {"task": task, "pdb": pdb.name, "declared_ligand": entry.get("ligand"),
           "n_motif_res_spec": len(parsed), "n_motif_res_tensor": int(mm.shape[0]),
           "n_motif_atoms_spec": sum(len(n) for _, _, n in parsed),
           "n_motif_atoms_tensor": int(mm.sum()),
           "residues": [], "max_coord_mismatch_nm": 0.0, "problems": []}

    for k, (chain, rid, names) in enumerate(parsed):
        sel = prot[(prot.chain_id == chain) & (prot.res_id == rid)]
        if len(sel) == 0:
            out["problems"].append(f"{chain}{rid} absent from PDB"); continue
        pdb_name = str(sel.res_name[0])
        rt = int(sm[k])
        tensor_name = RESTYPE_1TO3.get(restypes[rt], "UNK") if rt < len(restypes) else "UNK"
        found, missing, worst = [], [], 0.0
        for nm in names:
            at = sel[sel.atom_name == nm]
            if len(at) == 0:
                missing.append(nm); continue
            a37 = atom_order.get(nm)
            if a37 is None or not bool(mm[k, a37]):
                out["problems"].append(f"{chain}{rid}:{nm} not marked in motif_mask")
                continue
            d = float(np.linalg.norm(np.asarray(at.coord[0]) / 10.0
                                     - xm[k, a37].numpy()))
            worst = max(worst, d)
            found.append(nm)
        out["max_coord_mismatch_nm"] = max(out["max_coord_mismatch_nm"], worst)
        if missing:
            out["problems"].append(f"{chain}{rid} atoms absent from PDB: {missing}")
        if pdb_name != tensor_name:
            out["problems"].append(
                f"{chain}{rid} residue name PDB={pdb_name} tensor={tensor_name}")
        out["residues"].append({"chain": chain, "res_id": rid, "pdb_res_name": pdb_name,
                                "tensor_res_name": tensor_name, "atoms": names,
                                "atoms_found": found, "max_dev_nm": round(worst, 6)})

    lig_mask = s["target_mask"][0].bool()
    lig = s["x_target"][0][lig_mask]
    hn = sorted(set(het.res_name.tolist()))
    out["het_resnames_in_pdb"] = hn
    out["n_ligand_atoms_tensor"] = int(lig.shape[0])
    out["n_het_atoms_in_pdb"] = int(len(het))
    out["bond_graph_nonzero"] = int(s["target_bond_mask"].sum())
    out["bond_graph_available"] = bool(int(s["target_bond_mask"].sum()) > 0)
    if len(het):
        hp = np.asarray(het.coord) / 10.0
        d = np.linalg.norm(lig.numpy()[:, None] - hp[None], axis=-1)
        out["ligand_max_nn_mismatch_nm"] = float(d.min(axis=1).max())
        out["ligand_elements_pdb"] = sorted(set(
            (e if e else n[0]) for e, n in zip(het.element, het.atom_name)))
    out["ok"] = (not out["problems"]
                 and out["n_motif_atoms_spec"] == out["n_motif_atoms_tensor"]
                 and out["max_coord_mismatch_nm"] < 1e-4
                 and out.get("ligand_max_nn_mismatch_nm", 1.0) < 1e-4)
    return out

if __name__ == "__main__":
    root = Path(sys.argv[1]); dest = Path(sys.argv[2])
    res = []
    for td in sorted(root.iterdir()):
        if not td.is_dir():
            continue
        f = sorted(td.glob("native_seed*.pt"))
        if not f:
            continue
        try:
            res.append(check(td.name, f[0]))
        except Exception as exc:
            res.append({"task": td.name, "ok": False,
                        "problems": [f"{type(exc).__name__}: {exc}"[:300]]})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2, default=str))
    for r in res:
        print(f"{r['task']:16s} ok={str(r.get('ok')):5s} "
              f"atoms {r.get('n_motif_atoms_spec')}/{r.get('n_motif_atoms_tensor')} "
              f"coord_dev={r.get('max_coord_mismatch_nm')} "
              f"lig_dev={r.get('ligand_max_nn_mismatch_nm')} "
              f"bonds={r.get('bond_graph_available')} "
              f"{('PROBLEMS: '+ '; '.join(r.get('problems', [])[:2])) if r.get('problems') else ''}")
