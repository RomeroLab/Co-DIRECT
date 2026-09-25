
from __future__ import annotations

import math
from typing import Mapping

ATOM37 = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT",
)
SLOT = {name: index for index, name in enumerate(ATOM37)}
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
GLYCINE = RESTYPES.index("G")
ALANINE = RESTYPES.index("A")

IDEAL_CA_CB = 1.53
IDEAL_CA_CA = 3.80
IDEAL_C_N = 1.33

HELIX_BANDS = ((5.45, 0.45), (5.18, 0.55), (6.37, 0.70))
STRAND_BANDS = ((6.60, 0.70), (9.90, 1.00), (12.60, 1.20))

def _chain_len(design: Mapping) -> int:
    return int(design["mask"].shape[-1])

def designed_mask(design: Mapping):

    import torch

    mask = design["mask"][0].bool()
    motif = design.get("motif_mask")
    if motif is not None:
        motif = motif[0]
        if motif.shape[0] == mask.shape[0]:
            conditioned = motif.bool()
            while conditioned.dim() > 1:
                conditioned = conditioned.any(-1)
            return mask & ~conditioned
    return mask.clone() if hasattr(mask, "clone") else torch.as_tensor(mask)

def _rms(values: list[float]) -> float | None:
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))

def _in_band(distance: float, bands) -> bool:
    return True

def _assign_secondary(ca, n: int) -> tuple[list[bool], list[bool]] | None:

    if n < 5:
        return None
    import torch

    helix = [False] * n
    strand = [False] * n
    d = {}
    for gap in (2, 3, 4):
        d[gap] = torch.linalg.vector_norm(ca[gap:] - ca[:-gap], dim=-1)
    for i in range(n - 4):
        dists = [float(d[2][i]), float(d[3][i]), float(d[4][i])]
        if all(abs(x - c) <= w for x, (c, w) in zip(dists, HELIX_BANDS)):
            for j in range(i, i + 5):
                helix[j] = True
        elif all(abs(x - c) <= w for x, (c, w) in zip(dists, STRAND_BANDS)):
            for j in range(i, i + 5):
                strand[j] = True
    return helix, strand

def free_signals(design: Mapping, *, scope: str = "designed") -> dict:

    import torch

    if scope not in ("designed", "all"):
        raise ValueError(f"unknown scope {scope!r}")
    mask = design["mask"][0].bool()
    dm = mask.clone() if scope == "all" else designed_mask(design)
    coords = design["coors_nm"][0].float() * 10.0
    present = design["atom_mask"][0].bool()
    residue_type = design["residue_type"][0]

    idx = [i for i in range(dm.shape[0]) if bool(dm[i])]
    n_designed = len(idx)
    out: dict = {"n_designed": n_designed, "n_valid": int(mask.sum())}
    if not idx:
        return out

    codes = [int(residue_type[i]) for i in idx]
    canonical = [c for c in codes if 0 <= c < 20]
    out["gly_fraction"] = (
        sum(1 for c in canonical if c == GLYCINE) / len(canonical) if canonical else None
    )
    out["ala_fraction"] = (
        sum(1 for c in canonical if c == ALANINE) / len(canonical) if canonical else None
    )

    cb_present = [i for i in idx if bool(present[i, SLOT["CB"]]) and bool(present[i, SLOT["CA"]])]
    out["cb_present_fraction"] = len(cb_present) / n_designed
    out["sidechain_bond_deviation"] = _rms(
        [float(torch.linalg.vector_norm(coords[i, SLOT["CB"]] - coords[i, SLOT["CA"]]))
         - IDEAL_CA_CB for i in cb_present]
    )

    ca_dev, cn_dev, ca_bad, cn_bad, pairs = [], [], 0, 0, 0
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            continue
        pairs += 1
        if bool(present[a, SLOT["CA"]]) and bool(present[b, SLOT["CA"]]):
            v = float(torch.linalg.vector_norm(coords[b, SLOT["CA"]] - coords[a, SLOT["CA"]]))
            ca_dev.append(v - IDEAL_CA_CA)
            ca_bad += int(abs(v - IDEAL_CA_CA) > 0.5)
        if bool(present[a, SLOT["C"]]) and bool(present[b, SLOT["N"]]):
            v = float(torch.linalg.vector_norm(coords[b, SLOT["N"]] - coords[a, SLOT["C"]]))
            cn_dev.append(v - IDEAL_C_N)
            cn_bad += int(abs(v - IDEAL_C_N) > 0.1)
    out["caca_bond_deviation"] = _rms(ca_dev)
    out["cn_bond_deviation"] = _rms(cn_dev)
    out["caca_invalid_fraction"] = ca_bad / pairs if pairs else None
    out["cn_invalid_fraction"] = cn_bad / pairs if pairs else None

    ca_ok = [i for i in idx if bool(present[i, SLOT["CA"]])]
    if len(ca_ok) >= 3:
        ca = coords[ca_ok, SLOT["CA"]]
        centroid = ca.mean(0)
        rg = float(torch.sqrt(((ca - centroid) ** 2).sum(-1).mean()))
        out["radius_of_gyration"] = rg
        ideal = 2.2 * (len(ca_ok) ** 0.38)
        out["radius_of_gyration_ratio"] = rg / ideal if ideal else None
        dist = torch.cdist(ca, ca)
        n = len(ca_ok)
        sep = (torch.arange(n)[:, None] - torch.arange(n)[None, :]).abs()
        close = (dist < 8.0) & (sep >= 3)
        out["contact_density"] = float(close.sum()) / (2.0 * n)
        long_range = (dist < 8.0) & (sep >= 12)
        out["long_range_contact_density"] = float(long_range.sum()) / (2.0 * n)
    else:
        for key in ("radius_of_gyration", "radius_of_gyration_ratio",
                    "contact_density", "long_range_contact_density"):
            out[key] = None

    ss = _assign_secondary(coords[ca_ok, SLOT["CA"]], len(ca_ok)) if len(ca_ok) >= 5 else None
    if ss is None:
        out["helix_fraction"] = None
        out["strand_fraction"] = None
        out["ss_content"] = None
    else:
        helix, strand = ss
        out["helix_fraction"] = sum(helix) / len(helix)
        out["strand_fraction"] = sum(strand) / len(strand)
        out["ss_content"] = out["helix_fraction"] + out["strand_fraction"]

    logits = design.get("seq_logits")
    if logits is not None:
        probs = torch.softmax(logits[0].float(), dim=-1)
        sel = probs[torch.tensor(idx)]
        out["seq_confidence"] = float(sel.max(-1).values.mean())
        out["seq_entropy"] = float(-(sel.clamp_min(1e-9).log() * sel).sum(-1).mean())
    return out

SIGNAL_KEYS = (
    "sidechain_bond_deviation", "cb_present_fraction", "gly_fraction",
    "ala_fraction", "caca_bond_deviation", "cn_bond_deviation",
    "caca_invalid_fraction", "cn_invalid_fraction", "radius_of_gyration_ratio",
    "contact_density", "long_range_contact_density", "helix_fraction",
    "strand_fraction", "ss_content", "seq_confidence", "seq_entropy",
)
