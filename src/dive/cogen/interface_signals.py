
from __future__ import annotations

from collections.abc import Sequence

POSITIVE = set("KR")
NEGATIVE = set("DE")
AROMATIC = set("YWF")
HYDROPHOBIC = set("AVLIMFWC")

KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

def net_charge(sequence: str) -> float:
    return float(sum(1 for c in sequence if c in POSITIVE)
                 - sum(1 for c in sequence if c in NEGATIVE))

def interface_signals(designed: str, antigen: str) -> dict:

    n = len(designed)
    if n == 0:
        raise ValueError("empty designed span")
    if not antigen:
        raise ValueError("no antigen sequence; this signal set names an "
                         "interface that would not exist")
    charge = net_charge(designed)
    antigen_charge = net_charge(antigen)
    return {
        "h3_aromatic_fraction": sum(1 for c in designed if c in AROMATIC) / n,
        "h3_tyr_fraction": designed.count("Y") / n,
        "h3_hydrophobic_fraction": sum(1 for c in designed if c in HYDROPHOBIC) / n,
        "h3_kyte_doolittle": sum(KD.get(c, 0.0) for c in designed) / n,
        "h3_net_charge": charge,
        "h3_abs_net_charge": abs(charge),

        "h3_charge_complementarity": -(charge * antigen_charge) / n,
        "h3_polar_fraction": sum(1 for c in designed if c in "STNQYHKRDE") / n,
        "h3_glycine_fraction": designed.count("G") / n,
        "h3_proline_fraction": designed.count("P") / n,
    }

INTERFACE_SIGNAL_KEYS: Sequence[str] = (
    "h3_aromatic_fraction", "h3_tyr_fraction", "h3_hydrophobic_fraction",
    "h3_kyte_doolittle", "h3_net_charge", "h3_abs_net_charge",
    "h3_charge_complementarity", "h3_polar_fraction", "h3_glycine_fraction",
    "h3_proline_fraction",
)
