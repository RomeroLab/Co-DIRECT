
from __future__ import annotations

ALL_TASKS: tuple[str, ...] = (
    "01_PD1", "02_PDL1", "03_PDL1_AAV", "04_IFNAR2",
    "05_CD45", "06_CD45", "07_CD45", "08_CD45", "09_CD45", "10_CD45", "11_CD45",
    "12_Claudin1", "13_BBF14", "14_CrSAS6",
    "15_DerF7", "16_DerF7", "17_DerF7",
    "18_DerF21", "19_DerF21", "20_DerF21", "21_DerF21", "22_DerF21",
    "23_BetV1", "24_SpCas9", "25_CbAgo", "26_CbAgo",
    "27_HER2_AAV", "28_HER2_AAV", "29_BHRF1", "30_SC2RBD",
    "31_IL7RA", "31_IL7RA_FIX", "31_IL7RA_REPACK",
    "32_PDL1_ALPHA", "32_PDL1_ALPHA_FIX", "32_PDL1_ALPHA_REPACK",
    "33_TrkA", "34_Insulin", "35_H1", "36_VEGFA", "37_IL17A",
    "38_TNFalpha", "38_TNFalpha_FIX", "38_TNFalpha_REPACK",
)

_GROUP_OF_TASK: dict[str, str] = {
    "01_PD1": "PD1",
    "02_PDL1": "PDL1", "03_PDL1_AAV": "PDL1",
    "32_PDL1_ALPHA": "PDL1", "32_PDL1_ALPHA_FIX": "PDL1",
    "32_PDL1_ALPHA_REPACK": "PDL1",
    "04_IFNAR2": "IFNAR2",
    "05_CD45": "CD45", "06_CD45": "CD45", "07_CD45": "CD45", "08_CD45": "CD45",
    "09_CD45": "CD45", "10_CD45": "CD45", "11_CD45": "CD45",
    "12_Claudin1": "Claudin1", "13_BBF14": "BBF14", "14_CrSAS6": "CrSAS6",
    "15_DerF7": "DerF7", "16_DerF7": "DerF7", "17_DerF7": "DerF7",
    "18_DerF21": "DerF21", "19_DerF21": "DerF21", "20_DerF21": "DerF21",
    "21_DerF21": "DerF21", "22_DerF21": "DerF21",
    "23_BetV1": "BetV1", "24_SpCas9": "SpCas9",
    "25_CbAgo": "CbAgo", "26_CbAgo": "CbAgo",
    "27_HER2_AAV": "HER2", "28_HER2_AAV": "HER2",
    "29_BHRF1": "BHRF1", "30_SC2RBD": "SC2RBD",
    "31_IL7RA": "IL7RA", "31_IL7RA_FIX": "IL7RA", "31_IL7RA_REPACK": "IL7RA",
    "33_TrkA": "TrkA", "34_Insulin": "Insulin", "35_H1": "H1",
    "36_VEGFA": "VEGFA", "37_IL17A": "IL17A",
    "38_TNFalpha": "TNFalpha", "38_TNFalpha_FIX": "TNFalpha",
    "38_TNFalpha_REPACK": "TNFalpha",
}

EXPOSED_GROUPS: tuple[str, ...] = ("TrkA", "PD1")

CONFIRMATION_GROUPS: tuple[str, ...] = (
    "PDL1", "DerF21", "HER2", "SC2RBD", "Insulin", "IL17A", "CrSAS6",
)

DEVELOPMENT_GROUPS: tuple[str, ...] = (
    "TrkA", "PD1", "IFNAR2", "CD45", "Claudin1", "BBF14", "DerF7", "BetV1",
    "SpCas9", "CbAgo", "BHRF1", "IL7RA", "H1", "VEGFA", "TNFalpha",
)

def parent_group(task: str) -> str:

    try:
        return _GROUP_OF_TASK[task]
    except KeyError as error:
        raise KeyError(
            f"unknown binder task {task!r}; add it to _GROUP_OF_TASK with its "
            f"protein, because a guessed group silently leaks the split"
        ) from error

def split_for(task: str) -> str:
    group = parent_group(task)
    if group in CONFIRMATION_GROUPS:
        return "confirmation"
    if group in DEVELOPMENT_GROUPS:
        return "development"
    raise KeyError(f"group {group!r} is on neither side of the frozen split")

def tasks_in(side: str) -> tuple[str, ...]:
    return tuple(t for t in ALL_TASKS if split_for(t) == side)
