
from __future__ import annotations

from dataclasses import dataclass, field

class AntibodyTaskError(RuntimeError):
    pass

@dataclass(frozen=True)
class AntibodyTask:

    contig_string: str
    chain: str
    h3_author_ids: list[int]
    framework_author_ids: list[int]

    design_length: int
    total_length: int
    segments: list[str] = field(default_factory=list)

    def as_task_entry(self, motif_pdb_path: str) -> dict:

        return {
            "contig_string": self.contig_string,
            "motif_pdb_path": motif_pdb_path,
            "motif_only": False,
            "motif_min_length": self.total_length,
            "motif_max_length": self.total_length,
            "segment_order": self.chain,
            "atom_selection_mode": "bb3o",
        }

def _runs(values: list[int]) -> list[tuple[int, int]]:

    runs: list[tuple[int, int]] = []
    for value in values:
        if runs and value == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], value)
        else:
            runs.append((value, value))
    return runs

def framework_contig(
    author_res_ids,
    resolved_sequence: str,
    h3_sequence: str,
    *,
    chain: str,
    length_window: tuple[int, int] | None = None,
) -> AntibodyTask:

    author_res_ids = [int(v) for v in author_res_ids]
    if len(author_res_ids) != len(resolved_sequence):
        raise AntibodyTaskError(
            f"length mismatch: {len(author_res_ids)} residue ids against "
            f"{len(resolved_sequence)} sequence characters; a positional map "
            f"between them would place the loop wrongly"
        )
    if not h3_sequence:
        raise AntibodyTaskError("empty CDR-H3")

    occurrences = resolved_sequence.count(h3_sequence)
    if occurrences == 0:
        raise AntibodyTaskError(
            f"CDR-H3 {h3_sequence!r} not found in the resolved sequence; the "
            f"loop is partly unresolved in this file and cannot be stated"
        )
    if occurrences > 1:
        raise AntibodyTaskError(
            f"CDR-H3 {h3_sequence!r} appears {occurrences} times in the "
            f"resolved sequence; the window would be a guess"
        )

    start = resolved_sequence.index(h3_sequence)
    end = start + len(h3_sequence) - 1
    if start == 0 or end == len(resolved_sequence) - 1:
        raise AntibodyTaskError(
            "the CDR-H3 touches a chain terminus, so it has framework on one "
            "side only; closing a truncated chain is a different problem"
        )

    h3_ids = author_res_ids[start : end + 1]
    before = author_res_ids[:start]
    after = author_res_ids[end + 1 :]

    low, high = length_window if length_window else (len(h3_sequence), len(h3_sequence))
    if not (1 <= low <= high):
        raise AntibodyTaskError(f"invalid length window {length_window!r}")

    segments = [f"{chain}{lo}-{hi}" for lo, hi in _runs(before)]
    segments.append(f"{low}-{high}")
    segments.extend(f"{chain}{lo}-{hi}" for lo, hi in _runs(after))

    return AntibodyTask(
        contig_string="/".join(segments),
        chain=chain,
        h3_author_ids=h3_ids,
        framework_author_ids=before + after,
        design_length=len(h3_sequence),
        total_length=len(author_res_ids),
        segments=segments,
    )
