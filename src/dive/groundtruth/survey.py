
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dive.groundtruth.complex_spec import Eligibility, evaluate_eligibility
from dive.groundtruth.rcsb import EntryMetadata

GATE_A_MINIMUM = 4

class SurveyError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class GateVerdict:

    eligible_count: int
    minimum: int
    passed: bool
    statement: str

    def to_dict(self) -> dict[str, object]:

        return {
            "eligible_count": self.eligible_count,
            "minimum": self.minimum,
            "passed": self.passed,
            "statement": self.statement,
        }

def apply_gate_a(results: Sequence[Eligibility]) -> GateVerdict:

    count = sum(1 for result in results if result.eligible)
    passed = count >= GATE_A_MINIMUM
    if not passed:
        statement = (
            f"{count} eligible complexes is below the pre-registered minimum of "
            f"{GATE_A_MINIMUM}. The study is reported as DESCRIPTIVE ONLY and is "
            f"not called a validation."
        )
    else:
        statement = (
            f"{count} eligible complexes meets the pre-registered minimum of "
            f"{GATE_A_MINIMUM}. This is a small-n study and every verdict states "
            f"that limitation in its own sentence."
        )
    return GateVerdict(
        eligible_count=count, minimum=GATE_A_MINIMUM, passed=passed, statement=statement
    )

def survey_targets(
    targets: Mapping[str, Mapping], metadata_by_id: Mapping[str, EntryMetadata]
) -> tuple[Eligibility, ...]:

    results: list[Eligibility] = []
    for name in sorted(targets):
        entry = targets[name]
        pdb_id = entry.get("pdb_id")
        if not pdb_id:
            continue
        metadata = metadata_by_id.get(str(pdb_id))
        if metadata is None:
            raise SurveyError(f"no metadata supplied for {pdb_id} (target {name})")
        results.append(
            evaluate_eligibility(
                pdb_id=str(pdb_id),
                target=name,
                chains=metadata.chains,
                target_input=str(entry["target_input"]),
                binder_length=entry["binder_length"],
            )
        )
    return tuple(results)
