
from __future__ import annotations

import re

AMENDMENT = {
    "id": "A4-common-if-k",
    "status": "prospective",
    "declared_on": "2026-09-07",
    "pre_registered": False,
    "why_prospective": (
        "the registered baseline proteina_complexa_common_if_k was READY_EXACT "
        "but K was never pinned in any frozen artifact. K is fixed here, before "
        "any K-arm result is computed, and is reported as a prospective "
        "amendment; it is never presented as a past pre-registered value."
    ),
    "K": 8,
    "why_this_K": (
        "8 sequences per backbone is the redesign budget the ProteinMPNN "
        "authors use for self-consistency reporting. It is chosen on that "
        "external convention rather than on anything observed in this campaign."
    ),
    "seed": 42,
    "temperature": 0.1,
    "selection_rule": (
        "the inverse folder's own sequence score: minimum global_score for "
        "ProteinMPNN, maximum overall_confidence for LigandMPNN"
    ),
    "selection_uses_evaluator_metric": False,
    "why_not_best_of_k_on_the_metric": (
        "selecting on the evaluator's own quantity would make this arm an "
        "internal search scored by its own selection criterion, which is the "
        "coupling reported for BoltzGen and BindCraft."
    ),

    "record_all_k_sequences": True,
    "evaluate_selected_only": True,
    "compute_budget": (
        "K=8 inverse-folded sequences per backbone recorded; one AF2 evaluation "
        "per backbone, on the sequence the inverse folder scored best"
    ),
}

FAMILY = {
    "binder":   {"if1_dir": "if1_designed",    "tool": "proteinmpnn",
                 "glob": "binder:*"},
    "ame":      {"if1_dir": "if1_ligandmpnn",  "tool": "ligandmpnn",
                 "glob": "ame:*"},
    "antibody": {"if1_dir": "if1_designed",    "tool": "proteinmpnn",
                 "glob": "*"},
}

def assert_reproduces_if1(if1_sequence: str, first_k_sample: str, task: str) -> None:

    if if1_sequence != first_k_sample:
        raise CommonIfKError(
            f"{task}: the K run's first sample does not reproduce the K=1 "
            f"sequence ({len(if1_sequence)} vs {len(first_k_sample)} residues); "
            "the K call did not inherit the K=1 settings, so the two arms would "
            "differ by more than K"
        )

SCORE_FIELD = {
    "proteinmpnn": ("global_score", False),
    "ligandmpnn": ("overall_confidence", True),
}

_FIELD = re.compile(r"(\w+)\s*=\s*(-?\d+\.?\d*)")

def _canonical_designed_sequence(input_block: str, sample_block: str,
                                 tool: str, chain: str | None = None) -> str:

    from dive.benchmark.binder_af2_cells import (
        parse_ligandmpnn_fasta, parse_mpnn_fasta)
    two_record = f">{input_block}>{sample_block}"
    try:
        if tool == "ligandmpnn":
            return parse_ligandmpnn_fasta(two_record)["designed_sequence"]
        parsed = parse_mpnn_fasta(two_record)
        if chain is None:
            return parsed.designed_sequence

        if chain not in parsed.sequence_by_chain:
            raise CommonIfKError(
                f"chain {chain!r} is not in the record "
                f"(has {sorted(parsed.sequence_by_chain)})")
        return parsed.sequence_by_chain[chain].replace("X", "")
    except (ValueError, KeyError) as err:
        raise CommonIfKError(f"cannot read sample: {err}") from err

class CommonIfKError(RuntimeError):
    pass

def parse_mpnn_samples(text: str, tool: str,
                       chain: str | None = None) -> list[dict]:

    if tool not in SCORE_FIELD:
        raise CommonIfKError(f"unknown inverse folder: {tool!r}")
    field, _larger_is_better = SCORE_FIELD[tool]

    blocks = [b for b in text.split(">")[1:] if b.strip()]
    if len(blocks) < 2:
        raise CommonIfKError("FASTA has no sampled record (need input + samples)")

    out = []
    for block in blocks[1:]:
        header, _, _body = block.partition("\n")
        fields = {k: float(v) for k, v in _FIELD.findall(header)}
        if field not in fields:
            raise CommonIfKError(
                f"sample header has no {field!r}: {header.strip()!r}")

        sequence = _canonical_designed_sequence(blocks[0], block, tool, chain)
        out.append({
            "index": len(out) + 1,
            "sequence": sequence,
            "metrics": fields,
            "selection_value": fields[field],
            "selection_field": field,
            "tool": tool,
        })
    if not out:
        raise CommonIfKError("no sampled sequences in FASTA")
    return out

def select_best(samples, tool: str) -> dict:

    if tool not in SCORE_FIELD:
        raise CommonIfKError(f"unknown inverse folder: {tool!r}")
    if not samples:
        raise CommonIfKError("no samples to select from")
    _field, larger_is_better = SCORE_FIELD[tool]
    return (max if larger_is_better else min)(
        samples, key=lambda s: s["selection_value"])
