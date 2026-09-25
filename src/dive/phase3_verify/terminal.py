
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dive.phase3_verify.supplement import (
    CoverageDefinition,
    bind_decision,
    read_terminal,
)

class GoverningCoverage(Enum):

    APPROVED_SPEC = "approved_spec"

    IMPLEMENTATION = "implementation"

class TerminalAction(Enum):

    WAIT = "wait"

    VERIFY_AND_BIND = "verify_and_bind"

    PRESERVE_AND_FIX = "preserve_and_fix"

    REFUSE = "refuse"

@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    action: TerminalAction
    bind_allowed: bool
    reason: str
    refusals: list[str] = field(default_factory=list)

def _manifest_refusals(root: Path, expected: str | None) -> list[str]:

    manifest = root / "entity_manifest.parquet"
    if expected is None:
        return ["manifest sha256 was not checked: no expected value supplied"]
    if not manifest.is_file():
        return []
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != expected:
        return [
            f"manifest sha256 is {digest[:16]}... but {expected[:16]}... was "
            f"expected; the reconstruction described a different population"
        ]
    return []

def handle_terminal(
    evidence_root: Path,
    *,
    governing_coverage: GoverningCoverage | None = None,
    coverage_reconciled: bool = False,
    expect_manifest_sha256: str | None = None,
) -> TerminalOutcome:

    if coverage_reconciled is None:
        raise TypeError(
            "coverage_reconciled must be True or False; there is no undecided "
            "state that licenses a bind"
        )

    root = Path(evidence_root)
    terminal = read_terminal(root)
    if terminal is None:
        return TerminalOutcome(
            action=TerminalAction.WAIT,
            bind_allowed=False,
            reason="no terminal record: the run has not finished",
        )

    if terminal.status == "FAILED":
        cause = terminal.error_message or "(no cause recorded)"
        return TerminalOutcome(
            action=TerminalAction.PRESERVE_AND_FIX,
            bind_allowed=False,
            reason=f"FAILED: {cause}",
            refusals=[
                "preserve this root; do not delete it",
                "do not reduce membership when fixing",
                "relaunch under a NEW run id, not this one",
            ],
        )

    if terminal.status != "COMPLETE":
        return TerminalOutcome(
            action=TerminalAction.REFUSE,
            bind_allowed=False,
            reason=(
                f"terminal status {terminal.status!r} is not one the goal "
                f"defines; it is refused rather than mapped onto COMPLETE or "
                f"FAILED"
            ),
        )

    if governing_coverage is GoverningCoverage.APPROVED_SPEC:
        coverage = CoverageDefinition.RECONCILED
        coverage_refusal = None
    elif governing_coverage is GoverningCoverage.IMPLEMENTATION:
        coverage = CoverageDefinition.UNRECONCILED
        coverage_refusal = (
            "the implemented coverage rule governs but it misses 76 validation "
            "entities the approved coverage formula flags; binding it would "
            "call them clean"
        )
    elif governing_coverage is None:
        coverage = (
            CoverageDefinition.RECONCILED
            if coverage_reconciled
            else CoverageDefinition.UNRECONCILED
        )
        coverage_refusal = (
            None
            if coverage_reconciled
            else "governing coverage rule was not stated"
        )
    else:
        raise TypeError(f"unknown governing coverage {governing_coverage!r}")
    manifest_refusals = _manifest_refusals(root, expect_manifest_sha256)
    mismatched = any(r.startswith("manifest sha256 is") for r in manifest_refusals)
    if mismatched:

        refusals = list(manifest_refusals)
        if coverage_refusal is not None:
            refusals.append(coverage_refusal)
        return TerminalOutcome(
            action=TerminalAction.VERIFY_AND_BIND,
            bind_allowed=False,
            reason=f"COMPLETE but {len(refusals)} refusal(s) stand",
            refusals=refusals,
        )

    decision = bind_decision(root, coverage=coverage)
    refusals = [
        r for r in decision.refusals
        if "coverage definition is not reconciled" not in r
    ]
    if coverage_refusal is not None:
        refusals.append(coverage_refusal)
    refusals.extend(manifest_refusals)
    allowed = decision.allowed and not refusals
    return TerminalOutcome(
        action=TerminalAction.VERIFY_AND_BIND,
        bind_allowed=allowed,
        reason=(
            "COMPLETE: the hash may bind"
            if allowed
            else f"COMPLETE but {len(refusals)} refusal(s) stand"
        ),
        refusals=refusals,
    )

__all__ = (
    "GoverningCoverage",
    "TerminalAction",
    "TerminalOutcome",
    "handle_terminal",
)
