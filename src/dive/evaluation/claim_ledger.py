
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

STATUSES: frozenset[str] = frozenset(
    {"established", "unestablished", "blocked", "withdrawn"}
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "claim",
    "required_contrast",
    "raw_artifact",
    "result",
    "identification_limit",
    "permitted_wording",
    "status",
)

OPTIONAL_FIELDS: tuple[str, ...] = ("endpoint", "notes")

class ClaimLedgerError(RuntimeError):
    pass

def load_ledger(path: "str | Path") -> list[dict[str, Any]]:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    claims = data["claims"] if isinstance(data, dict) else data
    if not isinstance(claims, list):
        raise ClaimLedgerError(f"{path}: expected a list of claims")
    return claims

def validate_ledger(claims: Sequence[dict[str, Any]]) -> None:

    seen: set[str] = set()
    for index, claim in enumerate(claims):
        where = claim.get("id", f"#{index}")

        missing = [f for f in REQUIRED_FIELDS if f not in claim]
        if missing:
            raise ClaimLedgerError(f"{where}: missing field(s) {missing}")
        unknown = sorted(set(claim) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
        if unknown:
            raise ClaimLedgerError(
                f"{where}: unknown field(s) {unknown}; a typo here silently drops "
                f"the rule it was meant to set"
            )

        if claim["id"] in seen:
            raise ClaimLedgerError(f"duplicate claim id {claim['id']!r}")
        seen.add(claim["id"])

        status = claim["status"]
        if status not in STATUSES:
            raise ClaimLedgerError(
                f"{where}: unknown status {status!r}; expected one of {sorted(STATUSES)}"
            )
        if not str(claim["required_contrast"]).strip():
            raise ClaimLedgerError(
                f"{where}: required_contrast is empty. A claim with no stated "
                f"contrast can never be settled, only argued about."
            )

        wording = claim["permitted_wording"]
        if status == "established":
            if not wording:
                raise ClaimLedgerError(
                    f"{where}: established but carries no permitted_wording; say "
                    f"what the evidence licenses, or it will be overstated later"
                )
            if not claim["raw_artifact"]:
                raise ClaimLedgerError(
                    f"{where}: established but names no raw_artifact"
                )
        elif wording:
            raise ClaimLedgerError(
                f"{where}: status {status!r} but carries permitted_wording "
                f"{wording!r}. Wording is earned by a contrast, not by intent."
            )

        if status == "withdrawn" and not claim["raw_artifact"]:
            raise ClaimLedgerError(
                f"{where}: withdrawn but names no raw_artifact recording why"
            )
