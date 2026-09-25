
from __future__ import annotations

class BankError(RuntimeError):
    pass

REQUIRED_KEYS = ("native", "source", "batch")

def check_pair(payload, *, name: str = "<pair>") -> None:

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise BankError(
            f"{name} carries no {', '.join(missing)}. A pair that does not carry "
            f"the batch its source was generated against cannot be reconstructed: "
            f"re-deriving it from a loader index returns a different crop, and "
            f"every shape check downstream still passes."
        )
    if not payload["batch"]:
        raise BankError(f"{name} carries an empty batch")

def check_bank(payloads, *, names=None) -> int:

    payloads = list(payloads)
    if not payloads:
        raise BankError("no pair was supplied")
    names = list(names) if names is not None else [f"pair {i}" for i in range(len(payloads))]
    for payload, name in zip(payloads, names):
        check_pair(payload, name=name)
    return len(payloads)
