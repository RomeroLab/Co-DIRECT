
from __future__ import annotations

import re
from dataclasses import dataclass

ROLE_COLUMNS: tuple[str, ...] = ("generated", "context", "target")

_SELECTOR = re.compile(r"^(?P<chain>[^:]+)(?::(?P<start>\d+)-(?P<end>\d+))?$")

class RoleError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class Selector:

    chain: str
    start: int | None = None
    end: int | None = None

    @classmethod
    def parse(cls, text: str) -> "Selector":
        text = (text or "").strip()
        if not text:
            raise RoleError("empty selector")
        match = _SELECTOR.match(text)
        if match is None:
            raise RoleError(f"malformed selector {text!r}; expected chain or chain:start-end")
        chain = match.group("chain")
        if match.group("start") is None:
            return cls(chain=chain)
        start, end = int(match.group("start")), int(match.group("end"))
        if start > end:
            raise RoleError(f"inverted range in {text!r}")
        return cls(chain=chain, start=start, end=end)

    @property
    def is_whole_chain(self) -> bool:
        return self.start is None

    @property
    def length(self) -> int | None:
        return None if self.is_whole_chain else self.end - self.start + 1

    def as_text(self) -> str:
        if self.is_whole_chain:
            return self.chain
        return f"{self.chain}:{self.start}-{self.end}"

@dataclass(frozen=True, slots=True)
class RoleSpec:

    generated: str
    context: str
    target: str

    def as_columns(self) -> dict[str, str]:
        return {"generated": self.generated, "context": self.context, "target": self.target}

def binder_roles(row) -> RoleSpec:

    binder = _text(row, "binder_chain")
    targets = [c for c in _text(row, "target_chains").split(",") if c]
    if not binder:
        raise RoleError("no binder chain named")
    if not targets:
        raise RoleError("no target chain named")
    if binder in targets:
        raise RoleError(f"chain {binder!r} is both generated and target")
    return RoleSpec(generated=binder, context=binder, target=",".join(targets))

def ame_roles(row) -> RoleSpec:

    from dive.data.plinder import receptor_chains_of

    complex_name = _text(row, "complex_name")
    if not complex_name:
        raise RoleError("no complex_name to read roles from")
    try:
        receptor = receptor_chains_of(complex_name)
    except ValueError as error:
        raise RoleError(str(error)) from error
    if not receptor:
        raise RoleError(f"{complex_name} names no receptor chain")
    generated = ",".join(receptor)
    return RoleSpec(generated=generated, context=generated, target="")

def antibody_roles(row) -> RoleSpec:

    heavy_chain = _text(row, "Hchain") or "H"
    light_chain = _text(row, "Lchain") or "L"
    heavy_sequence = _text(row, "Hseq")
    loop = _text(row, "CDRH3")

    if not heavy_sequence or not loop:
        raise RoleError("heavy chain sequence or CDR-H3 is absent")

    occurrences = heavy_sequence.count(loop)
    if occurrences == 0:
        raise RoleError(f"CDR-H3 {loop!r} not found in the heavy chain sequence")
    if occurrences > 1:
        raise RoleError(
            f"CDR-H3 {loop!r} appears {occurrences} times in the heavy chain; the "
            f"range is ambiguous and guessing would place the mask wrongly"
        )

    start = heavy_sequence.index(loop)
    generated = Selector(chain=heavy_chain, start=start, end=start + len(loop) - 1)

    antigens = [c for c in _text(row, "agchains").split(",") if c]
    return RoleSpec(
        generated=generated.as_text(),
        context=f"{heavy_chain},{light_chain}",
        target=",".join(antigens),
    )

def _text(row, key: str) -> str:

    value = row.get(key) if hasattr(row, "get") else getattr(row, key, None)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text

_ROLE_MAP: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {

    "binder": (("binder",), ("target",)),

    "ame": (("scaffold",), ()),
    "antibody": (("heavy", "light"), ("antigen",)),
}

def roles_from_example(example) -> RoleSpec:

    family = str(example.family)
    if family not in _ROLE_MAP:
        raise RoleError(f"no role mapping for family {family!r}")
    designed_roles, conditioning_roles = _ROLE_MAP[family]

    designed = [c for c in example.chains if c.role in designed_roles]
    conditioning = [c.chain_id for c in example.chains if c.role in conditioning_roles]
    if not designed:
        raise RoleError(f"{example.example_id} has no chain to generate")

    context = ",".join(c.chain_id for c in designed)

    if family == "antibody":

        heavy = next((c for c in designed if c.role == "heavy"), None)
        loop = example.metadata.get("cdr_h3_sequence", "")
        if heavy is None or not loop:
            raise RoleError(f"{example.example_id} has no heavy chain or no CDR-H3")
        generated = _locate(heavy.chain_id, heavy.sequence, loop, example.example_id)
    else:

        generated = context

    return RoleSpec(generated=generated, context=context, target=",".join(conditioning))

def _locate(chain_id: str, sequence: str, loop: str, example_id: str) -> str:
    occurrences = sequence.count(loop)
    if occurrences == 0:
        raise RoleError(f"{example_id}: CDR-H3 not found in its heavy chain")
    if occurrences > 1:
        raise RoleError(
            f"{example_id}: CDR-H3 appears {occurrences} times; the range is "
            f"ambiguous and guessing would place the mask wrongly"
        )
    start = sequence.index(loop)
    return Selector(chain=chain_id, start=start, end=start + len(loop) - 1).as_text()
