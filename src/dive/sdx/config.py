
from __future__ import annotations

from dataclasses import asdict, dataclass

from dive.sdx.exchange import VARIANTS

SOURCES = ("cavity", "full")

CAVITIES = ("global", "local")

@dataclass(frozen=True)
class SdxConfig:

    deliver_message: bool = True

    variant: str = "full"

    source: str = "cavity"

    relation_weight: float = 1.0

    trains_relation_head: bool = True

    cavity: str = "global"

    query_fraction: float = 0.2

    stride: int = 1

    extra_identity_embedding: bool = False

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(
                f"unknown message variant {self.variant!r}; expected {sorted(VARIANTS)}"
            )
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected {list(SOURCES)}")
        if self.cavity not in CAVITIES:
            raise ValueError(f"unknown cavity {self.cavity!r}; expected {list(CAVITIES)}")
        if not 0.0 < self.query_fraction <= 1.0:
            raise ValueError("query_fraction must be in (0, 1]")
        if self.deliver_message and not self.trains_relation_head:
            raise ValueError(
                "an arm that delivers a message from an untrained relation head "
                "would be delivering noise; refusing to build it"
            )
        if self.stride < 1:
            raise ValueError("stride must be at least 1")

    def as_dict(self) -> dict:
        return asdict(self)
