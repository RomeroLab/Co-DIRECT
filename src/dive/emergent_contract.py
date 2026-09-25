
from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dive.signed_value.roots import DENIED_PREFIXES

def _expand_env(value):

    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value

class ContractError(RuntimeError):
    pass

MAX_GPU_COUNT = 4

CONTEMPLATED_LORA_RANKS = (8, 16)

AUTHORIZED_LORA_RANK = 8

REQUIRED_CHECKPOINT_KEYS = frozenset({"common", "autoencoder"})

TASK_SPECIFIC_CHECKPOINT_NAMES = frozenset(
    {
        "complexa_ame.ckpt",
        "complexa_ame_ae.ckpt",
        "complexa_ligand.ckpt",
        "complexa_ligand_ae.ckpt",
    }
)

_SECTIONS = frozenset({"upstream", "checkpoints", "model", "compute", "roots", "backup"})

@dataclass(frozen=True, slots=True)
class EmergentContract:

    upstream_root: Path
    upstream_commit: str
    common_checkpoint: Path
    autoencoder_checkpoint: Path
    architecture_v2: bool
    lora_rank: int
    gpu_count: int
    evidence_root: Path
    bulk_root: Path
    backup_deadline: date

    @classmethod
    def from_yaml(cls, path: Path) -> "EmergentContract":
        import yaml

        loaded = yaml.safe_load(Path(path).read_text())
        return cls.from_mapping(_expand_env(loaded))

    @classmethod
    def from_mapping(cls, config: Mapping) -> "EmergentContract":
        unknown = set(config) - _SECTIONS
        if unknown:
            raise ContractError(
                f"unknown section(s) {sorted(unknown)}; a typo'd section would "
                f"otherwise be ignored and its value silently lost"
            )
        missing = _SECTIONS - set(config)
        if missing:
            raise ContractError(f"missing section(s) {sorted(missing)}")

        checkpoints = cls._checkpoints(config["checkpoints"])
        model = config["model"]
        if not model.get("architecture_v2"):
            raise ContractError(
                "architecture v2 is required for every family; the v1 graph is "
                "never correct here, and `USE_V2_COMPLEXA_ARCH` is read once at "
                "import so a v1 fallback is otherwise silent"
            )

        rank = model["lora_rank"]
        if rank not in CONTEMPLATED_LORA_RANKS:
            raise ContractError(
                f"lora rank {rank} is not contemplated; "
                f"expected one of {list(CONTEMPLATED_LORA_RANKS)}"
            )

        gpu_count = config["compute"]["gpu_count"]
        if gpu_count < 1:
            raise ContractError(f"gpu count must be at least 1, got {gpu_count}")
        if gpu_count > MAX_GPU_COUNT:
            raise ContractError(
                f"gpu count must be at most {MAX_GPU_COUNT} on this shared host, "
                f"got {gpu_count}"
            )

        evidence_root, bulk_root = cls._roots(config["roots"])

        return cls(
            upstream_root=Path(config["upstream"]["root"]),
            upstream_commit=str(config["upstream"]["commit"]),
            common_checkpoint=checkpoints["common"],
            autoencoder_checkpoint=checkpoints["autoencoder"],
            architecture_v2=True,
            lora_rank=rank,
            gpu_count=gpu_count,
            evidence_root=evidence_root,
            bulk_root=bulk_root,
            backup_deadline=_as_date(config["backup"]["deadline"]),
        )

    @staticmethod
    def _checkpoints(section: Mapping) -> dict[str, Path]:
        extra = set(section) - REQUIRED_CHECKPOINT_KEYS
        if extra:
            raise ContractError(
                f"one common checkpoint is shared by every family; refusing the "
                f"per-family override(s) {sorted(extra)}"
            )
        missing = REQUIRED_CHECKPOINT_KEYS - set(section)
        if missing:
            raise ContractError(f"missing checkpoint(s) {sorted(missing)}")

        resolved = {key: Path(section[key]) for key in REQUIRED_CHECKPOINT_KEYS}
        for key, path in resolved.items():
            if path.name in TASK_SPECIFIC_CHECKPOINT_NAMES:
                raise ContractError(
                    f"task-specific checkpoint {path.name!r} named as {key!r}; "
                    f"the shared run initializes from the common checkpoint only"
                )
        return resolved

    @staticmethod
    def _roots(section: Mapping) -> tuple[Path, Path]:
        missing = {"evidence", "bulk"} - set(section)
        if missing:
            raise ContractError(f"missing root(s) {sorted(missing)}")

        evidence_root = Path(section["evidence"])
        bulk_root = Path(section["bulk"])

        for denied in DENIED_PREFIXES:
            if evidence_root == denied or denied in evidence_root.parents:
                raise ContractError(
                    f"evidence root {evidence_root} is under the purgeable prefix "
                    f"{denied}; terminal evidence must survive a purge"
                )

        if evidence_root == bulk_root:
            raise ContractError(
                "evidence and bulk roots must be distinct directories; collapsing "
                "them defeats the purgeable/terminal distinction they encode"
            )
        return evidence_root, bulk_root

    def assert_current(self) -> None:

        self._assert_upstream()

        for role, path in (
            ("common checkpoint", self.common_checkpoint),
            ("autoencoder checkpoint", self.autoencoder_checkpoint),
        ):
            if not path.is_file():
                raise ContractError(f"{role} is missing: {path}")

        for role, root in (("evidence", self.evidence_root), ("bulk", self.bulk_root)):
            if root.exists() and not root.is_dir():
                raise ContractError(f"{role} root exists and is not a directory: {root}")

        if (
            not self.architecture_v2
            or self.lora_rank != AUTHORIZED_LORA_RANK
            or self.gpu_count > MAX_GPU_COUNT
        ):
            raise ContractError("emergent leadership resource contract changed")

    def _assert_upstream(self) -> None:
        if not (self.upstream_root / ".git").exists():
            raise ContractError(f"upstream root is not a git checkout: {self.upstream_root}")

        head = _git(self.upstream_root, "rev-parse", "HEAD")
        if head != self.upstream_commit:
            raise ContractError(
                f"upstream commit is {head}, contract pins {self.upstream_commit}"
            )

        status = _git(self.upstream_root, "status", "--porcelain")
        if status:
            raise ContractError(
                f"upstream checkout is dirty; it must never be edited or written "
                f"to in place:\n{status}"
            )

    def as_provenance(self) -> dict[str, object]:

        return {
            "upstream_root": str(self.upstream_root),
            "upstream_commit": self.upstream_commit,
            "common_checkpoint": str(self.common_checkpoint),
            "autoencoder_checkpoint": str(self.autoencoder_checkpoint),
            "architecture_v2": self.architecture_v2,
            "lora_rank": self.lora_rank,
            "gpu_count": self.gpu_count,
            "max_gpu_count": MAX_GPU_COUNT,
            "evidence_root": str(self.evidence_root),
            "bulk_root": str(self.bulk_root),
            "backup_deadline": self.backup_deadline.isoformat(),
        }

def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))

def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ContractError(f"git {' '.join(args)} failed in {root}: {result.stderr.strip()}")
    return result.stdout.strip()
