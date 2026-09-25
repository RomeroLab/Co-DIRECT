
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

CAMPAIGN_SEEDS: tuple[int, ...] = (5, 6, 7)

_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

class CampaignPlanError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class Assignment:

    index: int
    target: str
    seed: int
    run_id: str

    def to_dict(self) -> dict[str, object]:

        return {
            "index": self.index,
            "target": self.target,
            "seed": self.seed,
            "run_id": self.run_id,
        }

def load_target_names(path: Path | str) -> tuple[str, ...]:

    raw = yaml.safe_load(Path(path).read_text()) or {}
    if "target_dict_cfg" not in raw:
        raise CampaignPlanError(f"{path} does not carry a 'target_dict_cfg' mapping")
    entries = raw["target_dict_cfg"] or {}
    if not entries:
        raise CampaignPlanError(f"{path} declares no targets under 'target_dict_cfg'")
    return tuple(sorted(str(name) for name in entries))

def build_assignments(
    targets: Sequence[str], seeds: Sequence[int], *, campaign_id: str
) -> tuple[Assignment, ...]:

    if not _CAMPAIGN_ID.match(campaign_id or ""):
        raise CampaignPlanError(
            f"campaign_id must be a non-empty identifier without spaces, observed {campaign_id!r}"
        )
    if not targets:
        raise CampaignPlanError("a campaign needs at least one target")
    if not seeds:
        raise CampaignPlanError("a campaign needs at least one seed")
    if len(set(targets)) != len(targets):
        raise CampaignPlanError("duplicate target in the campaign plan")
    if len(set(seeds)) != len(seeds):
        raise CampaignPlanError("duplicate seed in the campaign plan")

    total = len(targets) * len(seeds)
    width = max(3, len(str(total - 1)))
    return tuple(
        Assignment(
            index=index,
            target=targets[index // len(seeds)],
            seed=int(seeds[index % len(seeds)]),
            run_id=f"{campaign_id}-{index:0{width}d}",
        )
        for index in range(total)
    )

EXCLUDED_TARGETS: Mapping[str, str] = {
    "31_IL7RA_FIX": (
        "3di3_cropped_fixed.pdb carries chain A renumbered from 1, while the "
        "pinned config selects B17-209: 'No atoms found for selection: B/*/17'"
    ),
    "32_PDL1_ALPHA_FIX": (
        "5o45_cropped_fixed.pdb is renumbered from 1, while the pinned config "
        "selects A17-132: 'No atoms found for selection: A/*/117'"
    ),
    "38_TNFalpha_FIX": (
        "1tnf_cropped_fixed.pdb is a zero-byte file: 'The file has 0 models, "
        "the given model 1 does not exist'"
    ),
}

@dataclass(frozen=True, slots=True)
class CampaignEvidence:

    shard_paths: tuple[Path, ...]
    blockers: tuple[str, ...]
    manifests: Mapping[str, Mapping[str, object]]
    excluded_targets: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:

        return {
            "shards": [str(path) for path in self.shard_paths],
            "shard_count": len(self.shard_paths),
            "blockers": list(self.blockers),
            "complete": not self.blockers,
            "excluded_targets": {
                target: EXCLUDED_TARGETS.get(target, "excluded by the caller")
                for target in self.excluded_targets
            },
        }

def verify_campaign_evidence(
    plan: Sequence[Assignment],
    manifest_root: Path | str,
    *,
    expected_steps: int,
    excluded_targets: Sequence[str] = (),
) -> CampaignEvidence:

    root = Path(manifest_root)
    blockers: list[str] = []
    shard_paths: list[Path] = []
    manifests: dict[str, Mapping[str, object]] = {}

    planned = {assignment.target for assignment in plan}
    unknown = sorted(set(excluded_targets) - planned)
    if unknown:
        raise CampaignPlanError(
            "excluded target not in the campaign plan: " + ", ".join(unknown)
        )
    excluded = set(excluded_targets)

    for assignment in plan:
        if assignment.target in excluded:
            continue
        label = f"{assignment.run_id} ({assignment.target} seed {assignment.seed})"
        manifest_path = root / f"{assignment.run_id}.json"
        if not manifest_path.exists():
            blockers.append(f"{label}: manifest missing at {manifest_path}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            blockers.append(f"{label}: manifest unreadable: {error}")
            continue
        manifests[assignment.run_id] = manifest

        if manifest.get("target") != assignment.target:
            blockers.append(
                f"{label}: manifest names target {manifest.get('target')!r}"
            )
        if int(manifest.get("seed", -1)) != assignment.seed:
            blockers.append(f"{label}: manifest names seed {manifest.get('seed')!r}")
        if manifest.get("probe_enabled") is not True:
            blockers.append(f"{label}: probe was not enabled")
        if manifest.get("denoiser_calls_match") is not True:
            blockers.append(
                f"{label}: denoiser call count {manifest.get('denoiser_calls')!r} "
                f"!= expected {manifest.get('expected_denoiser_calls')!r}"
            )

        shard = manifest.get("shard")
        if not isinstance(shard, dict):
            blockers.append(f"{label}: manifest records no shard")
            continue
        if int(shard.get("steps", -1)) != expected_steps:
            blockers.append(
                f"{label}: shard recorded {shard.get('steps')!r} steps, "
                f"expected {expected_steps}"
            )

        shard_path = Path(str(shard.get("path", "")))
        if not shard_path.is_file():
            blockers.append(f"{label}: shard file missing at {shard_path}")
            continue

        observed = _sha256(shard_path)
        if observed != shard.get("sha256"):
            blockers.append(
                f"{label}: shard sha256 {observed} != manifest {shard.get('sha256')}"
            )
            continue
        shard_paths.append(shard_path)

    return CampaignEvidence(
        shard_paths=tuple(shard_paths),
        blockers=tuple(blockers),
        manifests=manifests,
        excluded_targets=tuple(sorted(excluded)),
    )

def _sha256(path: Path) -> str:

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def assignment_for(
    index: int, targets: Sequence[str], seeds: Sequence[int], *, campaign_id: str
) -> Assignment:

    plan = build_assignments(targets, seeds, campaign_id=campaign_id)
    if not 0 <= index < len(plan):
        raise CampaignPlanError(
            f"array index {index} is outside the plan's 0..{len(plan) - 1}"
        )
    return plan[index]
