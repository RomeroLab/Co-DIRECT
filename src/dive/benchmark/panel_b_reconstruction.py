
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat as stat_module
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dive.benchmark.contracts import ArtifactIdentity
from dive.benchmark.paper_baseline import panel_b_citation_allowed
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_BULK_ROOT,
    EMERGENT_EVIDENCE_ROOT,
)
from dive.training.preflight import canonical_json_bytes

PANEL_B_ARM = "base_common_native_v1"
REPLICA_ACCOUNTING = "64_generated_replica_pdbs"
RECORDED_SUMMARY_SHA256 = (
    "1815e7c75329e0a98b03f6424680e7ba1321c952e7e1e238947a8e07c0c8f471"
)
N_TARGETS = 4
N_LENGTH_SAMPLES = 8
N_REPLICAS = 2
N_CELLS = N_TARGETS * N_LENGTH_SAMPLES * N_REPLICAS
TARGET_RUNS: tuple[tuple[str, str], ...] = (
    ("01_PD1", "track-f-f0-native-binder-panel-pd1-20260827a"),
    ("02_PDL1", "track-f-f0-native-binder-panel-pdl1-20260827a"),
    ("04_IFNAR2", "track-f-f0-native-binder-panel-ifnar2-20260827a"),
    ("33_TrkA", "track-f-f0-native-binder-panel-trka-20260827a"),
)
TARGET_IDS: tuple[str, ...] = tuple(target_id for target_id, _run in TARGET_RUNS)
RUN_BY_TARGET = {target_id: run_id for target_id, run_id in TARGET_RUNS}
SUMMARY_RUN = "track-f-f0-native-binder-panel-summary-20260827a"
_PAPER_PREFIX = "paper_quality_baseline"
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_PDB_NAME = re.compile(
    r"^job_(?P<job>\d+)_n_(?P<length>\d+)_id_(?P<sample_id>\d+)"
    r"_bon_orig0_r(?P<replica>\d+)\.pdb$"
)
_SOURCE_ROLES: tuple[tuple[str, str, str], ...] = (
    *(
        item
        for target_id, run_id in TARGET_RUNS
        for item in (
            (f"target_evidence:{target_id}", "evidence", run_id),
            (f"target_bulk:{target_id}", "bulk", run_id),
        )
    ),
    ("summary", "evidence", SUMMARY_RUN),
)

class PanelBReconstructionError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class PanelBConstituent:
    target_id: str
    length: int
    replica: int
    sample_id: int
    pdb_path: str
    pdb_sha256: str
    complex_i_pAE: float
    complex_pLDDT: float
    binder_scRMSD_ca: float
    target_conditioned_success: bool

@dataclass(frozen=True, slots=True)
class PanelBSourceRole:
    role: str
    run_id: str
    path: Path
    present: bool

@dataclass(frozen=True, slots=True)
class PanelBProjection:
    run_id: str
    destination_evidence: Path
    destination_bulk: Path
    source_roles: tuple[PanelBSourceRole, ...]
    missing_roles: tuple[str, ...]
    executed: bool

@dataclass(frozen=True, slots=True)
class PanelBReconstruction:
    arm: str
    plan_cells: int
    raw_pdbs: int
    constituents: bool
    replica_accounting: str
    n_cells: int
    n_successes: int
    n_failed: int
    retries: int
    rebuilt_summary_sha256: str
    recorded_summary_sha256: str
    citation_allowed: bool
    hash_inventory: tuple[ArtifactIdentity, ...]
    constituents_rows: tuple[PanelBConstituent, ...]
    failures: tuple[str, ...]

def native_binder_success(
    complex_i_pAE: float, complex_pLDDT: float, binder_scRMSD_ca: float
) -> bool:
    return (
        float(complex_i_pAE) * 31 <= 7
        and float(complex_pLDDT) >= 0.9
        and float(binder_scRMSD_ca) < 1.5
    )

def project_native_binder_panel(
    *,
    run_id: str,
    evidence_source: Path,
    bulk_source: Path,
) -> PanelBProjection:
    _validate_run_id(run_id)
    evidence_source = Path(evidence_source)
    bulk_source = Path(bulk_source)
    destination_evidence = EMERGENT_EVIDENCE_ROOT / _PAPER_PREFIX / run_id
    destination_bulk = EMERGENT_BULK_ROOT / _PAPER_PREFIX / run_id
    _refuse_denied(destination_evidence)
    _refuse_denied(destination_bulk)
    roles = []
    missing: list[str] = []
    for role, kind, named_run in _SOURCE_ROLES:
        root = evidence_source if kind == "evidence" else bulk_source
        path = root / named_run
        present = _is_dir_nofollow(path)
        roles.append(
            PanelBSourceRole(role=role, run_id=named_run, path=path, present=present)
        )
        if not present:
            missing.append(role)
    return PanelBProjection(
        run_id=run_id,
        destination_evidence=destination_evidence,
        destination_bulk=destination_bulk,
        source_roles=tuple(roles),
        missing_roles=tuple(missing),
        executed=False,
    )

def reconstruct_native_binder_panel(
    *,
    evidence_source: Path,
    bulk_source: Path,
    recorded_summary_sha256: str | None = None,
) -> PanelBReconstruction:
    evidence_source = Path(evidence_source)
    bulk_source = Path(bulk_source)
    summary_path = evidence_source / SUMMARY_RUN / "summary.json"
    pdbs = _collect_pdbs(bulk_source)
    if not pdbs:
        if summary_path.is_file():
            raise PanelBReconstructionError(
                "summary-only evidence cannot authenticate raw pdbs"
            )
        raise PanelBReconstructionError("missing raw pdb files")
    if len(pdbs) == 32:
        raise PanelBReconstructionError("32 winners are not 64_generated_replica_pdbs")
    if len(pdbs) < N_CELLS:
        raise PanelBReconstructionError(
            f"missing generated replica pdbs: found {len(pdbs)} expected {N_CELLS}"
        )
    if len(pdbs) > N_CELLS:
        raise PanelBReconstructionError(
            f"extra generated replica pdbs: found {len(pdbs)} expected {N_CELLS}"
        )

    recorded_raw = _read_regular_bytes(summary_path, "recorded summary")
    recorded_digest = hashlib.sha256(recorded_raw).hexdigest()
    if (
        recorded_summary_sha256 is not None
        and recorded_digest != recorded_summary_sha256
    ):
        raise PanelBReconstructionError("recorded summary hash mismatch")
    recorded = _load_json(recorded_raw, "recorded summary")
    _assert_recorded_targets(recorded)
    recorded_request_ids = _recorded_request_ids(recorded)

    inventory: list[ArtifactIdentity] = [
        ArtifactIdentity(str(summary_path), recorded_digest, len(recorded_raw))
    ]
    summary_terminal = evidence_source / SUMMARY_RUN / "terminal.json"
    if summary_terminal.exists():
        inventory.append(_hash_artifact(summary_terminal, "summary terminal"))

    constituents: list[PanelBConstituent] = []
    requests: list[dict[str, object]] = []
    assets: list[dict[str, object]] = []
    retries = 0
    plan_cells = 0
    for target_id, run_id in TARGET_RUNS:
        evidence_run = evidence_source / run_id
        bulk_run = bulk_source / run_id
        attempt_path = evidence_run / "attempt.json"
        terminal_path = evidence_run / "terminal.json"
        telemetry_path = evidence_run / "telemetry.jsonl"
        rewards_path = bulk_run / "rewards_search_binder_local_pipeline_0.csv"
        timing_path = bulk_run / "timing_0.csv"
        attempt_raw = _read_regular_bytes(attempt_path, f"{run_id} plan")
        attempt = _load_json(attempt_raw, f"{run_id} plan")
        protocol = attempt.get("protocol")
        if not isinstance(protocol, Mapping):
            raise PanelBReconstructionError(f"{run_id} plan is missing protocol")
        if protocol.get("target_id") != target_id:
            raise PanelBReconstructionError(f"target drift in {run_id} plan")
        if protocol.get("samples") != N_LENGTH_SAMPLES:
            raise PanelBReconstructionError(f"length sample drift in {run_id} plan")
        if protocol.get("replicas") != N_REPLICAS:
            raise PanelBReconstructionError(
                f"{run_id} plan replicas are not {N_REPLICAS}"
            )
        plan_cells += int(protocol["samples"]) * int(protocol["replicas"])
        inventory.append(_identity_from_bytes(attempt_path, attempt_raw))
        hydra_overrides = evidence_run / "hydra" / ".hydra" / "overrides.yaml"
        if hydra_overrides.exists() or hydra_overrides.is_symlink():
            inventory.append(_hash_artifact(hydra_overrides, f"{run_id} hydra plan"))
        terminal_raw = _read_regular_bytes(terminal_path, f"{run_id} progress terminal")
        terminal = _load_json(terminal_raw, f"{run_id} progress terminal")
        inventory.append(_identity_from_bytes(terminal_path, terminal_raw))
        declared = _declared_outputs(terminal)
        if telemetry_path.exists() or telemetry_path.is_symlink():
            telemetry_raw = _read_regular_bytes(
                telemetry_path, f"{run_id} progress telemetry"
            )
            inventory.append(_identity_from_bytes(telemetry_path, telemetry_raw))
            retries += _retry_count(telemetry_raw)
            expected_telemetry = terminal.get("telemetry_sha256")
            if (
                isinstance(expected_telemetry, str)
                and hashlib.sha256(telemetry_raw).hexdigest() != expected_telemetry
            ):
                raise PanelBReconstructionError(f"{run_id} telemetry identity drifted")
        expected_attempt = terminal.get("attempt_sha256")
        if (
            isinstance(expected_attempt, str)
            and hashlib.sha256(attempt_raw).hexdigest() != expected_attempt
        ):
            raise PanelBReconstructionError(f"{run_id} plan identity drifted")
        rewards_raw = _read_regular_bytes(rewards_path, f"{run_id} evaluator")
        inventory.append(_identity_from_bytes(rewards_path, rewards_raw))
        _assert_declared_hash(declared, rewards_path, rewards_raw, run_id)
        reward_order, rewards = _load_rewards(rewards_raw, run_id)
        if timing_path.exists() or timing_path.is_symlink():
            timing_raw = _read_regular_bytes(timing_path, f"{run_id} progress timing")
            inventory.append(_identity_from_bytes(timing_path, timing_raw))
            _assert_declared_hash(declared, timing_path, timing_raw, run_id)
        run_pdbs = [path for path in pdbs if _is_relative_to(path, bulk_run)]
        run_rows = _constituents_for_run(
            target_id=target_id,
            run_id=run_id,
            pdbs=run_pdbs,
            reward_order=reward_order,
            rewards=rewards,
            declared=declared,
        )
        constituents.extend(run_rows)
        assets.extend(
            [
                {
                    "path": str(rewards_path),
                    "sha256": hashlib.sha256(rewards_raw).hexdigest(),
                },
                {
                    "path": str(terminal_path),
                    "sha256": hashlib.sha256(terminal_raw).hexdigest(),
                },
                {
                    "path": str(attempt_path),
                    "sha256": hashlib.sha256(attempt_raw).hexdigest(),
                },
            ]
        )
        for row in run_rows:
            try:
                request_id = recorded_request_ids[row.pdb_path]
            except KeyError as error:
                raise PanelBReconstructionError(
                    f"missing recorded request identity for {row.pdb_path}"
                ) from error
            requests.append(
                {
                    "binder_scRMSD_ca": row.binder_scRMSD_ca,
                    "complex_i_pAE": row.complex_i_pAE,
                    "complex_pLDDT": row.complex_pLDDT,
                    "pdb_path": row.pdb_path,
                    "request_id": request_id,
                    "run_id": run_id,
                    "target_conditioned_success": row.target_conditioned_success,
                    "target_id": row.target_id,
                }
            )

    if plan_cells != N_CELLS:
        raise PanelBReconstructionError(f"plan cells {plan_cells} are not {N_CELLS}")
    rebuilt_paths = {row.pdb_path for row in constituents}
    if set(recorded_request_ids) != rebuilt_paths:
        raise PanelBReconstructionError(
            "recorded request identities do not match replica pdb paths"
        )
    _assert_replica_accounting(constituents)
    hashes = [row.pdb_sha256 for row in constituents]
    if len(set(hashes)) != N_CELLS:
        raise PanelBReconstructionError("duplicate generated replica pdb hashes")
    n_successes = sum(1 for row in constituents if row.target_conditioned_success)
    n_failed = N_CELLS - n_successes
    rebuilt = _rebuild_summary(
        assets=assets,
        requests=requests,
        n_successes=n_successes,
        code_commit=str(recorded.get("code_commit") or ""),
    )
    rebuilt_raw = canonical_json_bytes(rebuilt)
    rebuilt_digest = hashlib.sha256(rebuilt_raw).hexdigest()
    if rebuilt_raw != recorded_raw:
        raise PanelBReconstructionError("rebuilt summary does not match recorded bytes")
    citation = (
        panel_b_citation_allowed(
            {
                "plan_cells": plan_cells,
                "raw_pdbs": len(constituents),
                "constituents": True,
                "replica_accounting": REPLICA_ACCOUNTING,
                "n_cells": N_CELLS,
                "rebuilt_summary_sha256": rebuilt_digest,
            }
        )
        and rebuilt_digest == recorded_digest
    )
    failures = tuple(
        row.pdb_path for row in constituents if not row.target_conditioned_success
    )
    return PanelBReconstruction(
        arm=PANEL_B_ARM,
        plan_cells=plan_cells,
        raw_pdbs=len(constituents),
        constituents=True,
        replica_accounting=REPLICA_ACCOUNTING,
        n_cells=N_CELLS,
        n_successes=n_successes,
        n_failed=n_failed,
        retries=retries,
        rebuilt_summary_sha256=rebuilt_digest,
        recorded_summary_sha256=recorded_digest,
        citation_allowed=citation,
        hash_inventory=tuple(sorted(inventory, key=lambda item: item.path)),
        constituents_rows=tuple(constituents),
        failures=failures,
    )

def format_projection(projection: PanelBProjection) -> str:
    payload = {
        "destination_bulk": str(projection.destination_bulk),
        "destination_evidence": str(projection.destination_evidence),
        "executed": False,
        "missing_roles": list(projection.missing_roles),
        "run_id": projection.run_id,
        "source_roles": [
            {
                "path": str(role.path),
                "present": role.present,
                "role": role.role,
                "run_id": role.run_id,
                "status": "present" if role.present else "missing",
            }
            for role in projection.source_roles
        ],
    }
    return canonical_json_bytes(payload).decode("utf-8")

def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reconstruct_native_binder_panel")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evidence-source", required=True, type=Path)
    parser.add_argument("--bulk-source", required=True, type=Path)
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        projection = project_native_binder_panel(
            run_id=args.run_id,
            evidence_source=args.evidence_source,
            bulk_source=args.bulk_source,
        )
    except PanelBReconstructionError:
        return 2
    print(format_projection(projection), end="")
    print("nothing executed")
    return 0

def _validate_run_id(run_id: str) -> None:
    if (
        type(run_id) is not str
        or _RUN_ID.fullmatch(run_id) is None
        or not (
            run_id.startswith("paper-baseline-") or run_id.startswith("paper-smoke-")
        )
    ):
        raise PanelBReconstructionError(f"invalid paper baseline run id {run_id}")

def _refuse_denied(path: Path) -> None:
    resolved = Path(path).resolve(strict=False)
    for denied in DENIED_PREFIXES:
        prefix = Path(denied).resolve(strict=False)
        if resolved == prefix or prefix in resolved.parents:
            raise PanelBReconstructionError(f"root is denied: {resolved}")

def _is_dir_nofollow(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    return stat_module.S_ISDIR(metadata.st_mode)

def _collect_pdbs(bulk_source: Path) -> list[Path]:
    found: list[Path] = []
    for _target_id, run_id in TARGET_RUNS:
        root = bulk_source / run_id
        if not _is_dir_nofollow(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [
                name for name in dirnames if not Path(dirpath, name).is_symlink()
            ]
            for name in filenames:
                if not name.endswith(".pdb"):
                    continue
                candidate = Path(dirpath) / name
                found.append(candidate)
    return found

def _constituents_for_run(
    *,
    target_id: str,
    run_id: str,
    pdbs: Sequence[Path],
    reward_order: Sequence[str],
    rewards: Mapping[str, tuple[float, float, float]],
    declared: Mapping[str, tuple[str, int]],
) -> list[PanelBConstituent]:
    pdb_by_path = {str(path): path for path in pdbs}
    rows: list[PanelBConstituent] = []
    groups: dict[tuple[str, int, int], set[int]] = defaultdict(set)
    seen_paths: set[str] = set()
    for key in reward_order:
        path = pdb_by_path.get(key)
        if path is None:
            raise PanelBReconstructionError(
                f"evaluator row target drift or extra path in {run_id}"
            )
        if path.is_symlink():
            raise PanelBReconstructionError(f"{path} is a symlink")
        parsed = _parse_pdb_identity(path, run_id)
        length, sample_id, replica = parsed
        groups[(target_id, length, sample_id // 2)].add(replica)
        raw = _read_regular_bytes(path, f"{run_id} pdb")
        digest = hashlib.sha256(raw).hexdigest()
        _assert_declared_hash(declared, path, raw, run_id)
        if key in seen_paths:
            raise PanelBReconstructionError(f"duplicate pdb path {path}")
        seen_paths.add(key)
        ipae, plddt, rmsd = rewards[key]
        rows.append(
            PanelBConstituent(
                target_id=target_id,
                length=length,
                replica=replica,
                sample_id=sample_id,
                pdb_path=key,
                pdb_sha256=digest,
                complex_i_pAE=ipae,
                complex_pLDDT=plddt,
                binder_scRMSD_ca=rmsd,
                target_conditioned_success=native_binder_success(ipae, plddt, rmsd),
            )
        )
    leftover = set(pdb_by_path) - seen_paths
    if leftover:
        raise PanelBReconstructionError(
            f"missing evaluator row for {sorted(leftover)[0]}"
        )
    if len(groups) != N_LENGTH_SAMPLES:
        raise PanelBReconstructionError(
            f"length drift in {run_id}: {len(groups)} length samples"
        )
    for group, replicas in groups.items():
        if replicas != {0, 1}:
            raise PanelBReconstructionError(
                f"length drift in {run_id} sample {group}: replicas {sorted(replicas)}"
            )
    return rows

def _assert_replica_accounting(rows: Sequence[PanelBConstituent]) -> None:
    if len(rows) != N_CELLS:
        raise PanelBReconstructionError(
            f"replica accounting is not {REPLICA_ACCOUNTING}"
        )
    identities = {
        (row.target_id, row.length, row.replica, row.sample_id) for row in rows
    }
    if len(identities) != N_CELLS:
        raise PanelBReconstructionError("duplicate replica identities")
    by_target: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        by_target[row.target_id].add((row.length, row.sample_id // 2))
    if set(by_target) != set(TARGET_IDS):
        raise PanelBReconstructionError("target drift across replica pdbs")
    for target_id, samples in by_target.items():
        if len(samples) != N_LENGTH_SAMPLES:
            raise PanelBReconstructionError(
                f"length drift for {target_id}: {len(samples)} samples"
            )

def _recorded_request_ids(recorded: Mapping[str, object]) -> dict[str, str]:
    requests = recorded.get("requests")
    if not isinstance(requests, list):
        raise PanelBReconstructionError("recorded summary is missing requests")
    identities: dict[str, str] = {}
    for row in requests:
        if not isinstance(row, Mapping):
            raise PanelBReconstructionError("recorded summary request is invalid")
        pdb_path = row.get("pdb_path")
        request_id = row.get("request_id")
        if not isinstance(pdb_path, str) or not isinstance(request_id, str):
            raise PanelBReconstructionError(
                "recorded request identity is missing pdb_path or request_id"
            )
        if pdb_path in identities:
            raise PanelBReconstructionError(
                f"duplicate recorded request identity for {pdb_path}"
            )
        identities[pdb_path] = request_id
    return identities

def _assert_recorded_targets(recorded: Mapping[str, object]) -> None:
    requests = recorded.get("requests")
    if not isinstance(requests, list):
        raise PanelBReconstructionError("recorded summary is missing requests")
    for row in requests:
        if not isinstance(row, Mapping):
            raise PanelBReconstructionError("recorded summary request is invalid")
        target_id = row.get("target_id")
        run_id = row.get("run_id")
        if target_id not in RUN_BY_TARGET:
            raise PanelBReconstructionError(f"target drift: unknown {target_id}")
        if run_id != RUN_BY_TARGET[target_id]:
            raise PanelBReconstructionError(
                f"target drift: {target_id} is not {run_id}"
            )

def _parse_pdb_identity(path: Path, run_id: str) -> tuple[int, int, int]:
    match = _PDB_NAME.fullmatch(path.name)
    if match is None:
        raise PanelBReconstructionError(f"unrecognized replica pdb name {path.name}")
    if path.parent.name != path.stem:
        raise PanelBReconstructionError(f"pdb directory identity drifted: {path}")
    replica = int(match.group("replica"))
    sample_id = int(match.group("sample_id"))
    length = int(match.group("length"))
    if replica not in {0, 1}:
        raise PanelBReconstructionError(f"replica {replica} is not r0/r1")
    if sample_id % 2 != replica:
        raise PanelBReconstructionError(
            f"length/replica identity drifted in {run_id}: {path.name}"
        )
    return length, sample_id, replica

def _load_rewards(
    raw: bytes, run_id: str
) -> tuple[list[str], dict[str, tuple[float, float, float]]]:
    text = io.StringIO(raw.decode("utf-8"))
    reader = csv.DictReader(text)
    order: list[str] = []
    rows: dict[str, tuple[float, float, float]] = {}
    for row in reader:
        pdb_path = row.get("pdb_path")
        if not pdb_path:
            raise PanelBReconstructionError(f"{run_id} evaluator row missing pdb_path")
        if pdb_path in rows:
            raise PanelBReconstructionError(f"duplicate evaluator row for {pdb_path}")
        try:
            ipae = float(row["af2folding_i_pae"])
            plddt = float(row["af2folding_plddt_log"])
            rmsd = float(row["af2folding_rmsd"])
        except (KeyError, TypeError, ValueError) as error:
            raise PanelBReconstructionError(
                f"{run_id} evaluator constituents are incomplete"
            ) from error
        order.append(pdb_path)
        rows[pdb_path] = (ipae, plddt, rmsd)
    return order, rows

def _declared_outputs(terminal: Mapping[str, object]) -> dict[str, tuple[str, int]]:
    outputs = terminal.get("outputs")
    if not isinstance(outputs, list):
        raise PanelBReconstructionError("progress terminal is missing outputs")
    declared: dict[str, tuple[str, int]] = {}
    for item in outputs:
        if not isinstance(item, Mapping):
            raise PanelBReconstructionError("progress terminal output is invalid")
        path = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise PanelBReconstructionError("progress terminal output hash is missing")
        declared[path] = (digest, int(size) if isinstance(size, int) else -1)
    return declared

def _assert_declared_hash(
    declared: Mapping[str, tuple[str, int]],
    path: Path,
    raw: bytes,
    run_id: str,
) -> None:
    key = str(path)
    if key not in declared:
        raise PanelBReconstructionError(f"missing {run_id} hash for {path}")
    digest, size = declared[key]
    live = hashlib.sha256(raw).hexdigest()
    if live != digest or (size != -1 and size != len(raw)):
        raise PanelBReconstructionError(
            f"{path} hash mutated during read against {run_id} progress"
        )

def _retry_count(raw: bytes) -> int:
    count = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise PanelBReconstructionError("truncated progress telemetry") from error
        if isinstance(payload, Mapping) and payload.get("event") == "retry":
            count += 1
    return count

def _rebuild_summary(
    *,
    assets: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    n_successes: int,
    code_commit: str,
) -> dict[str, object]:
    by_target: dict[str, dict[str, object]] = {}
    for target_id in TARGET_IDS:
        rows = [row for row in requests if row["target_id"] == target_id]
        successes = sum(1 for row in rows if row["target_conditioned_success"])
        by_target[target_id] = {
            "complete": len(rows),
            "failed_or_missing": 0,
            "requested": len(rows),
            "success_rate_all_requests": (successes / len(rows)) if rows else 0.0,
            "successes": successes,
        }
    return {
        "assets": list(assets),
        "blind_opened": False,
        "code_commit": code_commit,
        "fold4_opened": False,
        "requests": list(requests),
        "schema": "dive.track_f.native_binder_panel_artifact.v1",
        "status": "complete",
        "summary": {
            "complete": len(requests),
            "failed": 0,
            "requested": len(requests),
            "schema": "dive.track_f.native_binder_panel_summary.v1",
            "success_rate_all_requests": (n_successes / len(requests))
            if requests
            else 0.0,
            "successes": n_successes,
            "targets": by_target,
            "targets_with_success": sum(
                1 for row in by_target.values() if int(row["successes"]) > 0
            ),
        },
    }

def _hash_artifact(path: Path, label: str) -> ArtifactIdentity:
    raw = _read_regular_bytes(path, label)
    return _identity_from_bytes(path, raw)

def _identity_from_bytes(path: Path, raw: bytes) -> ArtifactIdentity:
    return ArtifactIdentity(str(path), hashlib.sha256(raw).hexdigest(), len(raw))

def _load_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PanelBReconstructionError(f"{label} is not JSON") from error
    if not isinstance(payload, dict):
        raise PanelBReconstructionError(f"{label} must be an object")
    return payload

def _read_regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink():
        raise PanelBReconstructionError(f"{label} is a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PanelBReconstructionError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode):
            raise PanelBReconstructionError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or len(raw) != before.st_size
    ):
        raise PanelBReconstructionError(f"{label} identity drifted")
    return raw

def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
