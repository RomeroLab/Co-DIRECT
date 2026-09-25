
from __future__ import annotations

from dive.codirect_paths import joined

import hashlib
import io
import json
import os
import re
import shlex
import shutil
import stat as stat_module
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dive.signed_value.roots import EMERGENT_EVIDENCE_ROOT
from dive.training.preflight import (
    PreflightError,
    assert_manifest_is_trainable,
    canonical_json_bytes,
    read_terminal_bytes,
    write_create_new_json,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_ROOT = Path(joined('PROTEINA_COMPLEXA_ROOT'))
_PROJECTION_EVIDENCE_ROOT = EMERGENT_EVIDENCE_ROOT
_READ_CHUNK = 1 << 20
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FAMILIES = ("binder", "ame", "antibody")
_PARTITIONS = ("train", "validation")

def _require_sha256(name: str, value: object) -> None:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise PreflightError(f"{name} must be lowercase sha256")

def _require_file_identity(name: str, sha256: str, size_bytes: int) -> None:
    _require_sha256(f"{name} sha256", sha256)
    if type(size_bytes) is not int or size_bytes <= 0:
        raise PreflightError(f"{name} size_bytes must be positive")

@dataclass(frozen=True, slots=True)
class ManifestSpec:

    name: str
    family: str
    partition: str
    sha256: str
    size_bytes: int
    row_count: int
    example_id_hash: str

    def __post_init__(self) -> None:
        expected_name = f"{self.family}_{self.partition}.parquet"
        if self.family not in _FAMILIES or self.partition not in _PARTITIONS:
            raise PreflightError("manifest family/partition is not directional")
        if self.name != expected_name:
            raise PreflightError(
                f"manifest name {self.name!r} does not match {expected_name!r}"
            )
        _require_file_identity(self.name, self.sha256, self.size_bytes)
        if type(self.row_count) is not int or self.row_count <= 0:
            raise PreflightError(f"manifest {self.name} row_count must be positive")
        if _HEX64.fullmatch(self.example_id_hash) is None:
            raise PreflightError(
                f"manifest {self.name} example_id_hash must be lowercase sha256"
            )

    def as_provenance(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class SourceSpec:

    name: str
    kind: str
    path: Path
    sha256: str
    size_bytes: int
    retain_bytes: bool = True

    def __post_init__(self) -> None:
        allowed = {"public_raw", "resolver_cache", "archive", "canonicalizer"}
        if self.kind not in allowed:
            raise PreflightError(f"source {self.name} has unknown kind {self.kind!r}")
        _require_file_identity(self.name, self.sha256, self.size_bytes)

    def identity(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

@dataclass(frozen=True, slots=True)
class ProjectionContract:

    schema_version: int
    upstream_commit: str
    manifests: tuple[ManifestSpec, ...]
    sources: tuple[SourceSpec, ...]
    code: tuple[SourceSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise PreflightError("parent projection schema_version must be exactly 1")
        if re.fullmatch(r"[0-9a-f]{40}", self.upstream_commit) is None:
            raise PreflightError("projection upstream_commit must be lowercase git sha")
        names = tuple(spec.name for spec in self.manifests)
        expected = tuple(
            f"{family}_{partition}.parquet"
            for family in _FAMILIES
            for partition in _PARTITIONS
        )
        if names != expected:
            raise PreflightError(
                "parent projection contract must bind the exact ordered six manifests"
            )
        all_names = [spec.name for spec in (*self.sources, *self.code)]
        if len(all_names) != len(set(all_names)):
            raise PreflightError("projection source/code names must be unique")

    @property
    def identity(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "upstream_commit": self.upstream_commit,
            "manifests": [item.as_provenance() for item in self.manifests],
            "sources": [item.identity() for item in self.sources],
            "code": [item.identity() for item in self.code],
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

@dataclass(frozen=True, slots=True)
class VerifiedManifest:

    spec: ManifestSpec
    example_ids: tuple[str, ...]
    records: tuple[Mapping[str, object], ...]

    def frame(self):

        import pandas as pd

        return pd.DataFrame([dict(row) for row in self.records])

@dataclass(frozen=True, slots=True)
class VerifiedSource:

    spec: SourceSpec
    content: bytes | None

@dataclass(frozen=True, slots=True)
class ReviewedProjectionSpec:

    schema_version: int
    identity: str
    projection_hash: str
    file_sha256: str
    size_bytes: int
    row_count: int
    builder_commit: str
    builder_code_identities: Mapping[str, str]
    contract_identity: str
    manifest_identities: tuple[Mapping[str, object], ...]
    source_identities: Mapping[str, str]
    canonicalizer_identities: Mapping[str, str]

@dataclass(frozen=True, slots=True)
class VerifiedProjectionSpec:

    path: Path
    sha256: str
    size_bytes: int
    spec: ReviewedProjectionSpec

@dataclass(frozen=True, slots=True)
class ProjectionApprovalPolicy:

    identity: str
    completion_sha256: str
    approval_commit: str
    builder_commit: str
    spec_relative_path: Path
    spec_sha256: str

_PROJECTION_APPROVAL_POLICY = ProjectionApprovalPolicy(
    identity="v1",
    completion_sha256=(
        "6c1f171023716d568917e4d4f6341b638b6d4da2ba95c6e1dfaf1202fa92bd6a"
    ),
    approval_commit="8b991d81e913145f28b97b57838fe95e9e602ad3",
    builder_commit="c69ec9bc4ca5c304834fa40d934e912538c959d2",
    spec_relative_path=Path("configs/emergent/directional_parent_projection_v1.json"),
    spec_sha256="64658c5f0ec1b3365e18a2d59b9c79891d580f6dcce0e286f93ed0c9ff041bf7",
)

_PROJECTION_V2_APPROVAL_POLICY = ProjectionApprovalPolicy(
    identity="v2",
    completion_sha256=(
        "fd133c25406f15fcc0a44e0218a2e8dbd88d9dd25f93062523d56844521154f4"
    ),
    approval_commit="9e851a2d92efbedd209b361941b75374bf79ac16",
    builder_commit="c63362fac4c605fd3c75ccb1fcfadf4066e04ed0",
    spec_relative_path=Path("configs/emergent/directional_parent_projection_v2.json"),
    spec_sha256="9d1101b952b429cf0e51382e3909d966c66fc281fb87bcd41d49202f64a19451",
)

PRODUCTION_CONTRACT = ProjectionContract(
    schema_version=1,
    upstream_commit="32b71ae1d9a8767414eae3921cf35969430db82f",
    manifests=(
        ManifestSpec(
            "binder_train.parquet",
            "binder",
            "train",
            "654757e70493a24608ded8490ddafdc21ae6225fb7e4e88632e974dca2c13935",
            238843,
            11587,
            "dfab4755fe85950c2c2be31e265216e08943894bc3eb8ec1c0ad3bef585746ce",
        ),
        ManifestSpec(
            "binder_validation.parquet",
            "binder",
            "validation",
            "7f44ade388fb35e325d7f8bac1ade149b3314dbeee1382e8cfa72864a159f8a0",
            66571,
            2976,
            "668e673f7837d6ababa453373393e663d1e3ff59421f418287def71d3a907a62",
        ),
        ManifestSpec(
            "ame_train.parquet",
            "ame",
            "train",
            "95549cc990f9e37c088f00d7d9d3359075031ae35189e38339545f4ddc35fc9a",
            101274,
            3808,
            "e606a0a255980562526681e627c70a1c5b0f37b12818bb6d971a4f39068f4f07",
        ),
        ManifestSpec(
            "ame_validation.parquet",
            "ame",
            "validation",
            "18837dadb8c14b1ee0fec1cd2ce2ed321a80d9e381fb304834ffebb1c7ee0dd6",
            99873,
            3705,
            "90f30ea6751f3a739dff132d1da93b1f92a28ad4c9910fb33a0c2910869d41b4",
        ),
        ManifestSpec(
            "antibody_train.parquet",
            "antibody",
            "train",
            "fd2eb30680b4dff2dac33435759711bb06f9755d058978e99004b3d716a32383",
            22331,
            784,
            "32cd0d013c064a7b0e706faf13a77ec804bf58244f9188db7dd69c364e841e72",
        ),
        ManifestSpec(
            "antibody_validation.parquet",
            "antibody",
            "validation",
            "137f5673d6e2c0ee89fbf0870f87f8cb35e1ed6c5ac9559d622038258b2d1fa7",
            11145,
            279,
            "8aa35a57298ad91f8aeff5317c956b2a111eb59752f4e29f22c06c0de22ebe4c",
        ),
    ),
    sources=(
        SourceSpec(
            "binder_metadata",
            "public_raw",
            Path(joined('PROTEINA_COMPLEXA_ROOT', 'assets', 'data', 'pdb_multimer.csv')),
            "e98bcea4c4f52c0860b99ffefbeb306ce556d858e56ab75b69360ccd9326b757",
            58894244,
        ),
        SourceSpec(
            "ame_metadata",
            "public_raw",
            Path(
                joined('PROTEINA_COMPLEXA_ROOT', 'assets', 'data', 'plinder_valid_dataset.csv')
            ),
            "a9401cae21be0cbe3e17c4a656f5def49a3347e315a9ab5e0591466bb9578b0a",
            23434438,
        ),
        SourceSpec(
            "ame_resolver_cache",
            "resolver_cache",
            Path(
                joined('CODIRECT_STRUCTURE_ROOT', 'sources', 'plinder-2024-06', 'resolved_systems.parquet')
            ),
            "085928368204c61b7d405ecf3e2d2be724927ece5dff1c1b77a943534ac5e63c",
            24476351,
        ),
        SourceSpec(
            "antibody_archive",
            "archive",
            Path(joined('CODIRECT_STRUCTURE_ROOT', 'sources', 'splits.tar.gz')),
            "54e3b9cdae5f5ff57a4210866f87e86686203556eea6a033200ff55c0f81e2b7",
            876381859,
            retain_bytes=False,
        ),
        SourceSpec(
            "antibody_split_table",
            "public_raw",
            Path(
                joined('CODIRECT_STRUCTURE_ROOT', 'sources', 'sabdab2-v0.1.0', 'splits_final', 'abag_split.csv')
            ),
            "01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb",
            140769855,
        ),
    ),
    code=(
        SourceSpec(
            "canonicalize_sources",
            "canonicalizer",
            _REPO_ROOT / "src/dive/data/sources.py",
            "6622339861eeb209e7ed06ca664465f46fce2f1dc82dff80c4ba55b686f95e96",
            11492,
        ),
        SourceSpec(
            "canonicalize_sabdab2",
            "canonicalizer",
            _REPO_ROOT / "src/dive/data/sabdab2.py",
            "61fd704b87299e3f7e5d7fd61b682d8021bc77d900d59b130115506827435142",
            17942,
        ),
        SourceSpec(
            "canonicalize_corpus",
            "canonicalizer",
            _REPO_ROOT / "src/dive/data/corpus.py",
            "a5518d5f5213bc2d71cdec4484c6c08e314b97671ad8a8a43e06382eb201d18b",
            6557,
        ),
        SourceSpec(
            "data_config",
            "canonicalizer",
            _REPO_ROOT / "configs/emergent/data.yaml",
            "bb043d6da528f92b9587785f791b653ac73a7dd5107469bbe6f65bb4d96f3350",
            5408,
        ),
    ),
)

PRODUCTION_CONTRACT_V2 = ProjectionContract(
    schema_version=PRODUCTION_CONTRACT.schema_version,
    upstream_commit=PRODUCTION_CONTRACT.upstream_commit,
    manifests=(
    ManifestSpec(
        "binder_train.parquet", "binder", "train",
        "15932aa1285eb5f37bf501f4ac38943ef6f56fbbe03be47af49e35c72d0a1c25",
        257454, 12803,
        "d707b822a0da86672005b06135d0a2ec6bf2564bd6e270f119b1ea6328abc2e0",
    ),
    ManifestSpec(
        "binder_validation.parquet", "binder", "validation",
        "02e4783ce2c8117fc7e8dad534ed96e9854e1320d17c7d2925de53798763d303",
        70445, 3226,
        "59d0aa588e36280b97230c93b8a9aa15c846e632a7ef266db0c93ed4c60279e7",
    ),
    ManifestSpec(
        "ame_train.parquet", "ame", "train",
        "1aeaf7eebec08cc5eae91f64757c3dac09602e809a0a3ed07a4d7f1d3ff1a22e",
        114626, 4780,
        "6febaa587da0094e27751e37d918d199de53aca59eceb5199df1677f3a79308b",
    ),
    ManifestSpec(
        "ame_validation.parquet", "ame", "validation",
        "173ddb3b4691904e7b11c3dfeee9bc25bb341fbb5b7fd56f2afc5b8aedd1d589",
        109103, 4435,
        "907f714ab5337630fab829f190447b5799aec8669f8e4b03dd918523916c808e",
    ),
    ManifestSpec(
        "antibody_train.parquet", "antibody", "train",
        "b778b51f87fda1c20473a11f923d5da010ae0f3a741eadbc062374b2213e7659",
        21978, 784,
        "92c4127ade5a74f4ffd75c80cb8335dc893339bd1a6fb65e96816468d9b73686",
    ),
    ManifestSpec(
        "antibody_validation.parquet", "antibody", "validation",
        "faccd821c755d40bbcfd11f176b9fcebba9c7919bc988bc166b6038ff38b6a1e",
        11170, 281,
        "ebdcc0aa24cb8129f47edbfab14bee9732f5dfe693bd4e81722c4210b07f4694",
    ),
    ),
    sources=PRODUCTION_CONTRACT.sources,
    code=PRODUCTION_CONTRACT.code,
)

def load_reviewed_projection_spec(
    path: Path | None = None,
    *,
    contract: ProjectionContract = PRODUCTION_CONTRACT,
) -> VerifiedProjectionSpec:

    raw: bytes | None = None
    policy = (
        _PROJECTION_APPROVAL_POLICY
        if contract is PRODUCTION_CONTRACT
        else _PROJECTION_V2_APPROVAL_POLICY
        if contract is PRODUCTION_CONTRACT_V2
        else None
    )
    if policy is not None:
        _verify_projection_approval_commits(_REPO_ROOT, policy)
        expected_path = _REPO_ROOT / policy.spec_relative_path
        if path is not None and Path(os.path.abspath(path)) != Path(
            os.path.abspath(expected_path)
        ):
            raise PreflightError(
                "production reviewed spec must use its fixed repo-relative path"
            )
        approval_raw = _read_git_blob(
            _REPO_ROOT, policy.approval_commit, policy.spec_relative_path
        )
        if hashlib.sha256(approval_raw).hexdigest() != policy.spec_sha256:
            raise PreflightError("approval commit spec sha256 differs from policy")
        current_raw = _read_bounded_regular_bytes(expected_path, maximum=64 * 1024)
        if hashlib.sha256(current_raw).hexdigest() != policy.spec_sha256:
            raise PreflightError(
                "current repo-relative spec sha256 differs from policy"
            )
        if current_raw != approval_raw:
            raise PreflightError("current spec bytes differ from approval commit blob")
        path = expected_path
        raw = current_raw
    elif path is None:
        raise PreflightError("nonproduction reviewed spec requires an explicit path")
    if raw is None:
        raw = _read_bounded_regular_bytes(Path(path), maximum=64 * 1024)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(
            "reviewed parent projection spec is invalid JSON"
        ) from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise PreflightError("reviewed parent projection spec is not canonical JSON")
    required = {
        "schema_version",
        "identity",
        "projection_hash",
        "file_sha256",
        "size_bytes",
        "row_count",
        "builder_commit",
        "builder_code_identities",
        "contract_identity",
        "manifest_identities",
        "source_identities",
        "canonicalizer_identities",
    }
    if set(payload) != required or payload["schema_version"] != 1:
        raise PreflightError("reviewed parent projection spec schema is invalid")
    _validate_projection_identity(payload["identity"])
    for field in ("projection_hash", "file_sha256", "contract_identity"):
        _require_sha256(field, payload[field])
    if re.fullmatch(r"[0-9a-f]{40}", str(payload["builder_commit"])) is None:
        raise PreflightError("reviewed projection builder_commit is invalid")
    if type(payload["size_bytes"]) is not int or payload["size_bytes"] <= 0:
        raise PreflightError("reviewed projection size_bytes must be positive")
    if type(payload["row_count"]) is not int or payload["row_count"] <= 0:
        raise PreflightError("reviewed projection row_count must be positive")
    expected_manifests = [item.as_provenance() for item in contract.manifests]
    expected_sources = {item.name: item.sha256 for item in contract.sources}
    expected_code = {item.name: item.sha256 for item in contract.code}
    if payload["contract_identity"] != contract.identity:
        raise PreflightError("reviewed projection contract identity drifted")
    if payload["manifest_identities"] != expected_manifests:
        raise PreflightError("reviewed projection manifest contract drifted")
    if payload["source_identities"] != expected_sources:
        raise PreflightError("reviewed projection source contract drifted")
    if payload["canonicalizer_identities"] != expected_code:
        raise PreflightError("reviewed projection canonicalizer contract drifted")
    builder_code = payload["builder_code_identities"]
    if (
        not isinstance(builder_code, Mapping)
        or set(builder_code)
        != {
            "scripts/data/build_directional_parent_projection.py",
            "src/dive/training/parent_projection.py",
        }
        or any(_HEX64.fullmatch(str(value)) is None for value in builder_code.values())
    ):
        raise PreflightError("reviewed projection builder code identity is invalid")
    for relative, expected_hash in builder_code.items():
        result = subprocess.run(
            (
                "git",
                "show",
                f"{payload['builder_commit']}:{relative}",
            ),
            cwd=_REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if (
            result.returncode != 0
            or hashlib.sha256(result.stdout).hexdigest() != expected_hash
        ):
            raise PreflightError(
                "reviewed projection builder source identity does not match "
                f"builder commit for {relative}"
            )
    spec = ReviewedProjectionSpec(
        schema_version=1,
        identity=str(payload["identity"]),
        projection_hash=str(payload["projection_hash"]),
        file_sha256=str(payload["file_sha256"]),
        size_bytes=int(payload["size_bytes"]),
        row_count=int(payload["row_count"]),
        builder_commit=str(payload["builder_commit"]),
        builder_code_identities=MappingProxyType(dict(builder_code)),
        contract_identity=str(payload["contract_identity"]),
        manifest_identities=tuple(
            MappingProxyType(dict(item)) for item in payload["manifest_identities"]
        ),
        source_identities=MappingProxyType(dict(payload["source_identities"])),
        canonicalizer_identities=MappingProxyType(
            dict(payload["canonicalizer_identities"])
        ),
    )
    return VerifiedProjectionSpec(
        path=Path(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        spec=spec,
    )

def _verify_projection_approval_commits(
    repo_root: Path, policy: ProjectionApprovalPolicy
) -> str:

    for label, commit in (
        ("builder", policy.builder_commit),
        ("approval", policy.approval_commit),
    ):
        result = subprocess.run(
            ("git", "cat-file", "-t", commit),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != "commit":
            raise PreflightError(
                f"projection {label} identity is not an existing commit object"
            )
    head = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    current = head.stdout.strip()
    if head.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", current) is None:
        raise PreflightError("projection verifier cannot resolve current checkout HEAD")
    for label, before, after in (
        ("builder -> approval", policy.builder_commit, policy.approval_commit),
        ("approval -> current HEAD", policy.approval_commit, current),
    ):
        result = subprocess.run(
            ("git", "merge-base", "--is-ancestor", before, after),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise PreflightError(f"projection commit ancestor relation failed: {label}")
    return current

def _read_git_blob(repo_root: Path, commit: str, relative: Path) -> bytes:

    object_name = f"{commit}:{relative.as_posix()}"
    resolved = subprocess.run(
        ("git", "rev-parse", "--verify", object_name),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    object_id = resolved.stdout.strip()
    if resolved.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", object_id) is None:
        raise PreflightError("approval commit does not contain the reviewed spec blob")
    kind = subprocess.run(
        ("git", "cat-file", "-t", object_id),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if kind.returncode != 0 or kind.stdout.strip() != "blob":
        raise PreflightError("reviewed approval spec object is not a Git blob")
    blob = subprocess.run(
        ("git", "cat-file", "blob", object_id),
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if blob.returncode != 0:
        raise PreflightError("cannot read reviewed approval spec Git blob")
    if not blob.stdout or len(blob.stdout) > 64 * 1024:
        raise PreflightError("reviewed approval spec Git blob size is invalid")
    return blob.stdout

def verify_manifest_inventory(
    root: Path, specs: tuple[ManifestSpec, ...]
) -> tuple[VerifiedManifest, ...]:

    import pandas as pd

    if type(specs) is not tuple or not specs:
        raise PreflightError("frozen manifest specs must be a nonempty tuple")
    root = Path(root)
    _reject_blind_path(root)
    _require_plain_directory(root, "manifest directory")
    entries = {entry.name: entry for entry in os.scandir(root)}
    expected = {spec.name for spec in specs}
    allowed = expected | {"coverage.json"}
    missing = sorted(expected - set(entries))
    unknown = sorted(set(entries) - allowed)
    if missing:
        raise PreflightError(f"frozen manifest inventory is missing {missing}")
    if unknown:
        raise PreflightError(f"frozen manifest inventory has unknown file(s) {unknown}")

    verified = []
    seen_global: set[str] = set()
    for spec in specs:
        entry = entries[spec.name]
        if entry.is_symlink():
            raise PreflightError(f"frozen manifest is a symlink: {entry.path}")
        raw = _read_exact_regular_bytes(Path(entry.path), spec)
        assert isinstance(raw, bytes)
        try:
            frame = pd.read_parquet(io.BytesIO(raw))
        except Exception as error:
            raise PreflightError(
                f"cannot parse frozen manifest {spec.name}: {error}"
            ) from error
        if len(frame) != spec.row_count:
            raise PreflightError(
                f"frozen manifest {spec.name} row_count changed: "
                f"{len(frame)} != {spec.row_count}"
            )
        ids = tuple(str(value) for value in frame["example_id"])
        if any(not value for value in ids):
            raise PreflightError(f"frozen manifest {spec.name} has empty example_id")
        if len(ids) != len(set(ids)):
            raise PreflightError(
                f"frozen manifest {spec.name} has duplicate example_id"
            )
        overlap = sorted(seen_global.intersection(ids))
        if overlap:
            raise PreflightError(
                f"frozen manifest inventory has duplicate example_id {overlap[:5]}"
            )
        seen_global.update(ids)
        if _example_id_hash(ids) != spec.example_id_hash:
            raise PreflightError(
                f"frozen manifest {spec.name} example_id identity changed"
            )
        inspected = (
            frame
            if "partition" in frame.columns
            else frame.assign(partition=spec.partition)
        )
        assert_manifest_is_trainable(Path(spec.name), inspected)
        if "partition" in frame.columns:
            present = {str(value) for value in frame["partition"].unique()}
            if present != {spec.partition}:
                raise PreflightError(
                    f"frozen manifest {spec.name} partition differs from contract"
                )
        from dive.data.loaders import assert_loader_manifest_is_clean

        assert_loader_manifest_is_clean(frame)
        records = tuple(MappingProxyType(dict(row)) for row in frame.to_dict("records"))
        verified.append(VerifiedManifest(spec=spec, example_ids=ids, records=records))
    return tuple(verified)

def verify_source_inputs(specs: Sequence[SourceSpec]) -> dict[str, VerifiedSource]:

    verified: dict[str, VerifiedSource] = {}
    for spec in specs:
        if spec.name in verified:
            raise PreflightError(f"duplicate frozen source name {spec.name!r}")
        _reject_blind_path(spec.path)
        content = _read_exact_regular_bytes(
            spec.path, spec, retain_bytes=spec.retain_bytes
        )
        verified[spec.name] = VerifiedSource(spec=spec, content=content)
    return verified

def regenerate_parent_maps(
    sources: Mapping[str, VerifiedSource],
) -> dict[str, dict[str, str]]:

    corpora = _canonicalize_verified_public_tables(sources)
    result: dict[str, dict[str, str]] = {}
    for family in _FAMILIES:
        corpus = corpora.get(family)
        if corpus is None:
            raise PreflightError(f"public canonicalization omitted family {family}")
        parents: dict[str, str] = {}
        for example in corpus.examples:
            example_id = str(example.example_id)
            parent_id = str(example.parent_id)
            if not example_id or not parent_id:
                raise PreflightError(
                    f"public canonicalization emitted empty identity for {family}"
                )
            if example_id in parents:
                raise PreflightError(
                    f"public canonicalization emitted duplicate example_id "
                    f"{example_id!r} for {family}"
                )
            parents[example_id] = parent_id
        if not parents:
            raise PreflightError(
                f"public canonicalization yielded no {family} examples"
            )
        result[family] = parents
    return result

def _canonicalize_verified_public_tables(
    sources: Mapping[str, VerifiedSource],
) -> dict[str, object]:

    import pandas as pd

    from dive.data.corpus import canonicalize_ame_tables, canonicalize_binder_table
    from dive.data.sabdab2 import canonicalize_sabdab2_table

    required = {
        "binder_metadata",
        "ame_metadata",
        "ame_resolver_cache",
        "antibody_archive",
        "antibody_split_table",
    }
    missing = sorted(required - set(sources))
    unknown = sorted(set(sources) - required)
    if missing:
        raise PreflightError(f"verified public inputs are missing {missing}")
    if unknown:
        raise PreflightError(f"verified public inputs have unknown source(s) {unknown}")

    def retained(name: str) -> bytes:
        content = sources[name].content
        if content is None:
            raise PreflightError(f"verified public input {name} did not retain bytes")
        return content

    binder_table = pd.read_csv(
        io.BytesIO(retained("binder_metadata")), low_memory=False
    )
    ame_table = pd.read_csv(io.BytesIO(retained("ame_metadata")), low_memory=False)
    ame_resolved = pd.read_parquet(io.BytesIO(retained("ame_resolver_cache")))
    antibody_table = pd.read_csv(
        io.BytesIO(retained("antibody_split_table")), low_memory=False
    )
    antibody_release = canonicalize_sabdab2_table(antibody_table)
    from dive.data.corpus import FamilyCorpus
    from dive.data.contracts import Family

    return {
        "binder": canonicalize_binder_table(binder_table),
        "ame": canonicalize_ame_tables(ame_table, ame_resolved),
        "antibody": FamilyCorpus(
            Family.ANTIBODY,
            antibody_release.examples,
            antibody_release.rejections,
        ),
    }

def build_projection_record(
    manifests: Sequence[VerifiedManifest],
    canonical_parents: Mapping[str, Mapping[str, str]],
    *,
    contract_identity: str,
    source_identities: Mapping[str, str],
    code_identities: Mapping[str, str],
) -> dict[str, Any]:

    _require_sha256("contract_identity", contract_identity)
    for group_name, identities in (
        ("source", source_identities),
        ("code", code_identities),
    ):
        if not identities:
            raise PreflightError(f"{group_name} identities are empty")
        for name, identity in identities.items():
            _require_sha256(f"{group_name} identity {name}", identity)

    rows = []
    seen: set[str] = set()
    for manifest in manifests:
        family_parents = canonical_parents.get(manifest.spec.family)
        if family_parents is None:
            raise PreflightError(
                f"canonical parent map is missing family {manifest.spec.family}"
            )
        for example_id in manifest.example_ids:
            if example_id in seen:
                raise PreflightError(f"duplicate loader example_id {example_id!r}")
            seen.add(example_id)
            parent_id = family_parents.get(example_id)
            if type(parent_id) is not str or not parent_id:
                raise PreflightError(
                    f"loader example_id {example_id!r} is unmapped in public parents"
                )
            rows.append(
                {
                    "family": manifest.spec.family,
                    "partition": manifest.spec.partition,
                    "example_id": example_id,
                    "parent_id": parent_id,
                }
            )
    rows.sort(key=lambda row: (row["family"], row["partition"], row["example_id"]))
    projection_hash = hashlib.sha256(canonical_json_bytes({"rows": rows})).hexdigest()
    return {
        "schema_version": 1,
        "contract_identity": contract_identity,
        "source_identities": dict(sorted(source_identities.items())),
        "code_identities": dict(sorted(code_identities.items())),
        "manifest_identities": [
            manifest.spec.as_provenance() for manifest in manifests
        ],
        "row_count": len(rows),
        "projection_hash": projection_hash,
        "rows": rows,
    }

def write_projection_create_new(identity: str, record: Mapping[str, Any]) -> Path:

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identity) is None:
        raise PreflightError(f"unsafe parent projection identity {identity!r}")
    if "blind" in identity.lower():
        raise PreflightError("parent projection identity must not be blind-named")
    path = (
        Path(_PROJECTION_EVIDENCE_ROOT)
        / "directional_parent_projections"
        / f"{identity}.json"
    )
    write_create_new_json(path, record)
    return path

def _load_reviewed_projection_artifact(
    identity: str,
    *,
    manifests: Sequence[VerifiedManifest],
    contract: ProjectionContract = PRODUCTION_CONTRACT,
    reviewed_spec: VerifiedProjectionSpec | None = None,
) -> tuple[Path, bytes, Mapping[str, Any], VerifiedProjectionSpec]:
    _validate_projection_identity(identity)
    reviewed = reviewed_spec or load_reviewed_projection_spec(contract=contract)
    if not isinstance(reviewed, VerifiedProjectionSpec):
        raise PreflightError("reviewed parent projection spec is absent")
    spec = reviewed.spec
    if spec.identity != identity:
        raise PreflightError("reviewed parent projection identity differs")
    if spec.contract_identity != contract.identity:
        raise PreflightError("reviewed parent projection contract differs")
    expected_manifests = tuple(item.spec.as_provenance() for item in manifests)
    if spec.manifest_identities != expected_manifests:
        raise PreflightError("reviewed parent projection manifests differ")
    if spec.source_identities != {
        item.name: item.sha256 for item in contract.sources
    } or spec.canonicalizer_identities != {
        item.name: item.sha256 for item in contract.code
    }:
        raise PreflightError("reviewed parent projection inputs differ")
    path = (
        Path(_PROJECTION_EVIDENCE_ROOT)
        / "directional_parent_projections"
        / f"{identity}.json"
    )
    raw = read_terminal_bytes(path)
    if (
        len(raw) != spec.size_bytes
        or hashlib.sha256(raw).hexdigest() != spec.file_sha256
    ):
        raise PreflightError("reviewed parent projection file identity changed")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError("parent projection is invalid JSON") from error
    if not isinstance(record, Mapping) or canonical_json_bytes(record) != raw:
        raise PreflightError("parent projection is not canonical JSON")
    required = {
        "schema_version",
        "contract_identity",
        "source_identities",
        "code_identities",
        "manifest_identities",
        "row_count",
        "projection_hash",
        "rows",
    }
    if set(record) != required or record["schema_version"] != 1:
        raise PreflightError("parent projection schema is invalid")
    if record["contract_identity"] != contract.identity:
        raise PreflightError("parent projection contract identity drifted")
    expected_sources = {spec.name: spec.sha256 for spec in contract.sources}
    expected_code = {spec.name: spec.sha256 for spec in contract.code}
    if record["source_identities"] != expected_sources:
        raise PreflightError("parent projection source identities drifted")
    if record["code_identities"] != expected_code:
        raise PreflightError("parent projection code identities drifted")
    if record["manifest_identities"] != list(expected_manifests):
        raise PreflightError("parent projection manifest identities drifted")
    rows = record["rows"]
    if not isinstance(rows, list) or record["row_count"] != len(rows):
        raise PreflightError("parent projection row count is inconsistent")
    observed_hash = hashlib.sha256(canonical_json_bytes({"rows": rows})).hexdigest()
    if record["projection_hash"] != observed_hash:
        raise PreflightError("parent projection row hash drifted")
    if (
        record["projection_hash"] != spec.projection_hash
        or record["row_count"] != spec.row_count
    ):
        raise PreflightError(
            "parent projection differs from reviewed semantic identity"
        )
    expected_keys = {
        (item.spec.family, item.spec.partition, example_id)
        for item in manifests
        for example_id in item.example_ids
    }
    observed_keys = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "family",
            "partition",
            "example_id",
            "parent_id",
        }:
            raise PreflightError("parent projection row schema is invalid")
        key = (row["family"], row["partition"], row["example_id"])
        if (
            key in observed_keys
            or type(row["parent_id"]) is not str
            or not row["parent_id"]
        ):
            raise PreflightError("parent projection has duplicate or empty identity")
        observed_keys.add(key)
    if observed_keys != expected_keys:
        raise PreflightError("parent projection rows differ from frozen loader IDs")
    return path, raw, record, reviewed

def load_parent_projection(
    identity: str,
    *,
    manifests: Sequence[VerifiedManifest],
    contract: ProjectionContract = PRODUCTION_CONTRACT,
    reviewed_spec: VerifiedProjectionSpec | None = None,
) -> Mapping[str, Any]:

    path, raw, record, reviewed = _load_reviewed_projection_artifact(
        identity,
        manifests=manifests,
        contract=contract,
        reviewed_spec=reviewed_spec,
    )
    _verify_reviewed_projection_completion(
        identity,
        projection_path=path,
        projection_raw=raw,
        reviewed=reviewed,
        contract=contract,
    )
    return record

def write_reviewed_projection_completion(
    identity: str,
    *,
    manifests: Sequence[VerifiedManifest],
    approving_commit: str,
    contract: ProjectionContract = PRODUCTION_CONTRACT,
    reviewed_spec: VerifiedProjectionSpec | None = None,
) -> Path:

    _validate_projection_identity(identity)
    if re.fullmatch(r"[0-9a-f]{40}", approving_commit) is None:
        raise PreflightError("projection approving_commit is invalid")
    current_commit = _current_repo_commit()
    if current_commit != approving_commit:
        raise PreflightError(
            "projection approving_commit must equal the clean current commit"
        )
    projection_path, projection_raw, _, reviewed = _load_reviewed_projection_artifact(
        identity,
        manifests=manifests,
        contract=contract,
        reviewed_spec=reviewed_spec,
    )
    directory = Path(_PROJECTION_EVIDENCE_ROOT) / "directional_parent_projections"
    attempt_path = directory / f"{identity}.attempt.json"
    attempt_raw = read_terminal_bytes(attempt_path)
    try:
        attempt = json.loads(attempt_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError("parent projection attempt is invalid JSON") from error
    if not isinstance(attempt, Mapping) or canonical_json_bytes(attempt) != attempt_raw:
        raise PreflightError("parent projection attempt is not canonical")
    if (
        attempt.get("schema_version") != 1
        or attempt.get("status") != "BUILDING"
        or attempt.get("identity") != identity
        or attempt.get("contract_identity") != contract.identity
        or attempt.get("output_path") != str(projection_path)
    ):
        raise PreflightError("parent projection attempt identity is inconsistent")
    spec = reviewed.spec
    completion_path = directory / f"{identity}.completion.json"
    write_create_new_json(
        completion_path,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "identity": identity,
            "contract_identity": contract.identity,
            "projection": {
                "path": str(projection_path),
                "sha256": hashlib.sha256(projection_raw).hexdigest(),
                "size_bytes": len(projection_raw),
                "semantic_sha256": spec.projection_hash,
                "row_count": spec.row_count,
            },
            "attempt": {
                "path": str(attempt_path),
                "sha256": hashlib.sha256(attempt_raw).hexdigest(),
                "size_bytes": len(attempt_raw),
            },
            "reviewed_spec": {
                "path": str(reviewed.path),
                "sha256": reviewed.sha256,
                "size_bytes": reviewed.size_bytes,
            },
            "builder_commit": spec.builder_commit,
            "approving_commit": approving_commit,
            "current_commit": current_commit,
        },
    )
    return completion_path

def _verify_reviewed_projection_completion(
    identity: str,
    *,
    projection_path: Path,
    projection_raw: bytes,
    reviewed: VerifiedProjectionSpec,
    contract: ProjectionContract,
) -> Mapping[str, Any]:
    directory = Path(_PROJECTION_EVIDENCE_ROOT) / "directional_parent_projections"
    completion_path = directory / f"{identity}.completion.json"
    try:
        raw = read_terminal_bytes(completion_path)
    except PreflightError as error:
        raise PreflightError(
            f"reviewed parent projection completion is unavailable: {completion_path}"
        ) from error
    policy = (
        _PROJECTION_APPROVAL_POLICY
        if contract is PRODUCTION_CONTRACT
        else _PROJECTION_V2_APPROVAL_POLICY
        if contract is PRODUCTION_CONTRACT_V2
        else None
    )
    if policy is not None and (
        identity != policy.identity
        or hashlib.sha256(raw).hexdigest() != policy.completion_sha256
    ):
        raise PreflightError("parent projection pinned completion sha256 changed")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError("parent projection completion is invalid JSON") from error
    if not isinstance(record, Mapping) or canonical_json_bytes(record) != raw:
        raise PreflightError("parent projection completion is not canonical")
    if set(record) != {
        "schema_version",
        "status",
        "identity",
        "contract_identity",
        "projection",
        "attempt",
        "reviewed_spec",
        "builder_commit",
        "approving_commit",
        "current_commit",
    }:
        raise PreflightError("parent projection completion schema is invalid")
    spec = reviewed.spec
    expected_projection = {
        "path": str(projection_path),
        "sha256": hashlib.sha256(projection_raw).hexdigest(),
        "size_bytes": len(projection_raw),
        "semantic_sha256": spec.projection_hash,
        "row_count": spec.row_count,
    }
    expected_spec = {
        "path": str(reviewed.path),
        "sha256": reviewed.sha256,
        "size_bytes": reviewed.size_bytes,
    }
    production_approval = policy is not None
    if production_approval:
        assert policy is not None
        recorded_spec = record["reviewed_spec"]
        relative_parts = policy.spec_relative_path.parts
        if (
            not isinstance(recorded_spec, Mapping)
            or set(recorded_spec) != {"path", "sha256", "size_bytes"}
            or recorded_spec["sha256"] != policy.spec_sha256
            or recorded_spec["sha256"] != reviewed.sha256
            or recorded_spec["size_bytes"] != reviewed.size_bytes
            or Path(str(recorded_spec["path"])).parts[-len(relative_parts) :]
            != relative_parts
        ):
            raise PreflightError(
                "parent projection completion reviewed spec identity is inconsistent"
            )
        spec_matches = True
        commits_match = (
            record["builder_commit"] == policy.builder_commit
            and record["approving_commit"] == policy.approval_commit
            and record["current_commit"] == policy.approval_commit
        )
    else:
        spec_matches = record["reviewed_spec"] == expected_spec
        commits_match = (
            record["builder_commit"] == spec.builder_commit
            and record["approving_commit"] == record["current_commit"]
            and re.fullmatch(r"[0-9a-f]{40}", str(record["current_commit"])) is not None
        )
    if (
        record["schema_version"] != 1
        or record["status"] != "COMPLETE"
        or record["identity"] != identity
        or record["contract_identity"] != contract.identity
        or record["projection"] != expected_projection
        or not spec_matches
        or not commits_match
    ):
        raise PreflightError("parent projection completion identity is inconsistent")
    attempt_identity = record["attempt"]
    attempt_path = directory / f"{identity}.attempt.json"
    if (
        not isinstance(attempt_identity, Mapping)
        or set(attempt_identity)
        != {
            "path",
            "sha256",
            "size_bytes",
        }
        or attempt_identity["path"] != str(attempt_path)
    ):
        raise PreflightError("parent projection completion attempt identity is invalid")
    attempt_raw = read_terminal_bytes(attempt_path)
    if (
        len(attempt_raw) != attempt_identity["size_bytes"]
        or hashlib.sha256(attempt_raw).hexdigest() != attempt_identity["sha256"]
    ):
        raise PreflightError("parent projection attempt changed after completion")
    return record

def _current_repo_commit() -> str:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=normal"),
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    observed = commit.stdout.strip()
    if (
        commit.returncode != 0
        or re.fullmatch(r"[0-9a-f]{40}", observed) is None
        or status.returncode != 0
        or status.stdout
    ):
        raise PreflightError("projection completion requires a clean exact commit")
    return observed

def attach_parent_projection(
    manifests: Sequence[VerifiedManifest], record: Mapping[str, Any]
) -> tuple[VerifiedManifest, ...]:

    parents = {
        (row["family"], row["partition"], row["example_id"]): row["parent_id"]
        for row in record["rows"]
    }
    attached = []
    for manifest in manifests:
        records = []
        for row in manifest.records:
            example_id = str(row["example_id"])
            key = (manifest.spec.family, manifest.spec.partition, example_id)
            if key not in parents:
                raise PreflightError(f"parent projection is missing {example_id!r}")
            records.append({**dict(row), "parent_id": parents[key]})
        attached.append(
            VerifiedManifest(
                spec=manifest.spec,
                example_ids=manifest.example_ids,
                records=tuple(MappingProxyType(row) for row in records),
            )
        )
    return tuple(attached)

def materialize_parent_projection(
    loader_root: Path,
    *,
    identity: str,
    argv: tuple[str, ...],
    contract: ProjectionContract = PRODUCTION_CONTRACT,
) -> tuple[Path, dict[str, Any]]:

    _validate_projection_identity(identity)
    if (
        type(argv) is not tuple
        or not argv
        or any(type(token) is not str or not token for token in argv)
    ):
        raise PreflightError("parent projection command argv must be an exact tuple")
    _verify_upstream_checkout(contract.upstream_commit)
    _assert_projection_capacity()
    manifests = verify_manifest_inventory(Path(loader_root), contract.manifests)
    sources = verify_source_inputs(contract.sources)
    code = verify_source_inputs(contract.code)

    directory = Path(_PROJECTION_EVIDENCE_ROOT) / "directional_parent_projections"
    final_path = directory / f"{identity}.json"
    attempt_path = directory / f"{identity}.attempt.json"
    if final_path.exists() or attempt_path.exists():
        raise PreflightError(
            f"create-new parent projection identity already exists: {identity}"
        )
    source_ids = {name: item.spec.sha256 for name, item in sources.items()}
    code_ids = {name: item.spec.sha256 for name, item in code.items()}
    attempt = {
        "schema_version": 1,
        "status": "BUILDING",
        "identity": identity,
        "contract_identity": contract.identity,
        "upstream_commit": contract.upstream_commit,
        "command_argv": list(argv),
        "command_rendered": shlex.join(argv),
        "loader_root": str(Path(loader_root)),
        "manifest_identities": [item.spec.as_provenance() for item in manifests],
        "source_identities": dict(sorted(source_ids.items())),
        "code_identities": dict(sorted(code_ids.items())),
        "output_path": str(final_path),
    }
    write_create_new_json(attempt_path, attempt)

    parents = regenerate_parent_maps(sources)
    record = build_projection_record(
        manifests,
        parents,
        contract_identity=contract.identity,
        source_identities=source_ids,
        code_identities=code_ids,
    )
    write_create_new_json(final_path, record)
    return final_path, record

def _verify_upstream_checkout(expected_commit: str) -> None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_UPSTREAM_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != expected_commit:
        raise PreflightError(
            "pinned public-source checkout commit drifted: "
            f"expected {expected_commit}, observed {result.stdout.strip()!r}"
        )

def _validate_projection_identity(identity: object) -> str:
    if (
        type(identity) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identity) is None
        or "blind" in identity.lower()
    ):
        raise PreflightError(f"unsafe parent projection identity {identity!r}")
    return identity

def _assert_projection_capacity() -> None:
    root = Path(_PROJECTION_EVIDENCE_ROOT)
    _reject_blind_path(root)
    _require_plain_directory(root, "fixed projection evidence root")
    free = shutil.disk_usage(root).free
    if free < 16 * 1024 * 1024:
        raise PreflightError(
            f"fixed projection evidence root has only {free} free bytes"
        )

def _example_id_hash(example_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(example_ids) + "\n").encode()).hexdigest()

def _reject_blind_path(path: Path) -> None:
    lexical = Path(os.path.abspath(Path(path)))
    if any("blind" in part.lower() for part in lexical.parts):
        raise PreflightError(f"blind-named/path input is forbidden: {lexical}")

def _require_plain_directory(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise PreflightError(f"{label} is missing: {path}") from error
    if stat_module.S_ISLNK(metadata.st_mode):
        raise PreflightError(f"{label} is a symlink: {path}")
    if not stat_module.S_ISDIR(metadata.st_mode):
        raise PreflightError(f"{label} is not a directory: {path}")

def _read_exact_regular_bytes(
    path: Path,
    spec: ManifestSpec | SourceSpec,
    *,
    retain_bytes: bool = True,
) -> bytes | None:
    fd = _open_regular_nofollow(path)
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] | None = [] if retain_bytes else None
    try:
        before = os.fstat(fd)
        if before.st_size <= 0:
            raise PreflightError(f"frozen input is empty: {path}")
        while chunk := os.read(fd, _READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise PreflightError(f"frozen input changed while reading: {path}")
    observed = digest.hexdigest()
    if size != spec.size_bytes or observed != spec.sha256:
        raise PreflightError(
            f"frozen input identity changed for {path}: size={size}, sha256={observed}"
        )
    return b"".join(chunks) if chunks is not None else None

def _read_bounded_regular_bytes(path: Path, *, maximum: int) -> bytes:

    if type(maximum) is not int or maximum <= 0:
        raise PreflightError("bounded read maximum must be positive")
    fd = _open_regular_nofollow(path)
    try:
        before = os.fstat(fd)
        if before.st_size <= 0 or before.st_size > maximum:
            raise PreflightError(
                f"reviewed spec size {before.st_size} is outside (0, {maximum}]"
            )
        raw = os.read(fd, before.st_size + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise PreflightError(f"reviewed spec changed while reading: {path}")
    return raw

def _open_regular_nofollow(path: Path) -> int:

    lexical = Path(os.path.abspath(Path(path)))
    if not lexical.is_absolute() or lexical.name in {"", ".", ".."}:
        raise PreflightError(f"unsafe frozen input path: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    current_fd = os.open("/", directory_flags)
    try:
        for part in lexical.parts[1:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except OSError as error:
                reason = "symlink" if error.errno in {20, 40} else str(error)
                raise PreflightError(
                    f"cannot open frozen input parent {part!r} ({reason}): {path}"
                ) from error
            os.close(current_fd)
            current_fd = next_fd
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(lexical.name, flags, dir_fd=current_fd)
        except OSError as error:
            reason = "symlink" if error.errno in {20, 40} else str(error)
            raise PreflightError(
                f"cannot open frozen input ({reason}): {path}"
            ) from error
        metadata = os.fstat(fd)
        if not stat_module.S_ISREG(metadata.st_mode):
            os.close(fd)
            raise PreflightError(f"frozen input is not a regular file: {path}")
        return fd
    finally:
        os.close(current_fd)
