
from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from dive.benchmark.contracts import ArtifactIdentity, file_identity
from dive.benchmark.evaluators.ame import AmeRedesign, reduce_ame
from dive.benchmark.evaluators.antibody import (
    AntibodyComplex,
    FrozenAntibodyRoles,
    ResidueKey,
    evaluate_antibody,
)
from dive.benchmark.evaluators.binder import reduce_binder
from dive.benchmark.evaluators.contracts import MetricUnavailable, MetricValue
from dive.benchmark.evaluators.worker_protocol import WorkerProtocolError
from dive.benchmark.paper_baseline import (
    FIXTURE_EVALUATOR_REGISTRY_SHA256,
    PRODUCTION_EVALUATOR_REGISTRY_SHA256,
    PaperBaselineError,
    PaperCell,
    paper_cell_identity,
)
from dive.benchmark.paper_batches import PaperBatch
from dive.benchmark.paper_export import (
    AME_CLASH_LIGAND_RESNAME,
    AME_EVALUATOR_LIGAND_CHAIN,
    AME_LIGAND_TRANSPORT_VERSION,
    BINDER_BACKBONE_NORMALIZATION_VERSION,
    AmeLigandTransport,
    BinderBackboneNormalization,
    PaperExportResult,
    _reclassify_ame_clash_ligand,
    ame_ligand_transport_from_mapping,
    authenticate_binder_geometry_repairs_against_generated_sample,
    authenticate_binder_native_inputs,
    authenticate_binder_native_repairs_against_batch,
    authenticate_binder_repaired_atoms_against_pdb,
    binder_backbone_normalization_from_mapping,
)
from dive.benchmark.paper_pdb_auth import (
    AuthenticatedPdb,
    PdbAuthenticationError,
    PrivateBinderInputs,
    publish_private_binder_inputs,
    read_authenticated_pdb,
)
from dive.benchmark.paper_workers import binder_redesigns_from_track_e_eval
from dive.signed_value.roots import (
    DENIED_PREFIXES,
    EMERGENT_UPSTREAM_ROOT,
    RUNTIME_PYTHON,
)

_BINDER_METHOD = "soluble_mpnn_2x_af2_multimer"
_BINDER_SELECTION = "or_over_two_sequence_redesigns"
_BINDER_REDESIGNS = 2
_AME_METHOD = "ligandmpnn_rf3_motif_ligand_joint"
_AME_SELECTION = "same_index_joint_or"
_AME_REDESIGNS = 2
_AME_SEQUENCE_TYPE = "mpnn_fixed"
_RF3_SENTINEL = 100.0
REQUIRED_AME_RAW_FIELDS = frozenset(
    {
        "binder_scRMSD_bb3",
        "motif_all_atom_rmsd",
        "exact_motif_sequence_recovery",
        "ligand_clash",
        "min_ipAE",
        "redesign_id",
        "sequence",
        "binder_scRMSD_bb3_source_present",
        "motif_all_atom_rmsd_source_present",
        "exact_motif_sequence_recovery_source_present",
        "ligand_clash_source_present",
        "min_ipAE_source_present",
    }
)
REQUIRED_AME_SOURCE_FLAGS = (
    "binder_scRMSD_bb3_source_present",
    "motif_all_atom_rmsd_source_present",
    "exact_motif_sequence_recovery_source_present",
    "ligand_clash_source_present",
    "min_ipAE_source_present",
)
_AME_SOURCE_FLAG_COLUMNS = {
    "binder_scRMSD_bb3": "mpnn_fixed_binder_scRMSD_bb3_source_present_all",
    "motif_all_atom_rmsd": "mpnn_fixed_motif_all_atom_rmsd_source_present_all",
    "exact_motif_sequence_recovery": (
        "mpnn_fixed_exact_motif_sequence_recovery_source_present_all"
    ),
    "ligand_clash": "mpnn_fixed_ligand_clash_source_present_all",
    "min_ipAE": "mpnn_fixed_min_ipAE_source_present_all",
}
_ANTIBODY_METHOD = "frozen_native_geometry"
_ANTIBODY_SELECTION = "in_process_native_geometry"
_ANTIBODY_SECONDARIES = (
    "all_atom_rmsd",
    "chi1_accuracy",
    "chi2_accuracy",
    "antigen_contact_recovery",
    "clash_rate",
)

@dataclass(frozen=True, slots=True)
class PaperEvaluationRequest:
    cell: PaperCell
    export: PaperExportResult
    registry_sha256: str
    work_dir: Path
    fixed_positions: tuple[str, ...] = ()
    run_binder_eval: Callable[..., object] | None = None
    run_ame_eval: Callable[..., object] | None = None
    expected_batch: PaperBatch | None = None
    generated_sample: ArtifactIdentity | None = None

@dataclass(frozen=True, slots=True)
class PaperEvaluationResult:
    family: str
    primary: MetricValue
    raw: Mapping[str, object]
    cell_id: str
    cell: PaperCell
    method: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))

    def to_mapping(self) -> dict[str, object]:
        return {
            "family": self.family,
            "primary": {
                "name": self.primary.name,
                "value": self.primary.value,
                "direction": self.primary.direction,
                "evaluator_manifest_hash": self.primary.evaluator_manifest_hash,
            },
            "raw": dict(self.raw),
            "cell_id": self.cell_id,
            "method": self.method,
        }

def evaluate_paper_binder(request: PaperEvaluationRequest) -> PaperEvaluationResult:

    complex_pdb, native_target = _authenticate_binder_request(request)
    try:
        private_inputs = publish_private_binder_inputs(
            complex_pdb=complex_pdb,
            native_target=native_target,
            work_dir=request.work_dir,
        )
    except PdbAuthenticationError as error:
        raise PaperBaselineError(
            f"binder private snapshot publication failed: {error}"
        ) from error
    with private_inputs:
        kwargs = _binder_eval_kwargs(
            request,
            private_inputs=private_inputs,
            complex_pdb=complex_pdb,
        )
        runner = request.run_binder_eval
        if runner is None:
            _refuse_denied(Path(kwargs["tmp_path"]))
            payload = _pinned_run_binder_eval(private_inputs, **kwargs)
        else:
            payload = runner(**kwargs)
    stats, sequences = _unpack_binder_eval(payload)
    sequence_type = kwargs["sequence_types"][0]
    rows = binder_redesigns_from_track_e_eval(
        stats, sequences, sequence_type=sequence_type
    )
    reduced = reduce_binder(rows, PRODUCTION_EVALUATOR_REGISTRY_SHA256)
    raw = dict(reduced.raw)
    raw["redesign_count"] = _BINDER_REDESIGNS
    raw["selection_rule"] = _BINDER_SELECTION
    raw["method"] = _BINDER_METHOD
    _assert_two_finite_vectors(raw)
    return PaperEvaluationResult(
        family="binder",
        primary=reduced.primary,
        raw=raw,
        cell_id=request.cell.cell_id,
        cell=request.cell,
        method=_BINDER_METHOD,
    )

def _authenticate_binder_request(
    request: PaperEvaluationRequest,
) -> tuple[AuthenticatedPdb, AuthenticatedPdb]:
    if type(request) is not PaperEvaluationRequest:
        raise PaperBaselineError("binder request must be a PaperEvaluationRequest")
    cell = request.cell
    if type(cell) is not PaperCell or cell.family != "binder":
        raise PaperBaselineError("binder adapter requires a binder cell")
    if (
        request.registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256
        or cell.evaluator_registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256
    ):
        raise PaperBaselineError("fixture hash cannot authenticate as production")
    if (
        request.registry_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256
        or cell.evaluator_registry_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256
    ):
        raise PaperBaselineError("production evaluator registry hash drifted")
    expected = paper_cell_identity(
        panel=cell.panel,
        arm=cell.arm,
        family=cell.family,
        partition=cell.partition,
        parent_id=cell.parent_id,
        example_id=cell.example_id,
        seed=cell.seed,
        sample_index=cell.sample_index,
        generation_replicas=cell.generation_replicas,
        steps=cell.steps,
        condition_count=cell.condition_count,
        denoiser_equivalent_passes=cell.denoiser_equivalent_passes,
        checkpoint=cell.checkpoint,
        splice_report_sha256=cell.splice_report_sha256,
        evaluator_registry_sha256=cell.evaluator_registry_sha256,
        view_semantic_hash=cell.view_semantic_hash,
        code_commit=cell.code_commit,
    )
    if cell.cell_id != expected:
        raise PaperBaselineError("cell identity drifted")
    export = request.export
    if type(export) is not PaperExportResult or export.family != "binder":
        raise PaperBaselineError("binder adapter requires a binder export")
    if export.native_target is None:
        raise PaperBaselineError("binder export is missing native target")
    live_manifest = file_identity(Path(export.manifest.path))
    if (
        live_manifest.sha256 != export.manifest.sha256
        or live_manifest.size_bytes != export.manifest.size_bytes
    ):
        raise PaperBaselineError("export identity drifted")
    normalization = export.binder_backbone_normalization
    if (
        type(normalization) is not BinderBackboneNormalization
        or normalization.version != BINDER_BACKBONE_NORMALIZATION_VERSION
        or normalization.roles_before_sha256 != normalization.roles_after_sha256
        or normalization.sequences_before_sha256 != normalization.sequences_after_sha256
    ):
        raise PaperBaselineError("binder backbone normalization identity drifted")
    _authenticate_binder_export_manifest(export, normalization, cell.example_id)
    try:
        parsed_normalization = binder_backbone_normalization_from_mapping(
            normalization.to_mapping()
        )
    except PaperBaselineError as error:
        raise PaperBaselineError("binder repair provenance drifted") from error
    if parsed_normalization != normalization:
        raise PaperBaselineError("binder repair provenance drifted")
    if request.expected_batch is not None and (
        request.expected_batch.target.example_id != cell.example_id
        or request.expected_batch.target.family != "binder"
    ):
        raise PaperBaselineError("binder expected batch identity drifted")
    try:
        complex_pdb = read_authenticated_pdb(
            export.complex_pdb, label="binder complex PDB"
        )
        native_target = read_authenticated_pdb(
            export.native_target, label="binder native-target PDB"
        )
    except PdbAuthenticationError as error:
        raise PaperBaselineError(
            f"binder PDB authentication failed: {error}"
        ) from error
    authenticate_binder_native_repairs_against_batch(
        normalization, request.expected_batch
    )
    authenticate_binder_geometry_repairs_against_generated_sample(
        normalization,
        request.generated_sample,
        cell,
        request.expected_batch,
        complex_pdb,
    )
    authenticate_binder_native_inputs(
        complex_pdb=complex_pdb,
        native_target=native_target,
        expected_batch=request.expected_batch,
    )
    for repair in normalization.repaired_atoms:
        if repair.source_artifact is None:
            continue
        live_source = file_identity(Path(repair.source_artifact.path))
        if (
            live_source.sha256 != repair.source_artifact.sha256
            or live_source.size_bytes != repair.source_artifact.size_bytes
        ):
            raise PaperBaselineError("binder native repair source artifact drifted")
    authenticate_binder_repaired_atoms_against_pdb(normalization, complex_pdb)
    _preflight_binder_four_atom(complex_pdb)
    return complex_pdb, native_target

def _authenticate_binder_export_manifest(
    export: PaperExportResult,
    normalization: BinderBackboneNormalization,
    example_id: str,
) -> None:
    try:
        payload = json.loads(Path(export.manifest.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperBaselineError(
            "binder export manifest cannot be authenticated"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "dive.paper.export_manifest.v1"
        or payload.get("family") != "binder"
        or payload.get("example_id") != example_id
        or payload.get("binder_backbone_normalization") != normalization.to_mapping()
    ):
        raise PaperBaselineError("binder manifest normalization identity drifted")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise PaperBaselineError("binder export manifest artifacts drifted")
    observed = {
        entry.get("kind"): entry
        for entry in artifacts
        if isinstance(entry, Mapping) and type(entry.get("kind")) is str
    }
    expected = {
        "complex_pdb": export.complex_pdb,
        "native_target": export.native_target,
    }
    for kind, identity in expected.items():
        assert identity is not None
        if observed.get(kind) != {
            "kind": kind,
            "path": identity.path,
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }:
            raise PaperBaselineError("binder export manifest artifacts drifted")

def _preflight_binder_four_atom(pdb: AuthenticatedPdb) -> None:
    required = frozenset({"N", "CA", "C", "O"})
    counts: dict[tuple[str, int], dict[str, int]] = {}
    coordinates: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    for pdb_atom in pdb.atoms:
        residue_key = (pdb_atom.chain, pdb_atom.residue)
        counts.setdefault(residue_key, {})
        if pdb_atom.atom not in required:
            continue
        counts[residue_key][pdb_atom.atom] = (
            counts[residue_key].get(pdb_atom.atom, 0) + 1
        )
        coordinates[(*residue_key, pdb_atom.atom)] = pdb_atom.coordinate
    if not counts:
        raise PaperBaselineError("binder four-atom preflight found no protein residues")
    for residue_key, atom_counts in counts.items():
        if set(atom_counts) != required or any(
            atom_counts[name] != 1 for name in required
        ):
            raise PaperBaselineError(
                "binder four-atom preflight requires exactly one N/CA/C/O "
                f"for {residue_key[0]}:{residue_key[1]}"
            )
        for name in required:
            if not all(
                math.isfinite(value) for value in coordinates[(*residue_key, name)]
            ):
                raise PaperBaselineError(
                    "binder four-atom preflight requires finite N/CA/C/O"
                )

def _binder_eval_kwargs(
    request: PaperEvaluationRequest,
    *,
    private_inputs: PrivateBinderInputs,
    complex_pdb: AuthenticatedPdb,
) -> dict[str, object]:
    chains = _authenticated_chains_in_file_order(complex_pdb)
    if len(chains) < 2:
        raise PaperBaselineError(
            "binder complex lacks exported target and binder chains"
        )
    binder_chain = chains[-1]
    target_chains = list(chains[:-1])
    fixed = tuple(request.fixed_positions)
    sequence_type = "mpnn_fixed" if fixed else "mpnn"
    af2 = _registry_asset("binder", "af2_multimer_v3_checkpoint")
    af2_path = Path(str(af2["path"]))
    return {
        "pdb_file_path": private_inputs.complex_pdb.path,
        "target_pdb_path": private_inputs.native_target.path,
        "folding_model_specs": {
            "model_name": "colabdesign",
            "af2_checkpoint_path": str(af2_path),
            "af2_checkpoint_sha256": af2["sha256"],
            "af2_params_dir": str(af2_path.parent),
        },
        "tmp_path": str(Path(request.work_dir)),
        "target_pdb_chain": target_chains,
        "sequence_types": [sequence_type],
        "interface_cutoff": 8.0,
        "is_target_ligand": False,
        "inverse_folding_model": "soluble_mpnn",
        "gen_target_chain": target_chains,
        "binder_chain": binder_chain,
        "num_redesign_seqs": _BINDER_REDESIGNS,
        "fixed_residues_override": list(fixed) if fixed else None,
    }

class _RetainedUpdatedPdbPath(str):

    def __new__(cls, updated_path: str, snapshot_path: str) -> _RetainedUpdatedPdbPath:
        obj = str.__new__(cls, updated_path)
        obj._snapshot_path = snapshot_path
        return obj

    def replace(self, old: str, new: str, count: int = -1) -> str:
        if old == "_updated.pdb" and new == ".pdb":
            return self._snapshot_path
        if count == -1:
            return str.replace(self, old, new)
        return str.replace(self, old, new, count)

def ensure_binder_chain_sorts_last(atom_array, binder_chain: str):

    import numpy as np

    chain_id = np.asarray(atom_array.chain_id)
    chains = sorted({str(item) for item in chain_id.tolist()})
    binder = str(binder_chain)
    if not chains:
        raise PaperBaselineError("generated complex has no chains")
    if binder not in chains:
        raise PaperBaselineError("binder chain missing from generated complex")
    if chains[-1] == binder:
        return atom_array
    last = chains[-1]
    if last >= "Z":
        raise PaperBaselineError("cannot rename binder chain to sort last")
    renamed = chr(ord(last) + 1)
    if renamed in chains:
        raise PaperBaselineError("cannot rename binder chain to sort last")
    remapped = np.array(
        [renamed if str(item) == binder else item for item in chain_id],
        dtype=chain_id.dtype,
    )
    atom_array.set_annotation("chain_id", remapped)
    return atom_array

def _pinned_run_binder_eval(private_inputs: PrivateBinderInputs, **kwargs):
    import sys

    source = str(EMERGENT_UPSTREAM_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    previous_cwd = Path.cwd()
    previous_python = os.environ.get("PYTHON_EXEC")
    previous_af2 = os.environ.get("AF2_DIR")
    specs = kwargs.get("folding_model_specs")
    os.chdir(EMERGENT_UPSTREAM_ROOT)
    os.environ["PYTHON_EXEC"] = str(RUNTIME_PYTHON)
    if isinstance(specs, Mapping) and specs.get("af2_params_dir"):
        os.environ["AF2_DIR"] = str(specs["af2_params_dir"])
    original_dirname = os.path.dirname
    original_join = os.path.join
    original_splitext = os.path.splitext
    binder_metrics_module = None
    pdb_utils_module = None
    inverse_folding_module = None
    original_binder_name = None
    original_utils_name = None
    original_inverse_name = None
    original_load_any = None
    original_subprocess_run = None
    original_get_interface = None
    try:
        from proteinfoundation.metrics import binder_metrics as binder_metrics_module
        from proteinfoundation.metrics import (
            inverse_folding_models as inverse_folding_module,
        )
        from proteinfoundation.metrics.binder_metrics import run_binder_eval
        from proteinfoundation.utils import pdb_utils as pdb_utils_module

        directory_path = f"/proc/{os.getpid()}/fd/{private_inputs.directory_descriptor}"
        complex_path = os.fspath(kwargs["pdb_file_path"])
        native_path = os.fspath(kwargs["target_pdb_path"])
        file_fd_paths = {complex_path, native_path}
        original_binder_name = binder_metrics_module.pdb_name_from_path
        original_utils_name = pdb_utils_module.pdb_name_from_path
        original_inverse_name = inverse_folding_module.pdb_name_from_path
        original_load_any = binder_metrics_module.load_any

        def dirname(path: str | os.PathLike[str]) -> str:
            if os.fspath(path) in file_fd_paths:
                return directory_path
            return original_dirname(path)

        def join(*parts: str | os.PathLike[str]) -> str:
            result = original_join(*parts)
            if not parts or os.fspath(parts[0]) != directory_path:
                return result
            leaf = os.fspath(parts[-1])
            if leaf == "complex_updated.pdb":
                return _RetainedUpdatedPdbPath(result, complex_path)
            if leaf == "native_target_updated.pdb":
                return _RetainedUpdatedPdbPath(result, native_path)
            return result

        def splitext(path: str | os.PathLike[str]) -> tuple[str, str]:
            normalized = os.fspath(path)
            if normalized in file_fd_paths:
                return (normalized, ".pdb")
            return original_splitext(path)

        def pdb_name_from_path(path: str | os.PathLike[str]) -> str:
            normalized = os.fspath(path)
            if normalized == complex_path:
                return "complex"
            if normalized == native_path:
                return "native_target"
            return original_utils_name(path)

        def load_any(
            file_or_buffer: object, file_type: str | None = None, **load_kwargs
        ):
            try:
                normalized = os.fspath(file_or_buffer)
            except TypeError:
                normalized = None
            if normalized in file_fd_paths and file_type is None:
                file_type = "pdb"
            loaded = original_load_any(file_or_buffer, file_type, **load_kwargs)
            binder_chain = kwargs.get("binder_chain")
            if normalized == complex_path and isinstance(binder_chain, str):
                structure = loaded[0] if isinstance(loaded, (list, tuple)) else loaded
                ensure_binder_chain_sorts_last(structure, binder_chain)
            return loaded

        os.path.dirname = dirname
        os.path.join = join
        os.path.splitext = splitext
        binder_metrics_module.pdb_name_from_path = pdb_name_from_path
        pdb_utils_module.pdb_name_from_path = pdb_name_from_path
        inverse_folding_module.pdb_name_from_path = pdb_name_from_path
        binder_metrics_module.load_any = load_any
        original_get_interface = binder_metrics_module.get_interface_residues
        binder_metrics_module.get_interface_residues = lambda *_args, **_kwargs: []
        tmp_root = kwargs.get("tmp_path")
        if tmp_root:
            alias_dir = Path(str(tmp_root)) / "ligandmpnn-fd"
            alias_dir.mkdir(mode=0o700, exist_ok=True)
            complex_alias = alias_dir / "complex.pdb"
            native_alias = alias_dir / "native_target.pdb"
            for alias, target in (
                (complex_alias, complex_path),
                (native_alias, native_path),
            ):
                if alias.exists() or alias.is_symlink():
                    alias.unlink()
                os.symlink(target, alias)
            original_subprocess_run = subprocess.run
            replacements = tuple(
                sorted(
                    (
                        (complex_path, str(complex_alias)),
                        (native_path, str(native_alias)),
                    ),
                    key=lambda item: len(item[0]),
                    reverse=True,
                )
            )

            def run_subprocess(command, *run_args, **run_kwargs):
                if isinstance(command, str):
                    for source, destination in replacements:
                        command = command.replace(source, destination)
                elif isinstance(command, (list, tuple)):
                    rewritten = []
                    for part in command:
                        if part == complex_path:
                            rewritten.append(str(complex_alias))
                        elif part == native_path:
                            rewritten.append(str(native_alias))
                        else:
                            rewritten.append(part)
                    command = type(command)(rewritten)
                return original_subprocess_run(command, *run_args, **run_kwargs)

            subprocess.run = run_subprocess
        return run_binder_eval(**kwargs)
    finally:
        if original_subprocess_run is not None:
            subprocess.run = original_subprocess_run
        os.path.dirname = original_dirname
        os.path.join = original_join
        os.path.splitext = original_splitext
        if binder_metrics_module is not None:
            if original_binder_name is not None:
                binder_metrics_module.pdb_name_from_path = original_binder_name
            if original_load_any is not None:
                binder_metrics_module.load_any = original_load_any
            if original_get_interface is not None:
                binder_metrics_module.get_interface_residues = original_get_interface
        if pdb_utils_module is not None and original_utils_name is not None:
            pdb_utils_module.pdb_name_from_path = original_utils_name
        if inverse_folding_module is not None and original_inverse_name is not None:
            inverse_folding_module.pdb_name_from_path = original_inverse_name
        os.chdir(previous_cwd)
        if previous_python is None:
            os.environ.pop("PYTHON_EXEC", None)
        else:
            os.environ["PYTHON_EXEC"] = previous_python
        if previous_af2 is None:
            os.environ.pop("AF2_DIR", None)
        else:
            os.environ["AF2_DIR"] = previous_af2

def _unpack_binder_eval(payload: object) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(payload, Sequence) or len(payload) != 4:
        raise PaperBaselineError("binder eval payload is not a four-tuple")
    stats = payload[2]
    sequences = payload[3]
    if not isinstance(stats, dict) or not isinstance(sequences, dict):
        raise PaperBaselineError("binder eval payload missing sequence type rows")
    return stats, sequences

def _assert_two_finite_vectors(raw: Mapping[str, object]) -> None:
    for name in ("i_pAE", "pLDDT", "binder_scRMSD_ca"):
        values = raw.get(name)
        if not isinstance(values, tuple) or len(values) != _BINDER_REDESIGNS:
            raise PaperBaselineError(f"invalid vector length: {name}")
        if any(type(item) is not float or not math.isfinite(item) for item in values):
            raise PaperBaselineError(f"{name} is non-finite")
    for name in ("redesign_id", "sequence"):
        values = raw.get(name)
        if not isinstance(values, tuple) or len(values) != _BINDER_REDESIGNS:
            raise PaperBaselineError(f"invalid vector length: {name}")
    if raw.get("redesign_count") != _BINDER_REDESIGNS:
        raise PaperBaselineError("binder production worker requires two redesigns")
    if raw.get("selection_rule") != _BINDER_SELECTION:
        raise PaperBaselineError("binder selection rule drifted")

def _authenticated_chains_in_file_order(pdb: AuthenticatedPdb) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for atom in pdb.atoms:
        if atom.chain not in seen:
            seen.add(atom.chain)
            ordered.append(atom.chain)
    return tuple(ordered)

def _chains_in_file_order(path: Path) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 22:
            continue
        chain = line[21].strip() or " "
        if chain not in seen:
            seen.add(chain)
            ordered.append(chain)
    return tuple(ordered)

def evaluate_paper_ame(request: PaperEvaluationRequest) -> PaperEvaluationResult:

    _authenticate_ame_request(request)
    kwargs = _ame_eval_kwargs(request)
    runner = request.run_ame_eval
    if runner is None:
        _refuse_generated_as_native_motif(kwargs)
        runner = _pinned_run_ame_eval
        _refuse_denied(Path(kwargs["tmp_path"]))
    payload = runner(**kwargs)
    try:
        rows = _ame_redesigns_from_motif_ligand_eval(payload)
        reduced = reduce_ame(rows, PRODUCTION_EVALUATOR_REGISTRY_SHA256)
    except WorkerProtocolError as error:
        raise PaperBaselineError(str(error)) from error
    raw = dict(reduced.raw)
    raw["redesign_count"] = _AME_REDESIGNS
    raw["selection_rule"] = _AME_SELECTION
    raw["method"] = _AME_METHOD
    _assert_two_ame_vectors(raw)
    return PaperEvaluationResult(
        family="ame",
        primary=reduced.primary,
        raw=raw,
        cell_id=request.cell.cell_id,
        cell=request.cell,
        method=_AME_METHOD,
    )

def _authenticate_ame_request(request: PaperEvaluationRequest) -> None:
    if type(request) is not PaperEvaluationRequest:
        raise PaperBaselineError("ame request must be a PaperEvaluationRequest")
    cell = request.cell
    if type(cell) is not PaperCell or cell.family != "ame":
        raise PaperBaselineError("ame adapter requires an ame cell")
    if (
        request.registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256
        or cell.evaluator_registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256
    ):
        raise PaperBaselineError("fixture hash cannot authenticate as production")
    if (
        request.registry_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256
        or cell.evaluator_registry_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256
    ):
        raise PaperBaselineError("production evaluator registry hash drifted")
    expected = paper_cell_identity(
        panel=cell.panel,
        arm=cell.arm,
        family=cell.family,
        partition=cell.partition,
        parent_id=cell.parent_id,
        example_id=cell.example_id,
        seed=cell.seed,
        sample_index=cell.sample_index,
        generation_replicas=cell.generation_replicas,
        steps=cell.steps,
        condition_count=cell.condition_count,
        denoiser_equivalent_passes=cell.denoiser_equivalent_passes,
        checkpoint=cell.checkpoint,
        splice_report_sha256=cell.splice_report_sha256,
        evaluator_registry_sha256=cell.evaluator_registry_sha256,
        view_semantic_hash=cell.view_semantic_hash,
        code_commit=cell.code_commit,
    )
    if cell.cell_id != expected:
        raise PaperBaselineError("cell identity drifted")
    export = request.export
    if type(export) is not PaperExportResult or export.family != "ame":
        raise PaperBaselineError("ame adapter requires an ame export")
    if export.motif_positions is None:
        raise PaperBaselineError("ame export is missing motif positions")
    identities = [export.complex_pdb, export.motif_positions, export.manifest]
    if export.native_motif is not None:
        identities.append(export.native_motif)
    for identity in identities:
        live = file_identity(Path(identity.path))
        if live.sha256 != identity.sha256 or live.size_bytes != identity.size_bytes:
            raise PaperBaselineError("export identity drifted")
    _authenticate_ame_ligand_transport(request)

def _authenticate_ame_ligand_transport(request: PaperEvaluationRequest) -> None:
    export = request.export
    transport = export.ame_ligand_transport
    if type(transport) is not AmeLigandTransport:
        raise PaperBaselineError("AME ligand transport identity is missing")
    try:
        parsed = ame_ligand_transport_from_mapping(transport.to_mapping())
    except PaperBaselineError as error:
        raise PaperBaselineError("AME ligand transport identity drifted") from error
    if parsed != transport:
        raise PaperBaselineError("AME ligand transport identity drifted")
    if (
        transport.version != AME_LIGAND_TRANSPORT_VERSION
        or not transport.canonical_component_token
        or not transport.ligand_smiles
        or not transport.ligand_identity
        or transport.evaluator_ligand_chain != AME_EVALUATOR_LIGAND_CHAIN
        or transport.clash_selector.get("kind") != "canonical_chain"
        or transport.clash_selector.get("chain_id") != AME_EVALUATOR_LIGAND_CHAIN
        or transport.clash_selector.get("component_token")
        != transport.canonical_component_token
    ):
        raise PaperBaselineError("AME ligand transport identity drifted")
    if (
        transport.original_representation == "polymer_peptide"
        and transport.native_hetatm_claimed
    ):
        raise PaperBaselineError("polymer peptide cannot be claimed as native HETATM")
    if (
        transport.atom_membership_sha256 != transport.normalized_atom_membership_sha256
        or transport.coordinate_sha256 != transport.normalized_coordinate_sha256
        or transport.clash_membership_before_sha256 != transport.atom_membership_sha256
        or transport.clash_coordinate_before_sha256 != transport.coordinate_sha256
        or transport.clash_coordinate_after_sha256 != transport.coordinate_sha256
    ):
        raise PaperBaselineError("AME ligand membership or coordinates changed")
    _authenticate_ame_export_manifest(export, transport, request.cell.example_id)
    expected_batch = request.expected_batch
    if expected_batch is not None:
        target = expected_batch.target
        if (
            target.family != "ame"
            or target.example_id != request.cell.example_id
            or type(target.canonical_component_token) is not str
            or not target.canonical_component_token
            or type(target.ligand_identity) is not str
            or not target.ligand_identity
            or type(target.ligand_smiles) is not str
            or not target.ligand_smiles
            or target.canonical_component_token != transport.canonical_component_token
            or target.ligand_identity != transport.ligand_identity
            or target.ligand_smiles != transport.ligand_smiles
        ):
            raise PaperBaselineError("AME ligand identity/SMILES is missing")

def _authenticate_ame_export_manifest(
    export: PaperExportResult, transport: AmeLigandTransport, example_id: str
) -> None:
    try:
        payload = json.loads(Path(export.manifest.path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperBaselineError(
            "ame export manifest cannot be authenticated"
        ) from error
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "dive.paper.export_manifest.v1"
        or payload.get("family") != "ame"
        or payload.get("example_id") != example_id
        or payload.get("ame_ligand_transport") != transport.to_mapping()
    ):
        raise PaperBaselineError("ame export manifest cannot be authenticated")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise PaperBaselineError("ame export manifest cannot be authenticated")
    observed = {
        entry.get("kind"): entry
        for entry in artifacts
        if isinstance(entry, Mapping) and type(entry.get("kind")) is str
    }
    expected = {
        "complex_pdb": export.complex_pdb,
        "motif_positions": export.motif_positions,
    }
    if export.native_motif is not None:
        expected["native_motif"] = export.native_motif
    for kind, identity in expected.items():
        assert identity is not None
        if observed.get(kind) != {
            "kind": kind,
            "path": identity.path,
            "sha256": identity.sha256,
            "size_bytes": identity.size_bytes,
        }:
            raise PaperBaselineError("ame export manifest cannot be authenticated")

def _ame_eval_kwargs(request: PaperEvaluationRequest) -> dict[str, object]:
    export = request.export
    assert export.motif_positions is not None
    transport = export.ame_ligand_transport
    if type(transport) is not AmeLigandTransport:
        raise PaperBaselineError("AME ligand transport identity is missing")
    chains = _chains_in_file_order(Path(export.complex_pdb.path))
    if len(chains) < 2:
        raise PaperBaselineError("ame complex lacks exported ligand and protein chains")
    if AME_EVALUATOR_LIGAND_CHAIN not in chains:
        raise PaperBaselineError("canonical AME ligand chain is missing from complex")
    binder_chain = chains[-1]
    target_chains = list(chains[:-1])
    frozen = _frozen_motif_positions(Path(export.motif_positions.path))
    rf3 = _registry_asset("ame", "rf3_checkpoint")
    rf3_exec = _registry_asset("ame", "rf3_executable")
    kwargs: dict[str, object] = {
        "pdb_file_path": export.complex_pdb.path,
        "target_pdb_path": export.complex_pdb.path,
        "motif_positions_path": export.motif_positions.path,
        "folding_model_specs": {
            "model_name": "rf3_latest",
            "rf3_checkpoint_path": rf3["path"],
            "rf3_checkpoint_sha256": rf3["sha256"],
            "rf3_executable_path": rf3_exec["path"],
            "rf3_executable_sha256": rf3_exec["sha256"],
        },
        "tmp_path": str(Path(request.work_dir)),
        "target_pdb_chain": target_chains,
        "sequence_types": [_AME_SEQUENCE_TYPE],
        "interface_cutoff": 6.0,
        "is_target_ligand": True,
        "inverse_folding_model": "ligand_mpnn",
        "gen_target_chain": target_chains,
        "binder_chain": binder_chain,
        "num_redesign_seqs": _AME_REDESIGNS,
        "fixed_residues_override": frozen,
        "ligand_smiles": transport.ligand_smiles,
        "ligand_component_token": transport.canonical_component_token,
        "ligand_clash_selector": dict(transport.clash_selector),
        "ligand_original_representation": transport.original_representation,
        "ligand_native_hetatm_claimed": transport.native_hetatm_claimed,
        "ligand_sequence": transport.sequence,
        "ligand_coordinate_sha256": transport.coordinate_sha256,
        "clash_membership_before_sha256": transport.clash_membership_before_sha256,
        "clash_membership_after_sha256": transport.clash_membership_after_sha256,
    }
    if export.native_motif is not None:
        kwargs["motif_pdb_path"] = export.native_motif.path
    return kwargs

def _frozen_motif_positions(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PaperBaselineError("AME motif positions are missing")
    chains = payload.get("chain_id")
    residues = payload.get("residue_index")
    if not isinstance(chains, list) or not isinstance(residues, list) or not chains:
        raise PaperBaselineError("AME motif positions are missing")
    if len(chains) != len(residues):
        raise PaperBaselineError("AME motif identity drifted")
    return [
        f"{chain}{int(resid)}" for chain, resid in zip(chains, residues, strict=True)
    ]

def _refuse_generated_as_native_motif(kwargs: Mapping[str, object]) -> None:
    generated = Path(str(kwargs["pdb_file_path"])).resolve(strict=False)
    motif = kwargs.get("motif_pdb_path")
    if type(motif) is not str or not motif:
        raise PaperBaselineError(
            "generated complex cannot authenticate as native motif"
        )
    if Path(motif).resolve(strict=False) == generated:
        raise PaperBaselineError(
            "generated complex cannot authenticate as native motif"
        )

def _canonical_clash_ligand_names(kwargs: Mapping[str, object]) -> list[str]:
    selector = kwargs.get("ligand_clash_selector")
    if not isinstance(selector, Mapping):
        raise PaperBaselineError("AME ligand clash selector drifted")
    kind = selector.get("kind")
    chain = selector.get("chain_id")
    token = selector.get("component_token")
    if (
        kind != "canonical_chain"
        or type(chain) is not str
        or chain != AME_EVALUATOR_LIGAND_CHAIN
        or type(token) is not str
        or not token
    ):
        raise PaperBaselineError("AME ligand clash selector drifted")
    return [f"CANONICAL_CHAIN_{chain}"]

def _ame_pinned_eval_config(kwargs: Mapping[str, object]) -> dict[str, object]:
    _refuse_generated_as_native_motif(kwargs)
    frozen = kwargs.get("fixed_residues_override")
    if not isinstance(frozen, list) or not frozen:
        raise PaperBaselineError("AME motif positions are missing")
    freeze = [str(item) for item in frozen]
    contig_atoms = _contig_atoms_from_native(
        Path(str(kwargs["motif_pdb_path"])), freeze
    )
    smiles = kwargs.get("ligand_smiles")
    token = kwargs.get("ligand_component_token")
    if type(smiles) is not str or not smiles or type(token) is not str or not token:
        raise PaperBaselineError("AME ligand identity/SMILES is missing")
    return {
        "metric": {
            "compute_binder_metrics": True,
            "compute_motif_binder_metrics": True,
            "binder_folding_method": "rf3_latest",
            "sequence_types": list(kwargs["sequence_types"]),
            "num_redesign_seqs": kwargs["num_redesign_seqs"],
            "interface_cutoff": kwargs["interface_cutoff"],
            "inverse_folding_model": kwargs["inverse_folding_model"],
        },
        "dataset": {
            "task_name": "paper_ame_cell",
            "motif_target_dict_cfg": {
                "paper_ame_cell": {
                    "target_path": str(kwargs["target_pdb_path"]),
                    "motif_pdb_path": str(kwargs["motif_pdb_path"]),
                    "contig_atoms": contig_atoms,
                    "ligand": _canonical_clash_ligand_names(kwargs),
                    "atom_selection_mode": "all_atom",
                    "motif_only": False,
                }
            },
        },
        "show_progress": False,
        "fixed_residues_override": freeze,
    }

_CATALYTIC_CONTIG_ATOMS = ("N", "CA", "C", "O", "CB")

def _contig_atoms_from_native(path: Path, freeze: Sequence[str]) -> str:
    atoms_by_res: dict[str, set[str]] = {str(item): set() for item in freeze}
    if not atoms_by_res:
        raise PaperBaselineError("AME motif positions are missing")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or len(line) < 26:
            continue
        chain = line[21].strip() or " "
        resid = int(line[22:26])
        name = line[12:16].strip()
        key = f"{chain}{resid}"
        if key in atoms_by_res and name in _CATALYTIC_CONTIG_ATOMS:
            atoms_by_res[key].add(name)
    missing = [key for key, atoms in atoms_by_res.items() if not atoms]
    if missing:
        raise PaperBaselineError("native motif residues are missing")
    return "; ".join(
        f"{key}: [{', '.join(atom for atom in _CATALYTIC_CONTIG_ATOMS if atom in atoms)}]"
        for key, atoms in atoms_by_res.items()
    )

def ame_rf3_binder_sequence(sequence: str, *, n_binder_residues: int) -> str:

    if type(sequence) is not str or not sequence:
        raise PaperBaselineError("AME RF3 binder sequence is empty")
    if type(n_binder_residues) is not int or n_binder_residues <= 0:
        raise PaperBaselineError("AME RF3 binder residue count is invalid")
    parts = [part for part in sequence.split(":") if part]
    matches = [part for part in parts if len(part) == n_binder_residues]
    if len(matches) != 1:
        raise PaperBaselineError(
            "AME RF3 binder sequence does not match the protein chain length"
        )
    return matches[0]

def _count_pdb_chain_residues(path: Path, chain: str) -> int:
    residues: set[int] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("ATOM") and len(line) > 25 and line[21] == chain:
            residues.add(int(line[22:26].strip()))
    return len(residues)

def _pinned_run_ame_eval(**kwargs):
    import sys

    config = _ame_pinned_eval_config(kwargs)
    frozen = list(config["fixed_residues_override"])
    source = str(EMERGENT_UPSTREAM_ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    previous_cwd = Path.cwd()
    previous_python = os.environ.get("PYTHON_EXEC")
    previous_ckpt = os.environ.get("RF3_CKPT_PATH")
    previous_exec = os.environ.get("RF3_EXEC_PATH")
    specs = kwargs.get("folding_model_specs")
    os.chdir(EMERGENT_UPSTREAM_ROOT)
    os.environ["PYTHON_EXEC"] = str(RUNTIME_PYTHON)
    if isinstance(specs, Mapping):
        if specs.get("rf3_checkpoint_path"):
            os.environ["RF3_CKPT_PATH"] = str(specs["rf3_checkpoint_path"])
        if specs.get("rf3_executable_path"):
            os.environ["RF3_EXEC_PATH"] = str(specs["rf3_executable_path"])
    original_clash = None
    original_run_rf3 = None
    original_inverse = None
    binder_metrics_module = None
    rf3_model = None
    try:
        from omegaconf import OmegaConf
        from proteinfoundation.evaluation.motif_binder_eval import (
            compute_motif_binder_metrics,
        )
        from proteinfoundation.evaluation import motif_binder_eval
        from proteinfoundation.metrics import binder_metrics as binder_metrics_module
        from proteinfoundation.utils import rf3_model as rf3_model_module

        original_freeze = motif_binder_eval.get_motif_fixed_residues
        original_clash = motif_binder_eval.check_ligand_clashes
        original_run_rf3 = rf3_model_module.run_rf3_eval
        original_inverse = binder_metrics_module.inverse_fold
        rf3_model = rf3_model_module
        smiles = str(kwargs["ligand_smiles"])
        selector = kwargs["ligand_clash_selector"]
        ligand_chain = str(selector["chain_id"])
        binder_chain = str(kwargs["binder_chain"])

        def get_fixed_residues_fn(_motif_info, _pdb_path, _binder_chain):
            return frozen

        def check_ligand_clashes_fn(
            pdb_file_path, clash_threshold=1.5, ligand_names=None
        ):
            return _check_ligand_clashes_by_canonical_chain(
                pdb_file_path,
                clash_threshold=clash_threshold,
                ligand_chain=ligand_chain,
                original=original_clash,
                work_dir=Path(kwargs["tmp_path"]),
                original_representation=str(kwargs["ligand_original_representation"]),
                native_hetatm_claimed=bool(kwargs["ligand_native_hetatm_claimed"]),
                sequence=str(kwargs.get("ligand_sequence") or ""),
                coordinate_sha256=str(kwargs.get("ligand_coordinate_sha256") or ""),
                membership_before_sha256=str(
                    kwargs.get("clash_membership_before_sha256") or ""
                ),
                membership_after_sha256=str(
                    kwargs.get("clash_membership_after_sha256") or ""
                ),
            )

        def inverse_fold_fn(*args, **fold_kwargs):
            sequences = original_inverse(*args, **fold_kwargs)
            pdb_path = fold_kwargs.get("pdb_file_path")
            if pdb_path is None:
                raise PaperBaselineError("AME inverse folding path is missing")
            n_res = _count_pdb_chain_residues(Path(str(pdb_path)), binder_chain)
            cleaned = []
            for item in sequences:
                if not isinstance(item, Mapping) or "seq" not in item:
                    raise PaperBaselineError(
                        "AME inverse-fold sequence payload is invalid"
                    )
                cleaned.append(
                    {
                        **dict(item),
                        "seq": ame_rf3_binder_sequence(
                            str(item["seq"]), n_binder_residues=n_res
                        ),
                    }
                )
            return cleaned

        def run_rf3_eval_fn(*args, **rf3_kwargs):
            rf3_kwargs["smiles"] = smiles
            pdb_path = rf3_kwargs.get("updated_pdb_path")
            binder_chain = rf3_kwargs.get("binder_chain_id")
            sequences = rf3_kwargs.get("binder_sequences")
            if pdb_path and binder_chain and sequences:
                n_res = _count_pdb_chain_residues(Path(str(pdb_path)), str(binder_chain))
                cleaned = []
                for item in sequences:
                    if not isinstance(item, Mapping) or "seq" not in item:
                        raise PaperBaselineError("AME RF3 sequence payload is invalid")
                    cleaned.append(
                        {
                            **dict(item),
                            "seq": ame_rf3_binder_sequence(
                                str(item["seq"]), n_binder_residues=n_res
                            ),
                        }
                    )
                rf3_kwargs["binder_sequences"] = cleaned
            return original_run_rf3(*args, **rf3_kwargs)

        motif_binder_eval.get_motif_fixed_residues = get_fixed_residues_fn
        motif_binder_eval.check_ligand_clashes = check_ligand_clashes_fn
        rf3_model_module.run_rf3_eval = run_rf3_eval_fn
        binder_metrics_module.inverse_fold = inverse_fold_fn
        sample_dir = Path(kwargs["tmp_path"]) / "cell"
        sample_dir.mkdir(parents=True, exist_ok=True)
        pdb_src = Path(kwargs["pdb_file_path"])
        pdb_dst = sample_dir / f"{sample_dir.name}.pdb"
        if not pdb_dst.exists():
            pdb_dst.write_bytes(pdb_src.read_bytes())
        hydra_config = {
            key: value
            for key, value in config.items()
            if key != "fixed_residues_override"
        }
        return compute_motif_binder_metrics(
            eval_config=OmegaConf.create(hydra_config),
            sample_root_paths=[str(sample_dir)],
            target_pdb_path=str(kwargs["target_pdb_path"]),
            target_pdb_chain=list(kwargs["target_pdb_chain"]),
            is_target_ligand=True,
        )
    finally:
        if "motif_binder_eval" in locals():
            motif_binder_eval.get_motif_fixed_residues = original_freeze
            if original_clash is not None:
                motif_binder_eval.check_ligand_clashes = original_clash
        if rf3_model is not None and original_run_rf3 is not None:
            rf3_model.run_rf3_eval = original_run_rf3
        if binder_metrics_module is not None and original_inverse is not None:
            binder_metrics_module.inverse_fold = original_inverse
        os.chdir(previous_cwd)
        if previous_python is None:
            os.environ.pop("PYTHON_EXEC", None)
        else:
            os.environ["PYTHON_EXEC"] = previous_python
        if previous_ckpt is None:
            os.environ.pop("RF3_CKPT_PATH", None)
        else:
            os.environ["RF3_CKPT_PATH"] = previous_ckpt
        if previous_exec is None:
            os.environ.pop("RF3_EXEC_PATH", None)
        else:
            os.environ["RF3_EXEC_PATH"] = previous_exec

def _ligand_resnames(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("HETATM") or len(line) < 20:
            continue
        name = line[17:20].strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names or ["LIG"]

def _check_ligand_clashes_by_canonical_chain(
    pdb_file_path,
    *,
    clash_threshold: float,
    ligand_chain: str,
    original,
    work_dir: Path,
    original_representation: str,
    native_hetatm_claimed: bool,
    sequence: str = "",
    coordinate_sha256: str | None = None,
    membership_before_sha256: str | None = None,
    membership_after_sha256: str | None = None,
):
    import secrets

    from atomworks.io.utils.io_utils import load_any
    from biotite.structure.io import save_structure

    from dive.benchmark.paper_export import PaperExportError

    if type(ligand_chain) is not str or ligand_chain != AME_EVALUATOR_LIGAND_CHAIN:
        raise PaperBaselineError("AME ligand clash selector drifted")
    if native_hetatm_claimed:
        raise PaperBaselineError("polymer peptide cannot be claimed as native HETATM")
    if original_representation not in {"polymer_peptide", "non_polymer"}:
        raise PaperBaselineError("AME ligand transport identity drifted")
    struct = load_any(pdb_file_path)[0].copy()
    chains = {str(chain) for chain in struct.chain_id.tolist()}
    if ligand_chain not in chains:
        raise PaperBaselineError("canonical AME ligand chain is missing from complex")
    ligand = struct[struct.chain_id == ligand_chain]
    try:
        _rewritten, hashes = _reclassify_ame_clash_ligand(ligand)
    except PaperExportError:
        return True
    if (
        original_representation == "polymer_peptide"
        and sequence
        and hashes["sequence"] != sequence
    ):
        return True
    if (
        membership_before_sha256
        and hashes["membership_before"] == membership_before_sha256
    ):
        if (
            membership_after_sha256
            and hashes["membership_after"] != membership_after_sha256
        ):
            return True
        if (
            coordinate_sha256
            and hashes["coordinate_before"] == coordinate_sha256
            and hashes["coordinate_after"] != coordinate_sha256
        ):
            return True
    struct.res_name[struct.chain_id == ligand_chain] = AME_CLASH_LIGAND_RESNAME
    destination = _refuse_denied(Path(work_dir))
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / f"ame_clash_{os.getpid()}_{secrets.token_hex(8)}.pdb"
    if snapshot.exists() or snapshot.is_symlink():
        raise PaperBaselineError("refusing to overwrite export artifact")
    save_structure(str(snapshot), struct)
    live = file_identity(snapshot)
    if live.size_bytes <= 0:
        raise PaperBaselineError("AME ligand membership or coordinates changed")
    return original(
        str(snapshot),
        clash_threshold=clash_threshold,
        ligand_names=[AME_CLASH_LIGAND_RESNAME],
    )

def _ame_row_mapping(payload: object) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    iloc = getattr(payload, "iloc", None)
    if iloc is not None:
        if len(payload) != 1:
            raise PaperBaselineError("ame production worker requires one cell")
        row = iloc[0]
        to_dict = getattr(row, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return dict(row)
    raise PaperBaselineError("ame eval payload is not a motif-ligand mapping")

def _same_index_vector(row: Mapping[str, object], name: str) -> tuple:
    value = row.get(name)
    if isinstance(value, (list, tuple)):
        sequence = value
    else:
        module = type(value).__module__
        if (
            value is not None
            and module.startswith("numpy")
            and hasattr(value, "tolist")
        ):
            sequence = value.tolist()
        else:
            raise PaperBaselineError(f"invalid vector length: {name}")
    if not isinstance(sequence, (list, tuple)) or len(sequence) != _AME_REDESIGNS:
        raise PaperBaselineError(f"invalid vector length: {name}")
    return tuple(sequence)

def _as_float(name: str, value: object) -> float:
    if type(value) is bool:
        raise PaperBaselineError(f"{name} must be a float")
    if type(value) is float:
        number = value
    elif type(value) is int:
        number = float(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise PaperBaselineError(f"{name} must be a float") from error
    if not math.isfinite(number):
        raise PaperBaselineError(f"{name} is non-finite")
    return number

def _as_bool(name: str, value: object) -> bool:
    if type(value) is bool:
        return value
    module = type(value).__module__
    if module == "numpy" and type(value).__name__.startswith("bool"):
        return bool(value)
    raise PaperBaselineError(f"{name} must be a boolean")

def _as_str(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PaperBaselineError(f"{name} must be a non-empty string")
    return value

def _structure_present(path: object) -> bool:
    return type(path) is str and bool(path)

def _ame_source_flags(
    row: Mapping[str, object],
    name: str,
    values: tuple,
    paths: tuple,
) -> tuple[bool, bool]:
    column = _AME_SOURCE_FLAG_COLUMNS[name]
    if column in row:
        flags = _same_index_vector(row, column)
        return (
            _as_bool(f"{name}_source_present", flags[0]),
            _as_bool(f"{name}_source_present", flags[1]),
        )
    derived: list[bool] = []
    for index, value in enumerate(values):
        present = _structure_present(paths[index])
        if type(value) is float and (
            value == _RF3_SENTINEL or value == 0.0 and not present
        ):
            present = False
        derived.append(present)
    return (derived[0], derived[1])

def _ame_redesigns_from_motif_ligand_eval(payload: object) -> tuple[AmeRedesign, ...]:
    row = _ame_row_mapping(payload)
    prefix = _AME_SEQUENCE_TYPE
    bb3 = tuple(
        _as_float("binder_scRMSD_bb3", item)
        for item in _same_index_vector(row, f"{prefix}_binder_scRMSD_bb3_all")
    )
    motif = tuple(
        _as_float("motif_all_atom_rmsd", item)
        for item in _same_index_vector(row, f"{prefix}_motif_rmsd_pred_all")
    )
    exact = tuple(
        _as_bool("exact_motif_sequence_recovery", item)
        for item in _same_index_vector(row, f"{prefix}_correct_motif_sequence_all")
    )
    clash = tuple(
        _as_bool("ligand_clash", item)
        for item in _same_index_vector(row, f"{prefix}_has_ligand_clashes_all")
    )
    ipae = tuple(
        _as_float("min_ipAE", item)
        for item in _same_index_vector(row, f"{prefix}_complex_min_ipAE_all")
    )
    sequences = tuple(
        _as_str("sequence", item)
        for item in _same_index_vector(row, f"{prefix}_sequence_all")
    )
    paths = _same_index_vector(row, f"{prefix}_complex_pdb_path_all")
    if any(value == _RF3_SENTINEL for value in (*bb3, *motif, *ipae)):
        raise PaperBaselineError("upstream_numeric_sentinel: RF3 sentinel 100")
    if not all(_structure_present(path) for path in paths):
        raise PaperBaselineError("upstream_numeric_sentinel: absent structure")
    flags = {
        "binder_scRMSD_bb3": _ame_source_flags(row, "binder_scRMSD_bb3", bb3, paths),
        "motif_all_atom_rmsd": _ame_source_flags(
            row, "motif_all_atom_rmsd", motif, paths
        ),
        "exact_motif_sequence_recovery": _ame_source_flags(
            row, "exact_motif_sequence_recovery", exact, paths
        ),
        "ligand_clash": _ame_source_flags(row, "ligand_clash", clash, paths),
        "min_ipAE": _ame_source_flags(row, "min_ipAE", ipae, paths),
    }
    if any(not flag for group in flags.values() for flag in group):
        raise PaperBaselineError(
            "upstream_numeric_sentinel: source field absent or defaulted"
        )
    rows = []
    for index in range(_AME_REDESIGNS):
        rows.append(
            AmeRedesign(
                f"{prefix}-{index}",
                sequences[index],
                bb3[index],
                motif[index],
                exact[index],
                clash[index],
                ipae[index],
                flags["binder_scRMSD_bb3"][index],
                flags["motif_all_atom_rmsd"][index],
                flags["exact_motif_sequence_recovery"][index],
                flags["ligand_clash"][index],
                flags["min_ipAE"][index],
                {},
            )
        )
    return tuple(rows)

def _assert_two_ame_vectors(raw: Mapping[str, object]) -> None:
    for name in (
        "binder_scRMSD_bb3",
        "motif_all_atom_rmsd",
        "min_ipAE",
    ):
        values = raw.get(name)
        if not isinstance(values, tuple) or len(values) != _AME_REDESIGNS:
            raise PaperBaselineError(f"invalid vector length: {name}")
        if any(type(item) is not float or not math.isfinite(item) for item in values):
            raise PaperBaselineError(f"{name} is non-finite")
    for name in (
        "exact_motif_sequence_recovery",
        "ligand_clash",
        *REQUIRED_AME_SOURCE_FLAGS,
    ):
        values = raw.get(name)
        if not isinstance(values, tuple) or len(values) != _AME_REDESIGNS:
            raise PaperBaselineError(f"invalid vector length: {name}")
        if any(item is not True and item is not False for item in values):
            raise PaperBaselineError(f"{name} must be a boolean")
    for name in ("redesign_id", "sequence"):
        values = raw.get(name)
        if not isinstance(values, tuple) or len(values) != _AME_REDESIGNS:
            raise PaperBaselineError(f"invalid vector length: {name}")
    if raw.get("redesign_count") != _AME_REDESIGNS:
        raise PaperBaselineError("ame production worker requires two redesigns")
    if raw.get("selection_rule") != _AME_SELECTION:
        raise PaperBaselineError("ame selection rule drifted")

def evaluate_paper_antibody(request: PaperEvaluationRequest) -> PaperEvaluationResult:

    _authenticate_antibody_request(request)
    export = request.export
    predicted = export.predicted_atoms
    native = export.native_atoms
    roles = export.antibody_roles
    if predicted is None or native is None or roles is None:
        raise PaperBaselineError("antibody export is missing atom records")
    evaluation = evaluate_antibody(
        AntibodyComplex(
            native=native,
            predicted=predicted,
            roles=roles,
            evaluator_manifest_hash=PRODUCTION_EVALUATOR_REGISTRY_SHA256,
        )
    )
    primary = evaluation.primary
    if isinstance(primary, MetricUnavailable):
        raise PaperBaselineError(f"antibody primary unavailable: {primary.reason_code}")
    if type(primary.value) is not float or not math.isfinite(primary.value):
        raise PaperBaselineError("cdr_h3_ca_rmsd is non-finite")
    if primary.name != "cdr_h3_ca_rmsd" or primary.direction != "lower":
        raise PaperBaselineError("antibody primary drifted")
    raw: dict[str, object] = {
        "cdr_h3_ca_rmsd": primary.value,
        "redesign_count": 0,
        "selection_rule": _ANTIBODY_SELECTION,
        "method": _ANTIBODY_METHOD,
    }
    for name in _ANTIBODY_SECONDARIES:
        metric = evaluation.secondary.get(name)
        if metric is None:
            raise PaperBaselineError(f"antibody secondary missing: {name}")
        raw[name] = metric
    return PaperEvaluationResult(
        family="antibody",
        primary=primary,
        raw=raw,
        cell_id=request.cell.cell_id,
        cell=request.cell,
        method=_ANTIBODY_METHOD,
    )

def _authenticate_antibody_request(request: PaperEvaluationRequest) -> None:
    if type(request) is not PaperEvaluationRequest:
        raise PaperBaselineError("antibody request must be a PaperEvaluationRequest")
    cell = request.cell
    if type(cell) is not PaperCell or cell.family != "antibody":
        raise PaperBaselineError("antibody adapter requires an antibody cell")
    if (
        request.registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256
        or cell.evaluator_registry_sha256 == FIXTURE_EVALUATOR_REGISTRY_SHA256
    ):
        raise PaperBaselineError("fixture hash cannot authenticate as production")
    if (
        request.registry_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256
        or cell.evaluator_registry_sha256 != PRODUCTION_EVALUATOR_REGISTRY_SHA256
    ):
        raise PaperBaselineError("production evaluator registry hash drifted")
    expected = paper_cell_identity(
        panel=cell.panel,
        arm=cell.arm,
        family=cell.family,
        partition=cell.partition,
        parent_id=cell.parent_id,
        example_id=cell.example_id,
        seed=cell.seed,
        sample_index=cell.sample_index,
        generation_replicas=cell.generation_replicas,
        steps=cell.steps,
        condition_count=cell.condition_count,
        denoiser_equivalent_passes=cell.denoiser_equivalent_passes,
        checkpoint=cell.checkpoint,
        splice_report_sha256=cell.splice_report_sha256,
        evaluator_registry_sha256=cell.evaluator_registry_sha256,
        view_semantic_hash=cell.view_semantic_hash,
        code_commit=cell.code_commit,
    )
    if cell.cell_id != expected:
        raise PaperBaselineError("cell identity drifted")
    export = request.export
    if type(export) is not PaperExportResult or export.family != "antibody":
        raise PaperBaselineError("antibody adapter requires an antibody export")
    live = file_identity(Path(export.complex_pdb.path))
    if live.sha256 != export.complex_pdb.sha256 or live.size_bytes != (
        export.complex_pdb.size_bytes
    ):
        raise PaperBaselineError("export identity drifted")
    if type(export.antibody_roles) is not FrozenAntibodyRoles:
        raise PaperBaselineError("antibody adapter requires frozen antibody roles")
    if not export.antibody_roles.heavy_chain:
        raise PaperBaselineError("antibody adapter requires frozen heavy-chain role")
    if not export.antibody_roles.antigen_chains:
        raise PaperBaselineError("antibody adapter requires frozen antigen role")
    if not export.antibody_roles.cdr_h3:
        raise PaperBaselineError("antibody adapter requires frozen CDR-H3 role")
    try:
        cdr_h3 = tuple(
            item
            if isinstance(item, ResidueKey)
            else ResidueKey(
                str(item[0]),
                int(item[1]),
                str(item[2]) if len(item) > 2 else "",
            )
            for item in export.antibody_roles.cdr_h3
        )
    except (IndexError, TypeError, ValueError) as error:
        raise PaperBaselineError(
            "antibody adapter requires insertion-aware CDR-H3 keys"
        ) from error
    if any(type(item.insertion_code) is not str for item in cdr_h3):
        raise PaperBaselineError(
            "antibody adapter requires insertion-aware CDR-H3 keys"
        )
    if type(export.predicted_atoms) is not tuple:
        raise PaperBaselineError("antibody export is missing atom records")
    if type(export.native_atoms) is not tuple:
        raise PaperBaselineError("antibody export is missing atom records")
    if not export.predicted_atoms or not export.native_atoms:
        raise PaperBaselineError("antibody export is missing atom records")
    for atom in (*export.predicted_atoms, *export.native_atoms):
        if type(getattr(atom, "insertion_code", None)) is not str:
            raise PaperBaselineError(
                "antibody adapter requires insertion-aware atom identities"
            )

def _refuse_denied(path: Path) -> Path:
    resolved = Path(path).resolve(strict=False)
    for denied in DENIED_PREFIXES:
        prefix = Path(denied).resolve(strict=False)
        if resolved == prefix or prefix in resolved.parents:
            raise PaperBaselineError(f"root is denied: {resolved}")
    return resolved

def _registry_asset(family: str, role: str) -> Mapping[str, object]:
    import yaml

    from dive.benchmark.paper_registry import DEFAULT_PAPER_EVALUATOR_REGISTRY_PATH

    raw = yaml.safe_load(
        DEFAULT_PAPER_EVALUATOR_REGISTRY_PATH.read_text(encoding="utf-8")
    )
    if not isinstance(raw, Mapping):
        raise PaperBaselineError("production evaluator registry is invalid")
    families = raw.get("families")
    if not isinstance(families, Mapping):
        raise PaperBaselineError("production evaluator registry is invalid")
    entry = families.get(family)
    if not isinstance(entry, Mapping):
        raise PaperBaselineError(f"{family} registry family is missing")
    assets = entry.get("assets")
    if not isinstance(assets, list):
        raise PaperBaselineError(f"{family} registry assets are missing")
    for item in assets:
        if isinstance(item, Mapping) and item.get("role") == role:
            return item
    raise PaperBaselineError(f"{family} registry asset {role} is missing")
