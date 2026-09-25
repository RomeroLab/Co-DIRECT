
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_CODIRECT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(_CODIRECT / "src") not in _sys.path:
    _sys.path.insert(0, str(_CODIRECT / "src"))
from dive.codirect_paths import benchv2_root, joined, proteina_root, proteina_src

ROOT = benchv2_root()
SPLITS = ROOT / "splits" / "v6"
ANTIBODY_STRUCTURES = joined(
    "CODIRECT_STRUCTURE_ROOT", "sources", "sabdab2-v0.1.0", "splits_final")
UPSTREAM = proteina_root()

AA1 = "ARNDCQEGHILKMFPSTWYV"
RESNAME3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

def ame_motif_record(*, specialist: bool, task: dict, places=None) -> dict:

    keys = sorted((task or {}).get("motif") or {})
    if specialist and places:
        return {
            "motif_positions": places,
            "motif_note": (
                "catalytic residues pinned inside the designed 180-mer at "
                "E3 .trb design_index; native identities on those rows. "
                "LigandMPNN not used. Contig walk not used."
            ),
            "motif_task_keys": keys,
        }
    note = (
        "unindexed: no pin; theozyme is x_motif extra tokens only"
        if specialist else
        "unindexed: complexa.ckpt v1 has enable_motif=False; the theozyme "
        "enters as target-axis coordinates only"
    )
    return {
        "motif_positions": None,
        "motif_note": note,
        "motif_task_keys": keys,
    }

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def build_batch(*, family, path, roles, example_id, seed, crop_size, target_budget,
                max_design_fraction):

    import pandas as pd
    from proteinfoundation.datasets.structure_data import structure_collate_fn
    from types import MappingProxyType

    from dive.benchmark.contracts import file_identity
    from dive.benchmark.paper_batches import PaperBatch
    from dive.benchmark.paper_inputs import PaperTargetRecord
    from dive.data.dataset import RoleAwareStructureDataset
    from dive.data.loaders import LOADER_COLUMNS, assert_loader_manifest_is_clean
    from dive.data.pipeline import family_loader_kwargs, family_transforms
    from dive.training.runtime import normalize_generation_masks

    identity = file_identity(Path(path))
    frame = pd.DataFrame(
        [{"example_id": example_id, "path": str(path), **roles}],
        columns=list(LOADER_COLUMNS),
    )
    assert_loader_manifest_is_clean(frame)
    transform_family = "binder" if family == "ame" else family
    dataset = RoleAwareStructureDataset(
        frame,
        transforms=family_transforms(
            transform_family,
            crop_size=crop_size,
            seed=seed,
            max_design_fraction=max_design_fraction,
            target_budget=target_budget,
        ),
        **family_loader_kwargs(family),
    )
    sample = dataset[0]
    if sample is None:
        raise RuntimeError("loader exclusion: the staged structure did not survive")
    collated = structure_collate_fn([sample])
    normalize_generation_masks(collated)
    target = PaperTargetRecord(
        parent_id=example_id,
        example_id=example_id,
        family=family,
        partition="validation",
        strict=True,
        source_path=str(path),
        source_sha256=identity.sha256,
        role_payload=dict(roles),
        loader_manifest_sha256=identity.sha256,
    )
    return PaperBatch(target, crop_size, seed, MappingProxyType(collated), identity, False)

def diagnose_loader(path, roles, family, crop_size, target_budget, max_design_fraction, seed):

    import biotite.structure as struc
    import numpy as np
    from atomworks.io.transforms.atom_array import remove_waters
    from proteinfoundation.datasets.structure_data import atomarray_to_atom37, load_structure

    from dive.data.dataset import _keep_role_chains
    from dive.data.pipeline import family_transforms

    row = {"example_id": "diagnose", "path": str(path), **roles}
    atoms = remove_waters(load_structure(str(path)))
    atoms = _keep_role_chains(atoms, row)
    sample = atomarray_to_atom37(atoms, sample_id="diagnose", atomworks_data=None)
    for column in ("generated", "context", "target"):
        setattr(sample, column, str(row[column]))
    transform_family = "binder" if family == "ame" else family
    for transform in family_transforms(
        transform_family, crop_size=crop_size, seed=seed,
        max_design_fraction=max_design_fraction, target_budget=target_budget,
    ):
        sample = transform(sample)
    return sample

def export(sample, batch, out_dir: Path, stem: str, design_chain: str | None = None):
    from dive.benchmark.paper_export import _pdb_text, _residue_rows

    rows = _residue_rows(sample, batch)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = out_dir / f"{stem}.pdb"
    pdb_path.write_text(_pdb_text(rows))
    if design_chain:
        designed = [r for r in rows if r["chain"] == design_chain]
    else:
        designed = [r for r in rows if r["generated"] or r["fixed"]]
    if not designed:
        raise RuntimeError("no designed residues in decoded sample")
    sequence = "".join(RESNAME3.get(str(r["resname"]), "X") for r in designed)
    fasta_path = out_dir / f"{stem}.fasta"
    fasta_path.write_text(f">{stem}|design|{len(sequence)}\n{sequence}\n")
    return pdb_path, fasta_path, sequence, len(rows)

def load_tasks(family):
    payload = json.loads((SPLITS / f"test_{family}.json").read_text())
    return payload["tasks"]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=("binder", "ame", "antibody"))
    parser.add_argument("--arm", required=True, choices=("frozen", "lora"))
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--trunk", default=None)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--target-budget", type=int, default=96)
    parser.add_argument("--max-design-fraction", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default=None, help="i/n")
    parser.add_argument("--only", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-contig-length", action="store_true",
                        help="AME: stage the designed chain at the task contig "
                             "length instead of the frozen 180")
    parser.add_argument("--ame-specialist", action="store_true",
                        help="load complexa_ame.ckpt natively (ligand+motif "
                             "pathways on) instead of the common checkpoint")
    parser.add_argument("--specialist-ckpt", default=None,
                        help="defaults to $PROTEINA_COMPLEXA_ROOT/ckpts/complexa_ame.ckpt")
    parser.add_argument("--specialist-ae", default=None,
                        help="defaults to $PROTEINA_COMPLEXA_ROOT/ckpts/complexa_ame_ae.ckpt")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--anticipate", action="store_true",
                        help="wrap generate in the shared anticipating sampler")
    parser.add_argument("--alpha", type=float, default=1.28)
    parser.add_argument("--interface-router", action="store_true",
                        help="shared residue-wise leadership router from live Cα–partner distance")
    parser.add_argument("--learned-router", default=None,
                        help="checkpoint from family joint train (router_stepN.pt)")
    parser.add_argument("--partner-only", action="store_true",
                        help="anticipation probe advances only the other modality")
    parser.add_argument("--alpha-latent", type=float, default=None,
                        help="separate anticipation weight for local_latents")
    parser.add_argument("--nsteps", type=int, default=None,
                        help="override diffusion steps; omit for the paper 400")
    parser.add_argument("--router-on-self-cond", action="store_true",
                        help="router scales self-conditioning; a missing "
                             "prediction is zeros, not the live state")
    parser.add_argument("--condition-router", action="store_true",
                        help="write router-gated values into trunk x_sc")
    parser.add_argument("--clean-hat", action="store_true",
                        help="extra pass reads x_sc = x_t+(1-t)v, the flow's "
                             "own clean endpoint; shared across families")
    parser.add_argument("--response-router", action="store_true",
                        help="weight the extra-pass by ||dv||/||v|| per residue, "
                             "not a native-contact MLP")
    parser.add_argument("--advance-frac", type=float, default=1.0,
                        help="extra-pass partner move as a fraction of dt; "
                             "0.5 is explicit midpoint (RK2), 1.0 is Euler")
    parser.add_argument("--alpha-schedule", choices=("t", "one_minus_t"),
                        default=None,
                        help="multiply extra-pass alpha by t or 1-t each step")
    parser.add_argument("--extra-pass-gate", default=None,
                        help="checkpoint that keeps extra-pass only where it "
                             "beats Jacobi on native velocity")
    parser.add_argument("--unclipped-extra-pass", action="store_true",
                        help="Gauss-Seidel: apply the full extra-pass delta, "
                             "not the 0.5-norm clip")
    parser.add_argument("--contact-reciprocal", action="store_true",
                        help="latent answers Cα snapped onto the partner; "
                             "backbone answers that latent's clean commitment "
                             "on designed near rows only. Motif rows stay Jacobi")
    parser.add_argument("--contact-clean", action="store_true",
                        help="extra pass shows the backbone's capped Jacobi "
                             "endpoint and replaces only the near-partner "
                             "latent velocity. Backbone velocity stays Jacobi")
    parser.add_argument("--contact-rigid", action="store_true",
                        help="extra pass translates the whole contact patch "
                             "by one vector toward the partner and replaces "
                             "only the latent velocity there")
    parser.add_argument("--snap-anticipation", action="store_true",
                        help="per-residue router who; backbone leader commits "
                             "the ligand snap, latent leader commits its clean step")
    parser.add_argument("--contact-snap-far-clean", action="store_true",
                        help="near rows: latent answers the ligand snap; "
                             "far rows: latent answers the capped Jacobi "
                             "backbone. Backbone velocity stays Jacobi")
    parser.add_argument("--contact-snap", action="store_true",
                        help="extra pass shows partner-near Cα snapped onto "
                             "the ligand and replaces only the latent velocity "
                             "there. Not a dt·v step")
    parser.add_argument("--residue-commit", action="store_true",
                        help="one probe: per residue the lower router weight "
                             "commits its clean endpoint and the higher one "
                             "ships the recomputed velocity. Not a gain")
    parser.add_argument("--commit-clean", action="store_true",
                        help="extra pass shows the other channel's capped "
                             "clean endpoint at t=1 and ships that answer "
                             "unscaled. Not a per-residue velocity gain")
    parser.add_argument("--leader-commit-probe", action="store_true",
                        help="extra-pass advances leader residues (w=0) and "
                             "holds followers; router then blends only on "
                             "followers. Stackelberg in the probe, not just "
                             "the blend")
    parser.add_argument("--invert-router", action="store_true",
                        help="swap leader/follower on designed rows "
                             "(peak-w). Interface commits; scaffold answers")
    parser.add_argument("--recycle-k", type=int, default=0,
                        help="re-evaluate trunk at the same x_t with "
                             "x_sc=x_t+(1-t)v this many times (Disco/AF "
                             "recycle). 0 is a no-op")
    parser.add_argument("--role-router", action="store_true",
                        help="split leadership by logit_iface: CA leads at "
                             "the interface, latents follow; reverse in bulk. "
                             "Disco cross-modal roles, no new loss")
    parser.add_argument("--persist-sc", action="store_true",
                        help="add extra-pass remaining path onto router "
                             "x_sc next step; does not replace the condition")
    parser.add_argument("--iface-cutoff", type=float, default=1.0)
    parser.add_argument("--iface-alpha-ca", type=float, default=0.0)
    parser.add_argument("--iface-alpha-lat", type=float, default=4.0)
    parser.add_argument("--bulk-alpha-ca", type=float, default=1.28)
    parser.add_argument("--bulk-alpha-lat", type=float, default=1.28)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.ame_specialist:

        os.environ["USE_V2_COMPLEXA_ARCH"] = "True"
    else:
        os.environ.pop("USE_V2_COMPLEXA_ARCH", None)
    sys.path.insert(0, str(proteina_src()))
    sys.path.insert(0, str(_CODIRECT / "src"))

    import proteinfoundation.patches.atomworks_patches

    from dive.benchmark import paper_generation
    from dive.benchmark.paper_baseline import PAPER_SEEDS, PaperArm
    arm = PaperArm.BASE_COMMON_EXACT_V1 if args.arm == "frozen" else PaperArm.BASE_COMMON_FINETUNE_V1
    if args.arm == "lora":
        if not args.trunk:
            raise SystemExit("--trunk is required for the lora arm")
        paper_generation.FINETUNE_TRUNK = str(args.trunk)

    label = "proteina_frozen" if args.arm == "frozen" else "proteina_lora"
    out_root = Path(args.out_root) if args.out_root else ROOT / "generation" / label / args.family
    out_root.mkdir(parents=True, exist_ok=True)

    from dive.generation import stage

    tasks = load_tasks(args.family)
    if args.only:
        wanted = set(args.only.split(","))
        key = {"binder": "name", "ame": "task_id", "antibody": "INSTANCE"}[args.family]
        tasks = [t for t in tasks if str(t[key]) in wanted]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        tasks = [t for index, t in enumerate(tasks) if index % n == i]
    if args.limit:
        tasks = tasks[: args.limit]

    lengths = {}
    if args.family == "binder":
        lengths = json.loads((ROOT / "generation" / "binder_lengths.json").read_text())["lengths"]

    staging_dir = ROOT / "generation" / "proteina_staged" / args.family
    staging_dir.mkdir(parents=True, exist_ok=True)

    if args.specialist_ckpt is None:
        args.specialist_ckpt = str(proteina_root() / "ckpts" / "complexa_ame.ckpt")
    if args.specialist_ae is None:
        args.specialist_ae = str(proteina_root() / "ckpts" / "complexa_ame_ae.ckpt")
    contract = None
    if args.ame_specialist:
        checkpoint_sha = sha256_file(Path(args.specialist_ckpt))
        autoencoder_sha = sha256_file(Path(args.specialist_ae))
    else:
        from dive.emergent_contract import EmergentContract
        contract = EmergentContract.from_yaml(Path("configs/emergent/resources.yaml"))
        checkpoint_sha = sha256_file(Path(contract.common_checkpoint))
        autoencoder_sha = sha256_file(Path(contract.autoencoder_checkpoint))
    trunk_sha = sha256_file(Path(args.trunk)) if args.arm == "lora" else None

    store = ROOT / "generation" / "proteina_store"
    store.mkdir(parents=True, exist_ok=True)
    model = report = None
    if not args.dry_run:
        started = time.perf_counter()
        if args.ame_specialist:

            from dive.generation.specialist import build_ame_specialist
            model, spec_report = build_ame_specialist(
                args.specialist_ckpt, args.specialist_ae, store)
            checkpoint_sha = sha256_file(Path(args.specialist_ckpt))
            autoencoder_sha = sha256_file(Path(args.specialist_ae))
            print(f"[model] AME specialist loaded: "
                  f"flags={spec_report['concat_feature_flags']} "
                  f"lora_tensors={spec_report['lora_tensors_in_checkpoint']}", flush=True)

            if args.trunk:
                import torch
                payload = torch.load(args.trunk, map_location="cpu", weights_only=False)
                state = payload["nn"] if isinstance(payload, dict) and "nn" in payload else payload
                try:
                    result = model.nn.load_state_dict(state, strict=True)
                    print(f"[trunk] specialist nn loaded from {args.trunk} "
                          f"missing={list(getattr(result,'missing_keys',[]))} "
                          f"unexpected={list(getattr(result,'unexpected_keys',[]))}", flush=True)
                except RuntimeError as exc:
                    print("[trunk] SKIP: r8-ame trunk is the common-checkpoint "
                          f"architecture, not complexa_ame. {exc}".split("\n")[0],
                          flush=True)
            class _R:
                live_parameter_sha256 = paper_generation.hash_live_parameters(model)
            report = _R()
        else:
            model, report = paper_generation.build_model_for_arm(arm, contract, store_dir=store)
        model = model.to("cuda").eval()
        print(f"[model] {arm.value} loaded in {time.perf_counter() - started:.1f}s "
              f"live_parameter_sha256={report.live_parameter_sha256}", flush=True)

    seeds = list(PAPER_SEEDS)[: args.candidates]
    summary = []
    for task in tasks:
        key = {"binder": "name", "ame": "task_id", "antibody": "INSTANCE"}[args.family]
        task_id = str(task[key])
        record = {"task_id": task_id, "family": args.family, "arm": arm.value}
        try:
            if args.family == "binder":
                staged = stage.stage_binder(
                    task, staging_dir, design_length=lengths[task_id]["binder_length"]
                )
            elif args.family == "ame":
                contig_n = None
                if args.use_contig_length and task.get("raw_contig"):
                    from dive.generation.indexed_motif import contig_length
                    contig_n = contig_length(task["raw_contig"])
                staged = stage.stage_ame(task, staging_dir, design_length=contig_n)
            else:
                staged = stage.stage_antibody(task, ANTIBODY_STRUCTURES)
            roles = {"generated": staged.generated, "context": staged.context,
                     "target": staged.target}
            record.update({"roles": roles, "design_length": staged.design_length,
                           "staged_path": str(staged.path), "staging": staged.notes})
        except Exception as error:
            record.update({"status": "stage_failed", "reason": type(error).__name__,
                           "detail": str(error)[:600]})
            print(f"[stage-fail] {task_id}: {type(error).__name__}: {str(error)[:200]}", flush=True)
            summary.append(record)
            continue

        if args.dry_run:
            try:
                probe = diagnose_loader(
                    staged.path, roles, args.family, args.crop_size,
                    args.target_budget, args.max_design_fraction, seeds[0],
                )
                record.update({
                    "status": "dry_ok",
                    "n_residues": int(probe.design_mask.shape[0]),
                    "n_design": int(probe.design_mask.sum()),
                    "n_target": int(probe.target_mask.sum(dim=-1).bool().sum()),
                })
            except Exception as error:
                record.update({"status": "dry_failed", "reason": type(error).__name__,
                               "detail": str(error)[:600]})
            print(f"[dry] {task_id}: {record.get('status')} "
                  f"n={record.get('n_residues')} design={record.get('n_design')} "
                  f"target={record.get('n_target')} {record.get('detail','')}", flush=True)
            summary.append(record)
            continue

        task_dir = out_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for index, seed in enumerate(seeds):
            stem = f"{task_id}_cand{index}_seed{seed}"
            entry = {"candidate_index": index, "seed": seed}
            started = time.perf_counter()
            import types as _types
            _st = _CODIRECT / "src" / "dive" / "stackelberg"
            if "dive.stackelberg" not in sys.modules:
                _pkg = _types.ModuleType("dive.stackelberg")
                _pkg.__path__ = [str(_st)]
                sys.modules["dive.stackelberg"] = _pkg
                import dive as _dive
                _dive.stackelberg = _pkg
            from dive.stackelberg.generate_skip import existing_design_pdb
            prior = existing_design_pdb(task_dir, stem)
            if prior is not None:
                print(f"[skip] {task_id} cand{index} exists", flush=True)
                candidates.append({**entry, "status": "ok", "path": str(prior)})
                continue
            try:
                batch = build_batch(
                    family=args.family, path=staged.path, roles=roles,
                    example_id=task_id, seed=seed, crop_size=args.crop_size,
                    target_budget=args.target_budget,
                    max_design_fraction=args.max_design_fraction,
                )
                raw = dict(batch.tensors)
                if args.ame_specialist and args.family == "ame":

                    from dive.generation.ligand_staging import (
                        build_features, inject_conditioning,
                    )
                    mf, lf, nres = build_features(task)
                    mf.setup(nres)
                    lf.setup(nres)
                    feats = {}
                    mf(feats, 0)
                    lf(feats, 0)
                    raw = inject_conditioning(raw, feats)
                    import types as _types
                    _st = _CODIRECT / "src" / "dive" / "stackelberg"
                    if "dive.stackelberg" not in sys.modules:
                        _pkg = _types.ModuleType("dive.stackelberg")
                        _pkg.__path__ = [str(_st)]
                        sys.modules["dive.stackelberg"] = _pkg
                        import dive as _dive
                        _dive.stackelberg = _pkg
                    from dive.stackelberg.ame_e3_match import (
                        e3_motif_positions, freeze_catalyst_rows, plant_residue_types,
                    )
                    from dive.generation.indexed_motif import native_types, placements_in_length
                    try:
                        places = e3_motif_positions(str(task["task_id"]), int(index))
                    except (FileNotFoundError, ValueError):
                        places = placements_in_length(
                            task["raw_contig"], task["motif"], length=180)
                    types = native_types(task["input_pdb"], task["motif"])
                    plant_residue_types(raw, places, types)
                    freeze_catalyst_rows(raw, places)
                    entry["motif_places"] = places
                tensors = {k: (v.to("cuda") if hasattr(v, "to") else v)
                           for k, v in raw.items()}
                from contextlib import ExitStack, nullcontext
                hook = ExitStack()
                if args.anticipate:
                    import types
                    from pathlib import Path as _P
                    _st = _CODIRECT / "src" / "dive" / "stackelberg"
                    if "dive.stackelberg" not in sys.modules:
                        _pkg = types.ModuleType("dive.stackelberg")
                        _pkg.__path__ = [str(_st)]
                        sys.modules["dive.stackelberg"] = _pkg
                        import dive as _dive
                        _dive.stackelberg = _pkg
                    from dive.stackelberg.sampling import anticipating_sampler
                    router = None
                    one = {k: (v[:1] if hasattr(v, "dim") and v.dim() >= 1 else v)
                           for k, v in tensors.items()}
                    if args.learned_router:
                        from types import SimpleNamespace
                        from dive.stackelberg.learned_router import make_generation_router
                        router = make_generation_router(
                            SimpleNamespace(
                                learned_router=args.learned_router,
                                role_router=bool(args.role_router),
                                interface_router=False,
                                iface_cutoff=args.iface_cutoff,
                                iface_alpha_ca=args.iface_alpha_ca,
                                iface_alpha_lat=args.iface_alpha_lat,
                                bulk_alpha_ca=args.bulk_alpha_ca,
                                bulk_alpha_lat=args.bulk_alpha_lat),
                            one)
                    elif args.interface_router:
                        from dive.stackelberg.interface_router import (
                            interface_router, partner_atoms_from_batch)

                        partner = partner_atoms_from_batch(one)
                        designed = tensors.get("generated_mask", tensors["mask"]).bool()
                        router = interface_router(
                            partner_coords=partner, designed=designed,
                            cutoff_nm=args.iface_cutoff,
                            interface={"bb_ca": args.iface_alpha_ca,
                                       "local_latents": args.iface_alpha_lat},
                            bulk={"bb_ca": args.bulk_alpha_ca,
                                  "local_latents": args.bulk_alpha_lat})
                    alpha = args.alpha
                    if args.alpha_latent is not None:
                        alpha = {"bb_ca": float(args.alpha),
                                 "local_latents": float(args.alpha_latent)}
                    if args.condition_router:
                        if router is None:
                            raise ValueError("--condition-router needs a router")
                        from dive.stackelberg.router_condition import (
                            trunk_reads_router_condition)
                        hook.enter_context(
                            trunk_reads_router_condition(
                                model, router,
                                missing=("zeros" if args.router_on_self_cond
                                         else "x_t")))
                    designed = tensors.get("generated_mask", tensors["mask"])
                    extra_gate = None
                    if args.extra_pass_gate:
                        from dive.stackelberg.extra_pass_gate import load_extra_pass_gate
                        from dive.stackelberg.interface_router import partner_atoms_from_batch
                        gnet = load_extra_pass_gate(args.extra_pass_gate)
                        gnet.to(device=tensors["mask"].device)
                        extra_gate = gnet.as_gate(
                            partner_coords=partner_atoms_from_batch(one),
                            designed=designed.bool())
                    hook.enter_context(anticipating_sampler(
                        model, mode="anticipate", alpha=alpha, router=router,
                        partner_only=bool(args.partner_only),
                        designed=designed.bool(),
                        clean_hat=bool(args.clean_hat),
                        response_router=bool(args.response_router),
                        advance_frac=float(args.advance_frac),
                        alpha_schedule=args.alpha_schedule,
                        extra_pass_gate=extra_gate,
                        clip=(None if args.unclipped_extra_pass else 0.5),
                        leader_commit=bool(args.leader_commit_probe),
                        commit_clean=bool(args.commit_clean),
                        residue_commit=bool(args.residue_commit),
                        contact_snap=bool(args.contact_snap),
                        contact_snap_far_clean=bool(args.contact_snap_far_clean),
                        snap_anticipation=bool(args.snap_anticipation),
                        contact_rigid=bool(args.contact_rigid),
                        contact_clean=bool(args.contact_clean),
                        contact_reciprocal=bool(args.contact_reciprocal),
                        invert_router=bool(args.invert_router),
                        recycle_k=int(args.recycle_k),
                        persist_sc=bool(args.persist_sc)))
                with hook:
                    sample = paper_generation.generate_paper_sample(
                        model, tensors, family=args.family, seed=seed, shipped_protocol=True,
                        nsteps_override=args.nsteps,
                    )
                pdb_path, fasta_path, sequence, n_rows = export(
                    sample, batch, task_dir, stem, design_chain=staged.design_chain)
                if args.ame_specialist and args.family == "ame" and entry.get("motif_places"):
                    from dive.stackelberg.ame_e3_match import graft_at_e3_indices
                    from dive.generation.indexed_motif import native_types
                    if len(sequence) != 180:
                        raise RuntimeError(
                            f"designed length {len(sequence)} != 180 after pinning motif "
                            "into the chain")
                    ungrafted = sequence
                    sequence = graft_at_e3_indices(
                        sequence, entry["motif_places"],
                        native_types(task["input_pdb"], task["motif"]))
                    fasta_path.write_text(f">{stem}|design|{len(sequence)}\n{sequence}\n")
                    entry["ungrafted_sequence"] = ungrafted
                entry.update({
                    "status": "complete",
                    "wall_seconds": round(time.perf_counter() - started, 2),
                    "pdb": str(pdb_path), "fasta": str(fasta_path),
                    "sequence": sequence, "design_residues": len(sequence),
                    "complex_residues": n_rows,
                    "n_target_tokens": int(raw["x_target"].shape[1]),
                    "ligand_conditioning": bool(args.ame_specialist and args.family == "ame"),
                    "ligands": task.get("ligands") if args.family == "ame" else None,
                    "n_main_tokens": int(raw["mask"].shape[1]),
                })
                meta = {
                    "task_id": task_id, "family": args.family, "arm": arm.value,
                    "candidate_index": index, "seed": seed, "steps": 400,
                    "shipped_protocol": True,
                    "wall_seconds": entry["wall_seconds"],
                    "sequence": sequence, "design_length": len(sequence),
                    "pdb": str(pdb_path), "fasta": str(fasta_path),
                    "roles": roles, "staging": staged.notes,
                    "design_chain": staged.design_chain,
                    "live_parameter_sha256": report.live_parameter_sha256,
                    "checkpoint": str(args.specialist_ckpt if args.ame_specialist else contract.common_checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "autoencoder_sha256": autoencoder_sha,
                    "trunk": str(args.trunk) if args.trunk else None,
                    "trunk_sha256": trunk_sha,
                    "crop_size": args.crop_size, "target_budget": args.target_budget,
                    "max_design_fraction": args.max_design_fraction,
                    "n_main_tokens": entry["n_main_tokens"],
                    "n_target_tokens": entry["n_target_tokens"],
                }
                if args.family == "ame":
                    meta.update(ame_motif_record(
                        specialist=bool(args.ame_specialist), task=task,
                        places=entry.get("motif_places")))
                    if entry.get("motif_places"):
                        meta["sequence_source"] = "proteina_codesign_e3_indices"
                (task_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
                entry["meta"] = str(task_dir / f"{stem}.meta.json")
                print(f"[ok] {task_id} cand{index} {entry['wall_seconds']}s "
                      f"L={len(sequence)} tok={entry['n_main_tokens']}+{entry['n_target_tokens']}",
                      flush=True)
            except Exception as error:
                import traceback
                entry.update({"status": "failed", "reason": type(error).__name__,
                              "detail": str(error)[:800],
                              "wall_seconds": round(time.perf_counter() - started, 2)})
                print(f"[fail] {task_id} cand{index}: {type(error).__name__}: {str(error)[:300]}",
                      flush=True)
                traceback.print_exc()
            candidates.append(entry)

        record.update({
            "status": "complete" if all(c["status"] == "complete" for c in candidates) else "partial",
            "candidates": candidates,
            "steps": 400,
            "shipped_protocol": True,
            "arm": arm.value,
            "live_parameter_sha256": report.live_parameter_sha256,
            "checkpoint": str(args.specialist_ckpt if args.ame_specialist else contract.common_checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "autoencoder_sha256": autoencoder_sha,
            "trunk": str(args.trunk) if args.trunk else None,
            "trunk_sha256": trunk_sha,
            "crop_size": args.crop_size,
            "target_budget": args.target_budget,
            "max_design_fraction": args.max_design_fraction,
        })
        (task_dir / "generation.json").write_text(json.dumps(record, indent=1))
        summary.append(record)

    name = f"summary_{args.arm}_{args.family}"
    if args.shard:
        name += "_shard" + args.shard.replace("/", "of")
    payload = {
        "arm": arm.value, "family": args.family, "candidates": args.candidates,
        "seeds": seeds, "dry_run": args.dry_run,
        "live_parameter_sha256": None if report is None else report.live_parameter_sha256,
        "trunk": str(args.trunk) if args.trunk else None, "trunk_sha256": trunk_sha,
        "checkpoint_sha256": checkpoint_sha, "rows": summary,
    }
    (out_root / f"{name}.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({"written": str(out_root / f'{name}.json'),
                      "n_tasks": len(summary)}), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
