#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FAMILY_INIT = {
    "ame": "complexa_ame.ckpt",
    "binder": "complexa.ckpt",
    "antibody": "complexa.ckpt",
}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=sorted(FAMILY_INIT))
    ap.add_argument("--lambda-iface", type=float, default=1.0)
    ap.add_argument("--snap-anticipation", action="store_true",
                    help="anticipation whose backbone commitment is the ligand "
                         "snap. The live router chooses who answers per residue")
    ap.add_argument("--lambda-far-clean", type=float, default=0.0,
                    help="latent loss on rows far from the ligand, given the "
                         "capped Jacobi backbone. Contact rows are not in it")
    ap.add_argument("--anchor-contact", action="store_true",
                    help="match a frozen copy's Jacobi velocity on rows near "
                         "the ligand, so the contact field cannot drift")
    ap.add_argument("--anchor-bulk", action="store_true",
                    help="match a frozen copy's Jacobi velocity on rows far "
                         "from the ligand. Contact rows stay free for the snap")
    ap.add_argument("--latent-head-only", action="store_true",
                    help="train local_latents_linear only. The trunk and the "
                         "Cα readout stay at the loaded checkpoint")
    ap.add_argument("--lambda-conditioned-ca", type=float, default=0.0,
                    help="Jacobi clean-sample loss on conditioned Cα rows "
                         "only. Does not rewrite backbone velocity at "
                         "generate. 0 leaves those rows on ordinary FM")
    ap.add_argument("--lambda-reciprocal", type=float, default=0.0,
                    help="latent answers a ligand snap; backbone answers the "
                         "clean latent that snap committed. Motif rows stay "
                         "on the Jacobi backbone. Not a velocity scale")
    ap.add_argument("--lambda-clean-backbone", type=float, default=0.0,
                    help="latent loss given the backbone's capped Jacobi "
                         "endpoint. Not a ligand snap and not a velocity scale")
    ap.add_argument("--lambda-rigid-contact", type=float, default=0.0,
                    help="latent loss given one rigid translation of the "
                         "contact patch toward the partner. Not a per-atom snap")
    ap.add_argument("--lambda-misalign", type=float, default=0.0,
                    help="latent loss given the backbone snapped onto the "
                         "partner. Not a velocity extra pass. 0 leaves the "
                         "Proteina step unchanged")
    ap.add_argument("--lambda-frame", type=float, default=0.0,
                    help="weight of the rigid interface-pose loss (centroid + "
                         "orientation). 0 leaves Jacobi FM unchanged. Not a "
                         "decoded structure or a predictor score")
    ap.add_argument("--iface-modalities", choices=("both", "ca", "lat"),
                    default="both",
                    help="which flow channels the interface extra hits. "
                         "both=S19/S25; ca=site geometry only (sequence stays "
                         "on ordinary FM); lat=site latents only")
    ap.add_argument("--contact-nm", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accumulate", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1401)
    ap.add_argument("--validate-every", type=int, default=100)
    ap.add_argument("--validate-batches", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--v6", action="store_true")
    ap.add_argument("--train-router", action="store_true",
                    help="jointly fit LearnedLeadershipRouter on native contacts "
                         "from the same family train batches; not iPTM")
    ap.add_argument("--lambda-router", type=float, default=1.0)
    ap.add_argument("--router-lr", type=float, default=1e-3)
    ap.add_argument("--init-trunk", type=Path, default=None,
                    help="overlay a previous family trunk after FAMILY_INIT load")
    ap.add_argument("--condition-router", type=Path, default=None,
                    help="learned-router checkpoint whose weights fill the trunk "
                         "x_sc slot; same wrap for every family, not a new head")
    ap.add_argument("--condition-live-router", action="store_true",
                    help="fill x_sc from the router being trained, when there is "
                         "no matching --condition-router checkpoint")
    ap.add_argument("--train-anticipate", action="store_true",
                    help="run the generate extra pass during native FM so the "
                         "trunk is fine-tuned under the wrap, not Jacobi-only")
    ap.add_argument("--anticipate-fm", action="store_true",
                    help="ADD extra-pass native-velocity FM on a native-ward "
                         "partner probe. Does not wrap Jacobi training_step "
                         "(S20 wrap dumped Motif). No iPTM / decode labels")
    ap.add_argument("--lambda-anticipate-fm", type=float, default=1.0)
    ap.add_argument("--anticipate-fm-exchange",
                    choices=("xz", "designed", "interface"),
                    default="xz",
                    help="x↔z extra-pass FM: xz/designed = generated rows; "
                         "interface = native-contact rows as the SITE of x↔z "
                         "exchange, not partner-as-leader")
    ap.add_argument("--anticipate-fm-followers", choices=("both", "ca", "lat"),
                    default="both",
                    help="which channel extra-passes: both; ca = z leads x "
                         "follows; lat = x leads z follows")
    ap.add_argument("--anticipate-fm-partner",
                    choices=("native", "jacobi", "commit", "residue"),
                    default="native",
                    help="native-ward, Jacobi step, whole-channel clean "
                         "endpoint, or per-residue hard who (one channel "
                         "commits, the other answers, no velocity gain)")
    ap.add_argument("--anticipate-fm-frac", type=float, default=1.0,
                    help="partner step as a fraction of sampler dt; 0.25 matches "
                         "generate --advance-frac 0.25")
    ap.add_argument("--anticipate-fm-objective",
                    choices=("velocity", "applied"), default="velocity",
                    help="velocity: raw extra-pass v vs native u; applied: "
                         "the generate blend (clip, router, schedule t) vs native u")
    ap.add_argument("--anticipate-alpha", type=float, default=1.0)
    ap.add_argument("--anticipate-alpha-latent", type=float, default=None)
    ap.add_argument("--anticipate-partner-only", action="store_true")
    ap.add_argument("--anticipate-clean-hat", action="store_true",
                    help="training extra pass reads x_sc = x_t+(1-t)v, matching "
                         "generate --clean-hat; shared across families")
    ap.add_argument("--train-extra-pass-gate", action="store_true",
                    help="learn a per-residue mix that keeps the extra-pass "
                         "only where it is closer to native velocity than Jacobi")
    ap.add_argument("--anticipate-router", type=Path, default=None,
                    help="learned-router checkpoint used as the generate router "
                         "during the training extra pass")
    ap.add_argument("--router-sc-dist", action="store_true",
                    help="concat native sidechain–partner min distance as a "
                         "router input; same for every family, not a new head")
    ap.add_argument("--router-label", choices=("ca", "sidechain", "union"),
                    default="ca",
                    help="native contact tensor used as the router BCE target")
    ap.add_argument("--router-time", action="store_true",
                    help="concat the flow clock t as a router input; same net "
                         "for every family")
    ap.add_argument("--router-trunk-condition", action="store_true",
                    help="concat the trunk's motif, ligand, and protein-target "
                         "concat features into the router; absent blocks are zeros")
    ap.add_argument("--router-no-residue-type", action="store_true",
                    help="drop the amino-acid one-hot; native letters are not "
                         "a router input")
    ap.add_argument("--router-on-self-cond", action="store_true",
                    help="scale the self-conditioning prediction by the router "
                         "weight; a missing prediction is zeros, not x_t")
    ap.add_argument("--frozen-reaction", action="store_true",
                    help="snap anticipation matches the frozen trunk velocity "
                         "on the probe, not the native velocity")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from dive.stackelberg.interface_fm import native_plus_interface_step
    from dive.training.sequence_recovery import refuse_outcome_labels
    iface_mods = {"both": ("bb_ca", "local_latents"),
                  "ca": ("bb_ca",),
                  "lat": ("local_latents",)}[args.iface_modalities]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.family == "ame":
        from dive.cre.model import build_ame_model
        from dive.cre.train import ame_dataloader

        model, report = build_ame_model(device=args.device, verify_sha=True, trainable=True)
        if not report.exact_load:
            raise SystemExit("refusing to train: checkpoint did not load exactly")
        ckpt_name = Path(report.checkpoint).name
        trunk_trainable = report.trunk_trainable_parameters
        train_loader = ame_dataloader(
            "train", batch_size=args.batch_size, seed=args.seed, v6=args.v6)
        val_loader = ame_dataloader(
            "validation", batch_size=args.batch_size, seed=args.seed,
            shuffle=False, v6=args.v6)
        build_info = {"checkpoint": report.checkpoint,
                      "trunk_trainable_parameters": trunk_trainable}
    else:

        sys.path.insert(0, str(REPO / "scripts" / "imr"))
        import _paths
        _paths.bootstrap()
        from dive.rce.binder_model import bootstrap_binder, load_binder
        from dive.stackelberg.family_data import family_role_dataloader

        bootstrap_binder()
        model, info = load_binder(device=args.device)
        model.train()
        for parameter in model.nn.parameters():
            parameter.requires_grad_(True)
        model.autoencoder.eval()
        for parameter in model.autoencoder.parameters():
            parameter.requires_grad_(False)
        ckpt_name = Path(info["checkpoint"]).name
        trunk_trainable = sum(p.numel() for p in model.nn.parameters() if p.requires_grad)
        train_loader = family_role_dataloader(
            args.family, "train", batch_size=args.batch_size, seed=args.seed)
        val_loader = family_role_dataloader(
            args.family, "validation", batch_size=args.batch_size, seed=args.seed,
            shuffle=False)
        build_info = {"checkpoint": info["checkpoint"],
                      "trunk_trainable_parameters": trunk_trainable,
                      "exact_load": info.get("exact_load")}

    if ckpt_name != FAMILY_INIT[args.family]:
        raise SystemExit(
            f"refusing: family {args.family} must init from {FAMILY_INIT[args.family]}, "
            f"got {ckpt_name}"
        )
    if args.train_extra_pass_gate and not args.train_anticipate:
        raise SystemExit(
            "--train-extra-pass-gate needs --train-anticipate so Jacobi and "
            "extra-pass velocities exist on the same t")
    if args.train_anticipate and args.anticipate_fm:
        raise SystemExit(
            "use --anticipate-fm (added native extra-pass FM) or "
            "--train-anticipate (wrap Jacobi), not both; wrapping dumped Motif")
    if args.init_trunk is not None:
        from dive.stackelberg.trunk_io import nn_state_for_train_continue
        payload = torch.load(args.init_trunk, map_location=args.device, weights_only=False)
        nn_state = payload["nn"] if isinstance(payload, dict) and "nn" in payload else payload
        model.nn.load_state_dict(nn_state_for_train_continue(nn_state), strict=False)
    if args.frozen_reaction and not args.snap_anticipation:
        raise SystemExit(
            "--frozen-reaction is the target of --snap-anticipation; "
            "without that probe there is no frozen reaction to match")

    anchor_teacher = None
    if args.anchor_bulk or args.anchor_contact:
        import copy
        anchor_teacher = copy.deepcopy(model)
        anchor_teacher.eval()
        for parameter in anchor_teacher.parameters():
            parameter.requires_grad_(False)
    frozen_reaction = None
    if args.frozen_reaction:
        import copy
        frozen_reaction = copy.deepcopy(model)
        frozen_reaction.eval()
        for parameter in frozen_reaction.parameters():
            parameter.requires_grad_(False)

    router_net = None
    router_opt = None
    cond_router_fn = None
    gate_net = None
    gate_opt = None
    if args.train_extra_pass_gate:
        from dive.stackelberg.extra_pass_gate import ExtraPassGate, extra_pass_gate_loss
        gate_net = ExtraPassGate().to(args.device)
        gate_opt = torch.optim.Adam(gate_net.parameters(), lr=args.router_lr)
    if args.train_router:
        from dive.stackelberg.learned_router import LearnedLeadershipRouter
        router_net = LearnedLeadershipRouter(
            use_sc_dist=bool(args.router_sc_dist),
            use_time=bool(args.router_time),
            use_trunk_condition=bool(args.router_trunk_condition),
            use_residue_type=not bool(args.router_no_residue_type)).to(args.device)
        router_opt = torch.optim.Adam(router_net.parameters(), lr=args.router_lr)
    if args.condition_router is not None:
        from dive.stackelberg.learned_router import load_learned_router
        from dive.stackelberg.router_condition import live_or_native_router
        loaded = load_learned_router(args.condition_router).to(args.device)
        if router_net is not None:
            router_net.load_state_dict(loaded.state_dict())
            cond_router_fn = live_or_native_router(router_net)
        else:
            loaded.eval()
            for parameter in loaded.parameters():
                parameter.requires_grad_(False)
            cond_router_fn = live_or_native_router(loaded)
    if args.condition_live_router:
        if router_net is None:
            raise SystemExit("--condition-live-router needs --train-router")
        if cond_router_fn is not None:
            raise SystemExit("pass --condition-router or --condition-live-router, not both")
        from dive.stackelberg.router_condition import live_or_native_router
        cond_router_fn = live_or_native_router(router_net)

    anticipate_router_fn = None
    if args.anticipate_router is not None:
        from dive.stackelberg.learned_router import load_learned_router
        from dive.stackelberg.router_condition import live_or_native_router
        anet = load_learned_router(args.anticipate_router).to(args.device)
        if router_net is not None:
            router_net.load_state_dict(anet.state_dict())
            anticipate_router_fn = live_or_native_router(router_net)
        else:
            anet.eval()
            for parameter in anet.parameters():
                parameter.requires_grad_(False)
            anticipate_router_fn = live_or_native_router(anet)

    if args.latent_head_only:
        from dive.stackelberg.latent_head import freeze_all_but_latent_head
        freeze_all_but_latent_head(model.nn)
    parameters = [p for p in model.nn.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    baseline = {k: v.detach().clone() for k, v in model.nn.state_dict().items()}
    fixed_val = []
    for batch in val_loader:
        if batch is None:
            continue
        refuse_outcome_labels(batch)
        fixed_val.append(batch)
        if len(fixed_val) >= args.validate_batches:
            break

    def drift():
        live = model.nn.state_dict()
        worst = 0.0
        for key, value in baseline.items():
            delta = (live[key].detach().cpu() - value.cpu()).abs()
            worst = max(worst, float(delta.max()))
        return {"max_abs_change": worst}

    def wrap_for_step():
        from contextlib import ExitStack
        stack = ExitStack()
        if cond_router_fn is not None:
            from dive.stackelberg.router_condition import trunk_reads_router_condition
            stack.enter_context(trunk_reads_router_condition(
                model, cond_router_fn,
                missing=("zeros" if args.router_on_self_cond else "x_t")))
        if args.train_anticipate:
            from dive.stackelberg.anticipate_train import train_under_anticipation
            stack.enter_context(train_under_anticipation(
                model, alpha=args.anticipate_alpha,
                alpha_latent=args.anticipate_alpha_latent,
                partner_only=args.anticipate_partner_only,
                router=anticipate_router_fn,
                clean_hat=bool(args.anticipate_clean_hat)))
        return stack

    def validate():
        model.eval()
        total, seen, extra = 0.0, 0, 0.0
        with torch.no_grad():
            for batch in fixed_val:
                try:
                    with wrap_for_step():
                        loss, parts = native_plus_interface_step(
                            model, batch, lambda_iface=args.lambda_iface,
                            contact_nm=args.contact_nm, batch_idx=-1,
                            modalities=iface_mods,
                            lambda_frame=args.lambda_frame)
                        if args.lambda_misalign:
                            from dive.stackelberg.interface_fm import contact_misalignment_latent_loss
                            fm_batch = getattr(model, "_last_fm_batch", None)
                            if fm_batch is not None and "x_1" in fm_batch:
                                loss = loss + args.lambda_misalign * contact_misalignment_latent_loss(
                                    model, fm_batch, contact_nm=args.contact_nm)
                        if anchor_teacher is not None:
                            from dive.stackelberg.bulk_anchor import (
                                bulk_anchor_loss, far_from_partner_mask,
                                near_partner_mask)
                            fm_batch = getattr(model, "_last_fm_batch", None)
                            live_v = getattr(model, "_last_jacobi_v", None)
                            if fm_batch is not None and live_v:
                                teacher_out = anchor_teacher.call_nn(fm_batch)
                                teacher_v = {
                                    key: teacher_out[key]["v"]
                                    for key in ("bb_ca", "local_latents")
                                    if isinstance(teacher_out.get(key), dict)
                                    and "v" in teacher_out[key]
                                }
                                if args.anchor_bulk:
                                    loss = loss + bulk_anchor_loss(
                                        live_v, teacher_v, far_from_partner_mask(
                                            fm_batch, contact_nm=args.contact_nm))
                                if args.anchor_contact:
                                    loss = loss + bulk_anchor_loss(
                                        live_v, teacher_v, near_partner_mask(
                                            fm_batch, contact_nm=args.contact_nm))
                        if args.lambda_far_clean:
                            from dive.stackelberg.interface_fm import far_clean_latent_loss
                            fm_batch = getattr(model, "_last_fm_batch", None)
                            jacobi_v = getattr(model, "_last_jacobi_v", None)
                            if fm_batch is not None and jacobi_v:
                                loss = loss + args.lambda_far_clean * far_clean_latent_loss(
                                    model, fm_batch, jacobi_v,
                                    contact_nm=args.contact_nm)
                        if args.lambda_rigid_contact:
                            from dive.stackelberg.interface_fm import rigid_contact_latent_loss
                            fm_batch = getattr(model, "_last_fm_batch", None)
                            if fm_batch is not None and "x_1" in fm_batch:
                                loss = loss + args.lambda_rigid_contact * rigid_contact_latent_loss(
                                    model, fm_batch, contact_nm=args.contact_nm)
                        if args.lambda_clean_backbone:
                            from dive.stackelberg.interface_fm import clean_backbone_latent_loss
                            fm_batch = getattr(model, "_last_fm_batch", None)
                            jacobi_v = getattr(model, "_last_jacobi_v", None)
                            if (fm_batch is not None and "x_1" in fm_batch
                                    and jacobi_v):
                                loss = loss + args.lambda_clean_backbone * clean_backbone_latent_loss(
                                    model, fm_batch, jacobi_v,
                                    contact_nm=args.contact_nm)
                        if args.lambda_reciprocal:
                            from dive.stackelberg.interface_fm import reciprocal_contact_loss
                            fm_batch = getattr(model, "_last_fm_batch", None)
                            jacobi_v = getattr(model, "_last_jacobi_v", None)
                            if (fm_batch is not None and "x_1" in fm_batch
                                    and jacobi_v):
                                loss = loss + args.lambda_reciprocal * reciprocal_contact_loss(
                                    model, fm_batch, jacobi_v,
                                    contact_nm=args.contact_nm)
                        if args.lambda_conditioned_ca:
                            from dive.stackelberg.interface_fm import conditioned_ca_loss
                            fm_batch = getattr(model, "_last_fm_batch", None)
                            nn_out = getattr(model, "_last_out", None)
                            if fm_batch is not None and nn_out is not None:
                                loss = loss + args.lambda_conditioned_ca * conditioned_ca_loss(
                                    model, fm_batch, nn_out)
                except Exception:
                    continue
                total += float(loss)
                extra += float(parts.get("interface_fm") or 0.0)
                seen += 1
        model.train()
        return {"mean_loss": total / seen if seen else None, "interface_fm": extra / seen if seen else None,
                "batches": seen}

    record = {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "family": args.family,
        "init_checkpoint": FAMILY_INIT[args.family],
        "lambda_iface": args.lambda_iface,
        "contact_nm": args.contact_nm,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "build": build_info,
        "history": [],
        "validation": [{"step": 0, **validate(), "drift": drift()}],
    }
    (args.out / "training.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"family": args.family, "init": FAMILY_INIT[args.family],
                      "lambda_iface": args.lambda_iface, "val0": record["validation"][0]}),
          flush=True)

    stream = iter(train_loader)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        if args.warmup:
            scale = min(1.0, step / args.warmup)
            for group in optimiser.param_groups:
                group["lr"] = args.lr * scale
        optimiser.zero_grad(set_to_none=True)
        if router_opt is not None:
            router_opt.zero_grad(set_to_none=True)
        if gate_opt is not None:
            gate_opt.zero_grad(set_to_none=True)
        losses, used = [], 0
        for _ in range(args.accumulate):
            try:
                batch = next(stream)
            except StopIteration:
                stream = iter(train_loader)
                batch = next(stream)
            if batch is None:
                continue
            refuse_outcome_labels(batch)
            with wrap_for_step():
                loss, parts = native_plus_interface_step(
                    model, batch, lambda_iface=args.lambda_iface,
                    contact_nm=args.contact_nm, modalities=iface_mods,
                    lambda_frame=args.lambda_frame)
            if args.anticipate_fm:
                from dive.stackelberg.anticipation_fm import anticipation_fm_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                if fm_batch is None or "x_1" not in fm_batch:
                    raise ValueError(
                        "anticipate-fm needs the captured training_step batch "
                        "with native x_1; the dataloader row is not the flow")
                rfn = None
                if router_net is not None:
                    from dive.stackelberg.learned_router import native_from_train_batch
                    native = native_from_train_batch(fm_batch)
                    rfn = router_net.as_router(
                        partner_coords=native["partner"],
                        designed=native["designed"])
                follow = {"both": ("bb_ca", "local_latents"),
                          "ca": ("bb_ca",),
                          "lat": ("local_latents",)}[args.anticipate_fm_followers]
                afm = anticipation_fm_loss(
                    model, fm_batch,
                    leadership=args.anticipate_fm_exchange,
                    router=rfn, followers=follow,
                    partner_advance=args.anticipate_fm_partner,
                    jacobi_v=getattr(model, "_last_jacobi_v", None),
                    advance_frac=args.anticipate_fm_frac,
                    objective=args.anticipate_fm_objective)
                loss = loss + args.lambda_anticipate_fm * afm
                parts["anticipate_fm"] = float(afm.detach())
            if args.snap_anticipation:
                if router_net is None:
                    raise ValueError("snap anticipation needs --train-router")
                from dive.stackelberg.anticipate_train import sampler_dt
                from dive.stackelberg.router_condition import (
                    live_or_native_router, trunk_reads_router_condition)
                from dive.stackelberg.snap_anticipation import snap_anticipation_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                live_v = getattr(model, "_last_jacobi_v", None)
                if fm_batch is None or not live_v:
                    raise ValueError("snap anticipation needs the Jacobi velocity")
                route = live_or_native_router(router_net)
                weights = route(fm_batch, {})
                with trunk_reads_router_condition(
                        model, route,
                        missing=("zeros" if args.router_on_self_cond else "x_t")):
                    sloss = snap_anticipation_loss(
                        model, fm_batch, live_v, weights,
                        dt=sampler_dt(fm_batch), contact_nm=args.contact_nm,
                        teacher=frozen_reaction)
                loss = loss + sloss
                parts["snap_anticipation"] = float(sloss.detach())
            if args.lambda_misalign:
                from dive.stackelberg.interface_fm import contact_misalignment_latent_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                if fm_batch is None or "x_1" not in fm_batch:
                    raise ValueError(
                        "misalignment latent target needs the captured "
                        "training_step batch")
                mloss = contact_misalignment_latent_loss(
                    model, fm_batch, contact_nm=args.contact_nm)
                loss = loss + args.lambda_misalign * mloss
                parts["misalign"] = float(mloss.detach())
            if args.anchor_bulk:
                from dive.stackelberg.bulk_anchor import (
                    bulk_anchor_loss, far_from_partner_mask)
                fm_batch = getattr(model, "_last_fm_batch", None)
                live_v = getattr(model, "_last_jacobi_v", None)
                if fm_batch is None or not live_v or anchor_teacher is None:
                    raise ValueError("bulk anchor needs the live Jacobi velocity")
                with torch.no_grad():
                    teacher_out = anchor_teacher.call_nn(fm_batch)
                teacher_v = {
                    key: teacher_out[key]["v"]
                    for key in ("bb_ca", "local_latents")
                    if isinstance(teacher_out.get(key), dict) and "v" in teacher_out[key]
                }
                far = far_from_partner_mask(fm_batch, contact_nm=args.contact_nm)
                aloss = bulk_anchor_loss(live_v, teacher_v, far)
                loss = loss + aloss
                parts["bulk_anchor"] = float(aloss.detach())
            if args.anchor_contact:
                from dive.stackelberg.bulk_anchor import (
                    bulk_anchor_loss, near_partner_mask)
                fm_batch = getattr(model, "_last_fm_batch", None)
                live_v = getattr(model, "_last_jacobi_v", None)
                if fm_batch is None or not live_v or anchor_teacher is None:
                    raise ValueError("contact anchor needs the live Jacobi velocity")
                with torch.no_grad():
                    teacher_out = anchor_teacher.call_nn(fm_batch)
                teacher_v = {
                    key: teacher_out[key]["v"]
                    for key in ("bb_ca", "local_latents")
                    if isinstance(teacher_out.get(key), dict) and "v" in teacher_out[key]
                }
                closs = bulk_anchor_loss(
                    live_v, teacher_v, near_partner_mask(
                        fm_batch, contact_nm=args.contact_nm))
                loss = loss + closs
                parts["contact_anchor"] = float(closs.detach())
            if args.lambda_far_clean:
                from dive.stackelberg.interface_fm import far_clean_latent_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                live_v = getattr(model, "_last_jacobi_v", None)
                if fm_batch is None or not live_v:
                    raise ValueError("far clean latent needs the Jacobi velocity")
                floss = far_clean_latent_loss(
                    model, fm_batch, live_v, contact_nm=args.contact_nm)
                loss = loss + args.lambda_far_clean * floss
                parts["far_clean"] = float(floss.detach())
            if args.lambda_rigid_contact:
                from dive.stackelberg.interface_fm import rigid_contact_latent_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                if fm_batch is None or "x_1" not in fm_batch:
                    raise ValueError(
                        "rigid contact latent target needs the captured "
                        "training_step batch")
                rloss = rigid_contact_latent_loss(
                    model, fm_batch, contact_nm=args.contact_nm)
                loss = loss + args.lambda_rigid_contact * rloss
                parts["rigid_contact"] = float(rloss.detach())
            if args.lambda_clean_backbone:
                from dive.stackelberg.interface_fm import clean_backbone_latent_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                jacobi_v = getattr(model, "_last_jacobi_v", None)
                if fm_batch is None or "x_1" not in fm_batch or not jacobi_v:
                    raise ValueError(
                        "clean-backbone latent target needs the captured "
                        "Jacobi velocity and native batch")
                closs = clean_backbone_latent_loss(
                    model, fm_batch, jacobi_v, contact_nm=args.contact_nm)
                loss = loss + args.lambda_clean_backbone * closs
                parts["clean_backbone"] = float(closs.detach())
            if args.lambda_reciprocal:
                from dive.stackelberg.interface_fm import reciprocal_contact_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                jacobi_v = getattr(model, "_last_jacobi_v", None)
                if fm_batch is None or "x_1" not in fm_batch or not jacobi_v:
                    raise ValueError(
                        "reciprocal contact needs the captured Jacobi "
                        "velocity and native batch")
                rloss = reciprocal_contact_loss(
                    model, fm_batch, jacobi_v, contact_nm=args.contact_nm)
                loss = loss + args.lambda_reciprocal * rloss
                parts["reciprocal"] = float(rloss.detach())
            if args.lambda_conditioned_ca:
                from dive.stackelberg.interface_fm import conditioned_ca_loss
                fm_batch = getattr(model, "_last_fm_batch", None)
                nn_out = getattr(model, "_last_out", None)
                if fm_batch is None or nn_out is None:
                    raise ValueError(
                        "conditioned Cα loss needs the captured Jacobi output")
                closs = conditioned_ca_loss(model, fm_batch, nn_out)
                loss = loss + args.lambda_conditioned_ca * closs
                parts["conditioned_ca"] = float(closs.detach())
            if router_net is not None:
                from dive.stackelberg.learned_router import (
                    native_from_train_batch, native_router_loss)
                r = native_router_loss(
                    router_net, native_from_train_batch(
                        batch, trunk_condition=bool(args.router_trunk_condition)),
                    label=args.router_label)
                loss = loss + args.lambda_router * r["loss"]
                parts["router"] = float(r["loss"].detach())
            if gate_net is not None:
                from dive.stackelberg.learned_router import native_from_train_batch
                from dive.stackelberg.extra_pass_gate import extra_pass_gate_loss
                jacobi = getattr(model, "_last_jacobi_v", None)
                extra = getattr(model, "_last_extra_v", None)
                xt = getattr(model, "_last_xt", None)
                clock = getattr(model, "_last_t", None)
                if jacobi is not None and extra is not None and xt is not None:
                    native = native_from_train_batch(batch)
                    ca = xt["bb_ca"]
                    partner = native["partner"].to(device=ca.device, dtype=ca.dtype)
                    dist = torch.cdist(ca, partner).min(dim=-1).values
                    x1 = native["bb_ca"].to(device=ca.device, dtype=ca.dtype)
                    restype = native["residue_type"].to(device=ca.device)
                    designed = native["designed"].to(device=ca.device)
                    lat = xt.get("local_latents", native["local_latents"])
                    if torch.is_tensor(lat):
                        lat = lat.to(device=ca.device, dtype=ca.dtype)
                    payload = {
                        "v_jacobi": jacobi["bb_ca"],
                        "v_extra": extra["bb_ca"],
                        "x_t": ca,
                        "x_1": x1,
                        "t": clock if clock is not None else native.get("t"),
                        "residue_type": restype,
                        "designed": designed,
                        "dist_nm": dist,
                        "local_latents": lat,
                    }
                    g = extra_pass_gate_loss(gate_net, payload)
                    loss = loss + g["loss"]
                    parts["extra_pass_gate"] = float(g["loss"].detach())
            (loss / args.accumulate).backward()
            losses.append(float(loss))
            used += 1
        if used:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimiser.step()
            if router_opt is not None:
                router_opt.step()
                router_opt.zero_grad(set_to_none=True)
            if gate_opt is not None:
                gate_opt.step()
                gate_opt.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            record["history"].append({
                "step": step, "loss": sum(losses) / len(losses) if losses else None,
                "microbatches": used, "elapsed_seconds": time.perf_counter() - started,
            })
            print(json.dumps(record["history"][-1]), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            record["validation"].append({"step": step, **validate(), "drift": drift()})
            (args.out / "training.json").write_text(json.dumps(record, indent=1) + "\n")
            print(json.dumps(record["validation"][-1]), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save({"nn": model.nn.state_dict(), "step": step,
                        "family": args.family, "lambda_iface": args.lambda_iface,
                        "init_checkpoint": FAMILY_INIT[args.family],
                        "train_router": bool(args.train_router)},
                       args.out / f"trunk_step{step}.pt")
            if router_net is not None:
                torch.save({"state_dict": router_net.state_dict(),
                            "config": {"d_latent": 8, "d_hidden": 64, "n_aa": 20,
                                       "use_sc_dist": bool(args.router_sc_dist),
                                       "use_time": bool(args.router_time),
                                       "use_trunk_condition": bool(args.router_trunk_condition),
                                       "use_residue_type": not bool(args.router_no_residue_type)},
                            "family": args.family, "step": step,
                            "router_label": args.router_label},
                           args.out / f"router_step{step}.pt")
            if gate_net is not None:
                torch.save({"state_dict": gate_net.state_dict(),
                            "config": {"d_latent": 8, "d_hidden": 64, "n_aa": 20},
                            "family": args.family, "step": step},
                           args.out / f"extra_pass_gate_step{step}.pt")

    record["wall_seconds"] = time.perf_counter() - started
    record["final_drift"] = drift()
    (args.out / "training.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"done": True, "family": args.family, "steps": args.steps,
                      "drift": record["final_drift"]}), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
