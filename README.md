# Co-DIRECT

Fine-tune the Proteina-Complexa AME trunk with snap anticipation, then sample
from the saved trunk and router.

The router does two things. It chooses, per residue, whether the backbone or
the sequence channel answers. It also writes that choice into the trunk as

```
x_sc = w × s
```

`s` is the self-conditioning prediction already stored for that step (the
previous clean sample). A step with no previous prediction uses zeros, not the
live coordinates and not the native structure `x_1`. The amino-acid one-hot is
not a router input. The trunk's motif and ligand condition features are not a
router input either. The anticipation loss still regresses the answering
channel toward the native flow-matching velocity `u = (x_1 - x_t) / (1 - t)`.
That velocity is a training target. It is not fed into the network, and
sampling does not compute it.

Python sources for this method and for Proteina are in this repository. The
only things that stay outside are model weights, Proteina `assets/`, and
structure files.

## Layout

- `src/dive/` — training, router, snap anticipation, v6 loader, sampling helpers
- `vendor/proteina-complexa/src/` — Proteina Python sources
- `scripts/train.py` — training entry
- `scripts/generate.py` — sampling entry
- `scripts/embed_latents.py` — the autoencoder call that builds `local_latents`
- `src/dive/generation/` — design-chain staging and AME conditioning used by `generate.py`
- `data/v6/` — AME train and validation manifests
- `environment.yml` — conda environment

## Environment

```bash
conda env create -f environment.yml
conda activate codirect
```

The export was taken from a CUDA 12.6 / PyTorch 2.7 environment. A fresh solve
can take a long time.

```bash
export PROTEINA_COMPLEXA_ROOT=<proteina-complexa>   # ckpts/ and assets/
export BENCHV2_ROOT=<benchv2>                       # task lists and staged complexes
export CODIRECT_STRUCTURE_ROOT=<structures>         # parent of structures/ame/
export PYTHONPATH=src
```

`PROTEINA_COMPLEXA_ROOT` must contain:

- `ckpts/complexa_ame.ckpt`
- `ckpts/complexa_ame_ae.ckpt`
- `assets/` (Proteina `DATA_PATH`)

`CODIRECT_V6_DATA` defaults to `data/v6` in this checkout. Those CSVs are
included. The `path` column is relative to `CODIRECT_STRUCTURE_ROOT`. The
structure files themselves are not in this repository. Training resolves the
paths when it reads the manifest.

`BENCHV2_ROOT` is required only for sampling. It holds the v6 task lists and
the staged complexes, including `generation/E3/ame` motif-index records.

## Latent embeddings

Latents are not a separate frozen table. For each training batch the
autoencoder encodes the structure and the flow matcher stores the result as
`x_1["local_latents"]` with shape `[batch, residues, 8]`. Cα coordinates are
`x_1["bb_ca"]` in nanometres. `scripts/embed_latents.py` runs that function on
one training example and writes both tensors.

```bash
python scripts/embed_latents.py runs/latent_one.pt
```

## Train

```bash
python scripts/train.py \
  --family ame --v6 \
  --lambda-iface 2 --lambda-frame 0 \
  --train-router --condition-live-router --router-on-self-cond \
  --router-no-residue-type \
  --snap-anticipation --lambda-router 1 --router-lr 1e-3 \
  --steps 400 --batch-size 1 --accumulate 8 --lr 1e-6 \
  --device cuda:0 \
  --out runs/train-out
```

Do not add `--router-trunk-condition` or `--frozen-reaction`. The run writes
`trunk_step*.pt` and `router_step*.pt`. Run every command from the repository
root, after the exports above.

Sampling does not read the raw training checkpoint. Merge LoRA into the trunk
first:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from dive.stackelberg.trunk_io import wrap_for_paper_generate
src = Path("runs/train-out/trunk_step400.pt")
wrap_for_paper_generate(src, src.with_name("trunk_step400.paper.pt"))
PY
```

Design-chain staging lives in `src/dive/generation/` and is called by
`generate.py`. It is not a separate command. `BENCHV2_ROOT` is only the
external directory of task lists and structures.

## Sample

```bash
python scripts/generate.py \
  --family ame --arm lora --gpu 0 --candidates 3 --ame-specialist \
  --trunk runs/train-out/trunk_step400.paper.pt \
  --learned-router runs/train-out/router_step400.pt \
  --anticipate --snap-anticipation \
  --condition-router --router-on-self-cond \
  --alpha 1.0 --alpha-latent 1.0 \
  --out-root runs/samples
```

`--nsteps 2` shortens diffusion for a load check. Omit it for the 400-step
protocol. The saved router must have `use_residue_type: false` and
`use_trunk_condition: false`.


