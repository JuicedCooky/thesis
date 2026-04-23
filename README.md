Repository of Frontend Demo: [`https://github.com/JuicedCooky/zscl_ui`]()

# MTIL — Multi-Task Incremental Learning with CLIP

Research framework for continual learning on vision-language models, built on top of OpenAI's CLIP.
Supports sequential fine-tuning across multiple datasets with methods designed to prevent catastrophic forgetting.

---

## Quick Start

```bash
# Fine-tune on DTD for 1000 iterations
python -m src.main \
    --method finetune \
    --train-mode whole \
    --train-dataset DTD \
    --iterations 1000 \
    --lr 1e-5 \
    --save ckpt/dtd_ft

# ZSCL with ImageNet reference
python -m src.main \
    --method ZSCL \
    --train-mode whole \
    --train-dataset DTD \
    --ref-dataset ImageNet \
    --ref-sentences conceptual_captions \
    --iterations 1000 \
    --lr 1e-5 \
    --save ckpt/dtd_zscl

# Evaluation only
python -m src.main \
    --load ckpt/dtd_ft/DTD.pth \
    --eval-datasets DTD,CIFAR100 \
    --eval-only

# Single image evaluation
python -m src.main \
    --load ckpt/dtd_ft/DTD.pth \
    --eval-single path/to/image.jpg \
    --class-names data/text_classes/imagenet_classes.txt \
    --eval-only
```

### SLURM

Edit `train.sh` with your arguments, then:

```bash
sbatch train.sh
```

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate <env_name>
# or
pip install -r requirements.txt
```

---

## Project Structure

```
mtil/
├── clip/                        # Modified OpenAI CLIP (model, tokenizer, loader)
├── src/
│   ├── main.py                  # Entry point
│   ├── args.py                  # All CLI arguments
│   ├── datasets/                # Dataset implementations
│   ├── models/
│   │   ├── training.py          # Main training loop (GradientTracker, TrainingState)
│   │   ├── finetune.py          # finetune / lwf / ZSCL pipelines
│   │   ├── icarl.py             # iCaRL pipeline
│   │   ├── evaluation.py        # Zero-shot + accuracy evaluation
│   │   ├── helpers.py           # WiSE-FT, L2, distillation utilities
│   │   ├── lora_injection.py    # Custom shared LoRA injection
│   │   └── modeling.py          # ImageEncoder, ClassificationHead
│   └── templates/               # Zero-shot prompt templates
├── scripts/                     # Experiment shell scripts
├── data/                        # Dataset storage root
├── ckpt/                        # Checkpoint outputs
├── logs/                        # Training logs
├── train.sh                     # SBATCH job template
├── list_clip_layers.py          # Utility: list CLIP model layers
├── datasets.md                  # Manual dataset download instructions
└── ARGUMENTS.md                 # Full argument reference
```

---

## Training Methods

| `--method`   | Description |
|--------------|-------------|
| `finetune`   | Standard full-model fine-tuning |
| `lwf`        | Learning Without Forgetting — distills from a frozen copy of the previous model |
| `ZSCL`       | Zero-Shot Continual Learning — regularizes against a reference dataset/model to preserve zero-shot ability |
| `icarl`      | iCaRL — exemplar memory-based incremental learning across sequential datasets |

---

## Train Modes

Controls which parts of CLIP are updated (`--train-mode`):

| Mode            | Trains |
|-----------------|--------|
| `whole`         | Image encoder + text encoder |
| `image`         | Image encoder only |
| `text`          | Text encoder only |
| `image-fc`      | Image encoder + classification head |
| `image-fc-fixed`| Image encoder (classification head frozen) |
| `fc`            | Classification head only (linear probe) |

---

## CLIP Model Variants

Passed via `--model` (default: `ViT-B/16`):

| Model      | Architecture | Notes |
|------------|--------------|-------|
| `ViT-B/32` | ViT, 32×32 patch | Fastest, lowest capacity |
| `ViT-B/16` | ViT, 16×16 patch | **Default** — best speed/accuracy balance |
| `RN50`     | ResNet-50    | CNN backbone |
| `RN101`    | ResNet-101   | Larger CNN |
| `RN50x4`   | ResNet-50 4× width | Wide CNN |
| `RN50x16`  | ResNet-50 16× width | Extra-wide CNN |

Models auto-download from OpenAI on first use.

---

## Datasets

### Auto-download (via torchvision / dataset loaders)

| Name | `--train-dataset` value | Classes |
|------|-------------------------|---------|
| Describable Textures | `DTD` | 47 |
| CIFAR-10 | `CIFAR10` | 10 |
| CIFAR-100 | `CIFAR100` | 100 |
| Flowers102 | `Flowers` | 102 |
| Food101 | `Food` | 101 |
| EuroSAT | `EuroSAT` | 10 |
| Stanford Cars | `StanfordCars` | 196 |
| Oxford Pet | `OxfordPet` | 37 |
| SUN397 | `SUN397` | 397 |
| Caltech101 | `Caltech101` | 102 |
| MNIST | `MNIST` | 10 |
| FGVC Aircraft | `Aircraft` | 100 |

### Manual download required

See [`datasets.md`](datasets.md) for full instructions.

| Dataset | `--train-dataset` / `--ref-dataset` |
|---------|--------------------------------------|
| ImageNet | `ImageNet` |
| ImageNet-A | `ImageNetA` |
| ImageNet-R | `ImageNetR` |
| ImageNet-Sketch | `ImageNetSketch` |
| ImageNet-V2 | `ImageNetV2` |
| ObjectNet | `ObjectNet` |
| YTBB-Robust | `YTBB` |
| ImageNet Vid-Robust | `ImageNetVidRobust` |
| Conceptual Captions | `conceptual_captions` (ref sentences) |
| FMOW (WILDS) | `FMOW` |
| IWildCam (WILDS) | `IWildCam` |

All datasets default to `./data/` — change with `--data-location`.

---

## Key Arguments

Full reference: [`ARGUMENTS.md`](ARGUMENTS.md)

### Core

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | — | Training method (`finetune`, `lwf`, `ZSCL`, `icarl`) |
| `--train-mode` | — | Which parts to train (see table above) |
| `--train-dataset` | — | Dataset to train on |
| `--iterations` | — | Total training steps (mutually exclusive with `--epochs`) |
| `--epochs` | — | Train for N epochs instead of a fixed iteration count |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 8 | Training batch size |
| `--save` | — | Directory to save checkpoints |
| `--load` | — | Checkpoint path to resume from |

### ZSCL

| Argument | Default | Description |
|----------|---------|-------------|
| `--ref-dataset` | — | Reference dataset (e.g., `ImageNet`) |
| `--ref-sentences` | — | Reference text source (e.g., `conceptual_captions`) |
| `--ref-model` | — | Path to reference model (defaults to zero-shot CLIP) |
| `--T` | 2.0 | Distillation temperature |
| `--image-loss` | False | Include image-side distillation loss |
| `--text-loss` | False | Include text-side distillation loss |

### Weight Averaging / WiSE-FT

| Argument | Default | Description |
|----------|---------|-------------|
| `--we` | False | Weight ensembling during training |
| `--we-wise` | False | WiSE-FT weight ensembling |
| `--we-wise-alpha` | 0.98 | WiSE interpolation coefficient |
| `--moving-avg` | False | Exponential moving average of weights |
| `--wise-merge` | False | Apply WiSE merge at end of training |

### OGD (Orthogonal Gradient Descent)

| Argument | Default | Description |
|----------|---------|-------------|
| `--orthogonal-gradients` | — | Number of OGD projections to sample |
| `--orthogonal-gradients-path` | — | Path to previous task's SVD gradient basis |

### LoRA

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora` | False | Enable LoRA adapter training |
| `--lora-r` | 8 | Low-rank dimension |
| `--lora-alpha` | 16 | Scaling factor (effective: alpha/r) |
| `--lora-dropout` | 0.1 | Dropout on LoRA layers (peft only) |
| `--lora-target-modules` | `attn.out_proj,mlp.c_fc,mlp.c_proj` | Comma-separated module names |
| `--lora-shared` | False | Use custom shared LoRA injection instead of peft — shares A/B bases across layers via a dimension-keyed lookup |
| `--lora-shared-split-qkvo` | False | With `--lora-shared`: give each projection type (q/k/v/o) its own shared master instead of one master per attention block shape |

### Evaluation

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval-datasets` | — | Comma-separated datasets to evaluate |
| `--eval-interval` | — | Evaluate every N iterations |
| `--eval-only` | False | Skip training, run evaluation only |
| `--eval-single` | — | Path to a single image to classify |

---

## LoRA

Two backends are available:

**peft library** (default when `--lora` is set):
```bash
python -m src.main --lora --lora-r 8 --lora-alpha 16 ...
```

**Custom shared LoRA** (`--lora-shared`):
Uses [`src/models/lora_injection.py`](src/models/lora_injection.py). After `inject_lora()` places LoRA wrappers, `apply_shared_lora()` walks the model and builds a **lookup registry** keyed by `(layer_type, dim_signature)`. The first layer seen for a given key becomes the *master*; every subsequent layer with the same key has its `lora_A` / `lora_B` tensors replaced with the master's, so all layers in the group literally share the same parameter objects. `layer_type` is inferred from the module path (`attn` vs `mlp`) so attention and MLP layers never share bases even when their dimensions coincide.

By default (`--lora-shared` without `--lora-shared-split-qkvo`) the key for a `MultiheadAttention` block is `("attn", embed_dim, num_heads)`, meaning q/k/v/o projections within each block all share a **single** master A and B — maximum parameter reuse across transformer layers.

```bash
python -m src.main --lora --lora-shared --lora-r 8 --lora-alpha 16 ...
```

**Split-QKVO mode** (`--lora-shared --lora-shared-split-qkvo`):
Adds the projection name to the lookup key: `("attn", proj, embed_dim, num_heads)`. Each projection type (q, k, v, o) gets its own shared master across blocks — q-layers share with q, k with k, etc. — but projections of different types do not share. This trades some parameter efficiency for greater expressiveness per projection type.

```bash
python -m src.main --lora --lora-shared --lora-shared-split-qkvo --lora-r 8 --lora-alpha 16 ...
```

Inspect available layer names for targeting:
```bash
python list_clip_layers.py --linear-only   # only nn.Linear layers
python list_clip_layers.py --params        # all parameters with shapes
```

---

## Continual Learning Workflow

A typical sequential multi-dataset experiment:

```bash
DATASETS="DTD Flowers EuroSAT MNIST StanfordCars Food CIFAR100 SUN397 Aircraft OxfordPet Caltech101"
PREV_CKPT=""

for DS in $DATASETS; do
    LOAD_ARG=""
    [ -n "$PREV_CKPT" ] && LOAD_ARG="--load $PREV_CKPT"

    python -m src.main \
        --method ZSCL \
        --train-mode whole \
        --train-dataset $DS \
        --ref-dataset ImageNet \
        --ref-sentences conceptual_captions \
        --iterations 1000 \
        --lr 1e-5 \
        --save ckpt/zscl_run \
        $LOAD_ARG

    PREV_CKPT="ckpt/zscl_run/${DS}.pth"
done
```

Pre-written scripts for common configurations are in [`scripts/`](scripts/).

---

## OGD Workflow

Collect gradients during training and project future updates to be orthogonal to past task gradients.

```bash
# Task 1: train and save gradient basis
python -m src.main \
    --method finetune \
    --train-dataset DTD \
    --orthogonal-gradients 10 \
    --save ckpt/ogd/task1 ...

# Task 2: project gradients away from task 1 subspace
python -m src.main \
    --method finetune \
    --train-dataset Flowers \
    --orthogonal-gradients 10 \
    --orthogonal-gradients-path ckpt/ogd/task1/grad_DTD.pth \
    --save ckpt/ogd/task2 ...
```

To manually convert raw gradient files to SVD basis:
```bash
python convert_grad_to_basis.py --input grad_DTD.pth --output basis_DTD.pth
```

---

## Evaluation Metrics

- **Top-1 accuracy** and **Top-5 accuracy** per dataset
- Results saved to `<save>/metrics_<dataset>.csv` (one row per evaluation checkpoint)
- Zero-shot classification uses dataset-specific prompt templates from `src/templates/`
