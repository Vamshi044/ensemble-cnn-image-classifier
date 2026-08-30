# Ensemble CNN Image Classifier

## Current Status

> **Stage 3B: a training run has been attempted but not completed.**
>
> **No trained model or training result is present in this repository.** No
> accuracy, loss, F1, confusion matrix, benchmark or model ranking for any
> classifier is reported anywhere in it. The only numbers recorded are dataset
> counts, tensor statistics, parameter counts, checkpoint file sizes, and
> wall-clock timings.
>
> Stage 1 built the data pipeline and Stage 2 built and verified the three
> models *(both approved)*. Stage 3A added the machinery a training run needs —
> the loop, optimiser, scheduler, mixed precision, checkpointing, resume, early
> stopping and metrics — together with its tests *(approved)*. A Stage 3B run
> was then started on a Colab T4 and ended before it finished when the runtime
> was lost. Checkpoints and results are gitignored and live only on that
> runtime's disk, so nothing from that run reached this repository and no figure
> from it is quoted here.
>
> Every execution performed against this infrastructure inside the repository
> used **synthetic random noise**, not CIFAR-10, purely to prove the code paths
> work. Any loss or accuracy printed by those smoke tests is a property of
> random tensors and says nothing about any model.
>
> `scripts/evaluate_ensemble.py` implements Stage 4 but has never been run
> against trained weights. **The official CIFAR-10 test set has not been used
> for anything.**

---

## Overview

A reproducible image-classification system for CIFAR-10 that fine-tunes three
ImageNet-pretrained convolutional networks independently and then combines their
predictions by fusing softmax probabilities.

The pipeline:

```
CIFAR-10
    -> Train / Validation / Test separation
    -> Data preprocessing + augmentation
    -> ImageNet-pretrained CNN models
    -> Fine-tuning
    -> Individual model predictions
    -> Softmax probabilities
    -> Probability-based ensemble
    -> Final prediction
    -> Evaluation
    -> Error analysis
```

## Objective

Determine whether fusing the softmax outputs of three architecturally different
CNNs produces a more accurate and more robust CIFAR-10 classifier than any of
the three individually, under a methodology strict enough that the final number
means something.

The methodological constraint that shapes the whole project: the official
10,000-image test set is used **exactly once**, at the very end. Every decision
before that point — hyperparameters, architecture handling, checkpoint
selection, early stopping, ensemble weighting — is made against the validation
split only.

## Models

| Architecture | torchvision source | Head replaced | Total params | Head params |
|---|---|---|---|---|
| GoogLeNet | `torchvision.models.googlenet` | `model.fc` | 5,610,154 | 10,250 |
| ResNet18 | `torchvision.models.resnet18` | `model.fc` | 11,181,642 | 5,130 |
| VGG11 | `torchvision.models.vgg11` | `model.classifier[6]` | 128,807,306 | 40,970 |

Counts are measured, not estimated (`scripts/verify_models.py`).

All three start from official ImageNet-pretrained weights loaded through the
modern torchvision weights API (`ResNet18_Weights.IMAGENET1K_V1` and friends —
no manual URLs), and have their 1000-class heads replaced with a 10-class head.

**Ordering matters and is enforced.** torchvision forces `num_classes=1000`
whenever `weights` is given, so the 10-class head cannot be requested at
construction time — weights are loaded first, then the head is swapped. Doing it
the other way round either fails on a shape mismatch or silently overwrites a
freshly initialised head. Verified by comparing a backbone tensor against the
official checkpoint after replacement: byte-identical for all three.

## Dataset

CIFAR-10 — 60,000 32×32 colour images across 10 classes, obtained via
`torchvision.datasets.CIFAR10`.

| Split | Images | Source |
|---|---|---|
| Train | 45,000 | Official training set |
| Validation | 5,000 | Official training set |
| Test | 10,000 | Official test set, untouched until final evaluation |

Only the official 50,000-image training set is split. The split is
class-stratified, giving exactly 4,500 training and 500 validation images per
class.

## Methodology

Transfer learning + fine-tuning + data augmentation + softmax probability
ensemble.

### Transfer-learning strategy

Two phases. Nothing below has been run as training yet; the code supports it and
the mechanics are verified, but hyperparameters remain unvalidated placeholders
to be selected on the validation split in Stage 3.

**Phase 1 — backbone frozen, head warm-up.** Only the new 10-class head trains.
A randomly initialised head produces large early gradients that would otherwise
damage pretrained features.

Freezing parameters is *not* sufficient on its own. BatchNorm running statistics
are buffers, not parameters, and keep updating in training mode — so a
"frozen" backbone would still drift as CIFAR-10 statistics overwrite the ImageNet
ones. `set_training_mode(model, training=True, backbone_frozen=True)` holds those
layers in eval mode, and a test asserts the buffers do not move. This affects
GoogLeNet (114 buffers) and ResNet18 (40); VGG11 has no normalisation layers.

**Phase 2 — full unfreeze, discriminative learning rates.** The backbone carries
pretrained features needing gentle adjustment; the head is new and must move
faster. `build_param_groups()` emits two optimiser groups accordingly.

Trainable parameters per phase (measured):

| Model | Phase 1 (head only) | Phase 2 (all) |
|---|---|---|
| GoogLeNet | 10,250 | 5,610,154 |
| ResNet18 | 5,130 | 11,181,642 |
| VGG11 | 40,970 | 128,807,306 |

A note on VGG11: only `classifier[6]` is new, so its two preceding 4096-wide
fully-connected layers count as backbone and stay frozen in Phase 1. That leaves
Phase 1 training just 0.03% of the network — expect very little to happen for
VGG11 until Phase 2.

**Head initialisation** is PyTorch's `nn.Linear` default (Kaiming-uniform,
`a=sqrt(5)`). No custom scheme is applied: the default is well-tested and
appropriate for a single linear layer feeding softmax, and overriding it without
a measured reason would be an unjustified deviation.

### Input resolution — the key preprocessing decision

CIFAR-10 is natively 32×32 while these architectures were designed for 224×224
inputs. Rather than resize to 224 by default, the resolution was chosen from a
measurement taken on the actual target hardware.

**Architectural constraint.** All three networks reduce spatial resolution by a
factor of 32 before their classifier. ResNet18 and GoogLeNet additionally open
with a 7×7 stride-2 convolution followed by a stride-2 max-pool, which discards
most of a 32×32 image immediately. The resulting final feature map is:

| Input | ResNet18 / GoogLeNet / VGG11 feature map | Assessment |
|---|---|---|
| 32px | 1×1 | Degenerate — global pooling becomes a no-op; GoogLeNet aux also fails on BN |
| 64px | 2×2 | Very coarse |
| 96px | 3×3 | Workable |
| 128px | 4×4 | Comfortable |
| 224px | 7×7 | Native design point |

(Measured by execution, not inferred. All three networks downsample by 32×.)

So some upsampling is architecturally necessary, not optional.

**A VGG11-specific caveat.** ResNet18 and GoogLeNet end in
`AdaptiveAvgPool2d((1,1))`, so any feature map ≥1×1 reduces cleanly to a 512- or
1024-dim vector. VGG11 instead uses `AdaptiveAvgPool2d((7,7))` and a classifier
expecting 25,088 inputs. Below 224px that pool is *upsampling*: at 96px a 3×3 map
is stretched to 7×7, so the 25,088-dim vector is a fixed function of only
3×3×512 = 4,608 numbers. It is well-defined and trains fine, but VGG11 is the one
model of the three whose classifier is genuinely under-fed below 224px. This is
the strongest ML argument for a larger input, and it is why 128px is recommended
as the upgrade if GPU capacity becomes available.

**Compute constraint.** Measured on this machine (CPU-only, see Environment),
forward+backward, batch of 16, random tensors, `scripts/benchmark_resolution.py`:

| Resolution | ResNet18 | GoogLeNet | VGG11 | **All three, min/epoch** |
|---|---|---|---|---|
| 32px | 12.70 ms/img | 5.62 ms/img | 26.55 ms/img | **33.6** |
| 64px | 12.90 ms/img | 8.50 ms/img | 28.12 ms/img | **37.2** |
| 96px | 11.85 ms/img | 13.31 ms/img | 52.56 ms/img | **58.3** |
| 128px | 18.27 ms/img | 23.69 ms/img | 101.34 ms/img | **107.5** |
| 224px | 52.34 ms/img | 83.55 ms/img | 261.97 ms/img | **298.5** |

(min/epoch = 45,000 training images, training cost only, summed across the three
models.)

**Decision: 128×128 for final training (approved Stage 2); 96×96 retained for
local CPU smoke tests only.**

128px was chosen over 96px on the VGG11 argument above: it lifts that model's
classifier utilisation from ~18% to ~33% of its 25,088 inputs, and gives all
three networks a 4×4 rather than 3×3 final feature map. It was chosen over 224px
because the source images are 32×32 — 224px is a 7× upsample that adds no
information while costing roughly 5× more compute than 96px.

Full training runs in a GPU environment. On this CPU-only machine 224px measured
~5 hours per epoch across the three models, so local runs are limited to smoke
tests at 96px, which must never produce reported results. Note that ResNet18 is essentially free between 32
and 96px — it is memory-bound rather than compute-bound at those sizes — so
dropping below 96 buys nothing for that model.

A point worth stating plainly: **upsampling adds no information.** The 32×32
source is the information ceiling regardless of the target size. Larger inputs
buy architectural compatibility with the pretrained weights, not extra detail.
This is why an intermediate resolution is a reasonable trade rather than a pure
loss.

*This is the single highest-impact parameter in the configuration and the most
likely candidate for revision if GPU hardware becomes available.*

### Normalisation

ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`),
for two reasons. The backbones are ImageNet-pretrained and their early filters
are calibrated for inputs in this range. Separately, using dataset-derived
statistics would require computing them from data — and those statistics would
then have to come from the training split alone to avoid leakage. Fixed
published constants remove that risk entirely.

For reference, the channel statistics of the training split alone are recorded
in `results/sanity_check_report.json`. They are documentation only; the pipeline
does not use them.

### Augmentation

Applied to the **training split only**:

- `RandomCrop(32, padding=4, padding_mode='reflect')` — the canonical CIFAR
  augmentation. Four pixels on a 32px image is up to a 12.5% translation.
  Reflect padding avoids black borders the network could learn as an artefact.
- `RandomHorizontalFlip(p=0.5)` — label-preserving for all ten classes.

Vertical flipping is deliberately excluded: an upside-down horse or ship is not
a plausible sample from the data distribution.

Augmentations run at native 32×32 **before** the resize, so the crop jitter has
a consistent physical meaning and the operation is cheaper. The resize,
tensor conversion and normalisation are then identical between training and
evaluation — augmentation is the only difference between the two paths.

Validation and test use a transform with no random component at all.

### GoogLeNet auxiliary classifiers

**Correction (Stage 1 QA).** An earlier revision of this README stated that the
auxiliary heads cannot be constructed below ~224px. That was wrong. torchvision's
`InceptionAux.forward` begins with `F.adaptive_avg_pool2d(x, (4, 4))`, so the
`Linear(2048, 1024)` always receives 128×4×4 = 2048 features regardless of input
size. The aux heads construct and run correctly at 128px — verified by execution
at 48/64/96/128/224px. Only 32px fails, and for an unrelated reason (BatchNorm
cannot compute statistics over a 1×1 spatial map in training mode).

They are nonetheless disabled, for reasons that are not about resolution:

1. torchvision documents the pretrained GoogLeNet's aux heads as **not
   pretrained**; enabling them emits a warning to that effect. They would start
   from effectively untrained weights and inject noise into a short fine-tune.
2. They affect **training only**. In `eval()` the model returns a single logits
   tensor either way, so this choice cannot alter ensemble inference.
3. Their purpose was to counter vanishing gradients when training a 22-layer
   pre-BatchNorm network from scratch in 2014. torchvision's `BasicConv2d`
   includes BatchNorm, and we fine-tune from pretrained weights, so that purpose
   is largely obsolete here.
4. It keeps a uniform single-tensor output across all three models.

Pretrained weights load correctly with the heads disabled: torchvision loads the
full checkpoint (which does contain 20 `aux.*` keys) and then discards them.

### GoogLeNet `transform_input`

Worth recording because it is easy to miss. When `googlenet(weights=...)` is
called, torchvision silently sets `transform_input=True`. The model then
internally re-maps its input from ImageNet normalisation to the TF-style
[-1, 1] convention its ported weights expect. This means feeding
ImageNet-normalised tensors is **correct** for this model — but only while
`transform_input` stays enabled. Stage 2 must not override it.

## Training Infrastructure (Stage 3A)

Implemented and tested; **no training run has been performed.** Every number in
this section is a file size, a parameter count or a test count — there are no
performance results, because none exist.

### Design

One loop serves all three architectures. Architecture-specific behaviour stays
in `src/models.py` (construction, head prefixes, parameter grouping), so
`src/training.py` contains no per-model branching.

| Module | Responsibility |
|---|---|
| `src/training.py` | Epoch loop, optimiser, scheduler, AMP, early stopping, orchestration |
| `src/metrics.py` | Sample-weighted loss and accuracy |
| `src/checkpointing.py` | Atomic save, load, and resume |
| `scripts/train.py` | Stage 3B entry point |
| `scripts/run_stage3b.py` | Stage 3B orchestrator: pre-flight gates, resume planning, the three runs |
| `scripts/recover_checkpoints.py` | Finds checkpoints left by a lost runtime and copies them somewhere persistent |
| `scripts/evaluate_ensemble.py` | Stage 4: the single evaluation of the official test set |
| `scripts/smoke_test_training.py` | Synthetic device / AMP / checkpoint smoke test |

No separate `reproducibility.py` was added — `src/seed.py` from Stage 1 already
seeds Python, NumPy, torch and CUDA and provides the DataLoader worker seeder, so
a second module would only have wrapped it.

### Validation-only tuning policy

`fit()` takes a training loader and a validation loader. **There is no third
parameter**, so no call site can pass the test set into training, and a test
asserts the signature contains no test-related parameter. Model selection, early
stopping and checkpoint promotion all read validation accuracy only. The cosine
schedule advances on epoch count rather than on a metric, so it cannot leak
information from any split.

### Metrics

Aggregation is weighted by **samples, not batches**: `total_correct / total_samples`
and a batch-size-weighted mean loss. The validation loader does not drop its last
batch, so unequal batch sizes are normal and a mean-of-batch-means would
overweight the small final batch. Accumulators are Python floats, never tensors,
so no autograd graph or CUDA tensor is held across an epoch. An empty pass
raises rather than reporting `0.0`, which would look like a measurement.

### Optimiser and scheduler

SGD with Nesterov momentum by default, as the project already specified; AdamW is
supported and selectable. Discriminative learning rates come from Stage 2's
`build_param_groups()`, which also excludes frozen parameters so a frozen
backbone carries no optimiser state.

The schedule is a linear warmup followed by cosine annealing, and **steps once
per epoch, after validation**. Because stepping is per epoch, a one-epoch warmup
means the entire first epoch runs at the reduced rate rather than ramping within
it — coarser than per-iteration warmup, and deliberate. `eta_min` is an absolute
floor applied to every group, so the backbone and head groups converge toward the
same final rate from different starting points.

**All learning rates, epoch counts and decay settings are placeholders.** None
has been validated.

### Mixed precision

Configurable and off by default. Uses the current `torch.amp` API
(`torch.amp.autocast`, `torch.amp.GradScaler`); the deprecated `torch.cuda.amp`
entry points are not used, and a test asserts no deprecation warning is emitted.
AMP is CUDA-only: requesting it on CPU is not an error but is disabled, with the
reason recorded rather than silently ignored. Training is numerically correct
with AMP off, and enabling it changes no methodology. It has **not** been
benchmarked and is not assumed to be beneficial.

### Gradient clipping

Configurable, default max-norm 5.0 — VGG11 has no batch normalisation and is the
most gradient-sensitive of the three. Clipping runs **after `backward()` and
before `optimizer.step()`**, and under AMP the gradients are unscaled first, or
the threshold would be applied to scaled values and mean nothing. When clipping
is disabled no clipping call is made at all; a test asserts it is not invoked.

### Checkpointing and resume

Two checkpoints per architecture: a rolling `last` for resume and a `best`
promoted on validation-accuracy improvement. Writes are atomic (temp file plus
`os.replace`), so an interrupted save cannot destroy the previous checkpoint. The
best checkpoint is written straight to disk on improvement rather than held in
memory, which for VGG11 avoids keeping roughly 490 MB resident for the whole run.

`last` carries model, optimiser, scheduler, AMP scaler, epoch, best metric, the
effective configuration and RNG state. `best` is weights-only by default, since
it exists to be loaded for final evaluation and the ensemble stage. Measured
`best` sizes, before and after that change:

| Model | Full state | Weights-only |
|---|---|---|
| GoogLeNet | 44,066 KB | 22,088 KB |
| ResNet18 | 87,465 KB | 43,756 KB |
| VGG11 | 1,006,335 KB | 503,162 KB |

`torch.load` is called with `weights_only=True` — a checkpoint is a pickle, and
this refuses to execute arbitrary objects on load. RNG state is therefore stored
as primitives and rebuilt on read.

**Resume restores the optimiser, not just the scheduler.** This is load-bearing:
`SequentialLR.load_state_dict` restores the scheduler's counters but does *not*
write the learning rate back into `optimizer.param_groups`, so restoring the
scheduler alone silently resumes at the warmup rate. The learning rate lives in
the optimiser state. `restore_training_state` refuses to restore a scheduler
without its optimiser, and a regression test asserts a resumed run reproduces the
uninterrupted learning-rate trajectory exactly. A missing or
architecture-mismatched checkpoint raises rather than silently restarting from
epoch 0.

### Surviving a lost runtime

A hosted GPU runtime can be reclaimed mid-run, which leaves an architecture
partly trained. Three things make that recoverable rather than fatal:

- **`scripts/run_stage3b.py --resume auto`** (the default) continues any
  architecture that has a `last` checkpoint and starts the others at epoch 1. A
  fresh start over existing checkpoints is refused unless `--resume off
  --force-restart` says so explicitly, so re-running the orchestrator after a
  disconnect can never overwrite the work it is meant to continue.
- **`--checkpoints-dir` / `--results-dir`** redirect output to persistent
  storage — a mounted Google Drive folder — so the next disconnect costs
  nothing. Neither flag touches a hyperparameter.
- **`scripts/recover_checkpoints.py`** searches for checkpoints a previous
  runtime left behind, reports the architecture and epoch each one holds, and
  copies the furthest-along copy of each into persistent storage. It never
  overwrites a destination file that is already at a later epoch.

`verify_resume_compatibility` compares the checkpoint's recorded configuration
against the current one before a single epoch runs. Verified by execution:
resuming a checkpoint written under a *different epoch budget* restores the
scheduler's counters into a schedule of a different length, and the learning
rate silently collapses to `scheduler_min_lr` instead of following the cosine
curve — no exception is raised and the numbers still look plausible. Resuming
across differing hyperparameters is therefore refused rather than warned about.

Equivalence was measured end to end on a real ResNet18: an interrupted run
resumed from its `last` checkpoint reproduces the uninterrupted run's final
weights bit-exactly (maximum absolute difference 0.0) and follows the same
learning-rate trajectory.

`notebooks/colab_stage3b.ipynb` drives the whole sequence on Colab — mount
Drive, gate on CUDA, recover, pre-flight, train or resume, verify, then the
single test-set evaluation.

### Early stopping

Configurable patience and minimum improvement, monitoring validation accuracy
only. **Disabled by default** until tuning decisions are made.

### Reproducibility

`set_global_seeds()` seeds Python, NumPy, torch and CUDA, and requests
deterministic cuDNN kernels. DataLoader workers are seeded through `seed_worker`,
which derives each worker's seed from `torch.initial_seed()` so augmentation is
reproducible without every worker drawing identical augmentations.

Honest limits: seeding makes a run repeatable on the *same* machine and library
versions. It does not make results identical across different hardware, cuDNN
versions or thread counts. Deterministic cuDNN also costs throughput, and
`torch.use_deterministic_algorithms` is deliberately **not** forced, because it
makes some convolution kernels unavailable on CUDA and would slow training for a
guarantee that does not survive a change of GPU anyway. The train/validation
split does not rely on seeded RNG alone — it is derived by an explicit
deterministic algorithm and checksummed, so it is stable regardless of
environment.

### Logging

Per epoch: architecture, epoch, train loss, train accuracy, validation loss,
validation accuracy, per-group learning rates, epoch duration, best validation
metric, sample counts, and full device information. Written as JSON and CSV into
the git-ignored `results/` directory. No log has been generated from real data.

### GPU execution plan

Training will run on a Google Colab Tesla T4. The infrastructure is
device-agnostic: `select_device()` resolves CUDA when present and CPU otherwise,
and no device index is hard-coded anywhere (a test greps the sources for
`cuda:0`). On the current CPU-only development machine the CUDA-specific tests
**skip with an explicit reason and have not been run**; they must be executed on
the T4 before the infrastructure is considered verified there:

```bash
python scripts/smoke_test_training.py --architecture resnet18 --amp
python -m pytest tests/ -v
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml              # All meaningful parameters, with rationale
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Typed config loading and validation
│   ├── seed.py                  # Python / NumPy / torch / CUDA seeding
│   ├── dataset.py               # Split, transforms, datasets, dataloaders
│   ├── models.py                # Model construction + transfer-learning utils
│   ├── training.py              # Training loop, optimiser, scheduler, AMP
│   ├── metrics.py               # Sample-weighted loss and accuracy
│   ├── checkpointing.py         # Save / load / resume
│   ├── audit.py                 # Data-leakage audit
│   └── utils.py                 # Environment capture, tensor checks, helpers
│
├── scripts/
│   ├── benchmark_resolution.py  # Timing measurement behind the resolution choice
│   ├── run_sanity_checks.py     # Stage 1 verification + figures
│   ├── verify_models.py         # Stage 2 architecture verification
│   ├── train.py                 # Stage 3B entry point (train + validation only)
│   ├── run_stage3b.py           # Stage 3B orchestrator (pre-flight, resume, three runs)
│   ├── recover_checkpoints.py   # Recovers checkpoints from a lost runtime
│   ├── evaluate_ensemble.py     # Stage 4 ensemble evaluation (reads the test set once)
│   └── smoke_test_training.py   # Synthetic infrastructure smoke test
│
├── tests/
│   ├── conftest.py              # Shared synthetic fixtures
│   ├── test_data_pipeline.py    # Stage 1 pipeline tests
│   ├── test_models.py           # Stage 2 model tests
│   ├── test_training.py         # Stage 3A training-loop tests
│   ├── test_metrics.py          # Stage 3A metric tests
│   ├── test_checkpointing.py    # Stage 3A checkpoint / resume tests
│   ├── test_stage3b_orchestrator.py  # Resume planning and compatibility gates
│   └── test_recover_checkpoints.py   # Recovery scan and no-regression rules
│
├── notebooks/
│   └── colab_stage3b.ipynb      # One-shot Colab driver: recover, resume, evaluate
├── data/                        # CIFAR-10 (gitignored, auto-downloaded)
├── results/                     # Generated reports and figures (gitignored)
└── checkpoints/                 # Model weights (gitignored)
```

Two additions to the originally proposed structure, both load-bearing:
`data/` is the dataset root and must exist for torchvision to download into, and
`src/audit.py` holds the leakage checks, which are substantial enough that
folding them into `dataset.py` would obscure both.

## Environment

Recorded from the working environment:

| Component | Version |
|---|---|
| Python | 3.13.0 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| NumPy | 2.4.0 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| PyYAML | 6.0.3 |
| Pillow | 12.0.0 |
| pytest | 9.1.1 |

**Hardware**

| | |
|---|---|
| CPU | 12th Gen Intel Core i5-1240P (16 threads) |
| RAM | 8 GB |
| GPU | Intel Iris Xe (integrated) |
| CUDA available | **No** |

> **This is a CPU-only machine with no CUDA device.** The installed PyTorch is a
> CPU build. This constraint drove the input-resolution decision above and the
> DataLoader worker count, and it sets a hard ceiling on the training budget.

### Setup

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

On a CUDA machine, install a matching CUDA build of `torch`/`torchvision`
instead — torchvision 0.28.0 pins torch 2.13.0 exactly.

## Dataset Setup

CIFAR-10 downloads automatically on first use (~170 MB) into `data/`:

```bash
python scripts/run_sanity_checks.py
```

This downloads the dataset if needed, runs every Stage 1 verification, writes
`results/sanity_check_report.json`, and saves two figures drawn from real
dataset images.

## Reproducibility

- Python, NumPy, PyTorch and CUDA generators are seeded from a single value.
- DataLoader workers are seeded individually so augmentation is reproducible
  without every worker drawing identical augmentations.
- The train/validation split uses a **separate seed** from the training seed, so
  re-seeding a training experiment can never move the validation boundary.
- The split is a pure function of `(seed, targets)` rather than a stored file,
  and uses `np.random.RandomState`, whose stream NumPy's compatibility policy
  guarantees to be stable across versions. A checksum is recorded so drift is
  detectable.
- cuDNN determinism flags are set where applicable.

## Data Leakage Controls

`src/audit.py` verifies, rather than assumes:

- Train and validation index sets are disjoint and partition the official
  training set exactly.
- No image **content** is shared between train, validation and test — compared
  by SHA-256 of raw pixel bytes, which would catch leakage arriving by any route.
- The validation and test transforms contain no random operation.
- Train and validation wrap **separate dataset objects**, guarding against the
  common `random_split` bug where both halves share one transform and the
  validation set is silently augmented.
- Normalisation constants are not derived from any data.

## Verifying the Models

```bash
python scripts/verify_models.py
```

Builds all three models, loads ImageNet weights, and exercises forward passes,
train/eval behaviour, freezing, gradient flow, parameter counts and
serialisation. Writes `results/model_verification.json`. It trains nothing: no
dataset is read and no optimiser step is taken.

## Running the Tests

```bash
python -m pytest tests/ -v
```

Tests that only exercise the split algorithm run against synthetic labels and
need no downloaded data. Tests requiring real CIFAR-10 files skip automatically
when the dataset is absent.

The Stage 3A training tests use a four-parameter toy network and random-noise
batches. None of them performs a real training experiment, and none reads
CIFAR-10. CUDA-specific tests skip on a CPU-only host with an explicit reason
and must be run on the GPU environment.

## Roadmap

- [x] **Stage 1** — Project foundation and data pipeline *(approved)*
- [x] **Stage 2** — Model construction and verification *(approved)*
- [x] **Stage 3A** — Training infrastructure *(approved)*
- [ ] **Stage 3B** — Training and fine-tuning on the Colab T4 GPU *(started; interrupted by the loss of the runtime, resumable, not completed)*
- [ ] **Stage 4** — Equal-weight softmax probability ensemble *(evaluator implemented and tested; not yet run against trained weights)*
- [ ] **Stage 5** — Final test evaluation and error analysis
