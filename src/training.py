"""Training infrastructure: loop, optimiser, scheduler, AMP, early stopping.

This module contains no experiment. It provides the machinery a training run
needs and nothing that decides what that run should be; every hyperparameter
arrives from :mod:`src.config`.

Ordering rules that are easy to get wrong, and are enforced here
-----------------------------------------------------------------

*Gradients are cleared before the forward pass of every batch*, with
``set_to_none=True`` so stale gradients cannot be silently reused and the
gradient buffers are actually released between steps.

*Gradient clipping happens after ``backward()`` and before ``optimizer.step()``.*
Under AMP the gradients are scaled, so ``scaler.unscale_(optimizer)`` must run
first or the clipping threshold would be applied to scaled values and would mean
nothing.

*The scheduler steps once per epoch, after validation.* See
:func:`build_scheduler` for what that implies for warmup.

*Validation runs under ``torch.inference_mode()`` with the model in eval mode,
and never touches the optimiser.* ``inference_mode`` is used rather than
``no_grad`` because it also skips version counter bookkeeping, which keeps
activation memory lower on the large VGG11 forward pass.

The test set has no code path through this module. :func:`fit` accepts a
training loader and a validation loader, and there is no third argument.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.checkpointing import (
    checkpoint_path,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from src.config import Config
from src.metrics import EpochMetrics, MetricTracker
from src.models import build_param_groups, set_training_mode
from src.utils import select_device

SUPPORTED_OPTIMIZERS: tuple[str, ...] = ("sgd", "adamw")
SUPPORTED_SCHEDULERS: tuple[str, ...] = ("cosine", "none")

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInfo:
    """What the run is executing on, recorded alongside every result."""

    device: str
    type: str
    cuda_available: bool
    gpu_name: str | None = None
    capability: str | None = None
    total_memory_gb: float | None = None
    free_memory_gb: float | None = None
    memory_query_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_device(device: torch.device | None = None) -> DeviceInfo:
    """Report the selected device and, on CUDA, the GPU and its free memory.

    Args:
        device: Device to describe. Defaults to :func:`src.utils.select_device`,
            which resolves to CUDA when available and CPU otherwise. No device
            index is hard-coded anywhere.

    Returns:
        A :class:`DeviceInfo`. On CPU the GPU fields are ``None`` rather than
        fabricated placeholders.
    """
    device = select_device() if device is None else torch.device(device)
    cuda_available = torch.cuda.is_available()

    if device.type != "cuda" or not cuda_available:
        return DeviceInfo(
            device=str(device), type=device.type, cuda_available=cuda_available
        )

    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)

    free_gb: float | None = None
    error: str | None = None
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        free_gb = round(free_bytes / 1024**3, 3)
    except (RuntimeError, AssertionError) as exc:
        # Not fatal, but recorded rather than swallowed: some drivers and some
        # virtualised GPUs refuse this query.
        error = f"{type(exc).__name__}: {exc}"

    return DeviceInfo(
        device=str(device),
        type=device.type,
        cuda_available=True,
        gpu_name=props.name,
        capability=f"{props.major}.{props.minor}",
        total_memory_gb=round(props.total_memory / 1024**3, 3),
        free_memory_gb=free_gb,
        memory_query_error=error,
    )


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmpSettings:
    """Resolved mixed-precision configuration for one run."""

    enabled: bool
    dtype: torch.dtype | None
    device_type: str
    reason: str


def resolve_amp(config: Config, device: torch.device) -> AmpSettings:
    """Decide whether mixed precision is actually active for this run.

    AMP is a CUDA feature here. Requesting it on CPU is not an error - the same
    config should run in both places - but it is disabled and the reason is
    reported rather than silently ignored.
    """
    requested = bool(config.training.amp)
    dtype_name = str(config.training.amp_dtype).lower()
    if dtype_name not in _AMP_DTYPES:
        raise ValueError(
            f"Unknown amp_dtype {config.training.amp_dtype!r}; "
            f"expected one of {tuple(_AMP_DTYPES)}."
        )

    if not requested:
        return AmpSettings(False, None, device.type, "disabled by configuration")
    if device.type != "cuda":
        return AmpSettings(
            False,
            None,
            device.type,
            f"requested but disabled: device is {device.type!r}, not CUDA",
        )
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        # The target GPU for this project is a Tesla T4 (Turing, sm_75), which
        # has no native bfloat16 path - torch only offers it there through slow
        # emulation. Falling back to fp16 silently would change the numerics of
        # a run without saying so, so AMP is disabled and the reason recorded,
        # matching how a CPU request is handled above.
        return AmpSettings(
            False,
            None,
            device.type,
            "requested but disabled: bfloat16 is not natively supported by "
            f"{torch.cuda.get_device_name(index)}; "
            "use amp_dtype: float16 on this GPU",
        )
    return AmpSettings(True, _AMP_DTYPES[dtype_name], "cuda", f"enabled ({dtype_name})")


def build_scaler(amp: AmpSettings) -> torch.amp.GradScaler:
    """Create the gradient scaler.

    ``torch.amp.GradScaler`` is the current API; ``torch.cuda.amp.GradScaler``
    is deprecated in this PyTorch build and emits a warning (verified by
    execution). A disabled scaler is still constructed so the training loop has
    one object to talk to in both modes - its ``scale``/``step``/``update`` calls
    become pass-throughs, and its state dict is empty.
    """
    return torch.amp.GradScaler(device=amp.device_type, enabled=amp.enabled)


# ---------------------------------------------------------------------------
# Criterion, optimiser, scheduler
# ---------------------------------------------------------------------------


def build_criterion(config: Config) -> nn.Module:
    """Cross-entropy with the configured label smoothing.

    ``label_smoothing`` is 0.0 by project decision: the final stage fuses
    softmax probabilities across models, and smoothing systematically flattens
    confidence in a way that would distort that fusion.
    """
    return nn.CrossEntropyLoss(label_smoothing=float(config.training.label_smoothing))


def build_optimizer(
    model: nn.Module, architecture: str, config: Config
) -> torch.optim.Optimizer:
    """Build the optimiser with discriminative learning rates.

    Parameter partitioning is delegated to
    :func:`src.models.build_param_groups`, which is the Stage 2 utility that
    knows each architecture's head prefix. It excludes frozen parameters, so a
    frozen backbone carries no optimiser state.

    The learning rates in the configuration are PLACEHOLDERS. They have not been
    tuned, and nothing here claims they are appropriate.
    """
    name = str(config.training.optimizer).lower()
    if name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer {config.training.optimizer!r}; "
            f"expected one of {SUPPORTED_OPTIMIZERS}."
        )

    groups = build_param_groups(
        model,
        architecture,
        backbone_lr=float(config.training.learning_rate),
        head_lr=float(config.training.head_learning_rate),
    )
    if not groups:
        raise ValueError(
            "No trainable parameters were found; the optimiser would have "
            "nothing to update. Check the freeze configuration."
        )

    weight_decay = float(config.training.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=float(config.training.momentum),
            weight_decay=weight_decay,
            nesterov=bool(config.training.nesterov),
        )
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Config, epochs: int
) -> Any | None:
    """Build the learning-rate schedule.

    **Stepping frequency: once per epoch, after validation.** This is stated
    here because the implementation and the documentation have to agree, and
    ``fit`` calls ``scheduler.step()`` exactly once per epoch.

    The schedule is a linear warmup followed by cosine annealing:

    * Warmup runs for ``scheduler_warmup_epochs`` epochs, starting at
      ``scheduler_warmup_start_factor`` times each group's base learning rate.
      Because stepping is per epoch, a one-epoch warmup means the whole first
      epoch runs at the reduced rate rather than ramping within it. That still
      serves the stated purpose - keeping the high head learning rate from
      destabilising the pretrained backbone at the start - but it is a coarser
      instrument than per-iteration warmup, and the difference is deliberate,
      not accidental.
    * Cosine annealing then runs over the remaining epochs, decaying each group
      from its own base learning rate toward ``scheduler_min_lr``. Note that
      ``eta_min`` is an absolute floor applied to every group, so the backbone
      and head groups converge to the same final value from different starts.

    Cosine is chosen because it is deterministic, has one meaningful knob, needs
    no validation signal to advance (so it cannot leak information), and
    restores exactly from a checkpoint. It is not claimed to be optimal.

    Returns:
        A scheduler, or ``None`` when scheduling is disabled.
    """
    name = str(config.training.scheduler).lower()
    if name not in SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler {config.training.scheduler!r}; "
            f"expected one of {SUPPORTED_SCHEDULERS}."
        )
    if name == "none":
        return None

    warmup_epochs = int(config.training.scheduler_warmup_epochs)
    if warmup_epochs < 0:
        raise ValueError("scheduler_warmup_epochs must be >= 0.")
    if warmup_epochs >= epochs:
        raise ValueError(
            f"scheduler_warmup_epochs ({warmup_epochs}) must be less than the "
            f"number of epochs ({epochs}), otherwise the whole run is warmup and "
            "the cosine phase never happens. For a short run set "
            "scheduler_warmup_epochs to 0, or set scheduler to 'none'."
        )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=float(config.training.scheduler_min_lr),
    )
    if warmup_epochs == 0:
        return cosine

    warmup = LinearLR(
        optimizer,
        start_factor=float(config.training.scheduler_warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def current_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Learning rate per parameter group, keyed by the group's name."""
    rates: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        rates[str(group.get("name", f"group{index}"))] = float(group["lr"])
    return rates


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop when a validation metric has not improved for ``patience`` epochs.

    Only validation metrics are ever passed to this class. The test set is not
    available to it, and model selection that consulted the test set would
    invalidate the final evaluation.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        mode: str = "max",
        enabled: bool = True,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}.")
        if patience < 0:
            raise ValueError("patience must be >= 0.")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0.")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.enabled = enabled
        self.best: float | None = None
        self.epochs_without_improvement = 0
        self.stopped = False

    def is_improvement(self, value: float) -> bool:
        """Whether ``value`` beats the best seen by more than ``min_delta``."""
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def update(self, value: float) -> bool:
        """Record a validation metric.

        Returns:
            True when this value is an improvement over the best so far.
        """
        improved = self.is_improvement(value)
        if improved:
            self.best = float(value)
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.enabled and self.epochs_without_improvement >= self.patience:
                self.stopped = True
        return improved

    @property
    def should_stop(self) -> bool:
        """True once patience has been exhausted and stopping is enabled."""
        return self.enabled and self.stopped


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _require_logits(output: Any, architecture: str) -> torch.Tensor:
    """Assert the model returned a plain logits tensor.

    GoogLeNet returns a ``GoogLeNetOutputs`` namedtuple in training mode when
    auxiliary classifiers are enabled. The approved configuration disables them,
    so a plain tensor is expected here and the loop contains no auxiliary-loss
    handling. Rather than quietly unwrapping such an output - which would drop
    the auxiliary losses and change the training objective without saying so -
    this raises.
    """
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(
        f"The {architecture!r} model returned {type(output).__name__} rather "
        "than a plain logits tensor. This happens when GoogLeNet is built with "
        "aux_logits=True. The approved configuration is aux_logits=False, and "
        "this training loop does not implement an auxiliary loss term; adding "
        "one would change the training objective and needs a Project Lead "
        "decision."
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    architecture: str,
    scaler: torch.amp.GradScaler | None = None,
    amp: AmpSettings | None = None,
    max_grad_norm: float | None = None,
    backbone_frozen: bool = False,
) -> EpochMetrics:
    """Run one training epoch and return its metrics.

    Args:
        model: Model to update. Put into train mode here.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimiser whose ``step`` is called once per batch.
        device: Device for the model and batches.
        architecture: Architecture identifier, used for output validation.
        scaler: AMP gradient scaler. May be a disabled scaler.
        amp: Resolved mixed-precision settings.
        max_grad_norm: Clip gradients to this norm. ``None`` or a
            non-positive value disables clipping entirely - no clipping call is
            made in that case.
        backbone_frozen: Passed through to
            :func:`src.models.set_training_mode` so a frozen backbone's
            normalisation layers stay in eval mode.

    Returns:
        Sample-weighted loss and accuracy over the epoch.
    """
    amp = amp or AmpSettings(False, None, device.type, "not supplied")
    set_training_mode(model, True, backbone_frozen=backbone_frozen)
    tracker = MetricTracker()
    clipping_enabled = max_grad_norm is not None and float(max_grad_norm) > 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Cleared before the forward pass; set_to_none releases the buffers.
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
        ):
            outputs = _require_logits(model(images), architecture)
            loss = criterion(outputs, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if clipping_enabled:
                # Gradients must be unscaled before the norm means anything.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clipping_enabled:
                nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()

        # detach() so no graph is retained; the tracker converts to float.
        tracker.update(outputs.detach(), targets, loss.detach())

    return tracker.compute()


def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    *,
    architecture: str,
    amp: AmpSettings | None = None,
) -> EpochMetrics:
    """Evaluate on a loader without touching the optimiser.

    The model is switched to eval mode and the whole pass runs under
    ``torch.inference_mode()``, so no gradients exist to apply even if an
    optimiser step were mistakenly attempted.

    This function is used for the VALIDATION split. Applying it to the test set
    is a final-evaluation-stage action and is not part of any training path.
    """
    amp = amp or AmpSettings(False, None, device.type, "not supplied")
    model.eval()
    tracker = MetricTracker()

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
            ):
                outputs = _require_logits(model(images), architecture)
                loss = criterion(outputs, targets)
            tracker.update(outputs, targets, loss)

    return tracker.compute()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochRecord:
    """One row of the training log."""

    architecture: str
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rates: dict[str, float]
    epoch_seconds: float
    best_val_accuracy: float
    is_best: bool
    train_samples: int
    val_samples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_row(self) -> dict[str, Any]:
        """Flatten for CSV, expanding the per-group learning rates."""
        row = self.as_dict()
        rates = row.pop("learning_rates")
        for name, value in rates.items():
            row[f"lr_{name}"] = value
        return row


@dataclass
class TrainingHistory:
    """Everything one run produced, apart from the checkpoints themselves."""

    architecture: str
    records: list[EpochRecord] = field(default_factory=list)
    best_val_accuracy: float | None = None
    best_epoch: int | None = None
    stopped_early: bool = False
    completed_epochs: int = 0
    device_info: dict[str, Any] = field(default_factory=dict)
    amp: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [r.as_dict() for r in self.records]
        return payload

    def as_rows(self) -> list[dict[str, Any]]:
        return [r.as_row() for r in self.records]


def fit(
    model: nn.Module,
    *,
    architecture: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    device: torch.device | None = None,
    epochs: int | None = None,
    checkpoint_dir: Path | str | None = None,
    resume_from: Path | str | None = None,
    backbone_frozen: bool | None = None,
    log: Callable[[str], None] = print,
) -> TrainingHistory:
    """Train ``model`` for a number of epochs, selecting on validation accuracy.

    Exactly two loaders are accepted. There is no test loader parameter, so no
    call site can accidentally pass the test set through this function.

    Args:
        model: The model to train. Moved to ``device`` here.
        architecture: One of the approved architecture identifiers.
        train_loader: Training data.
        val_loader: Validation data. Used for scheduling-independent model
            selection and early stopping, and for nothing else.
        config: Loaded project configuration; all hyperparameters come from it.
        device: Compute device. Defaults to :func:`src.utils.select_device`.
        epochs: Overrides ``config.training.epochs`` when given.
        checkpoint_dir: Overrides ``config.paths.checkpoints_dir`` when given.
        resume_from: Path to a checkpoint to resume from. When given and the
            file is missing or incompatible, this raises rather than silently
            restarting from epoch 0.
        backbone_frozen: Overrides ``config.training.freeze_backbone``.
        log: Where human-readable progress lines go.

    Returns:
        A :class:`TrainingHistory`. Checkpoints are written as a side effect
        according to the configured policy.
    """
    device = select_device() if device is None else torch.device(device)
    total_epochs = int(config.training.epochs if epochs is None else epochs)
    if total_epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {total_epochs}.")

    frozen = (
        bool(config.training.freeze_backbone)
        if backbone_frozen is None
        else bool(backbone_frozen)
    )
    directory = Path(
        config.paths.checkpoints_dir if checkpoint_dir is None else checkpoint_dir
    )

    model.to(device)
    device_info = describe_device(device)
    amp = resolve_amp(config, device)
    scaler = build_scaler(amp)

    criterion = build_criterion(config)
    optimizer = build_optimizer(model, architecture, config)
    scheduler = build_scheduler(optimizer, config, total_epochs)

    stopper = EarlyStopping(
        patience=int(config.training.early_stopping_patience),
        min_delta=float(config.training.early_stopping_min_delta),
        mode="max",
        enabled=bool(config.training.early_stopping_enabled),
    )

    history = TrainingHistory(
        architecture=architecture,
        device_info=device_info.as_dict(),
        amp={"enabled": amp.enabled, "reason": amp.reason},
    )

    log(f"device: {device_info.device} ({device_info.gpu_name or device_info.type})")
    log(f"mixed precision: {amp.reason}")
    if frozen:
        log("backbone frozen: normalisation layers held in eval mode")

    start_epoch = 0
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, map_location=device)
        state = restore_training_state(
            checkpoint,
            model=model,
            architecture=architecture,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = state.start_epoch
        if state.best_metric is not None:
            stopper.best = float(state.best_metric)
            history.best_val_accuracy = float(state.best_metric)
            history.best_epoch = state.best_epoch
        log(f"resumed from {resume_from} at epoch {start_epoch}")
        log(f"  restored: {', '.join(state.restored)}")
        if state.skipped:
            log(f"  skipped:  {', '.join(state.skipped)}")
        history.notes.append(f"resumed from {resume_from} at epoch {start_epoch}")
        if start_epoch >= total_epochs:
            log(
                f"checkpoint is already at epoch {start_epoch} of {total_epochs}; "
                "nothing to do."
            )
            history.completed_epochs = start_epoch
            return history

    save_best = bool(config.training.save_best_checkpoint)
    save_last = bool(config.training.save_last_checkpoint)
    save_every = int(config.training.save_every_n_epochs)

    for epoch in range(start_epoch, total_epochs):
        rates = current_learning_rates(optimizer)
        started = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            architecture=architecture,
            scaler=scaler,
            amp=amp,
            max_grad_norm=config.training.grad_clip_norm,
            backbone_frozen=frozen,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            architecture=architecture,
            amp=amp,
        )
        duration = time.perf_counter() - started

        # Stepped once per epoch, after validation, matching build_scheduler's
        # documented contract.
        if scheduler is not None:
            scheduler.step()

        improved = stopper.update(val_metrics.accuracy)
        if improved:
            history.best_val_accuracy = val_metrics.accuracy
            history.best_epoch = epoch

        history.records.append(
            EpochRecord(
                architecture=architecture,
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                val_loss=val_metrics.loss,
                val_accuracy=val_metrics.accuracy,
                learning_rates=rates,
                epoch_seconds=duration,
                best_val_accuracy=float(history.best_val_accuracy or 0.0),
                is_best=improved,
                train_samples=train_metrics.num_samples,
                val_samples=val_metrics.num_samples,
            )
        )
        history.completed_epochs = epoch + 1

        log(
            f"epoch {epoch + 1}/{total_epochs}  "
            f"train_loss {train_metrics.loss:.4f}  "
            f"train_acc {train_metrics.accuracy:.4f}  "
            f"val_loss {val_metrics.loss:.4f}  "
            f"val_acc {val_metrics.accuracy:.4f}  "
            f"lr {'/'.join(f'{v:.2e}' for v in rates.values())}  "
            f"{duration:.1f}s{'  *best' if improved else ''}"
        )

        snapshot_args: dict[str, Any] = {
            "model": model,
            "architecture": architecture,
            "epoch": epoch,
            "best_metric": history.best_val_accuracy,
            "best_epoch": history.best_epoch,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            # The dataclass, not config.raw["training"]: it reflects any
            # override the caller applied, so the snapshot records what the run
            # actually used rather than what the YAML file happened to say.
            "config_snapshot": asdict(config.training),
        }
        if save_best and improved:
            best_args = dict(snapshot_args)
            if config.training.best_checkpoint_weights_only:
                # The best checkpoint is loaded for final evaluation and for the
                # ensemble stage, neither of which needs optimiser momentum
                # buffers. Dropping them roughly halves the file - about 490 MB
                # per write for VGG11. `last` keeps full state for resume.
                best_args.update(
                    optimizer=None, scheduler=None, scaler=None, include_rng_state=False
                )
            save_checkpoint(
                checkpoint_path(directory, architecture, "best"), **best_args
            )
        periodic = save_every > 0 and (epoch + 1) % save_every == 0
        if save_last or periodic:
            save_checkpoint(
                checkpoint_path(directory, architecture, "last"), **snapshot_args
            )

        if stopper.should_stop:
            history.stopped_early = True
            message = (
                f"early stopping at epoch {epoch + 1}: validation accuracy has "
                f"not improved by more than {stopper.min_delta} for "
                f"{stopper.epochs_without_improvement} epochs "
                f"(patience {stopper.patience})"
            )
            log(message)
            history.notes.append(message)
            break

    return history
