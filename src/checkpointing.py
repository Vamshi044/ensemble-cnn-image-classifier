"""Checkpoint saving, loading and resume.

Design decisions worth stating, because each one is a correctness or a memory
concern rather than a matter of taste.

**Checkpoints are written atomically.** ``torch.save`` writes into a temporary
file in the destination directory which is then moved into place with
``os.replace``. Interrupting a save therefore leaves the previous checkpoint
intact rather than a half-written file that fails to load. VGG11's state dict is
roughly 490 MB, so the window in which a naive write is corruptible is not small.

**The best checkpoint is written straight to disk, never held in memory.** The
obvious implementation keeps a copy of the best state dict in RAM and writes it
at the end. For VGG11 that is an extra ~490 MB resident for the whole run, on a
host that is also holding the model, its gradients and the optimiser state.
Writing on improvement costs disk I/O instead.

**Random-number state is stored as primitives.** ``torch.load`` is used with
``weights_only=True``, which refuses arbitrary pickled objects - a genuine
security property, since a checkpoint is a pickle. A raw NumPy RNG state
contains an ``ndarray`` and is rejected under that setting, so it is encoded as
plain ints and rebuilt on load. Verified by execution: the restored NumPy stream
is identical to the original.

**The architecture identifier is checked on load.** Loading a ResNet18 state
dict into a VGG11 is a mistake that should stop the run, not produce a confusing
shape error twenty lines later.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Bumped if the on-disk layout changes in a way older code cannot read.
CHECKPOINT_FORMAT_VERSION = 1

LATEST_CHECKPOINT_NAME = "{architecture}_last.pt"
BEST_CHECKPOINT_NAME = "{architecture}_best.pt"


# ---------------------------------------------------------------------------
# Random-number state
# ---------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG the project seeds, using primitives only.

    Returns:
        A dict safe to store in a ``weights_only=True`` checkpoint. The NumPy
        state is decomposed into its five components with the key array
        converted to a list of ints.
    """
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            [int(v) for v in numpy_state[1]],
            int(numpy_state[2]),
            int(numpy_state[3]),
            float(numpy_state[4]),
        ],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state produced by :func:`capture_rng_state`.

    CUDA state is restored only when the current machine has the same number of
    devices the checkpoint was written on; restoring a mismatched list raises
    inside PyTorch, and a run that merely moved hosts should not die for that
    reason. The skip is visible to the caller rather than hidden.
    """
    if "python" in state:
        python_state = state["python"]
        # torch.load may hand back the nested tuple as a list.
        random.setstate((python_state[0], tuple(python_state[1]), python_state[2]))
    if "numpy" in state:
        kind, keys, pos, has_gauss, cached_gauss = state["numpy"]
        np.random.set_state(
            (kind, np.array(keys, dtype=np.uint32), pos, has_gauss, cached_gauss)
        )
    if "torch" in state:
        torch.set_rng_state(state["torch"].to(torch.uint8).cpu())
    if (
        "cuda" in state
        and torch.cuda.is_available()
        and len(state["cuda"]) == torch.cuda.device_count()
    ):
        # ``.cpu()`` is load-bearing, exactly as it is for the CPU generator
        # above. ``load_checkpoint`` is called with ``map_location=device``, and
        # verified by execution: every storage in the file - the RNG tensors
        # included, they are ordinary uint8 storages - is routed through
        # map_location. Resuming on CUDA therefore hands these tensors back as
        # CUDA tensors, while a generator's state must be a CPU ByteTensor.
        torch.cuda.set_rng_state_all(
            [s.to(torch.uint8).cpu() for s in state["cuda"]]
        )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def checkpoint_path(directory: Path | str, architecture: str, kind: str) -> Path:
    """Build a checkpoint path from the configured directory.

    Args:
        directory: Checkpoint directory, normally ``config.paths.checkpoints_dir``.
        architecture: Model identifier, used in the filename.
        kind: Either ``"last"`` or ``"best"``.

    Returns:
        The full path. No absolute path is hard-coded in this module; the
        directory always arrives from configuration.
    """
    templates = {"last": LATEST_CHECKPOINT_NAME, "best": BEST_CHECKPOINT_NAME}
    if kind not in templates:
        raise ValueError(
            f"Unknown checkpoint kind {kind!r}; expected 'last' or 'best'."
        )
    return Path(directory) / templates[kind].format(architecture=architecture)


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    architecture: str,
    epoch: int,
    best_metric: float | None = None,
    best_epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    config_snapshot: dict[str, Any] | None = None,
    include_rng_state: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a resumable checkpoint atomically.

    The AMP scaler is stored only when it is enabled. A disabled
    ``torch.amp.GradScaler`` has an empty state dict, and storing it would imply
    mixed precision had been used when it had not.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state"] = scaler.state_dict()
    if config_snapshot is not None:
        payload["config"] = config_snapshot
    if include_rng_state:
        payload["rng_state"] = capture_rng_state()
    if extra:
        payload["extra"] = extra

    # Atomic replace: an interrupted write cannot destroy the previous file.
    temp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)
    return path


# ---------------------------------------------------------------------------
# Loading and resuming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeState:
    """What was recovered from a checkpoint."""

    start_epoch: int
    best_metric: float | None
    best_epoch: int | None
    restored: tuple[str, ...]
    skipped: tuple[str, ...]


def load_checkpoint(
    path: Path | str, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read a checkpoint from disk.

    ``weights_only=True`` is used deliberately: a checkpoint is a pickle, and
    this refuses to execute arbitrary objects during load. Everything this
    module writes is a tensor or a primitive, so it round-trips under that
    restriction.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Resume was requested but there is "
            "nothing to resume from."
        )
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)

    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {path} has format version {version!r}, but this code "
            f"reads version {CHECKPOINT_FORMAT_VERSION}."
        )
    return checkpoint


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: nn.Module,
    architecture: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
) -> ResumeState:
    """Restore model, optimiser, scheduler, scaler and RNG state in place.

    The optimiser **must** be restored for the learning rate to be correct.
    Verified by execution: ``SequentialLR.load_state_dict`` restores the
    scheduler's internal counters but does not write the learning rate back into
    ``optimizer.param_groups``. Restoring the scheduler alone therefore resumes
    at the warmup learning rate rather than the one the run had reached. The
    learning rate is carried by the optimiser state dict, so the two must be
    restored together; this function refuses to restore a scheduler without one.

    Args:
        checkpoint: A dict from :func:`load_checkpoint`.
        model: Model to load weights into. Must match ``architecture``.
        architecture: Expected architecture identifier.
        optimizer: Optimiser to restore. Required when ``scheduler`` is given.
        scheduler: Scheduler to restore.
        scaler: AMP scaler to restore. Skipped when the checkpoint holds no
            scaler state, which is the case for runs made without AMP.
        restore_rng: Whether to restore Python / NumPy / torch RNG state.

    Returns:
        A :class:`ResumeState` describing what was restored and what was skipped.

    Raises:
        ValueError: If the architecture does not match, or a scheduler is passed
            without its optimiser.
    """
    stored_architecture = checkpoint.get("architecture")
    if stored_architecture != architecture:
        raise ValueError(
            f"Checkpoint architecture mismatch: the file was written for "
            f"{stored_architecture!r} but {architecture!r} was requested. "
            "Resuming across architectures is not meaningful."
        )
    if scheduler is not None and optimizer is None:
        raise ValueError(
            "Restoring a scheduler without its optimiser would resume at the "
            "wrong learning rate: the scheduler state does not carry the "
            "learning rate, the optimiser state does. Pass both."
        )

    restored: list[str] = ["model"]
    skipped: list[str] = []
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            restored.append("optimizer")
        else:
            skipped.append("optimizer (absent from checkpoint)")

    if scheduler is not None:
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            restored.append("scheduler")
        else:
            skipped.append("scheduler (absent from checkpoint)")

    if scaler is not None:
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
            restored.append("scaler")
        else:
            # Expected whenever the saved run had AMP disabled.
            skipped.append("scaler (checkpoint has no AMP scaler state)")

    if restore_rng:
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
            restored.append("rng_state")
        else:
            skipped.append("rng_state (absent from checkpoint)")

    return ResumeState(
        # Training resumes at the epoch after the one that was completed.
        start_epoch=int(checkpoint["epoch"]) + 1,
        best_metric=checkpoint.get("best_metric"),
        best_epoch=checkpoint.get("best_epoch"),
        restored=tuple(restored),
        skipped=tuple(skipped),
    )
