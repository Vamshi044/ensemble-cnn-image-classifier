"""Reproducibility controls.

Randomness in this project enters from four places: the Python ``random``
module, NumPy, PyTorch's CPU generator, and PyTorch's CUDA generators. All four
are seeded here from a single value so a run can be repeated.

A note on the limits of this: seeding makes a run repeatable on the *same*
machine and library versions. It does not make results identical across
different hardware, cuDNN versions, or thread counts. The train/validation
split is therefore *not* left to a seeded RNG alone - it is derived by an
explicit deterministic algorithm (see :mod:`src.dataset`) so it is stable
regardless of environment.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed every random source the project uses.

    Args:
        seed: The seed value applied to all generators.
        deterministic: When True, ask cuDNN for deterministic kernel selection
            and disable its autotuner. This costs some throughput but removes a
            source of run-to-run variation. On a CPU-only machine these flags
            are harmless no-ops.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker process.

    Each worker inherits a distinct ``torch.initial_seed()`` derived from the
    main process generator. Deriving the Python and NumPy seeds from it keeps
    augmentation reproducible without making every worker draw the *same*
    augmentations, which would silently reduce augmentation diversity.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed: int) -> torch.Generator:
    """Return a seeded generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
