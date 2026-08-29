"""CIFAR-10 data pipeline: splitting, transforms and DataLoaders.

Design notes that matter for correctness
----------------------------------------

**Only the official 50,000-image training set is split.** The official
10,000-image test set is loaded through a separate code path and is never
touched by the split logic, so it cannot leak into training or validation.

**Augmentation is bound to the split, not applied after it.** The common bug in
this kind of pipeline is to build one dataset object, wrap it in
``random_split``, and end up with the *same* transform on both halves - which
silently augments the validation set. This module avoids that by constructing
two independent ``CIFAR10`` objects over the same on-disk files, one carrying
the training transform and one carrying the deterministic evaluation transform,
and then indexing each with its own disjoint subset of indices.

**The split is derived, not stored.** ``build_stratified_split`` is a pure
function of ``(seed, targets)``, so the same split is reproduced on any machine.
A small manifest with a checksum is written alongside it so drift is detectable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode

from src.config import Config
from src.seed import build_generator, seed_worker

_INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint index arrays into the official CIFAR-10 training set."""

    train: np.ndarray
    val: np.ndarray

    def checksum(self) -> str:
        """Stable fingerprint of the split, for cross-machine verification."""
        payload = self.val.astype(np.int64).tobytes()
        return hashlib.sha256(payload).hexdigest()[:16]


def build_stratified_split(
    targets: list[int] | np.ndarray,
    val_size: int,
    seed: int,
    num_classes: int = 10,
) -> SplitIndices:
    """Split training indices into train/validation, preserving class balance.

    CIFAR-10's training set holds exactly 5,000 images per class, so a
    class-stratified 45,000/5,000 split yields exactly 4,500/500 per class and
    the validation set is a faithful miniature of the training distribution.
    Stratifying matters because a purely random 10% draw would leave validation
    class counts fluctuating by tens of images, adding avoidable noise to every
    model-selection decision made against it.

    ``np.random.RandomState`` is used deliberately rather than the newer
    ``default_rng``: NumPy's compatibility policy (NEP 19) guarantees the legacy
    generator's stream is stable across NumPy versions, so this split stays
    identical in a future environment. That guarantee is not extended to
    ``default_rng``.

    Args:
        targets: Class label for each image in the official training set.
        val_size: Total number of validation images. Must divide evenly across
            classes.
        seed: Seed controlling the within-class shuffle.
        num_classes: Number of distinct classes.

    Returns:
        A :class:`SplitIndices` with disjoint, sorted train and validation
        index arrays.
    """
    targets_array = np.asarray(targets)
    if val_size % num_classes != 0:
        raise ValueError(
            f"val_size ({val_size}) must be divisible by num_classes "
            f"({num_classes}) for an exactly balanced split."
        )
    per_class_val = val_size // num_classes

    rng = np.random.RandomState(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_id in range(num_classes):
        class_indices = np.where(targets_array == class_id)[0]
        if class_indices.size < per_class_val:
            raise ValueError(
                f"Class {class_id} has only {class_indices.size} images, fewer "
                f"than the {per_class_val} required for validation."
            )
        # Shuffle a copy so the source ordering is untouched, then take a
        # fixed-size slice. Iterating classes in ascending order keeps the RNG
        # consumption order deterministic.
        shuffled = rng.permutation(class_indices)
        val_parts.append(shuffled[:per_class_val])
        train_parts.append(shuffled[per_class_val:])

    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))

    assert np.intersect1d(train_idx, val_idx).size == 0, "split produced overlap"
    return SplitIndices(train=train_idx, val=val_idx)


def build_train_transform(
    config: Config, image_size: int | None = None
) -> transforms.Compose:
    """Augmentation + preprocessing for the training split only.

    Args:
        config: Loaded project configuration.
        image_size: Optional override for the output resolution. Defaults to
            ``config.image.size`` (the approved final training resolution).
            Pass ``config.image.smoke_test_size`` for local CPU smoke tests -
            results produced at that size must never be reported.

    Order is deliberate. The geometric augmentations run at the native 32x32
    resolution *before* the resize, for two reasons: a 4-pixel crop jitter is a
    meaningful translation at 32x32 (12.5% of the frame) whereas the equivalent
    after upsampling would have to be rescaled to mean the same thing, and
    operating on the small image is cheaper. The resize then happens once, at
    the end, so training and evaluation share an identical final geometry step.
    """
    size = config.image.size if image_size is None else image_size
    interpolation = _INTERPOLATION_MODES[config.image.interpolation]
    return transforms.Compose(
        [
            transforms.RandomCrop(
                config.image.native_size,
                padding=config.augmentation.random_crop_padding,
                padding_mode=config.augmentation.random_crop_padding_mode,
            ),
            transforms.RandomHorizontalFlip(p=config.augmentation.horizontal_flip_prob),
            transforms.Resize(
                (size, size),
                interpolation=interpolation,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.image.normalization_mean,
                std=config.image.normalization_std,
            ),
        ]
    )


def build_eval_transform(
    config: Config, image_size: int | None = None
) -> transforms.Compose:
    """Deterministic preprocessing for validation and test.

    Args:
        config: Loaded project configuration.
        image_size: Optional override for the output resolution. Must match the
            size used for training, or evaluation inputs would differ in scale
            from what the model was fitted on.

    Contains no random operation of any kind, so repeated passes over the
    validation or test set produce byte-identical tensors. The resize and
    normalisation exactly mirror the tail of the training transform, so the only
    difference between train and eval inputs is the augmentation itself.
    """
    size = config.image.size if image_size is None else image_size
    interpolation = _INTERPOLATION_MODES[config.image.interpolation]
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size),
                interpolation=interpolation,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.image.normalization_mean,
                std=config.image.normalization_std,
            ),
        ]
    )


@dataclass
class CIFAR10Splits:
    """The three datasets plus the indices and labels that produced them."""

    train: Dataset
    val: Dataset
    test: Dataset
    indices: SplitIndices
    train_targets: np.ndarray
    val_targets: np.ndarray
    test_targets: np.ndarray


def build_datasets(config: Config, download: bool | None = None) -> CIFAR10Splits:
    """Construct the train / validation / test datasets.

    Two distinct ``CIFAR10`` objects are created over the official training
    files. They read the same images but carry different transforms; each is
    then restricted to a disjoint index set. This is what guarantees that
    augmentation reaches training images only.
    """
    should_download = config.data.download if download is None else download
    root = str(config.data.root)

    train_transform = build_train_transform(config)
    eval_transform = build_eval_transform(config)

    # Same underlying files, different transforms. Only `train_base` augments.
    train_base = CIFAR10(root, train=True, download=should_download, transform=train_transform)
    val_base = CIFAR10(root, train=True, download=False, transform=eval_transform)
    test_dataset = CIFAR10(root, train=False, download=should_download, transform=eval_transform)

    if len(train_base) != config.data.official_train_size:
        raise RuntimeError(
            f"Expected {config.data.official_train_size} official training images, "
            f"found {len(train_base)}."
        )
    if len(test_dataset) != config.data.official_test_size:
        raise RuntimeError(
            f"Expected {config.data.official_test_size} official test images, "
            f"found {len(test_dataset)}."
        )

    indices = build_stratified_split(
        targets=train_base.targets,
        val_size=config.data.val_size,
        seed=config.reproducibility.split_seed,
        num_classes=config.data.num_classes,
    )

    all_train_targets = np.asarray(train_base.targets)
    return CIFAR10Splits(
        train=Subset(train_base, indices.train.tolist()),
        val=Subset(val_base, indices.val.tolist()),
        test=test_dataset,
        indices=indices,
        train_targets=all_train_targets[indices.train],
        val_targets=all_train_targets[indices.val],
        test_targets=np.asarray(test_dataset.targets),
    )


def _resolve_pin_memory(setting: str | bool) -> bool:
    """Pinned memory only helps when copying to a CUDA device."""
    if isinstance(setting, bool):
        return setting
    if setting == "auto":
        return torch.cuda.is_available()
    raise ValueError(f"Invalid pin_memory setting: {setting!r}")


def build_dataloaders(
    config: Config, splits: CIFAR10Splits
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create the train / validation / test DataLoaders.

    Only the training loader shuffles, and it does so through an explicitly
    seeded generator so epoch ordering is reproducible. Validation and test
    iterate in fixed dataset order, which additionally means a prediction array
    from either loader lines up positionally with its stored label array - a
    property the ensemble stage will rely on.
    """
    pin_memory = _resolve_pin_memory(config.dataloader.pin_memory)
    num_workers = config.dataloader.num_workers

    # persistent_workers and prefetch_factor are only legal when workers > 0.
    worker_kwargs: dict = {}
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = config.dataloader.persistent_workers
        if config.dataloader.prefetch_factor is not None:
            worker_kwargs["prefetch_factor"] = config.dataloader.prefetch_factor

    train_loader = DataLoader(
        splits.train,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=config.dataloader.drop_last,
        worker_init_fn=seed_worker,
        generator=build_generator(config.reproducibility.seed),
        **worker_kwargs,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=config.dataloader.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=config.dataloader.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **worker_kwargs,
    )
    return train_loader, val_loader, test_loader


def class_distribution(targets: np.ndarray, num_classes: int = 10) -> dict[int, int]:
    """Count images per class."""
    counts = np.bincount(np.asarray(targets), minlength=num_classes)
    return {int(i): int(counts[i]) for i in range(num_classes)}
