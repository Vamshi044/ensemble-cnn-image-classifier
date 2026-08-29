"""Data-leakage audit.

The project's core methodological claim is that the official 10,000-image test
set never influences anything until final evaluation. This module turns that
claim into checks that either pass or fail, rather than leaving it as an
assumption about how the pipeline was written.

Two levels of checking are performed:

*Index level* - the train and validation index sets are disjoint, they partition
the official training set exactly, and the split is reproducible.

*Content level* - images are compared by cryptographic hash of their raw pixel
bytes. This is the stronger check: it would catch leakage even if it arrived
through some route other than the index arrays, and it also surfaces genuine
duplicate images that exist within CIFAR-10 itself.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torchvision.datasets import CIFAR10

from src.config import Config
from src.dataset import (
    CIFAR10Splits,
    build_eval_transform,
    build_stratified_split,
    build_train_transform,
)


@dataclass
class AuditResult:
    """Outcome of the leakage audit."""

    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def record(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks[name] = bool(passed)
        if detail is not None:
            self.details[name] = detail

    @property
    def all_passed(self) -> bool:
        return all(self.checks.values())

    def summary(self) -> str:
        lines = []
        for name, passed in self.checks.items():
            marker = "PASS" if passed else "FAIL"
            lines.append(f"  [{marker}] {name}")
        return "\n".join(lines)


def hash_images(images: np.ndarray) -> list[str]:
    """Hash each image's raw pixel bytes.

    Args:
        images: Array of shape (N, H, W, C), dtype uint8.

    Returns:
        One hex digest per image, order preserved.
    """
    contiguous = np.ascontiguousarray(images)
    return [hashlib.sha256(img.tobytes()).hexdigest() for img in contiguous]


def audit_splits(config: Config, splits: CIFAR10Splits) -> AuditResult:
    """Run the full leakage audit and return a pass/fail record.

    Args:
        config: Loaded project configuration.
        splits: The datasets under audit.

    Returns:
        An :class:`AuditResult` with one entry per check.
    """
    result = AuditResult()
    idx = splits.indices

    # --- Index-level checks -------------------------------------------------
    train_set, val_set = set(idx.train.tolist()), set(idx.val.tolist())

    overlap = train_set & val_set
    result.record(
        "train_val_indices_disjoint", len(overlap) == 0, {"overlap_count": len(overlap)}
    )
    result.record(
        "split_partitions_official_train",
        len(train_set | val_set) == config.data.official_train_size,
        {"union_size": len(train_set | val_set)},
    )
    result.record("train_size_correct", len(train_set) == config.data.train_size,
                  {"actual": len(train_set), "expected": config.data.train_size})
    result.record("val_size_correct", len(val_set) == config.data.val_size,
                  {"actual": len(val_set), "expected": config.data.val_size})

    # Regenerating the split from the same seed must give the same partition.
    regenerated = build_stratified_split(
        targets=np.concatenate([splits.train_targets, splits.val_targets])[
            np.argsort(np.concatenate([idx.train, idx.val]))
        ],
        val_size=config.data.val_size,
        seed=config.reproducibility.split_seed,
        num_classes=config.data.num_classes,
    )
    result.record(
        "split_reproducible",
        np.array_equal(regenerated.val, idx.val),
        {"checksum": idx.checksum()},
    )

    # --- Content-level checks ----------------------------------------------
    # Read the raw uint8 arrays directly, bypassing transforms.
    root = str(config.data.root)
    raw_train = CIFAR10(root, train=True, download=False)
    raw_test = CIFAR10(root, train=False, download=False)

    train_hashes = hash_images(raw_train.data[idx.train])
    val_hashes = hash_images(raw_train.data[idx.val])
    test_hashes = hash_images(raw_test.data)

    train_h, val_h, test_h = set(train_hashes), set(val_hashes), set(test_hashes)

    tv_shared = train_h & val_h
    result.record(
        "no_train_val_image_content_overlap",
        len(tv_shared) == 0,
        {"shared_image_count": len(tv_shared)},
    )
    tt_shared = train_h & test_h
    result.record(
        "no_train_test_image_content_overlap",
        len(tt_shared) == 0,
        {"shared_image_count": len(tt_shared)},
    )
    vt_shared = val_h & test_h
    result.record(
        "no_val_test_image_content_overlap",
        len(vt_shared) == 0,
        {"shared_image_count": len(vt_shared)},
    )

    # Duplicates *within* a split are not leakage, but they are worth knowing
    # about because they slightly inflate the effective weight of some images.
    result.details["internal_duplicates"] = {
        "train": len(train_hashes) - len(train_h),
        "val": len(val_hashes) - len(val_h),
        "test": len(test_hashes) - len(test_h),
    }

    # --- Transform-level checks --------------------------------------------
    # The validation and test transforms must contain no random operation.
    random_op_names = ("Random", "Jitter", "Erasing", "AutoAugment", "TrivialAugment")
    eval_ops = [type(t).__name__ for t in build_eval_transform(config).transforms]
    train_ops = [type(t).__name__ for t in build_train_transform(config).transforms]

    eval_has_random = any(
        any(token in op for token in random_op_names) for op in eval_ops
    )
    train_has_random = any(
        any(token in op for token in random_op_names) for op in train_ops
    )
    result.record("eval_transform_is_deterministic", not eval_has_random,
                  {"ops": eval_ops})
    result.record("train_transform_has_augmentation", train_has_random,
                  {"ops": train_ops})

    # The val and test datasets must literally be carrying the eval transform.
    val_transform_ops = [
        type(t).__name__ for t in splits.val.dataset.transform.transforms
    ]
    test_transform_ops = [type(t).__name__ for t in splits.test.transform.transforms]
    result.record(
        "val_dataset_uses_eval_transform",
        val_transform_ops == eval_ops,
        {"ops": val_transform_ops},
    )
    result.record(
        "test_dataset_uses_eval_transform",
        test_transform_ops == eval_ops,
        {"ops": test_transform_ops},
    )

    # Guard against the classic Subset-sharing bug: the train and validation
    # subsets must wrap DIFFERENT underlying dataset objects, otherwise they
    # would share a single transform.
    result.record(
        "train_and_val_use_separate_dataset_objects",
        splits.train.dataset is not splits.val.dataset,
    )

    # --- Normalisation provenance ------------------------------------------
    # Statistics must not be derived from the test set. Using published ImageNet
    # constants makes this checkable rather than merely intended.
    result.record(
        "normalization_not_derived_from_data",
        config.image.normalization_source == "imagenet",
        {"source": config.image.normalization_source},
    )

    return result
