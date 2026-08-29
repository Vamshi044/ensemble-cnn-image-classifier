"""Automated tests for the Stage 1 data pipeline.

Run with:
    python -m pytest tests/ -v

The tests split into two groups. Those that only exercise the split algorithm
run against synthetic labels and need no downloaded data, so they work in any
environment. Those that need the real CIFAR-10 files are marked and skipped
automatically when the dataset is absent, so the suite never fails merely
because the data has not been fetched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset import (
    build_dataloaders,
    build_datasets,
    build_eval_transform,
    build_stratified_split,
    build_train_transform,
    class_distribution,
)
from src.seed import set_global_seeds


@pytest.fixture(scope="module")
def config():
    return load_config()


def cifar10_is_available(config) -> bool:
    """True when the extracted CIFAR-10 python batches are on disk."""
    return (Path(config.data.root) / "cifar-10-batches-py" / "test_batch").is_file()


@pytest.fixture(scope="module")
def splits(config):
    if not cifar10_is_available(config):
        pytest.skip("CIFAR-10 not downloaded; run scripts/run_sanity_checks.py first")
    set_global_seeds(config.reproducibility.seed)
    return build_datasets(config, download=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_declares_ten_classes(config):
    assert config.data.num_classes == 10
    assert len(config.data.class_names) == 10


def test_config_split_sizes_sum_to_official_train_set(config):
    assert config.data.train_size == 45000
    assert config.data.val_size == 5000
    assert config.data.train_size + config.data.val_size == config.data.official_train_size


def test_config_does_not_touch_the_test_set(config):
    """The test set size must be the official 10,000 and never be split."""
    assert config.data.official_test_size == 10000


def test_config_paths_are_absolute_and_not_machine_specific(config):
    """Paths resolve under the project root rather than being hard-coded."""
    assert config.data.root.is_absolute()
    assert str(PROJECT_ROOT) in str(config.data.root)


# ---------------------------------------------------------------------------
# Split algorithm - synthetic labels, no download required
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_targets():
    """50,000 labels, 5,000 per class, mirroring CIFAR-10's balance."""
    return np.repeat(np.arange(10), 5000)


def test_split_sizes(synthetic_targets):
    split = build_stratified_split(synthetic_targets, val_size=5000, seed=1337)
    assert split.train.size == 45000
    assert split.val.size == 5000


def test_split_has_no_overlap(synthetic_targets):
    split = build_stratified_split(synthetic_targets, val_size=5000, seed=1337)
    assert np.intersect1d(split.train, split.val).size == 0


def test_split_partitions_the_input_exactly(synthetic_targets):
    split = build_stratified_split(synthetic_targets, val_size=5000, seed=1337)
    union = np.union1d(split.train, split.val)
    assert union.size == synthetic_targets.size
    assert union.min() == 0 and union.max() == synthetic_targets.size - 1


def test_split_is_class_stratified(synthetic_targets):
    split = build_stratified_split(synthetic_targets, val_size=5000, seed=1337)
    val_counts = class_distribution(synthetic_targets[split.val])
    train_counts = class_distribution(synthetic_targets[split.train])
    assert set(val_counts.values()) == {500}
    assert set(train_counts.values()) == {4500}


def test_split_is_reproducible(synthetic_targets):
    """Same seed must give a byte-identical split."""
    a = build_stratified_split(synthetic_targets, val_size=5000, seed=1337)
    b = build_stratified_split(synthetic_targets, val_size=5000, seed=1337)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.train, b.train)
    assert a.checksum() == b.checksum()


def test_different_seeds_give_different_splits(synthetic_targets):
    a = build_stratified_split(synthetic_targets, val_size=5000, seed=1337)
    b = build_stratified_split(synthetic_targets, val_size=5000, seed=7)
    assert not np.array_equal(a.val, b.val)


def test_split_rejects_val_size_not_divisible_by_classes(synthetic_targets):
    with pytest.raises(ValueError, match="divisible"):
        build_stratified_split(synthetic_targets, val_size=5001, seed=1337)


# ---------------------------------------------------------------------------
# Transforms - no download required
# ---------------------------------------------------------------------------


def test_eval_transform_contains_no_random_operation(config):
    ops = [type(t).__name__ for t in build_eval_transform(config).transforms]
    assert not any("Random" in op for op in ops), ops


def test_train_transform_contains_augmentation(config):
    ops = [type(t).__name__ for t in build_train_transform(config).transforms]
    assert any("Random" in op for op in ops), ops


def test_train_and_eval_share_the_same_final_geometry(config):
    """Only the augmentation should differ; resize/normalise must match."""
    train_ops = [type(t).__name__ for t in build_train_transform(config).transforms]
    eval_ops = [type(t).__name__ for t in build_eval_transform(config).transforms]
    assert train_ops[-3:] == eval_ops[-3:] == ["Resize", "ToTensor", "Normalize"]


# ---------------------------------------------------------------------------
# Real dataset - skipped when CIFAR-10 has not been downloaded
# ---------------------------------------------------------------------------


def test_dataset_sizes(splits):
    assert len(splits.train) == 45000
    assert len(splits.val) == 5000
    assert len(splits.test) == 10000


def test_number_of_classes(splits, config):
    assert len(splits.test.classes) == 10
    assert list(splits.test.classes) == list(config.data.class_names)


def test_tensor_dimensions(splits, config):
    expected = (3, config.image.size, config.image.size)
    for dataset in (splits.train, splits.val, splits.test):
        image, _ = dataset[0]
        assert tuple(image.shape) == expected
        assert image.dtype == torch.float32


def test_label_range(splits):
    for targets in (splits.train_targets, splits.val_targets, splits.test_targets):
        assert int(np.min(targets)) >= 0
        assert int(np.max(targets)) <= 9


def test_no_train_val_index_overlap(splits):
    assert set(splits.indices.train.tolist()).isdisjoint(splits.indices.val.tolist())


def test_real_split_is_class_balanced(splits):
    assert set(class_distribution(splits.val_targets).values()) == {500}
    assert set(class_distribution(splits.train_targets).values()) == {4500}


def test_train_and_val_wrap_separate_dataset_objects(splits):
    """Guards the transform-sharing bug that random_split would introduce."""
    assert splits.train.dataset is not splits.val.dataset


def test_validation_reads_are_deterministic(splits):
    """Reading the same validation index twice must give identical tensors."""
    first, _ = splits.val[0]
    second, _ = splits.val[0]
    assert torch.equal(first, second)


def test_test_reads_are_deterministic(splits):
    first, _ = splits.test[0]
    second, _ = splits.test[0]
    assert torch.equal(first, second)


def test_training_reads_are_augmented(splits):
    """Reading the same training index repeatedly must vary.

    Ten draws are taken because any single pair could coincide by chance when
    the random crop happens to pick the centre and the flip does not fire.
    """
    draws = [splits.train[0][0] for _ in range(10)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_batches_contain_no_non_finite_values(splits, config):
    _, val_loader, _ = build_dataloaders(config, splits)
    images, labels = next(iter(val_loader))
    assert torch.isfinite(images).all()
    assert int(labels.min()) >= 0 and int(labels.max()) <= 9


def test_dataloaders_produce_expected_batch_shapes(splits, config):
    train_loader, val_loader, _ = build_dataloaders(config, splits)
    images, _ = next(iter(train_loader))
    assert tuple(images.shape) == (
        config.dataloader.batch_size, 3, config.image.size, config.image.size
    )
    images, _ = next(iter(val_loader))
    assert tuple(images.shape) == (
        config.dataloader.eval_batch_size, 3, config.image.size, config.image.size
    )


def test_eval_loader_order_is_stable(splits, config):
    """Fixed ordering lets stored labels align with later predictions."""
    _, val_loader, _ = build_dataloaders(config, splits)
    first = torch.cat([lbl for _, lbl in val_loader])
    second = torch.cat([lbl for _, lbl in val_loader])
    assert torch.equal(first, second)
    assert torch.equal(first, torch.as_tensor(splits.val_targets))


def test_no_content_overlap_between_splits(splits, config):
    """Compare raw pixel hashes, not just indices."""
    from src.audit import hash_images
    from torchvision.datasets import CIFAR10

    raw_train = CIFAR10(str(config.data.root), train=True, download=False)
    raw_test = CIFAR10(str(config.data.root), train=False, download=False)

    train_h = set(hash_images(raw_train.data[splits.indices.train]))
    val_h = set(hash_images(raw_train.data[splits.indices.val]))
    test_h = set(hash_images(raw_test.data))

    assert train_h.isdisjoint(val_h)
    assert train_h.isdisjoint(test_h)
    assert val_h.isdisjoint(test_h)
