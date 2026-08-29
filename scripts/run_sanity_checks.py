"""Stage 1 sanity checks over the CIFAR-10 pipeline.

Runs every verification the Stage 1 specification asks for, prints a readable
report, writes machine-readable results to ``results/sanity_check_report.json``
and saves two figures drawn from real dataset images.

This script trains nothing and reports no accuracy. It only establishes that the
data reaching a model would be correct.

Usage:
    python scripts/run_sanity_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display available; render straight to file.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import audit_splits  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dataset import (  # noqa: E402
    build_dataloaders,
    build_datasets,
    build_eval_transform,
    build_train_transform,
    class_distribution,
)
from src.seed import set_global_seeds  # noqa: E402
from src.utils import (  # noqa: E402
    collect_environment,
    denormalize,
    format_table,
    select_device,
    tensor_health,
    write_json,
)

SEPARATOR = "=" * 74


def section(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


def check(report: dict, name: str, passed: bool, detail: str = "") -> bool:
    """Print and record a single pass/fail check."""
    report[name] = bool(passed)
    marker = "PASS" if passed else "FAIL"
    print(f"  [{marker}] {name}{f' - {detail}' if detail else ''}")
    return bool(passed)


def visualize_training_augmentation(config, raw_train, indices, out_path: Path) -> Path:
    """Show real training images and several independent augmentations of each.

    Each row is one genuine CIFAR-10 training image: the leftmost panel is the
    unmodified 32x32 source, and the remaining panels are separate draws from
    the training transform. The panels differing across a row is the visual
    demonstration that augmentation is live and stochastic.
    """
    train_transform = build_train_transform(config)
    num_images, num_augs = 4, 5
    chosen = indices.train[:: max(1, len(indices.train) // num_images)][:num_images]

    fig, axes = plt.subplots(
        num_images, num_augs + 1, figsize=(2.0 * (num_augs + 1), 2.0 * num_images)
    )
    for row, dataset_index in enumerate(chosen):
        pil_image = raw_train[int(dataset_index)][0]
        label = raw_train.targets[int(dataset_index)]
        class_name = config.data.class_names[label]

        axes[row, 0].imshow(np.asarray(pil_image))
        axes[row, 0].set_title(f"source 32px\n{class_name}", fontsize=8)
        axes[row, 0].axis("off")

        for col in range(num_augs):
            tensor = train_transform(pil_image)
            shown = denormalize(
                tensor, config.image.normalization_mean, config.image.normalization_std
            )
            axes[row, col + 1].imshow(shown.permute(1, 2, 0).numpy())
            axes[row, col + 1].set_title(f"aug {col + 1}", fontsize=8)
            axes[row, col + 1].axis("off")

    fig.suptitle(
        f"Training augmentation - real CIFAR-10 images "
        f"(crop pad {config.augmentation.random_crop_padding} + h-flip, "
        f"resized to {config.image.size}px)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_eval_preprocessing(config, raw_train, raw_test, indices, out_path: Path) -> Path:
    """Show validation and test preprocessing, repeated, to prove it is fixed.

    Each row applies the evaluation transform to the same image three times. The
    panels being identical is the point: no random operation is in that path.
    """
    eval_transform = build_eval_transform(config)
    num_repeats = 3
    samples = [
        ("validation", raw_train, int(indices.val[0])),
        ("validation", raw_train, int(indices.val[1])),
        ("test", raw_test, 0),
        ("test", raw_test, 1),
    ]

    fig, axes = plt.subplots(
        len(samples), num_repeats + 1, figsize=(2.0 * (num_repeats + 1), 2.0 * len(samples))
    )
    for row, (split_name, dataset, index) in enumerate(samples):
        pil_image = dataset[index][0]
        class_name = config.data.class_names[dataset.targets[index]]

        axes[row, 0].imshow(np.asarray(pil_image))
        axes[row, 0].set_title(f"{split_name} source\n{class_name}", fontsize=8)
        axes[row, 0].axis("off")

        tensors = [eval_transform(pil_image) for _ in range(num_repeats)]
        identical = all(torch.equal(tensors[0], t) for t in tensors[1:])
        for col, tensor in enumerate(tensors):
            shown = denormalize(
                tensor, config.image.normalization_mean, config.image.normalization_std
            )
            axes[row, col + 1].imshow(shown.permute(1, 2, 0).numpy())
            axes[row, col + 1].set_title(
                f"pass {col + 1}{' (identical)' if identical else ' (DIFFERS)'}",
                fontsize=8,
            )
            axes[row, col + 1].axis("off")

    fig.suptitle(
        f"Validation / test preprocessing - deterministic, no augmentation "
        f"(resize to {config.image.size}px only)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    config = load_config()
    set_global_seeds(config.reproducibility.seed, config.reproducibility.deterministic)

    report: dict = {"checks": {}, "values": {}}
    checks = report["checks"]

    # ------------------------------------------------------------------ env
    section("1. ENVIRONMENT")
    environment = collect_environment()
    report["environment"] = environment
    for key, value in environment.items():
        print(f"  {key:<18} {value}")
    device = select_device()
    print(f"  {'selected device':<18} {device}")

    # ------------------------------------------------------------- datasets
    section("2. DATASET CONSTRUCTION")
    splits = build_datasets(config)
    print(f"  train      {len(splits.train):>6} images")
    print(f"  validation {len(splits.val):>6} images")
    print(f"  test       {len(splits.test):>6} images")
    report["values"]["split_checksum"] = splits.indices.checksum()
    print(f"  split checksum: {splits.indices.checksum()}")

    section("3. DATASET SIZES AND CLASSES")
    check(checks, "train_size_is_45000", len(splits.train) == 45000, f"got {len(splits.train)}")
    check(checks, "val_size_is_5000", len(splits.val) == 5000, f"got {len(splits.val)}")
    check(checks, "test_size_is_10000", len(splits.test) == 10000, f"got {len(splits.test)}")

    raw_test_classes = splits.test.classes
    check(
        checks,
        "num_classes_is_10",
        len(raw_test_classes) == config.data.num_classes == 10,
        f"got {len(raw_test_classes)}",
    )
    check(
        checks,
        "class_names_match_torchvision",
        list(raw_test_classes) == list(config.data.class_names),
        "config order matches dataset order",
    )

    # --------------------------------------------------------------- labels
    section("4. LABEL VALIDITY")
    for split_name, targets in (
        ("train", splits.train_targets),
        ("val", splits.val_targets),
        ("test", splits.test_targets),
    ):
        low, high = int(np.min(targets)), int(np.max(targets))
        valid = low >= 0 and high < config.data.num_classes
        integral = np.issubdtype(np.asarray(targets).dtype, np.integer)
        check(checks, f"{split_name}_labels_in_range", valid, f"range [{low}, {high}]")
        check(checks, f"{split_name}_labels_are_integers", integral)

    # ------------------------------------------------------- class balance
    section("5. CLASS DISTRIBUTIONS")
    distributions = {
        "train": class_distribution(splits.train_targets, config.data.num_classes),
        "val": class_distribution(splits.val_targets, config.data.num_classes),
        "test": class_distribution(splits.test_targets, config.data.num_classes),
    }
    report["values"]["class_distributions"] = distributions
    rows = [
        (
            f"{class_id}",
            config.data.class_names[class_id],
            distributions["train"][class_id],
            distributions["val"][class_id],
            distributions["test"][class_id],
        )
        for class_id in range(config.data.num_classes)
    ]
    print(format_table(rows, ("id", "class", "train", "val", "test")))
    check(
        checks,
        "train_split_is_class_balanced",
        len(set(distributions["train"].values())) == 1,
        f"{sorted(set(distributions['train'].values()))} per class",
    )
    check(
        checks,
        "val_split_is_class_balanced",
        len(set(distributions["val"].values())) == 1,
        f"{sorted(set(distributions['val'].values()))} per class",
    )

    # ----------------------------------------------------------- leakage
    section("6. DATA LEAKAGE AUDIT")
    audit = audit_splits(config, splits)
    print(audit.summary())
    report["leakage_audit"] = {"checks": audit.checks, "details": audit.details}
    checks.update({f"audit__{k}": v for k, v in audit.checks.items()})
    print(f"\n  internal duplicate images: {audit.details['internal_duplicates']}")

    # ------------------------------------------- augmentation determinism
    section("7. AUGMENTATION IS TRAINING-ONLY (numeric proof)")
    # Same index, repeated draws. Training must vary; evaluation must not.
    train_draws = [splits.train[0][0] for _ in range(4)]
    val_draws = [splits.val[0][0] for _ in range(4)]
    test_draws = [splits.test[0][0] for _ in range(4)]

    train_varies = any(not torch.equal(train_draws[0], t) for t in train_draws[1:])
    val_fixed = all(torch.equal(val_draws[0], t) for t in val_draws[1:])
    test_fixed = all(torch.equal(test_draws[0], t) for t in test_draws[1:])

    check(checks, "train_transform_varies_across_draws", train_varies,
          "repeated reads of the same index differ")
    check(checks, "val_transform_is_deterministic", val_fixed,
          "repeated reads of the same index are identical")
    check(checks, "test_transform_is_deterministic", test_fixed,
          "repeated reads of the same index are identical")

    # ------------------------------------------------------- tensor shapes
    section("8. TENSOR SHAPES AND HEALTH")
    expected_shape = (3, config.image.size, config.image.size)
    for split_name, dataset in (
        ("train", splits.train), ("val", splits.val), ("test", splits.test)
    ):
        image, label = dataset[0]
        check(
            checks,
            f"{split_name}_image_shape_is_{expected_shape}",
            tuple(image.shape) == expected_shape,
            f"got {tuple(image.shape)}",
        )
        check(checks, f"{split_name}_image_dtype_is_float32",
              image.dtype == torch.float32, str(image.dtype))
        check(checks, f"{split_name}_label_is_int", isinstance(label, int), str(type(label)))

    # ---------------------------------------------------------- dataloaders
    section("9. DATALOADERS")
    train_loader, val_loader, test_loader = build_dataloaders(config, splits)
    print(f"  train batches      {len(train_loader):>5} "
          f"(batch_size {config.dataloader.batch_size}, drop_last "
          f"{config.dataloader.drop_last})")
    print(f"  validation batches {len(val_loader):>5} "
          f"(batch_size {config.dataloader.eval_batch_size})")
    print(f"  test batches       {len(test_loader):>5} "
          f"(batch_size {config.dataloader.eval_batch_size})")
    print(f"  num_workers {config.dataloader.num_workers}, "
          f"pin_memory {train_loader.pin_memory}, "
          f"persistent_workers {config.dataloader.persistent_workers}")

    batch_health = {}
    for split_name, loader in (
        ("train", train_loader), ("val", val_loader), ("test", test_loader)
    ):
        images, labels = next(iter(loader))
        health = tensor_health(images)
        batch_health[split_name] = health
        expected_batch = (
            config.dataloader.batch_size if split_name == "train"
            else config.dataloader.eval_batch_size
        )
        check(
            checks,
            f"{split_name}_loader_yields_batch",
            tuple(images.shape) == (expected_batch, *expected_shape),
            f"shape {tuple(images.shape)}",
        )
        check(checks, f"{split_name}_batch_has_no_nan", not health["has_nan"])
        check(checks, f"{split_name}_batch_has_no_inf", not health["has_inf"])
        check(checks, f"{split_name}_batch_all_finite", health["all_finite"])
        check(
            checks,
            f"{split_name}_batch_labels_valid",
            bool(labels.min() >= 0 and labels.max() < config.data.num_classes),
            f"range [{int(labels.min())}, {int(labels.max())}]",
        )
        print(f"    {split_name}: mean {health['mean']:+.4f}  std {health['std']:.4f}  "
              f"min {health['min']:+.3f}  max {health['max']:+.3f}")

    report["values"]["batch_health"] = batch_health

    # ------------------------------------------------- shuffle behaviour
    section("10. LOADER ORDERING")
    val_labels_a = torch.cat([lbl for _, lbl in val_loader])
    val_labels_b = torch.cat([lbl for _, lbl in val_loader])
    check(checks, "val_loader_order_is_stable", torch.equal(val_labels_a, val_labels_b),
          "two passes give the same label sequence")
    check(
        checks,
        "val_loader_order_matches_stored_targets",
        torch.equal(val_labels_a, torch.as_tensor(splits.val_targets)),
        "predictions will align positionally with stored labels",
    )

    # ------------------------------------------------------ device transfer
    section("11. DEVICE TRANSFER")
    images, labels = next(iter(train_loader))
    moved_images, moved_labels = images.to(device), labels.to(device)
    check(checks, "batch_transfers_to_device", moved_images.device.type == device.type,
          f"moved to {moved_images.device}")
    check(checks, "transferred_batch_still_finite",
          bool(torch.isfinite(moved_images).all()))

    # ----------------------------- training-split-only channel statistics
    section("12. CHANNEL STATISTICS (training split only)")
    # Computed for documentation. NOT used by the pipeline, which normalises
    # with fixed ImageNet constants. Derived from the training split alone so
    # that even this descriptive number carries no validation or test influence.
    from torchvision.datasets import CIFAR10  # local: raw arrays, no transform

    raw_train = CIFAR10(str(config.data.root), train=True, download=False)
    raw_test = CIFAR10(str(config.data.root), train=False, download=False)
    train_pixels = raw_train.data[splits.indices.train].astype(np.float64) / 255.0
    train_mean = train_pixels.mean(axis=(0, 1, 2))
    train_std = train_pixels.std(axis=(0, 1, 2))
    report["values"]["train_split_channel_mean"] = [round(float(v), 4) for v in train_mean]
    report["values"]["train_split_channel_std"] = [round(float(v), 4) for v in train_std]
    print(f"  train-split mean  {np.round(train_mean, 4).tolist()}")
    print(f"  train-split std   {np.round(train_std, 4).tolist()}")
    print(f"  ImageNet mean     {list(config.image.normalization_mean)}  <- used")
    print(f"  ImageNet std      {list(config.image.normalization_std)}  <- used")

    # ------------------------------------------------------ visualisations
    section("13. VISUALISATIONS (real dataset images)")
    aug_path = visualize_training_augmentation(
        config, raw_train, splits.indices, config.paths.results_dir / "train_augmentation_samples.png"
    )
    eval_path = visualize_eval_preprocessing(
        config, raw_train, raw_test, splits.indices,
        config.paths.results_dir / "eval_preprocessing_samples.png",
    )
    print(f"  wrote {aug_path.name}")
    print(f"  wrote {eval_path.name}")
    report["values"]["figures"] = [aug_path.name, eval_path.name]

    # -------------------------------------------------------------- summary
    section("SUMMARY")
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    failed = [name for name, ok in checks.items() if not ok]
    report["summary"] = {"total": total, "passed": passed, "failed": len(failed),
                         "failed_names": failed}
    print(f"  {passed}/{total} checks passed")
    if failed:
        print("  FAILED:")
        for name in failed:
            print(f"    - {name}")

    out = write_json(report, config.paths.results_dir / "sanity_check_report.json")
    print(f"\n  report written to {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
