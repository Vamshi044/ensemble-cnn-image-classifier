"""Stage 4 - final ensemble evaluation on the official CIFAR-10 test set.

Loads the best checkpoint for each trained architecture, runs every model over
the same official 10,000-image test set under the deterministic evaluation
transform, converts logits to softmax probabilities, and fuses them by
equal-weight averaging:

    P_ensemble = (P_resnet18 + P_googlenet + P_vgg11) / 3
    prediction = argmax(P_ensemble)

This trains nothing, tunes nothing, and writes no checkpoint. It is the first
and only place the official test set is consumed, and it is consumed once,
after model selection has already been finalised against the validation split.

Run with:
    python scripts/evaluate_ensemble.py --dry-run   # gates + inventory only
    python scripts/evaluate_ensemble.py             # gates, then evaluate
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpointing import checkpoint_path, load_checkpoint
from src.config import load_config
from src.dataset import build_dataloaders, build_datasets
from src.models import build_model, count_parameters
from src.seed import set_global_seeds
from src.utils import collect_environment, select_device, write_json

# The split every Stage 3B run was fitted against. If this changes, the
# validation split that selected these checkpoints is not the split this
# process reconstructed, and "best" means nothing.
EXPECTED_SPLIT_CHECKSUM = "bdb035810af794a7"

EXPECTED_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Fixed order so the JSON, the CSV columns and the fusion all agree.
ARCH_ORDER: tuple[str, ...] = ("resnet18", "googlenet", "vgg11")

FUSION = "equal_weight_softmax_probability_average"


class PreflightError(RuntimeError):
    """A gate failed. Evaluation must not proceed."""


def gate(ok: bool, label: str, detail: Any = "") -> bool:
    """Print one pass/fail line and return the verdict."""
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


@dataclass
class ModelResult:
    """Everything recorded for a single architecture."""

    architecture: str
    checkpoint: str
    checkpoint_bytes: int
    checkpoint_epoch: int
    checkpoint_best_metric: float | None
    checkpoint_best_epoch: int | None
    weights_only: bool
    parameters: int
    test_accuracy: float
    test_correct: int
    probabilities: np.ndarray


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def inventory(checkpoints_dir: Path) -> dict[str, dict[str, Any]]:
    """Report what is on disk for every architecture, without loading it."""
    found: dict[str, dict[str, Any]] = {}
    for arch in ARCH_ORDER:
        entry: dict[str, Any] = {}
        for kind in ("best", "last"):
            path = checkpoint_path(checkpoints_dir, arch, kind)
            entry[kind] = {
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
        found[arch] = entry
    return found


def print_inventory(found: dict[str, dict[str, Any]]) -> None:
    """Print one line per checkpoint slot, present or not."""
    print("\n-- checkpoint inventory --")
    for arch, kinds in found.items():
        for kind, info in kinds.items():
            mark = "OK" if info["exists"] else "--"
            size = f"{info['bytes'] / 1e6:8.1f} MB" if info["exists"] else " missing"
            print(f"[{mark}] {arch:<10} {kind:<5} {size}  {info['path']}")


def preflight_data(config: Any, splits: Any) -> bool:
    """Verify the data this evaluation is about to consume.

    Every check guards a way the headline number could be wrong without
    anything raising: an altered split means the checkpoints were selected
    against different validation images, and a test set that is secretly a view
    of the training images would inflate accuracy silently.
    """
    ok = True
    checksum = splits.indices.checksum()
    ok &= gate(
        checksum == EXPECTED_SPLIT_CHECKSUM,
        "split checksum unchanged",
        f"{checksum} (expected {EXPECTED_SPLIT_CHECKSUM})",
    )
    ok &= gate(
        len(splits.train) == config.data.train_size
        and len(splits.val) == config.data.val_size,
        "train/val split sizes",
        f"{len(splits.train)} / {len(splits.val)}",
    )
    ok &= gate(
        len(splits.test) == config.data.official_test_size,
        "official test size",
        len(splits.test),
    )
    # The test dataset must be the official test files, not a Subset carved out
    # of the training files. torchvision records which file set it opened.
    ok &= gate(
        getattr(splits.test, "train", None) is False,
        "test set reads the official test files, not the training files",
        f"train flag = {getattr(splits.test, 'train', 'absent')}",
    )
    ok &= gate(
        np.intersect1d(splits.indices.train, splits.indices.val).size == 0,
        "train and validation indices are disjoint",
    )
    ok &= gate(
        tuple(splits.test.classes) == EXPECTED_CLASSES,
        "CIFAR-10 class order",
        ", ".join(f"{i}={c}" for i, c in enumerate(splits.test.classes)),
    )
    counts = np.bincount(splits.test_targets, minlength=config.data.num_classes)
    ok &= gate(
        bool((counts == 1000).all()),
        "test labels are the balanced 1000-per-class CIFAR-10 distribution",
        counts.tolist(),
    )
    return ok


def preflight_transform(config: Any, splits: Any) -> bool:
    """Confirm the evaluation transform is deterministic and correctly sized."""
    ok = True
    first, _ = splits.test[0]
    again, _ = splits.test[0]
    ok &= gate(
        torch.equal(first, again),
        "evaluation transform is deterministic (byte-identical on repeat)",
    )
    ok &= gate(
        tuple(first.shape) == (3, config.image.size, config.image.size),
        "evaluation tensor shape",
        tuple(first.shape),
    )
    # A random augmentation in the eval path would be a silent methodology
    # change, so the composition is inspected rather than trusted.
    names = [type(t).__name__ for t in splits.test.transform.transforms]
    ok &= gate(
        not any(n.startswith("Random") for n in names),
        "evaluation transform contains no random operation",
        " -> ".join(names),
    )
    return ok


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------


def load_trained_model(
    architecture: str, path: Path, config: Any, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild an architecture exactly as trained, then load its weights.

    ``pretrained=config.model.pretrained`` is deliberate and load-bearing even
    though every learned parameter is immediately overwritten by the
    checkpoint. Verified by execution: torchvision's GoogLeNet sets
    ``transform_input=True`` only on the pretrained construction path, and
    ``transform_input`` is a plain attribute that appears in no state dict.
    Building with ``pretrained=False`` therefore yields a key-identical state
    dict that loads cleanly under ``strict=True`` while silently changing the
    forward pass. Constructing the way training did removes that failure mode.
    """
    checkpoint = load_checkpoint(path, map_location="cpu")

    stored = checkpoint.get("architecture")
    if stored != architecture:
        raise PreflightError(
            f"{path} stores architecture {stored!r}, but it was loaded as "
            f"{architecture!r}. Refusing to evaluate a mislabelled checkpoint."
        )

    model = build_model(
        architecture,
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
        aux_logits=config.model.googlenet_aux_logits,
    )
    # strict=True: a silently dropped or unexpected tensor would mean the
    # evaluated network is not the trained network.
    model.load_state_dict(checkpoint["model_state"], strict=True)

    if architecture == "googlenet" and model.transform_input is not True:
        raise PreflightError(
            "GoogLeNet was built with transform_input=False, which does not "
            "match the approved training configuration."
        )

    model.to(device).eval()

    meta = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_metric": checkpoint.get("best_metric"),
        "best_epoch": checkpoint.get("best_epoch"),
        # A best checkpoint is written weights-only, so optimiser state is
        # expected to be absent. Recorded rather than asserted, because that
        # policy is a configuration choice.
        "weights_only": "optimizer_state" not in checkpoint,
    }
    return model, meta


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: Any, device: torch.device) -> np.ndarray:
    """Return softmax probabilities for the whole loader, in dataset order.

    The evaluation loader does not shuffle, so row *i* of the result matches
    test label *i* positionally.

    Inference runs in full float32 with no autocast. Mixed precision was a
    training-throughput decision; reintroducing it here would perturb the
    reported accuracy for no benefit, and float32 keeps the number stable
    across devices.
    """
    chunks: list[np.ndarray] = []
    for images, _targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise PreflightError(
                "Model produced non-finite logits on the test set; the "
                "checkpoint or the input pipeline is corrupt."
            )
        chunks.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 ensemble evaluation.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every gate and print the checkpoint inventory, then stop.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Compute device. 'auto' prefers CUDA when available.",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to the config file."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Override paths.checkpoints_dir - point this at the persistent "
             "directory the training run wrote to.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override paths.results_dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_config(args.config)
    set_global_seeds(
        config.reproducibility.seed,
        deterministic=config.reproducibility.deterministic,
    )

    device = select_device() if args.device == "auto" else torch.device(args.device)
    name = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    print(f"device: {device}{name}")

    checkpoints_dir = Path(args.checkpoints_dir or config.paths.checkpoints_dir)
    results_dir = Path(args.results_dir or config.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    found = inventory(checkpoints_dir)
    print_inventory(found)

    missing = [a for a in ARCH_ORDER if not found[a]["best"]["exists"]]
    if missing:
        print(
            "\nNo best checkpoint for: "
            + ", ".join(missing)
            + "\nThe ensemble is defined over all three architectures, so it "
            "cannot be evaluated until each has one. Nothing was evaluated and "
            "nothing was written."
        )
        return 3

    print("\n-- data integrity --")
    splits = build_datasets(config, download=False)
    ok = preflight_data(config, splits)
    ok = preflight_transform(config, splits) and ok
    if not ok:
        raise PreflightError("Data integrity gates failed; refusing to evaluate.")

    _train_loader, _val_loader, test_loader = build_dataloaders(config, splits)
    labels = np.asarray(splits.test_targets)

    if args.dry_run:
        print("\n--dry-run: gates passed. No test images were evaluated.")
        return 0

    print("\n-- per-model evaluation on the official test set --")
    results: list[ModelResult] = []
    for arch in ARCH_ORDER:
        path = Path(found[arch]["best"]["path"])
        model, meta = load_trained_model(arch, path, config, device)
        started = time.perf_counter()
        probabilities = predict(model, test_loader, device)
        elapsed = time.perf_counter() - started

        expected_shape = (len(labels), config.data.num_classes)
        if probabilities.shape != expected_shape:
            raise PreflightError(
                f"{arch} produced {probabilities.shape} probabilities, expected "
                f"{expected_shape}."
            )
        correct = int((probabilities.argmax(axis=1) == labels).sum())
        results.append(
            ModelResult(
                architecture=arch,
                checkpoint=str(path),
                checkpoint_bytes=found[arch]["best"]["bytes"],
                checkpoint_epoch=meta["epoch"],
                checkpoint_best_metric=meta["best_metric"],
                checkpoint_best_epoch=meta["best_epoch"],
                weights_only=meta["weights_only"],
                parameters=count_parameters(model, arch).total,
                test_accuracy=correct / len(labels),
                test_correct=correct,
                probabilities=probabilities,
            )
        )
        print(
            f"  {arch:<10} test_acc {correct / len(labels):.4f}  "
            f"({correct}/{len(labels)})  "
            f"val_acc_at_selection {meta['best_metric']}  {elapsed:.1f}s"
        )
        # The probabilities are kept; the network is not.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    weight = 1.0 / len(results)
    fused = np.zeros_like(results[0].probabilities)
    for result in results:
        fused += weight * result.probabilities

    predictions = fused.argmax(axis=1)
    ensemble_correct = int((predictions == labels).sum())
    ensemble_accuracy = ensemble_correct / len(labels)
    print(
        f"\n  ENSEMBLE   test_acc {ensemble_accuracy:.4f}  "
        f"({ensemble_correct}/{len(labels)})"
    )

    per_class = {}
    for class_id in range(config.data.num_classes):
        support = int((labels == class_id).sum())
        hits = int(((predictions == class_id) & (labels == class_id)).sum())
        per_class[splits.test.classes[class_id]] = {
            "support": support,
            "correct": hits,
            "accuracy": hits / support if support else 0.0,
        }

    # Rows are true classes, columns are ensemble predictions, so the diagonal
    # is per-class correct counts and row sums are the 1,000-image supports.
    confusion = np.zeros((config.data.num_classes, config.data.num_classes),
                         dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)

    print()
    print("  ensemble confusion matrix (rows: true, columns: predicted)")
    print("             " + " ".join(f"{c[:4]:>5}" for c in splits.test.classes))
    for class_id, name in enumerate(splits.test.classes):
        row = " ".join(f"{v:>5}" for v in confusion[class_id])
        print(f"  {name:<11}{row}")

    payload = {
        "stage": "4-ensemble-evaluation",
        "dataset": config.data.dataset_name,
        "test_size": int(len(labels)),
        "split_checksum": splits.indices.checksum(),
        "image_size": config.image.size,
        "device": str(device),
        "amp_used_for_inference": False,
        "architectures": list(ARCH_ORDER),
        "checkpoints": {r.architecture: r.checkpoint for r in results},
        "fusion": FUSION,
        "weights": {r.architecture: weight for r in results},
        "ensemble_test_accuracy": ensemble_accuracy,
        "ensemble_test_correct": ensemble_correct,
        "individual_test_accuracy": {
            r.architecture: r.test_accuracy for r in results
        },
        "per_model": [
            {
                "architecture": r.architecture,
                "checkpoint": r.checkpoint,
                "checkpoint_bytes": r.checkpoint_bytes,
                "checkpoint_epoch_index": r.checkpoint_epoch,
                "validation_accuracy_at_selection": r.checkpoint_best_metric,
                "best_epoch_index": r.checkpoint_best_epoch,
                "weights_only": r.weights_only,
                "parameters": r.parameters,
                "test_accuracy": r.test_accuracy,
                "test_correct": r.test_correct,
            }
            for r in results
        ],
        "ensemble_per_class": per_class,
        "ensemble_confusion_matrix": {
            "classes": list(splits.test.classes),
            "rows": "true label",
            "columns": "ensemble prediction",
            "matrix": confusion.tolist(),
        },
        "environment": collect_environment(),
        "notes": [
            "The official test set was consumed once, here, after all model "
            "selection had been completed against the validation split.",
            "No training, tuning, weight search, or checkpoint selection used "
            "these images.",
            "Fusion weights are fixed at 1/3 each and were not searched.",
        ],
    }
    json_path = write_json(payload, results_dir / "ensemble_evaluation.json")
    print(f"\nwrote {json_path}")

    csv_path = results_dir / "ensemble_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "true_label",
                "true_class",
                "ensemble_pred",
                "ensemble_confidence",
                *(f"{r.architecture}_pred" for r in results),
            ]
        )
        for i in range(len(labels)):
            writer.writerow(
                [
                    i,
                    int(labels[i]),
                    splits.test.classes[labels[i]],
                    int(predictions[i]),
                    f"{fused[i, predictions[i]]:.6f}",
                    *(int(r.probabilities[i].argmax()) for r in results),
                ]
            )
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(f"\nSTOPPED: {error}")
        raise SystemExit(2) from error
