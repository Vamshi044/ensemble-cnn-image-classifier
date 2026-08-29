"""Train one approved architecture on CIFAR-10 train/validation.

    python scripts/train.py --architecture resnet18
    python scripts/train.py --architecture vgg11 --epochs 15
    python scripts/train.py --architecture googlenet --resume auto

This is the Stage 3B entry point. Stage 3A built and tested it; running it is a
separate, explicitly authorised step.

**The test set is not reachable from this script.** ``build_dataloaders``
returns three loaders and the third is discarded on the line it is unpacked.
``src.training.fit`` accepts a training loader and a validation loader and has
no third parameter, so there is no argument position a test loader could occupy.
Model selection, early stopping and the learning-rate schedule all read the
validation split only.

Nothing here prints or stores an accuracy for anything but train and validation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpointing import checkpoint_path
from src.config import load_config
from src.dataset import build_dataloaders, build_datasets
from src.models import ARCHITECTURES, build_model
from src.seed import set_global_seeds
from src.training import fit
from src.utils import collect_environment, select_device, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune one approved architecture on CIFAR-10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default=None,
        help="Architecture to train. Defaults to config training.architecture.",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to an alternative config YAML."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override config training.epochs."
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH|auto",
        help="Resume from a checkpoint. 'auto' uses the architecture's "
        "last checkpoint in the configured checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force a device (e.g. cpu, cuda). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=None,
        help="Train the classifier head only, holding the backbone and its "
        "normalisation statistics fixed.",
    )
    return parser.parse_args(argv)


def resolve_resume(argument: str | None, directory: Path, architecture: str) -> Path | None:
    """Turn the --resume argument into a concrete path.

    'auto' resolves to the architecture's rolling checkpoint. A missing file is
    an error rather than a silent restart: the user explicitly asked to resume.
    """
    if argument is None:
        return None
    path = (
        checkpoint_path(directory, architecture, "last")
        if argument == "auto"
        else Path(argument)
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Resume was requested but no checkpoint exists at {path}. "
            "Remove --resume to start a new run."
        )
    return path


def write_history(history, results_dir: Path, architecture: str) -> tuple[Path, Path]:
    """Persist the epoch log as JSON and CSV.

    Both land in the configured results directory, which is git-ignored.
    """
    payload = history.as_dict()
    payload["environment"] = collect_environment()
    json_path = write_json(payload, results_dir / f"training_history_{architecture}.json")

    csv_path = results_dir / f"training_history_{architecture}.csv"
    rows = history.as_rows()
    if rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    architecture = args.architecture or config.training.architecture

    set_global_seeds(
        config.reproducibility.seed, deterministic=config.reproducibility.deterministic
    )

    device = select_device() if args.device is None else args.device

    splits = build_datasets(config)
    # Three loaders are returned; the test loader is discarded here and is never
    # bound to a name that could reach the training code.
    train_loader, val_loader, _discarded_test_loader = build_dataloaders(config, splits)
    del _discarded_test_loader

    model = build_model(
        architecture,
        num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
        aux_logits=config.model.googlenet_aux_logits,
    )

    resume_from = resolve_resume(
        args.resume, Path(config.paths.checkpoints_dir), architecture
    )

    print(f"architecture: {architecture}")
    print(f"train / val sizes: {len(splits.train)} / {len(splits.val)}")
    print(f"split checksum: {splits.indices.checksum()}")

    history = fit(
        model,
        architecture=architecture,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        epochs=args.epochs,
        resume_from=resume_from,
        backbone_frozen=args.freeze_backbone,
    )

    json_path, csv_path = write_history(
        history, Path(config.paths.results_dir), architecture
    )
    print(f"\nepoch log: {json_path}")
    print(f"epoch log: {csv_path}")
    if history.best_epoch is not None:
        print(
            f"best validation accuracy {history.best_val_accuracy:.4f} "
            f"at epoch {history.best_epoch + 1}"
        )
    print("\nThe official test set was not read. Final evaluation is a separate stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
