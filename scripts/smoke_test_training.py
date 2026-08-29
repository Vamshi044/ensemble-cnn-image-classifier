"""Tiny synthetic smoke test for the training infrastructure.

    python scripts/smoke_test_training.py
    python scripts/smoke_test_training.py --architecture vgg11 --amp

This proves the training machinery runs end to end on the selected device. It is
NOT training and it is NOT an experiment:

* the data is random noise from ``torch.randn`` with random labels - CIFAR-10 is
  never read, on any code path in this file;
* it runs two epochs over a handful of synthetic images;
* the loss and accuracy it prints are properties of random noise and mean
  nothing about any model's ability to classify images. They are shown only to
  demonstrate that numbers flow through the pipeline.

Its purpose is to answer one question: does the infrastructure execute correctly
on this device? On a CUDA host it additionally exercises autocast, the gradient
scaler and a checkpoint round-trip on the GPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpointing import checkpoint_path, load_checkpoint, restore_training_state
from src.config import load_config
from src.models import ARCHITECTURES, build_model
from src.seed import set_global_seeds
from src.training import build_optimizer, describe_device, fit, resolve_amp
from src.utils import select_device


def synthetic_loaders(
    image_size: int, batch_size: int, num_classes: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """Random-noise train and validation loaders.

    Deliberately uneven sizes: 12 training images and 10 validation images over
    a batch size of 4 leaves ragged final batches, which exercises the
    sample-weighted metric aggregation rather than the easy equal-batch case.
    """
    generator = torch.Generator().manual_seed(seed)

    def make(count: int) -> TensorDataset:
        images = torch.randn(count, 3, image_size, image_size, generator=generator)
        labels = torch.randint(0, num_classes, (count,), generator=generator)
        return TensorDataset(images, labels)

    return (
        DataLoader(make(12), batch_size=batch_size, shuffle=True),
        DataLoader(make(10), batch_size=batch_size, shuffle=False),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic smoke test for the training infrastructure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="resnet18")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Request mixed precision. Ignored (and reported) off CUDA.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Defaults to the config smoke-test size, not the training size.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Download ImageNet weights. Off by default so the smoke test needs "
        "no network.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    set_global_seeds(config.reproducibility.seed)

    device = select_device()
    info = describe_device(device)
    image_size = args.image_size or config.image.smoke_test_size

    print("=" * 72)
    print("TRAINING INFRASTRUCTURE SMOKE TEST (synthetic data - not an experiment)")
    print("=" * 72)
    print(f"device            : {info.device}")
    print(f"cuda available    : {info.cuda_available}")
    print(f"gpu name          : {info.gpu_name}")
    print(f"gpu memory (GB)   : {info.total_memory_gb}")
    print(f"gpu free (GB)     : {info.free_memory_gb}")
    if info.memory_query_error:
        print(f"memory query note : {info.memory_query_error}")
    print(f"capability        : {info.capability}")
    print(f"architecture      : {args.architecture}")
    print(f"image size        : {image_size}  (synthetic noise, not CIFAR-10)")
    print()

    if not info.cuda_available:
        print("!! CUDA IS NOT AVAILABLE ON THIS HOST.")
        print("!! The CUDA path of this smoke test is PENDING and has NOT been run.")
        print("!! Re-run this script on the Colab T4 to exercise it.")
        print("Continuing on CPU to verify the device-agnostic path.\n")

    # Build the model and confirm every parameter really moved to the device.
    model = build_model(
        args.architecture,
        num_classes=config.data.num_classes,
        pretrained=args.pretrained,
        aux_logits=config.model.googlenet_aux_logits,
    ).to(device)
    devices = {p.device.type for p in model.parameters()}
    print(f"parameter devices : {devices}")
    assert devices == {device.type}, f"model split across devices: {devices}"

    # Mixed precision is resolved from config; --amp flips the request on.
    # dataclasses.replace rather than mutating a frozen dataclass in place.
    if args.amp:
        config = replace(config, training=replace(config.training, amp=True))
    amp = resolve_amp(config, device)
    print(f"amp               : {amp.reason}")

    train_loader, val_loader = synthetic_loaders(
        image_size, args.batch_size, config.data.num_classes, config.reproducibility.seed
    )

    checkpoint_dir = Path(tempfile.mkdtemp(prefix="smoke_ckpt_"))
    try:
        before = next(model.parameters()).detach().clone()

        history = fit(
            model,
            architecture=args.architecture,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
        )

        after = next(model.parameters()).detach()
        print(f"\nparameters changed: {not torch.equal(before, after)}")
        print(f"epochs completed  : {history.completed_epochs}")
        print(f"train samples/ep  : {history.records[0].train_samples}")
        print(f"val samples/ep    : {history.records[0].val_samples}")

        # Checkpoint round-trip on this device.
        best = checkpoint_path(checkpoint_dir, args.architecture, "best")
        last = checkpoint_path(checkpoint_dir, args.architecture, "last")
        print(f"best checkpoint   : {best.is_file()} ({best.stat().st_size // 1024} KB)")
        print(f"last checkpoint   : {last.is_file()}")

        fresh = build_model(
            args.architecture,
            num_classes=config.data.num_classes,
            pretrained=False,
            aux_logits=config.model.googlenet_aux_logits,
        ).to(device)
        optimizer = build_optimizer(fresh, args.architecture, config)
        state = restore_training_state(
            load_checkpoint(last, map_location=device),
            model=fresh,
            architecture=args.architecture,
            optimizer=optimizer,
        )
        reloaded = all(
            torch.equal(a.to(device), b.to(device))
            for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True)
        )
        print(f"checkpoint reload : restored={state.restored} match={reloaded}")
        assert reloaded, "reloaded weights differ from the saved model"

        print("\nRESULT: infrastructure executed successfully on "
              f"{info.device}.")
        if not info.cuda_available:
            print("RESULT: CUDA path REMAINS PENDING - run this on the Colab T4.")
        print("No CIFAR-10 data was read. No accuracy was measured.")
        return 0
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
