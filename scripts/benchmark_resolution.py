"""Measure the real per-image cost of the three target architectures at several
input resolutions on THIS machine.

Task 5 of the project spec asks for a *justified* input-resolution decision
rather than a reflexive "ImageNet models want 224x224". The dominant constraint
here is compute, so this script produces the measurement the decision rests on.

Important scope notes:
  * This performs NO training. It runs forward+backward on RANDOM tensors purely
    to time the graph. No data is read, no weights are learned, no accuracy is
    produced or implied.
  * Models are built with ``weights=None``. Timing depends on the architecture,
    not on the values in the weights, so this avoids downloading ~700 MB of
    pretrained checkpoints that Stage 1 does not need.

Output: results/resolution_benchmark.json
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

# Small batch: this machine has 8 GB of RAM and VGG11 at 224x224 holds large
# activation maps during the backward pass. We report per-image cost anyway,
# so batch size only needs to be big enough to amortise per-call overhead.
BATCH_SIZE = 16
WARMUP_ITERS = 1
TIMED_ITERS = 3
RESOLUTIONS = (32, 64, 96, 128, 224)
NUM_CLASSES = 10


def build_model(name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Construct an untrained architecture with a 10-class head.

    The heads are adapted here only so the benchmark measures the same graph the
    project will actually train later; these models are discarded immediately.
    """
    if name == "googlenet":
        # aux_logits=False: the torchvision auxiliary classifiers contain a
        # Linear(2048, 1024) that assumes a 4x4 spatial map, which only occurs
        # for inputs of ~224px. They cannot be used at reduced resolution.
        model = tvm.googlenet(weights=None, aux_logits=False, init_weights=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "resnet18":
        model = tvm.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "vgg11":
        model = tvm.vgg11(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {name}")
    return model


def time_one(model: nn.Module, resolution: int) -> float:
    """Return mean seconds per image for a forward+backward pass."""
    model.train()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: nothing is learned
    criterion = nn.CrossEntropyLoss()
    images = torch.randn(BATCH_SIZE, 3, resolution, resolution)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    def step() -> None:
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimiser.step()

    for _ in range(WARMUP_ITERS):
        step()

    start = time.perf_counter()
    for _ in range(TIMED_ITERS):
        step()
    elapsed = time.perf_counter() - start
    return elapsed / (TIMED_ITERS * BATCH_SIZE)


def main() -> None:
    results: dict = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "protocol": {
            "batch_size": BATCH_SIZE,
            "warmup_iters": WARMUP_ITERS,
            "timed_iters": TIMED_ITERS,
            "note": "forward+backward on random tensors; no training, no metrics",
        },
        "seconds_per_image": {},
    }

    for arch in ("resnet18", "googlenet", "vgg11"):
        results["seconds_per_image"][arch] = {}
        for res in RESOLUTIONS:
            model = build_model(arch)
            try:
                spi = time_one(model, res)
                results["seconds_per_image"][arch][str(res)] = round(spi, 6)
                # 45,000 training images per epoch.
                print(
                    f"{arch:<10} {res:>4}px  {spi * 1000:8.2f} ms/img"
                    f"   -> {spi * 45000 / 60:7.1f} min/epoch (train only)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - record and continue
                results["seconds_per_image"][arch][str(res)] = f"FAILED: {exc}"
                print(f"{arch:<10} {res:>4}px  FAILED: {exc}", flush=True)
            finally:
                del model

    out = Path(__file__).resolve().parents[1] / "results" / "resolution_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
