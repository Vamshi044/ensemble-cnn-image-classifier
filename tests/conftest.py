"""Shared fixtures for the Stage 3A training-infrastructure tests.

Everything here is synthetic. No test in this suite reads CIFAR-10, and none
performs a real training experiment: the models are a four-parameter toy net
(or an untrained torchvision architecture where the real one is the point), and
the data is random noise. Losses and accuracies produced under these fixtures
describe random tensors and carry no information about model quality.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


class TinyNet(nn.Module):
    """A minimal stand-in for the real architectures.

    Its classifier is named ``fc`` so it shares the parameter-name layout that
    :data:`src.models._HEAD_PREFIX` maps for GoogLeNet and ResNet18. Tests that
    need parameter grouping therefore pass ``architecture="resnet18"`` and
    exercise the real Stage 2 splitting logic without paying for a real network.

    It contains a BatchNorm layer so frozen-backbone behaviour is testable.
    """

    def __init__(self, num_classes: int = 4, width: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, num_classes)
        # Recorded by forward() so tests can assert the mode the model was
        # actually called in, rather than the mode it happens to be in
        # afterwards.
        self.training_flags_seen: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.training_flags_seen.append(self.training)
        x = torch.relu(self.bn(self.conv(x)))
        return self.fc(torch.flatten(self.pool(x), 1))


def make_loader(
    count: int,
    *,
    batch_size: int,
    num_classes: int = 4,
    image_size: int = 8,
    seed: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """A loader over random noise with deterministic content."""
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(count, 3, image_size, image_size, generator=generator)
    labels = torch.randint(0, num_classes, (count,), generator=generator)
    return DataLoader(
        TensorDataset(images, labels), batch_size=batch_size, shuffle=shuffle
    )


@pytest.fixture
def tiny_model() -> TinyNet:
    torch.manual_seed(0)
    return TinyNet()


@pytest.fixture
def train_loader() -> DataLoader:
    # 10 images over a batch size of 4 gives batches of 4, 4, 2 - deliberately
    # ragged so sample-weighted aggregation is actually exercised.
    return make_loader(10, batch_size=4, seed=1)


@pytest.fixture
def val_loader() -> DataLoader:
    return make_loader(6, batch_size=4, seed=2)


@pytest.fixture(scope="session")
def base_config():
    """The real project configuration, loaded once."""
    return load_config()


@pytest.fixture
def tiny_config(base_config, tmp_path):
    """Project config narrowed to a fast, self-contained synthetic run.

    Checkpoint and results directories are redirected into pytest's ``tmp_path``
    so no test writes into the repository.
    """
    return replace(
        base_config,
        training=replace(
            base_config.training,
            epochs=2,
            scheduler_warmup_epochs=1,
            early_stopping_enabled=False,
        ),
        paths=replace(
            base_config.paths,
            checkpoints_dir=tmp_path / "checkpoints",
            results_dir=tmp_path / "results",
        ),
    )
