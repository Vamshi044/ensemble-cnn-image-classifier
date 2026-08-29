"""Centralised configuration.

Every parameter that matters to a result lives in ``configs/config.yaml`` and is
loaded into the typed dataclasses below. Nothing important should be hard-coded
elsewhere in the codebase: if a number changes an outcome, it belongs here.

Paths in the YAML file are relative to the project root and are resolved to
absolute paths at load time, so the project has no machine-specific paths
baked into it and can be checked out anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    """Dataset identity, location and split sizes."""

    dataset_name: str
    root: Path
    num_classes: int
    train_size: int
    val_size: int
    official_train_size: int
    official_test_size: int
    class_names: list[str]
    download: bool

    def __post_init__(self) -> None:
        if self.train_size + self.val_size != self.official_train_size:
            raise ValueError(
                f"train_size ({self.train_size}) + val_size ({self.val_size}) must "
                f"equal official_train_size ({self.official_train_size}); the "
                "official test set is never part of this split."
            )
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but num_classes "
                f"is {self.num_classes}."
            )


@dataclass(frozen=True)
class ImageConfig:
    """Input geometry and normalisation statistics."""

    size: int
    smoke_test_size: int
    native_size: int
    interpolation: str
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    normalization_source: str


@dataclass(frozen=True)
class AugmentationConfig:
    """Training-time augmentation. Applied to the training split only."""

    random_crop_padding: int
    random_crop_padding_mode: str
    horizontal_flip_prob: float


@dataclass(frozen=True)
class DataLoaderConfig:
    """Batching and worker behaviour."""

    batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: str | bool
    persistent_workers: bool
    drop_last: bool
    prefetch_factor: int | None


@dataclass(frozen=True)
class TrainingConfig:
    """Initial training hyperparameters.

    These are *starting points* recorded so Stage 2 has a defined baseline. They
    are not yet validated; any tuning must use the validation split only.
    """

    epochs: int
    optimizer: str
    learning_rate: float
    head_learning_rate: float
    momentum: float
    weight_decay: float
    nesterov: bool
    scheduler: str
    scheduler_warmup_epochs: int
    scheduler_min_lr: float
    label_smoothing: float
    grad_clip_norm: float | None
    early_stopping_patience: int
    amp: bool


@dataclass(frozen=True)
class ModelConfig:
    """Architectures and transfer-learning settings."""

    architectures: list[str]
    pretrained: bool
    googlenet_aux_logits: bool
    replace_classifier_head: bool


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Seeding and determinism."""

    seed: int
    deterministic: bool
    split_seed: int


@dataclass(frozen=True)
class PathsConfig:
    """Output locations."""

    results_dir: Path
    checkpoints_dir: Path


@dataclass(frozen=True)
class Config:
    """Top-level configuration object."""

    data: DataConfig
    image: ImageConfig
    augmentation: AugmentationConfig
    dataloader: DataLoaderConfig
    training: TrainingConfig
    model: ModelConfig
    reproducibility: ReproducibilityConfig
    paths: PathsConfig
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _resolve(path_value: str) -> Path:
    """Resolve a config path relative to the project root unless absolute."""
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate the project configuration.

    Args:
        path: Optional path to a YAML config. Defaults to ``configs/config.yaml``.

    Returns:
        A fully populated, validated :class:`Config`.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    data_raw = raw["data"]
    image_raw = raw["image"]
    paths_raw = raw["paths"]

    return Config(
        data=DataConfig(
            dataset_name=data_raw["dataset_name"],
            root=_resolve(data_raw["root"]),
            num_classes=int(data_raw["num_classes"]),
            train_size=int(data_raw["train_size"]),
            val_size=int(data_raw["val_size"]),
            official_train_size=int(data_raw["official_train_size"]),
            official_test_size=int(data_raw["official_test_size"]),
            class_names=list(data_raw["class_names"]),
            download=bool(data_raw["download"]),
        ),
        image=ImageConfig(
            size=int(image_raw["size"]),
            smoke_test_size=int(image_raw["smoke_test_size"]),
            native_size=int(image_raw["native_size"]),
            interpolation=image_raw["interpolation"],
            normalization_mean=tuple(float(v) for v in image_raw["normalization_mean"]),
            normalization_std=tuple(float(v) for v in image_raw["normalization_std"]),
            normalization_source=image_raw["normalization_source"],
        ),
        augmentation=AugmentationConfig(**raw["augmentation"]),
        dataloader=DataLoaderConfig(**raw["dataloader"]),
        training=TrainingConfig(**raw["training"]),
        model=ModelConfig(**raw["model"]),
        reproducibility=ReproducibilityConfig(**raw["reproducibility"]),
        paths=PathsConfig(
            results_dir=_resolve(paths_raw["results_dir"]),
            checkpoints_dir=_resolve(paths_raw["checkpoints_dir"]),
        ),
        raw=raw,
    )
