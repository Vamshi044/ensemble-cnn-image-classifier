"""Shared helpers: environment capture, device selection, tensor inspection.

Nothing here makes modelling decisions. These are the small utilities that the
scripts and tests share so the logic is written once.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def collect_environment() -> dict[str, Any]:
    """Record the versions and hardware that a result was produced on.

    Captured alongside every set of results so a number can always be traced
    back to the environment that produced it.
    """
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    # Optional at import time so this helper works in a minimal environment.
    for name, module_path in (
        ("torchvision", "torchvision"),
        ("sklearn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("yaml", "yaml"),
    ):
        try:
            module = __import__(module_path)
            info[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[name] = "not installed"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_count"] = torch.cuda.device_count()
    else:
        info["gpu_name"] = None
        info["gpu_memory_gb"] = None
        info["gpu_count"] = 0

    return info


def select_device() -> torch.device:
    """Return the best available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> torch.Tensor:
    """Invert ``transforms.Normalize`` so a tensor can be viewed as an image.

    Used only for visualisation. The result is clamped to [0, 1] because
    augmentation (reflect-padded cropping) can push individual pixels slightly
    outside the original range once normalisation is undone.

    Args:
        tensor: Normalised image tensor, shape (C, H, W) or (N, C, H, W).
        mean: Per-channel mean originally subtracted.
        std: Per-channel standard deviation originally divided by.

    Returns:
        A tensor of the same shape with values in [0, 1].
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    shape = (1, -1, 1, 1) if tensor.ndim == 4 else (-1, 1, 1)
    return (tensor * std_t.view(*shape) + mean_t.view(*shape)).clamp(0.0, 1.0)


def tensor_health(tensor: torch.Tensor) -> dict[str, Any]:
    """Summarise a tensor and flag non-finite values.

    A silent NaN in the input pipeline produces a model that trains to nothing
    for reasons that are painful to diagnose later, so batches are checked here
    rather than trusted.
    """
    finite = torch.isfinite(tensor)
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "has_nan": bool(torch.isnan(tensor).any()),
        "has_inf": bool(torch.isinf(tensor).any()),
        "all_finite": bool(finite.all()),
    }


def _json_default(obj: Any) -> Any:
    """Make numpy / Path / dataclass values JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def write_json(payload: Any, path: Path) -> Path:
    """Write ``payload`` to ``path`` as indented JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def format_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    """Render rows as a fixed-width text table for console output."""
    all_rows = [headers, *[tuple(str(c) for c in r) for r in rows]]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), line]
    out += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in all_rows[1:]]
    return "\n".join(out)
