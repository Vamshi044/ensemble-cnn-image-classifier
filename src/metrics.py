"""Metric accumulation for training and validation.

Two properties matter here and are the reason this is a module rather than a
few lines inlined in the training loop.

**Aggregation is sample-weighted, not batch-weighted.** Averaging per-batch
accuracies is only correct when every batch is the same size. The validation
loader deliberately does *not* drop its last batch (5,000 images over a batch
size of 128 leaves a final batch of 8), so a naive mean over batches would
weight those 8 images as heavily as a full 128. Everything here accumulates
totals and divides once at the end.

**Accumulators are Python floats and ints, never tensors.** Adding a CUDA
tensor to a running total each step keeps that tensor - and potentially its
autograd graph - alive for the whole epoch. Values are converted with
``float()`` on arrival, which both frees the tensor and forces a device
synchronisation point that would otherwise happen implicitly anyway.

Nothing in this module knows about the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EpochMetrics:
    """Aggregated results over one complete pass of a DataLoader."""

    loss: float
    accuracy: float
    num_samples: int
    num_correct: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
        }


def count_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Number of top-1 correct predictions in a batch.

    Args:
        logits: Raw model outputs, shape ``(batch, num_classes)``.
        targets: Integer class labels, shape ``(batch,)``.

    Returns:
        The count of correct predictions as a plain int.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits of shape (batch, num_classes), got {tuple(logits.shape)}."
        )
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Expected targets of shape ({logits.shape[0]},), "
            f"got {tuple(targets.shape)}."
        )
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


class MetricTracker:
    """Accumulate loss and accuracy across the batches of one epoch.

    Usage is one tracker per pass::

        tracker = MetricTracker()
        for images, targets in loader:
            ...
            tracker.update(logits, targets, loss)
        metrics = tracker.compute()
    """

    def __init__(self) -> None:
        self._loss_sum: float = 0.0
        self._correct: int = 0
        self._total: int = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor | float,
    ) -> None:
        """Fold one batch into the running totals.

        Args:
            logits: Model outputs for the batch. Detach before calling; this
                method does not hold a reference beyond computing the argmax.
            targets: Ground-truth labels for the batch.
            loss: The batch's mean loss. Converted to a Python float
                immediately so no tensor is retained across steps.

        The loss is multiplied by the batch size before accumulation, so the
        final average is over samples rather than over batches.
        """
        batch_size = int(targets.shape[0])
        if batch_size == 0:
            return

        # detach() before float(): converting a tensor that still carries
        # requires_grad warns, and the detach makes it explicit that no part of
        # the graph is being kept alive by this accumulator.
        loss_value = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        self._loss_sum += loss_value * batch_size
        self._correct += count_correct(logits, targets)
        self._total += batch_size

    def compute(self) -> EpochMetrics:
        """Return the aggregated metrics.

        Raises:
            ValueError: If no samples were seen. An empty pass has no accuracy,
                and returning 0.0 would look like a real measurement.
        """
        if self._total == 0:
            raise ValueError(
                "No samples were accumulated; there is no metric to report. "
                "This usually means the DataLoader was empty."
            )
        return EpochMetrics(
            loss=self._loss_sum / self._total,
            accuracy=self._correct / self._total,
            num_samples=self._total,
            num_correct=self._correct,
        )

    @property
    def num_samples(self) -> int:
        """Samples accumulated so far."""
        return self._total
