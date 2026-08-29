"""Tests for metric aggregation.

The property under test throughout is that aggregation is weighted by SAMPLES,
not by batches. The validation loader does not drop its last batch, so unequal
batch sizes are the normal case rather than an edge case, and a mean over
batches would silently overweight the small final batch.
"""

from __future__ import annotations

import pytest
import torch

from src.metrics import EpochMetrics, MetricTracker, count_correct

# ---------------------------------------------------------------------------
# count_correct
# ---------------------------------------------------------------------------


def test_count_correct_all_right():
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
    targets = torch.tensor([0, 1])
    assert count_correct(logits, targets) == 2


def test_count_correct_all_wrong():
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
    targets = torch.tensor([1, 0])
    assert count_correct(logits, targets) == 0


def test_count_correct_mixed():
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0], [9.0, 0.0]])
    targets = torch.tensor([0, 0, 0])
    assert count_correct(logits, targets) == 2


def test_count_correct_returns_plain_int():
    result = count_correct(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    assert isinstance(result, int) and not isinstance(result, torch.Tensor)


def test_count_correct_rejects_wrong_logit_rank():
    with pytest.raises(ValueError, match="batch, num_classes"):
        count_correct(torch.zeros(4), torch.zeros(4, dtype=torch.long))


def test_count_correct_rejects_mismatched_batch():
    with pytest.raises(ValueError, match="Expected targets"):
        count_correct(torch.zeros(3, 2), torch.zeros(4, dtype=torch.long))


# ---------------------------------------------------------------------------
# Sample weighting - the core correctness requirement
# ---------------------------------------------------------------------------


def test_accuracy_is_total_correct_over_total_samples():
    """Two ragged batches: 2/4 correct then 1/1 correct is 3/5, not the 0.75
    that averaging the two batch accuracies (0.5 and 1.0) would give."""
    tracker = MetricTracker()

    logits_a = torch.tensor([[9.0, 0.0], [9.0, 0.0], [0.0, 9.0], [0.0, 9.0]])
    tracker.update(logits_a, torch.tensor([0, 0, 0, 0]), loss=1.0)

    logits_b = torch.tensor([[0.0, 9.0]])
    tracker.update(logits_b, torch.tensor([1]), loss=1.0)

    metrics = tracker.compute()
    assert metrics.num_correct == 3
    assert metrics.num_samples == 5
    assert metrics.accuracy == pytest.approx(3 / 5)
    # The naive batch-mean would be 0.75; assert we did not produce it.
    assert metrics.accuracy != pytest.approx(0.75)


def test_loss_is_sample_weighted_across_unequal_batches():
    """A loss of 1.0 over 4 samples then 5.0 over 1 sample averages to
    (1*4 + 5*1) / 5 = 1.8, not the 3.0 a batch-mean would produce."""
    tracker = MetricTracker()
    tracker.update(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), loss=1.0)
    tracker.update(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), loss=5.0)

    metrics = tracker.compute()
    assert metrics.loss == pytest.approx(1.8)
    assert metrics.loss != pytest.approx(3.0)


def test_equal_batches_make_both_aggregations_agree():
    """Sanity check on the claim that the two agree when sizes are equal."""
    tracker = MetricTracker()
    tracker.update(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), loss=1.0)
    tracker.update(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long), loss=3.0)
    assert tracker.compute().loss == pytest.approx(2.0)


def test_perfect_and_zero_accuracy_bounds():
    perfect = MetricTracker()
    perfect.update(torch.tensor([[9.0, 0.0]]), torch.tensor([0]), loss=0.0)
    assert perfect.compute().accuracy == 1.0

    zero = MetricTracker()
    zero.update(torch.tensor([[9.0, 0.0]]), torch.tensor([1]), loss=0.0)
    assert zero.compute().accuracy == 0.0


# ---------------------------------------------------------------------------
# Numerical safety and tensor retention
# ---------------------------------------------------------------------------


def test_empty_tracker_raises_rather_than_reporting_zero():
    """An empty pass has no accuracy. Returning 0.0 would look like a result."""
    with pytest.raises(ValueError, match="No samples"):
        MetricTracker().compute()


def test_zero_length_batch_is_ignored():
    tracker = MetricTracker()
    tracker.update(torch.zeros(0, 2), torch.zeros(0, dtype=torch.long), loss=1.0)
    assert tracker.num_samples == 0


def test_tracker_accepts_tensor_loss_and_stores_a_float():
    tracker = MetricTracker()
    tracker.update(
        torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), loss=torch.tensor(2.5)
    )
    metrics = tracker.compute()
    assert isinstance(metrics.loss, float)
    assert metrics.loss == pytest.approx(2.5)


def test_tracker_does_not_retain_the_autograd_graph():
    """A loss carrying grad_fn must not keep the graph alive in the totals."""
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    loss = (parameter * 2).sum()
    assert loss.requires_grad

    tracker = MetricTracker()
    tracker.update(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), loss=loss)
    stored = tracker.compute().loss
    assert isinstance(stored, float)


def test_num_samples_tracks_incrementally():
    tracker = MetricTracker()
    assert tracker.num_samples == 0
    tracker.update(torch.zeros(3, 2), torch.zeros(3, dtype=torch.long), loss=1.0)
    assert tracker.num_samples == 3
    tracker.update(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), loss=1.0)
    assert tracker.num_samples == 5


def test_epoch_metrics_as_dict_round_trip():
    metrics = EpochMetrics(loss=1.5, accuracy=0.25, num_samples=8, num_correct=2)
    assert metrics.as_dict() == {
        "loss": 1.5,
        "accuracy": 0.25,
        "num_samples": 8,
        "num_correct": 2,
    }
