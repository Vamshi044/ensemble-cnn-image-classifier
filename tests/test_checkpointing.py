"""Tests for checkpoint saving, loading and resume.

The most important test here is
``test_resume_restores_the_learning_rate_not_just_the_scheduler``. It is a
regression test for a real trap found by execution during Stage 3A:
``SequentialLR.load_state_dict`` restores the scheduler's counters but does not
write the learning rate back into ``optimizer.param_groups``, so restoring the
scheduler alone silently resumes at the warmup learning rate.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
    checkpoint_path,
    load_checkpoint,
    restore_rng_state,
    restore_training_state,
    save_checkpoint,
)
from tests.conftest import TinyNet


def build_pieces(epochs: int = 10, warmup: int = 1):
    """A model, optimiser and warmup+cosine scheduler with distinct group LRs."""
    torch.manual_seed(0)
    model = TinyNet()
    optimizer = torch.optim.SGD(
        [
            {"params": list(model.conv.parameters()), "lr": 0.005, "name": "backbone"},
            {"params": list(model.fc.parameters()), "lr": 0.05, "name": "head"},
        ],
        momentum=0.9,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-5)
    warm = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    scheduler = SequentialLR(optimizer, [warm, cosine], milestones=[warmup])
    return model, optimizer, scheduler


# ---------------------------------------------------------------------------
# Paths - no hard-coded locations
# ---------------------------------------------------------------------------


def test_checkpoint_path_uses_the_supplied_directory(tmp_path):
    path = checkpoint_path(tmp_path, "resnet18", "best")
    assert path.parent == tmp_path
    assert "resnet18" in path.name


def test_checkpoint_path_distinguishes_best_and_last(tmp_path):
    best = checkpoint_path(tmp_path, "vgg11", "best")
    last = checkpoint_path(tmp_path, "vgg11", "last")
    assert best != last


def test_checkpoint_path_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="Unknown checkpoint kind"):
        checkpoint_path(tmp_path, "vgg11", "penultimate")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_creates_the_file(tmp_path):
    model, optimizer, scheduler = build_pieces()
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert path.is_file() and path.stat().st_size > 0


def test_save_creates_missing_parent_directories(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "nested" / "deeper" / "ckpt.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
    )
    assert path.is_file()


def test_save_leaves_no_temporary_file_behind(tmp_path):
    """The atomic write must not leave a .tmp artefact."""
    model, _, _ = build_pieces()
    save_checkpoint(
        tmp_path / "ckpt.pt", model=model, architecture="resnet18", epoch=0
    )
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_overwrites_atomically(tmp_path):
    """A second save replaces the first and remains loadable."""
    model, _, _ = build_pieces()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, architecture="resnet18", epoch=0)
    save_checkpoint(path, model=model, architecture="resnet18", epoch=7)
    assert load_checkpoint(path)["epoch"] == 7


def test_disabled_scaler_state_is_not_stored(tmp_path):
    """A run without AMP must not imply it used mixed precision."""
    model, _, _ = build_pieces()
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
        scaler=scaler,
    )
    assert "scaler_state" not in load_checkpoint(path)


def test_config_snapshot_is_stored(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
        config_snapshot={"epochs": 10, "optimizer": "sgd"},
    )
    assert load_checkpoint(path)["config"]["optimizer"] == "sgd"


def test_architecture_identifier_is_stored(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt", model=model, architecture="googlenet", epoch=0
    )
    assert load_checkpoint(path)["architecture"] == "googlenet"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_missing_file_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Resume was requested"):
        load_checkpoint(tmp_path / "absent.pt")


def test_load_rejects_a_foreign_format_version(tmp_path):
    torch.save({"format_version": 999, "architecture": "resnet18"}, tmp_path / "c.pt")
    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(tmp_path / "c.pt")


def test_checkpoint_loads_under_weights_only(tmp_path):
    """load_checkpoint uses weights_only=True; everything stored must comply."""
    model, optimizer, scheduler = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=1,
        optimizer=optimizer,
        scheduler=scheduler,
        config_snapshot={"epochs": 2},
    )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_model_weights_round_trip(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt", model=model, architecture="resnet18", epoch=0
    )

    fresh = TinyNet()
    torch.nn.init.constant_(fresh.fc.weight, 0.123)
    assert not torch.equal(fresh.fc.weight, model.fc.weight)

    restore_training_state(
        load_checkpoint(path), model=fresh, architecture="resnet18", restore_rng=False
    )
    assert torch.equal(fresh.fc.weight, model.fc.weight)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_restores_the_epoch(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt", model=model, architecture="resnet18", epoch=4
    )
    state = restore_training_state(
        load_checkpoint(path), model=TinyNet(), architecture="resnet18", restore_rng=False
    )
    # Epoch 4 completed, so training continues at epoch 5.
    assert state.start_epoch == 5


def test_resume_restores_the_best_metric(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=3,
        best_metric=0.4212,
        best_epoch=2,
    )
    state = restore_training_state(
        load_checkpoint(path), model=TinyNet(), architecture="resnet18", restore_rng=False
    )
    assert state.best_metric == pytest.approx(0.4212)
    assert state.best_epoch == 2


def test_optimizer_state_restores(tmp_path):
    """Momentum buffers must survive, or resume is not a continuation."""
    model, optimizer, _ = build_pieces()
    loss = model(torch.randn(2, 3, 8, 8)).sum()
    loss.backward()
    optimizer.step()

    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
        optimizer=optimizer,
    )

    fresh_model, fresh_optimizer, _ = build_pieces()
    assert len(fresh_optimizer.state) == 0

    state = restore_training_state(
        load_checkpoint(path),
        model=fresh_model,
        architecture="resnet18",
        optimizer=fresh_optimizer,
        restore_rng=False,
    )
    assert "optimizer" in state.restored
    assert len(fresh_optimizer.state) == len(optimizer.state) > 0


def test_scheduler_state_restores(tmp_path):
    model, optimizer, scheduler = build_pieces()
    for _ in range(3):
        optimizer.step()
        scheduler.step()

    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=2,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    fresh_model, fresh_optimizer, fresh_scheduler = build_pieces()
    state = restore_training_state(
        load_checkpoint(path),
        model=fresh_model,
        architecture="resnet18",
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        restore_rng=False,
    )
    assert "scheduler" in state.restored
    assert fresh_scheduler.last_epoch == scheduler.last_epoch


def test_resume_restores_the_learning_rate_not_just_the_scheduler(tmp_path):
    """Regression test for a real trap.

    SequentialLR.load_state_dict restores the scheduler's counters but does NOT
    write the learning rate back into optimizer.param_groups. The learning rate
    lives in the optimiser state, so both must be restored. Without the
    optimiser the run would silently resume at the warmup rate.
    """
    model, optimizer, scheduler = build_pieces()
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    expected = [group["lr"] for group in optimizer.param_groups]

    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=3,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    fresh_model, fresh_optimizer, fresh_scheduler = build_pieces()
    warmup_lrs = [group["lr"] for group in fresh_optimizer.param_groups]

    restore_training_state(
        load_checkpoint(path),
        model=fresh_model,
        architecture="resnet18",
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        restore_rng=False,
    )
    restored = [group["lr"] for group in fresh_optimizer.param_groups]

    assert restored == pytest.approx(expected)
    # And it is genuinely different from the un-restored warmup value, so the
    # assertion above is not vacuous.
    assert restored != pytest.approx(warmup_lrs)


def test_resumed_schedule_continues_identically(tmp_path):
    """The LR trajectory after a resume must match an uninterrupted run."""
    model, optimizer, scheduler = build_pieces()
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=3,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    uninterrupted = []
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        uninterrupted.append(optimizer.param_groups[1]["lr"])

    fresh_model, fresh_optimizer, fresh_scheduler = build_pieces()
    restore_training_state(
        load_checkpoint(path),
        model=fresh_model,
        architecture="resnet18",
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        restore_rng=False,
    )
    resumed = []
    for _ in range(5):
        fresh_optimizer.step()
        fresh_scheduler.step()
        resumed.append(fresh_optimizer.param_groups[1]["lr"])

    assert resumed == pytest.approx(uninterrupted)


def test_scaler_state_restores_when_amp_was_enabled(tmp_path):
    """An enabled scaler round-trips even when saved from a CPU host."""
    model, _, _ = build_pieces()
    scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    scaler.load_state_dict(
        {
            "scale": 512.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 0,
        }
    )
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
        scaler=scaler,
    )
    assert "scaler_state" in load_checkpoint(path)

    fresh_scaler = torch.amp.GradScaler(device="cpu", enabled=True)
    state = restore_training_state(
        load_checkpoint(path),
        model=TinyNet(),
        architecture="resnet18",
        scaler=fresh_scaler,
        restore_rng=False,
    )
    assert "scaler" in state.restored
    assert fresh_scaler.get_scale() == pytest.approx(512.0)


def test_scaler_absence_is_reported_not_hidden(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt", model=model, architecture="resnet18", epoch=0
    )
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    state = restore_training_state(
        load_checkpoint(path),
        model=TinyNet(),
        architecture="resnet18",
        scaler=scaler,
        restore_rng=False,
    )
    assert any("scaler" in item for item in state.skipped)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_architecture_mismatch_raises(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt", model=model, architecture="vgg11", epoch=0
    )
    with pytest.raises(ValueError, match="architecture mismatch"):
        restore_training_state(
            load_checkpoint(path),
            model=TinyNet(),
            architecture="resnet18",
            restore_rng=False,
        )


def test_scheduler_without_optimizer_raises(tmp_path):
    model, optimizer, scheduler = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    with pytest.raises(ValueError, match="wrong learning rate"):
        restore_training_state(
            load_checkpoint(path),
            model=TinyNet(),
            architecture="resnet18",
            scheduler=scheduler,
            restore_rng=False,
        )


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------


def test_rng_capture_uses_primitives_only():
    """Anything non-primitive would break weights_only=True loading."""
    state = capture_rng_state()
    assert isinstance(state["numpy"][1], list)
    assert all(isinstance(v, int) for v in state["numpy"][1][:8])
    assert isinstance(state["torch"], torch.Tensor)


def test_rng_restore_reproduces_all_three_streams():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    state = capture_rng_state()

    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    # Advance every stream so a no-op restore could not pass this test.
    for _ in range(5):
        random.random()
        np.random.rand()
        torch.rand(1)

    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert actual == pytest.approx(expected)


def test_rng_state_round_trips_through_a_checkpoint(tmp_path):
    model, _, _ = build_pieces()
    np.random.seed(7)

    path = save_checkpoint(
        tmp_path / "c.pt", model=model, architecture="resnet18", epoch=0
    )
    expected_after_save = float(np.random.rand())

    np.random.seed(999)
    restore_training_state(
        load_checkpoint(path),
        model=TinyNet(),
        architecture="resnet18",
        restore_rng=True,
    )
    assert float(np.random.rand()) == pytest.approx(expected_after_save)


def test_rng_restore_is_skippable(tmp_path):
    model, _, _ = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt", model=model, architecture="resnet18", epoch=0
    )
    state = restore_training_state(
        load_checkpoint(path),
        model=TinyNet(),
        architecture="resnet18",
        restore_rng=False,
    )
    assert "rng_state" not in state.restored


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA device on this host; this path is pending the Colab T4 run",
)


@requires_cuda
def test_rng_restores_when_the_checkpoint_is_mapped_onto_cuda(tmp_path):
    """Regression test for a defect found by inspection in Stage 3A.1.

    ``fit`` resumes with ``load_checkpoint(path, map_location=device)``. Verified
    by execution: every storage in the file is routed through ``map_location``,
    and the RNG states are ordinary uint8 storages, so on CUDA they come back as
    CUDA tensors. A generator's state must be a CPU ByteTensor, so
    ``restore_rng_state`` has to move them back. It did that for the CPU
    generator but not for the CUDA ones.
    """
    model, optimizer, scheduler = build_pieces()
    path = save_checkpoint(
        tmp_path / "c.pt",
        model=model,
        architecture="resnet18",
        epoch=0,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    device = torch.device("cuda")
    checkpoint = load_checkpoint(path, map_location=device)

    assert checkpoint["rng_state"]["torch"].device.type == "cuda", (
        "precondition: map_location should have moved the RNG state to CUDA"
    )

    # Must not raise.
    state = restore_training_state(
        checkpoint,
        model=TinyNet().to(device),
        architecture="resnet18",
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert "rng_state" in state.restored
