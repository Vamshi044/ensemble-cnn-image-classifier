"""Tests for the Stage 3B orchestrator's resume planning.

``src.training.fit`` has always been able to resume; the orchestrator was what
could not ask it to. These tests cover that wiring, because the failure it
prevents is silent and expensive: a hosted runtime is reclaimed mid-run, the
run is started again, and epoch 1 overwrites the only copy of the partially
trained model.

Nothing here trains. The checkpoint files are empty placeholders, because
:func:`resolve_resume` decides on existence alone, and the one test that
exercises :func:`run_one` replaces the model and the training loop with stubs.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_stage3b
from run_stage3b import PreflightError, resolve_resume, verify_resume_compatibility

from src.checkpointing import save_checkpoint


class StubHistory:
    """The minimum ``run_one`` consumes: no epochs were run."""

    def __init__(self, architecture: str) -> None:
        self.architecture = architecture
        self.records: list = []
        self.best_epoch = None
        self.best_val_accuracy = None

    def as_dict(self) -> dict:
        return {"architecture": self.architecture}

    def as_rows(self) -> list:
        return []


def touch(directory: Path, architecture: str, kind: str) -> Path:
    path = directory / f"{architecture}_{kind}.pt"
    path.write_bytes(b"")
    return path


def test_auto_resumes_when_a_last_checkpoint_exists(tmp_path):
    last = touch(tmp_path, "resnet18", "last")
    touch(tmp_path, "resnet18", "best")
    assert resolve_resume(tmp_path, "resnet18", "auto", False) == last


def test_auto_starts_fresh_when_nothing_exists(tmp_path):
    assert resolve_resume(tmp_path, "googlenet", "auto", False) is None


def test_fresh_start_refuses_to_overwrite_a_last_checkpoint(tmp_path):
    touch(tmp_path, "resnet18", "last")
    with pytest.raises(PreflightError, match="already exists"):
        resolve_resume(tmp_path, "resnet18", "off", False)


def test_auto_refuses_a_best_checkpoint_with_no_last(tmp_path):
    """A best without a last is a half-copied recovery, not a fresh start."""
    touch(tmp_path, "vgg11", "best")
    with pytest.raises(PreflightError, match="already exists"):
        resolve_resume(tmp_path, "vgg11", "auto", False)


def test_force_restart_permits_overwriting(tmp_path):
    touch(tmp_path, "resnet18", "last")
    assert resolve_resume(tmp_path, "resnet18", "off", True) is None


def test_force_restart_does_not_override_auto_resume(tmp_path):
    """--force-restart permits an overwrite; it does not request one."""
    last = touch(tmp_path, "resnet18", "last")
    assert resolve_resume(tmp_path, "resnet18", "auto", True) == last


def test_resume_is_only_decided_per_architecture(tmp_path):
    touch(tmp_path, "resnet18", "last")
    assert resolve_resume(tmp_path, "resnet18", "auto", False) is not None
    assert resolve_resume(tmp_path, "googlenet", "auto", False) is None
    assert resolve_resume(tmp_path, "vgg11", "auto", False) is None


def test_run_one_forwards_the_resume_path_to_fit(tmp_path, monkeypatch, base_config):
    """The wiring that was missing: a resolved checkpoint must reach ``fit``."""
    seen: dict = {}

    def fake_fit(model, **kwargs):
        seen.update(kwargs)
        return StubHistory("resnet18")

    monkeypatch.setattr(run_stage3b, "build_model", lambda *a, **k: nn.Linear(2, 2))
    monkeypatch.setattr(run_stage3b, "fit", fake_fit)
    monkeypatch.setattr(run_stage3b, "verify_checkpoints", lambda *a, **k: {})

    resume = tmp_path / "resnet18_last.pt"
    record = {"split_checksum": "bdb035810af794a7", "device": {}, "amp": {},
              "git_head": "test"}
    payload = run_stage3b.run_one(
        "resnet18", base_config, "cpu", None, None,
        tmp_path, tmp_path, record, resume_from=resume,
    )

    assert seen["resume_from"] == resume
    assert payload["resumed_from"] == str(resume)


def test_run_one_passes_none_when_starting_fresh(tmp_path, monkeypatch, base_config):
    seen: dict = {}

    def fake_fit(model, **kwargs):
        seen.update(kwargs)
        return StubHistory("googlenet")

    monkeypatch.setattr(run_stage3b, "build_model", lambda *a, **k: nn.Linear(2, 2))
    monkeypatch.setattr(run_stage3b, "fit", fake_fit)
    monkeypatch.setattr(run_stage3b, "verify_checkpoints", lambda *a, **k: {})

    record = {"split_checksum": "bdb035810af794a7", "device": {}, "amp": {},
              "git_head": "test"}
    payload = run_stage3b.run_one(
        "googlenet", base_config, "cpu", None, None, tmp_path, tmp_path, record,
    )

    assert seen["resume_from"] is None
    assert payload["resumed_from"] is None


# ---------------------------------------------------------------------------
# Resume compatibility
# ---------------------------------------------------------------------------


def write_checkpoint(directory: Path, architecture: str, training_config,
                     epoch: int = 1) -> Path:
    """A real resumable checkpoint, written by the real checkpoint writer."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return save_checkpoint(
        directory / f"{architecture}_last.pt",
        model=model, architecture=architecture, epoch=epoch,
        best_metric=0.9, best_epoch=epoch, optimizer=optimizer,
        config_snapshot=None if training_config is None else asdict(training_config),
    )


def test_compatible_checkpoint_reports_its_position(tmp_path, base_config):
    path = write_checkpoint(tmp_path, "resnet18", base_config.training, epoch=1)
    summary = verify_resume_compatibility(path, "resnet18", base_config)
    assert summary["completed_epochs"] == 2
    assert summary["total_epochs"] == base_config.training.epochs
    assert summary["has_optimizer"] is True
    assert summary["config_recorded"] is True


def test_a_different_epoch_budget_is_refused(tmp_path, base_config):
    """The failure this guard exists for: the same weights, a shorter schedule.

    Resuming across epoch budgets restores the scheduler counters into a
    schedule of a different length. Nothing raises inside PyTorch and the
    learning rate silently leaves the cosine curve, so the run has to be
    stopped here instead.
    """
    shorter = replace(base_config.training, epochs=base_config.training.epochs - 1)
    path = write_checkpoint(tmp_path, "resnet18", shorter)
    with pytest.raises(PreflightError, match="epochs"):
        verify_resume_compatibility(path, "resnet18", base_config)


def test_a_different_learning_rate_is_refused(tmp_path, base_config):
    tweaked = replace(base_config.training,
                      learning_rate=base_config.training.learning_rate * 2)
    path = write_checkpoint(tmp_path, "resnet18", tweaked)
    with pytest.raises(PreflightError, match="learning_rate"):
        verify_resume_compatibility(path, "resnet18", base_config)


def test_a_checkpoint_for_another_architecture_is_refused(tmp_path, base_config):
    path = write_checkpoint(tmp_path, "vgg11", base_config.training)
    with pytest.raises(PreflightError, match="written for"):
        verify_resume_compatibility(path, "resnet18", base_config)


def test_a_checkpoint_without_a_config_snapshot_still_resumes(tmp_path, base_config):
    """Older checkpoints carry no snapshot; they warn rather than block."""
    path = write_checkpoint(tmp_path, "resnet18", None)
    summary = verify_resume_compatibility(path, "resnet18", base_config)
    assert summary["config_recorded"] is False
