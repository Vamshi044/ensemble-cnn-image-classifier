"""Tests for the checkpoint recovery scan.

Recovery runs exactly once, on a machine that may be about to disappear, over
files that cannot be regenerated. The behaviour that matters is therefore not
"does it copy a file" but "can it ever lose one": every test here is about the
cases where a naive copy would destroy work - a later epoch already sitting at
the destination, a mislabelled file, a truncated file that must not abort the
scan before the good files are seen.

The checkpoints are real, written by :func:`src.checkpointing.save_checkpoint`
around a two-parameter model. Nothing here trains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import recover_checkpoints
from recover_checkpoints import best_per_slot, describe, recover, scan

from src.checkpointing import save_checkpoint


def write(directory: Path, architecture: str, kind: str, epoch: int,
          resumable: bool = True) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return save_checkpoint(
        directory / f"{architecture}_{kind}.pt",
        model=model, architecture=architecture, epoch=epoch,
        best_metric=0.9, best_epoch=epoch,
        optimizer=optimizer if resumable else None,
        scheduler=scheduler if resumable else None,
    )


def test_scan_finds_checkpoints_recursively_and_ignores_other_files(tmp_path):
    write(tmp_path / "run" / "checkpoints", "resnet18", "last", epoch=1)
    (tmp_path / "run" / "checkpoints" / "notes.pt").write_bytes(b"")
    (tmp_path / "run" / "checkpoints" / "resnet18_last.txt").write_bytes(b"")

    found = scan([tmp_path])

    assert [p.name for p in found] == ["resnet18_last.pt"]


def test_scan_skips_roots_that_do_not_exist(tmp_path):
    """Most default roots are absent on any given machine; that is not an error."""
    write(tmp_path, "vgg11", "best", epoch=3)

    found = scan([tmp_path / "nowhere", tmp_path, Path("/content/drive/MyDrive")])

    assert [p.name for p in found] == ["vgg11_best.pt"]


def test_describe_reports_the_epoch_and_resumability(tmp_path):
    path = write(tmp_path, "resnet18", "last", epoch=1)

    record = describe(path)

    assert record["readable"] is True
    assert record["architecture"] == "resnet18"
    assert record["epoch"] == 2  # stored zero-based, reported as epochs completed
    assert record["resumable"] is True
    assert record["has_rng"] is True


def test_describe_reports_a_weights_only_checkpoint_as_not_resumable(tmp_path):
    path = write(tmp_path, "resnet18", "best", epoch=1, resumable=False)

    assert describe(path)["resumable"] is False


def test_an_unreadable_file_is_reported_not_raised(tmp_path):
    good = write(tmp_path, "resnet18", "last", epoch=1)
    broken = tmp_path / "vgg11_last.pt"
    broken.write_bytes(b"not a checkpoint")

    records = [describe(p) for p in sorted(scan([tmp_path]))]

    assert [r["readable"] for r in records] == [True, False]
    assert "error" in records[1]
    assert describe(good)["readable"] is True


def test_the_later_epoch_wins_when_the_same_slot_is_found_twice(tmp_path):
    write(tmp_path / "stale", "resnet18", "last", epoch=0)
    newer = write(tmp_path / "fresh", "resnet18", "last", epoch=5)

    chosen = best_per_slot([describe(p) for p in scan([tmp_path])])

    assert chosen[("resnet18", "last")]["path"] == str(newer)
    assert chosen[("resnet18", "last")]["epoch"] == 6


def test_a_mislabelled_file_is_skipped(tmp_path):
    """A file named for one architecture holding another is not recoverable."""
    path = write(tmp_path, "resnet18", "last", epoch=1)
    path.rename(tmp_path / "vgg11_last.pt")

    records = [describe(p) for p in scan([tmp_path])]
    chosen = best_per_slot(records)

    assert chosen == {}
    assert "warning" in records[0]


def test_recover_copies_into_the_destination(tmp_path):
    source = write(tmp_path / "ephemeral", "resnet18", "last", epoch=1)
    destination = tmp_path / "drive"

    actions = recover(best_per_slot([describe(source)]), destination, dry_run=False)

    assert actions[0]["result"] == "copied"
    assert (destination / "resnet18_last.pt").is_file()
    assert describe(destination / "resnet18_last.pt")["epoch"] == 2


def test_recover_never_regresses_a_later_checkpoint(tmp_path):
    """The rule that makes recovery safe to run twice."""
    source = write(tmp_path / "ephemeral", "resnet18", "last", epoch=1)
    destination = tmp_path / "drive"
    write(destination, "resnet18", "last", epoch=7)

    actions = recover(best_per_slot([describe(source)]), destination, dry_run=False)

    assert "kept existing copy at epoch 8" in actions[0]["result"]
    assert describe(destination / "resnet18_last.pt")["epoch"] == 8


def test_recover_replaces_an_earlier_checkpoint(tmp_path):
    source = write(tmp_path / "ephemeral", "resnet18", "last", epoch=6)
    destination = tmp_path / "drive"
    write(destination, "resnet18", "last", epoch=1)

    actions = recover(best_per_slot([describe(source)]), destination, dry_run=False)

    assert actions[0]["result"] == "replaced an earlier copy"
    assert describe(destination / "resnet18_last.pt")["epoch"] == 7


def test_dry_run_copies_nothing(tmp_path):
    source = write(tmp_path / "ephemeral", "resnet18", "last", epoch=1)
    destination = tmp_path / "drive"

    actions = recover(best_per_slot([describe(source)]), destination, dry_run=True)

    assert actions[0]["result"] == "would copy"
    assert not destination.exists()


def test_main_reports_failure_when_nothing_is_found(tmp_path, capsys):
    exit_code = recover_checkpoints.main([
        "--search", str(tmp_path / "empty"),
        "--report", str(tmp_path / "recovery.json"),
        "--dry-run",
    ])

    assert exit_code == 1
    assert "No Stage 3B checkpoint was found" in capsys.readouterr().out


def test_main_recovers_and_writes_a_report(tmp_path):
    write(tmp_path / "ephemeral", "resnet18", "last", epoch=1)
    write(tmp_path / "ephemeral", "resnet18", "best", epoch=1, resumable=False)
    report = tmp_path / "recovery.json"

    exit_code = recover_checkpoints.main([
        "--search", str(tmp_path / "ephemeral"),
        "--destination", str(tmp_path / "drive"),
        "--report", str(report),
    ])

    assert exit_code == 0
    assert (tmp_path / "drive" / "resnet18_last.pt").is_file()
    assert (tmp_path / "drive" / "resnet18_best.pt").is_file()

    import json

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert sorted(payload["selected"]) == ["resnet18_best", "resnet18_last"]
    assert payload["missing"] == [
        "googlenet_best", "googlenet_last", "vgg11_best", "vgg11_last",
    ]
