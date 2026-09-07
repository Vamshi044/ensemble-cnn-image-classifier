"""Tests for the Stage 5 report generator.

The report is the artifact a reader will quote, so the risk it carries is not
crashing - it is stating something the underlying files do not support. These
tests pin the two behaviours that guard against that: the derived statistics are
computed correctly from a hand-checked confusion matrix, and a missing input
stops the report rather than producing one with a model quietly left out.

All fixtures here are synthetic. The accuracies in them are arbitrary numbers
chosen to make the arithmetic checkable, and describe no model.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_report
from generate_report import (
    MissingArtifact,
    ensemble_versus_members,
    load_inputs,
    most_confused,
    per_class_metrics,
)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
ARCHITECTURES = ["resnet18", "googlenet", "vgg11"]


def make_matrix() -> np.ndarray:
    """A matrix small enough to verify by hand.

    Class 0: 8 of 10 correct, 2 called class 1.
    Class 1: 6 of 10 correct, 4 called class 0.
    Every other class is perfect, so precision and recall for them are 1.0.
    """
    matrix = np.zeros((10, 10), dtype=np.int64)
    for index in range(10):
        matrix[index, index] = 10
    matrix[0, 0], matrix[0, 1] = 8, 2
    matrix[1, 1], matrix[1, 0] = 6, 4
    return matrix


def test_per_class_metrics_match_hand_computed_values():
    rows = per_class_metrics(make_matrix(), CLASSES)

    airplane = rows[0]
    assert airplane["support"] == 10
    assert airplane["predicted"] == 12          # 8 true + 4 from automobile
    assert airplane["precision"] == pytest.approx(8 / 12)
    assert airplane["recall"] == pytest.approx(0.8)
    assert airplane["f1"] == pytest.approx(2 * (8 / 12) * 0.8 / ((8 / 12) + 0.8))

    automobile = rows[1]
    assert automobile["precision"] == pytest.approx(6 / 8)
    assert automobile["recall"] == pytest.approx(0.6)

    assert rows[2]["precision"] == 1.0
    assert rows[2]["f1"] == 1.0


def test_a_class_that_is_never_predicted_scores_zero_not_nan():
    """Division by an empty column must not put a NaN into the report."""
    matrix = np.zeros((10, 10), dtype=np.int64)
    for index in range(10):
        matrix[index, (index + 1) % 10] = 10   # every class predicted as the next

    rows = per_class_metrics(matrix, CLASSES)

    assert all(row["correct"] == 0 for row in rows)
    assert all(row["precision"] == 0.0 for row in rows)
    assert all(row["f1"] == 0.0 for row in rows)


def test_most_confused_ranks_the_largest_off_diagonal_cells():
    pairs = most_confused(make_matrix(), CLASSES, limit=3)

    assert pairs[0] == {"true": "automobile", "predicted": "airplane",
                        "count": 4, "share_of_class": pytest.approx(0.4)}
    assert pairs[1]["count"] == 2
    assert all(pair["true"] != pair["predicted"] for pair in pairs)


def test_most_confused_omits_empty_cells():
    perfect = np.diag(np.full(10, 10, dtype=np.int64))

    assert most_confused(perfect, CLASSES) == []


def rows_for(cases: list[tuple[int, int, list[int]]]) -> list[dict]:
    """Build prediction rows: (truth, ensemble prediction, member predictions)."""
    return [
        {"true_label": str(truth), "ensemble_pred": str(ensemble),
         **{f"{a}_pred": str(p)
            for a, p in zip(ARCHITECTURES, members, strict=True)}}
        for truth, ensemble, members in cases
    ]


def test_ensemble_versus_members_counts_the_interesting_cases():
    predictions = rows_for([
        (3, 3, [3, 3, 3]),   # everyone right
        (3, 3, [3, 5, 5]),   # ensemble right, only a minority of members right
        (3, 5, [3, 5, 5]),   # ensemble wrong although a member was right
        (3, 5, [5, 5, 5]),   # everyone wrong, unanimous
    ])

    totals = ensemble_versus_members(predictions, ARCHITECTURES)

    assert totals["images"] == 4
    assert totals["ensemble_correct"] == 2
    assert totals["all_members_correct"] == 1
    assert totals["no_member_correct"] == 1
    assert totals["unanimous_members"] == 2
    assert totals["ensemble_correct_when_member_minority_correct"] == 1
    assert totals["ensemble_wrong_when_some_member_correct"] == 1
    assert totals["member_correct"] == {"resnet18": 3, "googlenet": 1, "vgg11": 1}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def write_artifacts(results_dir: Path, architectures=ARCHITECTURES) -> np.ndarray:
    """A complete, synthetic set of Stage 3B and Stage 4 artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)
    matrix = make_matrix()

    for index, architecture in enumerate(architectures):
        records = [
            {"architecture": architecture, "epoch": epoch,
             "train_loss": 1.0 - 0.05 * epoch, "train_accuracy": 0.5 + 0.02 * epoch,
             "val_loss": 1.1 - 0.05 * epoch, "val_accuracy": 0.4 + 0.02 * epoch,
             "learning_rates": {"backbone": 0.005, "head": 0.05},
             "epoch_seconds": 60.0, "best_val_accuracy": 0.4 + 0.02 * epoch,
             "is_best": True, "train_samples": 45000, "val_samples": 5000}
            for epoch in range(10)
        ]
        (results_dir / f"stage3b_{architecture}.json").write_text(json.dumps({
            "architecture": architecture, "records": records,
            "best_val_accuracy": 0.58 + 0.01 * index, "best_epoch": 9,
            "completed_epochs": 10, "split_checksum": "bdb035810af794a7",
            "device": {"device": "cuda:0"}, "amp": {"enabled": True},
            "git_head": "0123456789abcdef",
        }), encoding="utf-8")

    (results_dir / "ensemble_evaluation.json").write_text(json.dumps({
        "dataset": "CIFAR-10", "test_size": 100,
        "split_checksum": "bdb035810af794a7", "image_size": 128,
        "device": "cuda:0", "amp_used_for_inference": False,
        "architectures": architectures,
        "checkpoints": {a: f"{a}_best.pt" for a in architectures},
        "fusion": "equal_weight_softmax_probability_average",
        "weights": dict.fromkeys(architectures, 1 / 3),
        "ensemble_test_accuracy": 0.94, "ensemble_test_correct": 94,
        "individual_test_accuracy": dict.fromkeys(architectures, 0.9),
        "per_model": [
            {"architecture": a, "test_accuracy": 0.90 + 0.01 * i,
             "parameters": 11_000_000, "checkpoint": f"{a}_best.pt"}
            for i, a in enumerate(architectures)
        ],
        "ensemble_confusion_matrix": {
            "classes": CLASSES, "rows": "true label",
            "columns": "ensemble prediction", "matrix": matrix.tolist(),
        },
        "notes": ["The official test set was consumed once."],
    }), encoding="utf-8")

    with (results_dir / "ensemble_predictions.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "true_label", "true_class", "ensemble_pred",
                         "ensemble_confidence",
                         *(f"{a}_pred" for a in architectures)])
        for i in range(10):
            writer.writerow([i, 3, "cat", 3, 0.9, 3, 3, 3])
    return matrix


def test_report_is_written_with_figures(tmp_path):
    write_artifacts(tmp_path)

    exit_code = generate_report.main(["--results-dir", str(tmp_path)])

    assert exit_code == 0
    report = (tmp_path / "final_report.md").read_text(encoding="utf-8")
    assert "0.9400" in report                      # the ensemble accuracy
    assert "bdb035810af794a7" in report
    assert "automobile | airplane | 4" in report   # the top confusion
    for name in ("confusion_matrix.png", "model_comparison.png",
                 "training_curves.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 0

    summary = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
    assert summary["ensemble_test_accuracy"] == 0.94
    assert summary["macro_f1"] == pytest.approx(
        sum(r["f1"] for r in per_class_metrics(make_matrix(), CLASSES)) / 10
    )


def test_a_missing_training_result_stops_the_report(tmp_path):
    write_artifacts(tmp_path)
    (tmp_path / "stage3b_vgg11.json").unlink()

    exit_code = generate_report.main(["--results-dir", str(tmp_path)])

    assert exit_code == 2
    assert not (tmp_path / "final_report.md").exists()


def test_load_inputs_names_every_missing_artifact(tmp_path):
    with pytest.raises(MissingArtifact) as excinfo:
        load_inputs(tmp_path)

    message = str(excinfo.value)
    for architecture in ARCHITECTURES:
        assert f"stage3b_{architecture}.json" in message
    assert "ensemble_evaluation.json" in message
    assert "ensemble_predictions.csv" in message


def test_an_evaluation_without_a_confusion_matrix_stops_the_report(tmp_path):
    write_artifacts(tmp_path)
    path = tmp_path / "ensemble_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["ensemble_confusion_matrix"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert generate_report.main(["--results-dir", str(tmp_path)]) == 2
    assert not (tmp_path / "final_report.md").exists()
