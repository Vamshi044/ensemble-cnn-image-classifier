"""Stage 5: turn the recorded Stage 3B and Stage 4 artifacts into the final report.

    python scripts/generate_report.py
    python scripts/generate_report.py --results-dir DIR_ON_GOOGLE_DRIVE

This computes nothing about the models it did not already measure. It reads the
JSON and CSV that the training and evaluation stages wrote, derives the
descriptive statistics a report needs - per-class precision, recall and F1, the
most-confused class pairs, and where the ensemble differs from its members -
renders three figures, and writes ``final_report.md``.

Every number in the output is traceable to a file on disk. When an input is
missing the report is not written: an absent artifact is reported as absent
rather than filled in, because a report that quietly omits a model reads
exactly like a report about a model that was never trained.

No model is loaded, no image is classified, and the test set is not read here.
The predictions were produced once, by ``scripts/evaluate_ensemble.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # No display available; render straight to file.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import write_json  # noqa: E402

ARCH_ORDER = ("resnet18", "googlenet", "vgg11")


class MissingArtifact(RuntimeError):
    """Raised when an input the report depends on is not on disk."""


# ---------------------------------------------------------------------------
# Reading what the earlier stages recorded
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise MissingArtifact(f"{path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(results_dir: Path) -> dict:
    """Read every artifact the report needs, or say precisely which are absent."""
    missing: list[str] = []
    runs: dict[str, dict] = {}
    for architecture in ARCH_ORDER:
        path = results_dir / f"stage3b_{architecture}.json"
        if path.is_file():
            runs[architecture] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(str(path))

    evaluation_path = results_dir / "ensemble_evaluation.json"
    predictions_path = results_dir / "ensemble_predictions.csv"
    for path in (evaluation_path, predictions_path):
        if not path.is_file():
            missing.append(str(path))

    if missing:
        raise MissingArtifact(
            "The report cannot be written because these artifacts do not exist:\n  "
            + "\n  ".join(missing)
            + "\n\nRun scripts/run_stage3b.py to completion for every architecture, "
            "then scripts/evaluate_ensemble.py, and try again. Nothing was written."
        )

    return {
        "runs": runs,
        "evaluation": json.loads(evaluation_path.read_text(encoding="utf-8")),
        "predictions": read_predictions(predictions_path),
    }


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Derived statistics
# ---------------------------------------------------------------------------


def per_class_metrics(matrix: np.ndarray, classes: list[str]) -> list[dict]:
    """Precision, recall and F1 per class, straight from the confusion matrix.

    Rows are true labels and columns are predictions, so a row sum is the
    support and a column sum is how often that class was predicted. A class
    that is never predicted has undefined precision; it is reported as 0.0 and
    the ``predicted`` count makes that visible rather than implied.
    """
    rows: list[dict] = []
    for index, name in enumerate(classes):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        rows.append({"class": name, "support": support, "predicted": predicted,
                     "correct": true_positive, "precision": precision,
                     "recall": recall, "f1": f1})
    return rows


def most_confused(matrix: np.ndarray, classes: list[str], limit: int = 8) -> list[dict]:
    """The largest off-diagonal cells: what the ensemble actually mixes up."""
    pairs = [
        {"true": classes[i], "predicted": classes[j], "count": int(matrix[i, j]),
         "share_of_class": float(matrix[i, j] / matrix[i].sum())
         if matrix[i].sum() else 0.0}
        for i in range(len(classes)) for j in range(len(classes)) if i != j
    ]
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return [p for p in pairs[:limit] if p["count"] > 0]


def ensemble_versus_members(predictions: list[dict],
                            architectures: list[str]) -> dict:
    """Where averaging changed the answer.

    The interesting cases are the ones a single model cannot show: images the
    ensemble gets right that a minority of its members got right, and images it
    gets wrong that some member got right. Both are counted from the recorded
    predictions, not re-derived from the models.
    """
    member_columns = [f"{a}_pred" for a in architectures]
    totals = {
        "images": len(predictions),
        "ensemble_correct": 0,
        "all_members_correct": 0,
        "no_member_correct": 0,
        "ensemble_correct_when_member_minority_correct": 0,
        "ensemble_wrong_when_some_member_correct": 0,
        "unanimous_members": 0,
    }
    member_correct = Counter()

    for row in predictions:
        truth = int(row["true_label"])
        ensemble_hit = int(row["ensemble_pred"]) == truth
        member_hits = [int(row[column]) == truth for column in member_columns]
        hits = sum(member_hits)

        totals["ensemble_correct"] += ensemble_hit
        totals["all_members_correct"] += hits == len(member_columns)
        totals["no_member_correct"] += hits == 0
        totals["unanimous_members"] += len({row[c] for c in member_columns}) == 1
        if ensemble_hit and 0 < hits <= len(member_columns) // 2:
            totals["ensemble_correct_when_member_minority_correct"] += 1
        if not ensemble_hit and hits > 0:
            totals["ensemble_wrong_when_some_member_correct"] += 1
        for architecture, hit in zip(architectures, member_hits, strict=True):
            member_correct[architecture] += hit

    totals["member_correct"] = dict(member_correct)
    return totals


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_confusion(matrix: np.ndarray, classes: list[str], path: Path) -> Path:
    figure, axes = plt.subplots(figsize=(7.5, 6.5))
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    image = axes.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axes.set_yticks(range(len(classes)), classes)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title("Ensemble confusion matrix (row-normalised)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                      fontsize=7,
                      color="white" if normalised[i, j] > 0.5 else "black")
    figure.colorbar(image, ax=axes, fraction=0.046, label="share of true class")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_comparison(evaluation: dict, runs: dict, path: Path) -> Path:
    architectures = [m["architecture"] for m in evaluation["per_model"]]
    validation = [runs[a].get("best_val_accuracy") or 0.0 for a in architectures]
    test = [m["test_accuracy"] for m in evaluation["per_model"]]

    labels = [*architectures, "ensemble"]
    validation = [*validation, float("nan")]  # the ensemble has no validation figure
    test = [*test, evaluation["ensemble_test_accuracy"]]

    positions = np.arange(len(labels))
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.bar(positions - 0.2, validation, 0.4, label="best validation accuracy")
    axes.bar(positions + 0.2, test, 0.4, label="test accuracy")
    for x, value in zip(positions + 0.2, test, strict=True):
        axes.text(x, value + 0.01, f"{value:.4f}", ha="center", fontsize=8)
    axes.set_xticks(positions, labels)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("accuracy")
    axes.set_title("Per-model and ensemble accuracy")
    axes.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_training_curves(runs: dict, path: Path) -> Path:
    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(11, 4.2))
    for architecture, run in runs.items():
        records = run.get("records", [])
        if not records:
            continue
        epochs = [r["epoch"] + 1 for r in records]
        loss_axes.plot(epochs, [r["train_loss"] for r in records],
                       label=f"{architecture} train")
        loss_axes.plot(epochs, [r["val_loss"] for r in records], linestyle="--",
                       label=f"{architecture} val")
        accuracy_axes.plot(epochs, [r["val_accuracy"] for r in records],
                           label=architecture)
    loss_axes.set_xlabel("epoch")
    loss_axes.set_ylabel("loss")
    loss_axes.set_title("Loss")
    loss_axes.legend(fontsize=7)
    accuracy_axes.set_xlabel("epoch")
    accuracy_axes.set_ylabel("validation accuracy")
    accuracy_axes.set_title("Validation accuracy")
    accuracy_axes.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


def render_markdown(data: dict) -> str:
    evaluation = data["evaluation"]
    runs = data["runs"]
    lines: list[str] = []
    add = lines.append

    add("# Ensemble CNN Image Classifier - Final Report")
    add("")
    add(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        "from recorded artifacts. Every figure below is read from a file "
        "written by an earlier stage; nothing here is re-estimated.")
    add("")
    add("## Configuration")
    add("")
    add(f"- Dataset: {evaluation['dataset']}, "
        f"{evaluation['test_size']} official test images")
    add(f"- Split checksum: `{evaluation['split_checksum']}`")
    add(f"- Input resolution: {evaluation['image_size']}x{evaluation['image_size']}")
    add(f"- Fusion: {evaluation['fusion']}")
    add(f"- Fusion weights: {evaluation['weights']}")
    add(f"- Evaluation device: {evaluation['device']}, "
        f"AMP during inference: {evaluation['amp_used_for_inference']}")
    add("")

    add("## Results")
    add("")
    add("| model | epochs | best validation accuracy | test accuracy | parameters |")
    add("|---|---|---|---|---|")
    for member in evaluation["per_model"]:
        architecture = member["architecture"]
        run = runs.get(architecture, {})
        best = run.get("best_val_accuracy")
        add(f"| {architecture} | {run.get('completed_epochs', '-')} | "
            f"{'-' if best is None else f'{best:.4f}'} | "
            f"{member['test_accuracy']:.4f} | {member['parameters']:,} |")
    add(f"| **ensemble** | - | - | **"
        f"{evaluation['ensemble_test_accuracy']:.4f}** | - |")
    add("")

    best_member = max(evaluation["per_model"], key=lambda m: m["test_accuracy"])
    delta = evaluation["ensemble_test_accuracy"] - best_member["test_accuracy"]
    direction = "above" if delta >= 0 else "below"
    add(f"The ensemble is {abs(delta):.4f} {direction} the best single model "
        f"({best_member['architecture']}, {best_member['test_accuracy']:.4f}). "
        "The fusion weights are fixed at equal thirds and were not searched, so "
        "this difference is a property of averaging, not of tuning.")
    add("")

    add("## Per-class performance (ensemble)")
    add("")
    add("| class | support | precision | recall | F1 |")
    add("|---|---|---|---|---|")
    for row in data["per_class"]:
        add(f"| {row['class']} | {row['support']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} |")
    macro = sum(r["f1"] for r in data["per_class"]) / len(data["per_class"])
    add("")
    add(f"Macro-averaged F1: **{macro:.4f}**")
    add("")

    add("## Error analysis")
    add("")
    add("Most frequent confusions, as true class -> predicted class:")
    add("")
    add("| true | predicted | count | share of that class |")
    add("|---|---|---|---|")
    for pair in data["confusions"]:
        add(f"| {pair['true']} | {pair['predicted']} | {pair['count']} | "
            f"{pair['share_of_class']:.3f} |")
    add("")

    agreement = data["agreement"]
    add("Agreement between the ensemble and its members:")
    add("")
    add(f"- All three members correct: {agreement['all_members_correct']} images")
    add(f"- No member correct: {agreement['no_member_correct']} images")
    add("- Members unanimous (right or wrong): "
        f"{agreement['unanimous_members']} images")
    add("- Ensemble correct where only a minority of members were: "
        f"{agreement['ensemble_correct_when_member_minority_correct']} images")
    add("- Ensemble wrong where at least one member was right: "
        f"{agreement['ensemble_wrong_when_some_member_correct']} images")
    add("")

    add("## Provenance")
    add("")
    for architecture, run in runs.items():
        add(f"- {architecture}: split `{run.get('split_checksum')}`, "
            f"device `{run.get('device', {}).get('device')}`, "
            f"AMP {run.get('amp', {}).get('enabled')}, "
            f"git `{run.get('git_head', '')[:12]}`")
    add(f"- Checkpoints evaluated: {evaluation['checkpoints']}")
    add("")
    add("## Test-set discipline")
    add("")
    for note in evaluation.get("notes", []):
        add(f"- {note}")
    add("")
    add("## Figures")
    add("")
    for name, path in data["figures"].items():
        add(f"- {name}: `{Path(path).name}`")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=PROJECT_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=None,
                        help="defaults to <results-dir>/figures")
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    figures_dir = args.figures_dir or results_dir / "figures"

    try:
        data = load_inputs(results_dir)
    except MissingArtifact as exc:
        print(f"STOPPED: {exc}")
        return 2

    evaluation = data["evaluation"]
    confusion = evaluation.get("ensemble_confusion_matrix")
    if not confusion:
        print("STOPPED: ensemble_evaluation.json has no confusion matrix. "
              "Re-run scripts/evaluate_ensemble.py with the current code.")
        return 2

    classes = list(confusion["classes"])
    matrix = np.asarray(confusion["matrix"], dtype=np.int64)

    data["per_class"] = per_class_metrics(matrix, classes)
    data["confusions"] = most_confused(matrix, classes)
    data["agreement"] = ensemble_versus_members(
        data["predictions"], list(evaluation["architectures"])
    )
    data["figures"] = {
        "confusion matrix": str(
            plot_confusion(matrix, classes, figures_dir / "confusion_matrix.png")),
        "model comparison": str(
            plot_comparison(evaluation, data["runs"],
                            figures_dir / "model_comparison.png")),
        "training curves": str(
            plot_training_curves(data["runs"],
                                 figures_dir / "training_curves.png")),
    }

    report_path = results_dir / "final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(data), encoding="utf-8")

    summary_path = write_json(
        {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split_checksum": evaluation["split_checksum"],
            "test_size": evaluation["test_size"],
            "individual_test_accuracy": evaluation["individual_test_accuracy"],
            "ensemble_test_accuracy": evaluation["ensemble_test_accuracy"],
            "best_validation_accuracy": {
                a: run.get("best_val_accuracy") for a, run in data["runs"].items()
            },
            "per_class": data["per_class"],
            "macro_f1": sum(r["f1"] for r in data["per_class"]) / len(classes),
            "most_confused": data["confusions"],
            "ensemble_versus_members": data["agreement"],
            "figures": data["figures"],
        },
        results_dir / "final_report.json",
    )

    print(f"wrote {report_path}")
    print(f"wrote {summary_path}")
    for name, path in data["figures"].items():
        print(f"wrote {path}  ({name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
