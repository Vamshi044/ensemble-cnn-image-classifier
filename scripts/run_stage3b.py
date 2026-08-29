"""Stage 3B orchestrator: the three controlled real-data baseline runs.

    python scripts/run_stage3b.py --dry-run    # pre-flight gates only, no training
    python scripts/run_stage3b.py              # pre-flight, then train all three

This trains the three approved architectures on the approved CIFAR-10
train/validation split, one at a time, in the order resnet18 -> googlenet ->
vgg11. It adds no methodology of its own: every hyperparameter comes from
``configs/config.yaml`` and the training itself is ``src.training.fit``.

APPROVED PHASE CONFIGURATION (Project Lead, Stage 3B): single-phase full
fine-tune - ``freeze_backbone: false``, ``epochs: 10``. The two-phase
frozen-then-unfrozen experiment is deliberately NOT run here; it is a separate
experiment whose phase durations are not specified in the configuration, and
inventing them would change the experiment.

THE TEST SET IS NOT REACHABLE FROM THIS SCRIPT. ``build_dataloaders`` returns
three loaders and the third is discarded on the line it is unpacked;
``src.training.fit`` has no test-loader parameter. Nothing here computes,
prints or stores an accuracy for anything but train and validation.

Pre-flight gates run before any training starts, and a failure stops the run
rather than training under a configuration nobody verified.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.checkpointing import (  # noqa: E402
    checkpoint_path,
    load_checkpoint,
    restore_training_state,
)
from src.config import load_config  # noqa: E402
from src.dataset import build_dataloaders, build_datasets  # noqa: E402
from src.models import build_model  # noqa: E402
from src.seed import set_global_seeds  # noqa: E402
from src.training import (  # noqa: E402
    build_optimizer,
    build_scheduler,
    describe_device,
    fit,
    resolve_amp,
)
from src.utils import collect_environment, select_device, write_json  # noqa: E402

# ResNet18 first: it is the lightest of the three, so it validates the whole
# real-data path end to end before hours are committed to the larger models.
RUN_ORDER = ("resnet18", "googlenet", "vgg11")

EXPECTED_SPLIT_CHECKSUM = "bdb035810af794a7"


class PreflightError(RuntimeError):
    """Raised when a gate fails. Training must not start."""


def gate(checks: list[tuple[str, bool, str]], name: str, ok: bool, detail: str = "") -> bool:
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def preflight(config, device, splits, require_cuda: bool = True) -> dict:
    """Run the Stage 3B pre-flight gates. Raises PreflightError on any failure."""
    print("=" * 72)
    print("STAGE 3B PRE-FLIGHT")
    print("=" * 72)
    checks: list[tuple[str, bool, str]] = []

    # 1-2. Repository identity and cleanliness.
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise PreflightError(f"git is not usable here: {exc}") from exc

    gate(checks, "git working tree is clean", status == "",
         status.replace("\n", "; ") if status else "")
    gate(checks, "repository is the expected project",
         "ensemble-cnn-image-classifier" in remote, remote)
    print(f"         HEAD {head}")

    # 3. Split checksum.
    checksum = splits.indices.checksum()
    gate(checks, f"split checksum == {EXPECTED_SPLIT_CHECKSUM}",
         checksum == EXPECTED_SPLIT_CHECKSUM, checksum)
    gate(checks, "train/val sizes match the approved split",
         len(splits.train) == config.data.train_size
         and len(splits.val) == config.data.val_size,
         f"{len(splits.train)} / {len(splits.val)}")

    # 6. CUDA / GPU.
    info = describe_device(device)
    cuda_ok = info.type == "cuda"
    if require_cuda:
        gate(checks, "training device is CUDA", cuda_ok,
             f"{info.device} ({info.gpu_name or info.type})")
    else:
        # --allow-cpu was passed. Say so plainly rather than printing PASS next
        # to a device the approved configuration does not use.
        print(f"  [ -- ] training device is {info.device!r}, NOT the approved CUDA "
              "path (--allow-cpu)")
    if cuda_ok:
        print(f"         GPU {info.gpu_name}, capability {info.capability}, "
              f"{info.total_memory_gb:.2f} GB")

    # 7. AMP resolves to float16.
    amp = resolve_amp(config, torch.device(info.device))
    if require_cuda:
        gate(checks, "AMP resolves to enabled float16",
             amp.enabled and amp.dtype is torch.float16, amp.reason)
    else:
        print(f"  [ -- ] AMP: {amp.reason}")

    # 5. Approved model settings.
    gate(checks, "pretrained ImageNet weights enabled", bool(config.model.pretrained))
    gate(checks, "GoogLeNet aux_logits is False",
         config.model.googlenet_aux_logits is False)
    gate(checks, "image resolution is the approved 128", config.image.size == 128,
         str(config.image.size))
    gate(checks, "label smoothing is 0.0", config.training.label_smoothing == 0.0,
         str(config.training.label_smoothing))

    # Approved single-phase configuration.
    gate(checks, "phase configuration is the approved single-phase run",
         config.training.freeze_backbone is False,
         f"freeze_backbone={config.training.freeze_backbone}")

    # 8. No test-loader iteration is possible.
    import inspect
    fit_params = set(inspect.signature(fit).parameters)
    gate(checks, "fit() has no test-loader parameter",
         not any("test" in p for p in fit_params))

    # 9. Final configuration.
    print("\n" + "-" * 72)
    print("FINAL TRAINING CONFIGURATION")
    print("-" * 72)
    for key, value in asdict(config.training).items():
        print(f"  {key:<32} {value}")
    print(f"  {'image_size':<32} {config.image.size}")
    print(f"  {'batch_size':<32} {config.dataloader.batch_size}")
    print(f"  {'eval_batch_size':<32} {config.dataloader.eval_batch_size}")
    print(f"  {'seed':<32} {config.reproducibility.seed}")
    print(f"  {'split_seed':<32} {config.reproducibility.split_seed}")
    print(f"  {'split_checksum':<32} {checksum}")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        raise PreflightError(
            "Pre-flight failed, so no training was started. Failing gates: "
            + "; ".join(failed)
        )
    print("\n  all pre-flight gates passed")
    return {
        "git_head": head, "git_clean": status == "", "remote": remote,
        "split_checksum": checksum, "device": info.as_dict(),
        "amp": {"enabled": amp.enabled, "dtype": str(amp.dtype), "reason": amp.reason},
        "checks": [{"name": n, "passed": o, "detail": d} for n, o, d in checks],
    }


def verify_checkpoints(directory: Path, architecture: str, config, epochs: int) -> dict:
    """Verify LAST resumes and BEST is weights-only. Never touches the test set."""
    print(f"\n  verifying checkpoints for {architecture}")
    last_path = checkpoint_path(directory, architecture, "last")
    best_path = checkpoint_path(directory, architecture, "best")
    result: dict = {
        "last_path": str(last_path), "best_path": str(best_path),
        "last_exists": last_path.is_file(), "best_exists": best_path.is_file(),
    }
    if not last_path.is_file():
        print("    LAST checkpoint missing - cannot verify resume")
        return result

    result["last_bytes"] = last_path.stat().st_size
    last = load_checkpoint(last_path, map_location="cpu")
    result["last_architecture"] = last.get("architecture")
    result["last_has_optimizer"] = "optimizer_state" in last
    result["last_has_scheduler"] = "scheduler_state" in last
    result["last_has_scaler"] = "scaler_state" in last
    result["last_has_rng"] = "rng_state" in last

    # A real resume: rebuild the pieces and restore into them.
    model = build_model(architecture, num_classes=config.data.num_classes,
                        pretrained=False, aux_logits=config.model.googlenet_aux_logits)
    optimizer = build_optimizer(model, architecture, config)
    scheduler = build_scheduler(optimizer, config, epochs)
    state = restore_training_state(
        last, model=model, architecture=architecture,
        optimizer=optimizer, scheduler=scheduler,
    )
    result["resume_restored"] = list(state.restored)
    result["resume_skipped"] = list(state.skipped)
    result["resume_start_epoch"] = state.start_epoch
    print(f"    LAST  restored {state.restored}, start_epoch {state.start_epoch}")

    if best_path.is_file():
        result["best_bytes"] = best_path.stat().st_size
        best = load_checkpoint(best_path, map_location="cpu")
        result["best_architecture"] = best.get("architecture")
        result["best_is_weights_only"] = "optimizer_state" not in best
        result["best_epoch"] = best.get("best_epoch")
        print(f"    BEST  weights-only={result['best_is_weights_only']}, "
              f"{result['best_bytes'] / 1024**2:.0f} MB "
              f"(LAST {result['last_bytes'] / 1024**2:.0f} MB)")
    return result


def quality_control(history) -> list[str]:
    """Look for the failure modes the Project Lead listed. Returns problems found."""
    problems: list[str] = []
    for record in history.records:
        for field in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            value = getattr(record, field)
            if not math.isfinite(value):
                problems.append(f"epoch {record.epoch}: {field} is {value}")
        for name, lr in record.learning_rates.items():
            if not math.isfinite(lr) or lr < 0:
                problems.append(f"epoch {record.epoch}: lr_{name} is {lr}")
    return problems


def run_one(architecture: str, config, device, train_loader, val_loader,
            results_dir: Path, checkpoints_dir: Path, preflight_record: dict) -> dict:
    """Train one architecture and verify what it produced."""
    print("\n" + "=" * 72)
    print(f"TRAINING {architecture.upper()}")
    print("=" * 72)

    # Every run starts from the same seed, so the three are comparable.
    set_global_seeds(config.reproducibility.seed,
                     deterministic=config.reproducibility.deterministic)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = build_model(
        architecture, num_classes=config.data.num_classes,
        pretrained=config.model.pretrained,
        aux_logits=config.model.googlenet_aux_logits,
    )

    started = time.time()
    history = fit(
        model, architecture=architecture, train_loader=train_loader,
        val_loader=val_loader, config=config, device=device,
        checkpoint_dir=checkpoints_dir,
    )
    duration = time.time() - started

    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else None)

    problems = quality_control(history)
    if problems:
        print("\n  QUALITY CONTROL PROBLEMS:")
        for p in problems:
            print(f"    {p}")

    checkpoints = verify_checkpoints(
        checkpoints_dir, architecture, config, config.training.epochs
    )

    payload = history.as_dict()
    payload.update(
        configuration={
            "training": asdict(config.training),
            "image_size": config.image.size,
            "batch_size": config.dataloader.batch_size,
            "eval_batch_size": config.dataloader.eval_batch_size,
            "seed": config.reproducibility.seed,
            "split_seed": config.reproducibility.split_seed,
            "deterministic": config.reproducibility.deterministic,
            "phase": "single-phase full fine-tune (freeze_backbone=false)",
        },
        split_checksum=preflight_record["split_checksum"],
        device=preflight_record["device"],
        amp=preflight_record["amp"],
        git_head=preflight_record["git_head"],
        environment=collect_environment(),
        platform={"python": platform.python_version(),
                  "torch": torch.__version__,
                  "cuda": torch.version.cuda},
        duration_seconds=duration,
        peak_gpu_memory_mb=peak_mb,
        checkpoints=checkpoints,
        quality_control_problems=problems,
        note="Validation metrics only. The official test set was not read.",
    )
    json_path = write_json(payload, results_dir / f"stage3b_{architecture}.json")

    rows = history.as_rows()
    if rows:
        csv_path = results_dir / f"stage3b_{architecture}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n  duration {duration / 60:.1f} min"
          + (f", peak GPU memory {peak_mb:.0f} MB" if peak_mb else ""))
    if history.best_epoch is not None:
        print(f"  best VALIDATION accuracy {history.best_val_accuracy:.4f} "
              f"at epoch {history.best_epoch + 1}")
    print(f"  results {json_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="run the pre-flight gates and stop without training")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--architectures", nargs="+", default=list(RUN_ORDER),
                        help="subset to train, in the given order")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="permit a non-CUDA device; the approved run is CUDA+fp16")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    set_global_seeds(config.reproducibility.seed,
                     deterministic=config.reproducibility.deterministic)
    device = select_device()

    splits = build_datasets(config)
    # Three loaders come back; the test loader is discarded on this line and is
    # never bound to a name any training code can reach.
    train_loader, val_loader, _discarded_test_loader = build_dataloaders(config, splits)
    del _discarded_test_loader

    try:
        record = preflight(config, device, splits, require_cuda=not args.allow_cpu)
    except PreflightError as exc:
        print(f"\nSTOPPED: {exc}")
        return 2

    if args.dry_run:
        print("\n--dry-run: configuration verified, no training was started.")
        return 0

    results_dir = Path(config.paths.results_dir)
    checkpoints_dir = Path(config.paths.checkpoints_dir)
    summaries = []
    for architecture in args.architectures:
        summaries.append(
            run_one(architecture, config, device, train_loader, val_loader,
                    results_dir, checkpoints_dir, record)
        )

    print("\n" + "=" * 72)
    print("STAGE 3B SUMMARY - VALIDATION ONLY")
    print("=" * 72)
    print(f"  {'architecture':<14}{'best val acc':>14}{'best epoch':>12}{'minutes':>10}")
    for s in summaries:
        best = s.get("best_val_accuracy")
        epoch = s.get("best_epoch")
        print(f"  {s['architecture']:<14}"
              f"{best if best is None else f'{best:.4f}':>14}"
              f"{'-' if epoch is None else epoch + 1:>12}"
              f"{s['duration_seconds'] / 60:>10.1f}")
    write_json({"runs": summaries}, results_dir / "stage3b_summary.json")
    print("\n  These are VALIDATION figures. The official test set was not read,")
    print("  and no model is claimed 'best overall' on this basis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
