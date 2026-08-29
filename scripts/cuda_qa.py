"""Stage 3A.1 CUDA QA harness - Google Colab Tesla T4.

Runs the CUDA verification the Project Lead specified for Stage 3A.1, sections 1
through 9, and prints a report that can be pasted back verbatim.

SCOPE - this script does NOT train anything.

    * Every tensor it feeds a model is ``torch.randn`` noise with random labels.
    * CIFAR-10 is never downloaded, opened, or referenced.
    * No accuracy, F1 or loss value produced here describes a classifier. The
      losses printed are noise-on-noise and are reported only as evidence that
      the plumbing runs and stays finite.
    * The test split does not appear anywhere in this file.

It is an infrastructure smoke test. "PASS" means the machinery executed, not
that any model learned anything.

Usage::

    python scripts/cuda_qa.py             # sections 1 and 3-9
    python scripts/cuda_qa.py --pytest    # also run the full suite (section 2)

Exits non-zero if any check fails, so a failure cannot be missed by skimming.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.checkpointing import (  # noqa: E402
    checkpoint_path,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from src.config import load_config  # noqa: E402
from src.models import build_model, build_param_groups  # noqa: E402
from src.training import (  # noqa: E402
    build_criterion,
    build_optimizer,
    build_scaler,
    build_scheduler,
    current_learning_rates,
    describe_device,
    fit,
    resolve_amp,
    train_one_epoch,
)

ARCHITECTURES = ("googlenet", "resnet18", "vgg11")

# Small on purpose. This is a plumbing test, and a large batch would only make
# the VGG11 memory reading less informative about the model itself.
BATCH = 8
NUM_CLASSES = 10

_RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    _RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def header(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def synthetic_batch(
    size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random noise and random labels. Never CIFAR-10."""
    images = torch.randn(BATCH, 3, size, size, device=device)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,), device=device)
    return images, targets


def synthetic_loader(size: int, n: int, batch: int = BATCH) -> DataLoader:
    dataset = TensorDataset(
        torch.randn(n, 3, size, size), torch.randint(0, NUM_CLASSES, (n,))
    )
    return DataLoader(dataset, batch_size=batch, shuffle=False)


# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------


def section_1_environment() -> torch.device:
    header("1. ENVIRONMENT VERIFICATION")
    import torchvision

    print(f"  python              : {sys.version.split()[0]}")
    print(f"  torch               : {torch.__version__}")
    print(f"  torchvision         : {torchvision.__version__}")
    print(f"  cuda available      : {torch.cuda.is_available()}")
    print(f"  cuda (torch build)  : {torch.version.cuda}")
    print(f"  cudnn               : {torch.backends.cudnn.version()}")
    print(f"  gpu count           : {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        record("CUDA available", False,
               "no CUDA device - this is not the T4 environment")
        print("\n  STOPPING: sections 3-9 require CUDA and were NOT run.")
        summarise()
        raise SystemExit(2)

    device = torch.device("cuda")
    info = describe_device(device)
    props = torch.cuda.get_device_properties(device.index or 0)
    print(f"  gpu name            : {info.gpu_name}")
    print(f"  compute capability  : {info.capability}")
    print(f"  total VRAM          : {info.total_memory_gb:.2f} GB")
    print(f"  free VRAM           : {info.free_memory_gb}")
    print(f"  multiprocessors     : {props.multi_processor_count}")
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    print(f"  memory allocated now: {allocated:.1f} MB")
    print(f"  memory reserved now : {reserved:.1f} MB")
    print(f"  bf16 native support : {bf16}")

    record("CUDA available", True, info.gpu_name or "")
    is_t4 = "T4" in (info.gpu_name or "")
    record(
        "GPU is the expected Tesla T4",
        is_t4,
        f"reported {info.gpu_name!r}" if not is_t4 else "",
    )
    return device


# ---------------------------------------------------------------------------
# 2. Full test suite
# ---------------------------------------------------------------------------


def section_2_pytest() -> None:
    header("2. FULL TEST SUITE ON THE T4")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "-rs"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    print(proc.stdout[-8000:])
    if proc.stderr.strip():
        print("--- stderr ---")
        print(proc.stderr[-2000:])
    record("pytest suite", proc.returncode == 0, f"exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# 3. All three models on CUDA
# ---------------------------------------------------------------------------


def section_3_models(device: torch.device, size: int) -> None:
    header("3. ALL THREE MODELS ON CUDA (synthetic data)")
    for arch in ARCHITECTURES:
        print(f"\n-- {arch} --")
        try:
            torch.cuda.reset_peak_memory_stats(device)
            # 1. pretrained ImageNet weights, 2. 10-class head (build_model does both)
            # 3. move to CUDA
            model = build_model(arch, NUM_CLASSES, pretrained=True).to(device)
            record(f"{arch}: constructed pretrained + moved to CUDA", True)

            images, targets = synthetic_batch(size, device)
            model.train()
            outputs = model(images)  # 4. forward
            if not isinstance(outputs, torch.Tensor):
                record(f"{arch}: forward returns a plain tensor", False,
                       type(outputs).__name__)
                continue

            record(f"{arch}: output shape == ({BATCH}, {NUM_CLASSES})",
                   tuple(outputs.shape) == (BATCH, NUM_CLASSES),
                   str(tuple(outputs.shape)))  # 5
            record(f"{arch}: output finite", bool(torch.isfinite(outputs).all()))  # 6
            record(f"{arch}: output on CUDA", outputs.device.type == "cuda",
                   str(outputs.device))  # 7

            loss = nn.functional.cross_entropy(outputs, targets)  # 8. synthetic loss
            record(f"{arch}: loss finite", bool(torch.isfinite(loss)),
                   f"noise-on-noise value {loss.item():.4f} - NOT a result")

            before = [p.detach().clone() for p in model.parameters()]
            loss.backward()  # 9. backward

            grads = [p.grad for p in model.parameters() if p.grad is not None]
            record(f"{arch}: gradients exist", len(grads) > 0,
                   f"{len(grads)} tensors")  # 10
            record(f"{arch}: gradients finite",
                   all(bool(torch.isfinite(g).all()) for g in grads))

            optimizer = torch.optim.SGD(
                build_param_groups(model, arch, 0.005, 0.05), momentum=0.9
            )
            optimizer.step()  # 11. exactly ONE step

            after = list(model.parameters())
            changed = sum(
                1 for b, a in zip(before, after, strict=True)
                if not torch.equal(b, a.detach())
            )
            record(f"{arch}: parameters changed after one step", changed > 0,
                   f"{changed}/{len(before)} tensors changed")  # 12

            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"     peak CUDA memory allocated: {peak:.1f} MB")
        except Exception:
            record(f"{arch}: synthetic CUDA pass", False, "exception - see traceback")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 4. AMP + 5. gradient clipping
# ---------------------------------------------------------------------------


def _run_traced_epoch(
    device: torch.device, size: int, amp, config, max_grad_norm: float | None
) -> tuple[list[str], list[torch.Tensor], object]:
    """Drive the REAL ``train_one_epoch`` and record the call order it produces.

    The ordering claim has to be evidence about ``src/training.py``, not about a
    sequence written out again here - re-writing it in this script would only
    prove that this script can put four calls in the right order.
    """
    arch = "resnet18"
    model = build_model(arch, NUM_CLASSES, pretrained=True).to(device)
    optimizer = torch.optim.SGD(
        build_param_groups(model, arch, 0.005, 0.05), momentum=0.9
    )
    criterion = build_criterion(config)
    scaler = build_scaler(amp)
    loader = synthetic_loader(size, BATCH)  # exactly one batch

    order: list[str] = []
    real_clip = nn.utils.clip_grad_norm_
    real_backward = torch.Tensor.backward
    real_unscale = type(scaler).unscale_
    real_step = type(optimizer).step

    def spy_clip(params, max_norm, *a, **k):
        order.append("clip")
        return real_clip(params, max_norm, *a, **k)

    def spy_backward(self, *a, **k):
        order.append("backward")
        return real_backward(self, *a, **k)

    def spy_unscale(self, opt):
        order.append("unscale")
        return real_unscale(self, opt)

    def spy_step(self, *a, **k):
        order.append("step")
        return real_step(self, *a, **k)

    nn.utils.clip_grad_norm_ = spy_clip
    torch.Tensor.backward = spy_backward
    type(scaler).unscale_ = spy_unscale
    type(optimizer).step = spy_step
    try:
        metrics = train_one_epoch(
            model, loader, criterion, optimizer, device,
            architecture=arch, scaler=scaler, amp=amp, max_grad_norm=max_grad_norm,
        )
    finally:
        nn.utils.clip_grad_norm_ = real_clip
        torch.Tensor.backward = real_backward
        type(scaler).unscale_ = real_unscale
        type(optimizer).step = real_step

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return order, grads, metrics


def section_4_5_amp_and_clipping(device: torch.device, size: int) -> None:
    header("4. AMP CUDA TEST + 5. GRADIENT CLIPPING ORDER")
    config = load_config()
    amp_config = replace(config, training=replace(config.training, amp=True))
    amp = resolve_amp(amp_config, device)
    print(f"  resolved AMP: enabled={amp.enabled} dtype={amp.dtype}")
    print(f"  reason: {amp.reason!r}")
    record("AMP resolves to enabled on CUDA", amp.enabled, amp.reason)

    # Autocast really does reduce precision on this GPU.
    with torch.amp.autocast(
        device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
    ):
        probe = torch.randn(8, 8, device=device) @ torch.randn(8, 8, device=device)
    record("autocast produces reduced precision", probe.dtype == amp.dtype,
           str(probe.dtype))

    # No deprecated-AMP warning may originate from OUR code.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        order, grads, metrics = _run_traced_epoch(device, size, amp, config, 5.0)

    ours = [
        w for w in caught
        if "torch.cuda.amp" in str(w.message)
        and "src" in str(getattr(w, "filename", ""))
    ]
    deprecations = [
        str(w.message)[:100] for w in caught
        if issubclass(w.category, (DeprecationWarning, FutureWarning))
    ]
    record("no deprecated AMP API warning from our code", not ours,
           "; ".join(str(w.message)[:80] for w in ours))
    if deprecations:
        print("  (deprecation warnings seen, for the record):")
        for d in dict.fromkeys(deprecations):
            print(f"     {d}")

    record("GradScaler + autocast completed a real train_one_epoch",
           metrics is not None,
           f"loss {metrics.loss:.4f} on noise - NOT a result")
    record("gradients exist after AMP backward", len(grads) > 0,
           f"{len(grads)} tensors")
    record("gradients finite after unscale",
           all(bool(torch.isfinite(g).all()) for g in grads))

    expected = ["backward", "unscale", "clip", "step"]
    trimmed = [o for o in order if o in expected]
    record("ordering is backward -> unscale -> clip -> step",
           trimmed == expected, f"observed {trimmed}")

    # Clipping disabled: no clip call may be made at all.
    order_off, _, _ = _run_traced_epoch(device, size, amp, config, None)
    record("clipping disabled makes no clip call at all",
           "clip" not in order_off, f"observed {order_off}")


# ---------------------------------------------------------------------------
# 6. Checkpoints + 7. resume LR trajectory
# ---------------------------------------------------------------------------


def section_6_7_checkpoint_and_resume(device: torch.device, size: int) -> None:
    header("6. CHECKPOINT CUDA TEST + 7. RESUME LR TRAJECTORY")
    config = load_config()
    amp_config = replace(config, training=replace(config.training, amp=True))
    tmp = Path(tempfile.mkdtemp(prefix="cuda_qa_"))
    arch = "resnet18"
    epochs = 6
    split_at = 3
    try:
        torch.manual_seed(0)
        model = build_model(arch, NUM_CLASSES, pretrained=True).to(device)
        optimizer = build_optimizer(model, arch, config)
        scheduler = build_scheduler(optimizer, config, epochs)
        amp = resolve_amp(amp_config, device)
        scaler = build_scaler(amp)
        criterion = build_criterion(config)
        images, targets = synthetic_batch(size, device)

        def one_step() -> None:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp.dtype,
                                    enabled=amp.enabled):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        uninterrupted: list[dict[str, float]] = []
        for epoch in range(epochs):
            one_step()
            uninterrupted.append(current_learning_rates(optimizer))
            if scheduler is not None:
                scheduler.step()
            if epoch == split_at - 1:
                last = save_checkpoint(
                    checkpoint_path(tmp, arch, "last"),
                    model=model, architecture=arch, epoch=epoch,
                    best_metric=0.0, best_epoch=epoch,
                    optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler if amp.enabled else None,
                )
                best = save_checkpoint(
                    checkpoint_path(tmp, arch, "best"),
                    model=model, architecture=arch, epoch=epoch,
                    best_metric=0.0, best_epoch=epoch, include_rng_state=False,
                )
                ref_state = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}

        record("LAST checkpoint written", last.is_file(),
               f"{last.stat().st_size / 1024 ** 2:.1f} MB")
        record("BEST (weights-only) checkpoint written", best.is_file(),
               f"{best.stat().st_size / 1024**2:.1f} MB")
        record("weights-only BEST is smaller than LAST",
               best.stat().st_size < last.stat().st_size)

        # --- reload both, on CUDA. This is where the map_location/RNG bug bit. ---
        loaded_last = load_checkpoint(last, map_location=device)
        loaded_best = load_checkpoint(best, map_location=device)
        record("LAST carries optimizer state", "optimizer_state" in loaded_last)
        record("LAST carries scheduler state", "scheduler_state" in loaded_last)
        record("LAST carries scaler state (AMP on)", "scaler_state" in loaded_last)
        record("BEST carries no optimizer state", "optimizer_state" not in loaded_best)

        torch.manual_seed(0)
        fresh = build_model(arch, NUM_CLASSES, pretrained=True).to(device)
        fresh_opt = build_optimizer(fresh, arch, config)
        fresh_sched = build_scheduler(fresh_opt, config, epochs)
        fresh_scaler = build_scaler(amp)

        resume = restore_training_state(
            loaded_last, model=fresh, architecture=arch, optimizer=fresh_opt,
            scheduler=fresh_sched, scaler=fresh_scaler,
        )
        record("restore_training_state completed on CUDA", True, str(resume.restored))
        record("RNG state restored on CUDA (no device error)",
               "rng_state" in resume.restored, str(resume.skipped))

        same = all(torch.equal(ref_state[k].to(device), v)
                   for k, v in fresh.state_dict().items())
        record("reloaded model parameters match", same)
        record("resume start_epoch is the epoch after the saved one",
               resume.start_epoch == split_at, f"start_epoch={resume.start_epoch}")

        # --- 7. the SequentialLR trap, on CUDA ---
        resumed: list[dict[str, float]] = []
        for _ in range(split_at, epochs):
            model_ref, optimizer_ref = fresh, fresh_opt
            model_ref.train()
            optimizer_ref.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp.dtype,
                                    enabled=amp.enabled):
                loss = criterion(model_ref(images), targets)
            fresh_scaler.scale(loss).backward()
            fresh_scaler.step(optimizer_ref)
            fresh_scaler.update()
            resumed.append(current_learning_rates(optimizer_ref))
            if fresh_sched is not None:
                fresh_sched.step()

        expected_tail = uninterrupted[split_at:]
        print("\n  epoch | uninterrupted LRs            | resumed LRs")
        for i, (u, r) in enumerate(zip(expected_tail, resumed, strict=True)):
            print(f"  {split_at + i:>5} | {u} | {r}")
        record("resumed LR trajectory == uninterrupted LR trajectory",
               expected_tail == resumed,
               "" if expected_tail == resumed else "MISMATCH - the resume is wrong")
    except Exception:
        record("checkpoint/resume on CUDA", False, "exception - see traceback")
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        record("temporary checkpoint directory removed", not tmp.exists())
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 8. fit() + 9. memory
# ---------------------------------------------------------------------------


def section_8_9_fit_and_memory(device: torch.device, size: int) -> None:
    header("8. ALL THREE ARCHITECTURES THROUGH fit() + 9. MEMORY")
    config = load_config()
    run_config = replace(
        config,
        training=replace(
            config.training, amp=True, epochs=2, scheduler_warmup_epochs=1,
            save_best_checkpoint=True, save_last_checkpoint=True,
        ),
    )
    train_loader = synthetic_loader(size, 24)
    val_loader = synthetic_loader(size, 16)

    for arch in ARCHITECTURES:
        print(f"\n-- {arch} through fit() --")
        tmp = Path(tempfile.mkdtemp(prefix=f"cuda_qa_fit_{arch}_"))
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = build_model(arch, NUM_CLASSES, pretrained=True)
            record(f"{arch}: constructed on the T4", True)

            history = fit(
                model, architecture=arch, train_loader=train_loader,
                val_loader=val_loader, config=run_config, device=device,
                checkpoint_dir=tmp, log=lambda m: print(f"     {m}"),
            )
            record(f"{arch}: fit() completed 2 synthetic epochs",
                   len(history.records) == 2, f"{len(history.records)} epochs")
            record(f"{arch}: all epoch losses finite",
                   all(math.isfinite(r.train_loss) and math.isfinite(r.val_loss)
                       for r in history.records))
            record(f"{arch}: LAST checkpoint written",
                   checkpoint_path(tmp, arch, "last").is_file())
            record(f"{arch}: BEST checkpoint written",
                   checkpoint_path(tmp, arch, "best").is_file())

            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            free, total = torch.cuda.mem_get_info(device.index or 0)
            print(f"     peak allocated {peak:.0f} MB | "
                  f"peak reserved {reserved:.0f} MB | "
                  f"free {free / 1024**2:.0f} MB of {total / 1024**2:.0f} MB")
        except torch.cuda.OutOfMemoryError as exc:
            record(f"{arch}: fit() on CUDA", False, f"CUDA OOM - {str(exc)[:120]}")
        except Exception:
            record(f"{arch}: fit() on CUDA", False, "exception - see traceback")
            traceback.print_exc()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------


def summarise() -> None:
    header("SUMMARY")
    failures = [(n, d) for n, ok, d in _RESULTS if not ok]
    print(f"  checks run    : {len(_RESULTS)}")
    print(f"  passed        : {len(_RESULTS) - len(failures)}")
    print(f"  FAILED        : {len(failures)}")
    for name, detail in failures:
        print(f"    - {name}" + (f" ({detail})" if detail else ""))
    print("\n  NO CIFAR-10 data was loaded. NO classifier accuracy was produced.")
    print("  NO test split was touched. This was an infrastructure check only.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest", action="store_true",
                        help="also run the full test suite (section 2)")
    parser.add_argument("--size", type=int, default=None,
                        help="input resolution; defaults to config image.size")
    args = parser.parse_args()

    size = args.size if args.size else load_config().image.size
    print(f"Stage 3A.1 CUDA QA - synthetic data only, input {size}x{size}")

    device = section_1_environment()
    if args.pytest:
        section_2_pytest()
    section_3_models(device, size)
    section_4_5_amp_and_clipping(device, size)
    section_6_7_checkpoint_and_resume(device, size)
    section_8_9_fit_and_memory(device, size)
    summarise()
    return 1 if any(not ok for _, ok, _ in _RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
