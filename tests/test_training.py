"""Tests for the training infrastructure.

No test in this file performs a real training experiment. Every model is either
the toy :class:`tests.conftest.TinyNet` or an untrained torchvision
architecture, and every batch is random noise. Losses and accuracies that appear
here are properties of random tensors.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.checkpointing import checkpoint_path, load_checkpoint
from src.metrics import EpochMetrics
from src.models import build_model
from src.training import (
    SUPPORTED_OPTIMIZERS,
    AmpSettings,
    EarlyStopping,
    _require_logits,
    build_criterion,
    build_optimizer,
    build_scaler,
    build_scheduler,
    current_learning_rates,
    describe_device,
    evaluate,
    fit,
    resolve_amp,
    train_one_epoch,
)
from tests.conftest import TinyNet, make_loader

CPU = torch.device("cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA device on this host; this path is pending the Colab T4 run",
)


def optimizer_for(model, config, architecture="resnet18"):
    return build_optimizer(model, architecture, config)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------


def test_describe_device_reports_cpu_without_fabricating_gpu_fields():
    info = describe_device(CPU)
    assert info.type == "cpu"
    assert info.gpu_name is None
    assert info.total_memory_gb is None


def test_describe_device_defaults_to_select_device():
    info = describe_device()
    assert info.type in ("cpu", "cuda")
    assert info.cuda_available == torch.cuda.is_available()


def test_device_info_is_serialisable():
    assert isinstance(describe_device(CPU).as_dict(), dict)


def test_no_hard_coded_cuda_index_in_sources():
    """Guards the 'never hard-code cuda:0' rule across the training sources."""
    for name in ("training.py", "checkpointing.py", "metrics.py"):
        text = (PROJECT_ROOT / "src" / name).read_text(encoding="utf-8")
        assert "cuda:0" not in text, f"{name} hard-codes a device index"
    for name in ("train.py", "smoke_test_training.py"):
        text = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "cuda:0" not in text, f"{name} hard-codes a device index"


@requires_cuda
def test_describe_device_reports_gpu_details():
    info = describe_device(torch.device("cuda"))
    assert info.cuda_available is True
    assert info.gpu_name
    assert info.total_memory_gb and info.total_memory_gb > 0


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------


def test_amp_disabled_on_cpu_with_a_stated_reason(base_config):
    requested = replace(base_config, training=replace(base_config.training, amp=True))
    amp = resolve_amp(requested, CPU)
    assert amp.enabled is False
    assert "not CUDA" in amp.reason


def test_amp_disabled_by_configuration_is_reported(base_config):
    # Set amp=False explicitly rather than relying on the shipped default. This
    # test is about the "disabled by configuration" branch, and the default is a
    # Project Lead decision that has already changed once (false -> true for
    # Stage 3B); the test should not silently start exercising a different branch
    # when it changes again.
    disabled = replace(base_config, training=replace(base_config.training, amp=False))
    amp = resolve_amp(disabled, CPU)
    assert amp.enabled is False
    assert "configuration" in amp.reason


def test_amp_rejects_an_unknown_dtype(base_config):
    bad = replace(base_config, training=replace(base_config.training, amp_dtype="fp8"))
    with pytest.raises(ValueError, match="Unknown amp_dtype"):
        resolve_amp(bad, CPU)


def test_disabled_scaler_is_constructed_and_inert(base_config):
    scaler = build_scaler(resolve_amp(base_config, CPU))
    assert scaler.is_enabled() is False
    assert scaler.state_dict() == {}


def test_scaler_uses_the_non_deprecated_api():
    """torch.cuda.amp.GradScaler is deprecated; build_scaler must not use it."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_scaler(AmpSettings(False, None, "cpu", "test"))
    assert not [w for w in caught if "deprecated" in str(w.message)]


@requires_cuda
def test_amp_enabled_on_cuda(base_config):
    requested = replace(base_config, training=replace(base_config.training, amp=True))
    amp = resolve_amp(requested, torch.device("cuda"))
    assert amp.enabled is True
    assert amp.dtype == torch.float16
    assert build_scaler(amp).is_enabled() is True


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


def test_optimizer_uses_discriminative_learning_rates(tiny_model, tiny_config):
    optimizer = optimizer_for(tiny_model, tiny_config)
    rates = current_learning_rates(optimizer)
    assert rates["backbone"] == pytest.approx(tiny_config.training.learning_rate)
    assert rates["head"] == pytest.approx(tiny_config.training.head_learning_rate)
    assert rates["head"] > rates["backbone"]


def test_optimizer_supports_adamw(tiny_model, base_config):
    config = replace(base_config, training=replace(base_config.training, optimizer="adamw"))
    assert isinstance(optimizer_for(tiny_model, config), torch.optim.AdamW)


def test_optimizer_default_is_sgd_from_config(tiny_model, base_config):
    assert base_config.training.optimizer == "sgd"
    assert isinstance(optimizer_for(tiny_model, base_config), torch.optim.SGD)


def test_optimizer_rejects_unknown_names(tiny_model, base_config):
    config = replace(base_config, training=replace(base_config.training, optimizer="lion"))
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        optimizer_for(tiny_model, config)


def test_supported_optimizers_are_standard():
    assert set(SUPPORTED_OPTIMIZERS) == {"sgd", "adamw"}


def test_optimizer_excludes_frozen_parameters(tiny_model, tiny_config):
    for name, param in tiny_model.named_parameters():
        param.requires_grad = name.startswith("fc.")
    optimizer = optimizer_for(tiny_model, tiny_config)
    assert [g["name"] for g in optimizer.param_groups] == ["head"]


def test_optimizer_raises_when_nothing_is_trainable(tiny_model, tiny_config):
    for param in tiny_model.parameters():
        param.requires_grad = False
    with pytest.raises(ValueError, match="No trainable parameters"):
        optimizer_for(tiny_model, tiny_config)


def test_weight_decay_propagates(tiny_model, tiny_config):
    optimizer = optimizer_for(tiny_model, tiny_config)
    for group in optimizer.param_groups:
        assert group["weight_decay"] == pytest.approx(tiny_config.training.weight_decay)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_warmup_then_cosine_trajectory(tiny_model, base_config):
    config = replace(
        base_config,
        training=replace(base_config.training, scheduler_warmup_epochs=1),
    )
    optimizer = optimizer_for(tiny_model, config)
    scheduler = build_scheduler(optimizer, config, epochs=5)

    head = []
    for _ in range(5):
        head.append(optimizer.param_groups[1]["lr"])
        optimizer.step()
        scheduler.step()

    # Epoch 0 is the warmup epoch at start_factor x base.
    assert head[0] == pytest.approx(
        config.training.head_learning_rate * config.training.scheduler_warmup_start_factor
    )
    # Epoch 1 is the full base rate, and it decays monotonically thereafter.
    assert head[1] == pytest.approx(config.training.head_learning_rate)
    assert head[1] > head[2] > head[3] > head[4]


def test_scheduler_none_disables_scheduling(tiny_model, base_config):
    config = replace(base_config, training=replace(base_config.training, scheduler="none"))
    optimizer = optimizer_for(tiny_model, config)
    assert build_scheduler(optimizer, config, epochs=5) is None


def test_scheduler_rejects_unknown_names(tiny_model, base_config):
    config = replace(base_config, training=replace(base_config.training, scheduler="poly"))
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        build_scheduler(optimizer_for(tiny_model, config), config, epochs=5)


def test_scheduler_rejects_warmup_longer_than_the_run(tiny_model, base_config):
    config = replace(
        base_config, training=replace(base_config.training, scheduler_warmup_epochs=5)
    )
    with pytest.raises(ValueError, match="must be less than"):
        build_scheduler(optimizer_for(tiny_model, config), config, epochs=5)


def test_scheduler_without_warmup_starts_at_base_rate(tiny_model, base_config):
    config = replace(
        base_config, training=replace(base_config.training, scheduler_warmup_epochs=0)
    )
    optimizer = optimizer_for(tiny_model, config)
    build_scheduler(optimizer, config, epochs=5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(
        config.training.head_learning_rate
    )


def test_scheduler_steps_once_per_epoch_in_fit(tiny_model, tiny_config, train_loader, val_loader):
    """The documented contract is one step per epoch, after validation."""
    calls = []
    real_step = torch.optim.lr_scheduler.SequentialLR.step

    def counting_step(self, *args, **kwargs):
        calls.append(1)
        return real_step(self, *args, **kwargs)

    config = replace(tiny_config, training=replace(tiny_config.training, epochs=3))
    with patch.object(torch.optim.lr_scheduler.SequentialLR, "step", counting_step):
        fit(
            tiny_model,
            architecture="resnet18",
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=CPU,
            log=lambda _msg: None,
        )
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def test_one_epoch_runs_and_returns_metrics(tiny_model, tiny_config, train_loader):
    metrics = train_one_epoch(
        tiny_model,
        train_loader,
        build_criterion(tiny_config),
        optimizer_for(tiny_model, tiny_config),
        CPU,
        architecture="resnet18",
    )
    assert isinstance(metrics, EpochMetrics)
    assert metrics.num_samples == 10
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss > 0


def test_training_updates_parameters(tiny_model, tiny_config, train_loader):
    before = tiny_model.fc.weight.detach().clone()
    train_one_epoch(
        tiny_model,
        train_loader,
        build_criterion(tiny_config),
        optimizer_for(tiny_model, tiny_config),
        CPU,
        architecture="resnet18",
    )
    assert not torch.equal(before, tiny_model.fc.weight.detach())


def test_model_is_in_train_mode_during_every_training_forward(
    tiny_model, tiny_config, train_loader
):
    train_one_epoch(
        tiny_model,
        train_loader,
        build_criterion(tiny_config),
        optimizer_for(tiny_model, tiny_config),
        CPU,
        architecture="resnet18",
    )
    assert tiny_model.training_flags_seen
    assert all(tiny_model.training_flags_seen)


def test_gradients_are_cleared_between_steps(tiny_model, tiny_config, train_loader):
    """zero_grad(set_to_none=True) must run before each forward pass."""
    seen: list[bool] = []
    optimizer = optimizer_for(tiny_model, tiny_config)
    real_zero = optimizer.zero_grad

    def recording_zero_grad(*args, **kwargs):
        kwargs.setdefault("set_to_none", True)
        real_zero(*args, **kwargs)
        seen.append(all(p.grad is None for p in tiny_model.parameters()))

    optimizer.zero_grad = recording_zero_grad
    train_one_epoch(
        tiny_model,
        train_loader,
        build_criterion(tiny_config),
        optimizer,
        CPU,
        architecture="resnet18",
    )
    assert seen and all(seen), "gradients were not cleared before a forward pass"


def test_gradient_accumulation_does_not_leak_across_epochs(
    tiny_model, tiny_config, train_loader
):
    criterion = build_criterion(tiny_config)
    optimizer = optimizer_for(tiny_model, tiny_config)
    train_one_epoch(
        tiny_model, train_loader, criterion, optimizer, CPU, architecture="resnet18"
    )
    first = tiny_model.fc.weight.grad.detach().clone()
    train_one_epoch(
        tiny_model, train_loader, criterion, optimizer, CPU, architecture="resnet18"
    )
    # A fresh gradient, not a doubled one.
    assert not torch.equal(first, tiny_model.fc.weight.grad.detach())


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def test_clipping_is_invoked_when_configured(tiny_model, tiny_config, train_loader):
    with patch("torch.nn.utils.clip_grad_norm_", wraps=nn.utils.clip_grad_norm_) as spy:
        train_one_epoch(
            tiny_model,
            train_loader,
            build_criterion(tiny_config),
            optimizer_for(tiny_model, tiny_config),
            CPU,
            architecture="resnet18",
            max_grad_norm=5.0,
        )
    assert spy.call_count == 3  # three batches
    assert spy.call_args.args[1] == 5.0


def test_clipping_is_not_invoked_when_disabled(tiny_model, tiny_config, train_loader):
    """'Do not silently clip when the configuration says clipping is off.'"""
    with patch("torch.nn.utils.clip_grad_norm_") as spy:
        train_one_epoch(
            tiny_model,
            train_loader,
            build_criterion(tiny_config),
            optimizer_for(tiny_model, tiny_config),
            CPU,
            architecture="resnet18",
            max_grad_norm=None,
        )
    spy.assert_not_called()


def test_clipping_is_not_invoked_for_zero_norm(tiny_model, tiny_config, train_loader):
    with patch("torch.nn.utils.clip_grad_norm_") as spy:
        train_one_epoch(
            tiny_model,
            train_loader,
            build_criterion(tiny_config),
            optimizer_for(tiny_model, tiny_config),
            CPU,
            architecture="resnet18",
            max_grad_norm=0.0,
        )
    spy.assert_not_called()


def test_clipping_happens_after_backward_and_before_step(
    tiny_model, tiny_config, train_loader
):
    """Order matters: clipping an empty or already-applied gradient is useless."""
    events: list[str] = []
    optimizer = optimizer_for(tiny_model, tiny_config)
    real_step = optimizer.step
    # Bound before the patch, or the wrapper would call itself.
    real_clip = nn.utils.clip_grad_norm_

    def recording_step(*args, **kwargs):
        events.append("step")
        return real_step(*args, **kwargs)

    def recording_clip(parameters, max_norm, *args, **kwargs):
        grads_exist = any(
            p.grad is not None for p in tiny_model.parameters() if p.requires_grad
        )
        events.append(f"clip(grads={grads_exist})")
        return real_clip(parameters, max_norm, *args, **kwargs)

    optimizer.step = recording_step
    with patch("torch.nn.utils.clip_grad_norm_", recording_clip):
        train_one_epoch(
            tiny_model,
            train_loader,
            build_criterion(tiny_config),
            optimizer,
            CPU,
            architecture="resnet18",
            max_grad_norm=1.0,
        )

    assert events[0] == "clip(grads=True)", events
    assert events[1] == "step"
    # Strict alternation: exactly one clip before each step.
    assert events == ["clip(grads=True)", "step"] * 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validation_returns_metrics(tiny_model, tiny_config, val_loader):
    metrics = evaluate(
        tiny_model, val_loader, build_criterion(tiny_config), CPU, architecture="resnet18"
    )
    assert metrics.num_samples == 6
    assert 0.0 <= metrics.accuracy <= 1.0


def test_validation_does_not_change_parameters(tiny_model, tiny_config, val_loader):
    before = [p.detach().clone() for p in tiny_model.parameters()]
    evaluate(
        tiny_model, val_loader, build_criterion(tiny_config), CPU, architecture="resnet18"
    )
    after = list(tiny_model.parameters())
    assert all(torch.equal(b, a.detach()) for b, a in zip(before, after, strict=True))


def test_validation_leaves_no_gradients(tiny_model, tiny_config, val_loader):
    evaluate(
        tiny_model, val_loader, build_criterion(tiny_config), CPU, architecture="resnet18"
    )
    assert all(p.grad is None for p in tiny_model.parameters())


def test_validation_uses_eval_mode(tiny_model, tiny_config, val_loader):
    tiny_model.train()
    evaluate(
        tiny_model, val_loader, build_criterion(tiny_config), CPU, architecture="resnet18"
    )
    assert tiny_model.training_flags_seen
    assert not any(tiny_model.training_flags_seen)
    assert tiny_model.training is False


def test_validation_does_not_update_batchnorm_running_stats(
    tiny_model, tiny_config, val_loader
):
    before = tiny_model.bn.running_mean.detach().clone()
    evaluate(
        tiny_model, val_loader, build_criterion(tiny_config), CPU, architecture="resnet18"
    )
    assert torch.equal(before, tiny_model.bn.running_mean)


def test_frozen_backbone_holds_batchnorm_in_eval(tiny_model, tiny_config, train_loader):
    for name, param in tiny_model.named_parameters():
        param.requires_grad = name.startswith("fc.")
    before = tiny_model.bn.running_mean.detach().clone()
    train_one_epoch(
        tiny_model,
        train_loader,
        build_criterion(tiny_config),
        optimizer_for(tiny_model, tiny_config),
        CPU,
        architecture="resnet18",
        backbone_frozen=True,
    )
    assert torch.equal(before, tiny_model.bn.running_mean)


# ---------------------------------------------------------------------------
# GoogLeNet compatibility
# ---------------------------------------------------------------------------


def test_googlenet_output_flows_through_the_training_loop(tiny_config):
    """The approved aux_logits=False build returns a plain tensor."""
    model = build_model("googlenet", num_classes=10, pretrained=False, aux_logits=False)
    loader = make_loader(4, batch_size=2, num_classes=10, image_size=64, seed=3)
    metrics = train_one_epoch(
        model,
        loader,
        build_criterion(tiny_config),
        build_optimizer(model, "googlenet", tiny_config),
        CPU,
        architecture="googlenet",
    )
    assert metrics.num_samples == 4


def test_googlenet_evaluates_through_the_loop(tiny_config):
    model = build_model("googlenet", num_classes=10, pretrained=False, aux_logits=False)
    loader = make_loader(4, batch_size=2, num_classes=10, image_size=64, seed=4)
    assert evaluate(
        model, loader, build_criterion(tiny_config), CPU, architecture="googlenet"
    ).num_samples == 4


def test_require_logits_accepts_a_tensor():
    tensor = torch.zeros(2, 10)
    assert _require_logits(tensor, "googlenet") is tensor


def test_require_logits_rejects_auxiliary_outputs():
    """Silently unwrapping would drop the aux losses and change the objective."""
    from torchvision.models.googlenet import GoogLeNetOutputs

    output = GoogLeNetOutputs(torch.zeros(2, 10), torch.zeros(2, 10), torch.zeros(2, 10))
    with pytest.raises(TypeError, match="aux_logits=False"):
        _require_logits(output, "googlenet")


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.update(0.5) is True
    assert stopper.update(0.4) is False
    assert stopper.should_stop is False
    assert stopper.update(0.3) is False
    assert stopper.should_stop is True


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.update(0.5)
    stopper.update(0.4)
    stopper.update(0.6)
    assert stopper.epochs_without_improvement == 0
    assert stopper.should_stop is False


def test_early_stopping_honours_min_delta():
    stopper = EarlyStopping(patience=1, min_delta=0.05, mode="max")
    stopper.update(0.50)
    # +0.01 is real but smaller than min_delta, so not an improvement.
    assert stopper.update(0.51) is False
    assert stopper.should_stop is True


def test_early_stopping_can_be_disabled():
    stopper = EarlyStopping(patience=1, mode="max", enabled=False)
    stopper.update(0.5)
    stopper.update(0.1)
    stopper.update(0.1)
    assert stopper.should_stop is False


def test_early_stopping_min_mode():
    stopper = EarlyStopping(patience=1, mode="min")
    assert stopper.update(1.0) is True
    assert stopper.update(0.5) is True
    assert stopper.update(0.9) is False


def test_early_stopping_rejects_bad_arguments():
    with pytest.raises(ValueError, match="mode must be"):
        EarlyStopping(patience=1, mode="sideways")
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(patience=-1)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(patience=1, min_delta=-0.1)


def test_fit_stops_early_and_records_it(tiny_model, tiny_config, train_loader, val_loader):
    config = replace(
        tiny_config,
        training=replace(
            tiny_config.training,
            epochs=6,
            early_stopping_enabled=True,
            early_stopping_patience=1,
            # A huge min_delta makes every epoch a non-improvement, so stopping
            # is guaranteed without depending on what random noise produces.
            early_stopping_min_delta=10.0,
        ),
    )
    history = fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        log=lambda _msg: None,
    )
    assert history.stopped_early is True
    assert history.completed_epochs < 6
    assert any("early stopping" in note for note in history.notes)


# ---------------------------------------------------------------------------
# fit: orchestration, checkpointing, resume
# ---------------------------------------------------------------------------


def test_fit_has_no_test_loader_parameter():
    """Structural guarantee that the test set cannot be passed to training."""
    parameters = set(inspect.signature(fit).parameters)
    assert "test_loader" not in parameters
    assert not any("test" in name for name in parameters)


def test_train_script_discards_the_test_loader():
    source = (PROJECT_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "_discarded_test_loader" in source
    assert "del _discarded_test_loader" in source
    # The test loader is never handed to fit.
    assert "test_loader=" not in source


def test_fit_runs_the_configured_number_of_epochs(
    tiny_model, tiny_config, train_loader, val_loader
):
    config = replace(tiny_config, training=replace(tiny_config.training, epochs=3))
    history = fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        log=lambda _msg: None,
    )
    assert history.completed_epochs == 3
    assert len(history.records) == 3
    assert [r.epoch for r in history.records] == [0, 1, 2]


def test_fit_epochs_argument_overrides_config(
    tiny_model, tiny_config, train_loader, val_loader
):
    # Warmup is dropped to 0 because a single-epoch run cannot also contain a
    # one-epoch warmup; see the guard test below.
    config = replace(
        tiny_config, training=replace(tiny_config.training, scheduler_warmup_epochs=0)
    )
    history = fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        epochs=1,
        log=lambda _msg: None,
    )
    assert history.completed_epochs == 1


def test_single_epoch_run_with_warmup_is_rejected_not_silently_degraded(
    tiny_model, tiny_config, train_loader, val_loader
):
    """A one-epoch run whose warmup is also one epoch has no cosine phase.

    That is a misconfiguration rather than a schedule, so it raises with an
    actionable message instead of quietly running warmup-only.
    """
    with pytest.raises(ValueError, match="scheduler_warmup_epochs"):
        fit(
            tiny_model,
            architecture="resnet18",
            train_loader=train_loader,
            val_loader=val_loader,
            config=tiny_config,
            device=CPU,
            epochs=1,
            log=lambda _msg: None,
        )


def test_fit_records_every_required_log_field(
    tiny_model, tiny_config, train_loader, val_loader
):
    history = fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        log=lambda _msg: None,
    )
    record = history.records[0]
    for field in (
        "architecture",
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rates",
        "epoch_seconds",
        "best_val_accuracy",
    ):
        assert field in record.as_dict()
    assert history.device_info["type"] == "cpu"
    assert record.epoch_seconds > 0


def test_history_rows_flatten_learning_rates(
    tiny_model, tiny_config, train_loader, val_loader
):
    history = fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        log=lambda _msg: None,
    )
    row = history.as_rows()[0]
    assert "lr_backbone" in row and "lr_head" in row
    assert "learning_rates" not in row


def test_fit_writes_checkpoints_into_the_configured_directory(
    tiny_model, tiny_config, train_loader, val_loader
):
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        log=lambda _msg: None,
    )
    directory = tiny_config.paths.checkpoints_dir
    assert checkpoint_path(directory, "resnet18", "last").is_file()
    assert checkpoint_path(directory, "resnet18", "best").is_file()


def test_checkpoint_dir_argument_overrides_config(
    tiny_model, tiny_config, train_loader, val_loader, tmp_path
):
    elsewhere = tmp_path / "somewhere_else"
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        checkpoint_dir=elsewhere,
        log=lambda _msg: None,
    )
    assert checkpoint_path(elsewhere, "resnet18", "last").is_file()
    assert not tiny_config.paths.checkpoints_dir.exists()


def test_best_checkpoint_tracks_the_best_epoch(
    tiny_model, tiny_config, train_loader, val_loader
):
    history = fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        log=lambda _msg: None,
    )
    best = load_checkpoint(
        checkpoint_path(tiny_config.paths.checkpoints_dir, "resnet18", "best")
    )
    assert best["best_metric"] == pytest.approx(history.best_val_accuracy)
    assert best["epoch"] == history.best_epoch


def test_best_checkpoint_is_weights_only_by_default(
    tiny_model, tiny_config, train_loader, val_loader
):
    """The best checkpoint is for evaluation, so it carries no optimiser state.

    That roughly halves it; for VGG11 the optimiser buffers are ~490 MB.
    """
    assert tiny_config.training.best_checkpoint_weights_only is True
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        log=lambda _msg: None,
    )
    directory = tiny_config.paths.checkpoints_dir
    best = load_checkpoint(checkpoint_path(directory, "resnet18", "best"))
    last = load_checkpoint(checkpoint_path(directory, "resnet18", "last"))

    assert "optimizer_state" not in best
    assert "model_state" in best
    assert best["best_metric"] is not None
    # `last` must still be fully resumable.
    assert "optimizer_state" in last
    assert "scheduler_state" in last


def test_best_checkpoint_can_carry_full_state(
    tiny_model, tiny_config, train_loader, val_loader
):
    config = replace(
        tiny_config,
        training=replace(tiny_config.training, best_checkpoint_weights_only=False),
    )
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        log=lambda _msg: None,
    )
    best = load_checkpoint(
        checkpoint_path(config.paths.checkpoints_dir, "resnet18", "best")
    )
    assert "optimizer_state" in best


def test_weights_only_best_checkpoint_is_smaller(
    tiny_model, tiny_config, train_loader, val_loader, tmp_path
):
    """The size claim is asserted, not just stated."""
    slim_dir = tmp_path / "slim"
    full_dir = tmp_path / "full"
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        checkpoint_dir=slim_dir,
        log=lambda _msg: None,
    )
    full_config = replace(
        tiny_config,
        training=replace(tiny_config.training, best_checkpoint_weights_only=False),
    )
    fit(
        TinyNet(),
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=full_config,
        device=CPU,
        checkpoint_dir=full_dir,
        log=lambda _msg: None,
    )
    slim = checkpoint_path(slim_dir, "resnet18", "best").stat().st_size
    full = checkpoint_path(full_dir, "resnet18", "best").stat().st_size
    assert slim < full


def test_best_checkpoint_can_be_disabled(
    tiny_model, tiny_config, train_loader, val_loader
):
    config = replace(
        tiny_config, training=replace(tiny_config.training, save_best_checkpoint=False)
    )
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        log=lambda _msg: None,
    )
    assert not checkpoint_path(config.paths.checkpoints_dir, "resnet18", "best").is_file()


def test_checkpoint_stores_the_effective_config(
    tiny_model, tiny_config, train_loader, val_loader
):
    """The snapshot must reflect overrides, not just the YAML file."""
    config = replace(tiny_config, training=replace(tiny_config.training, epochs=2))
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        log=lambda _msg: None,
    )
    stored = load_checkpoint(
        checkpoint_path(config.paths.checkpoints_dir, "resnet18", "last")
    )["config"]
    assert stored["epochs"] == 2
    assert stored["optimizer"] == config.training.optimizer


def test_fit_resumes_from_a_checkpoint(
    tiny_model, tiny_config, train_loader, val_loader
):
    config = replace(tiny_config, training=replace(tiny_config.training, epochs=2))
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        log=lambda _msg: None,
    )
    last = checkpoint_path(config.paths.checkpoints_dir, "resnet18", "last")

    longer = replace(config, training=replace(config.training, epochs=4))
    resumed = fit(
        TinyNet(),
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=longer,
        device=CPU,
        resume_from=last,
        log=lambda _msg: None,
    )
    # Two epochs were already done, so only epochs 2 and 3 remain.
    assert [r.epoch for r in resumed.records] == [2, 3]
    assert resumed.completed_epochs == 4


def test_resume_from_a_missing_checkpoint_raises(
    tiny_model, tiny_config, train_loader, val_loader, tmp_path
):
    """It must not silently restart from epoch 0."""
    with pytest.raises(FileNotFoundError):
        fit(
            tiny_model,
            architecture="resnet18",
            train_loader=train_loader,
            val_loader=val_loader,
            config=tiny_config,
            device=CPU,
            resume_from=tmp_path / "nope.pt",
            log=lambda _msg: None,
        )


def test_resume_from_a_different_architecture_raises(
    tiny_model, tiny_config, train_loader, val_loader
):
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        log=lambda _msg: None,
    )
    last = checkpoint_path(tiny_config.paths.checkpoints_dir, "resnet18", "last")
    with pytest.raises(ValueError, match="architecture mismatch"):
        fit(
            TinyNet(),
            architecture="vgg11",
            train_loader=train_loader,
            val_loader=val_loader,
            config=tiny_config,
            device=CPU,
            resume_from=last,
            log=lambda _msg: None,
        )


def test_resume_past_the_epoch_budget_does_nothing(
    tiny_model, tiny_config, train_loader, val_loader
):
    config = replace(tiny_config, training=replace(tiny_config.training, epochs=2))
    fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        log=lambda _msg: None,
    )
    last = checkpoint_path(config.paths.checkpoints_dir, "resnet18", "last")
    history = fit(
        TinyNet(),
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=CPU,
        resume_from=last,
        log=lambda _msg: None,
    )
    assert history.records == []


def test_fit_rejects_a_non_positive_epoch_count(
    tiny_model, tiny_config, train_loader, val_loader
):
    with pytest.raises(ValueError, match="epochs must be"):
        fit(
            tiny_model,
            architecture="resnet18",
            train_loader=train_loader,
            val_loader=val_loader,
            config=tiny_config,
            device=CPU,
            epochs=0,
            log=lambda _msg: None,
        )


def test_fit_reports_amp_state(tiny_model, tiny_config, train_loader, val_loader):
    history = fit(
        tiny_model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=CPU,
        log=lambda _msg: None,
    )
    assert history.amp["enabled"] is False
    assert "reason" in history.amp


def test_label_smoothing_propagates_to_the_criterion(base_config):
    assert build_criterion(base_config).label_smoothing == 0.0
    smoothed = replace(base_config, training=replace(base_config.training, label_smoothing=0.1))
    assert build_criterion(smoothed).label_smoothing == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# CUDA - skipped off-GPU, must run on the Colab T4
# ---------------------------------------------------------------------------


@requires_cuda
def test_training_runs_on_cuda(tiny_config, train_loader, val_loader):
    model = TinyNet().cuda()
    history = fit(
        model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=tiny_config,
        device=torch.device("cuda"),
        log=lambda _msg: None,
    )
    assert history.completed_epochs == tiny_config.training.epochs
    assert history.device_info["type"] == "cuda"
    assert all(p.device.type == "cuda" for p in model.parameters())


@requires_cuda
def test_training_runs_on_cuda_with_amp(tiny_config, train_loader, val_loader):
    config = replace(tiny_config, training=replace(tiny_config.training, amp=True))
    model = TinyNet().cuda()
    history = fit(
        model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device("cuda"),
        log=lambda _msg: None,
    )
    assert history.amp["enabled"] is True
    assert history.completed_epochs == config.training.epochs


@requires_cuda
def test_amp_checkpoint_round_trip_on_cuda(tiny_config, train_loader, val_loader):
    config = replace(tiny_config, training=replace(tiny_config.training, amp=True))
    model = TinyNet().cuda()
    fit(
        model,
        architecture="resnet18",
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device("cuda"),
        log=lambda _msg: None,
    )
    stored = load_checkpoint(
        checkpoint_path(config.paths.checkpoints_dir, "resnet18", "last"),
        map_location="cuda",
    )
    assert "scaler_state" in stored


@requires_cuda
def test_bfloat16_amp_is_gated_on_native_gpu_support(base_config):
    """bfloat16 must not be silently accepted on a GPU that cannot do it natively.

    The target GPU is a Tesla T4 (Turing, sm_75), which has no native bfloat16
    path - torch only offers it there through emulation. Rather than fall back
    to fp16 and quietly change a run's numerics, AMP is disabled and the reason
    recorded, the same way a CPU request is handled.
    """
    config = replace(
        base_config,
        training=replace(base_config.training, amp=True, amp_dtype="bfloat16"),
    )
    amp = resolve_amp(config, torch.device("cuda"))
    native = torch.cuda.is_bf16_supported(including_emulation=False)

    if native:
        assert amp.enabled
        assert amp.dtype is torch.bfloat16
    else:
        assert not amp.enabled, "bfloat16 was accepted on a GPU without native support"
        assert amp.dtype is None
        assert "bfloat16" in amp.reason
        assert "float16" in amp.reason, "the reason should say what to use instead"


@requires_cuda
def test_float16_amp_stays_enabled_on_cuda(base_config):
    """The guard added for bfloat16 must not disturb the fp16 path."""
    config = replace(
        base_config,
        training=replace(base_config.training, amp=True, amp_dtype="float16"),
    )
    amp = resolve_amp(config, torch.device("cuda"))
    assert amp.enabled
    assert amp.dtype is torch.float16
