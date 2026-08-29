"""Automated tests for the Stage 2 model layer.

Run with:
    python -m pytest tests/ -v

Most tests build randomly-initialised models (``pretrained=False``). That is
deliberate: architecture, wiring, freezing and gradient behaviour do not depend
on the weight values, and building untrained models keeps the suite runnable
offline and fast. The handful of tests that genuinely need the ImageNet
checkpoints are marked and skip automatically when those files are not cached,
so the suite never fails merely because a download has not happened.

Nothing here trains a model. Backward passes are run to prove gradients exist
and flow to the right parameters; no optimiser step is ever taken.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.models import (
    ARCHITECTURES,
    build_model,
    build_param_groups,
    count_parameters,
    freeze_backbone,
    head_prefix,
    replace_classifier_head,
    set_training_mode,
    split_parameter_names,
    unfreeze_all,
)

NUM_CLASSES = 10
BATCH = 2
RESOLUTIONS = (32, 64, 96, 128, 224)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def size(config) -> int:
    """The approved final training resolution."""
    return config.image.size


@pytest.fixture(scope="module")
def untrained_models() -> dict[str, nn.Module]:
    """One randomly-initialised model per architecture, built once."""
    return {a: build_model(a, NUM_CLASSES, pretrained=False) for a in ARCHITECTURES}


def _weights_are_cached(architecture: str) -> bool:
    from torchvision.models import (
        GoogLeNet_Weights,
        ResNet18_Weights,
        VGG11_Weights,
    )

    enum = {"googlenet": GoogLeNet_Weights, "resnet18": ResNet18_Weights,
            "vgg11": VGG11_Weights}[architecture].IMAGENET1K_V1
    cache = Path(torch.hub.get_dir()) / "checkpoints" / enum.url.rsplit("/", 1)[-1]
    return cache.is_file()


def _logits(output) -> torch.Tensor:
    """Extract the main logits tensor regardless of GoogLeNet's output type."""
    return output if isinstance(output, torch.Tensor) else output.logits


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_final_training_resolution_is_128(config):
    assert config.image.size == 128


def test_smoke_test_resolution_is_available(config):
    assert config.image.smoke_test_size == 96


def test_config_lists_the_three_approved_architectures(config):
    assert set(config.model.architectures) == set(ARCHITECTURES)
    assert set(ARCHITECTURES) == {"googlenet", "resnet18", "vgg11"}


def test_vgg11_bn_is_not_substituted(config):
    """The approved spec is plain VGG11. Guard against a silent swap."""
    assert "vgg11_bn" not in config.model.architectures
    from src.models import _WEIGHTS

    assert "bn" not in _WEIGHTS["vgg11"].url


# ---------------------------------------------------------------------------
# Construction and head replacement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_model_builds(architecture, untrained_models):
    assert isinstance(untrained_models[architecture], nn.Module)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_head_outputs_ten_classes(architecture, untrained_models):
    model = untrained_models[architecture]
    head = model.fc if architecture != "vgg11" else model.classifier[6]
    assert isinstance(head, nn.Linear)
    assert head.out_features == NUM_CLASSES


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model("efficientnet_b0")


def test_replace_head_is_idempotent_in_shape(untrained_models):
    model = build_model("resnet18", NUM_CLASSES, pretrained=False)
    replace_classifier_head(model, "resnet18", NUM_CLASSES)
    assert model.fc.out_features == NUM_CLASSES


def test_head_prefixes_match_the_approved_targets():
    """ResNet18/GoogLeNet replace .fc; VGG11 replaces .classifier[6]."""
    assert head_prefix("resnet18") == "fc."
    assert head_prefix("googlenet") == "fc."
    assert head_prefix("vgg11") == "classifier.6."


# ---------------------------------------------------------------------------
# Forward passes (Task 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_shape_at_final_resolution(architecture, untrained_models, size):
    model = untrained_models[architecture].eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, size, size))
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_output_is_float32(architecture, untrained_models, size):
    model = untrained_models[architecture].eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, size, size))
    assert out.dtype == torch.float32


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_forward_output_has_no_nan_or_inf(architecture, untrained_models, size):
    model = untrained_models[architecture].eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, size, size))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_output_is_differentiable_in_train_mode(architecture, untrained_models, size):
    model = untrained_models[architecture]
    set_training_mode(model, training=True)
    out = _logits(model(torch.randn(BATCH, 3, size, size)))
    assert out.requires_grad
    assert out.grad_fn is not None


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_eval_mode_is_deterministic(architecture, untrained_models, size):
    model = untrained_models[architecture].eval()
    x = torch.randn(BATCH, 3, size, size)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_cpu_forward_works(architecture, untrained_models, size):
    model = untrained_models[architecture].to(torch.device("cpu")).eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, size, size))
    assert out.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_cuda_forward_works(architecture, size):
    model = build_model(architecture, NUM_CLASSES, pretrained=False).cuda().eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, size, size, device="cuda"))
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)
    assert out.device.type == "cuda"
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_forward_across_resolutions(architecture, resolution, untrained_models):
    """Architecture verification only - records that each size is accepted."""
    model = untrained_models[architecture].eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, resolution, resolution))
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Train / eval behaviour (Task 7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_train_and_eval_modes_toggle(architecture, untrained_models):
    model = untrained_models[architecture]
    set_training_mode(model, training=True)
    assert model.training
    set_training_mode(model, training=False)
    assert not model.training


def test_googlenet_returns_plain_tensor_when_aux_disabled(size):
    model = build_model("googlenet", NUM_CLASSES, pretrained=False, aux_logits=False)
    model.train()
    assert isinstance(model(torch.randn(BATCH, 3, size, size)), torch.Tensor)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, size, size))
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)


def test_googlenet_aux_modules_absent_when_disabled():
    model = build_model("googlenet", NUM_CLASSES, pretrained=False, aux_logits=False)
    assert model.aux_logits is False
    assert model.aux1 is None and model.aux2 is None


def test_googlenet_aux_works_at_128px_when_enabled(size):
    """Stage 1 correction: aux heads are input-size independent, not 224-only."""
    model = build_model("googlenet", NUM_CLASSES, pretrained=False, aux_logits=True)
    model.train()
    out = model(torch.randn(BATCH, 3, size, size))
    assert type(out).__name__ == "GoogLeNetOutputs"
    assert tuple(out.logits.shape) == (BATCH, NUM_CLASSES)
    assert tuple(out.aux_logits1.shape) == (BATCH, NUM_CLASSES)
    assert tuple(out.aux_logits2.shape) == (BATCH, NUM_CLASSES)


def test_googlenet_aux_is_training_only(size):
    """Even with aux enabled, eval() must return a single ensemble-ready tensor."""
    model = build_model("googlenet", NUM_CLASSES, pretrained=False, aux_logits=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, 3, size, size))
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (BATCH, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Freezing and gradient flow (Task 8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_freeze_backbone_leaves_only_head_trainable(architecture, untrained_models):
    model = untrained_models[architecture]
    freeze_backbone(model, architecture)
    prefix = head_prefix(architecture)
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith(prefix), name
    unfreeze_all(model)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_unfreeze_all_makes_everything_trainable(architecture, untrained_models):
    model = untrained_models[architecture]
    freeze_backbone(model, architecture)
    unfreeze_all(model)
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_phase1_gradients_reach_head_only(architecture, untrained_models, size):
    model = untrained_models[architecture]
    freeze_backbone(model, architecture)
    set_training_mode(model, training=True, backbone_frozen=True)
    model.zero_grad(set_to_none=True)

    out = _logits(model(torch.randn(BATCH, 3, size, size)))
    loss = nn.CrossEntropyLoss()(out, torch.randint(0, NUM_CLASSES, (BATCH,)))
    loss.backward()

    prefix = head_prefix(architecture)
    head = [n for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None]
    backbone = [n for n, p in model.named_parameters()
                if not n.startswith(prefix) and p.grad is not None]
    assert len(head) > 0, "head must receive gradients"
    assert backbone == [], f"frozen backbone must receive none, got {backbone[:3]}"

    model.zero_grad(set_to_none=True)
    unfreeze_all(model)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_phase2_gradients_reach_backbone(architecture, untrained_models, size):
    model = untrained_models[architecture]
    unfreeze_all(model)
    set_training_mode(model, training=True, backbone_frozen=False)
    model.zero_grad(set_to_none=True)

    out = _logits(model(torch.randn(BATCH, 3, size, size)))
    loss = nn.CrossEntropyLoss()(out, torch.randint(0, NUM_CLASSES, (BATCH,)))
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == [], f"all params should have gradients, missing {missing[:3]}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())

    prefix = head_prefix(architecture)
    first_backbone = next(n for n, _ in model.named_parameters()
                          if not n.startswith(prefix))
    grad = dict(model.named_parameters())[first_backbone].grad
    assert grad.abs().sum().item() > 0, "backbone gradient must be non-zero"

    model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("architecture", ["googlenet", "resnet18"])
def test_frozen_backbone_batchnorm_stats_do_not_drift(architecture, size):
    """Freezing parameters alone would not stop BN running stats updating."""
    model = build_model(architecture, NUM_CLASSES, pretrained=False)
    freeze_backbone(model, architecture)
    set_training_mode(model, training=True, backbone_frozen=True)

    before = {k: v.clone() for k, v in model.state_dict().items()
              if "running_mean" in k or "running_var" in k}
    assert before, "expected BatchNorm buffers in this architecture"

    model(torch.randn(BATCH, 3, size, size))
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())


def test_vgg11_has_no_batchnorm():
    """Documents why VGG11 is the LR-sensitive one, and guards against vgg11_bn."""
    model = build_model("vgg11", NUM_CLASSES, pretrained=False)
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    assert norms == []


# ---------------------------------------------------------------------------
# Parameter counting and optimiser groups (Task 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_total_equals_backbone_plus_head(architecture, untrained_models):
    counts = count_parameters(untrained_models[architecture], architecture)
    assert counts.total == counts.backbone + counts.head


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_frozen_trainable_count_equals_head_count(architecture, untrained_models):
    model = untrained_models[architecture]
    freeze_backbone(model, architecture)
    counts = count_parameters(model, architecture)
    assert counts.trainable == counts.head
    assert counts.frozen == counts.backbone
    unfreeze_all(model)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_head_parameter_count_matches_linear_arithmetic(architecture,
                                                        untrained_models):
    """head params must equal in_features*10 + 10."""
    model = untrained_models[architecture]
    head = model.fc if architecture != "vgg11" else model.classifier[6]
    expected = head.in_features * NUM_CLASSES + NUM_CLASSES
    assert count_parameters(model, architecture).head == expected


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_param_group_split(architecture, untrained_models):
    model = untrained_models[architecture]
    unfreeze_all(model)
    groups = build_param_groups(model, architecture, backbone_lr=0.005, head_lr=0.05)
    assert [g["name"] for g in groups] == ["backbone", "head"]
    assert groups[0]["lr"] < groups[1]["lr"]

    freeze_backbone(model, architecture)
    frozen_groups = build_param_groups(model, architecture, 0.005, 0.05)
    assert [g["name"] for g in frozen_groups] == ["head"]
    unfreeze_all(model)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_parameter_name_partition_is_complete(architecture, untrained_models):
    model = untrained_models[architecture]
    backbone, head = split_parameter_names(model, architecture)
    all_names = [n for n, _ in model.named_parameters()]
    assert sorted(backbone + head) == sorted(all_names)
    assert set(backbone).isdisjoint(head)
    assert len(head) == 2  # weight + bias of the replaced Linear


# ---------------------------------------------------------------------------
# Serialization (Task 13)
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip_is_exact(tmp_path):
    source = build_model("resnet18", NUM_CLASSES, pretrained=False)
    target = build_model("resnet18", NUM_CLASSES, pretrained=False)

    assert not torch.equal(
        source.state_dict()["fc.weight"], target.state_dict()["fc.weight"]
    ), "independently built models should differ before loading"

    path = tmp_path / "smoke.pt"
    torch.save(source.state_dict(), path)
    target.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key]), key


def test_saved_state_dict_has_ten_class_head(tmp_path):
    model = build_model("vgg11", NUM_CLASSES, pretrained=False)
    path = tmp_path / "vgg.pt"
    torch.save(model.state_dict(), path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(loaded["classifier.6.weight"].shape) == (NUM_CLASSES, 4096)


# ---------------------------------------------------------------------------
# Pretrained weights - skipped when checkpoints are not cached
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_pretrained_backbone_matches_official_checkpoint(architecture):
    if not _weights_are_cached(architecture):
        pytest.skip(f"{architecture} ImageNet weights not cached")
    from src.models import _WEIGHTS

    key = {"googlenet": "conv1.conv.weight", "resnet18": "conv1.weight",
           "vgg11": "features.0.weight"}[architecture]
    model = build_model(architecture, NUM_CLASSES, pretrained=True)
    reference = _WEIGHTS[architecture].get_state_dict(progress=False)
    assert torch.equal(dict(model.named_parameters())[key].detach(), reference[key])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_pretrained_head_is_replaced_not_pretrained(architecture):
    """The 10-class head must be newly initialised, not carried from ImageNet."""
    if not _weights_are_cached(architecture):
        pytest.skip(f"{architecture} ImageNet weights not cached")
    model = build_model(architecture, NUM_CLASSES, pretrained=True)
    head = model.fc if architecture != "vgg11" else model.classifier[6]
    assert head.out_features == NUM_CLASSES
    weight = head.weight.detach()
    assert abs(float(weight.mean())) < 0.01
    assert float(weight.std()) > 0.0


def test_pretrained_googlenet_keeps_transform_input_enabled():
    """Pretrained GoogLeNet expects its internal [-1,1] remap to stay on."""
    if not _weights_are_cached("googlenet"):
        pytest.skip("googlenet ImageNet weights not cached")
    model = build_model("googlenet", NUM_CLASSES, pretrained=True)
    assert model.transform_input is True
    assert model.aux_logits is False


# ---------------------------------------------------------------------------
# Resolution override wiring (smoke-test size must be usable, not just declared)
# ---------------------------------------------------------------------------


def test_transforms_default_to_the_final_training_resolution(config):
    from src.dataset import build_eval_transform, build_train_transform

    for build in (build_train_transform, build_eval_transform):
        resize = [t for t in build(config).transforms
                  if type(t).__name__ == "Resize"][0]
        assert tuple(resize.size) == (config.image.size, config.image.size)


def test_transforms_accept_the_smoke_test_resolution(config):
    from src.dataset import build_eval_transform, build_train_transform

    smoke = config.image.smoke_test_size
    for build in (build_train_transform, build_eval_transform):
        resize = [t for t in build(config, smoke).transforms
                  if type(t).__name__ == "Resize"][0]
        assert tuple(resize.size) == (smoke, smoke)


def test_models_accept_both_configured_resolutions(config, untrained_models):
    """Whatever the config offers must actually run through every model."""
    for arch in ARCHITECTURES:
        model = untrained_models[arch].eval()
        for res in (config.image.size, config.image.smoke_test_size):
            with torch.no_grad():
                out = model(torch.randn(BATCH, 3, res, res))
            assert tuple(out.shape) == (BATCH, NUM_CLASSES), (arch, res)
