"""Model construction and transfer-learning utilities.

Three ImageNet-pretrained torchvision architectures are adapted for CIFAR-10:
GoogLeNet, ResNet18 and VGG11. Nothing here trains anything - this module only
builds models, rewires their classifier heads, and exposes the freeze/unfreeze
and parameter-grouping helpers that the Stage 3 training loop will need.

Two ordering rules matter and are enforced rather than assumed:

*Weights are loaded before the head is replaced.* torchvision's builders force
``num_classes=1000`` whenever ``weights`` is given, so the 10-class head cannot
be requested at construction time. It must be swapped in afterwards, otherwise
either the load fails on a shape mismatch or a freshly initialised head is
silently overwritten by pretrained parameters.

*GoogLeNet's ``transform_input`` is left alone.* When pretrained weights are
requested torchvision sets it to True, which makes the model internally re-map
ImageNet-normalised input to the [-1, 1] convention its ported weights expect.
Feeding ImageNet-normalised tensors is correct only while that flag stays on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import (
    GoogLeNet_Weights,
    ResNet18_Weights,
    VGG11_Weights,
    googlenet,
    resnet18,
    vgg11,
)

ARCHITECTURES: tuple[str, ...] = ("googlenet", "resnet18", "vgg11")

# Normalisation layers whose running statistics must be held still when the
# backbone is frozen. VGG11 contains none of these.
_NORM_LAYERS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

# Dotted parameter-name prefix of the classifier head for each architecture.
# Every parameter whose name starts with this prefix is "head"; everything else
# is "backbone". Using name prefixes rather than module identity keeps the
# freeze/unfreeze logic working after a head has been replaced.
_HEAD_PREFIX: dict[str, str] = {
    "googlenet": "fc.",
    "resnet18": "fc.",
    "vgg11": "classifier.6.",
}

_WEIGHTS = {
    "googlenet": GoogLeNet_Weights.IMAGENET1K_V1,
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "vgg11": VGG11_Weights.IMAGENET1K_V1,
}


@dataclass(frozen=True)
class ParameterCounts:
    """Parameter totals for a model in a given freeze state."""

    total: int
    trainable: int
    frozen: int
    backbone: int
    head: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "backbone": self.backbone,
            "head": self.head,
        }


def _validate(architecture: str) -> str:
    key = architecture.lower()
    if key not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. Expected one of {ARCHITECTURES}."
        )
    return key


def head_prefix(architecture: str) -> str:
    """Return the dotted parameter-name prefix of the classifier head."""
    return _HEAD_PREFIX[_validate(architecture)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def replace_classifier_head(
    model: nn.Module, architecture: str, num_classes: int
) -> nn.Module:
    """Swap the 1000-class ImageNet head for a fresh ``num_classes`` head.

    The replacement ``nn.Linear`` carries PyTorch's default initialisation. That
    is deliberate - see :func:`describe_head_initialization`.

    Args:
        model: A constructed torchvision model.
        architecture: One of :data:`ARCHITECTURES`.
        num_classes: Output dimension of the new head.

    Returns:
        The same model, modified in place.
    """
    key = _validate(architecture)

    if key in ("googlenet", "resnet18"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:  # vgg11
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


def build_model(
    architecture: str,
    num_classes: int = 10,
    pretrained: bool = True,
    aux_logits: bool = False,
) -> nn.Module:
    """Build one architecture, adapted for ``num_classes``.

    Args:
        architecture: One of :data:`ARCHITECTURES`.
        num_classes: Number of output classes. 10 for CIFAR-10.
        pretrained: Load official ImageNet weights via the torchvision weights
            API. When False the network is randomly initialised, which is only
            useful for architecture tests.
        aux_logits: GoogLeNet only. See :func:`describe_googlenet_aux` for why
            this defaults to False.

    Returns:
        A model whose forward pass returns ``[batch_size, num_classes]`` in
        eval mode for every architecture.
    """
    key = _validate(architecture)
    weights = _WEIGHTS[key] if pretrained else None

    # Step 1: build and load pretrained weights at the ORIGINAL 1000-class head.
    if key == "googlenet":
        if pretrained:
            # Subtle torchvision contract. Its builder reads
            #     original_aux_logits = kwargs.get("aux_logits", False)
            # and then forces aux_logits=True so the full checkpoint (which
            # contains 20 aux.* keys) can be loaded, discarding the aux heads
            # afterwards when the caller did not ask for them. That forcing goes
            # through _ovewrite_named_param, which RAISES if the key is already
            # present with a different value. So passing aux_logits=False
            # explicitly is an error - the kwarg must be omitted entirely to get
            # the disabled-aux behaviour.
            #
            # transform_input and num_classes are likewise set by torchvision
            # here; passing num_classes would conflict with the checkpoint.
            kwargs = {"aux_logits": True} if aux_logits else {}
            model = googlenet(weights=weights, **kwargs)
        else:
            model = googlenet(
                weights=None,
                aux_logits=aux_logits,
                num_classes=num_classes,
                init_weights=True,
            )
    elif key == "resnet18":
        model = resnet18(weights=weights) if pretrained else resnet18(
            weights=None, num_classes=num_classes
        )
    else:  # vgg11
        model = vgg11(weights=weights) if pretrained else vgg11(
            weights=None, num_classes=num_classes
        )

    # Step 2: replace the head, but only when weights were loaded. In the
    # random-init path the head already has the right shape, and replacing it
    # again would be pointless churn.
    if pretrained:
        model = replace_classifier_head(model, key, num_classes)

    return model


def build_all_models(
    num_classes: int = 10, pretrained: bool = True, aux_logits: bool = False
) -> dict[str, nn.Module]:
    """Build all three architectures, keyed by name."""
    return {
        name: build_model(name, num_classes, pretrained, aux_logits)
        for name in ARCHITECTURES
    }


# ---------------------------------------------------------------------------
# Parameter partitioning and freezing
# ---------------------------------------------------------------------------


def split_parameter_names(
    model: nn.Module, architecture: str
) -> tuple[list[str], list[str]]:
    """Partition parameter names into (backbone, head)."""
    prefix = head_prefix(architecture)
    backbone, head = [], []
    for name, _ in model.named_parameters():
        (head if name.startswith(prefix) else backbone).append(name)
    return backbone, head


def split_parameters(
    model: nn.Module, architecture: str
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition parameters into (backbone, head)."""
    prefix = head_prefix(architecture)
    backbone, head = [], []
    for name, param in model.named_parameters():
        (head if name.startswith(prefix) else backbone).append(param)
    return backbone, head


def freeze_backbone(model: nn.Module, architecture: str) -> nn.Module:
    """Freeze every backbone parameter, leaving only the head trainable.

    This is Phase 1 of the staged fine-tune. Note that freezing parameters is
    not by itself enough to freeze a backbone containing BatchNorm: the running
    mean and variance are buffers, not parameters, and they keep updating in
    training mode. :func:`set_training_mode` handles that.
    """
    prefix = head_prefix(architecture)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefix)
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Make every parameter trainable. This is Phase 2 of the fine-tune."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def set_training_mode(
    model: nn.Module, training: bool, backbone_frozen: bool = False
) -> nn.Module:
    """Set train/eval mode, keeping a frozen backbone's BatchNorm in eval.

    ``model.train()`` puts every BatchNorm into training mode, where it updates
    its running statistics from the incoming batch. During Phase 1 that would
    let a supposedly frozen backbone drift as CIFAR-10 statistics overwrite the
    ImageNet ones, so the backbone would not really be frozen. Passing
    ``backbone_frozen=True`` puts the backbone's normalisation layers back into
    eval mode after the global switch.

    Only GoogLeNet and ResNet18 are affected; torchvision's VGG11 has no
    normalisation layers.

    Args:
        model: The model to switch.
        training: True for train mode, False for eval mode.
        backbone_frozen: When True and ``training`` is True, hold backbone
            normalisation layers in eval mode.

    Returns:
        The same model, modified in place.
    """
    model.train(training)

    # All three classifier heads are plain nn.Linear, so every normalisation
    # layer in these models belongs to the backbone. No head/backbone
    # discrimination is needed here.
    if training and backbone_frozen:
        for module in model.modules():
            if isinstance(module, _NORM_LAYERS):
                module.eval()

    return model


def count_parameters(model: nn.Module, architecture: str) -> ParameterCounts:
    """Count total / trainable / frozen / backbone / head parameters."""
    prefix = head_prefix(architecture)
    total = trainable = backbone = head = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        if name.startswith(prefix):
            head += n
        else:
            backbone += n
    return ParameterCounts(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        backbone=backbone,
        head=head,
    )


def build_param_groups(
    model: nn.Module,
    architecture: str,
    backbone_lr: float,
    head_lr: float,
) -> list[dict]:
    """Build optimiser parameter groups with discriminative learning rates.

    The backbone carries pretrained features that need gentle adjustment; the
    head is randomly initialised and needs to move much faster. Frozen
    parameters are excluded entirely so the optimiser holds no state for them.

    Returns:
        A list suitable for passing directly to a torch optimiser. Groups with
        no trainable parameters are omitted.
    """
    backbone_params, head_params = split_parameters(model, architecture)
    groups = []
    trainable_backbone = [p for p in backbone_params if p.requires_grad]
    trainable_head = [p for p in head_params if p.requires_grad]
    if trainable_backbone:
        groups.append({"params": trainable_backbone, "lr": backbone_lr,
                       "name": "backbone"})
    if trainable_head:
        groups.append({"params": trainable_head, "lr": head_lr, "name": "head"})
    return groups


# ---------------------------------------------------------------------------
# Documentation helpers - these return facts, they do not make decisions
# ---------------------------------------------------------------------------


def describe_head_initialization(model: nn.Module, architecture: str) -> dict:
    """Report how the replaced head is initialised.

    No custom initialisation is applied. ``nn.Linear`` uses Kaiming-uniform for
    the weight with ``a=sqrt(5)`` and a fan-in-scaled uniform for the bias,
    which is the standard, well-tested default and is appropriate for a single
    linear layer feeding softmax. Imposing a custom scheme here would be an
    unjustified deviation.
    """
    prefix = head_prefix(architecture)
    stats = {}
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            stats[name] = {
                "shape": tuple(param.shape),
                "mean": float(param.mean()),
                "std": float(param.std()),
                "min": float(param.min()),
                "max": float(param.max()),
            }
    return {"mechanism": "PyTorch nn.Linear default (Kaiming-uniform, a=sqrt(5))",
            "custom_initialization_applied": False,
            "parameters": stats}


def describe_googlenet_aux() -> dict:
    """Explain the auxiliary-classifier configuration.

    Verified in Stage 1 by reading torchvision's source and by execution: the
    aux heads are input-size INDEPENDENT, because ``InceptionAux.forward``
    begins with ``F.adaptive_avg_pool2d(x, (4, 4))``. They work at 128x128.
    They are disabled for reasons unrelated to resolution.
    """
    return {
        "aux_logits": False,
        "works_at_128px": True,
        "reasons_for_disabling": [
            "torchvision documents the pretrained aux heads as NOT pretrained; "
            "enabling them emits a warning and starts them from effectively "
            "untrained weights, injecting noise into a short fine-tune.",
            "They are training-only. In eval() the model returns a single "
            "logits tensor either way, so this cannot affect ensemble inference.",
            "Their purpose was countering vanishing gradients in a 22-layer "
            "pre-BatchNorm network trained from scratch; torchvision's "
            "BasicConv2d includes BatchNorm and we fine-tune from pretrained "
            "weights.",
            "Keeps a uniform single-tensor output across all three models.",
        ],
        "pretrained_weights_still_load": True,
    }
