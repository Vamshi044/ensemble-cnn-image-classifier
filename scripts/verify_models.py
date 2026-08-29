"""Stage 2 architecture verification.

Builds each model, exercises forward and backward passes, and reports what
actually happened. This trains nothing: no dataset is read, no optimiser step
is taken, no weights are learned, and no accuracy is produced or implied. Every
number printed comes from execution in this process.

Run with:
    python scripts/verify_models.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

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
    describe_googlenet_aux,
    describe_head_initialization,
    freeze_backbone,
    head_prefix,
    set_training_mode,
    unfreeze_all,
)
from src.seed import set_global_seeds
from src.utils import collect_environment, select_device, write_json

RESOLUTIONS = (32, 64, 96, 128, 224)
BANNER = "=" * 74


def _as_logits(output) -> torch.Tensor:
    """Main logits regardless of GoogLeNet's train-mode output type."""
    return output if isinstance(output, torch.Tensor) else output.logits


def section(title: str) -> None:
    print(f"\n{BANNER}\n{title}\n{BANNER}")


def check(report: dict, name: str, passed: bool, detail: object = None) -> bool:
    report[name] = bool(passed)
    marker = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail is not None else ""
    print(f"  [{marker}] {name}{suffix}")
    return bool(passed)


def main() -> int:
    config = load_config()
    set_global_seeds(config.reproducibility.seed)
    device = select_device()
    size = config.image.size
    results: dict = {"environment": collect_environment(), "checks": {}, "details": {}}
    checks = results["checks"]

    section("1. ENVIRONMENT AND TARGET RESOLUTION")
    print(f"  device                 {device}")
    print(f"  cuda_available         {torch.cuda.is_available()}")
    print(f"  final training size    {size}x{size}")
    print(f"  smoke-test size        {config.image.smoke_test_size}x"
          f"{config.image.smoke_test_size}")
    print(f"  architectures          {', '.join(ARCHITECTURES)}")
    print(f"  aux_logits (config)    {config.model.googlenet_aux_logits}")
    check(checks, "config_resolution_is_128", size == 128, f"got {size}")

    # -- Build all three with pretrained weights ----------------------------
    section("2. PRETRAINED WEIGHT LOADING AND HEAD REPLACEMENT")
    models: dict[str, nn.Module] = {}
    for arch in ARCHITECTURES:
        # Capture a reference backbone weight BEFORE and AFTER head replacement
        # to prove the head swap does not disturb loaded backbone weights.
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = build_model(
                arch,
                num_classes=config.data.num_classes,
                pretrained=config.model.pretrained,
                aux_logits=config.model.googlenet_aux_logits,
            )
        models[arch] = model
        warn_text = [str(w.message)[:90] for w in caught]

        prefix = head_prefix(arch)
        head_params = [n for n, _ in model.named_parameters() if n.startswith(prefix)]
        head_module = model.fc if arch != "vgg11" else model.classifier[6]

        print(f"\n  --- {arch} ---")
        print(f"    head module          {head_module}")
        print(f"    head param names     {head_params}")
        if warn_text:
            for w in warn_text:
                print(f"    WARNING              {w}")
        check(checks, f"{arch}_head_out_features_is_10",
              head_module.out_features == config.data.num_classes,
              f"out_features={head_module.out_features}")
        check(checks, f"{arch}_head_bias_present", head_module.bias is not None)

    # Prove pretrained weights are genuinely loaded: compare a backbone tensor
    # against a freshly downloaded reference state_dict.
    section("3. PRETRAINED WEIGHTS ARE GENUINELY PRESENT (not random init)")
    from torchvision.models import (
        GoogLeNet_Weights,
        ResNet18_Weights,
        VGG11_Weights,
    )

    reference = {
        "googlenet": (GoogLeNet_Weights.IMAGENET1K_V1, "conv1.conv.weight"),
        "resnet18": (ResNet18_Weights.IMAGENET1K_V1, "conv1.weight"),
        "vgg11": (VGG11_Weights.IMAGENET1K_V1, "features.0.weight"),
    }
    for arch, (weights, key) in reference.items():
        ref_sd = weights.get_state_dict(progress=False)
        live = dict(models[arch].named_parameters())[key].detach()
        identical = torch.equal(live, ref_sd[key])
        print(f"\n  --- {arch} : {key} ---")
        print(f"    checkpoint mean/std  {ref_sd[key].mean():+.6f} / "
              f"{ref_sd[key].std():.6f}")
        print(f"    live model mean/std  {live.mean():+.6f} / {live.std():.6f}")
        check(checks, f"{arch}_backbone_matches_official_checkpoint", identical)

        # Missing/unexpected key audit against the official checkpoint.
        model_keys = set(models[arch].state_dict().keys())
        ckpt_keys = set(ref_sd.keys())
        prefix = head_prefix(arch)
        missing = {k for k in ckpt_keys - model_keys if not k.startswith(prefix)}
        missing = {k for k in missing if not k.startswith("aux")}
        print(f"    backbone keys missing from model: "
              f"{sorted(missing) if missing else 'none'}")
        check(checks, f"{arch}_no_missing_backbone_keys", not missing)

    # -- Head initialisation ------------------------------------------------
    section("4. CLASSIFIER HEAD INITIALISATION")
    for arch in ARCHITECTURES:
        info = describe_head_initialization(models[arch], arch)
        print(f"\n  --- {arch} ---")
        print(f"    mechanism            {info['mechanism']}")
        print(f"    custom init applied  {info['custom_initialization_applied']}")
        for name, s in info["parameters"].items():
            print(f"    {name:22s} shape={s['shape']} mean={s['mean']:+.5f} "
                  f"std={s['std']:.5f} range=[{s['min']:+.4f}, {s['max']:+.4f}]")
        results["details"].setdefault("head_init", {})[arch] = info
        # A randomly initialised head is near-zero-mean and small; a pretrained
        # 1000-class head would not have 10 outputs at all.
        wstats = info["parameters"][f"{head_prefix(arch)}weight"]
        check(checks, f"{arch}_head_is_randomly_initialized",
              abs(wstats["mean"]) < 0.01 and wstats["std"] > 0.0)

    # -- Resolution sweep ---------------------------------------------------
    section("5. FORWARD PASS ACROSS RESOLUTIONS (eval mode)")
    shape_table: dict = {}
    for arch in ARCHITECTURES:
        model = models[arch].to(device).eval()
        shape_table[arch] = {}
        print(f"\n  --- {arch} ---")
        for res in RESOLUTIONS:
            x = torch.randn(2, 3, res, res, device=device)
            try:
                with torch.no_grad():
                    out = model(x)
                shape = tuple(out.shape)
                shape_table[arch][res] = str(shape)
                ok = shape == (2, config.data.num_classes)
                print(f"    {res:3d}px -> {str(shape):12s} "
                      f"dtype={out.dtype} finite={bool(torch.isfinite(out).all())}")
                if res == size:
                    check(checks, f"{arch}_forward_at_{size}px_is_2x10", ok, shape)
            except Exception as exc:  # noqa: BLE001 - reporting, not swallowing
                shape_table[arch][res] = f"FAIL: {type(exc).__name__}"
                print(f"    {res:3d}px -> FAIL {type(exc).__name__}: {str(exc)[:70]}")
    results["details"]["output_shapes_by_resolution"] = shape_table

    # -- Train/eval behaviour ----------------------------------------------
    section(f"6. TRAIN / EVAL BEHAVIOUR AT {size}x{size}")
    for arch in ARCHITECTURES:
        model = models[arch]
        x = torch.randn(2, 3, size, size, device=device)
        print(f"\n  --- {arch} ---")

        model.train()
        out_train = model(x)
        train_type = type(out_train).__name__
        train_tensor = isinstance(out_train, torch.Tensor)
        print(f"    train() output type  {train_type}")
        check(checks, f"{arch}_train_output_is_tensor", train_tensor, train_type)

        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        print(f"    eval()  output type  {type(out_eval).__name__} "
              f"shape={tuple(out_eval.shape)}")
        check(checks, f"{arch}_eval_output_is_2x10",
              tuple(out_eval.shape) == (2, config.data.num_classes))
        check(checks, f"{arch}_eval_output_finite",
              bool(torch.isfinite(out_eval).all()))
        check(checks, f"{arch}_eval_output_is_float32",
              out_eval.dtype == torch.float32, out_eval.dtype)

        # Train-mode stochasticity comes from Dropout only. BatchNorm in train
        # mode uses batch statistics, which are identical for a repeated input,
        # so a net without Dropout is legitimately deterministic here.
        # ResNet18 has no Dropout; GoogLeNet and VGG11 do.
        n_dropout = sum(1 for m in model.modules()
                        if isinstance(m, (nn.Dropout, nn.Dropout2d)))
        model.train()
        a_t = _as_logits(model(x))
        b_t = _as_logits(model(x))
        stochastic = not torch.equal(a_t, b_t)
        model.eval()
        with torch.no_grad():
            c, d = model(x), model(x)
        print(f"    dropout layers       {n_dropout}")
        print(f"    train mode varies    {stochastic} "
              f"({'expected: Dropout present' if n_dropout else 'expected: no Dropout'})")
        print(f"    eval mode repeatable {torch.equal(c, d)}")
        check(checks, f"{arch}_train_stochasticity_matches_dropout_presence",
              stochastic == (n_dropout > 0),
              f"dropout={n_dropout}, varies={stochastic}")
        check(checks, f"{arch}_eval_is_deterministic", torch.equal(c, d))

    # -- GoogLeNet aux specifics -------------------------------------------
    section("7. GOOGLENET AUXILIARY CLASSIFIER BEHAVIOUR")
    aux_info = describe_googlenet_aux()
    g = models["googlenet"]
    print(f"  aux_logits attribute   {g.aux_logits}")
    print(f"  model.aux1 / aux2      {g.aux1} / {g.aux2}")
    print(f"  transform_input        {g.transform_input}")
    check(checks, "googlenet_aux_disabled", g.aux_logits is False)
    check(checks, "googlenet_aux_modules_are_none",
          g.aux1 is None and g.aux2 is None)
    check(checks, "googlenet_transform_input_enabled", g.transform_input is True,
          "required: pretrained weights expect the internal [-1,1] remap")

    x = torch.randn(2, 3, size, size, device=device)
    g.train()
    t_out = g(x)
    g.eval()
    with torch.no_grad():
        e_out = g(x)
    print(f"  train() returns        {type(t_out).__name__} "
          f"shape={tuple(t_out.shape)}")
    print(f"  eval()  returns        {type(e_out).__name__} "
          f"shape={tuple(e_out.shape)}")
    check(checks, "googlenet_train_returns_plain_tensor",
          isinstance(t_out, torch.Tensor))
    check(checks, "googlenet_ensemble_compatible_output",
          tuple(e_out.shape) == (2, config.data.num_classes))

    # Control: with aux ENABLED the interface differs. Built untrained purely to
    # demonstrate the behavioural contrast; discarded immediately.
    g_aux = build_model("googlenet", 10, pretrained=False, aux_logits=True)
    g_aux.train()
    aux_out = g_aux(torch.randn(2, 3, size, size))
    g_aux.eval()
    with torch.no_grad():
        aux_eval = g_aux(torch.randn(2, 3, size, size))
    print(f"\n  [control, aux_logits=True, NOT pretrained]")
    print(f"    train() returns      {type(aux_out).__name__} with fields "
          f"{list(aux_out._fields)}")
    print(f"    eval()  returns      {type(aux_eval).__name__} "
          f"shape={tuple(aux_eval.shape)}")
    check(checks, "aux_enabled_train_returns_namedtuple",
          type(aux_out).__name__ == "GoogLeNetOutputs")
    check(checks, "aux_enabled_eval_still_returns_tensor",
          isinstance(aux_eval, torch.Tensor))
    results["details"]["googlenet_aux"] = aux_info
    del g_aux

    # -- Parameter counts ---------------------------------------------------
    section("8. PARAMETER COUNTS")
    param_table: dict = {}
    print(f"  {'model':<11}{'total':>14}{'backbone':>14}{'head':>10}"
          f"{'frozen-phase1':>16}{'unfrozen':>14}")
    print("  " + "-" * 77)
    for arch in ARCHITECTURES:
        model = models[arch]
        unfreeze_all(model)
        full = count_parameters(model, arch)
        freeze_backbone(model, arch)
        frozen = count_parameters(model, arch)
        param_table[arch] = {"full_unfrozen": full.as_dict(),
                             "backbone_frozen": frozen.as_dict()}
        print(f"  {arch:<11}{full.total:>14,}{full.backbone:>14,}{full.head:>10,}"
              f"{frozen.trainable:>16,}{full.trainable:>14,}")
        check(checks, f"{arch}_head_equals_phase1_trainable",
              frozen.trainable == full.head, f"{frozen.trainable} == {full.head}")
        check(checks, f"{arch}_total_equals_backbone_plus_head",
              full.total == full.backbone + full.head)
    results["details"]["parameter_counts"] = param_table

    # -- Gradient flow ------------------------------------------------------
    section("9. GRADIENT FLOW")
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, config.data.num_classes, (2,), device=device)
    grad_table: dict = {}
    for arch in ARCHITECTURES:
        model = models[arch].to(device)
        prefix = head_prefix(arch)
        print(f"\n  --- {arch} ---")

        # Phase 1: backbone frozen.
        freeze_backbone(model, arch)
        set_training_mode(model, training=True, backbone_frozen=True)
        model.zero_grad(set_to_none=True)
        out = _as_logits(model(torch.randn(2, 3, size, size, device=device)))
        loss_frozen = criterion(out, labels)
        loss_frozen.backward()

        head_with_grad = [n for n, p in model.named_parameters()
                          if n.startswith(prefix) and p.grad is not None]
        backbone_with_grad = [n for n, p in model.named_parameters()
                              if not n.startswith(prefix) and p.grad is not None]
        print(f"    PHASE 1 loss                 {loss_frozen.item():.6f}")
        print(f"    head params with grad        {len(head_with_grad)} "
              f"{head_with_grad}")
        print(f"    backbone params with grad    {len(backbone_with_grad)}")
        check(checks, f"{arch}_phase1_head_has_gradients", len(head_with_grad) > 0)
        check(checks, f"{arch}_phase1_backbone_has_no_gradients",
              len(backbone_with_grad) == 0)
        check(checks, f"{arch}_phase1_output_differentiable",
              loss_frozen.requires_grad and loss_frozen.grad_fn is not None)

        # Verify BN running stats really are held still while frozen.
        bn_buffers = {k: v.clone() for k, v in model.state_dict().items()
                      if "running_mean" in k or "running_var" in k}
        if bn_buffers:
            set_training_mode(model, training=True, backbone_frozen=True)
            model(torch.randn(2, 3, size, size, device=device))
            after = model.state_dict()
            unchanged = all(torch.equal(v, after[k]) for k, v in bn_buffers.items())
            print(f"    BN buffers ({len(bn_buffers):3d}) unchanged  {unchanged}")
            check(checks, f"{arch}_phase1_bn_running_stats_frozen", unchanged)
        else:
            print(f"    BN buffers                   none (VGG11 has no BatchNorm)")

        # Phase 2: everything unfrozen.
        unfreeze_all(model)
        set_training_mode(model, training=True, backbone_frozen=False)
        model.zero_grad(set_to_none=True)
        out = _as_logits(model(torch.randn(2, 3, size, size, device=device)))
        loss_full = criterion(out, labels)
        loss_full.backward()

        n_total = sum(1 for _ in model.parameters())
        n_grad = sum(1 for p in model.parameters() if p.grad is not None)
        first_backbone = next(n for n, _ in model.named_parameters()
                              if not n.startswith(prefix))
        fb_grad = dict(model.named_parameters())[first_backbone].grad
        print(f"    PHASE 2 loss                 {loss_full.item():.6f}")
        print(f"    params with grad             {n_grad}/{n_total}")
        print(f"    {first_backbone} grad norm  {fb_grad.norm().item():.6e}")
        check(checks, f"{arch}_phase2_all_params_have_gradients", n_grad == n_total)
        check(checks, f"{arch}_phase2_backbone_grad_is_nonzero",
              fb_grad.abs().sum().item() > 0)
        check(checks, f"{arch}_phase2_gradients_are_finite",
              all(torch.isfinite(p.grad).all() for p in model.parameters()
                  if p.grad is not None))
        grad_table[arch] = {"phase1_loss": loss_frozen.item(),
                            "phase2_loss": loss_full.item(),
                            "phase1_head_params_with_grad": len(head_with_grad),
                            "phase1_backbone_params_with_grad": len(backbone_with_grad),
                            "phase2_params_with_grad": f"{n_grad}/{n_total}"}
        model.zero_grad(set_to_none=True)
    results["details"]["gradient_flow"] = grad_table

    # -- Discriminative LR groups ------------------------------------------
    section("10. DISCRIMINATIVE LEARNING-RATE PARAMETER GROUPS")
    b_lr = config.training.learning_rate
    h_lr = config.training.head_learning_rate
    print(f"  configured backbone lr {b_lr}   head lr {h_lr}  "
          f"(ratio {h_lr / b_lr:.0f}x)  [PLACEHOLDER values, tuned in Stage 3]")
    for arch in ARCHITECTURES:
        model = models[arch]
        unfreeze_all(model)
        groups_full = build_param_groups(model, arch, b_lr, h_lr)
        freeze_backbone(model, arch)
        groups_frozen = build_param_groups(model, arch, b_lr, h_lr)
        print(f"\n  --- {arch} ---")
        for g_ in groups_full:
            n = sum(p.numel() for p in g_["params"])
            print(f"    unfrozen: {g_['name']:<9} lr={g_['lr']:<7} params={n:,}")
        names = [g_["name"] for g_ in groups_frozen]
        n = sum(p.numel() for g_ in groups_frozen for p in g_["params"])
        print(f"    frozen  : groups={names} params={n:,}")
        check(checks, f"{arch}_frozen_yields_head_group_only", names == ["head"])
        check(checks, f"{arch}_unfrozen_yields_two_groups",
              [g_["name"] for g_ in groups_full] == ["backbone", "head"])

    # -- Device handling ----------------------------------------------------
    section("11. DEVICE HANDLING")
    print(f"  select_device() -> {device}")
    m = models["resnet18"]
    m.to(torch.device("cpu"))
    on_cpu = all(p.device.type == "cpu" for p in m.parameters())
    check(checks, "model_moves_to_cpu", on_cpu)
    with torch.no_grad():
        m.eval()
        cpu_out = m(torch.randn(2, 3, size, size))
    check(checks, "cpu_forward_works", tuple(cpu_out.shape) == (2, 10))

    if torch.cuda.is_available():
        m.to(torch.device("cuda"))
        check(checks, "model_moves_to_cuda",
              all(p.device.type == "cuda" for p in m.parameters()))
        with torch.no_grad():
            cuda_out = m(torch.randn(2, 3, size, size, device="cuda"))
        check(checks, "cuda_forward_works", tuple(cuda_out.shape) == (2, 10))
        m.to(torch.device("cpu"))
    else:
        print("  CUDA not available on this machine - CUDA forward NOT tested.")
        print("  This is a real gap, not a pass. Must be exercised on the GPU host.")
        results["details"]["cuda_tested"] = False

    # -- Serialization smoke test ------------------------------------------
    section("12. STATE_DICT SAVE / LOAD SMOKE TEST")
    source = build_model("resnet18", 10, pretrained=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "resnet18_smoke.pt"
        torch.save(source.state_dict(), path)
        size_kb = path.stat().st_size / 1024
        target = build_model("resnet18", 10, pretrained=False)
        before = torch.equal(
            source.state_dict()["fc.weight"], target.state_dict()["fc.weight"]
        )
        target.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        after = all(
            torch.equal(v, target.state_dict()[k])
            for k, v in source.state_dict().items()
        )
        print(f"  written to temp dir    {path.name} ({size_kb:,.0f} KB)")
        print(f"  differed before load   {not before}")
        print(f"  all tensors equal      {after}")
        check(checks, "distinct_models_differ_before_load", not before)
        check(checks, "state_dict_roundtrip_is_exact", after)
    print(f"  temp dir removed       {not Path(path).exists()}")
    check(checks, "smoke_artifact_removed", not Path(path).exists())

    # -- Summary ------------------------------------------------------------
    section("SUMMARY")
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    failed = [k for k, v in checks.items() if not v]
    print(f"  {passed}/{total} checks passed")
    if failed:
        print("\n  FAILED:")
        for name in failed:
            print(f"    - {name}")
    out_path = write_json(results, config.paths.results_dir / "model_verification.json")
    print(f"\n  report written to {out_path}")
    print("\n  NOTE: no training was performed. No accuracy exists or is implied.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
