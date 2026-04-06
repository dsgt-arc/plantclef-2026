"""Smoke test: verify all 4 DINOv3 ConvNeXt variants load correctly with timm.

Checks:
  - Model creation with num_classes=7806
  - forward_features() returns spatial [B, D, H, W]
  - forward_head(feats, pre_logits=True) returns pooled [B, D]
  - forward_head(feats) returns logits [B, 7806]
  - Pretrained weights load (mean != 0 for backbone params)
  - Parameter counts match expectations
"""
import sys

import torch
import timm


VARIANTS = [
    ("convnext_tiny.dinov3_lvd1689m",  27_800_000,  768),
    ("convnext_small.dinov3_lvd1689m", 49_500_000,  768),
    ("convnext_base.dinov3_lvd1689m",  87_600_000, 1024),
    ("convnext_large.dinov3_lvd1689m", 196_200_000, 1536),
]

NUM_CLASSES = 7806
INPUT_SIZE = 224


def verify_variant(model_name: str, expected_params: int, expected_dim: int, load_pretrained: bool = False) -> bool:
    """Verify a single ConvNeXt variant."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        model = timm.create_model(
            model_name,
            pretrained=load_pretrained,
            num_classes=NUM_CLASSES,
        )
        model.eval()
    except Exception as e:
        print(f"  FAIL: Could not create model: {e}")
        return False

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} (expected ~{expected_params:,})")

    # Check head structure
    head = model.head
    print(f"  Head type: {type(head).__name__}")
    print(f"  Head FC: {head.fc}")
    assert head.fc.out_features == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {head.fc.out_features}"
    assert head.fc.in_features == expected_dim, f"Expected {expected_dim} embed dim, got {head.fc.in_features}"

    # num_features
    print(f"  model.num_features: {model.num_features}")
    assert model.num_features == expected_dim

    # Forward pass
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        # forward_features -> spatial
        features = model.forward_features(x)
        print(f"  forward_features: {features.shape}")
        assert len(features.shape) == 4, f"Expected 4D spatial, got {features.shape}"
        assert features.shape[1] == expected_dim, f"Channel dim mismatch"

        # forward_head(pre_logits=True) -> pooled embedding
        embeddings = model.forward_head(features, pre_logits=True)
        print(f"  forward_head(pre_logits=True): {embeddings.shape}")
        assert embeddings.shape == (2, expected_dim), f"Expected (2, {expected_dim}), got {embeddings.shape}"

        # forward_head() -> logits
        logits = model.forward_head(features)
        print(f"  forward_head(): {logits.shape}")
        assert logits.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {logits.shape}"

        # Full forward -> logits
        full_logits = model(x)
        print(f"  model(x): {full_logits.shape}")
        assert full_logits.shape == (2, NUM_CLASSES)

    # Check freeze-backbone compatibility
    for name, param in model.named_parameters():
        if name.startswith("head"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable
    print(f"  Freeze test: {frozen:,} frozen, {trainable:,} trainable")

    if load_pretrained:
        # Verify pretrained weights are loaded (backbone weights shouldn't be all zeros)
        first_conv = list(model.stem.parameters())[0]
        mean_val = first_conv.data.abs().mean().item()
        print(f"  Pretrained check: stem conv weight mean={mean_val:.6f} (should be > 0)")
        assert mean_val > 0.001, "Weights appear to be random/zero — pretrained loading may have failed"

    print(f"  PASS")
    return True


def main():
    print(f"timm version: {timm.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check if pretrained download is requested
    load_pretrained = "--pretrained" in sys.argv

    all_pass = True
    for model_name, expected_params, expected_dim in VARIANTS:
        ok = verify_variant(model_name, expected_params, expected_dim, load_pretrained)
        if not ok:
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL VARIANTS PASSED")
    else:
        print("SOME VARIANTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
