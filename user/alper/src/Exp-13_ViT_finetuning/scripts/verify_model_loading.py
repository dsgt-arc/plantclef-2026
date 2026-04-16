"""Smoke test: verify both ViT-Large variants load correctly with timm.

Checks:
  - Model creation with num_classes=7806 at the configured input resolution
    (518 for DINOv2 / 512 for DINOv3 -- patch16 divisibility)
  - forward_features() returns token sequence [B, seq, D]
  - forward_head(feats, pre_logits=True) returns pooled [B, D]  (CLS / global pool)
  - forward_head(feats) returns logits [B, 7806]
  - Freeze-backbone logic (prefix='head') leaves backbone frozen
  - Pretrained weights loaded (first-layer mean > 0) when --pretrained is passed

Usage:
  python scripts/verify_model_loading.py               # architecture-only smoke test
  python scripts/verify_model_loading.py --pretrained  # download weights and verify
"""
import sys

import torch
import timm


VARIANTS = [
    # (model_name, input_resolution, expected_embed_dim, expected_params)
    ("vit_large_patch14_reg4_dinov2.lvd142m", 518, 1024, 304_000_000),
    ("vit_large_patch16_dinov3.lvd1689m",     512, 1024, 303_000_000),
]

NUM_CLASSES = 7806


def verify_variant(
    model_name: str,
    input_size: int,
    expected_dim: int,
    expected_params: int,
    load_pretrained: bool = False,
) -> bool:
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} @ {input_size}x{input_size}")
    print(f"{'='*60}")

    try:
        model = timm.create_model(
            model_name,
            pretrained=load_pretrained,
            num_classes=NUM_CLASSES,
            img_size=input_size,
        )
        model.eval()
    except Exception as e:
        print(f"  FAIL: Could not create model: {e}")
        return False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} (expected ~{expected_params:,})")
    print(f"  model.num_features: {model.num_features}")
    assert model.num_features == expected_dim, (
        f"Expected embed_dim={expected_dim}, got {model.num_features}"
    )

    # Head should be a Linear (timm ViT) or ClassifierHead
    head = model.head
    print(f"  Head type: {type(head).__name__}")
    if hasattr(head, "out_features"):
        assert head.out_features == NUM_CLASSES
        assert head.in_features == expected_dim
    elif hasattr(head, "fc"):
        assert head.fc.out_features == NUM_CLASSES
        assert head.fc.in_features == expected_dim

    # Forward pass on a small batch
    x = torch.randn(2, 3, input_size, input_size)
    with torch.no_grad():
        features = model.forward_features(x)
        print(f"  forward_features: {tuple(features.shape)}")
        # ViT returns [B, tokens, D] -- 3 dims
        assert features.ndim == 3, (
            f"Expected 3D token sequence, got {features.shape}"
        )
        assert features.shape[-1] == expected_dim

        # timm-standard path: forward_head(pre_logits=True) + forward_head()
        embeddings = model.forward_head(features, pre_logits=True)
        print(f"  forward_head(pre_logits=True): {tuple(embeddings.shape)}")
        assert embeddings.shape == (2, expected_dim), (
            f"Expected (2, {expected_dim}), got {embeddings.shape}"
        )

        logits = model.forward_head(features)
        print(f"  forward_head(): {tuple(logits.shape)}")
        assert logits.shape == (2, NUM_CLASSES)

        # Extract-script path (mirrors Exp-7 / extract_vit_embeddings.py):
        # direct CLS indexing + model.head(cls). Must match what the inference
        # pipeline actually runs.
        cls_tokens = features[:, 0, :]
        print(f"  features[:, 0, :] (CLS): {tuple(cls_tokens.shape)}")
        assert cls_tokens.shape == (2, expected_dim)

        cls_logits = model.head(cls_tokens)
        print(f"  model.head(cls): {tuple(cls_logits.shape)}")
        assert cls_logits.shape == (2, NUM_CLASSES)

        # Full forward: model(x) == forward_features + forward_head (training path)
        full_logits = model(x)
        print(f"  model(x): {tuple(full_logits.shape)}")
        assert full_logits.shape == (2, NUM_CLASSES)

    # Freeze-backbone simulation (prefix='head')
    for name, param in model.named_parameters():
        if name.startswith("head"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable
    head_frac = trainable / total_params
    print(f"  Freeze test: {frozen:,} frozen, {trainable:,} trainable "
          f"({head_frac:.2%} is head)")
    assert trainable > 0, "Head has zero trainable params"
    assert frozen > 0, "Nothing got frozen"

    if load_pretrained:
        first_param = next(model.parameters())
        mean_val = first_param.data.abs().mean().item()
        print(f"  Pretrained check: first param |mean|={mean_val:.6f} (>0 expected)")
        assert mean_val > 1e-4, (
            "Weights look random; pretrained loading may have failed"
        )

    print(f"  PASS")
    return True


def main():
    print(f"timm version: {timm.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    load_pretrained = "--pretrained" in sys.argv

    all_pass = True
    for model_name, input_size, expected_dim, expected_params in VARIANTS:
        ok = verify_variant(
            model_name, input_size, expected_dim, expected_params, load_pretrained
        )
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
