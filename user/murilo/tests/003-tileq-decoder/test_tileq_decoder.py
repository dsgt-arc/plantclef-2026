"""Tests for plantclef.tileq_decoder.TileQDecoder.

Spec §2.1 properties:
- Forward shape: (B, T=16, 768) -> (B, 7806).
- Parameter count in the ~4 M range (spec rounds; allow 3.5-5.5 M).
- Group-FFN trim: 7806 / 64 = 122 per group -> 64*122 = 7808 -> trim to 7806.
- Backward computes finite gradients on every parameter.
- Forward is non-constant in the input.
- Forward is stable at both train (L2 ≈ 18.6) and test (L2 ≈ 25.4) input scales,
  matching the documented domain shift between train CLS and test tile CLS.
- Smaller configs work (parameterized for sanity checks).
"""

from __future__ import annotations

import pytest
import torch

from plantclef.tileq_decoder import TileQDecoder


def test_default_forward_shape():
    model = TileQDecoder()
    x = torch.randn(2, 16, 768)
    out = model(x)
    assert out.shape == (2, 7806)
    assert out.dtype == torch.float32


def test_param_count_around_4M():
    model = TileQDecoder()
    n_params = sum(p.numel() for p in model.parameters())
    # Computed: ~4.51 M (proj 295k + queries 25k + attn 591k + norms 1.5k +
    #                     ffn 591k + group_heads ~3.0M + group_bias 8k).
    # Spec rounds to "~4 M" — allow 3.5–5.5 M as the acceptance band.
    assert 3_500_000 < n_params < 5_500_000, (
        f"expected ~4 M params, got {n_params}"
    )


def test_group_ffn_classes_per_group():
    """ceil(7806 / 64) = 122 → group_heads shape (64, 384, 122) → trim to 7806."""
    model = TileQDecoder(n_classes=7806, n_groups=64)
    assert model.classes_per_group == 122
    assert model.group_heads.shape == (64, 384, 122)
    assert model.group_bias.shape == (64, 122)
    # 64 * 122 = 7808; output is trimmed to 7806
    assert model.n_groups * model.classes_per_group == 7808
    out = model(torch.randn(1, 16, 768))
    assert out.shape == (1, 7806)


def test_forward_no_nan():
    model = TileQDecoder()
    model.eval()
    out = model(torch.randn(4, 16, 768))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_backward_all_params_get_gradient():
    model = TileQDecoder()
    out = model(torch.randn(2, 16, 768))
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


def test_different_inputs_produce_different_outputs():
    """Sanity that the model is not collapsed to a constant function."""
    model = TileQDecoder()
    model.eval()
    x1 = torch.randn(1, 16, 768)
    x2 = torch.randn(1, 16, 768)
    with torch.no_grad():
        o1, o2 = model(x1), model(x2)
    assert not torch.allclose(o1, o2)


def test_input_layout_is_per_tile():
    """Permuting tiles should generally change the output (cross-attn over T)."""
    torch.manual_seed(0)
    model = TileQDecoder()
    model.eval()
    x = torch.randn(1, 16, 768)
    perm = torch.randperm(16)
    x_permuted = x[:, perm, :]
    with torch.no_grad():
        o = model(x)
        op = model(x_permuted)
    # Multi-head attention with no positional encoding is permutation-equivariant
    # in the keys/values, so output should be equal.  This is intentional —
    # a quadrat is a *set* of tiles, not a sequence. Document via the test.
    assert torch.allclose(o, op, atol=1e-5)


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param(18.6, id="train_scale"),    # train CLS mean L2
        pytest.param(25.4, id="test_scale"),     # test tile CLS mean L2
    ],
)
def test_stable_at_documented_input_scales(norm: float):
    """Both train and test CLS scales must produce finite outputs."""
    torch.manual_seed(0)
    model = TileQDecoder()
    model.eval()
    x = torch.randn(2, 16, 768)
    x = x / x.norm(dim=-1, keepdim=True) * norm
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    # And the output magnitudes shouldn't blow up at init (sanity)
    assert out.abs().max().item() < 100, f"logit overflow at L2 scale {norm}"


def test_smaller_config_for_overfit_one_batch():
    """A small config should exist to keep the overfit-one-batch sanity cheap."""
    model = TileQDecoder(
        n_classes=100, n_groups=10, d_in=64, d_model=32, n_heads=4
    )
    out = model(torch.randn(2, 8, 64))
    assert out.shape == (2, 100)
    assert model.classes_per_group == 10
    assert model.n_groups * model.classes_per_group == 100  # exact, no trim


def test_n_classes_indivisible_by_n_groups_trims_correctly():
    """If n_classes < n_groups * classes_per_group, output is trimmed."""
    model = TileQDecoder(
        n_classes=23, n_groups=4, d_in=16, d_model=8, n_heads=2
    )
    # ceil(23/4) = 6 → 4*6 = 24 → trim 1
    assert model.classes_per_group == 6
    out = model(torch.randn(1, 5, 16))
    assert out.shape == (1, 23)


def test_backbone_not_in_module():
    """The frozen ViT-B backbone must not be part of this module."""
    model = TileQDecoder()
    # No timm / DINOv2 / patch-embed style submodules should be present.
    for name, _ in model.named_modules():
        lname = name.lower()
        assert "vit" not in lname
        assert "patch_embed" not in lname
        assert "blocks" not in lname  # ViT uses .blocks for transformer stack
