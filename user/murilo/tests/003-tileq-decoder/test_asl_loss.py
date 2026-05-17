"""Tests for plantclef.asl_loss.AsymmetricLoss.

Verifies the spec §2.2 properties:
- Reduces to BCE when gamma_neg=gamma_pos=0 and margin=0.
- Margin clips negative loss to zero when p < margin (handles label noise).
- Asymmetric focal up-weights hard negatives relative to easy negatives.
- Numerical stability at extreme logits.
- Backward produces non-NaN gradients.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from plantclef.asl_loss import AsymmetricLoss


def test_default_forward_shape_and_sign():
    loss_fn = AsymmetricLoss()
    logits = torch.randn(8, 100)
    targets = torch.zeros(8, 100)
    targets[:, :3] = 1  # 3 positives per row
    out = loss_fn(logits, targets)
    assert out.dim() == 0
    assert out.item() > 0  # negative-log-likelihood is positive


def test_reduces_to_bce_when_gamma_zero_margin_zero():
    """With gamma_neg=gamma_pos=0 and margin=0, ASL should equal BCE."""
    loss_fn = AsymmetricLoss(
        gamma_neg=0.0, gamma_pos=0.0, margin=0.0, reduction="mean_per_sample"
    )
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    targets = (torch.randn(4, 50) > 0).float()

    asl = loss_fn(logits, targets).item()
    # mean_per_sample = sum(class) / batch → match BCE-sum / batch
    bce_sum = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum").item()
    bce_per_sample = bce_sum / 4

    assert abs(asl - bce_per_sample) < 1e-3, f"asl={asl}, bce={bce_per_sample}"


def test_margin_zeros_easy_negatives():
    """If p < margin and y=0, loss for that element must be effectively zero."""
    loss_fn = AsymmetricLoss(
        gamma_neg=4.0, gamma_pos=0.0, margin=0.05, reduction="none"
    )
    # logit = -10 → sigmoid ≈ 4.5e-5 < 0.05 (well inside margin)
    logits = torch.tensor([[-10.0]])
    targets = torch.tensor([[0.0]])
    out = loss_fn(logits, targets)
    assert out.item() < 1e-6, f"easy-negative loss should be ~0, got {out.item()}"


def test_margin_does_not_zero_hard_negatives():
    """If p > margin and y=0, loss must NOT be zero."""
    loss_fn = AsymmetricLoss(
        gamma_neg=4.0, gamma_pos=0.0, margin=0.05, reduction="none"
    )
    # logit = +2 → sigmoid ≈ 0.88, well above margin
    logits = torch.tensor([[2.0]])
    targets = torch.tensor([[0.0]])
    out = loss_fn(logits, targets)
    assert out.item() > 0.1


def test_asl_amplifies_hard_vs_easy_negative_ratio():
    """ASL with gamma_neg=4 sharpens the hard/easy ratio compared to plain BCE.

    Easy negative (p ≈ 0.047) and hard negative (p ≈ 0.88), both y=0.
    ASL should produce a much larger hard/easy loss ratio than BCE.
    """
    bce = AsymmetricLoss(gamma_neg=0.0, gamma_pos=0.0, margin=0.0, reduction="none")
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, margin=0.05, reduction="none")

    easy_logit = torch.tensor([[-3.0]])  # p ≈ 0.047
    hard_logit = torch.tensor([[2.0]])   # p ≈ 0.88
    y_neg = torch.tensor([[0.0]])

    bce_easy = bce(easy_logit, y_neg).item()
    bce_hard = bce(hard_logit, y_neg).item()
    asl_easy = asl(easy_logit, y_neg).item()
    asl_hard = asl(hard_logit, y_neg).item()

    # margin (m=0.05) zeros the easy negative for ASL → ratio explodes
    assert asl_easy < 1e-6
    bce_ratio = bce_hard / bce_easy
    assert bce_hard > 0 and bce_easy > 0  # sanity
    # ASL ratio is effectively infinite (easy → 0); confirm asl_hard > bce_hard scaling
    assert asl_hard > 0.1
    # Hard negative under ASL gets focused: weight (p_-)^4 ≈ 0.88^4 ≈ 0.6 of BCE
    # — the *amplification* shows up in the easy/hard contrast, not absolute hard loss.
    assert (asl_hard / max(asl_easy, 1e-12)) > bce_ratio * 100


def test_grad_flow_no_nan():
    loss_fn = AsymmetricLoss()
    logits = torch.randn(4, 100, requires_grad=True)
    targets = torch.zeros(4, 100)
    targets[:, :5] = 1
    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()


def test_extreme_logits_no_nan():
    """Logits at ±100 must not produce NaN or Inf."""
    loss_fn = AsymmetricLoss()
    logits = torch.tensor([[100.0, -100.0, 0.0, 50.0, -50.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_loss_decreases_with_correct_predictions():
    """Sanity: target-aligned logits give lower loss than random ones."""
    loss_fn = AsymmetricLoss()
    targets = torch.zeros(2, 10)
    targets[0, [1, 3]] = 1
    targets[1, [5, 7]] = 1

    torch.manual_seed(0)
    logits_bad = torch.randn(2, 10)
    loss_bad = loss_fn(logits_bad, targets).item()

    logits_good = torch.where(
        targets == 1, torch.full_like(targets, 5.0), torch.full_like(targets, -5.0)
    )
    loss_good = loss_fn(logits_good, targets).item()

    assert loss_good < loss_bad


def test_reduction_modes():
    """All four reduction modes produce consistent shapes / values."""
    torch.manual_seed(0)
    logits = torch.randn(3, 20)
    targets = (torch.randn(3, 20) > 0).float()

    none_loss = AsymmetricLoss(reduction="none")(logits, targets)
    sum_loss = AsymmetricLoss(reduction="sum")(logits, targets)
    mean_loss = AsymmetricLoss(reduction="mean")(logits, targets)
    mps_loss = AsymmetricLoss(reduction="mean_per_sample")(logits, targets)

    assert none_loss.shape == (3, 20)
    assert sum_loss.dim() == 0
    assert torch.allclose(sum_loss, none_loss.sum())
    assert torch.allclose(mean_loss, none_loss.mean())
    assert torch.allclose(mps_loss, none_loss.sum(dim=-1).mean())


def test_invalid_reduction_raises():
    with pytest.raises(ValueError):
        AsymmetricLoss(reduction="banana")
