"""Tests for experiments/003-tileq-decoder/train.py.

Targets only the testable pure helpers (lr schedule, validation eval).
The end-to-end training loop is exercised by sanity_check.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_MODULE_PATH = Path(__file__).parent.parent.parent / "experiments/003-tileq-decoder/train.py"
_spec = importlib.util.spec_from_file_location("train_module", _MODULE_PATH)
train_module = importlib.util.module_from_spec(_spec)
sys.modules["train_module"] = train_module
_spec.loader.exec_module(train_module)


def _make_optimizer(peak_lr: float = 3e-4):
    """Optimizer's base lr must equal peak_lr; LambdaLR multiplies that base."""
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([p], lr=peak_lr)


def test_lr_schedule_warmup_starts_at_zero():
    opt = _make_optimizer()
    sched = train_module.make_lr_schedule(
        opt, warmup_steps=100, total_steps=1000, peak_lr=3e-4, min_lr=1e-5
    )
    # step 0 -> lr should be 0 (or tiny)
    assert sched.get_last_lr()[0] == 0.0


def test_lr_schedule_warmup_reaches_peak():
    peak = 3e-4
    opt = _make_optimizer(peak)
    sched = train_module.make_lr_schedule(
        opt, warmup_steps=10, total_steps=100, peak_lr=peak, min_lr=1e-5
    )
    for _ in range(10):
        opt.step()
        sched.step()
    assert abs(sched.get_last_lr()[0] - peak) < 1e-7


def test_lr_schedule_cosine_decays_to_min_lr():
    peak, min_lr = 3e-4, 1e-5
    opt = _make_optimizer(peak)
    sched = train_module.make_lr_schedule(
        opt, warmup_steps=10, total_steps=100, peak_lr=peak, min_lr=min_lr
    )
    for _ in range(100):
        opt.step()
        sched.step()
    final = sched.get_last_lr()[0]
    assert abs(final - min_lr) < 1e-8


def test_lr_schedule_monotone_decreasing_after_warmup():
    peak = 3e-4
    opt = _make_optimizer(peak)
    sched = train_module.make_lr_schedule(
        opt, warmup_steps=5, total_steps=100, peak_lr=peak, min_lr=1e-5
    )
    lrs = []
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    # After warmup completes (step >= 5), cosine portion should be monotone non-increasing
    post_warmup = lrs[5:]
    for i in range(1, len(post_warmup)):
        assert post_warmup[i] <= post_warmup[i - 1] + 1e-12


def test_evaluate_returns_finite_values():
    """evaluate() should not return NaN/Inf for a fresh model on random tiles."""
    torch.manual_seed(0)
    from plantclef.asl_loss import AsymmetricLoss
    from plantclef.tileq_decoder import TileQDecoder

    model = TileQDecoder(n_classes=100, n_groups=10, d_in=64, d_model=32, n_heads=4)
    val_tiles = torch.randn(8, 4, 64)
    val_labels = torch.zeros(8, 100)
    val_labels[:, :3] = 1.0
    loss_fn = AsymmetricLoss()

    val_loss, val_acc = train_module.evaluate(
        model, val_tiles, val_labels, loss_fn,
        device=torch.device("cpu"),
        batch_size=4,
        autocast_dtype=None,
        top_k=10,
    )
    assert np.isfinite(val_loss)
    assert 0.0 <= val_acc <= 1.0
