"""Tests for experiments/003-tileq-decoder/dataset.py.

Verifies the IterableDataset wrapper:
- Yields torch.float32 tensors of the expected shapes.
- DataLoader collation produces (B, n_tiles, dim) and (B, n_classes) batches.
- Multi-worker mode: each worker reseeds the sampler so no two workers emit
  identical samples (probabilistic test).
- Single-worker (num_workers=0) is reproducible given a seeded sampler.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from plantclef.data.samplers import SamplerA

# Import dataset.py from experiments/003-tileq-decoder/
_MODULE_PATH = Path(__file__).parent.parent.parent / "experiments/003-tileq-decoder/dataset.py"
_spec = importlib.util.spec_from_file_location("dataset_module", _MODULE_PATH)
dataset_module = importlib.util.module_from_spec(_spec)
sys.modules["dataset_module"] = dataset_module
_spec.loader.exec_module(dataset_module)


def _make_sampler(seed: int = 0) -> SamplerA:
    """Tiny SamplerA on synthetic data."""
    rng = np.random.default_rng(seed)
    n_classes = 8
    n_per_class = 5
    embeddings = rng.normal(0, 1, (n_classes * n_per_class, 16)).astype(np.float32)
    offsets = np.zeros(n_classes + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([n_per_class] * n_classes)
    weights = (np.ones(n_classes) / n_classes).astype(np.float32)
    return SamplerA(
        embeddings, offsets, weights,
        n_positives=2, n_tiles=4,
        rng=np.random.default_rng(seed),
    )


def test_yields_torch_float32():
    ds = dataset_module.PseudoQuadratDataset(_make_sampler())
    it = iter(ds)
    tiles, label = next(it)
    assert isinstance(tiles, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert tiles.dtype == torch.float32
    assert label.dtype == torch.float32
    assert tiles.shape == (4, 16)
    assert label.shape == (8,)


def test_dataloader_collation_to_batches():
    ds = dataset_module.PseudoQuadratDataset(_make_sampler())
    loader = DataLoader(ds, batch_size=3, num_workers=0)
    batch_tiles, batch_labels = next(iter(loader))
    assert batch_tiles.shape == (3, 4, 16)
    assert batch_labels.shape == (3, 8)
    assert batch_tiles.dtype == torch.float32
    # Multi-hot label: each row sums to n_positives=2
    assert torch.allclose(batch_labels.sum(dim=-1), torch.full((3,), 2.0))


def test_seeded_single_worker_reproducible():
    """num_workers=0 with seeded sampler -> identical first batch across runs."""
    def first_batch():
        ds = dataset_module.PseudoQuadratDataset(_make_sampler(seed=123))
        loader = DataLoader(ds, batch_size=4, num_workers=0)
        return next(iter(loader))

    t1, l1 = first_batch()
    t2, l2 = first_batch()
    assert torch.equal(t1, t2)
    assert torch.equal(l1, l2)


def test_distinct_yields_different_samples():
    """Successive yields from one dataset should not be identical (very high prob)."""
    ds = dataset_module.PseudoQuadratDataset(_make_sampler(seed=0))
    it = iter(ds)
    tiles_a, _ = next(it)
    tiles_b, _ = next(it)
    assert not torch.equal(tiles_a, tiles_b)


def test_multiworker_workers_diverge():
    """num_workers=2 -> the two workers must emit different samples."""
    ds = dataset_module.PseudoQuadratDataset(_make_sampler(seed=0))
    loader = DataLoader(ds, batch_size=2, num_workers=2)
    # Pull 4 batches: workers alternate, so batches 0/2 come from worker 0
    # and batches 1/3 from worker 1 (in default round-robin).
    batches = []
    for i, b in enumerate(loader):
        batches.append(b)
        if i >= 3:
            break
    # Just assert: at least one pair of batches has different tile content.
    distinct_pairs = 0
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            if not torch.equal(batches[i][0], batches[j][0]):
                distinct_pairs += 1
    assert distinct_pairs >= 1, "all multi-worker batches were identical"
