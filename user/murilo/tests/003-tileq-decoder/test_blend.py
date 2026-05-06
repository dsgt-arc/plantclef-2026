"""Tests for experiments/003-tileq-decoder/blend.py.

Targets pure helpers:
- ``load_max_prob_per_image`` reduces (N, T, C) per-tile probs to (N, C) per-image max
- ``topk_decode`` returns indices of descending top-k
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_MODULE_PATH = Path(__file__).parent.parent.parent / "experiments/003-tileq-decoder/blend.py"
_spec = importlib.util.spec_from_file_location("blend_module", _MODULE_PATH)
blend_module = importlib.util.module_from_spec(_spec)
sys.modules["blend_module"] = blend_module
_spec.loader.exec_module(blend_module)


def test_topk_decode_returns_descending():
    probs = np.asarray(
        [[0.1, 0.5, 0.3, 0.9, 0.2],
         [0.7, 0.6, 0.5, 0.4, 0.3]],
        dtype=np.float32,
    )
    idx = blend_module.topk_decode(probs, k=3)
    assert idx.shape == (2, 3)
    # Row 0: top-3 are columns 3 (0.9), 1 (0.5), 2 (0.3) in descending order
    assert idx[0].tolist() == [3, 1, 2]
    # Row 1: top-3 are columns 0 (0.7), 1 (0.6), 2 (0.5)
    assert idx[1].tolist() == [0, 1, 2]


def test_topk_decode_k_equals_n_classes():
    probs = np.asarray([[0.3, 0.1, 0.5, 0.2]], dtype=np.float32)
    idx = blend_module.topk_decode(probs, k=4)
    assert idx[0].tolist() == [2, 0, 3, 1]


def test_load_max_prob_per_image_reduces_across_tiles(tmp_path: Path):
    """Plant a known tile-level signal and verify max_prob recovers it."""
    rng = np.random.default_rng(0)
    n_classes = 5
    n_tiles = 3
    img2tiles = {
        # img-a: tile 0 has high prob on class 2
        "img-a": np.full((n_tiles, n_classes), 0.1, dtype=np.float32),
        # img-b: tile 1 has high prob on class 4
        "img-b": np.full((n_tiles, n_classes), 0.2, dtype=np.float32),
    }
    img2tiles["img-a"][0, 2] = 0.9
    img2tiles["img-b"][1, 4] = 0.8

    # Write parquet matching grid_4x4 schema (image_name, tile_idx, probs[binary])
    rows = []
    for name, tiles in img2tiles.items():
        for t_idx in range(tiles.shape[0]):
            f16 = tiles[t_idx].astype(np.float16)
            rows.append((name, t_idx, f16.tobytes()))
    schema = pa.schema(
        [
            ("image_name", pa.string()),
            ("tile_idx", pa.int32()),
            ("probs", pa.binary()),
        ]
    )
    table = pa.table(
        {
            "image_name": [r[0] for r in rows],
            "tile_idx": pa.array([r[1] for r in rows], type=pa.int32()),
            "probs": [r[2] for r in rows],
        },
        schema=schema,
    )
    p = tmp_path / "grid_4x4.parquet"
    pq.write_table(table, p)

    image_names, max_prob = blend_module.load_max_prob_per_image(p)
    assert image_names.tolist() == ["img-a", "img-b"]
    assert max_prob.shape == (2, n_classes)
    # img-a's max for class 2 should be ~0.9 (within float16 round-trip tolerance)
    assert max_prob[0].argmax() == 2
    assert abs(max_prob[0, 2] - 0.9) < 0.005
    # img-b's max for class 4 should be ~0.8
    assert max_prob[1].argmax() == 4
    assert abs(max_prob[1, 4] - 0.8) < 0.005


def test_blend_alpha_zero_equals_max_prob():
    """alpha=0 -> blended == max_prob (sanity-duplicate of A0a)."""
    rng = np.random.default_rng(0)
    decoder = rng.uniform(0, 1, (4, 7)).astype(np.float32)
    mp = rng.uniform(0, 1, (4, 7)).astype(np.float32)
    alpha = 0.0
    blended = alpha * decoder + (1.0 - alpha) * mp
    assert np.allclose(blended, mp)


def test_blend_alpha_one_equals_decoder():
    rng = np.random.default_rng(1)
    decoder = rng.uniform(0, 1, (4, 7)).astype(np.float32)
    mp = rng.uniform(0, 1, (4, 7)).astype(np.float32)
    alpha = 1.0
    blended = alpha * decoder + (1.0 - alpha) * mp
    assert np.allclose(blended, decoder)
