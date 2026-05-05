"""Tests for experiments/003-tileq-decoder/build_faiss_background_map.py.

Unit tests on the pure helpers (l2_normalize_rows, train_row_to_class_idx,
apply_threshold, decode_test_parquet) plus a small end-to-end integration test
with a synthetic train + test parquet.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_MODULE_PATH = (
    Path(__file__).parent.parent.parent
    / "experiments/003-tileq-decoder/build_faiss_background_map.py"
)
_spec = importlib.util.spec_from_file_location("build_faiss_background_map", _MODULE_PATH)
faiss_module = importlib.util.module_from_spec(_spec)
sys.modules["build_faiss_background_map"] = faiss_module
_spec.loader.exec_module(faiss_module)


# ----- l2_normalize_rows -----


def test_l2_normalize_rows_unit_norm():
    arr = np.asarray([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    faiss_module.l2_normalize_rows(arr)
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_l2_normalize_rows_zero_handled():
    """A zero row must not produce NaN; the eps clamp keeps norm >= eps."""
    arr = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    faiss_module.l2_normalize_rows(arr)
    assert not np.isnan(arr).any()
    assert np.allclose(arr[1], [1.0, 0.0, 0.0])


def test_l2_normalize_rejects_non_float32():
    arr = np.zeros((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        faiss_module.l2_normalize_rows(arr)


# ----- train_row_to_class_idx -----


def test_train_row_to_class_idx_basic():
    """offsets [0, 3, 5, 7] -> rows {0,1,2}=0, {3,4}=1, {5,6}=2."""
    offsets = np.asarray([0, 3, 5, 7], dtype=np.int64)
    rows = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    out = faiss_module.train_row_to_class_idx(rows, offsets)
    assert out.dtype == np.int32
    assert out.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_train_row_to_class_idx_class_boundary():
    """Boundary rows (start of each class) must map to the new class."""
    offsets = np.asarray([0, 5, 10], dtype=np.int64)
    rows = np.asarray([4, 5, 9], dtype=np.int64)
    out = faiss_module.train_row_to_class_idx(rows, offsets)
    assert out.tolist() == [0, 1, 1]


# ----- apply_threshold -----


def test_apply_threshold_basic():
    sim = np.asarray([[0.9, 0.8, 0.7], [0.85, 0.95, 0.5]], dtype=np.float32)
    cls = np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    out = faiss_module.apply_threshold(sim, cls, threshold=0.85)
    # >= 0.85: positions (0,0), (1,0), (1,1)
    expected = np.asarray([[10, -1, -1], [40, 50, -1]], dtype=np.int32)
    assert np.array_equal(out, expected)
    assert out.dtype == np.int32


def test_apply_threshold_inclusive():
    """Tile with sim exactly equal to threshold gets labelled (>= comparison)."""
    sim = np.asarray([[0.85]], dtype=np.float32)
    cls = np.asarray([[7]], dtype=np.int32)
    out = faiss_module.apply_threshold(sim, cls, threshold=0.85)
    assert out.item() == 7


# ----- decode_test_parquet -----


def _write_test_parquet(path: Path, image_to_tiles: dict[str, np.ndarray]) -> None:
    """Write a cls_grid_4x4-style parquet from {image_name: (n_tiles, 768) float32}."""
    rows = []
    for name, tiles in image_to_tiles.items():
        for t_idx in range(tiles.shape[0]):
            f16 = tiles[t_idx].astype(np.float16)
            rows.append((name, t_idx, f16.tobytes()))
    schema = pa.schema(
        [
            ("image_name", pa.string()),
            ("tile_idx", pa.int32()),
            ("cls_embedding", pa.binary()),
        ]
    )
    table = pa.table(
        {
            "image_name": [r[0] for r in rows],
            "tile_idx": pa.array([r[1] for r in rows], type=pa.int32()),
            "cls_embedding": [r[2] for r in rows],
        },
        schema=schema,
    )
    pq.write_table(table, path)


def test_decode_test_parquet_shape_and_order(tmp_path: Path):
    rng = np.random.default_rng(0)
    img2tiles = {
        # Names intentionally not in alphabetical order in the parquet, to test sorting.
        "img-z": rng.normal(0, 1, (4, 768)).astype(np.float32),
        "img-a": rng.normal(0, 1, (4, 768)).astype(np.float32),
        "img-m": rng.normal(0, 1, (4, 768)).astype(np.float32),
    }
    p = tmp_path / "test.parquet"
    _write_test_parquet(p, img2tiles)
    names, test_cls = faiss_module.decode_test_parquet(p)
    # Sorted alphabetically -> img-a, img-m, img-z
    assert names.tolist() == ["img-a", "img-m", "img-z"]
    assert test_cls.shape == (3, 4, 768)
    assert test_cls.dtype == np.float32
    # img-a's tiles should be at index 0
    expected_a = img2tiles["img-a"].astype(np.float16).astype(np.float32)
    assert np.allclose(test_cls[0], expected_a, atol=1e-3)


def test_decode_test_parquet_uneven_tile_counts_rejected(tmp_path: Path):
    rng = np.random.default_rng(0)
    img2tiles = {
        "img-a": rng.normal(0, 1, (4, 768)).astype(np.float32),
        "img-b": rng.normal(0, 1, (3, 768)).astype(np.float32),  # only 3 tiles
    }
    p = tmp_path / "test.parquet"
    _write_test_parquet(p, img2tiles)
    with pytest.raises(ValueError, match="not uniform"):
        faiss_module.decode_test_parquet(p)


# ----- end-to-end -----


def test_main_end_to_end(tmp_path: Path):
    """Synthetic data: 3 train classes, 2 test images x 4 tiles.

    Construct the test embeddings so the expected nearest-train-row is known
    in advance, and verify the resulting tile_to_class_idx matches.
    """
    dim = 16
    rng = np.random.default_rng(42)

    # 3 train classes, 5 rows each = 15 rows total
    n_per_class = 5
    n_classes = 3
    train_emb = rng.normal(0, 1, (n_classes * n_per_class, dim)).astype(np.float32)
    offsets = np.asarray(
        [0, n_per_class, 2 * n_per_class, 3 * n_per_class], dtype=np.int64
    )
    class_to_species_id = np.asarray([100, 200, 300], dtype=np.int32)

    train_lookup_path = tmp_path / "train_lookup.npz"
    np.savez(
        train_lookup_path,
        embeddings=train_emb,
        offsets=offsets,
        class_to_species_id=class_to_species_id,
    )

    # Test image A: 4 tiles, each chosen so its nearest neighbour is a known
    # train row (high sim ~1.0). Test image B: 4 tiles of pure noise (low sim).
    img_a_tiles = train_emb[[0, 5, 10, 0]].copy() + 1e-3 * rng.normal(0, 1, (4, dim)).astype(np.float32)
    img_b_tiles = rng.normal(0, 1, (4, dim)).astype(np.float32) * 100  # very different scale, random direction
    test_path = tmp_path / "cls_grid_4x4.parquet"
    _write_test_parquet(test_path, {"img-a": img_a_tiles, "img-b": img_b_tiles})

    output_dir = tmp_path / "artifacts"
    faiss_module.main(
        train_lookup_path=train_lookup_path,
        test_parquet_path=test_path,
        output_dir=output_dir,
        threshold=0.95,
    )

    # Outputs exist
    out_npz = output_dir / "sampler_b_lookup.npz"
    assert out_npz.exists()
    assert (output_dir / "tile_top1_sim_histogram.png").exists()
    assert (output_dir / "faiss_threshold.txt").exists()

    data = np.load(out_npz, allow_pickle=False)
    expected_keys = {
        "image_names", "test_cls", "top1_sim", "top1_class_idx",
        "tile_to_class_idx", "threshold",
    }
    assert set(data.files) == expected_keys

    # Image-name ordering is alphabetical
    assert data["image_names"].tolist() == ["img-a", "img-b"]

    # Shapes
    assert data["test_cls"].shape == (2, 4, 16)
    assert data["top1_sim"].shape == (2, 4)
    assert data["top1_class_idx"].shape == (2, 4)
    assert data["tile_to_class_idx"].shape == (2, 4)
    assert float(data["threshold"]) == pytest.approx(0.95)

    # img-a's tiles were planted near train rows [0, 5, 10, 0] -> classes [0, 1, 2, 0]
    img_a_pred = data["top1_class_idx"][0]
    assert img_a_pred.tolist() == [0, 1, 2, 0]

    # All 4 img-a tiles should be above the 0.95 threshold (planted very close)
    assert (data["top1_sim"][0] >= 0.95).all()
    assert (data["tile_to_class_idx"][0] >= 0).all()

    # img-b's tiles are random — most should be below 0.95
    # (with random direction in 16-d space the expected sim is small)
    assert (data["tile_to_class_idx"][1] == -1).any()
