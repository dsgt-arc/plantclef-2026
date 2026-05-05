"""Tests for experiments/003-tileq-decoder/build_train_lookup.py.

Round-trips a small synthetic train-style dataset through the build pipeline
to verify:
- class_mapping.txt parsing.
- species_id -> class_idx vectorised lookup, with error paths.
- per-class sort produces contiguous slices via offsets[].
- 1/sqrt(count) species weights are correctly normalised.
- main() writes both artefacts and they reload cleanly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# build_train_lookup is in experiments/003-tileq-decoder/, not on PYTHONPATH.
# Import it explicitly so tests are runnable from user/murilo/.
_MODULE_PATH = (
    Path(__file__).parent.parent.parent
    / "experiments/003-tileq-decoder/build_train_lookup.py"
)
_spec = importlib.util.spec_from_file_location("build_train_lookup", _MODULE_PATH)
build_train_lookup = importlib.util.module_from_spec(_spec)
sys.modules["build_train_lookup"] = build_train_lookup
_spec.loader.exec_module(build_train_lookup)


def _make_synthetic_table(rows_species: list[int], rng: np.random.Generator, dim: int = 8):
    embs = rng.normal(0, 1, (len(rows_species), dim)).astype(np.float32)
    return (
        pa.table(
            {
                "species_id": pa.array(rows_species, type=pa.int32()),
                "cls_embedding": pa.array(
                    [list(e) for e in embs], type=pa.list_(pa.float32())
                ),
            }
        ),
        embs,
    )


# ----- load_class_mapping -----


def test_load_class_mapping(tmp_path: Path):
    p = tmp_path / "class_mapping.txt"
    p.write_text("100\n200\n300\n")
    arr = build_train_lookup.load_class_mapping(p)
    assert arr.dtype == np.int32
    assert arr.tolist() == [100, 200, 300]


def test_load_class_mapping_rejects_duplicates(tmp_path: Path):
    p = tmp_path / "class_mapping.txt"
    p.write_text("100\n200\n100\n")
    with pytest.raises(ValueError, match="duplicate"):
        build_train_lookup.load_class_mapping(p)


def test_load_class_mapping_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "class_mapping.txt"
    p.write_text("100\n\n200\n300\n\n")
    arr = build_train_lookup.load_class_mapping(p)
    assert arr.tolist() == [100, 200, 300]


# ----- species_ids_to_class_idxs -----


def test_species_ids_to_class_idxs_basic():
    class_to_species_id = np.asarray([100, 200, 300], dtype=np.int32)
    sids = np.asarray([200, 100, 300, 200, 100], dtype=np.int32)
    out = build_train_lookup.species_ids_to_class_idxs(sids, class_to_species_id)
    assert out.dtype == np.int32
    assert out.tolist() == [1, 0, 2, 1, 0]


def test_species_ids_to_class_idxs_in_range_but_unknown_raises():
    """species_id within [min, max] of the mapping but not present."""
    class_to_species_id = np.asarray([100, 200, 300], dtype=np.int32)
    sids = np.asarray([200, 250, 100], dtype=np.int32)  # 250 in range, not in mapping
    with pytest.raises(ValueError, match="not in class_mapping"):
        build_train_lookup.species_ids_to_class_idxs(sids, class_to_species_id)


def test_species_ids_to_class_idxs_out_of_range_raises():
    """species_id larger than max (would index past sids_sorted)."""
    class_to_species_id = np.asarray([100, 200, 300], dtype=np.int32)
    sids = np.asarray([200, 999, 100], dtype=np.int32)
    with pytest.raises(ValueError, match="exceed mapping range"):
        build_train_lookup.species_ids_to_class_idxs(sids, class_to_species_id)


def test_species_ids_to_class_idxs_handles_unsorted_mapping():
    """class_mapping order is the class_idx assignment; arbitrary values."""
    class_to_species_id = np.asarray([300, 100, 200], dtype=np.int32)
    sids = np.asarray([100, 200, 300], dtype=np.int32)
    out = build_train_lookup.species_ids_to_class_idxs(sids, class_to_species_id)
    # species 100 = class_idx 1, 200 = 2, 300 = 0
    assert out.tolist() == [1, 2, 0]


# ----- build_lookup_from_table -----


def test_build_lookup_from_table_synthetic():
    rng = np.random.default_rng(42)
    # 3 classes (species 100, 200, 300) -> 8 rows: [4, 3, 1] per class
    rows_species = [100, 200, 100, 300, 200, 100, 100, 200]
    table, raw_embs = _make_synthetic_table(rows_species, rng)
    class_to_species_id = np.asarray([100, 200, 300], dtype=np.int32)

    embs, offsets, counts = build_train_lookup.build_lookup_from_table(
        table, class_to_species_id, embedding_dim=8
    )

    assert embs.shape == (8, 8)
    assert embs.dtype == np.float32
    assert offsets.shape == (4,)
    assert offsets.dtype == np.int64
    assert offsets.tolist() == [0, 4, 7, 8]
    assert counts.tolist() == [4, 3, 1]

    # Class 0 (species 100) rows in input: indices 0, 2, 5, 6 — stable sort
    # preserves that order.
    assert np.array_equal(embs[0:4], raw_embs[[0, 2, 5, 6]])
    # Class 1 (species 200) rows: 1, 4, 7
    assert np.array_equal(embs[4:7], raw_embs[[1, 4, 7]])
    # Class 2 (species 300) rows: 3
    assert np.array_equal(embs[7:8], raw_embs[[3]])


def test_build_lookup_from_table_rejects_unknown_species_id():
    rng = np.random.default_rng(0)
    table, _ = _make_synthetic_table([100, 200, 250], rng)  # 250 in-range but unmapped
    class_to_species_id = np.asarray([100, 200, 300], dtype=np.int32)
    with pytest.raises(ValueError, match="not in class_mapping"):
        build_train_lookup.build_lookup_from_table(
            table, class_to_species_id, embedding_dim=8
        )


def test_build_lookup_from_table_handles_chunked_array():
    """Real Arrow datasets return ChunkedArrays; combine_chunks must be triggered."""
    rng = np.random.default_rng(1)
    t1, e1 = _make_synthetic_table([100, 200], rng)
    t2, e2 = _make_synthetic_table([300, 100], rng)
    table = pa.concat_tables([t1, t2])
    # Sanity: cls_embedding must now be a ChunkedArray with 2 chunks
    assert isinstance(table.column("cls_embedding"), pa.ChunkedArray)
    assert table.column("cls_embedding").num_chunks == 2

    class_to_species_id = np.asarray([100, 200, 300], dtype=np.int32)
    embs, offsets, counts = build_train_lookup.build_lookup_from_table(
        table, class_to_species_id, embedding_dim=8
    )
    assert embs.shape == (4, 8)
    assert offsets.tolist() == [0, 2, 3, 4]
    assert counts.tolist() == [2, 1, 1]
    # Class 0 (species 100): rows 0 (from t1) and 3 (from t2)
    expected_class0 = np.vstack([e1[0:1], e2[1:2]])
    assert np.array_equal(embs[0:2], expected_class0)


# ----- compute_species_weights -----


def test_compute_species_weights_basic():
    counts = np.asarray([100, 4, 16], dtype=np.int64)  # 1/sqrt = 0.1, 0.5, 0.25
    w = build_train_lookup.compute_species_weights(counts)
    assert w.shape == (3,)
    assert w.dtype == np.float32
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    # Largest weight goes to the rarest class
    assert w.argmax() == 1
    # Order of weights matches order of (1/sqrt(count))
    raw = 1.0 / np.sqrt(counts.astype(np.float64))
    assert np.allclose(w, raw / raw.sum(), atol=1e-6)


def test_compute_species_weights_zero_count_raises():
    counts = np.asarray([10, 0, 5], dtype=np.int64)
    with pytest.raises(ValueError, match="zero training rows"):
        build_train_lookup.compute_species_weights(counts)


# ----- main end-to-end -----


def test_main_end_to_end(tmp_path: Path):
    """Full round-trip: write tiny hive-partitioned parquet, run main(), reload."""
    rng = np.random.default_rng(123)
    train_dir = tmp_path / "train"
    sample0 = train_dir / "sample_id=0"
    sample1 = train_dir / "sample_id=1"
    sample0.mkdir(parents=True)
    sample1.mkdir(parents=True)

    # 3 classes, 6 rows total split across 2 hive partitions
    species_part0 = [100, 200, 300, 100]
    species_part1 = [200, 100]
    table0, _ = _make_synthetic_table(species_part0, rng)
    table1, _ = _make_synthetic_table(species_part1, rng)
    pq.write_table(table0, sample0 / "part-0.parquet")
    pq.write_table(table1, sample1 / "part-0.parquet")

    class_mapping = tmp_path / "class_mapping.txt"
    class_mapping.write_text("100\n200\n300\n")

    output_dir = tmp_path / "artifacts"

    build_train_lookup.main(
        train_dir=train_dir,
        class_mapping_path=class_mapping,
        output_dir=output_dir,
        expected_n_rows=6,
        embedding_dim=8,
    )

    lookup_path = output_dir / "train_cls_by_class_idx.npz"
    weights_path = output_dir / "species_weights.npy"
    assert lookup_path.exists()
    assert weights_path.exists()

    data = np.load(lookup_path)
    assert set(data.files) == {"embeddings", "offsets", "class_to_species_id"}
    assert data["embeddings"].shape == (6, 8)
    assert data["embeddings"].dtype == np.float32
    assert data["offsets"].shape == (4,)
    assert data["offsets"].tolist() == [0, 3, 5, 6]  # class 100:3, 200:2, 300:1
    assert data["class_to_species_id"].tolist() == [100, 200, 300]

    weights = np.load(weights_path)
    assert weights.shape == (3,)
    assert weights.dtype == np.float32
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    # 300 has fewest rows -> highest weight
    assert weights.argmax() == 2


def test_main_rejects_row_count_mismatch(tmp_path: Path):
    rng = np.random.default_rng(7)
    train_dir = tmp_path / "train" / "sample_id=0"
    train_dir.mkdir(parents=True)
    table, _ = _make_synthetic_table([100, 200], rng)
    pq.write_table(table, train_dir / "part-0.parquet")
    cm = tmp_path / "class_mapping.txt"
    cm.write_text("100\n200\n")

    with pytest.raises(RuntimeError, match="row count 2 != expected 999"):
        build_train_lookup.main(
            train_dir=tmp_path / "train",
            class_mapping_path=cm,
            output_dir=tmp_path / "artifacts",
            expected_n_rows=999,
            embedding_dim=8,
        )
