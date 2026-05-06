"""Tests for experiments/003-tileq-decoder/infer.py.

Targets the testable helpers:
- ``load_class_to_species_id`` round-trips a small text file.
- ``write_submission_csv`` writes the spec'd CSV format.
- ``load_test_cls`` reshapes a synthetic float16-blob parquet correctly.
"""

from __future__ import annotations

import csv as csv_module
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_MODULE_PATH = Path(__file__).parent.parent.parent / "experiments/003-tileq-decoder/infer.py"
_spec = importlib.util.spec_from_file_location("infer_module", _MODULE_PATH)
infer_module = importlib.util.module_from_spec(_spec)
sys.modules["infer_module"] = infer_module
_spec.loader.exec_module(infer_module)


def test_load_class_to_species_id(tmp_path: Path):
    p = tmp_path / "class_mapping.txt"
    p.write_text("100\n200\n300\n")
    arr = infer_module.load_class_to_species_id(p)
    assert arr.dtype == np.int64
    assert arr.tolist() == [100, 200, 300]


def test_write_submission_csv(tmp_path: Path):
    out = tmp_path / "submissions" / "x.csv"
    image_names = np.asarray(["img-a", "img-b"], dtype="<U64")
    species_id_lists = [[1, 2, 3], [4, 5]]
    infer_module.write_submission_csv(out, image_names, species_id_lists)
    assert out.exists()
    rows = list(csv_module.reader(out.open(), delimiter=";"))
    assert rows[0] == ["quadrat_id", "species_ids"]
    assert rows[1] == ["img-a", "[1, 2, 3]"]
    assert rows[2] == ["img-b", "[4, 5]"]


def test_load_test_cls_reshapes_correctly(tmp_path: Path):
    rng = np.random.default_rng(0)
    img2tiles = {
        "img-z": rng.normal(0, 1, (4, 768)).astype(np.float32),
        "img-a": rng.normal(0, 1, (4, 768)).astype(np.float32),
    }
    rows = []
    for name, tiles in img2tiles.items():
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
    p = tmp_path / "test.parquet"
    pq.write_table(table, p)
    image_names, test_cls = infer_module.load_test_cls(p)
    # Names are sorted alphabetically
    assert image_names.tolist() == ["img-a", "img-z"]
    assert test_cls.shape == (2, 4, 768)
    assert test_cls.dtype == np.float32
    # img-a is at index 0
    expected_a = img2tiles["img-a"].astype(np.float16).astype(np.float32)
    assert np.allclose(test_cls[0], expected_a, atol=1e-3)
