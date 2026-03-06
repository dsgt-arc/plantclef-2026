"""Aspect-ratio-aware tiling as a Spark Pandas UDF.

LUCAS (4:3 images): 4 cols x 3 rows grid = 12 tiles + 4 quadrant random crops = 16 tiles
Test (square images): 4x4 grid = 16 tiles
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from PIL import Image

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    IntegerType,
    StructField,
    StructType,
)

from utils.serde import deserialize_image, serialize_image

DINOV2_SIZE = 518

TILE_SCHEMA = ArrayType(
    StructType(
        [
            StructField("tile_index", IntegerType()),
            StructField("tile", BinaryType()),
        ]
    )
)


def _quadrant_crop(
    image: Image.Image,
    quadrant: int,
    seed: int,
    size: int = DINOV2_SIZE,
) -> Image.Image:
    """Random crop of `size x size` within one quadrant of the image, seeded for determinism."""
    w, h = image.size
    mid_w, mid_h = w // 2, h // 2

    # Quadrant bounding boxes (left, upper, right, lower)
    boxes = {
        0: (0, 0, mid_w, mid_h),
        1: (mid_w, 0, w, mid_h),
        2: (0, mid_h, mid_w, h),
        3: (mid_w, mid_h, w, h),
    }
    left, upper, right, lower = boxes[quadrant]
    qw = right - left
    qh = lower - upper

    crop_w = min(size, qw)
    crop_h = min(size, qh)

    rng = np.random.RandomState(seed + quadrant)
    x0 = rng.randint(0, max(qw - crop_w, 0) + 1) + left
    y0 = rng.randint(0, max(qh - crop_h, 0) + 1) + upper

    crop = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    if crop.size != (size, size):
        crop = crop.resize((size, size), Image.LANCZOS)
    return crop


def _tile_lucas(image: Image.Image, image_id: str) -> list[dict]:
    """4x3 grid + 4 quadrant random crops = 16 tiles for LUCAS 4:3 images."""
    w, h = image.size
    cols, rows = 4, 3
    grid_w = w // cols
    grid_h = h // rows

    tiles = []
    # Grid tiles (row-major): indices 0-11
    idx = 0
    for r in range(rows):
        for c in range(cols):
            left = c * grid_w
            upper = r * grid_h
            tile = image.crop((left, upper, left + grid_w, upper + grid_h))
            tile = tile.resize((DINOV2_SIZE, DINOV2_SIZE), Image.LANCZOS)
            tiles.append({"tile_index": idx, "tile": serialize_image(tile)})
            idx += 1

    # Random crops: indices 12-15 (one per quadrant)
    seed = int(hashlib.md5(image_id.encode()).hexdigest(), 16) & 0xFFFFFFFF
    for q in range(4):
        crop = _quadrant_crop(image, q, seed, DINOV2_SIZE)
        tiles.append({"tile_index": idx, "tile": serialize_image(crop)})
        idx += 1

    return tiles


def _tile_test(image: Image.Image) -> list[dict]:
    """4x4 grid = 16 tiles for square test images."""
    w, h = image.size
    grid_size = 4
    grid_w = w // grid_size
    grid_h = h // grid_size

    tiles = []
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            left = c * grid_w
            upper = r * grid_h
            tile = image.crop((left, upper, left + grid_w, upper + grid_h))
            tile = tile.resize((DINOV2_SIZE, DINOV2_SIZE), Image.LANCZOS)
            tiles.append({"tile_index": idx, "tile": serialize_image(tile)})
            idx += 1

    return tiles


def make_tiles_udf(dataset_type: str = "lucas"):
    """Returns a Spark Pandas UDF for aspect-ratio-aware tiling.

    Args:
        dataset_type: "lucas" for 4x3 grid + 4 random crops,
                      "test" for 4x4 grid.
    """

    @F.pandas_udf(TILE_SCHEMA)
    def tiles_udf(data_series: pd.Series, id_series: pd.Series) -> pd.Series:
        all_tiles = []
        for raw_bytes, image_id in zip(data_series, id_series):
            img = deserialize_image(raw_bytes)
            if img.mode != "RGB":
                img = img.convert("RGB")

            if dataset_type == "lucas":
                tile_structs = _tile_lucas(img, str(image_id))
            else:
                tile_structs = _tile_test(img)

            all_tiles.append(tile_structs)
        return pd.Series(all_tiles)

    @F.pandas_udf(TILE_SCHEMA)
    def tiles_udf_test_only(data_series: pd.Series) -> pd.Series:
        all_tiles = []
        for raw_bytes in data_series:
            img = deserialize_image(raw_bytes)
            if img.mode != "RGB":
                img = img.convert("RGB")
            all_tiles.append(_tile_test(img))
        return pd.Series(all_tiles)

    if dataset_type == "test":
        return tiles_udf_test_only
    return tiles_udf
