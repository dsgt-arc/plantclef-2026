"""Core re-aggregation engine. Reads per-tile parquet from the embedding
extraction step, applies configurable filtering, runs tile-to-image aggregation.
Designed to be loaded once and re-aggregated with different configs cheaply.

Note: parquet column names (e.g. 'teacher_logits_f16') are legacy from the
Exp-3 embedding extraction pipeline and kept for backward compatibility.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.aggregator import TemperatureNoisyORAggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix, f16_vector_to_bytes


@dataclass
class AggregationConfig:
    temperature: float = 2.0
    aug_filter: bool = True
    tile_filter_k: int = 16  # 16 = keep all tiles (no filter)
    geo_mask_path: Optional[str] = None
    image_topk: int = 64
    denoise_eps: float = 1e-6
    num_classes: int = 7806


def load_per_tile_parquet(
    parquet_root: str,
    pattern: str = "shard_*/*.parquet",
) -> pd.DataFrame:
    """Load all per-tile parquet files into a single DataFrame."""
    files = sorted(glob.glob(os.path.join(parquet_root, pattern)))
    if not files:
        # Try without subdirectory pattern
        files = sorted(glob.glob(os.path.join(parquet_root, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {parquet_root} matching {pattern}"
        )

    tables = []
    for fp in files:
        t = pq.read_table(fp)
        tables.append(t)

    combined = pa.concat_tables(tables)
    return combined.to_pandas()


def aggregate_one_image(
    tiles_df: pd.DataFrame,
    config: AggregationConfig,
    aggregator: TemperatureNoisyORAggregator,
    geo_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Filter + aggregate tiles for one image. Returns [num_classes] f16 probs or None."""
    pdf = tiles_df

    # Aug consistency filter
    if config.aug_filter and "passed_aug_consistency" in pdf.columns:
        pdf = pdf[pdf["passed_aug_consistency"] == True]  # noqa: E712

    if pdf.empty:
        return None

    # Tile confidence filter
    if "max_prob" in pdf.columns:
        pdf = pdf.sort_values("max_prob", ascending=False)
    if config.tile_filter_k > 0 and len(pdf) > config.tile_filter_k:
        pdf = pdf.head(config.tile_filter_k)

    # Aggregate (parquet column name is legacy from Exp-3 pipeline)
    logits_bufs = pdf["teacher_logits_f16"].tolist()
    tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, num_classes=config.num_classes)
    p_image_f16 = aggregator.aggregate(tile_logits)

    # Geo mask
    if geo_mask is not None:
        p_f32 = p_image_f16.astype(np.float32)
        p_f32[~geo_mask] = 0.0
        p_image_f16 = p_f32.astype(np.float16)

    return p_image_f16


def run_aggregation(
    tile_data: pd.DataFrame,
    config: AggregationConfig,
    image_col: str = "image_name",
) -> Dict[str, np.ndarray]:
    """Group by image, aggregate each, return {image_name: probs}."""
    image_topk = config.image_topk if config.image_topk > 0 else None

    aggregator = TemperatureNoisyORAggregator(
        num_classes=config.num_classes,
        temperature=config.temperature,
        denoise_eps=config.denoise_eps,
        output_dtype=np.float16,
        image_topk=image_topk,
        return_stats=False,
    )

    geo_mask = None
    if config.geo_mask_path:
        geo_mask = np.load(config.geo_mask_path)
        if geo_mask.dtype != np.bool_:
            geo_mask = geo_mask.astype(np.bool_)

    results = {}
    for name, group in tile_data.groupby(image_col):
        probs = aggregate_one_image(group, config, aggregator, geo_mask)
        if probs is not None:
            results[str(name)] = probs

    return results


def save_aggregated(
    image_probs: Dict[str, np.ndarray],
    out_path: str,
    id_col: str = "image_name",
) -> None:
    """Write aggregated probs to parquet."""
    rows_id = []
    rows_probs = []
    rows_ntiles = []
    for name, probs in image_probs.items():
        rows_id.append(name)
        rows_probs.append(f16_vector_to_bytes(probs))
        rows_ntiles.append(int(np.count_nonzero(probs)))

    table = pa.Table.from_pydict(
        {
            id_col: rows_id,
            "image_probs_f16": rows_probs,
            "num_nonzero": rows_ntiles,
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"Wrote {table.num_rows} images to {out_path}")
