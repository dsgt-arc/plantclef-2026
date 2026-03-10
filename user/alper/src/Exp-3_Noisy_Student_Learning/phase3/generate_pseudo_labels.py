"""Phase 3 — Generate pseudo-labels and student training dataset from LUCAS.

Reads raw per-tile parquet from Phase 1, re-runs Noisy-OR aggregation with
the best configuration from Phase 2, applies geo species masking and absolute
confidence thresholding, and produces the student training dataset.

Output: per-image parquet with packed tile embeddings (3 views), pseudo-labels,
and full teacher probability vectors for KL-div distillation.

Processing is done shard-by-shard to keep memory bounded (~2-3 GB per shard
instead of ~25 GB for all LUCAS tiles at once).
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.aggregator import TemperatureNoisyORAggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PseudoLabelConfig:
    temperature: float = 2.0
    aug_filter: bool = True
    tile_filter_k: int = 16       # for Noisy-OR aggregation; 16 = no filter
    threshold: float = 0.1        # absolute confidence threshold for pseudo-labels
    max_labels: int = 10          # cap pseudo-labels per image
    image_topk: int = 64          # top-K classes kept in Noisy-OR output
    num_classes: int = 7806


@dataclass
class PseudoLabelStats:
    total_images: int = 0
    images_with_labels: int = 0
    images_skipped_no_tiles: int = 0
    total_pseudo_labels: int = 0
    total_geo_blocked: int = 0
    total_tiles_input: int = 0
    total_tiles_aug_passed: int = 0
    species_seen: set = field(default_factory=set)
    label_counts: list = field(default_factory=list)
    pseudo_label_probs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_shard_dirs(parquet_root: str) -> List[str]:
    """Find shard directories under root, or treat root as a single shard."""
    shard_dirs = sorted(glob.glob(os.path.join(parquet_root, "shard_*")))
    if shard_dirs:
        return [d for d in shard_dirs if os.path.isdir(d)]
    return [parquet_root]


def load_shard_parquet(shard_path: str) -> pd.DataFrame:
    """Load all parquet files in a shard directory."""
    if os.path.isfile(shard_path):
        return pq.read_table(shard_path).to_pandas()

    files = sorted(glob.glob(os.path.join(shard_path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {shard_path}")

    tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables).to_pandas()


OUTPUT_SCHEMA = pa.schema([
    ("image_name", pa.string()),
    ("num_tiles", pa.int32()),
    ("tile_embeddings_orig_f16", pa.binary()),
    ("tile_embeddings_flip_f16", pa.binary()),
    ("tile_embeddings_rot90_f16", pa.binary()),
    ("tile_max_probs_f32", pa.binary()),
    ("tile_indices_i32", pa.binary()),
    ("pseudo_labels_i32", pa.binary()),
    ("num_pseudo_labels", pa.int32()),
    ("teacher_probs_f16", pa.binary()),
])


def write_output_parquet(records: List[dict], out_path: str) -> None:
    """Write processed per-image records to a single parquet file."""
    if not records:
        print("WARNING: No records to write!")
        return

    arrays = {name: [] for name in OUTPUT_SCHEMA.names}
    for rec in records:
        for name in OUTPUT_SCHEMA.names:
            arrays[name].append(rec[name])

    table = pa.table(arrays, schema=OUTPUT_SCHEMA)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"Wrote {table.num_rows} images to {out_path}")


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(
    image_name: str,
    tiles_df: pd.DataFrame,
    config: PseudoLabelConfig,
    aggregator: TemperatureNoisyORAggregator,
    geo_mask: Optional[np.ndarray],
    stats: PseudoLabelStats,
) -> Optional[dict]:
    """Process one image: aggregate teacher predictions, generate pseudo-labels,
    and pack all aug-passing tile embeddings for student training.

    Returns None if zero tiles survive aug filtering or zero pseudo-labels
    pass the confidence threshold (no supervisory signal → skip).
    """
    stats.total_tiles_input += len(tiles_df)

    # --- Aug consistency filter ---
    if config.aug_filter and "passed_aug_consistency" in tiles_df.columns:
        aug_passed = tiles_df[tiles_df["passed_aug_consistency"] == True]  # noqa: E712
    else:
        aug_passed = tiles_df

    if aug_passed.empty:
        stats.images_skipped_no_tiles += 1
        return None

    stats.total_tiles_aug_passed += len(aug_passed)

    # Sort by confidence (highest first) — used for tile selection below
    if "max_prob" in aug_passed.columns:
        aug_passed = aug_passed.sort_values("max_prob", ascending=False)

    # --- Tile selection for Noisy-OR aggregation ---
    # Only the top-K tiles contribute to aggregation. All aug-passing tiles
    # are still stored for the student dataloader.
    if 0 < config.tile_filter_k < len(aug_passed):
        agg_tiles = aug_passed.head(config.tile_filter_k)
    else:
        agg_tiles = aug_passed

    # --- Noisy-OR aggregation ---
    logits_bufs = agg_tiles["teacher_logits_f16"].tolist()
    tile_logits = f16_bytes_list_to_f32_matrix(
        logits_bufs, num_classes=config.num_classes
    )
    teacher_probs_f16 = aggregator.aggregate(tile_logits)   # [7806] f16
    teacher_probs = teacher_probs_f16.astype(np.float32)

    # --- Geo species masking ---
    pre_geo_nonzero = int(np.count_nonzero(teacher_probs))
    if geo_mask is not None:
        teacher_probs[~geo_mask] = 0.0
    post_geo_nonzero = int(np.count_nonzero(teacher_probs))
    geo_blocked = pre_geo_nonzero - post_geo_nonzero
    stats.total_geo_blocked += geo_blocked

    # --- Pseudo-label selection: threshold → sort → cap ---
    above_threshold = np.where(teacher_probs >= config.threshold)[0]
    if len(above_threshold) > 0:
        sorted_idx = above_threshold[np.argsort(-teacher_probs[above_threshold])]
        pseudo_labels = sorted_idx[: config.max_labels].astype(np.int32)
    else:
        pseudo_labels = np.array([], dtype=np.int32)

    # --- Skip images with no pseudo-labels ---
    n_labels = len(pseudo_labels)
    if n_labels == 0:
        stats.total_images += 1
        stats.label_counts.append(0)
        return None

    # --- Update statistics ---
    stats.total_images += 1
    stats.label_counts.append(n_labels)
    stats.total_pseudo_labels += n_labels
    stats.images_with_labels += 1
    for sp in pseudo_labels:
        stats.species_seen.add(int(sp))
        stats.pseudo_label_probs.append(float(teacher_probs[sp]))

    # --- Pack all aug-passing tile data for student training ---
    num_tiles = len(aug_passed)

    emb_orig_packed = b"".join(aug_passed["embedding_orig_f16"].tolist())
    emb_flip_packed = b"".join(aug_passed["embedding_flip_f16"].tolist())
    emb_rot90_packed = b"".join(aug_passed["embedding_rot90_f16"].tolist())

    tile_max_probs = aug_passed["max_prob"].values.astype(np.float32)
    tile_indices = aug_passed["tile_index"].values.astype(np.int32)

    teacher_probs_bytes = teacher_probs.astype(np.float16).tobytes()

    return {
        "image_name": image_name,
        "num_tiles": num_tiles,
        "tile_embeddings_orig_f16": emb_orig_packed,
        "tile_embeddings_flip_f16": emb_flip_packed,
        "tile_embeddings_rot90_f16": emb_rot90_packed,
        "tile_max_probs_f32": tile_max_probs.tobytes(),
        "tile_indices_i32": tile_indices.tobytes(),
        "pseudo_labels_i32": pseudo_labels.tobytes(),
        "num_pseudo_labels": n_labels,
        "teacher_probs_f16": teacher_probs_bytes,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(stats: PseudoLabelStats, config: PseudoLabelConfig) -> None:
    print(f"\n{'='*60}")
    print("PSEUDO-LABEL GENERATION STATISTICS")
    print(f"{'='*60}")

    print(f"\nConfiguration:")
    print(f"  Temperature:      {config.temperature}")
    print(f"  Aug filter:       {config.aug_filter}")
    print(f"  Tile filter K:    {config.tile_filter_k}")
    print(f"  Threshold:        {config.threshold}")
    print(f"  Max labels/image: {config.max_labels}")

    print(f"\nImage Statistics:")
    print(f"  Total images processed:   {stats.total_images}")
    print(f"  Images with >= 1 label:   {stats.images_with_labels}")
    print(f"  Images with 0 labels:     {stats.total_images - stats.images_with_labels}")
    print(f"  Images skipped (no tiles): {stats.images_skipped_no_tiles}")

    print(f"\nTile Statistics:")
    print(f"  Total tiles (input):      {stats.total_tiles_input}")
    print(f"  Tiles after aug filter:   {stats.total_tiles_aug_passed}")
    aug_rate = stats.total_tiles_aug_passed / max(stats.total_tiles_input, 1) * 100
    print(f"  Aug pass rate:            {aug_rate:.1f}%")
    avg_tiles = stats.total_tiles_aug_passed / max(stats.total_images, 1)
    print(f"  Avg tiles/image (stored): {avg_tiles:.1f}")

    print(f"\nPseudo-Label Statistics:")
    avg_labels = stats.total_pseudo_labels / max(stats.total_images, 1)
    print(f"  Total pseudo-labels:      {stats.total_pseudo_labels}")
    print(f"  Avg labels/image:         {avg_labels:.2f}")
    print(f"  Species coverage:         {len(stats.species_seen)} / {config.num_classes}")

    if stats.label_counts:
        counts = np.array(stats.label_counts)
        print(f"  Label count distribution:")
        print(f"    Min: {counts.min()}, Max: {counts.max()}, "
              f"Median: {np.median(counts):.1f}")
        for k in [0, 1, 2, 3, 4, 5, 10]:
            n = int(np.sum(counts == k))
            if n > 0 or k <= 5:
                pct = n / len(counts) * 100
                print(f"    {k} labels: {n} images ({pct:.1f}%)")

    print(f"\nGeo Mask Impact:")
    print(f"  Total predictions blocked: {stats.total_geo_blocked}")
    avg_blocked = stats.total_geo_blocked / max(stats.total_images, 1)
    print(f"  Avg blocked/image:         {avg_blocked:.1f}")

    if stats.pseudo_label_probs:
        probs = np.array(stats.pseudo_label_probs)
        print(f"\nPseudo-Label Confidence Distribution:")
        print(f"  Min: {probs.min():.4f}, Max: {probs.max():.4f}, "
              f"Mean: {probs.mean():.4f}")
        for lo, hi in [(0.1, 0.2), (0.2, 0.3), (0.3, 0.5),
                       (0.5, 0.7), (0.7, 1.0)]:
            n = int(np.sum((probs >= lo) & (probs < hi)))
            pct = n / len(probs) * 100
            print(f"    [{lo:.1f}, {hi:.1f}): {n} ({pct:.1f}%)")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 3: Generate pseudo-labels and student training dataset"
    )

    ap.add_argument(
        "--tile-parquet-root", required=True,
        help="Phase 1 raw per-tile parquet root (e.g., phase1_tiles_lucas/)",
    )
    ap.add_argument("--out-path", required=True, help="Output parquet file path")
    ap.add_argument(
        "--geo-mask-npy", default=None,
        help="Geo species mask .npy (bool array [7806])",
    )

    ap.add_argument(
        "--temperature", type=float, default=2.0,
        help="Noisy-OR temperature (use best from Phase 2 sweep)",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.1,
        help="Pseudo-label confidence threshold",
    )
    ap.add_argument(
        "--max-labels", type=int, default=10,
        help="Max pseudo-labels per image",
    )
    ap.add_argument(
        "--tile-filter-k", type=int, default=16,
        help="Top-K tiles for Noisy-OR aggregation (16 = no filter)",
    )
    ap.add_argument(
        "--no-aug-filter", action="store_true",
        help="Disable aug consistency filter",
    )
    ap.add_argument("--image-topk", type=int, default=64)
    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--id-col", default="image_name")

    args = ap.parse_args()

    config = PseudoLabelConfig(
        temperature=args.temperature,
        aug_filter=not args.no_aug_filter,
        tile_filter_k=args.tile_filter_k,
        threshold=args.threshold,
        max_labels=args.max_labels,
        image_topk=args.image_topk,
        num_classes=args.num_classes,
    )

    aggregator = TemperatureNoisyORAggregator(
        num_classes=config.num_classes,
        temperature=config.temperature,
        output_dtype=np.float16,
        image_topk=config.image_topk if config.image_topk > 0 else None,
        return_stats=False,
    )

    geo_mask = None
    if args.geo_mask_npy:
        geo_mask = np.load(args.geo_mask_npy)
        if geo_mask.dtype != np.bool_:
            geo_mask = geo_mask.astype(np.bool_)
        print(f"Loaded geo mask: {geo_mask.sum()} / {len(geo_mask)} species allowed")

    # --- Discover and process shards ---
    shard_dirs = find_shard_dirs(args.tile_parquet_root)
    print(f"Found {len(shard_dirs)} shard(s) in {args.tile_parquet_root}")

    stats = PseudoLabelStats()
    all_records: List[dict] = []

    for i, shard_path in enumerate(shard_dirs):
        shard_name = os.path.basename(shard_path)
        print(f"\nShard {i + 1}/{len(shard_dirs)}: {shard_name}")

        shard_df = load_shard_parquet(shard_path)
        print(f"  {len(shard_df)} tile rows")

        shard_records = []
        for name, group in shard_df.groupby(args.id_col):
            record = process_image(
                image_name=str(name),
                tiles_df=group,
                config=config,
                aggregator=aggregator,
                geo_mask=geo_mask,
                stats=stats,
            )
            if record is not None:
                shard_records.append(record)

        print(f"  -> {len(shard_records)} image records")
        all_records.extend(shard_records)

        del shard_df  # free memory before next shard

    print(f"\nTotal records: {len(all_records)}")

    # --- Write output ---
    write_output_parquet(all_records, args.out_path)

    # --- Print statistics ---
    print_statistics(stats, config)


if __name__ == "__main__":
    main()
