"""Extract additional augmentation view embeddings (rot180, rot270, vflip) for test tiles.

Standalone PyTorch script — no Spark/Luigi needed for the small test set (~2000 images).
Output: per-tile parquet with 3 new embedding columns that can be joined with
the original Phase 1 parquet for 6-view aggregation.

Augmentations (matching Phase 1 tensor conventions):
  rot180  — torch.rot90(t, 2, [1, 2])  — 180-degree rotation
  rot270  — torch.rot90(t, 3, [1, 2])  — 270-degree rotation (= 90 CW)
  vflip   — torch.flip(t, [1])         — vertical flip (H dimension)

Tiling: 4x4 grid for square test images, matching Phase 1 _tile_test() exactly.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import timm
import torch
from PIL import Image, ImageFile
import pyarrow as pa
import pyarrow.parquet as pq

ImageFile.LOAD_TRUNCATED_IMAGES = True

DINOV2_SIZE = 518
EMB_DIM = 768
NUM_CLASSES = 7806


def tile_test_image(image: Image.Image) -> list:
    """4x4 grid = 16 tiles for square test images (matches Phase 1 _tile_test)."""
    w, h = image.size
    grid_w, grid_h = w // 4, h // 4
    tiles = []
    for r in range(4):
        for c in range(4):
            left, upper = c * grid_w, r * grid_h
            tile = image.crop((left, upper, left + grid_w, upper + grid_h))
            tile = tile.resize((DINOV2_SIZE, DINOV2_SIZE), Image.LANCZOS)
            tiles.append(tile)
    return tiles


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract rot180/rot270/vflip embeddings for test tiles"
    )
    ap.add_argument(
        "--image-dir", required=True,
        help="Directory containing test images (*.jpg)",
    )
    ap.add_argument(
        "--output-path", required=True,
        help="Output parquet file path",
    )
    ap.add_argument(
        "--model-path", required=True,
        help="DINOv2 checkpoint (.pth)",
    )
    ap.add_argument(
        "--model-name",
        default="vit_base_patch14_reg4_dinov2.lvd142m",
    )
    ap.add_argument(
        "--batch-size", type=int, default=16,
        help="Tiles per GPU batch (each tile produces 3 augmented views)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}")
    print(f"  Checkpoint: {args.model_path}")
    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=NUM_CLASSES,
        checkpoint_path=args.model_path,
    )
    model = model.to(device).eval()

    # Use same eval transform as Phase 1 (resize + center crop + normalize)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    print(f"  Input size: {data_config['input_size']}")

    # Find all test images
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found in {args.image_dir}")
    print(f"Found {len(image_paths)} test images")

    # Process images one-by-one, batching tiles within each image
    results_names = []
    results_indices = []
    results_rot180 = []
    results_rot270 = []
    results_vflip = []

    t_start = time.time()
    n_tiles_total = 0

    for img_idx, img_path in enumerate(image_paths):
        image_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        tiles_pil = tile_test_image(img)

        # Transform tiles to tensors
        tile_tensors = [transform(t) for t in tiles_pil]

        # Process tiles in batches
        for batch_start in range(0, len(tile_tensors), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(tile_tensors))
            batch_tiles = tile_tensors[batch_start:batch_end]
            actual_bs = len(batch_tiles)

            # Create 3 augmented views per tile
            aug_tensors = []
            for t in batch_tiles:
                aug_tensors.append(torch.rot90(t, 2, [1, 2]))  # rot180
                aug_tensors.append(torch.rot90(t, 3, [1, 2]))  # rot270
                aug_tensors.append(torch.flip(t, [1]))          # vflip

            batch_input = torch.stack(aug_tensors).to(device)  # [3*bs, C, H, W]

            with torch.no_grad():
                features = model.forward_features(batch_input)
                cls_tokens = features[:, 0, :].cpu().numpy()  # [3*bs, 768]

            for i in range(actual_bs):
                tile_idx = batch_start + i
                results_names.append(image_name)
                results_indices.append(tile_idx)
                results_rot180.append(
                    cls_tokens[i * 3 + 0].astype(np.float16).tobytes()
                )
                results_rot270.append(
                    cls_tokens[i * 3 + 1].astype(np.float16).tobytes()
                )
                results_vflip.append(
                    cls_tokens[i * 3 + 2].astype(np.float16).tobytes()
                )

            n_tiles_total += actual_bs

        if (img_idx + 1) % 200 == 0 or img_idx + 1 == len(image_paths):
            elapsed = time.time() - t_start
            rate = n_tiles_total / elapsed if elapsed > 0 else 0
            print(
                f"  [{img_idx + 1}/{len(image_paths)}] "
                f"{n_tiles_total} tiles, {elapsed:.1f}s ({rate:.0f} tiles/s)"
            )

    elapsed_total = time.time() - t_start
    print(
        f"\nExtraction complete: {n_tiles_total} tiles from "
        f"{len(image_paths)} images in {elapsed_total:.1f}s"
    )

    # Write parquet
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    table = pa.Table.from_pydict({
        "image_name": results_names,
        "tile_index": results_indices,
        "embedding_rot180_f16": results_rot180,
        "embedding_rot270_f16": results_rot270,
        "embedding_vflip_f16": results_vflip,
    })
    pq.write_table(table, args.output_path, compression="snappy")
    print(f"Wrote {table.num_rows} rows to {args.output_path}")


if __name__ == "__main__":
    main()
