#!/usr/bin/env python3
"""Stage 1: GPU inference for PlantCLEF 2026 grid ablation.

Runs a frozen fine-tuned DINOv2 classifier with configurable grid tiling on
vegetation plot images. Saves full softmax probability vectors to parquet for
downstream classification (see classify.py in exp 002), or per-tile CLS
embeddings (--save-cls) for downstream decoding (see exp 003 TileQ-Decoder).

Usage:
    # Full image (no grid)
    python inference.py --grid-size 0

    # 4x4 grid
    python inference.py --grid-size 4

    # Pipeline validation (5 images)
    python inference.py --grid-size 3 --fast-dev-run

    # GPU profiling (20 images)
    python inference.py --grid-size 3 --gpu-check 20

    # Extract per-tile CLS embeddings (768-d) instead of probs
    python inference.py --grid-size 4 --save-cls
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from plantclef.gpu_monitor import InferenceGPUMonitor
from plantclef.inference.dataset import TileDataset
from plantclef.model import load_model

# Default paths on PACE shared storage
DEFAULT_MODEL_PATH = "/storage/project/ps-clef2026_plantclef-0/shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
DEFAULT_DATA_DIR = "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlanCLEF_test_data/images"
DEFAULT_OUTPUT_DIR = "/storage/project/ps-clef2026_plantclef-0/shared/inference/dinov2-vit-base"


def parse_args():
    parser = argparse.ArgumentParser(
        description="PlantCLEF 2026 — Stage 1: DINOv2 Grid Inference → Parquet"
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=0,
        help="Grid size N for NxN tiling (0 = no grid, full image)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (controls tiles per GPU forward pass)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, len(os.sched_getaffinity(0))),
        help="Number of DataLoader workers for parallel image loading",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to model checkpoint (.pth or .pth.tar)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save parquet output",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Process only 5 images for pipeline validation",
    )
    parser.add_argument(
        "--gpu-check",
        type=int,
        default=0,
        help="Process N images for GPU utilization profiling",
    )
    parser.add_argument(
        "--save-cls",
        action="store_true",
        help="Extract per-tile CLS embeddings (768-d) instead of probs; "
             "output to cls_<grid_label>.parquet alongside the existing probs parquet.",
    )
    return parser.parse_args()


def save_probs_to_parquet(
    image_names: list[str],
    image_probs: dict[int, torch.Tensor],
    grid_size: int,
    tiles_per_image: int,
    output_path: Path,
):
    """Save probability tensors to parquet with one row per tile.

    Schema: image_name (str), tile_idx (int), probs (binary float16 vector of 7806).
    Reconstruct with: np.frombuffer(row["probs"], dtype=np.float16)
    """
    rows = {
        "image_name": [],
        "tile_idx": [],
        "probs": [],
    }
    for idx, name in enumerate(image_names):
        probs_f16 = image_probs[idx].numpy().astype(np.float16)  # (T, 7806)
        for t in range(tiles_per_image):
            rows["image_name"].append(name)
            rows["tile_idx"].append(t)
            rows["probs"].append(probs_f16[t].tobytes())

    table = pa.table(
        {
            "image_name": pa.array(rows["image_name"], type=pa.string()),
            "tile_idx": pa.array(rows["tile_idx"], type=pa.int32()),
            "probs": pa.array(rows["probs"], type=pa.binary()),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Parquet saved: {output_path} ({file_size_mb:.1f} MB)")


def save_cls_to_parquet(
    image_names: list[str],
    image_cls: dict[int, torch.Tensor],
    tiles_per_image: int,
    output_path: Path,
):
    """Save per-tile CLS embeddings to parquet, schema mirrors grid_NxN.parquet.

    Schema: image_name (str), tile_idx (int32), cls_embedding (binary float16[768]).
    Reconstruct with: np.frombuffer(row["cls_embedding"], dtype=np.float16)
    """
    rows = {"image_name": [], "tile_idx": [], "cls_embedding": []}
    for idx, name in enumerate(image_names):
        cls_f16 = image_cls[idx].numpy().astype(np.float16)  # (T, 768)
        for t in range(tiles_per_image):
            rows["image_name"].append(name)
            rows["tile_idx"].append(t)
            rows["cls_embedding"].append(cls_f16[t].tobytes())
    table = pa.table(
        {
            "image_name": pa.array(rows["image_name"], type=pa.string()),
            "tile_idx": pa.array(rows["tile_idx"], type=pa.int32()),
            "cls_embedding": pa.array(rows["cls_embedding"], type=pa.binary()),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Parquet saved: {output_path} ({file_size_mb:.1f} MB)")


def main():
    args = parse_args()

    grid_size = args.grid_size
    grid_label = f"grid_{grid_size}x{grid_size}" if grid_size > 0 else "no_grid"
    dev_mode = args.fast_dev_run or args.gpu_check > 0

    print("=" * 60)
    print("PlantCLEF 2026 — Stage 1: DINOv2 Grid Inference")
    print("=" * 60)
    print(f"Grid size:    {grid_label}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Model:        {args.model_path}")
    print(f"Data dir:     {args.data_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Num workers:  {args.num_workers}")
    print(f"Dev mode:     {'fast-dev-run' if args.fast_dev_run else f'gpu-check({args.gpu_check})' if args.gpu_check else 'off'}")
    print(f"Output:       {'CLS embeddings (768-d)' if args.save_cls else 'softmax probs (7806-d)'}")
    print()

    # ---- Load model ----
    print("Loading model...")
    model, transforms = load_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # ---- Discover images ----
    data_dir = Path(args.data_dir)
    image_paths = sorted(data_dir.glob("*.jpg"))
    print(f"Images found: {len(image_paths)}")

    if args.fast_dev_run:
        image_paths = image_paths[:5]
        print(f"  [fast-dev-run] Limited to {len(image_paths)} images")
    elif args.gpu_check > 0:
        image_paths = image_paths[: args.gpu_check]
        print(f"  [gpu-check] Limited to {len(image_paths)} images")
    print()

    # ---- GPU monitor (for gpu-check mode) ----
    gpu_check_mode = args.gpu_check > 0
    gpu_monitor = None
    if gpu_check_mode:
        gpu_monitor = InferenceGPUMonitor(batch_size=args.batch_size)

    # ---- DataLoader ----
    num_images = len(image_paths)
    image_names = [p.stem for p in image_paths]

    dataset = TileDataset(image_paths, grid_size, transforms)
    tiles_per_image = dataset.tiles_per_image
    images_per_batch = max(1, args.batch_size // tiles_per_image)

    dataloader = DataLoader(
        dataset,
        batch_size=images_per_batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
        shuffle=False,
    )

    total_tiles = num_images * tiles_per_image
    print(f"Total tiles: {total_tiles} ({tiles_per_image} per image, {num_images} images)")
    print(f"Images per batch: {images_per_batch} ({images_per_batch * tiles_per_image} tiles)")
    print(f"Num workers: {args.num_workers}")
    print()

    # ---- Batched inference ----
    image_outputs: dict[int, torch.Tensor] = {}
    output_label = "CLS" if args.save_cls else "probs"
    t_start = time.time()

    for batch_indices, batch_tiles in tqdm(dataloader, desc=f"Inference ({grid_label}, {output_label})"):
        B, T = batch_tiles.shape[:2]
        flat_tiles = batch_tiles.view(B * T, *batch_tiles.shape[2:])

        if gpu_monitor:
            gpu_monitor.batch_start()

        flat_tiles = flat_tiles.to(device, non_blocking=True)
        with torch.no_grad():
            if args.save_cls:
                feats = model.forward_features(flat_tiles)
                output = model.forward_head(feats, pre_logits=True)  # (B*T, 768)
            else:
                logits = model(flat_tiles)
                output = torch.softmax(logits, dim=1)  # (B*T, 7806)

        if gpu_monitor:
            gpu_monitor.batch_end(batch_size=flat_tiles.shape[0])

        # Reshape back to (B, T, D) and store per image
        output = output.view(B, T, -1).cpu()
        for i, img_idx in enumerate(batch_indices.tolist()):
            image_outputs[img_idx] = output[i]  # (T, D) where D = 7806 (probs) or 768 (CLS)

    elapsed = time.time() - t_start

    # ---- Report ----
    print()
    print(f"Inference complete: {num_images} images, {total_tiles} tiles")
    print(f"Time: {elapsed:.1f}s ({elapsed / max(num_images, 1):.2f}s/image)")

    # ---- GPU profiling output ----
    if gpu_monitor:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        summary_file = summary_dir / f"{grid_label}_{job_id}.txt"
        gpu_monitor.print_summary(save_path=summary_file)
        gpu_monitor.shutdown()

    if dev_mode:
        print()
        print(f"=== Dev mode — sample {output_label} shape check ===")
        for idx in range(min(3, num_images)):
            p = image_outputs[idx]
            if args.save_cls:
                # Train-CLS mean L2 ≈ 18.6 (raw, pre-norm); test should be similar.
                print(f"  {image_names[idx]}: shape={tuple(p.shape)}, "
                      f"L2_norm={p.norm(dim=1).mean():.4f}, max={p.max():.4f}")
            else:
                print(f"  {image_names[idx]}: shape={tuple(p.shape)}, "
                      f"sum={p.sum(dim=1).mean():.4f}, max={p.max():.4f}")
        return

    # ---- Save to parquet ----
    if args.save_cls:
        output_path = Path(args.output_dir) / f"cls_{grid_label}.parquet"
        save_cls_to_parquet(image_names, image_outputs, tiles_per_image, output_path)
        print(f"\nCLS extraction complete. Used by exp 003 TileQ-Decoder.")
    else:
        output_path = Path(args.output_dir) / f"{grid_label}.parquet"
        save_probs_to_parquet(image_names, image_outputs, grid_size, tiles_per_image, output_path)
        print(f"\nStage 1 complete. Run stage 2 (exp 002):")
        print(f"  python ../002-classification-methods/classify.py --input {output_path} --method expanded-list --top-k 5 --max-species 20")


if __name__ == "__main__":
    main()
