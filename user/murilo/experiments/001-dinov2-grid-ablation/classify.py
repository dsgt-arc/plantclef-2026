#!/usr/bin/env python3
"""Grid ablation classification for PlantCLEF 2026.

Runs a frozen fine-tuned DINOv2 classifier with configurable grid tiling
on vegetation plot images. Each tile is classified independently, then
predictions are aggregated per image to produce a submission CSV.

Usage:
    # Full image (no grid)
    python classify.py --grid-size 0

    # 4x4 grid with union aggregation, no cap (default)
    python classify.py --grid-size 4

    # 4x4 grid with top-3 per tile instead of default top-5
    python classify.py --grid-size 4 --top-k 3

    # 4x4 grid capped at 20 species per image
    python classify.py --grid-size 4 --max-species 20

    # Pipeline validation (5 images)
    python classify.py --grid-size 3 --fast-dev-run

    # GPU profiling (20 images)
    python classify.py --grid-size 3 --gpu-check 20
"""

import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from plantclef.gpu_monitor import InferenceGPUMonitor
from plantclef.inference.dataset import TileDataset
from plantclef.model import load_class_mapping, load_model

# Default paths on PACE shared storage
DEFAULT_MODEL_PATH = "/storage/project/ps-clef2026_plantclef-0/shared/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth"
DEFAULT_CLASS_MAPPING = "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/class_id_to_species_mapping/class_mapping.txt"
DEFAULT_DATA_DIR = "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/PlanCLEF_test_data/images"


def parse_args():
    parser = argparse.ArgumentParser(
        description="PlantCLEF 2026 DINOv2 Grid Ablation Inference"
    )
    # Core parameters
    parser.add_argument(
        "--grid-size",
        type=int,
        default=0,
        help="Grid size N for NxN tiling (0 = no grid, full image)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="union",
        choices=["union"],
        help="Aggregation method for tile predictions (default: union)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K predictions per tile",
    )
    parser.add_argument(
        "--max-species",
        type=int,
        default=None,
        help="Max species per image (default: no cap)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, len(os.sched_getaffinity(0))),
        help="Number of DataLoader workers for parallel image loading",
    )

    # Paths
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to model checkpoint (.pth or .pth.tar)",
    )
    parser.add_argument(
        "--class-mapping-path",
        type=str,
        default=DEFAULT_CLASS_MAPPING,
        help="Path to class_mapping.txt",
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
        default="artifacts",
        help="Directory to save submission CSV",
    )

    # Dev/profiling modes
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

    return parser.parse_args()


def aggregate_union(
    tile_probs: torch.Tensor,
    top_k: int,
    cid_to_spid: dict[int, int],
    max_species: int | None = None,
) -> list[int]:
    """Aggregate tile predictions via union of top-k species.

    Each tile contributes its top-k predicted species. Species are sorted by
    frequency (how many tiles predicted each), then the expanded list (with
    repeats) is capped at max_species before deduplicating. This means
    high-frequency species consume more slots, naturally filtering out rare
    species that appeared in only 1-2 tiles.

    Example with 9 tiles, top-k=1, max_species=10:
        Expanded: [111]*8 + [222]*5 + [333]*3 + [444] + [555] + [666]
        After cap at 10: [111]*8 + [222]*2
        After dedup: [111, 222]

    Args:
        tile_probs: Tensor of shape (num_tiles, num_classes) with softmax probs.
        top_k: Number of top predictions per tile.
        cid_to_spid: Class index -> species ID mapping.
        max_species: Max entries in expanded frequency list before dedup,
            or None for no cap (returns all unique species).

    Returns:
        List of unique species IDs sorted by frequency (descending).
    """
    _, top_k_ids = tile_probs.topk(top_k, dim=1)
    species_counts = Counter()
    for tile_top_k in top_k_ids:
        for class_idx in tile_top_k.cpu().numpy():
            species_counts[cid_to_spid[int(class_idx)]] += 1

    if max_species is None:
        return [sp for sp, _ in species_counts.most_common()]

    # Build expanded list sorted by frequency, cap it, then deduplicate
    expanded = []
    for sp, count in species_counts.most_common():
        expanded.extend([sp] * count)
    expanded = expanded[:max_species]
    # Deduplicate preserving order (most frequent first)
    seen = set()
    result = []
    for sp in expanded:
        if sp not in seen:
            seen.add(sp)
            result.append(sp)
    return result


def main():
    args = parse_args()

    grid_label = (
        f"grid_{args.grid_size}x{args.grid_size}" if args.grid_size > 0 else "no_grid"
    )
    dev_mode = args.fast_dev_run or args.gpu_check > 0

    print("=" * 60)
    print("PlantCLEF 2026 — DINOv2 Grid Ablation")
    print("=" * 60)
    print(f"Grid size:    {grid_label}")
    print(f"Aggregation:  {args.aggregation}")
    print(f"Top-K:        {args.top_k}")
    print(f"Max species:  {args.max_species or 'no cap'}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Model:        {args.model_path}")
    print(f"Data dir:     {args.data_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Num workers:  {args.num_workers}")
    print(f"Dev mode:     {'fast-dev-run' if args.fast_dev_run else f'gpu-check({args.gpu_check})' if args.gpu_check else 'off'}")
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

    # ---- Load class mapping ----
    cid_to_spid = load_class_mapping(args.class_mapping_path)
    print(f"Class mapping loaded: {len(cid_to_spid)} classes")

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

    # ---- DataLoader: overlaps CPU (load/tile/transform) with GPU (inference) ----
    # Dataset is image-level: each item returns ALL tiles for one image.
    # --batch-size controls tiles per GPU forward pass; we compute images per
    # DataLoader batch so that total tiles per batch stays within budget.
    num_images = len(image_paths)
    image_names = [p.stem for p in image_paths]

    dataset = TileDataset(image_paths, args.grid_size, transforms)
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
    image_probs: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_images)}
    t_start = time.time()

    for batch_indices, batch_tiles in tqdm(dataloader, desc=f"Classifying ({grid_label})"):
        # batch_indices: (B,)
        # batch_tiles:   (B, tiles_per_image, C, H, W)
        B, T = batch_tiles.shape[:2]
        flat_tiles = batch_tiles.view(B * T, *batch_tiles.shape[2:])

        if gpu_monitor:
            gpu_monitor.batch_start()

        flat_tiles = flat_tiles.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(flat_tiles)
            probs = torch.softmax(logits, dim=1)

        if gpu_monitor:
            gpu_monitor.batch_end(batch_size=flat_tiles.shape[0])

        # Reshape back to (B, T, num_classes) and distribute to images
        probs = probs.view(B, T, -1).cpu()
        for i, img_idx in enumerate(batch_indices.tolist()):
            image_probs[img_idx] = probs[i]  # (T, num_classes)

    elapsed = time.time() - t_start

    # ---- Aggregate per image ----
    results = []
    for img_idx in range(num_images):
        tile_probs = image_probs[img_idx]  # (tiles_per_image, num_classes)
        species_ids = aggregate_union(
            tile_probs, args.top_k, cid_to_spid, args.max_species
        )
        results.append({"quadrat_id": image_names[img_idx], "species_ids": species_ids})

    # ---- Report ----
    print()
    print(f"Inference complete: {len(results)} images, {total_tiles} tiles")
    print(f"Time: {elapsed:.1f}s ({elapsed / len(results):.2f}s/image)")
    avg_species = sum(len(r["species_ids"]) for r in results) / max(len(results), 1)
    print(f"Avg species predicted per image: {avg_species:.1f}")

    # ---- GPU profiling output ----
    if gpu_monitor:
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        summary_file = summary_dir / f"{grid_label}_{job_id}.txt"
        gpu_monitor.print_summary(save_path=summary_file)
        gpu_monitor.shutdown()

    if dev_mode:
        print()
        print("=== Sample predictions (dev mode) ===")
        for r in results[:5]:
            sp = r["species_ids"][:10]
            print(f"  {r['quadrat_id']}: {sp}{'...' if len(r['species_ids']) > 10 else ''}")
        return

    # ---- Save submission CSV ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_label = f"_max{args.max_species}" if args.max_species else ""
    output_file = output_dir / f"{grid_label}_{args.aggregation}_topk{args.top_k}{max_label}.csv"

    df = pd.DataFrame(
        [
            {
                "quadrat_id": r["quadrat_id"],
                "species_ids": "[" + ", ".join(str(s) for s in r["species_ids"]) + "]",
            }
            for r in results
        ]
    )
    df.to_csv(output_file, sep=",", index=False, quoting=csv.QUOTE_ALL)

    print(f"\nSubmission saved to: {output_file}")
    print(f"  Rows: {len(df)}")


if __name__ == "__main__":
    main()
