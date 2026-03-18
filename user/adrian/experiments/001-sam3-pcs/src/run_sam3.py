#!/usr/bin/env python3
"""SAM3 Promptable Concept Segmentation — batch inference for PlantCLEF 2026.

Processes images using facebook/sam3 via 🤗 transformers.
Results are written to a single Parquet file (default) or .npz archive per
task to keep inode usage low on network scratch storage.

-- Local mode --
    python src/run_sam3.py \
        --input_dir  $TMPDIR/some_species_dir \
        --output_dir ~/scratch/plantclef-2026/sam3_outputs/task_00 \
        --prompt     "plant"

-- Remote streaming mode --
    python src/run_sam3.py \
        --remote      acheung46@login-phoenix.pace.gatech.edu \
        --remote_dir  ~/ps-clef2026_plantclef-0/train/PlantCLEF2024/train \
        --species_ids species_list.txt \
        --output_dir  ~/scratch/plantclef-2026/sam3_outputs/task_00 \
        --prompt      "plant"

Loading Parquet results:
    import pandas as pd, numpy as np
    from run_sam3 import rle_decode

    df = pd.read_parquet("results.parquet")
    row = df.iloc[0]
    rle   = np.frombuffer(row["rle"], dtype=np.int32)
    shape = (row["shape_h"], row["shape_w"])
    mask  = rle_decode(rle, shape)   # bool (H, W)

Loading NPZ results:
    data    = np.load("results.npz", allow_pickle=True)
    results = data["results"].item()
    from run_sam3 import rle_decode
    entry = results["1355868/abc123.jpg"]
    mask  = rle_decode(entry["rle"][0], entry["shape"])
"""

import argparse
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------
# One row per detected instance. Images with no detections get one row with
# mask_idx = -1 and null values for mask-specific fields.

PARQUET_SCHEMA = pa.schema([
    pa.field("species_id", pa.string()),
    pa.field("filename",   pa.string()),
    pa.field("prompt",     pa.string()),
    pa.field("shape_h",    pa.int32()),
    pa.field("shape_w",    pa.int32()),
    pa.field("mask_idx",   pa.int16()),   # -1 = no detection
    pa.field("rle",        pa.binary()),  # int32 RLE array as raw bytes; b"" if no detection
    pa.field("box_x1",     pa.float32()),
    pa.field("box_y1",     pa.float32()),
    pa.field("box_x2",     pa.float32()),
    pa.field("box_y2",     pa.float32()),
    pa.field("score",      pa.float32()),
])

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM3 batch segmentation for PlantCLEF 2026")

    # ── local mode ──
    p.add_argument("--input_dir",      type=Path, default=None,
                   help="Local directory of images (local mode)")

    # ── remote streaming mode ──
    p.add_argument("--remote",         default=None,
                   help="user@host to rsync from (remote mode)")
    p.add_argument("--remote_dir",     default=None,
                   help="Base species directory on remote host")
    p.add_argument("--species_ids",    type=Path, default=None,
                   help="Text file with one species ID per line to process")

    # ── shared ──
    p.add_argument("--output_dir",     required=True, type=Path,
                   help="Destination dir for output file")
    p.add_argument("--format",         default="parquet", choices=["parquet", "npz"],
                   help="Output format (default: parquet)")
    p.add_argument("--prompt",         default="plant",
                   help="Text concept prompt (default: 'plant')")
    p.add_argument("--hf_token",       default=None,
                   help="HuggingFace token (optional; falls back to huggingface-cli login)")
    p.add_argument("--batch_size",     type=int, default=8,
                   help="Images per forward pass (default: 8)")
    p.add_argument("--num_workers",    type=int, default=8,
                   help="DataLoader worker threads (default: 8)")
    p.add_argument("--threshold",      type=float, default=0.5,
                   help="Detection score threshold (default: 0.5)")
    p.add_argument("--mask_threshold", type=float, default=0.5,
                   help="Mask logit threshold (default: 0.5)")
    p.add_argument("--stage_dir",      type=Path,
                   default=Path(os.environ.get("TMPDIR", "/tmp")) / "sam3_staging",
                   help="Local staging dir for rsynced images (default: $TMPDIR/sam3_staging)")

    args = p.parse_args()

    remote_mode = args.remote is not None
    local_mode  = args.input_dir is not None
    if remote_mode == local_mode:
        p.error("Specify either --input_dir (local) or --remote + --remote_dir + --species_ids (remote)")
    if remote_mode and (args.remote_dir is None or args.species_ids is None):
        p.error("--remote requires --remote_dir and --species_ids")

    return args


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


class ImageChunkDataset(Dataset):
    def __init__(self, image_dir: Path) -> None:
        self.paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return path.name, img
        except (OSError, UnidentifiedImageError) as exc:
            log.warning(f"Skipping corrupt image '{path.name}': {exc}")
            return path.name, None


def collate_fn(batch):
    names, images, sizes = [], [], []
    for name, img in batch:
        if img is not None:
            names.append(name)
            images.append(img)
            sizes.append([img.height, img.width])
    return names, images, sizes


# ---------------------------------------------------------------------------
# RLE helpers  (column-major / COCO convention)
# ---------------------------------------------------------------------------

def rle_encode(mask: np.ndarray) -> np.ndarray:
    """Encode a boolean (H, W) mask as a COCO-style column-major RLE count array."""
    flat = mask.flatten(order="F").view(np.uint8)
    changes = np.concatenate(([True], flat[1:] != flat[:-1], [True]))
    counts = np.diff(np.where(changes)[0]).astype(np.int32)
    if flat[0]:
        counts = np.concatenate(([np.int32(0)], counts))
    return counts


def rle_decode(rle: np.ndarray, shape) -> np.ndarray:
    """Decode a COCO-style column-major RLE back to a boolean (H, W) mask."""
    h, w = int(shape[0]), int(shape[1])
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, length in enumerate(rle):
        if i % 2 == 1:
            flat[pos : pos + length] = True
        pos += length
    return flat.reshape((h, w), order="F")


# ---------------------------------------------------------------------------
# Remote helpers
# ---------------------------------------------------------------------------

def rsync_species(remote: str, remote_dir: str, species_id: str, stage_dir: Path) -> Path:
    local_path = stage_dir / species_id
    local_path.mkdir(parents=True, exist_ok=True)
    src = f"{remote}:{remote_dir}/{species_id}/"
    cmd = ["rsync", "-aq", "--no-perms", src, str(local_path) + "/"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"rsync failed for {species_id}: {result.stderr.strip()}")
    return local_path


def cleanup_species(local_path: Path) -> None:
    shutil.rmtree(local_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Result accumulators
# ---------------------------------------------------------------------------

def accumulate_parquet(rows: list, key: str, size: list, masks, boxes, scores, prompt: str = "") -> None:
    """Append Parquet rows for one image."""
    parts = key.split("/", 1)
    species_id = parts[0] if len(parts) == 2 else ""
    filename   = parts[1] if len(parts) == 2 else parts[0]
    h, w = size

    if len(masks) == 0:
        rows.append({
            "species_id": species_id, "filename": filename,
            "prompt": prompt,
            "shape_h": h, "shape_w": w,
            "mask_idx": -1, "rle": b"",
            "box_x1": float("nan"), "box_y1": float("nan"),
            "box_x2": float("nan"), "box_y2": float("nan"),
            "score": float("nan"),
        })
        return

    boxes_np  = boxes.float().cpu().numpy()
    scores_np = scores.float().cpu().numpy()
    for i, m in enumerate(masks):
        rle = rle_encode(m.cpu().numpy().astype(bool))
        rows.append({
            "species_id": species_id, "filename": filename,
            "prompt": prompt,
            "shape_h": h, "shape_w": w,
            "mask_idx": i, "rle": rle.tobytes(),
            "box_x1": float(boxes_np[i, 0]), "box_y1": float(boxes_np[i, 1]),
            "box_x2": float(boxes_np[i, 2]), "box_y2": float(boxes_np[i, 3]),
            "score": float(scores_np[i]),
        })


def accumulate_npz(results: dict, key: str, size: list, masks, boxes, scores) -> None:
    """Accumulate NPZ results dict for one image."""
    if len(masks) == 0:
        results[key] = {
            "rle":    np.empty(0, dtype=object),
            "shape":  np.array(size, dtype=np.int32),
            "boxes":  np.empty((0, 4), dtype=np.float32),
            "scores": np.empty((0,), dtype=np.float32),
        }
    else:
        results[key] = {
            "rle":    np.array(
                [rle_encode(m.cpu().numpy().astype(bool)) for m in masks],
                dtype=object,
            ),
            "shape":  np.array(size, dtype=np.int32),
            "boxes":  boxes.float().cpu().numpy(),
            "scores": scores.float().cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# Inference on one directory
# ---------------------------------------------------------------------------

def process_directory(
    image_dir: Path,
    processor,
    model,
    device,
    args,
    rows: list,
    npz_results: dict,
    key_prefix: str = "",
) -> tuple[int, int]:
    """Run SAM3 inference on all images in image_dir, appending to rows or npz_results."""
    dataset = ImageChunkDataset(image_dir)
    if len(dataset) == 0:
        log.warning(f"No images found in {image_dir} — skipping.")
        return 0, 0

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    n_processed = n_skipped = 0

    with torch.inference_mode():
        for names, pil_images, original_sizes in loader:
            if not names:
                n_skipped += args.batch_size
                continue
            try:
                inputs = processor(
                    images=pil_images,
                    text=[args.prompt] * len(pil_images),
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                batch_results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=args.threshold,
                    mask_threshold=args.mask_threshold,
                    target_sizes=original_sizes,
                )
            except Exception as exc:
                log.error(f"Batch failed — skipping {names}: {exc}")
                n_skipped += len(names)
                continue

            for name, res, size in zip(names, batch_results, original_sizes):
                key = f"{key_prefix}/{name}" if key_prefix else name
                if args.format == "parquet":
                    accumulate_parquet(rows, key, size, res["masks"], res["boxes"], res["scores"], prompt=args.prompt)
                else:
                    accumulate_npz(npz_results, key, size, res["masks"], res["boxes"], res["scores"])
                n_processed += 1

    return n_processed, n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {device}  |  Prompt: '{args.prompt}'  |  Batch: {args.batch_size}  |  Format: {args.format}")

    # ── Load model ──────────────────────────────────────────────────────────
    log.info("Loading SAM3 processor and model (facebook/sam3) ...")
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        token=hf_token,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    log.info("Model ready.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list        = []   # parquet mode
    npz_results: dict = {}   # npz mode

    n_total_processed = n_total_skipped = 0
    t_start = time.perf_counter()

    # ── Local mode ──────────────────────────────────────────────────────────
    if args.input_dir is not None:
        log.info(f"Local mode: {args.input_dir}")
        n_proc, n_skip = process_directory(
            args.input_dir, processor, model, device, args, rows, npz_results,
        )
        n_total_processed += n_proc
        n_total_skipped   += n_skip

    # ── Remote streaming mode ────────────────────────────────────────────────
    else:
        species_ids = [s.strip() for s in args.species_ids.read_text().splitlines() if s.strip()]
        log.info(f"Remote mode: {len(species_ids)} species from {args.remote}:{args.remote_dir}")
        args.stage_dir.mkdir(parents=True, exist_ok=True)

        for species_id in tqdm(species_ids, desc="Species", unit="sp"):
            try:
                local_path = rsync_species(args.remote, args.remote_dir, species_id, args.stage_dir)
            except RuntimeError as exc:
                log.error(f"Skipping species {species_id}: {exc}")
                continue

            n_imgs = sum(1 for p in local_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
            log.info(f"Species {species_id}: {n_imgs} images staged")

            n_proc, n_skip = process_directory(
                local_path, processor, model, device, args, rows, npz_results,
                key_prefix=species_id,
            )
            n_total_processed += n_proc
            n_total_skipped   += n_skip
            cleanup_species(local_path)

    # ── Throughput report ────────────────────────────────────────────────────
    elapsed    = time.perf_counter() - t_start
    throughput = n_total_processed / elapsed if elapsed > 0 else 0.0
    log.info(
        f"Done: {n_total_processed:,} processed, {n_total_skipped:,} skipped  "
        f"in {elapsed:.1f}s  ({throughput:.2f} img/s)"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.format == "parquet":
        out_file = args.output_dir / "results.parquet"
        table = pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)
        pq.write_table(table, out_file, compression="zstd")
        log.info(f"Saved {len(rows):,} rows → {out_file}  ({out_file.stat().st_size / 1e6:.1f} MB)")
    else:
        out_file = args.output_dir / "results.npz"
        np.savez_compressed(out_file, results=np.array(npz_results, dtype=object))
        log.info(f"Saved {len(npz_results):,} entries → {out_file}  ({out_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
