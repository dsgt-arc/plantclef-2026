#!/usr/bin/env python3
"""SAM3 Promptable Concept Segmentation — batch inference for PlantCLEF 2026.

Processes a directory of JPEG images using facebook/sam3 via 🤗 transformers.
All results for the chunk are written to a single compressed .npz archive to
keep inode usage low on network scratch storage.

Usage:
    python src/run_sam3.py \
        --input_dir  $TMPDIR/chunk_000042 \
        --output_dir ~/scratch/plantclef-2026/sam3_outputs/chunk_000042 \
        --prompt     "plant" \
        [--hf_token  hf_...] \
        [--batch_size 8] \
        [--num_workers 8]

Loading results:
    import numpy as np
    data    = np.load("results.npz", allow_pickle=True)
    results = data["results"].item()   # dict: filename -> {rle, shape, boxes, scores}

    from run_sam3 import rle_decode
    mask = rle_decode(results["photo.jpg"]["rle"][0], results["photo.jpg"]["shape"])
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM3 batch segmentation for PlantCLEF 2026")
    p.add_argument("--input_dir",     required=True,  type=Path,  help="Directory of input images")
    p.add_argument("--output_dir",    required=True,  type=Path,  help="Destination dir for results.npz")
    p.add_argument("--prompt",        default="plant",             help="Text concept prompt (default: 'plant')")
    p.add_argument("--hf_token",      default=None,                help="HuggingFace token (optional; falls back to huggingface-cli login)")
    p.add_argument("--batch_size",    type=int, default=8,         help="Images per forward pass (default: 8)")
    p.add_argument("--num_workers",   type=int, default=8,         help="DataLoader worker threads (default: 8)")
    p.add_argument("--threshold",     type=float, default=0.5,    help="Detection score threshold (default: 0.5)")
    p.add_argument("--mask_threshold",type=float, default=0.5,    help="Mask logit threshold (default: 0.5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


class ImageChunkDataset(Dataset):
    def __init__(self, image_dir: Path) -> None:
        self.paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        log.info(f"Found {len(self.paths):,} images in {image_dir}")

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
    """Filter out corrupt images; return (names, pil_images, original_sizes)."""
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
    """Encode a boolean (H, W) mask as a COCO-style column-major RLE count array.

    Returns an int32 array of run lengths, starting with the count of 0-pixels.
    """
    flat = mask.flatten(order="F").view(np.uint8)
    # Locate every transition between 0 and 1
    changes = np.concatenate(([True], flat[1:] != flat[:-1], [True]))
    counts = np.diff(np.where(changes)[0]).astype(np.int32)
    # COCO RLE always starts with count of zeros; prepend 0 if mask starts with 1
    if flat[0]:
        counts = np.concatenate(([np.int32(0)], counts))
    return counts


def rle_decode(rle: np.ndarray, shape) -> np.ndarray:
    """Decode a COCO-style column-major RLE back to a boolean (H, W) mask."""
    h, w = int(shape[0]), int(shape[1])
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, length in enumerate(rle):
        if i % 2 == 1:  # odd index = 1-run
            flat[pos : pos + length] = True
        pos += length
    return flat.reshape((h, w), order="F")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")  # None → uses huggingface-cli credentials
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Device: {device}  |  Prompt: '{args.prompt}'  |  Batch: {args.batch_size}")

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

    # ── DataLoader ──────────────────────────────────────────────────────────
    dataset = ImageChunkDataset(args.input_dir)
    if len(dataset) == 0:
        log.warning("No images found — exiting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ── Inference ───────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    n_processed = 0
    n_skipped = 0
    t_start = time.perf_counter()

    with torch.inference_mode():
        for names, pil_images, original_sizes in tqdm(loader, desc="Batches", unit="batch"):
            if not names:
                # All images in this batch were corrupt
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
                masks  = res["masks"]    # list of bool tensors (H, W)
                boxes  = res["boxes"]    # list of float tensors (4,)  — absolute xyxy
                scores = res["scores"]   # list of float tensors scalar

                if len(masks) == 0:
                    results[name] = {
                        "rle":    np.empty(0, dtype=object),
                        "shape":  np.array(size, dtype=np.int32),
                        "boxes":  np.empty((0, 4), dtype=np.float32),
                        "scores": np.empty(0, dtype=np.float32),
                    }
                else:
                    results[name] = {
                        "rle":    np.array(
                            [rle_encode(m.cpu().numpy().astype(bool)) for m in masks],
                            dtype=object,
                        ),
                        "shape":  np.array(size, dtype=np.int32),
                        "boxes":  torch.stack(boxes).cpu().numpy().astype(np.float32),
                        "scores": torch.stack(scores).cpu().numpy().astype(np.float32),
                    }
                n_processed += 1

    # ── Throughput report ───────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    throughput = n_processed / elapsed if elapsed > 0 else 0.0
    log.info(
        f"Done: {n_processed:,} processed, {n_skipped:,} skipped  "
        f"in {elapsed:.1f}s  ({throughput:.2f} img/s)"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    # Pack the entire results dict as a single 0-d object array.
    # Load with: results = np.load("results.npz", allow_pickle=True)["results"].item()
    out_file = args.output_dir / "results.npz"
    np.savez_compressed(out_file, results=np.array(results, dtype=object))
    log.info(f"Saved {len(results):,} entries → {out_file}  ({out_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
