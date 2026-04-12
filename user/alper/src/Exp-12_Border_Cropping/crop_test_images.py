"""Crop uniform border margins from test images.

Removes a fixed percentage of pixels from all four sides of each image.
Creates separate output directories for each crop percentage.

Usage:
  python crop_test_images.py \
    --image-dir /path/to/test/images \
    --output-root /path/to/output \
    --crop-pcts 3,5,7
"""
from __future__ import annotations

import argparse
import os
import time

from PIL import Image


def crop_border(
    img: Image.Image, pct: float, resize_back: bool = False,
) -> Image.Image:
    """Crop pct% of pixels from each side of the image.

    Args:
        img: Input PIL image.
        pct: Percentage to remove per side (e.g. 5.0 = 5%).
        resize_back: If True, resize the cropped image back to the
            original dimensions so the pipeline sees the same input
            size. Isolates the content-removal effect from the
            dimension-change effect.

    Returns:
        Cropped (and optionally resized) image.
    """
    w, h = img.size
    crop_w = int(w * pct / 100.0)
    crop_h = int(h * pct / 100.0)
    cropped = img.crop((crop_w, crop_h, w - crop_w, h - crop_h))
    if resize_back:
        cropped = cropped.resize((w, h), Image.LANCZOS)
    return cropped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Crop uniform border margins from test images"
    )
    ap.add_argument(
        "--image-dir", required=True,
        help="Directory containing original test images",
    )
    ap.add_argument(
        "--output-root", required=True,
        help="Root output directory (subdirs created per crop %%)",
    )
    ap.add_argument(
        "--crop-pcts", default="3,5,7",
        help="Comma-separated crop percentages (default: 3,5,7)",
    )
    ap.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG save quality for cropped images (default: 95)",
    )
    ap.add_argument(
        "--resize-back", action="store_true",
        help="Resize cropped images back to original dimensions "
             "(isolates content removal from dimension change)",
    )
    args = ap.parse_args()

    pcts = [float(p) for p in args.crop_pcts.split(",")]

    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(image_files)} images in {args.image_dir}")

    suffix = "_resized" if args.resize_back else ""

    for pct in pcts:
        out_dir = os.path.join(args.output_root, f"crop_{pct:.0f}pct{suffix}")
        os.makedirs(out_dir, exist_ok=True)

        mode = "crop + resize back" if args.resize_back else "crop only"
        print(f"\n=== Cropping at {pct}% per side ({mode}) -> {out_dir} ===")
        t0 = time.time()

        for i, fname in enumerate(image_files):
            img = Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
            cropped = crop_border(img, pct, resize_back=args.resize_back)
            cropped.save(
                os.path.join(out_dir, fname),
                format="JPEG",
                quality=args.jpeg_quality,
            )

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i + 1}/{len(image_files)}] {rate:.1f} img/s")

        elapsed = time.time() - t0
        # Show size change for first image
        sample = Image.open(os.path.join(args.image_dir, image_files[0]))
        orig_w, orig_h = sample.size
        crop_w = int(orig_w * pct / 100.0)
        crop_h = int(orig_h * pct / 100.0)
        new_w, new_h = orig_w - 2 * crop_w, orig_h - 2 * crop_h
        area_pct = (new_w * new_h) / (orig_w * orig_h) * 100

        print(f"  Done in {elapsed:.1f}s ({len(image_files) / elapsed:.1f} img/s)")
        print(f"  Sample: {orig_w}x{orig_h} -> {new_w}x{new_h} "
              f"({crop_w}px/side, {area_pct:.1f}% area retained)")


if __name__ == "__main__":
    main()
