"""Extract ViT-Large CLS embeddings at a specified tiling scale.

Mirrors Exp-7_Multiscale_tiling/extract_multiscale_embeddings.py and adds
support for two finetuned ViT-Large variants from Exp-13:

    - vit_large_patch14_reg4_dinov2.lvd142m  (tile_size=518, patch14)
    - vit_large_patch16_dinov3.lvd1689m      (tile_size=512, patch16)

Differences vs the ConvNext extractor (Exp-10):
  - CLS token obtained via direct indexing `features[:, 0, :]` (same as Exp-7),
    not via `forward_head(pre_logits=True)`. This is the authoritative pattern
    used by the production ensemble (Exp-7 / Exp-8 / Exp-11).
  - Logits computed as `model.head(cls_tokens)` (not `model.forward_head(features)`).
  - `img_size=tile_size` passed to `timm.create_model` so positional embeddings
    are interpolated to the inference resolution.

For each image:
  1. Center-crop to square
  2. Resize to (scale * tile_size) x (scale * tile_size) using Lanczos
  3. Cut into scale^2 non-overlapping tiles of tile_size x tile_size
  4. Optionally apply JPEG recompression per tile
  5. Run ViT -> CLS embedding (1024) + logits (7806)

Output parquet schema matches Exp-7 so the existing aggregation pipeline
(aggregate_tiles.py, multiscale_aggregate.py) consumes it unchanged.

Usage:
  python extract_vit_embeddings.py \\
    --scale 9 \\
    --tile-size 518 \\
    --image-dir /path/to/test/images \\
    --model-path /path/to/vit_dinov2_large_phase2_best.pth.tar \\
    --model-name vit_large_patch14_reg4_dinov2.lvd142m \\
    --output-dir /path/to/output/scale_9x9

  # DINOv3 variant uses tile_size=512 (patch16 divisibility)
  python extract_vit_embeddings.py \\
    --scale 9 --tile-size 512 \\
    --image-dir /path/to/test/images \\
    --model-path /path/to/vit_dinov3_large_phase2_best.pth.tar \\
    --model-name vit_large_patch16_dinov3.lvd1689m \\
    --output-dir /path/to/output/scale_9x9
"""
from __future__ import annotations

import argparse
import io
import os
import time
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import timm
import torch
from PIL import Image
from torchvision import transforms

DEFAULT_TILE_SIZE = 518  # DINOv2 ViT-L native; override to 512 for DINOv3
NUM_CLASSES = 7806

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def center_crop_square(img: Image.Image) -> Image.Image:
    """Crop the center square from a possibly rectangular image."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def tile_image(img: Image.Image, scale: int, tile_size: int) -> List[Image.Image]:
    """Cut a square image into scale x scale non-overlapping tiles.

    The image is first resized to (scale * tile_size) x (scale * tile_size)
    using Lanczos, then sliced into scale^2 tiles of tile_size x tile_size.
    """
    target_size = scale * tile_size
    img_resized = img.resize((target_size, target_size), Image.LANCZOS)

    tiles = []
    for row in range(scale):
        for col in range(scale):
            left = col * tile_size
            top = row * tile_size
            tile = img_resized.crop(
                (left, top, left + tile_size, top + tile_size)
            )
            tiles.append(tile)
    return tiles


def apply_jpeg_compression(
    img: Image.Image,
    quality: int = 85,
    subsampling: int = 1,
) -> Image.Image:
    """Round-trip JPEG compression on a PIL image.

    Aligns pixel statistics with the training distribution (which is JPEG).
    subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).copy()


def build_normalize_transform() -> transforms.Compose:
    """Standard ImageNet normalization (no resize -- tiles already tile_size)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def extract_embeddings_batch(
    model: torch.nn.Module,
    tiles: List[Image.Image],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ViT on a batch of tiles.

    Returns:
        embeddings: [N, embed_dim] float32  (CLS tokens, direct index)
        logits:     [N, 7806] float32
        max_probs:  [N] float32             (max softmax prob per tile)
    """
    all_embs = []
    all_logits = []
    all_max_probs = []

    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start : start + batch_size]
        tensors = torch.stack([transform(t) for t in batch_tiles]).to(device)

        # ViT forward_features returns [B, 1 + extra + N, D]
        #   DINOv2 (reg4): [B, 1 + 4 + N_patches, 1024]  -- 4 register tokens
        #   DINOv3:        [B, 1 + 4 + N_patches, 1024]  -- 4 storage tokens
        # CLS is at index 0 in both cases.
        features = model.forward_features(tensors)
        cls_tokens = features[:, 0, :]  # [B, 1024]

        logits = model.head(cls_tokens)  # [B, 7806]

        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # [B]

        all_embs.append(cls_tokens.cpu().float().numpy())
        all_logits.append(logits.cpu().float().numpy())
        all_max_probs.append(max_probs.cpu().float().numpy())

    return (
        np.concatenate(all_embs, axis=0),
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_max_probs, axis=0),
    )


def process_images(
    image_dir: str,
    scale: int,
    tile_size: int,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    jpeg_quality: int = 0,
    jpeg_subsampling: int = 1,
    batch_size: int = 16,
) -> pa.Table:
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Processing {len(image_files)} images at scale {scale}x{scale} "
          f"({scale**2} tiles/image, tile_size={tile_size})")

    all_image_names = []
    all_tile_indices = []
    all_embeddings = []
    all_logits_bytes = []
    all_max_probs = []

    t0 = time.time()
    for img_idx, fname in enumerate(image_files):
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert("RGB")

        img_sq = center_crop_square(img)
        tiles = tile_image(img_sq, scale, tile_size)

        if jpeg_quality > 0:
            tiles = [
                apply_jpeg_compression(t, jpeg_quality, jpeg_subsampling)
                for t in tiles
            ]

        embs, logits, max_p = extract_embeddings_batch(
            model, tiles, transform, device, batch_size
        )

        for tile_idx in range(len(tiles)):
            all_image_names.append(fname)
            all_tile_indices.append(tile_idx)
            emb_f16 = embs[tile_idx].astype(np.float16).tobytes()
            logit_f16 = logits[tile_idx].astype(np.float16).tobytes()
            all_embeddings.append(emb_f16)
            all_logits_bytes.append(logit_f16)
            all_max_probs.append(float(max_p[tile_idx]))

        if (img_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (img_idx + 1) / elapsed
            eta = (len(image_files) - img_idx - 1) / rate
            print(f"  [{img_idx + 1}/{len(image_files)}] "
                  f"{rate:.1f} img/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    total_tiles = len(all_image_names)
    print(f"\nExtracted {total_tiles} tiles from {len(image_files)} images "
          f"in {elapsed:.1f}s ({total_tiles / elapsed:.1f} tiles/s)")

    table = pa.Table.from_pydict({
        "image_name": all_image_names,
        "tile_index": all_tile_indices,
        "embedding_orig_f16": all_embeddings,
        "teacher_logits_f16": all_logits_bytes,
        "max_prob": all_max_probs,
        "passed_aug_consistency": [True] * total_tiles,
    })
    return table


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract ViT-Large CLS embeddings at a specified tiling scale"
    )
    ap.add_argument("--scale", type=int, required=True,
                    help="Tiling scale (1=1x1, 2=2x2, 5=5x5, 9=9x9)")
    ap.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                    help=f"Tile size in pixels (default: {DEFAULT_TILE_SIZE} "
                         "for DINOv2 patch14; use 512 for DINOv3 patch16)")
    ap.add_argument("--image-dir", required=True,
                    help="Directory containing test images")
    ap.add_argument("--model-path", required=True,
                    help="Path to finetuned ViT checkpoint (.pth or .pth.tar)")
    ap.add_argument("--model-name", required=True,
                    help="timm model name, e.g. "
                         "'vit_large_patch14_reg4_dinov2.lvd142m' or "
                         "'vit_large_patch16_dinov3.lvd1689m'")
    ap.add_argument("--output-dir", required=True,
                    help="Output directory for parquet")
    ap.add_argument("--jpeg-quality", type=int, default=85,
                    help="JPEG quality (0=disabled, 85=default)")
    ap.add_argument("--jpeg-subsampling", type=int, default=1,
                    help="JPEG subsampling (0=4:4:4, 1=4:2:2, 2=4:2:0)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="GPU batch size for inference")
    ap.add_argument("--output-filename", default=None,
                    help="Output parquet filename (default: scale_NxN.parquet)")

    args = ap.parse_args()

    if args.scale < 1:
        raise ValueError("Scale must be >= 1")

    print(f"Loading ViT model: {args.model_name}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Inference resolution: {args.tile_size}x{args.tile_size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # img_size matches tile_size so pretrained positional embeddings get
    # interpolated correctly (required for ViT; size must be divisible by patch).
    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=NUM_CLASSES,
        img_size=args.tile_size,
        checkpoint_path=args.model_path,
    )
    model = model.to(device).eval()

    embed_dim = model.num_features
    print(f"Model loaded on {device} (embed_dim={embed_dim}, "
          f"num_classes={NUM_CLASSES})")

    transform = build_normalize_transform()

    table = process_images(
        image_dir=args.image_dir,
        scale=args.scale,
        tile_size=args.tile_size,
        model=model,
        transform=transform,
        device=device,
        jpeg_quality=args.jpeg_quality,
        jpeg_subsampling=args.jpeg_subsampling,
        batch_size=args.batch_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.output_filename or f"scale_{args.scale}x{args.scale}.parquet"
    out_path = os.path.join(args.output_dir, out_name)
    pq.write_table(table, out_path, compression="snappy")
    print(f"\nWrote {table.num_rows} rows to {out_path}")

    n_images = len(set(table.column("image_name").to_pylist()))
    n_tiles_per = args.scale ** 2
    print(f"Scale: {args.scale}x{args.scale} ({n_tiles_per} tiles/image)")
    print(f"Images: {n_images}")
    print(f"Total tiles: {table.num_rows}")
    print(f"Embedding dim: {embed_dim}")
    if args.jpeg_quality > 0:
        print(f"JPEG: quality={args.jpeg_quality}, "
              f"subsampling={args.jpeg_subsampling}")
    else:
        print("JPEG: disabled")


if __name__ == "__main__":
    main()
