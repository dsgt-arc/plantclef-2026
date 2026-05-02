"""Extract 1024-d ViT-L LoRA+ArcFace embeddings for test tiles, merge with the
existing phase2v3 multi-scale parquets so teacher_logits_f16 (classifier branch)
is preserved alongside the new emb_arcface_f16 (kNN branch).

For each test image, per scale s:
  1. Center-crop to square
  2. Resize to (s * 518) x (s * 518), Lanczos
  3. Tile into s^2 non-overlapping 518 x 518 crops
  4. JPEG round-trip per tile (matches train distribution)
  5. Forward through phase2v3 ViT-L + LoRA + ArcFace head -> 1024-d L2-normalized

Output columns mirror the Exp-9B Tier-2 test parquets:
  [image_name, tile_index, embedding_orig_f16, teacher_logits_f16,
   max_prob, aug_consistency_overlap, passed_aug_consistency, emb_arcface_f16]

(The emb_arcface_f16 column will be 1024-d here vs Exp-9B's 768-d. Downstream
ensemble scripts in Exp-11C are already patched to handle 1024-d via VIT_EMB_DIM.)

Usage:
  python extract_embeddings_lora_test_vitL.py \
    --image-dir /storage/.../data/raw/PlanCLEF_test_data/images \
    --backbone-path /storage/.../checkpoints/exp13_vit/vit_dinov2_large_phase2v3/model_best.pth.tar \
    --checkpoint /storage/.../aerten3/exp11c_vitL_lora_arcface/lora_head_final.pt \
    --existing-parquet-dir /storage/.../parquet/exp13_vit_dinov2_large_phase2v3 \
    --out-dir /storage/.../aerten3/exp11c_vitL_lora_arcface/embeddings/test \
    --scales 4,5,6,7
"""
from __future__ import annotations

import argparse
import io
import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lora import inject_lora_into_attention, load_lora_state
from train_arcface_lora_vitL import (
    ArcFaceHead,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IN_DIM,
    DEFAULT_OUT_DIM,
    load_phase2v3_backbone,
)

DINOV2_SIZE = 518


@dataclass
class _MinimalCfg:
    image_size: int = 518
    grad_checkpoint: bool = False


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def tile_image(img: Image.Image, scale: int) -> List[Image.Image]:
    target = scale * DINOV2_SIZE
    img = img.resize((target, target), Image.LANCZOS)
    tiles = []
    for r in range(scale):
        for c in range(scale):
            tiles.append(
                img.crop((c * DINOV2_SIZE, r * DINOV2_SIZE,
                          (c + 1) * DINOV2_SIZE, (r + 1) * DINOV2_SIZE))
            )
    return tiles


def jpeg_compress(img: Image.Image, quality: int = 85, subsampling: int = 1) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).copy()


def build_normalize() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(backbone_path: str, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"[model] checkpoint epoch={ckpt.get('epoch')} step={ckpt.get('step')}")

    backbone = load_phase2v3_backbone(backbone_path, _MinimalCfg(), device)
    inject_lora_into_attention(
        backbone,
        r=int(cfg.get("lora_r", 8)),
        alpha=float(cfg.get("lora_alpha", 16.0)),
        dropout=0.0,
        include_proj=bool(cfg.get("lora_on_proj", False)),
    )
    backbone = backbone.to(device)
    load_lora_state(backbone, ckpt["lora"])
    backbone.eval()

    out_dim = int(cfg.get("out_dim", DEFAULT_OUT_DIM))
    head = ArcFaceHead(in_dim=IN_DIM, out_dim=out_dim).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()

    jpeg_q = int(cfg.get("jpeg_quality", 85))
    return backbone, head, jpeg_q, out_dim


@torch.no_grad()
def embed_tiles(backbone, head, tiles, transform, device, batch_size=16):
    out = []
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start : start + batch_size]
        x = torch.stack([transform(t) for t in batch]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = backbone.forward_features(x)
            cls = feats[:, 0, :]
            z = head(cls)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def process_scale(
    scale, image_dir, existing_parquet_dir, backbone, head, device,
    jpeg_quality, batch_size, out_dir,
):
    print(f"\n=== Scale {scale}x{scale} ===")
    existing_path = os.path.join(existing_parquet_dir, f"scale_{scale}x{scale}.parquet")
    if not os.path.exists(existing_path):
        raise FileNotFoundError(
            f"Existing parquet not found: {existing_path}\n"
            f"Run extract_multiscale_3to7.sbatch with EXP_SUFFIX=phase2v3 first."
        )

    print(f"[{scale}] loading existing parquet: {existing_path}")
    existing_df = pd.read_parquet(existing_path)
    print(f"[{scale}] existing rows: {len(existing_df)}")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"[{scale}] processing {len(image_files)} test images")

    transform = build_normalize()
    all_names: list[str] = []
    all_tile_idx: list[int] = []
    all_emb: list[np.ndarray] = []

    t0 = time.time()
    for i, fname in enumerate(image_files):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        img = center_crop_square(img)
        tiles = tile_image(img, scale)
        if jpeg_quality > 0:
            tiles = [jpeg_compress(t, jpeg_quality) for t in tiles]
        embs = embed_tiles(backbone, head, tiles, transform, device, batch_size)
        for tidx, e in enumerate(embs):
            all_names.append(fname)
            all_tile_idx.append(tidx)
            all_emb.append(e)
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(image_files) - i - 1) / rate
            print(f"[{scale}]   {i + 1}/{len(image_files)}  {rate:.1f} img/s  ETA {eta:.0f}s")

    new_embs = np.stack(all_emb, axis=0)
    print(f"[{scale}] extracted {len(new_embs)} tile embeddings  dim={new_embs.shape[1]}")

    new_df = pd.DataFrame({
        "image_name": all_names,
        "tile_index": all_tile_idx,
        "emb_arcface_f16": [e.astype(np.float16).tobytes() for e in new_embs],
    })

    merged = existing_df.merge(
        new_df, on=["image_name", "tile_index"], how="left",
        suffixes=("", "_new"),
    )
    if "emb_arcface_f16_new" in merged.columns:
        merged["emb_arcface_f16"] = merged["emb_arcface_f16_new"]
        merged = merged.drop(columns=["emb_arcface_f16_new"])

    missing = merged["emb_arcface_f16"].isna().sum()
    if missing > 0:
        print(f"[{scale}] WARNING: {missing} rows have no new embedding — check join keys")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"scale_{scale}x{scale}.parquet")
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                   out_path, compression="snappy")
    print(f"[{scale}] wrote {out_path}  rows={len(merged)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract ViT-L LoRA+ArcFace test tile embeddings "
                    "and merge with existing phase2v3 parquets"
    )
    ap.add_argument("--image-dir", required=True,
                    help="Raw test images directory.")
    ap.add_argument("--backbone-path", required=True,
                    help="Phase2v3 ViT-L .pth.tar.")
    ap.add_argument("--checkpoint", required=True,
                    help="lora_head_final.pt from train_arcface_lora_vitL.py.")
    ap.add_argument("--existing-parquet-dir", required=True,
                    help="Directory with phase2v3 scale_NxN.parquet files "
                         "(preserves teacher_logits_f16 etc).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scales", default="4,5,6,7",
                    help="Comma-separated scales (default: K scheme 4,5,6,7)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="Override JPEG quality (default: read from checkpoint).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    backbone, head, jpeg_q_ckpt, out_dim = load_model(args.backbone_path, args.checkpoint, device)
    jpeg_q = args.jpeg_quality if args.jpeg_quality is not None else jpeg_q_ckpt
    print(f"[init] JPEG quality: {jpeg_q}  out_dim: {out_dim}")

    scales = [int(s) for s in args.scales.split(",")]
    for scale in scales:
        process_scale(
            scale=scale,
            image_dir=args.image_dir,
            existing_parquet_dir=args.existing_parquet_dir,
            backbone=backbone,
            head=head,
            device=device,
            jpeg_quality=jpeg_q,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
        )

    print("\n[done] all scales processed")


if __name__ == "__main__":
    main()
