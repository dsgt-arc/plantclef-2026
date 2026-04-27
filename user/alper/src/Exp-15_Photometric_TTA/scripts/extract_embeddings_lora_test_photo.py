"""Phase-2 wrapper: full ViT test re-extraction with photometric preprocessing.

Clones Exp-9B/extract_embeddings_lora_test.py with two extensions:

1. `--photometric` flag selects one of {none, unsharp, clahe_lab, saturation}
   from preprocess.photometric. Variant transform is applied AFTER tile_image
   and BEFORE jpeg_compress, matching the insertion point used in the Phase-1
   probe.

2. The forward pass is extended to also produce 7806-d teacher_logits per tile
   (via backbone.forward_head, since the LoRA-injected backbone retains the
   classifier head). Both `teacher_logits_f16` and `emb_arcface_f16` columns
   in the output parquet are produced under the photometric variant. `max_prob`
   is recomputed from the new teacher_logits at temperature T=1.5 to keep
   downstream tile filters consistent. Other columns (image_name, tile_index,
   embedding_orig_f16, passed_aug_consistency) are carried over from the
   existing baseline parquet via the same merge pattern as Exp-9B.

This is the script the Phase-2 SLURM job invokes (one job per surviving
variant from Phase 1).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import timm  # noqa: F401
import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXP15_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP15_ROOT))
from preprocess import photometric  # noqa: E402

EXP9B_ROOT = EXP15_ROOT.parent / "Exp-9B_Improved_KNN_Retrieval"
sys.path.insert(0, str(EXP9B_ROOT))
from lora import inject_lora_into_attention, load_lora_state  # noqa: E402
from train_arcface_lora import (  # noqa: E402
    ArcFaceHead, IMAGENET_MEAN, IMAGENET_STD,
    MODEL_NAME, NUM_CLASSES, IN_DIM,
)

DINOV2_SIZE = 518


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
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    print(f"[model] checkpoint epoch={ckpt.get('epoch')} step={ckpt.get('step')}")

    backbone = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES,
        checkpoint_path=backbone_path,
    ).to(device)
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

    head = ArcFaceHead(in_dim=IN_DIM, out_dim=int(cfg.get("out_dim", 768))).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()

    return backbone, head, int(cfg.get("jpeg_quality", 85))


@torch.no_grad()
def forward_pass(
    backbone, head, tiles: List[Image.Image], transform_norm,
    device: torch.device, batch_size: int = 32,
):
    """Returns (logits_f32 [N, 7806], emb_f32 [N, 768], max_prob_f32 [N])."""
    all_logits, all_emb, all_max = [], [], []
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start: start + batch_size]
        x = torch.stack([transform_norm(t) for t in batch]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = backbone.forward_features(x)
            cls = feats[:, 0, :]
            logits = backbone.forward_head(feats, pre_logits=False)  # [B, 7806]
            emb = head(cls)  # [B, 768]
            # max_prob recomputed at T=1.5 (production temperature)
            T = 1.5
            probs = torch.softmax(logits / T, dim=-1)
            mp = probs.max(dim=-1).values
        all_logits.append(logits.float().cpu().numpy())
        all_emb.append(emb.float().cpu().numpy())
        all_max.append(mp.float().cpu().numpy())
    return (np.concatenate(all_logits), np.concatenate(all_emb), np.concatenate(all_max))


def process_scale(
    scale: int, image_dir: str, existing_parquet_dir: str,
    backbone, head, device, jpeg_quality: int, batch_size: int,
    photo_variant: str, out_dir: str,
) -> None:
    print(f"\n=== Scale {scale}x{scale}  (photo={photo_variant}) ===")
    existing_path = os.path.join(existing_parquet_dir, f"scale_{scale}x{scale}.parquet")
    if not os.path.exists(existing_path):
        raise FileNotFoundError(f"Existing parquet not found: {existing_path}")
    existing_df = pd.read_parquet(existing_path)
    print(f"[{scale}] existing rows: {len(existing_df)}")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"[{scale}] processing {len(image_files)} test images")

    transform_norm = build_normalize()

    all_names, all_tile_idx = [], []
    all_logits, all_emb, all_max = [], [], []

    t0 = time.time()
    for i, fname in enumerate(image_files):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        img = center_crop_square(img)
        tiles = tile_image(img, scale)
        tiles = [photometric.transform(t, photo_variant) for t in tiles]
        if jpeg_quality > 0:
            tiles = [jpeg_compress(t, jpeg_quality) for t in tiles]
        logits, emb, mp = forward_pass(
            backbone, head, tiles, transform_norm, device, batch_size,
        )
        for tidx in range(len(tiles)):
            all_names.append(fname)
            all_tile_idx.append(tidx)
            all_logits.append(logits[tidx])
            all_emb.append(emb[tidx])
            all_max.append(float(mp[tidx]))
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(image_files) - i - 1) / max(rate, 1e-6)
            print(f"[{scale}]   {i + 1}/{len(image_files)}  {rate:.1f} img/s  ETA {eta:.0f}s")

    print(f"[{scale}] extracted {len(all_logits)} tile logits/embeddings")
    new_df = pd.DataFrame({
        "image_name": all_names,
        "tile_index": all_tile_idx,
        "teacher_logits_f16": [l.astype(np.float16).tobytes() for l in all_logits],
        "emb_arcface_f16":     [e.astype(np.float16).tobytes() for e in all_emb],
        "max_prob": all_max,
    })

    # Merge: preserve image_name, tile_index, embedding_orig_f16,
    # passed_aug_consistency, and any other Exp-7 columns. Overwrite
    # teacher_logits_f16, emb_arcface_f16, max_prob with the new
    # photometric-aware values.
    keep = [c for c in existing_df.columns
            if c not in ("teacher_logits_f16", "emb_arcface_f16", "max_prob")]
    base = existing_df[keep]
    merged = base.merge(new_df, on=["image_name", "tile_index"], how="left")
    missing = merged["teacher_logits_f16"].isna().sum()
    if missing > 0:
        print(f"[{scale}] WARNING: {missing} rows missing new logits — join key issue")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"scale_{scale}x{scale}.parquet")
    pq.write_table(pa.Table.from_pandas(merged, preserve_index=False),
                   out_path, compression="snappy")
    print(f"[{scale}] wrote {out_path}  rows={len(merged)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--backbone-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--existing-parquet-dir", required=True,
                    help="Baseline (non-photometric) Exp-9B parquet dir to merge over.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scales", default="4,5,6")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--jpeg-quality", type=int, default=None)
    ap.add_argument("--photometric", required=True,
                    choices=["none", "unsharp", "clahe_lab", "saturation"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")
    print(f"[init] photometric variant: {args.photometric}")

    backbone, head, jpeg_q_ckpt = load_model(
        args.backbone_path, args.checkpoint, device
    )
    jpeg_q = args.jpeg_quality if args.jpeg_quality is not None else jpeg_q_ckpt
    print(f"[init] JPEG quality: {jpeg_q}")

    scales = [int(s) for s in args.scales.split(",")]
    for scale in scales:
        process_scale(
            scale, args.image_dir, args.existing_parquet_dir,
            backbone, head, device, jpeg_q, args.batch_size,
            args.photometric, args.out_dir,
        )


if __name__ == "__main__":
    main()
