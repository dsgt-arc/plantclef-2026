"""Phase-1 probe — run the LoRA-injected ViT-B + ArcFace head on a small
40-quadrat sample, applying each of {none, unsharp, clahe_lab, saturation}
photometric variants. Save per (quadrat, scale, tile_idx, variant):
    - teacher_logits_f16 (7806-d, from backbone classifier head)
    - emb_arcface_f16    (768-d, from ArcFace head)

Output: outputs/phase1/probe_logits.parquet

Mirrors `Exp-9B_Improved_KNN_Retrieval/extract_embeddings_lora_test.py`
(load_model, tile_image, jpeg_compress) but operates on a small sample
and applies all 4 photometric variants per tile in one go.
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

# Reuse Exp-9B model loading & helpers
EXP9B_ROOT = EXP15_ROOT.parent / "Exp-9B_Improved_KNN_Retrieval"
sys.path.insert(0, str(EXP9B_ROOT))
from lora import inject_lora_into_attention, load_lora_state  # noqa: E402
from train_arcface_lora import (  # noqa: E402
    ArcFaceHead,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_NAME,
    NUM_CLASSES,
    IN_DIM,
)

DINOV2_SIZE = 518
PHOTO_VARIANTS = ["none", "unsharp", "clahe_lab", "saturation"]


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
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
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
    device: torch.device, batch_size: int = 16,
):
    """Returns (logits [N, 7806] f32, arcface_emb [N, 768] f32)."""
    all_logits = []
    all_emb = []
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start: start + batch_size]
        x = torch.stack([transform_norm(t) for t in batch]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            feats = backbone.forward_features(x)
            cls = feats[:, 0, :]
            logits = backbone.forward_head(feats, pre_logits=False)
            arcface = head(cls)
        all_logits.append(logits.float().cpu().numpy())
        all_emb.append(arcface.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_emb, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-csv", required=True,
                    help="outputs/probe/probe_sample.csv from build_probe_sample.py")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--backbone-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True,
                    help="Output parquet path (e.g. outputs/phase1/probe_logits.parquet)")
    ap.add_argument("--scales", default="4,5,6")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--jpeg-quality", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    backbone, head, jpeg_q_ckpt = load_model(args.backbone_path, args.checkpoint, device)
    jpeg_q = args.jpeg_quality if args.jpeg_quality is not None else jpeg_q_ckpt
    print(f"[init] JPEG quality: {jpeg_q}")

    probe = pd.read_csv(args.probe_csv)
    print(f"[init] probing {len(probe)} quadrats × {len(PHOTO_VARIANTS)} variants × scales={args.scales}")

    transform_norm = build_normalize()
    scales = [int(s) for s in args.scales.split(",")]

    rows = []
    t0 = time.time()
    for i, qid in enumerate(probe["quadrat_id"]):
        path = os.path.join(args.image_dir, f"{qid}.jpg")
        if not os.path.exists(path):
            print(f"  ! missing {path}")
            continue
        img = Image.open(path).convert("RGB")
        img_sq = center_crop_square(img)

        for scale in scales:
            tiles = tile_image(img_sq, scale)
            for variant in PHOTO_VARIANTS:
                processed = [photometric.transform(t, variant) for t in tiles]
                if jpeg_q > 0:
                    processed = [jpeg_compress(t, jpeg_q) for t in processed]
                logits, emb = forward_pass(
                    backbone, head, processed, transform_norm,
                    device, args.batch_size,
                )
                for tile_idx, (lg, em) in enumerate(zip(logits, emb)):
                    rows.append({
                        "quadrat_id": qid,
                        "scale": scale,
                        "tile_index": tile_idx,
                        "variant": variant,
                        "teacher_logits_f16": lg.astype(np.float16).tobytes(),
                        "emb_arcface_f16": em.astype(np.float16).tobytes(),
                        "max_logit": float(lg.max()),
                    })
        if (i + 1) % 5 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(probe) - i - 1) / max(rate, 1e-6)
            print(f"  [{i + 1}/{len(probe)}]  {rate:.2f} qid/s  ETA {eta:.0f}s")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   args.out, compression="snappy")
    print(f"  wrote {len(df)} rows -> {args.out}")
    print(f"  total wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
