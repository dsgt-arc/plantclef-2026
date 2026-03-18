"""Phase 5 — Student inference on the test dataset.

Loads Phase 1 test tile embeddings (already extracted), runs them through the
trained CrossRegionTransformer, applies geo species masking, and produces:
  1. Aggregated parquet compatible with phase2/generate_submission.py
  2. Submission CSV (same format as Phase 2 submissions)

Only the original-view embeddings are used (flip/rot90 were for training only).
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from models import CrossRegionTransformer
from phase2.aggregate_tiles import load_per_tile_parquet
from phase2.generate_submission import (
    format_species_list,
    load_class_mask_npy,
    load_species_map,
)
from utils.io_bytes import f16_vector_to_bytes

EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Tile preparation
# ---------------------------------------------------------------------------

def prepare_test_images(
    tile_data,
    id_col: str,
    aug_filter: bool,
    tile_filter_k: int,
) -> List[dict]:
    """Group raw per-tile data by image and pack original-view embeddings."""
    images = []

    for name, group in tile_data.groupby(id_col):
        df = group

        # Aug consistency filter
        if aug_filter and "passed_aug_consistency" in df.columns:
            df = df[df["passed_aug_consistency"] == True]  # noqa: E712

        if df.empty:
            continue

        # Sort by confidence
        if "max_prob" in df.columns:
            df = df.sort_values("max_prob", ascending=False)

        # Tile confidence filter
        if 0 < tile_filter_k < len(df):
            df = df.head(tile_filter_k)

        # Unpack original-view embeddings only
        emb_bufs = df["embedding_orig_f16"].tolist()
        n = len(emb_bufs)
        emb = np.empty((n, EMBEDDING_DIM), dtype=np.float32)
        for i, buf in enumerate(emb_bufs):
            emb[i] = np.frombuffer(buf, dtype=np.float16).astype(np.float32)

        images.append({
            "image_name": str(name),
            "tile_embeddings": emb,  # [N, 768]
            "num_tiles": n,
        })

    return images


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: CrossRegionTransformer,
    images: List[dict],
    device: torch.device,
    batch_size: int,
    geo_mask: Optional[np.ndarray],
) -> dict[str, np.ndarray]:
    """Run student model on all images, return {image_name: probs_f16[7806]}."""
    model.eval()
    results = {}

    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        B = len(batch)
        max_tiles = max(img["num_tiles"] for img in batch)

        # Pad and build attention mask
        tile_tensor = torch.zeros(B, max_tiles, EMBEDDING_DIM)
        mask_tensor = torch.zeros(B, max_tiles, dtype=torch.bool)

        for i, img in enumerate(batch):
            n = img["num_tiles"]
            tile_tensor[i, :n] = torch.from_numpy(img["tile_embeddings"])
            mask_tensor[i, :n] = True

        tile_tensor = tile_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        logits = model(tile_tensor, mask_tensor)  # [B, 7806]
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        # Geo mask
        if geo_mask is not None:
            probs[:, ~geo_mask] = 0.0

        for i, img in enumerate(batch):
            results[img["image_name"]] = probs[i].astype(np.float16)

    return results


# ---------------------------------------------------------------------------
# Output (matches Phase 2 format)
# ---------------------------------------------------------------------------

def save_student_parquet(
    image_probs: dict[str, np.ndarray],
    out_path: str,
    id_col: str = "image_name",
    probs_col: str = "student_image_probs_f16",
) -> None:
    """Write student probs parquet compatible with phase2/generate_submission.py."""
    names = []
    probs_bytes = []

    for name, probs in image_probs.items():
        names.append(name)
        probs_bytes.append(f16_vector_to_bytes(probs))

    table = pa.table({
        id_col: names,
        probs_col: probs_bytes,
    })

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"Wrote {table.num_rows} images to {out_path}")


def write_submission_csv(
    image_probs: dict[str, np.ndarray],
    out_csv: str,
    species_map: List[int],
    topk: int = 10,
    threshold: float = 0.03,
    min_k: int = 1,
) -> int:
    """Write Kaggle submission CSV directly from student probs."""
    import csv

    rows = []

    for name, probs_f16 in image_probs.items():
        qid = os.path.splitext(name)[0]
        p = probs_f16.astype(np.float32)

        if topk > 0:
            k = min(topk, p.shape[0])
            idx = np.argpartition(p, -k)[-k:]
            idx = idx[np.argsort(-p[idx])]
            kept = [int(i) for i in idx if p[i] >= threshold]
        else:
            kept = [int(i) for i in np.where(p >= threshold)[0]]

        if len(kept) < min_k:
            k2 = min(min_k, p.shape[0])
            idx2 = np.argpartition(p, -k2)[-k2:]
            idx2 = idx2[np.argsort(-p[idx2])]
            kept = [int(i) for i in idx2]

        kept_species = [species_map[i] for i in kept]
        rows.append((qid, format_species_list(kept_species)))

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, s in rows:
            w.writerow([qid, s])

    print(f"Wrote {len(rows)} rows to {out_csv}")
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5: Student inference on test set")

    # Inputs
    ap.add_argument(
        "--tile-parquet-root", required=True,
        help="Phase 1 test tile parquet root (e.g., phase1_tiles_test/)",
    )
    ap.add_argument("--checkpoint", required=True, help="Student model checkpoint (.pt)")

    # Outputs
    ap.add_argument("--out-parquet", required=True, help="Student probs parquet")
    ap.add_argument("--out-csv", required=True, help="Submission CSV")

    # Submission config (match Phase 2 best settings)
    ap.add_argument("--species-map", required=True, help="Class-to-species JSON/CSV")
    ap.add_argument("--geo-mask-npy", default=None, help="Geo species mask .npy")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--min-k", type=int, default=1)

    # Tile filtering (should match training config)
    ap.add_argument("--aug-filter", action="store_true", default=True)
    ap.add_argument("--no-aug-filter", action="store_false", dest="aug_filter")
    ap.add_argument("--tile-filter-k", type=int, default=16, help="Top-K tiles (16=all)")

    # Model (overrides checkpoint config if provided)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-classes", type=int, default=7806)

    # System
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--id-col", default="image_name")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    hidden_dim = args.hidden_dim or ckpt_config.get("hidden_dim", 512)
    num_heads = args.num_heads or ckpt_config.get("num_heads", 8)
    num_layers = args.num_layers or ckpt_config.get("num_layers", 4)
    num_classes = args.num_classes
    dropout = ckpt_config.get("dropout", 0.1)
    embed_dropout = ckpt_config.get("embed_dropout", 0.1)

    model = CrossRegionTransformer(
        input_dim=768,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        embed_dropout=embed_dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters (epoch {ckpt.get('epoch', '?')})")

    # --- Load geo mask ---
    geo_mask = None
    if args.geo_mask_npy:
        geo_mask = load_class_mask_npy(args.geo_mask_npy, num_classes)
        print(f"Geo mask: {geo_mask.sum()} / {len(geo_mask)} species allowed")

    # --- Load species map ---
    species_map = load_species_map(args.species_map, num_classes)

    # --- Load and prepare test tiles ---
    print(f"Loading test tiles from {args.tile_parquet_root} ...")
    tile_data = load_per_tile_parquet(args.tile_parquet_root)
    print(f"Loaded {len(tile_data)} tile rows")

    images = prepare_test_images(
        tile_data,
        id_col=args.id_col,
        aug_filter=args.aug_filter,
        tile_filter_k=args.tile_filter_k,
    )
    print(f"Prepared {len(images)} test images")

    del tile_data  # free memory

    # --- Run inference ---
    print("Running student inference ...")
    image_probs = run_inference(
        model=model,
        images=images,
        device=device,
        batch_size=args.batch_size,
        geo_mask=geo_mask,
    )

    # --- Save outputs ---
    # 1. Parquet (compatible with phase2/generate_submission.py)
    save_student_parquet(image_probs, args.out_parquet, id_col=args.id_col)

    # 2. Submission CSV (direct)
    n_rows = write_submission_csv(
        image_probs=image_probs,
        out_csv=args.out_csv,
        species_map=species_map,
        topk=args.topk,
        threshold=args.threshold,
        min_k=args.min_k,
    )

    print(f"\nDone. {n_rows} images submitted.")
    print(f"  Parquet: {args.out_parquet}")
    print(f"  CSV:     {args.out_csv}")
    print(f"\nTo regenerate CSV with different topk/threshold:")
    print(f"  python phase2/generate_submission.py \\")
    print(f"    --in-parquet {args.out_parquet} \\")
    print(f"    --out-csv <new_csv> \\")
    print(f"    --id-col {args.id_col} \\")
    print(f"    --probs-col student_image_probs_f16 \\")
    print(f"    --species-map {args.species_map} \\")
    print(f"    --topk <K> --threshold <T>")


if __name__ == "__main__":
    main()
