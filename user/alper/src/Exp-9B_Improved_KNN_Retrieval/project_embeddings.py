"""Apply a trained ArcFace projection head to cached DINOv2 CLS embeddings.

Two modes:

  --mode train
    Input:  parquets with columns [image_name, species_id, cls_embedding]
            under <in-dir>/sample_id=*/*.parquet
    Output: parallel parquets under <out-dir>/sample_id=*/ with columns
            [image_name, species_id, emb512] (float32, L2-normalized, 512d)

  --mode test
    Input:  test-tile parquets carrying 'embedding_orig_f16' (bytes) and the
            standard Exp-7 tile metadata columns
    Output: parallel parquets under <out-dir>/ with an added 'emb_arcface_f16'
            column holding the projected 512-d embedding (float16 bytes),
            plus all original columns preserved.

Both modes load the head checkpoint produced by train_arcface_head.py.

Usage (train embeddings):
  python project_embeddings.py \
    --mode train \
    --checkpoint /path/to/arcface_head_final.pt \
    --in-dir     /storage/.../embeddings/train \
    --out-dir    /storage/.../exp9b_knn/embeddings_arcface/train

Usage (test tile embeddings):
  python project_embeddings.py \
    --mode test \
    --checkpoint /path/to/arcface_head_final.pt \
    --in-dir     /storage/.../multiscale_tiles_test_parquets \
    --out-dir    /storage/.../exp9b_knn/embeddings_arcface/test
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from train_arcface_head import ArcFaceHead, IN_DIM


def load_head(checkpoint_path: str, device: torch.device) -> ArcFaceHead:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    out_dim = int(cfg.get("out_dim", 512))
    head = ArcFaceHead(in_dim=IN_DIM, out_dim=out_dim).to(device)
    head.load_state_dict(ckpt["head"])
    head.eval()
    print(
        f"[head] loaded from {checkpoint_path} "
        f"(epoch={ckpt.get('epoch')} step={ckpt.get('step')} out_dim={out_dim})"
    )
    return head


@torch.no_grad()
def project_matrix(
    head: ArcFaceHead,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 32768,
) -> np.ndarray:
    """Forward rows of x [N, 768] through head. Return [N, out_dim] float32."""
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    out = np.empty((x.shape[0], head.out_dim), dtype=np.float32)
    for i in range(0, x.shape[0], batch_size):
        chunk = torch.from_numpy(x[i : i + batch_size]).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            z = head(chunk)
        out[i : i + batch_size] = z.float().cpu().numpy()
    return out


def process_train(
    head: ArcFaceHead,
    in_dir: str,
    out_dir: str,
    device: torch.device,
    batch_size: int,
) -> None:
    pattern = os.path.join(in_dir, "sample_id=*", "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files under {pattern}")
    print(f"[train] {len(files)} shards")

    for i, fp in enumerate(files):
        rel = os.path.relpath(fp, in_dir)
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        t0 = time.time()
        df = pd.read_parquet(fp, columns=["image_name", "species_id", "cls_embedding"])
        emb = np.stack(
            [np.asarray(e, dtype=np.float32) for e in df["cls_embedding"].to_numpy()],
            axis=0,
        )
        z = project_matrix(head, emb, device, batch_size=batch_size)

        out_df = pd.DataFrame(
            {
                "image_name": df["image_name"].astype(str),
                "species_id": df["species_id"].astype("int32"),
                "emb512": list(z),
            }
        )
        out_df.to_parquet(out_path, index=False, compression="snappy")
        print(
            f"[train] ({i + 1}/{len(files)}) {rel} n={len(df)} "
            f"dim={z.shape[1]} t={time.time() - t0:.1f}s"
        )


def _f16_bytes_to_f32_matrix(buffers: list[bytes], dim: int) -> np.ndarray:
    arr = np.empty((len(buffers), dim), dtype=np.float32)
    for i, b in enumerate(buffers):
        arr[i] = np.frombuffer(b, dtype=np.float16).astype(np.float32)
    return arr


def _f32_matrix_to_f16_bytes(x: np.ndarray) -> list[bytes]:
    x16 = x.astype(np.float16)
    return [row.tobytes() for row in x16]


def process_test(
    head: ArcFaceHead,
    in_dir: str,
    out_dir: str,
    device: torch.device,
    batch_size: int,
    emb_col_in: str,
    emb_col_out: str,
) -> None:
    pattern = os.path.join(in_dir, "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {pattern}")
    print(f"[test] {len(files)} parquet files matched")

    for i, fp in enumerate(files):
        rel = os.path.relpath(fp, in_dir)
        out_path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        t0 = time.time()
        df = pd.read_parquet(fp)
        if emb_col_in not in df.columns:
            raise KeyError(f"{fp}: column '{emb_col_in}' not present")

        buffers = df[emb_col_in].tolist()
        emb_f32 = _f16_bytes_to_f32_matrix(buffers, IN_DIM)
        z = project_matrix(head, emb_f32, device, batch_size=batch_size)

        df[emb_col_out] = _f32_matrix_to_f16_bytes(z)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")
        print(
            f"[test] ({i + 1}/{len(files)}) {rel} n={len(df)} "
            f"t={time.time() - t0:.1f}s"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-project cached CLS embeddings through trained ArcFace head"
    )
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32768)
    ap.add_argument(
        "--emb-col-in",
        default="embedding_orig_f16",
        help="Test mode only: name of input f16-bytes column carrying 768-d CLS",
    )
    ap.add_argument(
        "--emb-col-out",
        default="emb_arcface_f16",
        help="Test mode only: name of new f16-bytes column holding projected emb",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}")

    head = load_head(args.checkpoint, device)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "train":
        process_train(head, args.in_dir, args.out_dir, device, args.batch_size)
    else:
        process_test(
            head,
            args.in_dir,
            args.out_dir,
            device,
            args.batch_size,
            args.emb_col_in,
            args.emb_col_out,
        )


if __name__ == "__main__":
    main()
