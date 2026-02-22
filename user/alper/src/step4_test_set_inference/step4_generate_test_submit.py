#!/usr/bin/env python3

import os
import csv
import json
import glob
import argparse
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import pyarrow as pa
import pyarrow.parquet as pq

from Cross_Region_Transformer_v3 import CrossRegionTransformer


def load_species_map(path: Optional[str], num_classes: int) -> List[int]:
    """
    Returns a list that maps class_index to species_id.

    Supported formats
      json list: [1395806, 1351284, ...]
      csv with columns: class_index,species_id
      txt one species_id per line

    If path is None, uses identity mapping 0..num_classes-1.
    """
    if path is None:
        return list(range(num_classes))

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            raise ValueError("JSON species map must be a list")
        if len(arr) != num_classes:
            raise ValueError(f"JSON species map length {len(arr)} does not match num_classes {num_classes}")
        return [int(x) for x in arr]

    if ext == ".csv":
        df = pd.read_csv(path)
        if "class_index" not in df.columns or "species_id" not in df.columns:
            raise ValueError("CSV species map must have columns class_index and species_id")
        df = df.sort_values("class_index")
        if len(df) != num_classes:
            raise ValueError(f"CSV species map rows {len(df)} does not match num_classes {num_classes}")
        return [int(x) for x in df["species_id"].tolist()]

    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) != num_classes:
        raise ValueError(f"TXT species map lines {len(lines)} does not match num_classes {num_classes}")
    return [int(x) for x in lines]


def format_species_list(species_ids: List[int]) -> str:
    """
    Required format examples
      "[1395806]"
      "[1351284, 1494911]"
    """
    return "[" + ", ".join(str(s) for s in species_ids) + "]"


def _bytes_to_f16_matrix(b: bytes, n_tiles: int, emb_dim: int) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.float16)
    expected = n_tiles * emb_dim
    if arr.size != expected:
        raise ValueError(f"Embedding buffer has {arr.size} values, expected {expected} for n_tiles={n_tiles}")
    return arr.reshape(n_tiles, emb_dim)


class TestEmbeddingsParquetFixedN(Dataset):
    """
    Reads sharded dataset:
      root/shard_XXXX/data.parquet

    Required columns
      quadrat_id (string)
      tile_embeddings_f16 (binary) float16 bytes of shape [n_tiles, emb_dim]
      metadata_tensor_f32 (FixedSizeList[4] float32) or list[4] float32
    """

    def __init__(
        self,
        data_root: str,
        pattern: str = "shard_*/data.parquet",
        n_tiles: int = 16,
        emb_dim: int = 768,
        meta_dim: int = 4,
        cache_tables: bool = True,
    ):
        self.data_root = data_root
        self.pattern = pattern
        self.n_tiles = n_tiles
        self.emb_dim = emb_dim
        self.meta_dim = meta_dim
        self.cache_tables = cache_tables

        self.files = sorted(glob.glob(os.path.join(data_root, pattern)))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_root} with pattern {pattern}")

        self._tables: List[pa.Table] = []
        self._row_map: List[Tuple[int, int]] = []

        for fi, fp in enumerate(self.files):
            table = pq.read_table(fp)
            self._tables.append(table)
            for ri in range(table.num_rows):
                self._row_map.append((fi, ri))

        if len(self._row_map) == 0:
            raise RuntimeError("No rows found in parquet shards")

    def __len__(self) -> int:
        return len(self._row_map)

    def __getitem__(self, idx: int):
        fi, ri = self._row_map[idx]
        row = self._tables[fi].slice(ri, 1)

        quadrat_id = row["quadrat_id"][0].as_py()

        emb_bytes = row["tile_embeddings_f16"][0].as_py()
        tiles = _bytes_to_f16_matrix(emb_bytes, self.n_tiles, self.emb_dim).astype(np.float32)

        meta_val = row["metadata_tensor_f32"][0].as_py()
        if isinstance(meta_val, list):
            meta = np.array(meta_val, dtype=np.float32)
        else:
            meta = np.frombuffer(meta_val, dtype=np.float32)

        if meta.shape[0] != self.meta_dim:
            raise ValueError(f"{quadrat_id} metadata length {meta.shape[0]} expected {self.meta_dim}")

        return torch.from_numpy(tiles), torch.from_numpy(meta), quadrat_id


def collate_fn(batch):
    x_tiles = torch.stack([b[0] for b in batch], dim=0)  # [B, N, D]
    x_meta = torch.stack([b[1] for b in batch], dim=0)   # [B, M]
    qids = [b[2] for b in batch]
    return x_tiles, x_meta, qids


def load_checkpoint_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Checkpoint format not recognized. Expected dict or dict with key 'model'.")


@torch.no_grad()
def infer_to_dataframe(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    species_map: List[int],
    topk: int,
    threshold: float,
    min_k: int,
    amp: bool,
) -> pd.DataFrame:
    model.eval()

    rows = []
    use_cuda_amp = amp and device.type == "cuda"

    for x_tiles, x_meta, qids in loader:
        x_tiles = x_tiles.to(device, non_blocking=True)
        x_meta = x_meta.to(device, non_blocking=True)

        if use_cuda_amp:
            with torch.cuda.amp.autocast():
                logits = model(x_tiles, x_meta, None)
        else:
            logits = model(x_tiles, x_meta, None)

        probs = torch.sigmoid(logits).float().cpu()

        for i, qid in enumerate(qids):
            p = probs[i]

            if topk > 0:
                k = min(topk, p.numel())
                top_vals, top_idx = torch.topk(p, k=k)
                keep = top_vals >= threshold
                kept_idx = top_idx[keep].tolist()
            else:
                kept_idx = torch.nonzero(p >= threshold, as_tuple=False).view(-1).tolist()

            if len(kept_idx) < min_k:
                k2 = min(min_k, p.numel())
                _, top_idx2 = torch.topk(p, k=k2)
                kept_idx = top_idx2.tolist()

            kept_species = [species_map[j] for j in kept_idx]
            rows.append(
                {
                    "quadrat_id": qid,
                    "species_ids": format_species_list(kept_species),
                }
            )

    return pd.DataFrame(rows, columns=["quadrat_id", "species_ids"])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test-root", required=True, help="Root folder with test parquet shards")
    ap.add_argument("--pattern", default="shard_*/data.parquet", help="Glob pattern under test-root")

    ap.add_argument("--ckpt", required=True, help="Path to trained Step3B checkpoint")
    ap.add_argument("--out-csv", required=True, help="Output submission CSV path")

    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--species-map", default=None, help="Optional class_index to species_id mapping file")

    ap.add_argument("--n-tiles", type=int, default=16, help="Tile count per sample, example 16 or 25")
    ap.add_argument("--emb-dim", type=int, default=768)
    ap.add_argument("--meta-dim", type=int, default=4)

    ap.add_argument("--batch-rows", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)

    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--min-k", type=int, default=1)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--input-dim", type=int, default=768)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use-attention-pooling", action="store_true", help="Enable attention pooling. If omitted, CLS pooling is used")

    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    ds = TestEmbeddingsParquetFixedN(
        data_root=args.test_root,
        pattern=args.pattern,
        n_tiles=args.n_tiles,
        emb_dim=args.emb_dim,
        meta_dim=args.meta_dim,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_rows,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )

    species_map = load_species_map(args.species_map, args.num_classes)

    model = CrossRegionTransformer(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        metadata_dim=args.meta_dim,
        dropout=args.dropout,
        use_attention_pooling=bool(args.use_attention_pooling),
    )

    state = load_checkpoint_state(args.ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("State dict warning")
        if missing:
            print("Missing keys (first 25)", missing[:25])
        if unexpected:
            print("Unexpected keys (first 25)", unexpected[:25])

    model.to(device)

    df = infer_to_dataframe(
        model=model,
        loader=loader,
        device=device,
        species_map=species_map,
        topk=args.topk,
        threshold=args.threshold,
        min_k=args.min_k,
        amp=args.amp,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.out_csv, sep=",", index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


'''

python infer_step4_test_submit.py \
  --test-root /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/test_student_ready \
  --ckpt /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/checkpoints/lucas_212k_SSL/step3b/student_best.pt \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/submissions/run_step3b_best.csv \
  --num-classes 7806 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/mappings/class_to_species_id.json \
  --n-tiles 16 \
  --batch-rows 512 \
  --topk 64 \
  --threshold 0.0 \
  --min-k 1 \
  --amp

'''