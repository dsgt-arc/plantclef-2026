"""Run a trained TileQ-Decoder over the cached test CLSs and emit Kaggle CSVs.

One forward pass per quadrat -> 7806 sigmoid probabilities -> top-k decoding
at one or more k values. Also persists the per-image probability vector as
a numpy array so ``blend.py`` can α-mix with the cached ``max_prob`` baseline
without re-running inference.

Outputs (per checkpoint, all from one forward pass):
  submissions/<tag>_topk{k}_seed{seed}.csv  for each k in --topk
  artifacts/probs_<tag>_seed{seed}.npy      (2105, 7806) float32 — for blend.py

Usage:
  python infer.py \
      --checkpoint artifacts/checkpoints/decoder_samplerA.pt \
      --tag A1_samplerA \
      --topk 9,10
  python infer.py \
      --checkpoint artifacts/checkpoints/decoder_samplerB.pt \
      --tag A2_samplerB \
      --topk 5,7,9,10,12,15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from plantclef.tileq_decoder import TileQDecoder


def load_test_cls(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read cls_grid_4x4.parquet -> (image_names (n_q,), test_cls (n_q, 16, 768))."""
    table = pq.read_table(parquet_path)
    n_rows = table.num_rows
    names = np.asarray(table.column("image_name").to_pylist(), dtype="<U64")
    tile_idxs = table.column("tile_idx").to_numpy()
    blobs = table.column("cls_embedding").to_pylist()
    blob_size = len(blobs[0])
    if blob_size % 2 != 0:
        raise ValueError(f"blob size {blob_size} not divisible by 2 (float16)")
    dim = blob_size // 2
    flat = np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(n_rows, dim)
    flat = flat.astype(np.float32)
    order = np.lexsort((tile_idxs, names))
    names = names[order]
    flat = flat[order]
    image_names = np.unique(names)
    n_q = len(image_names)
    n_tiles = n_rows // n_q
    if n_q * n_tiles != n_rows:
        raise ValueError(f"{n_q} * {n_tiles} != {n_rows}")
    test_cls = flat.reshape(n_q, n_tiles, dim)
    return image_names, test_cls


def load_class_to_species_id(class_mapping_path: Path) -> np.ndarray:
    species_ids = [
        int(line.strip())
        for line in class_mapping_path.read_text().splitlines()
        if line.strip()
    ]
    return np.asarray(species_ids, dtype=np.int64)


def write_submission_csv(
    out_path: Path,
    image_names: np.ndarray,
    species_id_lists: list[list[int]],
) -> None:
    """Submission format: comma-separated, ALL fields quoted, species_ids
    encoded as Python list-repr. Matches exp 002's
    ``pandas.to_csv(sep=",", quoting=csv.QUOTE_ALL)`` writer byte-for-byte —
    Kaggle parses with that schema (verified by A0a/A0b acceptance)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_ALL)
        writer.writerow(["quadrat_id", "species_ids"])
        for name, sid_list in zip(image_names, species_id_lists):
            writer.writerow([name, str(list(sid_list))])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True,
                   help="CSV name prefix, e.g. 'A1_samplerA'")
    p.add_argument("--topk", type=str, default="9,10",
                   help="Comma-separated k values, e.g. '5,7,9,10,12,15'")
    p.add_argument("--seed", type=int, default=42,
                   help="Suffix tag, also seeds torch (for reproducibility on GPU).")
    p.add_argument(
        "--test-parquet",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/inference/"
            "dinov2-vit-base/cls_grid_4x4.parquet"
        ),
    )
    p.add_argument(
        "--class-mapping",
        type=Path,
        default=Path(
            "/storage/project/ps-clef2026_plantclef-0/shared/data/raw/"
            "class_id_to_species_mapping/class_mapping.txt"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "submissions",
    )
    p.add_argument(
        "--probs-dir",
        type=Path,
        default=Path(__file__).parent / "artifacts",
        help="Where to persist per-image probability vector (for blend.py).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-save-probs", action="store_true")
    args = p.parse_args()

    topk_values = [int(k) for k in args.topk.split(",") if k.strip()]
    if not topk_values:
        print("FATAL: --topk produced no values", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading class mapping: {args.class_mapping}")
    class_to_species_id = load_class_to_species_id(args.class_mapping)
    n_classes = len(class_to_species_id)
    print(f"  n_classes={n_classes}")

    print(f"Loading test CLS parquet: {args.test_parquet}")
    t0 = time.time()
    image_names, test_cls = load_test_cls(args.test_parquet)
    print(f"  image_names={len(image_names)} test_cls={test_cls.shape} "
          f"({time.time()-t0:.1f}s)")
    if test_cls.shape[2] != 768:
        print(f"FATAL: expected dim 768, got {test_cls.shape[2]}", file=sys.stderr)
        return 1

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    print(f"  trained sampler={cfg.get('sampler')} steps={cfg.get('steps')} "
          f"seed={cfg.get('seed')}")

    model = TileQDecoder(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_q, n_tiles, dim = test_cls.shape

    print(f"Running forward pass over {n_q} quadrats (batch={args.batch_size})...")
    t0 = time.time()
    probs = np.empty((n_q, n_classes), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_q, args.batch_size):
            j = min(i + args.batch_size, n_q)
            x = torch.from_numpy(test_cls[i:j]).to(device, non_blocking=True)
            logits = model(x)
            p = torch.sigmoid(logits).float().cpu().numpy()
            probs[i:j] = p
    print(f"  done ({time.time()-t0:.1f}s, "
          f"probs.shape={probs.shape} dtype={probs.dtype})")

    # Save per-image probability vector for blend.py
    if not args.no_save_probs:
        args.probs_dir.mkdir(parents=True, exist_ok=True)
        probs_path = args.probs_dir / f"probs_{args.tag}_seed{args.seed}.npy"
        np.save(probs_path, probs)
        print(f"  saved probs: {probs_path} ({probs_path.stat().st_size / 1e6:.1f} MB)")

    # Top-k decoding for each k value (one CSV each)
    print()
    print(f"Writing submission CSVs for top-k in {topk_values}...")
    for k in topk_values:
        if k > n_classes:
            print(f"  skip k={k} > n_classes={n_classes}")
            continue
        topk_idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
        # Sort the top-k by descending probability (for stable, deterministic output)
        for i in range(n_q):
            order = np.argsort(-probs[i, topk_idx[i]])
            topk_idx[i] = topk_idx[i][order]

        species_id_lists = [
            class_to_species_id[topk_idx[i]].tolist() for i in range(n_q)
        ]
        out_path = args.output_dir / f"{args.tag}_topk{k}_seed{args.seed}.csv"
        write_submission_csv(out_path, image_names, species_id_lists)

        avg_per_img = sum(len(s) for s in species_id_lists) / len(species_id_lists)
        print(f"  wrote {out_path}  "
              f"(avg species/img = {avg_per_img:.1f})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
