#!/usr/bin/env python3

import os
import glob
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from step2.aggregator import TemperatureNoisyORAggregator
from step2.io_bytes import f16_vector_to_bytes


def f16_bytes_to_f32_matrix(buf: bytes, n_tiles: int, num_classes: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.float16)
    expected = n_tiles * num_classes
    if arr.size != expected:
        raise ValueError(f"Expected {expected} f16 values, got {arr.size}")
    return arr.reshape(n_tiles, num_classes).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--in-parquet-root", required=True, help="Folder containing shard_0000 etc")
    p.add_argument("--pattern", default="shard_*/*.parquet", help="Glob under in-parquet-root for part files")
    p.add_argument("--out-parquet", required=True, help="Output parquet path or folder")

    p.add_argument("--id-col", default="quadrat_id")
    p.add_argument("--logits-col", default="teacher_logits_f16")

    p.add_argument("--num-classes", type=int, default=7806)
    p.add_argument("--n-tiles", type=int, default=16)

    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--denoise-eps", type=float, default=1e-6)

    p.add_argument("--image-topk", type=int, default=64, help="0 keeps dense, >0 keeps sparse topk at image level")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(glob.glob(os.path.join(args.in_parquet_root, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.in_parquet_root} matching {args.pattern}")

    image_topk = int(args.image_topk) if int(args.image_topk) > 0 else None

    agg = TemperatureNoisyORAggregator(
        num_classes=int(args.num_classes),
        temperature=float(args.temperature),
        denoise_eps=float(args.denoise_eps),
        output_dtype=np.float16,
        image_topk=image_topk,
        return_stats=False,
    )

    out_rows: List[Tuple[str, bytes, int]] = []

    for fp in files:
        table = pq.read_table(fp, columns=[args.id_col, args.logits_col])
        df = table.to_pandas()

        for _, r in df.iterrows():
            qid = str(r[args.id_col])
            buf = r[args.logits_col]
            if buf is None:
                continue

            tile_logits = f16_bytes_to_f32_matrix(buf, n_tiles=int(args.n_tiles), num_classes=int(args.num_classes))
            p_image_f16 = agg.aggregate(tile_logits)

            out_rows.append((qid, f16_vector_to_bytes(p_image_f16), int(args.n_tiles)))

    out_table = pa.Table.from_pydict(
        {
            args.id_col: [r[0] for r in out_rows],
            "teacher_image_probs_f16": [r[1] for r in out_rows],
            "num_tiles": [r[2] for r in out_rows],
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)), exist_ok=True)
    pq.write_table(out_table, args.out_parquet, compression="snappy")
    print(f"Wrote {out_table.num_rows} rows to {args.out_parquet}")


if __name__ == "__main__":
    main()


'''
python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step2_B_teacher_testing/run_step2_teacher_aggregation_on_test_set.py \
  --in-parquet-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/feature_bank \
  --pattern "shard_*/*.parquet" \
  --out-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_step2_agg/teacher_image_probs.parquet \
  --logits-col teacher_logits_f16 \
  --num-classes 7806 \
  --n-tiles 16 \
  --temperature 2.0 \
  --denoise-eps 1e-6 \
  --image-topk 64


'''