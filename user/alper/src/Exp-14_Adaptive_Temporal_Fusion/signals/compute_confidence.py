"""Photo-confidence stats from a per-branch image_probs_f16 parquet.

Reads image_name + image_probs_f16 (Exp-8 schema) and writes per-image
c_max, c_top5, c_entropy, plus min-max-normalized [0,1] versions
(c_max_norm, c_top5_norm, c_entropy_norm) using dataset percentiles
[q10, q90] as the calibration band.

Use this on either the ViT pre-fusion parquet or the ConvNeXt
aggregated parquet. Output is per-branch.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Reuse Exp-8's f16 helper instead of re-implementing
EXP8 = os.path.join(os.path.dirname(__file__), "..", "..",
                    "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, os.path.abspath(EXP8))
from generate_submission import f16_bytes_to_f32_vector  # noqa: E402

NUM_CLASSES = 7806


def stats_for_vec(p: np.ndarray) -> Dict[str, float]:
    p = p.astype(np.float32)
    p = np.clip(p, 0.0, 1.0)  # in case post-fusion blends produced edge values
    sorted_desc = np.sort(p)[::-1]
    eps = 1e-12
    return {
        "c_max": float(sorted_desc[0]),
        "c_top5": float(sorted_desc[:5].sum()),
        "c_entropy": float(-(p * np.log(p + eps)).sum()),
    }


def normalize_band(arr: np.ndarray, q_lo: float = 0.10, q_hi: float = 0.90) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    lo = np.quantile(a, q_lo)
    hi = np.quantile(a, q_hi)
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True,
                    help="Input parquet with [image_name, image_probs_f16].")
    ap.add_argument("--out", required=True, help="Output parquet path.")
    ap.add_argument("--id-col", default="image_name")
    ap.add_argument("--probs-col", default="image_probs_f16")
    args = ap.parse_args()

    df = pq.read_table(args.parquet,
                       columns=[args.id_col, args.probs_col]).to_pandas()
    rows = []
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r[args.id_col]))[0]
        p = f16_bytes_to_f32_vector(r[args.probs_col], NUM_CLASSES)
        s = stats_for_vec(p)
        rows.append({"quadrat_id": qid, **s})

    out = pd.DataFrame(rows)
    out["c_max_norm"] = normalize_band(out["c_max"].values)
    out["c_top5_norm"] = normalize_band(out["c_top5"].values)
    # Lower entropy = more confident; invert so c_entropy_norm follows the same convention
    out["c_entropy_norm"] = 1.0 - normalize_band(out["c_entropy"].values)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"  wrote {len(out)} rows -> {args.out}")
    for col in ["c_max", "c_top5", "c_entropy",
                "c_max_norm", "c_top5_norm", "c_entropy_norm"]:
        print(f"  {col:18s} mean={out[col].mean():.4f} "
              f"q10={out[col].quantile(0.1):.4f} q90={out[col].quantile(0.9):.4f}")


if __name__ == "__main__":
    main()
