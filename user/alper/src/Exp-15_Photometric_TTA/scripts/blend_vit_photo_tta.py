"""Phase-3 photometric TTA blend for the ViT branch.

Blends two pre-fusion ViT parquets (the original 0.45572-winner one and the
photometric-rerun one) at the per-image probability level:

    P_blend(qid) = beta * P_orig(qid) + (1 - beta) * P_photo(qid)

The `prefusion` parquets are the post-aggregation, pre-temporal-fusion
outputs from `Exp-9B/multiscale_aggregate_knn.py` (the file
`knn_bM0.85_bS1.00_T1.50_prefusion.parquet`). They contain
[image_name, image_probs_f16, num_nonzero] per quadrat.

Outputs one blended parquet per beta value; downstream the user runs
ensemble_v2.py with --vit-prefusion-parquet pointing at the blended file.

Beta = 1.0 reproduces the original; beta = 0.0 is pure-photometric.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXP15_ROOT = Path(__file__).resolve().parents[1]
EXP14_ROOT = EXP15_ROOT.parent / "Exp-14_Adaptive_Temporal_Fusion"
EXP8_ROOT = EXP15_ROOT.parent / "Exp-8_Quantile_Aggregation_SS"
sys.path.insert(0, str(EXP8_ROOT))
from generate_submission import f16_bytes_to_f32_vector  # noqa: E402
from aggregate_tiles import save_aggregated  # noqa: E402

NUM_CLASSES = 7806


def load_prefusion(path: str) -> Dict[str, np.ndarray]:
    df = pq.read_table(path, columns=["image_name", "image_probs_f16"]).to_pandas()
    out = {}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        out[qid] = f16_bytes_to_f32_vector(r["image_probs_f16"], NUM_CLASSES)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-prefusion", required=True,
                    help="Baseline ViT prefusion parquet "
                         "(knn_bM0.85_bS1.00_T1.50_prefusion.parquet).")
    ap.add_argument("--photo-prefusion", required=True,
                    help="Photometric ViT prefusion parquet (output of running "
                         "Exp-9B/multiscale_aggregate_knn.py on the new tile parquets).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--betas", default="0.30,0.45,0.55,0.65,0.75,0.85,1.00",
                    help="Comma-separated beta values to blend at "
                         "(P = beta*orig + (1-beta)*photo).")
    ap.add_argument("--tag", default="vit_photo_tta")
    args = ap.parse_args()

    print(f"  loading {args.orig_prefusion}")
    orig = load_prefusion(args.orig_prefusion)
    print(f"  loading {args.photo_prefusion}")
    photo = load_prefusion(args.photo_prefusion)

    common = sorted(set(orig.keys()) & set(photo.keys()))
    print(f"  orig: {len(orig)}  photo: {len(photo)}  common: {len(common)}")

    betas = [float(b) for b in args.betas.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)
    for beta in betas:
        blend = {}
        for qid in common:
            p = beta * orig[qid] + (1.0 - beta) * photo[qid]
            blend[qid] = p.astype(np.float16)
        out_path = os.path.join(
            args.out_dir, f"{args.tag}_beta{beta:.2f}_prefusion.parquet"
        )
        save_aggregated(blend, out_path)
        print(f"    beta={beta:.2f} -> {out_path}")

    print("\n  Next: re-run Exp-14 ensemble_v2.py with one of these blended parquets")
    print("        as --vit-prefusion-parquet, then apply_habitat_mask.py, then proxy_metrics.")


if __name__ == "__main__":
    main()
