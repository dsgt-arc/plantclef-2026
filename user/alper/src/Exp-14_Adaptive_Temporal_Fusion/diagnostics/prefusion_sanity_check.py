"""Phase 0.2 — Failure-mode-3 sanity check.

For the four RNNB pair sites the user flagged, load both visits' rows
from the pre-fusion ViT parquet (the parquet that comes BEFORE temporal
fusion is applied) and confirm their probability vectors are not
identical. If max-abs-diff is ~0 for any pair, the bug is upstream of
fusion (an extractor or dedupe step is reading the same image twice)
and the entire adaptive-alpha plan is invalid until that is fixed.

Run AFTER pulling submissions/exp9b_knn_diff_beta_phase2a/
knn_bM0.85_bS1.00_T1.50_prefusion.parquet from PACE to local.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXP8 = os.path.join(os.path.dirname(__file__), "..", "..",
                    "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, os.path.abspath(EXP8))
from generate_submission import f16_bytes_to_f32_vector  # noqa: E402

NUM_CLASSES = 7806

PAIR_SITES: List[Tuple[str, str]] = [
    ("RNNB-3-1-20230512", "RNNB-3-1-20240117"),
    ("RNNB-3-2-20230512", "RNNB-3-2-20240117"),
    ("RNNB-2-7-20230512", "RNNB-2-7-20240117"),
    ("RNNB-1-2-20230512", "RNNB-1-2-20240117"),
]

PASS_DIFF_THRESHOLD = 0.05  # max-abs-diff must exceed this; tighter than 0


def load_pair(parquet: str, qid_a: str, qid_b: str,
              id_col: str = "image_name",
              probs_col: str = "image_probs_f16") -> Tuple[np.ndarray, np.ndarray]:
    df = pq.read_table(parquet, columns=[id_col, probs_col]).to_pandas()
    # image_name may have ".jpg" suffix; quadrat_id strips it
    df["__qid"] = df[id_col].astype(str).str.replace(r"\.jpg$", "", regex=True)
    a_row = df[df["__qid"] == qid_a]
    b_row = df[df["__qid"] == qid_b]
    if a_row.empty or b_row.empty:
        raise KeyError(f"{qid_a} or {qid_b} missing from {parquet}")
    pa = f16_bytes_to_f32_vector(a_row.iloc[0][probs_col], NUM_CLASSES)
    pb = f16_bytes_to_f32_vector(b_row.iloc[0][probs_col], NUM_CLASSES)
    return pa, pb


def summarize(pa: np.ndarray, pb: np.ndarray) -> Dict[str, float]:
    diff = np.abs(pa.astype(np.float64) - pb.astype(np.float64))
    cos = float(np.dot(pa, pb) / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-12))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "l2_diff": float(np.linalg.norm(pa - pb)),
        "cosine_sim": cos,
        "top1_a": int(np.argmax(pa)),
        "top1_b": int(np.argmax(pb)),
        "top1_p_a": float(pa.max()),
        "top1_p_b": float(pb.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefusion-parquet", required=True,
                    help="Path to knn_bM0.85_bS1.00_T1.50_prefusion.parquet")
    ap.add_argument("--out-md", default=None,
                    help="Optional markdown report path.")
    args = ap.parse_args()

    print(f"=== Phase 0.2 sanity check on {args.prefusion_parquet} ===\n")
    rows: List[Dict] = []
    pass_all = True
    for qid_a, qid_b in PAIR_SITES:
        try:
            pa, pb = load_pair(args.prefusion_parquet, qid_a, qid_b)
            s = summarize(pa, pb)
            ok = s["max_abs_diff"] > PASS_DIFF_THRESHOLD
            pass_all &= ok
            print(f"  {qid_a}  vs  {qid_b}")
            print(f"    max_abs_diff = {s['max_abs_diff']:.6f}  "
                  f"{'PASS' if ok else '*** FAIL ***'}")
            print(f"    mean_abs_diff= {s['mean_abs_diff']:.6f}")
            print(f"    cos_sim      = {s['cosine_sim']:.6f}")
            print(f"    top1: {s['top1_a']} (p={s['top1_p_a']:.3f})  "
                  f"vs {s['top1_b']} (p={s['top1_p_b']:.3f})")
            rows.append({"qid_a": qid_a, "qid_b": qid_b, "ok": ok, **s})
        except KeyError as e:
            print(f"  ! {e}")
            rows.append({"qid_a": qid_a, "qid_b": qid_b, "ok": False,
                         "error": str(e)})
            pass_all = False
        print()

    if args.out_md:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".",
                    exist_ok=True)
        with open(args.out_md, "w") as fh:
            fh.write("# Phase 0.2 Sanity Check Results\n\n")
            fh.write(f"Source parquet: `{args.prefusion_parquet}`\n\n")
            fh.write(df.to_markdown(index=False))
            fh.write(f"\n\n**Overall: {'PASS' if pass_all else 'FAIL'}**\n")
        print(f"  wrote report -> {args.out_md}")

    if not pass_all:
        print("\n*** SANITY CHECK FAILED ***")
        print("    Pre-fusion vectors are too similar between paired visits.")
        print("    Stop the alpha plan; the bug is upstream of temporal fusion.")
        sys.exit(2)
    print("Sanity check PASSED — pre-fusion vectors differ meaningfully. "
          "Safe to proceed.")


if __name__ == "__main__":
    main()
