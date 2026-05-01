"""Exp-16 Phase 1 — apply vote-share rerank to a ViT prefusion parquet.

Mechanism in one sentence: multiply the production ViT prefusion
probability vector elementwise by `(vote_share + epsilon)^beta` and
rescale per-image to preserve the original L1 sum, so species with
broad multi-tile support are preferred over single-tile noise spikes.

beta=0 reproduces the input prefusion byte-for-byte (factor identically
1.0). This is the wiring sanity check.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXP14 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-14_Adaptive_Temporal_Fusion",
)
EXP8 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-8_Quantile_Aggregation_SS",
)
sys.path.insert(0, os.path.abspath(EXP14))
sys.path.insert(0, os.path.abspath(EXP8))

from generate_submission import f16_bytes_to_f32_vector  # noqa: E402

NUM_CLASSES = 7806


def f32_to_f16_bytes(v: np.ndarray) -> bytes:
    return v.astype(np.float16).tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefusion-parquet", required=True,
                    help="Input ViT prefusion parquet (knn_bM..._prefusion.parquet).")
    ap.add_argument("--vote-share-parquet", required=True,
                    help="Output of compute_vote_share.py.")
    ap.add_argument("--beta", type=float, required=True,
                    help="Exponent for (vote_share + epsilon)^beta. beta=0 reproduces baseline.")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="Floor added to vote_share before exponentiation.")
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--preserve", default="l1",
                    choices=["l1", "max", "none"],
                    help="Per-image rescale to preserve L1 sum (default), per-image max, or no rescale.")
    args = ap.parse_args()

    print("=== Exp-16 Phase 1: apply_vote_rerank ===")
    print(f"  prefusion:   {args.prefusion_parquet}")
    print(f"  vote-share:  {args.vote_share_parquet}")
    print(f"  beta:        {args.beta}")
    print(f"  epsilon:     {args.epsilon}")
    print(f"  preserve:    {args.preserve}")

    pref_df = pq.read_table(
        args.prefusion_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    pref_df["qid"] = pref_df["image_name"].apply(
        lambda s: os.path.splitext(str(s))[0]
    )
    pref_map: Dict[str, np.ndarray] = {}
    for _, r in pref_df.iterrows():
        pref_map[r["qid"]] = f16_bytes_to_f32_vector(
            r["image_probs_f16"], NUM_CLASSES
        )
    print(f"  loaded {len(pref_map)} prefusion vectors")

    vs_df = pd.read_parquet(
        args.vote_share_parquet,
        columns=["image_name", "vote_share_f16"],
    )
    vs_df["qid"] = vs_df["image_name"].apply(
        lambda s: os.path.splitext(str(s))[0]
    )
    vs_map: Dict[str, np.ndarray] = {}
    for _, r in vs_df.iterrows():
        v = np.frombuffer(r["vote_share_f16"], dtype=np.float16).astype(np.float32)
        if v.shape[0] != NUM_CLASSES:
            raise ValueError(
                f"vote_share dim {v.shape[0]} != {NUM_CLASSES} for {r['qid']}"
            )
        vs_map[r["qid"]] = v
    print(f"  loaded {len(vs_map)} vote-share vectors")

    common = set(pref_map.keys()) & set(vs_map.keys())
    pref_only = set(pref_map.keys()) - set(vs_map.keys())
    vs_only = set(vs_map.keys()) - set(pref_map.keys())
    print(f"  common: {len(common)}, pref-only: {len(pref_only)}, "
          f"vs-only: {len(vs_only)}")
    if pref_only:
        print(f"    WARN pref-only sample: {list(pref_only)[:3]}")
    if vs_only:
        print(f"    WARN vs-only sample: {list(vs_only)[:3]}")

    out_rows = []
    n_clip = 0
    for qid in sorted(pref_map.keys()):
        p = pref_map[qid].astype(np.float32)
        if qid in vs_map:
            vs = vs_map[qid]
        else:
            vs = np.zeros(NUM_CLASSES, dtype=np.float32)
        factor = np.power(vs + args.epsilon, args.beta).astype(np.float32)
        p_re = p * factor
        if args.preserve == "l1":
            s_orig = float(p.sum())
            s_new = float(p_re.sum())
            if s_new > 0 and s_orig > 0:
                p_re *= (s_orig / s_new)
        elif args.preserve == "max":
            m_orig = float(p.max())
            m_new = float(p_re.max())
            if m_new > 0 and m_orig > 0:
                p_re *= (m_orig / m_new)
        if (p_re > 1.0).any():
            n_clip += 1
            np.clip(p_re, 0.0, 1.0, out=p_re)
        # Mirror input "image_name" column convention (with .jpg suffix if present).
        # Look up the original key in the prefusion to keep row identity.
        nz = int(np.count_nonzero(p_re))
        out_rows.append({
            "image_name": qid + ".jpg",
            "image_probs_f16": f32_to_f16_bytes(p_re),
            "num_nonzero": nz,
        })

    if n_clip > 0:
        print(f"  clipped {n_clip}/{len(out_rows)} images at p>1.0 after rescale")

    out_df = pd.DataFrame(out_rows)
    Path(os.path.dirname(os.path.abspath(args.out_parquet))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"\n  wrote {len(out_df)} rows -> {args.out_parquet}")


if __name__ == "__main__":
    main()
