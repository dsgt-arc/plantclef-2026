"""Exp-16 — Post-blend vote-share rerank for clean LB testing.

Variant of apply_vote_rerank.py that operates on the LOCKED Stage 13
ensemble parquet (post-blend, post-temporal-fusion) rather than on the
ViT prefusion. This guarantees that beta=0 reproduces the Stage 13
winner CSV byte-for-byte, so LB deltas isolate the vote-share effect
without the historical-args drift in the orchestrator's reproduction
of the Stage 13 ensemble parquet.

Mechanism in one sentence: multiply the locked Stage 13 ensemble
probability vector elementwise by (vote_share + epsilon)^beta and
rescale per-image to preserve the original L1 sum, then apply the
production geo+aquatic+biogeo masks to emit a candidate CSV.
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

EXP14 = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-14_Adaptive_Temporal_Fusion",
))
EXP8 = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "Exp-8_Quantile_Aggregation_SS",
))
sys.path.insert(0, EXP14)
sys.path.insert(0, EXP8)

from generate_submission import (  # noqa: E402
    f16_bytes_to_f32_vector, load_class_mask_npy, load_species_map,
)

NUM_CLASSES = 7806
ALPINE_SOURCES = ("CBN-PdlC", "CBN-Pla")
ALL_SRC_PREFIXES = (
    "RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix", "2024-CEV3",
)


def source_of(qid: str) -> str:
    for p in ALL_SRC_PREFIXES:
        if qid.startswith(p):
            return p
    return "OTHER"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True,
                    help="Locked Stage 13 ensemble parquet (v_ld12RNNBonly_w0.77_ensemble.parquet).")
    ap.add_argument("--vote-share-parquet", required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True)
    ap.add_argument("--biogeo-npz", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    print("=" * 72)
    print("Exp-16: post-blend vote-share rerank")
    print("=" * 72)
    print(f"  ensemble:        {args.ensemble_parquet}")
    print(f"  vote-share:      {args.vote_share_parquet}")
    print(f"  beta:            {args.beta}")
    print(f"  epsilon:         {args.epsilon}")
    print(f"  threshold:       {args.threshold}")
    print(f"  topk:            {args.topk}")

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    combined_global = geo & aquatic
    combined_alpine = combined_global & mask_alpine
    print(f"\n  combined global: {combined_global.sum()} / {NUM_CLASSES}")
    print(f"  combined alpine: {combined_alpine.sum()} / {NUM_CLASSES}")

    species_map = load_species_map(args.species_map, NUM_CLASSES)

    ens_df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()
    ens_map: Dict[str, np.ndarray] = {}
    for _, r in ens_df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        ens_map[qid] = f16_bytes_to_f32_vector(
            r["image_probs_f16"], NUM_CLASSES
        ).astype(np.float32)
    print(f"  loaded {len(ens_map)} ensemble vectors")

    vs_df = pd.read_parquet(
        args.vote_share_parquet,
        columns=["image_name", "vote_share_f16"],
    )
    vs_map: Dict[str, np.ndarray] = {}
    for _, r in vs_df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        v = np.frombuffer(r["vote_share_f16"], dtype=np.float16).astype(np.float32)
        vs_map[qid] = v
    print(f"  loaded {len(vs_map)} vote-share vectors")

    rows = []
    for qid in sorted(ens_map.keys()):
        p = ens_map[qid].copy()
        if qid in vs_map:
            vs = vs_map[qid]
        else:
            vs = np.zeros(NUM_CLASSES, dtype=np.float32)

        if args.beta != 0.0:
            factor = np.power(vs + args.epsilon, args.beta).astype(np.float32)
            s_orig = float(p.sum())
            p = p * factor
            s_new = float(p.sum())
            if s_new > 0 and s_orig > 0:
                p *= (s_orig / s_new)
            np.clip(p, 0.0, 1.0, out=p)

        src = source_of(qid)
        if src in ALPINE_SOURCES:
            p[~combined_alpine] = 0.0
        else:
            p[~combined_global] = 0.0

        order = np.argsort(p)[::-1][:args.topk]
        selected: List[int] = [int(i) for i in order if p[int(i)] >= args.threshold]
        if not selected:
            selected = [int(order[0])]

        sids = [species_map[i] for i in selected]
        rows.append({"quadrat_id": qid, "species_ids": str(sids)})

    out_df = pd.DataFrame(rows).sort_values("quadrat_id")
    Path(os.path.dirname(os.path.abspath(args.out_csv))).mkdir(
        parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\n  wrote {len(out_df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
