"""Visual visit similarity from tile-level DINOv2 CLS embeddings.

For each test image, aggregate per-tile CLS embeddings into a single
per-image vector (mean pool + L2-norm). For each location, compute the
pairwise cosine-similarity matrix across visits. Output: JSON
{quadrat_id: {sibling_qid: cosine}} for use by prior_aggregators.sim_w.

Designed to run on PACE where the per-tile parquets live. Reads the
embedding column shared by Exp-7 multiscale parquets and Exp-9B
LoRA-test parquets ('embedding_orig_f16' = frozen DINOv2 — preferred
because it captures raw visual content, not species-decision-aligned
features).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

def parse_location(qid: str) -> str:
    parts = qid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[0]
    return qid


def _decode_f16_bytes_list(buffers: list, dim: int) -> np.ndarray:
    out = np.empty((len(buffers), dim), dtype=np.float32)
    for i, b in enumerate(buffers):
        v = np.frombuffer(b, dtype=np.float16)
        if v.size != dim:
            raise ValueError(f"Expected {dim} values but got {v.size}")
        out[i] = v.astype(np.float32)
    return out


def aggregate_per_image(parquet_paths: List[str], emb_col: str) -> Dict[str, np.ndarray]:
    """Mean-pool tile embeddings -> one f32 vector per quadrat_id.

    Embedding dimension is auto-detected from the first row (768 for
    DINOv2 ViT-B, 1536 for ConvNeXt-Large).
    """
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = defaultdict(int)
    dim = None
    for p in parquet_paths:
        df = pd.read_parquet(p, columns=["image_name", emb_col])
        print(f"  {os.path.basename(p)}: {len(df)} tiles, "
              f"{df['image_name'].nunique()} images")
        if dim is None:
            first = np.frombuffer(df[emb_col].iloc[0], dtype=np.float16)
            dim = int(first.size)
            print(f"  embedding dim auto-detected: {dim}")
        names = df["image_name"].astype(str).str.replace(r"\.jpg$", "", regex=True).values
        emb = _decode_f16_bytes_list(df[emb_col].tolist(), dim)
        for i, qid in enumerate(names):
            v = sums.get(qid)
            sums[qid] = emb[i] if v is None else v + emb[i]
            counts[qid] += 1
    out: Dict[str, np.ndarray] = {}
    for qid, s in sums.items():
        v = s / max(counts[qid], 1)
        n = np.linalg.norm(v) + 1e-12
        out[qid] = (v / n).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", nargs="+", required=True,
                    help="One or more per-tile parquets with embedding_orig_f16.")
    ap.add_argument("--emb-col", default="embedding_orig_f16",
                    help="Use 'emb_arcface_f16' for ArcFace-projected embeddings.")
    ap.add_argument("--out", required=True, help="Output JSON.")
    args = ap.parse_args()

    print(f"Aggregating per-image embeddings from {len(args.parquets)} parquet(s)")
    emb_per_qid = aggregate_per_image(args.parquets, args.emb_col)
    print(f"  pooled to {len(emb_per_qid)} per-image vectors")

    by_loc: Dict[str, List[str]] = defaultdict(list)
    for qid in emb_per_qid:
        by_loc[parse_location(qid)].append(qid)

    sim: Dict[str, Dict[str, float]] = {}
    n_pairs = 0
    for _loc, members in by_loc.items():
        if len(members) == 1:
            qid = members[0]
            sim[qid] = {qid: 1.0}
            continue
        mat = np.stack([emb_per_qid[q] for q in members], axis=0)  # already normed
        cos = mat @ mat.T  # NxN
        for i, qi in enumerate(members):
            sim[qi] = {qj: float(cos[i, j]) for j, qj in enumerate(members)}
            n_pairs += len(members)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(sim, fh)
    print(f"  wrote {len(sim)} qids ({n_pairs} pair entries) -> {args.out}")

    # Diagnostic: print one RNNB pair to confirm signal is present
    for probe in ("RNNB-3-2-20230512", "RNNB-1-2-20230512"):
        if probe in sim:
            sib = [k for k in sim[probe] if k != probe]
            for s in sib:
                print(f"  example: {probe} ~ {s} -> cos={sim[probe][s]:.4f}")


if __name__ == "__main__":
    main()
