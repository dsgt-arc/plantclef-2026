"""Apply geo + aquatic + per-source biogeographic masks to a Stage 11
ensemble probability parquet, then re-emit the threshold + topK CSV.

Stage 13 extension to apply_habitat_mask.py: same input parquet, same
geo+aquatic masks, plus a per-source biogeo mask applied per-quadrat
to the probability vector BEFORE the topK selection. Five hand-vetted
biogeographically-impossible-on-alpine species are zeroed on CBN-PdlC
and CBN-Pla quadrats only; all other sources are unchanged.

Mechanism in one sentence: on alpine sources, zero five species whose
biogeographic distribution excludes alpine summit habitat.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP14 = os.path.dirname(SCRIPT_DIR)
EXP8 = os.path.join(EXP14, "..", "Exp-8_Quantile_Aggregation_SS")
EXP11 = os.path.join(EXP14, "..", "Exp-11_Conv_ViT_Ensemble")
sys.path.insert(0, os.path.abspath(EXP8))
sys.path.insert(0, os.path.abspath(EXP11))

from generate_submission import (  # noqa: E402
    f16_bytes_to_f32_vector, load_class_mask_npy, load_species_map,
)
from ensemble_with_knn import generate_csv  # noqa: E402

NUM_CLASSES = 7806

ALPINE_SOURCES = ("CBN-PdlC", "CBN-Pla")


def source_of(qid: str) -> str:
    for p in ("RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix", "2024-CEV3"):
        if qid.startswith(p):
            return p
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-parquet", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--geo-mask", required=True)
    ap.add_argument("--habitat-mask", required=True,
                    help="Aquatic blacklist mask (Stage 12).")
    ap.add_argument("--biogeo-npz", required=True,
                    help="Per-source biogeo masks (mask_alpine, mask_default).")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.100)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    geo = load_class_mask_npy(args.geo_mask, NUM_CLASSES)
    aquatic = np.load(args.habitat_mask).astype(np.bool_)
    if aquatic.shape[0] != NUM_CLASSES:
        raise ValueError(f"habitat mask shape {aquatic.shape} != {NUM_CLASSES}")
    biogeo = np.load(args.biogeo_npz)
    mask_alpine = biogeo["mask_alpine"].astype(np.bool_)
    mask_default = biogeo["mask_default"].astype(np.bool_)
    for nm, m in (("alpine", mask_alpine), ("default", mask_default)):
        if m.shape[0] != NUM_CLASSES:
            raise ValueError(f"biogeo {nm} mask shape {m.shape} != {NUM_CLASSES}")

    combined_global = geo & aquatic
    print(f"  geo mask passes:           {geo.sum()} / {NUM_CLASSES}")
    print(f"  aquatic mask passes:       {aquatic.sum()} / {NUM_CLASSES}")
    print(f"  combined geo&aquatic:      {combined_global.sum()} / {NUM_CLASSES}")
    print(f"  biogeo mask_alpine passes: {mask_alpine.sum()} / {NUM_CLASSES}")
    print(f"    extra blocked on alpine: {(combined_global & ~mask_alpine).sum()} species")

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    df = pq.read_table(
        args.ensemble_parquet,
        columns=["image_name", "image_probs_f16"],
    ).to_pandas()

    probs: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        p = f16_bytes_to_f32_vector(r["image_probs_f16"], NUM_CLASSES).astype(np.float32)
        src = source_of(qid)
        if src in ALPINE_SOURCES:
            p[~mask_alpine] = 0.0
        probs[qid] = p
    print(f"  loaded {len(probs)} quadrats")

    n_alpine = sum(1 for q in probs if source_of(q) in ALPINE_SOURCES)
    print(f"  alpine-source quadrats: {n_alpine}")

    generate_csv(probs, species_map, combined_global, args.topk, args.threshold,
                 args.out_csv)
    print(f"  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
