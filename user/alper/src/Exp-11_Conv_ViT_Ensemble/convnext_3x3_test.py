"""Quick test: ConvNeXt with 3x3+4x4+5x5+6x6 vs 4x4+5x5+6x6 standalone."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP8_DIR = os.path.join(SCRIPT_DIR, "..", "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, EXP8_DIR)

from utils.aggregator import MaxProbabilityAggregator
from utils.io_bytes import f16_bytes_list_to_f32_matrix
from aggregate_tiles import save_aggregated
from temporal_fusion import (
    load_quadrat_probs, group_by_location,
    build_location_profiles, fuse_all,
)
from generate_submission import load_species_map, load_class_mask_npy

NUM_CLASSES = 7806


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-files", nargs="+", required=True,
                    help="Parquet files for each scale")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="test")
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--thresholds", default="0.075,0.100,0.125")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES) if args.class_mask_npy else None
    geo_mask = np.load(args.class_mask_npy).astype(np.bool_) if args.class_mask_npy else None

    # Load and combine all scales
    frames = []
    for f in args.scale_files:
        df = pd.read_parquet(f)
        print(f"  {os.path.basename(f)}: {len(df)} tiles, {df['image_name'].nunique()} images")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined)} tiles, {combined['image_name'].nunique()} images")

    # Aggregate
    aggregator = MaxProbabilityAggregator(num_classes=NUM_CLASSES, temperature=args.temperature, image_topk=64)
    image_probs = {}
    for name, group in combined.groupby("image_name"):
        logits_bufs = group["teacher_logits_f16"].tolist()
        tile_logits = f16_bytes_list_to_f32_matrix(logits_bufs, num_classes=NUM_CLASSES)
        p = aggregator.aggregate(tile_logits)
        if geo_mask is not None:
            p_f32 = p.astype(np.float32)
            p_f32[~geo_mask] = 0.0
            p = p_f32.astype(np.float16)
        image_probs[str(name)] = p
    print(f"  Aggregated {len(image_probs)} images")

    # Temporal fusion
    temp_path = os.path.join(args.out_dir, f"_temp_{args.tag}.parquet")
    save_aggregated(image_probs, temp_path)
    qp = load_quadrat_probs(temp_path)
    lg = group_by_location(list(qp.keys()))
    lp = build_location_profiles(qp, lg, NUM_CLASSES)
    fused = fuse_all(qp, lg, lp, strategy="prior_blend", params={"alpha": args.alpha})
    os.remove(temp_path)
    fused_f16 = {k: v.astype(np.float16) for k, v in fused.items()}

    # Generate submissions
    for thr in thresholds:
        rows = []
        for name, probs in sorted(fused_f16.items()):
            p = probs.astype(np.float32)
            if class_mask is not None:
                p[~class_mask] = 0.0
            top_idx = np.argsort(p)[::-1][:args.topk]
            selected = [i for i in top_idx if p[i] >= thr]
            if len(selected) < 1:
                selected = list(top_idx[:1])
            sids = [species_map[i] for i in selected]
            rows.append({"quadrat_id": os.path.splitext(name)[0], "species_ids": str(sids)})
        csv_path = os.path.join(args.out_dir, f"cnxt_{args.tag}_thr{thr:.3f}_top{args.topk}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  {os.path.basename(csv_path)}")

    print("Done")


if __name__ == "__main__":
    main()
