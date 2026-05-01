"""Exp-14 ensemble: adaptive temporal fusion per branch + probability blend.

Sibling of Exp-11/ensemble_with_knn.py. Differences:
  1. Reads the *pre-fusion* ViT parquet (not the post-fusion one).
  2. Re-fuses the ViT branch via fuse_v2.fuse_branch(controller_vit).
  3. Re-fuses the ConvNeXt branch via fuse_v2.fuse_branch(controller_cnxt).
  4. Blends with weight w (defaults to the Exp-11B optimum 0.77).
  5. Writes the same submission CSV format.

A run with default ControllerConfig (lambda=0, default_alpha=0.15 for ViT,
0.30 for ConvNeXt) and aggregator="max" must reproduce
ens_knn_w0.77_thr0.100_top10.csv to numerical tolerance — this is the
Stage-0 smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP14 = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXP14)
EXP8 = os.path.join(EXP14, "..", "Exp-8_Quantile_Aggregation_SS")
EXP11 = os.path.join(EXP14, "..", "Exp-11_Conv_ViT_Ensemble")
sys.path.insert(0, os.path.abspath(EXP8))
sys.path.insert(0, os.path.abspath(EXP11))

from controller.alpha_controller import ControllerConfig, baseline_config  # noqa: E402
from scripts.fuse_v2 import fuse_branch  # noqa: E402

# Re-use the heavy ConvNeXt aggregator and submission writer from Exp-11
from ensemble_with_knn import (  # noqa: E402
    aggregate as cnxt_aggregate,
    load_scales as cnxt_load_scales,
    generate_csv,
)
from generate_submission import load_class_mask_npy, load_species_map  # noqa: E402
from aggregate_tiles import save_aggregated  # noqa: E402
from signals import parse_source  # noqa: E402

NUM_CLASSES = 7806


def generate_csv_per_source_threshold(
    probs, species_map, class_mask, topk,
    threshold_default: float, threshold_per_source: dict,
    out_csv: str,
):
    """Sibling of generate_csv that applies a per-source threshold.

    A quadrat's threshold is looked up by parse_source(qid). Sources not
    in threshold_per_source use threshold_default.
    """
    rows = []
    for name, p in sorted(probs.items()):
        p = p.astype(np.float32)
        if class_mask is not None:
            p[~class_mask] = 0.0
        qid = os.path.splitext(name)[0]
        thr = threshold_per_source.get(parse_source(qid), threshold_default)
        top_idx = np.argsort(p)[::-1][:topk]
        selected = [i for i in top_idx if p[i] >= thr]
        if not selected:
            selected = [top_idx[0]]
        rows.append({
            "quadrat_id": qid,
            "species_ids": str([species_map[i] for i in selected])
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"    {os.path.basename(out_csv)}")


def _load_controller(path: Optional[str], branch: str) -> ControllerConfig:
    if path and os.path.exists(path):
        return ControllerConfig.from_json(path)
    return baseline_config(branch)


def _save_intermediate(probs: Dict[str, np.ndarray], out_dir: str, tag: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"_branch_{tag}.parquet")
    save_aggregated(probs, p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-prefusion-parquet", required=True,
                    help="Pre-fusion ViT parquet (knn_bM..._prefusion.parquet).")
    ap.add_argument("--convnext-scales", nargs="+", required=True,
                    help="Per-tile ConvNeXt parquets, one per scale.")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--weights", default="0.77",
                    help="Comma-separated ensemble weights w (P_final = w*ViT + (1-w)*CNXT).")
    ap.add_argument("--thresholds", default="0.100")
    ap.add_argument("--topk", type=int, default=10)
    # Controller configs (json) — optional, otherwise baseline (Exp-11B parity).
    ap.add_argument("--vit-controller", default=None,
                    help="JSON ControllerConfig for ViT branch.")
    ap.add_argument("--cnxt-controller", default=None,
                    help="JSON ControllerConfig for ConvNeXt branch.")
    # Prior aggregator (per branch).
    ap.add_argument("--vit-aggregator", default="max")
    ap.add_argument("--cnxt-aggregator", default="max")
    # Per-source aggregator overrides as JSON dict, e.g.
    #   '{"RNNB":"sim_w","OPTMix":"sim_w","GUARDEN":"sim_w"}'
    # Sources not in the dict use --vit-aggregator / --cnxt-aggregator as default.
    ap.add_argument("--vit-aggregator-per-source", default=None)
    ap.add_argument("--cnxt-aggregator-per-source", default=None)
    # Optional signals.
    ap.add_argument("--density", default=None)
    ap.add_argument("--vit-confidence", default=None)
    ap.add_argument("--cnxt-confidence", default=None)
    ap.add_argument("--sim-json", default=None)
    ap.add_argument("--season-json", default=None)
    ap.add_argument("--tag", default="exp14")
    # Per-source threshold override, e.g. '{"OPTMix":0.05,"GUARDEN":0.05}'.
    # Sources not in dict use the global --thresholds value.
    ap.add_argument("--threshold-per-source", default=None)
    # If set, save the post-blend ensemble probabilities to a parquet
    # before threshold+topK so they can be averaged across variants.
    ap.add_argument("--save-ensemble-parquet", action="store_true")
    args = ap.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES) \
        if args.class_mask_npy else None

    print("=== Exp-14 ensemble_v2 ===")
    print(f"  out-dir: {args.out_dir}")
    print(f"  tag:     {args.tag}")

    # --- ViT branch ---
    print("\n=== Fusing ViT branch ===")
    vit_ctrl = _load_controller(args.vit_controller, "vit")
    vit_per_src = json.loads(args.vit_aggregator_per_source) \
        if args.vit_aggregator_per_source else None
    print(f"  controller: {vit_ctrl.to_dict()}")
    print(f"  aggregator: {args.vit_aggregator}  per_source={vit_per_src}")
    vit_fused = fuse_branch(
        args.vit_prefusion_parquet,
        controller=vit_ctrl,
        aggregator=args.vit_aggregator,
        aggregator_per_source=vit_per_src,
        density_parquet=args.density,
        confidence_parquet=args.vit_confidence,
        sim_json=args.sim_json,
        season_json=args.season_json,
    )

    # --- ConvNeXt branch: aggregate tiles -> save -> re-load via fuse_v2 ---
    print("\n=== Aggregating ConvNeXt tiles ===")
    cnxt_tiles = cnxt_load_scales(args.convnext_scales)
    cnxt_probs = cnxt_aggregate(cnxt_tiles, args.temperature, args.class_mask_npy)
    print(f"  aggregated {len(cnxt_probs)} images")
    cnxt_prefusion_path = _save_intermediate(cnxt_probs, args.out_dir, "cnxt_prefusion")

    print("\n=== Fusing ConvNeXt branch ===")
    cnxt_ctrl = _load_controller(args.cnxt_controller, "convnext")
    cnxt_per_src = json.loads(args.cnxt_aggregator_per_source) \
        if args.cnxt_aggregator_per_source else None
    print(f"  controller: {cnxt_ctrl.to_dict()}")
    print(f"  aggregator: {args.cnxt_aggregator}  per_source={cnxt_per_src}")
    cnxt_fused = fuse_branch(
        cnxt_prefusion_path,
        controller=cnxt_ctrl,
        aggregator=args.cnxt_aggregator,
        aggregator_per_source=cnxt_per_src,
        density_parquet=args.density,
        confidence_parquet=args.cnxt_confidence,
        sim_json=args.sim_json,
        season_json=args.season_json,
    )
    try:
        os.remove(cnxt_prefusion_path)
    except OSError:
        pass

    # --- Blend + sweep ---
    common = set(vit_fused.keys()) & set(cnxt_fused.keys())
    print(f"\n=== Blending ===")
    print(f"  ViT:    {len(vit_fused)}")
    print(f"  CNXT:   {len(cnxt_fused)}")
    print(f"  common: {len(common)}")

    thr_per_src = json.loads(args.threshold_per_source) \
        if args.threshold_per_source else None
    for w in weights:
        ens: Dict[str, np.ndarray] = {}
        for name in common:
            pv = vit_fused[name].astype(np.float32)
            pc = cnxt_fused[name].astype(np.float32)
            ens[name] = (w * pv + (1 - w) * pc).astype(np.float16)
        if args.save_ensemble_parquet:
            pq_path = os.path.join(
                args.out_dir, f"{args.tag}_w{w:.2f}_ensemble.parquet")
            save_aggregated(ens, pq_path)
            print(f"  saved ensemble parquet -> {pq_path}")
        for thr in thresholds:
            csv = os.path.join(
                args.out_dir,
                f"{args.tag}_w{w:.2f}_thr{thr:.3f}_top{args.topk}.csv",
            )
            if thr_per_src:
                generate_csv_per_source_threshold(
                    ens, species_map, class_mask, args.topk,
                    threshold_default=thr,
                    threshold_per_source=thr_per_src,
                    out_csv=csv,
                )
            else:
                generate_csv(ens, species_map, class_mask, args.topk, thr, csv)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
