"""ViT-only fork of Exp-14/scripts/ensemble_v2.py.

Removes the ConvNeXt branch entirely. The fused ViT probabilities (with
sim_w-on-RNNB aggregator and the locked vit_rnnb030.json controller) are
saved directly as the "ensemble" parquet. The downstream apply_habitat_fit
expects exactly that parquet format (Dict[qid, float16 vector]).

Original blend formula: P_final = w * P_vit + (1-w) * P_cnxt
This script forces w = 1.0 and skips the ConvNeXt branch.

Reproducibility note: at w=1.0 the original ensemble_v2.py is numerically
equivalent to this script but still requires the ConvNeXt parquets and
runs ~20 min of wasted ConvNeXt aggregation/fusion compute.

Outputs:
  - <out-dir>/<tag>_w1.00_ensemble.parquet  (input to apply_habitat_fit.py)
  - <out-dir>/<tag>_w1.00_thr<THR>_top<K>.csv  (bare ViT-only submission CSV)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABL1 = os.path.dirname(SCRIPT_DIR)
SRC = os.path.dirname(ABL1)
EXP14 = os.path.join(SRC, "Exp-14_Adaptive_Temporal_Fusion")
EXP8 = os.path.join(SRC, "Exp-8_Quantile_Aggregation_SS")
EXP11 = os.path.join(SRC, "Exp-11_Conv_ViT_Ensemble")
sys.path.insert(0, EXP14)
sys.path.insert(0, EXP8)
sys.path.insert(0, EXP11)

from controller.alpha_controller import ControllerConfig, baseline_config  # noqa: E402
from scripts.fuse_v2 import fuse_branch  # noqa: E402
from ensemble_with_knn import generate_csv  # noqa: E402
from generate_submission import load_class_mask_npy, load_species_map  # noqa: E402
from aggregate_tiles import save_aggregated  # noqa: E402

NUM_CLASSES = 7806


def _load_controller(path: Optional[str]) -> ControllerConfig:
    if path and os.path.exists(path):
        return ControllerConfig.from_json(path)
    return baseline_config("vit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-prefusion-parquet", required=True,
                    help="Pre-fusion ViT parquet (knn_bM..._prefusion.parquet "
                         "or vitL_prefusion_b<BETA>.parquet).")
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresholds", default="0.085",
                    help="Comma-separated thresholds for the bare CSV(s).")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--vit-controller", default=None,
                    help="JSON ControllerConfig for ViT branch.")
    ap.add_argument("--vit-aggregator", default="max")
    ap.add_argument("--vit-aggregator-per-source", default=None,
                    help='JSON dict, e.g. \'{"RNNB":"sim_w"}\'.')
    ap.add_argument("--density", default=None)
    ap.add_argument("--vit-confidence", default=None)
    ap.add_argument("--sim-json", default=None)
    ap.add_argument("--season-json", default=None)
    ap.add_argument("--tag", default="abl1_vitonly")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    species_map = load_species_map(args.species_map, NUM_CLASSES)
    class_mask = load_class_mask_npy(args.class_mask_npy, NUM_CLASSES) \
        if args.class_mask_npy else None

    print("=== Abl-1 ensemble_vit_only (ConvNeXt knockout) ===")
    print(f"  out-dir: {args.out_dir}")
    print(f"  tag:     {args.tag}")
    print(f"  ViT prefusion: {args.vit_prefusion_parquet}")

    print("\n=== Fusing ViT branch ===")
    vit_ctrl = _load_controller(args.vit_controller)
    if args.vit_aggregator_per_source:
        # Accept either a JSON literal or a path to a JSON file (lets
        # PowerShell callers avoid double-quote-stripping headaches).
        if os.path.exists(args.vit_aggregator_per_source):
            with open(args.vit_aggregator_per_source, "r", encoding="utf-8") as fh:
                vit_per_src = json.load(fh)
        else:
            vit_per_src = json.loads(args.vit_aggregator_per_source)
    else:
        vit_per_src = None
    print(f"  controller: {vit_ctrl.to_dict()}")
    print(f"  aggregator: {args.vit_aggregator}  per_source={vit_per_src}")
    vit_fused: Dict[str, np.ndarray] = fuse_branch(
        args.vit_prefusion_parquet,
        controller=vit_ctrl,
        aggregator=args.vit_aggregator,
        aggregator_per_source=vit_per_src,
        density_parquet=args.density,
        confidence_parquet=args.vit_confidence,
        sim_json=args.sim_json,
        season_json=args.season_json,
    )
    print(f"  fused {len(vit_fused)} quadrats")

    pq_path = os.path.join(args.out_dir, f"{args.tag}_w1.00_ensemble.parquet")
    save_aggregated(vit_fused, pq_path)
    print(f"  saved ensemble parquet -> {pq_path}")

    for thr in thresholds:
        csv = os.path.join(
            args.out_dir,
            f"{args.tag}_w1.00_thr{thr:.3f}_top{args.topk}.csv",
        )
        generate_csv(vit_fused, species_map, class_mask, args.topk, thr, csv)
        print(f"  wrote bare CSV -> {csv}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
