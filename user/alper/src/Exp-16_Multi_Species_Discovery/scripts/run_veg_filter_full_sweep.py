"""Exp-16 veg-filter full sweep — orchestrator.

For each threshold (including 0.0 = unfiltered baseline at no-kNN beta=1):
  1. Aggregate ViT branch tiles locally at beta=1.0 (no kNN), with
     optional veg_frac filter applied to ViT tile parquets.
  2. Filter ConvNeXt tile parquets by veg_frac.
  3. Run ensemble_v2 with locally-aggregated ViT prefusion + filtered
     ConvNeXt tiles.
  4. Apply geo+aquatic+biogeo masks + N3_r0.5 admission.
  5. Output candidate CSV.

threshold=0.0 is the no-kNN baseline anchor; LB delta vs locked best
(0.46168 with kNN) tells us the kNN gap. Filtered variants tell us the
mechanism delta. If positive, push to PACE for kNN-aware filter test.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXP16 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP14 = os.path.abspath(os.path.join(EXP16, "..", "Exp-14_Adaptive_Temporal_Fusion"))
PACE = os.path.join(EXP14, "pace_downloads")
SIGNALS = os.path.join(EXP14, "outputs", "signals")
CTRL = os.path.join(EXP14, "stage_configs", "controllers")
CNXT_DIR = os.path.join(PACE, "ConvNext_large_parquets")
VIT_DIR = os.path.join(PACE, "ViT_classifier")

SPECIES_MAP = os.path.join(PACE, "class_to_species_id.json")
GEO_MASK = os.path.join(PACE, "sw_europe_mask_bool.npy")
HABITAT_MASK = os.path.join(SIGNALS, "aquatic_blacklist_mask.npy")
BIOGEO_NPZ = os.path.join(SIGNALS, "biogeo_per_source.npz")
SIM_JSON = os.path.join(SIGNALS, "visit_similarity.json")
DENSITY_PARQUET = os.path.join(SIGNALS, "density.parquet")
VIT_CTRL = os.path.join(CTRL, "vit_rnnb030_only.json")
CNXT_CTRL = os.path.join(CTRL, "cnxt_baseline_ld12_RNNBonly.json")
VEG_PARQUET = os.path.join(EXP16, "outputs", "image_audit",
                           "per_tile_veg_frac.parquet")


def run(cmd, cwd=None):
    print(f"\n  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veg-thresholds", default="0.0,0.02,0.05,0.10",
                    help="Comma-separated veg_frac thresholds.")
    ap.add_argument("--out-root", default=os.path.join(
        EXP16, "outputs", "veg_filter_full_sweep"))
    ap.add_argument("--skip-cnxt-filter", action="store_true",
                    help="Apply filter only on ViT branch (ConvNeXt unchanged).")
    args = ap.parse_args()

    if not os.path.isfile(VEG_PARQUET):
        raise SystemExit(f"missing veg_frac parquet: {VEG_PARQUET}")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    thresholds = [float(t) for t in args.veg_thresholds.split(",")]
    print(f"  veg-thresholds: {thresholds}")
    print(f"  skip cnxt filter: {args.skip_cnxt_filter}")

    for thr in thresholds:
        tag = f"veg{thr:.2f}"
        var_dir = os.path.join(args.out_root, tag)
        Path(var_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: aggregate ViT locally at beta=1.0 with optional filter
        vit_pq = os.path.join(var_dir, "vit_local_no_knn.parquet")
        cmd = [
            sys.executable, os.path.join(EXP16, "scripts", "local_vit_aggregate.py"),
            "--scale4-parquet", os.path.join(VIT_DIR, "scale_4x4.parquet"),
            "--scale5-parquet", os.path.join(VIT_DIR, "scale_5x5.parquet"),
            "--scale6-parquet", os.path.join(VIT_DIR, "scale_6x6.parquet"),
            "--temperature", "1.5",
            "--geo-mask", GEO_MASK,
            "--image-topk", "64",
            "--aug-filter",
            "--out-parquet", vit_pq,
        ]
        if thr > 0:
            cmd += ["--veg-frac-parquet", VEG_PARQUET,
                    "--veg-threshold", str(thr)]
        run(cmd)

        # Step 2: filter ConvNeXt scales
        if args.skip_cnxt_filter or thr == 0.0:
            cnxt_paths = [os.path.join(CNXT_DIR, f"scale_{s}x{s}.parquet")
                          for s in (4, 5, 6)]
        else:
            cnxt_paths = []
            for scale in (4, 5, 6):
                in_pq = os.path.join(CNXT_DIR, f"scale_{scale}x{scale}.parquet")
                out_pq = os.path.join(var_dir, f"cnxt_scale_{scale}x{scale}.parquet")
                cmd = [
                    sys.executable, os.path.join(EXP16, "scripts",
                                                  "filter_low_veg_tiles.py"),
                    "--in-parquet", in_pq,
                    "--out-parquet", out_pq,
                    "--scale", str(scale),
                    "--veg-frac-parquet", VEG_PARQUET,
                    "--veg-threshold", str(thr),
                ]
                run(cmd)
                cnxt_paths.append(out_pq)

        # Step 3: ensemble_v2 with locally-aggregated ViT prefusion
        ensemble_pq = os.path.join(var_dir, f"{tag}_w0.77_ensemble.parquet")
        cmd = [
            sys.executable, os.path.join(EXP14, "scripts", "ensemble_v2.py"),
            "--vit-prefusion-parquet", vit_pq,
            "--convnext-scales"] + cnxt_paths + [
            "--species-map", SPECIES_MAP,
            "--class-mask-npy", GEO_MASK,
            "--out-dir", var_dir,
            "--temperature", "1.5",
            "--weights", "0.77",
            "--thresholds", "0.100",
            "--topk", "10",
            "--vit-controller", VIT_CTRL,
            "--cnxt-controller", CNXT_CTRL,
            "--vit-aggregator", "max",
            "--cnxt-aggregator", "max",
            "--vit-aggregator-per-source", '{"RNNB":"sim_w"}',
            "--cnxt-aggregator-per-source", '{"RNNB":"sim_w"}',
            "--sim-json", SIM_JSON,
            "--density", DENSITY_PARQUET,
            "--tag", tag,
            "--save-ensemble-parquet",
        ]
        run(cmd, cwd=EXP14)

        # Step 4: apply N3_r0.5 admission for the count-bound source uplift
        for adm_tag, n_floor, ratio in [("plain", 0, 1000.0), ("N3r0.5", 3, 0.5)]:
            csv_out = os.path.join(var_dir, f"{tag}_{adm_tag}.csv")
            cmd = [
                sys.executable, os.path.join(EXP16, "scripts",
                                              "apply_n_floor_admission.py"),
                "--ensemble-parquet", ensemble_pq,
                "--species-map", SPECIES_MAP,
                "--geo-mask", GEO_MASK,
                "--habitat-mask", HABITAT_MASK,
                "--biogeo-npz", BIOGEO_NPZ,
                "--out-csv", csv_out,
                "--threshold", "0.100",
                "--topk", "10",
                "--n-floor", str(n_floor),
                "--ratio-floor", str(ratio),
            ]
            run(cmd, cwd=EXP14)


if __name__ == "__main__":
    main()
