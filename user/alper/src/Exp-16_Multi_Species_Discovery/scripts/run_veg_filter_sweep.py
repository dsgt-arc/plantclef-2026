"""Exp-16 veg-filter sweep — orchestrator.

For each veg_threshold in the sweep:
  1. Filter ConvNeXt scale_{4,5,6} parquets to drop tiles with veg_frac < threshold
  2. Run ensemble_v2 with the filtered ConvNeXt + production ViT prefusion
  3. Apply geo+aquatic+biogeo masks + N3_r0.5 admission
  4. Output candidate CSV + proxies vs current best (Stage 13 winner CSV)

Note: veg_threshold=0.0 reproduces the current best (no tiles dropped),
which is byte-equivalent to N3_r0.5 winner.
"""
from __future__ import annotations

import argparse
import filecmp
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

PRODUCTION_PREFUSION = os.path.join(PACE, "knn_bM0.85_bS1.00_T1.50_prefusion.parquet")
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
N3R05_BASELINE_CSV = os.path.join(
    EXP16, "outputs", "n_floor_sweep", "N3_r0.5.csv")


def run(cmd, cwd=None):
    print(f"\n  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veg-thresholds", default="0.02,0.05,0.10",
                    help="Comma-separated thresholds; tiles with veg_frac < thr are dropped.")
    ap.add_argument("--out-root", default=os.path.join(EXP16, "outputs", "veg_filter_sweep"))
    args = ap.parse_args()

    if not os.path.isfile(VEG_PARQUET):
        raise SystemExit(f"veg_frac parquet missing: {VEG_PARQUET}")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    thresholds = [float(t) for t in args.veg_thresholds.split(",")]
    print(f"  thresholds: {thresholds}")
    print(f"  out-root:   {args.out_root}")

    for thr in thresholds:
        tag = f"veg{thr:.2f}"
        var_dir = os.path.join(args.out_root, tag)
        Path(var_dir).mkdir(parents=True, exist_ok=True)

        filtered_paths = []
        for scale in (4, 5, 6):
            in_pq = os.path.join(CNXT_DIR, f"scale_{scale}x{scale}.parquet")
            out_pq = os.path.join(var_dir, f"scale_{scale}x{scale}.parquet")
            cmd = [
                sys.executable,
                os.path.join(EXP16, "scripts", "filter_low_veg_tiles.py"),
                "--in-parquet", in_pq,
                "--out-parquet", out_pq,
                "--scale", str(scale),
                "--veg-threshold", str(thr),
            ]
            run(cmd)
            filtered_paths.append(out_pq)

        ensemble_pq = os.path.join(var_dir, f"{tag}_w0.77_ensemble.parquet")
        cmd = [
            sys.executable, os.path.join(EXP14, "scripts", "ensemble_v2.py"),
            "--vit-prefusion-parquet", PRODUCTION_PREFUSION,
            "--convnext-scales"] + filtered_paths + [
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

        for adm_tag, n_floor, ratio in [("plain", 0, 1000.0), ("N3r0.5", 3, 0.5)]:
            csv_out = os.path.join(var_dir, f"{tag}_{adm_tag}.csv")
            cmd = [
                sys.executable,
                os.path.join(EXP16, "scripts", "apply_n_floor_admission.py"),
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

            if os.path.isfile(N3R05_BASELINE_CSV) and adm_tag == "N3r0.5":
                same = filecmp.cmp(csv_out, N3R05_BASELINE_CSV, shallow=False)
                print(f"\n  [{tag} {adm_tag}] byte-equivalent to N3_r0.5 baseline: {same}")


if __name__ == "__main__":
    main()
