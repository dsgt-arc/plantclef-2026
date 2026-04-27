"""Run one stage of the Exp-14 sweep.

A stage spec is a JSON file:
{
  "stage_name": "stage1_vit_aggregators",
  "axis": "vit_aggregator",
  "variants": [
    {"tag": "max",       "args": {"vit_aggregator": "max"}},
    {"tag": "mean",      "args": {"vit_aggregator": "mean"}},
    {"tag": "top2_mean", "args": {"vit_aggregator": "top2_mean"}},
    {"tag": "sim_w",     "args": {"vit_aggregator": "sim_w"}}
  ],
  "common_args": {
    "vit_controller": "configs/vit_baseline.json",
    "cnxt_controller": "configs/cnxt_baseline.json",
    "cnxt_aggregator": "max"
  }
}

For each variant, run ensemble_v2.py, then proxy_metrics.py on the
output CSV. Write a summary CSV with one row per variant. Winner
selection is left to the user — proxies are reported, not auto-picked.

This script does NOT submit to Kaggle. Inspect the summary CSV first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP14 = os.path.dirname(SCRIPT_DIR)


def run_variant(variant: dict, common_args: dict, base_args: argparse.Namespace,
                out_dir: str) -> dict:
    tag = variant["tag"]
    var_args = dict(common_args)
    var_args.update(variant.get("args", {}))

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "ensemble_v2.py")]
    cmd += ["--vit-prefusion-parquet", base_args.vit_prefusion_parquet]
    cmd += ["--convnext-scales"] + base_args.convnext_scales
    cmd += ["--species-map", base_args.species_map]
    if base_args.class_mask_npy:
        cmd += ["--class-mask-npy", base_args.class_mask_npy]
    cmd += ["--out-dir", out_dir]
    cmd += ["--temperature", str(base_args.temperature)]
    cmd += ["--weights", base_args.weights]
    cmd += ["--thresholds", base_args.thresholds]
    cmd += ["--topk", str(base_args.topk)]
    cmd += ["--tag", f"v_{tag}"]
    if base_args.density:
        cmd += ["--density", base_args.density]
    if base_args.sim_json:
        cmd += ["--sim-json", base_args.sim_json]
    if base_args.season_json:
        cmd += ["--season-json", base_args.season_json]
    if base_args.vit_confidence:
        cmd += ["--vit-confidence", base_args.vit_confidence]
    if base_args.cnxt_confidence:
        cmd += ["--cnxt-confidence", base_args.cnxt_confidence]

    for k, v in var_args.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]

    print("\n----- variant:", tag, "-----")
    print("  $", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  ensemble_v2 done in {time.time() - t0:.1f}s")

    # find the produced CSV (single weight x threshold)
    w = base_args.weights.split(",")[0]
    thr = base_args.thresholds.split(",")[0]
    csv = os.path.join(
        out_dir,
        f"v_{tag}_w{float(w):.2f}_thr{float(thr):.3f}_top{base_args.topk}.csv",
    )
    proxy_cmd = [sys.executable,
                 os.path.join(EXP14, "diagnostics", "proxy_metrics.py"),
                 "--csv", csv,
                 "--out-json", csv.replace(".csv", "_proxies.json")]
    if base_args.baseline_csv:
        proxy_cmd += ["--baseline-csv", base_args.baseline_csv]
    if base_args.density:
        proxy_cmd += ["--density", base_args.density]
    if base_args.vit_prefusion_parquet:
        proxy_cmd += ["--prefusion-parquet", base_args.vit_prefusion_parquet]
    if base_args.species_map:
        proxy_cmd += ["--species-map", base_args.species_map]

    print("  $", " ".join(proxy_cmd))
    subprocess.run(proxy_cmd, check=True)
    with open(csv.replace(".csv", "_proxies.json")) as fh:
        rep = json.load(fh)
    cand = rep["candidate"]
    return {
        "tag": tag,
        "csv": csv,
        "rnnb_pair_divergence": cand["rnnb_pair_divergence"]["value"],
        "stable_site_invariance": cand["stable_site_invariance"]["value"],
        "density_count_corr": cand["density_count_correlation"]["value"],
        "photo_faithfulness": cand["photo_faithfulness"]["value"],
        "n_predictions": cand["n_predictions"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-spec", required=True, help="JSON stage spec.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vit-prefusion-parquet", required=True)
    ap.add_argument("--convnext-scales", nargs="+", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--weights", default="0.77")
    ap.add_argument("--thresholds", default="0.100")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--baseline-csv", default=None)
    ap.add_argument("--density", default=None)
    ap.add_argument("--sim-json", default=None)
    ap.add_argument("--season-json", default=None)
    ap.add_argument("--vit-confidence", default=None)
    ap.add_argument("--cnxt-confidence", default=None)
    args = ap.parse_args()

    spec = json.load(open(args.stage_spec))
    print(f"=== Stage: {spec['stage_name']}  axis: {spec['axis']} ===")
    common = spec.get("common_args", {})
    os.makedirs(args.out_dir, exist_ok=True)

    rows: List[dict] = []
    for v in spec["variants"]:
        try:
            rows.append(run_variant(v, common, args, args.out_dir))
        except subprocess.CalledProcessError as e:
            print(f"  ! variant {v['tag']} failed: {e}")
            rows.append({"tag": v["tag"], "error": str(e)})

    summary = os.path.join(args.out_dir, f"{spec['stage_name']}_summary.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f"\n=== Summary -> {summary} ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
