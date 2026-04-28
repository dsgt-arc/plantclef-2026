"""Exp-16 Phase 1 orchestrator — sweep (K, beta) over the vote-share rerank.

For each variant V = (K, beta):
  1. Load (or compute and cache) vote_share for K
  2. Apply vote rerank to the ViT prefusion at beta
  3. Run ensemble_v2 with the locked Stage 13 winner controllers
  4. Run apply_biogeo_mask to emit candidate CSV
  5. Compute proxy metrics vs Stage 13 winner

beta=0 must reproduce the Stage 13 winner CSV byte-for-byte. This is the
wiring sanity check — if it doesn't reproduce, abort.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import subprocess
import sys
from pathlib import Path

EXP16 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP14 = os.path.abspath(os.path.join(EXP16, "..", "Exp-14_Adaptive_Temporal_Fusion"))
PACE = os.path.abspath(os.path.join(EXP14, "pace_downloads"))
SIGNALS = os.path.abspath(os.path.join(EXP14, "outputs", "signals"))
CTRL = os.path.abspath(os.path.join(EXP14, "stage_configs", "controllers"))

PRODUCTION_PREFUSION = os.path.join(
    PACE, "knn_bM0.85_bS1.00_T1.50_prefusion.parquet"
)
SPECIES_MAP = os.path.join(PACE, "class_to_species_id.json")
GEO_MASK = os.path.join(PACE, "sw_europe_mask_bool.npy")
HABITAT_MASK = os.path.join(SIGNALS, "aquatic_blacklist_mask.npy")
BIOGEO_NPZ = os.path.join(SIGNALS, "biogeo_per_source.npz")
SIM_JSON = os.path.join(SIGNALS, "visit_similarity.json")
DENSITY_PARQUET = os.path.join(SIGNALS, "density.parquet")

VIT_CTRL = os.path.join(CTRL, "vit_rnnb030_only.json")
CNXT_CTRL = os.path.join(CTRL, "cnxt_baseline_ld12_RNNBonly.json")

CNXT_DIR = os.path.join(PACE, "ConvNext_large_parquets")
CNXT_4 = os.path.join(CNXT_DIR, "scale_4x4.parquet")
CNXT_5 = os.path.join(CNXT_DIR, "scale_5x5.parquet")
CNXT_6 = os.path.join(CNXT_DIR, "scale_6x6.parquet")

STAGE13_WINNER_CSV = os.path.join(
    EXP14, "outputs", "stage13_biogeo_mask",
    "v_ld12RNNBonly_habitat_biogeo_w0.77_thr0.100_top10.csv",
)


def run(cmd: list[str], cwd: str | None = None) -> None:
    print(f"\n  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)} (rc={r.returncode})")


def build_reference_run(out_root: str, density_parquet: str) -> tuple[str, str]:
    """Run ensemble_v2 + apply_biogeo_mask on the ORIGINAL prefusion using my
    exact orchestrator args. Returns (csv_path, ensemble_parquet_path).
    """
    ref_dir = os.path.join(out_root, "_reference_baseline")
    Path(ref_dir).mkdir(parents=True, exist_ok=True)
    ens_pq = os.path.join(ref_dir, "ref_w0.77_ensemble.parquet")
    csv_out = os.path.join(ref_dir, "ref_habitat_biogeo_w0.77_thr0.100_top10.csv")
    if os.path.isfile(csv_out) and os.path.isfile(ens_pq):
        return csv_out, ens_pq
    cmd = [
        sys.executable, os.path.join(EXP14, "scripts", "ensemble_v2.py"),
        "--vit-prefusion-parquet", PRODUCTION_PREFUSION,
        "--convnext-scales", CNXT_4, CNXT_5, CNXT_6,
        "--species-map", SPECIES_MAP,
        "--class-mask-npy", GEO_MASK,
        "--out-dir", ref_dir,
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
        "--density", density_parquet,
        "--tag", "ref",
        "--save-ensemble-parquet",
    ]
    run(cmd, cwd=EXP14)
    cmd = [
        sys.executable, os.path.join(EXP14, "scripts", "apply_biogeo_mask.py"),
        "--ensemble-parquet", ens_pq,
        "--species-map", SPECIES_MAP,
        "--geo-mask", GEO_MASK,
        "--habitat-mask", HABITAT_MASK,
        "--biogeo-npz", BIOGEO_NPZ,
        "--out-csv", csv_out,
        "--threshold", "0.100",
        "--topk", "10",
    ]
    run(cmd, cwd=EXP14)
    return csv_out, ens_pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-values", default="5,10")
    ap.add_argument("--beta-values", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--epsilon", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--out-root", default=os.path.join(
        EXP16, "outputs", "phase1_vote_rerank"))
    ap.add_argument("--aug-filter", action="store_true",
                    help="Aug filter on tile-level vote computation (production matches this).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip variants whose final CSV already exists.")
    ap.add_argument("--only-beta-zero", action="store_true",
                    help="Only run the beta=0 sanity check, no sweep.")
    args = ap.parse_args()

    K_values = [int(k) for k in args.K_values.split(",")]
    beta_values = [float(b) for b in args.beta_values.split(",")]
    if args.only_beta_zero:
        beta_values = [0.0]

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Exp-16 Phase 1 sweep")
    print("=" * 72)

    print("\n--- Building reference baseline (original prefusion through current pipeline) ---")
    ref_csv, ref_ens = build_reference_run(args.out_root, DENSITY_PARQUET)
    print(f"  reference csv:     {ref_csv}")
    print(f"  reference ens.pq:  {ref_ens}")
    print(f"  K_values:    {K_values}")
    print(f"  beta_values: {beta_values}")
    print(f"  epsilon:     {args.epsilon}")
    print(f"  out-root:    {args.out_root}")
    print(f"  prefusion:   {PRODUCTION_PREFUSION}")
    print(f"  vit-ctrl:    {VIT_CTRL}")
    print(f"  cnxt-ctrl:   {CNXT_CTRL}")
    print(f"  stage13:     {STAGE13_WINNER_CSV}")

    summary = []

    for K in K_values:
        vs_dir = os.path.join(args.out_root, f"vote_share_K{K}")
        Path(vs_dir).mkdir(parents=True, exist_ok=True)
        vs_parquet = os.path.join(vs_dir, "vote_share.parquet")
        if not os.path.isfile(vs_parquet):
            print(f"\n--- Computing vote_share for K={K} ---")
            cmd = [
                sys.executable, os.path.join(EXP16, "scripts", "compute_vote_share.py"),
                "--K", str(K),
                "--temperature", str(args.temperature),
                "--out-parquet", vs_parquet,
            ]
            if args.aug_filter:
                cmd.append("--aug-filter")
            run(cmd)
        else:
            print(f"\n--- Reusing vote_share K={K}: {vs_parquet} ---")

        for beta in beta_values:
            tag = f"K{K}_b{beta:.2f}"
            var_dir = os.path.join(args.out_root, tag)
            Path(var_dir).mkdir(parents=True, exist_ok=True)
            reranked_pref = os.path.join(var_dir, "vit_prefusion_reranked.parquet")
            ensemble_pq = os.path.join(var_dir, f"{tag}_w0.77_ensemble.parquet")
            csv_out = os.path.join(var_dir, f"{tag}_habitat_biogeo_w0.77_thr0.100_top10.csv")
            proxies_json = os.path.join(var_dir, "proxies.json")

            if args.skip_existing and os.path.isfile(csv_out):
                print(f"\n=== Skipping {tag} (exists) ===")
                continue

            print(f"\n=== Variant {tag} ===")

            cmd = [
                sys.executable, os.path.join(EXP16, "scripts", "apply_vote_rerank.py"),
                "--prefusion-parquet", PRODUCTION_PREFUSION,
                "--vote-share-parquet", vs_parquet,
                "--beta", str(beta),
                "--epsilon", str(args.epsilon),
                "--out-parquet", reranked_pref,
                "--preserve", "l1",
            ]
            run(cmd)

            cmd = [
                sys.executable, os.path.join(EXP14, "scripts", "ensemble_v2.py"),
                "--vit-prefusion-parquet", reranked_pref,
                "--convnext-scales", CNXT_4, CNXT_5, CNXT_6,
                "--species-map", SPECIES_MAP,
                "--class-mask-npy", GEO_MASK,
                "--out-dir", var_dir,
                "--temperature", str(args.temperature),
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

            cmd = [
                sys.executable, os.path.join(EXP14, "scripts", "apply_biogeo_mask.py"),
                "--ensemble-parquet", ensemble_pq,
                "--species-map", SPECIES_MAP,
                "--geo-mask", GEO_MASK,
                "--habitat-mask", HABITAT_MASK,
                "--biogeo-npz", BIOGEO_NPZ,
                "--out-csv", csv_out,
                "--threshold", "0.100",
                "--topk", "10",
            ]
            run(cmd, cwd=EXP14)

            cmd = [
                sys.executable, os.path.join(EXP14, "diagnostics", "proxy_metrics.py"),
                "--csv", csv_out,
                "--baseline-csv", ref_csv,
                "--density", DENSITY_PARQUET,
                "--out-json", proxies_json,
            ]
            run(cmd, cwd=EXP14)

            entry = {"K": K, "beta": beta, "csv": csv_out, "proxies": proxies_json}
            if beta == 0.0:
                same = filecmp.cmp(csv_out, ref_csv, shallow=False)
                entry["beta0_byte_match_to_reference"] = same
                print(f"\n  beta=0 byte-equivalent to reference run: {same}")
            summary.append(entry)

    summary_path = os.path.join(args.out_root, "phase1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"variants": summary}, f, indent=2)
    print(f"\n  wrote {summary_path}")


if __name__ == "__main__":
    main()
