"""Phase-3 proxy metric extension: low_density_diff_concentration.

Of the rows that changed in the candidate submission CSV vs the Exp-14
0.45572 winner, what fraction have density < 0.15? The mechanism predicts
photometric TTA changes should concentrate on low-density (brownish,
low-contrast) images, NOT scatter uniformly across alpine plots.

Required to advance the candidate to LB:
    low_density_diff_concentration >= 0.45

Usage:
    python diagnostics/proxy_metrics_photo.py \
        --candidate-csv outputs/stage1_vit_photo_tta/v_blend_beta0.65_w0.77_thr0.100_top10.csv \
        --winner-csv 'C:/Users/alper/Desktop/exp14_adaptive_fusion/v_winner_habitat_w0.77_thr0.100_top10.csv' \
        --density-parquet ../Exp-14_Adaptive_Temporal_Fusion/outputs/signals/density.parquet \
        --out-json outputs/stage1_vit_photo_tta/proxy_photo.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

EXP15_ROOT = Path(__file__).resolve().parents[1]
EXP14_ROOT = EXP15_ROOT.parent / "Exp-14_Adaptive_Temporal_Fusion"
sys.path.insert(0, str(EXP14_ROOT))
from signals import parse_source  # noqa: E402


def load_csv(path: str) -> Dict[str, set]:
    df = pd.read_csv(path)
    out: Dict[str, set] = {}
    for _, r in df.iterrows():
        try:
            ids = ast.literal_eval(str(r["species_ids"]))
            out[str(r["quadrat_id"])] = set(int(x) for x in ids)
        except Exception:
            out[str(r["quadrat_id"])] = set()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-csv", required=True)
    ap.add_argument("--winner-csv", required=True,
                    help="Exp-14 v_winner_habitat CSV (LB 0.45572).")
    ap.add_argument("--density-parquet", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--low-density-threshold", type=float, default=0.15)
    args = ap.parse_args()

    cand = load_csv(args.candidate_csv)
    winner = load_csv(args.winner_csv)
    dens = pd.read_parquet(args.density_parquet).set_index("quadrat_id")["density"].to_dict()

    common = sorted(set(cand) & set(winner))
    diff_qids = [q for q in common if cand[q] != winner[q]]
    n_diff = len(diff_qids)
    n_low = sum(1 for q in diff_qids if dens.get(q, 1.0) < args.low_density_threshold)
    concentration = float(n_low / n_diff) if n_diff else 0.0

    # Distribution by source (for diagnostic context)
    from collections import Counter
    diff_by_source = Counter(parse_source(q) for q in diff_qids)
    diff_by_band = Counter(
        ("low" if dens.get(q, 1.0) < args.low_density_threshold else "high")
        for q in diff_qids
    )

    out = {
        "n_total_quadrats": len(common),
        "n_rows_diff": n_diff,
        "n_rows_diff_low_density": n_low,
        "low_density_diff_concentration": concentration,
        "low_density_threshold": args.low_density_threshold,
        "diff_by_source": dict(diff_by_source),
        "diff_by_density_band": dict(diff_by_band),
        "gate_pass": concentration >= 0.45,
    }
    print(json.dumps(out, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out_json}")
    if not out["gate_pass"]:
        print("  GATE FAILED: changes are not concentrating on low-density images")
        print("  -> photometric TTA may not be addressing the hypothesized error mode")


if __name__ == "__main__":
    main()
