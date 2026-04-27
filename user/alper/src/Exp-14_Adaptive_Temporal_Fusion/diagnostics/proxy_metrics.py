"""Internal proxy metrics for variant scoring without LB submission.

Each metric is computed on a candidate submission CSV (same schema as
ens_knn_w0.77_thr0.100_top10.csv: columns = quadrat_id, species_ids).

Metrics:
  rnnb_pair_divergence — mean (1 - jaccard(top10_a, top10_b)) across all
                          paired RNNB sites. Higher is better; baseline ~0.0-0.1.
  stable_site_invariance — mean Jaccard(top10) within each CBN-PdlC
                           site cluster. Higher = stable (good); the
                           baseline value of THIS metric is what we
                           compare against (we want delta close to 0).
  density_count_correlation — Spearman rho between vegetation density
                              and #species predicted per image. Optional
                              (requires density.parquet from
                              compute_density.py).
  photo_faithfulness — mean fraction of final top-10 species that are
                       also in P_photo top-20 (requires pre-fusion parquet).

The script can be run against a single CSV or two CSVs (baseline vs
candidate) to print deltas. Writes a JSON report.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EXP14 = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(EXP14))
from signals import parse_source  # noqa: E402

EXP8 = os.path.join(os.path.dirname(__file__), "..", "..",
                    "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, os.path.abspath(EXP8))
from generate_submission import f16_bytes_to_f32_vector  # noqa: E402

NUM_CLASSES = 7806


def parse_location(qid: str) -> str:
    parts = qid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[0]
    return qid


def parse_site(loc: str) -> str:
    parts = loc.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else loc


def load_csv(path: str) -> Dict[str, List[int]]:
    df = pd.read_csv(path)
    out: Dict[str, List[int]] = {}
    for _, r in df.iterrows():
        qid = str(r["quadrat_id"])
        try:
            ids = ast.literal_eval(str(r["species_ids"]))
            out[qid] = [int(x) for x in ids]
        except Exception:
            out[qid] = []
    return out


def jaccard(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def metric_rnnb_pair_divergence(preds: Dict[str, List[int]]) -> Dict[str, float]:
    by_loc = defaultdict(list)
    for q in preds:
        if parse_source(q) == "RNNB":
            by_loc[parse_location(q)].append(q)
    diffs, n = [], 0
    for _loc, members in by_loc.items():
        if len(members) < 2:
            continue
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                diffs.append(1.0 - jaccard(preds[members[i]], preds[members[j]]))
                n += 1
    return {"value": float(np.mean(diffs)) if diffs else 0.0, "n_pairs": n}


def metric_stable_site_invariance(preds: Dict[str, List[int]]) -> Dict[str, float]:
    by_site = defaultdict(list)
    for q in preds:
        if parse_source(q) == "CBN-PdlC":
            by_site[parse_site(parse_location(q))].append(q)
    sims, n = [], 0
    for _site, members in by_site.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                sims.append(jaccard(preds[members[i]], preds[members[j]]))
                n += 1
    return {"value": float(np.mean(sims)) if sims else 0.0, "n_pairs": n}


def metric_density_count_corr(preds: Dict[str, List[int]],
                              density_path: Optional[str]) -> Dict[str, float]:
    if not density_path or not os.path.exists(density_path):
        return {"value": float("nan"), "n": 0, "note": "density.parquet missing"}
    d = pd.read_parquet(density_path).set_index("quadrat_id")["density"].to_dict()
    rows = []
    for q, ids in preds.items():
        if q in d:
            rows.append((d[q], len(ids)))
    if len(rows) < 10:
        return {"value": float("nan"), "n": len(rows)}
    x = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    # Spearman = Pearson on ranks
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return {"value": rho, "n": len(rows)}


def metric_photo_faithfulness(preds: Dict[str, List[int]],
                              prefusion_parquet: Optional[str],
                              species_map: Optional[List[int]]) -> Dict[str, float]:
    if not prefusion_parquet or not os.path.exists(prefusion_parquet):
        return {"value": float("nan"), "n": 0, "note": "prefusion parquet missing"}
    if species_map is None:
        return {"value": float("nan"), "n": 0, "note": "species_map missing"}
    import pyarrow.parquet as pq
    df = pq.read_table(prefusion_parquet,
                       columns=["image_name", "image_probs_f16"]).to_pandas()
    sid_for_idx = {i: int(species_map[i]) for i in range(len(species_map))}
    fracs, n = [], 0
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        if qid not in preds:
            continue
        p = f16_bytes_to_f32_vector(r["image_probs_f16"], NUM_CLASSES)
        photo_top20 = {sid_for_idx[i] for i in np.argsort(p)[::-1][:20].tolist()}
        final_set = set(preds[qid])
        if not final_set:
            continue
        fracs.append(len(final_set & photo_top20) / len(final_set))
        n += 1
    return {"value": float(np.mean(fracs)) if fracs else 0.0, "n": n}


def per_source_count_distribution(preds: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    by_src = defaultdict(list)
    for q, ids in preds.items():
        by_src[parse_source(q)].append(len(ids))
    return {
        src: {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "n": len(v),
        }
        for src, v in by_src.items()
    }


def compute_all(csv_path: str,
                density_path: Optional[str] = None,
                prefusion_parquet: Optional[str] = None,
                species_map_path: Optional[str] = None) -> Dict:
    preds = load_csv(csv_path)
    print(f"  loaded {len(preds)} predictions from {csv_path}")
    sm = json.load(open(species_map_path)) if species_map_path else None
    return {
        "csv": csv_path,
        "n_predictions": len(preds),
        "rnnb_pair_divergence": metric_rnnb_pair_divergence(preds),
        "stable_site_invariance": metric_stable_site_invariance(preds),
        "density_count_correlation": metric_density_count_corr(preds, density_path),
        "photo_faithfulness": metric_photo_faithfulness(
            preds, prefusion_parquet, sm),
        "per_source_count_distribution": per_source_count_distribution(preds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--baseline-csv", default=None,
                    help="Optional second CSV for delta reporting.")
    ap.add_argument("--density", default=None)
    ap.add_argument("--prefusion-parquet", default=None)
    ap.add_argument("--species-map", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    candidate = compute_all(args.csv, args.density, args.prefusion_parquet,
                            args.species_map)
    report = {"candidate": candidate}
    print("\n=== Candidate ===")
    for k, v in candidate.items():
        if k != "per_source_count_distribution":
            print(f"  {k}: {v}")
    print("  per_source_count_distribution:")
    for src, st in candidate["per_source_count_distribution"].items():
        print(f"    {src:10s}: mean={st['mean']:.2f}  median={st['median']:.1f}  n={st['n']}")

    if args.baseline_csv:
        baseline = compute_all(args.baseline_csv, args.density,
                               args.prefusion_parquet, args.species_map)
        report["baseline"] = baseline
        print("\n=== Delta (candidate - baseline) ===")
        for k in ("rnnb_pair_divergence", "stable_site_invariance",
                  "density_count_correlation", "photo_faithfulness"):
            d = candidate[k]["value"] - baseline[k]["value"]
            print(f"  delta {k}: {d:+.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".",
                    exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
