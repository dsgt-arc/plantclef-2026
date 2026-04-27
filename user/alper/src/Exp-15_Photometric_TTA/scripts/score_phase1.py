"""Score the Phase-1 probe outputs: compute the 6 proxy metrics per
photometric variant vs the `none` baseline, and decide survivor(s).

Inputs:
    --probe-parquet : outputs/phase1/probe_logits.parquet
    --probe-csv     : outputs/probe/probe_sample.csv (band metadata)
    --species-map   : pace_downloads/class_to_species_id.json
    --aquatic-list  : Exp-14_Adaptive_Temporal_Fusion/outputs/signals/aquatic_blacklist.json

Outputs:
    --out-json : outputs/phase1/scores.json
    Also prints survival decision to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

EXP15_ROOT = Path(__file__).resolve().parents[1]
EXP14_ROOT = EXP15_ROOT.parent / "Exp-14_Adaptive_Temporal_Fusion"
EXP9B_ROOT = EXP15_ROOT.parent / "Exp-9B_Improved_KNN_Retrieval"

NUM_CLASSES = 7806
POSIDONIA_SID = 1361703


def softmax_f16(buf: bytes, num_classes: int = NUM_CLASSES, T: float = 1.5) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.float16).astype(np.float32) / T
    arr = arr - arr.max()
    e = np.exp(arr)
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-parquet", required=True)
    ap.add_argument("--probe-csv", required=True)
    ap.add_argument("--species-map", required=True,
                    help="JSON list mapping class_idx -> species_id")
    ap.add_argument("--aquatic-list", required=True,
                    help="JSON from Exp-14 build_habitat_mask.py listing aquatic species")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--temperature", type=float, default=1.5)
    args = ap.parse_args()

    df = pd.read_parquet(args.probe_parquet)
    print(f"  loaded {len(df)} (qid, scale, tile, variant) rows")

    # Bands
    bands = pd.read_csv(args.probe_csv).set_index("quadrat_id")["band"].to_dict()
    df["band"] = df["quadrat_id"].map(bands).fillna("unknown")

    # Class lookup
    with open(args.species_map, encoding="utf-8") as f:
        class_to_sid = json.load(f)
    sid_to_class = {int(sid): idx for idx, sid in enumerate(class_to_sid)}
    posidonia_idx = sid_to_class.get(POSIDONIA_SID)
    if posidonia_idx is None:
        raise SystemExit(f"Posidonia sid {POSIDONIA_SID} not found in species map")

    with open(args.aquatic_list, encoding="utf-8") as f:
        aquatic = json.load(f)
    aquatic_class_idx = [int(b["class_idx"]) for b in aquatic]
    print(f"  aquatic blacklist: {len(aquatic_class_idx)} class indices")

    # Pivot: (qid, scale, tile_index) x variant
    # Compute softmax once per row
    print("  computing softmax(logits/T)...")
    df["softmax"] = df["teacher_logits_f16"].apply(
        lambda b: softmax_f16(b, NUM_CLASSES, args.temperature)
    )
    df["top1_idx"] = df["softmax"].apply(lambda p: int(np.argmax(p)))
    df["top1_conf"] = df["softmax"].apply(lambda p: float(p.max()))
    df["posidonia_p"] = df["softmax"].apply(lambda p: float(p[posidonia_idx]))
    df["aquatic_mass"] = df["softmax"].apply(
        lambda p: float(p[aquatic_class_idx].sum())
    )

    # Helpers
    variants = [v for v in df["variant"].unique() if v != "none"]
    base = df[df["variant"] == "none"].set_index(["quadrat_id", "scale", "tile_index"])

    out: Dict[str, Dict] = {"variants": {}}
    for v in sorted(variants):
        sub = df[df["variant"] == v].set_index(["quadrat_id", "scale", "tile_index"])
        common = base.index.intersection(sub.index)
        b = base.loc[common]
        s = sub.loc[common]
        n = len(common)

        # Proxy 1: mean_top1_conf_shift (per band)
        shifts = s["top1_conf"].values - b["top1_conf"].values
        bands_idx = b["band"].values
        per_band = {}
        for band in ("brownish", "lush", "very_barren", "user_seed", "unknown"):
            mask = bands_idx == band
            if mask.sum() > 0:
                per_band[band] = float(shifts[mask].mean())
            else:
                per_band[band] = None
        mean_top1_conf_shift = float(shifts.mean())

        # Proxy 2: top1_stability
        stability = float((s["top1_idx"].values == b["top1_idx"].values).mean())

        # Proxy 3: posidonia_logit_shift
        posidonia_shift = float(
            s["posidonia_p"].values.mean() - b["posidonia_p"].values.mean()
        )

        # Proxy 4: aquatic_mass_shift
        aquatic_shift = float(
            s["aquatic_mass"].values.mean() - b["aquatic_mass"].values.mean()
        )

        # Proxy 5: cohort_topk_jaccard (per-image, max-aggregated)
        # For each (qid, scale), pool tiles by max prob per class, take top-10
        def topk_set(group: pd.DataFrame, k: int = 10) -> set:
            pooled = np.stack(group["softmax"].values, axis=0).max(axis=0)
            return set(np.argsort(pooled)[::-1][:k].tolist())

        jaccards = []
        for (qid, scale), gb in b.reset_index().groupby(["quadrat_id", "scale"]):
            gs = s.reset_index()
            gs = gs[(gs["quadrat_id"] == qid) & (gs["scale"] == scale)]
            if len(gs) == 0:
                continue
            sb = topk_set(gb)
            ss = topk_set(gs)
            j = len(sb & ss) / max(len(sb | ss), 1)
            jaccards.append(j)
        cohort_topk_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

        # Proxy 6: brownish_vs_barren_conf_gap
        brownish_shift = per_band.get("brownish")
        barren_shift = per_band.get("very_barren")
        if brownish_shift is not None and barren_shift is not None:
            gap = brownish_shift - barren_shift
        else:
            gap = None

        # Gates
        gates = {
            "top1_stability_>=0.55": stability >= 0.55,
            "cohort_topk_jaccard_in_[0.55,0.85]": 0.55 <= cohort_topk_jaccard <= 0.85,
            "aquatic_mass_shift_<=0": aquatic_shift <= 0,
            "brownish_vs_barren_gap_>+0.005": (gap is not None) and (gap > 0.005),
            "any_signal":
                (per_band.get("brownish") or 0) > 0.01
                or (aquatic_shift < -0.005),
        }
        all_pass = all(gates.values())

        out["variants"][v] = {
            "n_tiles": n,
            "mean_top1_conf_shift": mean_top1_conf_shift,
            "mean_top1_conf_shift_per_band": per_band,
            "top1_stability": stability,
            "posidonia_logit_shift": posidonia_shift,
            "aquatic_mass_shift": aquatic_shift,
            "cohort_topk_jaccard": cohort_topk_jaccard,
            "brownish_vs_barren_gap": gap,
            "gates": gates,
            "passes_all_gates": all_pass,
        }

        print(f"\n--- variant: {v} (n={n}) ---")
        print(f"  mean_top1_conf_shift     = {mean_top1_conf_shift:+.4f}")
        for band, val in per_band.items():
            if val is not None:
                print(f"     {band:12s} = {val:+.4f}")
        print(f"  top1_stability           = {stability:.4f}  "
              f"{'(pass)' if gates['top1_stability_>=0.55'] else '(FAIL)'}")
        print(f"  cohort_topk_jaccard      = {cohort_topk_jaccard:.4f}  "
              f"{'(pass)' if gates['cohort_topk_jaccard_in_[0.55,0.85]'] else '(FAIL)'}")
        print(f"  posidonia_logit_shift    = {posidonia_shift:+.6f}")
        print(f"  aquatic_mass_shift       = {aquatic_shift:+.6f}  "
              f"{'(pass)' if gates['aquatic_mass_shift_<=0'] else '(FAIL)'}")
        if gap is not None:
            print(f"  brownish_vs_barren_gap   = {gap:+.4f}  "
                  f"{'(pass)' if gates['brownish_vs_barren_gap_>+0.005'] else '(FAIL)'}")
        print(f"  any_signal               = {'(pass)' if gates['any_signal'] else '(FAIL)'}")
        print(f"  -> {'PASS all gates' if all_pass else 'FAIL'}")

    # Pick survivors
    survivors = [v for v, d in out["variants"].items() if d["passes_all_gates"]]
    out["survivors"] = survivors
    if survivors:
        print(f"\n=== SURVIVORS: {survivors} ===")
        print("Advance these variant(s) to Phase 2 full re-extraction.")
    else:
        print("\n=== NO SURVIVORS — close Exp-15 with negative-result report. ===")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    # Strip non-serializable softmax arrays before save
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [_clean(x) for x in d]
        if isinstance(d, (np.floating, np.integer)):
            return float(d) if isinstance(d, np.floating) else int(d)
        return d
    with open(args.out_json, "w") as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\n  wrote {args.out_json}")


if __name__ == "__main__":
    main()
