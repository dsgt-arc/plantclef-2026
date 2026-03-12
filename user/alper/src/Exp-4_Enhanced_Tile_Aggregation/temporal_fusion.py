"""Step 7: Temporal fusion — refine per-quadrat predictions using
cross-temporal location context.

75.8% of test quadrats (1596/2105) come from locations photographed
multiple times (up to 26 visits across 2013-2020). Predictions from
all visits at a location inform what species exist there.

Insertion point: AFTER multiview_aggregate (parquet with image_probs_f16)
and BEFORE generate_submission (CSV).

Strategies:
  location_mask  — P = P_Q * 1[P_loc > tau_loc]       (precision boost)
  cond_boost     — P = P_Q + beta * P_loc * 1[P_Q>tau] (safe recall boost)
  prior_blend    — P = alpha * P_Q + (1-alpha) * P_loc  (recall boost)
  blend_mask     — prior_blend then location_mask        (balanced)
  boost_mask     — cond_boost then location_mask         (conservative)
"""
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from aggregate_tiles import save_aggregated
from generate_submission import (
    f16_bytes_to_f32_vector,
    generate_submission_csv,
    load_class_mask_npy,
    load_species_map,
)

NUM_CLASSES = 7806


# ---------------------------------------------------------------------------
# Location parsing
# ---------------------------------------------------------------------------

def parse_location(quadrat_id: str) -> str:
    """Strip trailing 8-digit date from quadrat_id to get physical location.

    'CBN-PdlC-E3-20130723' -> 'CBN-PdlC-E3'
    """
    parts = quadrat_id.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[0]
    return quadrat_id


def group_by_location(
    quadrat_ids: List[str],
) -> Dict[str, List[str]]:
    """Group quadrat IDs by physical location.

    Returns {location: [qid1, qid2, ...]} sorted by location.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for qid in quadrat_ids:
        loc = parse_location(qid)
        groups[loc].append(qid)
    return dict(groups)


# ---------------------------------------------------------------------------
# Load + profile
# ---------------------------------------------------------------------------

def load_quadrat_probs(
    parquet_path: str,
    id_col: str = "image_name",
    probs_col: str = "image_probs_f16",
    num_classes: int = NUM_CLASSES,
) -> Dict[str, np.ndarray]:
    """Load parquet -> {quadrat_id: [C] f32 probability vector}.

    quadrat_id is derived from id_col by stripping the file extension.
    """
    t = pq.read_table(parquet_path, columns=[id_col, probs_col]).to_pandas()
    probs = {}
    for _, row in t.iterrows():
        qid = os.path.splitext(str(row[id_col]))[0]
        probs[qid] = f16_bytes_to_f32_vector(row[probs_col], num_classes)
    return probs


def build_location_profiles(
    quadrat_probs: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    num_classes: int = NUM_CLASSES,
) -> Dict[str, np.ndarray]:
    """Build P_loc(s) = element-wise max over all visits at each location.

    Returns {location: [C] f32 vector}.
    """
    profiles = {}
    for loc, qids in location_groups.items():
        vecs = [quadrat_probs[q] for q in qids if q in quadrat_probs]
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)  # [N_visits, C]
        profiles[loc] = np.max(stacked, axis=0)  # [C]
    return profiles


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

def apply_location_mask(
    p_q: np.ndarray,
    p_loc: np.ndarray,
    tau_loc: float,
) -> np.ndarray:
    """Strategy C: suppress species never confidently seen at this location."""
    return p_q * (p_loc > tau_loc).astype(p_q.dtype)


def apply_cond_boost(
    p_q: np.ndarray,
    p_loc: np.ndarray,
    beta: float,
    tau: float,
) -> np.ndarray:
    """Strategy B: boost species the model already somewhat detects."""
    boosted = p_q + beta * p_loc * (p_q > tau).astype(p_q.dtype)
    return np.clip(boosted, 0.0, 1.0)


def apply_prior_blend(
    p_q: np.ndarray,
    p_loc: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Strategy A: blend quadrat predictions with location prior."""
    return alpha * p_q + (1.0 - alpha) * p_loc


# ---------------------------------------------------------------------------
# Apply strategy to all quadrats
# ---------------------------------------------------------------------------

def fuse_all(
    quadrat_probs: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    location_profiles: Dict[str, np.ndarray],
    strategy: str,
    params: dict,
) -> Dict[str, np.ndarray]:
    """Apply a fusion strategy to all quadrats. Returns refined probs dict.

    Single-visit quadrats are passed through unchanged.
    """
    refined = {}
    loc_for_qid = {}
    for loc, qids in location_groups.items():
        for qid in qids:
            loc_for_qid[qid] = loc

    for qid, p_q in quadrat_probs.items():
        loc = loc_for_qid.get(qid)
        p_loc = location_profiles.get(loc) if loc else None

        # Single-visit or missing profile: pass through
        if p_loc is None or len(location_groups.get(loc, [])) < 2:
            refined[qid] = p_q.copy()
            continue

        if strategy == "location_mask":
            refined[qid] = apply_location_mask(p_q, p_loc, params["tau_loc"])
        elif strategy == "cond_boost":
            refined[qid] = apply_cond_boost(
                p_q, p_loc, params["beta"], params["tau"]
            )
        elif strategy == "prior_blend":
            refined[qid] = apply_prior_blend(p_q, p_loc, params["alpha"])
        elif strategy == "boost_mask":
            p = apply_cond_boost(p_q, p_loc, params["beta"], params["tau"])
            refined[qid] = apply_location_mask(p, p_loc, params["tau_loc"])
        elif strategy == "blend_mask":
            p = apply_prior_blend(p_q, p_loc, params["alpha"])
            refined[qid] = apply_location_mask(p, p_loc, params["tau_loc"])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return refined


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(
    original: Dict[str, np.ndarray],
    refined: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    threshold: float = 0.03,
) -> None:
    """Print diagnostic stats comparing original vs refined predictions."""
    multi_qids = [
        qid for loc, qids in location_groups.items()
        if len(qids) >= 2 for qid in qids
    ]
    single_qids = [
        qid for loc, qids in location_groups.items()
        if len(qids) < 2 for qid in qids
    ]

    # Check single-visit identity
    single_changed = 0
    for qid in single_qids:
        if qid in original and qid in refined:
            if not np.array_equal(original[qid], refined[qid]):
                single_changed += 1

    species_added = []
    species_removed = []
    prob_increases = []

    for qid in multi_qids:
        if qid not in original or qid not in refined:
            continue
        orig = original[qid]
        ref = refined[qid]
        orig_set = set(np.where(orig >= threshold)[0])
        ref_set = set(np.where(ref >= threshold)[0])
        species_added.append(len(ref_set - orig_set))
        species_removed.append(len(orig_set - ref_set))
        diff = ref - orig
        pos_mask = diff > 0
        if pos_mask.any():
            prob_increases.append(diff[pos_mask].mean())

    print(f"  Multi-visit quadrats: {len(multi_qids)}")
    print(f"  Single-visit quadrats: {len(single_qids)} "
          f"(changed: {single_changed})")
    if species_added:
        print(f"  Avg species added per quadrat (thr={threshold}): "
              f"{np.mean(species_added):.2f}")
    if species_removed:
        print(f"  Avg species removed per quadrat (thr={threshold}): "
              f"{np.mean(species_removed):.2f}")
    if prob_increases:
        print(f"  Avg prob increase (boosted species): "
              f"+{np.mean(prob_increases):.4f}")


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

def get_sweep_configs(tier: str = "all") -> List[Tuple[str, str, dict]]:
    """Return list of (tag, strategy_name, params) tuples.

    tier: 'tier1', 'tier2', or 'all'
    """
    tier1 = [
        ("C_mask_tau0.005", "location_mask", {"tau_loc": 0.005}),
        ("C_mask_tau0.01", "location_mask", {"tau_loc": 0.01}),
        ("B_boost_b0.2_tau0.005", "cond_boost", {"beta": 0.2, "tau": 0.005}),
        ("BC_boost_b0.2_tau0.005_mask0.01", "boost_mask",
         {"beta": 0.2, "tau": 0.005, "tau_loc": 0.01}),
    ]
    tier2 = [
        ("A_blend_a0.9", "prior_blend", {"alpha": 0.9}),
        ("A_blend_a0.8", "prior_blend", {"alpha": 0.8}),
        ("B_boost_b0.1_tau0.005", "cond_boost", {"beta": 0.1, "tau": 0.005}),
        ("B_boost_b0.3_tau0.005", "cond_boost", {"beta": 0.3, "tau": 0.005}),
        ("AC_blend_a0.9_mask0.01", "blend_mask",
         {"alpha": 0.9, "tau_loc": 0.01}),
    ]

    if tier == "tier1":
        return tier1
    elif tier == "tier2":
        return tier2
    return tier1 + tier2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Temporal fusion: refine predictions using location context"
    )
    ap.add_argument(
        "--in-parquet", required=True,
        help="Aggregated parquet from multiview_aggregate (best config)",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--species-map", required=True)
    ap.add_argument("--class-mask-npy", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--id-col", default="image_name")
    ap.add_argument("--probs-col", default="image_probs_f16")
    ap.add_argument(
        "--tier", default="all", choices=["tier1", "tier2", "all"],
        help="Which sweep tier to run (default: all)",
    )

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load data once ---
    print(f"Loading quadrat probabilities from {args.in_parquet}...")
    t0 = time.time()
    quadrat_probs = load_quadrat_probs(
        args.in_parquet, args.id_col, args.probs_col, args.num_classes
    )
    print(f"Loaded {len(quadrat_probs)} quadrats in {time.time() - t0:.1f}s")

    # --- Group by location ---
    location_groups = group_by_location(list(quadrat_probs.keys()))
    multi = {k: v for k, v in location_groups.items() if len(v) >= 2}
    single = {k: v for k, v in location_groups.items() if len(v) < 2}
    multi_q = sum(len(v) for v in multi.values())
    print(f"\nLocation grouping:")
    print(f"  Unique locations: {len(location_groups)}")
    print(f"  Multi-visit: {len(multi)} locations ({multi_q} quadrats)")
    print(f"  Single-visit: {len(single)} locations ({len(single)} quadrats)")

    # --- Build location profiles ---
    print("\nBuilding location profiles (element-wise max)...")
    location_profiles = build_location_profiles(
        quadrat_probs, location_groups, args.num_classes
    )
    avg_nonzero = np.mean([
        np.count_nonzero(p > 0.01) for p in location_profiles.values()
    ])
    print(f"  Avg species per location (>0.01): {avg_nonzero:.1f}")

    # --- Load submission helpers ---
    species_map = load_species_map(args.species_map, args.num_classes)
    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    # --- Generate baseline CSV (no fusion) ---
    baseline_csv = os.path.join(
        args.out_dir, f"baseline_no_fusion_thr{args.threshold:.2f}_top{args.topk}.csv"
    )
    baseline_parquet = os.path.join(args.out_dir, "baseline_no_fusion.parquet")
    save_aggregated(
        {qid: p.astype(np.float16) for qid, p in quadrat_probs.items()},
        baseline_parquet,
        id_col=args.id_col,
    )
    generate_submission_csv(
        in_parquet=baseline_parquet,
        out_csv=baseline_csv,
        species_map=species_map,
        num_classes=args.num_classes,
        class_mask=class_mask,
        topk=args.topk,
        threshold=args.threshold,
        id_col=args.id_col,
    )

    # --- Run sweep ---
    configs = get_sweep_configs(args.tier)
    print(f"\n{'='*80}")
    print(f"TEMPORAL FUSION SWEEP: {len(configs)} configs + baseline")
    print(f"{'='*80}")

    results_summary = []

    for tag, strategy, params in configs:
        print(f"\n{'='*70}")
        print(f"Config: {tag}")
        print(f"  Strategy: {strategy}  Params: {params}")
        print(f"{'='*70}")

        t0 = time.time()
        refined = fuse_all(
            quadrat_probs, location_groups, location_profiles,
            strategy, params,
        )
        elapsed = time.time() - t0
        print(f"  Fused {len(refined)} quadrats in {elapsed:.2f}s")

        # Diagnostics
        print_diagnostics(
            quadrat_probs, refined, location_groups, args.threshold
        )

        # Verification: bounds
        all_vals = np.concatenate([p for p in refined.values()])
        assert all_vals.min() >= -1e-6, f"Negative prob: {all_vals.min()}"
        assert all_vals.max() <= 1.0 + 1e-6, f"Prob > 1: {all_vals.max()}"

        # Save parquet
        parquet_path = os.path.join(args.out_dir, f"tf_{tag}.parquet")
        refined_f16 = {
            qid: p.astype(np.float16) for qid, p in refined.items()
        }
        save_aggregated(refined_f16, parquet_path, id_col=args.id_col)

        # Generate CSV
        csv_path = os.path.join(
            args.out_dir,
            f"tf_{tag}_thr{args.threshold:.2f}_top{args.topk}.csv",
        )
        n_rows = generate_submission_csv(
            in_parquet=parquet_path,
            out_csv=csv_path,
            species_map=species_map,
            num_classes=args.num_classes,
            class_mask=class_mask,
            topk=args.topk,
            threshold=args.threshold,
            id_col=args.id_col,
        )
        results_summary.append((tag, strategy, params, n_rows, csv_path))

    # --- Summary ---
    print(f"\n{'='*90}")
    print("TEMPORAL FUSION SWEEP SUMMARY")
    print(f"{'='*90}")
    print(f"Input: {args.in_parquet}")
    print(f"Baseline (no fusion): {baseline_csv}")
    print(f"Threshold: {args.threshold}  TopK: {args.topk}")
    print()
    fmt = "{:<45} {:>10} {:>8}  {}"
    print(fmt.format("Config", "Strategy", "Images", "CSV"))
    print(fmt.format("-" * 45, "-" * 10, "-" * 8, "-" * 40))
    for tag, strategy, params, n, path in results_summary:
        print(fmt.format(tag, strategy, str(n), os.path.basename(path)))
    print(f"{'='*90}")
    print(f"\nTotal CSVs generated: {len(results_summary) + 1} "
          f"(1 baseline + {len(results_summary)} fused)")
    print(f"Output directory: {args.out_dir}")
    print("Submit to Kaggle and compare against baseline F1=0.31487")


if __name__ == "__main__":
    main()
