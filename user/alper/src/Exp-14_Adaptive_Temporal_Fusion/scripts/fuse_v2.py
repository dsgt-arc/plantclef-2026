"""fuse_v2 — adaptive temporal fusion using a controller + new aggregators.

Reusable function. Call from ensemble_v2.py per branch.

Inputs are paths to artifacts already produced upstream:
  pre-fusion parquet (image_name, image_probs_f16)
  optional density.parquet (quadrat_id, density)
  optional confidence.parquet (quadrat_id, c_max_norm, ...)
  optional sim.json (quadrat_id -> {sibling: cosine})
  optional season.json (quadrat_id -> {sibling: weight})

Output: dict[quadrat_id, np.float16 vector of length NUM_CLASSES].
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

EXP14 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP14)
EXP8 = os.path.join(EXP14, "..", "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, os.path.abspath(EXP8))

from temporal_fusion import (  # noqa: E402
    build_location_profiles, fuse_all, group_by_location, load_quadrat_probs,
)
from controller.alpha_controller import (  # noqa: E402
    ControllerConfig, compute_alpha_map,
)
from controller.prior_aggregators import (  # noqa: E402
    aggregate_prior, aggregate_prior_per_source,
)
from signals import parse_source  # noqa: E402

NUM_CLASSES = 7806


def _load_optional_dict_parquet(path: Optional[str], col: str) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_parquet(path)
    if col not in df.columns:
        return {}
    return dict(zip(df["quadrat_id"].astype(str), df[col].astype(float)))


def _load_optional_json(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fuse_branch(
    prefusion_parquet: str,
    *,
    controller: ControllerConfig,
    aggregator: str = "max",
    aggregator_per_source: Optional[Dict[str, str]] = None,
    density_parquet: Optional[str] = None,
    density_col: str = "density",
    confidence_parquet: Optional[str] = None,
    confidence_col: str = "c_top5_norm",
    sim_json: Optional[str] = None,
    season_json: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Run fusion for one branch (ViT or ConvNeXt) and return {qid: f16 probs}."""
    quadrat_probs = load_quadrat_probs(prefusion_parquet)
    print(f"  loaded {len(quadrat_probs)} quadrats from {os.path.basename(prefusion_parquet)}")

    location_groups = group_by_location(list(quadrat_probs.keys()))

    # signals
    densities = _load_optional_dict_parquet(density_parquet, density_col)
    confidences = _load_optional_dict_parquet(confidence_parquet, confidence_col)
    sim_matrix = _load_optional_json(sim_json)
    season_weights = _load_optional_json(season_json)

    # season_max per qid (for the controller's lambda_s gate)
    season_max: Dict[str, float] = {}
    if season_weights:
        for qid, row in season_weights.items():
            others = [v for k, v in row.items() if k != qid]
            season_max[qid] = max(others) if others else 1.0

    # alpha_map from controller
    alpha_map = compute_alpha_map(
        list(quadrat_probs.keys()), controller,
        confidences=confidences if confidences else None,
        densities=densities if densities else None,
        season_max=season_max if season_max else None,
    )
    a_vals = np.fromiter(alpha_map.values(), dtype=np.float32)
    print(f"  controller: alpha mean={a_vals.mean():.3f} "
          f"min={a_vals.min():.3f} max={a_vals.max():.3f} "
          f"std={a_vals.std():.3f}")

    # prior aggregator
    if aggregator_per_source:
        print(f"  per-source aggregator: {aggregator_per_source} "
              f"(default={aggregator})")
        location_profiles = aggregate_prior_per_source(
            quadrat_probs, location_groups,
            source_to_mode=aggregator_per_source,
            parse_source_fn=parse_source,
            default_mode=aggregator,
            sim_matrix=sim_matrix, season_weights=season_weights,
            confidences=confidences if confidences else None,
            num_classes=NUM_CLASSES,
        )
    elif aggregator == "max":
        location_profiles = build_location_profiles(
            quadrat_probs, location_groups, NUM_CLASSES)
    else:
        # aggregate_prior may return a dict OR a callable
        location_profiles = aggregate_prior(
            quadrat_probs, location_groups, mode=aggregator,
            sim_matrix=sim_matrix, season_weights=season_weights,
            confidences=confidences if confidences else None,
            num_classes=NUM_CLASSES,
        )

    # fuse with per-qid alpha. params["alpha"] acts as a fallback only.
    fused = fuse_all(
        quadrat_probs, location_groups, location_profiles,
        strategy="prior_blend",
        params={"alpha": controller.default_alpha},
        alpha_map=alpha_map,
    )
    return {k: v.astype(np.float16) for k, v in fused.items()}
