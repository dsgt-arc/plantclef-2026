"""Prior-aggregation alternatives for temporal fusion.

The Exp-11B baseline uses element-wise max across visits at the same
location (Exp-8/temporal_fusion.py:116-132). This module exposes
alternative aggregators that can be selected via a single string flag.

For "sim_w" (visual-similarity-weighted) the prior is per-(location,
current_image) rather than per-location, so this module returns a
callable rather than a dict for those modes.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

PriorMap = Union[Dict[str, np.ndarray], Callable[[str], np.ndarray]]


def aggregate_prior(
    quadrat_probs: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    mode: str = "max",
    *,
    sim_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    season_weights: Optional[Dict[str, Dict[str, float]]] = None,
    confidences: Optional[Dict[str, float]] = None,
    num_classes: int = 7806,
) -> PriorMap:
    """Build the location prior under one of several aggregation modes.

    Returns either a Dict[location, P_loc] (location-only modes) or a
    callable f(qid) -> P_loc (per-image modes).

    mode:
        max           — element-wise max across visits (baseline)
        mean          — element-wise mean
        top2_mean     — element-wise mean of two largest per class
        conf_w        — confidence-weighted mean over visits
        sim_w         — per-image similarity-weighted (callable returned)
        season_w      — per-image season-weighted (callable returned)
        sim_season_w  — combined similarity * season weighting (callable)
    """
    if mode in ("max", "mean", "top2_mean"):
        return _aggregate_static(quadrat_probs, location_groups, mode)

    if mode == "conf_w":
        if confidences is None:
            raise ValueError("conf_w aggregator requires confidences")
        return _aggregate_conf_w(quadrat_probs, location_groups, confidences)

    if mode in ("sim_w", "season_w", "sim_season_w"):
        return _make_per_image_aggregator(
            quadrat_probs, location_groups, mode,
            sim_matrix=sim_matrix, season_weights=season_weights,
        )

    raise ValueError(f"Unknown aggregator mode: {mode}")


def aggregate_prior_per_source(
    quadrat_probs: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    source_to_mode: Dict[str, str],
    parse_source_fn,
    *,
    default_mode: str = "max",
    sim_matrix=None,
    season_weights=None,
    confidences=None,
    num_classes: int = 7806,
) -> Callable[[str], np.ndarray]:
    """Per-source prior aggregator.

    For each unique mode in source_to_mode (and the default), pre-compute
    the prior structure once. Return a callable f(qid) that dispatches
    to the right structure based on parse_source_fn(qid).
    """
    needed_modes = set(source_to_mode.values()) | {default_mode}
    built: Dict[str, PriorMap] = {}
    for m in needed_modes:
        built[m] = aggregate_prior(
            quadrat_probs, location_groups, m,
            sim_matrix=sim_matrix, season_weights=season_weights,
            confidences=confidences, num_classes=num_classes,
        )

    qid_to_loc: Dict[str, str] = {}
    for loc, qids in location_groups.items():
        for q in qids:
            qid_to_loc[q] = loc

    def f(qid: str):
        src = parse_source_fn(qid)
        m = source_to_mode.get(src, default_mode)
        prior = built[m]
        if callable(prior):
            return prior(qid)
        return prior.get(qid_to_loc.get(qid))

    return f


def _aggregate_static(
    quadrat_probs: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    mode: str,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for loc, qids in location_groups.items():
        vecs = [quadrat_probs[q] for q in qids if q in quadrat_probs]
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)
        if mode == "max":
            out[loc] = stacked.max(axis=0)
        elif mode == "mean":
            out[loc] = stacked.mean(axis=0)
        elif mode == "top2_mean":
            n = stacked.shape[0]
            if n == 1:
                out[loc] = stacked[0]
            else:
                # top-2 per class along axis 0 -> mean
                part = np.partition(stacked, n - 2, axis=0)[-2:]
                out[loc] = part.mean(axis=0)
    return out


def _aggregate_conf_w(
    quadrat_probs: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    confidences: Dict[str, float],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for loc, qids in location_groups.items():
        vecs, ws = [], []
        for q in qids:
            if q in quadrat_probs:
                vecs.append(quadrat_probs[q])
                ws.append(max(confidences.get(q, 0.0), 1e-6))
        if not vecs:
            continue
        stacked = np.stack(vecs, axis=0)
        w = np.asarray(ws, dtype=np.float32).reshape(-1, 1)
        out[loc] = (stacked * w).sum(axis=0) / w.sum()
    return out


def _make_per_image_aggregator(
    quadrat_probs: Dict[str, np.ndarray],
    location_groups: Dict[str, List[str]],
    mode: str,
    *,
    sim_matrix: Optional[Dict[str, Dict[str, float]]],
    season_weights: Optional[Dict[str, Dict[str, float]]],
) -> Callable[[str], np.ndarray]:
    """Build a callable f(qid)->P_loc with weights specific to that qid.

    Weights for each historical visit v of qid i:
        sim_w        : sim[i][v]
        season_w     : season_weights[i][v]
        sim_season_w : sim[i][v] * season_weights[i][v]

    A visit's weight always includes itself with weight=1.0 (self-similarity)
    so the aggregator never returns an empty prior. The blend keeps the
    current-image's own probabilities as part of the prior — this is fine
    because the controller still down-weights the prior overall.
    """
    qid_to_loc: Dict[str, str] = {}
    for loc, qids in location_groups.items():
        for q in qids:
            qid_to_loc[q] = loc

    def f(qid: str) -> Optional[np.ndarray]:
        loc = qid_to_loc.get(qid)
        if loc is None:
            return None
        siblings = location_groups.get(loc, [])
        siblings = [s for s in siblings if s in quadrat_probs]
        if not siblings:
            return None
        ws = []
        for s in siblings:
            if mode == "sim_w":
                w = (sim_matrix or {}).get(qid, {}).get(s, 1.0 if s == qid else 0.0)
            elif mode == "season_w":
                w = (season_weights or {}).get(qid, {}).get(s, 1.0 if s == qid else 0.0)
            else:  # sim_season_w
                ws_sim = (sim_matrix or {}).get(qid, {}).get(s, 1.0 if s == qid else 0.0)
                ws_season = (season_weights or {}).get(qid, {}).get(s, 1.0 if s == qid else 0.0)
                w = ws_sim * ws_season
            ws.append(max(w, 1e-6))
        stacked = np.stack([quadrat_probs[s] for s in siblings], axis=0)
        w_arr = np.asarray(ws, dtype=np.float32).reshape(-1, 1)
        return (stacked * w_arr).sum(axis=0) / w_arr.sum()

    return f
