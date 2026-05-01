"""Per-image alpha controller.

Computes alpha_i ∈ [alpha_min, alpha_max] per quadrat_id from a small
set of optional signals. Defaults that recover the Exp-11B baseline
exactly:

    lambda_c = lambda_d = lambda_s = 0
    alpha_src = {<source>: 0.15} for ViT branch  (or 0.30 for ConvNeXt)
    alpha_min = 0.05, alpha_max = 0.95

Formula:
    alpha_i = clip(
        alpha_src(source_i)
        * (1 + lambda_c * (conf_i  - 0.5))
        * (1 + lambda_d * (0.5     - dens_i))
        * (1 + lambda_s * (1.0     - season_max_i)),
        alpha_min, alpha_max,
    )

alpha is the *photo weight* in apply_prior_blend (alpha=1.0 ignores prior;
alpha=0.0 fully trusts prior).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Allow `from signals import parse_source` when imported as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signals import parse_source  # noqa: E402


@dataclass
class ControllerConfig:
    # Per-source baseline alpha (photo weight). Falls back to default_alpha.
    alpha_src: Dict[str, float] = field(default_factory=dict)
    default_alpha: float = 0.15
    # Gate strengths. 0.0 means "ignore this signal". Negative is allowed
    # if you want to invert (not recommended).
    lambda_c: float = 0.0
    lambda_d: float = 0.0
    lambda_s: float = 0.0
    # Per-source overrides for the gate strengths. If a source is in any of
    # these dicts, that value overrides the global lambda_* for that source.
    # Default: empty -> use global lambdas. Useful for restricting a gate
    # to specific sources where evidence supports it.
    lambda_c_src: Dict[str, float] = field(default_factory=dict)
    lambda_d_src: Dict[str, float] = field(default_factory=dict)
    lambda_s_src: Dict[str, float] = field(default_factory=dict)
    # Hard clip bounds.
    alpha_min: float = 0.05
    alpha_max: float = 0.95

    @classmethod
    def from_json(cls, path: str) -> "ControllerConfig":
        with open(path) as fh:
            return cls(**json.load(fh))

    def to_dict(self) -> dict:
        return {
            "alpha_src": dict(self.alpha_src),
            "default_alpha": self.default_alpha,
            "lambda_c": self.lambda_c,
            "lambda_d": self.lambda_d,
            "lambda_s": self.lambda_s,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
        }


def compute_alpha_map(
    quadrat_ids,
    config: ControllerConfig,
    *,
    confidences: Optional[Dict[str, float]] = None,
    densities: Optional[Dict[str, float]] = None,
    season_max: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return {qid: alpha_i} for every qid in quadrat_ids.

    Missing signals default to neutral (0.5 for centered scores, 0.0 for
    season_max). Result equals scalar config.default_alpha when all
    lambdas are 0, regardless of signal availability.
    """
    out: Dict[str, float] = {}
    for qid in quadrat_ids:
        src = parse_source(qid)
        a = config.alpha_src.get(src, config.default_alpha)

        lam_c = config.lambda_c_src.get(src, config.lambda_c)
        lam_d = config.lambda_d_src.get(src, config.lambda_d)
        lam_s = config.lambda_s_src.get(src, config.lambda_s)

        if lam_c != 0.0 and confidences is not None:
            c = confidences.get(qid, 0.5)
            a = a * (1.0 + lam_c * (c - 0.5))
        if lam_d != 0.0 and densities is not None:
            d = densities.get(qid, 0.5)
            a = a * (1.0 + lam_d * (0.5 - d))
        if lam_s != 0.0 and season_max is not None:
            sm = season_max.get(qid, 1.0)
            a = a * (1.0 + lam_s * (1.0 - sm))

        out[qid] = float(np.clip(a, config.alpha_min, config.alpha_max))
    return out


def baseline_config(branch: str) -> ControllerConfig:
    """Return a config that reproduces Exp-11B for the named branch."""
    if branch == "vit":
        return ControllerConfig(default_alpha=0.15)
    if branch == "convnext":
        return ControllerConfig(default_alpha=0.30)
    raise ValueError(f"Unknown branch: {branch}")
