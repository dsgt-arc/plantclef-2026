"""Render per-quadrat diagnostic pages to a multi-page PDF.

For each requested quadrat_id, one PDF page shows:
  - the raw image, downscaled
  - P_photo top-10 (pre-fusion), with species names if a map is given
  - P_loc top-10 (current max-aggregated prior over all sister visits)
  - P_final top-10 (alpha=ALPHA blend), color-coded by photo / prior / both
  - thumbnails of sister visits at the same location

Reads the same pre-fusion parquet used by Phase 0.2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

EXP8 = os.path.join(os.path.dirname(__file__), "..", "..",
                    "Exp-8_Quantile_Aggregation_SS")
sys.path.insert(0, os.path.abspath(EXP8))
from generate_submission import f16_bytes_to_f32_vector  # noqa: E402

EXP14 = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(EXP14))
from signals import parse_source  # noqa: E402

NUM_CLASSES = 7806
TOPK = 10


def parse_location(qid: str) -> str:
    parts = qid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[0]
    return qid


def load_probs_table(parquet: str) -> Dict[str, np.ndarray]:
    df = pq.read_table(parquet, columns=["image_name", "image_probs_f16"]).to_pandas()
    out = {}
    for _, r in df.iterrows():
        qid = os.path.splitext(str(r["image_name"]))[0]
        out[qid] = f16_bytes_to_f32_vector(r["image_probs_f16"], NUM_CLASSES)
    return out


def species_label(species_map: Optional[List[int]],
                  species_names: Optional[Dict[int, str]],
                  class_idx: int) -> str:
    if species_map is None:
        return f"id={class_idx}"
    sid = species_map[class_idx]
    if species_names and sid in species_names:
        return f"{sid}: {species_names[sid][:32]}"
    return str(sid)


def topk_bar(ax, p: np.ndarray, title: str, color: str,
             species_map, species_names, k: int = TOPK):
    idx = np.argsort(p)[::-1][:k]
    vals = p[idx]
    labels = [species_label(species_map, species_names, int(i)) for i in idx]
    y = np.arange(k)
    ax.barh(y, vals[::-1], color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlim(0, max(0.001, float(vals.max()) * 1.1))
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelsize=7)


def render_page(pdf, qid, image_path, p_photo, p_loc, p_final,
                sibling_imgs: List[str], sibling_probs: List[np.ndarray],
                species_map, species_names, alpha_used: float):
    src = parse_source(qid)
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0])

    # main image
    ax_img = fig.add_subplot(gs[0, 0])
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((800, 800))
        ax_img.imshow(img)
    ax_img.set_title(f"{qid}\nsource={src}, alpha={alpha_used:.2f}", fontsize=9)
    ax_img.axis("off")

    # three top-10 panels
    ax_photo = fig.add_subplot(gs[0, 1])
    topk_bar(ax_photo, p_photo, "P_photo (pre-fusion)", "#3a86ff",
             species_map, species_names)

    ax_loc = fig.add_subplot(gs[0, 2])
    topk_bar(ax_loc, p_loc, "P_loc (max prior)", "#ff006e",
             species_map, species_names)

    ax_final = fig.add_subplot(gs[0, 3])
    topk_bar(ax_final, p_final, f"P_final (alpha={alpha_used:.2f})",
             "#8338ec", species_map, species_names)

    # sister visits thumbnails
    n_sib = min(4, len(sibling_imgs))
    for i in range(n_sib):
        ax_s = fig.add_subplot(gs[1, i])
        if sibling_imgs[i] and os.path.exists(sibling_imgs[i]):
            img = Image.open(sibling_imgs[i]).convert("RGB")
            img.thumbnail((400, 400))
            ax_s.imshow(img)
        sib_id = os.path.splitext(os.path.basename(sibling_imgs[i] or ""))[0]
        # quick top-3
        if i < len(sibling_probs):
            top3 = np.argsort(sibling_probs[i])[::-1][:3]
            line = " | ".join(species_label(species_map, species_names, int(t))
                              for t in top3)
        else:
            line = ""
        ax_s.set_title(f"sibling {sib_id}\n{line}", fontsize=7)
        ax_s.axis("off")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefusion-parquet", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--quadrat-ids", nargs="+", required=True,
                    help="One or more quadrat_ids to render.")
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--alpha", type=float, default=0.15,
                    help="Static alpha for P_final visualization (Exp-11B baseline = 0.15 ViT).")
    ap.add_argument("--species-map", default=None,
                    help="JSON list mapping class_idx -> species_id.")
    ap.add_argument("--species-names", default=None,
                    help="JSON dict species_id -> Latin name.")
    args = ap.parse_args()

    print(f"Loading {args.prefusion_parquet} ...")
    probs = load_probs_table(args.prefusion_parquet)
    print(f"  {len(probs)} quadrats loaded")

    species_map = None
    if args.species_map:
        with open(args.species_map, encoding="utf-8") as fh:
            species_map = json.load(fh)
    species_names = None
    if args.species_names:
        with open(args.species_names, encoding="utf-8") as fh:
            d = json.load(fh)
            species_names = {int(k): v for k, v in d.items()}

    # group by location
    from collections import defaultdict
    by_loc = defaultdict(list)
    for q in probs:
        by_loc[parse_location(q)].append(q)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pdf)) or ".", exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        for qid in args.quadrat_ids:
            if qid not in probs:
                print(f"  ! {qid} not in parquet; skipping")
                continue
            loc = parse_location(qid)
            siblings = [s for s in by_loc.get(loc, []) if s != qid]
            sib_probs = [probs[s] for s in siblings]
            if siblings:
                p_loc = np.stack([probs[s] for s in siblings] + [probs[qid]],
                                 axis=0).max(axis=0)
            else:
                p_loc = probs[qid].copy()
            p_photo = probs[qid]
            p_final = args.alpha * p_photo + (1.0 - args.alpha) * p_loc

            sib_paths = [os.path.join(args.images_dir, f"{s}.jpg") for s in siblings]
            render_page(
                pdf, qid,
                os.path.join(args.images_dir, f"{qid}.jpg"),
                p_photo, p_loc, p_final,
                sib_paths, sib_probs,
                species_map, species_names, args.alpha,
            )
            print(f"  rendered {qid}  loc={loc}  siblings={len(siblings)}")
    print(f"  wrote {args.out_pdf}")


if __name__ == "__main__":
    main()
