"""Phase 5 — Ensemble student + teacher predictions.

Blends student and teacher probability vectors with a configurable alpha:
    ensemble_probs = alpha * teacher + (1 - alpha) * student

Sweeps multiple alpha values and generates a submission CSV for each.
Reuses generate_submission.py for species mapping, geo masking, and top-K.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from phase2.generate_submission import (
    f16_bytes_to_f32_vector,
    format_species_list,
    load_class_mask_npy,
    load_species_map,
)
from utils.io_bytes import f16_vector_to_bytes


NUM_CLASSES = 7806


def load_probs_parquet(
    path: str,
    id_col: str,
    probs_col: str,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, np.ndarray]:
    """Load a parquet with f16-encoded probability vectors → {image_name: f32[7806]}."""
    t = pq.read_table(path, columns=[id_col, probs_col]).to_pandas()
    probs = {}
    for _, row in t.iterrows():
        name = str(row[id_col])
        vec = f16_bytes_to_f32_vector(row[probs_col], num_classes)
        probs[name] = vec
    print(f"Loaded {len(probs)} images from {path} (col={probs_col})")
    return probs


def ensemble_blend(
    teacher_probs: Dict[str, np.ndarray],
    student_probs: Dict[str, np.ndarray],
    alpha: float,
    geo_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Blend: ensemble = alpha * teacher + (1 - alpha) * student."""
    common = set(teacher_probs.keys()) & set(student_probs.keys())
    teacher_only = set(teacher_probs.keys()) - common
    student_only = set(student_probs.keys()) - common

    if teacher_only:
        print(f"  WARNING: {len(teacher_only)} images in teacher only (using teacher probs)")
    if student_only:
        print(f"  WARNING: {len(student_only)} images in student only (using student probs)")

    result = {}

    for name in common:
        blended = alpha * teacher_probs[name] + (1.0 - alpha) * student_probs[name]
        if geo_mask is not None:
            blended[~geo_mask] = 0.0
        result[name] = blended

    # Include non-overlapping images (fallback to single source)
    for name in teacher_only:
        p = teacher_probs[name].copy()
        if geo_mask is not None:
            p[~geo_mask] = 0.0
        result[name] = p

    for name in student_only:
        p = student_probs[name].copy()
        if geo_mask is not None:
            p[~geo_mask] = 0.0
        result[name] = p

    return result


def write_ensemble_csv(
    image_probs: Dict[str, np.ndarray],
    out_csv: str,
    species_map: List[int],
    topk: int = 10,
    threshold: float = 0.03,
    min_k: int = 1,
) -> int:
    """Write submission CSV from ensemble probs."""
    rows = []
    for name, probs in image_probs.items():
        qid = os.path.splitext(name)[0]

        if topk > 0:
            k = min(topk, probs.shape[0])
            idx = np.argpartition(probs, -k)[-k:]
            idx = idx[np.argsort(-probs[idx])]
            kept = [int(i) for i in idx if probs[i] >= threshold]
        else:
            kept = [int(i) for i in np.where(probs >= threshold)[0]]

        if len(kept) < min_k:
            k2 = min(min_k, probs.shape[0])
            idx2 = np.argpartition(probs, -k2)[-k2:]
            idx2 = idx2[np.argsort(-probs[idx2])]
            kept = [int(i) for i in idx2]

        kept_species = [species_map[i] for i in kept]
        rows.append((qid, format_species_list(kept_species)))

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, s in rows:
            w.writerow([qid, s])

    print(f"  Wrote {len(rows)} rows to {out_csv}")
    return len(rows)


def save_ensemble_parquet(
    image_probs: Dict[str, np.ndarray],
    out_path: str,
    id_col: str = "image_name",
    probs_col: str = "ensemble_image_probs_f16",
) -> None:
    """Save blended probs as parquet (for downstream analysis)."""
    names = []
    probs_bytes = []
    for name, probs in image_probs.items():
        names.append(name)
        probs_bytes.append(f16_vector_to_bytes(probs))

    table = pa.table({id_col: names, probs_col: probs_bytes})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  Saved {table.num_rows} images to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble student + teacher predictions")

    ap.add_argument("--teacher-parquet", required=True, help="Teacher aggregated probs parquet")
    ap.add_argument("--student-parquet", required=True, help="Student probs parquet")
    ap.add_argument("--out-dir", required=True, help="Output directory for ensemble results")

    ap.add_argument("--teacher-id-col", default="image_name")
    ap.add_argument("--teacher-probs-col", default="teacher_image_probs_f16")
    ap.add_argument("--student-id-col", default="image_name")
    ap.add_argument("--student-probs-col", default="student_image_probs_f16")

    ap.add_argument(
        "--alphas", default="0.3,0.5,0.7,0.9",
        help="Comma-separated alpha values for teacher weight (default: 0.3,0.5,0.7,0.9)",
    )
    ap.add_argument("--species-map", required=True, help="Class-to-species JSON/CSV")
    ap.add_argument("--geo-mask-npy", default=None, help="Geo species mask .npy")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--min-k", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--save-parquet", action="store_true", help="Also save ensemble probs parquet")

    args = ap.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    teacher_probs = load_probs_parquet(
        args.teacher_parquet, args.teacher_id_col, args.teacher_probs_col, args.num_classes,
    )
    student_probs = load_probs_parquet(
        args.student_parquet, args.student_id_col, args.student_probs_col, args.num_classes,
    )

    # Load geo mask
    geo_mask = None
    if args.geo_mask_npy:
        geo_mask = load_class_mask_npy(args.geo_mask_npy, args.num_classes)
        print(f"Geo mask: {geo_mask.sum()} / {len(geo_mask)} species allowed")

    # Load species map
    species_map = load_species_map(args.species_map, args.num_classes)

    # Sweep alphas
    print(f"\nSweeping {len(alphas)} alpha values: {alphas}")
    print(f"{'='*60}")

    for alpha in alphas:
        print(f"\nalpha={alpha:.2f} (teacher={alpha:.0%}, student={1-alpha:.0%})")

        blended = ensemble_blend(teacher_probs, student_probs, alpha, geo_mask)

        csv_path = os.path.join(
            args.out_dir,
            f"ensemble_alpha{alpha:.2f}_top{args.topk}_thr{args.threshold:.2f}.csv",
        )
        write_ensemble_csv(
            blended, csv_path, species_map,
            topk=args.topk, threshold=args.threshold, min_k=args.min_k,
        )

        if args.save_parquet:
            pq_path = os.path.join(args.out_dir, f"ensemble_alpha{alpha:.2f}.parquet")
            save_ensemble_parquet(blended, pq_path)

    print(f"\n{'='*60}")
    print(f"Done. {len(alphas)} ensemble submissions in {args.out_dir}")


if __name__ == "__main__":
    main()
