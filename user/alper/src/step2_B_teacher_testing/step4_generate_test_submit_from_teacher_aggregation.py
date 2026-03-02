#!/usr/bin/env python3

import os
import csv
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _ensure_full_class_map(class_to_species: Dict[int, int], num_classes: int, src: str) -> List[int]:
    missing = [i for i in range(num_classes) if i not in class_to_species]
    if missing:
        raise ValueError(f"{src} missing {len(missing)} indices, first {missing[:10]}")
    return [int(class_to_species[i]) for i in range(num_classes)]


def load_species_map(path: Optional[str], num_classes: int) -> List[int]:
    if path is None:
        return list(range(num_classes))
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    src = f"species map {path}"

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list) or len(arr) != num_classes:
            raise ValueError(f"JSON species map must be a list of length {num_classes}")
        return [int(x) for x in arr]

    df = pd.read_csv(path)
    if "class_index" in df.columns and "species_id" in df.columns:
        class_to_species = dict(zip(df["class_index"].astype(int).tolist(), df["species_id"].astype(int).tolist()))
        return _ensure_full_class_map(class_to_species, num_classes, src)

    raise ValueError("species map must be json list or csv with class_index and species_id columns")


def load_class_mask_npy(path: str, num_classes: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = np.load(path)
    if m.ndim != 1 or m.shape[0] != num_classes:
        raise ValueError(f"class mask must be 1D length {num_classes}, got {m.shape}")
    if m.dtype != np.bool_:
        m = m.astype(np.bool_)
    if not m.any():
        raise ValueError("class mask has zero allowed classes")
    return m


def f16_bytes_to_f32_vector(buf: bytes, num_classes: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.float16)
    if arr.size != num_classes:
        raise ValueError(f"Expected {num_classes} f16 values, got {arr.size}")
    return arr.astype(np.float32, copy=False)


def format_species_list(species_ids: List[int]) -> str:
    return "[" + ", ".join(str(s) for s in species_ids) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in-parquet", required=True, help="Parquet containing quadrat_id and teacher_image_probs_f16")
    ap.add_argument("--out-csv", required=True)

    ap.add_argument("--id-col", default="quadrat_id")
    ap.add_argument("--probs-col", default="teacher_image_probs_f16")

    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--species-map", default=None)

    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--min-k", type=int, default=1)

    ap.add_argument("--class-mask-npy", default=None)

    args = ap.parse_args()

    t = pq.read_table(args.in_parquet, columns=[args.id_col, args.probs_col]).to_pandas()

    species_map = load_species_map(args.species_map, args.num_classes)

    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    rows: List[Tuple[str, str]] = []

    for _, r in t.iterrows():
        qid = str(r[args.id_col])
        p = f16_bytes_to_f32_vector(r[args.probs_col], args.num_classes)

        if class_mask is not None:
            p = p.copy()
            p[~class_mask] = 0.0

        if args.topk > 0:
            k = min(int(args.topk), p.shape[0])
            idx = np.argpartition(p, -k)[-k:]
            idx = idx[np.argsort(-p[idx])]
            kept = [int(i) for i in idx if p[i] >= float(args.threshold)]
        else:
            kept = [int(i) for i in np.where(p >= float(args.threshold))[0].tolist()]

        if len(kept) < int(args.min_k):
            k2 = min(int(args.min_k), p.shape[0])
            idx2 = np.argpartition(p, -k2)[-k2:]
            idx2 = idx2[np.argsort(-p[idx2])]
            kept = [int(i) for i in idx2.tolist()]

        kept_species = [species_map[i] for i in kept]
        rows.append((qid, format_species_list(kept_species)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["quadrat_id", "species_ids"])
        for qid, s in rows:
            w.writerow([qid, s])

    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


    
'''
## Submission without species mask

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step2_B_teacher_testing/step4_generate_test_submit_from_teacher_aggregation.py \
  --in-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_step2_agg/teacher_image_probs.parquet \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260301_teacher_step2_nomask.csv \
  --num-classes 7806 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --topk 8 \
  --threshold 0.05 \
  --min-k 1  
    

  ## Submission with species mask

  python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/step2_B_teacher_testing/step4_generate_test_submit_from_teacher_aggregation.py \
  --in-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_step2_agg/teacher_image_probs.parquet \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260301_teacher_step2_masked.csv \
  --num-classes 7806 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --class-mask-npy ps-clef2026_plantclef-0/species_masking/sw_europe_mask_bool.npy \
  --topk 8 \
  --threshold 0.05 \
  --min-k 1
    
'''