#!/usr/bin/env python3
import os
import csv
import json
import argparse
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cograph_apply import load_cograph_npz, graph_rerank, graph_mask_expand


def _try_parse_two_ints(line: str) -> Optional[Tuple[int, int]]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _ensure_full_class_map(class_to_species: Dict[int, int], num_classes: int, src: str) -> List[int]:
    missing = [i for i in range(num_classes) if i not in class_to_species]
    if missing:
        raise ValueError(
            f"{src} does not define all class indices. Missing {len(missing)} indices. First missing: {missing[:10]}"
        )
    extra = [k for k in class_to_species.keys() if k < 0 or k >= num_classes]
    if extra:
        raise ValueError(f"{src} contains out of range class_index values. Examples: {sorted(extra)[:10]}")
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

    if ext == ".csv":
        df = pd.read_csv(path)
        if "class_index" not in df.columns or "species_id" not in df.columns:
            raise ValueError("CSV species map must have columns class_index and species_id")
        df2 = df[["class_index", "species_id"]].copy()
        df2["class_index"] = df2["class_index"].astype(int)
        df2["species_id"] = df2["species_id"].astype(int)
        class_to_species = dict(zip(df2["class_index"].tolist(), df2["species_id"].tolist()))
        return _ensure_full_class_map(class_to_species, num_classes, src)

    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]

    one_int = True
    vals: List[int] = []
    for ln in raw_lines:
        try:
            vals.append(int(ln))
        except Exception:
            one_int = False
            break
    if one_int:
        if len(vals) != num_classes:
            raise ValueError(f"TXT species map lines {len(vals)} does not match num_classes {num_classes}")
        return [int(x) for x in vals]

    pairs: List[Tuple[int, int]] = []
    for ln in raw_lines:
        p = _try_parse_two_ints(ln)
        if p is None:
            raise ValueError(f"Unrecognized line in {path}: {ln}")
        pairs.append(p)

    class_to_species = {int(a): int(b) for a, b in pairs}
    return _ensure_full_class_map(class_to_species, num_classes, src)


def load_class_mask_npy(path: str, num_classes: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    m = np.load(path)
    if m.ndim != 1 or m.shape[0] != num_classes:
        raise ValueError(f"class mask must be shape [{num_classes}], got {m.shape}")
    if m.dtype != np.bool_:
        m = m.astype(np.bool_)
    if not m.any():
        raise ValueError("class mask has zero allowed classes")
    return m


def decode_probs(blob: bytes, num_classes: int) -> np.ndarray:
    v = np.frombuffer(blob, dtype=np.float16).astype(np.float32, copy=False)
    if v.shape[0] != num_classes:
        raise ValueError(f"teacher probs length {v.shape[0]} does not match num_classes {num_classes}")
    return v


def format_species_list(species_ids: List[int]) -> str:
    return "[" + ", ".join(str(s) for s in species_ids) + "]"


def pick_output_indices(p: np.ndarray, topk: int, threshold: float, min_k: int) -> List[int]:
    if topk > 0:
        k = min(topk, p.shape[0])
        idx = np.argpartition(p, -k)[-k:]
        idx = idx[np.argsort(p[idx])[::-1]]
        kept = [int(i) for i in idx.tolist() if float(p[i]) >= float(threshold)]
    else:
        kept = [int(i) for i in np.flatnonzero(p >= float(threshold)).tolist()]

    if len(kept) < int(min_k):
        k2 = min(int(min_k), p.shape[0])
        idx2 = np.argpartition(p, -k2)[-k2:]
        idx2 = idx2[np.argsort(p[idx2])[::-1]]
        kept = [int(i) for i in idx2.tolist()]

    return kept


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--out-csv", required=True)

    ap.add_argument("--num-classes", type=int, default=7806)
    ap.add_argument("--probs-col", default="teacher_image_probs_f16")

    ap.add_argument("--quadrat-col", default="quadrat_id")
    ap.add_argument("--image-name-col", default="image_name")

    ap.add_argument("--species-map", default=None)

    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--min-k", type=int, default=1)

    ap.add_argument("--class-mask-npy", default=None)

    ap.add_argument("--cograph-npz", default=None)
    ap.add_argument("--cograph-mode", choices=["none", "rerank", "mask", "inside"], default="none")
    ap.add_argument("--cograph-seed-k", type=int, default=8)
    ap.add_argument("--cograph-seed-alpha", type=float, default=0.0)
    ap.add_argument("--cograph-neigh-topn", type=int, default=50)
    ap.add_argument("--cograph-lambda", type=float, default=0.10)
    ap.add_argument("--cograph-weight-transform", choices=["log1p", "cap", "none"], default="log1p")
    ap.add_argument("--cograph-w-cap", type=float, default=5.0)

    args = ap.parse_args()

    species_map = load_species_map(args.species_map, args.num_classes)

    class_mask = None
    if args.class_mask_npy:
        class_mask = load_class_mask_npy(args.class_mask_npy, args.num_classes)

    g = None
    if args.cograph_mode != "none":
        if not args.cograph_npz:
            raise ValueError("cograph_mode is not none but cograph_npz was not provided")
        g = load_cograph_npz(args.cograph_npz, num_classes=args.num_classes)

    table = pq.read_table(args.in_parquet)
    df = table.to_pandas()

    if args.quadrat_col in df.columns:
        quad = df[args.quadrat_col].astype(str).tolist()
    elif args.image_name_col in df.columns:
        names = df[args.image_name_col].astype(str).tolist()
        quad = [n[:-4] if n.lower().endswith(".jpg") else n for n in names]
    else:
        raise ValueError(f"Need either {args.quadrat_col} or {args.image_name_col} in parquet")

    if args.probs_col not in df.columns:
        raise ValueError(f"Missing probs_col {args.probs_col} in parquet")

    out_rows = []

    for qid, blob in zip(quad, df[args.probs_col].tolist()):
        p = decode_probs(blob, args.num_classes)

        if class_mask is not None:
            p = p.copy()
            p[~class_mask] = 0.0

        if args.cograph_mode in ["rerank", "inside"]:
            p = graph_rerank(
                p=p,
                g=g,
                seed_k=args.cograph_seed_k,
                neigh_topn=args.cograph_neigh_topn,
                lam=args.cograph_lambda,
                seed_alpha=args.cograph_seed_alpha,
                weight_transform=args.cograph_weight_transform,
                w_cap=args.cograph_w_cap,
            )
        elif args.cograph_mode == "mask":
            p = graph_mask_expand(
                p=p,
                g=g,
                seed_k=args.cograph_seed_k,
                neigh_topn=args.cograph_neigh_topn,
                seed_alpha=args.cograph_seed_alpha,
            )

        kept_idx = pick_output_indices(p, topk=args.topk, threshold=args.threshold, min_k=args.min_k)
        kept_species = [species_map[i] for i in kept_idx]
        out_rows.append({"quadrat_id": qid, "species_ids": format_species_list(kept_species)})

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame(out_rows, columns=["quadrat_id", "species_ids"])
    out_df.to_csv(args.out_csv, sep=",", index=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(out_df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()


'''
## Option 1 graph rerank
   
python plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_C_teacher_testing_co-graph/generate_test_submission_from_aggregation.py \
  --in-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_step2_agg/teacher_image_probs.parquet \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260303_teacher_cograph_rerank_lam010.csv \
  --probs-col teacher_image_probs_f16 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --topk 8 \
  --threshold 0.05 \
  --min-k 1 \
  --cograph-npz /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/graphs/lucas_212k_cograph_ecology_top8_alpha060_pmi_min100_top50.npz \
  --cograph-mode rerank \
  --cograph-seed-k 8 \
  --cograph-neigh-topn 50 \
  --cograph-lambda 0.10 \
  --cograph-weight-transform log1p
    

## Option 2 graph mask expansion

python plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_C_teacher_testing_co-graph/generate_test_submission_from_aggregation.py \
  --in-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_step2_agg/teacher_image_probs.parquet \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260303_teacher_cograph_mask_seed8_neigh50.csv \
  --probs-col teacher_image_probs_f16 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --topk 8 \
  --threshold 0.05 \
  --min-k 1 \
  --cograph-npz /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/graphs/lucas_212k_cograph_ecology_top8_alpha060_pmi_min100_top50.npz \
  --cograph-mode mask \
  --cograph-seed-k 8 \
  --cograph-neigh-topn 50



  ## Baaeline for co-graph submissions, without the co-graph

  python plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_C_teacher_testing_co-graph/generate_test_submission_from_aggregation.py \
  --in-parquet /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/teacher_step2_agg/teacher_image_probs.parquet \
  --out-csv /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/submissions/20260303_teacher_baseline_wo_co-graph_JPEG_comp.csv \
  --probs-col teacher_image_probs_f16 \
  --species-map /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/class_id_to_species_mapping/class_to_species_id.json \
  --topk 8 \
  --threshold 0.05 \
  --min-k 1 \
  --cograph-mode none

'''
