#!/usr/bin/env python

import os
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def build_student_ready(
    step1_dir: str,
    step2_dir: str,
    out_dir: str,
    num_tiles: int = 16,
    emb_dim: int = 768,
):
    emb_bytes_per_tile = emb_dim * 2
    os.makedirs(out_dir, exist_ok=True)

    print("Reading Step 2 table")
    t2 = pq.read_table(
        step2_dir,
        columns=[
            "image_name",
            "teacher_image_probs_f16",
            "latitude",
            "longitude",
            "altitude",
        ],
    )
    t2_df = t2.to_pandas()

    lat = t2_df["latitude"].to_numpy()
    lon = t2_df["longitude"].to_numpy()
    alt = t2_df["altitude"].to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)

    lat_mu, lat_sd = np.nanmean(lat[valid]), np.nanstd(lat[valid]) + 1e-6
    lon_mu, lon_sd = np.nanmean(lon[valid]), np.nanstd(lon[valid]) + 1e-6
    alt_mu, alt_sd = np.nanmean(alt[valid]), np.nanstd(alt[valid]) + 1e-6

    t2_map = {}
    for _, r in t2_df.iterrows():
        img = r["image_name"]
        blob = r["teacher_image_probs_f16"]
        la, lo, al = r["latitude"], r["longitude"], r["altitude"]

        has = float(np.isfinite(la) and np.isfinite(lo) and np.isfinite(al))

        if has == 1.0:
            meta = np.array(
                [
                    (la - lat_mu) / lat_sd,
                    (lo - lon_mu) / lon_sd,
                    (al - alt_mu) / alt_sd,
                    1.0,
                ],
                dtype=np.float32,
            )
        else:
            meta = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        t2_map[img] = (blob, meta)

    print("Step 2 rows:", len(t2_map))

    print("Reading Step 1 dataset")
    ds = pq.ParquetDataset(step1_dir)

    tiles = {}
    rows_read = 0

    for frag in ds.fragments:
        tbl = frag.to_table(
            columns=["image_name", "tile_index", "embedding_f16"]
        )
        df = tbl.to_pandas()
        rows_read += len(df)

        for img, ti, emb_blob in zip(
            df["image_name"], df["tile_index"], df["embedding_f16"]
        ):
            if img not in tiles:
                tiles[img] = [None] * num_tiles

            if 0 <= int(ti) < num_tiles:
                tiles[img][int(ti)] = emb_blob

    print("Step 1 tile rows read:", rows_read)
    print("Unique images in Step 1:", len(tiles))

    out_image = []
    out_tile_blob = []
    out_teacher_blob = []
    out_meta = []

    missing_t2 = 0
    bad_tiles = 0

    for img, tile_list in tiles.items():
        if img not in t2_map:
            missing_t2 += 1
            continue

        if any(x is None for x in tile_list):
            bad_tiles += 1
            continue

        packed = b"".join(tile_list)
        expected = num_tiles * emb_bytes_per_tile

        if len(packed) != expected:
            bad_tiles += 1
            continue

        teacher_blob, meta = t2_map[img]

        out_image.append(img)
        out_tile_blob.append(packed)
        out_teacher_blob.append(teacher_blob)
        out_meta.append(meta)

    print("Images kept:", len(out_image))
    print("Missing Step 2 rows:", missing_t2)
    print("Bad or incomplete tiles:", bad_tiles)

    meta_arr = np.stack(out_meta, axis=0)

    table_out = pa.Table.from_arrays(
        [
            pa.array(out_image, type=pa.string()),
            pa.array(out_tile_blob, type=pa.binary()),
            pa.array(out_teacher_blob, type=pa.binary()),
            pa.FixedSizeListArray.from_arrays(
                pa.array(meta_arr.reshape(-1), type=pa.float32()),
                4,
            ),
        ],
        names=[
            "image_name",
            "tile_embeddings_f16",
            "teacher_image_probs_f16",
            "metadata_tensor_f32",
        ],
    )

    out_path = os.path.join(out_dir, "data.parquet")
    pq.write_table(table_out, out_path, compression="zstd")

    print("Wrote:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1-dir", required=True)
    parser.add_argument("--step2-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-tiles", type=int, default=16)
    parser.add_argument("--emb-dim", type=int, default=768)

    args = parser.parse_args()

    build_student_ready(
        step1_dir=args.step1_dir,
        step2_dir=args.step2_dir,
        out_dir=args.out_dir,
        num_tiles=args.num_tiles,
        emb_dim=args.emb_dim,
    )



''' 
python build_student_ready_merged_data_step3a.py \
--step1-dir /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_10k/feature_bank \
--step2-dir /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_10k/teacher_image_probs_step2_imageTopK64 \
--out-dir /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_10k/student_ready_merged_input_step3a

'''