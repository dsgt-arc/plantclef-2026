#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


def iter_shard_dirs(root_dir: str):
    """
    Yields shard directory names like shard_0000 in sorted order.
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")

    shards = []
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.startswith("shard_"):
            shards.append(name)
    shards.sort()
    return shards


def compute_global_mu_sd(step2_root: str, batch_size: int = 8192):
    """
    Streaming mean and std for latitude, longitude, altitude across all Step 2 shards.
    """
    dset = ds.dataset(step2_root, format="parquet")
    scanner = dset.scanner(columns=["latitude", "longitude", "altitude"], batch_size=batch_size)

    n = 0
    s = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)

    for rb in scanner.to_batches():
        lat = rb.column(0).to_numpy(zero_copy_only=False)
        lon = rb.column(1).to_numpy(zero_copy_only=False)
        alt = rb.column(2).to_numpy(zero_copy_only=False)

        arr = np.stack([lat, lon, alt], axis=1).astype(np.float64, copy=False)
        valid = np.isfinite(arr).all(axis=1)
        if not np.any(valid):
            continue

        v = arr[valid]
        n += v.shape[0]
        s += v.sum(axis=0)
        s2 += (v * v).sum(axis=0)

    if n == 0:
        mu = np.zeros(3, dtype=np.float64)
        sd = np.ones(3, dtype=np.float64)
        return mu, sd

    mu = s / n
    var = (s2 / n) - (mu * mu)
    var = np.maximum(var, 1e-12)
    sd = np.sqrt(var)
    return mu, sd


def make_meta(la, lo, al, mu, sd):
    has = float(np.isfinite(la) and np.isfinite(lo) and np.isfinite(al))
    if has == 1.0:
        return np.array(
            [(la - mu[0]) / sd[0], (lo - mu[1]) / sd[1], (al - mu[2]) / sd[2], 1.0],
            dtype=np.float32,
        )
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def build_student_ready_sharded(
    step1_root: str,
    step2_root: str,
    out_root: str,
    num_tiles: int = 16,
    emb_dim: int = 768,
    require_all_tiles: bool = True,
    step2_stats_batch_size: int = 8192,
):
    """
    Shard aligned builder.

    Expects:
      step1_root/shard_XXXX/*.parquet part files with columns:
        image_name, tile_index, embedding_f16

      step2_root/shard_XXXX/*.parquet part files with columns:
        image_name, teacher_image_probs_f16, latitude, longitude, altitude

    Writes:
      out_root/shard_XXXX/data.parquet with columns:
        image_name, tile_embeddings_f16, teacher_image_probs_f16, metadata_tensor_f32
    """
    os.makedirs(out_root, exist_ok=True)

    emb_bytes_per_tile = emb_dim * 2
    expected_packed_len = num_tiles * emb_bytes_per_tile

    print("Computing global metadata normalization from Step 2")
    mu, sd = compute_global_mu_sd(step2_root, batch_size=step2_stats_batch_size)
    print(f"lat_mu {mu[0]:.6f}  lat_sd {sd[0]:.6f}")
    print(f"lon_mu {mu[1]:.6f}  lon_sd {sd[1]:.6f}")
    print(f"alt_mu {mu[2]:.6f}  alt_sd {sd[2]:.6f}")

    step1_shards = set(iter_shard_dirs(step1_root))
    step2_shards = set(iter_shard_dirs(step2_root))

    common = sorted(step1_shards.intersection(step2_shards))
    if not common:
        raise ValueError(
            "No common shard directories found between Step 1 and Step 2 roots. "
            f"Step1 has {len(step1_shards)} shards, Step2 has {len(step2_shards)} shards."
        )

    print(f"Found {len(common)} common shards")

    total_kept = 0
    total_missing_step2 = 0
    total_bad_tiles = 0

    for shard_name in common:
        s1_dir = os.path.join(step1_root, shard_name)
        s2_dir = os.path.join(step2_root, shard_name)
        out_dir = os.path.join(out_root, shard_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nProcessing {shard_name}")
        print(f"  Step1 {s1_dir}")
        print(f"  Step2 {s2_dir}")

        # Load Step 2 shard into a dict for fast lookup
        t2 = pq.read_table(
            s2_dir,
            columns=["image_name", "teacher_image_probs_f16", "latitude", "longitude", "altitude"],
        )
        t2_df = t2.to_pandas()
        t2_map = {}
        for _, r in t2_df.iterrows():
            img = r["image_name"]
            blob = r["teacher_image_probs_f16"]
            meta = make_meta(r["latitude"], r["longitude"], r["altitude"], mu, sd)
            t2_map[img] = (blob, meta)

        print(f"  Step2 rows in shard {len(t2_map)}")

        # Stream Step 1 shard via Arrow dataset batches
        dset1 = ds.dataset(s1_dir, format="parquet")
        scanner1 = dset1.scanner(columns=["image_name", "tile_index", "embedding_f16"], batch_size=20000)

        tiles = {}
        rows_read = 0

        for rb in scanner1.to_batches():
            imgs = rb.column(0).to_pylist()
            tis = rb.column(1).to_pylist()
            embs = rb.column(2).to_pylist()
            rows_read += len(imgs)

            for img, ti, emb_blob in zip(imgs, tis, embs):
                if img not in tiles:
                    tiles[img] = [None] * num_tiles
                ti_int = int(ti)
                if 0 <= ti_int < num_tiles:
                    tiles[img][ti_int] = emb_blob

        print(f"  Step1 tile rows read {rows_read}")
        print(f"  Unique images in Step1 shard {len(tiles)}")

        out_image = []
        out_tile_blob = []
        out_teacher_blob = []
        out_meta = []

        missing_step2 = 0
        bad_tiles = 0

        for img, tile_list in tiles.items():
            if img not in t2_map:
                missing_step2 += 1
                continue

            if require_all_tiles and any(x is None for x in tile_list):
                bad_tiles += 1
                continue

            # If not requiring all tiles, drop images with any missing tiles anyway
            # because packing expects fixed size
            if any(x is None for x in tile_list):
                bad_tiles += 1
                continue

            packed = b"".join(tile_list)
            if len(packed) != expected_packed_len:
                bad_tiles += 1
                continue

            teacher_blob, meta = t2_map[img]
            out_image.append(img)
            out_tile_blob.append(packed)
            out_teacher_blob.append(teacher_blob)
            out_meta.append(meta)

        kept = len(out_image)
        total_kept += kept
        total_missing_step2 += missing_step2
        total_bad_tiles += bad_tiles

        print(f"  Images kept {kept}")
        print(f"  Missing Step2 rows {missing_step2}")
        print(f"  Bad or incomplete tiles {bad_tiles}")

        if kept == 0:
            print("  Nothing to write for this shard")
            continue

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
        print(f"  Wrote {out_path}")

        # release big objects
        del t2, t2_df, t2_map, tiles, table_out

    print("\nDone")
    print(f"Total images kept {total_kept}")
    print(f"Total missing Step2 rows {total_missing_step2}")
    print(f"Total bad or incomplete tiles {total_bad_tiles}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-root", required=True)
    ap.add_argument("--step2-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--num-tiles", type=int, default=16)
    ap.add_argument("--emb-dim", type=int, default=768)
    ap.add_argument("--require-all-tiles", action="store_true", default=True)
    ap.add_argument("--step2-stats-batch-size", type=int, default=8192)
    args = ap.parse_args()

    build_student_ready_sharded(
        step1_root=args.step1_root,
        step2_root=args.step2_root,
        out_root=args.out_root,
        num_tiles=args.num_tiles,
        emb_dim=args.emb_dim,
        require_all_tiles=args.require_all_tiles,
        step2_stats_batch_size=args.step2_stats_batch_size,
    )


if __name__ == "__main__":
    main()


"""
Example for your 212k layout

python src/step3_v2/build_student_ready_merged_data_step3a.py \
  --step1-root /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/feature_bank \
  --step2-root /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/step2_teacher_agg \
  --out-root  /storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k/student_ready_merged_input_step3a
"""
