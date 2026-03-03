#!/usr/bin/env python3

import os
import luigi
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from pyspark.sql.functions import udf

from utils.embedding.transform import WrappedFineTunedDINOv2
from utils.workflow import ProcessTransform, make_image_tiles_udf
from utils.spark import spark_resource

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable {name}. "
            f"Set it in your shell or in ~/.bashrc, for example:\n"
            f"export {name}=..."
        )
    return val


def _plantclef_root() -> Path:
    return Path(_require_env("PLANTCLEF_ROOT")).expanduser().resolve()


def _read_manifest_lines(manifest_path: Path) -> List[str]:
    """
    Reads a manifest file of absolute image paths, one per line.
    Ignores blank lines and comment lines starting with '#'.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    lines: List[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    if not lines:
        raise ValueError(f"Manifest is empty after filtering: {manifest_path}")
    return lines


def _resolve_manifest_for_shard(
    manifest_dir: Optional[str],
    manifest_path: Optional[str],
    shard_id: int,
) -> Optional[Path]:
    """
    Priority:
      1) manifest_path (explicit)
      2) manifest_dir + shard naming convention
      3) None (fallback to scanning input_path)
    """
    if manifest_path:
        return Path(manifest_path).expanduser().resolve()

    if manifest_dir:
        md = Path(manifest_dir).expanduser().resolve()
        candidate = md / f"shard_{shard_id:04d}.txt"
        return candidate

    return None


def _default_checkpoint_path(root: Path) -> str:
    return str(
        root
        / "pretrained_models"
        / "base_vit_model"
        / "vit_base_patch14_reg4_dinov2.pth"
    )


def _default_input_path(root: Path, dataset_name: str) -> str:
    if dataset_name.lower() in {"lucas_10k", "10k"}:
        return str(root / "data" / "raw" / "lucas_10k_images")

    if dataset_name.lower() in {"lucas_212k", "212k"}:
        return str(root / "data" / "raw" / "lucas_212k_images")

    return str(root / "data" / "raw")


def _default_output_root(root: Path, dataset_name: str) -> Path:
    return root / "parquet" / dataset_name / "feature_bank"


class ExtractLUCAS_Embeddings(ProcessTransform):
    """
    Embedding extraction that produces Step4 compatible parquet.

    Output schema per image:
      quadrat_id (string)
      tile_embeddings_f16 (binary) float16 bytes for [n_tiles, emb_dim]
      teacher_logits_f16  (binary) float16 bytes for [n_tiles, num_classes]
    Optional:
      image_name (string) if write_image_name=True
    """

    dataset_name = luigi.Parameter(default="lucas_10k")
    input_path = luigi.Parameter(default="")
    checkpoint_path = luigi.Parameter(default="")

    output_root = luigi.Parameter(default="")
    output_path = luigi.Parameter(default="")  # kept for backwards compatibility with your batch script

    shard_id = luigi.IntParameter(default=0)
    num_shards = luigi.IntParameter(default=1)
    overwrite = luigi.BoolParameter(default=True)

    manifest_dir = luigi.Parameter(default="")
    manifest_path = luigi.Parameter(default="")

    cpu_count = luigi.IntParameter(default=8)
    num_partitions = luigi.IntParameter(default=20)
    batch_size = luigi.IntParameter(default=32)
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")
    grid_size = luigi.IntParameter(default=4)

    # JPEG compression (0=disabled)
    jpeg_quality = luigi.IntParameter(default=0)
    jpeg_subsampling = luigi.IntParameter(default=1)

    debug_counts = luigi.BoolParameter(default=False)
    debug_sample_n = luigi.IntParameter(default=5)

    single_file = luigi.BoolParameter(default=False)
    write_image_name = luigi.BoolParameter(default=False)

    def _resolve_paths(self):
        root = _plantclef_root()

        input_path = self.input_path or _default_input_path(root, self.dataset_name)
        checkpoint_path = self.checkpoint_path or _default_checkpoint_path(root)

        output_root = (
            Path(self.output_root).expanduser().resolve()
            if self.output_root
            else _default_output_root(root, self.dataset_name)
        )

        # Preserve original behavior: if output_path is provided, use it exactly.
        # Otherwise fall back to output_root/shard_XXXX.
        if self.output_path:
            shard_out = Path(self.output_path).expanduser().resolve()
        else:
            shard_out = output_root / f"shard_{self.shard_id:04d}"

        manifest = _resolve_manifest_for_shard(
            manifest_dir=self.manifest_dir or None,
            manifest_path=self.manifest_path or None,
            shard_id=self.shard_id,
        )

        return (
            Path(input_path).expanduser().resolve(),
            Path(checkpoint_path).expanduser().resolve(),
            output_root,
            shard_out,
            manifest,
        )

    def run(self):
        input_path, checkpoint_path, output_root, shard_out, manifest = self._resolve_paths()

        n_tiles_expected = int(self.grid_size) * int(self.grid_size)

        logger.info("=" * 88)
        logger.info("Test Embedding Extraction (Step4 compatible) with Teacher Logits")
        logger.info("=" * 88)
        logger.info(f"Dataset name: {self.dataset_name}")
        logger.info(f"Shard: {self.shard_id} / {self.num_shards}")
        logger.info(f"Input path: {input_path}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Output root: {output_root}")
        logger.info(f"Output shard dir: {shard_out}")
        logger.info(f"Grid size: {self.grid_size}x{self.grid_size} -> n_tiles={n_tiles_expected}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Manifest: {manifest if manifest else '(none, will scan input_path)'}")
        logger.info(f"Overwrite: {self.overwrite}")
        logger.info(f"Single file: {self.single_file}")
        logger.info(f"Write image_name: {self.write_image_name}")
        logger.info(f"JPEG quality: {self.jpeg_quality} (0=disabled)")
        logger.info(f"JPEG subsampling: {self.jpeg_subsampling} (0=4:4:4, 1=4:2:2, 2=4:2:0)")
        logger.info("=" * 88)

        shard_out.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            kwargs = {"cores": self.cpu_count}
            with spark_resource(**kwargs) as spark:
                spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

                file_list: Optional[List[str]] = None
                if manifest is not None:
                    file_list = _read_manifest_lines(manifest)
                    logger.info(f"Loaded manifest lines: {len(file_list)}")

                logger.info("Step 1: Reading JPG files...")

                reader = (
                    spark.read.format("binaryFile")
                    .option("pathGlobFilter", "*.jpg")
                    .option("recursiveFileLookup", "true")
                )

                if file_list is not None:
                    df = (
                        reader.load(file_list)
                        .select(
                            F.regexp_extract(F.col("path"), r"([^/\\\\]+)$", 1).alias("image_name"),
                            F.col("content").alias("data"),
                        )
                    )
                else:
                    df = (
                        reader.load(str(input_path))
                        .select(
                            F.regexp_extract(F.col("path"), r"([^/\\\\]+)$", 1).alias("image_name"),
                            F.col("content").alias("data"),
                        )
                    )

                if self.debug_counts:
                    image_count = df.count()
                    logger.info(f"Debug: images found in this shard = {image_count}")
                else:
                    logger.info("Skipping full image count (debug_counts=false).")

                logger.info(f"Step 2: Repartitioning into {self.num_partitions} partitions...")
                df = df.repartition(self.num_partitions)

                if self.debug_sample_n > 0:
                    sample = df.select("image_name").limit(self.debug_sample_n).collect()
                    logger.info(f"Sample image names: {[r['image_name'] for r in sample]}")

                logger.info(f"Step 3: Applying {self.grid_size}x{self.grid_size} tiling...")
                tile_udf = make_image_tiles_udf(self.grid_size)
                df_tiles = (
                    df.withColumn("tiles", tile_udf("data"))
                    .select("image_name", F.explode("tiles").alias("tile_struct"))
                    .select(
                        "image_name",
                        F.col("tile_struct.tile_index").alias("tile_index"),
                        F.col("tile_struct.tile").alias("tile"),
                    )
                )

                if self.debug_counts:
                    tile_count = df_tiles.count()
                    logger.info(f"Debug: tiles generated in this shard = {tile_count}")
                else:
                    logger.info("Skipping full tile count (debug_counts=false).")

                logger.info("Step 4: Initializing DINOv2 embedder...")
                embedder = WrappedFineTunedDINOv2(
                    input_col="tile",
                    output_col="embeddings_raw",
                    model_path=str(checkpoint_path),
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    jpeg_quality=self.jpeg_quality,
                    jpeg_subsampling=self.jpeg_subsampling,
                )
                logger.info("Embedder initialized.")

                logger.info("Step 5: Extracting per tile embeddings and logits...")
                transformed = embedder.transform(df_tiles)

                expected_fields = {"cls_token", "logits"}
                actual_fields = set(transformed.select("embeddings_raw.*").columns)
                if not expected_fields.issubset(actual_fields):
                    raise ValueError(f"Schema validation failed. Expected {expected_fields}, got {actual_fields}")

                num_classes = int(getattr(embedder, "num_classes", 7806))

                def _pack_tiles(tile_rows):
                    if tile_rows is None:
                        return b""

                    tile_rows = sorted(tile_rows, key=lambda r: int(r["tile_index"]))

                    if len(tile_rows) != n_tiles_expected:
                        raise ValueError(f"Expected {n_tiles_expected} tiles, got {len(tile_rows)}")

                    first = tile_rows[0]["cls_token"]
                    emb_dim = len(first)

                    mat = np.zeros((n_tiles_expected, emb_dim), dtype=np.float16)
                    for i, r in enumerate(tile_rows):
                        tok = r["cls_token"]
                        if len(tok) != emb_dim:
                            raise ValueError(f"Inconsistent emb_dim: {len(tok)} vs {emb_dim}")
                        mat[i, :] = np.asarray(tok, dtype=np.float16)

                    return mat.tobytes()

                def _pack_logits(tile_rows):
                    if tile_rows is None:
                        return b""

                    tile_rows = sorted(tile_rows, key=lambda r: int(r["tile_index"]))

                    if len(tile_rows) != n_tiles_expected:
                        raise ValueError(f"Expected {n_tiles_expected} tiles, got {len(tile_rows)}")

                    first = tile_rows[0]["logits"]
                    c = len(first)
                    if c != num_classes:
                        raise ValueError(f"logits length {c} does not match num_classes {num_classes}")

                    mat = np.zeros((n_tiles_expected, num_classes), dtype=np.float16)
                    for i, r in enumerate(tile_rows):
                        lg = r["logits"]
                        if len(lg) != num_classes:
                            raise ValueError(f"Inconsistent num_classes in logits: {len(lg)} vs {num_classes}")
                        mat[i, :] = np.asarray(lg, dtype=np.float16)

                    return mat.tobytes()

                pack_tiles_udf = udf(_pack_tiles, BinaryType())
                pack_logits_udf = udf(_pack_logits, BinaryType())

                per_tile = transformed.select(
                    "image_name",
                    "tile_index",
                    F.col("embeddings_raw.cls_token").alias("cls_token"),
                    F.col("embeddings_raw.logits").alias("logits"),
                )

                grouped = (
                    per_tile.groupBy("image_name")
                    .agg(
                        F.collect_list(F.struct("tile_index", "cls_token", "logits")).alias("tile_rows")
                    )
                )

                base_cols = [
                    F.regexp_replace(F.col("image_name"), r"\.jpg$", "").alias("quadrat_id"),
                    pack_tiles_udf(F.col("tile_rows")).alias("tile_embeddings_f16"),
                    pack_logits_udf(F.col("tile_rows")).alias("teacher_logits_f16"),
                ]

                if self.write_image_name:
                    base_cols.insert(0, F.col("image_name"))

                final_df = grouped.select(*base_cols)

                write_mode = "overwrite" if self.overwrite else "errorifexists"
                logger.info(f"Step 6: Writing parquet ({write_mode}) to: {shard_out}")

                writer_df = final_df
                if self.single_file:
                    logger.info("Using coalesce(1) so Spark writes a single parquet part file.")
                    writer_df = writer_df.coalesce(1)
                else:
                    writer_df = writer_df.repartition(self.num_partitions, "quadrat_id").sortWithinPartitions("quadrat_id")

                (
                    writer_df.write.mode(write_mode)
                    .option("compression", "snappy")
                    .parquet(str(shard_out))
                )

                logger.info("=" * 88)
                logger.info("DONE")
                logger.info("=" * 88)
                logger.info(f"Output shard: {shard_out}")
                logger.info("=" * 88)

        except Exception as e:
            logger.error(f"Failed to extract embeddings for shard {self.shard_id}: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    luigi.run()



'''
python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step4_test_set_inference/extract_testdata_embeddings.py ExtractLUCAS_Embeddings \
  --task plantclef_2026_test_embed_shard0000 \
  --dataset-name plantclef_2026_test \
  --manifest-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/manifests/plantclef_2026_test_manifest.txt \
  --output-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/feature_bank_single \
  --output-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/feature_bank_JPEG_compressed/shard_0000 \
  --shard-id 0 \
  --num-shards 1 \
  --grid-size 4 \
  --batch-size 32 \
  --num-partitions 20 \
  --single-file \
  --write-image-name \
  --local-scheduler

'''