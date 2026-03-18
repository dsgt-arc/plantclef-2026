"""Luigi task orchestrating tile extraction + multi-view embedding.

Main GPU script run per SLURM shard. Reads a manifest of image paths,
applies aspect-ratio-aware tiling, runs 3-view DINOv2 inference, and
writes per-tile parquet with embeddings, logits, and aug consistency info.
"""
from __future__ import annotations

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

import luigi
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from pyspark.sql.functions import udf

from utils.spark import spark_resource
from phase1.tiling import make_tiles_udf
from phase1.multiview_embedder import MultiViewDINOv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing environment variable {name}")
    return val


def _plantclef_root() -> Path:
    return Path(_require_env("PLANTCLEF_ROOT")).expanduser().resolve()


def _read_manifest_lines(manifest_path: Path) -> List[str]:
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
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return lines


class ExtractPhase1Embeddings(luigi.Task):
    """Shard-safe multi-view embedding extraction."""

    # Dataset type
    dataset_type = luigi.Parameter(default="lucas")  # "lucas" or "test"

    # I/O
    input_path = luigi.Parameter(default="")
    checkpoint_path = luigi.Parameter(default="")
    output_root = luigi.Parameter(default="")

    # Shard
    shard_id = luigi.IntParameter(default=0)
    manifest_dir = luigi.Parameter(default="")
    manifest_path = luigi.Parameter(default="")
    overwrite = luigi.BoolParameter(default=True)

    # Compute
    cpu_count = luigi.IntParameter(default=6)
    num_partitions = luigi.IntParameter(default=1)
    batch_size = luigi.IntParameter(default=8)
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")

    # JPEG compression (0=disabled)
    jpeg_quality = luigi.IntParameter(default=0)
    jpeg_subsampling = luigi.IntParameter(default=1)

    # Aug consistency
    aug_consistency_topk = luigi.IntParameter(default=16)
    aug_consistency_threshold = luigi.IntParameter(default=8)

    def _resolve_paths(self):
        root = _plantclef_root()

        input_path = self.input_path
        if not input_path:
            if self.dataset_type == "test":
                input_path = str(root / "data" / "raw" / "PlanCLEF_test_data" / "images")
            else:
                input_path = str(root / "data" / "raw" / "lucas_212k_images")

        ckpt = self.checkpoint_path
        if not ckpt:
            ckpt = str(root / "pretrained_models" / "base_vit_model" / "vit_base_patch14_reg4_dinov2.pth")

        out_root = self.output_root
        if not out_root:
            dataset_label = "phase1_tiles_test" if self.dataset_type == "test" else "phase1_tiles_lucas"
            out_root = str(root / "aerten3" / "parquet" / "lucas_sw77k" / dataset_label)

        shard_out = Path(out_root) / f"shard_{self.shard_id:04d}"

        manifest = None
        if self.manifest_path:
            manifest = Path(self.manifest_path).expanduser().resolve()
        elif self.manifest_dir:
            manifest = Path(self.manifest_dir).expanduser().resolve() / f"shard_{self.shard_id:04d}.txt"

        return Path(input_path), Path(ckpt), shard_out, manifest

    def run(self):
        input_path, checkpoint, shard_out, manifest = self._resolve_paths()

        logger.info("=" * 88)
        logger.info("Phase 1: Multi-View Embedding Extraction")
        logger.info("=" * 88)
        logger.info(f"Dataset type: {self.dataset_type}")
        logger.info(f"Shard ID: {self.shard_id}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Checkpoint: {checkpoint}")
        logger.info(f"Output: {shard_out}")
        logger.info(f"Manifest: {manifest or '(none, will scan input_path)'}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"JPEG quality: {self.jpeg_quality} (0=disabled)")
        logger.info(f"Aug consistency: topk={self.aug_consistency_topk}, threshold={self.aug_consistency_threshold}")
        logger.info("=" * 88)

        shard_out.mkdir(parents=True, exist_ok=True)

        try:
            with spark_resource(cores=self.cpu_count) as spark:
                spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

                # Build file list
                file_list: Optional[List[str]] = None
                if manifest is not None:
                    file_list = _read_manifest_lines(manifest)
                    logger.info(f"Loaded manifest: {len(file_list)} files")

                # Read binary files
                logger.info("Reading image files...")
                reader = (
                    spark.read.format("binaryFile")
                    .option("pathGlobFilter", "*.jpg")
                    .option("recursiveFileLookup", "true")
                )

                if file_list is not None:
                    df = reader.load(file_list).select(
                        F.regexp_extract(F.col("path"), r"([^/\\]+)$", 1).alias("image_name"),
                        F.col("content").alias("data"),
                    )
                else:
                    df = reader.load(str(input_path)).select(
                        F.regexp_extract(F.col("path"), r"([^/\\]+)$", 1).alias("image_name"),
                        F.col("content").alias("data"),
                    )

                df = df.repartition(self.num_partitions)

                # Tiling
                logger.info(f"Applying tiling (dataset_type={self.dataset_type})...")
                tile_udf = make_tiles_udf(self.dataset_type)

                if self.dataset_type == "lucas":
                    # LUCAS tiling UDF takes (data, image_name)
                    df_tiles = (
                        df.withColumn("tiles", tile_udf("data", "image_name"))
                        .select("image_name", F.explode("tiles").alias("tile_struct"))
                        .select(
                            "image_name",
                            F.col("tile_struct.tile_index").alias("tile_index"),
                            F.col("tile_struct.tile").alias("tile"),
                        )
                    )
                else:
                    # Test tiling UDF takes (data) only
                    df_tiles = (
                        df.withColumn("tiles", tile_udf("data"))
                        .select("image_name", F.explode("tiles").alias("tile_struct"))
                        .select(
                            "image_name",
                            F.col("tile_struct.tile_index").alias("tile_index"),
                            F.col("tile_struct.tile").alias("tile"),
                        )
                    )

                # Multi-view embedder
                logger.info("Initializing MultiViewDINOv2 embedder...")
                embedder = MultiViewDINOv2(
                    input_col="tile",
                    output_col="mv_output",
                    model_path=str(checkpoint),
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    jpeg_quality=self.jpeg_quality,
                    jpeg_subsampling=self.jpeg_subsampling,
                    aug_consistency_topk=self.aug_consistency_topk,
                    aug_consistency_threshold=self.aug_consistency_threshold,
                )

                logger.info("Extracting multi-view embeddings...")
                transformed = embedder.transform(df_tiles)

                # Convert to f16 bytes for storage
                to_f16_bytes = udf(
                    lambda arr: np.asarray(arr, dtype=np.float16).tobytes(),
                    BinaryType(),
                )

                final_df = transformed.select(
                    "image_name",
                    "tile_index",
                    to_f16_bytes(F.col("mv_output.embedding_orig")).alias("embedding_orig_f16"),
                    to_f16_bytes(F.col("mv_output.embedding_flip")).alias("embedding_flip_f16"),
                    to_f16_bytes(F.col("mv_output.embedding_rot90")).alias("embedding_rot90_f16"),
                    to_f16_bytes(F.col("mv_output.logits_orig")).alias("teacher_logits_f16"),
                    F.col("mv_output.max_prob").alias("max_prob"),
                    F.col("mv_output.aug_consistency_overlap").alias("aug_consistency_overlap"),
                    F.col("mv_output.passed_aug_consistency").alias("passed_aug_consistency"),
                )

                # Write parquet
                write_mode = "overwrite" if self.overwrite else "errorifexists"
                logger.info(f"Writing parquet ({write_mode}) to: {shard_out}")

                (
                    final_df.repartition(self.num_partitions, "image_name")
                    .sortWithinPartitions("image_name", "tile_index")
                    .write.mode(write_mode)
                    .option("compression", "snappy")
                    .parquet(str(shard_out))
                )

                logger.info("=" * 88)
                logger.info("DONE")
                logger.info(f"Output shard: {shard_out}")
                logger.info("=" * 88)

        except Exception as e:
            logger.error(f"Failed for shard {self.shard_id}: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    luigi.run()
