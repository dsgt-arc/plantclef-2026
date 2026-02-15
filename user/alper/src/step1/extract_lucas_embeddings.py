import os
import luigi
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
from pyspark.sql.functions import udf

# Your current utils imports remain, assuming PYTHONPATH points to .../src
from utils.embedding.transform import WrappedFineTunedDINOv2
from utils.workflow import ProcessTransform, make_image_tiles_udf
from utils.spark import spark_resource

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Later to process 212,000 images, use shard manifest files to allow for parallel processing



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
      3) None (fallback to scanning input_path, OK for 10k, not recommended for 212k)
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
    # Keep your existing 10k layout
    if dataset_name.lower() in {"lucas_10k", "10k"}:
        return str(root / "data" / "raw" / "lucas_10k_images")

    # For 212k, you can change this to whatever folder you end up using
    if dataset_name.lower() in {"lucas_212k", "212k"}:
        return str(root / "data" / "raw" / "lucas_212k_images")

    # Fallback: let the user pass input_path explicitly
    return str(root / "data" / "raw")


def _default_output_root(root: Path, dataset_name: str) -> Path:
    return root / "parquet" / dataset_name / "feature_bank"


class ExtractLUCAS_Embeddings(ProcessTransform):
    """
    Shard-safe embedding extraction for LUCAS images.

    Designed to run as a SLURM job array:
      - each task processes one shard manifest
      - each task writes to its own output directory
      - later training can read the parent parquet folder
    """

    # Core I O
    dataset_name = luigi.Parameter(default="lucas_10k")
    input_path = luigi.Parameter(default="")
    checkpoint_path = luigi.Parameter(default="")

    # Output folder behavior
    output_root = luigi.Parameter(default="")  # parent folder for all shards
    shard_id = luigi.IntParameter(default=0)
    num_shards = luigi.IntParameter(default=1)
    overwrite = luigi.BoolParameter(default=True)

    # Manifest support
    manifest_dir = luigi.Parameter(default="")   # expects shard_0000.txt, shard_0001.txt...
    manifest_path = luigi.Parameter(default="")  # explicit shard manifest path

    # Compute settings
    cpu_count = luigi.IntParameter(default=8)
    num_partitions = luigi.IntParameter(default=20)
    batch_size = luigi.IntParameter(default=32)
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")
    grid_size = luigi.IntParameter(default=4)

    # Debug flags
    debug_counts = luigi.BoolParameter(default=False)
    debug_sample_n = luigi.IntParameter(default=5)

    def _resolve_paths(self):
        root = _plantclef_root()

        input_path = self.input_path or _default_input_path(root, self.dataset_name)
        checkpoint_path = self.checkpoint_path or _default_checkpoint_path(root)

        output_root = (
            Path(self.output_root).expanduser().resolve()
            if self.output_root
            else _default_output_root(root, self.dataset_name)
        )

        # shard-specific output directory prevents collisions across parallel jobs
        shard_out = output_root / f"shard_{self.shard_id:04d}"

        # Resolve manifest
        manifest = _resolve_manifest_for_shard(
            manifest_dir=self.manifest_dir or None,
            manifest_path=self.manifest_path or None,
            shard_id=self.shard_id,
        )

        return Path(input_path).expanduser().resolve(), Path(checkpoint_path).expanduser().resolve(), shard_out, manifest

    def run(self):
        input_path, checkpoint_path, shard_out, manifest = self._resolve_paths()

        logger.info("=" * 88)
        logger.info("LUCAS Embedding Extraction (Shard-Safe)")
        logger.info("=" * 88)
        logger.info(f"Dataset name: {self.dataset_name}")
        logger.info(f"Shard: {self.shard_id} / {self.num_shards}")
        logger.info(f"Input path: {input_path}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Output shard dir: {shard_out}")
        logger.info(f"Grid size: {self.grid_size}x{self.grid_size}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Manifest: {manifest if manifest else '(none, will scan input_path)'}")
        logger.info(f"Overwrite: {self.overwrite}")
        logger.info("=" * 88)

        shard_out.mkdir(parents=True, exist_ok=True)

        try:
            kwargs = {"cores": self.cpu_count}
            with spark_resource(**kwargs) as spark:
                spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

                # Build file list for this shard if manifest is provided
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

                # Preferred: read only the subset of files for this shard
                if file_list is not None:
                    df = (
                        reader.load(file_list)
                        .select(
                            F.regexp_extract(F.col("path"), r"([^/\\\\]+)$", 1).alias("image_name"),
                            F.col("content").alias("data"),
                        )
                    )
                else:
                    # Fallback: scan entire input_path
                    # OK for 10k, not recommended for 212k multi-job parallelization
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

                # Partitioning
                logger.info(f"Step 2: Repartitioning into {self.num_partitions} partitions...")
                df = df.repartition(self.num_partitions)

                # Optional small sample to sanity check
                if self.debug_sample_n > 0:
                    sample = df.select("image_name").limit(self.debug_sample_n).collect()
                    logger.info(f"Sample image names: {[r['image_name'] for r in sample]}")

                # Tiling
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

                # Embedder
                logger.info("Step 4: Initializing DINOv2 embedder...")
                embedder = WrappedFineTunedDINOv2(
                    input_col="tile",
                    output_col="embeddings_raw",
                    model_path=str(checkpoint_path),
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                )
                logger.info("Embedder initialized.")

                # Transform
                logger.info("Step 5: Extracting per-tile embeddings and logits...")
                transformed = embedder.transform(df_tiles)

                # Validate output schema lightly without forcing a full scan
                expected_fields = {"cls_token", "logits"}
                actual_fields = set(transformed.select("embeddings_raw.*").columns)
                if not expected_fields.issubset(actual_fields):
                    raise ValueError(
                        f"Schema validation failed. Expected {expected_fields}, got {actual_fields}"
                    )

                to_f16_bytes = udf(
                    lambda arr: np.asarray(arr, dtype=np.float16).tobytes(),
                    BinaryType(),
                )

                final_df = transformed.select(
                    "image_name",
                    "tile_index",
                    to_f16_bytes(F.col("embeddings_raw.cls_token")).alias("embedding_f16"),
                    to_f16_bytes(F.col("embeddings_raw.logits")).alias("teacher_logits_f16"),
                )

                # Write
                write_mode = "overwrite" if self.overwrite else "errorifexists"
                logger.info(f"Step 6: Writing parquet shard ({write_mode}) to: {shard_out}")

                (
                    final_df.repartition(self.num_partitions, "image_name")
                    .sortWithinPartitions("image_name", "tile_index")
                    .write.mode(write_mode)
                    .option("compression", "snappy")
                    .parquet(str(shard_out))
                )

                logger.info("=" * 88)
                logger.info("DONE")
                logger.info("=" * 88)
                logger.info(f"Output shard: {shard_out}")
                logger.info("Next step: point training to the parent folder:")
                logger.info(f"  {_default_output_root(_plantclef_root(), self.dataset_name)}")
                logger.info("=" * 88)

        except Exception as e:
            logger.error(f"Failed to extract embeddings for shard {self.shard_id}: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    luigi.run()
