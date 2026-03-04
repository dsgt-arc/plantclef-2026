#!/usr/bin/env python3
"""
Extract per-tile top-K species from test images with softmax applied during
PyTorch inference (float32 on GPU), matching the reference paper data flow.

Key difference from extract_testdata_embeddings.py:
  - Applies torch.softmax(logits, dim=1) in float32 during inference
  - Extracts top-K species per tile directly (torch.topk)
  - Stores per-tile species lists as JSON string (no float16 logit storage)

This matches the reference paper's classification/transform.py which applies
torch.softmax(logits, dim=1) during inference before any storage or aggregation.

Output parquet schema:
  quadrat_id (string)        - image ID (filename without .jpg)
  tile_species_json (string) - JSON array of N_TILES arrays, each with top-K species indices
"""

import io
import os
import sys
import json
import luigi
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional

import timm
import torch
from PIL import Image
from torchvision import transforms as T

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructType, StructField
from pyspark.sql.functions import udf
from pyspark.ml.functions import predict_batch_udf

# Add Exp-2-JPEG_Compression to import path so "utils" is visible
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.serde import deserialize_image
from utils.workflow import make_image_tiles_udf
from utils.spark import spark_resource

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


def _apply_jpeg_compression(image: Image.Image, quality: int, subsampling: int) -> Image.Image:
    """JPEG round-trip to align pixel statistics with DINOv2 training data."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).copy()


class TileSpeciesExtractor:
    """
    Extracts top-K species per tile with softmax applied in PyTorch (float32).
    Matches reference paper's classification/transform.py data flow.

    Unlike WrappedFineTunedDINOv2 which returns raw logits, this class:
      1. Applies torch.softmax(logits, dim=1) in float32 on GPU
      2. Extracts top-K species indices via torch.topk
      3. Returns only the species indices (no logits, no cls_token)
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 16,
        per_tile_topk: int = 9,
        jpeg_quality: int = 0,
        jpeg_subsampling: int = 1,
    ):
        self.num_classes = 7806
        self.per_tile_topk = per_tile_topk
        self.batch_size = batch_size
        self.jpeg_quality = jpeg_quality
        self.jpeg_subsampling = jpeg_subsampling

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            timm.create_model(
                model_name,
                pretrained=False,
                num_classes=self.num_classes,
                checkpoint_path=model_path,
            )
            .to(self.device)
            .eval()
        )

        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )

        # Custom transform for JPEG mode (Lanczos interpolation per paper Figure 5)
        img_size = self.data_config["input_size"][-1]  # 518
        self._pil_resize_crop = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(img_size),
        ])
        self._to_tensor_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=self.data_config.get("mean", (0.485, 0.456, 0.406)),
                std=self.data_config.get("std", (0.229, 0.224, 0.225)),
            ),
        ])

    def _make_predict_fn(self):
        per_tile_topk = self.per_tile_topk

        def predict(image_batch: np.ndarray) -> list:
            if self.jpeg_quality > 0:
                images = []
                for img in image_batch:
                    pil_img = deserialize_image(img)
                    pil_img = self._pil_resize_crop(pil_img)
                    pil_img = _apply_jpeg_compression(
                        pil_img, self.jpeg_quality, self.jpeg_subsampling
                    )
                    images.append(self._to_tensor_normalize(pil_img))
            else:
                images = [
                    self.transforms(deserialize_image(img))
                    for img in image_batch
                ]

            model_inputs = torch.stack(images).to(self.device)

            with torch.no_grad():
                logits = self.model(model_inputs)
                # KEY DIFFERENCE: softmax in float32 on GPU, matching reference paper
                probs = torch.softmax(logits, dim=1)
                _, topk_indices = torch.topk(probs, k=per_tile_topk, dim=1)

            return [
                {"top_species": topk_indices[i].cpu().tolist()}
                for i in range(len(image_batch))
            ]

        return predict

    def transform(self, df, input_col: str, output_col: str):
        schema = StructType([
            StructField("top_species", ArrayType(IntegerType())),
        ])
        return df.withColumn(
            output_col,
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=schema,
                batch_size=self.batch_size,
            )(input_col),
        )


class ExtractTileSpecies(luigi.Task):
    """
    Luigi task: extract per-tile top-K species with float32 softmax.
    Produces parquet with (quadrat_id, tile_species_json).
    """

    task = luigi.Parameter(default="tile_species")
    dataset_name = luigi.Parameter(default="plantclef_2026_test")
    input_path = luigi.Parameter(default="")
    checkpoint_path = luigi.Parameter(default="")

    output_root = luigi.Parameter(default="")
    output_path = luigi.Parameter(default="")

    shard_id = luigi.IntParameter(default=0)
    num_shards = luigi.IntParameter(default=1)
    overwrite = luigi.BoolParameter(default=True)

    manifest_dir = luigi.Parameter(default="")
    manifest_path = luigi.Parameter(default="")

    cpu_count = luigi.IntParameter(default=8)
    num_partitions = luigi.IntParameter(default=1)
    batch_size = luigi.IntParameter(default=16)
    model_name = luigi.Parameter(default="vit_base_patch14_reg4_dinov2.lvd142m")
    grid_size = luigi.IntParameter(default=4)
    per_tile_topk = luigi.IntParameter(default=9)

    jpeg_quality = luigi.IntParameter(default=0)
    jpeg_subsampling = luigi.IntParameter(default=1)

    single_file = luigi.BoolParameter(default=False)
    write_image_name = luigi.BoolParameter(default=False)

    def _resolve_paths(self):
        root = _plantclef_root()

        input_path = self.input_path or str(root / "data" / "raw")
        checkpoint_path = self.checkpoint_path or str(
            root / "pretrained_models" / "base_vit_model" / "vit_base_patch14_reg4_dinov2.pth"
        )

        output_root = (
            Path(self.output_root).expanduser().resolve()
            if self.output_root
            else root / "parquet" / self.dataset_name / "tile_species"
        )

        if self.output_path:
            shard_out = Path(self.output_path).expanduser().resolve()
        else:
            shard_out = output_root / f"shard_{self.shard_id:04d}"

        manifest = None
        if self.manifest_path:
            manifest = Path(self.manifest_path).expanduser().resolve()
        elif self.manifest_dir:
            md = Path(self.manifest_dir).expanduser().resolve()
            manifest = md / f"shard_{self.shard_id:04d}.txt"

        return (
            Path(input_path).expanduser().resolve(),
            Path(checkpoint_path).expanduser().resolve(),
            output_root,
            shard_out,
            manifest,
        )

    def run(self):
        input_path, checkpoint_path, output_root, shard_out, manifest = self._resolve_paths()
        n_tiles = int(self.grid_size) ** 2

        logger.info("=" * 88)
        logger.info("Tile Species Extraction (softmax in PyTorch float32, matching reference paper)")
        logger.info("=" * 88)
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Shard: {self.shard_id} / {self.num_shards}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Output root: {output_root}")
        logger.info(f"Output shard: {shard_out}")
        logger.info(f"Grid: {self.grid_size}x{self.grid_size} -> {n_tiles} tiles")
        logger.info(f"Per-tile top-K: {self.per_tile_topk}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Manifest: {manifest if manifest else '(none, scan input_path)'}")
        logger.info(f"JPEG quality: {self.jpeg_quality} (0=disabled)")
        logger.info(f"JPEG subsampling: {self.jpeg_subsampling}")
        logger.info("=" * 88)

        shard_out.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            with spark_resource(cores=self.cpu_count) as spark:
                spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

                # --- Read images ---
                file_list: Optional[List[str]] = None
                if manifest is not None:
                    file_list = _read_manifest_lines(manifest)
                    logger.info(f"Manifest: {len(file_list)} images")

                reader = (
                    spark.read.format("binaryFile")
                    .option("pathGlobFilter", "*.jpg")
                    .option("recursiveFileLookup", "true")
                )

                if file_list is not None:
                    df = reader.load(file_list).select(
                        F.regexp_extract(F.col("path"), r"([^/\\\\]+)$", 1).alias("image_name"),
                        F.col("content").alias("data"),
                    )
                else:
                    df = reader.load(str(input_path)).select(
                        F.regexp_extract(F.col("path"), r"([^/\\\\]+)$", 1).alias("image_name"),
                        F.col("content").alias("data"),
                    )

                df = df.repartition(self.num_partitions)

                # --- Tile ---
                logger.info(f"Tiling {self.grid_size}x{self.grid_size}...")
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

                # --- Extract per-tile species with float32 softmax ---
                logger.info("Extracting per-tile species (softmax in PyTorch float32)...")
                extractor = TileSpeciesExtractor(
                    model_path=str(checkpoint_path),
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    per_tile_topk=self.per_tile_topk,
                    jpeg_quality=self.jpeg_quality,
                    jpeg_subsampling=self.jpeg_subsampling,
                )

                transformed = extractor.transform(df_tiles, "tile", "tile_output")

                per_tile = transformed.select(
                    "image_name",
                    "tile_index",
                    F.col("tile_output.top_species").alias("top_species"),
                )

                # --- Pack tiles per image ---
                def _pack_tile_species(tile_rows):
                    import json as _json

                    if tile_rows is None:
                        return "[]"
                    tile_rows = sorted(tile_rows, key=lambda r: int(r["tile_index"]))
                    if len(tile_rows) != n_tiles:
                        raise ValueError(
                            f"Expected {n_tiles} tiles, got {len(tile_rows)}"
                        )
                    species_per_tile = [list(r["top_species"]) for r in tile_rows]
                    return _json.dumps(species_per_tile)

                pack_udf = udf(_pack_tile_species, StringType())

                grouped = per_tile.groupBy("image_name").agg(
                    F.collect_list(
                        F.struct("tile_index", "top_species")
                    ).alias("tile_rows")
                )

                base_cols = [
                    F.regexp_replace(F.col("image_name"), r"\.jpg$", "").alias("quadrat_id"),
                    pack_udf(F.col("tile_rows")).alias("tile_species_json"),
                ]

                if self.write_image_name:
                    base_cols.insert(0, F.col("image_name"))

                final_df = grouped.select(*base_cols)

                # --- Write parquet ---
                write_mode = "overwrite" if self.overwrite else "errorifexists"
                logger.info(f"Writing parquet ({write_mode}) to: {shard_out}")

                writer_df = final_df
                if self.single_file:
                    writer_df = writer_df.coalesce(1)
                else:
                    writer_df = writer_df.repartition(
                        self.num_partitions, "quadrat_id"
                    ).sortWithinPartitions("quadrat_id")

                (
                    writer_df.write.mode(write_mode)
                    .option("compression", "snappy")
                    .parquet(str(shard_out))
                )

                logger.info("=" * 88)
                logger.info("DONE")
                logger.info(f"Output shard: {shard_out}")
                logger.info("=" * 88)

        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    luigi.run()


'''

## No JPEG compression (match reference paper exactly)

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/extract_testdata_tile_species.py ExtractTileSpecies \
  --dataset-name plantclef_2026_test \
  --input-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/PlanCLEF_test_data/images \
  --output-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species \
  --output-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species/shard_0000 \
  --checkpoint-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth \
  --manifest-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/manifests/plantclef_2026_test/sharded_4/test_manifest_shard_00.txt \
  --shard-id 0 \
  --num-shards 1 \
  --cpu-count 6 \
  --num-partitions 1 \
  --batch-size 16 \
  --grid-size 4 \
  --per-tile-topk 9 \
  --jpeg-quality 0 \
  --single-file \
  --overwrite \
  --local-scheduler


## With JPEG compression (for later comparison)

python /storage/home/hcoda1/2/aerten3/plantclef-2026/user/alper/src/Exp-2-JPEG_Compression/step2_B_teacher_testing/extract_testdata_tile_species.py ExtractTileSpecies \
  --dataset-name plantclef_2026_test \
  --input-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/data/raw/PlanCLEF_test_data/images \
  --output-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species_JPEG \
  --output-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species_JPEG/shard_0000 \
  --checkpoint-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/pretrained_models/base_vit_model/vit_base_patch14_reg4_dinov2.pth \
  --manifest-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/manifests/plantclef_2026_test/sharded_4/test_manifest_shard_00.txt \
  --shard-id 0 \
  --num-shards 1 \
  --cpu-count 6 \
  --num-partitions 1 \
  --batch-size 16 \
  --grid-size 4 \
  --per-tile-topk 9 \
  --jpeg-quality 85 \
  --jpeg-subsampling 1 \
  --single-file \
  --overwrite \
  --local-scheduler

'''
