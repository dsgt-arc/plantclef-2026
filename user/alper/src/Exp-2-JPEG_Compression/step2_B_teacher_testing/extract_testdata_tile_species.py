#!/usr/bin/env python3
"""
Extract per-tile top-K species from test images with softmax applied during
PyTorch inference (float32 on GPU), matching the reference paper data flow.

Key difference from extract_testdata_embeddings.py:
  - Applies torch.softmax(logits, dim=1) in float32 during inference
  - Extracts top-K species per tile directly (torch.topk)
  - Stores per-tile species lists as JSON string (no float16 logit storage)

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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _ensure_exp_root_on_path() -> Path:
    """
    Ensure Exp-2-JPEG_Compression is on sys.path and PYTHONPATH so that:
      - driver can `import utils`
      - Spark executors can `import utils` when we set spark.executorEnv.PYTHONPATH

    Returns the experiment root directory (Exp-2-JPEG_Compression).
    """
    # Allow sbatch to override explicitly if desired
    env_root = os.environ.get("EXP_JPEG_ROOT", "").strip()
    if env_root:
        exp_root = Path(env_root).expanduser().resolve()
    else:
        # This script lives in:
        #   Exp-2-JPEG_Compression/step2_B_teacher_testing/extract_testdata_tile_species.py
        exp_root = Path(__file__).resolve().parents[1]

    if not (exp_root / "utils").is_dir():
        raise FileNotFoundError(f"Expected utils/ under: {exp_root}")

    exp_root_str = str(exp_root)

    # Make available to the current process
    if exp_root_str not in sys.path:
        sys.path.insert(0, exp_root_str)

    # Make available via environment for Spark executor propagation
    cur_pp = os.environ.get("PYTHONPATH", "")
    if cur_pp:
        parts = cur_pp.split(":")
        if exp_root_str not in parts:
            os.environ["PYTHONPATH"] = exp_root_str + ":" + cur_pp
    else:
        os.environ["PYTHONPATH"] = exp_root_str

    return exp_root


# Ensure path is set before importing repo-local modules
EXP_ROOT = _ensure_exp_root_on_path()

from utils.serde import deserialize_image
from utils.workflow import make_image_tiles_udf
from utils.spark import spark_resource


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
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).copy()


class TileSpeciesExtractor:
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
        self.transforms = timm.data.create_transform(**self.data_config, is_training=False)

        img_size = self.data_config["input_size"][-1]
        self._pil_resize_crop = T.Compose(
            [
                T.Resize(img_size, interpolation=T.InterpolationMode.LANCZOS),
                T.CenterCrop(img_size),
            ]
        )
        self._to_tensor_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=self.data_config.get("mean", (0.485, 0.456, 0.406)),
                    std=self.data_config.get("std", (0.229, 0.224, 0.225)),
                ),
            ]
        )

    def _make_predict_fn(self):
        per_tile_topk = self.per_tile_topk

        def predict(image_batch: np.ndarray) -> list:
            if self.jpeg_quality > 0:
                images = []
                for img in image_batch:
                    pil_img = deserialize_image(img)
                    pil_img = self._pil_resize_crop(pil_img)
                    pil_img = _apply_jpeg_compression(pil_img, self.jpeg_quality, self.jpeg_subsampling)
                    images.append(self._to_tensor_normalize(pil_img))
            else:
                images = [self.transforms(deserialize_image(img)) for img in image_batch]

            model_inputs = torch.stack(images).to(self.device)

            with torch.no_grad():
                logits = self.model(model_inputs)
                probs = torch.softmax(logits, dim=1)
                _, topk_indices = torch.topk(probs, k=per_tile_topk, dim=1)

            # Convert to numpy once for the whole batch
            topk_np = topk_indices.detach().cpu().to(torch.int32).numpy()

            return [
                {"top_species": topk_np[i]}  # 1D numpy array, has .shape
                for i in range(topk_np.shape[0])
            ]

        return predict

    def transform(self, df, input_col: str, output_col: str):
        schema = StructType([StructField("top_species", ArrayType(IntegerType()))])
        return df.withColumn(
            output_col,
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=schema,
                batch_size=self.batch_size,
            )(input_col),
        )


class ExtractTileSpecies(luigi.Task):
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
        logger.info("Tile Species Extraction")
        logger.info("=" * 88)
        logger.info(f"EXP_ROOT: {EXP_ROOT}")
        logger.info(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
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

                # Critical fix: propagate PYTHONPATH into Spark executors
                exec_pp = os.environ.get("SPARK_EXECUTOR_PYTHONPATH") or os.environ.get("PYTHONPATH")
                if exec_pp:
                    spark.conf.set("spark.executorEnv.PYTHONPATH", exec_pp)

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

                logger.info("Extracting per-tile species...")
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

                def _pack_tile_species(tile_rows):
                    import json as _json

                    if tile_rows is None:
                        return "[]"
                    tile_rows = sorted(tile_rows, key=lambda r: int(r["tile_index"]))
                    if len(tile_rows) != n_tiles:
                        raise ValueError(f"Expected {n_tiles} tiles, got {len(tile_rows)}")
                    species_per_tile = [list(r["top_species"]) for r in tile_rows]
                    return _json.dumps(species_per_tile)

                pack_udf = udf(_pack_tile_species, StringType())

                grouped = per_tile.groupBy("image_name").agg(
                    F.collect_list(F.struct("tile_index", "top_species")).alias("tile_rows")
                )

                base_cols = [
                    F.regexp_replace(F.col("image_name"), r"\.jpg$", "").alias("quadrat_id"),
                    pack_udf(F.col("tile_rows")).alias("tile_species_json"),
                ]

                if self.write_image_name:
                    base_cols.insert(0, F.col("image_name"))

                final_df = grouped.select(*base_cols)

                write_mode = "overwrite" if self.overwrite else "errorifexists"
                logger.info(f"Writing parquet ({write_mode}) to: {shard_out}")

                writer_df = final_df
                if self.single_file:
                    writer_df = writer_df.coalesce(1)
                else:
                    writer_df = writer_df.repartition(self.num_partitions, "quadrat_id").sortWithinPartitions(
                        "quadrat_id"
                    )

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
  --output-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species_direct \
  --output-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species_diret/shard_0000 \
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
  --output-root /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species_diect_JPEG \
  --output-path /storage/home/hcoda1/2/aerten3/ps-clef2026_plantclef-0/parquet/plantclef_2026_test/tile_species_direct_JPEG/shard_0000 \
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
