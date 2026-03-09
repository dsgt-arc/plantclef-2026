"""MultiViewDINOv2: Spark ML Transformer that runs 3-view DINOv2 inference
(original, horizontal flip, 90-degree rotation) + augmentation consistency check per tile.
"""
from __future__ import annotations

import io

import timm
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

from utils.serde import deserialize_image

from pyspark.sql import DataFrame
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.functions import predict_batch_udf
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)


OUTPUT_SCHEMA = StructType(
    [
        StructField("embedding_orig", ArrayType(FloatType())),
        StructField("embedding_flip", ArrayType(FloatType())),
        StructField("embedding_rot90", ArrayType(FloatType())),
        StructField("logits_orig", ArrayType(FloatType())),
        StructField("max_prob", FloatType()),
        StructField("aug_consistency_overlap", IntegerType()),
        StructField("passed_aug_consistency", BooleanType()),
    ]
)


class MultiViewDINOv2(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """3-view DINOv2 embedder with augmentation consistency check."""

    modelPath = Param(
        Params._dummy(),
        "modelPath",
        "Path to the fine-tuned DINOv2 checkpoint",
        typeConverter=TypeConverters.toString,
    )
    modelName = Param(
        Params._dummy(),
        "modelName",
        "Name of the DINOv2 model",
        typeConverter=TypeConverters.toString,
    )
    batchSize = Param(
        Params._dummy(),
        "batchSize",
        "Batch size for inference",
        typeConverter=TypeConverters.toInt,
    )

    def __init__(
        self,
        input_col: str = "tile",
        output_col: str = "multiview_output",
        model_path: str = "",
        model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        batch_size: int = 8,
        jpeg_quality: int = 0,
        jpeg_subsampling: int = 1,
        aug_consistency_topk: int = 16,
        aug_consistency_threshold: int = 8,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            modelPath=model_path,
            modelName=model_name,
            batchSize=batch_size,
        )
        self.num_classes = 7806
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.jpeg_quality = jpeg_quality
        self.jpeg_subsampling = jpeg_subsampling
        self.aug_consistency_topk = aug_consistency_topk
        self.aug_consistency_threshold = aug_consistency_threshold

        self.model = (
            timm.create_model(
                self.getOrDefault(self.modelName),
                pretrained=False,
                num_classes=self.num_classes,
                checkpoint_path=self.getOrDefault(self.modelPath),
            )
            .to(self.device)
            .eval()
        )

        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        img_size = self.data_config["input_size"][-1]  # 518
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

    def _apply_jpeg_compression(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(
            buf,
            format="JPEG",
            quality=self.jpeg_quality,
            subsampling=self.jpeg_subsampling,
        )
        buf.seek(0)
        return Image.open(buf).copy()

    def _preprocess_tile(self, raw_bytes: bytes) -> torch.Tensor:
        pil_img = deserialize_image(raw_bytes)
        if self.jpeg_quality > 0:
            pil_img = self._pil_resize_crop(pil_img)
            pil_img = self._apply_jpeg_compression(pil_img)
            return self._to_tensor_normalize(pil_img)
        else:
            return self.transforms(pil_img)

    def _make_predict_fn(self):
        model = self.model
        device = self.device
        topk = self.aug_consistency_topk
        threshold = self.aug_consistency_threshold

        def predict(image_batch: np.ndarray) -> list[dict]:
            n = len(image_batch)

            # Preprocess all tiles
            orig_tensors = []
            for raw in image_batch:
                t = self._preprocess_tile(raw)
                orig_tensors.append(t)

            # Create 3 views for each tile
            all_tensors = []
            for t in orig_tensors:
                all_tensors.append(t)  # original
                all_tensors.append(torch.flip(t, [2]))  # horizontal flip (W dim)
                all_tensors.append(torch.rot90(t, 1, [1, 2]))  # 90-degree rotation

            # Single batched forward pass: [3N, C, H, W]
            batch_input = torch.stack(all_tensors).to(device)

            with torch.no_grad():
                features = model.forward_features(batch_input)  # [3N, seq, 768]
                cls_tokens = features[:, 0, :]  # [3N, 768]
                logits = model.head(cls_tokens)  # [3N, 7806]

            cls_tokens = cls_tokens.cpu().numpy()
            logits = logits.cpu().numpy()

            results = []
            for i in range(n):
                idx_orig = i * 3
                idx_flip = i * 3 + 1
                idx_rot90 = i * 3 + 2

                emb_orig = cls_tokens[idx_orig]
                emb_flip = cls_tokens[idx_flip]
                emb_rot90 = cls_tokens[idx_rot90]
                logits_orig = logits[idx_orig]
                logits_flip = logits[idx_flip]

                # Aug consistency: top-K species overlap between orig and flip
                top_orig = set(np.argsort(logits_orig)[-topk:].tolist())
                top_flip = set(np.argsort(logits_flip)[-topk:].tolist())
                overlap = len(top_orig & top_flip)
                passed = overlap >= threshold

                # Max prob from softmax of original logits
                logits_f32 = logits_orig.astype(np.float32)
                logits_f32 = logits_f32 - logits_f32.max()
                exp_logits = np.exp(logits_f32)
                probs = exp_logits / exp_logits.sum()
                max_prob = float(probs.max())

                results.append(
                    {
                        "embedding_orig": emb_orig.astype(np.float32),
                        "embedding_flip": emb_flip.astype(np.float32),
                        "embedding_rot90": emb_rot90.astype(np.float32),
                        "logits_orig": logits_orig.astype(np.float32),
                        "max_prob": max_prob,
                        "aug_consistency_overlap": int(overlap),
                        "passed_aug_consistency": bool(passed),
                    }
                )

            return results

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            predict_batch_udf(
                make_predict_fn=self._make_predict_fn,
                return_type=OUTPUT_SCHEMA,
                batch_size=self.getOrDefault(self.batchSize),
            )(self.getInputCol()),
        )
