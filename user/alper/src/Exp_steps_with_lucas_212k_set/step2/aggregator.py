# src/step2/aggregator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AggregationStats:
    num_tiles: int
    num_classes: int
    image_topk: int
    num_nonzero_after_topk: int


class TemperatureNoisyORAggregator:
    """
    Tile logits [T, C] -> image probabilities [C] using:
      1) temperature softmax per tile over all classes
      2) Noisy OR across tiles: p_c = 1 - prod_t (1 - p_{t,c})

    Optional image_topk:
      After aggregation, keep only the TopK classes at image level and set the rest to 0.
      This preserves the meaning of probabilities and keeps vectors sparse for storage.

    Output is a dense length C vector, but if image_topk is set, most entries are exactly 0.
    """

    def __init__(
        self,
        num_classes: int = 7806,
        temperature: float = 2.0,
        denoise_eps: float = 1e-6,
        clip_probs: float = 1e-7,
        output_dtype: np.dtype = np.float16,
        image_topk: Optional[int] = None,
        return_stats: bool = False,
    ) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if denoise_eps < 0:
            raise ValueError("denoise_eps must be non negative")
        if clip_probs <= 0:
            raise ValueError("clip_probs must be positive")
        if image_topk is not None and image_topk <= 0:
            raise ValueError("image_topk must be positive when provided")

        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.denoise_eps = float(denoise_eps)
        self.clip_probs = float(clip_probs)
        self.output_dtype = output_dtype
        self.image_topk = int(image_topk) if image_topk is not None else None
        self.return_stats = bool(return_stats)

    def _softmax_full(self, logits_2d: np.ndarray) -> np.ndarray:
        x = logits_2d / self.temperature
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        denom = np.sum(ex, axis=1, keepdims=True, dtype=np.float32)
        probs = ex / denom
        return probs.astype(np.float32, copy=False)

    def _apply_image_topk_inplace(self, p_image: np.ndarray, k: int) -> int:
        c = p_image.shape[0]
        if k >= c:
            return int(np.count_nonzero(p_image))

        idx = np.argpartition(p_image, -k)[-k:]
        mask = np.zeros((c,), dtype=np.bool_)
        mask[idx] = True
        p_image[~mask] = 0.0
        return int(np.count_nonzero(p_image))

    def aggregate(self, tile_logits_2d: np.ndarray) -> np.ndarray:
        out = self.aggregate_with_stats(tile_logits_2d)
        if self.return_stats:
            return out  # type: ignore[return-value]
        return out[0]  # type: ignore[return-value]

    def aggregate_with_stats(self, tile_logits_2d: np.ndarray) -> Tuple[np.ndarray, AggregationStats]:
        if tile_logits_2d.ndim != 2:
            raise ValueError("tile_logits_2d must have shape [T, C]")
        if tile_logits_2d.shape[1] != self.num_classes:
            raise ValueError(f"Expected C {self.num_classes} but got {tile_logits_2d.shape[1]}")

        tile_logits_2d = tile_logits_2d.astype(np.float32, copy=False)
        t = int(tile_logits_2d.shape[0])
        c = self.num_classes

        probs = self._softmax_full(tile_logits_2d)  # [T, C]
        one_minus = 1.0 - probs
        np.clip(one_minus, self.clip_probs, 1.0, out=one_minus)

        log_prod = np.sum(np.log(one_minus, dtype=np.float32), axis=0, dtype=np.float32)

        p_image = (-np.expm1(log_prod)).astype(np.float32, copy=False)
        np.clip(p_image, 0.0, 1.0, out=p_image)

        if self.denoise_eps > 0:
            p_image[p_image < self.denoise_eps] = 0.0

        nnz_after_topk = int(np.count_nonzero(p_image))
        if self.image_topk is not None:
            k = min(self.image_topk, c)
            nnz_after_topk = self._apply_image_topk_inplace(p_image, k)

        p_image = p_image.astype(self.output_dtype, copy=False)

        stats = AggregationStats(
            num_tiles=t,
            num_classes=c,
            image_topk=(self.image_topk if self.image_topk is not None else c),
            num_nonzero_after_topk=nnz_after_topk,
        )
        return p_image, stats
