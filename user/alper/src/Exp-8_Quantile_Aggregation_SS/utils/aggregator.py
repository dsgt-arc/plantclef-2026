# utils/aggregator.py
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


class _BaseAggregator:
    """Shared init and helpers for alternative aggregation methods."""

    def __init__(
        self,
        num_classes: int = 7806,
        temperature: float = 2.0,
        denoise_eps: float = 1e-6,
        output_dtype: np.dtype = np.float16,
        image_topk: Optional[int] = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.denoise_eps = float(denoise_eps)
        self.output_dtype = output_dtype
        self.image_topk = int(image_topk) if image_topk is not None else None

    def _softmax(self, logits_2d: np.ndarray) -> np.ndarray:
        x = logits_2d.astype(np.float32, copy=False) / self.temperature
        x = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x, dtype=np.float32)
        return (ex / np.sum(ex, axis=1, keepdims=True, dtype=np.float32)).astype(
            np.float32, copy=False
        )

    def _postprocess(self, p_image: np.ndarray) -> np.ndarray:
        np.clip(p_image, 0.0, 1.0, out=p_image)
        if self.denoise_eps > 0:
            p_image[p_image < self.denoise_eps] = 0.0
        if self.image_topk is not None:
            k = min(self.image_topk, p_image.shape[0])
            if k < p_image.shape[0]:
                idx = np.argpartition(p_image, -k)[-k:]
                mask = np.zeros(p_image.shape[0], dtype=np.bool_)
                mask[idx] = True
                p_image[~mask] = 0.0
        return p_image.astype(self.output_dtype, copy=False)


class MeanProbabilityAggregator(_BaseAggregator):
    """P(c) = mean_t softmax(logit_{t,c} / T). Simple average of per-tile probabilities."""

    def aggregate(self, tile_logits_2d: np.ndarray) -> np.ndarray:
        probs = self._softmax(tile_logits_2d)  # [T, C]
        p_image = probs.mean(axis=0).astype(np.float32)
        return self._postprocess(p_image)


class MaxProbabilityAggregator(_BaseAggregator):
    """P(c) = max_t softmax(logit_{t,c} / T). One confident tile is enough."""

    def aggregate(self, tile_logits_2d: np.ndarray) -> np.ndarray:
        probs = self._softmax(tile_logits_2d)  # [T, C]
        p_image = probs.max(axis=0).astype(np.float32)
        return self._postprocess(p_image)


class TopMProbabilityAggregator(_BaseAggregator):
    """P(c) = mean of top-m tile probabilities for class c.

    Generalizes max_prob (m=1) and mean_prob (m=T).
    m=2 or m=3 uses the strongest evidence while getting confirmation
    from additional tiles, without diluting with noise from all tiles.
    """

    def __init__(self, m: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        if m < 1:
            raise ValueError("m must be >= 1")
        self.m = int(m)

    def aggregate(self, tile_logits_2d: np.ndarray) -> np.ndarray:
        probs = self._softmax(tile_logits_2d)  # [T, C]
        t = probs.shape[0]
        m = min(self.m, t)
        if m == 1:
            p_image = probs.max(axis=0).astype(np.float32)
        elif m >= t:
            p_image = probs.mean(axis=0).astype(np.float32)
        else:
            # Partition to find top-m per class, then average them
            # np.partition along axis=0: for each class, get top-m tile probs
            partitioned = np.partition(probs, -m, axis=0)[-m:]  # [m, C]
            p_image = partitioned.mean(axis=0).astype(np.float32)
        return self._postprocess(p_image)


class QuantileAggregator(_BaseAggregator):
    """P(c) = quantile_q over tiles of softmax(logit_{t,c} / T).

    Generalizes max_prob (q=1.0) and median (q=0.5).
    q=0.9 keeps most of the max_prob advantage (one confident tile drives
    the score) while being robust to a single spurious high-confidence tile.
    """

    def __init__(self, quantile: float = 0.9, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        self.quantile = float(quantile)

    def aggregate(self, tile_logits_2d: np.ndarray) -> np.ndarray:
        probs = self._softmax(tile_logits_2d)  # [T, C]
        p_image = np.quantile(probs, self.quantile, axis=0).astype(np.float32)
        return self._postprocess(p_image)


class WeightedNoisyORAggregator(_BaseAggregator):
    """P(c) = 1 - prod_t(1 - w_t * p_{t,c}) where w_t = max_prob_t / sum(max_prob)."""

    def __init__(self, clip_probs: float = 1e-7, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_probs = float(clip_probs)

    def aggregate(self, tile_logits_2d: np.ndarray) -> np.ndarray:
        probs = self._softmax(tile_logits_2d)  # [T, C]
        max_probs = probs.max(axis=1)  # [T]
        weights = max_probs / (max_probs.sum() + 1e-12)  # [T], sums to 1

        weighted = weights[:, None] * probs  # [T, C]
        one_minus = 1.0 - weighted
        np.clip(one_minus, self.clip_probs, 1.0, out=one_minus)
        log_prod = np.sum(np.log(one_minus, dtype=np.float32), axis=0, dtype=np.float32)
        p_image = (-np.expm1(log_prod)).astype(np.float32, copy=False)
        return self._postprocess(p_image)


def create_aggregator(
    method: str,
    num_classes: int = 7806,
    temperature: float = 2.0,
    denoise_eps: float = 1e-6,
    image_topk: Optional[int] = None,
    output_dtype: np.dtype = np.float16,
    agg_m: int = 2,
    agg_quantile: float = 0.9,
) -> object:
    """Factory for aggregator by name.

    Args:
        agg_m: For top_m_prob method, the number of top tiles to average per class.
        agg_quantile: For quantile method, the quantile value (0.0-1.0).
    """
    common = dict(
        num_classes=num_classes,
        temperature=temperature,
        denoise_eps=denoise_eps,
        image_topk=image_topk,
        output_dtype=output_dtype,
    )
    if method == "noisy_or":
        return TemperatureNoisyORAggregator(
            **common, clip_probs=1e-7, return_stats=False
        )
    elif method == "weighted_noisy_or":
        return WeightedNoisyORAggregator(clip_probs=1e-7, **common)
    elif method == "mean_prob":
        return MeanProbabilityAggregator(**common)
    elif method == "max_prob":
        return MaxProbabilityAggregator(**common)
    elif method == "top_m_prob":
        return TopMProbabilityAggregator(m=agg_m, **common)
    elif method == "quantile":
        return QuantileAggregator(quantile=agg_quantile, **common)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
