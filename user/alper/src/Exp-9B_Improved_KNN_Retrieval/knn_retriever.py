"""kNN retriever module using FAISS for species vote computation.

Loads a pre-built FAISS IndexFlatIP and parallel labels array. For each query
embedding, retrieves k nearest neighbors and computes distance-weighted species
vote probabilities.

Copied from Exp-9 unchanged — the retriever is embedding-dim agnostic. In
Exp-9B it is fed 512-d ArcFace-projected embeddings (queried against a 512-d
FAISS index) instead of Exp-9's 768-d frozen-DINOv2 embeddings.

Supports:
  - Inverse frequency weighting (rare species votes count more)
"""
from __future__ import annotations

import faiss
import numpy as np


class KNNRetriever:
    """FAISS-based kNN retriever with distance-weighted voting."""

    def __init__(
        self,
        index_path: str,
        labels_path: str,
        k: int = 20,
        tau: float = 0.07,
        num_classes: int = 7806,
        freq_weights_path: str | None = None,
    ) -> None:
        self.index = faiss.read_index(index_path)
        self.labels = np.load(labels_path).astype(np.int32)
        self.k = k
        self.tau = tau
        self.num_classes = num_classes
        self.dim = self.index.d

        if freq_weights_path is not None:
            class_counts = np.load(freq_weights_path).astype(np.float32)
            class_counts = np.maximum(class_counts, 1.0)
            self.inv_freq = 1.0 / np.sqrt(class_counts)
            self.inv_freq *= self.num_classes / self.inv_freq.sum()
        else:
            self.inv_freq = None

    def query_batch(
        self,
        embeddings: np.ndarray,
        k: int | None = None,
        tau: float | None = None,
        use_inv_freq: bool = False,
    ) -> np.ndarray:
        """Query FAISS index and return kNN vote probabilities.

        Args:
            embeddings: [N, D] float32 (raw; L2-normalized internally)
            k: override self.k for this query (optional)
            tau: override self.tau for this query (optional)
            use_inv_freq: if True, weight votes by inverse class frequency

        Returns:
            p_knn: [N, num_classes] float32 probability vectors
        """
        k = k if k is not None else self.k
        tau = tau if tau is not None else self.tau

        N = embeddings.shape[0]
        if N == 0:
            return np.zeros((0, self.num_classes), dtype=np.float32)

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Query dim {embeddings.shape[1]} does not match index dim {self.dim}"
            )

        emb = embeddings.astype(np.float32, copy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb /= norms + 1e-8

        similarities, indices = self.index.search(emb, k)

        sim_shifted = similarities - np.max(similarities, axis=1, keepdims=True)
        weights = np.exp(sim_shifted / tau)
        weights /= weights.sum(axis=1, keepdims=True)

        p_knn = np.zeros((N, self.num_classes), dtype=np.float32)
        for i in range(N):
            valid = indices[i] >= 0
            neighbor_labels = self.labels[indices[i, valid]]
            np.add.at(p_knn[i], neighbor_labels, weights[i, valid])

        if use_inv_freq and self.inv_freq is not None:
            p_knn *= self.inv_freq[np.newaxis, :]
            row_sums = p_knn.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-8)
            p_knn /= row_sums

        return p_knn
