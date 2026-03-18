"""kNN retriever module using FAISS for species vote computation.

Loads a pre-built FAISS IndexFlatIP and parallel labels array.
For each query embedding, retrieves k nearest neighbors and computes
distance-weighted species vote probabilities.

Supports:
  - Inverse frequency weighting (rare species votes count more)
  - Confidence gating (only use kNN when softmax is uncertain)
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

        # Inverse frequency weights: 1/sqrt(count) per class, normalized
        if freq_weights_path is not None:
            class_counts = np.load(freq_weights_path).astype(np.float32)  # [num_classes]
            # Avoid division by zero for classes with 0 images
            class_counts = np.maximum(class_counts, 1.0)
            self.inv_freq = 1.0 / np.sqrt(class_counts)
            # Normalize so weights sum to num_classes (preserves scale)
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
            embeddings: [N, 768] float32 (raw, will be L2-normalized internally)
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

        # L2-normalize queries (index vectors are already normalized)
        emb = embeddings.astype(np.float32, copy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb /= norms + 1e-8

        # FAISS search — inner product on L2-normed = cosine similarity
        similarities, indices = self.index.search(emb, k)
        # similarities: [N, k], indices: [N, k]

        # Distance-weighted voting via softmax over similarities
        # Subtract max for numerical stability
        sim_shifted = similarities - np.max(similarities, axis=1, keepdims=True)
        weights = np.exp(sim_shifted / tau)  # [N, k]
        weights /= weights.sum(axis=1, keepdims=True)  # normalize per query

        # Scatter-add weighted votes into class probabilities
        p_knn = np.zeros((N, self.num_classes), dtype=np.float32)
        for i in range(N):
            valid = indices[i] >= 0  # FAISS returns -1 for missing
            neighbor_labels = self.labels[indices[i, valid]]
            np.add.at(p_knn[i], neighbor_labels, weights[i, valid])

        # Apply inverse frequency weighting
        if use_inv_freq and self.inv_freq is not None:
            p_knn *= self.inv_freq[np.newaxis, :]  # [N, C] * [1, C]
            # Re-normalize each query's votes to sum to 1
            row_sums = p_knn.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-8)
            p_knn /= row_sums

        return p_knn
