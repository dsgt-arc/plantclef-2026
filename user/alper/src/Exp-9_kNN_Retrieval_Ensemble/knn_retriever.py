"""kNN retriever module using FAISS for species vote computation.

Loads a pre-built FAISS IndexFlatIP and parallel labels array.
For each query embedding, retrieves k nearest neighbors and computes
distance-weighted species vote probabilities.
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
    ) -> None:
        self.index = faiss.read_index(index_path)
        self.labels = np.load(labels_path).astype(np.int32)
        self.k = k
        self.tau = tau
        self.num_classes = num_classes

    def query_batch(
        self,
        embeddings: np.ndarray,
        k: int | None = None,
        tau: float | None = None,
    ) -> np.ndarray:
        """Query FAISS index and return kNN vote probabilities.

        Args:
            embeddings: [N, 768] float32 (raw, will be L2-normalized internally)
            k: override self.k for this query (optional)
            tau: override self.tau for this query (optional)

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

        return p_knn
