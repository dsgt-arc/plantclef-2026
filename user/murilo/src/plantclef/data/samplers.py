"""Pseudo-quadrat samplers for the TileQ-Decoder (spec §2.3).

Two samplers, both producing ``(tile_cls (16, 768) float32, label (7806,) float32)``:

- **Sampler A** (``SamplerA``) — naive random-background. Draws ``n_positives``
  species without replacement and ``n_tiles - n_positives`` background species
  with replacement (all from the train CLS pool, weighted by
  ``w_s ∝ 1/sqrt(count)``). Backgrounds are labelled negative even though they
  are real plant CLSs of other species; ASL's hard margin absorbs the noise.

- **Sampler B** (``SamplerB``) — test-anchored, FAISS-relabelled. Starts from a
  real test quadrat's 16 tile CLSs, overwrites ``n_positives`` random slots
  with train-CLS injections, and *adds* the FAISS-discovered species labels of
  the surviving (untouched) tiles to the multi-hot target. The test-anchored
  starting point is what makes H2 distinguishable from H1.

All embeddings are read **as-is** (not L2-normalised); the decoder's input
projection learns the appropriate scale (see CONTEXT.md and spec §2.3 common
parameters). L2 normalisation is only applied for FAISS retrieval (§3.4).
"""

from __future__ import annotations

import numpy as np


class SamplerA:
    """Naive random-background pseudo-quadrat sampler.

    All sampling is class_idx-indexed (0..n_classes-1). Conversion to/from
    Pl@ntNet species_id only happens at I/O boundaries (build_train_lookup
    persists ``class_to_species_id`` in the npz).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        offsets: np.ndarray,
        species_weights: np.ndarray,
        n_positives: int = 5,
        n_tiles: int = 16,
        rng: np.random.Generator | None = None,
    ):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}")
        if embeddings.dtype != np.float32:
            raise ValueError(f"embeddings dtype must be float32, got {embeddings.dtype}")
        if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != embeddings.shape[0]:
            raise ValueError(
                f"offsets must be 1-D with offsets[0]=0 and "
                f"offsets[-1]={embeddings.shape[0]}, got {offsets[:3].tolist()}..."
                f"{offsets[-1].item()}"
            )
        n_classes = len(offsets) - 1
        if species_weights.shape != (n_classes,):
            raise ValueError(
                f"species_weights shape {species_weights.shape} doesn't match "
                f"n_classes={n_classes} from offsets"
            )
        if n_positives < 1 or n_positives > n_tiles:
            raise ValueError(
                f"n_positives must be in [1, n_tiles={n_tiles}], got {n_positives}"
            )
        if n_positives > n_classes:
            raise ValueError(
                f"n_positives={n_positives} exceeds n_classes={n_classes} "
                f"(can't draw without replacement)"
            )
        # np.random.choice requires float64 + sum exactly 1.
        weights64 = species_weights.astype(np.float64, copy=True)
        weights64 /= weights64.sum()

        self.embeddings = embeddings
        self.offsets = offsets
        self.weights = weights64
        self.n_classes = n_classes
        self.n_positives = n_positives
        self.n_tiles = n_tiles
        self.embedding_dim = embeddings.shape[1]
        self.rng = rng if rng is not None else np.random.default_rng()

    def _draw_one_cls_for_class(self, class_idx: int) -> np.ndarray:
        """Sample one CLS uniformly from class_idx's pool. Returns (768,) view."""
        start = int(self.offsets[class_idx])
        end = int(self.offsets[class_idx + 1])
        row = start + int(self.rng.integers(0, end - start))
        return self.embeddings[row]

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw one pseudo-quadrat.

        Returns
        -------
        tiles : float32 (n_tiles, 768) — shuffled mix of positive + background CLSs
        label : float32 (n_classes,)   — multi-hot, exactly n_positives ones
        """
        pos_classes = self.rng.choice(
            self.n_classes, size=self.n_positives, replace=False, p=self.weights
        )
        bg_classes = self.rng.choice(
            self.n_classes,
            size=self.n_tiles - self.n_positives,
            replace=True,
            p=self.weights,
        )
        # Concatenate then shuffle so positives aren't always at slots 0..n_positives-1
        all_classes = np.concatenate([pos_classes, bg_classes])
        perm = self.rng.permutation(self.n_tiles)
        all_classes = all_classes[perm]

        tiles = np.empty((self.n_tiles, self.embedding_dim), dtype=np.float32)
        for i, c in enumerate(all_classes):
            tiles[i] = self._draw_one_cls_for_class(int(c))

        label = np.zeros(self.n_classes, dtype=np.float32)
        label[pos_classes] = 1.0
        return tiles, label


class SamplerB:
    """Test-anchored, FAISS-relabelled pseudo-quadrat sampler (spec §2.3).

    Each call:
      1. Picks one of ``n_quadrats`` test images uniformly at random.
      2. Starts with that image's ``(n_tiles, 768)`` test CLS tensor.
      3. Picks ``n_positives`` species (without replacement, weighted) and
         overwrites ``n_positives`` random tile slots with one CLS each drawn
         from the train pool of those species.
      4. Builds the multi-hot label as the union of:
           - the injected species (always positives),
           - the FAISS-discovered class_idx of every *surviving* (non-overwritten)
             tile whose ``tile_to_class_idx[q, t] >= 0``.

    Backgrounds whose tiles got overwritten are dropped from the label set
    (since the tile no longer carries that species' CLS).

    Inputs are class_idx-indexed throughout. ``tile_to_class_idx[q, t] == -1``
    means "unmatched / true background" (no label contribution).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        offsets: np.ndarray,
        species_weights: np.ndarray,
        test_cls: np.ndarray,
        tile_to_class_idx: np.ndarray,
        n_positives: int = 5,
        n_tiles: int = 16,
        rng: np.random.Generator | None = None,
    ):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}")
        if embeddings.dtype != np.float32:
            raise ValueError(f"embeddings dtype must be float32, got {embeddings.dtype}")
        if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != embeddings.shape[0]:
            raise ValueError(
                f"offsets must be 1-D with offsets[0]=0 and "
                f"offsets[-1]={embeddings.shape[0]}, got first={int(offsets[0])} "
                f"last={int(offsets[-1])}"
            )
        n_classes = len(offsets) - 1
        if species_weights.shape != (n_classes,):
            raise ValueError(
                f"species_weights shape {species_weights.shape} doesn't match "
                f"n_classes={n_classes} from offsets"
            )
        if test_cls.ndim != 3 or test_cls.shape[1] != n_tiles:
            raise ValueError(
                f"test_cls must be (n_quadrats, n_tiles={n_tiles}, dim), "
                f"got {test_cls.shape}"
            )
        if test_cls.dtype != np.float32:
            raise ValueError(f"test_cls dtype must be float32, got {test_cls.dtype}")
        if test_cls.shape[2] != embeddings.shape[1]:
            raise ValueError(
                f"test_cls dim {test_cls.shape[2]} != train dim {embeddings.shape[1]}"
            )
        if tile_to_class_idx.shape != test_cls.shape[:2]:
            raise ValueError(
                f"tile_to_class_idx shape {tile_to_class_idx.shape} must match "
                f"test_cls.shape[:2] {test_cls.shape[:2]}"
            )
        if n_positives < 1 or n_positives > n_tiles:
            raise ValueError(
                f"n_positives must be in [1, n_tiles={n_tiles}], got {n_positives}"
            )
        if n_positives > n_classes:
            raise ValueError(
                f"n_positives={n_positives} exceeds n_classes={n_classes}"
            )

        weights64 = species_weights.astype(np.float64, copy=True)
        weights64 /= weights64.sum()

        self.embeddings = embeddings
        self.offsets = offsets
        self.weights = weights64
        self.test_cls = test_cls
        self.tile_to_class_idx = tile_to_class_idx
        self.n_classes = n_classes
        self.n_quadrats = test_cls.shape[0]
        self.n_positives = n_positives
        self.n_tiles = n_tiles
        self.embedding_dim = embeddings.shape[1]
        self.rng = rng if rng is not None else np.random.default_rng()

    def _draw_one_cls_for_class(self, class_idx: int) -> np.ndarray:
        start = int(self.offsets[class_idx])
        end = int(self.offsets[class_idx + 1])
        row = start + int(self.rng.integers(0, end - start))
        return self.embeddings[row]

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw one test-anchored pseudo-quadrat.

        Returns
        -------
        tiles : float32 (n_tiles, 768)
        label : float32 (n_classes,)  — multi-hot; ones >= n_positives (background
                relabels add to it; overwritten background labels are dropped)
        """
        q = int(self.rng.integers(0, self.n_quadrats))
        # Copy the test quadrat's CLSs (we'll overwrite some slots in place)
        tiles = self.test_cls[q].copy()  # (n_tiles, embedding_dim) float32

        injected_classes = self.rng.choice(
            self.n_classes, size=self.n_positives, replace=False, p=self.weights
        )
        inject_slots = self.rng.choice(
            self.n_tiles, size=self.n_positives, replace=False
        )
        for slot, class_idx in zip(inject_slots, injected_classes):
            tiles[slot] = self._draw_one_cls_for_class(int(class_idx))

        # Build the label set
        injected_set = set(int(c) for c in injected_classes)

        # Background labels = the FAISS-discovered class_idxs on any tile
        # whose tile_to_class_idx is >= 0 AND whose slot was NOT overwritten.
        q_labels = self.tile_to_class_idx[q]  # (n_tiles,)
        inject_slot_set = set(int(s) for s in inject_slots)
        surviving_bg = set()
        for t in range(self.n_tiles):
            if t in inject_slot_set:
                continue
            c = int(q_labels[t])
            if c >= 0:
                surviving_bg.add(c)

        # Union: injected always wins (set deduplicates a coincident bg label).
        all_positives = injected_set | surviving_bg
        label = np.zeros(self.n_classes, dtype=np.float32)
        for c in all_positives:
            label[c] = 1.0
        return tiles, label
