"""Phase 4 — Student training dataset.

Reads per-image parquet from Phase 3. Each row contains packed tile embeddings
(3 views), pseudo-labels, and teacher probability vectors. The dataset applies
configurable tile filtering and returns tensors ready for collation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Tile filtering (applied per-image at __getitem__ time)
# ---------------------------------------------------------------------------

@dataclass
class TileFilterConfig:
    """Configurable tile filtering strategy applied by the dataloader.

    Strategies:
      none      — keep all aug-passing tiles (let transformer decide)
      fixed_k   — keep top-k tiles by max_prob
      threshold — keep tiles with max_prob >= min_prob
      relative  — keep tiles within alpha of the best tile's max_prob
    """
    strategy: str = "none"
    k: int = 8
    min_prob: float = 0.05
    alpha: float = 0.3
    min_tiles: int = 1


def filter_tiles(max_probs: np.ndarray, config: TileFilterConfig) -> np.ndarray:
    """Return boolean mask of tiles to keep."""
    n = len(max_probs)

    if config.strategy == "none":
        return np.ones(n, dtype=bool)

    if config.strategy == "fixed_k":
        if n <= config.k:
            return np.ones(n, dtype=bool)
        top_idx = np.argsort(max_probs)[-config.k :]
        mask = np.zeros(n, dtype=bool)
        mask[top_idx] = True
        return mask

    if config.strategy == "threshold":
        mask = max_probs >= config.min_prob
        if mask.sum() < config.min_tiles:
            top_idx = np.argsort(max_probs)[-config.min_tiles :]
            mask = np.zeros(n, dtype=bool)
            mask[top_idx] = True
        return mask

    if config.strategy == "relative":
        best = max_probs.max()
        cutoff = best * (1.0 - config.alpha)
        mask = max_probs >= cutoff
        if mask.sum() < config.min_tiles:
            top_idx = np.argsort(max_probs)[-config.min_tiles :]
            mask = np.zeros(n, dtype=bool)
            mask[top_idx] = True
        return mask

    raise ValueError(f"Unknown tile filter strategy: {config.strategy}")


# ---------------------------------------------------------------------------
# Image confidence filtering (applied at load time)
# ---------------------------------------------------------------------------

def _print_conf_histogram(max_confs: np.ndarray, threshold: float, mask: np.ndarray) -> None:
    """Print histogram of max image confidence and retention stats."""
    edges = [0.03, 0.04, 0.05, 0.07, 0.10, 0.20, 1.01]
    labels = ["[0.03,0.04)", "[0.04,0.05)", "[0.05,0.07)",
              "[0.07,0.10)", "[0.10,0.20)", "[0.20+)"]
    total = len(max_confs)
    print(f"\nImage confidence distribution (max Noisy-OR prob per image):")
    for i, label in enumerate(labels):
        count = int(((max_confs >= edges[i]) & (max_confs < edges[i + 1])).sum())
        pct = 100.0 * count / total
        marker = " <-- cutoff" if edges[i] <= threshold < edges[i + 1] else ""
        print(f"  {label}: {count:6d} ({pct:5.1f}%){marker}")
    kept = int(mask.sum())
    print(f"  Filter: min_image_conf={threshold:.3f} -> kept {kept}/{total} "
          f"({100.0 * kept / total:.1f}%)\n")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PseudoLabelDataset(Dataset):
    """PyTorch Dataset over Phase 3 pseudo-label parquet.

    Loads the entire parquet into memory (~1.5-2 GB for 77k LUCAS images).
    Each __getitem__ unpacks binary blobs, applies tile filtering, and
    returns numpy-backed tensors.
    """

    def __init__(
        self,
        parquet_path: str,
        tile_filter: TileFilterConfig | None = None,
        num_classes: int = 7806,
        min_image_conf: float = 0.0,
    ):
        self.num_classes = num_classes
        self.tile_filter = tile_filter or TileFilterConfig()

        table = pq.read_table(parquet_path)
        self.df = table.to_pandas()

        # Filter out low-confidence images based on max teacher probability
        if min_image_conf > 0:
            max_confs = np.array([
                float(np.frombuffer(buf, dtype=np.float16).max())
                for buf in self.df["teacher_probs_f16"]
            ], dtype=np.float32)
            mask = max_confs >= min_image_conf
            _print_conf_histogram(max_confs, min_image_conf, mask)
            self.df = self.df[mask].reset_index(drop=True)

        self.length = len(self.df)

    def __len__(self) -> int:
        return self.length

    def _unpack_embeddings(self, buf: bytes, num_tiles: int) -> np.ndarray:
        arr = np.frombuffer(buf, dtype=np.float16).reshape(num_tiles, EMBEDDING_DIM)
        return arr.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        num_tiles = int(row["num_tiles"])

        # Unpack tile embeddings (3 views)
        emb_orig = self._unpack_embeddings(row["tile_embeddings_orig_f16"], num_tiles)
        emb_flip = self._unpack_embeddings(row["tile_embeddings_flip_f16"], num_tiles)
        emb_rot90 = self._unpack_embeddings(row["tile_embeddings_rot90_f16"], num_tiles)

        # Unpack tile metadata
        tile_max_probs = np.frombuffer(
            row["tile_max_probs_f32"], dtype=np.float32
        ).copy()

        # Apply tile filtering
        keep = filter_tiles(tile_max_probs, self.tile_filter)
        emb_orig = emb_orig[keep]
        emb_flip = emb_flip[keep]
        emb_rot90 = emb_rot90[keep]
        num_kept = int(keep.sum())

        # Unpack pseudo-labels
        pseudo_labels = np.frombuffer(
            row["pseudo_labels_i32"], dtype=np.int32
        ).copy()

        # Unpack teacher probs
        teacher_probs = np.frombuffer(
            row["teacher_probs_f16"], dtype=np.float16
        ).astype(np.float32)

        return {
            "tile_embeddings_orig": torch.from_numpy(emb_orig),    # [N, 768]
            "tile_embeddings_flip": torch.from_numpy(emb_flip),    # [N, 768]
            "tile_embeddings_rot90": torch.from_numpy(emb_rot90),  # [N, 768]
            "num_tiles": num_kept,
            "pseudo_labels": torch.from_numpy(pseudo_labels),      # [L]
            "teacher_probs": torch.from_numpy(teacher_probs),      # [C]
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length tile sequences to batch max and build attention mask."""
    batch_size = len(batch)
    max_tiles = max(item["num_tiles"] for item in batch)
    num_classes = batch[0]["teacher_probs"].shape[0]

    tile_orig = torch.zeros(batch_size, max_tiles, EMBEDDING_DIM)
    tile_flip = torch.zeros(batch_size, max_tiles, EMBEDDING_DIM)
    tile_rot90 = torch.zeros(batch_size, max_tiles, EMBEDDING_DIM)
    attention_mask = torch.zeros(batch_size, max_tiles, dtype=torch.bool)
    teacher_probs = torch.zeros(batch_size, num_classes)

    pseudo_labels_list = []

    for i, item in enumerate(batch):
        n = item["num_tiles"]
        tile_orig[i, :n] = item["tile_embeddings_orig"]
        tile_flip[i, :n] = item["tile_embeddings_flip"]
        tile_rot90[i, :n] = item["tile_embeddings_rot90"]
        attention_mask[i, :n] = True
        teacher_probs[i] = item["teacher_probs"]
        pseudo_labels_list.append(item["pseudo_labels"])

    return {
        "tile_embeddings_orig": tile_orig,      # [B, max_N, 768]
        "tile_embeddings_flip": tile_flip,      # [B, max_N, 768]
        "tile_embeddings_rot90": tile_rot90,    # [B, max_N, 768]
        "attention_mask": attention_mask,         # [B, max_N]
        "teacher_probs": teacher_probs,           # [B, C]
        "pseudo_labels": pseudo_labels_list,      # list of [L_i] tensors
    }
