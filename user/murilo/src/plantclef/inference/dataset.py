"""Datasets for inference with grid tiling."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from plantclef.tiling import split_into_tiles


class TileDataset(Dataset):
    """Image-level dataset that loads each image once and returns all its tiles.

    Each item is (img_idx, tile_tensors) where tile_tensors has shape
    (tiles_per_image, C, H, W). The image is loaded once and tiled on the CPU
    worker, avoiding redundant NFS reads for the same image.

    With a DataLoader, a batch of B images produces a tensor of shape
    (B, tiles_per_image, C, H, W) which gets reshaped to
    (B * tiles_per_image, C, H, W) before feeding to the model.
    """

    def __init__(self, image_paths: list[Path], grid_size: int, transform):
        self.image_paths = image_paths
        self.grid_size = grid_size
        self.transform = transform
        self.tiles_per_image = grid_size * grid_size if grid_size > 0 else 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.grid_size > 0:
            tiles = split_into_tiles(img, self.grid_size)
        else:
            tiles = [img]
        tile_tensors = torch.stack([self.transform(t) for t in tiles])
        return idx, tile_tensors  # (tiles_per_image, C, H, W)
