"""Image tiling utilities for grid-based inference."""

from PIL import Image


def split_into_tiles(image: Image.Image, grid_size: int) -> list[Image.Image]:
    """Split an image into grid_size x grid_size non-overlapping tiles.

    Tiles are produced row-by-row, left-to-right. Any remainder pixels
    from non-divisible dimensions are discarded (right/bottom edges).

    Args:
        image: Input PIL image.
        grid_size: Number of divisions along each axis (e.g., 3 -> 3x3 = 9 tiles).

    Returns:
        List of PIL images, length = grid_size^2.
    """
    w, h = image.size
    tile_w = w // grid_size
    tile_h = h // grid_size
    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w
            lower = upper + tile_h
            tiles.append(image.crop((left, upper, right, lower)))
    return tiles
