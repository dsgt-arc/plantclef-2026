"""Vegetation density via ExG + HSV green-fraction.

Per JPEG: resize -> central crop 75% -> ExG mask (Otsu) and HSV
green-range mask -> density = 0.5*exg.mean() + 0.5*hsv.mean(). The
central crop drops the wooden frame visible on RNNB/GUARDEN that would
otherwise saturate the ExG signal on barren quadrats.

Output: density.parquet with columns [quadrat_id, density,
density_exg, density_hsv].

Local CPU. Parallel via multiprocessing. ~10-20 min for 2,105 images
on 8 cores. PIL-only — no cv2 dependency.
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

CROP_FRAC = 0.75   # central 75% by side -> 0.5625 of area
RESIZE_TO = 512


def _otsu(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    if finite.std() < 1e-6:
        return float(finite.mean())
    hist, edges = np.histogram(finite, bins=256)
    total = finite.size
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    sum_total = (hist * midpoints).sum()
    sum_b, w_b, var_max, threshold = 0.0, 0, -1.0, edges[0]
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += hist[i] * midpoints[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = midpoints[i]
    return float(threshold)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """rgb [H,W,3] uint8 -> hsv [H,W,3] float32 with H in [0,360], S,V in [0,1]."""
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    cmax = f.max(axis=-1)
    cmin = f.min(axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 1e-9
    rmax = mask & (cmax == r)
    gmax = mask & (cmax == g)
    bmax = mask & (cmax == b)
    h[rmax] = 60.0 * (((g[rmax] - b[rmax]) / delta[rmax]) % 6.0)
    h[gmax] = 60.0 * (((b[gmax] - r[gmax]) / delta[gmax]) + 2.0)
    h[bmax] = 60.0 * (((r[bmax] - g[bmax]) / delta[bmax]) + 4.0)

    s = np.where(cmax > 1e-9, delta / np.maximum(cmax, 1e-9), 0.0)
    v = cmax
    return np.stack([h, s, v], axis=-1)


def density_one(path: str) -> Tuple[str, float, float, float]:
    qid = os.path.splitext(os.path.basename(path))[0]
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return (qid, 0.0, 0.0, 0.0)

    # Resize to 512 on shorter side
    w, h = img.size
    s = RESIZE_TO / float(min(w, h))
    img = img.resize((int(round(w * s)), int(round(h * s))), Image.LANCZOS)

    # Central crop
    W, H = img.size
    cw, ch = int(W * CROP_FRAC), int(H * CROP_FRAC)
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    crop = np.asarray(img.crop((x0, y0, x0 + cw, y0 + ch)))  # [H,W,3] uint8

    # ExG = 2G - R - B (signed). On bare-sand patches the distribution is
    # unimodal near zero; Otsu would carve noise. We use a fixed positive
    # threshold (20 on a 0-255 channel scale) which separates green vegetation
    # from neutral surfaces robustly. ExG diagnostic kept for inspection.
    f = crop.astype(np.float32)
    exg = 2.0 * f[..., 1] - f[..., 0] - f[..., 2]
    exg_mask = (exg > 20.0)
    density_exg = float(exg_mask.mean())

    # HSV green-range. Saturation >= 0.15 filters bare-sand pixels (which
    # may have positive hue values but very low saturation).
    hsv = _rgb_to_hsv(crop)
    h_ch, s_ch, v_ch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsv_mask = (h_ch >= 40.0) & (h_ch <= 160.0) & (s_ch >= 0.15) & (v_ch >= 0.15)
    density_hsv = float(hsv_mask.mean())

    # AND of the two: a pixel counts as vegetation only if BOTH masks fire.
    # This is robust to Otsu unimodal artifacts and to HSV hue alone (sand
    # may have green-ish hue but low saturation; AND eliminates that).
    veg_mask = exg_mask & hsv_mask
    density = float(veg_mask.mean())
    return (qid, density, density_exg, density_hsv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths: List[str] = sorted(
        str(p) for p in Path(args.images_dir).iterdir()
        if p.suffix.lower() == args.ext
    )
    if args.limit:
        paths = paths[: args.limit]
    print(f"  {len(paths)} images, {args.workers} workers")

    if args.workers > 1:
        with Pool(args.workers) as pool:
            rows = pool.map(density_one, paths)
    else:
        rows = [density_one(p) for p in paths]

    df = pd.DataFrame(rows, columns=["quadrat_id", "density",
                                      "density_exg", "density_hsv"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"  wrote {len(df)} rows -> {args.out}")
    print(f"  density:     min={df.density.min():.3f} mean={df.density.mean():.3f} "
          f"max={df.density.max():.3f}")
    print(f"  density_exg: mean={df.density_exg.mean():.3f}")
    print(f"  density_hsv: mean={df.density_hsv.mean():.3f}")


if __name__ == "__main__":
    main()
