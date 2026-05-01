"""Sample test images and characterize properties that could motivate
specific test-time augmentations.

For each source group, sample N images and compute:
  - resolution (HxW), aspect ratio
  - mean RGB and per-channel std
  - HSV stats: mean saturation, value, hue distribution
  - per-image entropy proxy (laplacian variance = sharpness)
  - estimated vegetation fraction (HSV-green range AND ExG threshold)
  - ruler/measuring-tape detection: are bright bands present at borders?
  - shadow / lighting variance
  - is the image rotated (looking at quadrat-frame orientation if visible)
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

IMG_DIR = r"C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\images"
OUT_DIR = (r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub"
           r"\plantclef-2026\user\alper\src\Exp-16_Multi_Species_Discovery"
           r"\outputs\image_audit")

SOURCES = ("RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix",
           "2024-CEV3")
N_PER_SOURCE = 30


def source_of(name: str) -> str:
    for p in SOURCES:
        if name.startswith(p):
            return p
    return "OTHER"


def laplacian_variance(arr: np.ndarray) -> float:
    """Approximate sharpness via 3x3 Laplacian. arr is HxW grayscale."""
    a = arr.astype(np.float32)
    lap = (-4 * a[1:-1, 1:-1]
           + a[:-2, 1:-1] + a[2:, 1:-1]
           + a[1:-1, :-2] + a[1:-1, 2:])
    return float(lap.var())


def vegetation_fraction(rgb: np.ndarray) -> float:
    """Soft vegetation mask via ExG threshold."""
    r, g, b = rgb[..., 0].astype(np.int32), rgb[..., 1].astype(np.int32), rgb[..., 2].astype(np.int32)
    exg = 2 * g - r - b
    return float((exg > 20).mean())


def hsv_stats(rgb: np.ndarray) -> dict:
    """RGB->HSV (manual, no cv2)."""
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    delta = cmax - cmin
    sat = np.where(cmax > 1e-6, delta / np.maximum(cmax, 1e-6), 0.0)
    val = cmax
    return {"sat_mean": float(sat.mean()),
            "val_mean": float(val.mean()),
            "sat_std": float(sat.std()),
            "val_std": float(val.std())}


def ruler_band_score(arr: np.ndarray) -> float:
    """Heuristic: do top/bottom/left/right 5% bands have unusually
    high contrast variance vs the inner image? (rulers and tape measures
    appear as bright/dark bands at quadrat edges)."""
    h, w = arr.shape[:2]
    band_h = max(1, h // 20)
    band_w = max(1, w // 20)
    inner = arr[band_h:h - band_h, band_w:w - band_w].astype(np.float32)
    edges = [
        arr[:band_h, :].astype(np.float32),
        arr[h - band_h:, :].astype(np.float32),
        arr[:, :band_w].astype(np.float32),
        arr[:, w - band_w:].astype(np.float32),
    ]
    inner_var = inner.var()
    edge_var = float(np.mean([e.var() for e in edges]))
    return edge_var / max(1.0, inner_var)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    files_by_src = defaultdict(list)
    for f in os.listdir(IMG_DIR):
        if f.endswith(".jpg"):
            files_by_src[source_of(f)].append(f)

    rng = random.Random(42)
    results = defaultdict(list)
    for src, files in files_by_src.items():
        if not files:
            continue
        n = min(N_PER_SOURCE, len(files))
        sampled = rng.sample(files, n)
        print(f"=== {src}: {n} of {len(files)} sampled ===")
        for fn in sampled:
            try:
                img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
                w, h = img.size
                # Downscale for fast stats
                if max(w, h) > 1024:
                    scale = 1024 / max(w, h)
                    img_small = img.resize((int(w * scale), int(h * scale)),
                                           Image.LANCZOS)
                else:
                    img_small = img
                rgb = np.asarray(img_small)
                gray = rgb.mean(axis=2)
                stat = ImageStat.Stat(img_small)
                row = {
                    "image": fn,
                    "width": w,
                    "height": h,
                    "aspect": w / h,
                    "rgb_mean_r": stat.mean[0],
                    "rgb_mean_g": stat.mean[1],
                    "rgb_mean_b": stat.mean[2],
                    "rgb_std_r": stat.stddev[0],
                    "rgb_std_g": stat.stddev[1],
                    "rgb_std_b": stat.stddev[2],
                    "sharpness": laplacian_variance(gray),
                    "veg_frac": vegetation_fraction(rgb),
                    "ruler_band_score": ruler_band_score(gray),
                }
                row.update(hsv_stats(rgb))
                results[src].append(row)
            except Exception as e:
                print(f"  ERR {fn}: {e}")

    # Per-source summary
    print()
    print("=" * 88)
    print("Per-source summary (median values)")
    print("=" * 88)
    summary = {}
    keys = ("width", "height", "aspect", "sharpness", "veg_frac",
            "sat_mean", "val_mean", "rgb_mean_g", "ruler_band_score")
    print(f"{'source':<12} {'n':>4} " +
          " ".join(f"{k:>14s}" for k in keys))
    for src in SOURCES:
        if src not in results:
            continue
        rows = results[src]
        if not rows:
            continue
        vals = [tuple(r[k] for k in keys) for r in rows]
        med = [float(np.median([v[i] for v in vals])) for i in range(len(keys))]
        std = [float(np.std([v[i] for v in vals])) for i in range(len(keys))]
        print(f"{src:<12} {len(rows):>4} " +
              " ".join(f"{m:>14.2f}" for m in med))
        summary[src] = {
            "n": len(rows),
            "median": dict(zip(keys, med)),
            "std": dict(zip(keys, std)),
            "samples": rows,
        }

    out_path = os.path.join(OUT_DIR, "image_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
