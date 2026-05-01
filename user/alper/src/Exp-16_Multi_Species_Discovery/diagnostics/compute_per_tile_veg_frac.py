"""Compute per-tile vegetation fraction for every test image at scales
4x4, 5x5, 6x6. Uses the same tiling pipeline as the production extractor
(center-crop-square -> resize to scale * 518 -> slice into N*N tiles).

Vegetation fraction = mean(ExG > 20) where ExG = 2G - R - B (per tile).

Output: parquet with image_name, scale, tile_index, veg_frac.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DINOV2_SIZE = 518
IMG_DIR = r"C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\images"
OUT_PARQUET = (r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub"
               r"\plantclef-2026\user\alper\src\Exp-16_Multi_Species_Discovery"
               r"\outputs\image_audit\per_tile_veg_frac.parquet")
SCALES = (4, 5, 6)


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main() -> None:
    Path(os.path.dirname(OUT_PARQUET)).mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".jpg"))
    print(f"  {len(files)} images")
    rows = []
    t0 = time.time()
    last_report = t0
    for i, fn in enumerate(files):
        try:
            img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
            img = center_crop_square(img)
            for scale in SCALES:
                target = scale * DINOV2_SIZE
                resized = img.resize((target, target), Image.LANCZOS)
                arr = np.asarray(resized)  # H, W, 3 in RGB
                r_ch = arr[..., 0].astype(np.int32)
                g_ch = arr[..., 1].astype(np.int32)
                b_ch = arr[..., 2].astype(np.int32)
                exg = 2 * g_ch - r_ch - b_ch
                veg_mask = exg > 20
                tile_idx = 0
                for rr in range(scale):
                    for cc in range(scale):
                        sub = veg_mask[rr*DINOV2_SIZE:(rr+1)*DINOV2_SIZE,
                                       cc*DINOV2_SIZE:(cc+1)*DINOV2_SIZE]
                        rows.append({
                            "image_name": fn,
                            "scale": scale,
                            "tile_index": tile_idx,
                            "veg_frac": float(sub.mean()),
                        })
                        tile_idx += 1
            if time.time() - last_report > 5.0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(files) - i - 1) / rate
                print(f"    {i+1}/{len(files)}  rate={rate:.1f} img/s  eta={eta:.0f}s")
                last_report = time.time()
        except Exception as e:
            print(f"  ERR {fn}: {e}")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote {len(df):,} tile rows -> {OUT_PARQUET}")
    print(f"  total time: {time.time() - t0:.1f}s")
    # Per-source veg_frac distribution
    def src(q):
        for p in ['CBN-PdlC','CBN-Pla','RNNB','GUARDEN','LISAH','OPTMix','2024-CEV3']:
            if q.startswith(p): return p
        return 'OTHER'
    df['src'] = df['image_name'].apply(src)
    print("\n  veg_frac distribution per source (median, p10, p25, p75, p90):")
    for s, g in df.groupby('src'):
        med = g.veg_frac.median()
        p10 = g.veg_frac.quantile(0.10)
        p25 = g.veg_frac.quantile(0.25)
        p75 = g.veg_frac.quantile(0.75)
        p90 = g.veg_frac.quantile(0.90)
        n_low = (g.veg_frac < 0.05).sum()
        n_total = len(g)
        print(f"    {s:<12} med={med:.3f}  p10={p10:.3f}  p25={p25:.3f}  p75={p75:.3f}  p90={p90:.3f}  "
              f"<0.05: {n_low}/{n_total} ({100*n_low/n_total:.1f}%)")


if __name__ == "__main__":
    main()
