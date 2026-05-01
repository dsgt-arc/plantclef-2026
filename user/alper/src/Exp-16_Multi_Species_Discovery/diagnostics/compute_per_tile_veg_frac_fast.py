"""Fast per-tile vegetation fraction.

Skips the Lanczos resize-to-518N (which is the bottleneck). Computes
veg_frac directly on the source image after center-crop-square, using
pixel-coordinate tile boundaries. The ABSOLUTE veg_frac values differ
slightly from the production tile path, but the RANK ordering of
"which tiles have least vegetation" is preserved, which is all the
veg-filter mechanism needs.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_DIR = r"C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\images"
OUT_PARQUET = (r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub"
               r"\plantclef-2026\user\alper\src\Exp-16_Multi_Species_Discovery"
               r"\outputs\image_audit\per_tile_veg_frac.parquet")
SCALES = (4, 5, 6)


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
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            arr = np.asarray(img.crop((left, top, left + side, top + side)))
            r_ch = arr[..., 0].astype(np.int32)
            g_ch = arr[..., 1].astype(np.int32)
            b_ch = arr[..., 2].astype(np.int32)
            exg = 2 * g_ch - r_ch - b_ch
            veg_mask = exg > 20
            for scale in SCALES:
                tile_h = side // scale
                tile_w = side // scale
                tile_idx = 0
                for rr in range(scale):
                    for cc in range(scale):
                        sub = veg_mask[rr*tile_h:(rr+1)*tile_h,
                                       cc*tile_w:(cc+1)*tile_w]
                        rows.append({
                            "image_name": fn,
                            "scale": scale,
                            "tile_index": tile_idx,
                            "veg_frac": float(sub.mean()),
                        })
                        tile_idx += 1
            if time.time() - last_report > 5.0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(files) - i - 1) / max(rate, 0.1)
                print(f"    {i+1}/{len(files)}  rate={rate:.1f} img/s  eta={eta:.0f}s")
                last_report = time.time()
        except Exception as e:
            print(f"  ERR {fn}: {e}")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote {len(df):,} tile rows -> {OUT_PARQUET}")
    print(f"  total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
