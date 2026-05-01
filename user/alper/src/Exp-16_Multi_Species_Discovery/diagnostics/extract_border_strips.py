"""For 3 sample images per source, extract the outer 8% border strip
(top + bottom + left + right) and save as a single mosaic to inspect
whether rulers/tape are present.
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_DIR = r"C:\Users\alper\Documents\PlantCLEF_2026_Local\PlantCLEF2025test\images"
OUT_DIR = (r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub"
           r"\plantclef-2026\user\alper\src\Exp-16_Multi_Species_Discovery"
           r"\outputs\image_audit\borders")

SOURCES = ("RNNB", "CBN-PdlC", "CBN-Pla", "GUARDEN", "LISAH", "OPTMix")
N = 3
BORDER_FRAC = 0.08


def source_of(name):
    for p in SOURCES:
        if name.startswith(p):
            return p
    return None


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    files_by_src = defaultdict(list)
    for f in os.listdir(IMG_DIR):
        if f.endswith(".jpg"):
            s = source_of(f)
            if s:
                files_by_src[s].append(f)
    rng = random.Random(42)
    for src, files in files_by_src.items():
        sampled = rng.sample(files, min(N, len(files)))
        for fn in sampled:
            img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
            w, h = img.size
            bw = max(20, int(w * BORDER_FRAC))
            bh = max(20, int(h * BORDER_FRAC))
            top = img.crop((0, 0, w, bh))
            bot = img.crop((0, h - bh, w, h))
            left = img.crop((0, 0, bw, h))
            right = img.crop((w - bw, 0, w, h))
            # Resize all to small width for quick visual inspection
            target_w = 800
            ratio = target_w / w
            top = top.resize((target_w, int(bh * ratio)), Image.LANCZOS)
            bot = bot.resize((target_w, int(bh * ratio)), Image.LANCZOS)
            target_h = int(h * ratio)
            left = left.resize((int(bw * ratio), target_h), Image.LANCZOS)
            right = right.resize((int(bw * ratio), target_h), Image.LANCZOS)

            mosaic_h = top.height + target_h + bot.height
            mosaic_w = max(target_w, left.width + right.width)
            mosaic = Image.new("RGB", (mosaic_w, mosaic_h), (40, 40, 40))
            mosaic.paste(top, (0, 0))
            mosaic.paste(left, (0, top.height))
            mosaic.paste(right, (target_w - right.width, top.height))
            mosaic.paste(bot, (0, top.height + target_h))
            out_path = os.path.join(OUT_DIR, f"{src}_{fn.replace('.jpg', '')}_borders.jpg")
            mosaic.save(out_path, quality=85)
            print(f"  {out_path}")


if __name__ == "__main__":
    main()
