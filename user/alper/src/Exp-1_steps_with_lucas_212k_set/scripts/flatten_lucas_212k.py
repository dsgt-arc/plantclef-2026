from pathlib import Path
import hashlib
import shutil

# PACE paths
src = Path("/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/data/raw/lucas_212k_source")
dst = Path("/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/data/raw/lucas_212k_images")

dst.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

moved = 0
scanned = 0

print("Source:", src)
print("Exists:", src.exists())
print("Destination:", dst)

for p in src.rglob("*"):
    scanned += 1
    if p.is_file() and p.suffix.lower() in exts:
        h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:10]
        new_name = f"{p.stem}_{h}{p.suffix.lower()}"
        shutil.move(str(p), dst / new_name)   # <-- move instead of copy
        moved += 1

        if moved % 5000 == 0:
            print("Moved so far:", moved)

print("Scanned items:", scanned)
print("Moved images:", moved)
