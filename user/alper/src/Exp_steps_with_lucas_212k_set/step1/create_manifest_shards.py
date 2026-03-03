from pathlib import Path
import math

image_dir = Path("/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/data/raw/lucas_212k_images")
manifest_dir = Path("/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/manifests/lucas_212k")
manifest_dir.mkdir(parents=True, exist_ok=True)

# Collect images
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
images = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

print("Total images found:", len(images))

# Write master manifest
all_path = manifest_dir / "lucas_212k_all.txt"
with open(all_path, "w") as f:
    for img in images:
        f.write(str(img) + "\n")
print("Wrote:", all_path)

# Shard manifests
shard_size = 1000
num_shards = math.ceil(len(images) / shard_size)

for shard_id in range(num_shards):
    shard_path = manifest_dir / f"shard_{shard_id:04d}.txt"
    start = shard_id * shard_size
    end = min(start + shard_size, len(images))
    with open(shard_path, "w") as f:
        for img in images[start:end]:
            f.write(str(img) + "\n")

print("Created shards:", num_shards)
print("Last shard size:", len(images) - (num_shards - 1) * shard_size)
