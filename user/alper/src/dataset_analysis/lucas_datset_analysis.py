import numpy as np
import pyarrow.parquet as pq
from collections import Counter
import matplotlib.pyplot as plt

parquet_path = "/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k_SSL/step3b_pseudolabels/top8_rel085.parquet"

NUM_CLASSES = 7806
output_path = "step3b_species_longtail.jpg"

# ------------------------------------------------
# Load only necessary column
# ------------------------------------------------
table = pq.read_table(parquet_path, columns=["pos_idx_i32"])

species_counter = Counter()

for batch in table.to_batches():
    for blob in batch.column("pos_idx_i32").to_pylist():
        idx = np.frombuffer(blob, dtype=np.int32)
        species_counter.update(idx)

# ------------------------------------------------
# Compute coverage and frequency distribution
# ------------------------------------------------
unique_species = len(species_counter)
counts = np.array(sorted(species_counter.values(), reverse=True))

coverage_pct = unique_species / NUM_CLASSES * 100

# ------------------------------------------------
# Plot
# ------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, unique_species + 1), counts)
plt.yscale("log")

plt.axvline(unique_species, linestyle="--")

plt.title(
    f"Step 3B Species Frequency Long Tail\n"
    f"Coverage: {unique_species}/{NUM_CLASSES} "
    f"({coverage_pct:.1f}%)"
)
plt.xlabel("Species rank (1 = most frequent)")
plt.ylabel("Number of images containing species (log scale)")
plt.tight_layout()

# Save as JPG
plt.savefig(output_path, dpi=300, format="jpg")
plt.close()

print(f"Saved figure to: {output_path}")