import numpy as np
import pyarrow.parquet as pq
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parquet_path = "/storage/home/hcoda1/2/aerten3/ps-dsgt_clef2026-0/shared/plantclef_2026/parquet/lucas_212k_SSL/step3b_pseudolabels/top8_rel085.parquet"

NUM_CLASSES = 7806

# Set a threshold for the second figure
MIN_COUNT_FOR_BAR = 600

out_longtail_jpg = "step3b_species_longtail_filled.jpg"
out_bar_jpg = f"step3b_species_counts_ge_{MIN_COUNT_FOR_BAR}.jpg"

# Load only what we need
table = pq.read_table(parquet_path, columns=["pos_idx_i32"])

counter = Counter()
for batch in table.to_batches():
    for blob in batch.column("pos_idx_i32").to_pylist():
        idx = np.frombuffer(blob, dtype=np.int32)
        counter.update(idx)

unique_species = len(counter)
coverage_pct = 100.0 * unique_species / NUM_CLASSES

counts = np.array(sorted(counter.values(), reverse=True), dtype=np.int64)
ranks = np.arange(1, len(counts) + 1, dtype=np.int64)

# ----------------------------
# Figure 1: Filled long tail
# ----------------------------
plt.figure(figsize=(14, 7))
plt.fill_between(ranks, counts, step="pre", alpha=0.9)
plt.yscale("log")

plt.title(
    f"Step 3B Species Frequency Long Tail\n"
    f"Coverage: {unique_species}/{NUM_CLASSES} ({coverage_pct:.1f}%)",
    fontsize=22,
    weight="bold",
)
plt.xlabel("Species (sorted by frequency)", fontsize=18)
plt.ylabel("Count (number of images)", fontsize=18)

plt.tight_layout()
plt.savefig(out_longtail_jpg, dpi=300, format="jpg")
plt.close()

# -----------------------------------------
# Figure 2: Bar chart for counts >= thresh
# -----------------------------------------
counts_ge = counts[counts >= MIN_COUNT_FOR_BAR]

plt.figure(figsize=(14, 7))
plt.bar(np.arange(len(counts_ge)), counts_ge)

plt.title(
    f"Step 3B Species with Count >= {MIN_COUNT_FOR_BAR}\n"
    f"Coverage: {unique_species}/{NUM_CLASSES} ({coverage_pct:.1f}%)",
    fontsize=22,
    weight="bold",
)
plt.xlabel("Species (sorted, names omitted)", fontsize=18)
plt.ylabel("Count (number of images)", fontsize=18)

# Remove x tick labels since names are not needed
plt.xticks([])

plt.tight_layout()
plt.savefig(out_bar_jpg, dpi=300, format="jpg")
plt.close()

print("Saved:")
print(out_longtail_jpg)
print(out_bar_jpg)
print(f"Unique species: {unique_species} of {NUM_CLASSES} ({coverage_pct:.1f}%)")
print(f"Species with count >= {MIN_COUNT_FOR_BAR}: {len(counts_ge)}")