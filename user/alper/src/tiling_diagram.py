"""
Generate a clean diagram showing multi-scale n×n tiling of a vegetation plot image.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

M_TEAL   = "#009688"
M_BLUE   = "#2E86C1"
M_ORANGE = "#E67E22"
M_DARK   = "#333344"
M_GRAY   = "#888899"
M_LGRAY  = "#CCCCDD"
M_BG     = "#FFFFFF"
M_GREEN  = "#2ECC71"

def draw_grid(ax, n, color, label, tile_px):
    """Draw an n×n grid on the axes with labels."""
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")

    # Fill tiles with alternating light shading to show grid
    for i in range(n):
        for j in range(n):
            shade = color + "10" if (i + j) % 2 == 0 else color + "20"
            rect = plt.Rectangle((j, i), 1, 1, facecolor=shade,
                                  edgecolor=color, linewidth=1.2)
            ax.add_patch(rect)

    # Highlight a couple of tiles to show "plant detection"
    highlights = []
    if n == 4:
        highlights = [(1, 2), (2, 1), (3, 3)]
    elif n == 5:
        highlights = [(1, 3), (3, 1), (4, 4)]
    elif n == 6:
        highlights = [(0, 4), (3, 2), (5, 1)]

    for (r, c) in highlights:
        rect = plt.Rectangle((c, r), 1, 1, facecolor=color + "40",
                              edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(c + 0.5, r + 0.5, "\u2713", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

    # Title and info
    ax.set_title(f"{n}\u00d7{n}  ({n*n} tiles)", fontsize=12, fontweight="bold",
                 color=M_DARK, pad=8)
    ax.text(n / 2, -0.35, f"~{tile_px}px per tile", ha="center", fontsize=9, color=M_GRAY)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


fig = plt.figure(figsize=(10, 4), facecolor=M_BG)

# Layout: [original image] → [4×4] → [5×5] → [6×6]
gs = fig.add_gridspec(1, 5, width_ratios=[1.2, 0.3, 1, 1, 1],
                      wspace=0.15, left=0.03, right=0.97, top=0.82, bottom=0.12)

# Original image representation
ax0 = fig.add_subplot(gs[0, 0])
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.set_aspect("equal")

# Simulate a vegetation plot with colored patches
np.random.seed(42)
for _ in range(60):
    x, y = np.random.rand(), np.random.rand()
    size = np.random.uniform(0.03, 0.08)
    green = np.random.choice(["#2ECC71", "#27AE60", "#1ABC9C", "#16A085",
                               "#229954", "#1E8449", "#7DCEA0", "#A9DFBF"])
    circle = plt.Circle((x, y), size, facecolor=green, edgecolor="none", alpha=0.6)
    ax0.add_patch(circle)

# Brown soil background
ax0.set_facecolor("#D4C5A9")
ax0.set_title("Original Image", fontsize=12, fontweight="bold", color=M_DARK, pad=8)
ax0.text(0.5, -0.08, "~3000\u00d73000 px", ha="center", fontsize=9, color=M_GRAY,
         transform=ax0.transAxes)
ax0.set_xticks([])
ax0.set_yticks([])
for sp in ax0.spines.values():
    sp.set_color(M_DARK)
    sp.set_linewidth(1.5)

# Arrow
ax_arrow = fig.add_subplot(gs[0, 1])
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.axis("off")
ax_arrow.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                  arrowprops=dict(arrowstyle="-|>", color=M_DARK, lw=2))
ax_arrow.text(0.5, 0.62, "tile", ha="center", fontsize=9, color=M_GRAY, fontstyle="italic")

# 4×4 grid
ax1 = fig.add_subplot(gs[0, 2])
draw_grid(ax1, 4, M_BLUE, "4\u00d74", 750)

# 5×5 grid
ax2 = fig.add_subplot(gs[0, 3])
draw_grid(ax2, 5, M_TEAL, "5\u00d75", 600)

# 6×6 grid
ax3 = fig.add_subplot(gs[0, 4])
draw_grid(ax3, 6, M_ORANGE, "6\u00d76", 500)

# Overall title
fig.suptitle("Multi-Scale Tiling: 4\u00d74 + 5\u00d75 + 6\u00d76 = 77 tiles per image",
             fontsize=14, fontweight="bold", color=M_DARK, y=0.97)

# Bottom annotation
fig.text(0.5, 0.02,
         "\u2713 = tile where species detected  |  Finer tiles \u2192 closer match to single-plant training images  |  max_prob selects best detection per species",
         ha="center", fontsize=8.5, color=M_GRAY)

out = r"C:\Users\alper\Documents\PlantCLEF_2026_Local\_GitHub\plantclef-2026\user\alper\src\tiling_diagram.png"
fig.savefig(out, dpi=250, facecolor=M_BG, edgecolor="none")
plt.close(fig)
print(f"Saved: {out}")
